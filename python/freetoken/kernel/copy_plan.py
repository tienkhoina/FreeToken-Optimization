"""Prepared byte copies with stable pointers, using the device's copy engines."""

from functools import cache
from typing import Iterable

import torch

from .utils import load_jit


@cache
def _copy_plan_class():
    import tvm_ffi

    module = load_jit("byte_copy_plan", cuda_files=["copy_plan.cuh"])

    @tvm_ffi.register_object("freetoken.ByteCopyPlan")
    class NativeCopyPlan(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    NativeCopyPlan._module = module
    return NativeCopyPlan


class ByteCopyPlan:
    """Prepare independent copies once and enqueue them with one native call.

    All views must be contiguous, with equal shape/dtype per pair. CPU views
    must be pinned. Destinations must not overlap another copy's source or
    destination. The plan retains the tensor views for its lifetime; callers
    must fence buffer reuse and keep the plan alive until its work completes.
    Host descriptors are prepared only from information already known on the
    CPU, such as full-layer banks. Device-generated indices need the GPU copy
    kernel or an explicit host synchronization.
    """

    def __init__(self, destinations: Iterable[torch.Tensor], sources: Iterable[torch.Tensor]):
        self.destinations = tuple(destinations)
        self.sources = tuple(sources)
        if len(self.destinations) != len(self.sources):
            raise ValueError("source/destination counts differ")
        device = None
        for dst, src in zip(self.destinations, self.sources):
            if dst.shape != src.shape or dst.dtype != src.dtype:
                raise ValueError("copy views must have equal shape and dtype")
            if not dst.is_contiguous() or not src.is_contiguous():
                raise ValueError("copy views must be contiguous")
            for tensor in (dst, src):
                if tensor.device.type == "cpu":
                    if tensor.numel() and not tensor.is_pinned():
                        raise ValueError("CPU copy views must be pinned")
                elif tensor.device.type == "cuda":
                    if device is not None and tensor.device != device:
                        raise ValueError("copy views must use one CUDA device")
                    device = tensor.device
                else:
                    raise ValueError("copy views must be CPU or CUDA")
        if device is None:
            raise ValueError("a copy plan requires a CUDA view")
        self.device = device
        descriptors = [torch.tensor(values, dtype=torch.int64, device="cpu") for values in (
            [x.data_ptr() for x in self.destinations],
            [x.data_ptr() for x in self.sources],
            [x.numel() * x.element_size() for x in self.sources],
        )]
        with torch.cuda.device(device):
            self._native = _copy_plan_class()(*descriptors)

    def launch(self, stream: torch.cuda.Stream | None = None) -> None:
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        if stream.device != self.device:
            raise ValueError("copy stream and tensors must use the same CUDA device")
        self._native.launch(stream.cuda_stream)

    @property
    def bytes(self) -> int:
        return self._native.bytes()
