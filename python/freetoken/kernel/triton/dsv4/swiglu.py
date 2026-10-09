"""Fused clamped-SwiGLU for the DeepSeek-V4 shared (dense) expert.

Reference does ``silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)`` in fp32
(``w1(x).float()``, ``w3(x).float()``) then casts back to bf16 -- that's 2 upcasts,
clamp x2, silu, mul, downcast = 7 launches/layer x 43 layers. This kernel reads the
bf16 gate/up directly, does the math in fp32 internally, and writes bf16: bit-exact
to the reference (the inputs were exact bf16 upcasts) in a single launch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, N, limit, BLOCK: tl.constexpr,
                   HAS_LIMIT: tl.constexpr, OUT: tl.constexpr,
                   ROW_WIDTH: tl.constexpr, GATE_STRIDE: tl.constexpr, UP_STRIDE: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    rows, cols = offs // ROW_WIDTH, offs % ROW_WIDTH
    g = tl.load(gate_ptr + rows * GATE_STRIDE + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + rows * UP_STRIDE + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        u = tl.minimum(tl.maximum(u, -limit), limit)
        g = tl.minimum(g, limit)
    g = g * tl.sigmoid(g)
    tl.store(out_ptr + offs, (g * u).to(OUT), mask=mask)


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float,
                 out_dtype: torch.dtype) -> torch.Tensor:
    """``silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)`` -> ``out_dtype``."""
    out = torch.empty(gate.shape, dtype=out_dtype, device=gate.device)
    # The two halves of a merged projection share storage with a doubled row stride.
    gate, up = gate.view(-1, gate.shape[-1]), up.view(-1, up.shape[-1])
    if gate.stride(1) != 1 or up.stride(1) != 1:
        raise ValueError("DeepSeek-V4 SwiGLU requires contiguous columns")
    N = gate.numel()
    BLOCK = 512
    _swiglu_kernel[(triton.cdiv(N, BLOCK),)](
        gate, up, out, N, float(limit), BLOCK=BLOCK,
        HAS_LIMIT=limit > 0, OUT=_TL[out_dtype], num_warps=4,
        ROW_WIDTH=gate.shape[-1], GATE_STRIDE=gate.stride(0), UP_STRIDE=up.stride(0),
    )
    return out


__all__ = ["fused_swiglu"]
