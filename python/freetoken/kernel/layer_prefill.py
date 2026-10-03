from functools import cache
from pathlib import Path

from .utils import load_jit


@cache
def layer_prefill_module():
    return build_layer_prefill_module()


def build_layer_prefill_module(build_directory=None):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if torch.version.hip is not None:
        raise RuntimeError("native layer prefill currently uses CUDA graph/runtime APIs")
    if CUDA_HOME is None:
        raise RuntimeError("native layer dispatcher requires CUDA toolkit headers and runtime")
    root = Path(CUDA_HOME)
    lib = root / "lib64"
    if not (lib / "libcudart.so").exists():
        lib = root / "lib"
    return load_jit("layer_prefill", cpp_files=["layer_prefill.cuh"],
                    cpp_wrappers=[("run", "&LayerPrefill::run")],
                    extra_include_paths=[str(root / "include")],
                    extra_ldflags=[f"-L{lib}", "-lcudart", f"-Wl,-rpath,{lib}"],
                    build_directory=build_directory)
