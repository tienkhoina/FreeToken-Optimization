from functools import cache

from .utils import load_jit


@cache
def decode_metadata_module():
    return load_jit(
        "prepare_decode_metadata",
        cuda_files=["prepare_decode_metadata.cuh"],
        cuda_wrappers=[("launch", "&PrepareDecodeMetadata::run")],
    )
