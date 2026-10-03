from functools import cache

from .utils import load_jit


@cache
def runtime_batch_module():
    return load_jit(
        "runtime_batch", cuda_files=["runtime_batch.cuh"],
        cuda_wrappers=[("prepare", "&PrepareBatch::run"),
                       ("prefill_metadata", "&PrefillMetadata::run"),
                       ("qsa_metadata", "&QsaMetadata::run"),
                       ("copy_inputs", "&CopyBatchInputs::run"),
                       ("write_pages", "&WritePages::run"),
                       ("alloc_swa", "&AllocSwa::run"),
                       ("argmax", "&GreedyArgmax::run")],
    )
