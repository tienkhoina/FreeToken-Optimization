# Runtime batch preparation A/B

The runtime bundle removes repeated input/index preparation on the decode and
prefill paths. One pinned descriptor transfer and a CUDA kernel prepare positions,
token gathers, KV locations, write indices and optional GDN slots. Two persistent
buffers and copy/release events protect host DMA and overlapping GPU consumers.
The runtime still uses the existing scheduler admission and prefix-cache policies.

Captured decode copies four input arrays with one CUDA kernel and reuses the
descriptor for KV attention metadata. Triton prefill creates its ragged attention
metadata with one CUDA kernel. Page-table writes use a small descriptor rather than
two host index arrays. SWA slot allocation avoids temporary int64 casts/scatter.
Other attention backends continue using their existing metadata and math kernels.

Greedy sampling uses a CUDA argmax that writes int32 token IDs and preserves
PyTorch's first-index tie and NaN rules. Larger vocabularies use parallel partial
reductions. Scratch is preallocated at startup. Non-greedy sampling reuses its
parameter tensors when the ordered temperature/top-k/top-p values remain equal;
its probability/RNG operator is unchanged. CPU MoE health checks scan the error
flags in compiled C++ instead of creating CPU comparison/reduction tensors.

Build the native extension after applying the changes:

```bash
python setup.py build_ext --inplace
python -m pytest tests/scheduler/test_input_staging.py \
  tests/engine/test_sampling_native.py tests/moe/test_cpu_moe.py -q
```

Measure the same script against a saved source snapshot and the current runtime,
sequentially on an idle GPU:

```bash
PYTHONPATH=/path/to/baseline/python python benchmarks/bench_runtime_batch.py \
  --output /path/to/results/before.json --save-tensors
PYTHONPATH=python python benchmarks/bench_runtime_batch.py \
  --output /path/to/results/after.json --save-tensors
```

Use a CUDA toolkit matching PyTorch. Compilation and warmup are excluded; timed
steps include host dispatch, staging and GPU execution. Decode steps replay a
warmed attention graph; prefill steps measure input/metadata/sampling preparation.
The synthetic workload does not contain MLP/MoE math or replace a model benchmark.
PyTorch intra-op is one thread, matching inference with the CPU MoE worker pool.
All samples, input hashes and output hashes are saved. `--save-tensors` stores both
input and output tensors for byte comparisons; do not infer bit equality from
similar generated text. `--outputs-only` saves proof/memory data without timing.

Transient VRAM is measured with PyTorch allocated bytes during a warmed step and
reported separately from persistent input/sampling tensor payloads. It excludes
reserved allocator blocks, graph pools and full-model VRAM. Scratch/input buffers
trade a small persistent allocation for fewer per-step allocations.

For the real model A/B, keep checkpoint, expert cache slots/policy/preload seed,
KV pages, graph buckets, CPU worker count, sampling, prompts and request order
the same. This workspace preserves source and real GPT-OSS-120B results under
`result/decode_runtime_bundle_2026-10-01/` in the parent directory. Exact CLI flags
and environment are in `run_server.py` and the saved server JSON files. Compare
warm medians and exact logits/token IDs separately from loader/JIT/frontend costs.

The GDN optimization reuses the constant captured decode query indptr. Multimodal
positions retain the existing mrope builder. Prefix tree traversal/eviction,
periodic SWA frees, non-greedy probability kernels and server streaming remain
outside this bundle; this does not claim that every possible optimization in
every model architecture has been exhausted.
