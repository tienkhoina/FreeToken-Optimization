# Decode KV metadata benchmark

`bench_decode_metadata.py` measures KV metadata preparation with the Python dispatch
cost still present, and preparation followed by a warmed attention CUDA graph.
It does not replace real model throughput with an operator estimate.

Run the identical script against the saved source and current source, sequentially
on an idle GPU:

```bash
PYTHONPATH=/path/to/baseline/python python benchmarks/bench_decode_metadata.py \
  --output /path/to/results/before.json --save-inputs
PYTHONPATH=python python benchmarks/bench_decode_metadata.py \
  --output /path/to/results/after.json
```

Use the CUDA toolkit matching PyTorch. Kernel compilation and warmup happen before
timing. The output records all samples, median wall/host enqueue time, CUDA spans,
input hashes, and raw output hashes. CUDA spans include idle gaps while Python is
submitting work; they are not pure kernel execution time. Samples include repeated
host metadata creation and graph replay, rather than measuring a captured static
metadata graph that would hide the overhead being optimized.

The workloads cover batches 1/4/16, contexts 128/1024/8192, and optional full-to-SWA
translation. `--save-inputs` stores random inputs as `.pt` files. Independent runs
use the same seeds; compare every input hash before interpreting output equality.
The attention kernel and floating point arithmetic stay the same.
`--outputs-only --save-outputs` stores the output tensors and measures transient
allocated VRAM during one metadata staging call without repeating latency samples.
This counts PyTorch allocated bytes, not reserved allocator blocks or whole-model VRAM.

The fast runtime path applies to captured Triton decode batches with int32 positions.
It sends a small `(table_idx, device_len)` descriptor array to the GPU, then a CUDA
kernel writes ragged KV indices, cumulative lengths, positions and optional SWA
indices directly into persistent capture buffers. Constants and attention scratch
are reused. The engine stream stages these buffers after the prior replay; the
scheduler does not overwrite buffers while an earlier forward is reading them.
Prefill, other attention backends, and unsupported inputs keep the existing path.
The empty KV deferred-free region also skips an unnecessary `torch.cat` when no
pages were returned. Prefix tree and allocation policies remain the same.

For a model comparison, keep checkpoint, graph configuration, expert cache, preload
seed, KV pages, CPU worker count, prompts, sampling and request order the same. Save
source snapshots before edits and compare raw logits and delivered token IDs.
Warm request medians exclude server startup, JIT and frontend initialization; report
those separately. Operator gains alone do not establish a whole-model speedup.

Development artifacts are saved under `result/kv_decode_overhead_2026-10-01/` in the
parent workspace: source snapshot/manifest, input tensors, raw benchmark samples,
real GPT-OSS-120B server A/B, correctness proofs and `REPORT.md`.
