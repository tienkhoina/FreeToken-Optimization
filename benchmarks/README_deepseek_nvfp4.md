# DeepSeek-V4 NVFP4 and shared-expert fusion

The RedHatAI/DeepSeek-V4-Flash-0731-NVFP4 checkpoint now has a source path into
FreeToken's existing optimized NVFP4 expert operators. This change is locally
implemented and CPU-checked; full-model GPU execution and A/B throughput remain
unverified. The original DeepSeek MXFP4/e8m0 path remains available.

## Shared and architecture-specific optimizations

| Area | DeepSeek-V4 behavior |
|---|---|
| Prepared bank copies, pinned input staging, KV preparation and expert caching | Already shared with Qwen through the common runtime |
| NVFP4 wide-load decode and grouped prefill | Reuses the existing kernels after checkpoint names/scales are normalized |
| A one-token selective prefill | Uses the wide-load decode operator instead of expert sorting and grouped GEMM |
| Native hit/fetch/CPU route scheduling and ordered NVFP4 reduction | Available with `--moe-native-schedule`; decode and prefill use the common scheduler |
| Shared expert gate/up | `--dsv4-fuse-shared-expert` packs weights once, then uses one projection for both outputs |
| Whole-model bucket prefill graphs | `--dsv4-prefill-mode bucket` captures token/context shapes at startup, including sparse attention, compressor, indexer and MoE |
| Layer/block sweeps and ragged graph batches | Not implemented; multiple-request batches and uncovered contexts keep eager fallback |

The shared-expert fusion flag defaults off. It keeps the original w1/w2/w3
checkpoint keys as views into the merged gate/up allocation, and merges block
scales alongside weights. The SwiGLU kernel accepts those strided halves without
extra contiguous copies. A larger GEMM can change rounding; byte equality across
different projection shapes is not assumed.

## Checkpoint handling

DeepSeek-specific compressed-tensors handling recognizes the checkpoint's
w1/w2/w3 modules against gate/up/down target names, and recognizes wq_a/wkv
against fused_wqa_wkv. Explicit ignore entries and the original fp8 dialect are
preserved. NVFP4 experts use the common Qwen bank reader and packer: E2M1 bytes,
per-16 E4M3 scales and reciprocal tensor-global scales. MTP layers are excluded.

Attention/shared-expert `weight_scale` values are floating BF16 scales;
reference checkpoints' `.scale` values remain E8M0 codes. The wo_a conversion
distinguishes those formats. Model arguments can come from reference JSON or
the checkpoint's Hugging Face config; checkpoint files are not rewritten.

The optimized NVFP4 path uses BF16/FP16 activations (W4A16), rather than reproducing
the checkpoint publisher's NVFP4 activation quantization (W4A4). The publisher's
vLLM logits and evaluation scores are therefore not a bit-exact reference.

## Validation performed

CPU tests cover actual checkpoint configuration/index metadata, toy expert
shards, reciprocal global scales, floating/E8M0 block-scale conversion, original
checkpoint keys, merged BF16 output, FP8 weight/scale bytes, selective prefill
dispatch and the native scheduler's eager-prefill/configuration behavior.

The SwiGLU kernel is checked in Triton's CPU interpreter for strided and
contiguous inputs and can be compiled offline for sm_90. These checks do not
establish GPU numerical correctness or performance. No engine was started and
no GPU kernel or full model was run during this implementation.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python .venv-cpu-check/bin/python -m pytest \
  tests/models/deepseek_v4/test_optimization.py -q
CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 PYTHONPATH=python \
  .venv-cpu-check/bin/python -m pytest \
  tests/models/deepseek_v4/test_swiglu_interpreter.py -q
```

The tests and local evidence under `results/deepseek_optimization_2026-10-05/`
are excluded by this experimental repository's existing `.gitignore`.

## GPU A/B protocol to run separately

The following commands have **not** been run. Use a compatible PyTorch/CUDA/driver
environment and an available GPU. The synthetic benchmark isolates the shared
expert and excludes compilation, packing, warmup and capture from timing:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_dsv4_shared_expert.py \
  --format fp8_float --hidden 4096 --intermediate 2048 --tokens 1 32 128 \
  --repeats 51 --output /path/to/results/dsv4_shared_fp8.json
```

For end-to-end A/B, hold checkpoint revision, exact input IDs, output length,
sampling, expert/KV budgets, random preload seed and request order fixed. Run
the new source with fusion off versus on first. Test native scheduling separately
so its effect is distinguishable from projection fusion. Compare cold TTFT, warm
TTFT, decode tokens/s, logits/token IDs and GPU memory; retain all samples.

Example runtime settings for a later check, with GPU-only expert misses until a
CPU/GPU profile has been measured on that machine:

```bash
ft serve --model /path/to/DeepSeek-V4-Flash-0731-NVFP4 \
  --attention-backend dsv4_sparse --quant-backend moe.nvfp4=triton \
  --moe-strategy offload --moe-cache-auto --kv-reserve-tokens 8192 \
  --moe-cache-init random --moe-cache-init-seed 42 \
  --moe-native-schedule --moe-gpu-miss-fraction 1 \
  --moe-prefill-gpu-miss-fraction 1 --dsv4-fuse-shared-expert
```

The cache/KV numbers above are example workload settings, not a calibrated fit
for every device. Original `main` cannot load this NVFP4 export, so there is no
end-to-end NVFP4 A/B against unmodified `main`. A separate original MXFP4 checkpoint
is needed for a comparable `main` baseline of the shared-expert optimization.

## Bucket prefill and measured CPU/GPU split

Bucket mode is opt-in and currently requires DeepSeek NVFP4 with the native
Triton expert layout. Token and context buckets are window-page multiples
(128 for this checkpoint). A query uses the smallest covering token/context
pair. The graph accepts a GPU logical token count and prefix length; padding
cannot update window KV, compressed KV, carry rings or expert-cache routes.
Prefix lengths must be page aligned, matching radix reuse boundaries.

Only one-request text-prefill batches use these graphs. Ragged batches, contexts
beyond retained buckets and nonaligned continuation fall back to the existing
eager path. Default context coverage is bounded at 8192 (or the sequence ceiling),
not the checkpoint's full million-token maximum. More context buckets increase
the retained indexer-score workspaces and capture memory; measure their cost.
These are whole-model bucket graphs, not layer-major sweeps.
The physical context width changes the indexer's top-k candidate shape. Equal
scores can produce a different tie choice across shapes, in addition to GEMM
and pooling rounding differences; cross-shape logits are not promised bit exact.

Example flags for a later GPU check:

```bash
ft serve --model /path/to/DeepSeek-V4-Flash-0731-NVFP4 \
  --attention-backend dsv4_sparse --quant-backend moe.nvfp4=triton \
  --moe-strategy offload --moe-cache-auto --moe-native-schedule \
  --dsv4-prefill-mode bucket --moe-prefill-graph-max-tokens 512 \
  --dsv4-prefill-buckets 128,256,512 \
  --dsv4-prefill-context-buckets 512,2048,8192 \
  --moe-gpu-miss-fraction 1 --moe-prefill-gpu-miss-fraction 1
```

The runtime profile can now contain multiple `prefill_profiles`. Each row maps
distinct miss counts to integer GPU fetch counts. The GPU scheduler selects a
row by physical query-token bucket; entries outside its measured miss coverage
fall back to the configured fraction. A decode profile remains restricted to
one-token decode; a multi-request decode does not reuse a prefill profile.
Expert assignment preserves router IDs/weights. GPU hits, copied misses and CPU
misses execute on the existing streams and join before the next layer.

The calibration script reads a bounded set of actual DeepSeek experts, including
reciprocal compressed-tensors global scales. Prefill calibration can sweep more
distinct experts than top-k, while preserving at most top-k routes per token.
It measures complete CPU branch latency, weight copy and GPU operator separately,
then estimates `max(CPU branch, H2D copy + GPU compute)` for each integer split.
This model excludes integrated memory/stream contention and is not proof of an
optimal runtime split. The recorded CPU/GPU numerical errors also need review.

These measurement commands have **not** been run in this session:

```bash
PYTHONPATH=python python benchmarks/calibrate_moe_schedule.py \
  --workload dsv4-nvfp4 --format nvfp4 --model /path/to/DeepSeek-V4-Flash-0731-NVFP4 \
  --experts 32 --threads 0 --tokens 1 --output /path/to/results/decode.json
PYTHONPATH=python python benchmarks/calibrate_moe_schedule.py \
  --workload dsv4-nvfp4 --format nvfp4 --model /path/to/DeepSeek-V4-Flash-0731-NVFP4 \
  --experts 32 --threads 0 --tokens 128 --max-misses 32 \
  --output /path/to/results/prefill128.json
PYTHONPATH=python python benchmarks/calibrate_moe_schedule.py \
  --workload dsv4-nvfp4 --merge-profiles /path/to/results/decode.json \
  /path/to/results/prefill128.json --output /path/to/results/schedule.json
PYTHONPATH=python python benchmarks/bench_dsv4_prefill.py \
  --tokens 1 127 128 129 256 --prefix 0 128 --cap 256 --context 512 \
  --output /path/to/results/prefill_graph.json
```

Keep GPU model, CPU ISA/workers, expert dimensions, activation and top-k identical
when merging profiles. Use the merged JSON with `--moe-schedule-profile`. Each
bucket you intend to calibrate needs its own measured profile. No measured
DeepSeek profile or throughput gain is claimed from CPU testing/offline builds.
