# Qwen3.8-Flash-Next NVFP4 prefill

Bucket/block implementation and current flags are documented in
[README_bucket_block_prefill.md](README_bucket_block_prefill.md). The exact-length
results below are the preserved earlier experiment; `--qwen-prefill-mode exact`
selects that comparison path.

The native schedule supports Qwen's QSA/GDN/PLE text path: GPU hits, fetched
misses on a copy stream, and compiled CPU misses when calibration recommends them.
Random expert preload remains independent of LRU and runs after graph startup.

Exact query lengths avoid padding that would advance recurrent/conv/PLE state.
Single-request lengths 1..the cap, plus interior GDN snapshot variants above 64,
are warmed/captured before readiness. Long prompts use chunks capped at that size.
Prefix lengths, KV tables, state slots, freshness and snapshot destinations remain
runtime tensors. Multi-request length tuples capture their own exact graph on first
use; all possible ragged tuples are not pre-captured at startup. Multimodal
encoder/block-attention batches keep their prior fallback. Decode keeps QSA/GDN graphs.
More graphs cost startup time and VRAM; choose the cap for the workload.

QSA metadata is prepared by a CUDA kernel. Host-known FLA chunk plans avoid D2H
readbacks during capture. PLE convolution indices are retained with each graph.
Disk PLE fills persistent pinned staging, then a graph handshake/copy/dequant
consumes it. Capture warmup restores recurrent, conv, PLE and QSA ring state.
Sampling remains outside the model graph. Triton NVFP4 kernels skip the immutable
zero expert row for inactive branches; ordinary expert math is unchanged. Native
NVFP4 hit/fetch branches retain per-route results, then share one ordered FP32
reduction. Reducing each branch to BF16 first introduced avoidable rounding;
merging routes preserves the unsplit GPU operator's output bytes in mixed-route
decode and prefill tests. Hits still execute while misses are fetched.

## Calibrate and serve

```bash
python benchmarks/calibrate_moe_schedule.py \
  --workload qwen3.8-flash-next --format nvfp4 \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --threads 0 \
  --output /path/to/results/qwen_profile.json
ft serve --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --attention-backend qsa_sparse --ple-backend disk \
  --moe-strategy offload --quant-backend moe.nvfp4=triton \
  --moe-cache-auto --kv-reserve-tokens 8192 \
  --moe-cache-init random --moe-cache-init-seed 42 \
  --moe-native-schedule --moe-cpu-threads 0 \
  --moe-schedule-profile /path/to/results/qwen_profile.json \
  --moe-prefill-gpu-miss-fraction 1.0 --moe-prefill-graph-max-tokens 64
```

Calibration reads real Qwen expert weights through the production packer and uses
all allowed CPU workers. CPU compute is compiled C++; GPU math/copy and full CPU
branch measurements replay warmed graphs. GPU decode uses the production Triton
wide-load GEMV. Profile dimensions/format/ISA/GPU/workers must match the runtime.
The GPU-only prefill fraction above is a measured-machine choice, not universal.
The development profile chose GPU for 0..10 decode misses while keeping 64 CPU workers
ready; it did not demonstrate a CPU speedup. Keep cache/KV budgets identical in A/B.

## Validate and benchmark

```bash
python -m pytest tests/models/qwen4_exp/test_prefill_graph.py \
  tests/moe/test_nvfp4_backends.py tests/models/qwen4_exp/test_ple_disk.py -q
python benchmarks/bench_qwen_prefill_ops.py --output /path/to/results/ops.json
python benchmarks/bench_nvfp4_route_merge.py --output /path/to/results/route_merge.json
```

The operator benchmark uses a small Qwen model with BF16 experts to isolate eager dispatch
versus precompiled graph on the same shapes. Inputs/outputs and state hashes are
saved; reset/compile/capture are excluded from timing. It does not replace a real
checkpoint benchmark. Actual A/B artifacts are under
`result/qwen38_prefill_2026-10-01/` in the parent workspace: prior source, pinned
checkpoint revision, calibration, exact commands, prompts, raw logits/token IDs,
timing, graph audit and `REPORT.md`. Four pre-existing tests demanding bit equality
across different GEMM shapes also fail on the saved baseline; distinguish them
from same-shape graph/state correctness.

The route-merge benchmark uses random NVFP4 banks with default dimensions
H=2560, I=640 and top-k=10, at 1/39/64 tokens. It saves inputs, banks, outputs and
timing samples from warmed graphs. It requires byte equality with the unsplit
NVFP4 operator and reports the earlier separate-partial rounding error. It
isolates expert math/reduction; copy scheduling is measured by the server A/B.

## H100 NVFP4 measurements

The 2026-10-09 microbenchmark used an H100 80GB and a Xeon Platinum 8462Y+ with
16 CPU workers. GEMM rates below are general FP32/BF16 baselines, not NVFP4
throughput. The expert operator used synthetic Qwen-shaped NVFP4 banks, not
checkpoint weights.

| Measurement | CPU | H100 |
|---|---:|---:|
| GEMM FP32, 6144 x 6144 | 1.812 TFLOP/s | 51.06 TFLOP/s |
| GEMM BF16, 6144 x 6144 | 1.274 TFLOP/s | 800.54 TFLOP/s |
| RAM streaming read, 2 GiB | 181.0 GB/s | n/a |
| Pinned RAM -> VRAM | n/a | 55.32 GB/s |
| VRAM -> pinned RAM | n/a | 55.31 GB/s |

With H=2560, I=640, top-k=10 and 16 synthetic experts, warmed NVFP4 operator
latencies were:

| Query tokens | Unsplit | Hit/fetch with ordered reduction | Effective expert rate |
|---:|---:|---:|---:|
| 1 | 37.536 us | 50.304 us | 2.62 TFLOP/s |
| 39 | 527.232 us | 695.680 us | 7.27 TFLOP/s |
| 64 | 740.128 us | 769.824 us | 8.50 TFLOP/s |

One expert occupies 2,772,480 bytes (2.644 MiB) in the FreeToken NVFP4 bank
layout. At the measured pinned-copy rate, ten experts require about 0.501 ms
for a linear transfer; the measured gather graph took 0.560 ms for ten misses.
For the synthetic CPU/GPU calibration, the 10-miss CPU task took 2.221 ms and
its complete CPU branch graph took 3.773 ms, versus 0.0394 ms for GPU NVFP4
math. Those CPU timings are diagnostic only: CPU/GPU output relative L2 error
was about 0.99999, so they do not establish valid CPU NVFP4 throughput or a
usable split ratio.

These are microbenchmarks under a shared host load, not whole-model serving
results. Methodology and limitations are in
[the detailed H100 report](README_qwen_nvfp4_h100.md).

## Measured checkpoint result

On RTX A5000 24 GiB with 64 allowed CPU workers, Qwen NVFP4 revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`, 3072 cache slots and 8192 KV tokens:

| Repeat prompt | TTFT before | TTFT after | Decode before | Decode after |
|---|---:|---:|---:|---:|
| Math, 39 prompt tokens | 6.0070 s | 1.4144 s | 26.271 tok/s | 22.318 tok/s |
| Code, 28 prompt tokens | 6.0641 s | 1.2144 s | 25.072 tok/s | 29.203 tok/s |

Median of five repeats per topic, 16 greedy output tokens. Both prompts have
cached prefix length 0; repeated requests execute full prefill. All 12 saved
prefill logits match baseline bytes after conversion to FP32, and all delivered
token IDs match. All 12 prefills replay startup graphs, with no request-time
capture. Logical expert H2D drops from 63.457 GiB/request to 10.305/8.887 GiB.
The baseline snapshot includes earlier workspace optimizations, rather than
unmodified upstream. This combines selective expert copies and graph dispatch.
Decode can regress on a topic; the math median is 15.05% slower, while the code
median is 16.48% faster. Five short decode samples do not establish a universal gain.

Persistent allocated GPU bytes increase from 18.597 to 18.661 GiB; allocator
reservation increases from 18.723 to 19.795 GiB. Expert preload is 7.932 GiB in
both. `REPORT.md` includes raw artifact names, earlier separate-sum rounding
results, operator timings, calibration and test coverage. The final change has
29 passing dedicated tests and 75 passing BF16/MXFP4/FP8 shared-kernel tests.
Four broad-regression failures were also reproduced on the preserved baseline.
