# Local optimization experiments for FreeToken

Updated **2026-10-03**. This work builds on
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken), checked out at
`0d652e73a452d014ac5441a15baa75348e9fcb0a`. The upstream README, guidance and
technical documentation are preserved in [README.md](README.md) and [docs/](docs/).

**FreeToken Authors and the project's contributors created the original engine,
serving architecture, backends and inherited code.** The changes described here
were developed and tested in a local workspace under user direction, with Codex
assistance. This document identifies the additions and their measured results;
it does not claim authorship of the underlying platform or inherited algorithms.
This is a separate experimental repository, not an official FlashML-org release.
The optimization repository is
[tienkhoina/FreeToken-Optimization](https://github.com/tienkhoina/FreeToken-Optimization);
these changes have not been accepted upstream.

## Provenance and attribution

FreeToken's license is retained in [LICENSE](LICENSE), including the notice
`Copyright 2026 FreeToken Authors`. Existing copyright, SPDX and source references
continue to identify the origin of each component. These notes supplement those
notices; they do not replace them. [NOTICE](NOTICE) records this repository's
relationship to upstream and the scope of the local modifications.

| Inherited component | Credited source | Additions in this workspace |
|---|---|---|
| Loader, scheduler, expert offload, caches, APIs and model backends | [FreeToken](README.md) | Operator, transfer and metadata optimizations, plus experimental execution modes on the existing engine |
| Gated DeltaNet / Flash Linear Attention | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention); headers credit Songlin Yang and Yu Zhang in [chunk.py](python/freetoken/kernel/fla/chunk.py) | Graph metadata, chunk plans, padding masks and state snapshots; the inherited recurrence algorithm is retained |
| QSA sparse attention | [vLLM](https://github.com/vllm-project/vllm); SPDX and attribution in [score.py](python/freetoken/kernel/triton/qsa/score.py) | Runtime bucket metadata and protection against padded KV/ring writes |
| Causal convolution | sgl_kernel, credited in [causal_conv1d.py](python/freetoken/kernel/causal_conv1d.py) | Integration with the graph/state path, retaining attribution for the borrowed kernel |
| Selected GGUF kernels | vLLM and llama.cpp, cited in [mmq.cuh](python/freetoken/kernel/csrc/gguf/mmq.cuh) | Transfer/operator experiments for the corresponding layouts; execution remains within FreeToken |
| Checkpoints and quantization | Model authors, checkpoint publishers, PyTorch/Triton/CUDA and backend libraries | Execution optimizations; model weights were not created by this workspace |
| Chat interface | [Open WebUI](https://github.com/open-webui/open-webui) | Connection configuration and integration testing against FreeToken's API |

Offload, stream/event overlap, paged KV caching, inline dequantization and
quantized backends already have foundations in the upstream repository. This
workspace contributes the specific changes and experiments documented below.
Preserve source attribution, licenses and modification history when sharing them.

## Changes in this workspace

| Area | Local change | Implementation and documentation |
|---|---|---|
| CUDA/Triton MoE | Optimize FP4 loads/decoding, remove intermediate input/ID tensors, and fuse reduction/activation on validated paths while preserving rounding order where bit equality is required | [mxfp4_moe.py](python/freetoken/kernel/triton/mxfp4_moe.py), [moe_impl.py](python/freetoken/kernel/moe_impl.py) |
| NVFP4 | Tune decode geometry by tensor shape; retain per-route hit/fetch outputs and perform one ordered FP32 reduction, avoiding the addition of two separately rounded BF16 partial sums | [fused_nvfp4.py](python/freetoken/moe/fused_nvfp4.py), [route merge benchmark](benchmarks/bench_nvfp4_route_merge.py) |
| Full-bank transfers | Prepare ByteCopyPlan descriptors once and submit multiple memcpy operations through one C++ call, replacing a Python tensor-copy dispatch loop during prefill | [copy_plan.py](python/freetoken/kernel/copy_plan.py), [copy_plan.cuh](python/freetoken/kernel/csrc/jit/copy_plan.cuh) |
| Indexed transfers | Optimize raw-byte copying and add optional block allocation proportional to bank size; results include both gains and regressions when copying overlaps computation | [fast_index_copy_scheduled.cuh](python/freetoken/kernel/csrc/jit/fast_index_copy_scheduled.cuh), [benchmark](benchmarks/bench_copy_kernel_schedule.py) |
| Hot experts | Add adaptive_hot, which learns from online routing, protects part of the cache and decays scores without changing router decisions | [adaptive_hot.py](python/freetoken/moe/adaptive_hot.py), [adaptive_hot_cache.cuh](python/freetoken/kernel/csrc/jit/adaptive_hot_cache.cuh) |
| Expert-cache initialization | Add --moe-cache-init random after warmup/capture and before the first request; the preload seed is independent of the sampling RNG | [offload_cache.py](python/freetoken/moe/offload_cache.py), [engine.py](python/freetoken/engine/engine.py) |
| Native CPU/GPU scheduling | Classify hits/misses, build transfer descriptors and select CPU/GPU branches in C++/CUDA; graphs retain stream/event dependencies and calibration determines GPU fetch counts | [native_schedule.py](python/freetoken/moe/native_schedule.py), [calibration guide](benchmarks/README_native_moe_schedule.md) |
| Inputs, KV metadata and sampling | Use pinned descriptors and reusable buffers, prepare positions/pages on the GPU, reduce Python/ATen dispatch and allocations, and provide int32 greedy argmax | [runtime batch preparation](benchmarks/README_runtime_batch.md), [decode metadata](benchmarks/README_decode_metadata.md) |
| Qwen prefill | Capture QSA/GDN/PLE paths in exact, bucket, block and layer modes; logical lengths/masks protect KV, recurrence, history and snapshot state | [qwen_prefill.py](python/freetoken/engine/qwen_prefill.py), [bucket/block guide](benchmarks/README_bucket_block_prefill.md) |
| Layer prefill | Dispatch layer/block graphs from C++, retain the current expert bank across blocks and prefetch the next bank; place activation planes in GPU or pinned RAM according to the memory budget | [qwen_layer_prefill.py](python/freetoken/engine/qwen_layer_prefill.py), [layer_prefill.cuh](python/freetoken/kernel/csrc/jit/layer_prefill.cuh) |
| Local serving | Deploy a FreeToken service with four fixed buckets and connect Open WebUI through the OpenAI-compatible API; retain model/graphs between requests and disable per-token benchmark auditing | [local deployment](../deploy/qwen-webui/README.md) |

Transfer paths were tested with BF16, FP16, FP8 block, MXFP4, NVFP4, DS-FP4 and
Q4_0. This does not mean every arithmetic kernel in every format was optimized
or validated on a full model. NVFP4 donor/Marlin/b12x layouts have separate
byte-copy checks; correct movement does not establish compute-backend correctness.

The new choices depend on shape, dtype, layout and memory budget. Hardware names
in reports identify the measurement machine. Cache sizes, buckets and CPU/GPU
ratios require calibration on other machines. Untested devices and backends,
including ROCm paths, are not presented as validated.

## Validation method

Operator benchmarks retain input tensors, reference outputs, modified outputs,
timing samples and source hashes. Copy, metadata and arithmetic paths requiring
bit equality use raw-byte comparisons, including canaries and persistent state
where applicable. Compilation, warmup and state restoration are outside kernel
timing. Host enqueue latency, CUDA-event spans, allocator peaks and API throughput
are reported separately.

Full-model benchmarks retain commands, checkpoint IDs, prompts, output token IDs,
logits, cache/KV settings, graph configuration and repeat conditions. Many
baselines are **workspace snapshots containing earlier optimizations**, rather
than unmodified upstream. Speedup ratios from different rounds cannot be
multiplied into an overall speedup claim.

Changing buckets, GEMM shapes or recurrence segmentation can change rounding.
Graph replay and eager execution at the same physical shape have separate
byte/state checks; comparisons across shapes report numerical differences.
Matching generated tokens alone does not establish bit-identical logits or
forward passes. CPU branches use the accepted floating-point tolerance and
should be enabled when calibration demonstrates a benefit.

The main measurement machine has an RTX A5000 with 24 GiB VRAM, approximately
125.9 GiB RAM, 64 logical CPUs in process affinity, PyTorch 2.11/CUDA 13 and
Triton 3.6. These measurements do not establish performance on every device.

## Recorded results

### Kernels, transfers and runtime overhead

| Measurement | Before → after | Conditions and interpretation |
|---|---|---|
| MXFP4 expert decode | 0.148096 → 0.134400 ms | M=1, H=I=2,880, top-k=4; warmed graphs, 9.25% lower operator latency |
| MXFP4 expert prefill | 1.661440 → 1.570800 ms | M=128, same H/I/top-k; 5.46% lower operator latency |
| MoE with four forced expert misses | 4.430848 → 4.416512 ms | RAM→VRAM copies dominate; pipeline improvement is approximately 0.32% |
| MXFP4 full-bank transfer enqueue | 73.532 → 36.605 µs | ByteCopyPlan reduces host dispatch; bulk transfer time is essentially unchanged |
| Synthetic runtime decode | 1.1746 → 0.5067 ms | Batch 1, context capacity 8,192; input/metadata/attention graph/sampling, without MLP/MoE arithmetic |

Local evidence: [MXFP4 kernels](../result/cuda_moe_bitexact_2026-09-30/REPORT.md),
[copy plans](../result/copy_overlap_2026-09-30/REPORT.md),
[copy with computation](../result/copy_overlap_2026-09-30/SCHEDULE_REPORT.md),
[runtime bundle](../result/decode_runtime_bundle_2026-10-01/REPORT.md).

The MXFP4 round retained 389 comparisons across 39 cases with no differing bytes
in those cases. Weighted indexed copying helped some NVFP4/Q4_0 cases but slowed
BF16/FP16 cases, so it remains optional. The runtime bundle substantially reduced
synthetic preparation overhead, while full GPT-OSS throughput changed by only
approximately 0.34–0.37%, which does not demonstrate a material token-rate gain.

### Qwen: short prefill and long inputs

The short Qwen round used 3,072 expert slots, an 8,192-token KV budget, cap 64,
16 greedy output tokens and the median of five repeat requests per topic.
Prefill graphs and selective native loading were enabled together:

| Prompt | TTFT before → after | Decode before → after |
|---|---:|---:|
| Math | 6.007 → 1.414 s | 26.27 → 22.32 tok/s |
| Code | 6.064 → 1.214 s | 25.07 → 29.20 tok/s |

All 12 prefill-logit pairs and generated token sequences matched in that round.
Decode improved or regressed depending on the workload; this is not a uniform
decode improvement. [Local report](../result/qwen38_prefill_2026-10-01/REPORT.md).

A separate long-input round used 1,024 expert slots, a 131,072-token KV budget,
block cap 128 and eight output tokens. At 10k input tokens, whole-model bucket
passes had TTFT 270.26 s versus 28.32 s for the layer sweep. That 10k layer result
preceded the final fixes. The final source completed 100k new tokens with zero
prefix hits, TTFT 310.85 s and 3.818 GiB of activation planes in pinned RAM.
There is no cold 100k full-model A/B from which to calculate a speedup.
[Conditions and limits](../result/bucket_block_prefill_2026-10-02/SPEED_REPORT_VI.md).

The later 10k round used 2,048 expert slots, KV capacity 16,384 and 32 output
tokens. Among completed configurations, full-layer eager prefill with chunk
8,192 was fastest: warm TTFT **14.39 s**. Full-layer chunk 128 took 457.64 s;
selective eager chunk 128 took 209.85 s. Bucket/layer configurations did not
finish in that matrix. The earlier 9.54× layer result therefore compares with
the earlier bucket-cap-128 configuration, not with default chunk 8,192.
[Published matrix](../result/qwen_10k_matrix_2026-10-02/REPORT.md).

### Four buckets: 128, 512, 2,048 and 8,192

These measurements used 2,048 expert slots, KV/context capacity 16,384, batch 1
and one request per prompt generating 32 tokens. KV/prefix state and expert
cache were reset to the same random initialization before each request, outside
TTFT timing:

| Input tokens | Physical bucket | TTFT | Prefill | Decode |
|---:|---:|---:|---:|---:|
| 32 | 128 | 1.68 s | 19.70 tok/s | 14.14 tok/s |
| 128 | 128 | 3.67 s | 42.02 tok/s | 16.45 tok/s |
| 1,024 | 2,048 | 6.59 s | 158.22 tok/s | 10.28 tok/s |
| 4,096 | 8,192 | 11.22 s | 374.67 tok/s | 10.17 tok/s |

All four prefill graphs and one decode graph were retained, with no additional
capture during measured requests. These are individual samples, not medians or
same-shape graph/eager A/B results.
[Three prompts](../result/qwen_bucket_128_512_2048_8192_2026-10-02/REPORT.md),
[32-token prompt](../result/qwen_bucket_128_512_2048_8192_2026-10-02/REPORT_32.md).

### Expert-cache capacity affects decode

The old baseline source and script were rerun with expert cache retained between
requests: KV budget 8,192, context 2,048, eager prefill cap 64, one decode graph
and native scheduling disabled. The inference setting changed was cache size,
2,048 → 4,096 slots:

| Prompt | Decode with 2,048 slots | Decode with 4,096 slots | Measurement |
|---|---:|---:|---|
| Math | 15.34 tok/s | 27.25 tok/s | Median of five repeats, 16 output tokens |
| Code | 21.18 tok/s | 26.90 tok/s | Median of five repeats, 16 output tokens |
| Identical raw prompt, 32 input / 32 output | 14.84 tok/s | 20.53 tok/s | One sample after the same math/code request sequence |

Token IDs and prefill logits matched across all 13 compared requests. The servers
started separately, so kernel/page-cache warmth and measurement timing may still
differ. Increasing slots changes the VRAM budget; it is not a kernel speedup.
Expert-cache warmth is also distinct from prefix/KV reuse: these short prompts
all had zero prefix hits.
[4,096-slot report](../result/qwen_old_script_cache4096_2026-10-03/REPORT.md),
[identical 32-token prompt](../result/qwen_old_script_cache4096_2026-10-03/REPORT_32.md).

## Limits when interpreting the results

- Random preload does not know which experts will become hot. Routing updates
  the cache during a request; reset/rebuild discards that history.
- adaptive_hot helps some request groups and regresses on some topic changes;
  LRU remains available. [Hot-expert report](../result/adaptive_hot_2026-09-30/REPORT.md).
- The next layer's banks are known and can be prepared early. Its exact expert
  selection must wait for the router; experiments do not use future expert IDs.
- GPU hits may execute while misses copy or the CPU works, but PCIe, RAM and
  data dependencies still limit throughput. Qwen calibration on this machine
  selects the GPU for every decode miss; having 64 CPU workers ready does not
  mean they perform expert arithmetic in the deployment.
- Hit-D2D helped one GPT-OSS round, whereas Qwen 10k cap 128 copied fewer bytes
  but had higher TTFT. [GPT-OSS hit-D2D](../result/prefill_hit_d2d_2026-10-01/REPORT.md).
- Layer prefill blocks decode admission until the admitted prompt finishes.
  State and KV still have host-side management. Metadata optimizations neither
  move the entire scheduler to native code nor add automatic RAM KV spill.
- Broad tests include cross-shape equality failures reproduced on the baseline;
  test results and skipped cases are retained in each round's local report.
- Other checkpoints, multimodal inputs, backends and concurrent requests require
  separate validation. Context capacity is not the number of tokens always attended.

Compiled calibration for one real Qwen NVFP4 expert recorded a 5.119 ms CPU task,
0.281 ms H2D transfer and 0.0798 ms GPU arithmetic on the measurement machine.
Complete CPU-branch and dispatch timings are also retained; CPU/GPU splits
should not be chosen from multiplication time alone.
[Calibration evidence](../result/qwen38_prefill_2026-10-01/REPORT.md).

## Validated local deployment

The [local deployment](../deploy/qwen-webui/README.md) uses the current source,
2,048 slots, LRU, random preload seed 42, native GPU miss fraction 1, QSA sparse,
disk PLE, NVFP4 Triton and KV/context capacity 16,384. FreeToken runs as a systemd
user service and Open WebUI runs in Docker. A browser integration test signed in,
selected the model and received a chat response. This validates serving integration.

The four fixed shapes are selected by the
[deployment hook](../deploy/qwen-webui/instrument/sitecustomize.py). The normal
engine bucket mode creates power-of-two shapes from 8 through the cap; setting
--moe-prefill-graph-max-tokens 8192 alone does not select exactly these four x4
shapes. Runtime reuses graphs by bucket and masks logical lengths. Padding still
requires computation.

The deployment sets max-running-requests=1 and captures decode batch 1. It
accepts multiple HTTP requests and queues them; this configuration does not
execute multiple requests concurrently. Higher concurrency needs a revised
hook, larger batch graphs and new VRAM/throughput measurements.

```bash
# From FreeToken; requires the installed deployment in the same local workspace.
systemctl --user status freetoken-qwen.service
cd ../deploy/qwen-webui
docker compose ps
curl http://127.0.0.1:8000/health
```

Credentials remain in private deployment files. Passwords, tokens, browser
sessions and chat data are excluded from the README and published source.

## Reproducing measurements on another machine

Choose cache capacity after reserving KV, state and peak graph/workspace memory.
Keep the checkpoint, input IDs, output length, cache state, sampling and graph
settings identical for A/B comparisons. Report cold requests, warm repeats and
startup separately. Do not compare a warm median with a cache-reset sample
without accounting for the different starting state.

```bash
# From the repository; the CUDA toolkit major must match PyTorch's CUDA major.
python benchmarks/calibrate_moe_schedule.py \
  --workload qwen3.8-flash-next --format nvfp4 --threads 0 \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --output /path/to/results/moe_profile.json
```

Select a supported workload and shapes corresponding to the checkpoint; consult
--help and the [calibration guide](benchmarks/README_native_moe_schedule.md).
Compiled CPU tasks and warmed GPU graphs are measured separately. Synthetic
samples do not replace real-model routing/transfer measurements, and another
machine needs its own profile.

The [kernel](benchmarks/bench_moe_bitexact.py),
[copy](benchmarks/bench_copy_overlap.py),
[runtime](benchmarks/bench_runtime_batch.py) and
[Qwen graph](benchmarks/bench_qwen_prefill_ops.py) benchmarks measure individual
components. Publish only the coverage actually run and retain tensors, source
hashes, logs and original samples.

## Local evidence and publication scope

[result/](../result/) retains reports, before/after snapshots, test XML, raw
tensors, logits, token IDs, commands and traces. Paths under ../result/ and
../deploy/ refer to the original experiment workspace. They are not part of
upstream or this source publication: local tests, result data, checkpoints and
Open WebUI deployment/account data are excluded by .gitignore.

Links to these artifacts work in that workspace. A standalone clone contains
the published source and documentation; it does not include those local files.
Share appropriate artifacts separately or replace their paths with public links
when distributing an independently reproducible report. Preserve attribution
and the upstream authors' documentation.

These notes summarize existing evidence; no new benchmark was run for this
documentation update. The original upstream README, [LICENSE](LICENSE),
[CONTRIBUTING.md](CONTRIBUTING.md) and other existing guides remain unchanged.
