# Native MoE scheduling and calibration

This experimental path keeps the random expert preload and schedules each MoE layer in three concurrent branches: GPU cache hits, GPU fetched misses, and CPU misses. A CUDA/C++ kernel classifies routes, chooses LRU slots, writes copy descriptors and assigns branches. CPU work runs in the compiled C++ executor. The captured graph owns the stream/event dependencies; replay does not run per-layer Python scheduling.

The default path remains unchanged. Enable the experiment with `--moe-native-schedule`. CPU/GPU floating point differences are accepted, but must be measured. Packed weight copies remain exact bytes.

## Calibrate on another machine

Use the CUDA toolkit matching the installed PyTorch build. Rebuild the extension after changing the CPU executor:

```bash
python setup.py build_ext --inplace
python benchmarks/calibrate_moe_schedule.py \
  --workload gpt-oss-120b --format mxfp4_triton \
  --model /path/to/gpt-oss-120b \
  --threads 0 --output /path/to/results/moe_profile.json
```

`--threads 0` uses every logical CPU available in process affinity. It does not reserve a worker for the coordinator. The executor still detects the CPU ISA. The actual checkpoint reader currently supports GPT-OSS MXFP4. Without `--model`, the script creates finite synthetic banks in the production layout. For other implemented CPU formats:

```bash
python benchmarks/calibrate_moe_schedule.py \
  --workload qwen3-30b --format bf16 --threads 0 \
  --output /path/to/results/bf16_profile.json
python benchmarks/calibrate_moe_schedule.py \
  --workload dsv4 --format ds_fp4 --threads 0 \
  --output /path/to/results/ds_fp4_profile.json
```

NVFP4 is also supported by the calibration script. FP8-block and FP16 have no corresponding CPU operator in the executor; the native runtime keeps their misses on the GPU. Do not interpret an untested donor backend as having passed its compute benchmark.

For NVFP4, this script times the repository's production Triton wide-load decode operator and grouped prefill operator. The production donor/backend selected by the checkpoint may differ, so its integrated timings must be checked before applying a calibration profile. CPU format support does not imply every GPU donor uses the same bank layout.

All GPU measurements use CUDA graph replay after kernel compilation and warmup. CPU has two columns: compiled C++ task time including dispatch/barriers, and `cpu_graph_ms` for the complete captured D2H/handshake/CPU/H2D branch. The split recommendation uses `cpu_graph_ms`, because CPU math time alone can miss substantial synchronization cost. Neither GPU JIT nor eager GPU math is included in timed samples. Expert IDs rotate over the working set to reduce unrealistic CPU LLC reuse. Use enough `--experts` to exceed LLC; the default is 32, capped at the workload's expert count.

The output contains per-expert-count medians and all samples, hardware/ISA/worker information, `recommend_fetch_counts`, and a coarse fraction. For example, `[0, 1, 2, 2, 3]` means that 0/1/2/3/4 distinct missing experts send 0/1/2/2/3 experts to the GPU; the remainder goes to the CPU. The recommendation minimizes the rough estimate `max(CPU branch graph time, copy time + GPU math time)`. It does not fully account for shared RAM contention, GPU copy/kernel contention or cache reuse across future tokens. It is a starting point, not a proven optimum.

By default policy estimates use CPU graph p90 and require at least 15% predicted gain before offloading a miss to CPU (`--cpu-timing-percentile`, `--minimum-split-gain`). This avoids relying on a lucky low median when the full worker pool has long delays. It can recommend fetching all small misses on GPU on a noisy machine; the CPU pool and native branch remain available. `coarse_gpu_fraction` is only a rough diagnostic, not a replacement for the integer lookup table.

For a short prefill estimate with multiple tokens sharing an expert, run one additional calibration, not a deep ratio search:

```bash
python benchmarks/calibrate_moe_schedule.py \
  --workload gpt-oss-120b --format mxfp4_triton \
  --model /path/to/gpt-oss-120b --threads 0 --tokens 16 \
  --output /path/to/results/prefill16_profile.json
```

Keep the single-token profile for `--moe-schedule-profile`. Use the multi-token result to choose `--moe-prefill-gpu-miss-fraction` manually; its shared-expert pattern is a rough estimate of prefill reuse, not an actual prompt routing distribution.

Large variance or non-monotonic CPU timings mean the small task is dominated by worker scheduling or interference. Keep the raw samples and validate the proposed split under graph replay of the integrated operator. A bandwidth ratio alone is insufficient, especially with many workers and one expert. Do not reuse one machine's profile on another machine; rerun the script there.

## Serve with the measured profile

```bash
ft serve --model /path/to/gpt-oss-120b \
  --attention-backend triton --moe-strategy offload \
  --moe-cache-auto --kv-reserve-tokens 8192 \
  --moe-cache-init random --moe-cache-init-seed 42 \
  --moe-native-schedule --moe-cpu-threads 0 \
  --moe-schedule-profile /path/to/results/moe_profile.json \
  --moe-gpu-miss-fraction 0.8 \
  --moe-prefill-gpu-miss-fraction 1.0 \
  --moe-prefill-graph-max-tokens 128
```

Random preload runs after startup graph capture/warmup and before ready. The preload RNG is independent of model sampling. With native scheduling, prefill uses the shared slot cache directly instead of streaming full expert layers into the two borrowed buffers. It therefore does not invalidate those buffers on each prefill.

Prefill fetches choose victims only in the first two expert-layer regions. Those regions act as a temporary working set; resident owners outside them remain available for subsequent decode. Active hit slots within the temporary region are protected during the layer as well. Decode can choose victims across the whole cache. This prevents a prompt with many one-off experts from replacing the decode working set merely because its selective prefill visited them.

The profile's small-miss table applies when the actual query has one token, including a one-token prefix-cache extension. Larger prefill queries use `--moe-prefill-gpu-miss-fraction` if provided, otherwise `--moe-gpu-miss-fraction`, rounded to a distinct-expert count. A large prefill needs separate validation: CPU cost depends on tokens per expert, while one GPU weight copy can serve many tokens. The `1.0` example above is appropriate only when calibration shows GPU copy+compute beats CPU for those multi-token experts; it is not a universal recommendation. The fraction and lookup live in GPU tensors, so updating them at an idle replay boundary does not require recompilation or capture. The runtime does not continuously search for a ratio.

## Graphs, startup and KV cache

Decode uses the existing batch-size CUDA graphs. Text prefill is captured into power-of-two token buckets and exact request-count buckets at startup. The configured prefill graph cap also caps the scheduler's chunk token budget, so longer prompts are chunked into covered buckets instead of quietly using eager for large chunks. Padding routes are masked by a runtime valid-token count. Attention metadata and page indices are copied into persistent buffers before replay; a query with prefix hits still uses its actual prefix lengths.

All branch kernels and CPU task buffers are warmed before capture. Graph capture temporarily routes dummy misses to the GPU to avoid expensive CPU computation of repeated dummy prompts; the runtime fraction is restored before serving. CPU submit/wait nodes remain present in the graph and read the current routing on replay. Stable pointers and fixed graph topology allow the hit/miss counts to change between tokens.

When the native count assigns no expert to CPU, a small native doorbell sets its completion flag directly and does not wake the worker pool. The merge masks the stale CPU result. If stream-memory handshake is unavailable, the existing host-callback fallback remains functional but has different overhead; calibrate that configuration again.

The current whole-prefill graph implementation requires Triton paged attention without linear-state or mrope models. Multimodal encoder/block-attention batches are not captured by this implementation. This limitation is explicit; do not describe it as full graph support for every architecture. Cache movement and native route classification are bank based, but each model/backend still needs validation.

KV allocation and prefix-cache management stay in the existing scheduler. All expert branches join on the engine stream before the next layer; copy misses cannot evict active hit slots. Each GPU branch has its own input copy because some existing expert operators overwrite input. An extra immutable zero row per bank handles inactive GPU routes without reading a cache slot concurrently being evicted.

The MXFP4 kernels skip that inactive row and explicitly write zero outputs/partials, so inactive routes do not execute their dot products. Existing operator calls keep the original math when no inactive row is specified. Other formats retain zero-row masking and need their own inactive-work optimization; no claim is made that every format's math kernel has the same speed improvement.

More buckets increase startup time and persistent graph memory. Budget graph/workspace memory together with expert and KV cache. The cap is a workload choice, not a GPU architecture constant. If changing pool sizes, the engine re-captures against the new tensors and then repeats the configured random preload.

The extra zero expert row is charged in the automatic cache budget and counts against donor slot limits. Explicit cache sizes must leave one addressable row for it. Do not choose a donor's maximum slot count blindly when enabling native scheduling.

The present native eviction implementation uses LRU. It does not currently combine with the earlier experimental `adaptive_hot` eviction mode. Random initialization is independent and is preserved. This separation is an explicit current implementation limit, not a requirement of CUDA graphs.

## Validation and benchmark protocol

```bash
python -m pytest tests/moe/test_native_schedule.py tests/moe/test_cpu_moe.py -q
```

Measure latency after server ready; report startup and preload cost separately. Verify graph use for both prefill and decode, changed routing on replay, zero misses, all misses, the CPU-only miss assignment, prefills with padding, chunked prompts, prefix KV hits and cache rebuild. Compare operator output and logits with a stated numerical tolerance. Free-running generated token strings can diverge under small floating point differences, so also compare logits on the same input/routing and record top-token agreement.

Preserve the old source/config/results before edits. Keep model, input tensors, request order, cache size, KV pages, random seed and CPU count the same for A/B. A CUPTI trace can demonstrate concurrency but adds overhead; exclude profiled requests from latency medians. The experiment artifacts in the current workspace are under `result/native_moe_schedule_2026-10-01/`; copy results separately when moving machines.

## Measured limits on the development machine

RTX A5000 24 GiB, GPT-OSS-120B, 64 allowed logical CPUs/64 worker pool, LRU1241 expert slots, KV8192 pages, batch1, random preload seed42. Five repeat requests per topic, greedy32 output tokens. The same saved baseline is used for the successive experimental variants; these numbers are not medians across multiple server restarts.

| Repeat group | Legacy prefill/first text | Native prefill/first text | Legacy decode | Native decode |
|---|---:|---:|---:|---:|
| Math | 4.461 s / 4.868 s | 0.143 s / 0.437 s | 20.47 tok/s | 18.42 tok/s |
| Coding | 4.468 s / 4.891 s | 0.120 s / 0.393 s | 26.84 tok/s | 22.72 tok/s |

Warm prefill H2D expert bytes fell from44.916GiB to1.356GiB for math and1.122GiB for coding. All15 recorded native prefill requests used graph replay. The native route schedule eliminated the unnecessary full-layer stream for short queries. Decode still regressed10-15% versus the legacy path, despite inactive-route skipping and protecting resident owners from prefill eviction. This is why the new path stays opt-in; a whole-model speedup from splitting work onto CPU has not been demonstrated here.

The full-branch CPU calibration was noisy with all64 workers. The conservative p90+15%gain table recommended fetching all0..4 distinct decode misses on GPU. The final profile therefore assigned no CPU expert in this model benchmark, while keeping the64-worker native CPU branch ready. Explicit CPU/GPU splits were exercised under CUDA graph in operator tests and earlier model variants, but those variants decoded more slowly on this machine. Do not claim the final speedup is due to CPU compute.

For the final15 prefill logit comparisons on identical prompts: maximum normalized max error0.03891, maximum relative L2 error0.07271, first argmax15/15 equal. Free-running32-token responses matched12/15 pairs. The CPU/GPU split operators have their own numerical tests. These figures are observed errors, not a universally safe accuracy bound; evaluate longer prompts and application tasks before adopting a profile. The final regression suite passed122tests; one existing FlashInfer-dependent config test was deselected after verifying the missing dependency on the earlier baseline.

Small-engine checks also exercised graph prefill with2requests, prefix lengths and22valid tokens in a32token bucket. Replay matched eager execution of the same padded native bucket exactly; the padded bucket differed from unpadded model output by normalized max0.07292 on those random dummy weights. Padding can change selected GEMM/reduction shapes and floating point results even with correct KV metadata. Keep this separate from graph replay correctness and validate batched model accuracy; the120B performance table above is batch1.

Saved artifacts include `resident_server_ab.json`, `resident_summary.json`, `FINAL_REPORT.md`, raw logits, token IDs, per-request graph/H2D/route stats and CUPTI traces. Earlier unsuccessful profiles are preserved as separate results. Startup preload and graph capture are outside request latency; do not hide those costs when comparing time from process launch to first output.
