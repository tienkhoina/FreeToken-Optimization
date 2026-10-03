# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

**`calibrate_moe_schedule.py`** — compiled CPU branch latency and GPU graph-replay copy/math timing,
with a coarse integer CPU/GPU miss split recommendation. It uses all CPUs in process affinity by default.
See [native scheduling and calibration](README_native_moe_schedule.md) for commands, graph coverage,
numerical validation and the measured performance limits of the experimental runtime.

**`bench_decode_metadata.py`** — KV preparation overhead and preparation plus warmed attention
graph replay, with saved random inputs and output hashes. See [decode metadata benchmark](README_decode_metadata.md)
for A/B commands, stream ordering and timing boundaries.

**`bench_runtime_batch.py`** — combined input/KV/attention-graph/sampling overhead after the
runtime preparation bundle. See [runtime batch benchmark](README_runtime_batch.md) for the
native build, saved tensor proof, memory reporting and full-model comparison protocol.

**`bench_qwen_prefill_ops.py`** — same-shape QSA/GDN/PLE prefill dispatch versus warmed
CUDA graph, with tensor/state proof. See [Qwen prefill](README_qwen_prefill.md) for
calibration, startup coverage, disk PLE and real-checkpoint A/B.

**`bench_nvfp4_route_merge.py`** — saved random NVFP4 expert inputs, banks and
outputs, comparing separate BF16 partial sums with one ordered FP32 route sum
after warmed graph replay. The single sum must match the unsplit operator bytes.

**`bench_bucket_block_prefill.py`** — 10k/100k input scaling on a small Qwen
QSA/GDN/PLE model, comparing block-major bucket graphs with native layer sweeps.
Inputs, logits, final states, graph counts and memory are retained. See
[bucket/block prefill](README_bucket_block_prefill.md) for flags and checkpoint evidence.
