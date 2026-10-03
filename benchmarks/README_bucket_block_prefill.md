# Bucket and layer-block Qwen prefill

The native Qwen text path has four modes selected by `--qwen-prefill-mode`:

- `exact`: the preserved per-length implementation for comparison.
- `bucket`: power-of-two token/request shapes with runtime logical lengths.
- `block`: whole-model blocks at one physical shape, including the padded tail.
- `layer`: native C++ block/layer scheduling, with full expert-layer double buffers.

`bucket` compiles 8,16,32,... up to `--moe-prefill-graph-max-tokens`, and request
counts 1,2,4,... up to the concurrency ceiling. Real tokens are packed first;
padding is excluded from KV/ring/index writes, expert fetch classification,
GDN recurrence endpoints, PLE history/ngram commit and tracked snapshots.
Padding is computed by dense operators. Token length, prefix and page/state
addresses are runtime metadata; ragged lengths do not create a new graph.

`layer` accepts a whole admitted text-prefill batch, then internally divides
each request into capped blocks. It embeds the inputs, finishes the block sweep
of one decoder layer, then proceeds to the next. C++ calls CUDA graphs without
Python dispatch per block/layer. The current expert bank remains in its protected
buffer; the next layer's bank copies on a separate stream. Two activation staging
windows pipeline H2D/D2H copies against compute, fenced by events before reuse.
Layer mode uses one configured physical block shape and pads the tail to it.
This keeps two graphs per layer rather than retaining every quantized GEMM
workspace for every tail bucket. Only final real token rows reach the LM head.
Decode retains the existing native scheduling and graphs.

Activation planes use two reusable GPU tensors when they fit. With
`--qwen-prefill-activation-device auto`, they use pinned RAM otherwise; `gpu` and
`cpu` force a placement for measurement. Their payload scales with the admitted
prompt, while GPU staging scales with the block cap. PLE disk rows are hashed
and read through the existing native store once for the admitted prefill, then
staged at the PLE layer. This initial implementation does not overlap that initial
disk fill with early-layer compute. Layer mode currently supports at most one PLE
table, matching the loaded Qwen checkpoint.

KV is separately paged and grows with actual context; the graph block cap does
not reserve KV for the whole prompt. Set a sufficient maximum sequence length
and KV token budget, including output. Model position limits still apply. Full
layer buffers require at least `2 * num_experts` cache slots; the random cache
preload remains active before requests. Layer-buffer owners are invalidated before
overwrite, and stable slots outside the two buffers remain available to decode.

Each graph has an isolated private allocator pool. Input/staging buffers are
shared only with event-protected lifetimes. Many layer/bucket graphs can increase
VRAM and initialization time: measure this before raising the block cap. The
runtime does not silently share graph temporaries across arbitrary replay order.
The QSA indexer retains its bounded score workspace and exact existing top-k;
it still scans visible old keys. No approximate window is introduced.

## Serve and test

```bash
ft serve --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --attention-backend qsa_sparse --ple-backend disk \
  --moe-strategy offload --quant-backend moe.nvfp4=triton \
  --moe-native-schedule --moe-prefill-gpu-miss-fraction 1 \
  --moe-gpu-miss-fraction 1 \
  --qwen-prefill-mode layer --moe-prefill-graph-max-tokens 128 \
  --qwen-prefill-activation-device auto \
  --moe-cache-size 1024 --moe-cache-init random --moe-cache-init-seed 42 \
  --max-running-requests 1 --cuda-graph-max-bs 1 \
  --max-seq-len-override 131072 --num-tokens 131072
python -m pytest tests/models/qwen4_exp/test_bucket_graph.py \
  tests/models/qwen4_exp/test_prefill_graph.py tests/models/qwen4_exp/test_ple_disk.py -q
python benchmarks/bench_bucket_block_prefill.py \
  --tokens 10000 100000 --block 256 --activation-device auto \
  --output /path/to/results/operator.json
```

The numeric cache/block choices above reproduce this workspace's measurements;
they are not fixed CPU/GPU architecture rules. Calibrate CPU split with the earlier
script and use the appropriate profile on another machine. GPU-only layer prefill
does not use the decode CPU miss profile as a multi-token model.

## Saved evidence and limits

`result/bucket_block_prefill_2026-10-02/` in the parent workspace retains the
baseline sources, graph/state equality tests, packed NVFP4 regression, long-input
operator tensors/timings/traces and real-checkpoint server scripts/logs/audits.
The operator model is a small four-layer Qwen with BF16 experts; distinguish it
from the full NVFP4 checkpoint. Graph-vs-eager comparisons on the same padded
shapes are strict byte comparisons. Different bucket sizes and FLA chunking can
change arithmetic; their full-model numerical comparisons are recorded separately.
Short synthetic results do not establish long-context answer quality.

The measured 10k checkpoint comparison of `bucket` versus `layer` produced the
same eight delivered tokens but different final logits (normalized max error
0.08893). `bucket` used a 16-row tail while `layer` padded it to 128, selecting
different dense NVFP4 GEMM paths. This is a possible source of the error, not a
verified attribution. A `block` reference keeps that physical shape
fixed to isolate execution order. The saved 10k fixed-shape comparison has
normalized max logit error 0.002564 and relative L2 error 0.001561, with the
same delivered tokens. It is a tolerance comparison, not full-model bit equality.

The earlier broad-regression PLE/QSA cross-shape equality failures were reproduced
on the prior baseline. They are retained and are not described as passing.
Multimodal encoder/block-attention batches retain eager fallback.

Layer mode currently blocks decode admission until that admitted prompt completes.
Its kernel dispatch is native, but request admission, descriptor construction,
initial PLE disk fill and the final LM-head dispatch are still host-side. CPU/RAM
activation mode pipelines bounded GPU staging; it still retains two full hidden
planes in RAM. KV is currently GPU-resident, so growing context requires enough
KV pages rather than an automatic RAM KV spill.

## Real checkpoint results

Same 10k input, 1024 expert cache slots, 131072 KV token capacity and a128-token
block cap: whole-model bucket TTFT270.262s versus layer-sweep TTFT28.321s
(9.54x in one cold sample). Logical expert payload drops1715.615GiB to63.457GiB.
Delivered eight-token text agrees. With the same physical128-row tail, normalized
max logit error is0.002564; cross-strategy logits are not bit exact. The layer10k
sample preceded the final expert-prefetch ordering/snapshot-stride corrections;
their strict graph/state tests are retained separately.

Final source also completed a cold100000-token input with0 prefix-cached tokens:
TTFT310.850s, audited prefill306.758s,8.918tok/s for8 decode outputs,63.457GiB
expert payload,3.818GiB pinned activation planes,16.551GiB peak allocated GPU
memory and20.881GiB allocator reservation. No request-time captures occurred.
The server had98 prefill graphs; initialization535.2s is excluded from TTFT.
This is a repetitive text benchmark, not a long-context quality evaluation.

52 dedicated tests passed,5 skipped,1 deselected. The earlier broad suite has
262 passes and the4 known baseline cross-shape failures. Full commands, source
hashes, prompts, raw logits and run limitations are in the saved `REPORT.md`.
