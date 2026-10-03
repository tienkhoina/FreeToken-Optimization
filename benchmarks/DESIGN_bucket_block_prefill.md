# Bucket and block prefill for long contexts

Original design, 2026-10-02. The user accepts computing padded rows to improve
speed. The first bucket/block implementation is now described in
[README_bucket_block_prefill.md](README_bucket_block_prefill.md), with evidence
under `result/bucket_block_prefill_2026-10-02/` in the parent workspace. Layer
mode uses one physical block shape and two staging parities to bound graph
workspace; the full token-bucket set remains in whole-model bucket mode.
Existing exact-length results remain at `result/qwen38_prefill_2026-10-01/`.

## Three independent sizes

- N: logical prompt/context length, plus the requested generation reserve.
- C: tokens admitted to a forward block. Choose a maximum from measured VRAM and
  throughput, for example 512, 1024, 2048 or 4096; these are tuning candidates.
- P: physical KV page size. Follow the model/backend; QSA currently requires 64.

Compile/capture buckets `8,16,32,64,128,256,512,1024,2048` if C_max is 2048.
Select the smallest bucket at least as large as the current block's real rows.
Do not create a 100000-row activation graph or capture every query/prefix length.
Context length, prefix length, block offsets and page IDs remain runtime data.

For C_max=2048:

| Prompt | Full blocks | Last block | Last bucket | Total computed rows |
|---|---:|---:|---:|---:|
| 10000 | 4 x 2048 | 1808 | 2048 | 10240 |
| 100000 | 48 x 2048 | 1696 | 2048 | 100352 |

Only the tail is padded for an uninterrupted single request. Aggregate ragged
batches can have more padding. The block cap does not cap total context length.
The pool still needs enough real KV pages to retain the admitted contexts.

## Fixed graph inputs and logical metadata

Graph key: `(execution mode, layer or model, token bucket, request bucket,
staging-buffer parity if used)`. No per-request length tuple or prefix length.
Batch-size buckets can be 1,2,4,... up to the configured concurrency limit.

Input activations are `[token_bucket, hidden_width]` in fixed staging buffers.
Real request rows are packed first; padding occupies the tail. Fixed-address
metadata contains:

```text
valid_tokens                 scalar on device
valid_requests               scalar on device
cu_seqlens[request_bucket+1]  true packed request starts/ends
prefix_lens[request_bucket]   true context prefix lengths
state_slots, ring_slots       valid request state addressing
token_to_req[token_bucket]    -1 for inactive rows
positions[token_bucket]       true absolute positions; safe dummy positions for padding
out_loc[token_bucket]         real KV addresses plus masked inactive rows
last_indices[request_bucket] true last-token indices
chunk_plan[fixed_capacity]    (request, local chunk), with invalid entries masked
track_plan[request_bucket]    destinations/boundaries plus an enabled mask
block_table[request_bucket, maximum_pages]
```

A native CUDA preparation kernel rewrites metadata in place. FLA launch-plan
capacity covers `ceil(token_bucket/64) + request_bucket` chunks, with equivalent
capacities for its 16/32-token internal work. Kernel grids use capacity, while
true lengths and validity determine reads, writes and recurrence endpoints.
No D2H reads or Python shape decisions are needed inside replay.

## Padding semantics

Dense GEMM/norm/activation can compute padded rows. Stateful operations must
exclude them from logical progress:

| Operation | Required change |
|---|---|
| Router and expert movement | Compute router logits at bucket size if useful; mask padding before unique expert classification and loading. |
| Expert GEMM/reduction | Fixed launch capacity; inactive routes cannot affect a real token. Preserve route-order FP32 reduction. |
| QSA KV, compressed keys, ring, mrope positions | Mask inactive writes. Padding must not close an index group or overwrite a pending ring. |
| GDN recurrence | cu_seqlens ends at real rows. Final state is after the final real token. Zero-valued padding alone is insufficient because decay remains active. |
| GDN convolution | Launch at bucket capacity; update history from the real endpoint, without advancing it through dummy inputs. |
| PLE convolution/ngram | Generate indices on device from true lengths. Native bounded convolution replaces the current host-length-specific packed F.conv1d plan. Commit context at the true endpoint. |
| PLE disk/UVA lookup | Fetch rows for real token IDs. Fixed-size staging copies may include harmless initialized tail bytes. |
| Snapshot donation | Fixed descriptor capacity with masked inactive writes; publish a snapshot only after every relevant layer has written it. |
| LM head/sampling | Gather true last-token rows. Intermediate prompt blocks do not need vocabulary logits or sampling. |

Padding rows and unused state slots must not share unmasked writes to a single
dummy address: concurrent writes can race even if the destination is a sink.
Zero-length padded requests need explicit inactive handling in state kernels.

## Execution order and expert traffic

For short interactive requests, keep whole-model block graphs: finish every
layer for a block, then continue the next block. This bounds activation storage
to C, but may reload the same expert layer for each block of a long prompt.

For a long admitted prompt, offer layer-major execution:

```text
create embedding/hyper-connection activation plane A
for each decoder layer L:
    make expert bank L available in a protected current-layer buffer
    prefetch expert bank L+1 to the other buffer on the copy stream
    for prompt blocks in increasing token order:
        stage a block of A and its true metadata
        replay the graph for layer L and the selected bucket
        store the valid outputs into activation plane B
    complete layer L; swap activation planes A and B
run final mixer/LM head for each request's last real token
publish completed prefill and admit decode
```

This is a native C++ dispatcher with CUDA graphs/events, not a Python call to
every layer/block. Each layer's weights and graph buffers have stable addresses.
For 48 layers and nine token buckets, there are 432 small layer graphs per
request bucket/parity, with many shared kernel binaries. Graph counts do not
grow with N or with the number of blocks. Separate long-prefill and decode graph
families avoid changing decode behavior when only prefill is layer-major.

Causality permits this order: a layer's block needs the previous layer's hidden
states and that same layer's earlier KV/GDN/conv history. Process blocks in
chronological order inside each layer. PLE n-gram IDs are known from prompt IDs,
while PLE conv state still advances in order at its own layer. Refactor its
current end-of-model ngram commit accordingly. Keep per-layer progress private
until all layers finish; existing Req.cached_len cannot advertise partial-layer
completion as a reusable prefix.

Sparse expert selection for future layers is not known until their routers run.
For large blocks/windows, stage the entire next expert layer: its identity and
weight addresses are already known, so no future router prediction is required.
Keep that layer resident throughout its block sweep. Small sparse workloads
retain the measured selective hit/miss schedule. Choose the mode from measured
whole-layer DMA vs selective gather cost; do not assume uniformly hot experts.

Expert buffers need leases until every consuming graph finishes. Publish cache
owners after copies finish; evictions cannot touch in-flight rows. Resident hot
experts can occupy the remaining separately budgeted slots. Layer-wide prefetch
can use native cudaMemcpyAsync/batched DMA; sparse gather may use the existing
GPU kernel over registered RAM. They have different resource costs.

PLE table row IDs for subsequent text-prefill blocks are already derivable from
prompt tokens and immutable hash constants, including the preceding two token
IDs. A native host worker can prefetch a bounded number of blocks' disk rows to
pinned staging while GPU work proceeds. This opportunity differs from future
layer expert routing, which is data-dependent. Keep graph-parity buffers and
ready/release events so an early fill cannot overwrite in-flight PLE rows.

## Activation storage choices

Layer-major execution trades expert transfers for storage of the whole prompt's
hidden states. For Qwen's four hyper-connection streams, each plane has
`N * 4 * hidden_size * dtype_bytes` payload.

1. GPU planes when two planes plus KV, current/next expert banks and maximum
   workspace fit. Buffers are reused across layers, never retained per layer.
2. Pinned RAM planes otherwise, with two or more fixed-size GPU staging windows.
   Prefetch block j+1 while j computes; copy j-1's valid outputs to RAM after its
   completion event. Dependency/event chains control reuse. Account for shared
   PCIe/RAM bandwidth with expert prefetch; overlap is not a guarantee of speed.
3. Whole-model block mode when RAM capacity or activation movement makes the
   layer-major mode slower. Choose a larger measured-safe block to amortize
   expert copies.

GPU/RAM planes can round capacity to a block boundary; metadata and KV still
refer to N real tokens. Round-up adds at most one block to a long request.
Do not copy or reserve every hidden layer: use two planes and bounded staging.

## QSA at long context

The current backend already chunks score rows to a 128 MiB raw-score workspace.
It still scores each real query against all visible compressed index keys and
stores full K/V. Selecting 2048 attended tokens does not eliminate the index
scan or permit deleting older KV without changing model behavior.

Keep the query bucket small and scan key pages in fixed tiles. The initial
implementation can retain the bounded current score workspace with runtime
visible limits. A later fused scorer/top-k can store tile-local candidates and
merge them globally, avoiding a full score matrix for the query bucket.

Exact top-k of tile-local top-k candidates is sufficient for global top-k. Use
the same score arithmetic and deterministic score/index tie rule. Do not change
to a restricted old-context window or approximate selection silently. Fixed
maximum-context grids or bounded loops inside kernels consume true visible
counts, so 100000 versus 100001 context tokens does not trigger recompilation.

Workspace is bounded, but indexer work remains approximately O(N^2/ratio) over
the entire prompt. Sparse attention bounds the final attended set, not all
prefill computation. This design makes memory and compilation manageable; it
does not establish short TTFT at 100k tokens.

## Memory budget

Use actual dtype, tensor geometry and measured graph peak memory:

```text
weights + KV(real_context_capacity) + bounded recurrent/snapshot slots
  + protected expert buffers/cache + activation planes/staging
  + maximum graph workspace + runtime headroom <= available VRAM
```

Reserve KV for the sum of admitted real context/output lengths. The bucket cap
is independent of this reservation. Keep GDN/PLE snapshots bounded: the Qwen
GDN recurrent tensor alone costs 108 MiB per full-model state slot, so retaining
one complete state at every 64-token boundary of 100k input would be impractical.
Use a fixed slot budget and sparse reusable checkpoints, with normal eviction.

The local Qwen checkpoint declares a 262144-token position limit. That is a
configuration limit, not a long-context quality test. Its current BF16 QSA pool
cost is 25356 B/token, excluding fixed rings/state/scratch:

| Real context | Page-rounded KV payload |
|---|---:|
| 10000 | 0.237 GiB |
| 100000 | 2.362 GiB |
| 131072 | 3.095 GiB |
| 262144 | 6.190 GiB |

At 100000 tokens, two unpadded BF16 hyper-connection planes cost 3.815 GiB; two
full NVFP4 expert-layer buffers cost 2.644 GiB. Replacing the measured 3072-slot
cache and 8192-token KV with these terms gives roughly 19.36 GiB allocated
before extra long-block graph workspace, rounding, allocator reservation and
runtime headroom. This is arithmetic from saved short-run allocations, not a
tested 100k fit or a hardware-specific rule.

If every block uses all experts, 49 blocks of 2048 tokens can move about
3109.4 GiB of expert payload for a 100k prompt. One layer sweep per prompt moves
about 63.46 GiB, excluding cache hits. If both activation planes are on RAM,
their round trips over 48 layers add roughly 183.1 GiB. These are payload
estimates, not measured timings, and assume this Qwen's tensor geometry.

## Graph memory and initialization

Precompile operator variants and capture only supported buckets before serving.
Use explicit scratch arenas/out buffers with known lifetimes, or isolated graph
pools. Do not depend on private allocator pool sharing for arbitrarily ordered
bucket/layer replays or concurrent prefill/decode. PyTorch documents same replay
order and no concurrent replay as the safe shared-private-pool conditions:
https://docs.pytorch.org/docs/2.11/notes/cuda.html#sharing-memory-across-captures

If two staging windows permit concurrent work, each has distinct storage and
completion events. Copies can overlap compute without launching two model
graphs against the same recurrent state or scratch. KV-pool growth changes
addresses; reserve the intended maximum at startup or recapture during an idle
resize, rather than growing captured allocations silently.

The current scheduler refuses a request if its whole input/output KV reservation
does not fit. Engine.max_seq_len is also clamped by KV token capacity. The
earlier benchmark used max_seq_len=2048 and 8192 KV tokens; neither configuration
can admit 100k input just by increasing a graph bucket.

## Implementation and evidence sequence

1. Fixed token/request buckets plus CUDA logical metadata and explicit scratch.
2. Padding-aware QSA writes, GDN endpoint handling, PLE native conv/indices,
   inactive requests, snapshot masks and router load masking.
3. Byte comparison of graph versus eager on identical padded shapes, with cold
   prefixes, continued prefixes, ragged batches, changing inputs and final states.
4. Cross-shape/chunk/layer-order numerical checks with documented tolerances;
   do not assume bit equality after changing GEMM shapes or FLA segmentation.
5. Whole-model block benchmark, then layer-major dispatcher and bounded
   activation planes; benchmark H2D/D2H expert/activation traffic separately.
6. Grow real admitted lengths through 10k, 32k, 100k and, if resources allow,
   131k. Save input IDs, final logits/states, TTFT, token/s, graph counts,
   initialization time, allocated/reserved/peak memory and per-stream traces.

This design's payload estimates are arithmetic, not benchmark results. Consult
the implementation report for the exact completed long-input runs and limitations.
