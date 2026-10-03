"""Compare an immutable source baseline with current MoE CUDA kernels, byte for byte.

Run with --baseline-root pointing at the baseline snapshot saved before edits.
Both variants run on one GPU in alternating order; compilation and validation are
outside the CUDA-event timing window. No model weights or HTTP server are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

import torch


REPO = Path(__file__).resolve().parents[1]


def load_python(name, path, replacements=()):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    source = path.read_text()
    for old, new in replacements:
        source = source.replace(old, new)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_baseline(root):
    base = root / "python/freetoken"
    load_python("freetoken.kernel.triton._bitexact_baseline_mxfp4", base / "kernel/triton/mxfp4_moe.py")
    impl = load_python(
        "freetoken.kernel._bitexact_baseline_impl", base / "kernel/moe_impl.py",
        [("from .triton.mxfp4_moe import", "from .triton._bitexact_baseline_mxfp4 import")],
    )
    fused = load_python(
        "freetoken.moe._bitexact_baseline_fused", base / "moe/fused_mxfp4.py",
        [("from freetoken.kernel import", "from freetoken.kernel._bitexact_baseline_impl import")],
    )
    load_python("freetoken.moe._bitexact_baseline_cache", base / "moe/offload_cache.py")
    return impl, fused


def baseline_copy(root):
    from tvm_ffi.cpp import load_inline
    header = root / "python/freetoken/kernel/csrc/jit/fast_index_copy.cuh"
    stamp = hashlib.sha256(header.read_bytes()).hexdigest()[:12]
    return load_inline(
        name=f"freetoken_bitexact_baseline_copy_{stamp}",
        cuda_sources=[f'#include "{header}"',
                      "TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, (&MultiIndexCopyKernel<1024,8>::run));"],
        extra_include_paths=[str(REPO / "python/freetoken/kernel/csrc/include")],
        extra_cuda_cflags=["-std=c++20", "-O3", "--expt-relaxed-constexpr"],
    )


def digest(tensor):
    raw = tensor.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def assert_bits(reference, actual, label):
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise AssertionError(f"{label}: incompatible shape/dtype")
    a = reference.contiguous().reshape(-1).view(torch.uint8)
    b = actual.contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(a, b):
        mismatches = int((a != b).sum().item())
        raise AssertionError(f"{label}: {mismatches} differing bytes")
    return {"label": label, "bytes": a.numel(), "mismatched_bytes": 0, "sha256": digest(reference)}


def capture(function, cycles=1):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.graph(graph):
        for _ in range(cycles):
            output = function()
    graph.replay()
    torch.cuda.synchronize()
    memory = {
        "graph_live_extra_bytes": max(0, torch.cuda.memory_allocated() - before),
        "peak_extra_allocated_bytes": max(0, torch.cuda.max_memory_allocated() - before),
    }
    return graph, output, memory


def paired_time(a, b, repeats, cycles=1, prepare_a=None, prepare_b=None):
    samples = {"before_ms": [], "after_ms": []}
    graphs = {"before_ms": a, "after_ms": b}
    preps = {"before_ms": prepare_a, "after_ms": prepare_b}
    for i in range(repeats):
        order = ("before_ms", "after_ms") if i % 2 == 0 else ("after_ms", "before_ms")
        for name in order:
            if preps[name] is not None:
                preps[name]()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / cycles)
    old, new = (statistics.median(samples[k]) for k in ("before_ms", "after_ms"))
    ratios = [x / y for x, y in zip(samples["before_ms"], samples["after_ms"])]
    return {**samples, "before_median_ms": old, "after_median_ms": new, "speedup": old / new,
            "paired_speedup_min": min(ratios), "paired_speedup_max": max(ratios)}


def weights(e, h, inter, pinned=False):
    roles = {
        "gate_up": (h // 2, 2 * inter),
        "gate_up_scale": (h // 32, 2 * inter),
        "gate_up_bias": (2 * inter,),
        "down": (inter // 2, h),
        "down_scale": (inter // 32, h),
        "down_bias": (h,),
    }
    result = {}
    for role, shape in roles.items():
        device = "cpu" if pinned else "cuda"
        if role.endswith("bias"):
            value = torch.randn(e, *shape, device=device, dtype=torch.bfloat16) * 0.05
        else:
            low, high = (124, 129) if role.endswith("scale") else (0, 256)
            value = torch.randint(low, high, (e, *shape), device=device, dtype=torch.uint8)
        result[role] = value.pin_memory() if pinned else value
    return result


def views(banks):
    from freetoken.moe.legacy_format import canonical_role
    roles = {canonical_role(k): v for k, v in banks.items()}
    return [roles[k] for k in ("gate_up", "gate_up_scale", "gate_up_bias", "down", "down_scale", "down_bias")]


def intermediate_proof(old_impl, old_fused, x, w, ids, banks, alpha, limit):
    from freetoken.kernel import mxfp4_splitk_gemv_triton

    batch, h = x.shape
    routes, inter = ids.numel(), banks["down"].shape[1] * 2
    experts = ids.reshape(-1).long()
    routed_x = x if batch == 1 else x.repeat_interleave(ids.shape[1], dim=0)
    stride = 0 if batch == 1 else routed_x.stride(0)
    splits = old_fused._decode_split_count(routes, h // 32, target_programs=180)
    pa = torch.empty(routes * splits, 2 * inter, device=x.device)
    pb = torch.empty_like(pa)
    args = (routed_x, banks["gate_up"], banks["gate_up_scale"], banks["gate_up_bias"], experts)
    kw = {"N": 2 * inter, "K": h, "stride_xe": stride, "num_splits": splits}
    gu = old_impl.mxfp4_splitk_gemv_triton(*args, partial=pa, **kw)
    gu_new = torch.empty_like(gu)
    activated = torch.empty(routes, inter, device=x.device, dtype=x.dtype)
    old_impl.gpt_oss_swiglu_triton(gu, activated, alpha=alpha, limit=limit, compute_type=x.dtype)
    new_args = (x, *args[1:-1], ids.reshape(-1))
    new_kw = {**kw, "stride_xe": x.stride(0), "routes_per_token": ids.shape[1]}
    act_new = mxfp4_splitk_gemv_triton(*new_args, partial=pb, gate_up_out=gu_new,
                                    swiglu=(alpha, limit, x.dtype), **new_kw)
    proof = [assert_bits(pa, pb, "gate_up_fp32_partials"), assert_bits(gu, gu_new, "gate_up_bf16"),
             assert_bits(activated, act_new, "swiglu_activation")]
    splits = old_fused._decode_split_count(routes, inter // 32, target_programs=72)
    pa = torch.empty(routes * splits, h, device=x.device)
    pb = torch.empty_like(pa)
    kw = {"N": h, "K": inter, "stride_xe": inter, "num_splits": splits, "expert_wts": w.reshape(-1)}
    args = (activated, banks["down"], banks["down_scale"], banks["down_bias"], experts)
    da = old_impl.mxfp4_splitk_gemv_triton(*args, partial=pa, **kw)
    args = (act_new, *args[1:])
    db = mxfp4_splitk_gemv_triton(*args, partial=pb, **kw)
    proof.extend([assert_bits(pa, pb, "down_fp32_partials"), assert_bits(da, db, "weighted_down_bf16")])
    return proof


def compute_case(old_impl, old_fused, batch, h, inter, repeats, mode="decode"):
    from freetoken.moe import fused_mxfp4
    e, topk = 16, 4
    banks = weights(e, h, inter)
    x = torch.randn(batch, h, device="cuda", dtype=torch.bfloat16)
    ids = (torch.arange(batch * topk, device="cuda") % e).reshape(batch, topk).int()
    w = torch.rand(batch, topk, device="cuda")
    args = (x, w, ids, *views(banks))
    kw = {"top_k": topk, "hidden_act_alpha": 1.702, "swiglu_limit": 7.0}
    function = "run_mxfp4_prefill_experts_t" if mode == "prefill" else "run_mxfp4_splitk_decode_experts"
    old = lambda: getattr(old_fused, function)(*args, **kw)
    new = lambda: getattr(fused_mxfp4, function)(*args, **kw)
    proof = (intermediate_proof(old_impl, old_fused, x, w, ids, banks, 1.702, 7.0) if mode == "decode"
             else prefill_proof(old_impl, x, w, ids, banks))
    proof.append(assert_bits(old(), new(), "expert_output"))
    cycles = 16 if mode == "decode" else 4
    ga, ya, ma = capture(old, cycles=cycles)
    gb, yb, mb = capture(new, cycles=cycles)
    proof.append(assert_bits(ya, yb, "expert_graph_output"))
    return {"operation": "expert_compute", "mode": mode, "batch": batch, "hidden": h, "intermediate": inter,
            "proof": proof, "memory_before": ma, "memory_after": mb,
            "eliminated_gate_up_tensor_bytes": batch * topk * 2 * inter * 2 if mode == "decode" else 0,
            "eliminated_routed_input_bytes": batch * topk * h * x.element_size() if mode == "decode" and batch > 1 else 0,
            "eliminated_route_tokens_bytes": batch * topk * 8 if mode == "decode" and batch > 1 else 0,
            "eliminated_expert_id_cast_bytes": batch * topk * 8 if mode == "decode" else 0,
            **paired_time(ga, gb, repeats, cycles=cycles)}


def prefill_proof(old_impl, x, w, ids, banks):
    from freetoken.kernel import moe_impl
    from freetoken.moe.fused import moe_align_block_size, try_get_optimal_moe_config
    tokens, h = x.shape
    e, inter = banks["gate_up"].shape[0], banks["down"].shape[1] * 2
    topk = ids.shape[1]
    config = try_get_optimal_moe_config((e, 2 * inter, h), (e, h, inter), topk, tokens)
    if config["BLOCK_SIZE_K"] % 32:
        config = {**config, "BLOCK_SIZE_K": 64}
    aligned = moe_align_block_size(ids, config["BLOCK_SIZE_M"], e)
    gu = torch.empty(tokens * topk, 2 * inter, device=x.device, dtype=x.dtype)
    gu_new = torch.empty_like(gu)
    tail = (w, ids, *aligned, False, topk, config, x.dtype)
    old_impl.mxfp4_fused_moe_kernel_t_triton(x, banks["gate_up"], banks["gate_up_scale"], banks["gate_up_bias"], gu, *tail)
    moe_impl.mxfp4_fused_moe_kernel_t_triton(x, banks["gate_up"], banks["gate_up_scale"], banks["gate_up_bias"], gu_new, *tail)
    proof = [assert_bits(gu, gu_new, "prefill_gate_up_gemm")]
    act, act_new = (torch.empty(tokens * topk, inter, device=x.device, dtype=x.dtype) for _ in range(2))
    old_impl.gpt_oss_swiglu_triton(gu, act, alpha=1.702, limit=7.0, compute_type=x.dtype)
    moe_impl.gpt_oss_swiglu_triton(gu_new, act_new, alpha=1.702, limit=7.0, compute_type=x.dtype)
    proof.append(assert_bits(act, act_new, "prefill_swiglu"))
    down, down_new = (torch.empty(tokens * topk, h, device=x.device, dtype=x.dtype) for _ in range(2))
    tail = (w, ids, *aligned, True, 1, config, x.dtype)
    old_impl.mxfp4_fused_moe_kernel_t_triton(act, banks["down"], banks["down_scale"], banks["down_bias"], down, *tail)
    moe_impl.mxfp4_fused_moe_kernel_t_triton(act_new, banks["down"], banks["down_scale"], banks["down_bias"], down_new, *tail)
    proof.append(assert_bits(down, down_new, "prefill_weighted_down_gemm"))
    return proof


def copy_case(old_copy, features, count, host, repeats):
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
    from freetoken.kernel.pinned import device_ptr
    e = max(count, 4)
    sources = [torch.randint(0, 256, (e, feat), dtype=torch.uint8,
                             device="cpu" if host else "cuda") for feat in features]
    if host:
        sources = [x.pin_memory() for x in sources]
    before = [torch.full((e + 2, feat), 165, dtype=torch.uint8, device="cuda") for feat in features]
    after = [x.clone() for x in before]
    planned = max(count, 1)
    src = torch.arange(planned, device="cuda", dtype=torch.int32).flip(0)
    dst = torch.arange(1, planned + 1, device="cuda", dtype=torch.int32)
    num = torch.tensor([count], device="cuda", dtype=torch.int64)
    ptrs = torch.tensor([device_ptr(x) for x in sources], device="cuda", dtype=torch.int64)
    feats = torch.tensor(features, device="cuda", dtype=torch.int64)
    ap = torch.tensor([x.data_ptr() for x in before], device="cuda", dtype=torch.int64)
    bp = torch.tensor([x.data_ptr() for x in after], device="cuda", dtype=torch.int64)
    old = lambda: old_copy.launch(ap, ptrs, feats, dst, src, num)
    new = lambda: fast_index_copy_multi_jit(bp, ptrs, feats, dst, src, num)
    old()
    new()
    torch.cuda.synchronize()
    proofs = []
    for i, (a, b) in enumerate(zip(before, after)):
        expected = torch.full_like(a, 165)
        if count:
            expected[1:count + 1].copy_(sources[i].to("cuda").index_select(0, src.long()))
        proofs.append(assert_bits(expected, a, f"bank_{i}_baseline_and_canary"))
        proofs.append(assert_bits(a, b, f"bank_{i}_candidate_and_canary"))
    ga, _, _ = capture(old, cycles=4)
    gb, _, _ = capture(new, cycles=4)
    return {"operation": "host_to_vram" if host else "vram_to_vram", "features": features,
            "rows": count, "bytes_copied": sum(features) * count, "proof": proofs,
            **paired_time(ga, gb, repeats, cycles=4)}


def pipeline_case(old_impl, old_fused, old_copy, batch, h, inter, miss, repeats):
    from freetoken.kernel import gpt_oss_fused_routing
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts
    from freetoken.moe.offload_cache import OffloadMoeCache
    e, topk = 128, 4
    sources = {k: [v] for k, v in weights(e, h, inter, pinned=True).items()}
    a = OffloadMoeCache(1, e, 2 * e, torch.device("cuda"), prefill_overlap=False, quant_format="mxfp4_triton")
    b = OffloadMoeCache(1, e, 2 * e, torch.device("cuda"), prefill_overlap=False, quant_format="mxfp4_triton")
    a.set_bank_sources(sources)
    b.set_bank_sources(sources)
    x = torch.randn(batch, h, device="cuda", dtype=torch.bfloat16)
    router = torch.randn(e, h, device="cuda", dtype=torch.bfloat16) / h**0.5
    bias = torch.randn(e, device="cuda", dtype=torch.bfloat16) * 0.01
    kw = {"top_k": topk, "hidden_act_alpha": 1.702, "swiglu_limit": 7.0}
    initial_slots = torch.arange(e, device="cuda", dtype=torch.int32)
    mask = torch.zeros(e, device="cuda", dtype=torch.bool) if miss else torch.ones(e, device="cuda", dtype=torch.bool)
    initial = torch.where(mask, initial_slots, -1)
    source_ids = torch.arange(e, device="cuda", dtype=torch.int32)
    all_n = torch.tensor([e], device="cuda", dtype=torch.int64)
    for cache in (a, b):
        for bank in cache.bank_caches.values():
            bank.view(torch.uint8).fill_(165)
        old_copy.launch(cache._copy_dst_ptrs, cache._copy_src_ptrs[0], cache._copy_feat_bytes,
                        source_ids, source_ids, all_n)

    def prepare(cache):
        cache.slot_for_id[0].copy_(initial)
        cache.id_of_slot.fill_(-1)
        cache.id_of_slot[:e].copy_(initial)
        cache.usage.zero_()
        cache.step.zero_()

    def execute(cache, is_old):
        logits = torch.nn.functional.linear(x, router, bias)
        route = old_impl.gpt_oss_fused_routing if is_old else gpt_oss_fused_routing
        rw, ids = route(logits, topk)
        cache.ensure_experts(0, ids)
        if is_old:
            old_copy.launch(cache._copy_dst_ptrs, cache._copy_src_ptrs[0], cache._copy_feat_bytes,
                            cache.evict_slots, cache.src_indices, cache.num_indices)
        else:
            cache.copy_missing()
        compute = old_fused.run_mxfp4_splitk_decode_experts if is_old else run_mxfp4_splitk_decode_experts
        out = compute(x, rw, ids, *views(cache.bank_caches), **kw)
        return logits, rw, ids, out

    prepare(a)
    ya = execute(a, True)
    prepare(b)
    yb = execute(b, False)
    proofs = [assert_bits(p, q, label) for p, q, label in zip(ya, yb, ("router_logits", "router_weights", "slot_ids", "output"))]
    for name in ("slot_for_id", "id_of_slot", "usage", "step", "num_indices"):
        proofs.append(assert_bits(getattr(a, name), getattr(b, name), name))
    copied = int(a.num_indices.item())
    for role in a.bank_caches:
        proofs.append(assert_bits(a.bank_caches[role], b.bank_caches[role], f"cache_{role}"))
    ga, oa, ma = capture(lambda: execute(a, True))
    gb, ob, mb = capture(lambda: execute(b, False))
    timings = paired_time(ga, gb, repeats, prepare_a=lambda: prepare(a), prepare_b=lambda: prepare(b))
    for p, q, label in zip(oa, ob, ("graph_logits", "graph_weights", "graph_ids", "graph_output")):
        proofs.append(assert_bits(p, q, label))
    return {"operation": "moe_pipeline", "batch": batch, "hidden": h, "intermediate": inter,
            "miss": miss, "distinct_experts_copied": copied, "proof": proofs,
            "num_experts": e, "memory_before": ma, "memory_after": mb, **timings}


def routing_case(old_impl, tokens, experts, topk, repeats):
    from freetoken.kernel import gpt_oss_fused_routing
    logits = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
    logits[0].zero_()
    if tokens > 1:
        logits[1, ::3] = 1.0
    old = lambda: old_impl.gpt_oss_fused_routing(logits, topk)
    new = lambda: gpt_oss_fused_routing(logits, topk)
    proof = [assert_bits(p, q, name) for p, q, name in zip(old(), new(), ("route_weights", "expert_ids"))]
    ga, _, _ = capture(old, cycles=8)
    gb, _, _ = capture(new, cycles=8)
    return {"operation": "routing", "tokens": tokens, "num_experts": experts, "topk": topk,
            "proof": proof, **paired_time(ga, gb, repeats, cycles=8)}


def fusion_ablation(old_fused, baseline_root, batch, h, inter, repeats):
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts
    unfused = load_python("freetoken.moe._bitexact_unfused_current", baseline_root / "python/freetoken/moe/fused_mxfp4.py")
    banks = weights(16, h, inter)
    x = torch.randn(batch, h, device="cuda", dtype=torch.bfloat16)
    ids = (torch.arange(batch * 4, device="cuda") % 16).reshape(batch, 4).int()
    w = torch.rand(batch, 4, device="cuda")
    args = (x, w, ids, *views(banks))
    kw = {"top_k": 4, "hidden_act_alpha": 1.702, "swiglu_limit": 7.0}
    funcs = [lambda: old_fused.run_mxfp4_splitk_decode_experts(*args, **kw),
             lambda: unfused.run_mxfp4_splitk_decode_experts(*args, **kw),
             lambda: run_mxfp4_splitk_decode_experts(*args, **kw)]
    graphs, outputs = [], []
    for fun in funcs:
        graph, output, _ = capture(fun, cycles=16)
        graphs.append(graph)
        outputs.append(output)
    proof = [assert_bits(outputs[0], output, name) for output, name in zip(outputs[1:], ("lut_only", "lut_and_fusion"))]
    return {"operation": "fusion_ablation", "batch": batch, "hidden": h, "intermediate": inter,
            "proof": proof, "original_vs_lut": paired_time(graphs[0], graphs[1], repeats, cycles=16),
            "lut_vs_fused": paired_time(graphs[1], graphs[2], repeats, cycles=16),
            "original_vs_final": paired_time(graphs[0], graphs[2], repeats, cycles=16)}


def invalidation_case(experts, repeats):
    from freetoken.moe._bitexact_baseline_cache import OffloadMoeCache as OldCache
    from freetoken.moe.offload_cache import OffloadMoeCache
    a = OldCache(3, experts, 3 * experts, torch.device("cuda"), prefill_overlap=False)
    b = OffloadMoeCache(3, experts, 3 * experts, torch.device("cuda"), prefill_overlap=False)
    ids = torch.arange(3 * experts, device="cuda", dtype=torch.int32).roll(experts // 2)
    ids[::3] = -1
    slots = torch.full((3 * experts,), -1, device="cuda", dtype=torch.int32)
    valid = ids >= 0
    slots[ids[valid].long()] = torch.arange(3 * experts, device="cuda", dtype=torch.int32)[valid]
    age = torch.arange(3 * experts, device="cuda", dtype=torch.int64) + 19

    def prepare(cache):
        cache.id_of_slot.copy_(ids)
        cache.slot_for_id.view(-1).copy_(slots)
        cache.usage.copy_(age)

    old = lambda: a._invalidate_prefill_buffer(1)
    new = lambda: b._invalidate_prefill_buffer(1)
    prepare(a)
    prepare(b)
    old()
    new()
    proofs = [assert_bits(getattr(a, n), getattr(b, n), n) for n in ("slot_for_id", "id_of_slot", "usage")]
    # The original boolean indexing cannot be captured; time both variants eagerly.
    for _ in range(3):
        prepare(a)
        old()
        prepare(b)
        new()
    samples = {"before_ms": [], "after_ms": [], "before_host_enqueue_ms": [], "after_host_enqueue_ms": []}
    for i in range(repeats):
        for name, cache, fun in (("before", a, old), ("after", b, new)) if i % 2 == 0 else (("after", b, new), ("before", a, old)):
            prepare(cache)
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            t = time.perf_counter_ns()
            fun()
            samples[name + "_host_enqueue_ms"].append((time.perf_counter_ns() - t) / 1e6)
            end.record()
            end.synchronize()
            samples[name + "_ms"].append(begin.elapsed_time(end))
    old_ms, new_ms = (statistics.median(samples[n + "_ms"]) for n in ("before", "after"))
    old_host, new_host = (statistics.median(samples[n + "_host_enqueue_ms"]) for n in ("before", "after"))
    graph, _, _ = capture(new)
    prepare(b)
    graph.replay()
    torch.cuda.synchronize()
    proofs.extend(assert_bits(getattr(a, n), getattr(b, n), "graph_" + n) for n in ("slot_for_id", "id_of_slot", "usage"))
    return {"operation": "prefill_cache_invalidation", "num_experts": experts, "proof": proofs,
            "timing_note": "eager CUDA events include launch gaps; host enqueue measured separately",
            **samples, "before_median_ms": old_ms, "after_median_ms": new_ms, "speedup": old_ms / new_ms,
            "before_host_enqueue_median_ms": old_host, "after_host_enqueue_median_ms": new_host}


def profile_case(old_fused, output):
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts

    batch, h, inter, topk = 4, 2880, 2880, 4
    banks = weights(16, h, inter)
    x = torch.randn(batch, h, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(batch * topk, device="cuda", dtype=torch.int32).reshape(batch, topk)
    w = torch.rand(batch, topk, device="cuda")
    args = (x, w, ids, *views(banks))
    kw = dict(top_k=topk, hidden_act_alpha=1.702, swiglu_limit=7.0)
    result = {"operation": "profile_decode", "batch": batch, "hidden": h, "intermediate": inter,
              "note": "profiler timings contain instrumentation overhead; use A/B event cases for performance"}
    outputs = []
    for label, fun in (("before", old_fused.run_mxfp4_splitk_decode_experts), ("after", run_mxfp4_splitk_decode_experts)):
        for _ in range(3):
            fun(*args, **kw)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    profile_memory=True, record_shapes=True) as profile:
            y = fun(*args, **kw)
            torch.cuda.synchronize()
        outputs.append(y)
        path = output.with_name(output.stem + "_" + label + "_trace.json")
        profile.export_chrome_trace(str(path))
        result[label + "_trace"] = str(path)
        result[label + "_events"] = [{"name": e.key, "count": e.count, "cpu_time_us": e.cpu_time_total,
                                      "device_time_us": e.device_time_total,
                                      "self_cpu_memory_bytes": e.self_cpu_memory_usage,
                                      "self_device_memory_bytes": e.self_device_memory_usage}
                                     for e in profile.key_averages()]
    result["proof"] = [assert_bits(outputs[0], outputs[1], "profile_output")]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", choices=("all", "compute", "prefill", "copy", "pipeline", "routing", "invalidation", "ablation", "profile"), default="all")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    torch.cuda.init()
    torch.manual_seed(73019)
    old_impl, old_fused = load_baseline(args.baseline_root.resolve())
    old_copy = baseline_copy(args.baseline_root.resolve()) if args.only in ("all", "copy", "pipeline") else None
    result = {"baseline": json.loads((args.baseline_root / "manifest.json").read_text()),
              "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "current_source_sha256": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in (REPO / "python/freetoken").rglob("*")
                                        if path.is_file() and path.suffix in (".py", ".cuh", ".cpp")},
              "source_root": str(REPO), "command": sys.argv, "timing": "alternating CUDA graphs/events; compile/validation/state preparation excluded",
              "bitwise": "uint8 comparison including signed zeros, NaN payloads, and untouched canary rows", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "proof" and not isinstance(v, list)}), flush=True)

    if args.only == "profile":
        record(profile_case(old_fused, args.output))

    if args.only in ("all", "compute"):
        shapes = [(1, 2880, 2880), (4, 2880, 2880)] if args.quick else [(1, 1024, 512), (4, 2048, 768), (1, 2880, 2880), (4, 2880, 2880), (16, 4096, 1024)]
        for shape in shapes:
            record(compute_case(old_impl, old_fused, *shape, args.repeats))
    if args.only == "ablation":
        for shape in ((1, 2880, 2880), (4, 2880, 2880)):
            record(fusion_ablation(old_fused, args.baseline_root, *shape, args.repeats))
    if args.only in ("all", "prefill"):
        shapes = [(24, 1024, 512), (128, 2880, 2880)] if args.quick else [(24, 1024, 512), (128, 2048, 768), (128, 2880, 2880), (512, 2880, 2880)]
        for shape in shapes:
            record(compute_case(old_impl, old_fused, *shape, args.repeats, mode="prefill"))
    if args.only in ("all", "routing"):
        profiles = [(1, 128, 4), (1024, 128, 4)] if args.quick else [(1, 40, 4), (1, 128, 4), (16, 128, 4), (1024, 128, 4), (16, 512, 8)]
        for profile in profiles:
            record(routing_case(old_impl, *profile, args.repeats))
    if args.only in ("all", "invalidation"):
        for experts in ((128,) if args.quick else (4, 128, 512)):
            record(invalidation_case(experts, args.repeats))
    if args.only in ("all", "copy"):
        profiles = [[8192, 512, 256, 4096, 512, 256], [8294400, 518400, 11520, 4147200, 259200, 5760]]
        for host in (False, True):
            for features in profiles:
                for count in ((0, 4) if args.quick else (0, 1, 4, 16)):
                    record(copy_case(old_copy, features, count, host, args.repeats))
    if args.only in ("all", "pipeline"):
        shapes = [(1, 2880, 2880)] if args.quick else [(1, 2048, 768), (1, 2880, 2880), (4, 2880, 2880)]
        for shape in shapes:
            for miss in (False, True):
                record(pipeline_case(old_impl, old_fused, old_copy, *shape, miss, args.repeats))
    print(f"All {len(result['cases'])} cases passed bitwise checks; results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
