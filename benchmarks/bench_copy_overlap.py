"""Byte-exact copy/compute experiments with fixed saved tensors and unchanged math kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

REPO = Path(__file__).resolve().parents[1]


def raw(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def checksum(tensor):
    return hashlib.sha256(raw(tensor).cpu().numpy().tobytes()).hexdigest()


def exact(a, b, label):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(f"{label}: shape/dtype mismatch")
    x, y = raw(a), raw(b)
    mismatch = int((x != y).sum().item())
    if mismatch:
        raise AssertionError(f"{label}: {mismatch} differing bytes")
    return {"label": label, "bytes": x.numel(), "different_bytes": mismatch, "sha256": checksum(a)}


def make_banks(fmt, layers, experts, h, inter, seed):
    from freetoken.kernel.aot_models import fp8_block_scale_pad
    g = torch.Generator().manual_seed(seed)
    s = layers * experts
    def u8(*shape, low=0, high=256):
        return torch.randint(low, high, shape, dtype=torch.uint8, generator=g)
    def dense(*shape, dtype=torch.bfloat16, scale=0.02):
        return (torch.randn(shape, generator=g) * scale).to(dtype)
    if fmt in ("bf16", "fp16"):
        dt = torch.bfloat16 if fmt == "bf16" else torch.float16
        banks = {"gate_up": dense(s, 2 * inter, h, dtype=dt, scale=h**-0.5),
                 "down": dense(s, h, inter, dtype=dt, scale=inter**-0.5)}
    elif fmt == "fp8_block":
        banks = {"gate_up": dense(s, 2 * inter, h, dtype=torch.float8_e4m3fn, scale=0.5),
                 "gate_up_scale": torch.full((s, 2 * inter // 128, fp8_block_scale_pad(2 * inter // 128, h // 128)), h**-0.5 * 2, dtype=torch.bfloat16),
                 "down": dense(s, h, inter, dtype=torch.float8_e4m3fn, scale=0.5),
                 "down_scale": torch.full((s, h // 128, fp8_block_scale_pad(h // 128, inter // 128)), inter**-0.5 * 2, dtype=torch.bfloat16)}
    elif fmt == "nvfp4":
        banks = {"gate_up": u8(s, 2 * inter, h // 2),
                 "gate_up_scale": torch.full((s, 2 * inter, h // 16), 0.5, dtype=torch.float8_e4m3fn),
                 "gate_up_global": torch.full((s, 2 * inter), 0.02, dtype=torch.float16),
                 "down": u8(s, h, inter // 2),
                 "down_scale": torch.full((s, h, inter // 16), 0.5, dtype=torch.float8_e4m3fn),
                 "down_global": torch.full((s, h), 0.02, dtype=torch.float16)}
    elif fmt in ("nvfp4_marlin", "nvfp4_b12x"):
        from freetoken.layers.quantization.moe.base import MoEConfig
        from freetoken.layers.quantization.moe.nvfp4 import MarlinNvfp4MoEKernel, B12xNvfp4MoEKernel
        kernel = MarlinNvfp4MoEKernel() if fmt == "nvfp4_marlin" else B12xNvfp4MoEKernel()
        specs = kernel.layout(MoEConfig(experts, h, inter, top_k=4))
        banks = {}
        for name, spec in specs.items():
            if spec.resident:
                continue
            t = torch.empty((s, *spec.shape), dtype=spec.dtype)
            t.view(torch.uint8).random_(0, 256, generator=g)
            banks[name] = t
    elif fmt == "mxfp4":
        banks = {"gate_up": u8(s, h // 2, 2 * inter),
                 "gate_up_scale": u8(s, h // 32, 2 * inter, low=120, high=123),
                 "gate_up_bias": dense(s, 2 * inter, scale=0.01),
                 "down": u8(s, inter // 2, h),
                 "down_scale": u8(s, inter // 32, h, low=120, high=123),
                 "down_bias": dense(s, h, scale=0.01)}
    elif fmt == "ds_fp4":
        banks = {"gate_up": u8(s, 2 * inter, h // 2),
                 "gate_up_scale": u8(s, 2 * inter, h // 32, low=120, high=123),
                 "down": u8(s, h, inter // 2),
                 "down_scale": u8(s, h, inter // 32, low=120, high=123)}
    elif fmt == "q4_0":
        def packed(n, k):
            codes = u8(s, n, k // 32, 16)
            scales = torch.full((s, n, k // 32), 0.1 / k**0.5, dtype=torch.float16)
            return torch.cat((scales.view(torch.uint8).reshape(s, n, k // 32, 2), codes), dim=-1).reshape(s, n, k // 32 * 18)
        banks = {"gate_up": packed(2 * inter, h), "down": packed(h, inter)}
    else:
        raise ValueError(fmt)
    return {name: tensor.reshape(layers, experts, *tensor.shape[1:]).contiguous().pin_memory()
            for name, tensor in banks.items()}


def compute(fmt, x, ids, w, banks, prefill=False):
    if fmt in ("bf16", "fp16"):
        from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl
        fun = fused_experts_impl if prefill else fused_experts_decode_impl
        return fun(x, banks["gate_up"], banks["down"], w, ids)
    if fmt == "fp8_block":
        from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block, fused_experts_fp8_block
        if prefill:
            return fused_experts_fp8_block(x, banks["gate_up"], banks["gate_up_scale"], banks["down"], banks["down_scale"], w, ids, banks["gate_up"].shape[0])
        return fused_experts_decode_fp8_block(x, banks["gate_up"], banks["gate_up_scale"], banks["down"], banks["down_scale"], w, ids)
    if fmt == "nvfp4":
        from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin, fused_experts_nvfp4
        inputs = (x, banks["gate_up"], banks["gate_up_scale"], banks["gate_up_global"], banks["down"], banks["down_scale"], banks["down_global"], w, ids)
        return fused_experts_nvfp4(*inputs, banks["gate_up"].shape[0]) if prefill else fused_experts_decode_nvfp4_marlin(*inputs)
    if fmt == "mxfp4":
        from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts, run_mxfp4_prefill_experts_t
        fun = run_mxfp4_prefill_experts_t if prefill else run_mxfp4_splitk_decode_experts
        return fun(x, w, ids, banks["gate_up"], banks["gate_up_scale"], banks["gate_up_bias"],
                                             banks["down"], banks["down_scale"], banks["down_bias"], top_k=ids.shape[1], hidden_act_alpha=1.702, swiglu_limit=7.0)
    if fmt == "ds_fp4":
        from freetoken.moe.fused_ds_fp4 import routed_experts_fp4, routed_experts_fp4_prefill
        inputs = (x, ids, w, banks["gate_up"], banks["gate_up_scale"], banks["down"], banks["down_scale"], 7.0)
        return routed_experts_fp4_prefill(*inputs, banks["gate_up"].shape[0]) if prefill else routed_experts_fp4(*inputs)
    if fmt == "q4_0":
        from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0
        return fused_experts_gguf_q4_0(x, banks["gate_up"], banks["down"], w, ids, "silu")
    raise ValueError(fmt)


def gate_stage(fmt, x, ids, w, banks):
    from freetoken.kernel import fused_moe_decode_kernel_triton, mxfp4_splitk_gemv_triton
    from freetoken.layers import gated_act_and_mul
    m, h = x.shape
    topk = ids.shape[1]
    routes = m * topk
    inter = banks["gate_up"].shape[-1] // 2 if fmt == "mxfp4" else banks["gate_up"].shape[1] // 2
    if fmt == "mxfp4":
        from freetoken.moe.fused_mxfp4 import _decode_split_count
        return mxfp4_splitk_gemv_triton(x, banks["gate_up"], banks["gate_up_scale"], banks["gate_up_bias"],
                                      ids.reshape(-1), N=2 * inter, K=h, stride_xe=h,
                                      num_splits=_decode_split_count(routes, h // 32, 180),
                                      swiglu=(1.702, 7.0, x.dtype), routes_per_token=topk)
    if fmt == "ds_fp4":
        from freetoken.moe.fused_ds_fp4 import _grouped_decode, act_quant_fp8_roundtrip, fused_swiglu, act_quant_fp8_inplace
        value = act_quant_fp8_roundtrip(x, 128)
        gu = _grouped_decode(value, banks["gate_up"], banks["gate_up_scale"], ids, None,
                             a_row_is_route=False, mul_routed_weight=False)
        act = fused_swiglu(gu, 7.0).reshape(routes, inter)
        act_quant_fp8_inplace(act, 128)
        return act
    if fmt == "q4_0":
        from freetoken.kernel.gguf import ggml_moe_a8_vec
        from freetoken.layers.activation import silu_and_mul
        return silu_and_mul(ggml_moe_a8_vec(x, banks["gate_up"], ids, topk, 2, 2 * inter, m))
    gu = torch.empty(m, topk, 2 * inter, device=x.device, dtype=x.dtype)
    if fmt in ("bf16", "fp16"):
        config = dict(BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, num_warps=8)
        fused_moe_decode_kernel_triton(x, banks["gate_up"], gu, w, ids, False, False, config, x.dtype)
    elif fmt == "fp8_block":
        from freetoken.kernel.triton.fp8_blockscale_moe import _decode_gemm
        _decode_gemm(x, banks["gate_up"], banks["gate_up_scale"], gu, w, ids, False, False)
    elif fmt == "nvfp4":
        from freetoken.moe.fused_nvfp4 import _decode_gemm_marlin
        _decode_gemm_marlin(x, banks["gate_up"], banks["gate_up_scale"], banks["gate_up_global"], gu, w, ids, False, False)
    else:
        raise ValueError(fmt)
    act = torch.empty(routes, inter, device=x.device, dtype=x.dtype)
    gated_act_and_mul("silu", gu.view(routes, 2 * inter), act, alpha=1.702 if fmt == "nvfp4" else 1.0,
                     limit=7.0 if fmt == "nvfp4" else float("inf"))
    return act


def down_stage(fmt, act, ids, w, banks):
    from freetoken.kernel import fused_moe_decode_kernel_triton, moe_sum_reduce_triton, mxfp4_splitk_gemv_triton
    m, topk = ids.shape
    routes, inter = act.shape
    h = banks["down"].shape[-1] if fmt == "mxfp4" else banks["down"].shape[1]
    if fmt == "mxfp4":
        from freetoken.moe.fused_mxfp4 import _decode_split_count
        down = mxfp4_splitk_gemv_triton(act, banks["down"], banks["down_scale"], banks["down_bias"], ids.reshape(-1),
                                        N=h, K=inter, stride_xe=inter,
                                        num_splits=_decode_split_count(routes, inter // 32, 72), expert_wts=w.reshape(-1))
        return down.view(m, topk, h).sum(dim=1).to(act.dtype)
    if fmt == "ds_fp4":
        from freetoken.moe.fused_ds_fp4 import _grouped_decode
        return _grouped_decode(act, banks["down"], banks["down_scale"], ids, w,
                               a_row_is_route=True, mul_routed_weight=True).sum(dim=1)
    if fmt == "q4_0":
        from freetoken.kernel.gguf import ggml_moe_a8_vec
        out = ggml_moe_a8_vec(act, banks["down"], ids, 1, 2, h, routes)
        out = out.reshape(m, topk, h) * w.reshape(m, topk, 1).to(out.dtype)
        return out.sum(dim=1)
    down = torch.empty(m, topk, h, device=act.device, dtype=act.dtype)
    if fmt in ("bf16", "fp16"):
        config = dict(BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, num_warps=8)
        fused_moe_decode_kernel_triton(act, banks["down"], down, w, ids, True, True, config, act.dtype)
    elif fmt == "fp8_block":
        from freetoken.kernel.triton.fp8_blockscale_moe import _decode_gemm
        _decode_gemm(act, banks["down"], banks["down_scale"], down, w, ids, True, True)
    elif fmt == "nvfp4":
        from freetoken.moe.fused_nvfp4 import _decode_gemm_marlin
        _decode_gemm_marlin(act, banks["down"], banks["down_scale"], banks["down_global"], down, w, ids, True, True)
    out = torch.empty(m, h, device=act.device, dtype=act.dtype)
    moe_sum_reduce_triton(down, out)
    return out


def capture(fun):
    for _ in range(3):
        fun()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.cuda.graph(graph):
        out = fun()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out, max(0, torch.cuda.max_memory_allocated() - before)


def time_graphs(graphs, repeats):
    samples = {name: [] for name in graphs}
    for i in range(repeats):
        names = list(graphs)
        names = names[i % len(names):] + names[:i % len(names)]
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    return {name: {"median_ms": statistics.median(v), "samples_ms": v} for name, v in samples.items()}


def experiment(fmt, args):
    from freetoken.kernel.copy_plan import ByteCopyPlan
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
    from freetoken.kernel.pinned import device_ptr

    l, e, h, inter, batch, topk = args.layers, args.experts, args.hidden, args.intermediate, args.batch, args.topk
    case = args.output.parent / "inputs" / f"{fmt}_L{l}_E{e}_H{h}_I{inter}_B{batch}_seed{args.seed}_v2"
    case.mkdir(parents=True, exist_ok=True)
    datafile = case / "inputs.pt"
    if datafile.exists():
        data = torch.load(datafile, weights_only=True)
        host = {k: v.pin_memory() for k, v in data["banks"].items()}
    else:
        host = make_banks(fmt, l, e, h, inter, args.seed)
        gen = torch.Generator().manual_seed(args.seed + 1)
        dt = torch.float16 if fmt == "fp16" else torch.bfloat16
        x = torch.randn(batch, h, generator=gen).to(dt)
        ids = torch.stack([torch.randperm(e, generator=gen)[:topk] for _ in range(batch)]).int()
        w = torch.rand(batch, topk, generator=gen)
        w /= w.sum(-1, keepdim=True)
        data = {"banks": {k: v.cpu() for k, v in host.items()}, "x": x, "ids": ids, "weights": w,
                "seed": args.seed, "format": fmt}
        torch.save(data, datafile)
    x, ids, w = (data[name].to("cuda") for name in ("x", "ids", "weights"))
    manifest = {name: {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
                       "bytes": t.numel() * t.element_size(), "sha256": checksum(t)} for name, t in host.items()}
    manifest.update({name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype), "sha256": checksum(data[name])}
                     for name in ("x", "ids", "weights")})
    (case / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    math = lambda value, banks: compute(fmt, value, ids, w, banks, prefill=args.mode == "prefill")
    # Separate ring buffers per variant keep captures and output checks independent.
    variants = ["serial_sm", "serial_dma", "overlap_sm", "overlap_sm_narrow", "overlap_sm_weighted", "overlap_dma", "resident", "prefix1_sm", "prefix1_dma"]
    if args.mode == "decode":
        variants.extend(("staged_sm", "staged_dma"))
    records, graphs, outputs, rings = {}, {}, {}, {}
    keepalive = []
    for name in variants:
        resident_layers = l if name == "resident" else (1 if name.startswith("prefix1") else 0)
        nbuf = resident_layers + min(2, l - resident_layers)
        buffer_of = [layer if layer < resident_layers else resident_layers + (layer - resident_layers) % (nbuf - resident_layers)
                     for layer in range(l)]
        cache = {role: torch.empty((nbuf, *tensor.shape[1:]), device="cuda", dtype=tensor.dtype) for role, tensor in host.items()}
        for tensor in cache.values():
            raw(tensor).fill_(165)
        rings[name] = cache
        stream = torch.cuda.Stream()
        ready = [torch.cuda.Event() for _ in range(nbuf)]
        free = [torch.cuda.Event() for _ in range(nbuf)]
        index = torch.arange(e, device="cuda", dtype=torch.int32)
        valid = torch.tensor([e], device="cuda", dtype=torch.int64)
        feat = torch.tensor([t[0, 0].numel() * t.element_size() for t in host.values()], device="cuda", dtype=torch.int64)
        source_ptrs = [torch.tensor([device_ptr(t[layer]) for t in host.values()], device="cuda", dtype=torch.int64) for layer in range(l)]
        dst_ptrs = [torch.tensor([t[buf].data_ptr() for t in cache.values()], device="cuda", dtype=torch.int64) for buf in range(nbuf)]
        plans = [ByteCopyPlan([t[buffer_of[layer]] for t in cache.values()], [t[layer] for t in host.values()]) for layer in range(l)]
        grouped = {}
        for group in ("gate_up", "down"):
            names = [role for role in host if role.startswith(group)]
            grouped[group] = {
                "plans": [ByteCopyPlan([cache[role][buffer_of[layer]] for role in names], [host[role][layer] for role in names]) for layer in range(l)],
                "src": [torch.tensor([device_ptr(host[role][layer]) for role in names], device="cuda", dtype=torch.int64) for layer in range(l)],
                "dst": [torch.tensor([cache[role][buf].data_ptr() for role in names], device="cuda", dtype=torch.int64) for buf in range(nbuf)],
                "feat": torch.tensor([host[role][0, 0].numel() * host[role].element_size() for role in names], device="cuda", dtype=torch.int64),
            }
        down_ready = [torch.cuda.Event() for _ in range(nbuf)]
        keepalive.append((stream, ready, free, down_ready, index, valid, feat, source_ptrs, dst_ptrs, plans, grouped))
        def copy(layer):
            buf = buffer_of[layer]
            if "sm" in name:
                fast_index_copy_multi_jit(dst_ptrs[buf], source_ptrs[layer], feat, index, index, valid,
                                          num_threads=256 if name.endswith(("narrow", "weighted")) else 1024,
                                          blocks_per_bank=2 if name.endswith("weighted") else (4 if name.endswith("narrow") else 8),
                                          weighted=name.endswith("weighted"))
            else:
                plans[layer].launch()
        def copy_group(layer, group):
            plan = grouped[group]
            if name.endswith("sm"):
                fast_index_copy_multi_jit(plan["dst"][buffer_of[layer]], plan["src"][layer], plan["feat"], index, index, valid)
            else:
                plan["plans"][layer].launch()
        preload_ms = 0.0
        if resident_layers:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for layer in range(resident_layers):
                copy(layer)
            stop.record()
            stop.synchronize()
            preload_ms = start.elapsed_time(stop)

        def forward():
            out = x
            stages = []
            main = torch.cuda.current_stream()
            if name.startswith("prefix1"):
                stream.wait_stream(main)
                def enqueue_next(layer):
                    buf = buffer_of[layer]
                    with torch.cuda.stream(stream):
                        if layer >= nbuf:
                            stream.wait_event(free[buf])
                        copy(layer)
                        ready[buf].record()
                for layer in range(l):
                    buf = buffer_of[layer]
                    if layer >= resident_layers:
                        main.wait_event(ready[buf])
                    if layer + 1 < l:
                        enqueue_next(layer + 1)
                    out = math(out, {role: t[buf] for role, t in cache.items()})
                    stages.append(out)
                    free[buf].record(main)
                main.wait_stream(stream)
            elif name.startswith("overlap"):
                stream.wait_stream(main)
                with torch.cuda.stream(stream):
                    copy(0)
                    ready[0].record()
                for layer in range(l):
                    buf = layer % nbuf
                    main.wait_event(ready[buf])
                    if layer + 1 < l:
                        next_buf = (layer + 1) % nbuf
                        with torch.cuda.stream(stream):
                            if layer + 1 >= nbuf:
                                stream.wait_event(free[next_buf])
                            copy(layer + 1)
                            ready[next_buf].record()
                    out = math(out, {role: t[buf] for role, t in cache.items()})
                    stages.append(out)
                    free[buf].record(main)
                main.wait_stream(stream)
            elif name.startswith("staged"):
                stream.wait_stream(main)
                def enqueue(layer):
                    buf = layer % nbuf
                    with torch.cuda.stream(stream):
                        if layer >= nbuf:
                            stream.wait_event(free[buf])
                        copy_group(layer, "gate_up")
                        ready[buf].record()
                        copy_group(layer, "down")
                        down_ready[buf].record()
                enqueue(0)
                for layer in range(l):
                    buf = layer % nbuf
                    main.wait_event(ready[buf])
                    if layer + 1 < l:
                        enqueue(layer + 1)
                    view = {role: t[buf] for role, t in cache.items()}
                    act = gate_stage(fmt, out, ids, w, view)
                    main.wait_event(down_ready[buf])
                    out = down_stage(fmt, act, ids, w, view)
                    stages.append(out)
                    free[buf].record(main)
                main.wait_stream(stream)
            else:
                for layer in range(l):
                    if name != "resident":
                        copy(layer)
                    out = math(out, {role: t[buffer_of[layer]] for role, t in cache.items()})
                    stages.append(out)
            return stages

        try:
            graph, out, peak = capture(forward)
        except Exception:
            torch.cuda.synchronize()
            raise
        graphs[name] = graph
        outputs[name] = out
        records[name] = {"ring_buffers": nbuf, "cache_bytes": sum(t.numel() * t.element_size() for t in cache.values()),
                         "buffer_of_layer": buffer_of, "resident_layers": resident_layers,
                         "preload_ms": preload_ms, "peak_capture_extra_bytes": peak}
    proofs = []
    base = outputs["serial_sm"]
    for name, stages in outputs.items():
        for layer, (reference, output) in enumerate(zip(base, stages)):
            if not torch.isfinite(output).all():
                raise AssertionError(f"{fmt}/{name}: nonfinite output")
            proofs.append(exact(reference, output, f"{name}_layer_{layer}"))
        cache = rings[name]
        nbuf = records[name]["ring_buffers"]
        for buf in range(nbuf):
            last = max(i for i in range(l) if records[name]["buffer_of_layer"][i] == buf)
            for role, tensor in cache.items():
                proofs.append(exact(host[role][last].to("cuda"), tensor[buf], f"{name}_buffer_{buf}_{role}"))
    torch.save({"stages": [v.cpu() for v in base]}, case / "outputs_before.pt")
    timings = time_graphs(graphs, args.repeats)
    for name in timings:
        timings[name]["amortized_forward_ms"] = {
            str(steps): timings[name]["median_ms"] + records[name]["preload_ms"] / steps
            for steps in (1, 16, 256)
        }
    # Recheck after alternating replays: stale ready/release events must not hide a race.
    for name, graph in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        for layer, (reference, output) in enumerate(zip(base, outputs[name])):
            proofs.append(exact(reference, output, f"replay_{name}_layer_{layer}"))
    traces = {}
    if args.trace:
        for name in ("serial_sm", "overlap_sm", "overlap_sm_narrow", "overlap_sm_weighted", "overlap_dma", "staged_sm", "staged_dma"):
            if name not in graphs:
                continue
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                graphs[name].replay()
                torch.cuda.synchronize()
            path = args.output.with_name(args.output.stem + "_" + fmt + "_" + name + "_trace.json")
            profile.export_chrome_trace(str(path))
            traces[name] = str(path)
    micro = {}
    if args.measure_pair:
        cache = rings["resident"]
        math_graph, _, _ = capture(lambda: math(x, {role: t[0] for role, t in cache.items()}))
        pair_graphs = {"compute_alone": math_graph}
        retain = []
        for label, threads, blocks in (("sm", 1024, 8), ("sm_narrow", 256, 4), ("dma", 0, 0)):
            targets = [torch.empty_like(t[0], device="cuda") for t in host.values()]
            plan = ByteCopyPlan(targets, [t[1] for t in host.values()])
            srcptr = torch.tensor([device_ptr(t[1]) for t in host.values()], device="cuda", dtype=torch.int64)
            dstptr = torch.tensor([t.data_ptr() for t in targets], device="cuda", dtype=torch.int64)
            stream = torch.cuda.Stream()
            def move():
                if label == "dma":
                    plan.launch()
                else:
                    fast_index_copy_multi_jit(dstptr, srcptr, feat, index, index, valid,
                                              num_threads=threads, blocks_per_bank=blocks)
            graph, _, _ = capture(move)
            pair_graphs[label + "_copy_alone"] = graph
            def concurrent():
                main = torch.cuda.current_stream()
                stream.wait_stream(main)
                with torch.cuda.stream(stream):
                    move()
                out = math(x, {role: t[0] for role, t in cache.items()})
                main.wait_stream(stream)
                return out
            graph, out, _ = capture(concurrent)
            proofs.append(exact(outputs["resident"][0], out, label + "_concurrent_output"))
            pair_graphs[label + "_copy_and_compute"] = graph
            retain.append((targets, plan, srcptr, dstptr, stream))
        micro = time_graphs(pair_graphs, args.repeats)
    bytes_per_layer = sum(t[0].numel() * t.element_size() for t in host.values())
    return {"format": fmt, "input_file": str(datafile), "layers": l, "experts": e, "hidden": h, "intermediate": inter,
            "batch": batch, "topk": topk, "mode": args.mode, "bytes_per_layer": bytes_per_layer, "proof": proofs,
            "variants": {name: {**records[name], **timings[name]} for name in variants},
            "traces": traces,
            "copy_compute_pair": micro,
            "note": "full-layer lookahead, no future route oracle; same math kernels and chained layer outputs; resident preload/setup excluded"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--formats", default="bf16,fp16,fp8_block,nvfp4,mxfp4,ds_fp4,q4_0")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--seed", type=int, default=92307)
    p.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--measure-pair", action="store_true")
    args = p.parse_args()
    if args.layers < 2 or args.experts < args.topk:
        raise ValueError("need >=2 layers and experts >= topk")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "command": sys.argv, "timing": "alternating captured forwards; CUDA events cover fork/join of both streams",
              "scope": "standalone byte-copy operators and unchanged MoE kernels; no engine/server/scheduler changes", "cases": []}
    for fmt in args.formats.split(","):
        row = experiment(fmt.strip(), args)
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"format": row["format"], "bytes_per_layer": row["bytes_per_layer"],
                          "times_ms": {k: v["median_ms"] for k, v in row["variants"].items()},
                          "exact_comparisons": len(row["proof"])}), flush=True)
        torch.cuda.empty_cache()
    print(f"All {len(result['cases'])} format cases passed byte checks", flush=True)


if __name__ == "__main__":
    main()
