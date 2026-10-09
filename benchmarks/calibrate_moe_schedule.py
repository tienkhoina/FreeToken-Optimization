"""Measure compiled CPU/GPU operators and recommend a coarse miss split for this machine."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from dataclasses import replace
from types import SimpleNamespace

import torch

from freetoken.moe.benchbw import WORKLOADS, _cpu_moe_bank_sources
from freetoken.moe.cpu_executor import CpuMoeExecutor
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.schedule_profile import merge_profiles, recommend_fetch_counts


def calibration_routes(tokens, top_k, misses, working_set, offset=0):
    if min(tokens, top_k, misses) <= 0 or misses > min(tokens * top_k, working_set):
        raise ValueError("Expert count exceeds query routes or working set")
    active = min(top_k, misses)
    slots = torch.arange(tokens * active, dtype=torch.int32).reshape(tokens, active) % misses
    sources = (torch.arange(misses, dtype=torch.int32) + offset) % working_set
    return slots, sources[slots.long()], sources


def resolve_workload(args):
    workload = WORKLOADS[args.workload]
    if args.workload in ("dsv4", "dsv4-nvfp4") and args.format == "nvfp4":
        workload = WORKLOADS["dsv4-nvfp4"]
        if args.model is not None:
            from freetoken.models.deepseek_v4.args import load_args

            cfg = load_args(str(args.model))
            if args.layer < 0 or args.layer >= cfg.n_layers:
                raise ValueError("Calibration layer is outside DeepSeek's decoder")
            workload = replace(workload, hidden=cfg.dim, inter=cfg.moe_inter_dim,
                               experts=cfg.n_routed_experts, top_k=cfg.n_activated_experts,
                               swiglu_limit=cfg.swiglu_limit,
                               activation="swiglu_clamp" if cfg.swiglu_limit > 0 else "silu")
    if args.format not in workload.formats:
        raise ValueError(f"{args.workload} has no {args.format} workload")
    return workload


def load_banks(args, workload):
    if args.model is None:
        banks = _cpu_moe_bank_sources(args.format, workload.hidden, workload.inter, args.experts)
        for name, tensor in banks.items():
            if tensor.dtype == torch.uint8 and "scale" not in name:
                tensor.random_(0, 256)
            elif tensor.dtype == torch.uint8 and args.format == "nvfp4":
                tensor.fill_(56)
        return {name: [tensor] for name, tensor in banks.items()}, "synthetic finite banks, production layouts"
    if args.format == "nvfp4" and args.workload in ("qwen3.8-flash-next", "dsv4", "dsv4-nvfp4"):
        from safetensors import safe_open
        from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

        weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
        kernel = TritonNvfp4MoEKernel()
        cfg = SimpleNamespace(hidden=workload.hidden, intermediate=workload.inter)
        layout = kernel.layout(cfg)
        banks = {name: torch.empty((args.experts, *spec.shape), dtype=spec.dtype, pin_memory=torch.cuda.is_available())
                 for name, spec in layout.items()}
        deepseek = args.workload in ("dsv4", "dsv4-nvfp4")
        projections = (("w1", "gate"), ("w3", "up"), ("w2", "down")) if deepseek else (
            ("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down"))
        suffixes = (("weight_packed", ""), ("weight_scale", "_scale"), ("weight_global_scale", "_global")) if deepseek else (
            ("weight", ""), ("weight_scale", "_scale"), ("weight_scale_2", "_global"))
        handles = {}
        try:
            for expert in range(args.experts):
                pieces = {}
                for projection, role in projections:
                    for suffix, end in suffixes:
                        root = f"layers.{args.layer}.ffn" if deepseek else f"model.language_model.layers.{args.layer}.mlp"
                        name = f"{root}.experts.{expert}.{projection}.{suffix}"
                        shard = weight_map[name]
                        if shard not in handles:
                            handles[shard] = safe_open(args.model / shard, framework="pt", device="cpu")
                        tensor = handles[shard].get_tensor(name)
                        if deepseek and end == "_global":
                            tensor = tensor.float().reciprocal().to(torch.float16)
                        pieces[role + end] = tensor.unsqueeze(0)
                kernel.pack(pieces, cfg, {name: bank[expert:expert + 1] for name, bank in banks.items()})
        finally:
            handles.clear()
        return {name: [tensor] for name, tensor in banks.items()}, f"actual {'DeepSeek' if deepseek else 'Qwen'} NVFP4 checkpoint layer {args.layer}"
    if args.format != "mxfp4_triton":
        raise ValueError("Actual checkpoint reader currently supports GPT-OSS MXFP4; other formats use production-layout synthetic banks")
    from safetensors import safe_open

    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    roles = {"gate_up_blocks": "gate_up_proj_blocks", "gate_up_scales": "gate_up_proj_scales",
             "gate_up_bias": "gate_up_proj_bias", "down_blocks": "down_proj_blocks",
             "down_scales": "down_proj_scales", "down_bias": "down_proj_bias"}
    banks = {}
    for role, suffix in roles.items():
        key = f"model.layers.{args.layer}.mlp.experts.{suffix}"
        with safe_open(args.model / weight_map[key], framework="pt", device="cpu") as source:
            tensor = source.get_slice(key)[:args.experts].clone()
        if role.endswith("blocks"):
            tensor = tensor.reshape(args.experts, tensor.shape[1], -1).permute(0, 2, 1).contiguous()
        elif role.endswith("scales"):
            tensor = tensor.permute(0, 2, 1).contiguous()
        banks[role] = [tensor.pin_memory()]
    return banks, f"actual checkpoint layer {args.layer}"


def gpu_compute(fmt, x, weights, ids, banks, workload, inactive_expert=-1):
    if fmt == "mxfp4_triton":
        from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts, run_mxfp4_prefill_experts_t

        run = run_mxfp4_prefill_experts_t if x.shape[0] > 1 else run_mxfp4_splitk_decode_experts
        return run(x, weights, ids, *banks, top_k=ids.shape[1],
                                              hidden_act_alpha=workload.swiglu_alpha, swiglu_limit=workload.swiglu_limit)
    if fmt == "bf16":
        from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

        run = fused_experts_impl if x.shape[0] > 1 else fused_experts_decode_impl
        return run(x.clone(), *banks, weights, ids, workload.activation, False,
                                         workload.swiglu_alpha if workload.activation == "gpt_oss_swiglu" else 1.0,
                                         workload.swiglu_limit or float("inf"))
    if fmt == "ds_fp4":
        from freetoken.moe.fused_ds_fp4 import routed_experts_fp4, routed_experts_fp4_prefill

        if x.shape[0] > 1:
            return routed_experts_fp4_prefill(x, ids, weights, *banks, workload.swiglu_limit or float("inf"), banks[0].shape[0])
        return routed_experts_fp4(x, ids, weights, *banks, workload.swiglu_limit or float("inf"))
    if fmt == "nvfp4":
        from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4, fused_experts_decode_nvfp4_marlin

        if x.shape[0] == 1:
            return fused_experts_decode_nvfp4_marlin(x, *banks, weights, ids, workload.activation, False,
                                                   workload.swiglu_alpha, workload.swiglu_limit or float("inf"), inactive_expert)
        return fused_experts_nvfp4(x, *banks, weights, ids, banks[0].shape[0], workload.activation, False,
                                   workload.swiglu_alpha, workload.swiglu_limit or float("inf"), inactive_expert)
    raise ValueError(fmt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, default="gpt-oss-120b")
    parser.add_argument("--format", choices=["bf16", "mxfp4_triton", "nvfp4", "ds_fp4"], default="mxfp4_triton")
    parser.add_argument("--model", type=Path, help="Optional local GPT-OSS MXFP4, Qwen3.8 NVFP4 or DeepSeek NVFP4 weights")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", type=int, default=32, help="Working-set expert count; choose enough to exceed CPU LLC")
    parser.add_argument("--threads", type=int, default=0, help="0 uses every CPU in process affinity")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--tokens", type=int, default=1, help="Query tokens sharing each expert; use a small prefill sample separately")
    parser.add_argument("--max-misses", type=int, help="Maximum distinct missing experts; default top_k for decode, up to working-set size for prefill")
    parser.add_argument("--merge-profiles", type=Path, nargs="+", help="Merge measured decode/prefill JSON files without GPU execution")
    parser.add_argument("--cpu-timing-percentile", type=int, choices=[50, 75, 90], default=90,
                        help="Use a conservative CPU graph percentile for miss assignment when workers are noisy")
    parser.add_argument("--minimum-split-gain", type=float, default=0.15,
                        help="Require this predicted latency reduction before assigning misses to the CPU")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.merge_profiles:
        if args.output.exists():
            raise FileExistsError(args.output)
        profiles = [json.loads(path.read_text()) for path in args.merge_profiles]
        result = merge_profiles(profiles, WORKLOADS[args.workload].experts)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    if args.tokens < 1 or args.repeats < 5 or not 0 <= args.minimum_split_gain < 1:
        raise ValueError("tokens must be positive, repeats >=5 and minimum split gain in [0,1)")
    if args.output.exists():
        raise FileExistsError(f"Preserving old result: {args.output}")
    workload = resolve_workload(args)
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    threads = args.threads or len(allowed)
    args.experts = min(workload.experts, args.experts)
    if args.experts < workload.top_k:
        raise ValueError("working set must hold at least top_k experts")
    coverage = min(args.experts, args.tokens * workload.top_k)
    max_misses = args.max_misses or coverage
    if max_misses < 1 or max_misses > coverage:
        raise ValueError("max-misses exceeds the query's distinct-expert coverage")
    torch.manual_seed(3901)
    banks, source_note = load_banks(args, workload)
    dev = torch.device("cuda")
    cache = OffloadMoeCache(1, args.experts, args.experts, dev, quant_format=args.format,
                           native_schedule=args.format == "nvfp4")
    cache.set_bank_sources(banks)
    cache._pending_src_layer = 0
    cache.src_indices[:args.experts].copy_(torch.arange(args.experts, device=dev, dtype=torch.int32))
    cache.evict_slots[:args.experts].copy_(torch.arange(args.experts, device=dev, dtype=torch.int32))
    cache.num_indices.fill_(args.experts)
    cache.copy_missing()
    torch.cuda.synchronize()
    ex = CpuMoeExecutor(cache, top_k=workload.top_k, activation=workload.activation, apply_router_weight_on_input=False,
                        num_threads=threads, max_tokens=args.tokens, device=dev, swiglu_alpha=workload.swiglu_alpha,
                        swiglu_limit=workload.swiglu_limit)
    io = ex._io_for(args.tokens)
    io["x"].copy_(torch.randn(args.tokens, workload.hidden, dtype=torch.bfloat16) * 0.1)
    io["w"].fill_(1 / workload.top_k)
    task = ex._task_for(0, args.tokens)
    x = io["x"].cuda()
    cpu_ids_gpu = torch.arange(workload.top_k, device=dev, dtype=torch.int32).view(1, workload.top_k).expand(args.tokens, -1).contiguous()
    cpu_weights_gpu = io["w"].cuda()
    ex.decode(0, x, cpu_weights_gpu, cpu_ids_gpu)
    torch.cuda.synchronize()
    cpu_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cpu_graph):
        cpu_result = ex.decode(0, x, cpu_weights_gpu, cpu_ids_gpu)
    copy_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(copy_graph):
        cache.copy_missing()
    table = []
    for count in range(1, max_misses + 1):
        slots, _, _ = calibration_routes(args.tokens, workload.top_k, count, args.experts)
        if args.format == "nvfp4":
            ids = torch.full((args.tokens,workload.top_k),args.experts,dtype=torch.int32,device=dev)
            ids[:,:slots.shape[1]].copy_(slots.to(dev))
            weights = torch.where(ids < args.experts, 1 / workload.top_k, 0.0)
            gpu_banks = cache.scheduled_bank_views()
            inactive = args.experts
        else:
            ids = slots.to(dev)
            weights = torch.full(ids.shape, 1 / workload.top_k, device=dev)
            gpu_banks, inactive = cache.bank_views(), -1
        for _ in range(2):
            gpu_compute(args.format, x, weights, ids, gpu_banks, workload, inactive)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = gpu_compute(args.format, x, weights, ids, gpu_banks, workload, inactive)
        cache.num_indices.fill_(count)
        samples = {"cpu_ms": [], "cpu_graph_ms": [], "copy_ms": [], "gpu_ms": []}
        for repeat in range(args.repeats + 5):
            _, source_ids, rows = calibration_routes(args.tokens, workload.top_k, count, args.experts,
                                                       repeat * workload.top_k)
            active = source_ids.shape[1]
            io["ids"].fill_(-1)
            io["ids"][:, :active].copy_(source_ids)
            cache.src_indices[:count].copy_(rows.to(dev))
            cpu_ids_gpu.fill_(-1)
            cpu_ids_gpu[:, :active].copy_(source_ids.to(dev))
            torch.cuda.synchronize()
            start = time.perf_counter()
            ex._ext.run_task(task)
            cpu_ms = (time.perf_counter() - start) * 1000
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            cpu_graph.replay()
            end.record()
            end.synchronize()
            cpu_graph_ms = begin.elapsed_time(end)
            begin.record()
            copy_graph.replay()
            end.record()
            end.synchronize()
            copy_ms = begin.elapsed_time(end)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            if repeat >= 5:
                samples["cpu_ms"].append(cpu_ms)
                samples["cpu_graph_ms"].append(cpu_graph_ms)
                samples["copy_ms"].append(copy_ms)
                samples["gpu_ms"].append(begin.elapsed_time(end))
        cpu_output, gpu_output = cpu_result.float().cpu(), out.float().cpu()
        if not torch.isfinite(cpu_output).all() or not torch.isfinite(gpu_output).all():
            raise ValueError("Calibration produced a nonfinite expert output")
        difference = gpu_output - cpu_output
        table.append({"experts": count, **{name: statistics.median(values) for name, values in samples.items()}, "samples": samples,
                      "cpu_gpu_max_abs_error": difference.abs().max().item(),
                      "cpu_gpu_relative_l2_error": (difference.norm() / gpu_output.norm().clamp_min(1e-12)).item()})
        print(json.dumps({k: v for k, v in table[-1].items() if k != "samples"}), flush=True)
    by_count = {row["experts"]: row for row in table}
    for row in table:
        ordered = sorted(row["samples"]["cpu_graph_ms"])
        row["cpu_graph_policy_ms"] = ordered[min(len(ordered) - 1, (len(ordered) - 1) * args.cpu_timing_percentile // 100)]
    fetch_counts, estimates = recommend_fetch_counts(table, max_misses, args.cpu_timing_percentile, args.minimum_split_gain)
    last = by_count[max_misses]
    simple_fraction = last["cpu_graph_policy_ms"] / (last["cpu_graph_policy_ms"] + last["copy_ms"] + last["gpu_ms"])
    result = {"format": args.format, "workload": args.workload, "query_tokens": args.tokens, "hidden": workload.hidden, "intermediate": workload.inter,
              "activation": workload.activation, "swiglu_alpha": workload.swiglu_alpha, "swiglu_limit": workload.swiglu_limit,
              "top_k": workload.top_k, "max_misses": max_misses,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "cpu_affinity": allowed, "cpu_workers": ex.num_threads, "cpu_isa": ex.isa, "source": source_note,
              "cpu_timing_percentile": args.cpu_timing_percentile,
              "minimum_split_gain": args.minimum_split_gain,
              "table": table, "recommend_fetch_counts": fetch_counts, "split_estimates": estimates,
              "coarse_gpu_fraction": simple_fraction,
              "method": "CPU C++ task and complete CPU branch graph latency (D2H+handshake+native workers+H2D), all detected workers by default; GPU math/copy CUDA graph replay after JIT/warmup. No eager GPU math in timed regions. Split uses CPU graph latency; still a rough independent-duration estimate, not a continuous search or integrated optimum."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Suggested GPU fetch counts for 0..{max_misses} misses: {fetch_counts}", flush=True)
    print(f"Coarse large-miss GPU fraction: {simple_fraction:.3f}; validate prefill separately by tokens per expert.", flush=True)


if __name__ == "__main__":
    main()
