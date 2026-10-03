"""A/B native index-copy schedules alone and beside unchanged expert compute."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from tvm_ffi.cpp import load_inline

from bench_copy_overlap import capture, compute, exact, make_banks, raw, time_graphs
from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.kernel.pinned import device_ptr

REPO = Path(__file__).resolve().parents[1]


def variants():
    header = REPO / "python/freetoken/kernel/csrc/jit/fast_index_copy_scheduled.cuh"
    stamp = hashlib.sha256(header.read_bytes()).hexdigest()[:12]
    specs = [(256, 2, 1, n) for n in (0, 8, 16, 24, 32)] + [(256, 2, 4, n) for n in (0, 16, 24)] + [(256, 4, 1, 0), (512, 4, 1, 0), (1024, 8, 1, 0), (1024, 64, 1, 0)]
    wrappers = [f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(v{t}_{b}_{u}_{n}, (&ScheduledIndexCopyKernel<{t},{b},{u},{n}>::run));" for t, b, u, n in specs]
    module = load_inline(name="freetoken_copy_schedule_" + stamp,
                         cuda_sources=[f'#include "{header}"', *wrappers],
                         extra_include_paths=[str(REPO / "python/freetoken/kernel/csrc/include")],
                         extra_cuda_cflags=["-std=c++20", "-O3", "--expt-relaxed-constexpr"])
    return module, {f"scheduled_{t}_{b}_{u}_min{n}": getattr(module, f"v{t}_{b}_{u}_{n}") for t, b, u, n in specs}


def case(fmt, host, rows, funs, args):
    data_path = args.output.parent / "inputs" / f"schedule_{fmt}_E8_H2048_I1024.pt"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        data = torch.load(data_path, weights_only=True)
        banks = {k: v.pin_memory() for k, v in data["banks"].items()}
    else:
        banks = make_banks(fmt, 2, 8, 2048, 1024, 918721)
        generator = torch.Generator().manual_seed(918722)
        x = torch.randn(128, 2048, generator=generator).bfloat16()
        if fmt == "fp16":
            x = x.half()
        ids = torch.stack([torch.randperm(8, generator=generator)[:4] for _ in range(128)]).int()
        w = torch.rand(128, 4, generator=generator)
        w /= w.sum(-1, keepdim=True)
        data = {"banks": {k: v.cpu() for k, v in banks.items()}, "x": x, "ids": ids, "w": w}
        torch.save(data, data_path)
    source = [t[1] if host else t[1].to("cuda") for t in banks.values()]
    dst = [torch.empty_like(t[0], device="cuda") for t in banks.values()]
    for t in dst:
        raw(t).fill_(165)
    planned = max(1, rows)
    if args.offset + planned > 8:
        raise ValueError("copy rows exceed saved bank")
    si = (torch.arange(planned, device="cuda", dtype=torch.int32) + args.offset).flip(0)
    di = torch.arange(planned, device="cuda", dtype=torch.int32) + args.offset
    num = torch.tensor([rows], device="cuda", dtype=torch.int64)
    ptrs = [torch.tensor(values, device="cuda", dtype=torch.int64) for values in (
        [t.data_ptr() for t in dst], [device_ptr(t) for t in source],
        [t[0].numel() * t.element_size() for t in source],
    )]
    inputs = (*ptrs, di, si, num)
    baseline = lambda: fast_index_copy_multi_jit(*inputs)
    baseline()
    torch.cuda.synchronize()
    expected = [t.clone() for t in dst]
    torch.save({"outputs_before": [t.cpu() for t in expected], "src_indices": si.cpu(), "dst_indices": di.cpu(), "num_indices": num.cpu()},
               data_path.with_name(data_path.stem + ("_host" if host else "_device") + f"_rows{rows}_before.pt"))
    proof = []
    graphs = {}
    funs = {"baseline_1024_8": lambda *x: fast_index_copy_multi_jit(*x),
            "baseline_1024_64": lambda *x: fast_index_copy_multi_jit(*x, blocks_per_bank=64),
            "uniform_256_4": lambda *x: fast_index_copy_multi_jit(*x, num_threads=256, blocks_per_bank=4),
            **funs}
    keepalive = []
    if args.concurrent:
        resident = {name: t[0].to("cuda") for name, t in banks.items()}
        x, ids, w = (data[name].to("cuda") for name in ("x", "ids", "w"))
        math = lambda: compute(fmt, x, ids, w, resident, prefill=args.prefill)
        graph, math_output, _ = capture(math)
        graphs["compute_alone"] = graph
    for name, func in funs.items():
        for t in dst:
            raw(t).fill_(165)
        func(*inputs)
        torch.cuda.synchronize()
        for role, a, b in zip(banks, expected, dst):
            proof.append(exact(a, b, name + "_" + role))
        graph, _, _ = capture(lambda: func(*inputs))
        graphs[name + "_copy"] = graph
        if args.concurrent:
            stream = torch.cuda.Stream()
            def joint():
                main = torch.cuda.current_stream()
                stream.wait_stream(main)
                with torch.cuda.stream(stream):
                    func(*inputs)
                result = math()
                main.wait_stream(stream)
                return result
            graph, out, _ = capture(joint)
            proof.append(exact(math_output, out, name + "_joint_output"))
            graphs[name + "_joint"] = graph
            keepalive.append(stream)
    times = time_graphs(graphs, args.repeats)
    for name, func in funs.items():
        graphs[name + "_copy"].replay()
        torch.cuda.synchronize()
        for role, a, b in zip(banks, expected, dst):
            proof.append(exact(a, b, name + "_graph_" + role))
    traces = {}
    if args.trace and args.concurrent:
        for name in ("baseline_1024_8_joint", "scheduled_256_2_1_min0_joint"):
            if name not in graphs:
                continue
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                graphs[name].replay()
                torch.cuda.synchronize()
            path = args.output.with_name(f"{args.output.stem}_{fmt}_{rows}_{name}_trace.json")
            profile.export_chrome_trace(str(path))
            traces[name] = str(path)
    base = times[f"baseline_1024_{args.baseline_blocks}_copy"]["median_ms"]
    jointbase = times.get(f"baseline_1024_{args.baseline_blocks}_joint", {}).get("median_ms")
    return {"format": fmt, "host_source": host, "rows": rows, "saved_inputs": str(data_path),
            "row_offset": args.offset, "hidden": 2048, "intermediate": 1024,
            "compute_batch": 128 if args.concurrent else None, "compute_mode": "prefill" if args.prefill else "decode",
            "bytes_copied": sum(t[0].numel() * t.element_size() for t in source) * rows,
            "timings": times, "proof": proof, "traces": traces,
            "ranked": sorted([{"variant": k, "copy_ms": times[k + "_copy"]["median_ms"],
                                "copy_speedup": base / times[k + "_copy"]["median_ms"],
                                **({"joint_ms": times[k + "_joint"]["median_ms"],
                                    "joint_speedup": jointbase / times[k + "_joint"]["median_ms"]} if args.concurrent else {})}
                               for k in funs], key=lambda x: x.get("joint_ms", x["copy_ms"]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--formats", default="nvfp4,mxfp4,bf16")
    p.add_argument("--rows", default="4,8")
    p.add_argument("--location", choices=("host", "device", "both"), default="host")
    p.add_argument("--concurrent", action="store_true")
    p.add_argument("--prefill", action="store_true")
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--baseline-blocks", type=int, choices=(8, 64), default=8)
    p.add_argument("--variants", help="comma-separated scheduled variants; default all candidates")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--offset", type=int, default=0, help="start copy at this bank row")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    module, funs = variants()
    if args.variants:
        wanted = set(args.variants.split(","))
        missing = wanted - funs.keys()
        if missing:
            raise ValueError(f"unknown variants: {sorted(missing)}")
        funs = {k: v for k, v in funs.items() if k in wanted}
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "command": sys.argv,
              "source_sha256": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                  REPO / "python/freetoken/kernel/csrc/jit/fast_index_copy_scheduled.cuh",
                  REPO / "python/freetoken/kernel/csrc/jit/fast_index_copy.cuh",
                  REPO / "python/freetoken/kernel/fast_index_copy.py")},
              "timing": "alternating CUDA graphs; both streams join before stop event; copy and compute independent", "cases": []}
    for fmt in args.formats.split(","):
        for host in ((True, False) if args.location == "both" else (args.location == "host",)):
            for rows in map(int, args.rows.split(",")):
                row = case(fmt, host, rows, funs, args)
                result["cases"].append(row)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"format": fmt, "host": host, "rows": rows, "best": row["ranked"][:4]}), flush=True)
    print("All scheduled-copy variants are byte-exact", flush=True)


if __name__ == "__main__":
    main()
