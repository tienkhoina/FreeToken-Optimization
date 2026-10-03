"""Compare the existing tensor copy loop with a prepared native byte-copy operator."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from bench_copy_overlap import exact, make_banks, raw
from freetoken.kernel.copy_plan import ByteCopyPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--formats", default="bf16,fp16,fp8_block,nvfp4,mxfp4,ds_fp4,q4_0,nvfp4_marlin,nvfp4_b12x")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"scope": "unchanged cudaMemcpyAsync byte semantics, one prepared native call instead of a tensor loop", "cases": []}
    for fmt in args.formats.split(","):
        host = make_banks(fmt, 1, 8, 2048, 1024, 92811)
        sources = [t[0] for t in host.values()]
        baseline = [torch.empty_like(s, device="cuda") for s in sources]
        candidate = [torch.empty_like(s, device="cuda") for s in sources]
        for dst in baseline + candidate:
            raw(dst).fill_(165)
        plan = ByteCopyPlan(candidate, sources)
        def old():
            for dst, src in zip(baseline, sources):
                dst.copy_(src, non_blocking=True)
        old()
        plan.launch()
        torch.cuda.synchronize()
        proof = [exact(a, b, role) for role, a, b in zip(host, baseline, candidate)]
        path = args.output.parent / "inputs" / (fmt + "_copy_operator.pt")
        path.parent.mkdir(exist_ok=True)
        torch.save({"sources": [s.cpu() for s in sources], "outputs_before": [d.cpu() for d in baseline]}, path)
        samples = {label: {"host_enqueue_ms": [], "device_completion_ms": []} for label in ("before", "after")}
        for fun in (old, plan.launch):
            for _ in range(3):
                fun()
        torch.cuda.synchronize()
        for i in range(args.repeats):
            variants = (("before", old), ("after", plan.launch))
            for label, fun in variants if i % 2 == 0 else variants[::-1]:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                start = time.perf_counter_ns()
                fun()
                samples[label]["host_enqueue_ms"].append((time.perf_counter_ns() - start) / 1e6)
                end.record()
                end.synchronize()
                samples[label]["device_completion_ms"].append(begin.elapsed_time(end))
        graphs = {}
        for label, fun in (("before", old), ("after", plan.launch)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fun()
            graph.replay()
            torch.cuda.synchronize()
            graphs[label] = graph
        graph_samples = {"before": [], "after": []}
        for i in range(args.repeats):
            for label in ("before", "after") if i % 2 == 0 else ("after", "before"):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                graphs[label].replay()
                end.record()
                end.synchronize()
                graph_samples[label].append(begin.elapsed_time(end))
        proof.extend(exact(a, b, "graph_" + role) for role, a, b in zip(host, baseline, candidate))
        row = {"format": fmt, "bytes": plan.bytes, "banks": len(sources), "input_output_file": str(path), "proof": proof,
               "variants": {label: {**samples[label], "graph_samples_ms": graph_samples[label],
                                    "host_enqueue_median_ms": statistics.median(samples[label]["host_enqueue_ms"]),
                                    "device_completion_median_ms": statistics.median(samples[label]["device_completion_ms"]),
                                    "graph_median_ms": statistics.median(graph_samples[label])} for label in samples}}
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"format": fmt, "bytes": plan.bytes, "medians": {label: {k: v for k, v in values.items() if k.endswith("median_ms")} for label, values in row["variants"].items()}}), flush=True)
    print("All prepared-copy operator outputs are byte-exact", flush=True)


if __name__ == "__main__":
    main()
