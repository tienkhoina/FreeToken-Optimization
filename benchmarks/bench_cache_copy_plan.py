"""A/B the original and native full-layer copy at the existing cache boundary."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import types

import torch

from bench_copy_overlap import compute, exact, make_banks
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.legacy_format import canonical_role


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    old = types.ModuleType("freetoken.moe._copy_task_baseline_cache")
    old.__package__ = "freetoken.moe"
    sys.modules[old.__name__] = old
    path = args.baseline_root / "python/freetoken/moe/offload_cache.py"
    exec(compile(path.read_text(), str(path), "exec"), old.__dict__)
    result = {"scope": "same cache/prefetch/event policy and unchanged expert math; only full-bank copy enqueue changed", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tags = {"bf16": "bf16", "fp16": "bf16", "fp8_block": "fp8_block", "nvfp4": "nvfp4",
            "mxfp4": "mxfp4_triton", "ds_fp4": "ds_fp4", "q4_0": "q4_0"}
    for fmt, tag in tags.items():
        layers, experts, h, inter, batch, topk = 4, 8, 1024, 512, 128, 4
        host = make_banks(fmt, layers, experts, h, inter, 92309)
        sources = {name: list(t.unbind()) for name, t in host.items()}
        caches = {label: cls(layers, experts, 2 * experts, torch.device("cuda"), prefill_overlap=True, quant_format=tag)
                  for label, cls in (("before", old.OffloadMoeCache), ("after", OffloadMoeCache))}
        for cache in caches.values():
            cache.set_bank_sources(sources)
        torch.manual_seed(92310)
        dtype = torch.float16 if fmt == "fp16" else torch.bfloat16
        x = torch.randn(batch, h, device="cuda", dtype=dtype)
        ids = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(batch)]).int()
        w = torch.rand(batch, topk, device="cuda")
        w /= w.sum(-1, keepdim=True)
        def forward(cache):
            cache.begin_prefill()
            output = x
            stages = []
            for layer in range(layers):
                cache.prefetch_prefill_layer(layer)
                if layer + 1 < layers:
                    cache.prefetch_prefill_layer(layer + 1)
                banks = dict(zip((canonical_role(n) for n in cache.bank_schema), cache.wait_prefill_layer(layer)))
                output = compute(fmt, output, ids, w, banks, prefill=True)
                stages.append(output)
                cache.release_prefill_layer(layer)
            return stages
        for cache in caches.values():
            forward(cache)
        torch.cuda.synchronize()
        before = forward(caches["before"])
        after = forward(caches["after"])
        torch.cuda.synchronize()
        proof = [exact(a, b, f"layer_{i}") for i, (a, b) in enumerate(zip(before, after))]
        for name in ("slot_for_id", "id_of_slot", "usage"):
            proof.append(exact(getattr(caches["before"], name), getattr(caches["after"], name), name))
        datafile = args.output.parent / "inputs" / (fmt + "_cache_forward.pt")
        torch.save({"banks": {n: t.cpu() for n, t in host.items()}, "x": x.cpu(), "ids": ids.cpu(), "weights": w.cpu(),
                    "outputs_before": [t.cpu() for t in before]}, datafile)
        samples = {label: {"completion_ms": [], "host_enqueue_ms": []} for label in caches}
        for i in range(args.repeats):
            for label in ("before", "after") if i % 2 == 0 else ("after", "before"):
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                start = time.perf_counter_ns()
                output = forward(caches[label])
                samples[label]["host_enqueue_ms"].append((time.perf_counter_ns() - start) / 1e6)
                end.record()
                end.synchronize()
                samples[label]["completion_ms"].append(begin.elapsed_time(end))
                del output
        row = {"format": fmt, "layers": layers, "experts": experts, "hidden": h, "intermediate": inter,
               "batch": batch, "proof": proof, "saved_tensors": str(datafile),
               "variants": {label: {**values, "completion_median_ms": statistics.median(values["completion_ms"]),
                                    "host_enqueue_median_ms": statistics.median(values["host_enqueue_ms"])} for label, values in samples.items()}}
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"format": fmt, "medians": {label: {k: v for k, v in values.items() if k.endswith("median_ms")} for label, values in row["variants"].items()}}), flush=True)
    print("All existing-cache forward stages match byte for byte", flush=True)


if __name__ == "__main__":
    main()
