"""Identical byte payload/routing sequences from cold cache; policy learns online."""

import argparse
import json
from pathlib import Path

import torch

from bench_copy_overlap import exact
from freetoken.moe.offload_cache import OffloadMoeCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    l, e, c, routes, steps = 4, 16, 32, 4, 192
    g = torch.Generator().manual_seed(83067)
    banks = {name: [torch.randint(0, 256, (e, width), dtype=torch.uint8, generator=g).pin_memory() for _ in range(l)]
             for name, width in (("gate_up", 524288), ("down", 262144))}
    workloads = {}
    for kind in ("skew", "uniform", "shift"):
        seq = []
        for t in range(steps):
            for layer in range(l):
                if kind == "uniform":
                    ids = torch.randperm(e, generator=g)[:routes]
                else:
                    hot = 0 if kind == "skew" or t < steps // 2 else 8
                    # Repeated hot group interrupted by long scan bursts to stress recency-only eviction.
                    if t % 16 < 5:
                        ids = torch.tensor([hot, hot + 1, hot + 2, hot + 3])
                    else:
                        pool = torch.tensor([v for v in range(e) if v not in (hot, hot + 1, hot + 2, hot + 3)])
                        ids = pool[torch.randperm(pool.numel(), generator=g)[:routes]]
                seq.append((t, layer, ids.int()))
        workloads[kind] = seq
    saved = args.output.parent / "inputs.pt"
    torch.save({"banks": {k: [v.cpu() for v in vals] for k, vals in banks.items()},
                "workloads": {k: [(t, layer, ids) for t, layer, ids in seq] for k, seq in workloads.items()}}, saved)
    result = {"slots": c, "layers": l, "experts": e, "saved_tensors": str(saved), "cases": []}
    for workload, seq in workloads.items():
        graphs, caches, query_inputs, observations, out = {}, {}, {}, {}, {}
        proof = []
        for policy in ("lru", "adaptive_hot"):
            cache = OffloadMoeCache(l, e, c, torch.device("cuda"), cache_policy=policy, prefill_overlap=False)
            cache.set_bank_sources(banks)
            cache.collect_stats = True
            caches[policy] = cache
            ids = torch.zeros(routes, device="cuda", dtype=torch.int32)
            query_inputs[policy] = ids
            outputs = []
            graph_by_layer = {}
            counts = torch.zeros(l, device="cuda", dtype=torch.int64)
            for layer in range(l):
                ids.zero_()
                cache.ensure_experts(layer, ids)
                cache.copy_missing()
                ids.zero_()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    cache.ensure_experts(layer, ids)
                    cache.copy_missing()
                    payload = torch.cat([bank.index_select(0, ids.long()).reshape(routes, -1) for _, bank in cache.banks], dim=-1)
                    counts[layer].copy_(cache.num_indices.view(()))
                graph_by_layer[layer] = graph
                outputs.append(payload)
            cache.reset()
            cache.reset_stats()
            graphs[policy] = graph_by_layer
            out[policy] = outputs
            observations[policy] = {"misses": [], "layer_times_ms": [], "hot_count": []}
            for t, layer, query in seq:
                ids.copy_(query)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                graph_by_layer[layer].replay()
                end.record()
                end.synchronize()
                observations[policy]["layer_times_ms"].append(begin.elapsed_time(end))
                observations[policy]["misses"].append(int(counts[layer].item()))
                if policy == "adaptive_hot":
                    observations[policy]["hot_count"].append(int(cache._adaptive_hot.count.item()))
                if t in (0, 15, 31, 95, 96, 111, 191):
                    expected = torch.cat([banks[name][layer].index_select(0, query.long()).reshape(routes, -1) for name in banks], dim=-1).to("cuda")
                    proof.append(exact(expected, outputs[layer], f"{policy}_{t}_{layer}"))
            observations[policy]["miss_rate"] = sum(observations[policy]["misses"]) / (len(seq) * routes)
            observations[policy]["completion_ms"] = sum(observations[policy]["layer_times_ms"])
            observations[policy]["bytes_fetched"] = sum(observations[policy]["misses"]) * 786432
        row = {"workload": workload, "proof": proof, "policies": observations}
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"workload": workload, "results": {p: {k: v for k, v in values.items() if not isinstance(v, list)} for p, values in observations.items()}}), flush=True)
    print("All checked expert payloads match byte-exactly", flush=True)


if __name__ == "__main__":
    main()
