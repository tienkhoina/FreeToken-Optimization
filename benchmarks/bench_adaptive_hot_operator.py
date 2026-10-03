"""Measure cache admission overhead separately from expert byte copies and model math."""

import argparse
import json
from pathlib import Path

import torch

from bench_copy_overlap import time_graphs
from freetoken.moe.offload_cache import OffloadMoeCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"scope": "admission/query remap only; fixed graph buffers, no weight copy", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    l, e, c = 36, 128, 1241
    for scenario in ("warm", "forced_miss"):
        graphs, objects = {}, []
        for policy in ("lru", "adaptive_hot"):
            cache = OffloadMoeCache(l, e, c, torch.device("cuda"), cache_policy=policy, prefill_overlap=True)
            query = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32)
            ids = query.clone()
            for _ in range(8):
                ids.copy_(query)
                cache.ensure_experts(0, ids)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(32):
                    if scenario == "forced_miss":
                        cache.slot_for_id.fill_(-1)
                        cache.id_of_slot.fill_(-1)
                        cache.usage.zero_()
                        if cache._adaptive_hot is not None:
                            cache._adaptive_hot.hot.zero_()
                            cache._adaptive_hot.count.zero_()
                    ids.copy_(query)
                    cache.ensure_experts(0, ids)
            graph.replay()
            torch.cuda.synchronize()
            owner = cache.id_of_slot.index_select(0, ids.long())
            if not torch.equal(owner, query):
                raise AssertionError("cache admission returned the wrong expert")
            graphs[policy] = graph
            objects.append((cache, query, ids))
        samples = time_graphs(graphs, 41)
        row = {"scenario": scenario, "cycles_per_replay": 32,
               "note": "forced_miss includes the same bookkeeping fills plus policy mask reset; not pure miss-kernel latency",
               "median_us": {p: t["median_ms"] * 1000 / 32 for p, t in samples.items()}, "samples": samples}
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row["median_us"]), flush=True)


if __name__ == "__main__":
    main()
