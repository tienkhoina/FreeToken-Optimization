"""Benchmark-only request-boundary snapshots; enabled through PYTHONPATH and one env var."""

import atexit
import json
import os
from pathlib import Path
import sys


if os.getenv("FREETOKEN_HOT_AUDIT_DIR"):
    # The CLI frontend is torch-free. Only the spawned engine process needs this hook.
    argv = " ".join(sys.argv)
    if "spawn_main" in argv or "--multiprocessing-fork" in sys.argv:
        import torch
        from freetoken.engine.engine import Engine

        original_init = Engine.__init__
        original_forward = Engine.forward_batch
        current = {"engine": None, "uid": None, "stats": None, "before": None, "routes": 0}

        def snapshot(engine):
            cache = engine.moe_offload_cache
            if cache is None:
                return {}
            hot = cache._adaptive_hot
            result = {"policy": cache.cache_policy, "cache_size": cache.cache_size,
                      "expert_bytes": sum(bank[0].numel() * bank.element_size() for _, bank in cache.banks),
                      "lru_stats": cache.lru_stats.cpu().tolist()}
            if hot is not None:
                epoch = int(cache.step.item()) // hot.decay_calls
                effective = hot.scores >> (epoch - hot.epochs).clamp(0, 31).int()
                result.update(hot_limit=hot.limit, protected_count=int(hot.count.item()),
                              score_epoch=epoch, top_experts=effective.view(cache.num_layers, cache.num_experts).topk(8, dim=-1).indices.cpu().tolist(),
                              hot_stats=hot.stats.cpu().tolist())
            return result

        def save():
            engine = current["engine"]
            if engine is None or current["uid"] is None:
                return
            payload = {"pid": os.getpid(), "uid": current["uid"], "before": current["before"],
                       "after": snapshot(engine), "decode_forwards": current["routes"]}
            directory = Path(os.environ["FREETOKEN_HOT_AUDIT_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"engine_{os.getpid()}.jsonl").open("a") as out:
                out.write(json.dumps(payload) + "\n")

        def init(self, *args, **kwargs):
            config = args[0] if args else kwargs["config"]
            object.__setattr__(config, "moe_collect_stats", True)
            original_init(self, *args, **kwargs)
            current["engine"] = self

        def forward(self, batch, args):
            uid = tuple(req.uid for req in batch.reqs)
            if current["uid"] != uid:
                save()
                current["uid"] = uid
                current["before"] = snapshot(self)
                current["routes"] = 0
            output = original_forward(self, batch, args)
            if not batch.is_prefill:
                current["routes"] += 1
            target = int(os.getenv("FREETOKEN_HOT_AUDIT_TOKENS", "128")) - 1
            if not batch.is_prefill and current["routes"] == target:
                save()
                current["uid"] = None
            return output

        Engine.__init__ = init
        Engine.forward_batch = forward
        atexit.register(save)
