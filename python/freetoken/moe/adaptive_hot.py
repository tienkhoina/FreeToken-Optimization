"""Online protection of frequently reused expert slots, with a bounded LRU remainder."""

from functools import cache

import torch

from freetoken.kernel.utils import load_jit, make_cpp_args


@cache
def _module(queries):
    args = make_cpp_args(queries)
    return load_jit("adaptive_hot_ensure", *args, cuda_files=["adaptive_hot_cache.cuh"],
                    cuda_wrappers=[("launch", f"&AdaptiveHotEnsure<{args}>::run")])


class AdaptiveHotState:
    def __init__(self, cache):
        self.scores = torch.zeros(cache.num_layers * cache.num_experts, dtype=torch.int32, device=cache.device)
        self.epochs = torch.zeros_like(self.scores, dtype=torch.int64)
        self.hot = torch.zeros(cache.cache_size, dtype=torch.uint8, device=cache.device)
        self.count = torch.zeros(1, dtype=torch.int32, device=cache.device)
        self.stats = torch.zeros(cache.num_layers, 4, dtype=torch.int64, device=cache.device)
        self.reserved = (2 if cache.prefill_overlap else 1) * cache.num_experts
        self.limit = max(0, min(cache.cache_size // 2, cache.cache_size - self.reserved,
                                cache.cache_size - cache.num_experts))
        self.decay_calls = 64 * cache.num_layers
        self.min_score = 4

    def ensure(self, cache, layer_id, expert_ids):
        query = expert_ids.view(-1)
        if query.numel() == 0:
            cache.num_indices.zero_()
            return
        bucket = max(1, 1 << (query.numel() - 1).bit_length())
        if bucket > 2048:
            raise ValueError("adaptive_hot supports at most 2048 expert routes per layer call")
        _module(bucket).launch(query, cache.slot_for_id.view(-1), cache.id_of_slot, cache.usage,
                               cache.step.view(1), cache.src_indices, cache.evict_slots, cache.num_indices,
                               self.scores, self.epochs, self.hot, self.count, cache.lru_stats[layer_id],
                               self.stats[layer_id], layer_id * cache.num_experts, self.limit,
                               self.reserved, self.decay_calls, self.min_score, cache.collect_stats)

    def reset(self):
        self.scores.zero_()
        self.epochs.zero_()
        self.hot.zero_()
        self.count.zero_()
        self.stats.zero_()
