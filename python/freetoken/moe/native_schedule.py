"""Fixed graph topology with native routing and concurrent hit, fetch and CPU branches."""

from functools import cache
import json

import torch

from freetoken.kernel.utils import load_jit
from .schedule_profile import profile_rows


@cache
def schedule_module():
    return load_jit("moe_schedule", "token_buckets", cuda_files=["moe_schedule.cuh"],
                    cuda_wrappers=[("launch", "&MoeSchedule::run"), ("cpu_doorbell", "&MoeCpuDoorbell::run")])


class NativeMoeSchedule:
    def __init__(self, cache, gpu_fraction=0.8, profile=None, prefill_fraction=None):
        self.cache = cache
        self.fraction = torch.tensor([round(gpu_fraction * 65536)], device=cache.device, dtype=torch.int32)
        self.prefill_fraction = torch.tensor([round((gpu_fraction if prefill_fraction is None else prefill_fraction) * 65536)],
                                             device=cache.device, dtype=torch.int32)
        self.token_buckets = torch.tensor([1], dtype=torch.int32, device=cache.device)
        self.fetch_table = torch.full((1, cache.num_experts + 1), -1, dtype=torch.int32, device=cache.device)
        self.stats = torch.zeros(cache.num_layers, 4, dtype=torch.int64, device=cache.device)
        if profile is not None:
            with open(profile) as source:
                data = json.load(source)
            expected = getattr(cache.cpu_executor, "quant_format", cache.quant_format)
            if data.get("format") != expected or data.get("gpu") != torch.cuda.get_device_name(cache.device):
                raise ValueError("schedule profile format/GPU differs; rerun calibration on this machine")
            executor = cache.cpu_executor
            if executor is not None and (data.get("cpu_workers") != executor.num_threads
                                         or data.get("hidden") != executor.H or data.get("intermediate") != executor.I
                                         or data.get("cpu_isa") != executor.isa):
                raise ValueError("schedule profile CPU ISA/worker count/expert dimensions differ; rerun calibration")
            if executor is not None:
                for name in ("top_k", "activation", "swiglu_alpha", "swiglu_limit"):
                    if name in data and data[name] != getattr(executor, name):
                        raise ValueError(f"schedule profile {name} differs; rerun calibration")
            rows = profile_rows(data, cache.num_experts)
            self.token_buckets = torch.tensor([tokens for tokens, _ in rows], device=cache.device, dtype=torch.int32)
            self.fetch_table = torch.full((len(rows), cache.num_experts + 1), -1, device=cache.device, dtype=torch.int32)
            for index, (_, counts) in enumerate(rows):
                self.fetch_table[index, :len(counts)] = torch.tensor(counts, device=cache.device, dtype=torch.int32)
        self.copy_stream = torch.cuda.Stream(device=cache.device)
        self.cpu_stream = torch.cuda.Stream(device=cache.device)
        self.route_ready = torch.cuda.Event()
        self.fetch_ready = torch.cuda.Event()
        self.cpu_ready = torch.cuda.Event()
        self.buffers = {}
        self.launch = schedule_module().launch

    def _buffers(self, tokens, top_k):
        key = tokens, top_k
        if key not in self.buffers:
            shape = tokens, top_k
            ids = [torch.empty(shape, dtype=torch.int32, device=self.cache.device) for _ in range(3)]
            weights = [torch.empty(shape, dtype=torch.float32, device=self.cache.device) for _ in range(2)]
            counts = torch.empty(4, dtype=torch.int64, device=self.cache.device)
            valid = torch.tensor([tokens], dtype=torch.int32, device=self.cache.device)
            self.buffers[key] = (*ids, *weights, counts, valid)
        return self.buffers[key]

    def route(self, layer_id, topk_ids, topk_weights, valid_tokens=None, is_prefill=False):
        c = self.cache
        hit_ids, fetch_ids, cpu_ids, hit_w, fetch_w, counts, full = self._buffers(*topk_ids.shape)
        self.launch(topk_ids, topk_weights, hit_ids, hit_w, fetch_ids, fetch_w, cpu_ids,
                    c.slot_for_id.view(-1), c.id_of_slot, c.usage, c.step.view(1), c.src_indices,
                    c.evict_slots, c.num_indices, counts, self.stats[layer_id],
                    self.prefill_fraction if is_prefill else self.fraction, self.fetch_table, self.token_buckets,
                    full if valid_tokens is None else valid_tokens, c.num_experts, layer_id * c.num_experts,
                    min(c.cache_size, 2 * c.num_experts) if is_prefill else c.cache_size,
                    topk_ids.shape[0] if is_prefill else -1)
        c._pending_src_layer = layer_id
        c._pending_whole_layer = False
        return hit_ids, fetch_ids, cpu_ids, hit_w, fetch_w, counts

    def forward(self, layer, x, weights, ids, *, is_prefill, valid_tokens=None):
        c = self.cache
        method = layer.quant_method
        route_outputs = method is not None and method.kernel.supports_route_outputs
        hit_ids, fetch_ids, cpu_ids, hit_w, fetch_w, counts = self.route(layer.layer_id, ids, weights, valid_tokens, is_prefill)
        main = torch.cuda.current_stream(c.device)
        self.route_ready.record(main)
        if c.cpu_executor is not None:
            with torch.cuda.stream(self.cpu_stream):
                self.cpu_stream.wait_event(self.route_ready)
                pending = c.cpu_executor.decode_submit(layer.layer_id, x, weights, cpu_ids, active_count=counts[3:])
                cpu = c.cpu_executor.decode_sync(pending)
                self.cpu_ready.record(self.cpu_stream)
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.route_ready)
            c.copy_missing()
            # Some expert operators overwrite x; neither branch may change the other's input.
            fetch_x = x.clone()
            fetched = layer._expert_gemm(c, fetch_x, fetch_w, fetch_ids,
                                         views=c.scheduled_bank_views(), n=c.cache_size + 1 if is_prefill else None,
                                         alphas=c.scheduled_alphas(layer.layer_id), is_prefill=is_prefill,
                                         route_outputs=route_outputs)
            self.fetch_ready.record(self.copy_stream)
        hit_x = x.clone()
        hit = layer._expert_gemm(c, hit_x, hit_w, hit_ids,
                                views=c.scheduled_bank_views(), n=c.cache_size + 1 if is_prefill else None,
                                alphas=c.scheduled_alphas(layer.layer_id), is_prefill=is_prefill,
                                route_outputs=route_outputs)
        main.wait_event(self.fetch_ready)
        if route_outputs:
            from freetoken.kernel import moe_sum_reduce_triton

            result = torch.empty_like(x)
            moe_sum_reduce_triton(hit, result, branch=fetched)
        else:
            result = hit + fetched
        if c.cpu_executor is not None:
            main.wait_event(self.cpu_ready)
            result = result + torch.where(counts[3] > 0, cpu, torch.zeros_like(cpu))
        return result
