"""Startup-captured DeepSeek prefill graphs with token and context buckets."""

from types import SimpleNamespace

import torch

from freetoken.attention.dsv4_sparse import DSV4AttnMetadata
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

logger = init_logger(__name__)


def bucket_shapes(config):
    page = config.model_config.dsv4_args.window_size
    cap = config.moe_prefill_graph_max_tokens or getattr(config, "max_extend_tokens", 8192)
    if cap < page:
        raise ValueError("DeepSeek prefill graph cap must hold at least one window page")
    tokens = getattr(config, "dsv4_prefill_buckets", None)
    if tokens is None:
        tokens = [page * (2 ** exponent) for exponent in range((cap // page).bit_length())]
    contexts = getattr(config, "dsv4_prefill_context_buckets", None)
    if contexts is None:
        contexts = [min(config.max_seq_len, max(max(tokens), 8192))]
    for name, values in (("token", tokens), ("context", contexts)):
        if not values or any(type(value) is not int or value < page or value % page for value in values):
            raise ValueError(f"DeepSeek {name} buckets must be positive window-page multiples")
        if list(values) != sorted(set(values)):
            raise ValueError(f"DeepSeek {name} buckets must be strictly increasing")
    if max(tokens) > cap or max(contexts) > config.max_seq_len:
        raise ValueError("DeepSeek buckets exceed the configured token cap or sequence limit")
    if any(not any(context >= token for context in contexts) for token in tokens):
        raise ValueError("Each DeepSeek token bucket needs a covering context bucket")
    return [(token, context) for token in tokens for context in contexts if token <= context]


def select_bucket(shapes, new_tokens, prefix, page):
    if new_tokens <= 0 or prefix < 0 or prefix % page:
        return None
    choices = [(token, context) for token, context in shapes
               if new_tokens <= token and prefix + new_tokens <= context]
    return min(choices, default=None)


class Dsv4PrefillGraphs:
    def __init__(self, runner, model, config):
        self.runner, self.model, self.config = runner, model, config
        self.shapes = bucket_shapes(config)
        self.page = config.model_config.dsv4_args.window_size
        self.graphs = {}
        self.capture_count = 0
        self.replay_count = 0

    def key(self, batch):
        if (not batch.is_prefill or batch.size != 1 or batch.mm_embeds is not None
                or batch.mm_gather_plan is not None or batch.mm_block_ends is not None):
            return None
        request = batch.reqs[0]
        return select_bucket(self.shapes, request.extend_len, request.cached_len, self.page)

    def can_use(self, batch):
        return self.key(batch) in self.graphs

    def _batch(self, token, context):
        req = SimpleNamespace(extend_len=token, cached_len=0, table_idx=self.runner.dummy_req.table_idx)
        batch = Batch([req], "prefill")
        batch.padded_reqs = batch.reqs
        batch.input_ids = torch.zeros(token, dtype=torch.int32, device=self.runner.device)
        batch.positions = torch.zeros(token, dtype=torch.int64, device=self.runner.device)
        batch.out_loc = torch.zeros(token, dtype=torch.int32, device=self.runner.device)
        descriptor = torch.zeros(2, dtype=torch.int32, device=self.runner.device)
        batch.attn_metadata = DSV4AttnMetadata(
            last_indices=torch.zeros(1, dtype=torch.int64, device=self.runner.device),
            descriptor=descriptor,
            full_snap=torch.full((1, context), -1, dtype=torch.int64, device=self.runner.device),
            window_snap=torch.full((context,), -1, dtype=torch.int64, device=self.runner.device),
            cu_seqlens_q_gpu=torch.zeros(2, dtype=torch.int32, device=self.runner.device),
        )
        return batch

    def capture_startup(self):
        cache = self.runner.moe_offload_cache
        if cache is None or cache.scheduler is None or cache.quant_format != "nvfp4":
            raise ValueError("DeepSeek bucket prefill requires native MoE scheduling")
        ctx = get_global_ctx()
        previous = ctx._batch
        self.model._ensure_bound()
        try:
            ctx._batch = None
            for token, context in self.shapes:
                batch = self._batch(token, context)
                with ctx.forward_batch(batch):
                    for _ in range(2):
                        self.model.forward()
                    torch.cuda.synchronize(self.runner.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.runner.stream):
                        output = self.model.forward()
                self.graphs[token, context] = graph, batch, output
                self.capture_count += 1
                logger.info_rank0(f"Captured DeepSeek prefill graph: tokens={token}, context={context}")
        finally:
            ctx._batch = previous
            cache.reset()
        torch.cuda.synchronize(self.runner.device)

    def prepare(self, batch):
        key = self.key(batch)
        graph, captured, output = self.graphs[key]
        n, prefix = batch.reqs[0].extend_len, batch.reqs[0].cached_len
        if batch.input_ids.numel() != n or batch.positions.numel() != n:
            raise ValueError("DeepSeek bucket input does not match its logical token length")
        captured.input_ids.zero_()
        captured.input_ids[:n].copy_(batch.input_ids)
        captured.positions.zero_()
        captured.positions[:n].copy_(batch.positions)
        meta = captured.attn_metadata
        meta.descriptor.copy_(torch.tensor([n, prefix], dtype=torch.int32, pin_memory=True), non_blocking=True)
        meta.cu_seqlens_q_gpu[1:].copy_(meta.valid_tokens)
        meta.last_indices.copy_((meta.valid_tokens - 1).long())
        pool = get_global_ctx().kv_cache
        source = pool.full_loc_map[batch.reqs[0].table_idx, :prefix + n]
        meta.full_snap.fill_(-1)
        meta.full_snap[0, :prefix + n].copy_(source)
        meta.window_snap.copy_(pool.translate_full_to_window(meta.full_snap[0]))
        return graph, captured, output

    def replay(self, batch):
        graph, _, output = self.prepare(batch)
        graph.replay()
        self.replay_count += 1
        return output

    def destroy(self):
        self.graphs.clear()
