"""Exact query-length prefill graphs for the QSA/GDN/PLE model family."""

from copy import copy
from dataclasses import fields

import torch

from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.kernel.fla.index import attach_chunk_plans
from freetoken.utils import init_logger

logger = init_logger(__name__)


def _clone_metadata(metadata):
    values = {}
    for field in fields(metadata):
        value = getattr(metadata, field.name)
        values[field.name] = value.clone() if isinstance(value, torch.Tensor) else value
    return type(metadata)(**values)


class QwenExactPrefillGraphs:
    def __init__(self, runner, model, config):
        self.runner = runner
        self.model = model
        self.config = config
        self.max_tokens = min(config.moe_prefill_graph_max_tokens or config.max_extend_tokens,
                              config.max_extend_tokens, config.max_seq_len)
        self.max_requests = config.max_running_req
        self.graphs = {}
        self.pool = next(iter(runner.graph_map.values())).pool() if runner.graph_map else None
        self.padding_slot = get_global_ctx().linear_state_pool.padding_slot
        self.schedule = runner.moe_offload_cache.scheduler
        self.captures_at_runtime = 0

    def key(self, batch):
        lengths = tuple(req.extend_len for req in batch.reqs)
        track = batch.fla_metadata.track_dst
        return lengths, 0 if track is None else track.numel()

    def can_use(self, batch):
        return (batch.is_prefill and batch.size <= self.max_requests
                and batch.input_ids.numel() <= self.max_tokens
                and batch.mm_embeds is None and batch.mm_gather_plan is None
                and batch.mm_block_ends is None)

    def _capture_batch(self, batch):
        reqs = [copy(req) for req in batch.reqs]
        captured = Batch(reqs=reqs, phase="prefill")
        captured.padded_reqs = reqs
        captured.input_ids = batch.input_ids.clone()
        captured.out_loc = batch.out_loc.clone()
        captured.positions = batch.positions.clone()
        if batch.mrope_positions is not None:
            captured.mrope_positions = batch.mrope_positions.clone()
        captured.attn_metadata = _clone_metadata(batch.attn_metadata)
        captured.fla_metadata = _clone_metadata(batch.fla_metadata)
        fla = captured.fla_metadata
        attach_chunk_plans(fla.cu_seqlens, [req.extend_len for req in reqs])
        # Fixed-size freshness addressing lets one graph serve cold and continued prefixes.
        fla.fresh_state_indices = torch.where(
            fla.has_initial_state, torch.full_like(fla.cache_indices, self.padding_slot),
            fla.cache_indices,
        ).to(torch.int64)
        fla.seq_lens = tuple(req.extend_len for req in reqs)
        # PLE plans are allocated before capture; retain them with the graph even
        # if the bounded eager cache later evicts this query length.
        captured.ple_index_plans = tuple(
            ple._prefill_indices(list(fla.seq_lens), self.runner.device)
            for ple in self.model.model.ple_layers
        )
        return captured

    def capture(self, batch):
        key = self.key(batch)
        if key in self.graphs:
            return self.graphs[key]
        captured = self._capture_batch(batch)
        logits = torch.empty(captured.size, self.config.model_config.vocab_size,
                             dtype=torch.float32, device=self.runner.device)
        # Warmups advance GDN/PLE/ring/cache state; restore before the real replay.
        ctx = get_global_ctx()
        linear = ctx.linear_state_pool
        saved = [(tensor, tensor.clone()) for tensor in
                 [linear.conv_states, linear.recurrent_states, *linear.slot_states.values()]]
        qsa = ctx.kv_cache
        rings = [qsa.pending_ring(slot) for slot in range(len(self.runner.attn_backend._idx_slot))]
        saved.extend((tensor, tensor.clone()) for tensor in rings)
        previous_batch = ctx._batch
        ctx._batch = None
        try:
            with ctx.forward_batch(captured):
                for _ in range(2):
                    logits.copy_(self._forward(captured))
                torch.cuda.synchronize(self.runner.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.runner.stream, pool=self.pool):
                    logits.copy_(self._forward(captured))
        finally:
            ctx._batch = previous_batch
            for tensor, snapshot in saved:
                tensor.copy_(snapshot)
        if self.pool is None:
            self.pool = graph.pool()
        self.graphs[key] = graph, captured, logits
        logger.info_rank0(f"Captured Qwen prefill graph: key={key}")
        return self.graphs[key]

    def _forward(self, captured):
        return self.model.forward()

    def capture_startup(self):
        runner = self.runner
        old_fraction = self.schedule.fraction.clone()
        old_prefill = self.schedule.prefill_fraction.clone()
        old_table = self.schedule.fetch_table.clone()
        self.schedule.fraction.fill_(65536)
        self.schedule.prefill_fraction.fill_(65536)
        self.schedule.fetch_table.fill_(-1)
        try:
            for tokens in range(self.max_tokens, 0, -1):
                req = Req(torch.zeros(tokens, dtype=torch.int32), runner.dummy_req.table_idx,
                          0, 1, -1, None, None)
                req.linear_slot_idx = self.padding_slot
                batch = Batch(reqs=[req], phase="prefill")
                batch.padded_reqs = batch.reqs
                batch.input_ids = torch.zeros(tokens, dtype=torch.int32, device=runner.device)
                batch.positions = torch.arange(tokens, dtype=torch.int32, device=runner.device)
                batch.out_loc = torch.full((tokens,), int(get_global_ctx().page_table[req.table_idx, 0].item()),
                                           dtype=torch.int32, device=runner.device)
                if runner.mrope:
                    batch.mrope_positions = batch.positions.unsqueeze(0).repeat(3, 1)
                runner.attn_backend.prepare_metadata(batch)
                batch.fla_metadata = build_fla_metadata(batch, runner.device)
                self.capture(batch)
                if tokens > 64:
                    # Hybrid radix may snapshot an interior chunk boundary. Warm the
                    # tracked topology before requests, using only the padding sink.
                    fla = batch.fla_metadata
                    boundary = ((tokens - 1) // 64) * 64
                    km1 = get_global_ctx().linear_state_pool.conv_states.shape[-1]
                    fla.track_dst = torch.tensor([self.padding_slot], dtype=torch.int64, device=runner.device)
                    fla.track_h_row = torch.tensor([boundary // 64], dtype=torch.int64, device=runner.device)
                    fla.track_conv_src = torch.arange(boundary - km1, boundary, device=runner.device).view(1, -1)
                    fla.track_boundary_row = torch.tensor([boundary], dtype=torch.int64, device=runner.device)
                    self.capture(batch)
                runner._reset_moe_offload_cache()
        finally:
            self.schedule.fraction.copy_(old_fraction)
            self.schedule.prefill_fraction.copy_(old_prefill)
            self.schedule.fetch_table.copy_(old_table)
        torch.cuda.synchronize(runner.device)

    def replay(self, batch):
        if self.key(batch) not in self.graphs:
            self.captures_at_runtime += 1
        graph, captured, logits = self.capture(batch)
        disk_table = getattr(self.model, "_ple_table", None)
        if (disk_table is not None and hasattr(disk_table, "_graph_pinned")
                and captured.input_ids.numel() * disk_table._token_bytes > disk_table._graph_pinned.numel()):
            raise RuntimeError("PLE graph staging capacity is smaller than the prefill")
        captured.input_ids.copy_(batch.input_ids)
        captured.out_loc.copy_(batch.out_loc)
        captured.positions.copy_(batch.positions)
        if captured.mrope_positions is not None:
            captured.mrope_positions.copy_(batch.mrope_positions)
        src, dst = batch.attn_metadata, captured.attn_metadata
        for name in ("seq_lens", "ring_slots", "block_table"):
            getattr(dst, name).copy_(getattr(src, name))
        src, dst = batch.fla_metadata, captured.fla_metadata
        dst.cache_indices.copy_(src.cache_indices)
        dst.has_initial_state.copy_(src.has_initial_state)
        dst.fresh_state_indices.copy_(torch.where(
            src.has_initial_state, torch.full_like(src.cache_indices, self.padding_slot),
            src.cache_indices,
        ))
        for name in ("track_dst", "track_h_row", "track_conv_src", "track_boundary_row"):
            target = getattr(dst, name)
            if target is not None:
                target.copy_(getattr(src, name))
        graph.replay()
        return logits


class QwenBucketPrefillGraphs(QwenExactPrefillGraphs):
    """Physical power-of-two shapes with logical lengths staged entirely on device."""

    def __init__(self, runner, model, config):
        super().__init__(runner, model, config)
        self.max_tokens = 1 << (max(1, self.max_tokens).bit_length() - 1)
        # Each bucket can replay independently. Private graph pools avoid aliasing
        # temporaries across arbitrarily ordered token/request buckets.
        self.pool = None

    def key(self, batch):
        tokens = (self.max_tokens if getattr(self.config, "qwen_prefill_mode", "bucket") == "block" else
                  max(8, 1 << (max(1, batch.input_ids.numel()) - 1).bit_length()))
        bs = 1 << (batch.size - 1).bit_length()
        return bs, tokens

    def can_use(self, batch):
        return super().can_use(batch) and self.key(batch)[1] <= self.max_tokens

    def _capture_batch(self, batch):
        from types import SimpleNamespace
        from freetoken.attention.linear import FLAMetadata
        from freetoken.attention.qsa_sparse import QSASparseMetadata
        from freetoken.kernel.triton.prefill_bucket import allocate_chunk_plans

        bs, tokens = self.key(batch)
        device = self.runner.device
        reqs = [SimpleNamespace(extend_len=tokens, cached_len=0, device_len=tokens,
                                table_idx=0, linear_slot_idx=self.padding_slot,
                                mamba_ping_pong=None) for _ in range(bs)]
        captured = Batch(reqs, "prefill")
        captured.padded_reqs = reqs
        def empty(*shape, dtype=torch.int32):
            return torch.empty(*shape, dtype=dtype, device=device)
        captured.input_ids = empty(tokens)
        captured.out_loc, captured.positions = empty(tokens), empty(tokens)
        if batch.mrope_positions is not None:
            captured.mrope_positions = empty(3, tokens)
        captured.fla_metadata = FLAMetadata(
            cu_seqlens=empty(bs + 1, dtype=torch.int64), cache_indices=empty(bs),
            has_initial_state=empty(bs, dtype=torch.bool), seq_lens=(tokens,) * bs,
            graph_padded=True, track_dst=empty(bs, dtype=torch.int64),
            track_h_row=empty(bs, dtype=torch.int64), track_boundary_row=empty(bs, dtype=torch.int64))
        allocate_chunk_plans(captured.fla_metadata.cu_seqlens, tokens, bs)
        pages = get_global_ctx().page_table
        captured.attn_metadata = QSASparseMetadata(
            is_decode=False, last_indices=empty(bs),
            qo_indptr_cpu=torch.zeros(bs + 1, dtype=torch.int32),
            kv_len_cpu=torch.zeros(bs, dtype=torch.int32),
            token_to_req=empty(tokens), cu_seqlens=empty(bs + 1), seq_lens=empty(bs),
            ring_slots=empty(bs), block_table=empty(bs, -(-pages.shape[1] // get_global_ctx().page_size)))
        captured.bucket_desc = empty(bs, 8)
        captured.bucket_desc_host = torch.empty(bs, 8, dtype=torch.int32, pin_memory=True)
        captured.bucket_raw = (empty(tokens), empty(tokens), empty(tokens), empty(3, tokens))
        self._stage(captured, batch)
        return captured

    def _stage(self, captured, batch):
        n, bs = batch.input_ids.numel(), batch.size
        raw = captured.bucket_raw
        raw[0][:n].copy_(batch.input_ids)
        raw[1][:n].copy_(batch.out_loc)
        raw[2][:n].copy_(batch.positions)
        if batch.mrope_positions is not None:
            raw[3][:, :n].copy_(batch.mrope_positions)
        pending = getattr(captured, "bucket_pending_descriptors", [])
        pending = [(event, value) for event, value in pending if not event.query()]
        # The scheduler can enqueue a second prefill while the preceding graph
        # still runs. Keep each pinned descriptor immutable until its H2D ends.
        desc = torch.empty_like(captured.bucket_desc_host, pin_memory=True)
        desc.zero_()
        desc[:, 3].fill_(n)
        desc[:, 6:].fill_(-1)
        offset = 0
        fla = batch.fla_metadata
        tracks = {}
        if fla.track_specs:
            offset = 0
            starts = []
            for req in batch.reqs:
                starts.append(offset)
                offset += req.extend_len
            tracks = {i: (dst, starts[i] + boundary) for i, dst, boundary in fla.track_specs}
        elif fla.track_dst is not None:
            # These small descriptors are scheduler data. The bulk token/FLA plans
            # remain on device and do not read back per-token metadata.
            dst = fla.track_dst.cpu().tolist()
            boundaries = fla.track_boundary_row.cpu().tolist()
            j = 0
            for i, req in enumerate(batch.reqs):
                if getattr(req, "mamba_ping_pong", None) is not None and req.extend_len > 64:
                    tracks[i] = (dst[j], boundaries[j])
                    j += 1
            if not tracks and bs == 1 and dst:
                tracks[0] = (dst[0], boundaries[0])
        offset = 0
        for i, req in enumerate(batch.reqs):
            slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
            desc[i, :6] = torch.tensor([req.table_idx, req.cached_len, req.extend_len,
                                        offset, slot, offset], dtype=torch.int32)
            if i in tracks:
                desc[i, 6] = tracks[i][0]
                desc[i, 7] = tracks[i][1] - offset
            offset += req.extend_len
        captured.bucket_desc.copy_(desc, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record()
        pending.append((ready, desc))
        captured.bucket_pending_descriptors = pending

    def _forward(self, captured):
        from freetoken.kernel.triton.prefill_bucket import prepare_bucket

        ctx = get_global_ctx()
        prepare_bucket(captured, captured.bucket_desc, captured.bucket_raw, ctx.page_table, ctx.page_size)
        return self.model.forward()

    def capture(self, batch):
        self.pool = None
        return super().capture(batch)

    def capture_startup(self):
        from types import SimpleNamespace
        from freetoken.attention.linear import build_fla_metadata

        old = (self.schedule.fraction.clone(), self.schedule.prefill_fraction.clone(), self.schedule.fetch_table.clone())
        self.schedule.fraction.fill_(65536)
        self.schedule.prefill_fraction.fill_(65536)
        self.schedule.fetch_table.fill_(-1)
        try:
            bs_cap = 1 << (max(1, self.max_requests) - 1).bit_length()
            for bs in (1 << i for i in range(bs_cap.bit_length())):
                for power in range(self.max_tokens.bit_length() - 1, 2, -1):
                    tokens = 1 << power
                    if getattr(self.config, "qwen_prefill_mode", "bucket") == "block" and tokens != self.max_tokens:
                        continue
                    if tokens < bs:
                        continue
                    lengths = [tokens // bs + int(i < tokens % bs) for i in range(bs)]
                    reqs = [SimpleNamespace(extend_len=n, cached_len=0, device_len=n,
                                            table_idx=self.runner.dummy_req.table_idx,
                                            linear_slot_idx=self.padding_slot, mamba_ping_pong=None) for n in lengths]
                    batch = Batch(reqs, "prefill")
                    batch.padded_reqs = reqs
                    batch.input_ids = torch.zeros(tokens, dtype=torch.int32, device=self.runner.device)
                    batch.positions = torch.cat([torch.arange(n, dtype=torch.int32, device=self.runner.device) for n in lengths])
                    batch.out_loc = torch.full_like(batch.input_ids, int(get_global_ctx().page_table[reqs[0].table_idx, 0]))
                    if self.runner.mrope:
                        batch.mrope_positions = batch.positions.unsqueeze(0).repeat(3, 1)
                    self.runner.attn_backend.prepare_metadata(batch)
                    batch.fla_metadata = build_fla_metadata(batch, self.runner.device)
                    self.capture(batch)
                    self.runner._reset_moe_offload_cache()
        finally:
            for dst, value in zip((self.schedule.fraction, self.schedule.prefill_fraction, self.schedule.fetch_table), old):
                dst.copy_(value)
        torch.cuda.synchronize(self.runner.device)

    def replay(self, batch):
        if self.key(batch) not in self.graphs:
            self.captures_at_runtime += 1
        graph, captured, logits = self.capture(batch)
        self._stage(captured, batch)
        graph.replay()
        return logits[:batch.size]


def QwenPrefillGraphs(runner, model, config):
    mode = getattr(config, "qwen_prefill_mode", "bucket")
    if mode == "layer":
        from .qwen_layer_prefill import QwenLayerPrefillGraphs

        return QwenLayerPrefillGraphs(runner, model, config)
    cls = QwenExactPrefillGraphs if mode == "exact" else QwenBucketPrefillGraphs
    return cls(runner, model, config)
