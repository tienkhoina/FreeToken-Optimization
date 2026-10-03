"""Layer-major long-prefill graphs dispatched by native code over bounded blocks."""

from copy import copy
from types import SimpleNamespace

import torch

from freetoken.core import Batch, get_global_ctx
from freetoken.kernel.layer_prefill import layer_prefill_module
from freetoken.kernel.triton.prefill_bucket import prepare_bucket
from freetoken.utils import init_logger

from .qwen_prefill import QwenBucketPrefillGraphs

logger = init_logger(__name__)


class QwenLayerPrefillGraphs(QwenBucketPrefillGraphs):
    def __init__(self, runner, model, config):
        super().__init__(runner, model, config)
        if self.max_tokens < 64:
            raise ValueError("layer prefill needs a block cap of at least 64 tokens")
        self.layers = model.model.layers.op_list
        cache = runner.moe_offload_cache
        if cache.cache_size < 2 * cache.num_experts:
            raise ValueError("layer prefill requires two expert-layer buffers in the cache")
        self.staging = {}
        # Layer sweeps use one block shape. Quantized dense GEMMs can retain a
        # large workspace per graph, so capture two parities instead of every tail.
        self.buckets = [self.max_tokens]
        self.native_run = layer_prefill_module().run
        self.planes = None
        self.pending = None
        self.peak_plane_bytes = 0
        self.plane_device = getattr(config, "qwen_prefill_activation_device", "auto")
        self.expert_bytes = 0
        self.runtime_blocks = 0
        self.disk_table = getattr(model, "_ple_table", None)
        if self.disk_table is not None and not hasattr(self.disk_table, "_store"):
            self.disk_table = None
        if len(model.model.ple_layers) > 1:
            raise ValueError("layer prefill currently requires at most one PLE table")

    def can_use(self, batch):
        return (batch.is_prefill and batch.size <= self.max_requests
                and batch.input_ids.numel() > 0
                and batch.mm_embeds is None and batch.mm_gather_plan is None
                and batch.mm_block_ends is None)

    def _capture_staging(self, bucket):
        req = SimpleNamespace(extend_len=bucket, cached_len=0, device_len=bucket,
                              table_idx=self.runner.dummy_req.table_idx,
                              linear_slot_idx=self.padding_slot, mamba_ping_pong=None)
        batch = Batch([req], "prefill")
        batch.padded_reqs = batch.reqs
        batch.input_ids = torch.zeros(bucket, dtype=torch.int32, device=self.runner.device)
        batch.positions = torch.arange(bucket, dtype=torch.int32, device=self.runner.device)
        batch.out_loc = torch.full_like(batch.input_ids, int(get_global_ctx().page_table[req.table_idx, 0]))
        if self.runner.mrope:
            batch.mrope_positions = batch.positions.unsqueeze(0).repeat(3, 1)
        self.runner.attn_backend.prepare_metadata(batch)
        from freetoken.attention.linear import build_fla_metadata

        batch.fla_metadata = build_fla_metadata(batch, self.runner.device)
        captured = self._capture_batch(batch)
        width = self.config.model_config.hidden_size * self.config.model_config.qwen4_args.hc_count
        dtype = self.model.model.embed_tokens.weight.dtype
        captured.layer_input = torch.zeros(bucket, width, dtype=dtype, device=self.runner.device)
        captured.layer_output = torch.empty_like(captured.layer_input)
        if self.disk_table is not None:
            captured.ple_raw = torch.zeros(bucket, self.disk_table._token_bytes,
                                          dtype=torch.uint8, device=self.runner.device)
        return captured

    def _layer_forward(self, captured, index):
        ctx = get_global_ctx()
        prepare_bucket(captured, captured.bucket_desc, captured.bucket_raw, ctx.page_table, ctx.page_size)
        captured.attn_metadata.cmp_rows = None
        captured.ple_preloaded_embeddings = None
        if index < 0:
            hidden = self.model.model.embed_tokens.forward(captured.input_ids)
            result = hidden.repeat(1, self.model.model.hc_count)
        else:
            layer = self.layers[index]
            if layer.ple is not None:
                from freetoken.models.qwen4_exp.ple import build_ple_metadata, commit_ngram_context

                if self.disk_table is not None:
                    values = captured.ple_raw.view(torch.float8_e4m3fn).to(captured.layer_input.dtype)
                    captured.ple_preloaded_embeddings = values * self.disk_table.scale
                meta = build_ple_metadata(captured, layer.ple.args, self.runner.device)
                layer.ple.start_prefetch(captured, meta)
            result = layer.forward(captured.layer_input, captured)
            if layer.ple is not None:
                commit_ngram_context(meta, captured.fla_metadata)
        captured.layer_output.copy_(result)
        return captured.layer_output

    def capture_startup(self):
        cache = self.runner.moe_offload_cache
        if not cache.prefill_bank_buffers:
            cache._init_prefill_overlap_buffers()
        dtype = self.model.model.embed_tokens.weight.dtype
        graph_handles = []
        ctx = get_global_ctx()
        for index in range(-1, len(self.layers)):
            row = []
            for bucket, parity in ((bucket, parity) for bucket in self.buckets for parity in (0, 1)):
                base = self.staging.get((bucket, parity))
                if base is None:
                    base = self.staging[bucket, parity] = self._capture_staging(bucket)
                captured = copy(base)
                captured.attn_metadata = copy(base.attn_metadata)
                if index >= 0:
                    captured.moe_layer_banks = tuple(buf[index % 2] for buf in cache.prefill_bank_buffers)
                    for (per_layer, _), buffer in zip(cache.banks, cache.prefill_bank_buffers):
                        buffer[index % 2].copy_(per_layer[index], non_blocking=True)
                else:
                    captured.moe_layer_banks = None
                state = ctx.linear_state_pool
                saved = []
                if index >= 0:
                    layer = self.layers[index]
                    if layer._is_linear:
                        li = state.local_index(layer._layer_id)
                        saved.extend((tensor, tensor.clone()) for tensor in
                                     (state.conv_states[li, self.padding_slot],
                                      state.recurrent_states[li, self.padding_slot]))
                    if layer.ple is not None:
                        saved.extend((tensor, tensor.clone()) for tensor in
                                     (state.slot_state("ple_conv", layer._layer_id)[self.padding_slot],
                                      state.slot_state("ple_ngram_ctx")[self.padding_slot]))
                    if not layer._is_linear:
                        idx = self.runner.attn_backend._idx_slot[layer._layer_id]
                        ring = ctx.kv_cache.pending_ring(idx)[self.runner.dummy_req.table_idx]
                        saved.append((ring, ring.clone()))
                previous = ctx._batch
                ctx._batch = None
                try:
                    with ctx.forward_batch(captured):
                        self._layer_forward(captured, index)
                        torch.cuda.synchronize(self.runner.device)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=self.runner.stream):
                            self._layer_forward(captured, index)
                finally:
                    ctx._batch = previous
                    for tensor, snapshot in saved:
                        tensor.copy_(snapshot)
                self.graphs[index, bucket, parity] = graph, captured, captured.layer_output
                row.append(graph.raw_cuda_graph_exec())
            graph_handles.append(row)
            logger.info_rank0(f"Captured layer prefill: layer={index}, buckets={self.buckets}")
        self.graph_handles = torch.tensor(graph_handles, dtype=torch.int64)
        self.buffer_handles = torch.tensor([
            [*(tensor.data_ptr() for tensor in base.bucket_raw),
             base.layer_input.data_ptr(), base.layer_output.data_ptr(), base.bucket_desc.data_ptr(),
             getattr(base, "ple_raw", base.layer_input).data_ptr() if self.disk_table is not None else 0,
             0, bucket] for (bucket, parity), base in self.staging.items()], dtype=torch.int64)
        self.bank_copies = torch.tensor([
            [[buffer[layer % 2].data_ptr(), per_layer[layer].data_ptr(),
              per_layer[layer].numel() * per_layer[layer].element_size()]
             for (per_layer, _), buffer in zip(cache.banks, cache.prefill_bank_buffers)]
            for layer in range(len(self.layers))], dtype=torch.int64)
        for parity in (0, 1):
            cache._invalidate_prefill_buffer(parity)
        torch.cuda.synchronize(self.runner.device)

    def _plan(self, batch):
        blocks, descriptors = [], []
        tracks = {i: (dst, boundary) for i, dst, boundary in batch.fla_metadata.track_specs}
        start = 0
        for r, req in enumerate(batch.reqs):
            slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
            for offset in range(0, req.extend_len, self.max_tokens):
                n = min(self.max_tokens, req.extend_len - offset)
                blocks.append([start + offset, n, 0])
                dst, boundary = -1, -1
                if r in tracks:
                    target, at = tracks[r]
                    if offset < at <= offset + n:
                        dst, boundary = target, at - offset
                descriptors.append([req.table_idx, req.cached_len + offset, n, 0, slot, 0, dst, boundary])
            start += req.extend_len
        return torch.tensor(blocks, dtype=torch.int64), torch.tensor(descriptors, dtype=torch.int32, pin_memory=True)

    def prepare(self, batch):
        if self.pending is not None:
            self.pending[0].synchronize()
            self.pending = None
        n = batch.input_ids.numel()
        width = self.config.model_config.hidden_size * self.model.model.hc_count
        dtype = self.model.model.embed_tokens.weight.dtype
        capacity = ((n + self.max_tokens - 1) // self.max_tokens) * self.max_tokens
        need = capacity * width * torch.empty((), dtype=dtype).element_size() * 2
        if self.planes is None or self.planes[0].shape[0] < n:
            self.planes = None
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(self.runner.device)
            workspace_margin = max(512 << 20, free // 5)
            gpu = self.plane_device == "gpu" or (self.plane_device == "auto" and need + workspace_margin < free)
            if gpu and need + workspace_margin >= free:
                raise RuntimeError("layer prefill activation planes exceed VRAM; use auto or cpu placement")
            kwargs = {"device": self.runner.device} if gpu else {"device": "cpu", "pin_memory": True}
            self.planes = tuple(torch.empty(capacity, width, dtype=dtype, **kwargs) for _ in range(2))
            self.peak_plane_bytes = need
            logger.info_rank0(f"Layer activation planes: device={'gpu' if gpu else 'cpu'}, rows={capacity}, bytes={need}")
        ple = torch.empty(0, 0, dtype=torch.uint8)
        if self.disk_table is not None:
            table = self.disk_table
            ple = torch.empty(n, table._token_bytes, dtype=torch.uint8, pin_memory=True)
            offset = 0
            for req in batch.reqs:
                ids = table._ple_ids(req.input_ids[req.cached_len:req.device_len]).to(torch.int64)
                run = torch.cat([torch.tensor(table._ple_context(req.input_ids, req.cached_len), dtype=torch.int64), ids])
                table._store.stage(run.data_ptr(), ids.numel(), ple.data_ptr() + offset * table._token_bytes)
                offset += ids.numel()
            table._store.flush(0)
        plan, descriptors = self._plan(batch)
        self.prepared = batch, plan, descriptors, descriptors.to(self.runner.device, non_blocking=True), ple

    def replay(self, batch):
        if getattr(self, "prepared", None) is None or self.prepared[0] is not batch:
            self.prepare(batch)
        _, plan, host_desc, desc, ple = self.prepared
        self.prepared = None
        cache = self.runner.moe_offload_cache
        for parity in (0, 1):
            cache._invalidate_prefill_buffer(parity)
        mrope = batch.mrope_positions
        if mrope is None:
            mrope = batch.positions.unsqueeze(0).repeat(3, 1)
        ple_layer = next((i for i, layer in enumerate(self.layers) if layer.ple is not None), -1)
        self.native_run(self.graph_handles, self.buffer_handles, self.bank_copies, plan, desc,
                        batch.input_ids, batch.out_loc, batch.positions, mrope, *self.planes, ple,
                        torch.cuda.current_stream(self.runner.device).cuda_stream,
                        cache.prefill_copy_stream.cuda_stream,
                        getattr(self.config, "moe_prefill_overlap", True), ple_layer)
        self.expert_bytes += int(self.bank_copies[:, :, 2].sum())
        self.runtime_blocks += plan.shape[0] * len(self.layers)
        final = self.planes[len(self.layers) % 2]
        last = torch.empty(batch.size, final.shape[1], dtype=final.dtype, device=self.runner.device)
        offset = 0
        for i, req in enumerate(batch.reqs):
            offset += req.extend_len
            last[i:i + 1].copy_(final[offset - 1:offset], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending = event, (host_desc, desc, ple, plan, batch, mrope)
        tail = copy(batch)
        tail.attn_metadata = copy(batch.attn_metadata)
        tail.attn_metadata.last_indices = torch.arange(batch.size, dtype=torch.int32, device=self.runner.device)
        ctx = get_global_ctx()
        previous = ctx._batch
        ctx._batch = tail
        try:
            mixed = self.model.model.hyper_connection_mixer.mix(last)[0]
            return self.model.lm_head.forward(mixed).float()
        finally:
            ctx._batch = previous

    def destroy(self):
        if self.pending is not None:
            self.pending[0].synchronize()
        self.graphs.clear()
        self.staging.clear()
        self.planes = None
