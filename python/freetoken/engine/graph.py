# Local modifications: FreeToken-Optimization workspace, 2026-10-03.
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    # [3, bs] t/h/w rope positions; allocated only for mrope models (else None).
    mrope_positions: torch.Tensor | None
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(
        cls, bs: int, vocab_size: int, device: torch.device, mrope: bool = False
    ) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            mrope_positions=(
                torch.zeros(3, bs, dtype=torch.int32, device=device) if mrope else None
            ),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        if self.mrope_positions is not None:
            batch.mrope_positions = self.mrope_positions[:, _slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        if hasattr(batch, "native_input_slot") and self.mrope_positions is None:
            from freetoken.kernel.runtime_batch import runtime_batch_module

            runtime_batch_module().copy_inputs(
                batch.input_ids, batch.out_loc, batch.positions, batch.native_linear_slots,
                self.input_ids[_slice], self.out_loc[_slice], self.positions[_slice], self.table_idx[_slice],
            )
            return
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if self.mrope_positions is not None:
            self.mrope_positions[:, _slice] = batch.mrope_positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mrope: bool = False,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.mrope = mrope
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)
        self.prefill_graph_map = {}
        self.prefill_max_tokens = 0

    def capture_prefill(self, model, config, page_table):
        if getattr(config.model_config, "dsv4_args", None) is not None:
            if getattr(config, "dsv4_prefill_mode", "eager") == "bucket":
                from .dsv4_prefill import Dsv4PrefillGraphs

                self.dsv4_prefill = Dsv4PrefillGraphs(self, model, config)
                self.prefill_max_tokens = max(token for token, _ in self.dsv4_prefill.shapes)
                self.dsv4_prefill.capture_startup()
                return
            self.prefill_max_tokens = 0
            logger.info_rank0("DeepSeek-V4 native MoE scheduling uses decode graphs; sparse prefill remains eager")
            return
        if config.attention_backend == "qsa_sparse" and config.model_config.qwen4_args is not None:
            from .qwen_prefill import QwenPrefillGraphs

            self.qwen_prefill = QwenPrefillGraphs(self, model, config)
            self.prefill_max_tokens = self.qwen_prefill.max_tokens
            self.qwen_prefill.capture_startup()
            return
        if config.attention_backend.split(",")[0] != "triton" or config.model_config.has_linear_attention or config.model_config.model_is_mrope:
            raise ValueError("native scheduled prefill graphs currently require Triton paged attention without linear/mrope state")
        cap = config.moe_prefill_graph_max_tokens or config.max_extend_tokens
        cap = min(cap, config.max_extend_tokens, config.max_seq_len)
        cap = 1 << (max(1, cap).bit_length() - 1)
        self.prefill_max_tokens = 1 << (max(1, cap) - 1).bit_length()
        self.prefill_graph_map = {}
        sizes = []
        tokens = 1
        while tokens <= self.prefill_max_tokens:
            sizes.append(tokens)
            tokens *= 2
        bs_list = list(range(1, config.max_running_req + 1))
        dummy = int(page_table[self.dummy_req.table_idx, 0].item())
        schedule = self.moe_offload_cache.scheduler
        old_fraction = schedule.fraction.clone()
        old_prefill_fraction = schedule.prefill_fraction.clone()
        old_table = schedule.fetch_table.clone()
        # Warm/capture every CPU handshake without spending startup time computing repeated dummy routes.
        schedule.fraction.fill_(65536)
        schedule.prefill_fraction.fill_(65536)
        schedule.fetch_table.fill_(-1)
        pool = next(iter(self.graph_map.values())).pool() if self.graph_map else None
        for bs in bs_list:
            for tokens in sizes:
                if tokens < bs:
                    continue
                lengths = [tokens // bs + (i < tokens % bs) for i in range(bs)]
                reqs = [Req(torch.zeros(n, dtype=torch.int32), self.dummy_req.table_idx, 0, 1, -1, None, None)
                        for n in lengths]
                batch = Batch(reqs=reqs, phase="prefill")
                batch.padded_reqs = reqs
                batch.input_ids = torch.zeros(tokens, dtype=torch.int32, device=self.device)
                batch.positions = torch.cat([torch.arange(n, dtype=torch.int32, device=self.device) for n in lengths])
                batch.out_loc = torch.full((tokens,), dummy, dtype=torch.int32, device=self.device)
                self.attn_backend.prepare_metadata(batch)
                meta = batch.attn_metadata
                meta.cu_seqlens_q_gpu = meta.cu_seqlens_q_gpu.clone()
                meta.indptr = meta.indptr.clone()
                index_capacity = bs * min(config.max_seq_len, page_table.shape[1])
                indices = torch.full((index_capacity,), dummy, dtype=torch.int32, device=self.device)
                indices[: meta.indices.numel()].copy_(meta.indices)
                meta.indices = indices
                if meta.swa_indices is not None:
                    swa = torch.zeros_like(indices)
                    swa[: meta.swa_indices.numel()].copy_(meta.swa_indices)
                    meta.swa_indices = swa
                meta.q_to_req = torch.zeros(tokens, dtype=torch.int32, device=self.device)
                meta.q_positions = batch.positions
                # All buckets share the extend kernel; lengths/prefixes remain runtime data on replay.
                meta.is_decode = False
                meta.max_q_len = tokens
                meta.graph_padded = True
                logits = torch.empty(bs, config.model_config.vocab_size, dtype=torch.float32, device=self.device)
                with get_global_ctx().forward_batch(batch):
                    for _ in range(2):
                        logits.copy_(model.forward())
                    torch.cuda.synchronize(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream, pool=pool):
                        logits.copy_(model.forward())
                if pool is None:
                    pool = graph.pool()
                self.prefill_graph_map[bs, tokens] = (graph, batch, logits)
                self._reset_moe_offload_cache()
                logger.info_rank0(f"Captured native MoE prefill graph: requests={bs}, tokens={tokens}")
        torch.cuda.synchronize(self.device)
        schedule.fraction.copy_(old_fraction)
        schedule.prefill_fraction.copy_(old_prefill_fraction)
        schedule.fetch_table.copy_(old_table)

    def _prefill_key(self, batch):
        tokens = int(batch.input_ids.numel())
        bucket = 1 << (max(tokens, 1) - 1).bit_length()
        return batch.size, bucket

    def _stage_prefill(self, batch):
        graph, captured, logits = self.prefill_graph_map[self._prefill_key(batch)]
        n = batch.input_ids.numel()
        captured.input_ids.zero_()
        captured.input_ids[:n].copy_(batch.input_ids)
        captured.positions.zero_()
        captured.positions[:n].copy_(batch.positions)
        dummy = get_global_ctx().page_table[self.dummy_req.table_idx, 0]
        captured.out_loc.copy_(dummy.expand_as(captured.out_loc))
        captured.out_loc[:n].copy_(batch.out_loc)
        src, dst = batch.attn_metadata, captured.attn_metadata
        dst.cu_seqlens_q_gpu.copy_(src.cu_seqlens_q_gpu)
        dst.indptr.copy_(src.indptr)
        dst.prefix_lens.copy_(src.prefix_lens)
        dst.indices[: src.indices.numel()].copy_(src.indices)
        if dst.swa_indices is not None:
            dst.swa_indices[: src.swa_indices.numel()].copy_(src.swa_indices)
        return graph, captured, logits

    def replay_prefill(self, batch):
        if hasattr(self, "dsv4_prefill"):
            return self.dsv4_prefill.replay(batch)
        if hasattr(self, "qwen_prefill"):
            return self.qwen_prefill.replay(batch)
        graph, captured, logits = self._stage_prefill(batch)
        graph.replay()
        return logits[:batch.size]

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs, vocab_size, self.device, mrope=self.mrope
        )
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if batch.is_prefill:
            if hasattr(self, "dsv4_prefill"):
                return self.dsv4_prefill.can_use(batch)
            if hasattr(self, "qwen_prefill"):
                return self.qwen_prefill.can_use(batch)
            return (batch.mm_embeds is None and batch.mm_gather_plan is None and batch.mm_block_ends is None
                    and hasattr(batch, "input_ids") and self._prefill_key(batch) in self.prefill_graph_map)
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        if batch.is_prefill:
            return self.replay_prefill(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        if batch.is_prefill:
            batch.padded_reqs = batch.reqs
            return
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        if hasattr(self, "dsv4_prefill"):
            self.dsv4_prefill.destroy()
            del self.dsv4_prefill
        if hasattr(self, "qwen_prefill"):
            if hasattr(self.qwen_prefill, "destroy"):
                self.qwen_prefill.destroy()
            self.qwen_prefill.graphs.clear()
            del self.qwen_prefill
        self.prefill_graph_map = {}
        self.buffer = None
        gc.collect()
