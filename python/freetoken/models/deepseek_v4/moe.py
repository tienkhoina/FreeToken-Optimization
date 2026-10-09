"""DSV4 MoE: sqrtsoftplus/hash router, shared SwiGLU expert, offloaded MXFP4 routed
experts (GPU slot-cache / cpu / hybrid decode paths)."""

from __future__ import annotations

from copy import copy
from dataclasses import replace

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearRowParallel, OffloadMoELayer
from freetoken.layers.quantization import QuantKind

from .args import DeepseekV4Args


class Gate(BaseOP):
    """MoE router: sqrtsoftplus scoring + hash routing (first ``n_hash_layers``)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16)
        if self.hash:
            self.tid2eid = torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int64)
            self.bias = None
        else:
            self.bias = torch.empty(args.n_routed_experts, dtype=torch.float32)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        scores = bf16_linear_fp32(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(BaseOP):
    """Dense SwiGLU expert (the shared expert; routed experts are offloaded FP4)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float, *, quant_config=None, prefix: str = "", fuse_gate_up: bool = False):
        self.w1 = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w1")
        self.w2 = LinearRowParallel(inter_dim, dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w2")
        self.w3 = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w3")
        self.swiglu_limit = swiglu_limit
        self._gate_up = None
        self._fuse_gate_up = fuse_gate_up
        if fuse_gate_up:
            for projection in (self.w1, self.w3):
                if projection.quant_method.kind not in (QuantKind.NONE, QuantKind.FP8_BLOCK):
                    raise ValueError("DeepSeek-V4 shared-expert fusion requires BF16 or block-FP8 weights")
            if self.w1.quant_method.scheme != self.w3.quant_method.scheme:
                raise ValueError("DeepSeek-V4 shared gate/up use different quantization schemes")

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        if getattr(self, "_fuse_gate_up", False):
            self._prepare_gate_up()

    def _prepare_gate_up(self):
        weight = torch.cat((self.w1.weight, self.w3.weight), dim=0)
        # Preserve the original checkpoint keys as views, without retaining duplicate weights.
        self.w1.weight, self.w3.weight = weight.chunk(2, dim=0)
        merged = self._gate_up = copy(self.w1)
        merged.weight = weight
        for name in ("full_output_size", "local_output_size", "out_features"):
            setattr(merged, name, 2 * getattr(merged, name))
        merged.output_sizes = self.w1.output_sizes + self.w3.output_sizes
        merged.quant_method = copy(self.w1.quant_method)
        merged.quant_method.cfg = replace(self.w1.quant_method.cfg, out_features=merged.out_features,
                                           output_sizes=merged.output_sizes)
        if hasattr(self.w1, "weight_scale_inv"):
            scale = torch.cat((self.w1.weight_scale_inv, self.w3.weight_scale_inv), dim=0)
            self.w1.weight_scale_inv, self.w3.weight_scale_inv = scale.chunk(2, dim=0)
            merged.weight_scale_inv = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gate_up is None:
            gate, up = self.w1.forward(x), self.w3.forward(x)
        else:
            gate, up = self._gate_up.forward(x).chunk(2, dim=-1)
        h = fused_swiglu(gate, up, self.swiglu_limit, x.dtype)
        return self.w2.forward(h)


class DSV4OffloadMoELayer(OffloadMoELayer):
    """Routed MXFP4 experts on the shared offload cache: the base whole-layer
    streaming prefill (grouped inline-dequant GEMM for dense chunks, GEMV
    below the route crossover) and slot-cache / cpu / hybrid decode paths
    (per-route dequant GEMV)."""

    prefill_decode_max_tokens = 1

    def __init__(self, layer_id: int, args: DeepseekV4Args, *, strategy: str = "offload", decode_target: str = "gpu", quant_config=None, prefix: str = ""):
        scheme = quant_config.scheme_for(prefix) if quant_config is not None else None
        nvfp4 = scheme is not None and scheme.kind is QuantKind.NVFP4
        super().__init__(
            layer_id=layer_id,
            num_experts=args.n_routed_experts,
            top_k=args.n_activated_experts,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=True,
            activation="swiglu_clamp" if nvfp4 and args.swiglu_limit > 0 else "silu",
            limit=args.swiglu_limit if not nvfp4 or args.swiglu_limit > 0 else None,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Whole-layer streaming moves all num_experts rows per layer; a small
        # chunk touches at most T*top_k of them, so below that crossover the
        # decode-style on-demand slot path strictly moves fewer bytes (and
        # keeps short-prompt slot residency -- hence hybrid decode's GPU/CPU
        # route split -- unchanged). Mixing modes across chunks is safe: the
        # streaming buffers disown their borrowed slots on invalidation.
        cache = self.offload_cache
        assert cache is not None
        if cache.scheduler is not None:
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        # unpinned (LOCKED) layers must take the base materialize path: their copy_missing is the whole-layer pageable branch with position == expert id, which ensure_experts's LRU slot remap would contradict (the GEMM would gather other experts' weights)
        if (
            hidden_states.shape[0] * self.top_k >= self.num_experts
            or cache.is_unpinned_layer(self.layer_id)
        ):
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)  # in-place expert-id -> slot
        cache.copy_missing()
        if cache.collect_stats:
            cache.record_decode_stats(self.layer_id)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=True,
        )


class MoE(BaseOP):
    """Sparse MoE: hash/score router -> offloaded MXFP4 routed experts + shared expert."""

    def __init__(self, layer_id: int, args: DeepseekV4Args, *, strategy: str = "offload", decode_target: str = "gpu", quant_config=None, prefix: str = ""):
        self.dim = args.dim
        self.gate = Gate(layer_id, args)
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit, quant_config=quant_config, prefix=f"{prefix}.shared_experts", fuse_gate_up=args.fuse_shared_expert)
        self.experts = DSV4OffloadMoELayer(layer_id, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.experts")

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate.forward(x, input_ids.flatten())
        # Shared expert enqueued before routed_forward: hybrid decode blocks on the
        # CPU pool inside routed_forward, so this GEMM must already be on the stream
        # to overlap the CPU overflow compute.
        shared = self.shared_experts.forward(x)
        # routed_forward may mutate the ids in place (offload decode slot remap);
        # indices.to(int32) always copies (int64 source), so no clone needed here.
        routed = self.experts.routed_forward(
            x, weights.float().contiguous(), indices.to(torch.int32).contiguous()
        )
        return (routed + shared).view(shape)
