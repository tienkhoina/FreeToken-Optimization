"""DeepSeek-V4-Flash hyperparameters.

Field names mirror the authors' ``inference/config.json`` (consumed by the
reference ``ModelArgs``) so the port stays 1:1 with the reference. ``load_args``
reads that file from the checkpoint directory (it ships alongside the weights);
the few runtime knobs (batch / sequence length) are overlaid by the runner.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Literal, Tuple


@dataclass
class DeepseekV4Args:
    # ----- runtime -----
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = "fp4"
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    fuse_shared_expert: bool = False

    # ----- shape -----
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_hash_layers: int = 3
    n_mtp_layers: int = 1
    n_heads: int = 64

    # ----- moe -----
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    # ----- mla -----
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int, ...] = (0, 0, 4, 128, 4, 128, 4, 0)

    # ----- rope / yarn -----
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1

    # ----- lightning indexer -----
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512

    # ----- hyper-connections -----
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    def __post_init__(self) -> None:
        # JSON lists -> tuple so the dataclass stays hashable / immutable-ish.
        if isinstance(self.compress_ratios, list):
            self.compress_ratios = tuple(self.compress_ratios)

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim


def _config_path(model_path: str) -> str:
    """Prefer the reference ModelArgs JSON, falling back to a Hugging Face export."""
    candidates = [
        os.path.join(model_path, "inference", "config.json"),
        os.path.join(model_path, "model_args.json"),
        os.path.join(model_path, "config.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No DeepSeek-V4 ModelArgs JSON found under {model_path} "
        f"(looked for inference/config.json, model_args.json and config.json)"
    )


def load_args(model_path: str, **overrides) -> DeepseekV4Args:
    """Build :class:`DeepseekV4Args` from reference or Hugging Face configuration.

    ``overrides`` (e.g. ``max_seq_len``, ``max_batch_size``) take precedence over the
    file, letting the runner size the per-request caches.
    """
    with open(_config_path(model_path)) as f:
        raw = json.load(f)
    if "model_type" in raw:
        if raw["model_type"] != "deepseek_v4":
            raise ValueError(f"Expected deepseek_v4 config, got {raw['model_type']!r}")
        aliases = {
            "hidden_size": "dim", "moe_intermediate_size": "moe_inter_dim",
            "num_hidden_layers": "n_layers", "num_hash_layers": "n_hash_layers",
            "num_nextn_predict_layers": "n_mtp_layers", "num_attention_heads": "n_heads",
            "num_experts_per_tok": "n_activated_experts", "scoring_func": "score_func",
            "routed_scaling_factor": "route_scale", "qk_rope_head_dim": "rope_head_dim",
            "rms_norm_eps": "norm_eps", "sliding_window": "window_size",
        }
        raw = {aliases.get(key, key): value for key, value in raw.items()}
        rope = raw.get("rope_scaling") or raw.get("rope_parameters") or {}
        for source, target in (("factor", "rope_factor"), ("original_max_position_embeddings", "original_seq_len"),
                               ("beta_fast", "beta_fast"), ("beta_slow", "beta_slow")):
            if source in rope:
                raw[target] = rope[source]
    valid = {f.name for f in fields(DeepseekV4Args)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    kwargs.update(overrides)
    args = DeepseekV4Args(**kwargs)
    if len(args.compress_ratios) < args.n_layers:
        raise ValueError(f"compress_ratios has {len(args.compress_ratios)} entries for {args.n_layers} layers")
    return args


__all__ = ["DeepseekV4Args", "load_args"]
