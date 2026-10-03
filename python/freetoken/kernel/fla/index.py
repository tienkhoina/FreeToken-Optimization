# Local modifications: FreeToken-Optimization workspace, 2026-10-03.
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from freetoken.kernel.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    plans = getattr(cu_seqlens, "_freetoken_chunk_plans", None)
    if plans is not None and chunk_size in plans:
        return plans[chunk_size][0]
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    plans = getattr(cu_seqlens, "_freetoken_chunk_plans", None)
    if plans is not None and chunk_size in plans:
        return plans[chunk_size][1]
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


def attach_chunk_plans(cu_seqlens: torch.Tensor, lengths, chunk_sizes=(16, 32, 64)):
    """Build host-known launch geometry before capture, without device readback."""
    plans = {}
    for size in chunk_sizes:
        rows, offsets = [], [0]
        for request, length in enumerate(lengths):
            chunks = (int(length) + size - 1) // size
            rows.extend((request, index) for index in range(chunks))
            offsets.append(offsets[-1] + chunks)
        pin = cu_seqlens.is_cuda
        indices = torch.tensor(rows, dtype=cu_seqlens.dtype, pin_memory=pin).reshape(-1, 2)
        off = torch.tensor(offsets, dtype=cu_seqlens.dtype, pin_memory=pin)
        plans[size] = (indices.to(cu_seqlens.device, non_blocking=True),
                       off.to(cu_seqlens.device, non_blocking=True))
    cu_seqlens._freetoken_chunk_plans = plans
    return cu_seqlens
