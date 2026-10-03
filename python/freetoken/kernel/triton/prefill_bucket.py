"""Fixed-capacity prefill plans and state operations with runtime logical lengths."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bucket_metadata(desc, raw_ids, raw_loc, raw_pos, raw_mrope, pages,
                     ids, loc, pos, mrope, cu, slots, has_initial, qcu, mapping,
                     seq_lens, ring_slots, table, last, track_dst, track_h, track_end,
                     ROWS: tl.constexpr, BS: tl.constexpr, PAGE_WIDTH: tl.constexpr,
                     PAGE_SIZE: tl.constexpr, TABLE_WIDTH: tl.constexpr,
                     RAW_CAP: tl.constexpr, MROPE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    request = tl.full((BLOCK,), -1, tl.int32)
    source = tl.zeros((BLOCK,), tl.int64)
    for r in range(BS):
        offset = tl.load(desc + r * 8 + 3)
        n = tl.load(desc + r * 8 + 2)
        source_start = tl.load(desc + r * 8 + 5)
        selected = (i >= offset) & (i < offset + n)
        request = tl.where(selected, r, request)
        source = tl.where(selected, source_start + i - offset, source)
    valid = (i < ROWS) & (request >= 0)
    tl.store(ids + i, tl.load(raw_ids + source, valid, other=0), i < ROWS)
    tl.store(loc + i, tl.load(raw_loc + source, valid, other=-1), i < ROWS)
    tl.store(pos + i, tl.load(raw_pos + source, valid, other=0), i < ROWS)
    tl.store(mapping + i, request, i < ROWS)
    if MROPE:
        for axis in tl.static_range(3):
            value = tl.load(raw_mrope + axis * RAW_CAP + source, valid, other=0)
            tl.store(mrope + axis * ROWS + i, value, i < ROWS)
    r = i
    active = r < BS
    offset = tl.load(desc + r * 8 + 3, active, other=0)
    n = tl.load(desc + r * 8 + 2, active, other=0)
    prefix = tl.load(desc + r * 8 + 1, active, other=0)
    slot = tl.load(desc + r * 8 + 4, active, other=0)
    page_row = tl.load(desc + r * 8, active, other=0)
    tl.store(cu + r, offset, active)
    tl.store(qcu + r, offset, active)
    final = tl.load(desc + (BS - 1) * 8 + 3) + tl.load(desc + (BS - 1) * 8 + 2)
    if tl.program_id(0) == 0:
        tl.store(cu + BS, final)
        tl.store(qcu + BS, final)
    tl.store(slots + r, slot, active)
    tl.store(has_initial + r, prefix > 0, active)
    tl.store(seq_lens + r, prefix + n, active)
    tl.store(ring_slots + r, page_row, active)
    tl.store(last + r, tl.maximum(offset + n - 1, 0), active)
    dst = tl.load(desc + r * 8 + 6, active, other=-1)
    boundary = tl.load(desc + r * 8 + 7, active, other=-1)
    h_offset = tl.zeros((BLOCK,), tl.int32)
    for previous in range(BS):
        previous_n = tl.load(desc + previous * 8 + 2)
        h_offset += tl.where(previous < r, tl.cdiv(previous_n, 64), 0)
    tl.store(track_dst + r, dst, active)
    tl.store(track_h + r, tl.where(boundary == n, -1, h_offset + boundary // 64), active)
    tl.store(track_end + r, offset + boundary, active)
    total_table = BS * TABLE_WIDTH
    table_req, column = i // TABLE_WIDTH, i % TABLE_WIDTH
    src_row = tl.load(desc + table_req * 8, i < total_table, other=0)
    page = tl.load(pages + src_row.to(tl.int64) * PAGE_WIDTH + column * PAGE_SIZE,
                   i < total_table, other=0)
    tl.store(table + i, page // PAGE_SIZE, i < total_table)


def prepare_bucket(batch, desc, raw, page_table, page_size):
    md, fla = batch.attn_metadata, batch.fla_metadata
    rows, bs = batch.input_ids.numel(), fla.cache_indices.numel()
    mrope = batch.mrope_positions
    _bucket_metadata[(triton.cdiv(max(rows, bs * md.block_table.shape[1]), 128),)](
        desc, raw[0], raw[1], raw[2], raw[3], page_table,
        batch.input_ids, batch.out_loc, batch.positions, mrope if mrope is not None else batch.positions,
        fla.cu_seqlens, fla.cache_indices, fla.has_initial_state, md.cu_seqlens, md.token_to_req,
        md.seq_lens, md.ring_slots, md.block_table, md.last_indices,
        fla.track_dst, fla.track_h_row, fla.track_boundary_row,
        rows, bs, page_table.shape[1], page_size, md.block_table.shape[1],
        raw[0].numel(), mrope is not None, 128, num_warps=1)
    update_chunk_plans(fla.cu_seqlens)


@triton.jit
def _chunk_plan(cu, indices, offsets, CAP: tl.constexpr, BS: tl.constexpr,
                SIZE: tl.constexpr, BLOCK: tl.constexpr):
    requests = tl.arange(0, BLOCK)
    starts = tl.load(cu + requests, requests < BS, other=0)
    ends = tl.load(cu + requests + 1, requests < BS, other=0)
    counts = tl.cdiv(ends - starts, SIZE)
    cumulative = tl.cumsum(counts)
    previous = cumulative - counts
    row = tl.program_id(0)
    selected = (row >= previous) & (row < cumulative) & (requests < BS)
    req = tl.sum(tl.where(selected, requests, 0))
    local = row - tl.sum(tl.where(selected, previous, 0))
    valid = tl.sum(selected.to(tl.int32)) > 0
    tl.store(indices + 2 * row, req)
    tl.store(indices + 2 * row + 1, tl.where(valid, local, CAP + 1))
    if row == 0:
        tl.store(offsets + requests, previous, requests < BS)
        tl.store(offsets + BS, tl.sum(counts))


def allocate_chunk_plans(cu, tokens, requests):
    plans = {
        size: (torch.empty((triton.cdiv(tokens, size) + requests, 2), dtype=cu.dtype, device=cu.device),
               torch.empty(requests + 1, dtype=cu.dtype, device=cu.device))
        for size in (16, 32, 64)
    }
    cu._freetoken_chunk_plans = plans
    return plans


def update_chunk_plans(cu):
    bs = cu.numel() - 1
    for size, (indices, offsets) in cu._freetoken_chunk_plans.items():
        _chunk_plan[(indices.shape[0],)](cu, indices, offsets, indices.shape[0], bs,
                                       size, triton.next_power_of_2(bs), num_warps=1)


@triton.jit
def _token_requests(cu, mapping, TOKENS: tl.constexpr, BS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.full((BLOCK,), -1, tl.int32)
    for i in range(BS):
        start, end = tl.load(cu + i), tl.load(cu + i + 1)
        req = tl.where((rows >= start) & (rows < end), i, req)
    tl.store(mapping + rows, req, rows < TOKENS)


def update_token_requests(cu, mapping):
    _token_requests[(triton.cdiv(mapping.numel(), 128),)](
        cu, mapping, mapping.numel(), cu.numel() - 1, 128, num_warps=1)


@triton.jit
def _clear_fresh(state, cu, slots, has_initial, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    start, end = tl.load(cu + req), tl.load(cu + req + 1)
    if end <= start or tl.load(has_initial + req):
        return
    slot = tl.load(slots + req).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(state + slot * WIDTH + cols, 0, cols < WIDTH)


def clear_fresh_state(state, fla):
    width = state[0].numel()
    _clear_fresh[(fla.cache_indices.numel(), triton.cdiv(width, 256))](
        state, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state, width, 256)


@triton.jit
def _copy_history(x, states, ends, slots, XWIDTH: tl.constexpr, XROW: tl.constexpr, HISTORY: tl.constexpr,
                  BLOCK: tl.constexpr):
    req = tl.program_id(0)
    slot = tl.load(slots + req).to(tl.int64)
    end = tl.load(ends + req).to(tl.int64)
    if slot < 0 or end < HISTORY:
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    for j in tl.static_range(HISTORY):
        value = tl.load(x + (end - HISTORY + j) * XROW + cols, cols < XWIDTH, other=0)
        tl.store(states + (slot * XWIDTH + cols) * HISTORY + j, value, cols < XWIDTH)


def copy_history_window(x, states, ends, slots):
    _copy_history[(slots.numel(), triton.cdiv(x.shape[1], 128))](
        x, states, ends, slots, x.shape[1], x.stride(0), states.shape[-1], 128)


@triton.jit
def _copy_recurrent(h, states, rows, destinations, live_slots, WIDTH: tl.constexpr,
                    BLOCK: tl.constexpr, ROUND_LIVE: tl.constexpr):
    req = tl.program_id(0)
    dst = tl.load(destinations + req).to(tl.int64)
    if dst < 0:
        return
    row = tl.load(rows + req).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if row >= 0:
        value = tl.load(h + row * WIDTH + cols, cols < WIDTH, other=0)
    else:
        live = tl.load(live_slots + req).to(tl.int64)
        value = tl.load(states + live * WIDTH + cols, cols < WIDTH, other=0)
        if ROUND_LIVE:
            value = value.to(h.dtype.element_ty)
    tl.store(states + dst * WIDTH + cols, value, cols < WIDTH)


def copy_recurrent_snapshot(h, states, fla):
    width = states[0].numel()
    _copy_recurrent[(fla.track_dst.numel(), triton.cdiv(width, 256))](
        h, states, fla.track_h_row, fla.track_dst, fla.cache_indices, width, 256, True)


@triton.jit
def _ple_conv(x, old_state, weights, cu, mapping, out,
              WIDTH: tl.constexpr, HISTORY: tl.constexpr, TAPS: tl.constexpr,
              DILATION: tl.constexpr, TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(mapping + row)
    if req < 0:
        tl.store(out + row * WIDTH + cols, 0, cols < WIDTH)
        return
    start = tl.load(cu + req).to(tl.int64)
    local = row - start
    acc = tl.zeros((BLOCK,), tl.float32)
    for tap in tl.static_range(TAPS):
        relative = local - HISTORY + tap * DILATION
        current = tl.load(x + (start + relative) * WIDTH + cols,
                          (relative >= 0) & (cols < WIDTH), other=0).to(tl.float32)
        previous = tl.load(old_state + (req * WIDTH + cols) * HISTORY + relative + HISTORY,
                           (relative < 0) & (relative >= -HISTORY) & (cols < WIDTH), other=0).to(tl.float32)
        value = tl.where(relative >= 0, current, previous)
        weight = tl.load(weights + cols * TAPS + tap, cols < WIDTH, other=0).to(tl.float32)
        acc += value * weight
    acc = acc.to(x.dtype.element_ty).to(tl.float32)
    acc = acc / (1.0 + tl.exp(-acc))
    tl.store(out + row * WIDTH + cols, acc, cols < WIDTH)


@triton.jit
def _ple_conv_state(x, old_state, states, cu, slots, WIDTH: tl.constexpr,
                    HISTORY: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    start, end = tl.load(cu + req).to(tl.int64), tl.load(cu + req + 1).to(tl.int64)
    if end <= start:
        return
    dst = tl.load(slots + req).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    for j in tl.static_range(HISTORY):
        pos = end - HISTORY + j
        value = tl.load(x + pos * WIDTH + cols, (pos >= start) & (cols < WIDTH), other=0)
        prev = tl.load(old_state + (req * WIDTH + cols) * HISTORY + pos - start + HISTORY,
                       (pos < start) & (pos - start + HISTORY >= 0) & (cols < WIDTH), other=0)
        tl.store(states + (dst * WIDTH + cols) * HISTORY + j,
                 tl.where(pos >= start, value, prev), cols < WIDTH)


def ple_conv_varlen(x, old_state, states, weights, meta, dilation):
    out = torch.empty_like(x)
    width, history = x.shape[1], states.shape[-1]
    _ple_conv[(x.shape[0], triton.cdiv(width, 128))](
        x, old_state, weights, meta.cu_seqlens, meta.token_to_req, out,
        width, history, weights.shape[-1], dilation, x.shape[0], 128)
    _ple_conv_state[(meta.state_slots.numel(), triton.cdiv(width, 128))](
        x, old_state, states, meta.cu_seqlens, meta.state_slots, width, history, 128)
    return out


@triton.jit
def _ngram_hash(ids, cu, mapping, context, multipliers, vocab, offsets, out,
                TOKENS: tl.constexpr, HEADS: tl.constexpr, PER_NGRAM: tl.constexpr,
                NGRAM: tl.constexpr, EOS: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    heads = tl.arange(0, BLOCK_H)
    req = tl.load(mapping + row)
    if req < 0:
        tl.store(out + row * HEADS + heads, -1, heads < HEADS)
        return
    start = tl.load(cu + req).to(tl.int64)
    length = 2 + heads // PER_NGRAM
    mixed = tl.full((BLOCK_H,), 0, tl.int64)
    crossed = False
    for shift in tl.static_range(NGRAM):
        pos = row - start - shift
        value = tl.load(ids + row - shift, pos >= 0, other=EOS).to(tl.int64)
        old = tl.load(context + req * (NGRAM - 1) + pos + NGRAM - 1,
                      (pos < 0) & (pos >= -(NGRAM - 1)), other=EOS)
        value = tl.where(pos >= 0, value, old)
        if shift > 0:
            crossed = crossed | (value == EOS)
            value = tl.where(crossed, EOS, value)
        multiplier = tl.load(multipliers + shift)
        mixed ^= tl.where(shift < length, value * multiplier, 0)
    modulus = tl.load(vocab + heads, heads < HEADS, other=1)
    base = tl.load(offsets + heads, heads < HEADS, other=0)
    result = mixed % modulus
    result = tl.where(result < 0, result + modulus, result)
    tl.store(out + row * HEADS + heads, result + base, heads < HEADS)


def ngram_hash_varlen(meta, embedding):
    ids = meta.input_ids
    out = torch.empty(ids.numel(), embedding.num_heads, dtype=torch.int64, device=ids.device)
    _ngram_hash[(ids.numel(),)](
        ids, meta.cu_seqlens, meta.token_to_req, meta.ngram_context,
        embedding.layer_multipliers, embedding.ngram_heads_vocab_sizes, embedding.ngram_heads_offsets,
        out, ids.numel(), embedding.num_heads, embedding.heads_per_ngram,
        embedding.ngram_size, embedding.eos_token_id, triton.next_power_of_2(embedding.num_heads), num_warps=1)
    return out


@triton.jit
def _commit_context(ids, cu, old, states, slots, track_ends, track_slots,
                    HISTORY: tl.constexpr, TRACK: tl.constexpr):
    req = tl.program_id(0)
    start, end = tl.load(cu + req).to(tl.int64), tl.load(cu + req + 1).to(tl.int64)
    if end <= start:
        return
    dst = tl.load(slots + req).to(tl.int64)
    for j in tl.static_range(HISTORY):
        pos = end - HISTORY + j
        value = tl.load(ids + pos, pos >= start, other=0)
        previous = tl.load(old + req * HISTORY + pos - start + HISTORY,
                           (pos < start) & (pos - start + HISTORY >= 0), other=0)
        tl.store(states + dst * HISTORY + j, tl.where(pos >= start, value, previous))
    if TRACK:
        target = tl.load(track_slots + req).to(tl.int64)
        boundary = tl.load(track_ends + req).to(tl.int64)
        if target >= 0 and boundary >= HISTORY:
            for j in tl.static_range(HISTORY):
                tl.store(states + target * HISTORY + j, tl.load(ids + boundary - HISTORY + j))


def commit_context(meta, fla, states):
    track = fla.track_dst is not None
    _commit_context[(meta.state_slots.numel(),)](
        meta.input_ids, meta.cu_seqlens, meta.ngram_context, states, meta.state_slots,
        fla.track_boundary_row if track else meta.cu_seqlens,
        fla.track_dst if track else meta.state_slots, meta.ngram_context.shape[1], track, num_warps=1)
