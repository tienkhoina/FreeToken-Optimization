"""Device-length prefill stores and compressor pooling for padded DeepSeek buckets."""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_rows(source, destination, slots, count, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                SOURCE_STRIDE: tl.constexpr, DEST_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(count)
    slot = tl.load(slots + row, mask=row < n, other=-1).to(tl.int64)
    valid = (row < n) & (slot >= 0) & (col < WIDTH)
    value = tl.load(source + row * SOURCE_STRIDE + col, mask=valid, other=0)
    tl.store(destination + slot * DEST_STRIDE + col, value, mask=valid)


def store_rows(source, destination, slots, count):
    _store_rows[(source.shape[0], triton.cdiv(source.shape[1], 128))](
        source, destination, slots, count, source.shape[0], source.shape[1],
        source.stride(0), destination.stride(0), 128)


@triton.jit
def _pool_blocks(kv, score, ape, ring, windows, descriptor, output,
                 WIDTH: tl.constexpr, RATIO: tl.constexpr, OVERLAP: tl.constexpr,
                 PAGE: tl.constexpr, CONTEXT: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    row = tl.arange(0, RATIO * (2 if OVERLAP else 1))
    n, start = tl.load(descriptor), tl.load(descriptor + 1)
    valid = block < n // RATIO
    item = WIDTH * (2 if OVERLAP else 1)
    if OVERLAP:
        previous = row < RATIO
        token = (block - previous.to(tl.int32)) * RATIO + row % RATIO
        column = col[None, :] + tl.where(previous[:, None], 0, WIDTH)
        seeded = previous & (block == 0)
    else:
        token = block * RATIO + row
        column = col[None, :]
        seeded = tl.full((RATIO,), False, tl.int1)
    memory = valid & (token[:, None] >= 0) & (col[None, :] < WIDTH)
    values = tl.load(kv + token[:, None] * item + column, mask=memory, other=0.0).to(tl.float32)
    scores = tl.load(score + token[:, None] * item + column, mask=memory, other=float('-inf')).to(tl.float32)
    offset = tl.load(ape + (row % RATIO)[:, None] * item + column,
                     mask=col[None, :] < WIDTH, other=0.0)
    scores += offset
    tail = tl.load(windows + tl.maximum(start - 1, 0), mask=(start > 0) & (start <= CONTEXT), other=-1).to(tl.int64)
    ring_row = (tail // PAGE) * RATIO * (2 if OVERLAP else 1) + row % RATIO
    seed_mask = valid & seeded[:, None] & (start > 0) & (tail >= 0) & (col[None, :] < WIDTH)
    seed_k = tl.load(ring + ring_row[:, None] * (2 * item) + col[None, :], mask=seed_mask, other=0.0)
    seed_s = tl.load(ring + ring_row[:, None] * (2 * item) + item + col[None, :],
                     mask=seed_mask, other=float('-inf'))
    values = tl.where(seeded[:, None], seed_k, values)
    scores = tl.where(seeded[:, None], seed_s, scores)
    maximum = tl.max(scores, axis=0)
    maximum = tl.where(maximum == float('-inf'), 0.0, maximum)
    weights = tl.exp(scores - maximum[None, :])
    denominator = tl.sum(weights, axis=0)
    result = tl.sum(weights * values, axis=0) / tl.maximum(denominator, 1.0e-20)
    tl.store(output + block * WIDTH + col, result, mask=col < WIDTH)


@triton.jit
def _commit_carries(kv, score, ape, ring, windows, descriptor,
                    WIDTH: tl.constexpr, RATIO: tl.constexpr, OVERLAP: tl.constexpr,
                    PAGE: tl.constexpr, CONTEXT: tl.constexpr, BLOCK: tl.constexpr):
    page, r = tl.program_id(0), tl.program_id(1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    n, start = tl.load(descriptor), tl.load(descriptor + 1)
    end = tl.minimum((page + 1) * PAGE, n)
    live = page * PAGE < n
    cutoff = (end // RATIO) * RATIO
    remainder = end - cutoff
    item = WIDTH * (2 if OVERLAP else 1)
    if OVERLAP:
        first = r < RATIO
        token = tl.where(first, cutoff - RATIO + r, cutoff + r - RATIO)
        present = tl.where(first, cutoff >= RATIO, r - RATIO < remainder)
        seeded = first & (cutoff == 0) & (start > 0)
    else:
        token = cutoff + r
        present = r < remainder
        seeded = False
    mask = live & present & (col < item)
    value = tl.load(kv + token.to(tl.int64) * item + col, mask=mask, other=0.0)
    scores = tl.load(score + token.to(tl.int64) * item + col, mask=mask, other=float('-inf'))
    offset = tl.load(ape + (r % RATIO) * item + col, mask=col < item, other=0.0)
    scores += offset
    tail = tl.load(windows + tl.maximum(start - 1, 0), mask=(start > 0) & (start <= CONTEXT), other=-1).to(tl.int64)
    seed_row = (tail // PAGE) * RATIO * (2 if OVERLAP else 1) + r
    seed_mask = live & seeded & (tail >= 0) & (col < item)
    seed_k = tl.load(ring + seed_row * (2 * item) + col, mask=seed_mask, other=0.0)
    seed_s = tl.load(ring + seed_row * (2 * item) + item + col, mask=seed_mask, other=float('-inf'))
    value = tl.where(seeded, seed_k, value)
    scores = tl.where(seeded, seed_s, scores)
    slot = tl.load(windows + start + end - 1, mask=live & (start + end <= CONTEXT), other=-1).to(tl.int64)
    destination = (slot // PAGE) * RATIO * (2 if OVERLAP else 1) + r
    write = live & (slot >= 0) & (col < item)
    tl.store(ring + destination * (2 * item) + col, value, mask=write)
    tl.store(ring + destination * (2 * item) + item + col, scores, mask=write)


def pool_blocks(kv, score, compressor, metadata, dtype):
    ratio, width = compressor.compress_ratio, compressor.head_dim
    output = torch.empty((1, triton.cdiv(kv.shape[1], ratio), width), dtype=dtype, device=kv.device)
    ring = compressor.state_ring.buffer
    _pool_blocks[(output.shape[1], triton.cdiv(width, 64))](
        kv, score, compressor.ape, ring, metadata.window_snap, metadata.descriptor, output,
        width, ratio, compressor.overlap, compressor.P, metadata.window_snap.numel(), 64, num_warps=4)
    return output


def commit_carries(kv, score, compressor, metadata):
    ring = compressor.state_ring.buffer
    _commit_carries[(triton.cdiv(kv.shape[1], compressor.P), compressor.ring_size,
                     triton.cdiv(compressor.item_size, 128))](
        kv, score, compressor.ape, ring, metadata.window_snap, metadata.descriptor,
        compressor.head_dim, compressor.compress_ratio, compressor.overlap, compressor.P,
        metadata.window_snap.numel(), 128, num_warps=4)
