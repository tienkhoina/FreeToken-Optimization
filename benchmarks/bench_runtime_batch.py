"""Measure the combined input/KV/metadata/attention-graph/sampling runtime path."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from freetoken.attention.triton import TritonAttentionBackend
from freetoken.core import Batch, Context, SamplingParams, set_global_ctx
from freetoken.engine.graph import GraphCaptureBuffer
from freetoken.engine.sample import BatchSamplingArgs, Sampler
from freetoken.kernel.triton.attention import decode_paged_attention
from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
from freetoken.scheduler.cache import _write_page_table
from freetoken.scheduler.scheduler import _make_input_tuple, _make_positions, _make_write_tuple


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def measure(fn, samples, iterations):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    rows = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        enqueue = time.perf_counter()
        torch.cuda.synchronize()
        rows.append({"wall_ms": (time.perf_counter() - start) * 1000 / iterations,
                     "host_enqueue_ms": (enqueue - start) * 1000 / iterations})
    return {"median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
            "median_host_enqueue_ms": statistics.median(row["host_enqueue_ms"] for row in rows), "samples": rows}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--save-tensors", action="store_true")
    parser.add_argument("--outputs-only", action="store_true", help="save correctness/memory artifacts without latency samples")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    native = importlib.util.find_spec("freetoken.scheduler.input_staging") is not None
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "torch_threads": 1,
              "native_input_staging": native, "samples": args.samples, "iterations": args.iterations,
              "method": "Compilation/warmup excluded; each step runs host input staging, KV metadata, graph input copy, warmed attention graph and greedy sampling. Includes host work and GPU work; model MLP/MoE math not present. No profiler.",
              "cases": [], "components": {}}
    if native:
        from freetoken.scheduler.input_staging import BatchInputStager
    for bs, width, decode in [(1, 1024, True), (1, 8192, True), (4, 1024, True), (16, 1024, True), (1, 1024, False), (4, 1024, False)]:
        torch.manual_seed(919 + bs + width + decode)
        ctx.page_table = torch.randperm((bs + 1) * width, device=device).to(torch.int32).view(bs + 1, width)
        token_pool = torch.randint(0, 201088, ctx.page_table.shape, dtype=torch.int32, device=device)
        mapping = torch.randperm(ctx.page_table.numel(), device=device).to(torch.int64)
        ctx.kv_cache = SimpleNamespace(device=device, swa_paged=True, full_to_swa_index_mapping=mapping,
                        translate_loc_from_full_to_swa=lambda ids: mapping[ids.long()].to(torch.int32))
        backend = TritonAttentionBackend(SimpleNamespace(num_qo_heads=4, head_dim=64))
        backend.init_capture_graph(width, [bs])
        buffer = GraphCaptureBuffer.init(bs, 1, device)
        reqs = [SimpleNamespace(table_idx=i, cached_len=width - 1 - i * 7 if decode else i * 11,
                                device_len=width - i * 7 if decode else i * 11 + 257 - i * 23,
                                extend_len=1 if decode else 257 - i * 23, can_decode=True,
                                sampling_params=SamplingParams()) for i in range(bs)]
        batch = Batch(reqs=reqs, phase="decode" if decode else "prefill")
        batch.padded_reqs = reqs
        stager = BatchInputStager(device, bs, max(bs, sum(r.extend_len for r in reqs))) if native else None
        sampler = Sampler(device, 201088)
        if native:
            sampler.warmup(bs)
        logits = torch.randn(bs, 201088, device=device, dtype=torch.float32)
        if decode:
            captured = SimpleNamespace(size=bs)
            backend.prepare_for_capture(captured)
            md = captured.attn_metadata
            q = torch.randn(bs, 4, 64, device=device, dtype=torch.bfloat16)
            k = torch.randn(token_pool.numel(), 2, 64, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            def attention():
                return decode_paged_attention(q=q, k_cache=k, v_cache=v, indptr=md.indptr,
                    indices=md.swa_indices, q_positions=md.q_positions, attn_logits=md.attn_logits,
                    attn_lse=md.attn_lse, num_kv_splits=md.num_kv_splits,
                    max_kv_splits=backend.max_kv_splits, sm_scale=0.125)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                attention()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = attention()
            torch.cuda.current_stream().wait_stream(stream)
        def prepare():
            if native:
                incoming, outgoing = stager.prepare(batch, ctx.page_table, token_pool)
            else:
                batch.positions = _make_positions(batch, device)
                incoming = _make_input_tuple(batch, device)
                outgoing = _make_write_tuple(batch, device)
                batch.out_loc = ctx.page_table[incoming]
                batch.input_ids = token_pool[incoming]
            backend.prepare_metadata(batch)
            if decode:
                buffer.copy_from(batch)
                backend.prepare_for_replay(batch)
            return incoming, outgoing
        def step():
            prepare()
            if decode:
                graph.replay()
            tokens = sampler.sample(logits, sampler.prepare(batch)).to(torch.int32)
            if native:
                stager.release(batch)
            return tokens
        if args.outputs_only:
            for _ in range(20):
                step()
            torch.cuda.synchronize()
            timing = None
        else:
            timing = measure(step, args.samples, args.iterations)
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        tokens = step()
        torch.cuda.synchronize()
        transient = torch.cuda.max_memory_allocated() - allocated
        md_actual = batch.attn_metadata
        tensors = {"input_ids": batch.input_ids, "out_loc": batch.out_loc, "positions": batch.positions,
                   "indices": md_actual.indices[:sum(r.device_len for r in reqs)],
                   "indptr": md_actual.indptr, "cu_q": md_actual.cu_seqlens_q_gpu,
                   "swa_indices": md_actual.swa_indices[:sum(r.device_len for r in reqs)], "tokens": tokens}
        if decode:
            tensors["attention"] = out
        else:
            tensors["prefix"] = md_actual.prefix_lens
            tensors["q_to_req"] = md_actual.q_to_req
        inputs = {"page_table": ctx.page_table, "token_pool": token_pool, "mapping": mapping, "logits": logits}
        if decode:
            inputs.update(q=q, k=k, v=v)
        row = {"batch": bs, "context": width, "decode": decode, "timing": timing,
               "peak_extra_allocated_bytes": transient,
               "input_staging_payload_bytes": sum(t.numel() * t.element_size() for slot in stager.slots
                   for t in slot.values() if isinstance(t, torch.Tensor) and t.is_cuda) if native else 0,
               "greedy_scratch_payload_bytes": sum(t.numel() * t.element_size() for t in sampler._greedy_buffers) if native else 0,
               "input_hashes": {key: digest(tensor) for key, tensor in inputs.items()},
               "output_hashes": {key: digest(tensor) for key, tensor in tensors.items()}}
        if args.save_tensors:
            path = args.output.parent / f"{args.output.stem}_bs{bs}_ctx{width}_decode{int(decode)}.pt"
            if path.exists():
                raise FileExistsError(path)
            torch.save({"inputs": {key: tensor.cpu() for key, tensor in inputs.items()},
                        "outputs": {key: tensor.cpu() for key, tensor in tensors.items()}}, path)
        result["cases"].append(row)
        if timing is None:
            print(f"bs={bs} context={width} decode={decode}: extra allocated={transient} bytes", flush=True)
        else:
            print(f"bs={bs} context={width} decode={decode}: {timing['median_wall_ms']:.4f}ms", flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.outputs_only:
        return
    for bs in [1, 4, 16]:
        logits = torch.randn(bs, 201088, device=device)
        sampler = Sampler(device, 201088)
        if native:
            sampler.warmup(bs)
        fn = lambda: sampler.sample(logits, BatchSamplingArgs(None)).to(torch.int32)
        result["components"][f"greedy_bs{bs}"] = measure(fn, args.samples, args.iterations)
    pages = torch.zeros(5, 2048, dtype=torch.int32, device=device)
    allocated = torch.arange(8, dtype=torch.int32, device=device)
    result["components"]["write_pages_decode4"] = measure(
        lambda: _write_page_table(pages, allocated[:4], [(i, 100, 101) for i in range(4)], 1), args.samples, args.iterations)
    result["components"]["write_pages_prefill4"] = measure(
        lambda: _write_page_table(pages, allocated, [(i, 100, 102) for i in range(4)], 1), args.samples, args.iterations)
    free = torch.arange(1, 65, dtype=torch.int32, device=device)
    pool = SimpleNamespace(_swa_paged=True, _swa_free=free, full_to_swa_index_mapping=torch.zeros(128, dtype=torch.int64, device=device))
    def swa_alloc():
        pool._swa_free = free
        HybridSWAKVCache.alloc_swa(pool, allocated)
    result["components"]["alloc_swa8"] = measure(swa_alloc, args.samples, args.iterations)
    result["component_output_hashes"] = {
        "page_table": digest(pages), "swa_mapping": digest(pool.full_to_swa_index_mapping),
        "remaining_swa_slots": digest(pool._swa_free),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
