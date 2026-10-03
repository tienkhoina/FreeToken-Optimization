"""Time host KV preparation followed by a warmed attention CUDA graph."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from freetoken.attention.triton import TritonAttentionBackend
from freetoken.core import Context, set_global_ctx
from freetoken.kernel.triton.attention import decode_paged_attention
from freetoken.scheduler.cache import CacheManager


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--outputs-only", action="store_true", help="save correctness/memory artifacts without repeating latency samples")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    device = torch.device("cuda")
    results = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "method": "Kernels warmed/compiled before timing. Host prepares metadata each iteration, then warmed attention graph replays. Wall time includes Python, allocation and GPU work; CUDA span includes idle gaps from host submission. No profiler.",
               "samples": args.samples, "iterations_per_sample": args.iterations, "cases": []}
    for bs, length in [(1, 128), (1, 1024), (1, 8192), (4, 1024), (16, 1024)]:
        for swa in [False, True]:
            torch.manual_seed(7301 + bs + length)
            slots = (bs + 1) * length
            ctx.page_table = torch.randperm(slots, device=device).to(torch.int32).view(bs + 1, length)
            mapping = torch.randperm(slots, device=device).to(torch.int64)
            pool = SimpleNamespace(device=device, swa_paged=swa, full_to_swa_index_mapping=mapping,
                                   translate_loc_from_full_to_swa=lambda ids: mapping[ids.long()].to(torch.int32))
            ctx.kv_cache = pool
            backend = TritonAttentionBackend(SimpleNamespace(num_qo_heads=4, head_dim=64))
            backend.init_capture_graph(length, [bs])
            rows = list(range(bs - 1, -1, -1))
            lengths = [max(1, length - 11 * i) for i in range(bs)]
            reqs = [SimpleNamespace(table_idx=row, device_len=n, cached_len=n - 1, extend_len=1)
                    for row, n in zip(rows, lengths)]
            batch = SimpleNamespace(is_decode=True, padded_reqs=reqs, padded_size=bs,
                                    positions=torch.tensor([n - 1 for n in lengths], dtype=torch.int32, device=device))
            captured = SimpleNamespace(size=bs)
            backend.prepare_for_capture(captured)
            md = captured.attn_metadata
            q = torch.randn(bs, 4, 64, dtype=torch.bfloat16, device=device)
            k = torch.randn(slots, 2, 64, dtype=torch.bfloat16, device=device)
            v = torch.randn_like(k)
            def attention():
                return decode_paged_attention(q=q, k_cache=k, v_cache=v, indptr=md.indptr,
                        indices=md.swa_indices if swa else md.indices, q_positions=md.q_positions,
                        attn_logits=md.attn_logits, attn_lse=md.attn_lse,
                        num_kv_splits=md.num_kv_splits, max_kv_splits=backend.max_kv_splits,
                        sm_scale=0.125)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                attention()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = attention()
            torch.cuda.current_stream().wait_stream(stream)
            def stage():
                backend.prepare_metadata(batch)
                backend.prepare_for_replay(batch)
            def step():
                stage()
                graph.replay()
            for _ in range(30):
                step()
            torch.cuda.synchronize()
            expected = torch.cat([ctx.page_table[row, :n] for row, n in zip(rows, lengths)])
            actual = batch.attn_metadata
            torch.testing.assert_close(actual.indices[:sum(lengths)], expected, rtol=0, atol=0)
            if swa:
                torch.testing.assert_close(actual.swa_indices[:sum(lengths)], mapping[expected.long()].to(torch.int32), rtol=0, atol=0)
            proof = {"attention_sha256": digest(out), "indices_sha256": digest(actual.indices[:sum(lengths)]),
                     "indptr_sha256": digest(actual.indptr), "positions_sha256": digest(actual.q_positions)}
            inputs = {"page_table": ctx.page_table, "mapping": mapping, "positions": batch.positions, "q": q, "k": k, "v": v}
            proof["input_sha256"] = {name: digest(tensor) for name, tensor in inputs.items()}
            if args.save_inputs:
                path = args.output.parent / f"inputs_bs{bs}_len{length}_swa{int(swa)}.pt"
                if path.exists():
                    raise FileExistsError(path)
                torch.save({name: tensor.cpu() for name, tensor in inputs.items()} | {"rows": rows, "lengths": lengths}, path)
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            stage()
            torch.cuda.synchronize()
            peak_extra = torch.cuda.max_memory_allocated() - allocated
            if args.save_outputs:
                path = args.output.parent / f"{args.output.stem}_bs{bs}_len{length}_swa{int(swa)}.pt"
                if path.exists():
                    raise FileExistsError(path)
                torch.save({"attention": out.cpu(), "indices": actual.indices[:sum(lengths)].cpu(),
                            "indptr": actual.indptr.cpu(), "positions": actual.q_positions.cpu(),
                            "swa_indices": actual.swa_indices[:sum(lengths)].cpu() if swa else None}, path)
            if args.outputs_only:
                results["cases"].append({"batch": bs, "context": length, "swa": swa,
                                         "proof": proof, "metadata_peak_extra_bytes": peak_extra})
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(f"bs={bs} context={length} swa={swa}: extra allocated={peak_extra} bytes", flush=True)
                continue
            timing = {}
            for name, fn in [("metadata", stage), ("metadata_attention_graph", step)]:
                samples = []
                for _ in range(args.samples):
                    torch.cuda.synchronize()
                    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    begin.record()
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        fn()
                    enqueue_end = time.perf_counter()
                    end.record()
                    end.synchronize()
                    samples.append({"wall_ms": (time.perf_counter() - start) * 1000 / args.iterations,
                                    "host_enqueue_ms": (enqueue_end - start) * 1000 / args.iterations,
                                    "cuda_span_ms": begin.elapsed_time(end) / args.iterations})
                timing[name] = {"median_wall_ms": statistics.median(row["wall_ms"] for row in samples),
                                "median_host_enqueue_ms": statistics.median(row["host_enqueue_ms"] for row in samples),
                                "samples": samples}
            row = {"batch": bs, "context": length, "swa": swa, "native_metadata": actual.decode_requests is not None if hasattr(actual, "decode_requests") else False,
                   "proof": proof, "timing": timing, "metadata_peak_extra_bytes": peak_extra}
            results["cases"].append(row)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(f"bs={bs} context={length} swa={swa}: metadata={timing['metadata']['median_wall_ms']:.4f}ms full={timing['metadata_attention_graph']['median_wall_ms']:.4f}ms", flush=True)
    if args.outputs_only:
        return
    ctx.kv_cache = SimpleNamespace(device=device)
    cm = CacheManager(8192, 1, ctx.page_table, "naive")
    for _ in range(30):
        with cm.lazy_free_region():
            pass
    samples = []
    for _ in range(args.samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            with cm.lazy_free_region():
                pass
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / 200)
    results["empty_lazy_free"] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
