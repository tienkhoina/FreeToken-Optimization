"""Saved long-input Qwen operator test: native layer sweep vs block-major bucket graphs."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import statistics

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[10000, 100000])
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--activation-device", choices=["gpu", "cpu", "auto"], default="auto")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import sys
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from tests.models.qwen4_exp.test_prefill_graph import _setup, _batch, _state
    from freetoken.engine.qwen_layer_prefill import QwenLayerPrefillGraphs
    from freetoken.engine.qwen_prefill import QwenBucketPrefillGraphs
    from freetoken.core import Req, Batch
    from freetoken.attention.linear import build_fla_metadata

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    maximum = max(args.tokens)
    position_cap = 1 << (maximum - 1).bit_length()
    fixture, model, runner, config = _setup(max_tokens=args.block,
                                          num_pages=(maximum + 63) // 64 + 16,
                                          max_position=position_cap)
    config.qwen_prefill_mode = "layer"
    config.qwen_prefill_activation_device = args.activation_device
    runner.dummy_req = Req(torch.zeros(1, dtype=torch.int32), 2, 0, 1, -1, None, None)
    layer = QwenLayerPrefillGraphs(runner, model, config)
    bucket = QwenBucketPrefillGraphs(runner, model, config)
    stream = runner.stream
    stream.wait_stream(torch.cuda.current_stream())
    start = time.perf_counter()
    with torch.cuda.stream(stream):
        layer.capture_startup()
        bucket.capture_startup()
    torch.cuda.synchronize()
    result = {"gpu": torch.cuda.get_device_name(), "model": "small Qwen QSA/GDN/PLE model with BF16 experts",
              "compile_capture_seconds": time.perf_counter() - start,
              "block_cap": args.block, "layer_graphs": len(layer.graphs), "bucket_graphs": len(bucket.graphs),
              "cases": [], "scope": "operator/scheduler scaling; not the full Qwen checkpoint or a quality evaluation"}
    for tokens in args.tokens:
        fixture._free = list(range(fixture.pool._kv_buffer.shape[2] - 1))
        batch = _batch(fixture, (tokens,))
        req = batch.reqs[0]
        state = _state(fixture)
        outputs, states, timings = {}, {}, {}
        for mode in ("block_major", "layer_major"):
            samples = []
            for repetition in range(args.repeats + 1):
                with torch.cuda.stream(stream):
                    for tensor, value in state:
                        tensor.copy_(value)
                    fixture.ctx.moe_offload_cache.reset()
                stream.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                context = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                           if args.trace and mode == "layer_major" and repetition == args.repeats else __import__("contextlib").nullcontext())
                with context as trace, torch.cuda.stream(stream):
                    begin.record()
                    start = time.perf_counter()
                    if mode == "layer_major":
                        with fixture.ctx.forward_batch(batch):
                            logits = layer.replay(batch)
                    else:
                        for offset in range(0, tokens, args.block):
                            length = min(args.block, tokens - offset)
                            block_req = __import__("copy").copy(req)
                            block_req.cached_len = offset
                            block_req.device_len = offset + length
                            block_req.extend_len = length
                            block_req.input_ids = req.input_ids[:offset + length]
                            block = Batch([block_req], "prefill")
                            block.padded_reqs = block.reqs
                            block.input_ids = batch.input_ids[offset:offset + length]
                            block.out_loc = batch.out_loc[offset:offset + length]
                            block.positions = batch.positions[offset:offset + length]
                            fixture.backend.prepare_metadata(block)
                            block.fla_metadata = build_fla_metadata(block, fixture.device)
                            with fixture.ctx.forward_batch(block):
                                logits = bucket.replay(block)
                    enqueue_ms = (time.perf_counter() - start) * 1000
                    end.record()
                    end.synchronize()
                    wall_ms = (time.perf_counter() - start) * 1000
                if args.trace and mode == "layer_major" and repetition == args.repeats:
                    trace.export_chrome_trace(str(args.output.with_name(args.output.stem + f"_{tokens}_trace.json")))
                outputs[mode] = logits.cpu()
                states[mode] = [tensor.cpu().clone() for tensor, _ in state]
                sample = {"host_enqueue_ms": enqueue_ms, "wall_ms": wall_ms,
                                 "cuda_span_ms": begin.elapsed_time(end),
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "reserved_bytes": torch.cuda.memory_reserved()}
                if repetition > 0:
                    samples.append(sample)
            timings[mode] = {key: statistics.median(row[key] for row in samples) for key in samples[0]}
            timings[mode]["samples"] = samples
        x, y = outputs["block_major"], outputs["layer_major"]
        error = (x - y).abs()
        assert torch.isfinite(y).all()
        assert error.max() / x.abs().max().clamp_min(1e-6) < 0.05
        file = args.output.with_name(args.output.stem + f"_{tokens}.pt")
        torch.save({"input_ids": batch.input_ids.cpu(), "outputs": outputs, "final_states": states}, file)
        row = {"tokens": tokens, "timings": timings,
               "normalized_max_logit_error": float(error.max() / x.abs().max().clamp_min(1e-6)),
               "argmax_equal": int(x.argmax()) == int(y.argmax()),
               "logits_bytes_equal": torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
               "activation_plane_bytes": layer.peak_plane_bytes,
               "activation_device": str(layer.planes[0].device),
               "file": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
               "runtime_layer_captures": layer.captures_at_runtime,
               "runtime_bucket_captures": bucket.captures_at_runtime}
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(tokens, timings, "normalized error", row["normalized_max_logit_error"], flush=True)


if __name__ == "__main__":
    main()
