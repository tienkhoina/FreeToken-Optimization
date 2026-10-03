"""Same-shape prefill eager vs warmed graph; preserve logits and every state tensor."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(repo))
    from tests.models.qwen4_exp.test_prefill_graph import _setup, _batch, _state
    from freetoken.engine.qwen_prefill import QwenPrefillGraphs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    results = {"gpu": torch.cuda.get_device_name(), "method": "Same QSA/GDN/PLE/operator shapes and native MoE, eager dispatch versus precompiled exact-length CUDA graph. Compilation/capture and state resets excluded; full checkpoint/server measured separately.", "cases": []}
    for lengths in [(13,), (64,), (73,), (7, 19)]:
        fixture, model, runner, config = _setup(max_tokens=96)
        graphs = QwenPrefillGraphs(runner, model, config)
        batch = _batch(fixture, lengths)
        original = _state(fixture)
        stream = runner.stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graphs.capture(batch)
            for _ in range(3):
                graphs.replay(batch)
        torch.cuda.synchronize()
        samples = {"eager": [], "graph": []}
        outputs = {}
        state_hashes = {}
        inputs = {"token_ids": batch.input_ids.cpu(), "positions": batch.positions.cpu(), "locations": batch.out_loc.cpu()}
        for name in ["eager", "graph"]:
            for _ in range(args.repeats):
                with torch.cuda.stream(stream):
                    for tensor, value in original:
                        tensor.copy_(value)
                stream.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream), fixture.ctx.forward_batch(batch):
                    begin.record()
                    start = time.perf_counter()
                    logits = model.forward().float() if name == "eager" else graphs.replay(batch)
                    enqueue_ms = (time.perf_counter() - start) * 1000
                    end.record()
                end.synchronize()
                samples[name].append({"wall_ms": (time.perf_counter() - start) * 1000,
                                      "host_enqueue_ms": enqueue_ms, "cuda_span_ms": begin.elapsed_time(end)})
            outputs[name] = logits.clone().cpu()
            state_hashes[name] = [hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                                  for t, _ in original]
        assert torch.equal(outputs["eager"].view(torch.uint8), outputs["graph"].view(torch.uint8))
        assert state_hashes["eager"] == state_hashes["graph"]
        tensors_path = args.output.parent / (args.output.stem + "_" + "-".join(map(str, lengths)) + ".pt")
        torch.save({"inputs": inputs, "outputs": outputs}, tensors_path)
        row = {"lengths": lengths, "samples": samples, "logits_and_state_bit_exact": True,
               "state_hashes": state_hashes, "tensors_path": str(tensors_path),
               "medians": {name: {key: statistics.median(sample[key] for sample in values)
                                  for key in values[0]} for name, values in samples.items()}}
        results["cases"].append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(lengths, row["medians"], flush=True)


if __name__ == "__main__":
    main()
