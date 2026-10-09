"""A/B the production DeepSeek-V4 shared expert with separate and merged gate/up."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--format", choices=["bf16", "fp8_float", "fp8_e8m0"], default="fp8_float")
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats <= 0 or any(tokens <= 0 for tokens in args.tokens):
        parser.error("tokens and repeats must be positive")

    from freetoken.distributed import set_tp_info
    from freetoken.layers.quantization import QuantConfig
    from freetoken.models.deepseek_v4.moe import Expert
    from freetoken.utils.torch_utils import torch_dtype

    set_tp_info(0, 1)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quant = None
    if args.format != "bf16":
        raw = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        if args.format == "fp8_e8m0":
            raw["scale_fmt"] = "ue8m0"
        quant = QuantConfig.from_hf(SimpleNamespace(quantization_config=raw))
    with torch.device("cuda"), torch_dtype(torch.bfloat16):
        separate = Expert(args.hidden, args.intermediate, 10, quant_config=quant)
        merged = Expert(args.hidden, args.intermediate, 10, quant_config=quant, fuse_gate_up=True)
    state = {}
    for name, tensor in separate.state_dict().items():
        if name.endswith("weight_scale_inv"):
            value = (torch.full(tensor.shape, 120, dtype=torch.uint8, device="cuda").view(tensor.dtype)
                     if args.format == "fp8_e8m0" else torch.full_like(tensor, 0.02))
        else:
            value = torch.randn(tensor.shape, device="cuda", dtype=torch.float32).mul(0.1).to(tensor.dtype)
        state[name] = value
    separate.load_state_dict(dict(state))
    merged.load_state_dict(dict(state))
    report = {
        "scope": "synthetic shared expert only; no model checkpoint or end-to-end inference",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "seed": 42,
        "hidden": args.hidden, "intermediate": args.intermediate, "format": args.format,
        "limit": 10, "repeats": args.repeats, "cases": [],
    }
    for tokens in args.tokens:
        x = torch.randn(tokens, args.hidden, device="cuda", dtype=torch.bfloat16).mul(0.1)
        outputs, samples = {}, {}
        for name, expert in (("separate", separate), ("merged", merged)):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    expert.forward(x)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = expert.forward(x)
            graph.replay()
            torch.cuda.synchronize()
            rows = []
            for _ in range(args.repeats):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                start = time.perf_counter()
                graph.replay()
                enqueue_us = (time.perf_counter() - start) * 1e6
                end.record()
                end.synchronize()
                rows.append({"cuda_ms": begin.elapsed_time(end), "host_enqueue_us": enqueue_us})
            outputs[name] = output.clone().cpu()
            samples[name] = rows
        reference, modified = outputs["separate"].float(), outputs["merged"].float()
        difference = modified - reference
        assert torch.isfinite(reference).all() and torch.isfinite(modified).all()
        proof = args.output.with_name(args.output.stem + f"_{tokens}.pt")
        torch.save({"x": x.cpu(), "weights": {key: value.cpu() for key, value in state.items()},
                    "outputs": outputs}, proof)
        report["cases"].append({
            "tokens": tokens, "samples": samples,
            "medians": {name: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
                        for name, rows in samples.items()},
            "byte_equal": torch.equal(outputs["separate"].view(torch.uint8), outputs["merged"].view(torch.uint8)),
            "max_absolute_error": difference.abs().max().item(),
            "relative_l2_error": (difference.norm() / reference.norm().clamp_min(1e-12)).item(),
            "proof_file": str(proof), "proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        })
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(tokens, report["cases"][-1]["medians"], flush=True)


if __name__ == "__main__":
    main()
