"""Large NVFP4 hit/fetch math, old partial sums vs a single ordered reduction."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch


@torch.inference_mode()
def main():
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin, fused_experts_nvfp4

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--intermediate", type=int, default=640)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(905)
    h, i, e, k = args.hidden, args.intermediate, 16, 10
    assert h % 32 == i % 32 == 0
    banks = [
        torch.randint(0, 256, (e + 1, 2 * i, h // 2), dtype=torch.uint8, device="cuda"),
        (torch.rand(e + 1, 2 * i, h // 16, device="cuda") + 0.25).to(torch.float8_e4m3fn),
        torch.full((e + 1, 2 * i), 0.02, dtype=torch.float16, device="cuda"),
        torch.randint(0, 256, (e + 1, h, i // 2), dtype=torch.uint8, device="cuda"),
        (torch.rand(e + 1, h, i // 16, device="cuda") + 0.25).to(torch.float8_e4m3fn),
        torch.full((e + 1, h), 0.02, dtype=torch.float16, device="cuda"),
    ]
    for bank in banks:
        bank[e].zero_()
    result = {"gpu": torch.cuda.get_device_name(), "seed": 905, "hidden": h, "intermediate": i,
              "experts": e, "top_k": k, "method": "Warmed CUDA graphs; compile/capture excluded. Same active expert math and layout. Weight copies are excluded here; real-model A/B includes them.", "cases": []}
    for tokens in (1, 39, 64):
        x = torch.randn(tokens, h, dtype=torch.bfloat16, device="cuda") * 0.2
        ids = torch.randint(0, e, (tokens, k), dtype=torch.int32, device="cuda")
        weights = torch.softmax(torch.randn(tokens, k, device="cuda"), dim=-1)
        hit_mask = ids < e // 2
        hit_ids, fetch_ids = torch.where(hit_mask, ids, e), torch.where(hit_mask, e, ids)
        hit_w, fetch_w = torch.where(hit_mask, weights, 0), torch.where(hit_mask, 0, weights)

        def experts(route_ids, route_w, routes=False):
            if tokens == 1:
                return fused_experts_decode_nvfp4_marlin(x, *banks, route_w, route_ids,
                                                        inactive_expert=e, route_outputs=routes)
            return fused_experts_nvfp4(x, *banks, route_w, route_ids, e + 1,
                                      inactive_expert=e, route_outputs=routes)

        def old():
            return experts(hit_ids, hit_w) + experts(fetch_ids, fetch_w)

        def merged():
            hit, fetched = experts(hit_ids, hit_w, True), experts(fetch_ids, fetch_w, True)
            output = torch.empty_like(x)
            moe_sum_reduce_triton(hit, output, branch=fetched)
            return output

        samples, outputs = {}, {}
        for name, fn in (("separate_partial_sums", old), ("single_route_reduction", merged),
                         ("unsplit_reference", lambda: experts(ids, weights))):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = fn()
            graph.replay()
            torch.cuda.synchronize()
            rows = []
            for _ in range(args.repeats):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                start = time.perf_counter()
                graph.replay()
                enqueue_ms = (time.perf_counter() - start) * 1000
                end.record()
                end.synchronize()
                rows.append({"wall_ms": (time.perf_counter() - start) * 1000,
                             "host_enqueue_ms": enqueue_ms, "cuda_span_ms": begin.elapsed_time(end)})
            samples[name] = rows
            outputs[name] = output.clone().cpu()
        reference = outputs["unsplit_reference"]
        exact = torch.equal(outputs["single_route_reduction"].view(torch.uint8), reference.view(torch.uint8))
        assert exact
        inputs_path = args.output.with_name(args.output.stem + f"_{tokens}.pt")
        torch.save({"inputs": {"hidden": x.cpu(), "expert_ids": ids.cpu(), "weights": weights.cpu(),
                               "banks": [bank.cpu() for bank in banks]}, "outputs": outputs}, inputs_path)
        row = {"query_tokens": tokens, "single_reduction_matches_unsplit_bytes": exact,
               "separate_partial_matches_unsplit_bytes": torch.equal(outputs["separate_partial_sums"].view(torch.uint8), reference.view(torch.uint8)),
               "separate_partial_max_abs_error": float((outputs["separate_partial_sums"].float() - reference.float()).abs().max()),
               "input_output_file": str(inputs_path),
               "input_output_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(), "samples": samples,
               "medians": {name: {metric: statistics.median(r[metric] for r in values) for metric in values[0]}
                           for name, values in samples.items()}}
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(tokens, row["medians"], flush=True)


if __name__ == "__main__":
    main()
