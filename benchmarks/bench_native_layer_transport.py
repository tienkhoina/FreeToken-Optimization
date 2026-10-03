"""Trace native expert and activation transfer overlap with substantial graph compute."""

import argparse
import json
from pathlib import Path
import time

import torch


@torch.inference_mode()
def main():
    from freetoken.kernel.layer_prefill import layer_prefill_module

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(711)
    h, c, n, layers = args.width, args.block, args.block * args.blocks, args.layers
    host_banks = [torch.randn(h, h, dtype=torch.bfloat16).mul_(0.01).pin_memory() for _ in range(layers)]
    banks = [torch.empty(h, h, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    bases = []
    pointers = []
    for parity in range(2):
        raw = [torch.empty(c, dtype=torch.int32, device="cuda") for _ in range(3)]
        raw.append(torch.empty(3, c, dtype=torch.int32, device="cuda"))
        hidden, output = [torch.empty(c, h, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
        desc = torch.empty(1, 8, dtype=torch.int32, device="cuda")
        bases.append((raw, hidden, output, desc))
        pointers.append([*(t.data_ptr() for t in raw), hidden.data_ptr(), output.data_ptr(), desc.data_ptr(), 0, 0, c])
    stream, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
    graphs = []
    handles = []
    stream.wait_stream(torch.cuda.current_stream())
    for layer in range(-1, layers):
        rows = []
        for raw, hidden, output, desc in bases:
            def compute():
                if layer < 0:
                    output.copy_((raw[0].float() % 17 / 17).view(-1, 1).expand(c, h))
                else:
                    x = hidden
                    for _ in range(4):
                        x = torch.relu(x @ banks[layer % 2])
                    output.copy_(x)
            with torch.cuda.stream(stream):
                compute()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                compute()
            graphs.append(graph)
            rows.append(graph.raw_cuda_graph_exec())
        handles.append(rows)
    handles = torch.tensor(handles, dtype=torch.int64)
    pointer_table = torch.tensor(pointers, dtype=torch.int64)
    copies = torch.tensor([[[banks[i % 2].data_ptr(), host_banks[i].data_ptr(), host_banks[i].numel() * 2]]
                          for i in range(layers)], dtype=torch.int64)
    plan = torch.tensor([[b * c, c, 0] for b in range(args.blocks)], dtype=torch.int64)
    desc = torch.tensor([[0, b * c, c, 0, 1, 0, -1, -1] for b in range(args.blocks)], dtype=torch.int32, device="cuda")
    ids = torch.arange(n, dtype=torch.int32, device="cuda")
    pos, loc = ids.clone(), ids.clone()
    mrope = ids.unsqueeze(0).repeat(3, 1)
    planes = [torch.empty(n, h, dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
    empty_ple = torch.empty(0, 0, dtype=torch.uint8)
    run = layer_prefill_module().run
    result = {"scope": "native transport/graph protocol with synthetic dense math; not model throughput",
              "width": h, "block": c, "blocks": args.blocks, "layers": layers, "samples": []}
    for overlap in (False, True):
        with torch.cuda.stream(stream):
            run(handles, pointer_table, copies, plan, desc, ids, loc, pos, mrope,
                *planes, empty_ple, stream.cuda_stream, copy_stream.cuda_stream, overlap, -1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.cuda.stream(stream):
                run(handles, pointer_table, copies, plan, desc, ids, loc, pos, mrope,
                    *planes, empty_ple, stream.cuda_stream, copy_stream.cuda_stream, overlap, -1)
            torch.cuda.synchronize()
        trace = args.output.with_name(args.output.stem + f"_{int(overlap)}_trace.json")
        profile.export_chrome_trace(str(trace))
        result["samples"].append({"overlap": overlap, "wall_s_with_profiler": time.perf_counter() - start,
                                  "trace": str(trace), "finite_output": bool(torch.isfinite(planes[layers % 2]).all())})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
