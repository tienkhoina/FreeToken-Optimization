"""Compare original and current MoE stages on an actual local checkpoint layer."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from bench_moe_bitexact import (
    assert_bits, digest, intermediate_proof, load_baseline, prefill_proof, views,
)
from freetoken.moe.fused_mxfp4 import (
    _transpose_mxfp4_for_decode,
    run_mxfp4_prefill_experts_t,
    run_mxfp4_splitk_decode_experts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((args.model / "config.json").read_text())
    h, topk = config["hidden_size"], config["num_experts_per_tok"]
    old_impl, old_fused = load_baseline(args.baseline_root)
    result = {"model": str(args.model), "baseline_root": str(args.baseline_root), "cases": []}
    roles = {"gate_up": "gate_up_proj_blocks", "gate_up_scale": "gate_up_proj_scales",
             "gate_up_bias": "gate_up_proj_bias", "down": "down_proj_blocks",
             "down_scale": "down_proj_scales", "down_bias": "down_proj_bias"}
    torch.manual_seed(77331)
    for layer in (0, 17, 35):
        banks = {}
        prefix = f"model.layers.{layer}.mlp."
        for role, name in roles.items():
            key = prefix + "experts." + name
            with safe_open(args.model / index[key], framework="pt", device="cpu") as file:
                banks[role] = file.get_tensor(key).to("cuda")
        banks["gate_up"], banks["gate_up_scale"] = _transpose_mxfp4_for_decode(banks["gate_up"], banks["gate_up_scale"])
        banks["down"], banks["down_scale"] = _transpose_mxfp4_for_decode(banks["down"], banks["down_scale"])
        with safe_open(args.model / index[prefix + "router.weight"], framework="pt", device="cpu") as file:
            router = file.get_tensor(prefix + "router.weight").to("cuda")
            bias = file.get_tensor(prefix + "router.bias").to("cuda")
        for tokens in (1, 4, 16, 116, 464):
            x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)
            logits = torch.nn.functional.linear(x, router, bias)
            weights, ids = old_impl.gpt_oss_fused_routing(logits, topk)
            mode = "decode" if tokens <= 16 else "prefill"
            proof = (intermediate_proof(old_impl, old_fused, x, weights, ids, banks, 1.702, 7.0) if mode == "decode"
                     else prefill_proof(old_impl, x, weights, ids, banks))
            name = "run_mxfp4_splitk_decode_experts" if mode == "decode" else "run_mxfp4_prefill_experts_t"
            current = run_mxfp4_splitk_decode_experts if mode == "decode" else run_mxfp4_prefill_experts_t
            inputs = (x, weights, ids, *views(banks))
            kwargs = dict(top_k=topk, hidden_act_alpha=1.702, swiglu_limit=7.0)
            proof.append(assert_bits(getattr(old_fused, name)(*inputs, **kwargs), current(*inputs, **kwargs), "output"))
            row = {"layer": layer, "tokens": tokens, "mode": mode, "proof": proof, "input_sha256": digest(x)}
            result["cases"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(f"layer={layer} tokens={tokens}: all {len(proof)} stages byte-identical", flush=True)
        del banks, router, bias
    print(f"All {len(result['cases'])} real-weight cases passed", flush=True)


if __name__ == "__main__":
    main()
