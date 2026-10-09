"""Synthetic DeepSeek bucket prefill: eager reference, same-shape eager and graph replay."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch


def setup(context, cap):
    from freetoken.attention.dsv4_sparse import DSV4SparseAttnBackend
    from freetoken.core import Context, set_global_ctx
    from freetoken.distributed import set_tp_info
    from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    from freetoken.layers.quantization import NameMap
    from freetoken.models.config import ModelConfig
    from freetoken.models.deepseek_v4.args import DeepseekV4Args
    from freetoken.models.deepseek_v4.config import build_quant_config
    from freetoken.models.deepseek_v4.model import DeepseekV4ForCausalLM
    from freetoken.moe.native_schedule import NativeMoeSchedule
    from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache
    from freetoken.utils.torch_utils import torch_dtype

    set_tp_info(0, 1)
    torch.manual_seed(41)
    device = torch.device('cuda')
    args = DeepseekV4Args(dim=128, moe_inter_dim=128, n_layers=3, n_heads=2,
        n_routed_experts=8, n_activated_experts=2, n_hash_layers=1, vocab_size=128,
        q_lora_rank=128, head_dim=128, o_groups=1, o_lora_rank=128,
        index_n_heads=4, index_head_dim=128, index_topk=16, compress_ratios=(0, 4, 128),
        max_seq_len=context, max_batch_size=2)
    hf = SimpleNamespace(quantization_config={
        'quant_method':'compressed-tensors', 'config_groups':{'experts':{
            'format':'nvfp4-pack-quantized', 'targets':['re:.*ffn.*(gate|up|down)_proj$'],
            'weights':{'num_bits':4, 'type':'float', 'group_size':16, 'strategy':'tensor_group'}}}})
    quant = build_quant_config(hf, name_map=NameMap(roots=(('model.layers','layers'),),
        packed=(('experts',('experts.0.w1','experts.0.w2','experts.0.w3')),)))
    model_config = ModelConfig(num_layers=3, num_qo_heads=2, num_kv_heads=1, head_dim=128,
        hidden_size=128, vocab_size=128, intermediate_size=128, hidden_act='silu',
        rms_norm_eps=1e-6, tie_word_embeddings=False, rotary_config=None,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=128,
        norm_topk_prob=True, model_type='deepseek_v4', architectures=['DeepseekV4ForCausalLM'],
        dsv4_args=args, quant=quant, moe_strategy='offload', decode_target='gpu')
    ctx = Context(128)
    set_global_ctx(ctx)
    pool = ctx.kv_cache = DSV4PagedKVCache(dsv4_pool_sizes(context // 128 + 1, args, 1.0), args, device)
    pool.full_loc_map = torch.stack((torch.arange(context, device=device) % 128,
                                    torch.arange(context, device=device) + 128)).int()
    pool.full_to_window[:-1].copy_(torch.arange(pool.full_to_window.numel() - 1, device=device))
    ctx.page_table = pool.full_loc_map
    ctx.attn_backend = DSV4SparseAttnBackend(model_config)
    with torch.device(device), torch_dtype(torch.bfloat16):
        model = DeepseekV4ForCausalLM(model_config)
    state = {}
    for name, tensor in model.state_dict().items():
        if name.endswith('tid2eid'):
            value = torch.randint(0, 8, tensor.shape, device=device, dtype=tensor.dtype)
        elif 'norm.weight' in name:
            value = torch.ones_like(tensor)
        else:
            value = torch.randn(tensor.shape, device=device).mul(0.02).to(tensor.dtype)
        state[name] = value
    model.load_state_dict(state)
    method = model.model.layers.op_list[0].ffn.experts.quant_method
    banks = {}
    for name, spec in method.layout().items():
        layers = []
        for _ in range(3):
            tensor = torch.empty((8, *spec.shape), dtype=spec.dtype, pin_memory=True)
            if spec.dtype is torch.uint8:
                tensor.random_(0, 256)
            else:
                tensor.copy_(torch.full(tensor.shape, 0.05 if 'global' in name else 1.0).to(spec.dtype))
            layers.append(tensor)
        banks[name] = layers
    cache = ctx.moe_offload_cache = OffloadMoeCache(3, 8, 24, device, native_schedule=True,
                                                  quant_format='nvfp4', layout=method.layout())
    cache.set_bank_sources(banks)
    attach_offload_moe_cache(model, cache)
    cache.scheduler = NativeMoeSchedule(cache, gpu_fraction=1.0, prefill_fraction=1.0)
    model._ensure_bound()
    runner = SimpleNamespace(device=device, stream=torch.cuda.Stream(), moe_offload_cache=cache,
                             dummy_req=SimpleNamespace(table_idx=0))
    config = SimpleNamespace(model_config=model_config, max_seq_len=context,
        moe_prefill_graph_max_tokens=cap, dsv4_prefill_buckets=tuple(128 * 2**i for i in range((cap // 128).bit_length())),
        dsv4_prefill_context_buckets=(context,))
    return ctx, model, runner, config


def batch_for(ctx, ids, prefix):
    from freetoken.core import Batch
    req = SimpleNamespace(extend_len=ids.numel(), cached_len=prefix, table_idx=1)
    batch = Batch([req], 'prefill')
    batch.padded_reqs = batch.reqs
    batch.input_ids, batch.positions = ids, torch.arange(prefix, prefix + ids.numel(), device=ids.device)
    batch.out_loc = ctx.kv_cache.full_loc_map[1, prefix:prefix + ids.numel()]
    ctx.attn_backend.prepare_metadata(batch)
    return batch


def pool_state(pool):
    result = {}
    for category in ('window_pool','cmp_pool','idx_pool','state_ring','indexer_state_ring'):
        for layer, item in enumerate(getattr(pool, category)):
            if item is not None:
                result[f'{category}.{layer}'] = item if isinstance(item, torch.Tensor) else item.buffer
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, nargs='+', default=[1, 127, 128, 129, 256])
    parser.add_argument('--prefix', type=int, nargs='+', default=[0, 128])
    parser.add_argument('--cap', type=int, default=256)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 1 or any(n < 1 or n > args.cap for n in args.tokens):
        parser.error('tokens must be positive and within cap; repeats must be positive')
    if any(prefix < 0 or prefix % 128 or prefix + max(args.tokens) > args.context for prefix in args.prefix):
        parser.error('prefix must be page aligned and context must cover prefix plus query')
    from freetoken.engine.dsv4_prefill import Dsv4PrefillGraphs

    torch.set_num_threads(1)
    ctx, model, runner, config = setup(args.context, args.cap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graphs = Dsv4PrefillGraphs(runner, model, config)
    runner.stream.wait_stream(torch.cuda.current_stream())
    start = time.perf_counter()
    with torch.cuda.stream(runner.stream):
        graphs.capture_startup()
    runner.stream.synchronize()
    tensors = pool_state(ctx.kv_cache)
    report = dict(scope='synthetic three-layer BF16 dense / NVFP4 expert DeepSeek; no full checkpoint',
        gpu=torch.cuda.get_device_name(), shapes=graphs.shapes, capture_seconds=time.perf_counter()-start, cases=[])
    for prefix in args.prefix:
        for n in args.tokens:
            for tensor in tensors.values():
                tensor.zero_()
            prefix_ids = torch.randint(0, 128, (prefix,), dtype=torch.int32, device=runner.device)
            runner.stream.wait_stream(torch.cuda.current_stream())
            if prefix:
                with torch.cuda.stream(runner.stream), ctx.forward_batch(batch_for(ctx, prefix_ids, 0)):
                    model.forward()
            runner.stream.synchronize()
            initial = {name: tensor.clone() for name, tensor in tensors.items()}
            ids = torch.randint(0, 128, (n,), dtype=torch.int32, device=runner.device)
            batch = batch_for(ctx, ids, prefix)
            runner.stream.wait_stream(torch.cuda.current_stream())
            outputs, states, samples = {}, {}, {}
            for mode in ('eager_reference', 'eager_bucket', 'graph'):
                rows = []
                for repeat in range(args.repeats + 1):
                    with torch.cuda.stream(runner.stream):
                        for name, tensor in tensors.items():
                            tensor.copy_(initial[name])
                        ctx.moe_offload_cache.reset()
                        _, captured, _ = graphs.prepare(batch)
                    runner.stream.synchronize()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(runner.stream), ctx.forward_batch(batch if mode == 'eager_reference' else captured):
                        begin.record()
                        start = time.perf_counter()
                        logits = graphs.replay(batch) if mode == 'graph' else model.forward()
                        enqueue_ms = (time.perf_counter() - start) * 1000
                        end.record()
                    end.synchronize()
                    if repeat:
                        rows.append(dict(cuda_ms=begin.elapsed_time(end), host_enqueue_ms=enqueue_ms))
                outputs[mode] = logits.clone().cpu()
                states[mode] = {name: tensor.clone().cpu() for name, tensor in tensors.items()}
                samples[mode] = rows
            same_logits = torch.equal(outputs['eager_bucket'].view(torch.uint8), outputs['graph'].view(torch.uint8))
            same_state = all(torch.equal(states['eager_bucket'][name].view(torch.uint8), states['graph'][name].view(torch.uint8))
                             for name in states['graph'])
            if not same_logits or not same_state:
                raise AssertionError('Same-shape eager/graph output or persistent state differs')
            difference = outputs['graph'].float() - outputs['eager_reference'].float()
            proof = args.output.with_name(f'{args.output.stem}_p{prefix}_n{n}.pt')
            torch.save(dict(ids=ids.cpu(), prefix_ids=prefix_ids.cpu(), outputs=outputs, states=states,
                            weights={name: tensor.cpu() for name,tensor in model.state_dict().items()},
                            expert_banks={name:[tensor.cpu() for tensor in layers]
                                          for name,layers in ctx.moe_offload_cache.bank_sources.items()}), proof)
            report['cases'].append(dict(prefix=prefix, new_tokens=n, samples=samples,
                medians={mode:{key:statistics.median(row[key] for row in rows) for key in rows[0]} for mode,rows in samples.items()},
                same_shape_logits_equal=same_logits, same_shape_state_equal=same_state,
                cross_shape_max_abs_error=difference.abs().max().item(),
                proof=str(proof), proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest()))
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(prefix, n, report['cases'][-1]['medians'], flush=True)


if __name__ == '__main__':
    main()
