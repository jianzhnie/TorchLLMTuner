"""HCA sliding/compressed Ulysses attention vs explicit full attention.

Run with torchrun --nproc_per_node=2. CPU/Gloo FP32; no full model claims.
"""

from tests.caps import require_env

require_env("spmd_types")

import torch
import torch.distributed as dist

from llmtuner.parallel.context_parallel.compressor_attention import (
    compressor_ulysses_attention,
)


def run_case(length, rate, window):
    size, rank = dist.get_world_size(), dist.get_rank()
    full_length, heads, width = length * size, 4, 3
    torch.manual_seed(83)
    query = torch.randn(2, heads, full_length, width, requires_grad=True)
    sliding = torch.randn(2, 1, full_length, width, requires_grad=True)
    compressed = torch.randn(2, 1, full_length // rate, width, requires_grad=True)
    sinks = torch.randn(heads, requires_grad=True)
    shard = slice(rank * length, (rank + 1) * length)
    first, last = rank * length // rate, (rank + 1) * length // rate
    q = query.detach()[:, :, shard].clone().requires_grad_()
    kv = sliding.detach()[:, :, shard].clone().requires_grad_()
    c = compressed.detach()[:, :, first:last].clone().requires_grad_()
    s = sinks.detach().clone().requires_grad_()
    actual = compressor_ulysses_attention(
        q,
        kv,
        c,
        sinks=s,
        compress_rate=rate,
        sliding_window=window,
        scaling=0.5,
        group=dist.group.WORLD,
    )
    # Independent scalar index rules for the sliding and compressed visibility.
    allowed = torch.tensor(
        [
            [j <= i and i - j < window for j in range(full_length)]
            + [(j + 1) * rate <= i + 1 for j in range(full_length // rate)]
            for i in range(full_length)
        ]
    )
    all_kv = torch.cat((sliding, compressed), dim=2).expand(-1, heads, -1, -1)
    scores = (query @ all_kv.transpose(-1, -2)) * 0.5
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    sink_scores = sinks[None, :, None, None].expand(2, -1, full_length, 1)
    probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(-1)[..., :-1]
    expected = (probabilities @ all_kv).transpose(1, 2)
    torch.testing.assert_close(actual, expected[:, shard], atol=1e-6, rtol=1e-6)
    gradient = torch.randn_like(expected)
    actual.backward(gradient[:, shard])
    expected.backward(gradient)
    for actual_gradient, expected_gradient in (
        (q.grad, query.grad[:, :, shard]),
        (kv.grad, sliding.grad[:, :, shard]),
        (c.grad, compressed.grad[:, :, first:last]),
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=2e-6, rtol=2e-6
        )
    dist.all_reduce(s.grad)
    torch.testing.assert_close(s.grad, sinks.grad, atol=2e-6, rtol=2e-6)
    print(
        f"rank {rank}: HCA attention length={length}, "
        f"rate={rate}, window={window} passed",
        flush=True,
    )


def run_module_case(*, csa=False):
    from copy import deepcopy

    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Attention,
        DeepseekV4RotaryEmbedding,
    )

    from llmtuner.parallel.context_parallel.compressor_attention import (
        compressor_attention_forward,
    )

    config = DeepseekV4Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_groups=2,
        o_lora_rank=4,
        sliding_window=3,
        layer_types=[
            "compressed_sparse_attention" if csa else "heavily_compressed_attention"
        ],
        index_n_heads=2,
        index_head_dim=8,
        index_topk=1,
        mlp_layer_types=["moe"],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 4,
        },
    )
    config._attn_implementation = "eager"
    torch.manual_seed(91)
    model = DeepseekV4Attention(config, 0)
    with torch.no_grad():
        model.sinks.normal_()
        model.compressor.position_bias.normal_()
        if csa:
            model.compressor.indexer.position_bias.normal_()
    reference = deepcopy(model)
    rotary = DeepseekV4RotaryEmbedding(config)
    rank, size = dist.get_rank(), dist.get_world_size()
    length = 5 if csa else 3
    full = torch.randn(2, length * size, 16, requires_grad=True)
    positions = torch.arange(length * size)[None].expand(2, -1)
    cos, sin = rotary(full, position_ids=positions, layer_type="compress")
    indices = torch.arange(length * size)
    allowed = (indices[None] <= indices[:, None]) & (
        indices[None] > indices[:, None] - config.sliding_window
    )
    mask = torch.zeros(2, 1, length * size, length * size).masked_fill(
        ~allowed, float("-inf")
    )
    expected, _ = reference(full, {"compress": (cos, sin)}, positions, mask)
    shard = slice(rank * length, (rank + 1) * length)
    local = full.detach()[:, shard].clone().requires_grad_()
    actual, _ = compressor_attention_forward(dist.group.WORLD)(
        model, local, {"compress": (cos[:, shard], sin[:, shard])}, positions[:, shard]
    )
    torch.testing.assert_close(actual, expected[:, shard], atol=2e-6, rtol=2e-6)
    gradient = torch.randn_like(expected)
    actual.backward(gradient[:, shard])
    expected.backward(gradient)
    torch.testing.assert_close(local.grad, full.grad[:, shard], atol=3e-6, rtol=3e-6)
    for (name, p), (_, r) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if p.grad is None or r.grad is None:
            assert p.grad is None and r.grad is None, name
            continue
        dist.all_reduce(p.grad)
        torch.testing.assert_close(p.grad, r.grad, atol=4e-6, rtol=4e-6, msg=name)
    label = "CSA" if csa else "HCA"
    print(f"rank {rank}: full {label} module passed", flush=True)
    if csa:
        print(
            f"rank {rank}: real CSA module selection/output/gradients passed",
            flush=True,
        )
        return

    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model

    config.vocab_size = 32
    config.moe_intermediate_size = 16
    config.n_routed_experts = 4
    config.num_experts_per_tok = 2
    config.n_shared_experts = 1
    config.hc_mult = 2
    decoder = DeepseekV4Model(config)
    complete = deepcopy(decoder)
    from types import SimpleNamespace

    from torch.distributed.device_mesh import init_device_mesh

    from llmtuner.config.parallel_config import ContextParallelConfig, ParallelConfig
    from llmtuner.parallel.context_parallel.context_parallel import apply_cp

    cfg = ParallelConfig(
        context_parallel_ulysses_size=size,
        cp_config=ContextParallelConfig(context_parallel_strategy="ulysses"),
    )
    mesh = init_device_mesh("cpu", (size,), mesh_dim_names=("cp",))
    installed = {}
    wrapper = SimpleNamespace(
        model=SimpleNamespace(config=config, model=decoder),
        set_cp_mesh=lambda mesh, **kwargs: installed.update(kwargs),
    )
    before = dict(decoder.named_parameters())
    assert apply_cp(wrapper, mesh=mesh, cfg=cfg) is wrapper
    assert installed["strategy"] == "compressor_ulysses"
    assert all(
        dict(decoder.named_parameters())[key] is value for key, value in before.items()
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    expected_decoder = complete(input_ids=tokens, use_cache=False).last_hidden_state
    actual_decoder = decoder(
        input_ids=tokens[:, shard], position_ids=positions[:1, shard]
    ).last_hidden_state
    torch.testing.assert_close(
        actual_decoder, expected_decoder[:, shard], atol=3e-6, rtol=3e-6
    )
    upstream = torch.randn_like(expected_decoder)
    actual_decoder.backward(upstream[:, shard])
    expected_decoder.backward(upstream)
    for (name, parameter), (_, wanted) in zip(
        decoder.named_parameters(), complete.named_parameters(), strict=True
    ):
        if parameter.grad is None or wanted.grad is None:
            assert parameter.grad is None and wanted.grad is None, name
            continue
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(
            parameter.grad, wanted.grad, atol=6e-6, rtol=6e-6, msg=name
        )
    print(f"rank {rank}: stateless HCA decoder passed", flush=True)

    from llmtuner.models.hf.model import HFTransformerModel

    config.num_hidden_layers = 3
    config.layer_types = [
        "sliding_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]
    config.mlp_layer_types = ["moe"] * 3
    wrapper_model = HFTransformerModel(deepcopy(config))
    full_wrapper = deepcopy(wrapper_model)
    apply_cp(wrapper_model, mesh=mesh, cfg=cfg)
    expected_logits = full_wrapper(tokens[0], positions=positions[0])
    actual_logits = wrapper_model(tokens[0, shard], positions=positions[0, shard])
    torch.testing.assert_close(
        actual_logits, expected_logits[shard], atol=3e-6, rtol=3e-6
    )
    upstream = torch.randn_like(expected_logits)
    actual_logits.backward(upstream[shard])
    expected_logits.backward(upstream)
    for (name, parameter), (_, wanted) in zip(
        wrapper_model.named_parameters(), full_wrapper.named_parameters(), strict=True
    ):
        if parameter.grad is None or wanted.grad is None:
            assert parameter.grad is None and wanted.grad is None, name
            continue
        dist.all_reduce(parameter.grad)
        torch.testing.assert_close(
            parameter.grad, wanted.grad, atol=6e-6, rtol=6e-6, msg=name
        )
    optimizer = torch.optim.SGD(wrapper_model.parameters(), lr=0.01)
    full_optimizer = torch.optim.SGD(full_wrapper.parameters(), lr=0.01)
    optimizer.step()
    full_optimizer.step()
    for (name, parameter), (_, wanted) in zip(
        wrapper_model.named_parameters(), full_wrapper.named_parameters(), strict=True
    ):
        torch.testing.assert_close(parameter, wanted, atol=2e-6, rtol=2e-6, msg=name)
    print(f"rank {rank}: full HF wrapper logits/gradients/update passed", flush=True)


def main():
    dist.init_process_group("gloo")
    try:
        for length, rate, window in ((3, 4, 2), (4, 4, 4), (5, 3, 20), (2, 7, 1)):
            run_case(length, rate, window)
        run_module_case()
        run_module_case(csa=True)
        for mismatch in ("local_kv", "peer_dtype"):
            rank = dist.get_rank()
            dtype = (
                torch.float64 if mismatch == "peer_dtype" and rank else torch.float32
            )
            q = torch.ones(1, 4, 3, 2, dtype=dtype)
            kv = torch.ones(1, 1, 3, 2, dtype=dtype)
            if mismatch == "local_kv" and rank == 0:
                kv = kv.double()
            count = (rank + 1) * 3 // 4 - rank * 3 // 4
            c = torch.ones(1, 1, count, 2, dtype=dtype)
            try:
                compressor_ulysses_attention(
                    q,
                    kv,
                    c,
                    sinks=torch.zeros(4, dtype=dtype),
                    compress_rate=4,
                    sliding_window=3,
                    scaling=0.5,
                    group=dist.group.WORLD,
                )
            except (ValueError, RuntimeError) as error:
                assert any(
                    text in str(error)
                    for text in ("dtype", "another rank", "matching CP")
                )
            else:
                raise AssertionError("Incompatible HCA dtype must fail on every rank")
            dist.barrier()
            print(f"rank {rank}: {mismatch} coordinated", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
