"""Capture the first update differences when changing compilation on resume.

Run full/resume exactly as pp_checkpoint_equivalence.py, using the same WORKDIR.
This observer saves local parameter/gradient snapshots around optimizer.step;
all original checkpoint equivalence assertions remain enabled. Traces are
written under trace-<phase>-<backend>/step-<N>-rank-<R>.pt.
Use --isolated-backward for a single-process native HF/eager/AOT comparison.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.distributed.tensor import DTensor

from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.pp_checkpoint_equivalence import main


def parameters(trainer, *, gradients=False):
    result = {}
    for part in trainer.model_parts:
        for name, parameter in part.named_parameters():
            name = canonical_fqn(name)
            if name in result:
                raise AssertionError(f"Duplicate trace parameter {name}")
            value = parameter.grad if gradients else parameter
            if value is None:
                result[name] = None
            else:
                if isinstance(value, DTensor):
                    value = value._local_tensor
                result[name] = value.detach().cpu().clone()
    return result


def isolated_backward(*, norm_override=None):
    """Compare native HF, eager wrapper and AOT without process groups."""
    import copy

    from torch._dynamo.utils import counters

    from llmtuner.config.compile_config import CompileConfig
    from llmtuner.models.hf.factory import build_model_config
    from llmtuner.models.hf.model import HFTransformerModel
    from llmtuner.models.hf.model_setup import _isolate_norm_input
    from llmtuner.parallel.model_compile import apply_compile

    torch.set_num_threads(1)
    torch._dynamo.reset()
    counters.clear()
    torch.manual_seed(42)
    config = build_model_config(
        "qwen3",
        seq_len=32,
        arch_overrides=dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
    )
    eager = HFTransformerModel(config)
    native = copy.deepcopy(eager.model)
    for module in native.modules():
        for key, hook in tuple(module._forward_pre_hooks.items()):
            if hook is _isolate_norm_input:
                del module._forward_pre_hooks[key]
                module._forward_pre_hooks_with_kwargs.pop(key, None)
    if norm_override is not None:
        from types import MethodType

        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        for module in eager.modules():
            if type(module) is Qwen3RMSNorm:
                module.forward = MethodType(norm_override, module)
    compiled_base = copy.deepcopy(eager)
    compiled = apply_compile(
        compiled_base,
        compile_config=CompileConfig(
            backend=os.environ.get("LLMTUNER_TEST_COMPILE_BACKEND", "aot_eager")
        ),
    )
    generator = torch.Generator().manual_seed(1234)
    tokens = torch.randint(128, (32,), generator=generator)
    targets = torch.randint(128, (32,), generator=generator)
    outputs = [
        eager(tokens),
        compiled(tokens),
        native(tokens.unsqueeze(0), use_cache=False).logits.squeeze(0),
    ]
    for value in outputs:
        torch.nn.functional.cross_entropy(value, targets).backward()
    assert counters["stats"]["unique_graphs"] > 0
    torch.testing.assert_close(outputs[1], outputs[2])
    native_parameters = dict(native.named_parameters())
    for name, parameter in compiled_base.model.named_parameters():
        torch.testing.assert_close(parameter.grad, native_parameters[name].grad)
    for label, model, output in (
        ("compiled", compiled_base.model, outputs[1]),
        ("native", native, outputs[2]),
    ):
        torch.testing.assert_close(outputs[0], output)
        others = dict(model.named_parameters())
        differences = []
        for name, parameter in eager.model.named_parameters():
            other = others[name]
            torch.testing.assert_close(parameter.grad, other.grad)
            if not torch.equal(parameter.grad, other.grad):
                differences.append(
                    (name, (parameter.grad - other.grad).abs().max().item())
                )
        print(
            f"{label}: logits_exact={torch.equal(outputs[0], output)} "
            f"gradient_differences={differences}",
            flush=True,
        )


def replay_backward(workdir: str, rank: int):
    """Replay a captured dense Qwen3 batch with full checkpoint parameters."""
    import copy
    from contextlib import nullcontext

    import torch.distributed.checkpoint as dcp

    from llmtuner.config.compile_config import CompileConfig
    from llmtuner.models.hf.factory import build_model_config
    from llmtuner.models.hf.model import HFTransformerModel
    from llmtuner.parallel.model_compile import apply_compile

    root = Path(workdir)
    torch.set_num_threads(1)
    config = build_model_config(
        "qwen3",
        seq_len=32,
        arch_overrides=dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
    )
    eager = HFTransformerModel(config)
    state = eager.state_dict()
    dcp.load(state, checkpoint_id=root / "checkpoint" / "step-2")
    eager.load_state_dict(state)
    view_prenorm = os.environ.get("LLMTUNER_TEST_VIEW_PRENORM") == "1"
    clone_prenorm = os.environ.get("LLMTUNER_TEST_CLONE_PRENORM") == "1"
    if clone_prenorm or view_prenorm:

        def clone_input(module, args):
            value = args[0]
            return (value.view_as(value) if view_prenorm else value.clone(), *args[1:])

        for layer in eager.layers:
            layer.input_layernorm.register_forward_pre_hook(clone_input)
            layer.post_attention_layernorm.register_forward_pre_hook(clone_input)
    base = copy.deepcopy(eager)
    if os.environ.get("LLMTUNER_TEST_REPLAY_Q_BREAK") == "1":
        for layer in base.layers:
            layer.self_attn.q_proj.forward = torch.compiler.disable(
                layer.self_attn.q_proj.forward
            )
    compiled = apply_compile(base, compile_config=CompileConfig(backend="aot_eager"))
    trace = torch.load(
        root / "trace-resume-aot_eager" / f"step-3-rank-{rank}.pt",
        weights_only=True,
    )
    assert len(trace["batches"]) == 1
    batch = trace["batches"][0]
    actual = compiled(batch["inputs"], positions=batch["positions"])
    expected = eager(batch["inputs"], positions=batch["positions"])
    additions = {}
    capture_path = os.environ.get("LLMTUNER_TEST_REPLAY_ADDS")
    if capture_path:
        from torch.utils._python_dispatch import TorchDispatchMode

        class CaptureAdditions(TorchDispatchMode):
            def __init__(self, records):
                super().__init__()
                self.records = records

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func in (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor):
                    if args[0].shape == (1, 128, 64):
                        self.records.append(
                            (str(func), *(arg.detach().clone() for arg in args[:2]))
                        )
                return func(*args, **(kwargs or {}))

    for backend, output in (("aot_eager", actual), ("eager", expected)):
        loss = (
            torch.nn.functional.cross_entropy(
                output, batch["labels"], reduction="sum", ignore_index=-100
            )
            / batch["global_valid_tokens"]
        )
        records = additions.setdefault(backend, [])
        with CaptureAdditions(records) if capture_path else nullcontext():
            loss.backward()
    if capture_path:
        torch.save(additions, capture_path)
    torch.testing.assert_close(actual, expected)
    parameters_by_name = dict(base.named_parameters())
    differences = []
    for name, parameter in eager.named_parameters():
        other = parameters_by_name[name]
        torch.testing.assert_close(parameter.grad, other.grad)
        if not torch.equal(parameter.grad, other.grad):
            differences.append((name, (parameter.grad - other.grad).abs().max().item()))
    if os.environ.get("LLMTUNER_TEST_REPLAY_Q_BREAK") == "1" and not (
        clone_prenorm or view_prenorm
    ):
        # Compare complete local gradients before any distributed reduction.
        # This is diagnostic evidence, not a replacement for resume assertions.
        for backend, model in (("eager", eager), ("aot_eager", base)):
            captured = torch.load(
                root / f"trace-resume-{backend}" / f"step-3-rank-{rank}.pt",
                weights_only=True,
            )["gradients_before_reduce"]
            captured = {key.split(":", 1)[1]: value for key, value in captured.items()}
            replayed = dict(model.named_parameters())
            assert captured.keys() == replayed.keys()
            mismatches = []
            for name, parameter in replayed.items():
                gradient = parameter.grad
                reference = captured[name]
                torch.testing.assert_close(gradient, reference)
                if not torch.equal(gradient, reference):
                    mismatches.append((name, (gradient - reference).abs().max().item()))
            print(
                f"rank={rank} backend={backend} "
                f"replay_vs_fsdp_gradient_differences={mismatches}",
                flush=True,
            )
    print(
        f"rank={rank} replay logits_exact={torch.equal(actual, expected)} "
        f"gradient_differences={differences}",
        flush=True,
    )
    if clone_prenorm or view_prenorm:
        assert torch.equal(actual, expected)
        assert not differences


def inductor_norm():
    """Locate eager/Inductor rounding differences in an isolated forward norm."""
    from torch._dynamo.utils import counters

    torch.set_num_threads(1)
    torch.manual_seed(42)
    inputs = torch.randn(128, 64)

    def parts(value):
        square = value.pow(2)
        mean = square.mean(-1, keepdim=True)
        scale = torch.rsqrt(mean + 1e-6)
        return square, mean, scale, value * scale

    expected = parts(inputs)
    for name, options in (("default", {}), ("scalar", {"cpp.simdlen": 1})):
        actual = torch.compile(parts, backend="inductor", options=options)(inputs)
        for reference, result in zip(expected, actual, strict=True):
            torch.testing.assert_close(reference, result)
        print(
            name,
            [
                (
                    torch.equal(reference, result),
                    (reference - result).abs().max().item(),
                )
                for reference, result in zip(expected, actual, strict=True)
            ],
            flush=True,
        )
    assert counters["stats"]["unique_graphs"] > 0


def inductor_mean_boundary():
    """Keep only mean opaque to isolate forward and backward rounding."""
    from torch._dynamo.utils import counters

    torch.set_num_threads(1)
    torch.manual_seed(42)

    @torch.library.custom_op("llmtuner_diagnostic::mean_last", mutates_args=())
    def mean_last(x: torch.Tensor) -> torch.Tensor:
        return x.mean(-1, keepdim=True)

    @mean_last.register_fake
    def _(x):
        return x.new_empty((*x.shape[:-1], 1))

    def setup(ctx, inputs, output):
        ctx.shape = inputs[0].shape

    def backward(ctx, grad):
        return grad.expand(ctx.shape) / ctx.shape[-1]

    mean_last.register_autograd(backward, setup_context=setup)
    x = torch.randn(128, 64, requires_grad=True)

    def native(x):
        square = x.pow(2)
        mean = square.mean(-1, keepdim=True)
        scale = torch.rsqrt(mean + 1e-6)
        return square, mean, scale, x * scale

    def boundary(x):
        square = x.pow(2)
        mean = mean_last(square)
        scale = torch.rsqrt(mean + 1e-6)
        return square, mean, scale, x * scale

    ref = native(x)
    y = x.detach().clone().requires_grad_(True)
    out = torch.compile(boundary, backend="inductor")(y)
    print(
        "parts",
        [
            (torch.equal(a, b), (a - b).abs().max().item())
            for a, b in zip(ref, out, strict=False)
        ],
        flush=True,
    )
    assert all(torch.equal(a, b) for a, b in zip(ref, out, strict=False))
    ref[-1].square().sum().backward()
    out[-1].square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    assert counters["stats"]["unique_graphs"] > 0
    print(
        "grad_exact",
        torch.equal(x.grad, y.grad),
        "maxabs",
        (x.grad - y.grad).abs().max().item(),
        "graphs",
        counters["stats"]["unique_graphs"],
        flush=True,
    )


def inductor_norm_backward():
    """Compare explicit backward pieces with fixed upstream gradient."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.randn(128, 64)
    g = torch.randn_like(x)
    scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

    @torch.library.custom_op("llmtuner_diagnostic::sum_last", mutates_args=())
    def sum_last(value: torch.Tensor) -> torch.Tensor:
        return value.sum(-1, keepdim=True)

    @sum_last.register_fake
    def _(value):
        return value.new_empty((*value.shape[:-1], 1))

    def parts(x, g, scale, keep_sum=False):
        direct = g * scale
        product = g * x
        reduced = sum_last(product) if keep_sum else product.sum(-1, keepdim=True)
        cube = scale.pow(3)
        variance = (-0.5 * reduced) * cube
        mean_grad = variance.expand_as(x) / x.shape[-1]
        square_grad = 2 * x
        indirect = mean_grad * square_grad
        return (
            direct,
            reduced,
            cube,
            variance,
            mean_grad,
            square_grad,
            indirect,
            direct + indirect,
        )

    ref = parts(x, g, scale)
    out = torch.compile(parts, backend="inductor")(x, g, scale)
    print(
        [
            (torch.equal(a, b), (a - b).abs().max().item())
            for a, b in zip(ref, out, strict=True)
        ],
        flush=True,
    )
    # Compare the decomposed reference against autograd using the same upstream g.
    z = x.clone().requires_grad_()
    norm = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)
    norm.backward(g)
    print(
        "formula_vs_autograd",
        torch.equal(z.grad, ref[-1]),
        (z.grad - ref[-1]).abs().max().item(),
        flush=True,
    )

    assert torch.equal(z.grad, ref[-1])
    for a, b in zip(ref, out, strict=True):
        torch.testing.assert_close(a, b)
    bounded = torch.compile(parts, backend="inductor")(x, g, scale, True)
    print(
        "aten_sum_boundary",
        [
            (torch.equal(a, b), (a - b).abs().max().item())
            for a, b in zip(ref, bounded, strict=True)
        ],
        flush=True,
    )
    for a, b in zip(ref, bounded, strict=True):
        torch.testing.assert_close(a, b)


def inductor_opaque_norm(*, model_probe=False, trainer_probe=False):
    """Validate a first-order opaque norm against eager and finite differences."""
    torch.set_num_threads(1)
    torch.manual_seed(42)

    def native(x, w, eps):
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scale = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
        return w * (value * scale).to(x.dtype)

    @torch.library.custom_op("llmtuner_diagnostic::norm_backward", mutates_args=())
    def norm_backward(
        x: torch.Tensor, w: torch.Tensor, g: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scale = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
        norm = (value * scale).to(x.dtype)
        gw = g * norm
        if x.ndim > 1:
            gw = gw.sum(tuple(range(x.ndim - 1)))
        gw = gw.to(w.dtype)
        gn = (g * w).to(x.dtype).to(value.dtype)
        direct = gn * scale
        gs = (gn * value).sum(-1, keepdim=True)
        variance = (-0.5 * gs) * scale.pow(3)
        indirect = (variance.expand_as(value) / x.shape[-1]) * (2 * value)
        return (direct + indirect).to(x.dtype).contiguous(), gw.contiguous()

    @norm_backward.register_fake
    def _(x, w, g, eps):
        return (
            torch.empty_like(x, memory_format=torch.contiguous_format),
            torch.empty_like(w, memory_format=torch.contiguous_format),
        )

    @torch.library.custom_op("llmtuner_diagnostic::norm", mutates_args=())
    def norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        return native(x, w, eps)

    @norm.register_fake
    def _(x, w, eps):
        return torch.empty_like(x, dtype=torch.promote_types(x.dtype, w.dtype))

    def setup(ctx, inputs, output):
        x, w, ctx.eps = inputs
        ctx.save_for_backward(x, w)

    def backward(ctx, g):
        return *norm_backward(*ctx.saved_tensors, g, ctx.eps), None

    norm.register_autograd(backward, setup_context=setup)
    if os.environ.get("LLMTUNER_TEST_NORM_BENCHMARK") == "1":
        from torch.utils.benchmark import Timer

        for shape in ((128, 64), (512, 1024)):
            value = torch.randn(shape, requires_grad=True)
            weight = torch.randn(shape[-1], requires_grad=True)
            gradient = torch.randn_like(value)
            for label, compute in (
                ("native", native),
                ("fused_inductor", torch.compile(native, backend="inductor")),
                ("opaque_inductor", torch.compile(norm, backend="inductor")),
            ):

                def step(
                    value=value, weight=weight, compute=compute, gradient=gradient
                ):
                    value.grad = None
                    weight.grad = None
                    compute(value, weight, 1e-6).backward(gradient)

                for _ in range(3):
                    step()
                measurement = Timer(
                    stmt="step()", globals={"step": step}
                ).blocked_autorange(min_run_time=0.3)
                print("benchmark_cpu", shape, label, measurement.median, flush=True)
        return
    if os.environ.get("LLMTUNER_TEST_NORM_DTENSOR") == "1":
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import (
            Partial,
            Replicate,
            Shard,
            distribute_tensor,
        )
        from torch.distributed.tensor.experimental import register_sharding

        @register_sharding(torch.ops.llmtuner_diagnostic.norm.default)
        def norm_sharding(x, w, eps):
            return [([Replicate()], [Replicate(), Replicate(), None])] + [
                ([Shard(dim)], [Shard(dim), Replicate(), None])
                for dim in range(x.ndim - 1)
            ]

        @register_sharding(torch.ops.llmtuner_diagnostic.norm_backward.default)
        def backward_sharding(x, w, g, eps):
            return [
                (
                    [Replicate(), Replicate()],
                    [Replicate(), Replicate(), Replicate(), None],
                )
            ] + [
                ([Shard(dim), Partial()], [Shard(dim), Replicate(), Shard(dim), None])
                for dim in range(x.ndim - 1)
            ]

        dist.init_process_group("gloo")
        try:
            multi_axis = os.environ.get("LLMTUNER_TEST_NORM_MESH_2D") == "1"
            mesh = init_device_mesh(
                "cpu", (2, 2) if multi_axis else (dist.get_world_size(),)
            )
            layouts = (
                (
                    (Replicate(), Replicate()),
                    (Shard(0), Replicate()),
                    (Shard(0), Shard(0)),
                )
                if multi_axis
                else ((Replicate(),), (Shard(0),), (Shard(1),))
            )
            replicated = [Replicate()] * mesh.ndim
            for placements in layouts:
                compile_dtensor = (
                    os.environ.get("LLMTUNER_TEST_NORM_COMPILE_DTENSOR") == "1"
                )
                if compile_dtensor:
                    from torch._dynamo.utils import counters

                    torch._dynamo.reset()
                    counters.clear()
                dense = torch.randn(4, 8, requires_grad=True)
                dense_weight = torch.randn(8, requires_grad=True)
                gradient = torch.randn_like(dense)
                value = distribute_tensor(dense.detach(), mesh, placements)
                value.requires_grad_()
                weight = distribute_tensor(dense_weight.detach(), mesh, replicated)
                weight.requires_grad_()
                compute = (
                    torch.compile(norm, backend="inductor", fullgraph=True)
                    if compile_dtensor
                    else norm
                )
                output = compute(value, weight, 1e-6)
                reference = native(dense, dense_weight, 1e-6)
                torch.testing.assert_close(output.full_tensor(), reference)
                output.backward(distribute_tensor(gradient, mesh, placements))
                reference.backward(gradient)
                if compile_dtensor:
                    assert counters["stats"]["unique_graphs"] > 0
                torch.testing.assert_close(value.grad.full_tensor(), dense.grad)
                torch.testing.assert_close(weight.grad.full_tensor(), dense_weight.grad)
                if all(item in (Shard(0), Replicate()) for item in placements):
                    assert weight.grad.placements == tuple(
                        Partial() if item == Shard(0) else Replicate()
                        for item in placements
                    )
                if placements == (Shard(1),):
                    assert output.placements == (Replicate(),)
                print(
                    "DTensor contract passed", dist.get_rank(), placements, flush=True
                )
        finally:
            dist.destroy_process_group()
        return
    if os.environ.get("LLMTUNER_TEST_NORM_DTYPES") == "1":
        from torch._dynamo.utils import counters
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        for input_dtype in (torch.float16, torch.bfloat16, torch.float32):
            for weight_dtype in (torch.float16, torch.bfloat16, torch.float32):
                reference = Qwen3RMSNorm(8, eps=1e-6).to(weight_dtype)
                value = torch.randn(2, 3, 8, dtype=input_dtype, requires_grad=True)
                other = value.detach().clone().requires_grad_()
                weight = reference.weight.detach().clone().requires_grad_()
                expected = reference(value)
                compile_dtypes = (
                    os.environ.get("LLMTUNER_TEST_NORM_COMPILE_DTYPES") == "1"
                )
                if compile_dtypes:
                    torch._dynamo.reset()
                    counters.clear()
                compute = (
                    torch.compile(norm, backend="inductor", fullgraph=True)
                    if compile_dtypes
                    else norm
                )
                actual = compute(other, weight, 1e-6)
                assert actual.dtype == expected.dtype
                assert torch.equal(actual, expected)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                actual.backward(gradient)
                assert torch.equal(value.grad, other.grad)
                assert torch.equal(reference.weight.grad, weight.grad)
                if compile_dtypes:
                    assert counters["stats"]["unique_graphs"] > 0
                print("dtype_contract", input_dtype, weight_dtype, "passed", flush=True)
    if os.environ.get("LLMTUNER_TEST_NORM_SHAPES") == "1":
        for shape in ((4,), (0, 4), (3, 4), (2, 3, 4)):
            for input_grad, weight_grad in ((True, True), (True, False), (False, True)):
                value = torch.randn(shape, requires_grad=input_grad)
                weight = torch.randn(4, requires_grad=weight_grad)
                reference_value = value.detach().clone().requires_grad_(input_grad)
                reference_weight = weight.detach().clone().requires_grad_(weight_grad)
                output = norm(value, weight, 1e-6)
                reference = native(reference_value, reference_weight, 1e-6)
                assert torch.equal(output, reference)
                gradient = torch.randn_like(output)
                output.backward(gradient)
                reference.backward(gradient)
                for actual, expected in (
                    (value.grad, reference_value.grad),
                    (weight.grad, reference_weight.grad),
                ):
                    if expected is None:
                        assert actual is None
                    else:
                        assert torch.equal(actual, expected)
                print("shape_contract", shape, input_grad, weight_grad, "passed")
    if os.environ.get("LLMTUNER_TEST_NORM_OPCHECK") == "1":
        for noncontiguous in (False, True):
            value = torch.randn(2, 3, 4)
            if noncontiguous:
                value = value.transpose(0, 1)
            value.requires_grad_()
            weight = torch.randn(4, requires_grad=True)
            gradient = torch.randn_like(value)
            print(
                "opcheck_forward",
                noncontiguous,
                torch.library.opcheck(norm, (value, weight, 1e-6)),
                flush=True,
            )
            print(
                "opcheck_backward",
                noncontiguous,
                torch.library.opcheck(
                    norm_backward,
                    (value.detach(), weight.detach(), gradient, 1e-6),
                ),
                flush=True,
            )
    if model_probe or trainer_probe:

        def forward(module, hidden_states):
            assert hidden_states.dtype == torch.float32
            return norm(hidden_states, module.weight, module.variance_epsilon)

        if trainer_probe:
            from types import MethodType

            from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

            from llmtuner.trainer.runtime import assembly

            apply = assembly._apply_parallelism

            def configure(trainer, model, **kwargs):
                for module in model.modules():
                    if type(module) is Qwen3RMSNorm:
                        module.forward = MethodType(forward, module)
                return apply(trainer, model, **kwargs)

            with patch.object(assembly, "_apply_parallelism", configure):
                run()
        else:
            isolated_backward(norm_override=forward)
        return
    x = torch.randn(128, 64, requires_grad=True)
    w = torch.randn(64, requires_grad=True)
    g = torch.randn_like(x)
    y = x.detach().clone().requires_grad_()
    v = w.detach().clone().requires_grad_()
    a = native(x, w, 1e-6)
    b = torch.compile(norm, backend="inductor", fullgraph=True)(y, v, 1e-6)
    a.backward(g)
    b.backward(g)
    for label, left, right in [
        ("output", a, b),
        ("input_grad", x.grad, y.grad),
        ("weight_grad", w.grad, v.grad),
    ]:
        print(
            label,
            torch.equal(left, right),
            (left - right).abs().max().item(),
            flush=True,
        )
        assert torch.equal(left, right)
    u = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    z = torch.randn(4, dtype=torch.float64, requires_grad=True)
    print(
        "gradcheck",
        torch.autograd.gradcheck(lambda a, b: norm(a, b, 1e-6), (u, z)),
        flush=True,
    )


def residual_backward():
    """Isolate shared prenorm, graph boundary, and residual gradient merging."""
    import copy

    from torch import nn
    from torch.utils._python_dispatch import TorchDispatchMode

    class CaptureAdditions(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.records = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.add.Tensor and args[0].shape == (128, 64):
                self.records.append(tuple(arg.detach().clone() for arg in args[:2]))
            return func(*args, **(kwargs or {}))

    torch.set_num_threads(1)
    torch.manual_seed(7)
    clone_prenorm = os.environ.get("LLMTUNER_TEST_CLONE_PRENORM") == "1"

    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(64, 64, bias=False)
            self.k = nn.Linear(64, 64, bias=False)
            self.v = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            residual = x
            if clone_prenorm:
                x = x.clone()
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-6)
            return residual + (q * k + v)

    model = Branches()
    base = copy.deepcopy(model)
    base.q.forward = torch.compiler.disable(base.q.forward)
    compiled = torch.compile(base, backend="aot_eager")
    x = torch.randn(128, 64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    g = torch.randn(128, 64)
    a = model(x)
    b = compiled(y)
    eager_adds = CaptureAdditions()
    compiled_adds = CaptureAdditions()
    with eager_adds:
        a.backward(g)
    with compiled_adds:
        b.backward(g)
    # In this fixed diagnostic, the final three-way merge consists of the
    # residual plus the direct and variance paths of the shared RMSNorm.
    assert len(eager_adds.records) == len(compiled_adds.records) == 6
    for eager_inputs, compiled_inputs in zip(
        eager_adds.records[:4], compiled_adds.records[:4], strict=True
    ):
        for eager_input, compiled_input in zip(
            eager_inputs, compiled_inputs, strict=True
        ):
            assert torch.equal(eager_input, compiled_input)
    if clone_prenorm:
        assert torch.equal(x.grad, y.grad)
    else:
        residual, direct = eager_adds.records[4]
        variance = eager_adds.records[5][1]
        assert torch.equal(residual, g)
        assert torch.equal(direct, compiled_adds.records[4][0])
        assert torch.equal(variance, compiled_adds.records[4][1])
        assert torch.equal(residual, compiled_adds.records[5][0])
        assert torch.equal(x.grad, (residual + direct) + variance)
        assert torch.equal(y.grad, residual + (direct + variance))
    for index, (left, right) in enumerate(eager_adds.records):
        print(
            "eager_add",
            index,
            "matching_compiled_inputs",
            [
                other_index
                for other_index, (other_left, other_right) in enumerate(
                    compiled_adds.records
                )
                if torch.equal(left, other_left) and torch.equal(right, other_right)
            ],
        )
    capture_path = os.environ.get("LLMTUNER_TEST_REPLAY_ADDS")
    if capture_path:
        torch.save(
            {"eager": eager_adds.records, "aot_eager": compiled_adds.records},
            capture_path,
        )
    print(
        "outputs_exact",
        torch.equal(a, b),
        "input_grad_exact",
        torch.equal(x.grad, y.grad),
        "maxabs",
        (x.grad - y.grad).abs().max().item(),
    )


def run():
    phase, workdir = sys.argv[1:3]
    backend = os.environ.get("LLMTUNER_TEST_COMPILE_BACKEND", "eager")
    directory = Path(workdir) / f"trace-{phase}-{backend}"
    directory.mkdir(parents=True, exist_ok=True)
    train_step = Trainer.train_step

    def observed_step(trainer, *args, **kwargs):
        update = trainer.optimizer.step
        body = trainer._forward_backward_body
        batches = []

        def observed_body(inputs, labels, *, extra_kwargs, global_valid_tokens):
            # Tensor-only input record; masks stay on the live execution path.
            batches.append(
                {
                    "inputs": inputs.detach().cpu().clone(),
                    "labels": labels.detach().cpu().clone(),
                    "global_valid_tokens": global_valid_tokens.detach().cpu().clone(),
                    "positions": extra_kwargs["positions"].detach().cpu().clone()
                    if isinstance(extra_kwargs.get("positions"), torch.Tensor)
                    else None,
                }
            )
            return body(
                inputs,
                labels,
                extra_kwargs=extra_kwargs,
                global_valid_tokens=global_valid_tokens,
            )

        clip = trainer._clip_and_check_finite
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group

        reduce = _fsdp_param_group.foreach_reduce
        reductions = {}
        reduction_order = []

        def observed_reduce(fsdp_params, unsharded_grads, *args, **kwargs):
            names = []
            for parameter, gradient in zip(fsdp_params, unsharded_grads, strict=True):
                name = canonical_fqn(parameter._param_fqn)
                names.append(name)
                key = f"{len(reduction_order)}:{name}"
                reductions[key] = gradient.detach().cpu().clone()
            reduction_order.append(names)
            return reduce(fsdp_params, unsharded_grads, *args, **kwargs)

        preclip = {}
        loss_sum = trainer.loss_sum
        logits_seen = []

        def observed_loss(logits, *args, **kwargs):
            value = logits._local_tensor if isinstance(logits, DTensor) else logits
            logits_seen.append(value.detach().cpu().clone())
            return loss_sum(logits, *args, **kwargs)

        def observed_clip(*args, **kwargs):
            preclip["gradients_before_clip"] = parameters(trainer, gradients=True)
            result = clip(*args, **kwargs)
            preclip["gradient_norm"] = result.detach().cpu().clone()
            return result

        def observed_update(*args, **kwargs):
            snapshot = {
                **preclip,
                "batches": batches,
                "gradients_before_reduce": reductions,
                "logits": {str(i): value for i, value in enumerate(logits_seen)},
                "parameters_before": parameters(trainer),
                "gradients": parameters(trainer, gradients=True),
            }
            result = update(*args, **kwargs)
            snapshot["parameters_after"] = parameters(trainer)
            torch.save(
                snapshot,
                directory / f"step-{trainer.step}-rank-{trainer.rank}.pt",
            )
            return result

        with (
            patch.object(trainer.optimizer, "step", observed_update),
            patch.object(trainer, "_forward_backward_body", observed_body),
            patch.object(trainer, "_clip_and_check_finite", observed_clip),
            patch.object(trainer, "loss_sum", observed_loss),
            patch.object(_fsdp_param_group, "foreach_reduce", observed_reduce),
        ):
            return train_step(trainer, *args, **kwargs)

    with patch.object(Trainer, "train_step", observed_step):
        projections = os.environ.get("LLMTUNER_TEST_TRACE_PROJECTIONS")
        if projections is None:
            main()
            return
        from llmtuner.trainer.runtime import assembly
        from tests.integration_tests.pp_checkpoint_equivalence import _run

        selected = projections.split(",")
        assert selected and set(selected) <= {"q_proj", "k_proj", "v_proj", "o_proj"}
        implementation = os.environ.get("LLMTUNER_TEST_FSDP_IMPLEMENTATION")
        assert implementation in {"native", "custom"}
        assert "LLMTUNER_TEST_RELOCATED_HOOKS" not in os.environ
        local_hooks = os.environ.get("LLMTUNER_TEST_TRACE_LOCAL_HOOKS") == "1"
        assert implementation == "custom" or local_hooks
        apply = assembly._apply_parallelism

        def configure(trainer, model, **kwargs):
            view_prenorm = os.environ.get("LLMTUNER_TEST_VIEW_PRENORM") == "1"
            if os.environ.get("LLMTUNER_TEST_CLONE_PRENORM") == "1" or view_prenorm:

                def clone_input(module, args):
                    value = args[0]
                    return (
                        value.view_as(value) if view_prenorm else value.clone(),
                        *args[1:],
                    )

                for layer in model.layers:
                    layer.input_layernorm.register_forward_pre_hook(clone_input)
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        clone_input
                    )
            names = {module: name for name, module in model.named_modules()}
            trainer.cfg.parallel.fsdp_config.fsdp_apply_modules = {
                names[getattr(layer.self_attn, name)]: {
                    "hook_module": getattr(layer.self_attn, name)
                    if local_hooks
                    else layer.self_attn
                }
                if implementation == "custom"
                else {}
                for layer in model.layers
                for name in selected
            }
            apply(trainer, model, **kwargs)
            for layer in model.layers:
                for name in selected:
                    state = getattr(layer.self_attn, name)._get_fsdp_state()
                    expected = (
                        getattr(layer.self_attn, name)
                        if local_hooks
                        else layer.self_attn
                    )
                    if implementation == "custom":
                        assert state.hook_module() is expected
                    else:
                        assert state._modules == (expected,)

        with patch.object(assembly, "_apply_parallelism", configure):
            _run()


if __name__ == "__main__":
    if sys.argv[1:] == ["--isolated-backward"]:
        isolated_backward()
    elif sys.argv[1:] == ["--residual-backward"]:
        residual_backward()
    elif sys.argv[1:] == ["--inductor-opaque-qwen"]:
        inductor_opaque_norm(model_probe=True)
    elif sys.argv[1:] == ["--inductor-opaque-norm"]:
        inductor_opaque_norm()
    elif sys.argv[1:] == ["--inductor-norm-backward"]:
        inductor_norm_backward()
    elif sys.argv[1:] == ["--inductor-mean-boundary"]:
        inductor_mean_boundary()
    elif sys.argv[1:] == ["--inductor-norm"]:
        inductor_norm()
    elif sys.argv[1:2] == ["--replay-backward"]:
        replay_backward(sys.argv[2], int(sys.argv[3]))
    else:
        if os.environ.get("LLMTUNER_TEST_OPAQUE_NORM") == "1":
            inductor_opaque_norm(trainer_probe=True)
        else:
            run()
