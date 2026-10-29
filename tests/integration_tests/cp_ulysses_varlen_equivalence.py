"""Ulysses CP x packed (varlen) sequences: full-length mask, full-sequence math.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/cp_ulysses_varlen_equivalence.py

Where ``cp_ulysses_equivalence.py`` pins the single-document ulysses wiring,
this pins the varlen/packed combination that used to be refused at attach
time. The premise: ulysses all-to-all's every rank's token shard into a head
shard BEFORE attention, so each rank attends the full, unpermuted token stream
with ``heads / cp`` heads -- and the packed document structure (llmtuner's
varlen metadata, baked into the BlockMask from the full positions) therefore
applies to the kernel FULL-LENGTH and unsharded, exactly as upstream's ulysses
``cp_shard`` lifts ``attention_masks`` out of the sharded inputs.

Three layers of checks:

* the seam: ``preprocess_inputs`` on a multi-document packed row must hand the
  forward a full-length (unsharded) document mask under ulysses -- Q-sharded
  would index the wrong queries;
* the wiring: sharded logits gathered in rank order equal the single-rank
  dense block-causal reference (documents of uneven lengths, one of length 1,
  boundaries off the CP split and off the mask's 128-block grid). Non-vacuity:
  a causal-only reference -- attention crossing document boundaries -- must
  differ materially, so a kernel that silently dropped the document structure
  could not pass;
* the gradients: torch's flex attention has no CPU backward, so the collective
  duality is pinned around an SDPA inner attention with the same dense
  document mask: loss and q/k/v gradients of the sharded path
  (``SeqToHead`` -> attention -> ``HeadToSeq``) must equal the full-sequence
  reference's shards, and must be nonzero (non-vacuity).

CPU uses float64 and explicit flex; NPU uses float32 and production CP SDPA.
The NPU arm also compares all model parameter gradients to the dense reference.
Set LLMTUNER_TEST_DUAL_CP=1 with four accelerator ranks for KV2/Ulysses2,
including the production assembly and full document-mask path.
Add LLMTUNER_TEST_TP_SIZE=2 with four ranks for single-axis CP, or eight
ranks for dual CP, to verify TP inside each CP shard on an accelerator.
Set LLMTUNER_TEST_DP_MODE=replicate or shard to verify production FSDP
gradients against the mean of distinct DP inputs: use four ranks for single-axis
Ulysses, or eight ranks with LLMTUNER_TEST_DUAL_CP=1 for dual CP.
"""

from __future__ import annotations

import inspect
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

from llmtuner.config import (
    LLMTunerConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.config.parallel_config import ContextParallelConfig
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.context_parallel import apply_cp
from llmtuner.parallel.context_parallel.attention import HeadToSeq, SeqToHead
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.harness import device, init_distributed, moved

SEQ = 256  # torch's CP BlockMask path requires Q_LEN % (cp * 128) == 0
VOCAB = 128
DOC_LENS = (90, 1, 165)  # uneven, off the cp split (128) and the block grid

DEVICE = device()
DTYPE = torch.float64 if DEVICE.type == "cpu" else torch.float32
TOL = 1e-9 if DEVICE.type == "cpu" else 1e-5
NONVACUITY = 1e-3
DUAL = os.environ.get("LLMTUNER_TEST_DUAL_CP") == "1"
TP = int(os.environ.get("LLMTUNER_TEST_TP_SIZE", "1"))
DP_MODE = os.environ.get("LLMTUNER_TEST_DP_MODE", "none")
DP = 1 if DP_MODE == "none" else 2
KV_ONLY = os.environ.get("LLMTUNER_TEST_KV_ONLY") == "1"
KV_PACKED = os.environ.get("LLMTUNER_TEST_KV_PACKED") == "1"
PACKED = not KV_ONLY or KV_PACKED


def _cfg() -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            data_parallel_replicate_size=2 if DP_MODE == "replicate" else 1,
            data_parallel_shard_size=2 if DP_MODE == "shard" else 1,
            tensor_parallel_size=TP,
            context_parallel_kv_allgather_size=2 if DUAL or KV_ONLY else 1,
            context_parallel_ulysses_size=1 if KV_ONLY else 2,
            cp_config=ContextParallelConfig(
                context_parallel_strategy="kv_allgather" if KV_ONLY else "ulysses",
                context_parallel_load_balancer=None,
            ),
        ),
        training=TrainingConfig(max_seq_len=SEQ, steps=1),
    )


def _build_model(cfg: LLMTunerConfig, *, flex: bool, seed: int = 0):
    torch.manual_seed(seed)
    model = moved(HFTransformerModel(build_model_config_for(cfg)).to(DTYPE))
    # The packed mask is a document mask; declare the corpus shape the way
    # ``build_model_config_for`` would for any non-random dataset.
    model.model.config.attn_mask_type = "block_causal" if PACKED else "causal"
    if flex:
        model.model.config._attn_implementation = "flex_torchtitan"
    return model.eval()


def _packed_data(seed: int = 0):
    """A multi-document packed row: ids/labels plus positions resetting per doc."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(VOCAB, (SEQ,), generator=g)
    labels = torch.randint(VOCAB, (SEQ,), generator=g)
    positions = (
        torch.cat([torch.arange(n) for n in DOC_LENS]) if PACKED else torch.arange(SEQ)
    )
    assert positions.shape[0] == SEQ
    return moved(ids), moved(labels), moved(positions)


def _packed_additive_mask(positions: torch.Tensor, *, document: bool) -> torch.Tensor:
    """Dense (1, 1, T, T) block-causal additive mask, float64.

    ``document=False`` drops the document half -- the cross-boundary reference
    the non-vacuity check needs.
    """
    t = positions.shape[0]
    allow = torch.tril(torch.ones(t, t, dtype=torch.bool, device=DEVICE))
    if document:
        # Same doc-id convention as masks.get_document_mask_mod.
        doc_ids = torch.cumsum((positions == 0).int(), dim=0) - 1
        allow &= doc_ids[:, None] == doc_ids[None, :]
    additive = torch.zeros(t, t, dtype=DTYPE, device=DEVICE)
    return additive.masked_fill(~allow, float("-inf"))[None, None]


def _ref_forward(ref, ids, positions, *, document: bool = True):
    """Full-sequence forward through HF's own sdpa path with a dense mask."""
    hidden = ref.model.model(
        ids.unsqueeze(0),
        position_ids=positions.unsqueeze(0),
        attention_mask=_packed_additive_mask(positions, document=document),
        use_cache=False,
    ).last_hidden_state.squeeze(0)
    return ref.model.lm_head(hidden)


def _uncompiled_create_attention_mask(*args, **kwargs):
    """The production mask builder minus ``torch.compile`` (CPU inductor bug)."""
    from torch.nn.attention.flex_attention import create_block_mask

    if "separate_full_blocks" not in inspect.signature(create_block_mask).parameters:
        kwargs.pop("separate_full_blocks", None)
    return create_block_mask(*args, **kwargs)


class _CpOnlyDims:
    def __init__(self, mesh):
        self._mesh = mesh

    def get_optional_mesh(self, name: str):
        assert name in ("cp", "tp"), f"preprocess_inputs asked for '{name}'"
        return self._mesh["cp"] if name == "cp" else None


def _run_ulysses_packed(
    mesh, failures: list[str], parallel_dims=None
) -> dict[str, float]:
    cfg = _cfg()
    model = _build_model(cfg, flex=DEVICE.type == "cpu")
    ref = _build_model(cfg, flex=False)
    tp_mesh = parallel_dims.get_optional_mesh("tp") if parallel_dims else None
    if tp_mesh is not None:
        model = apply_tp(model, tp_mesh, cfg.parallel)
        from llmtuner.parallel.tensor_parallel.tensor_parallel import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        checked_projections = 0
        for name, module in model.named_modules():
            if not isinstance(module, ColumnParallelLinear | RowParallelLinear):
                continue
            reference_weight = ref.get_submodule(name).weight
            axis = 0 if isinstance(module, ColumnParallelLinear) else 1
            expected_weight = reference_weight.chunk(TP, dim=axis)[
                tp_mesh.get_local_rank()
            ]
            assert module.tp_size == TP
            assert module.weight.numel() * TP == reference_weight.numel()
            torch.testing.assert_close(module.weight, expected_weight, rtol=0, atol=0)
            checked_projections += 1
        assert checked_projections >= 4 * cfg.model.num_hidden_layers, (
            f"TP did not shard the expected projections: {checked_projections}"
        )
    apply_cp(model, mesh=mesh, cfg=cfg.parallel, parallel_dims=parallel_dims)
    if KV_ONLY:
        from llmtuner.parallel.context_parallel.sdpa_attention import CPSdpaKernel

        assert all(
            isinstance(layer.self_attn._titan_cp_sdpa_kernel, CPSdpaKernel)
            for layer in model.layers
        )
    if DUAL:
        from llmtuner.parallel.context_parallel.sdpa_attention import DualCPSdpaKernel

        assert model.model.config._cp_sdpa_dual
        assert all(
            isinstance(layer.self_attn._titan_cp_sdpa_kernel, DualCPSdpaKernel)
            for layer in model.layers
        )

    ids, labels, positions = _packed_data()
    if DP > 1:
        dp_mesh = parallel_dims.get_mesh("dp")
        dp_rank = dp_mesh.get_local_rank()
        ids = (ids + dp_rank * 17) % VOCAB
        labels = (labels + dp_rank * 17) % VOCAB
        model = apply_fsdp(model, cfg.parallel, parallel_dims)
        for parameter in model.parameters():
            assert isinstance(parameter, DTensor)
            assert parameter.to_local().numel() < parameter.numel()
            if DP_MODE == "replicate":
                axis = parameter.device_mesh.mesh_dim_names.index("dp_replicate")
                assert parameter.device_mesh.size(axis) == 2
                assert isinstance(parameter.placements[axis], Replicate)
    mask_builder = mock.patch(
        "llmtuner.models.hf.model.create_attention_mask",
        _uncompiled_create_attention_mask,
    )
    with mask_builder if DEVICE.type == "cpu" else nullcontext():
        inputs, out_labels, extra = model.preprocess_inputs(
            {"input": ids, "labels": labels, "positions": positions},
            parallel_dims=parallel_dims or _CpOnlyDims(mesh),
        )

    # THE pin: under ulysses the packed mask must reach the forward FULL-LENGTH
    # (unsharded) -- the varlen input contract. Q-sharded would be (SEQ/cp, SEQ).
    mask = extra.get("attention_masks")
    if mask is None and PACKED:
        failures.append("preprocess_inputs built no mask for a packed ulysses row")
        return {}
    mask_shape = (
        tuple(mask.shape[-2:])
        if isinstance(mask, torch.Tensor)
        else tuple(mask.seq_lengths)
        if mask is not None
        else None
    )
    if PACKED and mask_shape != (SEQ, SEQ):
        failures.append(
            f"packed ulysses mask seq_lengths {mask_shape}, "
            f"expected the full-length ({SEQ}, {SEQ})"
        )

    with torch.no_grad():
        logits_local = model(
            inputs,
            positions=extra["positions"],
            attention_masks=mask,
        )
        ref_logits = _ref_forward(ref, ids, positions)
        cross_boundary = _ref_forward(ref, ids, positions, document=False)

    logits_cp = logits_local
    if tp_mesh is not None:
        tp_blocks = [torch.empty_like(logits_local) for _ in range(tp_mesh.size())]
        dist.all_gather(
            tp_blocks, logits_local.detach().contiguous(), group=tp_mesh.get_group()
        )
        logits_cp = torch.cat(tp_blocks, dim=0)
    gathered = [torch.empty_like(logits_cp) for _ in range(mesh["cp"].size())]
    assert torch.isfinite(logits_local).all().item()
    assert torch.isfinite(ref_logits).all().item()
    dist.all_gather(
        gathered, logits_cp.detach().contiguous(), group=mesh["cp"].get_group()
    )
    logit_diff = (torch.cat(gathered, dim=0) - ref_logits).abs().max().item()
    if logit_diff > TOL:
        failures.append(f"ulysses/packed: logits max abs diff {logit_diff:.3e}")

    # Non-vacuity: dropping the document structure must change the answer
    # materially, so a kernel attending across boundaries cannot pass above.
    doc_gap = (ref_logits - cross_boundary).abs().max().item()
    if PACKED and doc_gap < NONVACUITY:
        failures.append(f"document mask is a no-op ({doc_gap:.3e}), vacuous")

    # Summed token losses agree after the CP reduction.
    with torch.no_grad():
        loss_local = F.cross_entropy(logits_local, out_labels, reduction="sum")
        loss_ref = F.cross_entropy(ref_logits, labels, reduction="sum")
    dist.all_reduce(loss_local, group=mesh["cp"].get_group())
    if tp_mesh is not None:
        dist.all_reduce(loss_local, group=tp_mesh.get_group())
    loss_diff = abs(loss_local.item() - loss_ref.item())
    if DEVICE.type != "cpu":
        loss_diff /= labels.numel()
    if loss_diff > TOL:
        failures.append(f"ulysses/packed: loss diff {loss_diff:.3e}")

    param_grad_diff = 0.0
    if DEVICE.type != "cpu":
        actual_logits = model(
            inputs, positions=extra["positions"], attention_masks=mask
        )
        expected_logits = _ref_forward(ref, ids, positions)
        (
            F.cross_entropy(actual_logits, out_labels, reduction="sum") / (SEQ * DP)
        ).backward()
        F.cross_entropy(expected_logits, labels).backward()
        if DP == 1:
            for parameter in model.parameters():
                assert parameter.grad is not None
                dist.all_reduce(parameter.grad, group=mesh["cp"].get_group())
        else:
            dp_gradient_gap = 0.0
            for parameter in ref.parameters():
                assert parameter.grad is not None
                peer_gradients = [torch.empty_like(parameter.grad) for _ in range(DP)]
                dist.all_gather(
                    peer_gradients, parameter.grad, group=dp_mesh.get_group()
                )
                dp_gradient_gap = max(
                    dp_gradient_gap,
                    (peer_gradients[0] - peer_gradients[1]).abs().max().item(),
                )
                dist.all_reduce(parameter.grad, group=dp_mesh.get_group())
                parameter.grad.div_(DP)
            assert dp_gradient_gap > NONVACUITY, dp_gradient_gap
        if tp_mesh is not None:
            Trainer._allreduce_replicated_tp_grads(
                SimpleNamespace(parallel_dims=parallel_dims, model_parts=[model])
            )
        for (name, parameter), (ref_name, reference) in zip(
            model.named_parameters(), ref.named_parameters(), strict=True
        ):
            assert name == ref_name
            assert parameter.grad is not None and reference.grad is not None
            actual_gradient = (
                parameter.grad.full_tensor()
                if isinstance(parameter.grad, DTensor)
                else parameter.grad
            )
            assert torch.isfinite(actual_gradient).all().item()
            assert torch.isfinite(reference.grad).all().item()
            target = reference.grad
            if tp_mesh is not None:
                if any(
                    name.endswith(f".{projection}.weight")
                    for projection in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "gate_proj",
                        "up_proj",
                    )
                ):
                    target = target.chunk(TP, dim=0)[tp_mesh.get_local_rank()]
                elif any(
                    name.endswith(f".{projection}.weight")
                    for projection in ("o_proj", "down_proj")
                ):
                    target = target.chunk(TP, dim=1)[tp_mesh.get_local_rank()]
            difference = (actual_gradient - target).abs().max().item()
            param_grad_diff = max(param_grad_diff, difference)
            if not torch.allclose(actual_gradient, target, atol=TOL, rtol=TOL):
                failures.append(
                    f"packed model gradient {name}: max AE {difference:.3e}"
                )

    return {
        "logits": logit_diff,
        "loss": loss_diff,
        "doc_gap": doc_gap,
        "parameter_grad": param_grad_diff,
    }


def _check_gradients_under_varlen_mask(
    cp_mesh, failures: list[str]
) -> dict[str, float]:
    """Loss and q/k/v grads: sharded path vs full-sequence reference, varlen mask.

    Flex has no CPU backward, so the inner attention here is SDPA against the
    same dense document mask -- what is pinned is the collective duality
    (``SeqToHead`` / ``HeadToSeq`` forward-backward pairing) carrying correct
    gradients under a varlen mask, not the flex kernel.
    """
    rank = cp_mesh.get_local_rank()
    cp = cp_mesh.size()
    group = cp_mesh.get_group()
    b, h, s, d = 1, 4, SEQ, 16
    g = torch.Generator().manual_seed(0)
    qkv_full = [moved(torch.randn(b, h, s, d, generator=g, dtype=DTYPE)) for _ in "qkv"]
    w = moved(torch.randn(b, h, s, d, generator=g, dtype=DTYPE))
    _, _, positions = _packed_data()
    mask = _packed_additive_mask(positions, document=True)

    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # Reference: full sequence, per-rank token shard of the loss weight.
    lo, hi = rank * s // cp, (rank + 1) * s // cp
    ref_qkv = [x.clone().requires_grad_(True) for x in qkv_full]
    ref_out = sdpa(*ref_qkv)
    (ref_out * w).sum().backward()

    # Sharded: token shard -> head shard -> attention -> token shard.
    sh_qkv = [x[:, :, lo:hi].clone().requires_grad_(True) for x in qkv_full]
    a2a_qkv = [SeqToHead.apply(x, group) for x in sh_qkv]
    out = sdpa(*a2a_qkv)
    out = HeadToSeq.apply(out, group)
    (out * w[:, :, lo:hi]).sum().backward()

    out_diff = (out - ref_out[:, :, lo:hi]).abs().max().item()
    assert torch.isfinite(out).all().item()
    assert torch.isfinite(ref_out).all().item()
    if out_diff > TOL:
        failures.append(f"varlen grad path: output diff {out_diff:.3e}")
    grad_diff = max(
        (sh.grad - ref.grad[:, :, lo:hi]).abs().max().item()
        for sh, ref in zip(sh_qkv, ref_qkv, strict=True)
    )
    for sharded, reference in zip(sh_qkv, ref_qkv, strict=True):
        assert torch.isfinite(sharded.grad).all().item()
        assert torch.isfinite(reference.grad).all().item()
    if grad_diff > TOL:
        failures.append(f"varlen grad path: grad diff {grad_diff:.3e}")
    grad_norm = min(sh.grad.abs().max().item() for sh in sh_qkv)
    if grad_norm == 0.0:
        failures.append("varlen grad path: an all-zero gradient, vacuous")

    return {"output": out_diff, "grad": grad_diff}


def main() -> None:
    init_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert TP in {1, 2}
    assert not KV_ONLY or (not DUAL and DEVICE.type != "cpu")
    assert not KV_PACKED or KV_ONLY, "KV_PACKED requires KV_ONLY=1"
    assert DP_MODE in {"none", "replicate", "shard"}
    assert DP == 1 or TP == 1
    cp_size = 4 if DUAL else 2
    expected_world = cp_size * TP * DP
    assert world == expected_world, f"expected {expected_world} ranks, got {world}"
    parallel_dims = None
    if DUAL or DP > 1 or KV_ONLY or TP > 1:
        assert DEVICE.type != "cpu", "Parallel model test requires accelerator SDPA"
        parallel_dims = ParallelTopology(
            dp_replicate=2 if DP_MODE == "replicate" else 1,
            dp_shard=2 if DP_MODE == "shard" else 1,
            cp=cp_size,
            tp=TP,
            pp=1,
            ep=1,
            world_size=expected_world,
            cp_kv=2 if DUAL or KV_ONLY else 1,
            cp_ulysses=1 if KV_ONLY else 2,
        )
        parallel_dims.build_device_mesh()
        mesh = parallel_dims.get_mesh("cp")
    else:
        mesh = init_device_mesh(DEVICE.type, (world,), mesh_dim_names=("cp",))
    failures: list[str] = []

    stats = _run_ulysses_packed(mesh, failures, parallel_dims)
    grad_stats = _check_gradients_under_varlen_mask(mesh["cp"], failures)

    local_ok = moved(torch.tensor([0.0 if not failures else 1.0]))
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"seq={SEQ} cp_size={mesh['cp'].size()} tp_size={TP} "
            f"kv_only={KV_ONLY} dp_mode={DP_MODE} "
            f"packed={PACKED} doc_lens={DOC_LENS if PACKED else (SEQ,)} "
            f"dtype={DTYPE} tol={TOL:.0e}"
        )
        if stats:
            print(
                f"{'kv' if KV_ONLY else 'ulysses'}/{'packed' if PACKED else 'causal'} "
                f"logits={stats['logits']:.3e} "
                f"loss={stats['loss']:.3e} doc_gap={stats['doc_gap']:.3e}"
            )
            if DEVICE.type != "cpu":
                print(f"parameter_grad={stats['parameter_grad']:.3e}")
        print(
            f"varlen grad path out={grad_stats['output']:.3e} "
            f"grad={grad_stats['grad']:.3e}"
        )
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "CP ulysses varlen equivalence check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
