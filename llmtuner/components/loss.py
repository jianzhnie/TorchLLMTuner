"""Summed training losses and per-token log-probabilities.

Adapted from torchtitan ``components/loss.py``. Callers normalize summed losses
by the global valid-token count. Vocab-parallel CE uses local tensors and TP
collectives; provide both the TP group and global vocabulary size to select it
when the logits contain a vocabulary shard.

``next_token_targets`` shifts synthetic labels within rows during model
preprocessing. Packed dataset labels are already aligned. ``vocab_shard_bounds``
is shared by embedding and loss code to keep vocabulary ownership consistent.
Chunked CE performs backward internally; the other loss functions return tensors
for the caller to backpropagate.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F

from ..accelerator import distributed

__all__ = [
    "IGNORE_INDEX",
    "chunked_lm_head_cross_entropy",
    "compute_logprobs",
    "cross_entropy_loss",
    "mse_loss",
    "next_token_targets",
    "vocab_shard_bounds",
]

# PyTorch's default ignore index for cross-entropy loss.
IGNORE_INDEX = -100


def vocab_shard_bounds(
    global_vocab_size: int, tp_world_size: int, tp_rank: int
) -> tuple[int, int]:
    """The ``[start, end)`` slice of the vocabulary that ``tp_rank`` owns.

    One definition, used by both the vocab-parallel embedding and the
    vocab-parallel loss. They *must* agree -- if the embedding gathers token ids
    from one shard and the loss attributes the target log-probability to another,
    every loss value is wrong while every shape still checks out.

    The shards are even-sized except possibly the last: with ``V=10, tp=3`` the
    chunks are ``[0,4)``, ``[4,8)``, ``[8,10)``. Every rank's start is clamped to
    ``V`` as well as its end, so a TP degree greater than the vocabulary yields
    empty (not negative) slices, which the callers reject explicitly.
    """
    if (
        not isinstance(global_vocab_size, int)
        or isinstance(global_vocab_size, bool)
        or global_vocab_size < 1
    ):
        raise ValueError(f"global_vocab_size must be >= 1, got {global_vocab_size}")
    if (
        not isinstance(tp_world_size, int)
        or isinstance(tp_world_size, bool)
        or tp_world_size < 1
    ):
        raise ValueError(f"tp_world_size must be >= 1, got {tp_world_size}")
    if (
        not isinstance(tp_rank, int)
        or isinstance(tp_rank, bool)
        or not 0 <= tp_rank < tp_world_size
    ):
        raise ValueError(f"tp_rank {tp_rank} is outside [0, {tp_world_size})")
    chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
    start = min(global_vocab_size, chunk_size * tp_rank)
    end = min(global_vocab_size, start + chunk_size)
    return start, max(start, end)


def next_token_targets(labels: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Shift ``labels`` into next-token targets, one document per row.

    ``labels`` is the flat ``(B * T,)`` token stream and ``seq_len`` the row
    length. The model emits ``logits[t]`` predicting ``labels[t + 1]``, and the
    shift is therefore within a *row*, not across the flattened sequence:
    position ``t`` of a row predicts position ``t + 1`` of that same row. The
    last position of every row would predict the first token of the *next*
    document, which is not a prediction the model was given context for, so it
    is marked ``IGNORE_INDEX`` rather than dropped -- keeping the tensor
    rectangular means the ignored entries are excluded from both the loss and
    its denominator (``(targets != IGNORE_INDEX).sum()``), so nothing is
    silently over- or under-counted.
    """
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if labels.numel() % seq_len != 0:
        raise ValueError(
            f"next_token_targets got {labels.numel()} labels, which is not a "
            f"whole number of rows of length {seq_len}"
        )
    targets = torch.full_like(labels, IGNORE_INDEX)
    # Row r of the (B, T) layout occupies [r*T, (r+1)*T); its predictions are the
    # positions [r*T + 1, (r+1)*T) and its targets are [r*T + 1, (r+1)*T).
    targets = targets.reshape(-1, seq_len)
    targets[:, :-1] = labels.reshape(-1, seq_len)[:, 1:]
    targets = targets.reshape(labels.shape)
    return targets


def cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    tp_group: dist.ProcessGroup | None = None,
    global_vocab_size: int | None = None,
) -> torch.Tensor:
    """Cross-entropy over ``pred[T, V]`` and ``labels[T]`` with sum reduction.

    The vocab-parallel path is selected by *shape*, not by a flag: it runs
    exactly when the logits hold fewer than ``global_vocab_size`` classes, which
    is what a sharded lm_head produces and nothing else does. A flag could
    disagree with the tensors; ``pred.shape[-1]`` cannot.

    ``tp_group`` and ``global_vocab_size`` must be provided together: with only
    one of them the shape test cannot be evaluated, and a sharded lm_head would
    silently score against the LOCAL softmax denominator.
    """
    if (tp_group is None) != (global_vocab_size is None):
        raise ValueError(
            "tp_group and global_vocab_size must be provided together: with "
            f"tp_group={tp_group} and global_vocab_size={global_vocab_size}, a "
            "vocab-sharded lm_head cannot be told from a replicated one."
        )
    if tp_group is not None and global_vocab_size is not None:
        if pred.shape[-1] != global_vocab_size:
            return LossParallelCrossEntropy.apply(
                pred.float(), labels, tp_group, global_vocab_size, "sum"
            )

    return F.cross_entropy(
        pred.float(),
        labels,
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def shard_local_labels(
    labels: torch.Tensor, vocab_start: int, local_vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global labels to shard-local indices, masking non-owned tokens.

    Returns ``(local_labels, out_of_range)``: ``local_labels`` shifted by
    ``vocab_start`` with out-of-shard and ``IGNORE_INDEX`` entries clamped to 0
    (safe to index with), and ``out_of_range`` marking exactly those entries.
    Shared by forward and backward so both attribute a target to the same shard.
    """
    safe_labels = torch.where(labels != IGNORE_INDEX, labels, 0)
    out_of_range = (safe_labels < vocab_start) | (
        safe_labels >= vocab_start + local_vocab_size
    )
    local_labels = safe_labels - vocab_start
    local_labels[out_of_range] = 0
    return local_labels, out_of_range


class LossParallelCrossEntropy(torch.autograd.Function):
    """Vocab-parallel cross-entropy on local ``[T, V_local]`` logits.

    For tensor parallelism that shards the lm_head weight on its vocab dim:
    each rank holds ``V/tp`` output classes and a target token is known to
    exactly one rank, so the softmax denominator has to be assembled across the
    group before any rank can produce a loss.

    Forward uses three TP all-reduces -- max (for a numerically stable shifted
    softmax), sum-of-exp (the denominator), and gather (picking the owner rank's
    log-probability for each target). Backward is fused (NLL + log-softmax
    derivative) with **zero** collectives: each rank already holds the full
    ``[T, V_local]`` log-probability slice it needs the gradient for.

    Supports uneven vocab sharding (the last TP rank may hold fewer classes) and
    ``IGNORE_INDEX`` labels. All inputs and outputs are plain local tensors, not
    DTensors.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
        global_vocab_size: int,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Exact CE from local vocab shards via TP all-reduces.

        ``reduction="sum"`` returns the scalar summed loss. ``reduction="none"``
        returns the per-token NLL ``[T]``, which GRPO negates to get per-token
        logprobs without ever all-gathering the vocabulary.
        """
        if reduction not in ("sum", "none"):
            raise ValueError(
                f"reduction must be 'sum' or 'none', got {reduction!r}: the "
                "vocab-parallel sums are token-summed, and 'mean' here would "
                "divide by the wrong count."
            )
        logits_dtype = logits.dtype
        logits = logits.float()

        # This rank's slice of the vocabulary, by the same bounds the
        # vocab-parallel embedding uses -- the two must agree or the embedding
        # and the loss disagree about which rank owns which token.
        tp_world_size = distributed.get_world_size(tp_group)
        tp_rank = distributed.get_rank(tp_group)
        vocab_start, vocab_end = vocab_shard_bounds(
            global_vocab_size, tp_world_size, tp_rank
        )
        local_vocab_size = max(0, vocab_end - vocab_start)
        if logits.shape[-1] != local_vocab_size:
            raise ValueError(
                "LossParallelCrossEntropy expected local vocab size "
                f"{local_vocab_size} for global vocab size {global_vocab_size}, "
                f"got {logits.shape[-1]}."
            )
        if local_vocab_size == 0:
            raise ValueError(
                "LossParallelCrossEntropy does not support empty vocab shards. "
                f"Global vocab {global_vocab_size} is smaller than the TP degree "
                f"{tp_world_size}."
            )

        # A label outside [0, global_vocab_size) would be silently dropped by the
        # shard mask below rather than reported, so check it on device -- no host
        # sync, and no invalid target can reach the gather.
        torch._assert_async(
            torch.all(
                (labels == IGNORE_INDEX)
                | ((labels >= 0) & (labels < global_vocab_size))
            ),
            f"labels must be {IGNORE_INDEX} or in [0, {global_vocab_size})",
        )

        # All-reduce max for a numerically stable distributed log-softmax.
        local_max = torch.amax(logits, dim=-1, keepdim=True)
        local_max = funcol.all_reduce(
            local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
        )

        # All-reduce the shifted sum-of-exp: the global softmax denominator.
        shifted = logits - local_max
        shifted_sumexp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
        shifted_sumexp = funcol.all_reduce(
            shifted_sumexp, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )
        log_probs = shifted - torch.log(shifted_sumexp)

        # Mask labels outside this shard; the all-reduce below then selects the
        # owner rank's log-probability for each target.
        local_labels, out_of_range = shard_local_labels(
            labels, vocab_start, local_vocab_size
        )

        local_result = torch.gather(log_probs, -1, local_labels.unsqueeze(-1))
        local_result[out_of_range.unsqueeze(-1)] = 0
        local_result = funcol.all_reduce(
            local_result, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )

        # Per-token NLL, with ignored labels zeroed (their log-prob is 0 above).
        result = -local_result.squeeze(-1)
        result = torch.where(labels != IGNORE_INDEX, result, 0)

        ctx.save_for_backward(log_probs, labels)
        ctx.logits_dtype = logits_dtype
        ctx.vocab_start = vocab_start
        ctx.local_vocab_size = local_vocab_size
        ctx.reduction = reduction
        if reduction == "none":
            return result
        return result.sum()

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        log_probs, labels = ctx.saved_tensors
        local_labels, out_of_range = shard_local_labels(
            labels, ctx.vocab_start, ctx.local_vocab_size
        )

        # d/dz [ -log_softmax(z)_y ] = softmax(z) - onehot(y), assembled only for
        # this rank's slice -- targets outside the shard contribute
        # softmax(z) * 1, which is what the out_of_range branch adds back.
        grad_input = torch.zeros_like(log_probs)
        row_idx = torch.arange(local_labels.shape[0], device=local_labels.device)
        grad_update = out_of_range.to(grad_input.dtype) - 1.0
        grad_input[row_idx, local_labels] = grad_update

        # reduction="none" hands back a per-token [T] upstream grad; unsqueeze it
        # to broadcast over the local vocab. "sum" gives the scalar, which does.
        if ctx.reduction == "none":
            grad_output = grad_output.unsqueeze(-1)
        grad_output = torch.where(
            (labels != IGNORE_INDEX).unsqueeze(-1), grad_output, 0
        )
        grad_logits = (grad_input + torch.exp(log_probs)) * grad_output
        return grad_logits.to(ctx.logits_dtype), None, None, None, None


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE loss with sum reduction, for models trained on continuous targets."""
    return F.mse_loss(pred.float(), labels.float().detach(), reduction="sum")


def chunked_lm_head_cross_entropy(
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_chunks: int,
    grad_scale: torch.Tensor | float,
    tp_group: dist.ProcessGroup | None = None,
    global_vocab_size: int | None = None,
) -> torch.Tensor:
    """Compute summed CE in token chunks and perform scaled backward.

    Inputs are aligned ``hidden_states[T, H]`` and ``labels[T]``. Returns a
    detached, unnormalized loss; gradients are scaled by ``grad_scale``.
    Hidden states must require grad. The caller must not backward the result.

    Each chunk runs the head and backward independently, bounding logits memory
    while accumulating head gradients. A single backward then propagates the
    assembled hidden-state gradients through the decoder. Chunks need not divide
    T, and CP/TP sequence shards can be chunked locally.

    ``tp_group`` and ``global_vocab_size`` follow ``cross_entropy_loss``'s
    vocabulary-shard contract. FSDP head resharding and gradient synchronization
    are not deferred across chunks, so each chunk can incur extra collectives.
    """
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be (T, H), got shape {tuple(hidden_states.shape)}"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels must be (T,), got shape {tuple(labels.shape)}")
    if labels.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match the token axis of "
            f"hidden_states ({hidden_states.shape[0]})"
        )
    if not hidden_states.requires_grad:
        raise ValueError(
            "chunked_lm_head_cross_entropy is a training path: hidden_states "
            "must require grad."
        )

    hidden_chunks = hidden_states.chunk(num_chunks, dim=0)
    label_chunks = labels.chunk(num_chunks, dim=0)
    total = hidden_states.new_zeros((), dtype=torch.float32)
    hidden_grads: list[torch.Tensor] = []
    for hidden_chunk, label_chunk in zip(hidden_chunks, label_chunks, strict=True):
        # detach + requires_grad_ makes the chunk a leaf, so its backward
        # stops at the lm_head boundary and the chunk's hidden gradient lands
        # in ``.grad`` for assembly below.
        detached = hidden_chunk.detach().requires_grad_(True)
        logits = lm_head(detached)
        chunk_loss = cross_entropy_loss(
            logits,
            label_chunk,
            tp_group=tp_group,
            global_vocab_size=global_vocab_size,
        )
        total = total + chunk_loss.detach()
        (chunk_loss * grad_scale).backward()
        assert detached.grad is not None
        hidden_grads.append(detached.grad)

    torch.autograd.backward(hidden_states, grad_tensors=torch.cat(hidden_grads))
    return total


def vocab_parallel_entropy(
    logits: torch.Tensor, tp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Exact per-token Shannon entropy from vocab-sharded logits, no gather.

    ``H(p) = logsumexp(z) - sum(softmax(z) * z)`` needs two cross-shard sums:
    the softmax denominator and the first moment ``sum(p * z)``. Both are
    assembled with all-reduces against a shared max, so the vocabulary never
    has to be gathered to compute the metric.
    """
    logits = logits.float()

    local_max = torch.amax(logits, dim=-1, keepdim=True)
    global_max = funcol.all_reduce(
        local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
    )

    shifted = logits - global_max
    shifted_exp = torch.exp(shifted)
    local_sumexp = shifted_exp.sum(dim=-1)
    # Avoid 0 * -inf for finite distributions with masked logits while
    # preserving NaNs for invalid distributions such as all -inf logits.
    shifted_weighted = torch.where(
        torch.isneginf(shifted),
        torch.zeros_like(shifted),
        shifted_exp * shifted,
    )
    local_weighted_sum = shifted_weighted.sum(dim=-1)
    global_stats = funcol.all_reduce(
        torch.stack((local_sumexp, local_weighted_sum)),
        reduceOp=dist.ReduceOp.SUM.name,
        group=tp_group,
    )
    sumexp, weighted_sum = global_stats.unbind()
    return torch.log(sumexp) - weighted_sum / sumexp


def compute_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    tp_group: dist.ProcessGroup | None = None,
    global_vocab_size: int | None = None,
    return_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities from ``logits[T, V]`` and ``labels[T]``.

    With a sharded vocabulary each rank holds only its own classes, so the
    log-probabilities come from the vocab-parallel path and the entropy from
    ``vocab_parallel_entropy`` -- neither gathers the vocabulary. Reachability
    note: nothing in llmtuner shards the lm_head yet, so today the sharded
    branch is unreachable in training -- it is here because the local-vocab case
    is the whole reason llmtuner has an ``LossParallelCrossEntropy`` at all.

    When ``return_entropy`` is set, also returns per-token Shannon entropy
    ``H(p) = logsumexp(logits) - sum(softmax(logits) * logits)``, shape ``[T]``.
    Entropy is a metric only, so it is computed under ``no_grad``: it contributes
    no gradient and must not extend the autograd graph over the logits softmax.

    Returns ``logprobs``, or ``(logprobs, entropy)`` when ``return_entropy``.

    ``tp_group`` and ``global_vocab_size`` must be provided together (see
    ``cross_entropy_loss``): half a pair makes the vocab-parallel shape test
    unevaluable and would silently score against the local softmax.
    """
    if (tp_group is None) != (global_vocab_size is None):
        raise ValueError("tp_group and global_vocab_size must be provided together.")
    if (
        tp_group is not None
        and global_vocab_size is not None
        and logits.shape[-1] != global_vocab_size
    ):
        # reduction="none" is the vocab-parallel path's per-token form: it
        # returns -NLL directly, so no vocab all-gather is needed.
        logprobs = -LossParallelCrossEntropy.apply(
            logits, labels, tp_group, global_vocab_size, "none"
        )
        if not return_entropy:
            return logprobs
        with torch.no_grad():
            entropy = vocab_parallel_entropy(logits, tp_group)
        return logprobs, entropy

    # One bf16 -> fp32 upcast, shared by the logprobs and (if asked) the entropy.
    logits = logits.float()
    logprobs = -F.cross_entropy(
        logits,
        labels,
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    if not return_entropy:
        return logprobs
    with torch.no_grad():
        log_probs = F.log_softmax(logits, dim=-1)
        # Masked classes have p=0 and contribute zero, rather than 0 * -inf.
        # Keep NaNs for invalid distributions such as an all-masked row.
        entropy = -torch.where(
            torch.isneginf(log_probs),
            torch.zeros_like(log_probs),
            log_probs.exp() * log_probs,
        ).sum(dim=-1)
    return logprobs, entropy
