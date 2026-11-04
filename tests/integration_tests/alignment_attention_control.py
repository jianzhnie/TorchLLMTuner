"""Diagnostic attention geometry control; never a production backend."""

import torch


def batched_attention_control(query, key, value, attention_mask, geometry, **kwargs):
    batch_size, sequence = geometry
    total = batch_size * sequence
    if query.shape[0] != 1 or query.shape[2] != total:
        raise ValueError("Attention control received unexpected TP geometry")
    indices = torch.arange(total, device=query.device)
    expected_mask = (indices[:, None] >= indices[None, :]) & (
        indices[:, None] // sequence == indices[None, :] // sequence
    )
    if attention_mask is None:
        raise ValueError("Attention control requires explicit mask")
    if attention_mask.dtype == torch.bool:
        allowed = attention_mask
    elif attention_mask.is_floating_point():
        blocked = torch.isneginf(attention_mask) | (
            attention_mask == torch.finfo(attention_mask.dtype).min
        )
        if not torch.all((attention_mask == 0) | blocked):
            raise ValueError("Attention control rejects attention mask biases")
        allowed = attention_mask == 0
    else:
        raise ValueError("Attention control requires boolean or additive mask")
    if not torch.equal(allowed.squeeze(), expected_mask):
        raise ValueError("Attention control mask differs from document causal")

    def unflatten(tensor):
        return (
            tensor.transpose(1, 2)
            .reshape(batch_size, sequence, tensor.shape[1], tensor.shape[-1])
            .transpose(1, 2)
        )

    output = torch.nn.functional.scaled_dot_product_attention(
        unflatten(query),
        unflatten(key),
        unflatten(value),
        is_causal=True,
        enable_gqa=True,
        dropout_p=kwargs.get("dropout", 0.0),
        scale=kwargs.get("scaling"),
    )
    output = (
        output.transpose(1, 2)
        .reshape(1, total, query.shape[1], query.shape[-1])
        .contiguous()
    )
    return output, None
