"""Pipeline-only combination guards."""

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.parallel.compatibility import (
    chunked_loss_fsdp as chunked_loss_fsdp,
)
from llmtuner.parallel.compatibility import (
    ep_hf_initial_load as ep_hf_initial_load,
)


def chunked_loss_pp(chunks: int, pp: int) -> None:
    """Under PP the last stage's loss runs inside the schedule on materialized logits;
    rewiring that seam for hidden states plus a per-chunk backward is a PP-side
    change.
    """
    raise UnsupportedCombinationError(
        f"chunked_loss_num_chunks={chunks} with "
        f"pipeline_parallel_size={pp} is not "
        "supported: the pipeline last stage's loss runs inside the "
        "schedule on materialized logits. Run chunked loss without "
        "pipeline parallelism."
    )


def pp_packed_microbatch_split(num_microbatches: int) -> None:
    """A packed (1-D) corpus batch cannot be split into PP microbatches: the
    split would cut documents at the token level, silently truncating their
    context. Row-batched corpora (synthetic rows, multimodal rows) split by
    rows and are unaffected.
    """
    raise UnsupportedCombinationError(
        f"pp > 1 with num_pp_microbatches={num_microbatches} > 1 over a packed "
        "corpus is not supported: the batch is a flat token stream and the "
        "microbatch split would cut documents mid-context. Use the configured "
        "text loader, which independently packs each pipeline microbatch, "
        "provide a row-batched corpus, or set num_pp_microbatches=1."
    )
