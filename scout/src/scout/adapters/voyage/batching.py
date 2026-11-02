"""Token-budget batching for Voyage embedding requests (design section 13, AC-3.7)."""

from collections.abc import Sequence

__all__ = ["BATCH_MAX_DOCS", "BATCH_TOKEN_BUDGET", "plan_batches"]

BATCH_TOKEN_BUDGET = 100_000
BATCH_MAX_DOCS = 512


def plan_batches(
    counts: Sequence[int],
    *,
    token_budget: int = BATCH_TOKEN_BUDGET,
    max_docs: int = BATCH_MAX_DOCS,
) -> list[range]:
    """Split item indexes into consecutive batches of <= `max_docs` items and <= `token_budget` tokens
    (by the real per-text `counts`). Order is preserved, so concatenating the batches' results restores
    input order. A single text over the budget cannot be split: it gets a batch of its own (the API
    truncates it to the model's context length)."""
    batches: list[range] = []
    start = 0
    docs = tokens = 0
    for i, c in enumerate(counts):
        if docs and (docs >= max_docs or tokens + c > token_budget):
            batches.append(range(start, i))
            start, docs, tokens = i, 0, 0
        docs += 1
        tokens += c
    if docs:
        batches.append(range(start, len(counts)))
    return batches
