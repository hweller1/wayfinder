"""Embedder port (design section 3). GATEWAY side only."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class EmbedResult:
    vectors: list[list[float]]
    tokens: int


class Embedder(Protocol):
    dims: int
    model: str

    async def count_tokens(self, texts: Sequence[str]) -> int: ...

    async def embed(self, texts: Sequence[str], *, input_type: Literal["document", "query"]) -> EmbedResult:
        """Returns vectors in input order. Adapter owns batching (<=100K real tokens by count_tokens,
        <=512 docs), 6 attempts, min(2**n,60) s backoff on 429/timeout/50x, and asserts len(vec)==dims
        (mismatch -> INTERNAL, no retry). Terminal failure -> UPSTREAM_UNAVAILABLE (retryable)."""
        ...
