"""`FakeEmbedder`: deterministic in-memory `Embedder` for tests (design section 3, Testability).

Vectors are a hashed bag of identifier tokens (whole lowercase word plus camel/snake/digit sub-tokens, each
mapped to a signed bucket by SHA-256), L2-normalised. Texts that share identifiers therefore have a higher
cosine than unrelated ones, which lets retrieval tests tell relevant from irrelevant chunks. No network, no
randomness, no wall clock.

Limits mirror the port docstring: the adapter owns batching (<= 512 docs and <= 100K tokens per upstream
request); a single text above the token cap cannot be batched and is rejected (INVALID_ARGUMENT). Counters
(`calls`, `batches`, `tokens`, `embedded`) let sync tests assert that unchanged chunks are never re-embedded.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.embedder import EmbedResult

MAX_BATCH_DOCS = 512
MAX_BATCH_TOKENS = 100_000

_WORD = re.compile(r"[A-Za-z0-9_]+")
_PIECE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


def approx_tokens(text: str) -> int:
    """`ceil(chars / 3.5)`, as specified for the fake."""
    return math.ceil(len(text) / 3.5)


def _terms(text: str) -> list[str]:
    out: list[str] = []
    for word in _WORD.findall(text):
        out.append(word.lower())
        pieces = [p.lower() for chunk in word.split("_") for p in _PIECE.findall(chunk)]
        if len(pieces) > 1:
            out.extend(pieces)
    return out


def _bucket(term: str, dims: int) -> tuple[int, float]:
    digest = hashlib.sha256(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dims, 1.0 if digest[8] & 1 else -1.0


@dataclass(frozen=True, slots=True)
class EmbedCall:
    input_type: Literal["document", "query"]
    texts: tuple[str, ...]
    batches: int


class FakeEmbedder:
    def __init__(self, dims: int = 1024, model: str = "fake-embed") -> None:
        if dims < 1:
            raise ValueError("dims must be >= 1")
        self.dims = dims
        self.model = model
        self.calls: list[EmbedCall] = []
        self.batches = 0
        self.tokens = 0
        self.count_calls = 0

    @property
    def embedded(self) -> list[str]:
        """Every text embedded so far, in order (duplicates preserved)."""
        return [t for call in self.calls for t in call.texts]

    def vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for term in _terms(text):
            idx, sign = _bucket(term, self.dims)
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:  # empty text or exact cancellation: fall back to a hash of the raw text
            idx, _ = _bucket("\0" + text, self.dims)
            vec[idx] = 1.0
            return vec
        return [x / norm for x in vec]

    async def count_tokens(self, texts: Sequence[str]) -> int:
        self.count_calls += 1
        return sum(approx_tokens(t) for t in texts)

    async def embed(self, texts: Sequence[str], *, input_type: Literal["document", "query"]) -> EmbedResult:
        if input_type not in ("document", "query"):
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "input_type must be 'document' or 'query'")
        counts = [approx_tokens(t) for t in texts]
        if any(c > MAX_BATCH_TOKENS for c in counts):
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "a single text exceeds the per-request token cap")
        batches = 0
        docs = toks = 0
        for c in counts:
            if docs == 0 or (docs < MAX_BATCH_DOCS and toks + c <= MAX_BATCH_TOKENS):
                docs += 1
                toks += c
            else:
                batches += 1
                docs, toks = 1, c
        batches += 1 if docs else 0
        vectors = [self.vector(t) for t in texts]
        for v in vectors:
            if len(v) != self.dims:  # the real adapter asserts this; keep the fake honest
                raise ScoutError(ErrorCode.INTERNAL, "embedding dimension mismatch", retryable=False)
        total = sum(counts)
        self.calls.append(EmbedCall(input_type, tuple(texts), batches))
        self.batches += batches
        self.tokens += total
        return EmbedResult(vectors=vectors, tokens=total)
