"""`VoyageEmbedder`: the `Embedder` port over the `voyageai` SDK (design sections 3 and 13). Gateway only.

- `voyageai` is imported lazily. The SDK's own retries are off (`max_retries=0`); backoff lives in
  `retry.py` and sleeps through the injected `Clock`.
- Documents (`input_type="document"`): split into batches of <= 512 texts and <= 100,000 tokens by the
  real per-text token counts (`batching.plan_batches`), embedded in order, one SDK call per batch, up to 6
  retries per call with 1, 2, 4, 8, 16, 32 s backoff. Queries: SDK timeout 30 s, up to 3 retries (1, 2, 4 s).
- Every vector is checked: `len(vec) == dims` else INTERNAL with no retry (a wrong-dimension vector would
  corrupt the namespace), and the vector count must match the batch size.
- Terminal transport failure -> UPSTREAM_UNAVAILABLE. Key routing (`al-` Atlas keys vs `pa-` Voyage keys)
  is left to the SDK.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from scout.adapters.voyage.batching import BATCH_MAX_DOCS, BATCH_TOKEN_BUDGET, plan_batches
from scout.adapters.voyage.retry import Sleep, call_with_retry, to_scout_error
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.clock import Clock
from scout.core.ports.embedder import EmbedResult

__all__ = ["VoyageEmbedder"]

DEFAULT_MODEL = "voyage-code-4"
DOC_RETRIES = 6
QUERY_RETRIES = 3
DOC_TIMEOUT_S = 120.0
QUERY_TIMEOUT_S = 30.0


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        dims: int = 1024,
        model: str = DEFAULT_MODEL,
        clock: Clock | None = None,
        client: Any = None,
        query_client: Any = None,
        token_budget: int = BATCH_TOKEN_BUDGET,
        max_docs: int = BATCH_MAX_DOCS,
        doc_retries: int = DOC_RETRIES,
        query_retries: int = QUERY_RETRIES,
    ) -> None:
        """`client` / `query_client` inject `AsyncClient`-shaped objects (tests); `query_client` defaults to
        `client`. Otherwise two SDK clients are built lazily from `api_key` (documents and queries differ in
        timeout). `clock` supplies `sleep` (default `asyncio.sleep`)."""
        if dims < 1:
            raise ValueError("dims must be >= 1")
        self.dims = dims
        self.model = model
        self._api_key = api_key
        self._doc_client: Any = client
        self._query_client: Any = query_client if query_client is not None else client
        self._sleep: Sleep = clock.sleep if clock is not None else asyncio.sleep
        self._budget = token_budget
        self._max_docs = max_docs
        self._doc_retries = doc_retries
        self._query_retries = query_retries

    @classmethod
    def from_env(cls, env: Mapping[str, str], *, clock: Clock | None = None) -> "VoyageEmbedder":
        """`VOYAGE_API_KEY` (required), `VOYAGE_MODEL` (default `voyage-code-4`), `VOYAGE_DIMS` (1024)."""
        key = env.get("VOYAGE_API_KEY")
        if not key:
            raise ScoutError(ErrorCode.INTERNAL, "VOYAGE_API_KEY is not set")
        return cls(
            key,
            dims=int(env.get("VOYAGE_DIMS") or 1024),
            model=env.get("VOYAGE_MODEL") or DEFAULT_MODEL,
            clock=clock,
        )

    def _sdk(self, input_type: str) -> Any:
        if input_type == "query":
            if self._query_client is None:
                self._query_client = self._build(QUERY_TIMEOUT_S)
            return self._query_client
        if self._doc_client is None:
            self._doc_client = self._build(DOC_TIMEOUT_S)
        return self._doc_client

    def _build(self, timeout: float) -> Any:
        if not self._api_key:
            raise ScoutError(ErrorCode.INTERNAL, "Voyage API key is not configured")
        from voyageai.client_async import AsyncClient

        return AsyncClient(api_key=self._api_key, max_retries=0, timeout=timeout)

    async def count_tokens(self, texts: Sequence[str]) -> int:
        if not texts:
            return 0
        client = self._sdk("document")
        try:  # the SDK tokenizer is synchronous and may load vocabulary files: keep it off the loop
            return int(await asyncio.to_thread(client.count_tokens, list(texts), self.model))
        except Exception as exc:
            raise to_scout_error(exc) from None

    async def _counts(self, texts: Sequence[str]) -> list[int]:
        client = self._sdk("document")
        try:
            encoded = await asyncio.to_thread(client.tokenize, list(texts), self.model)
        except Exception as exc:
            raise to_scout_error(exc) from None
        counts = [len(e) for e in encoded]
        if len(counts) != len(texts):
            raise ScoutError(ErrorCode.INTERNAL, "tokenizer returned the wrong number of results")
        return counts

    async def embed(self, texts: Sequence[str], *, input_type: Literal["document", "query"]) -> EmbedResult:
        if input_type not in ("document", "query"):
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "input_type must be 'document' or 'query'")
        if not texts:
            return EmbedResult(vectors=[], tokens=0)
        if input_type == "query":
            return await self._embed_batch(list(texts), "query", self._query_retries)
        counts = await self._counts(texts)
        vectors: list[list[float]] = []
        tokens = 0
        for batch in plan_batches(counts, token_budget=self._budget, max_docs=self._max_docs):
            res = await self._embed_batch([texts[i] for i in batch], "document", self._doc_retries)
            vectors.extend(res.vectors)
            tokens += res.tokens
        return EmbedResult(vectors=vectors, tokens=tokens)

    async def _embed_batch(self, texts: list[str], input_type: str, retries: int) -> EmbedResult:
        client = self._sdk(input_type)

        async def call() -> Any:
            return await client.embed(
                texts, model=self.model, input_type=input_type, output_dimension=self.dims
            )

        res = await call_with_retry(call, retries=retries, sleep=self._sleep)
        vectors = [[float(x) for x in v] for v in res.embeddings]
        if len(vectors) != len(texts):
            raise ScoutError(ErrorCode.INTERNAL, "embedding count does not match the request")
        if any(len(v) != self.dims for v in vectors):
            raise ScoutError(ErrorCode.INTERNAL, "embedding dimension mismatch", retryable=False)
        return EmbedResult(vectors=vectors, tokens=int(res.total_tokens))
