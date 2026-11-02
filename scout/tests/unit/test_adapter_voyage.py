"""Voyage adapter: batching, retry/backoff, dims assertion (task 7.2). FR-P6, AC-5.5, AC-3.7.

A stub SDK client stands in for `voyageai.AsyncClient`; failures are the REAL `voyageai.error` classes
(constructed offline). `FakeClock` records the backoff sleeps, so nothing waits. No network.
"""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import voyageai.error as verr

from scout.adapters.fake_clock import FakeClock
from scout.adapters.voyage import VoyageEmbedder
from scout.adapters.voyage.batching import BATCH_MAX_DOCS, BATCH_TOKEN_BUDGET, plan_batches
from scout.adapters.voyage.retry import backoff_delay, is_retryable
from scout.core.contracts.errors import ErrorCode, ScoutError

DIMS = 4


class StubVoyage:
    """`tokenize` returns a list of length `len(text.split())` per text (so tokens = words); `embed`
    replays `script` (exceptions raised, anything else ignored) then returns well-formed vectors."""

    def __init__(self, *, dims: int = DIMS, script: Sequence[Exception | None] = ()) -> None:
        self.dims = dims
        self.script = list(script)
        self.embed_calls: list[dict[str, Any]] = []
        self.count_calls = 0

    def tokenize(self, texts: list[str], model: str | None = None) -> list[list[int]]:
        return [[0] * len(t.split()) for t in texts]

    def count_tokens(self, texts: list[str], model: str | None = None) -> int:
        self.count_calls += 1
        return sum(len(t) for t in self.tokenize(texts, model))

    async def embed(self, texts: list[str], **kw: Any) -> Any:
        self.embed_calls.append({"texts": list(texts), **kw})
        if self.script:
            exc = self.script.pop(0)
            if exc is not None:
                raise exc
        vecs = [[float(len(t.split())), 1.0] + [0.0] * (self.dims - 2) for t in texts]
        return SimpleNamespace(embeddings=vecs, total_tokens=sum(len(t.split()) for t in texts))


def make(stub: StubVoyage, **kw: Any) -> tuple[VoyageEmbedder, FakeClock]:
    clock = FakeClock()
    return VoyageEmbedder(client=stub, dims=DIMS, clock=clock, **kw), clock


def rate_limit() -> Exception:
    return verr.RateLimitError("VENDOR-TEXT secret", http_status=429)


# --- batching (AC-3.7) --------------------------------------------------------------------------------
def test_plan_batches_splits_on_token_budget_and_doc_count() -> None:
    assert plan_batches([]) == []
    assert plan_batches([5, 5, 5]) == [range(0, 3)]
    assert plan_batches([60, 60, 60], token_budget=100) == [range(0, 1), range(1, 2), range(2, 3)]
    assert plan_batches([50, 50, 1], token_budget=100) == [range(0, 2), range(2, 3)]  # exactly 100 fits
    assert plan_batches([1] * 5, max_docs=2) == [range(0, 2), range(2, 4), range(4, 5)]
    assert plan_batches([500], token_budget=100) == [range(0, 1)]  # oversize single text: own batch
    assert plan_batches([1, 500, 1], token_budget=100) == [range(0, 1), range(1, 2), range(2, 3)]
    assert (BATCH_TOKEN_BUDGET, BATCH_MAX_DOCS) == (100_000, 512)


@pytest.mark.parametrize("n_docs", [1, 511, 512, 513, 1200])
def test_plan_batches_never_exceeds_512_docs_and_covers_everything(n_docs: int) -> None:
    batches = plan_batches([1] * n_docs)
    assert all(len(b) <= 512 for b in batches)
    assert [i for b in batches for i in b] == list(range(n_docs))


async def test_embed_splits_at_100k_tokens_by_real_counts_and_keeps_order() -> None:
    stub = StubVoyage()
    emb, _ = make(stub)
    texts = [("w " * 40_000).strip(), ("v " * 40_000).strip(), ("u " * 40_000).strip(), "x y"]
    res = await emb.embed(texts, input_type="document")
    sent = [len(c["texts"]) for c in stub.embed_calls]
    assert sent == [2, 2]  # 40k+40k fits, +40k does not; then 40k + 2
    for c in stub.embed_calls:
        assert sum(len(t.split()) for t in c["texts"]) <= 100_000
    assert [v[0] for v in res.vectors] == [40_000.0, 40_000.0, 40_000.0, 2.0]  # input order
    assert res.tokens == 120_002
    assert all(c["input_type"] == "document" and c["model"] == "voyage-code-4" for c in stub.embed_calls)
    assert all(c["output_dimension"] == DIMS for c in stub.embed_calls)


async def test_embed_splits_at_512_docs() -> None:
    stub = StubVoyage()
    emb, _ = make(stub)
    res = await emb.embed([f"t{i}" for i in range(1030)], input_type="document")
    assert [len(c["texts"]) for c in stub.embed_calls] == [512, 512, 6]
    assert len(res.vectors) == 1030


async def test_embed_empty_makes_no_call() -> None:
    stub = StubVoyage()
    emb, _ = make(stub)
    assert (await emb.embed([], input_type="document")).vectors == []
    assert stub.embed_calls == []
    assert await emb.count_tokens([]) == 0


async def test_count_tokens_delegates_to_the_sdk() -> None:
    stub = StubVoyage()
    emb, _ = make(stub)
    assert await emb.count_tokens(["a b c", "d e"]) == 5
    assert stub.count_calls == 1


async def test_invalid_input_type_rejected() -> None:
    emb, _ = make(StubVoyage())
    with pytest.raises(ScoutError) as ei:
        await emb.embed(["x"], input_type="other")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT


# --- retry (AC-5.5) -----------------------------------------------------------------------------------
def test_backoff_delay_is_min_2n_60() -> None:
    assert [backoff_delay(n) for n in range(10)] == [1, 2, 4, 8, 16, 32, 60, 60, 60, 60]


async def test_six_retries_sleep_1_2_4_8_16_32_then_succeed() -> None:
    stub = StubVoyage(script=[rate_limit()] * 6)
    emb, clock = make(stub)
    res = await emb.embed(["a b"], input_type="document")
    assert clock.sleeps == [1, 2, 4, 8, 16, 32]
    assert len(stub.embed_calls) == 7
    assert res.vectors[0][0] == 2.0


@pytest.mark.parametrize(
    "exc",
    [
        verr.RateLimitError("x", http_status=429),
        verr.Timeout("x"),
        verr.APIConnectionError("x"),
        verr.ServiceUnavailableError("x", http_status=503),
        verr.ServerError("x", http_status=500),
        verr.APIError("x", http_status=502),
        verr.TryAgain("x"),
        TimeoutError(),
        ConnectionResetError(),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_each_transient_failure_kind_is_retried_then_terminal_is_unavailable(exc: Exception) -> None:
    stub = StubVoyage(script=[exc] * 20)
    emb, clock = make(stub)
    with pytest.raises(ScoutError) as ei:
        await emb.embed(["a"], input_type="document")
    assert ei.value.code == ErrorCode.UPSTREAM_UNAVAILABLE
    assert ei.value.retryable
    assert "VENDOR-TEXT" not in str(ei.value)
    assert clock.sleeps == [1, 2, 4, 8, 16, 32]  # 6 retries, no sleep after the last failure
    assert len(stub.embed_calls) == 7


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (verr.AuthenticationError("VENDOR-TEXT", http_status=401), ErrorCode.INTERNAL),
        (verr.InvalidRequestError("VENDOR-TEXT", http_status=400), ErrorCode.INVALID_ARGUMENT),
        (verr.MalformedRequestError("VENDOR-TEXT", http_status=400), ErrorCode.INVALID_ARGUMENT),
        (RuntimeError("VENDOR-TEXT"), ErrorCode.INTERNAL),
    ],
    ids=lambda p: type(p).__name__ if isinstance(p, Exception) else str(p),
)
async def test_non_transient_failures_are_not_retried(exc: Exception, code: ErrorCode) -> None:
    stub = StubVoyage(script=[exc])
    emb, clock = make(stub)
    with pytest.raises(ScoutError) as ei:
        await emb.embed(["a"], input_type="document")
    assert ei.value.code == code
    assert "VENDOR-TEXT" not in str(ei.value)
    assert ei.value.__cause__ is None
    assert clock.sleeps == []
    assert len(stub.embed_calls) == 1


def test_is_retryable_classification() -> None:
    assert not is_retryable(ScoutError(ErrorCode.INTERNAL, "x"))
    assert not is_retryable(verr.APIError("x", http_status=400))
    assert is_retryable(verr.APIError("x"))  # no status: transport-level


async def test_query_path_retries_three_times_1_2_4() -> None:
    stub = StubVoyage(script=[rate_limit()] * 10)
    emb, clock = make(stub)
    with pytest.raises(ScoutError) as ei:
        await emb.embed(["find it"], input_type="query")
    assert ei.value.code == ErrorCode.UPSTREAM_UNAVAILABLE
    assert clock.sleeps == [1, 2, 4]
    assert len(stub.embed_calls) == 4
    assert stub.embed_calls[0]["input_type"] == "query"


async def test_query_path_recovers_and_skips_tokenizer() -> None:
    stub = StubVoyage(script=[verr.Timeout("x"), None])
    emb, clock = make(stub)
    res = await emb.embed(["q"], input_type="query")
    assert clock.sleeps == [1]
    assert len(res.vectors) == 1
    assert stub.count_calls == 0


def test_query_and_document_clients_are_separate_when_built_lazily() -> None:
    emb = VoyageEmbedder("pa-key", dims=DIMS)
    doc, query = emb._sdk("document"), emb._sdk("query")  # pyright: ignore[reportPrivateUsage]
    assert doc is not query
    assert emb._sdk("document") is doc  # pyright: ignore[reportPrivateUsage]
    assert type(doc).__name__ == "AsyncClient"


# --- dims assertion -----------------------------------------------------------------------------------
async def test_dims_mismatch_stops_immediately_with_no_retry_and_no_sleep() -> None:
    stub = StubVoyage(dims=DIMS + 1)
    emb, clock = make(stub)
    with pytest.raises(ScoutError) as ei:
        await emb.embed(["a", "b"], input_type="document")
    assert ei.value.code == ErrorCode.INTERNAL
    assert not ei.value.retryable
    assert clock.sleeps == []
    assert len(stub.embed_calls) == 1


async def test_dims_mismatch_in_a_later_batch_stops_the_remaining_batches() -> None:
    class Drift(StubVoyage):
        async def embed(self, texts: list[str], **kw: Any) -> Any:
            res = await super().embed(texts, **kw)
            if len(self.embed_calls) == 2:
                res.embeddings[0] = res.embeddings[0][:-1]
            return res

    stub = Drift()
    emb, _ = make(stub, max_docs=1)
    with pytest.raises(ScoutError) as ei:
        await emb.embed(["a", "b", "c", "d"], input_type="document")
    assert ei.value.code == ErrorCode.INTERNAL
    assert len(stub.embed_calls) == 2  # c and d were never sent


async def test_query_dims_mismatch_is_not_retried() -> None:
    stub = StubVoyage(dims=DIMS * 2)
    emb, clock = make(stub)
    with pytest.raises(ScoutError) as ei:
        await emb.embed(["q"], input_type="query")
    assert ei.value.code == ErrorCode.INTERNAL
    assert clock.sleeps == []
    assert len(stub.embed_calls) == 1


async def test_vector_count_mismatch_is_internal() -> None:
    class Short(StubVoyage):
        async def embed(self, texts: list[str], **kw: Any) -> Any:
            res = await super().embed(texts, **kw)
            res.embeddings.pop()
            return res

    emb, _ = make(Short())
    with pytest.raises(ScoutError) as ei:
        await emb.embed(["a", "b"], input_type="document")
    assert ei.value.code == ErrorCode.INTERNAL


# --- construction -------------------------------------------------------------------------------------
def test_from_env() -> None:
    with pytest.raises(ScoutError):
        VoyageEmbedder.from_env({})
    e = VoyageEmbedder.from_env({"VOYAGE_API_KEY": "k"})
    assert (e.dims, e.model) == (1024, "voyage-code-4")
    e2 = VoyageEmbedder.from_env({"VOYAGE_API_KEY": "k", "VOYAGE_DIMS": "256", "VOYAGE_MODEL": "m"})
    assert (e2.dims, e2.model) == (256, "m")
    with pytest.raises(ValueError, match="dims"):
        VoyageEmbedder("k", dims=0)
