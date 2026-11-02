"""FR-P1: FakeEmbedder (deterministic, unit norm, counters, limits) and FakeKeyStore (dict, no disk)."""

import builtins
import math
from pathlib import Path

import pytest

from scout.adapters.fake_embedder import MAX_BATCH_DOCS, MAX_BATCH_TOKENS, FakeEmbedder, approx_tokens
from scout.adapters.fake_keystore import FakeKeyStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.embedder import Embedder
from scout.core.ports.keystore import KeyStore

_check_embedder: Embedder = FakeEmbedder()
_check_keystore: KeyStore = FakeKeyStore()


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_deterministic_unit_norm_and_dims() -> None:
    for dims in (16, 1024):
        e = FakeEmbedder(dims=dims)
        r1 = await e.embed(["def getUserSession(): pass", ""], input_type="document")
        r2 = await FakeEmbedder(dims=dims).embed(["def getUserSession(): pass", ""], input_type="query")
        assert r1.vectors == r2.vectors
        assert all(len(v) == dims for v in r1.vectors)
        assert all(math.isclose(math.sqrt(_cos(v, v)), 1.0, rel_tol=1e-9) for v in r1.vectors)


async def test_order_preserved() -> None:
    e = FakeEmbedder(dims=64)
    texts = ["alpha_one", "beta_two", "gamma_three"]
    both = await e.embed(texts, input_type="document")
    for i, t in enumerate(texts):
        single = await e.embed([t], input_type="document")
        assert both.vectors[i] == single.vectors[0]
    rev = await e.embed(texts[::-1], input_type="document")
    assert rev.vectors == both.vectors[::-1]


async def test_shared_identifiers_rank_above_unrelated() -> None:
    e = FakeEmbedder()
    chunks = [
        "def refresh_auth_token(session): return session.token",
        "class InvoiceRenderer: def render_pdf(self): ...",
        "SELECT name FROM customers WHERE region = 'emea'",
    ]
    q = await e.embed(["where do we refreshAuthToken for a session"], input_type="query")
    docs = await e.embed(chunks, input_type="document")
    scores = [_cos(q.vectors[0], d) for d in docs.vectors]
    assert scores[0] == max(scores)
    assert scores[0] > scores[1] + 0.1
    assert scores[0] > scores[2] + 0.1


async def test_counters_and_input_type() -> None:
    e = FakeEmbedder(dims=16)
    assert await e.count_tokens(["abcdefg", "x"]) == math.ceil(7 / 3.5) + 1
    r = await e.embed(["abcdefg", "hello world"], input_type="document")
    await e.embed(["q"], input_type="query")
    assert r.tokens == approx_tokens("abcdefg") + approx_tokens("hello world")
    assert len(e.calls) == 2
    assert e.batches == 2
    assert [c.input_type for c in e.calls] == ["document", "query"]
    assert e.embedded == ["abcdefg", "hello world", "q"]
    assert e.tokens == r.tokens + approx_tokens("q")
    assert e.count_calls == 1


async def test_empty_input_is_a_noop_call() -> None:
    e = FakeEmbedder(dims=16)
    r = await e.embed([], input_type="document")
    assert r.vectors == []
    assert r.tokens == 0
    assert e.batches == 0


async def test_batching_limits_enforced() -> None:
    e = FakeEmbedder(dims=16)
    n = MAX_BATCH_DOCS * 2 + 1
    r = await e.embed(["a"] * n, input_type="document")
    assert len(r.vectors) == n
    assert e.batches == 3  # 512 + 512 + 1
    big = "x" * int(3.5 * (MAX_BATCH_TOKENS // 2 + 1))
    e2 = FakeEmbedder(dims=16)
    await e2.embed([big, big, big], input_type="document")
    assert e2.batches == 3  # two texts together exceed the token cap
    with pytest.raises(ScoutError) as ei:
        await e2.embed(["y" * int(3.5 * (MAX_BATCH_TOKENS + 1))], input_type="document")
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert e2.batches == 3  # rejected call is not counted


def test_keystore_roundtrip_and_no_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[object] = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda *a, **k: (opened.append(a), real_open(*a, **k))[1])
    monkeypatch.chdir(tmp_path)
    ks = FakeKeyStore()
    assert ks.get("scout_key") is None
    ks.set("scout_key", b"abc")
    ks.set("k_team:dev", b"\x00" * 64)
    assert ks.get("scout_key") == b"abc"
    assert ks.get("k_team:dev") == b"\x00" * 64
    ks.set("scout_key", b"def")
    assert ks.get("scout_key") == b"def"
    ks.delete("scout_key")
    ks.delete("scout_key")  # idempotent
    assert ks.get("scout_key") is None
    assert FakeKeyStore().get("k_team:dev") is None  # instances do not share state
    assert opened == []
    assert list(tmp_path.iterdir()) == []


def test_keystore_copies_mutable_values() -> None:
    ks = FakeKeyStore()
    buf = bytearray(b"abc")
    ks.set("scout_key", buf)  # type: ignore[arg-type]
    buf[0] = 0
    assert ks.get("scout_key") == b"abc"
