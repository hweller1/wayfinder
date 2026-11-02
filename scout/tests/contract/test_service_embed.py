"""Task 3.3: `sync_nodes`, `sync_plan`, `embed` with the per-tenant embedding cache.

Covers FR-C11, FR-C12, AC-3.5 (server half: unchanged or moved chunks are never re-embedded), AC-3.6, AC-3.7
(insert-if-absent under concurrent embeds), PRD 4.3 items 4-5 and the 3.1 embed rate limit.
"""

import asyncio
import hashlib
from collections.abc import Sequence
from typing import Any, Literal

import pytest

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder, approx_tokens
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import EmbedReq, NodesReq, PlanReq, ReplaceFile, WireRow
from scout.core.crypto import ids
from scout.core.ports.embedder import EmbedResult
from scout.core.ports.store import Filter, Row
from scout.core.service.gateway import GatewayService, TenantGateway
from scout.core.service.ratelimit import Limits, RateLimiter
from scout.core.service.schema import ROOT_ID, TREE_SCHEMA
from scout.core.service.syncsvc import MAX_BATCH_TOKENS, cache_namespace

MASTER = b"m" * 32
REPO = "a" * 32
BRANCH = "c" * 16
Call = tuple[str, ...]

PARSE = "def parse_invoice_total(lines):\n    return sum(line.amount for line in lines)\n"
RENDER = "def render_receipt_header(order):\n    return f'Order {order.id}'\n"
AUTH = "def verify_session_token(token):\n    return token in ACTIVE\n"
FIXTURE_IDENT = "parse_invoice_total"


class RecordingStore(FakeStore):
    def __init__(self, *, yield_reads: bool = False) -> None:
        super().__init__()
        self.calls: list[Call] = []
        self.yield_reads = yield_reads
        self.before_delete: list[Any] = []

    def ops(self, *names: str) -> list[Call]:
        return [c for c in self.calls if c[0] in names]

    async def upsert(
        self, ns: str, rows: Sequence[Row], *, if_absent: bool = False, condition: Filter | None = None
    ) -> int:
        self.calls.append(("upsert", ns, "absent" if if_absent else "all"))
        assert all(r.vector is not None for r in rows) or ns.endswith("__tree")
        return await super().upsert(ns, rows, if_absent=if_absent, condition=condition)

    async def get_by_ids(self, ns: str, ids: Sequence[str], **kw: Any) -> list[Row]:
        self.calls.append(("get_by_ids", ns))
        rows = await super().get_by_ids(ns, ids, **kw)
        if self.yield_reads:
            await asyncio.sleep(0)  # every caller reads "absent" before any caller writes
        return rows

    async def delete_by_filter(self, ns: str, flt: Filter) -> int:
        self.calls.append(("delete_by_filter", ns))
        for hook in self.before_delete:
            await hook()
        return await super().delete_by_filter(ns, flt)

    async def ensure_namespace(self, ns: str, schema: Any) -> None:
        self.calls.append(("ensure_namespace", ns))
        await super().ensure_namespace(ns, schema)

    async def namespace_exists(self, ns: str) -> bool:
        self.calls.append(("namespace_exists", ns))
        return await super().namespace_exists(ns)


class DistinctEmbedder(FakeEmbedder):
    """Every embed call returns different vectors for the same text, and yields mid-call, so concurrent embeds
    of one key only agree if the cache's first write wins."""

    def __init__(self) -> None:
        super().__init__()
        self.serial = 0

    async def embed(self, texts: Sequence[str], *, input_type: Literal["document", "query"]) -> EmbedResult:
        await asyncio.sleep(0)
        self.serial += 1
        res = await super().embed(texts, input_type=input_type)
        return EmbedResult(vectors=[[v + self.serial for v in vec] for vec in res.vectors], tokens=res.tokens)


class SkewedEmbedder(FakeEmbedder):
    """`count_tokens` disagrees with character length: texts containing HEAVY cost 40K tokens each."""

    async def count_tokens(self, texts: Sequence[str]) -> int:
        self.count_calls += 1
        return sum(40_000 if "HEAVY" in t else 1 for t in texts)


def key_of(text: str) -> str:
    """Stand-in for `HMAC(K_chunk, body)[:32]` (the body, never the path)."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def file_of(path: str) -> str:
    return hashlib.sha256(b"file\0" + path.encode()).hexdigest()[:32]


def wire_row(path: str, text: str, file_hash: str, occurrence: int = 0) -> WireRow:
    fid, ck = file_of(path), key_of(text)
    return WireRow(
        id=hashlib.sha256(f"{fid}{ck}{occurrence}".encode()).hexdigest()[:32],
        file_id=fid,
        path_enc=f"enc/{fid[:8]}",
        ext_tag="e" * 16,
        dir_tags=["d" * 16],
        file_hash=file_hash,
        chunk_key=ck,
        loc_enc="bG9j",
        tokens=["t" * 32],
    )


def make(
    *, embedder: FakeEmbedder | None = None, store: RecordingStore | None = None, limits: Limits | None = None
) -> tuple[RecordingStore, FakeEmbedder, GatewayService]:
    store = store or RecordingStore()
    embedder = embedder or FakeEmbedder()
    clock = FakeClock()
    svc = GatewayService(store, clock, MASTER, embedder=embedder, limiter=RateLimiter(clock, limits))
    return store, embedder, svc


async def begun(svc: GatewayService, tenant: str = "acme", key_id: str = "k1") -> TenantGateway:
    from scout.core.contracts.wire import SyncBeginReq
    from scout.core.service.tenants import Tenant

    gw = svc.for_tenant(Tenant(tenant=tenant, key_id=key_id))
    await gw.sync_begin(SyncBeginReq(repo_id=REPO, branch_id=BRANCH, root="0" * 64, expected_chunks=0))
    return gw


def req(
    files: dict[str, tuple[str, list[str]]],
    texts: dict[str, str] | None = None,
    *,
    replace: bool = False,
    delete: Sequence[str] = (),
) -> EmbedReq:
    """`files`: path -> (file_hash, [chunk texts]); `texts`: chunk_key -> text sent for misses."""
    rows = [wire_row(p, t, h, i) for p, (h, chunks) in files.items() for i, t in enumerate(chunks)]
    reps = [ReplaceFile(file_id=file_of(p), file_hash=h) for p, (h, _) in files.items()] if replace else []
    return EmbedReq(
        repo_id=REPO,
        branch_id=BRANCH,
        texts=texts or {},
        rows=rows,
        replace=reps,
        delete_files=[file_of(p) for p in delete],
    )


async def sync_tree(gw: TenantGateway, files: dict[str, tuple[str, list[str]]], **kw: Any) -> Any:
    """The client's two steps: plan, then embed texts for exactly the missing keys."""
    by_key = {key_of(t): t for _, chunks in files.values() for t in chunks}
    plan = await gw.sync_plan(PlanReq(repo_id=REPO, branch_id=BRANCH, chunk_keys=list(by_key)))
    return await gw.embed(req(files, {k: by_key[k] for k in plan.missing}, **kw))


def chunk_rows(store: FakeStore, ns: str) -> dict[str, Any]:
    return store._ns[ns].rows  # pyright: ignore[reportPrivateUsage]


NS = ids.namespace("acme", REPO, BRANCH)
TREE = ids.tree_namespace(NS)
CACHE = cache_namespace("acme")
TREE_FILES = {"src/a.py": ("h-a", [PARSE, RENDER]), "src/b.py": ("h-b", [AUTH])}


# --- the core property: never re-embed ------------------------------------------------------------------


async def test_first_sync_embeds_each_distinct_chunk_once_and_writes_rows_with_vectors():
    """AC-3.5: baseline counts, and every written row carries the cached vector."""
    store, emb, svc = make()
    gw = await begun(svc)
    res = await sync_tree(gw, TREE_FILES)
    assert sorted(emb.embedded) == sorted([PARSE, RENDER, AUTH])
    assert [c.input_type for c in emb.calls] == ["document"]
    assert (res.written, res.embedded_tokens) == (3, emb.tokens)
    assert len(chunk_rows(store, CACHE)) == 3
    for row in await store.get_by_ids(NS, list(chunk_rows(store, NS)), include_vectors=True):
        cached = (await store.get_by_ids(CACHE, [row.attrs["chunk_key"]], include_vectors=True))[0]
        assert row.vector == cached.vector


async def test_resync_of_an_identical_tree_makes_zero_embed_calls():
    """AC-3.5: unchanged chunks are never re-embedded."""
    store, emb, svc = make()
    gw = await begun(svc)
    await sync_tree(gw, TREE_FILES)
    calls, count_calls = len(emb.calls), emb.count_calls
    plan = await gw.sync_plan(
        PlanReq(repo_id=REPO, branch_id=BRANCH, chunk_keys=[key_of(PARSE), key_of(AUTH)])
    )
    assert plan.missing == []
    res = await sync_tree(gw, TREE_FILES, replace=True)
    assert (len(emb.calls), emb.count_calls) == (calls, count_calls) == (1, emb.count_calls)
    assert (res.written, res.embedded_tokens) == (3, 0)
    assert len(chunk_rows(store, NS)) == 3


async def test_moving_a_file_makes_zero_embed_calls_and_reuses_cached_vectors():
    """AC-3.5: a moved file keeps its chunk keys; only its rows are rewritten under the new file id."""
    store, emb, svc = make()
    gw = await begun(svc)
    await sync_tree(gw, TREE_FILES)
    before = len(emb.calls)
    moved = {"lib/renamed.py": ("h-a", [PARSE, RENDER]), "src/b.py": TREE_FILES["src/b.py"]}
    res = await sync_tree(gw, moved, delete=["src/a.py"])
    assert len(emb.calls) == before
    assert res.embedded_tokens == 0
    files = {r.attrs["file_id"] for r in chunk_rows(store, NS).values()}
    assert files == {file_of("lib/renamed.py"), file_of("src/b.py")}
    assert len(chunk_rows(store, NS)) == 3
    assert len(chunk_rows(store, CACHE)) == 3  # no new cache rows either


async def test_editing_one_function_embeds_only_that_chunk():
    """AC-3.5: only the new body's key misses; its sibling and the other file are untouched."""
    store, emb, svc = make()
    gw = await begun(svc)
    await sync_tree(gw, TREE_FILES)
    before = len(emb.calls)
    edited = PARSE.replace("sum(", "max(")
    files = {"src/a.py": ("h-a2", [edited, RENDER]), "src/b.py": TREE_FILES["src/b.py"]}
    res = await sync_tree(gw, files, replace=True)
    assert [c.texts for c in emb.calls[before:]] == [(edited,)]
    assert res.embedded_tokens == approx_tokens(edited)
    assert len(chunk_rows(store, CACHE)) == 4
    a_rows = [r for r in chunk_rows(store, NS).values() if r.attrs["file_id"] == file_of("src/a.py")]
    assert {r.attrs["file_hash"] for r in a_rows} == {"h-a2"}  # old rows of the edited file are gone
    assert {r.attrs["chunk_key"] for r in a_rows} == {key_of(edited), key_of(RENDER)}
    b_rows = [r for r in chunk_rows(store, NS).values() if r.attrs["file_id"] == file_of("src/b.py")]
    assert len(b_rows) == 1


async def test_two_tenants_with_identical_code_each_embed_their_own_chunks():
    """The cache is per tenant: tenant B never hits tenant A's cache."""
    store, emb, svc = make()
    a = await begun(svc, "acme", "ka")
    await sync_tree(a, TREE_FILES)
    after_a = len(emb.calls)
    b = await begun(svc, "beta", "kb")
    plan = await b.sync_plan(
        PlanReq(repo_id=REPO, branch_id=BRANCH, chunk_keys=[key_of(PARSE), key_of(AUTH)])
    )
    assert sorted(plan.missing) == sorted([key_of(PARSE), key_of(AUTH)])
    await sync_tree(b, TREE_FILES)
    assert len(emb.calls) == after_a + 1
    assert sorted(emb.calls[-1].texts) == sorted([PARSE, RENDER, AUTH])
    assert len(chunk_rows(store, cache_namespace("acme"))) == 3
    assert len(chunk_rows(store, cache_namespace("beta"))) == 3
    assert cache_namespace("acme") != cache_namespace("beta")
    # and B's rows landed only in B's namespace
    assert len(chunk_rows(store, ids.namespace("beta", REPO, BRANCH))) == 3


async def test_text_sent_for_an_already_cached_key_is_not_embedded_again():
    store, emb, svc = make()
    gw = await begun(svc)
    await sync_tree(gw, TREE_FILES)
    before = len(emb.calls)
    keys = {key_of(PARSE): PARSE, key_of(AUTH): AUTH}  # a stale client sends text it did not need to
    res = await gw.embed(req(TREE_FILES, keys))
    assert len(emb.calls) == before
    assert res.embedded_tokens == 0


# --- concurrency (AC-3.7) -------------------------------------------------------------------------------


async def test_concurrent_embeds_of_the_same_key_leave_one_cache_row_and_identical_vectors():
    """AC-3.7: both callers miss, both embed, insert-if-absent keeps the first vector for everyone."""
    store = RecordingStore(yield_reads=True)
    emb = DistinctEmbedder()
    _, _, svc = make(embedder=emb, store=store)
    gw = await begun(svc)
    files_a = {"src/a.py": ("h", [PARSE])}
    files_b = {"src/copy.py": ("h", [PARSE])}
    texts = {key_of(PARSE): PARSE}
    await asyncio.gather(gw.embed(req(files_a, texts)), gw.embed(req(files_b, texts)))
    assert len(emb.calls) == 2  # the race really happened: both embedded
    assert len(chunk_rows(store, CACHE)) == 1
    rows = await store.get_by_ids(NS, list(chunk_rows(store, NS)), include_vectors=True)
    assert len(rows) == 2
    assert rows[0].vector == rows[1].vector
    cached = (await store.get_by_ids(CACHE, [key_of(PARSE)], include_vectors=True))[0]
    assert rows[0].vector == cached.vector
    assert ("upsert", CACHE, "absent") in store.calls
    assert ("upsert", CACHE, "all") not in store.calls


# --- plaintext, limits, batching ------------------------------------------------------------------------


async def test_chunk_text_is_never_written_to_any_store():
    """FR-X7: plaintext is in transit only; byte-scan every store row after an embed."""
    store, emb, svc = make()
    gw = await begun(svc)
    await sync_tree(gw, TREE_FILES)
    assert any(FIXTURE_IDENT in t for t in emb.embedded)  # positive control: the embedder saw it
    dump = repr(store._ns).encode()  # pyright: ignore[reportPrivateUsage]
    for needle in (FIXTURE_IDENT, "verify_session_token", "def ", "amount", "return"):
        assert needle.encode() not in dump
    assert store.ops("upsert")  # something was written, so the scan means something


async def test_embed_never_touches_the_tree_and_every_upserted_row_has_a_vector():
    store, _, svc = make()
    gw = await begun(svc)
    store.calls.clear()
    await sync_tree(gw, TREE_FILES, replace=True, delete=["src/gone.py"])
    assert {c[1] for c in store.calls} <= {NS, CACHE}
    assert not any(c[1].endswith("__tree") for c in store.calls)
    # the recording store itself asserts `vector is not None` on every non-tree upsert
    assert {c[1] for c in store.ops("upsert")} == {NS, CACHE}


async def test_embed_rate_limit_is_per_key_precharged_and_reconciled_with_real_tokens():
    limits = Limits(embed_tokens_per_min=10_000.0)
    store, emb, svc = make(limits=limits)
    gw = await begun(svc, "acme", "k1")
    other = await begun(svc, "acme", "k2")
    big = "x = 1  # " + "pad " * 1500  # ~6K chars -> ~1.7K tokens
    files = {"src/big.py": ("h", [big])}
    await gw.embed(req(files, {key_of(big): big}))
    bucket = svc._sync._limiter._embed_bucket("k1")  # pyright: ignore[reportPrivateUsage]
    assert bucket.tokens == pytest.approx(10_000 - emb.tokens)  # reconciled to the embedder's count
    # k1 drains its bucket; k2 is unaffected
    drain = "y = 2  # " + "word " * 4000
    calls = len(emb.calls)

    async def hammer() -> None:
        for i in range(10):
            t = f"{drain}{i}"
            await gw.embed(req({f"d{i}.py": ("h", [t])}, {key_of(t): t}))

    with pytest.raises(ScoutError) as ei:
        await hammer()
    assert ei.value.code == ErrorCode.RATE_LIMITED
    assert ei.value.details["retry_after"] > 0
    assert len(emb.calls) > calls
    n = len(emb.calls)
    fresh = "w = 3  # " + "word " * 4000 + "k2"
    await other.embed(req({"o.py": ("h", [fresh])}, {key_of(fresh): fresh}))  # k2 has its own bucket
    assert len(emb.calls) == n + 1


async def test_oversize_request_is_rejected_before_the_embedder_is_called():
    store, emb, svc = make(limits=Limits(embed_tokens_per_min=1_000.0))
    gw = await begun(svc)
    huge = "z = 0  # " + "q" * 5000  # ~1.4K tokens > the 1K bucket capacity
    with pytest.raises(ScoutError) as ei:
        await gw.embed(req({"h.py": ("h", [huge])}, {key_of(huge): huge}))
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert emb.calls == []
    assert emb.count_calls == 0
    assert chunk_rows(store, NS) == {}
    assert not await store.namespace_exists(CACHE)


async def test_failed_embed_refunds_the_precharge():
    class Boom(FakeEmbedder):
        async def embed(
            self, texts: Sequence[str], *, input_type: Literal["document", "query"]
        ) -> EmbedResult:
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "down", retryable=True)

    _, _, svc = make(embedder=Boom(), limits=Limits(embed_tokens_per_min=10_000.0))
    gw = await begun(svc)
    with pytest.raises(ScoutError) as ei:
        await gw.embed(req({"a.py": ("h", [PARSE])}, {key_of(PARSE): PARSE}))
    assert ei.value.code == ErrorCode.UPSTREAM_UNAVAILABLE
    assert svc._sync._limiter._embed_bucket("k1").tokens == pytest.approx(10_000.0)  # pyright: ignore[reportPrivateUsage]


async def test_batches_are_sized_by_count_tokens_not_characters():
    """Three short-by-chars texts that count 40K tokens each: 2 + 1, never one 120K request."""
    emb = SkewedEmbedder()
    _, _, svc = make(embedder=emb)
    gw = await begun(svc)
    texts = [f"HEAVY {i}" for i in range(3)]
    files = {"a.py": ("h", texts)}
    await gw.embed(req(files, {key_of(t): t for t in texts}))
    assert [len(c.texts) for c in emb.calls] == [2, 1]
    assert emb.count_calls >= 3
    # and many cheap texts (by count_tokens) that are long by characters stay in ONE batch
    emb2 = SkewedEmbedder()
    _, _, svc2 = make(embedder=emb2)
    gw2 = await begun(svc2)
    longs = [f"light {i} " + "c" * 20_000 for i in range(5)]
    await gw2.embed(req({"b.py": ("h", longs)}, {key_of(t): t for t in longs}))
    assert [len(c.texts) for c in emb2.calls] == [5]
    assert MAX_BATCH_TOKENS == 100_000


async def test_a_single_chunk_over_the_token_cap_is_rejected_before_any_embed_call():
    emb = SkewedEmbedder()
    _, _, svc = make(embedder=emb)
    gw = await begun(svc)

    class Over(SkewedEmbedder):
        async def count_tokens(self, texts: Sequence[str]) -> int:
            return MAX_BATCH_TOKENS + 1

    _, over, svc3 = make(embedder=Over())
    gw3 = await begun(svc3)
    with pytest.raises(ScoutError) as ei:
        await gw3.embed(req({"a.py": ("h", [PARSE])}, {key_of(PARSE): PARSE}))
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert over.calls == []
    assert gw is not None


# --- CONFLICT, replace ordering, idempotency, validation -----------------------------------------------


async def test_row_without_text_and_without_cache_entry_is_a_conflict_that_writes_nothing():
    """FR-C11: the client replans. Nothing is embedded or written."""
    store, emb, svc = make()
    gw = await begun(svc)
    await sync_tree(gw, {"src/b.py": TREE_FILES["src/b.py"]})
    store.calls.clear()
    before = len(emb.calls)
    files = {"src/b.py": ("h-b", [AUTH]), "src/c.py": ("h-c", [PARSE])}  # PARSE: no text, not cached
    with pytest.raises(ScoutError) as ei:
        await gw.embed(req(files, {}, replace=True))
    assert ei.value.code == ErrorCode.CONFLICT
    assert ei.value.retryable is True
    assert len(emb.calls) == before
    assert not store.ops("upsert", "delete_by_filter")
    assert len(chunk_rows(store, NS)) == 1


async def test_replace_deletes_old_rows_only_after_the_new_rows_are_written():
    """Design 14: upsert then delete_by_file, so a search never sees the file with no rows."""
    store, _, svc = make()
    gw = await begun(svc)
    await sync_tree(gw, {"src/a.py": ("h-old", [PARSE])})
    seen: list[set[str]] = []

    async def snapshot() -> None:
        seen.append({r.attrs["file_hash"] for r in chunk_rows(store, NS).values()})

    store.before_delete.append(snapshot)
    store.calls.clear()
    new = PARSE.replace("sum(", "min(")
    await sync_tree(gw, {"src/a.py": ("h-new", [new])}, replace=True)
    assert seen == [{"h-old", "h-new"}]  # at the delete, old and new coexist: no empty window
    ops = [c[0] for c in store.calls if c[1] == NS and c[0] in ("upsert", "delete_by_filter")]
    assert ops == ["upsert", "delete_by_filter"]
    assert {r.attrs["file_hash"] for r in chunk_rows(store, NS).values()} == {"h-new"}


async def test_delete_files_removes_every_row_of_a_removed_file():
    store, _, svc = make()
    gw = await begun(svc)
    await sync_tree(gw, TREE_FILES)
    await gw.embed(req({}, delete=["src/a.py"]))
    assert {r.attrs["file_id"] for r in chunk_rows(store, NS).values()} == {file_of("src/b.py")}


async def test_idempotency_key_replay_returns_the_stored_result_without_work():
    store, emb, svc = make()
    gw = await begun(svc)
    r = req(TREE_FILES, {key_of(t): t for _, ch in TREE_FILES.values() for t in ch})
    first = await gw.embed(r, idempotency_key="11111111-aaaa")
    n, upserts = len(emb.calls), len(store.ops("upsert"))
    again = await gw.embed(r, idempotency_key="11111111-aaaa")
    assert again == first
    assert (len(emb.calls), len(store.ops("upsert"))) == (n, upserts)
    with pytest.raises(ScoutError) as ei:  # same key, different body: a client bug, not a replay
        await gw.embed(req({"x.py": ("h", [AUTH])}, {}), idempotency_key="11111111-aaaa")
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError):
        await gw.embed(r, idempotency_key="k" * 200)
    # without a key nothing is replayed: the same body runs again, and still embeds nothing (cache)
    await gw.embed(r)
    assert len(emb.calls) == n


async def test_replay_expires_after_24_hours():
    store, emb, svc = make()
    gw = await begun(svc)
    r = req({"a.py": ("h", [PARSE])}, {key_of(PARSE): PARSE})
    await gw.embed(r, idempotency_key="idem-1")
    chunk_rows(store, NS).clear()  # an external delete: a live replay would NOT restore it
    await gw.embed(r, idempotency_key="idem-1")
    assert chunk_rows(store, NS) == {}
    svc._sync._replays._clock.advance(24 * 3600 + 1)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType]
    await gw.embed(r, idempotency_key="idem-1")
    assert len(chunk_rows(store, NS)) == 1


async def test_embed_into_a_branch_that_was_never_begun_is_not_found():
    store, emb, svc = make()
    gw = svc.for_tenant("acme")
    with pytest.raises(ScoutError) as ei:
        await gw.embed(req({"a.py": ("h", [PARSE])}, {key_of(PARSE): PARSE}))
    assert ei.value.code == ErrorCode.NOT_FOUND
    assert emb.calls == []


async def test_embed_without_a_configured_embedder_fails_closed():
    store = RecordingStore()
    svc = GatewayService(store, FakeClock(), MASTER)
    gw = await begun(svc)
    with pytest.raises(ScoutError) as ei:
        await gw.embed(req({"a.py": ("h", [PARSE])}, {key_of(PARSE): PARSE}))
    assert ei.value.code == ErrorCode.INTERNAL


# --- sync_plan ------------------------------------------------------------------------------------------


async def test_plan_lists_missing_keys_deduplicated_and_reads_only_the_tenant_cache():
    store, _, svc = make()
    gw = await begun(svc)
    keys = [key_of(PARSE), key_of(RENDER), key_of(PARSE)]
    store.calls.clear()
    first = await gw.sync_plan(PlanReq(repo_id=REPO, branch_id=BRANCH, chunk_keys=keys))  # cache absent
    assert first.missing == [key_of(PARSE), key_of(RENDER)]
    await sync_tree(gw, {"a.py": ("h", [PARSE])})
    store.calls.clear()
    second = await gw.sync_plan(PlanReq(repo_id=REPO, branch_id=BRANCH, chunk_keys=keys))
    assert second.missing == [key_of(RENDER)]
    assert {c[1] for c in store.calls} == {CACHE}


async def test_plan_rejects_bad_branch_ids_without_touching_the_store():
    store, _, svc = make()
    gw = svc.for_tenant("acme")
    with pytest.raises(ScoutError) as ei:
        await gw.sync_plan(PlanReq.model_construct(repo_id="Z" * 32, branch_id=BRANCH, chunk_keys=[]))
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert store.calls == []


# --- sync_nodes -----------------------------------------------------------------------------------------


async def seed_tree(store: FakeStore, tenant: str = "acme") -> None:
    tree = ids.tree_namespace(ids.namespace(tenant, REPO, BRANCH))
    await store.ensure_namespace(tree, TREE_SCHEMA)
    rows = [
        Row(id="1" * 32, attrs={"kind": "node", "children_enc": "YmxvYjE=", "node_hash": "n1"}),
        Row(id="2" * 32, attrs={"kind": "node", "children_enc": "YmxvYjI=", "node_hash": "n2"}),
        Row(id=ROOT_ID, attrs={"kind": "root", "root_hash": "r" * 64, "expected_chunks": 1}),
    ]
    await store.upsert(tree, rows)


async def test_nodes_returns_blobs_by_id_from_the_tree_only():
    store, _, svc = make()
    gw = await begun(svc)
    await seed_tree(store)
    store.calls.clear()
    res = await gw.sync_nodes(NodesReq(repo_id=REPO, branch_id=BRANCH, ids=["1" * 32, "2" * 32, "3" * 32]))
    assert res.blobs == {"1" * 32: "YmxvYjE=", "2" * 32: "YmxvYjI="}  # unknown id absent
    assert {c[1] for c in store.calls} == {TREE}
    root = NodesReq.model_construct(repo_id=REPO, branch_id=BRANCH, ids=[ROOT_ID])  # bypasses the hex pattern
    assert (await gw.sync_nodes(root)).blobs == {}  # the root row has no children_enc


async def test_nodes_for_an_id_that_exists_only_as_a_chunk_row_returns_nothing():
    store, emb, svc = make()
    gw = await begun(svc)
    await seed_tree(store)
    await sync_tree(gw, {"a.py": ("h", [PARSE])})
    chunk_id = next(iter(chunk_rows(store, NS)))
    assert chunk_id not in chunk_rows(store, TREE)
    res = await gw.sync_nodes(NodesReq(repo_id=REPO, branch_id=BRANCH, ids=[chunk_id]))
    assert res.blobs == {}
    # a row in the chunk namespace WITH a children_enc attribute is still invisible to sync_nodes
    await store.upsert(
        NS, [Row(id="4" * 32, attrs={"kind": "chunk", "children_enc": "bGVhaw=="}, vector=[0.1] * 1024)]
    )
    assert (await gw.sync_nodes(NodesReq(repo_id=REPO, branch_id=BRANCH, ids=["4" * 32]))).blobs == {}


async def test_nodes_are_isolated_per_tenant_and_a_missing_tree_is_empty():
    store, _, svc = make()
    await seed_tree(store, "acme")
    other = svc.for_tenant("beta")
    res = await other.sync_nodes(NodesReq(repo_id=REPO, branch_id=BRANCH, ids=["1" * 32]))
    assert res.blobs == {}
    empty = await svc.for_tenant("acme").sync_nodes(NodesReq(repo_id=REPO, branch_id=BRANCH, ids=[]))
    assert empty.blobs == {}
