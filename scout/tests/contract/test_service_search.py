"""Task 3.5: `search` (hybrid ANN + BM25, RRF k=60, whitelisted filters, proof filter, overfetch, rate limit).

Covers FR-G1, FR-X4, FR-C17, AC-4.3 (first half: hits outside the proof set are dropped server-side) and
AC-9.3 (per-key search limit with Retry-After).
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import EmbedReq, SearchReq, SearchScope, SyncBeginReq, WireRow
from scout.core.crypto import ids
from scout.core.ports.store import And, Contains, Eq, Filter, Glob, Row
from scout.core.service.gateway import GatewayService, TenantGateway
from scout.core.service.ratelimit import Limits, RateLimiter
from scout.core.service.schema import META_SCHEMA
from scout.core.service.searchsvc import RRF_K, build_filter, rrf
from scout.core.service.syncsvc import meta_namespace, proofs_row_id
from scout.core.service.tenants import Tenant

MASTER = b"m" * 32
REPO = "a" * 32
BRANCH = "c" * 16
NS = ids.namespace("acme", REPO, BRANCH)
TREE = ids.tree_namespace(NS)

QUERY = "parse invoice total"
SECRET_IDENT = "parse_invoice_total_secret_name"  # noqa: S105
TOK_Q = "q" * 32  # a pre-hashed query token (the client sends HMAC output, never identifiers)
TOK_OTHER = "z" * 32

# (id, chunk text, tokens, file_hash). ANN order for QUERY is A, C, D, then B; BM25 for [TOK_Q] is B, C.
CHUNKS = {
    "A": ("parse invoice total", [TOK_OTHER], "h-a"),
    "C": ("parse invoice", [TOK_Q], "h-c"),
    "D": ("invoice", [TOK_OTHER], "h-d"),
    "B": ("render receipt header order", [TOK_Q], "h-b"),
}


class RecordingStore(FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def search_calls(self) -> list[tuple[str, str, dict[str, Any]]]:
        return [c for c in self.calls if c[0] in ("query_ann", "query_bm25")]

    async def query_ann(
        self, ns: str, vector: Sequence[float], *, top_k: int, flt: Filter | None = None, **kw: Any
    ) -> list[Row]:
        self.calls.append(("query_ann", ns, {"top_k": top_k, "flt": flt, "vector": list(vector)}))
        return await super().query_ann(ns, vector, top_k=top_k, flt=flt, **kw)

    async def query_bm25(
        self, ns: str, field: str, tokens: Sequence[str], *, top_k: int, flt: Filter | None = None, **kw: Any
    ) -> list[Row]:
        self.calls.append(("query_bm25", ns, {"top_k": top_k, "flt": flt, "tokens": list(tokens)}))
        return await super().query_bm25(ns, field, tokens, top_k=top_k, flt=flt, **kw)

    async def get_by_ids(self, ns: str, ids: Sequence[str], **kw: Any) -> list[Row]:
        self.calls.append(("get_by_ids", ns, {"ids": list(ids)}))
        return await super().get_by_ids(ns, ids, **kw)


def wire_row(name: str, text: str, tokens: list[str], file_hash: str, **over: Any) -> WireRow:
    fid = hashlib.sha256(f"file{name}".encode()).hexdigest()[:32]
    ck = hashlib.sha256(text.encode()).hexdigest()[:32]
    base: dict[str, Any] = {
        "id": hashlib.sha256(f"row{name}".encode()).hexdigest()[:32],
        "file_id": fid,
        "path_enc": f"enc/{name}.py",
        "ext_tag": "e" * 16,
        "dir_tags": ["d" * 16],
        "file_hash": file_hash,
        "chunk_key": ck,
        "loc_enc": "bG9j",
        "tokens": tokens,
    }
    return WireRow(**(base | over))


def make(
    limits: Limits | None = None,
) -> tuple[RecordingStore, FakeEmbedder, GatewayService, FakeClock]:
    store, embedder, clock = RecordingStore(), FakeEmbedder(), FakeClock()
    svc = GatewayService(store, clock, MASTER, embedder=embedder, limiter=RateLimiter(clock, limits))
    return store, embedder, svc, clock


async def populated(
    svc: GatewayService,
    chunks: dict[str, tuple[str, list[str], str]] | None = None,
    *,
    tenant: str = "acme",
    key_id: str = "k1",
    **row_over: Any,
) -> TenantGateway:
    gw = svc.for_tenant(Tenant(tenant=tenant, key_id=key_id))
    await gw.sync_begin(SyncBeginReq(repo_id=REPO, branch_id=BRANCH, root="0" * 64, expected_chunks=0))
    rows = [wire_row(n, t, tok, h, **row_over) for n, (t, tok, h) in (chunks or CHUNKS).items()]
    await gw.embed(
        EmbedReq(
            repo_id=REPO,
            branch_id=BRANCH,
            texts={r.chunk_key: t for r, (t, _, _) in zip(rows, (chunks or CHUNKS).values(), strict=True)},
            rows=rows,
        )
    )
    return gw


def search_req(**kw: Any) -> SearchReq:
    return SearchReq(repo_id=REPO, branch_id=BRANCH, **({"query": QUERY, "tokens": [TOK_Q]} | kw))


def row_id(name: str) -> str:
    return hashlib.sha256(f"row{name}".encode()).hexdigest()[:32]


def chunk_key(name: str) -> str:
    return hashlib.sha256(CHUNKS[name][0].encode()).hexdigest()[:32]


async def seed_proofs(store: FakeStore, leaves: list[str], *, required: bool = True) -> None:
    meta = meta_namespace("acme")
    await store.ensure_namespace(meta, META_SCHEMA)
    attrs = {"kind": "proofs", "leaves": leaves, "proofs_required": required}
    await store.upsert(meta, [Row(id=proofs_row_id(NS), attrs=attrs)])


# --- hybrid ranking -------------------------------------------------------------------------------------


async def test_hybrid_ranks_a_vector_only_and_a_token_only_match_into_top_k():
    """ANN and BM25 disagree: A is ANN-only, B is BM25-only; each alone misses the other."""
    store, _, svc, _ = make()
    gw = await populated(svc)
    ann = await store.query_ann(NS, FakeEmbedder().vector(QUERY), top_k=3)
    bm25 = await store.query_bm25(NS, "tokens", [TOK_Q], top_k=3)
    assert row_id("B") not in [r.id for r in ann]
    assert row_id("A") not in [r.id for r in bm25]
    res = await gw.search(search_req(k=3))
    got = [h.chunk_key for h in res.hits]
    assert chunk_key("C") == got[0]  # in both lists: highest fused score
    assert {chunk_key("A"), chunk_key("B")} <= set(got)  # one from each list survives the fusion
    # exact fused order: rank sums, ties by row id
    expected = rrf([[r.id for r in ann], [r.id for r in bm25]])
    by_row = {row_id(n): chunk_key(n) for n in CHUNKS}
    assert got == [by_row[i] for i, _ in expected][:3]
    assert [h.score for h in res.hits] == pytest.approx([s for _, s in expected][:3])


async def test_hybrid_is_deterministic_across_repeats():
    _, _, svc, _ = make()
    gw = await populated(svc)
    first = await gw.search(search_req(k=4))
    assert [await gw.search(search_req(k=4)) for _ in range(3)] == [first] * 3


def test_rrf_uses_k60_one_based_ranks_and_breaks_ties_by_id():
    assert RRF_K == 60
    fused = dict(rrf([["x", "y"], ["y", "z"]]))
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert [i for i, _ in rrf([["b"], ["a"]])] == ["a", "b"]  # equal scores: id order
    assert rrf([]) == []
    assert rrf([["a", "a"]]) == [("a", pytest.approx(1 / 61))]  # a repeat counts once


async def test_query_embeds_once_with_input_type_query_and_no_tokens_means_ann_only():
    store, emb, svc, _ = make()
    gw = await populated(svc)
    before = len(emb.calls)
    res = await gw.search(search_req(tokens=[]))
    assert [(c.input_type, c.texts) for c in emb.calls[before:]] == [("query", (QUERY,))]
    assert [c[0] for c in store.search_calls()] == ["query_ann"]
    assert res.hits[0].chunk_key == chunk_key("A")


async def test_search_before_sync_begin_is_not_found():
    _, _, svc, _ = make()
    gw = svc.for_tenant(Tenant(tenant="acme", key_id="k1"))
    with pytest.raises(ScoutError) as exc:
        await gw.search(search_req())
    assert exc.value.code is ErrorCode.NOT_FOUND


# --- whitelist -------------------------------------------------------------------------------------------


async def test_filter_is_built_only_from_the_whitelist_plus_forced_kind_chunk():
    store, _, svc, _ = make()
    gw = await populated(svc)
    scope = SearchScope(path_glob_enc="enc/*", ext_tag="e" * 16, dir_tag="d" * 16)
    await gw.search(search_req(scope=scope))
    expected = And(
        [
            Eq("kind", "chunk"),
            Glob("path_enc", "enc/*"),
            Eq("ext_tag", "e" * 16),
            Contains("dir_tags", "d" * 16),
        ]
    )
    assert [c[2]["flt"] for c in store.search_calls()] == [expected, expected]
    assert build_filter(SearchScope()) == And([Eq("kind", "chunk")])


def test_a_client_supplied_attribute_name_is_rejected_at_parse_time():
    for scope in ({"kind": "node"}, {"file_hash": "x"}, {"path_enc": "x"}, {"tenant": "other"}):
        with pytest.raises(ValidationError):
            SearchReq.model_validate({"repo_id": REPO, "branch_id": BRANCH, "query": "q", "scope": scope})
    for extra in ({"namespace": "scout_other_x_y"}, {"tenant": "other"}, {"filter": ["kind", "Eq", "node"]}):
        with pytest.raises(ValidationError):
            SearchReq.model_validate({"repo_id": REPO, "branch_id": BRANCH, "query": "q", **extra})


async def test_searches_touch_only_the_branch_chunk_namespace_never_the_tree_or_another_tenant():
    store, _, svc, _ = make()
    gw = await populated(svc)
    other = await populated(svc, tenant="bob", key_id="kb")
    store.calls.clear()
    await gw.search(search_req())
    await other.search(search_req())
    namespaces = {c[1] for c in store.search_calls()}
    assert namespaces == {NS, ids.namespace("bob", REPO, BRANCH)}
    assert all(not c[1].endswith("__tree") for c in store.calls)
    # a tenant that never synced this branch cannot reach acme's rows with the same ids
    eve = svc.for_tenant(Tenant(tenant="eve", key_id="ke"))
    with pytest.raises(ScoutError) as exc:
        await eve.search(search_req())
    assert exc.value.code is ErrorCode.NOT_FOUND


async def test_search_results_are_scoped_to_the_callers_tenant():
    _, _, svc, _ = make()
    acme = await populated(svc)
    bob = await populated(svc, {"X": ("only bobs chunk", [TOK_Q], "h-x")}, tenant="bob", key_id="kb")
    assert {h.file_hash for h in (await bob.search(search_req(k=5))).hits} == {"h-x"}
    assert "h-x" not in {h.file_hash for h in (await acme.search(search_req(k=5))).hits}


async def test_a_node_style_row_cannot_enter_the_chunk_namespace_so_ann_never_returns_it():
    store, _, svc, _ = make()
    await populated(svc)
    node = Row(id="node1", attrs={"kind": "node", "children_enc": "x", "node_hash": "y"})
    with pytest.raises(ScoutError) as exc:
        await store.upsert(NS, [node])  # vector-less row in a vector namespace: rejected (spike 0a-A)
    assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    assert "node1" not in store._ns[NS].rows  # pyright: ignore[reportPrivateUsage]
    ann = await store.query_ann(NS, FakeEmbedder().vector(QUERY), top_k=30)
    assert {r.attrs["kind"] for r in ann} == {"chunk"}


async def test_an_oversize_scope_value_is_invalid_argument_with_a_hint():
    _, _, svc, _ = make()
    gw = await populated(svc)
    with pytest.raises(ScoutError) as exc:
        await gw.search(search_req(scope=SearchScope(path_glob_enc="x" * 5000)))
    assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    assert "4 KiB" in exc.value.message
    assert "dir_tag" in exc.value.message


# --- proof filter (AC-4.3 first half) --------------------------------------------------------------------


async def test_hits_outside_the_proof_set_are_dropped_before_the_response():
    store, _, svc, _ = make()
    gw = await populated(svc)
    unfiltered = await gw.search(search_req(k=4))
    assert len(unfiltered.hits) == 4
    await seed_proofs(store, ["h-a", "h-c"])
    res = await gw.search(search_req(k=4))
    assert {h.file_hash for h in res.hits} == {"h-a", "h-c"}
    assert len(res.hits) == 2  # counts reflect the drop
    wire = res.model_dump_json()
    for name in ("B", "D"):
        for leaked in (chunk_key(name), f"enc/{name}.py", f"h-{name.lower()}"):
            assert leaked not in wire


async def test_proofs_not_required_or_absent_means_no_filtering_and_empty_proofs_return_nothing():
    store, _, svc, _ = make()
    gw = await populated(svc)
    await seed_proofs(store, ["h-a"], required=False)
    assert len((await gw.search(search_req(k=4))).hits) == 4
    await seed_proofs(store, [], required=True)  # fail closed: a seeded ns with no proofs shows nothing
    assert (await gw.search(search_req(k=4))).hits == []
    await store.delete_ids(meta_namespace("acme"), [proofs_row_id(NS)])  # dropped on commit
    assert len((await gw.search(search_req(k=4))).hits) == 4


async def test_proof_rows_of_another_namespace_do_not_filter_this_one():
    store, _, svc, _ = make()
    gw = await populated(svc)
    meta = meta_namespace("acme")
    await store.ensure_namespace(meta, META_SCHEMA)
    other = ids.namespace("acme", REPO, "d" * 16)
    attrs = {"kind": "proofs", "leaves": [], "proofs_required": True}
    await store.upsert(meta, [Row(id=proofs_row_id(other), attrs=attrs)])
    assert len((await gw.search(search_req(k=4))).hits) == 4


# --- overfetch ------------------------------------------------------------------------------------------


async def test_overfetch_fetches_k_times_overfetch_and_returns_that_many_for_the_client_post_filter():
    store, _, svc, _ = make()
    chunks = {f"n{i}": (f"parse invoice {i}", [TOK_Q], f"h{i}") for i in range(10)}
    gw = await populated(svc, chunks)
    store.calls.clear()
    res = await gw.search(search_req(k=2, overfetch=3))
    assert [c[2]["top_k"] for c in store.search_calls()] == [6, 6]
    assert len(res.hits) == 6  # the client post-filters these down to k
    store.calls.clear()
    assert len((await gw.search(search_req(k=2))).hits) == 2
    assert [c[2]["top_k"] for c in store.search_calls()] == [2, 2]


# --- privacy --------------------------------------------------------------------------------------------


async def test_no_plaintext_query_or_identifier_reaches_the_store():
    store, emb, svc, _ = make()
    gw = await populated(svc)
    store.calls.clear()
    query = f"how does {SECRET_IDENT} work"
    await gw.search(search_req(query=query))
    wire = json.dumps([(c[0], c[1], repr(c[2])) for c in store.calls]).encode()
    for secret in (query, SECRET_IDENT, "work", "how does"):
        assert secret.encode() not in wire
    stored = repr([(ns, repr(n.rows)) for ns, n in store._ns.items()]).encode()  # pyright: ignore[reportPrivateUsage]
    assert query.encode() not in stored
    assert SECRET_IDENT.encode() not in stored
    assert emb.calls[-1].texts == (query,)  # only the embedder sees the text


# --- rate limit -----------------------------------------------------------------------------------------


async def test_search_is_rate_limited_per_key_with_retry_after():
    _, _, svc, clock = make(Limits(search_per_s=10.0, search_burst=3.0))
    gw = await populated(svc)
    other = svc.for_tenant(Tenant(tenant="acme", key_id="k2"))
    for _ in range(3):
        await gw.search(search_req())
    with pytest.raises(ScoutError) as exc:
        await gw.search(search_req())
    assert exc.value.code is ErrorCode.RATE_LIMITED
    assert exc.value.details["retry_after"] == pytest.approx(0.1)
    await other.search(search_req())  # another key has its own bucket
    clock.advance(0.1)
    await gw.search(search_req())  # refilled at 10 req/s
