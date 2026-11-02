"""Backend-agnostic `VectorStore` contract suite (design section 3, AC-5.3, FR-P3, FR-X4).

Runs unchanged against `FakeStore` (tests/contract/test_fake_store_suite.py) and, in task 7.10, against the
real turbopuffer adapter. Rules that keep it portable:
- Imports only the port (`scout.core.ports.store`) and `ScoutError`/`ErrorCode`; never a fake or an adapter.
- Every case is `async def test_*(store: VectorStore, ns_prefix: str)`; namespaces are named
  `<ns_prefix><suffix>` and deleted in a `finally` (the live runner passes `scout_dev_...` prefixes).
- Only behaviour the spike (artifacts/spike/0a_tpuf.json) verified or the port docstrings guarantee is
  asserted. Unspecified details are tolerant: ANN/BM25 ties and score magnitudes are never compared
  exactly (sets, ranking prefixes with well-separated inputs, monotone scores). Client-side-only limits
  (4 KiB filter values: the real service accepted 1 MiB) and fake-only strictness (Glob on a non-glob
  attribute) are NOT here; they live in test_fake_store_suite.py.
- `ALL_CASES` lists every case so a runner can parametrize without pytest collecting this module.
"""

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

import pytest

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.store import (
    And,
    AttrSpec,
    Contains,
    Eq,
    Filter,
    Glob,
    In,
    NamespaceSchema,
    Not,
    Row,
    VectorStore,
)

ATTRS = {
    "kind": AttrSpec("string"),
    "file_id": AttrSpec("string"),
    "file_hash": AttrSpec("string"),
    "path_enc": AttrSpec("string", glob=True),
    "dir_tags": AttrSpec("[]string"),
    "tokens": AttrSpec("[]string", filterable=False, fts="pre_tokenized_array"),
    "note": AttrSpec("string", filterable=False),
}
VEC = NamespaceSchema(dims=3, attrs=ATTRS)
TREE = NamespaceSchema(
    dims=None,
    attrs={"kind": AttrSpec("string"), "gen": AttrSpec("int"), "expected_chunks": AttrSpec("int")},
)
TOK_A = "a" * 64  # spike c: 64-char pre-hashed tokens (longer than the default tokenizer limit)
TOK_B = "b" * 64
TOK_C = "c" * 64

type Case = Callable[[VectorStore, str], Awaitable[None]]


def chunk(i: str, vec: Sequence[float] = (1.0, 0.0, 0.0), **attrs: object) -> Row:
    return Row(id=i, attrs={"kind": "chunk", **attrs}, vector=list(vec))


def node(i: str, **attrs: object) -> Row:
    return Row(id=i, attrs=dict(attrs))


def ids(rows: Sequence[Row]) -> list[str]:
    return [r.id for r in rows]


async def code_of(coro: Awaitable[object]) -> ErrorCode:
    with pytest.raises(ScoutError) as ei:
        await coro
    return ei.value.code


@asynccontextmanager
async def namespaces(store: VectorStore, ns_prefix: str) -> AsyncIterator[Callable[[str], str]]:
    """Yields `name(suffix)`; every name handed out (and its `__tree` sibling) is deleted on exit."""
    used: list[str] = []

    def name(suffix: str) -> str:
        n = f"{ns_prefix}{suffix}"
        used.append(n)
        return n

    try:
        yield name
    finally:
        for n in used:
            await store.delete_namespace(n)


async def seeded(store: VectorStore, ns: str) -> None:
    """Chunk namespace with well-separated vectors (cosine to (1,0,0): c1 1.0, c2 0.8, c3 0, c4 0)."""
    await store.ensure_namespace(ns, VEC)
    await store.upsert(
        ns,
        [
            chunk(
                "c1",
                (1.0, 0.0, 0.0),
                file_id="F1",
                file_hash="H1",
                path_enc="aa11/bb22/cc33",
                dir_tags=["d1", "d2"],
                tokens=[TOK_A, TOK_B],
                note="n1",
            ),
            chunk(
                "c2",
                (0.8, 0.6, 0.0),
                file_id="F1",
                file_hash="H1",
                path_enc="aa11/bb22/dd44",
                dir_tags=["d1", "d2"],
                tokens=[TOK_A],
                note="n2",
            ),
            chunk(
                "c3",
                (0.0, 1.0, 0.0),
                file_id="F2",
                file_hash="H2",
                path_enc="ee55/ff66",
                dir_tags=["d3"],
                tokens=[TOK_A, TOK_C],
                note="n3",
            ),
            chunk(
                "c4",
                (0.0, 0.0, 1.0),
                file_id="F3",
                file_hash="H3",
                path_enc="aa11/zz99",
                dir_tags=["d1"],
                tokens=[TOK_C],
                note="n4",
            ),
        ],
    )


async def ann_ids(store: VectorStore, ns: str, flt: Filter | None) -> set[str]:
    return set(ids(await store.query_ann(ns, [1.0, 0.0, 0.0], top_k=100, flt=flt)))


# --- namespaces ---------------------------------------------------------------------------------------
async def test_ensure_namespace_idempotent_and_mismatch(store: VectorStore, ns_prefix: str) -> None:
    """Port: idempotent create; existing ns with a different schema -> INTERNAL."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("ens")
        assert not await store.namespace_exists(ns)
        await store.ensure_namespace(ns, VEC)
        await store.ensure_namespace(ns, VEC)
        assert await store.namespace_exists(ns)
        assert await code_of(store.ensure_namespace(ns, TREE)) == ErrorCode.INTERNAL
        assert await code_of(store.ensure_namespace(ns, NamespaceSchema(dims=4, attrs=ATTRS))) == (
            ErrorCode.INTERNAL
        )


async def test_list_and_delete_namespace(store: VectorStore, ns_prefix: str) -> None:
    """Port: list_namespaces(prefix) filters by prefix; delete_namespace is idempotent."""
    async with namespaces(store, ns_prefix) as name:
        a, b = name("list_a"), name("list_b")
        await store.ensure_namespace(a, VEC)
        await store.ensure_namespace(b, TREE)
        listed = await store.list_namespaces(ns_prefix)
        assert {a, b} <= set(listed)
        assert all(n.startswith(ns_prefix) for n in listed)
        assert a in await store.list_namespaces(a)
        await store.delete_namespace(a)
        await store.delete_namespace(a)
        assert not await store.namespace_exists(a)
        assert a not in await store.list_namespaces(ns_prefix)
        await store.delete_namespace(name("never_created"))


# --- writes and get_by_ids ----------------------------------------------------------------------------
async def test_upsert_and_get_by_ids(store: VectorStore, ns_prefix: str) -> None:
    """Port: missing ids omitted; attrs projection; include_vectors; upsert overwrites and counts rows."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("rows")
        await store.ensure_namespace(ns, VEC)
        assert await store.upsert(ns, [chunk("c1", file_id="F1", note="x"), chunk("c2", file_id="F1")]) == 2
        got = {r.id: r for r in await store.get_by_ids(ns, ["c1", "c2", "ghost"])}
        assert set(got) == {"c1", "c2"}
        assert got["c1"].attrs["file_id"] == "F1"
        assert got["c1"].vector is None  # include_vectors defaults to False
        only = await store.get_by_ids(ns, ["c1"], attrs=["file_id"])
        assert set(only[0].attrs) <= {"file_id"}
        assert only[0].attrs["file_id"] == "F1"
        withvec = (await store.get_by_ids(ns, ["c1"], include_vectors=True))[0]
        assert withvec.vector is not None
        assert [round(x, 4) for x in withvec.vector] == [1.0, 0.0, 0.0]
        assert await store.upsert(ns, [chunk("c1", file_id="F9")]) == 1
        assert (await store.get_by_ids(ns, ["c1"]))[0].attrs["file_id"] == "F9"
        assert await store.get_by_ids(ns, []) == []


async def test_if_absent_upsert(store: VectorStore, ns_prefix: str) -> None:
    """Spike f: insert-if-absent returns the count actually inserted and keeps the first value."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("absent")
        await store.ensure_namespace(ns, VEC)
        assert await store.upsert(ns, [chunk("k1", note="cache")], if_absent=True) == 1
        assert await store.upsert(ns, [chunk("k1", note="other")], if_absent=True) == 0
        assert (await store.get_by_ids(ns, ["k1"]))[0].attrs["note"] == "cache"
        assert await store.upsert(ns, [chunk("k1", note="a"), chunk("k2", note="b")], if_absent=True) == 1


async def test_conditional_upsert(store: VectorStore, ns_prefix: str) -> None:
    """Port `condition` (turbopuffer upsert_condition, evaluated against the EXISTING row): passes and
    overwrites, fails and skips (not counted), and a missing row passes (docs; basis of the root CAS)."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("cond")
        await store.ensure_namespace(ns, VEC)
        base = Eq("file_hash", "h1")
        assert (
            await store.upsert(ns, [chunk("r", file_hash="h1", note="v1")], condition=base) == 1
        )  # missing row
        assert (await store.get_by_ids(ns, ["r"]))[0].attrs["note"] == "v1"
        assert await store.upsert(ns, [chunk("r", file_hash="h2", note="v2")], condition=base) == 1  # passes
        assert (await store.get_by_ids(ns, ["r"]))[0].attrs["file_hash"] == "h2"
        assert (
            await store.upsert(ns, [chunk("r", file_hash="h3", note="v3")], condition=base) == 0
        )  # now fails
        got = (await store.get_by_ids(ns, ["r"]))[0].attrs
        assert (got["file_hash"], got["note"]) == ("h2", "v2")  # a failed condition changes nothing
        mixed = [chunk("r", file_hash="h4", note="v4"), chunk("fresh", file_hash="h9", note="n")]
        assert (
            await store.upsert(ns, mixed, condition=Eq("file_hash", "h2")) == 2
        )  # per row: existing + missing
        assert await store.upsert(ns, mixed, condition=Eq("file_hash", "h2")) == 0  # r is h4, fresh is h9
        assert [r.id for r in await store.get_by_ids(ns, ["r", "fresh"])] == ["r", "fresh"]


async def test_conditional_upsert_on_vectorless_namespace_is_a_cas(
    store: VectorStore, ns_prefix: str
) -> None:
    """The root-row shape: vector-less row, condition on the stored value, only one of N racers wins."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("cas")
        await store.ensure_namespace(ns, TREE)
        assert await store.upsert(ns, [node("root", kind="root", gen=1)], if_absent=True) == 1
        racers = [node("root", kind="root", gen=2 + i) for i in range(3)]
        wins = [await store.upsert(ns, [r], condition=Eq("gen", 1)) for r in racers]
        assert wins == [1, 0, 0]  # the first moved gen off 1; the others lost
        assert (await store.get_by_ids(ns, ["root"]))[0].attrs["gen"] == 2


async def test_delete_ids(store: VectorStore, ns_prefix: str) -> None:
    """Port: delete_ids returns rows deleted; unknown ids are not counted."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("delids")
        await seeded(store, ns)
        assert await store.delete_ids(ns, ["c1", "c2", "ghost"]) == 2
        assert set(ids(await store.get_by_ids(ns, ["c1", "c2", "c3", "c4"]))) == {"c3", "c4"}
        assert await store.delete_ids(ns, ["c1"]) == 0


async def test_delete_by_file_and_missing_ns(store: VectorStore, ns_prefix: str) -> None:
    """Port: delete_by_file(file_id==X [AND file_hash!=except]) returns rows deleted; missing ns -> 0.
    Spike d: Not(Eq) inside And is accepted."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("delfile")
        await seeded(store, ns)
        await store.upsert(ns, [chunk("c5", file_id="F1", file_hash="H1b", dir_tags=["d1"])])
        assert await store.delete_by_file(ns, "F1", except_file_hash="H1b") == 2  # c1, c2 stale
        assert set(ids(await store.get_by_ids(ns, ["c1", "c2", "c3", "c4", "c5"]))) == {"c3", "c4", "c5"}
        assert await store.delete_by_file(ns, "F2") == 1
        assert await store.delete_by_file(ns, "F2") == 0
        assert await store.delete_by_file(name("no_such_ns"), "F1") == 0


async def test_delete_by_filter(store: VectorStore, ns_prefix: str) -> None:
    """Port: delete_by_filter over the Filter AST returns rows deleted."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("delflt")
        await seeded(store, ns)
        assert await store.delete_by_filter(ns, Contains("dir_tags", "d3")) == 1
        assert await store.delete_by_filter(ns, And([Eq("file_id", "F1"), Not(Eq("file_hash", "H1"))])) == 0
        assert await store.delete_by_filter(ns, Eq("file_id", "F1")) == 2
        assert set(ids(await store.get_by_ids(ns, ["c1", "c2", "c3", "c4"]))) == {"c4"}


# --- filters (observed through ANN, which every real search uses) -------------------------------------
async def test_filters_eq_in_contains_and_not(store: VectorStore, ns_prefix: str) -> None:
    """Spike c/d: Eq, In, Contains on []string, And, Not(Eq) inside And."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("flt")
        await seeded(store, ns)
        assert await ann_ids(store, ns, Eq("file_id", "F1")) == {"c1", "c2"}
        assert await ann_ids(store, ns, In("file_id", ["F2", "F3"])) == {"c3", "c4"}
        assert await ann_ids(store, ns, Contains("dir_tags", "d2")) == {"c1", "c2"}
        assert await ann_ids(store, ns, And([Contains("dir_tags", "d1"), Eq("kind", "chunk")])) == {
            "c1",
            "c2",
            "c4",
        }
        assert await ann_ids(store, ns, And([Eq("file_id", "F1"), Not(Eq("file_hash", "H9"))])) == {
            "c1",
            "c2",
        }
        assert await ann_ids(store, ns, And([Contains("dir_tags", "d1"), Not(Eq("file_id", "F1"))])) == {"c4"}
        assert await ann_ids(store, ns, Eq("file_id", "nope")) == set()


async def test_glob_whole_segment_prefix(store: VectorStore, ns_prefix: str) -> None:
    """Spike e: Glob on a glob=True attr with whole-segment prefixes (`aa11/**`, `aa11/bb22/*`, exact)."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("glob")
        await seeded(store, ns)
        assert await ann_ids(store, ns, Glob("path_enc", "aa11/**")) == {"c1", "c2", "c4"}
        assert await ann_ids(store, ns, Glob("path_enc", "aa11/bb22/*")) == {"c1", "c2"}
        assert await ann_ids(store, ns, Glob("path_enc", "ee55/ff66")) == {"c3"}
        assert await ann_ids(store, ns, Glob("path_enc", "zz00/**")) == set()
        assert await ann_ids(store, ns, Glob("path_enc", "**33")) == {"c1"}  # spike e: in-segment suffix
        both = And([Glob("path_enc", "aa11/**"), Not(Eq("file_id", "F1"))])
        assert await ann_ids(store, ns, both) == {"c4"}


# --- ANN ----------------------------------------------------------------------------------------------
async def test_query_ann_ranking_topk_filter_attrs(store: VectorStore, ns_prefix: str) -> None:
    """Port: cosine ANN, top_k, filter, attrs projection, score set with higher = better.
    Tolerant: only the well-separated head is ranked; tie order and score magnitudes are not asserted."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("ann")
        await seeded(store, ns)
        top2 = await store.query_ann(ns, [1.0, 0.0, 0.0], top_k=2)
        assert ids(top2) == ["c1", "c2"]
        scores = [r.score for r in top2]
        assert all(s is not None for s in scores)
        assert scores[0] is not None
        assert scores[1] is not None
        assert scores[0] >= scores[1]
        assert len(await store.query_ann(ns, [1.0, 0.0, 0.0], top_k=1)) == 1
        assert set(ids(await store.query_ann(ns, [1.0, 0.0, 0.0], top_k=100))) == {"c1", "c2", "c3", "c4"}
        filtered = await store.query_ann(ns, [1.0, 0.0, 0.0], top_k=1, flt=Not(Eq("file_id", "F1")))
        assert ids(filtered)[0] in {"c3", "c4"}
        near_c3 = await store.query_ann(ns, [0.0, 1.0, 0.0], top_k=2)
        assert near_c3[0].id == "c3"
        assert near_c3[1].id == "c2"
        proj = await store.query_ann(ns, [1.0, 0.0, 0.0], top_k=1, attrs=["file_id"])
        assert set(proj[0].attrs) <= {"file_id"}
        assert proj[0].attrs["file_id"] == "F1"
        assert await store.query_ann(ns, [1.0, 0.0, 0.0], top_k=5, flt=Eq("file_id", "nope")) == []


# --- BM25 ---------------------------------------------------------------------------------------------
async def test_query_bm25_exact_tokens_or_ranking(store: VectorStore, ns_prefix: str) -> None:
    """Spike c + port: pre_tokenized_array exact-token matching (64-char tokens), OR across query tokens,
    case-sensitive, no stemming, filter applies. Tolerant: sets, and a ranking prefix where one chunk
    matches strictly more query tokens than any other (score values never compared)."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("bm25")
        await seeded(store, ns)
        assert ids(await store.query_bm25(ns, "tokens", [TOK_B], top_k=10)) == ["c1"]
        assert set(ids(await store.query_bm25(ns, "tokens", [TOK_A], top_k=10))) == {"c1", "c2", "c3"}
        assert set(ids(await store.query_bm25(ns, "tokens", [TOK_B, TOK_C], top_k=10))) == {"c1", "c3", "c4"}
        two = await store.query_bm25(ns, "tokens", [TOK_A, TOK_C], top_k=10)
        assert set(ids(two)) == {"c1", "c2", "c3", "c4"}
        assert two[0].id == "c3"  # only chunk with both tokens
        assert [r.score is not None for r in two] == [True] * len(two)
        assert len(await store.query_bm25(ns, "tokens", [TOK_A], top_k=1)) == 1
        assert await store.query_bm25(ns, "tokens", [TOK_A.upper()], top_k=10) == []
        assert await store.query_bm25(ns, "tokens", ["a" * 63], top_k=10) == []
        assert await store.query_bm25(ns, "tokens", ["nope"], top_k=10) == []
        flt = await store.query_bm25(ns, "tokens", [TOK_A], top_k=10, flt=Eq("file_id", "F1"))
        assert set(ids(flt)) == {"c1", "c2"}
        proj = await store.query_bm25(ns, "tokens", [TOK_B], top_k=10, attrs=["file_id"])
        assert set(proj[0].attrs) <= {"file_id"}


async def test_query_bm25_errors(store: VectorStore, ns_prefix: str) -> None:
    """Port: missing ns -> NOT_FOUND. Non-FTS field -> INVALID_ARGUMENT (adapter maps the 400)."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("bm25err")
        await seeded(store, ns)
        assert await code_of(store.query_bm25(name("bm25_missing"), "tokens", [TOK_A], top_k=5)) == (
            ErrorCode.NOT_FOUND
        )
        assert await code_of(store.query_bm25(ns, "note", [TOK_A], top_k=5)) == ErrorCode.INVALID_ARGUMENT
        assert await code_of(store.query_bm25(ns, "file_id", [TOK_A], top_k=5)) == ErrorCode.INVALID_ARGUMENT


# --- vector-less rows (spike a/b/h) -------------------------------------------------------------------
async def test_vectorless_rejected_in_vector_ns(store: VectorStore, ns_prefix: str) -> None:
    """Spike a: turbopuffer returns 400 for a vector-less row in a vector namespace (separate write,
    mixed batch, explicit vector=None); the namespace is unchanged afterwards (batch atomic). A store
    more lenient than turbopuffer fails here."""
    async with namespaces(store, ns_prefix) as name:
        ns = name("vl_reject")
        await store.ensure_namespace(ns, VEC)
        await store.upsert(ns, [chunk("c1", file_id="F1")])
        bad = Row(id="root", attrs={"kind": "root"})
        assert await code_of(store.upsert(ns, [bad])) == ErrorCode.INVALID_ARGUMENT
        assert await code_of(store.upsert(ns, [chunk("c2"), bad])) == ErrorCode.INVALID_ARGUMENT
        explicit = Row(id="n1", attrs={"kind": "node"}, vector=None)
        assert await code_of(store.upsert(ns, [explicit])) == ErrorCode.INVALID_ARGUMENT
        assert await store.get_by_ids(ns, ["root", "n1", "c2"]) == []
        assert ids(await store.get_by_ids(ns, ["c1"])) == ["c1"]


async def test_vectorless_ns_accepts_tree_rows(store: VectorStore, ns_prefix: str) -> None:
    """Spike h: a dims=None namespace accepts node/root-style rows and get_by_ids returns them."""
    async with namespaces(store, ns_prefix) as name:
        tree = name("vl_tree__tree")
        await store.ensure_namespace(tree, TREE)
        rows = [node("root", kind="root", gen=1, expected_chunks=2), node("n1", kind="node")]
        assert await store.upsert(tree, rows) == 2
        got = {r.id: r for r in await store.get_by_ids(tree, ["root", "n1", "ghost"])}
        assert set(got) == {"root", "n1"}
        assert got["root"].attrs["kind"] == "root"
        assert got["root"].attrs["gen"] == 1
        assert got["root"].vector is None
        assert await store.delete_ids(tree, ["n1"]) == 1


async def test_branch_from_vectorless_copies_rows(store: VectorStore, ns_prefix: str) -> None:
    """Spike h: branch_from of a vector-less namespace copies its rows; the branch stays vector-less
    (accepts vector-less rows) and edits never touch the source."""
    async with namespaces(store, ns_prefix) as name:
        src, dst = name("vl_src__tree"), name("vl_dst__tree")
        await store.ensure_namespace(src, TREE)
        await store.upsert(src, [node("root", kind="root", gen=1), node("n1", kind="node")])
        await store.branch_from(dst, src)
        assert set(ids(await store.get_by_ids(dst, ["root", "n1"]))) == {"root", "n1"}
        assert await store.upsert(dst, [node("n2", kind="node")]) == 1
        assert ids(await store.get_by_ids(src, ["n2"])) == []


# --- branch_from --------------------------------------------------------------------------------------
async def test_branch_from_copies_rows_schema_and_isolates(store: VectorStore, ns_prefix: str) -> None:
    """Port: schema and rows inherited (including vectors); source missing -> NOT_FOUND; existing
    destination -> CONFLICT; edits to either side never touch the other."""
    async with namespaces(store, ns_prefix) as name:
        src, dst = name("br_src"), name("br_dst")
        await seeded(store, src)
        await store.branch_from(dst, src)
        assert await store.namespace_exists(dst)
        assert set(ids(await store.get_by_ids(dst, ["c1", "c2", "c3", "c4"]))) == {"c1", "c2", "c3", "c4"}
        assert ids(await store.query_ann(dst, [1.0, 0.0, 0.0], top_k=2)) == ["c1", "c2"]
        await store.ensure_namespace(dst, VEC)  # schema inherited: same schema is a no-op
        assert await code_of(store.branch_from(dst, src)) == ErrorCode.CONFLICT
        assert await code_of(store.branch_from(name("br_new"), name("br_ghost"))) == ErrorCode.NOT_FOUND
        await store.upsert(dst, [chunk("c9", file_id="F9")])
        await store.delete_ids(dst, ["c1"])
        await store.upsert(src, [chunk("c8", file_id="F8")])
        assert set(ids(await store.get_by_ids(src, ["c1", "c8", "c9"]))) == {"c1", "c8"}
        assert set(ids(await store.get_by_ids(dst, ["c1", "c8", "c9"]))) == {"c9"}


async def test_branch_pair_ordering(store: VectorStore, ns_prefix: str) -> None:
    """Design 5 (AC-5.3): callers copy `{ns}__tree` before `{ns}`. A writer commit (new chunk rows, then a
    new root row) lands between the two branch_from calls. Tree first -> branch has the OLD root and a
    chunk set that is a superset of what the old root describes (tree older than chunks: safe). Reverse
    order (negative control) -> branch has the NEW root but too few chunks (tree newer: unsafe)."""
    async with namespaces(store, ns_prefix) as name:

        async def make_source(base: str) -> tuple[str, str]:
            chunks_ns, tree_ns = name(base), name(base + "__tree")
            await store.ensure_namespace(chunks_ns, VEC)
            await store.ensure_namespace(tree_ns, TREE)
            await store.upsert(chunks_ns, [chunk("k1", gen=1), chunk("k2", gen=1)])
            await store.upsert(tree_ns, [node("root", kind="root", gen=1, expected_chunks=2)])
            return chunks_ns, tree_ns

        async def writer_commit(chunks_ns: str, tree_ns: str) -> None:
            await store.upsert(chunks_ns, [chunk("k3", gen=2)])  # chunks first ...
            await store.upsert(
                tree_ns, [node("root", kind="root", gen=2, expected_chunks=3)]
            )  # ... then root

        async def branch_pair(base: str, *, tree_first: bool) -> tuple[Row, set[str]]:
            src_c, src_t = await make_source(base)
            dst_c, dst_t = name(base + "_br"), name(base + "_br__tree")
            first, second = (
                ((dst_t, src_t), (dst_c, src_c)) if tree_first else ((dst_c, src_c), (dst_t, src_t))
            )
            await store.branch_from(*first)
            await writer_commit(src_c, src_t)  # the hook between the two calls
            await store.branch_from(*second)
            root = (await store.get_by_ids(dst_t, ["root"]))[0]
            have = set(ids(await store.get_by_ids(dst_c, ["k1", "k2", "k3"])))
            return root, have

        root, have = await branch_pair("ord_ok", tree_first=True)
        assert root.attrs["gen"] == 1
        assert root.attrs["expected_chunks"] == 2
        assert {"k1", "k2"} <= have  # superset of what the old root describes
        assert len(have) >= root.attrs["expected_chunks"]

        root, have = await branch_pair("ord_bad", tree_first=False)
        assert root.attrs["gen"] == 2
        assert root.attrs["expected_chunks"] == 3
        assert have == {"k1", "k2"}  # chunks older than the tree: fewer than the root expects
        assert len(have) < root.attrs["expected_chunks"]


ALL_CASES: list[Case] = [
    test_ensure_namespace_idempotent_and_mismatch,
    test_list_and_delete_namespace,
    test_upsert_and_get_by_ids,
    test_if_absent_upsert,
    test_conditional_upsert,
    test_conditional_upsert_on_vectorless_namespace_is_a_cas,
    test_delete_ids,
    test_delete_by_file_and_missing_ns,
    test_delete_by_filter,
    test_filters_eq_in_contains_and_not,
    test_glob_whole_segment_prefix,
    test_query_ann_ranking_topk_filter_attrs,
    test_query_bm25_exact_tokens_or_ranking,
    test_query_bm25_errors,
    test_vectorless_rejected_in_vector_ns,
    test_vectorless_ns_accepts_tree_rows,
    test_branch_from_vectorless_copies_rows,
    test_branch_from_copies_rows_schema_and_isolates,
    test_branch_pair_ordering,
]
