"""FakeStore part 1 (task 2.3): namespaces, rows, filters, branch_from, vector-less rules.

AC-5.3, FR-P3. Spike 0a-A (artifacts/spike/0a_tpuf.json) behaviours are named in each test docstring.
"""

import pytest

from scout.adapters.fake_store import FakeStore
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

_check_store: VectorStore = FakeStore()

ATTRS = {
    "kind": AttrSpec("string"),
    "file_id": AttrSpec("string"),
    "file_hash": AttrSpec("string"),
    "dir_tags": AttrSpec("[]string"),
    "path": AttrSpec("string", glob=True),
    "secret": AttrSpec("string", filterable=False),
}
VEC = NamespaceSchema(dims=3, attrs=ATTRS)
TREE = NamespaceSchema(dims=None, attrs=ATTRS)


def chunk(i: str, **attrs: object) -> Row:
    return Row(id=i, attrs=dict(attrs), vector=[1.0, 0.0, 0.0])


def node(i: str, **attrs: object) -> Row:
    return Row(id=i, attrs=dict(attrs))


async def codes(coro: object) -> ErrorCode:
    with pytest.raises(ScoutError) as ei:
        await coro  # type: ignore[misc]
    return ei.value.code


# --- namespaces -------------------------------------------------------------------------------------
async def test_ensure_namespace_idempotent_and_schema_mismatch() -> None:
    s = FakeStore()
    await s.ensure_namespace("scout_dev_a", VEC)
    await s.ensure_namespace("scout_dev_a", VEC)
    assert await codes(s.ensure_namespace("scout_dev_a", TREE)) == ErrorCode.INTERNAL
    other_dims = NamespaceSchema(dims=4, attrs=ATTRS)
    assert await codes(s.ensure_namespace("scout_dev_a", other_dims)) == ErrorCode.INTERNAL


async def test_exists_list_delete_idempotent() -> None:
    s = FakeStore()
    for n in ("scout_dev_a", "scout_dev_b", "scout_eval_c"):
        await s.ensure_namespace(n, TREE)
    assert await s.namespace_exists("scout_dev_a")
    assert not await s.namespace_exists("nope")
    assert await s.list_namespaces("scout_dev_") == ["scout_dev_a", "scout_dev_b"]
    await s.delete_namespace("scout_dev_a")
    await s.delete_namespace("scout_dev_a")  # idempotent
    await s.delete_namespace("never_existed")
    assert await s.list_namespaces("scout_dev_") == ["scout_dev_b"]


# --- rows -------------------------------------------------------------------------------------------
async def test_upsert_get_by_ids_projection_and_missing_ids() -> None:
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    assert await s.upsert("n", [chunk("a", kind="chunk", file_id="f1"), chunk("b", kind="chunk")]) == 2
    got = await s.get_by_ids("n", ["a", "zz", "b"])
    assert [r.id for r in got] == ["a", "b"]  # missing ids omitted
    assert got[0].vector is None  # include_vectors False
    assert [r.vector for r in await s.get_by_ids("n", ["a"], include_vectors=True)] == [[1.0, 0.0, 0.0]]
    only = await s.get_by_ids("n", ["a"], attrs=["file_id"])
    assert dict(only[0].attrs) == {"file_id": "f1"}
    assert await codes(s.get_by_ids("missing", ["a"])) == ErrorCode.NOT_FOUND


async def test_upsert_overwrites_and_if_absent_counts_inserted_only() -> None:
    """Spike f: if_absent first=1 second=0, value kept."""
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    assert await s.upsert("n", [chunk("a", v="old")], if_absent=True) == 1
    assert await s.upsert("n", [chunk("a", v="new")], if_absent=True) == 0
    assert (await s.get_by_ids("n", ["a"]))[0].attrs["v"] == "old"
    assert await s.upsert("n", [chunk("a", v="new")]) == 1  # plain upsert overwrites
    assert (await s.get_by_ids("n", ["a"]))[0].attrs["v"] == "new"
    assert await s.upsert("n", [chunk("a"), chunk("b")], if_absent=True) == 1  # mixed batch


async def test_rows_are_copy_safe_in_and_out() -> None:
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    tags = ["d1"]
    vec = [1.0, 0.0, 0.0]
    await s.upsert("n", [Row(id="a", attrs={"dir_tags": tags}, vector=vec)])
    tags.append("mutated-after-write")
    vec[0] = 9.0
    got = (await s.get_by_ids("n", ["a"], include_vectors=True))[0]
    assert got.attrs["dir_tags"] == ["d1"]
    assert got.vector == [1.0, 0.0, 0.0]
    got.attrs["dir_tags"].append("mutated-after-read")  # type: ignore[union-attr]
    again = (await s.get_by_ids("n", ["a"]))[0]
    assert again.attrs["dir_tags"] == ["d1"]


async def test_row_id_over_64_bytes_rejected_atomically() -> None:
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    assert await codes(s.upsert("n", [chunk("ok"), chunk("x" * 65)])) == ErrorCode.INVALID_ARGUMENT
    assert await s.get_by_ids("n", ["ok"]) == []


async def test_delete_ids_and_missing_namespace() -> None:
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    await s.upsert("n", [chunk("a"), chunk("b")])
    assert await s.delete_ids("n", ["a", "zz"]) == 1
    assert [r.id for r in await s.get_by_ids("n", ["a", "b"])] == ["b"]
    assert await s.delete_ids("missing", ["a"]) == 0
    assert await s.delete_by_file("missing", "f") == 0
    assert await s.delete_by_filter("missing", Eq("kind", "chunk")) == 0


async def test_delete_by_file_except_hash() -> None:
    """Spike d: And(Eq, Not(Eq)) in delete_by_filter; other file and new hash kept."""
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    await s.upsert(
        "n",
        [
            chunk("c1", file_id="F1", file_hash="H1"),
            chunk("c2", file_id="F1", file_hash="H1"),
            chunk("c3", file_id="F2", file_hash="H1"),
            chunk("c4", file_id="F1", file_hash="H2"),
        ],
    )
    assert await s.delete_by_file("n", "F1", except_file_hash="H2") == 2
    assert [r.id for r in await s.get_by_ids("n", ["c1", "c2", "c3", "c4"])] == ["c3", "c4"]
    assert await s.delete_by_file("n", "F1") == 1  # no except: all of the file
    assert [r.id for r in await s.get_by_ids("n", ["c3", "c4"])] == ["c3"]


# --- vector-less rules (spike a, h) -----------------------------------------------------------------
async def test_vectorless_ns_accepts_rows_and_get_by_ids_returns_them() -> None:
    """Spike h: dims=None namespace accepts node rows; get_by_ids returns them."""
    s = FakeStore()
    await s.ensure_namespace("n__tree", TREE)
    assert await s.upsert("n__tree", [node("root", kind="root"), node("n1", kind="node")]) == 2
    got = await s.get_by_ids("n__tree", ["n1", "root", "x"], include_vectors=True)
    assert [(r.id, r.vector) for r in got] == [("n1", None), ("root", None)]


async def test_vectorless_rejected_in_vector_ns_separate_write() -> None:
    """Spike a (separate_write): HTTP 400; namespace unchanged."""
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    await s.upsert("n", [chunk("c1")])
    assert await codes(s.upsert("n", [node("root"), node("n1")])) == ErrorCode.INVALID_ARGUMENT
    assert await s.get_by_ids("n", ["root", "n1"]) == []
    assert [r.id for r in await s.get_by_ids("n", ["c1"])] == ["c1"]


async def test_vectorless_rejected_in_vector_ns_mixed_batch_is_atomic() -> None:
    """Spike a (mixed_batch): the whole batch is rejected, the valid chunk row is not written."""
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    assert await codes(s.upsert("n", [chunk("c1"), node("root")])) == ErrorCode.INVALID_ARGUMENT
    assert await codes(s.upsert("n", [node("root"), chunk("c1")])) == ErrorCode.INVALID_ARGUMENT
    assert await s.get_by_ids("n", ["c1", "root"]) == []


async def test_vectorless_rejected_in_vector_ns_explicit_none_vector() -> None:
    """Spike a (explicit_none): Row(vector=None) rejected; so is if_absent."""
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    row = Row(id="r", attrs={}, vector=None)
    assert await codes(s.upsert("n", [row])) == ErrorCode.INVALID_ARGUMENT
    assert await codes(s.upsert("n", [row], if_absent=True)) == ErrorCode.INVALID_ARGUMENT
    assert await s.get_by_ids("n", ["r"]) == []


async def test_vector_row_rejected_in_vectorless_ns_and_dims_checked() -> None:
    s = FakeStore()
    await s.ensure_namespace("t", TREE)
    assert await codes(s.upsert("t", [chunk("c1")])) == ErrorCode.INVALID_ARGUMENT  # port-level guard
    assert await s.get_by_ids("t", ["c1"]) == []
    await s.ensure_namespace("v", VEC)
    bad = Row(id="c", attrs={}, vector=[1.0, 2.0])
    assert await codes(s.upsert("v", [bad])) == ErrorCode.INVALID_ARGUMENT


async def test_implicitly_created_vector_ns_rejects_later_vectorless_rows() -> None:
    """Spike a: tpuf creates a namespace on first write; a first write with vectors makes it a vector ns."""
    s = FakeStore()
    assert await s.upsert("auto", [chunk("c1")]) == 1
    assert await codes(s.upsert("auto", [node("root")])) == ErrorCode.INVALID_ARGUMENT
    assert await s.upsert("auto", [chunk("c2")]) == 1
    # and the other way: a first vector-less write makes a vector-less ns
    assert await s.upsert("auto_tree", [node("root")]) == 1
    assert await codes(s.upsert("auto_tree", [chunk("c1")])) == ErrorCode.INVALID_ARGUMENT


# --- filters ----------------------------------------------------------------------------------------
async def _filter_ns() -> FakeStore:
    s = FakeStore()
    await s.ensure_namespace("n", VEC)
    await s.upsert(
        "n",
        [
            chunk("c1", kind="chunk", file_id="F1", dir_tags=["d1", "d2"], n=1, ok=True),
            chunk("c2", kind="chunk", file_id="F2", dir_tags=["d2"], n=2, ok=False),
            chunk("c3", kind="other", file_id="F1", dir_tags=[]),
            chunk("c4"),  # no attrs at all
        ],
    )
    return s


async def _ids(s: FakeStore, flt: Filter) -> list[str]:
    n = await s.delete_by_filter("n", flt)
    left = {r.id for r in await s.get_by_ids("n", ["c1", "c2", "c3", "c4"])}
    return sorted({"c1", "c2", "c3", "c4"} - left) if n else []


@pytest.mark.parametrize(
    ("flt", "expected"),
    [
        (Eq("kind", "chunk"), ["c1", "c2"]),
        (Eq("n", 2), ["c2"]),
        (Eq("ok", True), ["c1"]),
        (In("file_id", ["F1", "F9"]), ["c1", "c3"]),
        (Contains("dir_tags", "d2"), ["c1", "c2"]),  # spike c: Contains on []string
        (And([Contains("dir_tags", "d1"), Eq("kind", "chunk")]), ["c1"]),  # spike c
        (And([Eq("file_id", "F1"), Not(Eq("kind", "chunk"))]), ["c3"]),  # spike d
        (Not(Eq("kind", "chunk")), ["c3", "c4"]),  # Not negates; a missing attr is not equal
        (Eq("missing", "x"), []),  # missing attr = no match
        (In("missing", ["x"]), []),
        (Contains("missing", "x"), []),
        (And([]), ["c1", "c2", "c3", "c4"]),
    ],
)
async def test_filter_semantics(flt: Filter, expected: list[str]) -> None:
    assert await _ids(await _filter_ns(), flt) == expected


async def test_contains_on_non_array_attr_and_unfilterable_attr_rejected() -> None:
    s = await _filter_ns()
    assert await codes(s.delete_by_filter("n", Contains("kind", "chunk"))) == ErrorCode.INVALID_ARGUMENT
    assert await codes(s.delete_by_filter("n", Eq("secret", "x"))) == ErrorCode.INVALID_ARGUMENT


async def test_glob_filter_deletes_whole_segment_prefix() -> None:
    """Task 2.4 replaced the NotImplementedError stub: Glob evaluates (full coverage in tests/contract)."""
    s = await _filter_ns()
    assert await s.delete_by_filter("n", Glob("path", "zz/**")) == 0


# --- branch_from ------------------------------------------------------------------------------------
async def test_branch_from_copies_rows_schema_and_isolates_edits() -> None:
    s = FakeStore()
    await s.ensure_namespace("src", VEC)
    await s.upsert("src", [chunk("c1", kind="chunk", dir_tags=["d"]), chunk("c2", kind="chunk")])
    await s.branch_from("dst", "src")
    assert [r.id for r in await s.get_by_ids("dst", ["c1", "c2"], include_vectors=True)] == ["c1", "c2"]
    await s.ensure_namespace("dst", VEC)  # schema copied: identical ensure is a no-op
    await s.upsert("dst", [chunk("c3")])
    await s.delete_ids("dst", ["c1"])
    await s.upsert("dst", [chunk("c2", kind="edited")])
    assert [r.id for r in await s.get_by_ids("src", ["c1", "c2", "c3"])] == ["c1", "c2"]
    assert (await s.get_by_ids("src", ["c2"]))[0].attrs["kind"] == "chunk"
    await s.upsert("src", [chunk("c4")])  # and source edits do not reach the branch
    assert await s.get_by_ids("dst", ["c4"]) == []
    assert await codes(s.upsert("dst", [node("root")])) == ErrorCode.INVALID_ARGUMENT  # still a vector ns


async def test_branch_from_vectorless_stays_vectorless_and_copies_rows() -> None:
    """Spike h: branch_from copies vector-less node rows."""
    s = FakeStore()
    await s.ensure_namespace("n__tree", TREE)
    await s.upsert("n__tree", [node("root", kind="root"), node("n1", kind="node")])
    await s.branch_from("b__tree", "n__tree")
    assert [r.id for r in await s.get_by_ids("b__tree", ["n1", "root"])] == ["n1", "root"]
    assert await s.upsert("b__tree", [node("n2")]) == 1  # still accepts vector-less rows
    assert await codes(s.upsert("b__tree", [chunk("c1")])) == ErrorCode.INVALID_ARGUMENT
    assert await s.get_by_ids("n__tree", ["n2"]) == []


async def test_branch_from_errors() -> None:
    s = FakeStore()
    assert await codes(s.branch_from("dst", "missing")) == ErrorCode.NOT_FOUND
    await s.ensure_namespace("src", VEC)
    await s.ensure_namespace("dst", VEC)
    assert await codes(s.branch_from("dst", "src")) == ErrorCode.CONFLICT  # destination must not exist
    assert await codes(s.branch_from("src", "src")) == ErrorCode.CONFLICT
    assert not await s.namespace_exists("other")
    assert await codes(s.branch_from("src", "missing")) == ErrorCode.NOT_FOUND  # source checked first
