"""Runs the backend-agnostic store suite on `FakeStore` (task 2.4) plus fake-only checks.

AC-5.3, FR-P3, FR-X4. The live runner (task 7.10) imports `store_suite.ALL_CASES` the same way.
"""

import uuid

import pytest
import store_suite

from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.store import AttrSpec, Contains, Eq, Glob, In, NamespaceSchema, Row


@pytest.mark.parametrize("case", store_suite.ALL_CASES, ids=lambda c: c.__name__)
async def test_store_suite_on_fake_store(case: store_suite.Case) -> None:
    await case(FakeStore(), f"scout_dev_{uuid.uuid4().hex[:8]}_")


def test_suite_lists_every_case() -> None:
    """A case added to store_suite.py but missing from ALL_CASES would never run."""
    defined = {n for n in dir(store_suite) if n.startswith("test_")}
    assert defined == {c.__name__ for c in store_suite.ALL_CASES}


# --- fake-only: client-side limits and strictness the real service does not share ------------------------
async def _ns() -> tuple[FakeStore, str]:
    s = FakeStore()
    await s.ensure_namespace("n", store_suite.VEC)
    await s.upsert("n", [Row(id="a", attrs={"path_enc": "x/y", "file_id": "F"}, vector=[1.0, 0.0, 0.0])])
    return s, "n"


@pytest.mark.parametrize("big", [4097, 10_000])
async def test_filter_value_over_4kib_rejected(big: int) -> None:
    """Design 3: 4 KiB filter-value check (docs limit; spike g saw none up to 1 MiB, so fake-only)."""
    s, ns = await _ns()
    for flt in (
        Eq("file_id", "x" * big),
        In("file_id", ["ok", "x" * big]),
        Glob("path_enc", "a" * big),
        Contains("dir_tags", "x" * big),
    ):
        with pytest.raises(ScoutError) as ei:
            await s.query_ann(ns, [1.0, 0.0, 0.0], top_k=1, flt=flt)
        assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError):
        await s.delete_by_filter(ns, Eq("file_id", "x" * big))


async def test_filter_value_at_4kib_accepted() -> None:
    s, ns = await _ns()
    assert await s.query_ann(ns, [1.0, 0.0, 0.0], top_k=1, flt=Eq("file_id", "x" * 4096)) == []


async def test_glob_on_non_glob_attribute_rejected() -> None:
    """Task 2.4: attr must be glob=True (schema-known attrs only)."""
    s, ns = await _ns()
    with pytest.raises(ScoutError) as ei:
        await s.query_ann(ns, [1.0, 0.0, 0.0], top_k=1, flt=Glob("file_id", "F*"))
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT


async def test_glob_segment_semantics() -> None:
    """`*` and `?` stay inside a `/`-segment, `**` crosses segments, match is whole-string."""
    s = FakeStore()
    schema = NamespaceSchema(dims=3, attrs={"p": AttrSpec("string", glob=True)})
    await s.ensure_namespace("g", schema)
    paths = {"1": "aa/bb/cc", "2": "aa/bb", "3": "aa/cc/dd", "4": "ab/bb/cc", "5": "aa"}
    await s.upsert("g", [Row(id=i, attrs={"p": p}, vector=[1.0, 0.0, 0.0]) for i, p in paths.items()])

    async def hits(pattern: str) -> set[str]:
        rows = await s.query_ann("g", [1.0, 0.0, 0.0], top_k=10, flt=Glob("p", pattern))
        return {r.id for r in rows}

    assert await hits("aa/*") == {"2"}
    assert await hits("aa/**") == {"1", "2", "3"}
    assert await hits("aa/*/cc") == {"1"}
    assert await hits("a?/bb/*") == {"1", "4"}
    assert await hits("**/cc") == {"1", "4"}
    assert await hits("aa") == {"5"}
    assert await hits("a") == set()
    assert await hits("aa/b") == set()
    assert await hits("**33") == set()
