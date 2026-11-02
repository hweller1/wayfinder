"""turbopuffer adapter: translation, chunking, errors (task 7.1). FR-P7, AC-5.3.

Uses a recording stub SDK (`tpuf_stub.py`) and the REAL SDK exception classes (constructed offline) so the
error mapping is checked against what the SDK actually raises. No network.
"""

import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest
import turbopuffer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))  # shared stub SDK (tpuf_stub.py)
from tpuf_stub import VECTORLESS_400, StubTurbopuffer, no_sleep  # noqa: E402  # pyright: ignore[reportMissingImports]

from scout.adapters.turbopuffer import TurbopufferStore
from scout.adapters.turbopuffer.client import schema_payload
from scout.adapters.turbopuffer.filters import to_tpuf_filter
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.store import And, AttrSpec, Contains, Eq, Glob, In, NamespaceSchema, Not, Row

VEC = NamespaceSchema(
    dims=3,
    attrs={
        "kind": AttrSpec("string"),
        "path_enc": AttrSpec("string", glob=True),
        "dir_tags": AttrSpec("[]string"),
        "tokens": AttrSpec("[]string", filterable=False, fts="pre_tokenized_array"),
        "note": AttrSpec("string", filterable=False),
    },
)
TREE = NamespaceSchema(dims=None, attrs={"kind": AttrSpec("string"), "gen": AttrSpec("int")})


def make(**kw: Any) -> tuple[TurbopufferStore, StubTurbopuffer, list[float]]:
    sleep, slept = no_sleep()
    stub = StubTurbopuffer()
    return TurbopufferStore(client=stub, sleep=sleep, **kw), stub, slept


def sdk_error(cls: type[turbopuffer.APIStatusError], status: int, body: str = "VENDOR-TEXT") -> Exception:
    req = httpx2.Request("POST", "https://api.turbopuffer.test/v2/namespaces/x")
    return cls("VENDOR-TEXT", response=httpx2.Response(status, request=req, text=body), body=body)


# --- filter translation: every AST node ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("flt", "expected"),
    [
        (Eq("file_id", "F1"), ("file_id", "Eq", "F1")),
        (Eq("gen", 3), ("gen", "Eq", 3)),
        (Eq("flag", True), ("flag", "Eq", True)),
        (In("file_id", ["a", "b"]), ("file_id", "In", ["a", "b"])),
        (Glob("path_enc", "aa/**"), ("path_enc", "Glob", "aa/**")),
        (Contains("dir_tags", "d1"), ("dir_tags", "Contains", "d1")),
        (Not(Eq("file_hash", "H")), ("Not", ("file_hash", "Eq", "H"))),
        (
            And([Eq("file_id", "F"), Not(Eq("file_hash", "H"))]),
            ("And", [("file_id", "Eq", "F"), ("Not", ("file_hash", "Eq", "H"))]),
        ),
        (
            And([And([Contains("dir_tags", "d")]), In("k", [])]),
            ("And", [("And", [("dir_tags", "Contains", "d")]), ("k", "In", [])]),
        ),
    ],
)
def test_filter_translation_table(flt: Any, expected: Any) -> None:
    assert to_tpuf_filter(flt) == expected


def test_filter_value_over_4kib_refused_in_every_node() -> None:
    big = "x" * 4097
    for flt in (Eq("a", big), In("a", ["ok", big]), Glob("a", big), Contains("a", big)):
        with pytest.raises(ScoutError) as ei:
            to_tpuf_filter(And([flt]))
        assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert to_tpuf_filter(Eq("a", "x" * 4096)) == ("a", "Eq", "x" * 4096)


# --- schema translation -------------------------------------------------------------------------------
def test_schema_payload_vector_namespace_exact() -> None:
    assert schema_payload(VEC) == {
        "kind": {"type": "string", "filterable": True},
        "path_enc": {"type": "string", "filterable": True, "glob": True},
        "dir_tags": {"type": "[]string", "filterable": True},
        "tokens": {
            "type": "[]string",
            "filterable": False,
            "full_text_search": {"tokenizer": "pre_tokenized_array"},
        },
        "note": {"type": "string", "filterable": False},
        "vector": {"type": "[3]f32", "ann": True},
    }


def test_schema_payload_vectorless_has_no_vector_attribute() -> None:
    payload = schema_payload(TREE)
    assert "vector" not in payload
    assert payload == {
        "kind": {"type": "string", "filterable": True},
        "gen": {"type": "int", "filterable": True},
    }


async def test_ensure_namespace_sends_the_payload_and_creates_empty() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("n", VEC)
    await store.ensure_namespace("t__tree", TREE)
    writes = stub.calls_of("write")
    assert writes[0]["create_namespace"] is True
    assert writes[0]["schema"] == {"id": "string", **schema_payload(VEC)}
    assert writes[0]["distance_metric"] == "cosine_distance"
    assert "vector" not in writes[1]["schema"]
    assert "distance_metric" not in writes[1]
    assert len(writes) == 2
    await store.ensure_namespace("n", VEC)  # idempotent: no third write
    assert len(stub.calls_of("write")) == 2


# --- single-namespace branch --------------------------------------------------------------------------
async def test_branch_from_is_one_call_on_one_namespace_and_never_touches_a_sibling() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("a", VEC)
    await store.ensure_namespace("a__tree", TREE)
    await store.branch_from("b", "a")
    assert stub.calls_of("branch_from") == [{"source_namespace": "a"}]
    assert "b" in stub.data
    assert "b__tree" not in stub.data  # the sibling is the service's job
    with pytest.raises(ScoutError) as ei:
        await store.branch_from("b", "a")
    assert ei.value.code == ErrorCode.CONFLICT
    with pytest.raises(ScoutError) as ei2:
        await store.branch_from("c", "ghost")
    assert ei2.value.code == ErrorCode.NOT_FOUND


async def test_delete_namespace_uses_delete_all_and_is_idempotent() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("n", VEC)
    await store.delete_namespace("n")
    await store.delete_namespace("n")
    assert [c for c in stub.calls if c[0] == "delete_all"][0][1] == "n"
    assert not await store.namespace_exists("n")


# --- upsert mapping -----------------------------------------------------------------------------------
async def test_if_absent_and_condition_map_to_upsert_condition() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("n", VEC)
    row = Row(id="k", attrs={"kind": "cache"}, vector=[1.0, 0.0, 0.0])
    assert await store.upsert("n", [row], if_absent=True) == 1
    assert await store.upsert("n", [row], if_absent=True) == 0
    assert stub.calls_of("write")[-1]["upsert_condition"] == ("id", "Eq", None)
    cond = And([Eq("kind", "cache"), Not(Eq("note", "x"))])
    assert await store.upsert("n", [row], condition=cond) == 1
    assert stub.calls_of("write")[-1]["upsert_condition"] == (
        "And",
        [("kind", "Eq", "cache"), ("Not", ("note", "Eq", "x"))],
    )
    await store.upsert("n", [row])
    assert "upsert_condition" not in stub.calls_of("write")[-1]
    with pytest.raises(ScoutError) as ei:
        await store.upsert("n", [row], if_absent=True, condition=Eq("kind", "x"))
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert await store.upsert("n", []) == 0


async def test_upsert_row_payload_carries_id_attrs_and_vector() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("n", VEC)
    await store.upsert("n", [Row(id="c1", attrs={"kind": "chunk", "dir_tags": ("a", "b")}, vector=[1, 0, 0])])
    sent = stub.calls_of("write")[-1]["upsert_rows"][0]
    assert sent == {"id": "c1", "kind": "chunk", "dir_tags": ["a", "b"], "vector": [1.0, 0.0, 0.0]}


async def test_vector_row_into_a_vectorless_namespace_is_refused_before_the_call() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("t__tree", TREE)
    before = len(stub.calls_of("write"))
    with pytest.raises(ScoutError) as ei:
        await store.upsert("t__tree", [Row(id="x", vector=[1.0, 0.0, 0.0])])
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert len(stub.calls_of("write")) == before


# --- id chunking at 5,000 -----------------------------------------------------------------------------
async def test_get_by_ids_chunks_at_5000() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("n", VEC)
    await store.upsert("n", [Row(id="c1", attrs={"kind": "chunk"}, vector=[1.0, 0.0, 0.0])])
    ids = [f"i{n}" for n in range(11_000)] + ["c1"]
    got = await store.get_by_ids("n", ids)
    assert [r.id for r in got] == ["c1"]
    queries = stub.calls_of("query")
    assert [len(q["filters"][2]) for q in queries] == [5000, 5000, 1001]
    assert all(q["filters"][:2] == ("id", "In") for q in queries)
    assert await store.get_by_ids("n", []) == []
    assert len(stub.calls_of("query")) == 3


async def test_get_by_ids_dedups_projects_and_returns_vectors_on_request() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("n", VEC)
    await store.upsert("n", [Row(id="c1", attrs={"kind": "chunk", "note": "z"}, vector=[0.5, 0.5, 0.0])])
    r = (await store.get_by_ids("n", ["c1", "c1"], attrs=["kind"]))[0]
    assert r.attrs == {"kind": "chunk"}
    assert stub.calls_of("query")[-1]["include_attributes"] == ["kind"]
    v = (await store.get_by_ids("n", ["c1"], include_vectors=True))[0]
    assert v.vector == [0.5, 0.5, 0.0]
    assert "vector" in stub.calls_of("query")[-1]["include_attributes"]
    assert (await store.get_by_ids("n", ["c1"]))[0].vector is None


async def test_delete_ids_counts_rows_actually_deleted_and_chunks() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("n", VEC)
    await store.upsert("n", [Row(id="a", vector=[1.0, 0.0, 0.0]), Row(id="b", vector=[1.0, 0.0, 0.0])])
    assert await store.delete_ids("n", ["a", "ghost", "a"]) == 1
    assert stub.calls_of("write")[-1]["delete_by_filter"] == ("id", "In", ["a", "ghost"])
    assert await store.delete_ids("n", [f"x{i}" for i in range(5001)]) == 0
    assert len(stub.calls_of("write")[-2:]) == 2
    assert await store.delete_ids("missing", ["a"]) == 0


# --- queries ------------------------------------------------------------------------------------------
async def test_query_shapes_and_score_direction() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("n", VEC)
    await store.upsert(
        "n",
        [
            Row(id="c1", attrs={"kind": "chunk", "tokens": ["t1"]}, vector=[1.0, 0.0, 0.0]),
            Row(id="c2", attrs={"kind": "chunk", "tokens": ["t2"]}, vector=[0.0, 1.0, 0.0]),
        ],
    )
    top = await store.query_ann("n", [1, 0, 0], top_k=2, flt=Eq("kind", "chunk"), attrs=["kind"])
    q = stub.calls_of("query")[-1]
    assert q["rank_by"] == ("vector", "ANN", [1.0, 0.0, 0.0])
    assert q["top_k"] == 2
    assert q["filters"] == ("kind", "Eq", "chunk")
    assert q["include_attributes"] == ["kind"]
    assert [r.id for r in top] == ["c1", "c2"]
    assert top[0].score is not None
    assert top[1].score is not None
    assert top[0].score > top[1].score  # cosine distance flipped: higher = better
    assert top[0].attrs == {"kind": "chunk"}
    hits = await store.query_bm25("n", "tokens", ["t1", "t1", "zz"], top_k=5)
    q = stub.calls_of("query")[-1]
    assert q["rank_by"] == ("tokens", "BM25", ["t1", "zz"])  # list[str] operand, de-duplicated
    assert [r.id for r in hits] == ["c1"]
    assert hits[0].score is not None
    assert hits[0].score > 0
    assert await store.query_bm25("n", "tokens", [], top_k=5) == []
    for bad in (0, -1):
        with pytest.raises(ScoutError) as ei:
            await store.query_ann("n", [1, 0, 0], top_k=bad)
        assert ei.value.code == ErrorCode.INVALID_ARGUMENT


# --- error translation --------------------------------------------------------------------------------
async def test_vectorless_400_maps_to_invalid_argument_without_vendor_text() -> None:
    store, stub, _ = make()
    await store.ensure_namespace("n", VEC)
    for body in (VECTORLESS_400, "\U0001fab9 An empty vector was passed! Each row must have a vector."):
        stub.fail.append(sdk_error(turbopuffer.BadRequestError, 400, body))
        with pytest.raises(ScoutError) as ei:
            await store.upsert("n", [Row(id="x", vector=[1.0, 0.0, 0.0])])
        assert ei.value.code == ErrorCode.INVALID_ARGUMENT
        assert ei.value.message == "the vector store rejected the request"
        assert "VENDOR-TEXT" not in str(ei.value)
        assert not ei.value.retryable
    # and the stub's own genuine rejection of a vector-less row in a vector namespace takes the same path
    with pytest.raises(ScoutError) as ei2:
        await store.upsert("n", [Row(id="root", attrs={"kind": "root"})])
    assert ei2.value.code == ErrorCode.INVALID_ARGUMENT
    assert VECTORLESS_400 not in str(ei2.value)


@pytest.mark.parametrize(
    ("cls", "status", "code"),
    [
        (turbopuffer.BadRequestError, 400, ErrorCode.INVALID_ARGUMENT),
        (turbopuffer.UnprocessableEntityError, 422, ErrorCode.INVALID_ARGUMENT),
        (turbopuffer.NotFoundError, 404, ErrorCode.NOT_FOUND),
        (turbopuffer.ConflictError, 409, ErrorCode.CONFLICT),
        (turbopuffer.AuthenticationError, 401, ErrorCode.INTERNAL),
        (turbopuffer.PermissionDeniedError, 403, ErrorCode.INTERNAL),
    ],
)
async def test_non_retryable_statuses(cls: type[turbopuffer.APIStatusError], status: int, code: Any) -> None:
    store, stub, slept = make()
    stub.fail.append(sdk_error(cls, status))
    with pytest.raises(ScoutError) as ei:
        await store.query_ann("n", [1.0, 0.0, 0.0], top_k=1)
    assert ei.value.code == code
    assert "VENDOR-TEXT" not in str(ei.value)
    assert slept == []
    assert len(stub.calls_of("query")) == 1  # no retry


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: sdk_error(turbopuffer.RateLimitError, 429),
        lambda: sdk_error(turbopuffer.InternalServerError, 503),
        lambda: sdk_error(turbopuffer.InternalServerError, 500),
        lambda: turbopuffer.APIConnectionError(request=httpx2.Request("GET", "https://x.test")),
    ],
    ids=["429", "503", "500", "connection"],
)
async def test_backoff_then_upstream_unavailable(make_exc: Any) -> None:
    store, stub, slept = make(max_attempts=4)
    stub.fail.extend(make_exc() for _ in range(4))
    with pytest.raises(ScoutError) as ei:
        await store.delete_by_filter("n", Eq("file_id", "F"))
    assert ei.value.code == ErrorCode.UPSTREAM_UNAVAILABLE
    assert ei.value.retryable
    assert "VENDOR-TEXT" not in str(ei.value)
    assert slept == [0.5, 1.0, 2.0]  # backoff between the 4 attempts, none after the last
    assert len(stub.calls_of("write")) == 4


async def test_transient_failure_then_success() -> None:
    store, stub, slept = make()
    await store.ensure_namespace("n", VEC)
    stub.fail.extend(
        [sdk_error(turbopuffer.RateLimitError, 429), sdk_error(turbopuffer.InternalServerError, 502)]
    )
    assert await store.upsert("n", [Row(id="a", vector=[1.0, 0.0, 0.0])]) == 1
    assert slept == [0.5, 1.0]


async def test_timeout_maps_to_upstream_timeout() -> None:
    store, stub, slept = make(max_attempts=2)
    stub.fail.extend(turbopuffer.APITimeoutError(httpx2.Request("GET", "https://x.test")) for _ in range(2))
    with pytest.raises(ScoutError) as ei:
        await store.namespace_exists("n")
    assert ei.value.code == ErrorCode.UPSTREAM_TIMEOUT
    assert slept == [0.5]


async def test_unknown_exceptions_never_leak() -> None:
    store, stub, _ = make()
    stub.fail.append(RuntimeError("VENDOR-TEXT secret"))
    with pytest.raises(ScoutError) as ei:
        await store.namespace_exists("n")
    assert ei.value.code == ErrorCode.INTERNAL
    assert "VENDOR-TEXT" not in str(ei.value)
    assert ei.value.__cause__ is None


async def test_schema_mismatch_is_internal() -> None:
    store, _, _ = make()
    await store.ensure_namespace("n", VEC)
    for other in (
        TREE,
        NamespaceSchema(dims=4, attrs=VEC.attrs),
        NamespaceSchema(dims=3, attrs={"kind": AttrSpec("int")}),
    ):
        with pytest.raises(ScoutError) as ei:
            await store.ensure_namespace("n", other)
        assert ei.value.code == ErrorCode.INTERNAL


# --- construction -------------------------------------------------------------------------------------
def test_from_env_and_lazy_sdk_import() -> None:
    with pytest.raises(ScoutError):
        TurbopufferStore.from_env({})
    store = TurbopufferStore.from_env({"TURBOPUFFER_API_KEY": "k"})
    assert store._region == "gcp-us-central1"  # pyright: ignore[reportPrivateUsage]
    assert store._client is None  # pyright: ignore[reportPrivateUsage]
    sdk = store._sdk()  # pyright: ignore[reportPrivateUsage]  # builds the client; makes no request
    assert type(sdk).__name__ == "AsyncTurbopuffer"
