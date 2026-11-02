"""Task 3.2: GatewayService team key, sync_begin, status, delete_branch. FR-G1, FR-C15, AC-3.10 (server)."""

import asyncio
import base64
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import StatusReq, SyncBeginReq
from scout.core.crypto import ids, keys
from scout.core.ports.gateway import Gateway
from scout.core.ports.store import Filter, Row
from scout.core.service.gateway import GatewayService
from scout.core.service.schema import CHUNK_SCHEMA, META_SCHEMA, ROOT_ID, TREE_SCHEMA
from scout.core.service.tenants import KEYS_NAMESPACE, Tenant

MASTER = b"m" * 32
REPO = "a" * 32
BR_BASE = "b" * 16
BR_NEW = "c" * 16
Call = tuple[str, ...]


class RecordingStore(FakeStore):
    """FakeStore that logs every call as (op, ns, ...) and can run hooks / inject one failure."""

    def __init__(self, *, yield_reads: bool = False) -> None:
        super().__init__()
        self.calls: list[Call] = []
        self.yield_reads = yield_reads
        self.fail_once: dict[Call, Exception] = {}
        self.after: dict[Call, Callable[[], Awaitable[None]]] = {}

    async def _enter(self, *call: str) -> None:
        self.calls.append(call)
        err = self.fail_once.pop(call, None)
        if err is not None:
            raise err

    async def _exit(self, *call: str) -> None:
        hook = self.after.pop(call, None)
        if hook is not None:
            await hook()

    async def ensure_namespace(self, ns: str, schema: Any) -> None:
        await self._enter("ensure_namespace", ns)
        await super().ensure_namespace(ns, schema)

    async def namespace_exists(self, ns: str) -> bool:
        await self._enter("namespace_exists", ns)
        return await super().namespace_exists(ns)

    async def delete_namespace(self, ns: str) -> None:
        await self._enter("delete_namespace", ns)
        await super().delete_namespace(ns)

    async def branch_from(self, ns: str, source: str) -> None:
        await self._enter("branch_from", ns, source)
        await super().branch_from(ns, source)
        await self._exit("branch_from", ns, source)

    async def upsert(
        self, ns: str, rows: Sequence[Row], *, if_absent: bool = False, condition: Filter | None = None
    ) -> int:
        await self._enter("upsert", ns)
        return await super().upsert(ns, rows, if_absent=if_absent, condition=condition)

    async def get_by_ids(self, ns: str, ids: Sequence[str], **kw: Any) -> list[Row]:
        await self._enter("get_by_ids", ns)
        rows = await super().get_by_ids(ns, ids, **kw)
        if self.yield_reads:
            await asyncio.sleep(0)  # all callers read "absent" before any of them writes
        return rows

    def ops(self, *names: str) -> list[Call]:
        return [c for c in self.calls if c[0] in names]


def make(store: RecordingStore | None = None, **kw: Any) -> tuple[RecordingStore, GatewayService]:
    store = store or RecordingStore()
    return store, GatewayService(store, FakeClock(), MASTER, **kw)


def names(tenant: str = "acme", branch: str = BR_NEW) -> tuple[str, str]:
    ns = ids.namespace(tenant, REPO, branch)
    return ns, ids.tree_namespace(ns)


def begin_req(branch: str = BR_NEW, base: str | None = None) -> SyncBeginReq:
    return SyncBeginReq(repo_id=REPO, branch_id=branch, base_branch_id=base, root="0" * 64, expected_chunks=0)


async def seed_pair(
    store: FakeStore, branch: str, root_hash: str, chunks: list[str], tenant: str = "acme"
) -> None:
    """A synced base branch: chunk rows in `{ns}`, then the root row in `{ns}__tree` (the commit point)."""
    ns, tree = names(tenant, branch)
    await store.ensure_namespace(ns, CHUNK_SCHEMA)
    await store.ensure_namespace(tree, TREE_SCHEMA)
    await store.upsert(ns, [Row(id=c, attrs={"kind": "chunk"}, vector=[0.1] * 1024) for c in chunks])
    root = {"kind": "root", "root_hash": root_hash, "expected_chunks": 4, "written_chunks": 4, "gen": 1}
    await store.upsert(tree, [Row(id=ROOT_ID, attrs=root | {"commit": "abc"})])


async def chunk_ids(store: FakeStore, ns: str) -> set[str]:
    return {r.id for r in await store.get_by_ids(ns, ["c1", "c2", "c3"])}


# --- team key ------------------------------------------------------------------------------------------


async def test_team_key_concurrent_first_calls_converge_on_one_key():
    """FR-G1: ten simultaneous first calls (reads interleave with writes) end with ONE stored key."""
    store, svc = make(RecordingStore(yield_reads=True))
    gw = svc.for_tenant("acme")
    got = await asyncio.gather(*(gw.team_key() for _ in range(10)))
    assert len({bytes(k) for k in got}) == 1
    assert len(got[0]) == keys.TEAM_KEY_LEN
    assert len(await store.list_namespaces(KEYS_NAMESPACE)) == 1
    assert len(store._ns[KEYS_NAMESPACE].rows) == 1  # pyright: ignore[reportPrivateUsage]
    assert await gw.team_key() == got[0]


async def test_team_key_is_wrapped_at_rest_and_bound_to_tenant():
    store, svc = make()
    k_acme = await svc.for_tenant("acme").team_key()
    k_beta = await svc.for_tenant("beta").team_key()
    assert k_acme != k_beta
    rows = store._ns[KEYS_NAMESPACE].rows  # pyright: ignore[reportPrivateUsage]
    assert len(rows) == 2
    for stored in rows.values():
        assert k_acme not in base64.b64decode(stored.attrs["wrapped_k_team"])
    # a second process with the same master key unwraps the same key; another master key cannot
    assert await GatewayService(store, FakeClock(), MASTER).for_tenant("acme").team_key() == k_acme
    with pytest.raises(ScoutError) as ei:
        await GatewayService(store, FakeClock(), b"x" * 32).for_tenant("acme").team_key()
    assert ei.value.code == ErrorCode.INTERNAL


async def test_team_key_wrapped_blob_moved_between_tenants_is_refused():
    store, svc = make()
    await svc.for_tenant("acme").team_key()
    await svc.for_tenant("beta").team_key()
    rows = store._ns[KEYS_NAMESPACE].rows  # pyright: ignore[reportPrivateUsage]
    (acme_id,) = [i for i, r in rows.items() if r.attrs["tenant"] == "acme"]
    (beta_id,) = [i for i, r in rows.items() if r.attrs["tenant"] == "beta"]
    rows[beta_id].attrs["wrapped_k_team"] = rows[acme_id].attrs["wrapped_k_team"]  # AAD=acme, read as beta
    with pytest.raises(ScoutError) as ei:
        await svc.for_tenant("beta").team_key()
    assert ei.value.code == ErrorCode.INTERNAL


async def test_keys_namespace_is_injected_for_every_key_read_and_write():
    """Parent note (3.1 review): a non-default keys namespace is used for all reads and writes."""
    store, svc = make(keys_namespace="scout_dev_keys")
    gw = svc.for_tenant("acme")
    first = await gw.team_key()
    assert await gw.team_key() == first
    touched = {c[1] for c in store.calls}
    assert touched == {"scout_dev_keys"}
    assert not await store.namespace_exists(KEYS_NAMESPACE)


def test_core_service_never_reads_environment():
    import inspect

    from scout.core.service import gateway

    src = inspect.getsource(gateway)
    assert "import os" not in src
    assert "os.environ" not in src
    assert "getenv" not in src


async def test_for_tenant_accepts_an_authenticated_tenant_and_rejects_bad_names():
    _, svc = make()
    info = await svc.for_tenant(Tenant(tenant="acme", key_id="k1")).whoami()
    assert (info.tenant, info.key_id) == ("acme", "k1")
    for bad in ("", "acme_x", "ac me", "a/b", "x" * 200):
        with pytest.raises(ScoutError) as ei:
            svc.for_tenant(bad)
        assert ei.value.code == ErrorCode.INVALID_ARGUMENT


def test_gateway_port_is_structurally_satisfied():
    _, svc = make()
    gw: Gateway = svc.for_tenant("acme")  # pyright checks the structural match
    assert gw is not None


# --- namespace naming ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("repo", "branch"),
    [
        ("A" * 32, BR_NEW),  # uppercase hex
        (REPO, "c" * 15),  # wrong length
        (REPO, "c" * 17),
        ("scout_acme_x", BR_NEW),  # client-chosen namespace string
        (REPO, "../" + "c" * 13),
        (REPO, "c" * 15 + "g"),
        (REPO, ""),
    ],
)
async def test_client_ids_must_be_exact_hex_and_never_reach_the_store(repo: str, branch: str):
    store, svc = make()
    gw = svc.for_tenant("acme")
    with pytest.raises(ScoutError) as ei:
        await gw.sync_begin(
            SyncBeginReq.model_construct(repo_id=repo, branch_id=branch, root="0" * 64, expected_chunks=0)
        )
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError):
        await gw.index_status(StatusReq.model_construct(repo_id=repo, branch_id=branch))
    with pytest.raises(ScoutError):
        await gw.delete_branch(repo, branch)
    assert store.calls == []


async def test_base_branch_id_is_validated_too():
    store, svc = make()
    with pytest.raises(ScoutError) as ei:
        await svc.for_tenant("acme").sync_begin(
            SyncBeginReq.model_construct(
                repo_id=REPO, branch_id=BR_NEW, base_branch_id="x", root="0" * 64, expected_chunks=0
            )
        )
    assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert store.calls == []


async def test_namespaces_derive_from_the_authenticated_tenant_only():
    store, svc = make()
    await svc.for_tenant("acme").sync_begin(begin_req())
    await svc.for_tenant("beta").sync_begin(begin_req())
    assert sorted(store._ns) == sorted([*names("acme"), *names("beta")])  # pyright: ignore[reportPrivateUsage]


# --- sync_begin ------------------------------------------------------------------------------------------


async def test_begin_creates_both_namespaces_once_with_the_section_7_schemas():
    store, svc = make()
    gw = svc.for_tenant("acme")
    ns, tree = names()
    res = await gw.sync_begin(begin_req())
    assert (res.ns_state, res.server_root, res.coverage) == ("empty", None, 0.0)
    assert store._ns[ns].schema == CHUNK_SCHEMA
    assert store._ns[ns].schema.dims == 1024
    assert store._ns[tree].schema == TREE_SCHEMA
    assert store._ns[tree].schema.dims is None
    assert META_SCHEMA.dims is None
    assert store.ops("ensure_namespace") == [("ensure_namespace", ns), ("ensure_namespace", tree)]
    again = await gw.sync_begin(begin_req())  # idempotent
    assert again == res
    assert sorted(store._ns) == sorted([ns, tree])


async def test_begin_reports_populated_with_server_root_and_coverage():
    store, svc = make()
    await seed_pair(store, BR_NEW, "r" * 64, ["c1"])
    res = await svc.for_tenant("acme").sync_begin(begin_req())
    assert (res.ns_state, res.server_root, res.coverage) == ("populated", "r" * 64, 1.0)


async def test_branch_creation_copies_tree_before_chunks_in_exact_order():
    """AC-3.10 server half, ordering invariant (5.3): tree first, then chunks; nothing else mutates."""
    store, svc = make()
    base_ns, base_tree = names(branch=BR_BASE)
    await seed_pair(store, BR_BASE, "r" * 64, ["c1", "c2"])
    store.calls.clear()
    ns, tree = names()
    res = await svc.for_tenant("acme").sync_begin(begin_req(base=BR_BASE))
    assert (res.ns_state, res.server_root) == ("populated", "r" * 64)
    assert store.ops("ensure_namespace", "delete_namespace", "branch_from", "upsert") == [
        ("branch_from", tree, base_tree),
        ("branch_from", ns, base_ns),
    ]
    assert await chunk_ids(store, ns) == {"c1", "c2"}
    assert await chunk_ids(store, base_ns) == {"c1", "c2"}  # the base is untouched


async def test_begin_with_absent_base_creates_empty_pair_and_never_branches():
    store, svc = make()
    res = await svc.for_tenant("acme").sync_begin(begin_req(base=BR_BASE))
    assert res.ns_state == "empty"
    assert not store.ops("branch_from")


async def test_begin_with_existing_ns_ignores_base():
    store, svc = make()
    await seed_pair(store, BR_BASE, "b" * 64, ["c1"])
    await seed_pair(store, BR_NEW, "n" * 64, ["c2"])
    store.calls.clear()
    res = await svc.for_tenant("acme").sync_begin(begin_req(base=BR_BASE))
    assert res.server_root == "n" * 64
    assert not store.ops("branch_from", "delete_namespace")


async def test_ordering_a_commit_between_the_two_copies_leaves_old_root_with_newer_chunks():
    """Hook between the tree copy and the chunk copy commits to the base pair (chunks, then root)."""
    store, svc = make()
    base_ns, base_tree = names(branch=BR_BASE)
    ns, tree = names()
    await seed_pair(store, BR_BASE, "old" + "0" * 61, ["c1"])

    async def commit_to_base() -> None:
        await store.upsert(base_ns, [Row(id="c2", attrs={"kind": "chunk"}, vector=[0.2] * 1024)])  # chunks
        await store.upsert(
            base_tree, [Row(id=ROOT_ID, attrs={"kind": "root", "root_hash": "new" + "0" * 61})]
        )

    store.after[("branch_from", tree, base_tree)] = commit_to_base
    res = await svc.for_tenant("acme").sync_begin(begin_req(base=BR_BASE))
    assert (res.ns_state, res.server_root) == ("populated", "old" + "0" * 61)
    assert await chunk_ids(store, ns) == {"c1", "c2"}  # newer chunks, older root: the next diff re-syncs
    (root,) = await store.get_by_ids(base_tree, [ROOT_ID])
    assert root.attrs["root_hash"] == "new" + "0" * 61  # the base itself advanced


async def test_orphan_tree_after_crash_between_copies_is_deleted_and_recopied():
    """Crash after the first copy: the orphan `{ns}__tree` is never `populated`; the retry repairs it."""
    store, svc = make()
    base_ns, base_tree = names(branch=BR_BASE)
    ns, tree = names()
    await seed_pair(store, BR_BASE, "r" * 64, ["c1"])
    gw = svc.for_tenant("acme")
    store.fail_once[("branch_from", ns, base_ns)] = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "crash")
    with pytest.raises(ScoutError):
        await gw.sync_begin(begin_req(base=BR_BASE))
    assert await store.namespace_exists(tree)
    assert not await store.namespace_exists(ns)
    store.calls.clear()
    res = await gw.sync_begin(begin_req(base=BR_BASE))
    assert (res.ns_state, res.server_root) == ("populated", "r" * 64)
    assert store.ops("delete_namespace", "branch_from") == [
        ("delete_namespace", tree),
        ("branch_from", tree, base_tree),
        ("branch_from", ns, base_ns),
    ]
    assert await chunk_ids(store, ns) == {"c1"}


async def test_orphan_tree_without_a_base_is_dropped_so_its_stale_root_is_never_reported():
    store, svc = make()
    ns, tree = names()
    await store.ensure_namespace(tree, TREE_SCHEMA)
    await store.upsert(tree, [Row(id=ROOT_ID, attrs={"kind": "root", "root_hash": "s" * 64})])
    res = await svc.for_tenant("acme").sync_begin(begin_req())
    assert (res.ns_state, res.server_root) == ("empty", None)
    assert await store.get_by_ids(tree, [ROOT_ID]) == []
    assert await store.namespace_exists(ns)


# --- index_status ----------------------------------------------------------------------------------------


async def test_index_status_absent_then_ready_then_partial():
    store, svc = make()
    gw = svc.for_tenant("acme")
    req = StatusReq(repo_id=REPO, branch_id=BR_NEW)
    assert (await gw.index_status(req)).state == "absent"
    await gw.sync_begin(begin_req())
    assert (await gw.index_status(req)).state == "absent"  # namespaces exist, no root row yet
    _, tree = names()
    root = {
        "kind": "root",
        "root_hash": "r" * 64,
        "expected_chunks": 10,
        "written_chunks": 9,
        "commit": "abc",
    }
    await store.upsert(tree, [Row(id=ROOT_ID, attrs=root)])
    st = await gw.index_status(req)
    assert (st.state, st.coverage, st.semantic_enabled, st.commit) == ("ready", 0.9, True, "abc")
    await store.upsert(tree, [Row(id=ROOT_ID, attrs=root | {"written_chunks": 7})])
    st = await gw.index_status(req)
    assert (st.state, st.coverage, st.semantic_enabled) == ("partial", 0.7, False)


# --- delete_branch ---------------------------------------------------------------------------------------


async def test_delete_branch_removes_chunks_before_tree_in_exact_order():
    store, svc = make()
    ns, tree = names()
    await seed_pair(store, BR_NEW, "r" * 64, ["c1"])
    store.calls.clear()
    await svc.for_tenant("acme").delete_branch(REPO, BR_NEW)
    assert store.ops("delete_namespace") == [("delete_namespace", ns), ("delete_namespace", tree)]
    assert not await store.namespace_exists(ns)
    assert not await store.namespace_exists(tree)


async def test_delete_branch_is_idempotent_and_ok_on_a_nonexistent_branch():
    store, svc = make()
    gw = svc.for_tenant("acme")
    await gw.delete_branch(REPO, BR_NEW)  # never existed
    await seed_pair(store, BR_NEW, "r" * 64, ["c1"])
    await gw.delete_branch(REPO, BR_NEW)
    await gw.delete_branch(REPO, BR_NEW)
    assert await store.list_namespaces("scout_") == []


async def test_delete_branch_never_touches_other_tenants_or_branches():
    store, svc = make()
    await seed_pair(store, BR_NEW, "r" * 64, ["c1"], tenant="acme")
    await seed_pair(store, BR_NEW, "r" * 64, ["c1"], tenant="beta")
    await seed_pair(store, BR_BASE, "r" * 64, ["c1"], tenant="acme")
    await svc.for_tenant("acme").delete_branch(REPO, BR_NEW)
    assert sorted(await store.list_namespaces("scout_")) == sorted([*names("beta"), *names("acme", BR_BASE)])


async def test_delete_branch_crash_between_the_deletes_is_finished_by_the_retry():
    store, svc = make()
    gw = svc.for_tenant("acme")
    ns, tree = names()
    await seed_pair(store, BR_NEW, "r" * 64, ["c1"])
    store.fail_once[("delete_namespace", tree)] = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "crash")
    store.calls.clear()
    with pytest.raises(ScoutError):
        await gw.delete_branch(REPO, BR_NEW)
    assert not await store.namespace_exists(ns)
    assert await store.namespace_exists(tree)
    await gw.delete_branch(REPO, BR_NEW)  # retry
    assert not await store.namespace_exists(tree)
    assert store.ops("delete_namespace") == [
        ("delete_namespace", ns),
        ("delete_namespace", tree),
        ("delete_namespace", ns),
        ("delete_namespace", tree),
    ]
    # and the leftover from a crash is also repaired by the next sync_begin on that branch
    await seed_pair(store, BR_NEW, "r" * 64, ["c1"])
    store.fail_once[("delete_namespace", tree)] = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "crash")
    with pytest.raises(ScoutError):
        await gw.delete_branch(REPO, BR_NEW)
    res = await gw.sync_begin(begin_req())
    assert (res.ns_state, res.server_root) == ("empty", None)


def test_unimplemented_methods_fail_loudly_not_silently():
    _, svc = make()
    gw = svc.for_tenant("acme")
    with pytest.raises(NotImplementedError):
        gw.stream(None)  # type: ignore[arg-type]
