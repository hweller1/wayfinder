"""Task 3.9: tenant isolation across read / write / branch / seed. FR-X6, AC-13.1.

Two tenants share one `RecordingStore` (one `FakeStore`) behind one `GatewayService`; tenants come out of
`AuthService.resolve` on real issued keys, never from a request. Tenant B owns a synced branch, a branch
seeded (page 0 of 2, so its proofs row is open and fail-closed), a meta namespace (sketches, proofs), a cache
namespace and key records. Tenant A then attacks with B's exact ids (B's `repo_id`/`branch_id` values are
also A's), with B's sealed values (tags, path ciphertexts, chunk keys, node ids, sketch, leaf hashes), and
with namespace names smuggled in as ids.

After EVERY attempt:
- B's namespaces (chunks, `__tree`, meta, cache) and B's key records are byte-identical to the snapshot taken
  before (full `FakeStore` contents: schema, vector-ness, every row's attrs and vector);
- every namespace the store was asked about for A's call has A's exact tenant segment and an expected shape
  (`scout_{A}_meta|cache|{repo}_{branch}[__tree]`), and `list_namespaces` prefixes are exactly `scout_{A}_`.
  The segment is compared for equality, never with `startswith` (`scout_acme_` vs `scout_acme-x_`).

Tenant pairs include prefix-sharing names (`acme`/`acme-x`, both directions). Each pair runs with A empty
(its own namespaces absent) and A populated (own data under the SAME ids, different content), so a request
that wrongly resolved to B's namespace would show B's data or change B's rows.
"""

import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pydantic
import pytest
from svc_helpers import (
    NEW_BR,
    REPO,
    SRC_BR,
    Clone,
    RecordingStore,
    edited,
    make,
    ns_pair,
    src,
    text_key,
)

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import (
    CommitReq,
    EmbedReq,
    NodesReq,
    NodeUpsert,
    PlanReq,
    SearchReq,
    SearchScope,
    SeedReq,
    StageNodesReq,
    StatusReq,
    SyncBeginReq,
)
from scout.core.ports.chat import ChatRequest, ChatTurn
from scout.core.service.auth import AuthService
from scout.core.service.gateway import GatewayService
from scout.core.service.logging import LOGGER
from scout.core.service.schema import ROOT_ID
from scout.core.service.seedsvc import DEFAULT_THRESHOLD
from scout.core.service.syncsvc import cache_namespace, meta_namespace, sketch_row_id
from scout.core.service.tenants import KEYS_NAMESPACE, Tenant, TenantDirectory

PEPPER = b"isolation-pepper-0123456789"
OTHER_BR = "d" * 16
SEED_BR = "e" * 16
PAIRS = [("acme", "beta"), ("acme", "acme-x"), ("acme-x", "acme")]  # (A the attacker, B the victim)
VARIANTS = ["a_empty", "a_populated"]
_NS_SHAPE = re.compile(r"^scout_([A-Za-z0-9\-]+)_(meta|cache|[0-9a-f]{32}_[0-9a-f]{16}(?:__tree)?)$")


# --- isolation checkers -----------------------------------------------------------------------------------


def tenant_of(ns: str) -> str | None:
    """The tenant segment of a `scout_{tenant}_{rest}` name (tenants contain no `_`), else None."""
    m = _NS_SHAPE.match(ns)
    return m.group(1) if m else None


def snapshot(store: RecordingStore, tenant: str) -> str:
    """Canonical text of everything `tenant` owns: its namespaces in full plus its key records."""
    owned = {n: store.dump(n) for n in store.names() if tenant_of(n) == tenant}
    keys_ns = store.dump(KEYS_NAMESPACE)["rows"] if KEYS_NAMESPACE in store.names() else {}
    key_rows = {i: v for i, v in keys_ns.items() if v[0].get("tenant") == tenant}
    return repr((owned, key_rows))


def assert_only_own_namespaces(store: RecordingStore, tenant: str) -> None:
    """Every namespace a store call named belongs to `tenant` (exact segment equality), and no other."""
    for op, name in store.touched:
        if op == "list_namespaces":
            assert name == f"scout_{tenant}_", f"{op} scanned prefix {name!r}"
        else:
            assert tenant_of(name) == tenant, f"{op} touched {name!r}, which is not tenant {tenant!r}'s"


async def denied(call: Awaitable[Any], *codes: ErrorCode) -> ScoutError:
    with pytest.raises(ScoutError) as ei:
        await call
    assert ei.value.code in codes, (ei.value.code, ei.value.message)
    return ei.value


@contextmanager
def request_logs():
    """The service's request log records (INFO, which pytest does not capture by default) as text."""
    seen: list[str] = []

    class Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    handler, level = Grab(), LOGGER.level
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    try:
        yield seen
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(level)


# --- the world: tenant B's data, tenant A's attacker -----------------------------------------------------


@dataclass
class World:
    store: RecordingStore
    svc: GatewayService
    chat: ScriptedChat
    a: Clone  # the attacker; its gateway is bound to the tenant that `AuthService` resolved
    b: Clone  # the victim's synced branch (REPO, SRC_BR)
    b_seeded: Clone  # the victim's branch (REPO, NEW_BR): seeded, proofs still open
    a_tenant: Tenant
    b_tenant: Tenant
    a_files: dict[str, str]
    b_files: dict[str, str]
    a_populated: bool

    @property
    def A(self) -> str:
        return self.a_tenant.tenant

    @property
    def B(self) -> str:
        return self.b_tenant.tenant

    def b_rows(self, ns: str) -> list[Any]:
        return list(self.store.rows(ns).values())

    @property
    def b_chunks(self) -> list[Any]:
        return self.b_rows(self.b.ns)

    @property
    def b_root_hash(self) -> str:
        return self.store.rows(self.b.tree_ns)[ROOT_ID].attrs["root_hash"]

    @property
    def b_node_ids(self) -> list[str]:
        rows = self.store.rows(self.b.tree_ns)
        return [i for i, r in rows.items() if r.attrs.get("kind") == "node"]

    @property
    def b_sketch(self) -> str:
        return self.store.rows(meta_namespace(self.B))[sketch_row_id(self.b.ns)].attrs["sketch"]

    def a_hashes(self) -> set[str]:
        return self.a.file_hashes(self.a_files)

    def b_hashes(self) -> set[str]:
        return {r.attrs["file_hash"] for r in self.b_chunks}


async def build_world(a_tenant: str, b_tenant: str, a_populated: bool) -> World:
    chat = ScriptedChat([ChatTurn(content="ok", tool_calls=(), tokens_in=3, tokens_out=2, finish="stop")] * 4)
    store, svc, _ = make(chat=chat)
    directory = TenantDirectory(store, FakeClock(), PEPPER)
    auth = AuthService(directory, FakeClock())
    tenants = {t: await auth.resolve(await directory.issue_key(t)) for t in (a_tenant, b_tenant)}

    async def clone(tenant: str, branch: str) -> Clone:
        c = await Clone.create(svc, store, tenant, branch)
        c.gw = svc.for_tenant(tenants[tenant])  # the tenant AuthService resolved, never a request field
        return c

    b_files, a_files = src(12, "b-secret"), src(12, "a-own")
    b = await clone(b_tenant, SRC_BR)
    await b.sync(b_files)
    b_seeded = await clone(b_tenant, NEW_BR)
    leaves = b_seeded.leaves(edited(b_files, 1))
    res = await b_seeded.gw.seed(b_seeded.seed_req(edited(b_files, 1), page=0, pages=2, proofs=leaves))
    assert res.seeded
    assert not res.complete
    await b.gw.complete(ChatRequest(messages=[{"role": "user", "content": "B's prompt"}]))
    a = await clone(a_tenant, SRC_BR)
    if a_populated:  # same repo_id and branch_ids as B, different content (and different keys)
        await a.sync(a_files)
        await (await clone(a_tenant, NEW_BR)).sync(a_files)
    world = World(
        store, svc, chat, a, b, b_seeded, tenants[a_tenant], tenants[b_tenant], a_files, b_files, a_populated
    )
    assert len(snapshot(store, b_tenant)) > 10_000  # chunks, tree, meta, cache, key rows: not vacuous
    for ns in (
        b.ns,
        b.tree_ns,
        b_seeded.ns,
        b_seeded.tree_ns,
        meta_namespace(b_tenant),
        cache_namespace(b_tenant),
    ):
        assert tenant_of(ns) == b_tenant
        assert await store.namespace_exists(ns)
    return world


# --- read -------------------------------------------------------------------------------------------------


async def search_with_b_ids(w: World) -> None:
    """Same ids as B's synced and seeded branches. A sees only its own rows: B's proofs row (open, which
    would hide every hit in a namespace named like B's) is never consulted for A's namespace."""
    for branch in (SRC_BR, NEW_BR):
        req = SearchReq(repo_id=REPO, branch_id=branch, query="fn", k=30, tokens=["t" * 32])
        if not w.a_populated:
            await denied(w.a.gw.search(req), ErrorCode.NOT_FOUND)
            continue
        hits = {h.file_hash for h in (await w.a.gw.search(req)).hits}
        assert hits == w.a_hashes()  # all of A's files: nothing hidden by B's proofs, nothing of B's
        assert not hits & w.b_hashes()


async def search_scope_naming_b(w: World) -> None:
    """`scope` carrying B's own sealed path, ext tag and dir tag, and B's namespace names as globs."""
    row = w.b_chunks[0].attrs
    scopes = [
        SearchScope(path_glob_enc=row["path_enc"]),
        SearchScope(ext_tag=row["ext_tag"], dir_tag=row["dir_tags"][0]),
        SearchScope(path_glob_enc=f"scout_{w.B}_*"),
        SearchScope(path_glob_enc=w.b.ns, dir_tag=w.b.tree_ns),
    ]
    for scope in scopes:
        req = SearchReq(repo_id=REPO, branch_id=SRC_BR, query="fn", k=30, tokens=["t" * 32], scope=scope)
        if not w.a_populated:
            await denied(w.a.gw.search(req), ErrorCode.NOT_FOUND)
            continue
        hits = {h.file_hash for h in (await w.a.gw.search(req)).hits}
        assert not hits & w.b_hashes()
        assert hits <= w.a_hashes()


async def plan_with_b_chunk_keys(w: World) -> None:
    """B's chunk keys are not in A's cache namespace, so none of them is reported as already embedded."""
    keys = [r.attrs["chunk_key"] for r in w.b_chunks]
    assert len(keys) >= 12
    res = await w.a.gw.sync_plan(PlanReq(repo_id=REPO, branch_id=SRC_BR, chunk_keys=keys))
    assert res.missing == list(dict.fromkeys(keys))


async def nodes_of_b(w: World) -> None:
    """B's real node ids, asked for through A's request: nothing of B's tree comes back."""
    b_blobs = {r.attrs["children_enc"] for r in w.b_rows(w.b.tree_ns) if r.attrs.get("kind") == "node"}
    assert b_blobs
    res = await w.a.gw.sync_nodes(NodesReq(repo_id=REPO, branch_id=SRC_BR, ids=w.b_node_ids[:500]))
    assert not set(res.blobs.values()) & b_blobs
    if not w.a_populated:
        assert res.blobs == {}


async def status_of_b_ids(w: World) -> None:
    status = await w.a.gw.index_status(StatusReq(repo_id=REPO, branch_id=SRC_BR))
    if w.a_populated:
        assert status.state == "ready"  # A's own index, which merely lives under the same ids
    else:
        assert status.state == "absent"
        assert status.coverage == 0.0
        assert status.commit is None


async def begin_with_b_root(w: World) -> None:
    """`sync_begin` on B's ids with B's real root hash: the answer is about A's namespace, never B's."""
    req = SyncBeginReq(repo_id=REPO, branch_id=SRC_BR, root=w.b_root_hash, expected_chunks=12)
    res = await w.a.gw.sync_begin(req)
    assert res.server_root != w.b_root_hash
    assert res.ns_state == ("populated" if w.a_populated else "empty")
    if not w.a_populated:
        assert res.server_root is None


async def chat_usage_visibility(w: World) -> None:
    """B already made a chat call. A's call is logged under A's key only, with no content, and the model
    receives exactly A's request."""
    mine = ChatRequest(messages=[{"role": "user", "content": "A's prompt"}])
    with request_logs() as seen:
        turn = await w.a.gw.complete(mine)
    assert turn.content == "ok"
    assert len(seen) == 1
    assert f"key={w.a_tenant.key_id}" in seen[0]
    assert "op=chat" in seen[0]
    assert w.b_tenant.key_id not in seen[0]
    assert "prompt" not in seen[0]
    assert w.chat.requests[-1].messages == mine.messages
    assert all("A's prompt" not in str(m) for r in w.chat.requests[:-1] for m in r.messages)


# --- write ------------------------------------------------------------------------------------------------


async def embed_into_b_ids(w: World) -> None:
    """Embed to B's ids, naming B's file ids in `delete_files` and `replace`: only A's namespace can move."""
    text = "def fn_mine():\n    return 1\n"
    row = w.a.wire_row("src/mine.py", text)
    b_file_ids = sorted({r.attrs["file_id"] for r in w.b_chunks})
    from scout.core.contracts.wire import ReplaceFile

    req = EmbedReq(
        repo_id=REPO,
        branch_id=SRC_BR,
        texts={text_key(text): text},
        rows=[row],
        replace=[ReplaceFile(file_id=f, file_hash="0" * 64) for f in b_file_ids[:3]],
        delete_files=b_file_ids,
    )
    if not w.a_populated:
        await denied(w.a.gw.embed(req), ErrorCode.NOT_FOUND)
        assert w.a.ns not in w.store.names()
        return
    before = set(w.store.rows(w.a.ns))
    assert (await w.a.gw.embed(req)).written == 1
    assert set(w.store.rows(w.a.ns)) == before | {row.id}  # landed in A's own namespace


async def embed_rows_resolved_from_b_cache(w: World) -> None:
    """Rows naming B's chunk keys with no text must not resolve through B's embed cache."""
    key = w.b_chunks[0].attrs["chunk_key"]
    row = w.a.wire_row("src/mine.py", "x = 1\n").model_copy(update={"chunk_key": key})
    req = EmbedReq(repo_id=REPO, branch_id=SRC_BR, texts={}, rows=[row])
    expected = ErrorCode.CONFLICT if w.a_populated else ErrorCode.NOT_FOUND
    await denied(w.a.gw.embed(req), expected)


async def commit_with_b_root(w: World) -> None:
    req = CommitReq(
        repo_id=REPO, branch_id=SRC_BR, expected_root=w.b_root_hash, new_root="f" * 64, written_chunks=1
    )
    if not w.a_populated:
        await denied(w.a.gw.sync_commit(req), ErrorCode.NOT_FOUND)
        assert w.a.ns not in w.store.names()
        return
    err = await denied(w.a.gw.sync_commit(req), ErrorCode.CONFLICT)
    assert err.details["server_root"] != w.b_root_hash  # the conflict reports A's root, never B's
    assert err.details["server_root"] == w.store.rows(w.a.tree_ns)[ROOT_ID].attrs["root_hash"]


async def stage_nodes_into_b(w: World) -> None:
    """A stages node pages at B's ids, naming B's root as `base_root`/`new_root` and B's node ids as rows and
    as deletes; then commits with `staged_pages` over them. Only A's own tree can move."""
    b_id = w.b_node_ids[0]
    row = NodeUpsert(id=b_id, node_hash="f" * 64, children_enc="dGFtcGVy")
    forge = {"repo_id": REPO, "branch_id": SRC_BR, "pages": 1}
    pages = [
        StageNodesReq(base_root=w.b_root_hash, new_root="f" * 64, page=0, node_upserts=[row], **forge),
        StageNodesReq(base_root=None, new_root="f" * 64, page=0, node_upserts=[row], **forge),
        StageNodesReq(base_root=w.b_root_hash, new_root=w.b_root_hash, page=0, node_upserts=[row], **forge),
        StageNodesReq(base_root=None, new_root="e" * 64, page=0, node_deletes=w.b_node_ids[:50], **forge),
    ]
    for req in pages:
        if not w.a_populated:
            await denied(w.a.gw.sync_stage_nodes(req), ErrorCode.NOT_FOUND)
        else:
            await denied(w.a.gw.sync_stage_nodes(req), ErrorCode.CONFLICT)  # A's root is neither base nor new
    commit = CommitReq(
        repo_id=REPO,
        branch_id=SRC_BR,
        expected_root=w.b_root_hash,
        new_root=w.b_root_hash,
        staged_pages=1,
        written_chunks=1,
    )
    await denied(w.a.gw.sync_commit(commit), ErrorCode.NOT_FOUND if not w.a_populated else ErrorCode.CONFLICT)
    if not w.a_populated:
        assert w.a.ns not in w.store.names()
        assert w.a.tree_ns not in w.store.names()


async def delete_b_branches(w: World) -> None:
    """`delete_branch` on B's ids removes at most A's own pair."""
    for branch in (SRC_BR, NEW_BR):
        assert await w.a.gw.delete_branch(REPO, branch) is None
    assert ns_pair(w.A, SRC_BR)[0] not in w.store.names()
    assert w.b.ns in w.store.names()
    assert w.b_seeded.tree_ns in w.store.names()


# --- branch -----------------------------------------------------------------------------------------------


async def begin_with_b_base(w: World) -> None:
    """`sync_begin` of a new branch with B's `base_branch_id`: a base can only be A's own namespace pair."""
    new_ns, new_tree = ns_pair(w.A, OTHER_BR)
    for base in (SRC_BR, NEW_BR):
        await w.store.delete_namespace(new_ns)  # a fresh branch each round
        await w.store.delete_namespace(new_tree)
        req = SyncBeginReq(
            repo_id=REPO, branch_id=OTHER_BR, base_branch_id=base, root="0" * 64, expected_chunks=12
        )
        res = await w.a.gw.sync_begin(req)
        a_base_ns, a_base_tree = ns_pair(w.A, base)
        if w.a_populated:
            assert res.ns_state == "populated"
            assert w.store.dump(new_ns)["rows"] == w.store.dump(a_base_ns)["rows"]  # A's rows, copied
            assert w.store.dump(new_tree)["rows"] == w.store.dump(a_base_tree)["rows"]
        else:
            assert res.ns_state == "empty"
            assert res.server_root is None
            assert w.store.rows(new_ns) == {}
            assert w.store.rows(new_tree) == {}
    branched = w.store.ops("branch_from")
    assert all(tenant_of(c[1]) == w.A and tenant_of(c[2]) == w.A for c in branched)
    assert bool(branched) == w.a_populated


# --- seed -------------------------------------------------------------------------------------------------


async def seed_with_b_sketch(w: World) -> None:
    """A presents B's exact stored sketch and B's leaf hashes as proofs: B's sketch is never a candidate
    (A's scan reads only A's meta namespace), so nothing is copied and A's branch is not created."""
    leaves = w.b.leaves(w.b_files)
    req = SeedReq(repo_id=REPO, branch_id=SEED_BR, minhash_b64=w.b_sketch, proofs=leaves)
    res = await w.a.gw.seed(req)
    assert res.seeded is False
    assert res.jaccard is None or res.jaccard < DEFAULT_THRESHOLD
    if not w.a_populated:
        assert res.jaccard is None  # A has no sketches at all, and B's are invisible
    assert ns_pair(w.A, SEED_BR)[0] not in w.store.names()
    assert w.store.ops("branch_from") == []


async def seed_pages_into_b_seed(w: World) -> None:
    """B's seed upload is open at (REPO, NEW_BR). A cannot continue it, and a page 0 of A's own never
    resumes or reuses B's proofs row."""
    leaves = w.b_seeded.leaves(edited(w.b_files, 1))
    page1 = SeedReq(repo_id=REPO, branch_id=NEW_BR, minhash_b64=w.b_sketch, proofs=leaves, page=1, pages=2)
    await denied(w.a.gw.seed(page1), ErrorCode.CONFLICT)  # "no seed in progress for this branch"
    page0 = SeedReq(repo_id=REPO, branch_id=NEW_BR, minhash_b64=w.b_sketch, proofs=leaves, page=0, pages=2)
    res = await w.a.gw.seed(page0)
    assert res.seeded is False  # nothing of B's to match; A's own namespace (if any) is already committed
    assert w.store.ops("branch_from") == []


# --- namespace names never come from the client -----------------------------------------------------------


async def smuggled_names(w: World) -> None:
    """Namespace names, `__tree` names and path tricks in id fields (past the wire regex, via
    `model_construct`, as a buggy surface might) are rejected by the service before any store call."""
    bad_values = [w.b.ns, w.b.tree_ns, f"{SRC_BR}__tree", f"../{w.b.ns}", meta_namespace(w.B), w.B, ""]
    gw = w.a.gw
    for bad in bad_values:
        for repo, branch in ((bad, SRC_BR), (REPO, bad)):
            calls: list[Awaitable[Any]] = [
                gw.search(SearchReq.model_construct(repo_id=repo, branch_id=branch, query="fn")),
                gw.sync_nodes(NodesReq.model_construct(repo_id=repo, branch_id=branch, ids=[])),
                gw.index_status(StatusReq.model_construct(repo_id=repo, branch_id=branch)),
                gw.sync_plan(PlanReq.model_construct(repo_id=repo, branch_id=branch, chunk_keys=[])),
                gw.sync_commit(
                    CommitReq.model_construct(repo_id=repo, branch_id=branch, new_root="f", written_chunks=0)
                ),
                gw.sync_stage_nodes(
                    StageNodesReq.model_construct(
                        repo_id=repo, branch_id=branch, new_root="f", page=0, pages=1, node_upserts=[]
                    )
                ),
                gw.embed(EmbedReq.model_construct(repo_id=repo, branch_id=branch, texts={}, rows=[])),
                gw.seed(SeedReq.model_construct(repo_id=repo, branch_id=branch, minhash_b64="", proofs=[])),
                gw.sync_begin(
                    SyncBeginReq.model_construct(repo_id=repo, branch_id=branch, root="0", expected_chunks=0)
                ),
                gw.delete_branch(repo, branch),
            ]
            for call in calls:
                await denied(call, ErrorCode.INVALID_ARGUMENT)
        await denied(  # a bad base id, with a valid repo and branch
            gw.sync_begin(
                SyncBeginReq.model_construct(
                    repo_id=REPO, branch_id=OTHER_BR, base_branch_id=bad, root="0", expected_chunks=0
                )
            ),
            ErrorCode.INVALID_ARGUMENT,
        )
    for bad in bad_values[:5]:  # and the wire models themselves refuse them
        with pytest.raises(pydantic.ValidationError):
            SearchReq(repo_id=bad, branch_id=SRC_BR, query="fn")
        with pytest.raises(pydantic.ValidationError):
            SyncBeginReq(repo_id=REPO, branch_id=OTHER_BR, base_branch_id=bad, root="0", expected_chunks=0)
    assert w.store.touched == []  # all of that was rejected before the store was asked anything


ATTACKS: dict[str, Callable[[World], Awaitable[None]]] = {
    f.__name__: f
    for f in (
        # read
        search_with_b_ids,
        search_scope_naming_b,
        plan_with_b_chunk_keys,
        nodes_of_b,
        status_of_b_ids,
        begin_with_b_root,
        chat_usage_visibility,
        # write
        embed_into_b_ids,
        embed_rows_resolved_from_b_cache,
        commit_with_b_root,
        stage_nodes_into_b,
        delete_b_branches,
        # branch
        begin_with_b_base,
        # seed
        seed_with_b_sketch,
        seed_pages_into_b_seed,
        # names
        smuggled_names,
    )
}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}-vs-{p[1]}")
@pytest.mark.parametrize("attack", ATTACKS)
async def test_tenant_a_cannot_reach_tenant_b(attack: str, pair: tuple[str, str], variant: str):
    """AC-13.1, FR-X6: every attempt fails or is answered from A's own namespaces; B is byte-identical."""
    w = await build_world(pair[0], pair[1], variant == "a_populated")
    before = snapshot(w.store, w.B)
    w.store.touched.clear()
    w.store.calls.clear()
    await ATTACKS[attack](w)
    assert snapshot(w.store, w.B) == before
    assert_only_own_namespaces(w.store, w.A)


# --- the checkers themselves ------------------------------------------------------------------------------


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}-vs-{p[1]}")
async def test_checkers_catch_a_leak_and_a_write(pair: tuple[str, str]):
    """Controls: a read of B's namespace, a prefix scan that also matches B, and a write to B's rows are each
    detected, so a green suite is not vacuous (including for the `acme` / `acme-x` prefix pair)."""
    w = await build_world(pair[0], pair[1], True)
    w.store.touched.clear()
    await w.store.get_by_ids(w.b.ns, [])
    with pytest.raises(AssertionError):
        assert_only_own_namespaces(w.store, w.A)
    w.store.touched.clear()
    await w.store.list_namespaces(f"scout_{w.A[:2]}")  # a looser prefix than `scout_{A}_`
    with pytest.raises(AssertionError):
        assert_only_own_namespaces(w.store, w.A)
    w.store.touched.clear()
    await w.a.gw.index_status(StatusReq(repo_id=REPO, branch_id=SRC_BR))
    assert_only_own_namespaces(w.store, w.A)  # while an honest call passes

    before = snapshot(w.store, w.B)
    victim = w.b_chunks[0]
    victim.attrs["loc_enc"] = "tampered"
    assert snapshot(w.store, w.B) != before
    victim.attrs["loc_enc"] = "bG9j"
    assert snapshot(w.store, w.B) == before
    await w.store.delete_namespace(meta_namespace(w.B))
    assert snapshot(w.store, w.B) != before


def test_tenant_segment_is_compared_for_equality_not_prefix():
    assert tenant_of("scout_acme-x_meta") == "acme-x" != "acme"
    assert tenant_of(ns_pair("acme", SRC_BR)[1]) == "acme"
    assert tenant_of("scout_acme_x_meta") is None  # an underscore name has no valid tenant segment
    assert tenant_of(KEYS_NAMESPACE) is None


@pytest.mark.parametrize("tenant", ["acme_x", "acme_", "_acme", "acme x", "acme/x", "", "a" * 200])
async def test_tenant_names_that_could_alias_another_tenants_prefix_are_rejected(tenant: str):
    """`acme_x` would fall inside tenant `acme`'s `scout_acme_` prefix scan, so such names cannot exist."""
    _, svc, _ = make()
    with pytest.raises(ScoutError) as ei:
        svc.for_tenant(tenant)
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    directory = TenantDirectory(RecordingStore(), FakeClock(), PEPPER)
    with pytest.raises(ScoutError):
        await directory.issue_key(tenant)
