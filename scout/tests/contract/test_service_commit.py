"""Task 3.4: `sync_commit` (CAS on the root row), coverage readiness, proof drop, crash safety.

Covers FR-C9, FR-C13, AC-3.8 (partial/ready at 80%), AC-4.3 second half (proofs dropped on commit), and the
ordering invariant of design 5 (root row strictly after every chunk write; node rows before the root row).
The client side is the real keyed Merkle tree (`build_tree`/`seal_tree`/`diff`), so "the next diff converges"
is checked against what the server stored, not against a stub.
"""

import asyncio
import base64
import hashlib
from collections.abc import Sequence
from typing import Any

import pytest

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import IndexStatus
from scout.core.contracts.wire import (
    CommitReq,
    EmbedReq,
    NodesReq,
    NodeUpsert,
    ReplaceFile,
    StageNodesReq,
    StatusReq,
    SyncBeginReq,
    WireRow,
)
from scout.core.crypto import ids, keys, paths
from scout.core.crypto.minhash import SKETCH_BYTES
from scout.core.ports.store import Eq, Filter, Row
from scout.core.service.gateway import GatewayService, TenantGateway
from scout.core.service.schema import META_SCHEMA, ROOT_ID, TREE_SCHEMA
from scout.core.service.syncsvc import meta_namespace, proofs_row_id, sketch_row_id
from scout.core.sync import merkle

MASTER = b"m" * 32
TENANT = "acme"
REPO = "a" * 32
BR = "c" * 16
BR_BASE = "b" * 16
SKETCH = base64.b64encode(bytes(range(256)) * 2).decode()
Call = tuple[str, ...]

V1 = {"src/a.py": "def alpha_one():\n    return 1\n", "src/b.py": "def beta_one():\n    return 2\n"}


class RecordingStore(FakeStore):
    """Logs every call, injects one failure per (op, ns), runs hooks, snapshots state after each write."""

    def __init__(self, *, yield_ops: bool = False) -> None:
        super().__init__()
        self.calls: list[Call] = []
        self.fail_once: dict[Call, Exception] = {}
        self.after: dict[Call, Any] = {}
        self.yield_ops = yield_ops
        self.snapshot: Any = None
        self.trace: list[Any] = []

    async def _enter(self, *call: str) -> None:
        self.calls.append(call)
        err = self.fail_once.pop(call, None)
        if err is not None:
            raise err
        if self.yield_ops:
            await asyncio.sleep(0)

    async def _exit(self, *call: str) -> None:
        if self.snapshot is not None:
            self.trace.append((call, self.snapshot()))
        hook = self.after.pop(call, None)
        if hook is not None:
            await hook()

    async def branch_from(self, ns: str, source: str) -> None:
        await self._enter("branch_from", ns, source)
        await super().branch_from(ns, source)
        await self._exit("branch_from", ns, source)

    async def upsert(
        self, ns: str, rows: Sequence[Row], *, if_absent: bool = False, condition: Filter | None = None
    ) -> int:
        label = ("upsert", ns, ROOT_ID if any(r.id == ROOT_ID for r in rows) else "rows")
        await self._enter(*label)
        out = await super().upsert(ns, rows, if_absent=if_absent, condition=condition)
        await self._exit(*label)
        return out

    async def delete_ids(self, ns: str, ids: Sequence[str]) -> int:
        await self._enter("delete_ids", ns)
        out = await super().delete_ids(ns, ids)
        await self._exit("delete_ids", ns)
        return out

    async def delete_by_filter(self, ns: str, flt: Filter) -> int:
        await self._enter("delete_by_filter", ns)
        out = await super().delete_by_filter(ns, flt)
        await self._exit("delete_by_filter", ns)
        return out

    async def get_by_ids(self, ns: str, ids: Sequence[str], **kw: Any) -> list[Row]:
        await self._enter("get_by_ids", ns)
        return await super().get_by_ids(ns, ids, **kw)

    def ops(self, *names: str) -> list[Call]:
        return [c for c in self.calls if c[0] in names]

    def rows(self, ns: str) -> dict[str, Any]:
        cur = self._ns.get(ns)  # pyright: ignore[reportPrivateUsage]
        return cur.rows if cur is not None else {}


def names(branch: str = BR) -> tuple[str, str]:
    ns = ids.namespace(TENANT, REPO, branch)
    return ns, ids.tree_namespace(ns)


NS, TREE = names()
META = meta_namespace(TENANT)


def make(store: RecordingStore | None = None) -> tuple[RecordingStore, GatewayService]:
    store = store or RecordingStore()
    clock = FakeClock()
    return store, GatewayService(store, clock, MASTER, embedder=FakeEmbedder())


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


class Client:
    """One clone: builds the keyed tree, diffs against the server, embeds, commits."""

    def __init__(self, gw: TenantGateway, store: RecordingStore, k: keys.Subkeys, branch: str = BR) -> None:
        self.gw, self.store, self.k, self.branch = gw, store, k, branch
        self.ns, self.tree_ns = names(branch)

    @classmethod
    async def create(cls, svc: GatewayService, store: RecordingStore, branch: str = BR) -> "Client":
        gw = svc.for_tenant(TENANT)
        return cls(gw, store, keys.derive(await gw.team_key()), branch)

    def tree(self, files: dict[str, str]) -> merkle.Tree:
        leaves = [(p, merkle.leaf_hash(self.k, t.encode())) for p, t in files.items()]
        return merkle.build_tree(leaves, self.k, TENANT, REPO)

    def remote_nodes(self, node_ids: Sequence[str]) -> dict[str, merkle.SealedNode]:
        rows = self.store.rows(self.tree_ns)
        return {
            i: merkle.SealedNode(rows[i].attrs["children_enc"], rows[i].attrs["node_hash"])
            for i in node_ids
            if i in rows and rows[i].attrs.get("kind") == "node"
        }

    async def begin(self, tree: merkle.Tree, base: str | None = None) -> str | None:
        res = await self.gw.sync_begin(
            SyncBeginReq(
                repo_id=REPO,
                branch_id=self.branch,
                base_branch_id=base,
                root=tree.root_hash,
                expected_chunks=len(tree.leaves),
            )
        )
        return res.server_root

    def wire_row(self, path: str, text: str) -> WireRow:
        fid, ck = ids.file_id(self.k, path), text_key(text)
        return WireRow(
            id=ids.row_id(self.k, fid, ck, 0),
            file_id=fid,
            path_enc=paths.enc_path(self.k, TENANT, REPO, path),
            ext_tag=ids.ext_tag(self.k, ".py"),
            dir_tags=[ids.dir_tag(self.k, "src")],
            file_hash=merkle.leaf_hash(self.k, text.encode()),
            chunk_key=ck,
            loc_enc="bG9j",
            tokens=["t" * 32],
        )

    def embed_reqs(self, delta: merkle.Delta, files: dict[str, str], per_batch: int = 100) -> list[EmbedReq]:
        todo = [*delta.added, *delta.changed]
        reqs: list[EmbedReq] = []
        for i in range(0, max(len(todo), 1), per_batch):
            part = todo[i : i + per_batch]
            rows = [self.wire_row(p, files[p]) for p in part]
            reqs.append(
                EmbedReq(
                    repo_id=REPO,
                    branch_id=self.branch,
                    texts={text_key(files[p]): files[p] for p in part},
                    rows=rows,
                    # replace for EVERY written file (not only `changed`): a lost commit race can leave
                    # another version of an "added" file behind, and replace is idempotent
                    replace=[ReplaceFile(file_id=r.file_id, file_hash=r.file_hash) for r in rows],
                )
            )
        reqs[-1] = reqs[-1].model_copy(
            update={"delete_files": [ids.file_id(self.k, p) for p in delta.removed]}
        )
        return reqs

    def commit_req(self, tree: merkle.Tree, expected: str | None, **kw: Any) -> CommitReq:
        sealed = merkle.seal_tree(tree, self.k)
        stored = {i for i, r in self.store.rows(self.tree_ns).items() if r.attrs.get("kind") == "node"}
        total = len(tree.leaves)
        fields: dict[str, Any] = {
            "repo_id": REPO,
            "branch_id": self.branch,
            "expected_root": expected,
            "new_root": tree.root_hash,
            "node_upserts": [
                NodeUpsert(id=i, node_hash=n.node_hash, children_enc=n.children_enc)
                for i, n in sealed.items()
            ],
            "node_deletes": sorted(stored - sealed.keys()),
            "written_chunks": total,
            "expected_chunks": total,
        }
        return CommitReq(**(fields | kw))

    def staged(
        self, tree: merkle.Tree, expected: str | None, per_page: int = 2
    ) -> tuple[list[StageNodesReq], CommitReq]:
        """Node pages of `per_page` rows and the matching commit (no inline rows; deletes ride on page 0)."""
        full = self.commit_req(tree, expected)
        rows = full.node_upserts
        chunks = [rows[i : i + per_page] for i in range(0, len(rows), per_page)] or [[]]
        pages = [
            StageNodesReq(
                repo_id=REPO,
                branch_id=self.branch,
                base_root=expected,
                new_root=tree.root_hash,
                page=n,
                pages=len(chunks),
                node_upserts=part,
                node_deletes=full.node_deletes if n == 0 else [],
            )
            for n, part in enumerate(chunks)
        ]
        return pages, full.model_copy(
            update={"node_upserts": [], "node_deletes": [], "staged_pages": len(pages)}
        )

    async def sync(
        self, files: dict[str, str], *, base: str | None = None, redo: Sequence[str] = (), **kw: Any
    ) -> IndexStatus:
        """`redo`: files of a previous, CONFLICTed attempt. Their chunk rows may have overwritten the
        winner's, so the retry re-embeds them even when the winner's tree already agrees (cache hit)."""
        tree = self.tree(files)
        server_root = await self.begin(tree, base)
        delta = merkle.diff(tree, server_root, self.remote_nodes, self.k)
        again = {p: tree.leaves[p] for p in redo if p in tree.leaves and p not in delta.added | delta.changed}
        delta = merkle.Delta(delta.added, delta.changed | again, delta.removed)
        for req in self.embed_reqs(delta, files):
            await self.gw.embed(req)
        return await self.gw.sync_commit(self.commit_req(tree, server_root, **kw))


def chunk_state(store: RecordingStore, c: Client) -> dict[str, set[str]]:
    """file_id -> set of file_hash values present among the chunk rows."""
    out: dict[str, set[str]] = {}
    for r in store.rows(c.ns).values():
        out.setdefault(r.attrs["file_id"], set()).add(r.attrs["file_hash"])
    return out


def server_root(store: RecordingStore, tree_ns: str = TREE) -> str | None:
    row = store.rows(tree_ns).get(ROOT_ID)
    return row.attrs["root_hash"] if row else None


async def converged(c: Client, files: dict[str, str]) -> None:
    """The next diff is empty, the chunk rows are exactly the tree's files, with no duplicates."""
    tree = c.tree(files)
    root = await c.begin(tree)
    assert root == tree.root_hash
    assert merkle.diff(tree, root, c.remote_nodes, c.k).empty
    expect = {ids.file_id(c.k, p): {merkle.leaf_hash(c.k, t.encode())} for p, t in files.items()}
    assert chunk_state(c.store, c) == expect
    assert len(c.store.rows(c.ns)) == len(files)  # one chunk per file: no lost, no duplicated rows
    status = await c.gw.index_status(StatusReq(repo_id=REPO, branch_id=c.branch))
    assert status.state == "ready"


# --- compare-and-set -----------------------------------------------------------------------------------


async def test_first_commit_then_incremental_commit_advance_the_root():
    """FR-C9: root row written on commit; the next commit CASes from it."""
    store, svc = make()
    c = await Client.create(svc, store)
    first = await c.sync(V1)
    assert (first.state, first.coverage, first.semantic_enabled) == ("ready", 1.0, True)
    assert server_root(store) == c.tree(V1).root_hash
    v2 = V1 | {"src/a.py": "def alpha_two():\n    return 3\n"}
    await c.sync(v2)
    assert server_root(store) == c.tree(v2).root_hash
    assert store.rows(TREE)[ROOT_ID].attrs["gen"] == 2
    await converged(c, v2)


async def test_commit_requires_the_begun_namespace():
    _, svc = make()
    gw = svc.for_tenant(TENANT)
    with pytest.raises(ScoutError) as ei:
        await gw.sync_commit(CommitReq(repo_id=REPO, branch_id=BR, new_root="r1", written_chunks=0))
    assert ei.value.code == ErrorCode.NOT_FOUND


@pytest.mark.parametrize("base_committed", [False, True])
async def test_two_clients_from_one_base_exactly_one_wins_and_the_loser_can_rediff(base_committed: bool):
    """FR-C9, design 15: gather of two commits from one base; one CONFLICT; the loser re-diffs."""
    store, svc = make(RecordingStore(yield_ops=True))
    a = await Client.create(svc, store)
    b = Client(a.gw, store, a.k)  # a second process of the same clone/branch, same expected base root
    base = V1 | {"src/c.py": "def gamma():\n    pass\n"}
    if base_committed:
        await a.sync(base)
    mine = base | {"src/a.py": "def alpha_mine():\n    return 'a'\n"}
    theirs = base | {"src/b.py": "def beta_theirs():\n    return 'b'\n"}
    expected = server_root(store)
    for client, files in ((a, mine), (b, theirs)):
        tree = client.tree(files)
        await client.begin(tree)
        delta = merkle.diff(tree, expected, client.remote_nodes, client.k)
        for req in client.embed_reqs(delta, files):
            await client.gw.embed(req)
    reqs = [a.commit_req(a.tree(mine), expected), b.commit_req(b.tree(theirs), expected)]
    results = await asyncio.gather(*(a.gw.sync_commit(r) for r in reqs), return_exceptions=True)
    wins = [(r, q) for r, q in zip(results, reqs, strict=True) if not isinstance(r, BaseException)]
    losses = [(r, q) for r, q in zip(results, reqs, strict=True) if isinstance(r, ScoutError)]
    assert len(wins) == 1
    assert len(losses) == 1
    err, lost_req = losses[0]
    assert isinstance(err, ScoutError)
    assert err.code == ErrorCode.CONFLICT
    assert err.retryable
    winner_root = wins[0][1].new_root
    assert err.details["server_root"] == winner_root == server_root(store)
    # the loser re-diffs against the winner's root and commits on top of it
    assert lost_req.new_root != winner_root
    merged = base | {"src/a.py": mine["src/a.py"], "src/b.py": theirs["src/b.py"]}
    lost_files = mine if lost_req.new_root == a.tree(mine).root_hash else theirs
    await a.sync(
        merged, redo=list(lost_files)
    )  # the CONFLICT protocol: redo own delta (see syncsvc docstring)
    await converged(a, merged)


async def test_two_service_instances_race_every_commit_and_exactly_one_wins_each_time():
    """AC-3.8, design 5.2: two `GatewayService` objects over ONE store model two Modal containers (separate
    in-process locks), so only the store-side conditional upsert can decide. Three successive commits, the
    first from no root and the next two from the previous winner's root, each raced with `asyncio.gather`."""
    store = RecordingStore(yield_ops=True)
    _, svc_a = make(store)
    _, svc_b = make(store)
    assert svc_a is not svc_b
    a, b = await Client.create(svc_a, store), await Client.create(svc_b, store)
    assert a.k == b.k  # same master secret, same tenant: the same derived keys
    files, expected, winners = dict(V1), None, []
    for rnd in range(3):
        mine = files | {f"src/a{rnd}.py": f"def a{rnd}():\n    return {rnd}\n"}
        theirs = files | {f"src/b{rnd}.py": f"def b{rnd}():\n    return {rnd}\n"}
        entries = [(a, mine), (b, theirs)]
        for client, new_files in entries:
            tree = client.tree(new_files)
            await client.begin(tree)
            added = {p: tree.leaves[p] for p in new_files.keys() - files.keys()}
            for req in client.embed_reqs(merkle.Delta(added=added), new_files):
                await client.gw.embed(req)
        reqs = [c.commit_req(c.tree(f), expected) for c, f in entries]
        results = await asyncio.gather(
            *(c.gw.sync_commit(r) for (c, _), r in zip(entries, reqs, strict=True)), return_exceptions=True
        )
        wins = [i for i, r in enumerate(results) if not isinstance(r, BaseException)]
        assert len(wins) == 1, f"round {rnd}: {[type(r).__name__ for r in results]}"
        loser = results[1 - wins[0]]
        assert isinstance(loser, ScoutError)
        assert loser.code == ErrorCode.CONFLICT
        assert loser.details["server_root"] == reqs[wins[0]].new_root == server_root(store)
        winners.append(wins[0])
        files, expected = entries[wins[0]][1], reqs[wins[0]].new_root
        assert store.rows(TREE)[ROOT_ID].attrs["gen"] == rnd + 1
    assert len(winners) == 3
    assert server_root(store) == a.tree(files).root_hash
    assert (await a.gw.index_status(StatusReq(repo_id=REPO, branch_id=BR))).state == "ready"


async def test_root_write_is_conditional_on_the_base_root_and_first_commit_is_insert_if_absent():
    """The root CAS is a store-side `condition=Eq(root_hash, base)`, not a lock: a stale writer that already
    passed the read-check is rejected by the store."""
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    v2 = V1 | {"src/a.py": "def alpha_two():\n    return 3\n"}
    await c.sync(v2)
    roots = [call for call in store.calls if call[:3] == ("upsert", TREE, ROOT_ID)]
    assert len(roots) == 2
    stale = await store.upsert(
        TREE, [Row(id=ROOT_ID, attrs={"root_hash": "x"})], condition=Eq("root_hash", c.tree(V1).root_hash)
    )
    assert stale == 0
    assert server_root(store) == c.tree(v2).root_hash


async def test_stale_expected_root_is_a_conflict_and_changes_nothing():
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    root_before, rows_before = server_root(store), dict(store.rows(TREE))
    tree = c.tree(V1 | {"src/z.py": "def z():\n    pass\n"})
    for expected in (None, "f" * 64):
        with pytest.raises(ScoutError) as ei:
            await c.gw.sync_commit(c.commit_req(tree, expected))
        assert ei.value.code == ErrorCode.CONFLICT
        assert ei.value.details["server_root"] == root_before
    assert server_root(store) == root_before
    assert set(store.rows(TREE)) == set(rows_before)


async def test_expected_root_on_a_tree_without_root_row_is_a_conflict():
    store, svc = make()
    c = await Client.create(svc, store)
    tree = c.tree(V1)
    await c.begin(tree)
    with pytest.raises(ScoutError) as ei:
        await c.gw.sync_commit(c.commit_req(tree, "e" * 64))
    assert (ei.value.code, ei.value.details["server_root"]) == (ErrorCode.CONFLICT, None)
    assert ROOT_ID not in store.rows(TREE)


async def test_replay_of_the_same_new_root_is_ok_and_idempotent():
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    tree = c.tree(V1)
    snapshot = {i: dict(r.attrs) for i, r in store.rows(TREE).items()}
    again = await c.gw.sync_commit(c.commit_req(tree, None))  # the lost response case: client retries blindly
    assert again.state == "ready"
    assert {i: dict(r.attrs) for i, r in store.rows(TREE).items()} == snapshot  # gen did not move


async def test_commit_rejects_overlapping_upsert_and_delete_and_bad_sketch_before_any_write():
    store, svc = make()
    c = await Client.create(svc, store)
    tree = c.tree(V1)
    await c.begin(tree)
    req = c.commit_req(tree, None)
    store.calls.clear()
    clash = req.model_copy(update={"node_deletes": [req.node_upserts[0].id]})
    bad = [
        clash,
        req.model_copy(update={"minhash_b64": base64.b64encode(b"x" * 10).decode()}),
        req.model_copy(update={"minhash_b64": "!!!not-base64"}),
    ]
    for r in bad:
        with pytest.raises(ScoutError) as ei:
            await c.gw.sync_commit(r)
        assert ei.value.code == ErrorCode.INVALID_ARGUMENT
    assert store.ops("upsert", "delete_ids") == []
    assert ROOT_ID not in store.rows(TREE)


# --- write order ---------------------------------------------------------------------------------------


async def test_root_row_is_the_last_tree_upsert_after_every_chunk_write_and_node_rows():
    """Design 5 rule 2: chunks, then node rows, then the root row; only node deletes follow the root."""
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    v2 = {"src/a.py": V1["src/a.py"], "lib/new.py": "def fresh():\n    pass\n"}  # b.py removed, new dir added
    store.calls.clear()
    await c.sync(v2)
    writes = store.ops("upsert", "delete_ids", "delete_by_filter")
    chunk_writes = [i for i, w in enumerate(writes) if w[1] == NS]
    tree_writes = [i for i, w in enumerate(writes) if w[1] == TREE]
    assert chunk_writes
    assert max(chunk_writes) < min(tree_writes)
    tree_ops = [writes[i] for i in tree_writes]
    assert tree_ops == [("upsert", TREE, "rows"), ("upsert", TREE, ROOT_ID)]  # node rows, then the root
    assert not any(w[0] == "delete_ids" and w[1] == TREE for w in writes)  # no node became unreferenced


async def test_node_deletes_run_after_the_root_so_server_root_never_names_a_missing_node():
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1 | {"deep/x/y.py": "def y():\n    pass\n"})
    v2 = dict(V1)  # the whole `deep` subtree disappears: its nodes are deleted
    store.calls.clear()
    store.snapshot = lambda: server_root(store)
    await c.sync(v2)
    tree_writes = [w for w in store.ops("upsert", "delete_ids") if w[1] == TREE]
    assert tree_writes[-1][0] == "delete_ids"
    assert tree_writes[-2] == ("upsert", TREE, ROOT_ID)
    gone = {i for i in c.commit_req(c.tree(v2), None).node_deletes}
    assert not gone  # nothing left to delete after the commit: the stale nodes are really gone
    # at the instant the root landed, every node the new tree names already existed
    root_landed = next(t for t in store.trace if t[0] == ("upsert", TREE, ROOT_ID))
    assert root_landed[1] == c.tree(v2).root_hash
    await converged(c, v2)


# --- crash safety --------------------------------------------------------------------------------------


async def test_crash_after_chunk_writes_before_root_commit_then_retry_converges():
    """The client dies after `embed` and before `sync_commit`: server root unchanged; the retry converges."""
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    v2 = V1 | {"src/a.py": "def alpha_two():\n    return 3\n", "src/n.py": "def newcomer():\n    pass\n"}
    tree = c.tree(v2)
    base_root = await c.begin(tree)
    delta = merkle.diff(tree, base_root, c.remote_nodes, c.k)
    for req in c.embed_reqs(delta, v2):
        await c.gw.embed(req)
    # --- crash: no commit ---
    assert server_root(store) == c.tree(V1).root_hash
    status = await c.gw.index_status(StatusReq(repo_id=REPO, branch_id=BR))
    assert status.state == "ready"  # still the old, internally consistent tree
    embedded_before = len(store.rows(NS))
    # --- restart: fresh client, same state on the server ---
    c2 = Client(c.gw, store, c.k)
    store.calls.clear()
    await c2.sync(v2)
    assert len(store.rows(NS)) == len(v2) == embedded_before  # replayed embed: no duplicated rows
    await converged(c2, v2)


@pytest.mark.parametrize(
    "failing", [("upsert", TREE, "rows"), ("upsert", TREE, ROOT_ID), ("delete_ids", TREE)]
)
async def test_failure_inside_commit_is_healed_by_the_retry(failing: Call):
    """A store failure at each step of the commit (nodes, root, node deletes): retry reaches a clean end."""
    store, svc = make()
    c = await Client.create(svc, store)
    base = V1 | {"deep/x/y.py": "def y():\n    pass\n"}
    await c.sync(base)
    v2 = {"src/a.py": "def alpha_two():\n    return 3\n"}  # b.py and deep/ removed, a.py changed
    tree = c.tree(v2)
    root = await c.begin(tree)
    for req in c.embed_reqs(merkle.diff(tree, root, c.remote_nodes, c.k), v2):
        await c.gw.embed(req)
    store.fail_once[failing] = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "injected")
    req = c.commit_req(tree, root)
    with pytest.raises(ScoutError):
        await c.gw.sync_commit(req)
    again = await c.gw.sync_commit(req)  # the client retries the identical commit
    assert again.state == "ready"
    await converged(c, v2)
    stored = {i for i, r in store.rows(TREE).items() if r.attrs.get("kind") == "node"}
    assert stored == set(merkle.seal_tree(tree, c.k))  # no stale or missing node rows


async def test_crash_after_root_write_before_proofs_drop_is_finished_by_the_replay():
    store, svc = make()
    c = await Client.create(svc, store)
    tree = c.tree(V1)
    await c.begin(tree)
    for req in c.embed_reqs(merkle.diff(tree, None, c.remote_nodes, c.k), V1):
        await c.gw.embed(req)
    await store.ensure_namespace(META, META_SCHEMA)
    await store.upsert(META, [Row(id=proofs_row_id(NS), attrs={"kind": "proofs", "proofs_required": True})])
    store.fail_once[("delete_ids", META)] = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "injected")
    req = c.commit_req(tree, None, minhash_b64=SKETCH)
    with pytest.raises(ScoutError):
        await c.gw.sync_commit(req)
    assert server_root(store) == tree.root_hash  # root landed
    assert proofs_row_id(NS) in store.rows(META)
    assert sketch_row_id(NS) not in store.rows(META)
    await c.gw.sync_commit(req)  # replay finishes the tail
    assert proofs_row_id(NS) not in store.rows(META)
    assert sketch_row_id(NS) in store.rows(META)


@pytest.mark.parametrize("where", ["first_batch", "second_batch", "delete_after_upsert"])
async def test_failure_mid_chunk_write_then_retry_loses_and_duplicates_nothing(where: str):
    """Embed fails partway (second batch / between upsert and replace-delete); the retry converges."""
    store, svc = make()
    c = await Client.create(svc, store)
    v1 = {f"src/f{i}.py": f"def fn_{i}():\n    return {i}\n" for i in range(4)}
    await c.sync(v1)
    v2 = {p: t.replace("return", "return 100 +") for p, t in v1.items()} | {
        "src/g.py": "def gee():\n    pass\n"
    }
    tree = c.tree(v2)
    root = await c.begin(tree)
    delta = merkle.diff(tree, root, c.remote_nodes, c.k)
    reqs = c.embed_reqs(delta, v2, per_batch=2)
    assert len(reqs) == 3
    boom = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "injected")
    if where == "first_batch":
        store.fail_once[("upsert", NS, "rows")] = boom
        with pytest.raises(ScoutError):
            await c.gw.embed(reqs[0])
    elif where == "second_batch":
        await c.gw.embed(reqs[0])
        store.fail_once[("upsert", NS, "rows")] = boom
        with pytest.raises(ScoutError):
            await c.gw.embed(reqs[1])
    else:
        store.fail_once[("delete_by_filter", NS)] = boom  # new rows are in, old ones not yet deleted
        with pytest.raises(ScoutError):
            await c.gw.embed(reqs[0])
        assert any(len(h) == 2 for h in chunk_state(store, c).values())  # old and new coexist
    assert server_root(store) == c.tree(v1).root_hash  # never committed
    await c.sync(v2)  # the retry from scratch: begin, diff (same delta), embed everything, commit
    await converged(c, v2)


async def test_branch_commit_after_a_base_commit_landed_between_the_two_copies():
    """3.4 interleave: base advances between the tree and chunk copies; the branch keeps the OLDER root and a
    later `sync_commit(expected_root=older)` on the branch succeeds (no false CONFLICT)."""
    store, svc = make()
    base = await Client.create(svc, store, BR_BASE)
    await base.sync(V1)
    old_root = base.tree(V1).root_hash
    v_base2 = V1 | {"src/c.py": "def gamma_base():\n    pass\n"}
    base_tree = names(BR_BASE)[1]

    async def base_commits_meanwhile() -> None:  # a full sync lands on the base pair between the copies
        await base.sync(v_base2)

    store.after[("branch_from", TREE, base_tree)] = base_commits_meanwhile
    branch = Client(base.gw, store, base.k, BR)
    tree = branch.tree(V1)
    assert await branch.begin(tree, base=BR_BASE) == old_root
    assert server_root(store, base_tree) == base.tree(v_base2).root_hash  # the base moved on
    # branch chunks are newer than its tree: c.py exists as a chunk, but the tree does not name it
    assert len(chunk_state(store, branch)) == 3
    files = V1 | {"src/a.py": "def alpha_branch():\n    return 'br'\n"}
    status = await branch.sync(files)  # expected_root = old_root, as the branch's server_root said
    assert status.state == "ready"
    assert server_root(store) == branch.tree(files).root_hash


# --- stale chunks vs the committed tree ----------------------------------------------------------------


async def test_stale_chunk_window_is_embed_to_commit_and_old_rows_never_outlive_the_embed():
    """Documented window (syncsvc docstring): for a changed file, old+new rows coexist for ONE delete
    round trip inside `embed`; from the end of `embed` to the root write only the new rows exist while the
    tree still names the old hash (tree older than chunks); after the root write chunk rows equal the tree.
    Removed files lose their
    rows in `embed`, before the root write. At no point after the root write is an old-hash row present."""
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync({**V1, "src/c.py": "def gone():\n    pass\n"})
    fid_a, fid_c = ids.file_id(c.k, "src/a.py"), ids.file_id(c.k, "src/c.py")
    old_a = merkle.leaf_hash(c.k, V1["src/a.py"].encode())
    v2 = {"src/a.py": "def alpha_two():\n    return 3\n", "src/b.py": V1["src/b.py"]}  # a changed, c removed
    new_a = merkle.leaf_hash(c.k, v2["src/a.py"].encode())
    old_root, new_root = c.tree({**V1, "src/c.py": "def gone():\n    pass\n"}).root_hash, c.tree(v2).root_hash

    def snap() -> tuple[frozenset[str], bool, str | None]:
        st = chunk_state(store, c)
        return frozenset(st.get(fid_a, set())), fid_c in st, server_root(store)

    store.snapshot = snap
    store.trace.clear()
    await c.sync(v2)
    steps = [(call, s) for call, s in store.trace if call[1] in (NS, TREE)]
    labels = [(call[0], call[2] if len(call) > 2 else "") for call, _ in steps]
    assert labels == [  # exactly one embed (upsert, replace delete, removal delete), then the commit
        ("upsert", "rows"),
        ("delete_by_filter", ""),
        ("delete_by_filter", ""),
        ("upsert", "rows"),
        ("upsert", ROOT_ID),
    ]
    (after_upsert, after_replace, after_remove, after_nodes, after_root) = [s for _, s in steps]
    assert after_upsert == (frozenset({old_a, new_a}), True, old_root)  # the one-delete-round-trip overlap
    assert after_replace == (frozenset({new_a}), True, old_root)  # old a rows gone as soon as the replace ran
    assert after_remove == (
        frozenset({new_a}),
        False,
        old_root,
    )  # removed file: rows gone, tree still names it
    assert after_nodes == (frozenset({new_a}), False, old_root)  # node rows do not move the root
    assert after_root == (frozenset({new_a}), False, new_root)  # window closed: chunks == committed tree
    # the delete used the spike-verified shape: file_id == X AND file_hash != new (never a bare file delete)
    assert store.ops("delete_by_filter")[0][1] == NS


async def test_replace_delete_filter_is_file_and_not_new_hash():
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    seen: list[Filter] = []
    real = store.delete_by_filter

    async def spy(ns: str, flt: Filter) -> int:
        seen.append(flt)
        return await real(ns, flt)

    store.delete_by_filter = spy  # type: ignore[method-assign]
    v2 = V1 | {"src/a.py": "def alpha_two():\n    return 3\n"}
    await c.sync(v2)
    from scout.core.ports.store import And, Eq, Not

    new_hash = merkle.leaf_hash(c.k, v2["src/a.py"].encode())
    assert seen == [And([Eq("file_id", ids.file_id(c.k, "src/a.py")), Not(Eq("file_hash", new_hash))])]


# --- coverage and readiness ----------------------------------------------------------------------------


async def status(c: Client) -> IndexStatus:
    return await c.gw.index_status(StatusReq(repo_id=REPO, branch_id=c.branch))


@pytest.mark.parametrize(
    ("written", "expected", "state"),
    [
        (0, 10, "partial"),
        (7, 10, "partial"),
        (79, 100, "partial"),
        (8, 10, "ready"),
        (80, 100, "ready"),
        (81, 100, "ready"),
        (10, 10, "ready"),
        (4, 5, "ready"),
        (3, 5, "partial"),
    ],
)
async def test_readiness_boundary_at_80_percent(written: int, expected: int, state: str):
    """AC-3.8: partial strictly below 0.8, ready at exactly 0.8 and above; semantic search follows."""
    store, svc = make()
    c = await Client.create(svc, store)
    tree = c.tree(V1)
    await c.begin(tree)
    res = await c.gw.sync_commit(c.commit_req(tree, None, written_chunks=written, expected_chunks=expected))
    assert res.state == state
    assert res.semantic_enabled == (state == "ready")
    assert res.coverage == pytest.approx(written / expected)
    assert (await status(c)).state == state


async def test_index_state_walks_absent_partial_ready_as_coverage_grows():
    """AC-3.8 / FR-C13: absent (no root) -> partial (< 80%) -> ready, via commits of the same tree."""
    store, svc = make()
    c = await Client.create(svc, store)
    tree = c.tree(V1)
    assert (await status(c)).state == "absent"
    await c.begin(tree)
    assert (await status(c)).state == "absent"  # namespace begun, nothing committed
    p = await c.gw.sync_commit(c.commit_req(tree, None, written_chunks=5, expected_chunks=10))
    assert (p.state, p.coverage, p.semantic_enabled) == ("partial", 0.5, False)
    more = await c.gw.sync_commit(c.commit_req(tree, None, written_chunks=8, expected_chunks=10))
    assert (more.state, more.coverage, more.semantic_enabled) == ("ready", 0.8, True)
    assert server_root(store) == tree.root_hash
    lower = await c.gw.sync_commit(c.commit_req(tree, None, written_chunks=3, expected_chunks=10))
    assert lower.coverage == 0.8  # a late duplicate never lowers coverage under the same root
    moved = await c.gw.sync_commit(c.commit_req(tree, None, written_chunks=3, expected_chunks=20))
    assert (moved.coverage, moved.state) == (0.4, "partial")  # new total, but written never goes below 8


async def test_coverage_and_commit_come_from_the_committed_root_and_status_survives_a_new_gateway():
    store, svc = make()
    c = await Client.create(svc, store)
    tree = c.tree(V1)
    await c.begin(tree)
    res = await c.gw.sync_commit(
        c.commit_req(tree, None, commit="abc123", written_chunks=9, expected_chunks=10)
    )
    assert res.commit == "abc123"
    fresh = GatewayService(store, FakeClock(), MASTER).for_tenant(TENANT)  # another process
    got = await fresh.index_status(StatusReq(repo_id=REPO, branch_id=BR))
    assert (got.state, got.coverage, got.commit) == ("ready", 0.9, "abc123")


# --- proofs and sketch ---------------------------------------------------------------------------------


async def test_proofs_row_is_dropped_when_the_root_commits_and_sketch_is_written():
    """AC-4.3 second half: the seed proofs row is deleted on commit; a vector-less sketch row appears."""
    store, svc = make()
    c = await Client.create(svc, store)
    other_ns = names(BR_BASE)[0]
    await store.ensure_namespace(META, META_SCHEMA)
    await store.upsert(
        META,
        [
            Row(id=proofs_row_id(NS), attrs={"kind": "proofs", "leaves": ["l1"], "proofs_required": True}),
            Row(id=proofs_row_id(other_ns), attrs={"kind": "proofs", "proofs_required": True}),
        ],
    )
    tree = c.tree(V1)
    await c.begin(tree)
    for req in c.embed_reqs(merkle.diff(tree, None, c.remote_nodes, c.k), V1):
        await c.gw.embed(req)
    await c.gw.sync_commit(c.commit_req(tree, None, minhash_b64=SKETCH))
    rows = store.rows(META)
    assert proofs_row_id(NS) not in rows  # dropped
    assert proofs_row_id(other_ns) in rows  # another namespace's proofs untouched
    sketch = rows[sketch_row_id(NS)]
    assert sketch.vector is None
    assert sketch.attrs == {"kind": "sketch", "sketch": SKETCH}
    assert len(base64.b64decode(sketch.attrs["sketch"])) == SKETCH_BYTES


async def test_proofs_survive_a_conflicting_commit_and_other_tenants_meta_is_untouched():
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    await store.upsert(META, [Row(id=proofs_row_id(NS), attrs={"kind": "proofs", "proofs_required": True})])
    other_meta = meta_namespace("beta")
    await store.ensure_namespace(other_meta, META_SCHEMA)
    await store.upsert(other_meta, [Row(id=proofs_row_id(NS), attrs={"kind": "proofs"})])
    tree = c.tree(V1 | {"src/z.py": "def z():\n    pass\n"})
    with pytest.raises(ScoutError):
        await c.gw.sync_commit(c.commit_req(tree, "e" * 64, minhash_b64=SKETCH))
    assert proofs_row_id(NS) in store.rows(META)  # the loser did not commit, so nothing was spent
    assert sketch_row_id(NS) not in store.rows(META)
    await c.gw.sync_commit(c.commit_req(tree, server_root(store)))
    assert proofs_row_id(NS) not in store.rows(META)
    assert proofs_row_id(NS) in store.rows(other_meta)  # a tenant only ever touches its own meta namespace


async def test_commit_without_a_meta_namespace_or_sketch_creates_no_meta_namespace():
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    assert not await store.namespace_exists(META)


async def test_tree_namespace_stays_vector_less_and_chunks_never_land_in_it():
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    assert store._ns[TREE].has_vector is False  # pyright: ignore[reportPrivateUsage]
    assert store._ns[TREE].schema == TREE_SCHEMA  # pyright: ignore[reportPrivateUsage]
    assert {r.attrs["kind"] for r in store.rows(TREE).values()} == {"node", "root"}
    assert {r.attrs["kind"] for r in store.rows(NS).values()} == {"chunk"}


# --- node paging (task 4.1a) ---------------------------------------------------------------------------

DEEP = {
    f"d{i}/f{i}.py": f"def f{i}():\n    return {i}\n" for i in range(7)
}  # 8 node rows (7 dirs + root dir)


def tree_attrs(store: RecordingStore) -> dict[str, Any]:
    return {i: dict(r.attrs) for i, r in store.rows(TREE).items()}


async def embedded(
    c: Client, files: dict[str, str], base: str | None = None
) -> tuple[merkle.Tree, str | None]:
    tree = c.tree(files)
    root = await c.begin(tree, base)
    for req in c.embed_reqs(merkle.diff(tree, root, c.remote_nodes, c.k), files):
        await c.gw.embed(req)
    return tree, root


async def test_staged_pages_then_commit_equals_an_inline_commit():
    """AC-3.7: the same tree via pages (2 rows each) and via one inline commit (other branch, same keys)
    gives the same node ids and hashes and the same root; no marker row is left."""
    store, svc = make()
    inline = await Client.create(svc, store, BR_BASE)
    await inline.sync(DEEP)
    c = await Client.create(svc, store)
    tree, root = await embedded(c, DEEP)
    pages, commit = c.staged(tree, root)
    assert len(pages) == 4
    for p in pages:
        assert (await c.gw.sync_stage_nodes(p)).staged == len(p.node_upserts)
    assert server_root(store) is None  # pages alone never move the root

    assert (await c.gw.sync_commit(commit)).state == "ready"

    def shape(tree_ns: str) -> dict[str, Any]:
        rows = store.rows(tree_ns)
        return {i: r.attrs.get("node_hash", r.attrs.get("root_hash")) for i, r in rows.items()}

    assert shape(TREE) == shape(inline.tree_ns)
    assert len(shape(TREE)) == 8 + 1
    await converged(c, DEEP)


async def test_commit_with_a_missing_page_is_refused_and_writes_nothing():
    store, svc = make()
    c = await Client.create(svc, store)
    tree, root = await embedded(c, DEEP)
    pages, commit = c.staged(tree, root)
    for p in (pages[0], pages[1], pages[3]):  # page 2 never arrives
        await c.gw.sync_stage_nodes(p)
    store.calls.clear()
    with pytest.raises(ScoutError) as ei:
        await c.gw.sync_commit(commit)
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    assert ei.value.details["missing_pages"] == [2]
    assert server_root(store) is None
    assert not [x for x in store.ops("upsert", "delete_ids") if x[1] == TREE]  # refused before any write
    await c.gw.sync_stage_nodes(pages[2])  # the page arrives late (any order): now it commits
    assert (await c.gw.sync_commit(commit)).state == "ready"
    await converged(c, DEEP)


async def test_commit_claiming_more_pages_than_staged_or_pages_of_another_sync_is_refused():
    store, svc = make()
    c = await Client.create(svc, store)
    tree, root = await embedded(c, DEEP)
    pages, commit = c.staged(tree, root)
    for p in pages:
        await c.gw.sync_stage_nodes(p)
    with pytest.raises(ScoutError) as ei:
        await c.gw.sync_commit(commit.model_copy(update={"staged_pages": len(pages) + 1}))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError) as ei:  # pages staged under another page count do not satisfy this commit
        await c.gw.sync_commit(commit.model_copy(update={"staged_pages": len(pages) - 1}))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    other = c.tree({"zz/other.py": "def other():\n    pass\n"})  # pages are scoped by new_root
    with pytest.raises(ScoutError) as ei:
        await c.gw.sync_commit(commit.model_copy(update={"new_root": other.root_hash}))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    assert server_root(store) is None


@pytest.mark.parametrize("k", [0, 1, 2, 3])
async def test_crash_after_page_k_then_retry_converges(k: int):
    """A client dying after page k (k+1 pages sent, no commit) leaves the old root; the retry restages."""
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    v2 = V1 | DEEP
    tree, root = await embedded(c, v2)
    pages, commit = c.staged(tree, root)
    for p in pages[: k + 1]:
        await c.gw.sync_stage_nodes(p)
    assert server_root(store) == c.tree(V1).root_hash  # old root still names the old tree
    old = c.tree(V1)
    assert not merkle.diff(old, server_root(store), c.remote_nodes, c.k).added  # and it still resolves
    if k < len(pages) - 1:
        with pytest.raises(ScoutError):
            await c.gw.sync_commit(commit)
    c2 = Client(c.gw, store, c.k)  # a fresh client re-sends every page, including the ones already there
    for p in pages:
        await c2.gw.sync_stage_nodes(p)
    assert (await c2.gw.sync_commit(commit)).state == "ready"
    await converged(c2, v2)
    assert {r.attrs["kind"] for r in store.rows(TREE).values()} == {"node", "root"}


async def test_staging_a_page_twice_changes_nothing():
    store, svc = make()
    c = await Client.create(svc, store)
    tree, root = await embedded(c, DEEP)
    pages, _ = c.staged(tree, root)
    await c.gw.sync_stage_nodes(pages[0])
    before = tree_attrs(store)
    await c.gw.sync_stage_nodes(pages[0])
    assert tree_attrs(store) == before


async def test_page_checks_base_root_and_is_a_noop_for_a_committed_root():
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1)
    tree, root = await embedded(c, V1 | DEEP)
    pages, commit = c.staged(tree, root)
    wrong = pages[0].model_copy(update={"base_root": "e" * 64})
    with pytest.raises(ScoutError) as ei:
        await c.gw.sync_stage_nodes(wrong)
    assert ei.value.code is ErrorCode.CONFLICT
    assert ei.value.details["server_root"] == root
    assert not [r for r in store.rows(TREE).values() if r.attrs.get("kind") == "nodepage"]
    for p in pages:
        await c.gw.sync_stage_nodes(p)
    await c.gw.sync_commit(commit)
    before = tree_attrs(store)
    res = await c.gw.sync_stage_nodes(pages[0])  # a late duplicate after the commit
    assert res.staged == 0
    assert tree_attrs(store) == before
    await c.gw.sync_commit(commit)  # and a replayed commit is OK too
    assert tree_attrs(store) == before


async def test_page_and_commit_validation():
    store, svc = make()
    c = await Client.create(svc, store)
    tree, root = await embedded(c, DEEP)
    pages, commit = c.staged(tree, root)
    with pytest.raises(ScoutError) as ei:
        await c.gw.sync_stage_nodes(pages[0].model_copy(update={"page": 4}))
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    clash = pages[0].model_copy(update={"node_deletes": [pages[0].node_upserts[0].id]})
    with pytest.raises(ScoutError) as ei:
        await c.gw.sync_stage_nodes(clash)
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    for p in pages:
        await c.gw.sync_stage_nodes(p)
    cross = commit.model_copy(
        update={"node_deletes": [pages[1].node_upserts[0].id]}
    )  # staged upsert, inline delete
    with pytest.raises(ScoutError) as ei:
        await c.gw.sync_commit(cross)
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    assert server_root(store) is None


async def test_staged_node_deletes_run_after_the_root_write():
    """Deletes ride on a page marker; the commit runs them strictly after the root (ordering invariant)."""
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1 | DEEP)
    v2 = dict(V1)  # every `d*` directory disappears
    tree, root = await embedded(c, v2)
    pages, commit = c.staged(tree, root)
    gone = set(pages[0].node_deletes)
    assert gone
    store.calls.clear()
    store.snapshot = lambda: server_root(store)
    for p in pages:
        await c.gw.sync_stage_nodes(p)
    assert gone <= set(store.rows(TREE))  # staging deleted nothing
    await c.gw.sync_commit(commit)
    tree_writes = [w for w in store.ops("upsert", "delete_ids") if w[1] == TREE]
    root_at = tree_writes.index(("upsert", TREE, ROOT_ID))
    assert [w[0] for w in tree_writes[root_at + 1 :]] == ["delete_ids", "delete_ids"]  # node deletes, markers
    assert not gone & set(store.rows(TREE))
    await converged(c, v2)


@pytest.mark.parametrize(
    "failing_call",
    [("upsert", TREE, ROOT_ID), ("delete_ids", TREE)],
    ids=["root_write", "after_root_deletes"],
)
async def test_failure_inside_a_staged_commit_is_healed_by_the_replay(failing_call: Call):
    store, svc = make()
    c = await Client.create(svc, store)
    await c.sync(V1 | DEEP)
    v2 = V1 | {"src/a.py": "def alpha_two():\n    return 3\n"}
    tree, root = await embedded(c, v2)
    pages, commit = c.staged(tree, root)
    for p in pages:
        await c.gw.sync_stage_nodes(p)
    store.fail_once[failing_call] = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "injected")
    with pytest.raises(ScoutError):
        await c.gw.sync_commit(commit)
    assert (await c.gw.sync_commit(commit)).state == "ready"  # the replay finishes the tail
    await converged(c, v2)
    stored = {i for i, r in store.rows(TREE).items() if r.attrs.get("kind") == "node"}
    assert stored == set(merkle.seal_tree(tree, c.k))  # no stale node, no marker row


async def test_staged_commit_cas_still_has_exactly_one_winner():
    """Two service instances over one store, both staging their own pages from one base: one commit wins."""
    store = RecordingStore(yield_ops=True)
    _, svc_a = make(store)
    _, svc_b = make(store)
    a, b = await Client.create(svc_a, store), await Client.create(svc_b, store)
    mine, theirs = V1 | DEEP, V1 | {"src/b.py": "def beta_theirs():\n    return 'b'\n"}
    staged = []
    for client, files in ((a, mine), (b, theirs)):
        tree, root = await embedded(client, files)
        pages, commit = client.staged(tree, root)
        for p in pages:
            await client.gw.sync_stage_nodes(p)
        staged.append((client, commit))
    results = await asyncio.gather(*(c.gw.sync_commit(q) for c, q in staged), return_exceptions=True)
    assert sum(not isinstance(r, BaseException) for r in results) == 1
    loser = next(r for r in results if isinstance(r, ScoutError))
    assert loser.code is ErrorCode.CONFLICT


async def test_node_and_marker_rows_stay_out_of_sync_nodes():
    store, svc = make()
    c = await Client.create(svc, store)
    tree, root = await embedded(c, DEEP)
    pages, _ = c.staged(tree, root)
    await c.gw.sync_stage_nodes(pages[0])
    marker = next(i for i, r in store.rows(TREE).items() if r.attrs.get("kind") == "nodepage")
    req = NodesReq.model_construct(repo_id=REPO, branch_id=BR, ids=[marker])  # the wire regex forbids this id
    got = await c.gw.sync_nodes(req)
    assert got.blobs == {}
