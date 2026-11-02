"""Task 4.3: `SyncEngine` branches, moves, CONFLICT retry, MerkleError recovery and partial coverage, through
the real `GatewayService` over fakes.

FR-C15, AC-3.5, AC-3.8, AC-3.10 (design 5.3, 15). Every scenario is judged against a from-scratch sync of the
same checkout (same keys, same clone id, separate store): chunk and tree namespaces must be identical.
MinHash is only stored, never compared, so the 3-sigma margin rule does not apply here.
"""

import shutil
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from svc_helpers import RecordingStore
from test_sync_first import Spy, World, chunk_rows, disk_status, gen, root_attrs, tree_rows, world
from test_sync_incremental import BIG, EditFS, OrderStore, dirs_of, index_view, tree_repo

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import EmbedReq, EmbedRes, SearchReq, StatusReq
from scout.core.crypto import ids, rows
from scout.core.crypto.tokens import hmac_tokens, identifier_tokens
from scout.core.ports.fs import GitIds
from scout.core.ports.gateway import Gateway
from scout.core.ports.store import Row
from scout.core.service.schema import ROOT_ID
from scout.core.sync import merkle
from scout.core.sync.engine import SyncConfig, SyncEngine, WireMeter

TENANT = "dev"
FEAT = "feature/login"


class BranchFS(EditFS):
    """`EditFS` whose git branch can change (a checkout). Files are edited by the test, not by `checkout`."""

    def checkout(self, branch: str, *, detached: bool = False) -> None:
        head = ("2" if detached else "1") * 40
        name = f"detached-{head[:12]}" if detached else branch
        self._git = replace(cast(GitIds, self._git), branch=name, head=head, detached=detached)

    def adopt(self, git: GitIds) -> None:
        self._git = git


def bfs(w: World) -> BranchFS:
    return cast(BranchFS, w.fs)


async def branch_world(tmp_path: Path, files: dict[str, str | bytes], name: str = "w") -> World:
    w = await world(tmp_path, files, fs_cls=BranchFS, name=name)
    await w.engine().sync()
    return w


async def fresh_ref(tmp_path: Path, w: World, name: str = "ref") -> World:
    """A from-scratch sync of the checkout `w` has now: same git state, keys and clone id, empty store."""
    ref = await world(tmp_path, dict(bfs(w).files), k=w.k, fs_cls=BranchFS, name=name)
    ref.state_dir.mkdir(parents=True)
    shutil.copy(w.state_dir / "state.json", ref.state_dir / "state.json")
    bfs(ref).adopt(bfs(w).git_ids())
    await ref.engine().sync()
    return ref


def branch_ns(w: World, name: str) -> tuple[str, str]:
    """`(ns, tree)` of a branch of this clone."""
    st = w.state.read_state()
    assert st.repo_id
    ns = ids.namespace(TENANT, st.repo_id, ids.branch_id(w.k, st.repo_id, st.clone_id, name))
    return ns, ids.tree_namespace(ns)


def chunk_view(w: World) -> list[tuple[str, str, str]]:
    return sorted((i, r.attrs["chunk_key"], r.attrs["file_hash"]) for i, r in chunk_rows(w).items())


def edit_branch(fs: BranchFS) -> list[str]:
    """One edited file, one added, one removed: the branch's whole diff against main."""
    fs.write("svc1/mod1/f2.py", "def renamed_handler(request):\n    return request * 3\n")
    fs.write("svc1/mod4/fresh.py", "def fresh_one():\n    return 'FRESH'\n")
    fs.remove("svc2/mod2/f1.py")
    return ["svc1/mod1/f2.py", "svc1/mod4/fresh.py", "svc2/mod2/f1.py"]


# --- AC-3.10: a new branch is branched from the default branch and embeds only its own diff ----------------
async def test_branch_switch_branches_the_default_namespace_and_embeds_only_the_diff(tmp_path: Path) -> None:
    """AC-3.10, FR-C15: first sync on a new branch -> `branch_from` of the tree THEN the chunks of main, a
    server-tree diff (`sync_nodes`), only the branch-changed files embedded; the result equals a from-scratch
    sync of the branch tree and main is untouched."""
    w = await branch_world(tmp_path, tree_repo())
    main_ns = w.ids[2]
    main_view = index_view(w)
    old_texts = set(w.emb.embedded)
    first_total = len(w.emb.embedded)
    assert first_total > 20
    w.store.calls.clear()

    fs = bfs(w)
    fs.checkout(FEAT)
    touched = edit_branch(fs)
    meter = WireMeter(w.gw)
    status = await w.engine(spy=cast(Spy, meter)).sync()

    feat_ns, feat_tree = branch_ns(w, FEAT)
    assert w.ids[2] == feat_ns != main_ns
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert w.store.ops("branch_from") == [  # tree first, then chunks (design 5)
        ("branch_from", feat_tree, ids.tree_namespace(main_ns)),
        ("branch_from", feat_ns, main_ns),
    ]
    first_upsert = next(i for i, c in enumerate(w.store.calls) if c[0] == "upsert" and c[1] == feat_ns)
    assert max(i for i, c in enumerate(w.store.calls) if c[0] == "branch_from") < first_upsert
    ref = await fresh_ref(tmp_path, w)
    new_texts = w.emb.embedded[first_total:]
    assert sorted(new_texts) == sorted(set(ref.emb.embedded) - old_texts)  # new bodies only, each once
    assert len(new_texts) == 2  # the edited file and the added one; the removed file embeds nothing
    print(f"[4.3] branch switch: {len(new_texts)} of {first_total} chunk bodies embedded")
    assert index_view(w) == index_view(ref)
    names = [name for name, _, _ in meter.calls]
    assert "sync_nodes" in names  # the diff walked the server tree
    assert names.count("sync_nodes") <= 5  # one round trip per level (depth 3 here), not per node
    leaves = {r.attrs["file_id"]: r.attrs["file_hash"] for r in chunk_rows(w).values()}
    removed = ids.file_id(w.k, touched[2])
    assert removed not in leaves  # the file the branch deleted has no rows left
    assert {ids.file_id(w.k, p) for p in touched[:2]} <= leaves.keys()
    # main's namespaces are exactly as they were
    w.state.update_state(last_branch=ids.branch_id(w.k, w.ids[0], w.state.read_state().clone_id, "main"))
    assert index_view(w) == main_view


async def test_switching_back_reuses_the_old_branch_namespace_with_no_embed(tmp_path: Path) -> None:
    """FR-C15: main -> feature -> main (files restored) and feature again: no embed call, no `branch_from`,
    no commit of new content; each branch keeps its own index."""
    files = tree_repo()
    w = await branch_world(tmp_path, dict(files))
    fs = bfs(w)
    fs.checkout(FEAT)
    edit_branch(fs)
    await w.engine().sync()
    feat_view = index_view(w)
    feat_files = dict(fs.files)

    fs.files = {p: (d.encode() if isinstance(d, str) else d) for p, d in files.items()}
    fs.checkout("main")
    fs.mtimes.clear()
    for p in fs.files:
        fs.touch(p)  # a checkout rewrites files: new mtimes, the same bytes
    n_calls, n_embeds, n_commits = len(w.emb.calls), len(w.spy.embeds), len(w.spy.commits)
    w.store.calls.clear()
    status = await w.engine().sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert (len(w.emb.calls), len(w.spy.embeds), len(w.spy.commits)) == (n_calls, n_embeds, n_commits)
    assert not w.store.ops("branch_from")

    fs.files = feat_files
    fs.checkout(FEAT)
    for p in fs.files:
        fs.touch(p)
    status = await w.engine().sync()
    print(
        f"[4.3] switch back: embed calls {len(w.emb.calls) - n_calls}, "
        f"commits {len(w.spy.commits) - n_commits}"
    )
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert (len(w.emb.calls), len(w.spy.embeds), len(w.spy.commits)) == (n_calls, n_embeds, n_commits)
    assert not w.store.ops("branch_from")
    assert index_view(w) == feat_view


async def test_a_branch_with_no_indexed_default_branch_is_a_plain_full_sync(tmp_path: Path) -> None:
    """The default branch was never synced here: `sync_begin` creates an empty namespace, no `branch_from`."""
    w = await world(tmp_path, tree_repo(), fs_cls=BranchFS)
    bfs(w).checkout(FEAT)
    status = await w.engine().sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert not w.store.ops("branch_from")
    assert index_view(w) == index_view(await fresh_ref(tmp_path, w))


async def test_detached_head_and_look_alike_branch_names_get_distinct_namespaces(tmp_path: Path) -> None:
    """FR-C15: names that differ only by case or by `/` and a detached HEAD each map to their own branch id
    and namespace; each is branched from main, so none re-embeds anything."""
    files = {k: v for k, v in list(tree_repo().items())[:12]}
    w = await branch_world(tmp_path, files)
    embedded = len(w.emb.embedded)
    seen = {w.ids[1]}
    fs = bfs(w)
    for name in ("Feature", "feature", "feat/a", "feat-a", "feat/A", "detached"):
        fs.checkout(name, detached=name == "detached")
        status = await w.engine().sync()
        assert (status.state, status.coverage) == ("ready", 1.0)
        seen.add(w.ids[1])
    git = fs.git_ids()
    assert git.detached
    assert git.branch.startswith("detached-")
    assert len(seen) == 7  # main + six names, all different
    chunk_namespaces = [n for n in w.store.names() if n.endswith(tuple(seen)) and not n.endswith("__tree")]
    assert len(chunk_namespaces) == 7
    assert len(w.emb.embedded) == embedded  # every branch came from main's rows
    assert len(w.store.ops("branch_from")) == 12  # six branches, each a tree copy and a chunk copy of main
    repo_id, clone = w.ids[0], w.state.read_state().clone_id
    names = ["main", "Feature", "feature", "feat/a", "feat-a", "feat/A", git.branch]
    assert len({ids.branch_id(w.k, repo_id, clone, n) for n in names}) == 7


# --- AC-3.5: moves ----------------------------------------------------------------------------------------
async def hit_paths(w: World, query: str) -> list[str]:
    repo_id, branch_id, _ = w.ids
    res = await w.gw.search(
        SearchReq(
            repo_id=repo_id,
            branch_id=branch_id,
            query=query,
            k=10,
            tokens=hmac_tokens(w.k, identifier_tokens(query)),
        )
    )
    return [rows.open_hit(w.k, TENANT, repo_id, h)[0] for h in res.hits]


@pytest.mark.parametrize("kind", ["file", "directory"])
async def test_a_moved_file_embeds_nothing_and_search_reports_the_new_path(tmp_path: Path, kind: str) -> None:
    """AC-3.5: unchanged content under a new path costs zero embed calls (chunk key = body only); search then
    returns the NEW path and never the stale one; the old file's rows are gone."""
    w = await branch_world(tmp_path, tree_repo())
    fs = bfs(w)
    if kind == "file":
        moves = {"svc0/mod0/f0.py": "svc0/mod3/moved_handler.py"}
    else:
        moves = {f"svc1/mod1/f{i}.py": f"svc1/renamed/f{i}.py" for i in range(5)}
    query = (
        " ".join(f"handler_{10 + i}_{j}" for i in range(1) for j in range(3))
        if kind == "directory"
        else ("handler_0_0 handler_0_1 handler_0_2 request_0 value_0_0")
    )
    before = await hit_paths(w, query)
    probe = next(iter(moves))
    assert before[0] == probe  # the probe file is the best hit for its own identifiers
    n_reqs = len(w.spy.embeds)
    doc_calls = lambda: sum(1 for c in w.emb.calls if c.input_type == "document")  # noqa: E731
    n_calls = doc_calls()
    for old, new in moves.items():
        fs.write(new, fs.files.pop(old))
    status = await w.engine().sync()
    after = await hit_paths(w, query)

    print(f"[4.3] move ({kind}, {len(moves)} files): document embed calls {doc_calls() - n_calls}")
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert doc_calls() == n_calls  # the embedder was never asked for a document
    assert all(req.texts == {} for req in w.spy.embeds[n_reqs:])  # and no text travelled either
    assert after[0] == moves[probe]
    assert not set(moves) & set(after)  # a stale path is never returned
    for old in moves:
        assert not [r for r in chunk_rows(w).values() if r.attrs["file_id"] == ids.file_id(w.k, old)]
    assert index_view(w, vectors=False) == index_view(await fresh_ref(tmp_path, w), vectors=False)


# --- ordering invariant on every write path (recorded at the store) -----------------------------------------
def kinds(log: list[tuple[str, str, tuple[str, ...]]], ns: str) -> list[str]:
    """C chunk write/delete, N tree upsert (node page or marker), R root upsert, D tree delete."""
    tree, out = ids.tree_namespace(ns), []
    for op, name, row_ids in log:
        if name == ns:
            out.append("C")
        elif name == tree and op == "upsert":
            out.append("R" if ROOT_ID in row_ids else "N")
        elif name == tree:
            out.append("D")
    return out


async def ordered_world(tmp_path: Path, files: dict[str, str | bytes]) -> tuple[World, OrderStore]:
    store = OrderStore()
    w = await world(tmp_path, files, fs_cls=BranchFS)
    w.store = store
    w.gw = make_fake_gateway(TENANT, store=store, embedder=w.emb, clock=w.clock)
    w.spy = Spy(w.gw)
    await w.engine().sync()
    return w, store


async def test_branch_sync_write_order_chunks_then_nodes_then_root(tmp_path: Path) -> None:
    """Design 5 ordering invariant on the branch path: copies tree-first, every chunk write before any node
    page, the root CAS last among tree upserts, node deletes after it."""
    w, store = await ordered_world(tmp_path, tree_repo())
    mark = len(store.log)
    bfs(w).checkout(FEAT)
    edit_branch(bfs(w))
    await w.engine().sync()
    seq = kinds(store.log[mark:], w.ids[2])
    assert seq.count("R") == 1
    assert "C" in seq
    assert seq.index("R") == max(i for i, k in enumerate(seq) if k in "CNR")  # nothing but deletes after
    assert max(i for i, k in enumerate(seq) if k == "C") < min(i for i, k in enumerate(seq) if k == "N")
    assert all(k == "D" for k in seq[seq.index("R") + 1 :])


# --- conflicts ---------------------------------------------------------------------------------------------
class Rival:
    """Another writer of the same branch (same clone id, its own checkout and state dir, same gateway)."""

    def __init__(self, tmp_path: Path, w: World, name: str = "rival") -> None:
        self.dir = tmp_path / name / "git" / "scout"
        shutil.copytree(w.state_dir, self.dir)
        self.fs = BranchFS(dict(bfs(w).files), state_dir=str(self.dir))
        self.fs.adopt(bfs(w).git_ids())
        self.fs.mtimes = dict(bfs(w).mtimes)
        self.fs.inos = dict(bfs(w).inos)
        self.engine = SyncEngine(self.fs, w.gw, w.k, w.clock)

    async def sync(self) -> None:
        assert (await self.engine.sync()).state == "ready"


class ConflictSpy(Spy):
    """Runs `before_commit(n)` ahead of the n-th (0-based) `sync_commit`, so a rival can win the CAS."""

    def __init__(self, gw: Gateway, before_commit: Callable[[int], Awaitable[None]]) -> None:
        super().__init__(gw)
        self.before_commit = before_commit

    async def sync_commit(self, req: Any) -> Any:
        await self.before_commit(len(self.commits))
        return await super().sync_commit(req)


async def test_conflict_retries_rediffs_and_reembeds_the_whole_failed_delta(tmp_path: Path) -> None:
    """AC-3.8 / 3.4 note (a): the root moves between the loser's embed and its CAS. The retry re-diffs against
    the winner's root AND re-sends every file of its own failed delta, even F1 which the winner's tree
    already agrees with, because the winner's chunk rows for it may have been overwritten (simulated)."""
    w = await branch_world(tmp_path, tree_repo())
    fs = bfs(w)
    rival = Rival(tmp_path, w)
    f1, f2, f3 = "svc1/mod1/f1.py", "svc1/mod1/f2.py", "svc2/mod2/f3.py"
    shared = "def shared_edit(request):\n    return 'SHARED'\n"
    rival.fs.write(f1, shared)  # the winner made the same edit to f1 ...
    rival.fs.write(f3, "def rival_only():\n    return 'RIVAL'\n")  # ... and one of its own
    fs.write(f1, shared)
    fs.write(f2, "def loser_only():\n    return 'LOSER'\n")
    old_texts = len(w.emb.embedded)
    first_commits = len(w.spy.commits)

    async def win(n: int) -> None:
        if n != first_commits:
            return
        await rival.sync()
        for rid in [i for i, r in chunk_rows(w).items() if r.attrs["file_id"] == ids.file_id(w.k, f1)]:
            await w.store.delete_ids(w.ids[2], [rid])  # the loser's embed "overwrote" the winner's rows of f1

    spy = ConflictSpy(w.gw, win)
    spy.commits = list(w.spy.commits)
    status = await w.engine(spy=spy).sync()

    assert (status.state, status.coverage) == ("ready", 1.0)
    assert len(spy.commits) == first_commits + 2  # CONFLICT, then the retry
    ref = await fresh_ref(tmp_path, w)
    assert index_view(w) == index_view(ref)  # f1's rows are back, f2 and f3 are right
    retry_rows = {r.file_id for req in spy.embeds[1:] for r in req.rows}
    assert retry_rows == {ids.file_id(w.k, p) for p in (f1, f2, f3)}  # f1 re-sent although the diff skips it
    assert len(w.emb.embedded) - old_texts == 3  # shared, loser-only and rival-only bodies, once each
    print(f"[4.3] conflict retry: commits {len(spy.commits) - first_commits}, bodies embedded {3}")
    ev = spy.events
    commits_at = [i for i, e in enumerate(ev) if e == "commit"]
    assert len(commits_at) == 2
    for lo, hi in zip([0, *commits_at], commits_at, strict=False):  # per attempt: embeds, stage, then commit
        seg = ev[lo:hi]
        assert "embed_end" in seg
        assert max(i for i, e in enumerate(seg) if e == "embed_end") < max(
            (i for i, e in enumerate(seg) if e == "stage"), default=len(seg)
        )


async def test_conflict_retry_gives_up_after_three_and_restores_the_status(tmp_path: Path) -> None:
    """Task 4.3 item 2: at most 3 retries (4 commits); the last CONFLICT is raised, status.json restored."""
    w = await branch_world(tmp_path, tree_repo())
    rival = Rival(tmp_path, w)
    bfs(w).write("svc1/mod1/f2.py", "def loser_only():\n    return 'LOSER'\n")
    wins = 0

    async def win(n: int) -> None:
        nonlocal wins
        text = f"def rival_{n}():\n    return {n}\n"
        rival.fs.write(f"svc2/extra{n}.py", text)
        bfs(w).write(f"svc2/extra{n}.py", text)  # the loser's checkout has the winner's files too
        await rival.sync()
        wins += 1

    spy = ConflictSpy(w.gw, win)
    spy.commits = list(w.spy.commits)
    base = len(spy.commits)
    with pytest.raises(ScoutError) as err:
        await w.engine(spy=spy).sync()
    assert err.value.code is ErrorCode.CONFLICT
    assert len(spy.commits) - base == 4
    assert wins == 4
    doc = disk_status(w)
    assert (doc["state"], doc["syncing"]) == ("ready", False)
    # the losing attempts left their node rows under the winners' root (the cross-container residual), so this
    # sync may meet a MerkleError and rebuild the tree; either way it converges
    fresh = await w.engine().sync()  # no rival now: the next sync converges
    assert fresh.state == "ready"
    a, b = index_view(w), index_view(await fresh_ref(tmp_path, w))
    tree = ids.tree_namespace(w.ids[2])
    left = {k for k in a[tree]["rows"] if k.startswith("nodepage-")}
    assert len(left) == 4  # the four abandoned attempts' markers stay (syncsvc: a few small rows, harmless)
    a[tree]["rows"] = {k: v for k, v in a[tree]["rows"].items() if k not in left}
    assert a == b


# --- MerkleError: node rows overwritten by a cross-container race ------------------------------------------
NODE_DIR = "svc0/mod0"


async def corrupt_node(w: World, how: str) -> None:
    """Leave the winner's root over a loser's (or missing, or unreadable) node row for `NODE_DIR`."""
    repo_id, _, ns = w.ids
    tree = ids.tree_namespace(ns)
    nid = ids.node_id(w.k, repo_id, NODE_DIR)
    if how == "missing":
        await w.store.delete_ids(tree, [nid])
        return
    if how == "unreadable":
        donor = next(r for i, r in tree_rows(w).items() if r.attrs.get("kind") == "node" and i != nid)
        await w.store.upsert(tree, [Row(id=nid, attrs=dict(donor.attrs))])  # sealed for another node id
        return
    loser = {p: merkle.leaf_hash(w.k, d) for p, d in bfs(w).files.items()}
    loser[BIG] = merkle.leaf_hash(w.k, b"the loser's version of big.py")
    tree_obj = merkle.build_tree(loser.items(), w.k, TENANT, repo_id)
    sealed = merkle.seal_tree(tree_obj, w.k)[nid]
    attrs = {"kind": "node", "children_enc": sealed.children_enc, "node_hash": sealed.node_hash}
    await w.store.upsert(tree, [Row(id=nid, attrs=attrs)])  # parent hash no longer matches this node


def forget_local_tree(w: World) -> None:
    (w.state_dir / "trees" / f"{w.ids[1]}.json").unlink()


@pytest.mark.parametrize("how", ["overwritten", "missing", "unreadable"])
async def test_merkle_error_falls_back_to_a_full_resync_and_converges(tmp_path: Path, how: str) -> None:
    """Task 4.3 note (b): a server tree that fails its hash checks is detected (MerkleError), logged, and
    answered with ONE full re-sync of the tree (every node row rewritten, root CAS'd against the current
    root); the end state equals a from-scratch sync and the next diff against the server tree is clean."""
    w = await branch_world(tmp_path, tree_repo())
    old_root = root_attrs(w)["root_hash"]  # type: ignore[index]
    await corrupt_node(w, how)
    fs = bfs(w)
    fs.write("svc0/mod0/f1.py", "def edited_in_the_broken_dir():\n    return 'EDIT'\n")
    forget_local_tree(w)  # a lost state dir: the sync must diff against the server tree, and meets the damage
    embedded, staged = len(w.emb.embedded), len(w.spy.staged)

    eng = w.engine()
    status = await eng.sync()

    assert eng.merkle_resyncs == 1
    assert any("full re-sync" in m for m in eng.warnings)
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert w.spy.commits[-1].expected_root == old_root  # CAS'd against the root the server had
    nodes = sum(len(p.node_upserts) for p in w.spy.staged[staged:])
    assert nodes == len(dirs_of(list(fs.files)))  # every node row of the tree was rewritten
    assert len(w.emb.embedded) - embedded == 1  # only the edited body; the rest were cache hits
    assert index_view(w) == index_view(await fresh_ref(tmp_path, w))
    print(f"[4.3] merkle recovery ({how}): resyncs {eng.merkle_resyncs}, nodes rewritten {nodes}")

    # diff-clean: a second lost-state sync walks the same directory without any MerkleError
    fs.write("svc0/mod0/f2.py", "def edited_again():\n    return 'AGAIN'\n")
    forget_local_tree(w)
    eng2 = w.engine()
    assert (await eng2.sync()).state == "ready"
    assert eng2.merkle_resyncs == 0
    assert index_view(w) == index_view(await fresh_ref(tmp_path, w, "ref2"))


async def test_merkle_error_after_another_clone_commits_over_our_saved_tree(tmp_path: Path) -> None:
    """The race of syncsvc's residual note: a rival container moved the root and left node rows that do not
    match it; our saved tree is older than the server root, so the diff walks the server tree, fails its
    hash check once, and the full re-sync converges (the local state dir is intact here)."""
    w = await branch_world(tmp_path, tree_repo())
    rival = Rival(tmp_path, w)
    rival.fs.write("svc2/mod2/f4.py", "def rival_edit():\n    return 'RIVAL'\n")
    await rival.sync()
    bfs(w).write("svc2/mod2/f4.py", "def rival_edit():\n    return 'RIVAL'\n")  # the checkout follows main
    await corrupt_node(w, "overwritten")
    bfs(w).write("svc0/mod0/f1.py", "def our_edit():\n    return 'OURS'\n")
    assert w.state.load_tree(w.ids[1]) is not None
    eng = w.engine()
    status = await eng.sync()
    assert (eng.merkle_resyncs, status.state, status.coverage) == (1, "ready", 1.0)
    assert index_view(w) == index_view(await fresh_ref(tmp_path, w))


async def test_merkle_recovery_keeps_the_write_order(tmp_path: Path) -> None:
    w, store = await ordered_world(tmp_path, tree_repo())
    await corrupt_node(w, "overwritten")
    bfs(w).write("svc0/mod0/f1.py", "def edited_in_the_broken_dir():\n    return 'EDIT'\n")
    forget_local_tree(w)
    mark = len(store.log)
    await w.engine().sync()
    seq = kinds(store.log[mark:], w.ids[2])
    assert seq.count("R") == 1
    assert max(i for i, k in enumerate(seq) if k == "C") < min(i for i, k in enumerate(seq) if k == "N")
    assert seq.index("R") == max(i for i, k in enumerate(seq) if k in "CNR")


# --- branch repair: the server tree older than the server chunks ---------------------------------------
async def reversed_pair(store: Any, ns: str, tree: str, src_ns: str, src_tree: str) -> None:
    """Test-only gateway behaviour: copy the chunks BEFORE the tree (the order design 5 forbids)."""
    await store.branch_from(ns, src_ns)
    await store.branch_from(tree, src_tree)


async def interleaved_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reverse: bool
) -> tuple[World, World]:
    """Main commits an edit to F between the two branch copies; the branch checkout has F's new text."""
    w = await branch_world(tmp_path, tree_repo())
    rival = Rival(tmp_path, w)
    changed, new_text = "svc1/mod1/f1.py", "def moved_on_in_main():\n    return 'MAIN2'\n"
    rival.fs.write(changed, new_text)
    main_ns, main_tree = w.ids[2], ids.tree_namespace(w.ids[2])
    feat_ns, feat_tree = branch_ns(w, FEAT)
    store = cast(RecordingStore, w.store)

    async def main_commits() -> None:
        await rival.sync()

    if reverse:
        monkeypatch.setattr("scout.core.service.gateway.branch_pair", reversed_pair)
        store.after[("branch_from", feat_ns, main_ns)] = main_commits  # after the chunk copy
    else:
        store.after[("branch_from", feat_tree, main_tree)] = main_commits  # after the tree copy
    fs = bfs(w)
    fs.checkout(FEAT)
    fs.write(changed, new_text)  # the branch is rebased on main's new tip ...
    fs.write("svc1/mod4/fresh.py", "def fresh_one():\n    return 'FRESH'\n")  # ... plus its own file
    assert (await w.engine().sync()).state == "ready"
    return w, await fresh_ref(tmp_path, w)


async def test_branch_converges_when_main_commits_between_the_two_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task 4.3 item 2a / ordering repair: tree-first copies leave the branch tree OLDER than its chunks; the
    diff against that tree re-syncs the affected file, so the branch ends equal to a from-scratch sync (sorted
    chunk keys included, no stale chunk for the file main edited)."""
    w, ref = await interleaved_branch(tmp_path, monkeypatch, reverse=False)
    assert chunk_view(w) == chunk_view(ref)
    assert sorted(r.attrs["chunk_key"] for r in chunk_rows(w).values()) == sorted(
        r.attrs["chunk_key"] for r in chunk_rows(ref).values()
    )
    assert index_view(w) == index_view(ref)


async def test_reversed_copy_order_leaves_a_stale_chunk_so_the_test_can_detect_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: with the chunks copied BEFORE the tree the branch tree is newer than its chunks, the
    diff sees nothing to do for the file main edited, and its old chunk survives. The assertion above fails
    on this state, which is what proves it detects the bug."""
    w, ref = await interleaved_branch(tmp_path, monkeypatch, reverse=True)
    assert chunk_view(w) != chunk_view(ref)
    stale = set(chunk_view(w)) - set(chunk_view(ref))
    assert stale
    fid = ids.file_id(w.k, "svc1/mod1/f1.py")
    assert {chunk_rows(w)[s[0]].attrs["file_id"] for s in stale} == {fid}


# --- AC-3.8: partial coverage -------------------------------------------------------------------------------
class FlakySpy(Spy):
    """`embed` fails (after the engine's retries) for any batch holding a row of a doomed file."""

    def __init__(self, gw: Gateway, doomed: set[str]) -> None:
        super().__init__(gw)
        self.doomed = doomed
        self.refused = 0

    async def embed(self, req: EmbedReq, *, idempotency_key: str | None = None) -> EmbedRes:
        if any(r.file_id in self.doomed for r in req.rows):
            self.refused += 1
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "embedding upstream 503")
        return await super().embed(req, idempotency_key=idempotency_key)


PARTIAL_CFG = SyncConfig(batch_tokens=700, workers=2, retries=1)


async def test_failed_embed_batches_leave_partial_with_honest_coverage_and_the_next_sync_completes(
    tmp_path: Path,
) -> None:
    """AC-3.8, PRD 4.3 item 6: batches that fail after their retries are skipped; the root counters carry the
    rows actually written, so `partial` and the coverage number agree everywhere (status.json, `index_status`,
    the store); the next sync embeds exactly what is missing and reaches `ready`."""
    files = gen.make_repo(60)
    w = await world(tmp_path, dict(files), fs_cls=BranchFS)
    doomed = {ids.file_id(w.k, p) for p in sorted(files)[:36]}  # the first 60% of the files
    flaky = FlakySpy(w.gw, doomed)
    eng = w.engine(PARTIAL_CFG, flaky)
    status = await eng.sync()

    root = root_attrs(w)
    assert root
    written, expected = root["written_chunks"], root["expected_chunks"]
    assert written == len(chunk_rows(w)) < expected  # the counter is what is really in the store
    assert 0.0 < status.coverage < 0.8
    assert status.coverage == pytest.approx(written / expected)
    assert (status.state, status.semantic_enabled) == ("partial", False)
    assert flaky.refused >= 2  # more than one batch failed, each after its retry
    assert any("embed batches failed" in m for m in eng.warnings)
    repo_id, branch_id, _ = w.ids
    served = await w.gw.index_status(StatusReq(repo_id=repo_id, branch_id=branch_id))
    doc = disk_status(w)
    assert served.state == doc["state"] == "partial"
    assert served.coverage == pytest.approx(doc["coverage"]) == pytest.approx(written / expected)
    print(f"[4.3] partial: coverage {status.coverage:.3f} ({written}/{expected})")

    first_run = list(w.emb.embedded)
    healed = await w.engine(PARTIAL_CFG, Spy(w.gw)).sync()  # the upstream is back
    second_run = w.emb.embedded[len(first_run) :]
    assert (healed.state, healed.coverage) == ("ready", 1.0)
    assert set(first_run).isdisjoint(second_run)  # nothing embedded twice
    assert second_run  # and the missing bodies were embedded now
    assert root_attrs(w)["written_chunks"] == root_attrs(w)["expected_chunks"] == len(chunk_rows(w))  # type: ignore[index]
    assert index_view(w) == index_view(await fresh_ref(tmp_path, w))
    assert disk_status(w)["state"] == "ready"


async def test_every_batch_failing_commits_nothing(tmp_path: Path) -> None:
    """No batch succeeded: the error is raised and no root exists, so the index still reads `absent`."""
    files = gen.make_repo(20)
    w = await world(tmp_path, dict(files), fs_cls=BranchFS)
    flaky = FlakySpy(w.gw, {ids.file_id(w.k, p) for p in files})
    with pytest.raises(ScoutError) as err:
        await w.engine(PARTIAL_CFG, flaky).sync()
    assert err.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert root_attrs(w) is None
    assert (disk_status(w)["state"], disk_status(w)["syncing"]) == ("absent", False)


async def test_partial_write_order_is_unchanged(tmp_path: Path) -> None:
    store = OrderStore()
    files = gen.make_repo(40)
    w = await world(tmp_path, dict(files), fs_cls=BranchFS)
    w.store = store
    w.gw = make_fake_gateway(TENANT, store=store, embedder=FakeEmbedder(), clock=FakeClock())
    flaky = FlakySpy(w.gw, {ids.file_id(w.k, p) for p in sorted(files)[:10]})
    assert (await w.engine(PARTIAL_CFG, flaky).sync()).state in ("partial", "ready")
    seq = kinds(store.log, w.ids[2])
    assert seq.count("R") == 1
    assert max(i for i, k in enumerate(seq) if k == "C") < min(i for i, k in enumerate(seq) if k == "N")
    assert seq.index("R") == max(i for i, k in enumerate(seq) if k in "CNR")
