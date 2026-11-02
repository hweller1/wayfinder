"""Task 4.2: `SyncEngine` incremental sync through the real `GatewayService` over fakes.

FR-C9, FR-C12, AC-3.1, AC-3.6, NFR-5 (wire bytes scale with changed files x depth, not repo size) and the
design 8 stat-cache rule. Every incremental scenario is judged against a from-scratch sync of the edited tree
(same keys, same clone id): the chunk namespace and the tree namespace must be identical, so stale rows,
missing rows and wrong nodes all fail one assertion. MinHash is only stored, never compared (no 3-sigma rule).
"""

import json
import math
import shutil
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from svc_helpers import RecordingStore
from test_sync_first import (
    FULL,
    HookFS,
    Spy,
    World,
    chunk_rows,
    disk_status,
    gen,
    normalized,
    root_attrs,
    world,
)

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_gateway import make_fake_gateway
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import IndexStatus
from scout.core.contracts.wire import StatusReq, WireHit
from scout.core.crypto import ids, keys, paths, rows
from scout.core.local.ignore import MAX_FILE_BYTES
from scout.core.ports.fs import FileStat
from scout.core.ports.gateway import Gateway
from scout.core.ports.store import Filter, NamespaceSchema
from scout.core.service.schema import CHUNK_SCHEMA, ROOT_ID
from scout.core.sync import chunker
from scout.core.sync.engine import SyncEngine, WireMeter
from scout.core.sync.state import StateDir

TENANT = "dev"
BASE_MTIME = 1_700_000_000_000_000_000
BIG = "svc0/mod0/big.py"


class EditFS(HookFS):
    """In-memory checkout with real stat semantics: `mtime_ns` and `ino` are per file and only change when a
    test says so (`write`, `touch`, `new_inode`), so the stat cache sees exactly what a disk would show."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.mtimes: dict[str, int] = {}
        self.inos: dict[str, int] = {}
        self._tick = 0

    def stat(self, rel: str) -> FileStat:
        real = self._confine(rel)
        data = self.files.get(real)
        if data is None:
            raise ScoutError(ErrorCode.NOT_FOUND, "file not found")
        ino = self.inos.setdefault(real, len(self.inos) + 1)
        return FileStat(len(data), self.mtimes.get(real, BASE_MTIME), ino, False)

    def touch(self, path: str) -> None:
        self._tick += 1
        self.mtimes[path] = BASE_MTIME + self._tick

    def write(self, path: str, content: str | bytes, *, mtime: bool = True) -> None:
        self.files[path] = content.encode() if isinstance(content, str) else content
        if mtime:
            self.touch(path)

    def remove(self, path: str) -> None:
        del self.files[path]

    def new_inode(self, path: str) -> None:
        self.inos[path] = len(self.inos) + 1_000_000


class OrderStore(RecordingStore):
    """Logs every write with its row ids, across namespaces, in call order."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[tuple[str, str, tuple[str, ...]]] = []

    async def upsert(self, ns: str, rows: Any, **kw: Any) -> int:
        self.log.append(("upsert", ns, tuple(r.id for r in rows)))
        return await super().upsert(ns, rows, **kw)

    async def delete_ids(self, ns: str, ids: Any) -> int:
        self.log.append(("delete_ids", ns, tuple(ids)))
        return await super().delete_ids(ns, ids)

    async def delete_by_filter(self, ns: str, flt: Filter) -> int:
        self.log.append(("delete_by_filter", ns, ()))
        return await super().delete_by_filter(ns, flt)


def big_module(tag: str = "a") -> str:
    """Four functions of ~350 chars each: four chunks; `tag` changes only the third one."""
    out: list[str] = []
    for j in range(4):
        body = "\n".join(f"    step_{j}_{k} = payload.get('key_{j}_{k}', {j * 7 + k})" for k in range(5))
        end = f"return step_{j}_0 + len('{tag}')" if j == 2 else f"return step_{j}_0 + {j}"
        out.append(f"def stage_{j}(payload):\n{body}\n    {end}\n")
    return "\n".join(out)


def tree_repo() -> dict[str, str | bytes]:
    """30 small modules over 6 directories (two levels), a four-chunk file and a lone file in its own dirs."""
    files: dict[str, str | bytes] = {}
    for d in range(6):
        for f in range(5):
            files[f"svc{d % 3}/mod{d}/f{f}.py"] = gen._module(d * 10 + f, 12)  # pyright: ignore[reportPrivateUsage]
    files[BIG] = big_module()
    files["solo/only/one.py"] = "def only_one():\n    return 'ONLY_ONE'\n"
    return files


async def edit_world(tmp_path: Path, files: dict[str, str | bytes], name: str = "w") -> World:
    w = await world(tmp_path, files, fs_cls=EditFS, name=name)
    await w.engine().sync()
    return w


def efs(w: World) -> EditFS:
    return cast(EditFS, w.fs)


async def reference(tmp_path: Path, w: World) -> World:
    """A from-scratch sync of the checkout `w` has now (same keys and clone id, separate store)."""
    ref = await world(tmp_path, dict(efs(w).files), k=w.k, fs_cls=EditFS, name="ref")
    ref.state_dir.mkdir(parents=True)
    shutil.copy(w.state_dir / "state.json", ref.state_dir / "state.json")
    await ref.engine().sync()
    return ref


def index_view(w: World, *, vectors: bool = True) -> dict[str, Any]:
    """Chunk + tree namespaces, nonce-free; the root `gen` counter is history, not state. `vectors=False`
    drops chunk vectors: a moved file keeps the vector cached under its body key (the chunk key hashes the
    body, not the path header, so AC-3.5 embeds nothing), which differs from a fresh embed of the new path."""
    ns = w.ids[2]
    view = {n: v for n, v in normalized(w).items() if n in (ns, ids.tree_namespace(ns))}
    if not vectors:
        view[ns]["rows"] = {r: (a, None) for r, (a, _) in view[ns]["rows"].items()}
    tree = view[ids.tree_namespace(ns)]["rows"]
    attrs, vector = tree[ROOT_ID]
    tree[ROOT_ID] = ({k: v for k, v in attrs.items() if k != "gen"}, vector)
    return view


def dirs_of(files: list[str]) -> set[str]:
    return {""} | {d for p in files for d in paths.ancestors(p)}


# --- scenarios: each mutates the checkout and returns the paths it touched ---
def edit_one_chunk(fs: EditFS) -> list[str]:
    fs.write(BIG, big_module("b"))
    return [BIG]


def edit_small_file(fs: EditFS) -> list[str]:
    fs.write("svc1/mod1/f2.py", "def renamed_handler(request):\n    return request * 3\n")
    return ["svc1/mod1/f2.py"]


def add_file(fs: EditFS) -> list[str]:
    fs.write("svc1/mod4/fresh.py", "def fresh_one():\n    return 'FRESH'\n")
    return ["svc1/mod4/fresh.py"]


def add_in_new_dirs(fs: EditFS) -> list[str]:
    fs.write("svc9/new/deep/er/leaf.py", "def deep_leaf():\n    return 'LEAF'\n")
    return ["svc9/new/deep/er/leaf.py"]


def remove_file(fs: EditFS) -> list[str]:
    fs.remove("svc2/mod2/f1.py")
    return ["svc2/mod2/f1.py"]


def remove_last_file_of_a_dir(fs: EditFS) -> list[str]:
    fs.remove("solo/only/one.py")  # `solo/only` and `solo` vanish: two node deletes
    return ["solo/only/one.py"]


def move_file(fs: EditFS) -> list[str]:
    fs.write("svc0/mod3/moved.py", fs.files.pop("svc0/mod0/f0.py"))
    return ["svc0/mod0/f0.py", "svc0/mod3/moved.py"]


def empty_a_file(fs: EditFS) -> list[str]:
    fs.write("svc1/mod1/f0.py", "\n\n")  # no chunks left: its rows all go, the file stays in the tree
    return ["svc1/mod1/f0.py"]


def many(fs: EditFS) -> list[str]:
    out = edit_one_chunk(fs) + edit_small_file(fs) + add_file(fs) + add_in_new_dirs(fs)
    return out + remove_file(fs) + remove_last_file_of_a_dir(fs)


SCENARIOS: dict[str, Callable[[EditFS], list[str]]] = {
    f.__name__: f
    for f in (
        edit_one_chunk,
        edit_small_file,
        add_file,
        add_in_new_dirs,
        remove_file,
        remove_last_file_of_a_dir,
        move_file,
        empty_a_file,
        many,
    )
}


# --- AC-3.6: only the changed chunk bodies are embedded; the index equals a from-scratch sync ---
@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_incremental_sync_equals_a_from_scratch_sync(tmp_path: Path, name: str) -> None:
    """AC-3.6, FR-C12: after any mix of edits, adds, removals and moves the chunk and tree namespaces equal
    a from-scratch sync of the new tree; only genuinely new bodies reach the embedder, only changed files'
    rows are sent, and only the changed paths' nodes are staged."""
    w = await edit_world(tmp_path, tree_repo())
    before_root = root_attrs(w)
    assert before_root
    old_texts = set(w.emb.embedded)
    first_embeds, first_stages = len(w.spy.embeds), len(w.spy.staged)

    touched = SCENARIOS[name](efs(w))
    status = await w.engine().sync()

    ref = await reference(tmp_path, w)
    assert (status.state, status.coverage, status.syncing) == ("ready", 1.0, False)
    moved = name == "move_file"
    assert index_view(w, vectors=not moved) == index_view(ref, vectors=not moved)
    new_texts = w.emb.embedded[len(old_texts) :]
    expect = [] if moved else sorted(set(ref.emb.embedded) - old_texts)  # AC-3.5: a move embeds nothing
    assert sorted(new_texts) == expect  # new bodies only, each once
    allowed = {ids.file_id(w.k, p) for p in touched}
    sent = [r for req in w.spy.embeds[first_embeds:] for r in req.rows]
    assert {r.file_id for r in sent} <= allowed  # no row of an untouched file crosses the wire
    # nodes: the path from each touched file to the root, vanished directories as deletes
    ref_saved = ref.state.load_tree(ref.ids[1])
    assert ref_saved
    new_dirs = dirs_of(list(ref_saved.files))
    path_dirs = dirs_of(touched)
    staged = w.spy.staged[first_stages:]
    assert sum(len(p.node_upserts) for p in staged) == len(path_dirs & new_dirs)
    assert sum(len(p.node_deletes) for p in staged) == len(path_dirs - new_dirs)
    commit = w.spy.commits[-1]
    assert commit.expected_root == before_root["root_hash"]
    assert commit.node_upserts == []
    assert commit.node_deletes == []
    saved = w.state.load_tree(w.ids[1])
    assert saved is not None
    assert saved.root == root_attrs(w)["root_hash"]  # type: ignore[index]
    assert set(saved.files) == set(ref_saved.files)
    assert disk_status(w)["state"] == "ready"
    assert (await w.gw.index_status(StatusReq(repo_id=w.ids[0], branch_id=w.ids[1]))).coverage == 1.0


async def test_one_changed_chunk_embeds_one_text_and_removes_the_stale_row(tmp_path: Path) -> None:
    """AC-3.6: one edited function of a four-chunk file -> one embedded text, three rows reused from the
    cache, the file's old-hash row removed (`replace` -> `delete_by_file`), the other 33 files untouched."""
    w = await edit_world(tmp_path, tree_repo())
    fid = ids.file_id(w.k, BIG)
    old = [r for r in chunk_rows(w).values() if r.attrs["file_id"] == fid]
    assert len(old) == 4
    old_hash = old[0].attrs["file_hash"]
    n_embeds = len(w.emb.embedded)
    n_reqs = len(w.spy.embeds)

    efs(w).write(BIG, big_module("b"))
    await w.engine().sync()

    assert len(w.emb.embedded) - n_embeds == 1
    assert "len('b')" in w.emb.embedded[-1]
    assert w.emb.embedded[-1].startswith(f"{BIG}\n")
    reqs = w.spy.embeds[n_reqs:]
    assert len(reqs) == 1
    assert len(reqs[0].texts) == 1  # one fresh body; the three cache-hit rows go without text
    assert len(reqs[0].rows) == 4
    assert [(r.file_id, r.file_hash) for r in reqs[0].replace] == [(fid, reqs[0].rows[0].file_hash)]
    assert reqs[0].delete_files == []
    new = [r for r in chunk_rows(w).values() if r.attrs["file_id"] == fid]
    assert len(new) == 4
    assert old_hash not in {r.attrs["file_hash"] for r in new}  # the old rows are gone
    assert len({r.attrs["file_hash"] for r in chunk_rows(w).values() if r.attrs["file_id"] == fid}) == 1
    assert len(chunk_rows(w)) == len(index_view(w)[w.ids[2]]["rows"])  # no extra rows anywhere


async def test_nothing_changed_sends_nothing(tmp_path: Path) -> None:
    w = await edit_world(tmp_path, tree_repo())
    calls = len(w.spy.embeds), len(w.spy.staged), len(w.spy.commits), len(w.emb.calls)
    status = await w.engine().sync()
    assert status.state == "ready"
    assert (len(w.spy.embeds), len(w.spy.staged), len(w.spy.commits), len(w.emb.calls)) == calls


# --- ordering invariant: chunk writes, then node pages, then the root CAS, then node deletes ---
@pytest.mark.parametrize("name", ["edit_one_chunk", "remove_last_file_of_a_dir", "many"])
async def test_incremental_write_order_chunks_then_nodes_then_root(tmp_path: Path, name: str) -> None:
    """Design 5 / 8 ordering invariant, recorded at the store: every chunk upsert and delete precedes every
    tree write, the root row is the last tree upsert, and node deletes come after it."""
    store = OrderStore()
    w = await world(tmp_path, tree_repo(), fs_cls=EditFS)
    w.store = store
    w.gw = make_fake_gateway(TENANT, store=store, embedder=w.emb, clock=w.clock)
    w.spy = Spy(w.gw)
    await w.engine().sync()
    mark, ev_mark = len(store.log), len(w.spy.events)
    SCENARIOS[name](efs(w))
    await w.engine().sync()

    ns = w.ids[2]
    tree = ids.tree_namespace(ns)
    log = store.log[mark:]
    chunk_ops = [i for i, (_, n, _) in enumerate(log) if n == ns]
    tree_up = [i for i, (op, n, _) in enumerate(log) if n == tree and op == "upsert"]
    root_at = [i for i in tree_up if ROOT_ID in log[i][2]]
    tree_del = [i for i, (op, n, _) in enumerate(log) if n == tree and op == "delete_ids"]
    assert chunk_ops
    assert len(root_at) == 1
    assert max(chunk_ops) < min(tree_up)  # chunks before any node page
    assert root_at[0] == max(tree_up)  # the root CAS is the last tree upsert
    assert all(i > root_at[0] for i in tree_del)  # node deletes and marker cleanup after the root
    if name != "edit_one_chunk":
        assert any(log[i][2] and ROOT_ID not in log[i][2] for i in tree_del)
    ev = w.spy.events[ev_mark:]  # the same order as seen by the gateway port
    assert max(i for i, e in enumerate(ev) if e == "embed_end") < ev.index("stage") < ev.index("commit")


# --- stat cache (design 8): what it skips, what it never skips, and the accepted risk ---
async def test_unchanged_files_are_not_read_or_hashed(tmp_path: Path) -> None:
    """Design 8: a file whose (size, mtime_ns, ino) match its saved entry gets no full read at all."""
    w = await edit_world(tmp_path, tree_repo())
    fs = efs(w)
    fs.reads.clear()
    fs.write(BIG, big_module("b"))
    await w.engine().sync()
    assert [r for r in fs.reads if r[1] == FULL] == [(BIG, FULL)]  # the matcher's 8 KiB sniff reads aside


async def test_touched_file_with_same_bytes_is_hashed_but_not_embedded(tmp_path: Path) -> None:
    """A new mtime forces a re-hash; the same bytes keep the saved chunk count and send nothing."""
    w = await edit_world(tmp_path, tree_repo())
    fs = efs(w)
    before = w.state.load_tree(w.ids[1])
    assert before
    n = len(w.emb.calls), len(w.spy.embeds), len(w.spy.commits)
    fs.reads.clear()
    fs.touch(BIG)
    status = await w.engine().sync()
    assert status.state == "ready"
    assert [r for r in fs.reads if r[1] == FULL] == [(BIG, FULL)]
    assert (len(w.emb.calls), len(w.spy.embeds), len(w.spy.commits)) == n
    after = w.state.load_tree(w.ids[1])
    assert after
    assert after.files[BIG].mtime_ns == fs.mtimes[BIG] != before.files[BIG].mtime_ns  # the new stat is saved
    assert (after.files[BIG].leaf, after.files[BIG].chunks) == (before.files[BIG].leaf, 4)
    fs.reads.clear()
    await w.engine().sync()
    assert not [r for r in fs.reads if r[1] == FULL]  # and the next sync skips it again


async def test_a_touched_file_next_to_a_real_edit_sends_only_the_edit(tmp_path: Path) -> None:
    """The touched file hashes to its saved leaf, so it is not chunked or sent; only the edited file is."""
    w = await edit_world(tmp_path, tree_repo())
    n = len(w.spy.embeds)
    efs(w).touch("svc1/mod1/f3.py")
    efs(w).write("svc1/mod1/f4.py", "def changed():\n    return 'CHANGED'\n")
    await w.engine().sync()
    sent = {r.file_id for req in w.spy.embeds[n:] for r in req.rows}
    assert sent == {ids.file_id(w.k, "svc1/mod1/f4.py")}
    assert index_view(w) == index_view(await reference(tmp_path, w))


@pytest.mark.parametrize("how", ["size", "mtime", "inode"])
async def test_a_changed_size_mtime_or_inode_is_always_picked_up(tmp_path: Path, how: str) -> None:
    """Design 8 / AC-3.6: any one of (size, mtime_ns, ino) differing means the edit is seen, even when the
    other two are equal to the saved entry."""
    w = await edit_world(tmp_path, tree_repo())
    fs = efs(w)
    old = fs.files[BIG]
    new = big_module("b") if how != "size" else big_module("bb")
    fs.files[BIG] = new.encode()  # content swapped without touching the stat ...
    assert (len(old) == len(new)) == (how != "size")
    if how == "size":
        pass  # the length alone changed
    elif how == "mtime":
        fs.touch(BIG)
    else:
        fs.new_inode(BIG)
    before = len(w.emb.embedded)
    await w.engine().sync()
    assert len(w.emb.embedded) - before == 1
    assert "len('b" in w.emb.embedded[-1]
    ref = await reference(tmp_path, w)
    assert index_view(w) == index_view(ref)


async def test_same_size_mtime_and_inode_is_the_accepted_stat_cache_risk(tmp_path: Path) -> None:
    """Design 8 accepted risk, pinned so it cannot change silently: bytes rewritten with the same size, the
    same mtime_ns AND the same inode are not noticed by a normal sync (it is the documented price of not
    re-reading every file), and `sync(full=True)` (`scout sync --full`) re-hashes everything and heals it."""
    w = await edit_world(tmp_path, tree_repo())
    fs = efs(w)
    fs.write(BIG, big_module("b"), mtime=False)  # same size, same mtime, same inode, different bytes
    before = len(w.emb.embedded)
    await w.engine().sync()
    assert len(w.emb.embedded) == before  # skipped: accepted risk

    status = await w.engine().sync(full=True)
    assert status.state == "ready"
    assert len(w.emb.embedded) - before == 1
    ref = await reference(tmp_path, w)
    assert index_view(w) == index_view(ref)
    again = len(w.emb.calls)
    await w.engine().sync(full=True)  # a full re-hash of an unchanged tree embeds nothing
    assert len(w.emb.calls) == again


# --- CRLF, binary and oversize files ---
async def test_crlf_edit_keeps_lf_based_line_numbers(tmp_path: Path) -> None:
    """FR-C10 through the incremental path: a file turned CRLF is re-chunked on its bytes and the stored line
    ranges are the `\\n`-based ones (a `\\r` never shifts a number)."""
    w = await edit_world(tmp_path, tree_repo())
    crlf = big_module("b").replace("\n", "\r\n").encode()
    efs(w).write(BIG, crlf)
    await w.engine().sync()
    spans = chunker.chunk_file(BIG, crlf)
    assert len(spans) == 4
    repo_id = w.ids[0]
    fid = ids.file_id(w.k, BIG)
    got = sorted(
        rows.open_hit(
            w.k,
            TENANT,
            repo_id,
            WireHit(
                path_enc=r.attrs["path_enc"],
                loc_enc=r.attrs["loc_enc"],
                chunk_key=r.attrs["chunk_key"],
                file_hash=r.attrs["file_hash"],
            ),
        )
        for r in chunk_rows(w).values()
        if r.attrs["file_id"] == fid
    )
    assert got == sorted((BIG, s.start_line, s.end_line) for s in spans)
    assert index_view(w) == index_view(await reference(tmp_path, w))


async def test_binary_and_oversize_files_are_skipped_on_the_incremental_path(tmp_path: Path) -> None:
    """AC-13.2 on the incremental path: a file that turns binary or grows past 1 MiB leaves the index (rows
    and node entry), and a new binary or oversize file is never embedded or added to the tree."""
    files = tree_repo()
    files["svc1/mod1/was_text.py"] = "def was_text():\n    return 'WAS_TEXT'\n"
    files["svc1/mod1/was_small.py"] = "def was_small():\n    return 'WAS_SMALL'\n"
    w = await edit_world(tmp_path, files)
    fs = efs(w)
    n = len(w.emb.embedded)
    fs.write("svc1/mod1/was_text.py", b"BINARY_MARKER\x00\x01\x02" + bytes(range(256)))
    fs.write("svc1/mod1/was_small.py", "BIGFILE_MARKER " + "x" * MAX_FILE_BYTES)
    fs.write("svc1/mod1/new.bin", b"NEW_BINARY_MARKER\x00\xff")
    fs.write("svc1/mod1/new_big.txt", "NEW_BIG_MARKER " + "y" * MAX_FILE_BYTES)
    await w.engine().sync()

    assert w.emb.embedded[n:] == []
    blob = "\n".join(w.emb.embedded)
    assert not any(m in blob for m in ("BINARY_MARKER", "BIGFILE_MARKER", "NEW_BIG_MARKER"))
    saved = w.state.load_tree(w.ids[1])
    assert saved
    assert not [
        p
        for p in saved.files
        if p.startswith("svc1/mod1/")
        and p.rsplit("/", 1)[1] in ("was_text.py", "was_small.py", "new.bin", "new_big.txt")
    ]
    for gone in ("was_text.py", "was_small.py"):
        assert not [
            r for r in chunk_rows(w).values() if r.attrs["file_id"] == ids.file_id(w.k, f"svc1/mod1/{gone}")
        ]
    assert index_view(w) == index_view(await reference(tmp_path, w))


# --- failures, healing, history ---
@pytest.mark.parametrize("where", ["embed", "stage"])
async def test_crash_before_the_root_keeps_the_old_root_and_the_rerun_converges(
    tmp_path: Path, where: str
) -> None:
    """AC-3.8: a crash between the chunk writes and the root leaves `ready` on the OLD root (the saved tree is
    not advanced); the rerun converges to a from-scratch sync of the edited tree."""
    w = await edit_world(tmp_path, tree_repo())
    old_root = root_attrs(w)["root_hash"]  # type: ignore[index]
    efs(w).write(BIG, big_module("b"))
    crashing = Spy(w.gw)
    if where == "embed":
        crashing.fail_after = 0
    else:
        crashing.stage_fail_at = 0
    with pytest.raises(ScoutError, match="injected"):
        await w.engine(spy=crashing).sync()
    assert root_attrs(w)["root_hash"] == old_root  # type: ignore[index]
    saved = w.state.load_tree(w.ids[1])
    assert saved
    assert saved.root == old_root
    assert (disk_status(w)["state"], disk_status(w)["syncing"]) == ("ready", False)
    status = await w.engine().sync()
    assert status.state == "ready"
    assert index_view(w) == index_view(await reference(tmp_path, w))


async def test_an_edit_reverted_and_redone_leaves_exactly_the_current_rows(tmp_path: Path) -> None:
    """A -> B -> A -> B -> A: every sync's embed (replace + delete) really runs. The Idempotency-Key is the
    hash of the request body, which includes the per-row sealed `loc_enc` (fresh nonce each time), so no two
    syncs can collide on a key and replay a stored result over a newer state."""
    w = await edit_world(tmp_path, tree_repo())
    fs = efs(w)
    for tag in ("b", "a", "b", "a"):
        fs.write(BIG, big_module(tag))
        await w.engine().sync()
        fid = ids.file_id(w.k, BIG)
        hashes = {r.attrs["file_hash"] for r in chunk_rows(w).values() if r.attrs["file_id"] == fid}
        saved = w.state.load_tree(w.ids[1])
        assert saved
        assert hashes == {saved.files[BIG].leaf}  # exactly the current version, never a stale one
    assert index_view(w) == index_view(await reference(tmp_path, w))
    assert len(set(w.spy.idem)) == len(w.spy.idem)  # no key was ever reused


async def test_a_wiped_server_is_rebuilt_from_a_stale_local_tree(tmp_path: Path) -> None:
    """The stat cache says nothing changed but the server has no index: every file is chunked and sent."""
    w = await edit_world(tmp_path, tree_repo())
    ns = w.ids[2]
    await w.store.delete_namespace(ns)
    await w.store.delete_namespace(ids.tree_namespace(ns))
    status = await w.engine().sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert index_view(w) == index_view(await reference(tmp_path, w))


async def test_a_partial_index_heals_together_with_a_new_edit(tmp_path: Path) -> None:
    """Coverage < 1: the chunks of files the stat cache skips may be missing, so a later sync re-sends every
    file (nodes still only the changed path) and reaches `ready`."""
    w = await world(tmp_path, tree_repo(), fs_cls=EditFS)
    lossy = Spy(w.gw)
    lossy.keep_rows = 20
    assert (await w.engine(spy=lossy).sync()).state == "partial"
    efs(w).write(BIG, big_module("b"))
    status = await w.engine(spy=Spy(w.gw)).sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert index_view(w) == index_view(await reference(tmp_path, w))


async def test_an_old_state_file_without_chunk_counts_is_a_full_resync(tmp_path: Path) -> None:
    """A `trees/<branch>.json` written before 4.2 has no chunk counts: it loads as None, not as zeros."""
    w = await edit_world(tmp_path, tree_repo())
    p = w.state_dir / "trees" / f"{w.ids[1]}.json"
    doc = json.loads(p.read_text())
    for e in doc["files"].values():
        del e["chunks"]
    p.write_text(json.dumps(doc))
    assert StateDir(str(w.state_dir)).load_tree(w.ids[1]) is None
    n = len(w.emb.calls)
    status = await w.engine().sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert len(w.emb.calls) == n  # every text was cached
    assert index_view(w) == index_view(await reference(tmp_path, w))


# --- AC-3.1 / NFR-5: wire bytes of a one-file edit do not grow with repo size ---
DEPTH = 5
PER_LEAF = 10
FILES_AT = (300, 3_000, 30_000)


def deep_repo(files: int, depth: int = DEPTH) -> dict[str, str | bytes]:
    """`files` modules, `PER_LEAF` per directory, every path exactly `depth` directories deep; the tree's
    branching factor grows with `files` (so the fan-out of each node does), the depth does not."""
    leaves = math.ceil(files / PER_LEAF)
    fan = max(2, math.ceil(leaves ** (1 / depth)))
    out: dict[str, str | bytes] = {}
    for i in range(files):
        leaf, segs = i // PER_LEAF, []
        for _ in range(depth):
            segs.append(f"d{leaf % fan}")
            leaf //= fan
        out["/".join(segs) + f"/m{i}.py"] = f"def handler_{i}(request_{i}):\n    return request_{i} + {i}\n"
    return out


class SeedStore(FakeStore):
    """First syncs write into an empty namespace, so there is nothing to delete: `delete_by_file` is a no-op
    until `seeded` (the fake's O(rows) scan per file would make a 30k seed quadratic)."""

    seeded = False

    async def delete_by_file(self, ns: str, file_id: str, *, except_file_hash: str | None = None) -> int:
        if not self.seeded:
            return 0
        return await super().delete_by_file(ns, file_id, except_file_hash=except_file_hash)


class CheapEmbedder(FakeEmbedder):
    """One-hot vectors from a crc32: the scaling test measures wire bytes, not retrieval quality."""

    def vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        vec[zlib.crc32(text.encode()) % self.dims] = 1.0
        return vec


@dataclass
class Measured:
    files: int
    wire: int
    by_method: dict[str, int]
    stage_pages: int
    wall_s: float
    seed_s: float


async def measure_edit(tmp_path: Path, files: int, mp: pytest.MonkeyPatch, edits: int = 1) -> Measured:
    """First-sync `files` modules, then edit `edits` of them and meter the incremental sync."""
    dims = 16  # 1024-dim fake vectors for 30k rows would need ~2 GB
    mp.setattr(
        "scout.core.service.gateway.CHUNK_SCHEMA", NamespaceSchema(dims=dims, attrs=CHUNK_SCHEMA.attrs)
    )
    mp.setattr("scout.core.service.syncsvc.CACHE_SCHEMA", NamespaceSchema(dims=dims, attrs={}))
    store, emb, clock = SeedStore(), CheapEmbedder(dims=dims), FakeClock()
    gw = make_fake_gateway(TENANT, store=store, embedder=emb, clock=clock)
    state_dir = tmp_path / f"scale{files}x{edits}" / "git" / "scout"
    fs = EditFS(deep_repo(files), state_dir=str(state_dir))
    meter = WireMeter(gw)
    engine = SyncEngine(fs, cast(Gateway, meter), keys.derive(await gw.team_key()), clock)
    t0 = time.perf_counter()
    assert (await engine.sync()).state == "ready"
    seed_s = time.perf_counter() - t0
    store.seeded = True
    meter.reset()
    stride = len(fs.files) // (edits + 1)
    for n, target in enumerate(sorted(fs.files)[stride::stride][:edits]):
        fs.write(target, f"def edited_{n}(request):\n    return {7 + n} * request\n")
    t0 = time.perf_counter()
    status = await engine.sync()
    wall = time.perf_counter() - t0
    assert (status.state, status.coverage, status.files) == ("ready", 1.0, files)
    assert len(emb.embedded) == files + edits  # one new body per edited file, nothing else
    return Measured(
        files,
        meter.total_bytes,
        meter.by_method(),
        sum(1 for name, _, _ in meter.calls if name == "sync_stage_nodes"),
        wall,
        seed_s,
    )


@pytest.mark.slow
async def test_one_file_edit_wire_bytes_do_not_grow_with_repo_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC-3.1, NFR-5, PRD 4.3 item 2 (the Cursor property): the same one-file edit in repos of 300, 3,000 and
    30,000 files of equal depth costs bytes within 1.5x of each other (request + response, every gateway
    call), and the incremental sync stays far under the 10 s NFR against the fake."""
    results = [await measure_edit(tmp_path, n, monkeypatch) for n in FILES_AT]
    small, mid, big = results
    with capsys.disabled():
        print("\n[4.2] one-file edit, wire bytes (request + response):")
        for r in results:
            print(
                f"  {r.files:>6} files: {r.wire:>6} B  {r.by_method}  incremental {r.wall_s:.2f}s "
                f"(first sync {r.seed_s:.1f}s)"
            )
        print(f"  ratio 3k/300 = {mid.wire / small.wire:.2f}x, 30k/300 = {big.wire / small.wire:.2f}x")
    assert max(r.wire for r in results) <= 1.5 * min(r.wire for r in results)
    assert big.wall_s < 10  # NFR: incremental sync < 10 s (the fake is a lower bound)
    for r in results:
        assert r.stage_pages == 1  # a path's nodes fit one page
        assert "sync_nodes" not in r.by_method  # fast path: the server root is the saved root, no tree walk
        assert r.by_method["embed"] < 3_000  # one edited chunk body, not the repo
        assert r.by_method["sync_stage_nodes"] < 10_000  # depth x fan-out node rows, not the tree


async def test_wire_bytes_follow_changed_files_not_repo_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NFR-5, AC-3.1: in one 1,000-file repo, 8 edited files cost more than 1 but at most 8 times the
    single-edit bytes (the fixed per-sync calls amortise): bytes track the changed files, not the rest."""
    one = await measure_edit(tmp_path, 1_000, monkeypatch, edits=1)
    eight = await measure_edit(tmp_path, 1_000, monkeypatch, edits=8)
    assert one.wire < eight.wire <= 8 * one.wire
    assert eight.by_method["embed"] > one.by_method["embed"]  # more changed bodies, more embed bytes


async def test_wire_meter_counts_request_and_response_json_bytes(tmp_path: Path) -> None:
    w = await edit_world(tmp_path, {"a/b.py": "def f():\n    return 1\n"})
    meter = WireMeter(cast(Gateway, w.spy))
    req = StatusReq(repo_id=w.ids[0], branch_id=w.ids[1])
    res = cast(IndexStatus, await meter.index_status(req))
    assert meter.calls == [("index_status", len(req.model_dump_json()), len(res.model_dump_json()))]
    assert meter.total_bytes == meter.request_bytes + meter.response_bytes > 0
    meter.reset()
    assert meter.calls == []
