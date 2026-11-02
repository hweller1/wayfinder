"""Task 4.1: `SyncEngine` first full sync through the real `GatewayService` over fakes (FakeGateway).

FR-C8, FR-C9, FR-C11, AC-3.7, AC-3.8, AC-13.2. The engine is client-side: it talks only to the `Gateway`
port, so every assertion about the index reads the fake store the gateway wrote (never engine internals).
MinHash is not compared to a threshold here (only stored), so the 3-sigma margin rule does not apply.
"""

import ast
import asyncio
import importlib.util
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jsonschema
import pytest
from svc_helpers import RecordingStore

from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_fs import FakeFileSystem
from scout.adapters.fake_gateway import make_fake_gateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import EmbedReq, EmbedRes, StatusReq, WireHit
from scout.core.crypto import ids, keys, rows
from scout.core.local.ignore import MAX_FILE_BYTES
from scout.core.ports.gateway import Gateway
from scout.core.service.schema import ROOT_ID
from scout.core.sync import chunker, merkle
from scout.core.sync.engine import SyncConfig, SyncEngine, estimate_tokens
from scout.core.sync.state import StateDir

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
STATUS_SCHEMA = json.loads((ROOT / "schema" / "v1" / "status_file.json").read_text())
TENANT = "dev"

_spec = importlib.util.spec_from_file_location("gen_repo", FIXTURES / "gen_repo.py")
assert _spec
assert _spec.loader
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class Spy:
    """Wraps a `Gateway`: records the call order, can fail or perturb `embed`, tracks in-flight requests."""

    def __init__(self, gw: Gateway) -> None:
        self._gw = gw
        self.events: list[str] = []
        self.embeds: list[EmbedReq] = []
        self.idem: list[str | None] = []
        self.fail_after: int | None = None  # raise INTERNAL on the embed call after this many successes
        self.fail_with: list[ScoutError] = []  # raised (one per call, in order) before the real embed
        self.keep_rows: int | None = None  # drop rows beyond this total (a gateway that wrote only part)
        self.on_embed: Callable[[], None] | None = None
        self.stage_fail_at: int | None = None  # raise INTERNAL on the stage call with this 0-based index
        self.staged: list[Any] = []  # every StageNodesReq, in order
        self.commits: list[Any] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._ok = 0
        self._kept = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gw, name)

    async def embed(self, req: EmbedReq, *, idempotency_key: str | None = None) -> EmbedRes:
        self.events.append("embed_start")
        self.embeds.append(req)
        self.idem.append(idempotency_key)
        if self.on_embed:
            self.on_embed()
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)  # a real network call yields; lets the workers overlap
            if self.fail_with:
                raise self.fail_with.pop(0)
            if self.fail_after is not None and self._ok >= self.fail_after:
                raise ScoutError(ErrorCode.INTERNAL, "injected crash")
            if self.keep_rows is not None:
                keep = max(self.keep_rows - self._kept, 0)
                self._kept += min(keep, len(req.rows))
                req = req.model_copy(update={"rows": req.rows[:keep]})
            res = await self._gw.embed(req, idempotency_key=idempotency_key)
            self._ok += 1
            return res
        finally:
            self.in_flight -= 1
            self.events.append("embed_end")

    async def sync_stage_nodes(self, req: Any) -> Any:
        self.events.append("stage")
        if self.stage_fail_at is not None and len(self.staged) == self.stage_fail_at:
            self.staged.append(req)
            raise ScoutError(ErrorCode.INTERNAL, "injected stage crash")
        self.staged.append(req)
        return await self._gw.sync_stage_nodes(req)

    async def sync_commit(self, req: Any) -> Any:
        self.commits.append(req)
        self.events.append("commit")
        return await self._gw.sync_commit(req)


class HookFS(FakeFileSystem):
    """FakeFileSystem whose `read_bytes` calls a hook after every read (to mutate a file mid-sync)."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.reads: list[tuple[str, int | None]] = []
        self.after_read: Callable[[str, int | None], None] | None = None

    def read_bytes(self, rel: str, *, max_bytes: int | None = None) -> bytes:
        data = super().read_bytes(rel, max_bytes=max_bytes)
        self.reads.append((rel, max_bytes))
        if self.after_read:
            self.after_read(rel, max_bytes)
        return data


@dataclass
class World:
    store: RecordingStore
    emb: FakeEmbedder
    clock: FakeClock
    gw: Gateway
    spy: Spy
    fs: FakeFileSystem
    state_dir: Path
    k: keys.Subkeys

    def engine(self, cfg: SyncConfig | None = None, spy: Spy | None = None) -> SyncEngine:
        return SyncEngine(self.fs, cast(Gateway, spy or self.spy), self.k, self.clock, cfg)

    @property
    def state(self) -> StateDir:
        return StateDir(str(self.state_dir))

    @property
    def ids(self) -> tuple[str, str, str]:
        st = self.state.read_state()
        assert st.repo_id
        assert st.last_branch
        return st.repo_id, st.last_branch, ids.namespace(TENANT, st.repo_id, st.last_branch)


async def world(
    tmp_path: Path,
    files: dict[str, str | bytes],
    *,
    k: keys.Subkeys | None = None,
    fs_cls: type[FakeFileSystem] = FakeFileSystem,
    name: str = "w",
) -> World:
    store, emb, clock = RecordingStore(), FakeEmbedder(), FakeClock()
    gw = make_fake_gateway(TENANT, store=store, embedder=emb, clock=clock)
    state_dir = tmp_path / name / "git" / "scout"
    fs = fs_cls(files, state_dir=str(state_dir))
    return World(store, emb, clock, gw, Spy(gw), fs, state_dir, k or keys.derive(await gw.team_key()))


def repo_small() -> dict[str, str | bytes]:
    return dict(gen.load_dir(FIXTURES / "repo_small"))


def tree_rows(w: World) -> dict[str, Any]:
    _, _, ns = w.ids
    return w.store.rows(ids.tree_namespace(ns))


def chunk_rows(w: World) -> dict[str, Any]:
    return w.store.rows(w.ids[2])


def root_attrs(w: World) -> dict[str, Any] | None:
    r = tree_rows(w).get(ROOT_ID)
    return dict(r.attrs) if r else None


def disk_status(w: World) -> dict[str, Any]:
    doc = json.loads((w.state_dir / "status.json").read_text())
    jsonschema.validate(doc, STATUS_SCHEMA)
    return doc


def indexable_chunks(files: dict[str, str | bytes], skip: tuple[str, ...] = ()) -> int:
    return sum(
        len(chunker.chunk_file(p, t.encode() if isinstance(t, str) else t))
        for p, t in files.items()
        if p not in skip
    )


# --- AC-3.7 / FR-C8: first sync reaches ready on the fixture repo ---
async def test_first_sync_of_repo_small_is_ready_and_status_validates(tmp_path: Path) -> None:
    """AC-3.7, FR-C9: index ready, rows exist, status.json schema-valid, `index_status` agrees."""
    w = await world(tmp_path, repo_small())
    status = await w.engine().sync()

    repo_id, branch_id, ns = w.ids
    assert (status.state, status.coverage, status.semantic_enabled, status.syncing) == (
        "ready",
        1.0,
        True,
        False,
    )
    assert status.files == 13  # 15 fixture files minus `.env` (denylist) and `generated/session_gen.py`
    served = await w.gw.index_status(StatusReq(repo_id=repo_id, branch_id=branch_id))
    assert (served.state, served.coverage, served.semantic_enabled) == ("ready", 1.0, True)
    assert len(chunk_rows(w)) > 0
    assert await w.store.namespace_exists(ns)
    assert await w.store.namespace_exists(ids.tree_namespace(ns))
    doc = disk_status(w)
    assert (doc["state"], doc["coverage"], doc["syncing"]) == ("ready", 1.0, False)
    assert doc["last_sync_at"]
    assert doc["gateway_ok_at"]
    assert doc["commit"] == "1" * 40
    saved = w.state.load_tree(branch_id)
    assert saved is not None
    assert saved.root == root_attrs(w)["root_hash"]  # type: ignore[index]
    assert len(saved.files) == status.files


async def test_secrets_and_ignored_files_are_never_embedded(tmp_path: Path) -> None:
    """AC-13.2, FR-C8: denylist files, `.scoutignore`d dirs, > 1 MB and binaries are never embedded."""
    files = repo_small()
    files.update(gen.SECRET_FILES)
    files["blob/big.txt"] = "BIGFILE_MARKER " + "x" * MAX_FILE_BYTES
    files["blob/data.bin"] = b"BINARY_MARKER\x00\x01\x02" + bytes(range(256))
    w = await world(tmp_path, files)
    status = await w.engine().sync()

    assert status.state == "ready"
    blob = "\n".join(w.emb.embedded)
    for marker in (*gen.SECRET_MARKERS, "BIGFILE_MARKER", "BINARY_MARKER", "SESSION_LOOKUP", "API_TOKEN"):
        assert marker not in blob
    generated = (FIXTURES / "repo_small" / "generated" / "session_gen.py").read_text().splitlines()[0]
    assert generated
    assert generated not in blob  # `.scoutignore` excludes generated/
    saved = w.state.load_tree(w.ids[1])
    assert saved is not None
    excluded = ("generated/", "blob/", "keys/", "deploy/", "config/")
    assert not [p for p in saved.files if p.startswith(excluded) or p == ".env"]
    # nothing in the store carries a secret either (sealed rows hold ciphertext only)
    dumped = repr([w.store.dump(n) for n in w.store.names() if n.startswith(f"scout_{TENANT}_")])
    assert not any(m in dumped for m in gen.SECRET_MARKERS)


# --- 300-file fixture ---
async def test_first_sync_of_300_generated_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = gen.make_repo(300)
    w = await world(tmp_path, dict(files))
    t0 = time.perf_counter()
    status = await w.engine().sync()
    wall = time.perf_counter() - t0

    expected = indexable_chunks(dict(files))
    assert expected >= 300
    root = root_attrs(w)
    assert root is not None
    assert (root["expected_chunks"], root["written_chunks"]) == (expected, expected)
    assert len(chunk_rows(w)) == expected
    assert (status.state, status.files, status.coverage) == ("ready", 300, 1.0)
    assert len(w.emb.embedded) == len(set(w.emb.embedded)) == expected  # every unique text exactly once
    assert w.emb.batches >= 1
    assert wall < 60
    with capsys.disabled():
        print(
            f"\n[4.1] 300 files: {expected} chunks, {len(w.emb.embedded)} embedded texts, "
            f"{w.emb.batches} embedder batches, {len(w.spy.embeds)} embed requests, {wall:.2f}s"
        )


def dir_repo(dirs: int) -> dict[str, str | bytes]:
    return {f"d{i}/f.py": f"def fn_{i}():\n    return {i}\n" for i in range(dirs)}


@pytest.mark.slow
async def test_2000_directories_first_sync_to_ready(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """FR-C8, AC-3.7, task 4.1a: 2,000 directories (2,001 node rows, 5 pages of <= 500) sync to `ready`; the
    node pages all precede the commit, which carries no node rows and the page count."""
    w = await world(tmp_path, dir_repo(2000))
    t0 = time.perf_counter()
    status = await w.engine().sync()
    wall = time.perf_counter() - t0

    assert (status.state, status.files, status.coverage) == ("ready", 2000, 1.0)
    nodes = [r for r in tree_rows(w).values() if r.attrs.get("kind") == "node"]
    assert len(nodes) == 2001  # 2000 directories + the root directory
    assert all(r.attrs.get("kind") in ("node", "root") for r in tree_rows(w).values())  # markers cleaned up
    pages = w.spy.staged
    assert [(p.page, p.pages) for p in pages] == [(n, 5) for n in range(5)]
    assert [len(p.node_upserts) for p in pages] == [500, 500, 500, 500, 1]
    commit = w.spy.commits[-1]
    assert (commit.staged_pages, commit.node_upserts, commit.node_deletes) == (5, [], [])
    ev = w.spy.events
    assert max(i for i, e in enumerate(ev) if e == "embed_end") < ev.index("stage")
    assert ev.index("stage") < ev.index("commit")
    assert ev.count("stage") == 5
    assert wall < 60
    saved = w.state.load_tree(w.ids[1])
    assert saved is not None
    assert saved.root == root_attrs(w)["root_hash"]  # type: ignore[index]
    again = await w.engine().sync()  # the stored tree is exactly the local one: nothing to do
    assert again.state == "ready"
    assert len(w.spy.staged) == 5
    with capsys.disabled():
        print(f"\n[4.1a] 2000 directories: {len(nodes)} node rows in {len(pages)} pages, {wall:.2f}s")


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.slow
async def test_crash_after_node_page_k_then_rerun_converges(tmp_path: Path, k: int) -> None:
    """AC-3.8, task 4.1a: a crash after page k leaves no root (index absent); the re-run embeds nothing new,
    restages every page and ends with a store dump equal to an uninterrupted run."""
    files = dir_repo(1200)  # 1,201 node rows: 3 pages
    ref = await world(tmp_path, dict(files), name="ref")
    await ref.engine().sync()
    w = await world(tmp_path, dict(files), k=ref.k, name="crash")
    w.state_dir.mkdir(parents=True)  # same clone_id as the reference run, so the namespaces match
    shutil.copy(ref.state_dir / "state.json", w.state_dir / "state.json")
    crashing = Spy(w.gw)
    crashing.stage_fail_at = k  # fail ON page index k: pages before it are staged
    with pytest.raises(ScoutError, match="injected stage crash"):
        await w.engine(spy=crashing).sync()
    assert root_attrs(w) is None  # no root until commit
    repo_id, branch_id, _ = w.ids
    assert (await w.gw.index_status(StatusReq(repo_id=repo_id, branch_id=branch_id))).state == "absent"
    assert (disk_status(w)["state"], disk_status(w)["syncing"]) == ("absent", False)
    staged_before = len([r for r in tree_rows(w).values() if r.attrs.get("kind") == "node"])
    assert staged_before == 500 * k  # exactly the pages before the failing one landed
    first_run = list(w.emb.embedded)

    status = await w.engine().sync()
    assert status.state == "ready"
    assert w.emb.embedded == first_run  # the re-run embedded nothing: every text is cached
    assert normalized(w) == normalized(ref)


async def test_a_gateway_that_loses_a_page_makes_the_commit_fail_not_succeed(tmp_path: Path) -> None:
    """Task 4.1a: a gateway that acknowledges but drops page 1 gets a refused commit and writes no root."""
    w = await world(tmp_path, dir_repo(1200))

    class DropsPage(Spy):
        async def sync_stage_nodes(self, req: Any) -> Any:
            if req.page == 1:
                self.staged.append(req)
                return None  # acknowledged but never written
            return await super().sync_stage_nodes(req)

    with pytest.raises(ScoutError) as ei:
        await w.engine(spy=DropsPage(w.gw)).sync()
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    assert root_attrs(w) is None
    assert disk_status(w)["state"] == "absent"


async def test_over_the_stage_cap_is_refused_before_any_spend(tmp_path: Path) -> None:
    """The service takes at most 400 pages of 500 node rows: a bigger tree fails before embedding."""
    w = await world(tmp_path, dir_repo(40))
    cfg = SyncConfig()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("scout.core.sync.engine.MAX_NODE_PAGES", 0)
        with pytest.raises(ScoutError) as ei:
            await w.engine(cfg).sync()
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT
    assert w.emb.calls == []
    assert disk_status(w)["state"] == "absent"


# --- no TOCTOU: one read per file ---
FULL = MAX_FILE_BYTES + 1  # `max_bytes` of the engine's one full read (the matcher's sniff passes 8192)
TARGET = "pkg/target.py"
ORIGINAL = "def original_marker_fn():\n    return 'ORIGINAL_BODY'\n"
MUTATED = "def mutated_marker_fn():\n    return 'MUTATED_BODY'\n"


def twin_files() -> dict[str, str | bytes]:
    return {TARGET: ORIGINAL, "pkg/other.py": "def other():\n    return 2\n"}


async def test_bytes_are_read_once_and_hash_chunks_tokens_agree(tmp_path: Path) -> None:
    """FR-C9, design 8: a file edited right after the engine read it still yields a consistent record."""
    w = await world(tmp_path, twin_files(), fs_cls=HookFS)
    fs = cast(HookFS, w.fs)

    def mutate(rel: str, max_bytes: int | None) -> None:
        if rel == TARGET and max_bytes == FULL:
            fs.files[TARGET] = MUTATED.encode()

    fs.after_read = mutate
    await w.engine().sync()

    assert [r for r in fs.reads if r == (TARGET, FULL)] == [(TARGET, FULL)]  # exactly one full read
    assert not [r for r in fs.reads if r[0] == TARGET and r[1] is None]
    leaf = merkle.leaf_hash(w.k, ORIGINAL.encode())
    saved = w.state.load_tree(w.ids[1])
    assert saved is not None
    assert saved.files[TARGET].leaf == leaf
    assert saved.files[TARGET].size == len(ORIGINAL)  # stat taken before the read: next sync sees the edit
    target_rows = [r for r in chunk_rows(w).values() if r.attrs["file_id"] == ids.file_id(w.k, TARGET)]
    assert target_rows
    assert {r.attrs["file_hash"] for r in target_rows} == {leaf}
    blob = "\n".join(w.emb.embedded)
    assert "ORIGINAL_BODY" in blob
    assert "MUTATED_BODY" not in blob
    toks = set().union(*(set(r.attrs["tokens"]) for r in target_rows))
    from scout.core.crypto.tokens import hmac_tokens, identifier_tokens

    assert set(hmac_tokens(w.k, identifier_tokens("original_marker_fn"))) <= toks
    assert not set(hmac_tokens(w.k, ["mutatedmarkerfn"])) & toks


async def test_an_edit_before_the_read_is_seen_consistently(tmp_path: Path) -> None:
    """The mutation lands after the matcher's sniff but before the engine's read: all four views agree."""
    w = await world(tmp_path, twin_files(), fs_cls=HookFS)
    fs = cast(HookFS, w.fs)

    def mutate(rel: str, max_bytes: int | None) -> None:
        if rel == TARGET and max_bytes == 8192:
            fs.files[TARGET] = MUTATED.encode()

    fs.after_read = mutate
    await w.engine().sync()
    saved = w.state.load_tree(w.ids[1])
    assert saved is not None
    assert saved.files[TARGET].leaf == merkle.leaf_hash(w.k, MUTATED.encode())
    blob = "\n".join(w.emb.embedded)
    assert "MUTATED_BODY" in blob
    assert "ORIGINAL_BODY" not in blob


# --- AC-3.7: batches ---
async def test_batches_are_token_capped_ordered_and_parallel(tmp_path: Path) -> None:
    """Each embed request <= 100K estimated tokens, <= 6 in flight, commit strictly after the last embed."""
    files = {
        f"big/f{i}.py": "\n".join(f"def fn_{i}_{j}(x):\n    return x + {i * 1000 + j}\n" for j in range(1500))
        for i in range(24)
    }
    w = await world(tmp_path, dict(files))
    await w.engine().sync()

    per_req = [sum(estimate_tokens(t) for t in req.texts.values()) for req in w.spy.embeds]
    assert len(per_req) >= 3
    assert max(per_req) <= 100_000
    assert 1 < w.spy.max_in_flight <= 6
    ev = w.spy.events
    assert ev.count("commit") == 1
    assert ev.index("commit") > max(i for i, e in enumerate(ev) if e == "embed_end")
    assert all(k and 0 < len(k) <= 128 for k in w.spy.idem)
    assert len(set(w.spy.idem)) == len(w.spy.idem)


async def test_worker_count_is_configurable_and_bounded(tmp_path: Path) -> None:
    files = gen.make_repo(60)
    w = await world(tmp_path, dict(files))
    await w.engine(SyncConfig(batch_tokens=600, workers=2)).sync()
    assert len(w.spy.embeds) > 6
    assert w.spy.max_in_flight == 2


async def test_transient_gateway_errors_are_retried_with_backoff(tmp_path: Path) -> None:
    w = await world(tmp_path, gen.make_repo(20))
    w.spy.fail_with = [ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "tpuf 503")]
    status = await w.engine().sync()
    assert status.state == "ready"
    assert w.clock.sleeps == [1.0]


async def test_conflict_replans_by_resending_every_text(tmp_path: Path) -> None:
    w = await world(tmp_path, gen.make_repo(20))
    w.spy.fail_with = [ScoutError(ErrorCode.CONFLICT, "cache entry missing", retryable=True)]
    status = await w.engine().sync()
    assert status.state == "ready"
    first, retry = w.spy.embeds[0], w.spy.embeds[1]
    assert retry.rows == first.rows
    assert set(retry.texts) >= {r.chunk_key for r in retry.rows}  # every row has its text again


# --- crash-resume ---
def normalized(w: World) -> dict[str, Any]:
    """Every namespace of the tenant, nonce-free: chunk rows (attrs, decrypted loc, vector), tree rows
    (decrypted children, hashes, root counters) and the cache/meta rows."""
    repo_id, _, ns = w.ids
    out: dict[str, Any] = {}
    for name in w.store.names():
        if not name.startswith(f"scout_{TENANT}_"):
            continue
        dump = w.store.dump(name)
        norm: dict[str, Any] = {}
        for rid, (attrs, vector) in dump["rows"].items():
            a = dict(attrs)
            if name == ns:
                hit = WireHit(
                    path_enc=a["path_enc"],
                    loc_enc=a.pop("loc_enc"),
                    chunk_key=a["chunk_key"],
                    file_hash=a["file_hash"],
                )
                a["loc"] = rows.open_hit(w.k, TENANT, repo_id, hit)
            elif a.get("kind") == "node":
                a["children"] = rows.open_node(w.k, TENANT, repo_id, rid, a.pop("children_enc"))
            norm[rid] = (a, vector)
        out[name] = {"has_vector": dump["has_vector"], "rows": norm}
    return out


async def test_crash_mid_embed_then_rerun_embeds_only_the_remainder(tmp_path: Path) -> None:
    """AC-3.8: a crash reads absent, never ready; the re-run embeds only what is missing; same end state."""
    files = gen.make_repo(80)
    cfg = SyncConfig(batch_tokens=700, workers=2)

    ref = await world(tmp_path, dict(files), name="ref")
    await ref.engine(cfg).sync()
    total = list(ref.emb.embedded)
    assert len(ref.spy.embeds) >= 8

    w = await world(tmp_path, dict(files), k=ref.k, name="crash")
    (w.state_dir).mkdir(parents=True)  # same clone_id as the reference run, so the namespaces match
    shutil.copy(ref.state_dir / "state.json", w.state_dir / "state.json")
    crashing = Spy(w.gw)
    crashing.fail_after = 3
    with pytest.raises(ScoutError, match="injected crash"):
        await w.engine(cfg, crashing).sync()

    first_run = list(w.emb.embedded)
    assert 0 < len(first_run) < len(total)
    assert root_attrs(w) is None  # no root row until commit: the crashed first sync is not `ready`
    repo_id, branch_id, _ = w.ids
    assert (await w.gw.index_status(StatusReq(repo_id=repo_id, branch_id=branch_id))).state == "absent"
    doc = disk_status(w)
    assert (doc["state"], doc["syncing"]) == ("absent", False)

    status = await w.engine(cfg).sync()  # re-run on the same store, embedder and state dir
    second_run = w.emb.embedded[len(first_run) :]
    assert status.state == "ready"
    assert set(first_run).isdisjoint(second_run)  # nothing embedded twice
    assert sorted(first_run + second_run) == sorted(total)  # and exactly the remainder was embedded
    assert normalized(w) == normalized(ref)  # end state identical to an uninterrupted run
    resumed = [t for req in w.spy.embeds for t in req.texts.values()]
    assert sorted(resumed) == sorted(second_run)  # text only travels for cache misses (sync_plan is honoured)


async def test_edit_between_crash_and_rerun_replaces_the_stale_rows(tmp_path: Path) -> None:
    """FR-C9: rows of a file's older version written before a crash are deleted by `replace` on the re-run."""
    files = gen.make_repo(40)
    w = await world(tmp_path, dict(files))
    crashing = Spy(w.gw)
    crashing.fail_after = 3
    cfg = SyncConfig(batch_tokens=700, workers=1)
    with pytest.raises(ScoutError, match="injected crash"):
        await w.engine(cfg, crashing).sync()
    victim = sorted(files)[0]  # chunked first, so its old rows were written before the crash
    old_hash = merkle.leaf_hash(w.k, files[victim].encode())
    fid = ids.file_id(w.k, victim)
    before = [r for r in chunk_rows(w).values() if r.attrs["file_id"] == fid]
    assert before
    assert {r.attrs["file_hash"] for r in before} == {old_hash}

    edited = files[victim] + "\n\ndef added_after_crash():\n    return 'EDITED_AFTER_CRASH'\n"
    cast(FakeFileSystem, w.fs).files[victim] = edited.encode()
    status = await w.engine(cfg).sync()

    new_hash = merkle.leaf_hash(w.k, edited.encode())
    after = [r for r in chunk_rows(w).values() if r.attrs["file_id"] == fid]
    assert status.state == "ready"
    assert {r.attrs["file_hash"] for r in after} == {new_hash}
    expected = indexable_chunks({**files, victim: edited})
    assert len(chunk_rows(w)) == expected
    assert root_attrs(w)["written_chunks"] == expected  # type: ignore[index]


async def test_rerun_after_a_complete_sync_embeds_nothing(tmp_path: Path) -> None:
    w = await world(tmp_path, gen.make_repo(30))
    await w.engine().sync()
    embedded, requests = len(w.emb.embedded), len(w.spy.embeds)
    status = await w.engine().sync()
    assert (status.state, status.coverage, status.files) == ("ready", 1.0, 30)
    assert (len(w.emb.embedded), len(w.spy.embeds)) == (embedded, requests)


# --- readiness: the 80% rule ---
@pytest.mark.parametrize(
    ("keep", "state", "semantic"), [(79, "partial", False), (80, "ready", True), (100, "ready", True)]
)
async def test_status_json_and_index_status_follow_the_80_percent_rule(
    tmp_path: Path, keep: int, state: str, semantic: bool
) -> None:
    """AC-3.8, PRD 4.3 item 6: coverage = written/expected; < 0.8 partial, >= 0.8 ready; all views agree."""
    files = {f"src/m{i}.py": f"def fn_{i}():\n    return {i}\n" for i in range(100)}
    w = await world(tmp_path, dict(files))
    w.spy.keep_rows = keep
    status = await w.engine().sync()

    repo_id, branch_id, _ = w.ids
    served = await w.gw.index_status(StatusReq(repo_id=repo_id, branch_id=branch_id))
    doc = disk_status(w)
    root = root_attrs(w)
    assert root is not None
    assert (root["expected_chunks"], root["written_chunks"]) == (100, keep)
    assert status.state == served.state == doc["state"] == state
    assert status.semantic_enabled == served.semantic_enabled == semantic
    assert status.coverage == pytest.approx(keep / 100)
    assert doc["coverage"] == pytest.approx(keep / 100)
    assert served.coverage == pytest.approx(keep / 100)


async def test_partial_index_heals_on_rerun(tmp_path: Path) -> None:
    files = {f"src/m{i}.py": f"def fn_{i}():\n    return {i}\n" for i in range(100)}
    w = await world(tmp_path, dict(files))
    w.spy.keep_rows = 50
    assert (await w.engine().sync()).state == "partial"
    healed = await w.engine(spy=Spy(w.gw)).sync()  # a gateway that writes everything this time
    assert (healed.state, healed.coverage) == ("ready", 1.0)
    assert len(chunk_rows(w)) == 100
    assert disk_status(w)["state"] == "ready"


async def test_status_json_shows_syncing_during_a_sync_and_is_restored_on_failure(tmp_path: Path) -> None:
    w = await world(tmp_path, gen.make_repo(10))
    seen: list[dict[str, Any]] = []
    w.spy.on_embed = lambda: seen.append(disk_status(w))
    await w.engine().sync()
    assert seen
    assert all(s["syncing"] is True and s["state"] == "absent" for s in seen)
    assert disk_status(w)["syncing"] is False

    w2 = await world(tmp_path, gen.make_repo(10), name="w2")
    w2.spy.fail_after = 0
    with pytest.raises(ScoutError):
        await w2.engine().sync()
    assert (disk_status(w2)["state"], disk_status(w2)["syncing"]) == ("absent", False)


# --- file cap, lock, wait=False, empty repo, layering ---
async def test_file_cap_is_honoured_with_a_warning(tmp_path: Path) -> None:
    """Design 15: 100k-file cap with a warning list (configurable here so the test stays small)."""
    assert SyncConfig().max_files == 100_000
    files = {f"src/m{i:03d}.py": f"def fn_{i}():\n    return {i}\n" for i in range(50)}
    w = await world(tmp_path, dict(files))
    eng = w.engine(SyncConfig(max_files=20))
    status = await eng.sync()
    assert status.files == 20
    assert len(w.state.load_tree(w.ids[1]).files) == 20  # type: ignore[union-attr]
    assert any("file cap 20" in m and "30 more" in m for m in eng.warnings)
    w2 = await world(tmp_path, dict(files), name="w2")
    eng2 = w2.engine()
    await eng2.sync()
    assert not any("file cap" in m for m in eng2.warnings)


async def test_a_second_sync_while_the_lock_is_held_does_not_embed(tmp_path: Path) -> None:
    w = await world(tmp_path, gen.make_repo(10))
    w.state_dir.mkdir(parents=True)
    with w.state.lock() as ok:
        assert ok
        status = await w.engine().sync()
    assert status.syncing is True
    assert status.state == "absent"
    assert w.emb.calls == []


async def test_wait_false_returns_early_and_join_finishes(tmp_path: Path) -> None:
    w = await world(tmp_path, gen.make_repo(10))
    eng = w.engine()
    early = await eng.sync(wait=False)
    assert early.syncing is True
    assert early.state == "absent"
    done = await eng.join()
    assert done is not None
    assert (done.state, done.syncing) == ("ready", False)
    assert root_attrs(w) is not None


async def test_empty_repo_stays_absent_and_makes_no_gateway_writes(tmp_path: Path) -> None:
    w = await world(tmp_path, {".env": "A=1\n", "bin.dat": b"\x00\x01"})
    status = await w.engine().sync()
    assert (status.state, status.files, status.semantic_enabled) == ("absent", 0, False)
    assert w.emb.calls == []
    assert disk_status(w)["state"] == "absent"
    assert w.spy.events == []


def test_engine_never_touches_the_vector_store_or_embedder() -> None:
    """Design port table / lint-imports contract 4: the client sync engine speaks only the Gateway port."""
    import scout.core.sync.engine as engine_mod

    imported: set[str] = set()
    for node in ast.walk(ast.parse(Path(engine_mod.__file__).read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert "scout.core.ports.gateway" in imported
    assert not {"scout.core.ports.store", "scout.core.ports.embedder"} & imported
    assert not [m for m in imported if m.startswith(("scout.adapters", "scout.core.service"))]
