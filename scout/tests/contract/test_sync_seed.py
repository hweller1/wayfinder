"""Task 4.4: `SyncEngine` teammate seed flow (MinHash match, paged proofs, fail-closed search) through the
real `GatewayService` over fakes. FR-C16, FR-C17, AC-4.2, AC-4.3 (design 5.4, 8; PRD 4.4).

Clone A and clone B share one store, one gateway and one `K_team`; each has its own state dir (so its own
`clone_id` and branch namespace). MinHash margin rule (3 standard errors, SE ~ 0.04 at 128 slots): the related
checkouts have true Jaccard >= 0.86 (A=40 files, B shares 37 -> 37/43), the unrelated one has a disjoint leaf
set (true Jaccard 0), against the 0.55 threshold.
"""

import statistics
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from svc_helpers import proofs_row
from test_sync_branch import BranchFS, Rival, bfs, branch_world, chunk_view, fresh_ref, reversed_pair
from test_sync_first import Spy, World, chunk_rows, disk_status, gen, root_attrs
from test_sync_incremental import index_view

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import MAX_PROOFS, EmbedReq, SearchReq, SeedReq, SeedRes, WireHit
from scout.core.crypto import ids, rows
from scout.core.crypto.tokens import hmac_tokens, identifier_tokens
from scout.core.ports.gateway import Gateway
from scout.core.service.syncsvc import meta_namespace, proofs_row_id, sketch_row_id
from scout.core.sync import merkle
from scout.core.sync.state import StateDir

TENANT = "dev"
Hook = Callable[[], Awaitable[None]]


class SeedSpy(Spy):
    """`Spy` that records `seed` calls and can run a coroutine at three moments of B's sync: right after
    the seed completed, before the first embed request and before the root commit."""

    def __init__(self, gw: Gateway) -> None:
        super().__init__(gw)
        self.seeds: list[tuple[SeedReq, SeedRes]] = []
        self.fail_seed: list[ScoutError] = []
        self.after_seed: Hook | None = None
        self.first_embed: Hook | None = None
        self.before_commit: Hook | None = None

    async def seed(self, req: SeedReq) -> SeedRes:
        if self.fail_seed:
            raise self.fail_seed.pop(0)
        res = await self._gw.seed(req)
        self.seeds.append((req, res))
        hook = self.after_seed
        if hook is not None and res.seeded and res.complete:
            self.after_seed = None
            await hook()
        return res

    async def embed(self, req: EmbedReq, *, idempotency_key: str | None = None) -> Any:
        hook, self.first_embed = self.first_embed, None
        if hook is not None:
            await hook()
        return await super().embed(req, idempotency_key=idempotency_key)

    async def sync_commit(self, req: Any) -> Any:
        hook, self.before_commit = self.before_commit, None
        if hook is not None:
            await hook()
        return await super().sync_commit(req)


def seed_spy(w: World) -> SeedSpy:
    assert isinstance(w.spy, SeedSpy)
    return w.spy


def repo(n: int = 40) -> dict[str, str | bytes]:
    return dict(gen.make_repo(n))


def unrelated(n: int = 40) -> dict[str, str | bytes]:
    """No file in common with `repo()`: other paths and other bodies (true Jaccard 0)."""
    return {f"lib{i % 4}/other_{i}.py": gen._module(1000 + i, 12) for i in range(n)}  # pyright: ignore[reportPrivateUsage]


def edit(files: dict[str, str | bytes], count: int = 3) -> tuple[dict[str, str | bytes], list[str]]:
    out, changed = dict(files), sorted(files)[:count]
    for p in changed:
        out[p] = f"{files[p]!s}\n# edited by B\n"
    return out, changed


async def teammate(tmp_path: Path, a: World, files: dict[str, str | bytes], name: str = "b") -> World:
    """A second checkout of the same tenant: same store/gateway/keys, its own state dir and fs."""
    state_dir = tmp_path / name / "git" / "scout"
    fs = BranchFS(files, state_dir=str(state_dir))
    return World(a.store, a.emb, a.clock, a.gw, SeedSpy(a.gw), fs, state_dir, a.k)


def leaf_set(w: World, files: dict[str, str | bytes]) -> set[str]:
    return {merkle.leaf_hash(w.k, v if isinstance(v, bytes) else v.encode()) for v in files.values()}


def b_namespaces(b: World, repo_id: str) -> tuple[str, str]:
    """`(ns, tree)` B's first sync will use: its clone id is created now (the engine reads the same file)."""
    clone_id = StateDir.open(b.fs).clone_id()
    ns = ids.namespace(TENANT, repo_id, ids.branch_id(b.k, repo_id, clone_id, "main"))
    return ns, ids.tree_namespace(ns)


async def search(w: World, query: str, k: int = 30) -> list[WireHit]:
    repo_id, branch_id, _ns = w.ids
    w.clock.advance(1.0)  # the gateway rate-limits searches per key; the fake clock refills the bucket
    toks = hmac_tokens(w.k, identifier_tokens(query))
    res = await w.gw.search(
        SearchReq(repo_id=repo_id, branch_id=branch_id, query=query, k=k, tokens=toks, overfetch=5)
    )
    return res.hits


def hit_path(w: World, hit: WireHit) -> str:
    return rows.open_hit(w.k, TENANT, w.ids[0], hit)[0]


def doc_bodies(w: World) -> list[str]:
    """Chunk texts this checkout's engine sent for embedding (the embedder's own log also holds queries)."""
    return [t for req in w.spy.embeds for t in req.texts.values()]


# --- the end-to-end seed (AC-4.3, FR-C16, FR-C17) -------------------------------------------------------
async def test_teammate_seeds_embeds_only_the_changed_files_and_searches_are_proof_filtered(
    tmp_path: Path,
) -> None:
    """AC-4.3: A syncs; B (new clone id, same K_team, 3 files changed) seeds, embeds only the changed files,
    while seeded-but-uncommitted B's searches return only hits for files B can prove (never A's version of a
    file B changed); the proofs are gone after the commit and every B file is then searchable."""
    files = repo()
    a = await branch_world(tmp_path, files, "a")
    a_total = len(doc_bodies(a))
    b_files, changed = edit(files)
    a_old = {p: merkle.leaf_hash(a.k, files[p].encode()) for p in changed}  # type: ignore[union-attr]
    b_leaves = leaf_set(a, b_files)
    b = await teammate(tmp_path, a, b_files)
    spy = seed_spy(b)
    b_ns, _ = b_namespaces(b, a.ids[0])
    probes: dict[str, list[WireHit]] = {}

    async def probe(tag: str) -> None:
        """Right now B's namespace is a copy of A's: A's rows of the changed files are still in it."""
        seen = {r.attrs["file_hash"] for r in b.store.rows(b_ns).values()}
        probes[tag] = await search(b, "def handler request value return field", 30)
        for p in changed:  # a query aimed at each file A had
            probes[tag] += await search(b, f"handler_{p.split('_')[-1].split('.')[0]}_0", 30)
        probes[f"{tag}:seen"] = [WireHit(path_enc="", loc_enc="", chunk_key="", file_hash=h) for h in seen]
        row = proofs_row(b.store, TENANT, b_ns)
        assert row is not None
        assert row.attrs["proofs_required"] is True

    spy.first_embed = lambda: probe("embed")
    spy.before_commit = lambda: probe("commit")
    status = await b.engine(spy=spy).sync()

    assert (status.state, status.coverage) == ("ready", 1.0)
    assert len(spy.seeds) == 1
    sreq, sres = spy.seeds[0]
    assert (sres.seeded, sres.complete) == (True, True)
    assert sres.jaccard is not None
    assert sres.jaccard >= 0.75  # true 37/43 = 0.86; 0.55 threshold is >7 SE away
    assert (sreq.page, sreq.pages, len(sreq.proofs)) == (0, 1, len(b_leaves))
    assert set(sreq.proofs) == b_leaves
    # A's version of every changed file was in the copied namespace (so the filter is what hid it) ...
    seen_at_embed = {h.file_hash for h in probes["embed:seen"]}
    assert set(a_old.values()) <= seen_at_embed
    for tag in ("embed", "commit"):  # ... and no search ever returned it, nor any file B cannot prove
        got = {h.file_hash for h in probes[tag]}
        assert got
        assert got <= b_leaves
        assert not got & set(a_old.values())
    # only the changed files were embedded
    sent = {r.file_id for req in spy.embeds for r in req.rows}
    assert sent == {ids.file_id(b.k, p) for p in changed}
    ref = await fresh_ref(tmp_path, b)
    bodies = doc_bodies(b)
    assert sorted(bodies) == sorted(set(doc_bodies(ref)) - set(doc_bodies(a)))
    assert 0 < len(bodies) < a_total
    print(
        f"[4.4] B embedded {len(bodies)} of {a_total} chunk bodies ({len(changed)} of {len(b_files)} files)"
    )
    # converged: equal to a from-scratch sync of B's checkout, proofs gone, sketch stored
    assert index_view(b) == index_view(ref)
    assert chunk_view(b) == chunk_view(ref)
    assert proofs_row(b.store, TENANT, b_ns) is None
    assert sketch_row_id(b_ns) in b.store.rows(meta_namespace(TENANT))
    assert disk_status(b)["state"] == "ready"
    # after the commit: no filter, every B file is found, and never A's version of a changed file
    for p in b_files:
        stem = p.rsplit("_", 1)[-1].removesuffix(".py")
        got = [hit_path(b, h) for h in await search(b, f"def handler_{stem}_0 request_{stem}", 10)]
        assert p in got
    everything = await search(b, "def handler request value return field", 30)
    assert {h.file_hash for h in everything} <= b_leaves
    assert not {h.file_hash for h in everything} & set(a_old.values())


async def test_proofs_are_paged_and_every_page_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fast stand-in for the slow 12k test: a tiny page size forces several proof pages; all must be sent."""
    import scout.core.sync.engine as engine_mod

    monkeypatch.setattr(engine_mod, "MAX_PROOFS", 7)
    files = repo()
    a = await branch_world(tmp_path, files, "a")
    b_files, _ = edit(files)
    b_leaves = leaf_set(a, b_files)
    b = await teammate(tmp_path, a, b_files)
    spy = seed_spy(b)
    status = await b.engine(spy=spy).sync()
    assert status.state == "ready"
    pages = [req for req, _ in spy.seeds]
    n = -(-len(b_leaves) // 7)
    assert n > 1
    assert [r.page for r in pages] == list(range(n))
    assert {r.pages for r in pages} == {n}
    assert set().union(*(set(r.proofs) for r in pages)) == b_leaves
    assert spy.seeds[-1][1].complete


async def test_proof_filter_is_what_hides_a_files_version_b_does_not_have(tmp_path: Path) -> None:
    """Control for AC-4.3: with B's proofs removed from the gateway's view the same query DOES return A's
    version of a changed file, so the assertions above can fail."""
    files = repo()
    a = await branch_world(tmp_path, files, "a")
    b_files, changed = edit(files, 1)
    old_hash = merkle.leaf_hash(a.k, files[changed[0]].encode())  # type: ignore[union-attr]
    b = await teammate(tmp_path, a, b_files)
    spy = seed_spy(b)
    b_ns, _ = b_namespaces(b, a.ids[0])
    seen: dict[str, set[str]] = {}

    async def probe() -> None:
        stem = changed[0].rsplit("_", 1)[-1].removesuffix(".py")
        q = f"def handler_{stem}_0 request_{stem}"
        seen["filtered"] = {h.file_hash for h in await search(b, q, 30)}
        row = proofs_row(b.store, TENANT, b_ns)
        assert row is not None
        await b.store.delete_ids(meta_namespace(TENANT), [proofs_row_id(b_ns)])  # drop the fail-closed row
        seen["open"] = {h.file_hash for h in await search(b, q, 30)}

    spy.first_embed = probe
    await b.engine(spy=spy).sync()
    assert old_hash not in seen["filtered"]
    assert old_hash in seen["open"]


# --- AC-4.2: unrelated repo does not seed ---------------------------------------------------------------
async def test_unrelated_repo_does_not_seed_and_does_a_normal_full_sync(tmp_path: Path) -> None:
    """AC-4.2: true Jaccard 0 vs 0.55: the gateway answers seeded=false, nothing is copied, the client runs a
    normal full sync (every body embedded) and no proofs row ever exists."""
    a = await branch_world(tmp_path, repo(), "a")
    other = unrelated()
    c = await teammate(tmp_path, a, other, "c")
    spy = seed_spy(c)
    a.store.calls.clear()
    status = await c.engine(spy=spy).sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert len(spy.seeds) == 1
    assert (spy.seeds[0][1].seeded, spy.seeds[0][1].jaccard is not None) == (False, True)
    assert spy.seeds[0][1].jaccard < 0.35  # type: ignore[operator]
    c_ns = c.ids[2]
    assert [call for call in c.store.ops("branch_from") if call[1] in (c_ns, ids.tree_namespace(c_ns))] == []
    assert proofs_row(c.store, TENANT, c_ns) is None
    ref = await fresh_ref(tmp_path, c)
    assert len(doc_bodies(c)) == len(doc_bodies(ref)) > 0  # a full sync: every body embedded
    assert index_view(c) == index_view(ref)


async def test_seed_failure_degrades_to_a_full_sync_with_a_warning(tmp_path: Path) -> None:
    """A seed that keeps failing upstream is skipped, not fatal: the sync converges as a normal full sync."""
    files = repo()
    a = await branch_world(tmp_path, files, "a")
    b_files, _ = edit(files)
    b = await teammate(tmp_path, a, b_files)
    spy = seed_spy(b)
    spy.fail_seed = [ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "down", retryable=True) for _ in range(9)]
    eng = b.engine(spy=spy)
    status = await eng.sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert any("teammate seed skipped" in m for m in eng.warnings)
    assert index_view(b) == index_view(await fresh_ref(tmp_path, b))


# --- seed only for a brand-new namespace ---------------------------------------------------------------
async def test_a_clone_that_already_has_an_index_never_re_seeds(tmp_path: Path) -> None:
    """Brand-new namespace only: later syncs, an edit, and a lost local tree (namespace populated) never call
    `seed`; a seeded clone whose first sync crashed resumes through the diff, not a second seed."""
    files = repo()
    a = await branch_world(tmp_path, files, "a")
    b_files, changed = edit(files)
    b = await teammate(tmp_path, a, b_files)
    first = seed_spy(b)
    await b.engine(spy=first).sync()
    assert len(first.seeds) == 1

    again = SeedSpy(b.gw)
    await b.engine(spy=again).sync()  # unchanged
    efs = bfs(b)
    efs.write(changed[0], "def after_seed():\n    return 'AFTER'\n")
    await b.engine(spy=again).sync()  # edited
    import shutil

    shutil.rmtree(b.state_dir / "trees")
    (b.state_dir / "trees").mkdir()
    await b.engine(spy=again).sync()  # local tree lost, server index there
    assert again.seeds == []
    assert index_view(b) == index_view(await fresh_ref(tmp_path, b))


async def test_crash_after_seed_resumes_through_the_diff_without_a_second_seed(tmp_path: Path) -> None:
    """B's first sync dies at its first embed (the namespace is seeded, proofs still required): search stays
    proof-filtered, the retry does not seed again, converges equal to a from-scratch sync, drops proofs."""
    files = repo()
    a = await branch_world(tmp_path, files, "a")
    b_files, changed = edit(files)
    b_leaves = leaf_set(a, b_files)
    b = await teammate(tmp_path, a, b_files)
    crash = seed_spy(b)
    crash.fail_after = 0
    with pytest.raises(ScoutError):
        await b.engine(spy=crash).sync()
    assert len(crash.seeds) == 1
    assert disk_status(b)["state"] == "absent"
    b_ns = b.ids[2]
    assert proofs_row(b.store, TENANT, b_ns) is not None
    window = await search(b, "def handler request value return field", 30)
    assert window
    assert {h.file_hash for h in window} <= b_leaves
    assert not {h.file_hash for h in window} & {merkle.leaf_hash(b.k, files[p].encode()) for p in changed}  # type: ignore[union-attr]

    retry = SeedSpy(b.gw)
    status = await b.engine(spy=retry).sync()
    assert retry.seeds == []
    assert (status.state, status.coverage) == ("ready", 1.0)
    assert proofs_row(b.store, TENANT, b_ns) is None
    ref = await fresh_ref(tmp_path, b)
    assert index_view(b) == index_view(ref)


# --- interleaved: A commits an edit between the gateway's two seed copies ---------------------------------
async def interleaved_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reverse: bool
) -> tuple[World, World]:
    """A commits an edit to F between the tree copy and the chunk copy of B's seed; B's checkout has F's new
    text (rebased on A's tip) plus two edits of its own."""
    files = repo()
    a = await branch_world(tmp_path, files, "a")
    rival = Rival(tmp_path, a)
    f = sorted(files)[5]
    new_f = "def moved_on_in_a():\n    return 'A2'\n"
    rival.fs.write(f, new_f)
    b_files, _ = edit(files, 2)
    b_files[f] = new_f
    b = await teammate(tmp_path, a, b_files)
    a_ns, a_tree = a.ids[2], ids.tree_namespace(a.ids[2])
    b_ns, b_tree = b_namespaces(b, a.ids[0])

    async def a_commits() -> None:
        await rival.sync()

    if reverse:
        monkeypatch.setattr("scout.core.service.seedsvc.branch_pair", reversed_pair)
        b.store.after[("branch_from", b_ns, a_ns)] = a_commits  # type: ignore[attr-defined]  # after the chunk copy
    else:
        b.store.after[("branch_from", b_tree, a_tree)] = a_commits  # type: ignore[attr-defined]  # after the tree copy
    status = await b.engine(spy=seed_spy(b)).sync()
    assert status.state == "ready"
    assert len(seed_spy(b).seeds) == 1
    assert seed_spy(b).seeds[0][1].seeded
    return b, await fresh_ref(tmp_path, b)


async def test_seed_converges_when_a_commits_between_the_two_seed_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task 4.4 ordering repair: the tree copy (first) is older than the chunk copy (second); B's diff against
    the copied tree re-embeds the file A moved on, and B ends `ready` with chunk keys equal to a from-scratch
    sync (no stale chunk)."""
    b, ref = await interleaved_seed(tmp_path, monkeypatch, reverse=False)
    assert chunk_view(b) == chunk_view(ref)
    assert sorted(r.attrs["chunk_key"] for r in chunk_rows(b).values()) == sorted(
        r.attrs["chunk_key"] for r in chunk_rows(ref).values()
    )
    assert index_view(b) == index_view(ref)
    assert proofs_row(b.store, TENANT, b.ids[2]) is None
    assert root_attrs(b)


async def test_reversed_seed_copy_order_leaves_a_stale_chunk_so_the_test_can_detect_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: with the chunks copied BEFORE the tree the copied tree is newer than its chunks, B's
    diff sees nothing to do for F, and A's old chunk of F survives: the assertion above fails here."""
    b, ref = await interleaved_seed(tmp_path, monkeypatch, reverse=True)
    assert chunk_view(b) != chunk_view(ref)
    stale = set(chunk_view(b)) - set(chunk_view(ref))
    assert stale
    assert len({chunk_rows(b)[s[0]].attrs["file_id"] for s in stale}) == 1


# --- time to first query (PRD 4.4: < 5 s median; the fake is a lower bound) --------------------------------
async def test_time_to_first_query_after_seed_on_the_fake_system(tmp_path: Path) -> None:
    """PRD 4.4: from `sync` start on clone B to the first search returning proof-covered hits. On the fake
    gateway this is a lower bound (no network, no model): it is the scan + sketch + seed + one search."""
    files = repo()
    a = await branch_world(tmp_path, files, "a")
    times: list[float] = []
    for i in range(5):
        b_files, _ = edit(files, 3)
        b_files[sorted(files)[10 + i]] = f"def uniq_{i}():\n    return {i}\n"
        b = await teammate(tmp_path, a, b_files, f"b{i}")
        spy = seed_spy(b)
        b_leaves = leaf_set(a, b_files)
        first: list[float] = []
        t0 = time.perf_counter()

        async def first_query(
            b: World = b, t0: float = t0, first: list[float] = first, b_leaves: set[str] = b_leaves
        ) -> None:
            hits = await search(b, "def handler request value return field", 10)
            first.append(time.perf_counter() - t0)
            assert hits
            assert {h.file_hash for h in hits} <= b_leaves

        spy.after_seed = first_query
        await b.engine(spy=spy).sync()
        assert len(first) == 1
        times.append(first[0])
    median = statistics.median(times)
    runs = [round(t * 1000) for t in times]
    print(f"[4.4] time to first query (fake, 40 files): median {median * 1000:.0f} ms, runs {runs} ms")
    assert median < 5.0


# --- paged proofs: a clone above 5k files seeds ---------------------------------------------------------
def big_repo(n: int) -> dict[str, str | bytes]:
    return {f"d{i // 100}/f{i}.py": f"def fn_{i}():\n    return {i}\n" for i in range(n)}


@pytest.mark.slow
async def test_a_12k_file_clone_seeds_through_paged_proofs(tmp_path: Path) -> None:
    """Proofs go out in pages of `MAX_PROOFS` (3 pages for ~12k files): fail-closed until the last page, then
    B embeds only its 3 changed files and equals a from-scratch sync's chunk rows."""
    n = 12_000
    files = big_repo(n)
    a = await branch_world(tmp_path, files, "a")
    a_embedded = len(doc_bodies(a))
    assert a_embedded >= n
    b_files, changed = edit(files)
    b_leaves = leaf_set(a, b_files)
    b = await teammate(tmp_path, a, b_files)
    spy = seed_spy(b)
    first: list[float] = []
    t0 = time.perf_counter()

    async def first_query() -> None:
        hits = await search(b, "def fn return", 10)
        first.append(time.perf_counter() - t0)
        assert hits
        assert {h.file_hash for h in hits} <= b_leaves

    spy.after_seed = first_query
    status = await b.engine(spy=spy).sync()
    elapsed = time.perf_counter() - t0
    assert (status.state, status.coverage) == ("ready", 1.0)
    pages = [(req.page, req.pages, len(req.proofs), res.complete) for req, res in spy.seeds]
    assert [p[:2] for p in pages] == [(0, 3), (1, 3), (2, 3)]
    assert all(p[2] <= MAX_PROOFS for p in pages)
    assert sum(p[2] for p in pages) == len(b_leaves)
    assert [p[3] for p in pages] == [False, False, True]  # search stays closed until the last page
    assert len(first) == 1
    sent = {r.file_id for req in spy.embeds for r in req.rows}
    assert sent == {ids.file_id(b.k, p) for p in changed}
    assert len(doc_bodies(b)) == len(changed)
    assert proofs_row(b.store, TENANT, b.ids[2]) is None
    assert len(chunk_rows(b)) == len(chunk_rows(a))
    sizes = [p[2] for p in pages]
    print(
        f"[4.4] 12k-file seed: pages {sizes}, {len(doc_bodies(b))} of {a_embedded} bodies embedded, "
        f"first query {first[0]:.2f} s, whole sync {elapsed:.2f} s"
    )
