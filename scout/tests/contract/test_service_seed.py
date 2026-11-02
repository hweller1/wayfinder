"""Task 3.6: `seed` (MinHash scan, tree-then-chunks branch, paged proofs, fail-closed search filter).

Covers FR-C16, FR-C17, AC-4.2 (above/below threshold), AC-4.3 (proof filtering until commit, then dropped),
tenant isolation of sketches, crash/retry idempotency and the 20k-file paged upload (eval repos exceed the
5k `wire.MAX_PROOFS` per request). The client side is the real keyed Merkle tree and MinHash.
"""

import base64
import hashlib
import time
from typing import Any

import pytest
from svc_helpers import (
    MASTER,
    NEW_BR,
    REPO,
    SRC_BR,
    Clone,
    RecordingStore,
    edited,
    make,
    ns_pair,
    proofs_row,
    server_root,
    source_clone,
    src,
)

from scout.adapters.fake_clock import FakeClock
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import (
    MAX_PROOFS,
    SearchReq,
    SeedReq,
    StatusReq,
    SyncBeginReq,
)
from scout.core.crypto import keys, minhash
from scout.core.ports.store import Row
from scout.core.service.gateway import GatewayService
from scout.core.service.schema import CHUNK_SCHEMA, META_SCHEMA, ROOT_ID, TREE_SCHEMA
from scout.core.service.seedsvc import DEFAULT_THRESHOLD
from scout.core.service.syncsvc import meta_namespace, sketch_row_id

# --- match decision (AC-4.2) ----------------------------------------------------------------------------


async def test_above_threshold_seeds_and_below_threshold_does_not():
    """AC-4.2: a checkout sharing most files seeds; an unrelated one gets seeded=false, nothing written."""
    store, svc, _ = make()
    files = src(40)
    await source_clone(svc, store, files)
    mate = await Clone.create(svc, store)
    res = await mate.gw.seed(mate.seed_req(edited(files, 3)))  # 37 of 43 files shared: J ~ 0.86
    assert res.seeded
    assert res.complete
    assert res.jaccard is not None
    assert abs(res.jaccard - 37 / 43) < 0.12
    assert res.jaccard >= DEFAULT_THRESHOLD
    assert await store.namespace_exists(mate.ns)
    assert await store.namespace_exists(mate.tree_ns)

    other = await Clone.create(svc, store, "acme", "d" * 16)
    unrelated = src(40, "other")
    before = list(store.calls)
    res = await other.gw.seed(other.seed_req(unrelated))
    assert (res.seeded, res.complete) == (False, True)
    assert res.jaccard is not None
    assert res.jaccard < DEFAULT_THRESHOLD
    assert not await store.namespace_exists(other.ns)
    assert not await store.namespace_exists(other.tree_ns)
    assert proofs_row(store, "acme", other.ns) is None
    assert [c for c in store.calls[len(before) :] if c[0] in ("branch_from", "upsert")] == []


async def test_threshold_is_configurable_and_a_missing_meta_namespace_means_no_seed():
    store, svc, _ = make()
    mate = await Clone.create(svc, store)
    res = await mate.gw.seed(mate.seed_req(src(5)))  # nobody has synced: no sketches at all
    assert (res.seeded, res.jaccard) == (False, None)
    assert not await store.namespace_exists(meta_namespace("acme"))
    assert store.ops("branch_from") == []

    store2, strict, _ = make(seed_threshold=0.99)
    files = src(40)
    await source_clone(strict, store2, files)
    mate2 = await Clone.create(strict, store2)
    res = await mate2.gw.seed(mate2.seed_req(edited(files, 3)))
    assert res.seeded is False
    assert res.jaccard is not None
    assert 0.5 < res.jaccard < 0.99
    with pytest.raises(ValueError, match="threshold"):
        GatewayService(RecordingStore(), FakeClock(), MASTER, seed_threshold=0.0)


async def test_below_threshold_leaves_a_normal_empty_namespace_and_a_full_first_sync_works():
    store, svc, _ = make()
    await source_clone(svc, store, src(20))
    mate = await Clone.create(svc, store)
    files = src(20, "other")
    assert (await mate.gw.seed(mate.seed_req(files))).seeded is False
    res = await mate.gw.sync_begin(
        SyncBeginReq(repo_id=REPO, branch_id=NEW_BR, root="0" * 64, expected_chunks=0)
    )
    assert (res.ns_state, res.server_root) == ("empty", None)
    delta = await mate.sync(files)
    assert len(delta.added) == 20
    assert (await mate.gw.index_status(StatusReq(repo_id=REPO, branch_id=NEW_BR))).state == "ready"
    assert await mate.hits() == mate.file_hashes(files)  # no proof filter ever applied


async def test_best_match_wins_and_a_candidate_without_a_root_row_is_skipped():
    store, svc, _ = make()
    files = src(40)
    close = await Clone.create(svc, store, "acme", "1" * 16)
    await close.sync(edited(files, 2))
    far = await Clone.create(svc, store, "acme", "2" * 16)
    await far.sync(edited(files, 8))
    mate = await Clone.create(svc, store)
    await mate.gw.seed(mate.seed_req(files))
    assert ("branch_from", mate.tree_ns, close.tree_ns) in store.calls
    assert not any(c[0] == "branch_from" and c[2] == far.tree_ns for c in store.calls)

    # the best candidate's sketch survives a branch delete but its tree has no root: next best is used
    store.rows(close.tree_ns).pop(ROOT_ID)
    again = await Clone.create(svc, store, "acme", "3" * 16)
    await again.gw.seed(again.seed_req(files))
    assert ("branch_from", again.tree_ns, far.tree_ns) in store.calls


# --- branch copy and proofs ------------------------------------------------------------------------------


async def test_seed_writes_proofs_first_then_copies_tree_before_chunks_and_never_a_vector():
    store, svc, _ = make()
    files = src(12)
    source = await source_clone(svc, store, files)
    mate = await Clone.create(svc, store)
    req = mate.seed_req(edited(files, 2))
    await mate.begin(mate.tree(edited(files, 2)))  # design 5.4: sync_begin first leaves an empty pair
    store.calls.clear()
    await mate.gw.seed(req)
    meta = meta_namespace("acme")
    order = [c for c in store.calls if c[0] in ("branch_from", "upsert") and c[1] != meta or c[1] == meta]
    labels = [(c[0], c[1]) for c in order]
    first_copy = labels.index(("branch_from", mate.tree_ns))
    assert labels.index(("upsert", meta)) < first_copy  # fail-closed proofs row BEFORE any copy
    assert first_copy + 1 == labels.index(("branch_from", mate.ns))  # tree, THEN chunks
    assert store.ops("branch_from") == [
        ("branch_from", mate.tree_ns, source.tree_ns),
        ("branch_from", mate.ns, source.ns),
    ]
    row = proofs_row(store, "acme", mate.ns)
    assert row is not None
    assert row.vector is None
    assert row.attrs["proofs_required"] is True
    assert row.attrs["complete"] is True
    assert row.attrs["leaves"] == mate.leaves(edited(files, 2))  # sorted leaf hashes, nothing else
    assert META_SCHEMA.dims is None
    assert CHUNK_SCHEMA.dims == 1024
    assert TREE_SCHEMA.dims is None
    res = await mate.gw.sync_begin(
        SyncBeginReq(repo_id=REPO, branch_id=NEW_BR, root="0" * 64, expected_chunks=0)
    )
    assert (res.ns_state, res.server_root) == ("populated", server_root(store, source.tree_ns))


async def test_seeded_namespace_is_searchable_only_through_proofs_until_commit():
    """AC-4.3: hits outside the client's proof set are dropped; after commit the proofs are gone."""
    store, svc, _ = make()
    files = src(24)  # 22 of 26 shared: J ~ 0.85, far above 0.55 (12/3 sat at 0.6 and flaked)
    source = await source_clone(svc, store, files)
    mate = await Clone.create(svc, store)
    mine = edited(files, 2)
    tree = mate.tree(mine)
    await mate.begin(tree)  # absent, no base: an empty pair
    assert (await mate.gw.seed(mate.seed_req(mine))).seeded
    stale = source.file_hashes(files) - mate.file_hashes(mine)  # the 2 files the teammate has edited
    assert len(stale) == 2
    seen = await mate.hits()
    assert seen == source.file_hashes(files) & mate.file_hashes(mine)  # 22 shared files, no stale version
    assert not stale & seen
    # the client diff re-embeds only the 2 edited files and ends ready with no stale chunks
    emb_before = len(store.rows(mate.ns))
    delta = await mate.sync(mine)
    assert len(delta.added) + len(delta.changed) == 2
    assert not delta.removed
    assert proofs_row(store, "acme", mate.ns) is None  # commit spent the proofs
    assert (await mate.gw.index_status(StatusReq(repo_id=REPO, branch_id=NEW_BR))).state == "ready"
    assert await mate.hits() == mate.file_hashes(mine)
    assert len(store.rows(mate.ns)) == emb_before  # 24 chunks: stale rows replaced, none duplicated
    assert {r.attrs["file_hash"] for r in store.rows(mate.ns).values()} == mate.file_hashes(mine)
    assert sketch_row_id(mate.ns) in store.rows(meta_namespace("acme"))
    assert sketch_row_id(source.ns) in store.rows(meta_namespace("acme"))


async def test_commit_landing_between_the_two_seed_copies_leaves_older_tree_newer_chunks_and_converges():
    store, svc, _ = make()
    files = src(12)
    source = await source_clone(svc, store, files)
    old_root = server_root(store, source.tree_ns)
    newer = files | {
        "src/m0.py": "def fn_s_0():\n    return 'newer'\n",
        "src/extra.py": "def extra():\n    pass\n",
    }

    async def source_commits_meanwhile() -> None:
        await source.sync(newer)  # a full sync lands on the best namespace between the copies

    mate = await Clone.create(svc, store)
    store.after[("branch_from", mate.tree_ns, source.tree_ns)] = source_commits_meanwhile
    mine = newer  # the teammate already has the newer files
    await mate.gw.seed(mate.seed_req(mine))
    assert server_root(store, mate.tree_ns) == old_root  # tree older ...
    assert server_root(store, source.tree_ns) != old_root
    assert len(store.rows(mate.ns)) == 13  # ... chunks newer (extra.py and the new m0 are already there)
    delta = await mate.sync(mine)  # next diff re-embeds only the affected files and ends ready
    assert set(delta.added) == {"src/extra.py"}
    assert set(delta.changed) == {"src/m0.py"}
    assert (await mate.gw.index_status(StatusReq(repo_id=REPO, branch_id=NEW_BR))).state == "ready"
    assert {r.attrs["file_hash"] for r in store.rows(mate.ns).values()} == mate.file_hashes(mine)
    assert len(store.rows(mate.ns)) == 13


# --- isolation -------------------------------------------------------------------------------------------


async def test_seed_never_crosses_tenants_even_for_identical_repos_and_identical_sketches():
    store, svc, _ = make()
    files = src(30)
    a = await source_clone(svc, store, files, "acme")
    b_src = await Clone.create(svc, store, "beta", SRC_BR)
    await b_src.sync(src(30, "beta-only"))  # beta has its own, unrelated, indexed repo
    b = await Clone.create(svc, store, "beta", NEW_BR)
    a_meta_before = {i: (r.attrs, r.vector) for i, r in store.rows(meta_namespace("acme")).items()}
    a_sketch = base64.b64encode(minhash.sketch(a.k, [bytes.fromhex(x) for x in a.leaves(files)])).decode()
    # beta presents tenant acme's EXACT sketch (and acme's namespace ids are the same hex ids)
    res = await b.gw.seed(b.seed_req(files, minhash_b64=a_sketch))
    assert res.seeded is False  # beta's meta namespace holds only beta's sketch; nothing near it
    assert not await store.namespace_exists(b.ns)
    assert not any(c[0] == "branch_from" for c in store.calls)
    assert {i: (r.attrs, r.vector) for i, r in store.rows(meta_namespace("acme")).items()} == a_meta_before
    assert a.ns != b.ns
    assert ns_pair("acme", SRC_BR)[0] in await store.list_namespaces("scout_acme_")
    # and symmetrically a tenant with an identical repo as its own source seeds only from itself
    a2 = await Clone.create(svc, store, "acme", NEW_BR)
    assert (await a2.gw.seed(a2.seed_req(files))).seeded is True
    assert ("branch_from", a2.ns, a.ns) in store.calls
    assert not any(c[0] == "branch_from" and "scout_beta_" in c[2] for c in store.calls)


# --- sketch hygiene --------------------------------------------------------------------------------------


async def test_sketch_row_holds_only_the_keyed_sketch_never_leaves_or_paths():
    store, svc, _ = make()
    files = src(10)
    source = await source_clone(svc, store, files)
    rows = store.rows(meta_namespace("acme"))
    sketch = rows[sketch_row_id(source.ns)]
    assert sketch.vector is None
    assert set(sketch.attrs) == {"kind", "sketch"}
    assert len(base64.b64decode(sketch.attrs["sketch"])) == minhash.SKETCH_BYTES
    blob = repr(sketch.attrs)
    for leaf in source.leaves(files):
        assert leaf not in blob
        assert leaf[:16] not in blob
    for path in files:
        assert path not in blob
    # a seed only reads sketches; it never writes one (commit does)
    mate = await Clone.create(svc, store)
    await mate.gw.seed(mate.seed_req(files))
    assert sketch_row_id(mate.ns) not in store.rows(meta_namespace("acme"))


async def test_meta_namespace_rejects_a_row_with_a_vector_and_a_bad_sketch_fails_before_any_write():
    store, svc, _ = make()
    files = src(10)
    await source_clone(svc, store, files)
    with pytest.raises(ScoutError) as exc:
        await store.upsert(meta_namespace("acme"), [Row(id="x", attrs={"kind": "sketch"}, vector=[0.1] * 4)])
    assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    mate = await Clone.create(svc, store)
    store.calls.clear()
    for bad in ("not base64!", base64.b64encode(b"short").decode()):
        with pytest.raises(ScoutError) as exc:
            await mate.gw.seed(mate.seed_req(files, minhash_b64=bad))
        assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError) as exc:
        await mate.gw.seed(mate.seed_req(files, proofs=["x" * 200]))
    assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(ScoutError) as exc:
        await mate.gw.seed(mate.seed_req(files, page=1, pages=1))
    assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    assert [c for c in store.calls if c[0] in ("branch_from", "upsert", "delete_namespace")] == []


# --- idempotency and crash recovery ----------------------------------------------------------------------


async def test_seed_retry_is_idempotent_and_does_not_copy_twice():
    store, svc, _ = make()
    files = src(12)
    source = await source_clone(svc, store, files)
    mate = await Clone.create(svc, store)
    req = mate.seed_req(files)
    first = await mate.gw.seed(req)
    snapshot = {i: r.attrs for i, r in store.rows(meta_namespace("acme")).items()}
    chunks = {i: r.attrs for i, r in store.rows(mate.ns).items()}
    second = await mate.gw.seed(req)
    assert (first.seeded, second.seeded, first.complete, second.complete) == (True, True, True, True)
    assert first.jaccard == second.jaccard
    assert len(store.ops("branch_from")) == 2  # one pair, ever
    assert {i: r.attrs for i, r in store.rows(meta_namespace("acme")).items()} == snapshot
    assert {i: r.attrs for i, r in store.rows(mate.ns).items()} == chunks
    assert server_root(store, mate.tree_ns) == server_root(store, source.tree_ns)
    await mate.sync(files)  # commit; a late retry of the same seed is now a harmless no-seed answer
    assert proofs_row(store, "acme", mate.ns) is None
    late = await mate.gw.seed(req)
    assert late.seeded is False
    assert proofs_row(store, "acme", mate.ns) is None  # no proofs resurrected: search stays unfiltered
    assert await mate.hits() == mate.file_hashes(files)


@pytest.mark.parametrize("crash", ["after_tree_copy", "chunk_copy_fails", "proofs_write_fails"])
async def test_crash_mid_seed_then_retry_converges(crash: str):
    store, svc, _ = make()
    files = src(12)
    source = await source_clone(svc, store, files)
    mate = await Clone.create(svc, store)
    boom = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "crash")
    if crash == "after_tree_copy":
        store.fail_once[("branch_from", mate.ns, source.ns)] = boom
    elif crash == "chunk_copy_fails":
        store.fail_once[("delete_namespace", mate.tree_ns)] = boom  # fails while clearing a stale pair
        await mate.begin(mate.tree(files))  # sync_begin first: empty pair exists
    else:
        store.fail_once[("upsert", meta_namespace("acme"), "proofpage")] = boom
    with pytest.raises(ScoutError):
        await mate.gw.seed(mate.seed_req(files))
    if crash == "after_tree_copy":
        assert await store.namespace_exists(mate.tree_ns)
        assert not await store.namespace_exists(mate.ns)
        # the proofs row was written first, fail-closed and empty
        row = proofs_row(store, "acme", mate.ns)
        assert row is not None
        assert row.attrs["leaves"] == []
        assert row.attrs["complete"] is False
    res = await mate.gw.seed(mate.seed_req(files))
    assert (res.seeded, res.complete) == (True, True)
    assert server_root(store, mate.tree_ns) == server_root(store, source.tree_ns)
    assert len(store.rows(mate.ns)) == 12
    row = proofs_row(store, "acme", mate.ns)
    assert row is not None
    assert row.attrs["complete"] is True
    assert row.attrs["leaves"] == mate.leaves(files)
    assert await mate.hits() == mate.file_hashes(files)


async def test_orphan_tree_from_a_crashed_seed_is_also_repaired_by_sync_begin():
    store, svc, _ = make()
    files = src(8)
    source = await source_clone(svc, store, files)
    mate = await Clone.create(svc, store)
    store.fail_once[("branch_from", mate.ns, source.ns)] = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "x")
    with pytest.raises(ScoutError):
        await mate.gw.seed(mate.seed_req(files))
    res = await mate.gw.sync_begin(
        SyncBeginReq(repo_id=REPO, branch_id=NEW_BR, root="0" * 64, expected_chunks=0)
    )
    assert (res.ns_state, res.server_root) == ("empty", None)  # never "populated" from an orphan tree
    assert await store.namespace_exists(mate.ns)


# --- paged proofs ----------------------------------------------------------------------------------------


def pages_of(leaves: list[str], size: int) -> list[list[str]]:
    return [leaves[i : i + size] for i in range(0, len(leaves), size)]


async def test_partial_proof_upload_never_widens_what_search_can_see():
    """All-or-nothing: until every page has arrived the seeded namespace returns no hits at all."""
    store, svc, _ = make()
    files = src(12)
    await source_clone(svc, store, files)
    mate = await Clone.create(svc, store)
    leaves = mate.leaves(files)
    parts = pages_of(leaves, 4)  # 3 pages
    assert len(parts) == 3

    async def send(i: int) -> Any:
        return await mate.gw.seed(mate.seed_req(files, proofs=parts[i], page=i, pages=3))

    res = await send(0)
    assert (res.seeded, res.complete) == (True, False)
    assert await mate.hits() == set()  # page 0 alone: closed
    res = await send(2)  # the LAST page arrives before page 1: still closed
    assert res.complete is False
    assert await mate.hits() == set()
    row = proofs_row(store, "acme", mate.ns)
    assert row is not None
    assert row.attrs["leaves"] == []
    assert row.attrs["proofs_required"] is True
    res = await send(2)  # a repeated page does not complete the set either
    assert res.complete is False
    assert await mate.hits() == set()
    res = await send(1)
    assert (res.seeded, res.complete) == (True, True)
    assert await mate.hits() == mate.file_hashes(files)  # the complete set, exactly
    row = proofs_row(store, "acme", mate.ns)
    assert row is not None
    assert row.attrs["leaves"] == leaves

    # a restart (page 0 again) closes the namespace again until the set is complete once more
    res = await send(0)
    assert res.complete is False
    assert await mate.hits() == set()
    await send(1)
    assert (await send(2)).complete is True
    assert await mate.hits() == mate.file_hashes(files)
    assert len(store.ops("branch_from")) == 2  # restarts never re-copy


async def test_proof_pages_need_page_zero_first_and_a_consistent_page_count():
    store, svc, _ = make()
    files = src(12)
    await source_clone(svc, store, files)
    mate = await Clone.create(svc, store)
    with pytest.raises(ScoutError) as exc:  # no seed session yet: pages without page 0 are refused
        await mate.gw.seed(mate.seed_req(files, proofs=[], page=1, pages=2))
    assert exc.value.code is ErrorCode.CONFLICT
    assert not await store.namespace_exists(mate.ns)
    leaves = mate.leaves(files)
    await mate.gw.seed(mate.seed_req(files, proofs=leaves[:6], page=0, pages=2))
    with pytest.raises(ScoutError) as exc:
        await mate.gw.seed(mate.seed_req(files, proofs=leaves[6:], page=1, pages=3))
    assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    assert await mate.hits() == set()
    with pytest.raises(ValueError, match="pages"):
        SeedReq(repo_id=REPO, branch_id=NEW_BR, minhash_b64="AAAA", page=0, pages=21)
    assert (await mate.gw.seed(mate.seed_req(files, proofs=leaves[6:], page=1, pages=2))).complete is True


async def test_seed_of_a_20k_file_tree_in_four_pages():
    """FR-C17 (parent note, review of 1.4): eval repos exceed MAX_PROOFS=5000 files, so proofs are paged."""
    store, svc, _ = make()
    k = keys.derive(await svc.for_tenant("acme").team_key())
    leaves = sorted(hashlib.sha256(f"leaf-{i}".encode()).hexdigest() for i in range(20_000))
    sketch = base64.b64encode(minhash.sketch(k, [bytes.fromhex(x) for x in leaves])).decode()
    # the source: a committed namespace with a sketch row and three chunks (two proven, one not)
    src_ns, src_tree = ns_pair("acme", SRC_BR)
    await store.ensure_namespace(src_ns, CHUNK_SCHEMA)
    await store.ensure_namespace(src_tree, TREE_SCHEMA)
    await store.ensure_namespace(meta_namespace("acme"), META_SCHEMA)
    in_set = [leaves[0], leaves[19_999]]
    outside = hashlib.sha256(b"not-in-the-client-set").hexdigest()
    assert outside not in set(leaves)
    for n, fh in enumerate([*in_set, outside]):
        attrs = {
            "kind": "chunk",
            "file_id": f"f{n}",
            "path_enc": f"enc/{n}",
            "ext_tag": "e",
            "dir_tags": ["d"],
            "file_hash": fh,
            "chunk_key": f"ck{n}",
            "loc_enc": "bG9j",
            "tokens": ["t" * 32],
        }
        await store.upsert(src_ns, [Row(id=f"c{n}", attrs=attrs, vector=[1.0] + [0.0] * 1023)])
    root = {"kind": "root", "root_hash": "r" * 64, "expected_chunks": 3, "written_chunks": 3, "gen": 1}
    await store.upsert(src_tree, [Row(id=ROOT_ID, attrs=root)])
    await store.upsert(
        meta_namespace("acme"), [Row(id=sketch_row_id(src_ns), attrs={"kind": "sketch", "sketch": sketch})]
    )

    mate_ns, mate_tree = ns_pair("acme", NEW_BR)
    gw = svc.for_tenant("acme")
    parts = pages_of(leaves, MAX_PROOFS)
    assert len(parts) == 4
    assert all(len(p) == MAX_PROOFS for p in parts)
    with pytest.raises(ValueError, match="proofs"):  # one request still cannot carry more than 5k proofs
        SeedReq(repo_id=REPO, branch_id=NEW_BR, minhash_b64=sketch, proofs=leaves[: MAX_PROOFS + 1])

    async def search() -> set[str]:
        res = await gw.search(SearchReq(repo_id=REPO, branch_id=NEW_BR, query="fn", k=10))
        return {h.file_hash for h in res.hits}

    started = time.perf_counter()
    for i, part in enumerate(parts):
        res = await gw.seed(
            SeedReq(repo_id=REPO, branch_id=NEW_BR, minhash_b64=sketch, proofs=part, page=i, pages=4)
        )
        assert res.seeded
        assert res.jaccard == 1.0
        assert res.complete is (i == 3)
        if i < 3:
            assert await search() == set()  # 5k, 10k, 15k proofs uploaded: still closed
    elapsed = time.perf_counter() - started
    assert await search() == set(in_set)  # the full 20k set: proven hits only, the outside file is dropped
    row = proofs_row(store, "acme", mate_ns)
    assert row is not None
    assert len(row.attrs["leaves"]) == 20_000
    assert row.attrs["leaves"] == leaves
    assert server_root(store, mate_tree) == "r" * 64
    assert elapsed < 30.0, f"20k-file paged seed took {elapsed:.1f}s"
    print(f"seed of 20k files in 4 pages: {elapsed:.2f}s")


async def test_sketch_of_a_deleted_branch_is_never_a_candidate():
    """`delete_branch` leaves the sketch row behind; the seed must not branch from (or crash on) it."""
    store, svc, _ = make()
    files = src(30)
    gone = await source_clone(svc, store, files)
    await gone.gw.delete_branch(REPO, SRC_BR)
    assert sketch_row_id(gone.ns) in store.rows(meta_namespace("acme"))  # the stale row is still there
    mate = await Clone.create(svc, store)
    res = await mate.gw.seed(mate.seed_req(files))
    assert (res.seeded, res.jaccard) == (False, None)
    assert store.ops("branch_from") == []
