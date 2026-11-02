"""Task 8.3: the whole client stack over HTTP. `HttpGateway` over `httpx.ASGITransport(app)` against the real
FastAPI app over the real `GatewayService` over fakes.

FR-G1, AC-3.1, AC-13.1, AC-6.1. First sync, one-file incremental, branch switch, seed, semantic explore
and the tenant isolation matrix run through HTTP; each result is judged against the in-process fake gateway (a
from-scratch sync of the same checkout with the same keys and clone id) and must be equal. Wire bytes are the
real HTTP body sizes, and still scale with changed files, not repo size.
"""

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from gateway_app_helpers import Rig, make_rig
from svc_helpers import NEW_BR, REPO, Clone, src
from test_sync_branch import FEAT, BranchFS, bfs, branch_ns, edit_branch, fresh_ref, tree_repo
from test_sync_first import Spy, World, chunk_rows
from test_sync_incremental import (
    CheapEmbedder,
    deep_repo,
    index_view,
)
from test_sync_seed import SeedSpy, doc_bodies, edit, leaf_set, repo, teammate
from test_sync_seed import search as world_search

from scout import Scout
from scout.adapters.fake_clock import FakeClock
from scout.adapters.gateway_http import HttpGateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import ExploreRequest, IndexStatusRequest
from scout.core.contracts.wire import (
    CommitReq,
    EmbedReq,
    NodesReq,
    PlanReq,
    SearchReq,
    SeedReq,
    StageNodesReq,
    StatusReq,
)
from scout.core.crypto import ids, keys
from scout.core.ports.gateway import Gateway
from scout.core.ports.store import NamespaceSchema
from scout.core.service.schema import CHUNK_SCHEMA
from scout.core.sync.engine import SyncEngine
from scout.wiring import Config, build_app

TENANT = "dev"
URL = "http://gw.test"


async def http_world(
    tmp_path: Path, files: dict[str, str | bytes], *, name: str = "w", rig: Rig | None = None
) -> World:
    """A `World` whose gateway is the HTTP client (state dir, fs and keys as in the in-process worlds)."""
    rig = rig or make_rig()
    gw = HttpGateway(URL, await rig.key(TENANT), transport=rig.transport())
    state_dir = tmp_path / name / "git" / "scout"
    fs = BranchFS(files, state_dir=str(state_dir))
    k = keys.derive(await gw.team_key())
    return World(
        rig.store, rig.embedder, rig.clock, cast(Gateway, gw), Spy(cast(Gateway, gw)), fs, state_dir, k
    )


async def synced(tmp_path: Path, files: dict[str, str | bytes], name: str = "w") -> World:
    w = await http_world(tmp_path, files, name=name)
    status = await w.engine().sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    return w


# --- the sync flows over HTTP equal the in-process gateway --------------------------------------------
async def test_first_sync_over_http_equals_the_in_process_index(tmp_path: Path) -> None:
    w = await synced(tmp_path, tree_repo())
    ref = await fresh_ref(tmp_path, w)
    assert index_view(w) == index_view(ref)
    assert sorted(w.emb.embedded) == sorted(ref.emb.embedded)
    assert len(chunk_rows(w)) == len(chunk_rows(ref)) > 20
    st = await w.gw.index_status(StatusReq(repo_id=w.ids[0], branch_id=w.ids[1]))
    assert (st.state, st.coverage, st.semantic_enabled) == ("ready", 1.0, True)


async def test_one_file_incremental_over_http(tmp_path: Path) -> None:
    w = await synced(tmp_path, tree_repo())
    first_total = len(w.emb.embedded)
    bfs(w).write("svc1/mod1/f2.py", "def renamed_handler(request):\n    return request * 3\n")
    assert (await w.engine().sync()).state == "ready"
    assert len(w.emb.embedded) == first_total + 1  # one new chunk body, nothing else
    ref = await fresh_ref(tmp_path, w)
    assert index_view(w) == index_view(ref)


async def test_branch_switch_over_http_copies_tree_then_chunks(tmp_path: Path) -> None:
    w = await synced(tmp_path, tree_repo())
    main_ns = w.ids[2]
    main_view = index_view(w)
    first_total = len(w.emb.embedded)
    w.store.calls.clear()
    bfs(w).checkout(FEAT)
    edit_branch(bfs(w))
    assert (await w.engine().sync()).state == "ready"
    feat_ns, feat_tree = branch_ns(w, FEAT)
    assert w.store.ops("branch_from") == [
        ("branch_from", feat_tree, ids.tree_namespace(main_ns)),
        ("branch_from", feat_ns, main_ns),
    ]
    assert len(w.emb.embedded) - first_total == 2  # the edited file and the added one
    ref = await fresh_ref(tmp_path, w)
    assert index_view(w) == index_view(ref)
    w.state.update_state(last_branch=ids.branch_id(w.k, w.ids[0], w.state.read_state().clone_id, "main"))
    assert index_view(w) == main_view  # main untouched


async def test_teammate_seed_over_http(tmp_path: Path) -> None:
    """AC-4.3 over HTTP: B seeds from A's index, embeds only its 3 changed files, converges."""
    files = repo()
    rig = make_rig()
    a = await http_world(tmp_path, files, name="a", rig=rig)
    assert (await a.engine().sync()).state == "ready"
    a_total = len(doc_bodies(a))
    b_files, changed = edit(files)
    b = await teammate(tmp_path, a, b_files)
    spy = b.spy
    assert isinstance(spy, SeedSpy)
    status = await b.engine(spy=spy).sync()
    assert (status.state, status.coverage) == ("ready", 1.0)
    ((sreq, sres),) = spy.seeds
    assert (sres.seeded, sres.complete) == (True, True)
    assert set(sreq.proofs) == leaf_set(a, b_files)
    sent = {r.file_id for req in spy.embeds for r in req.rows}
    assert sent == {ids.file_id(b.k, p) for p in changed}
    assert 0 < len(doc_bodies(b)) < a_total
    ref = await fresh_ref(tmp_path, b)
    assert index_view(b) == index_view(ref)
    hits = await world_search(b, "def handler request value return field", 30)
    assert {h.file_hash for h in hits} <= leaf_set(a, b_files)


# --- semantic explore over HTTP equals the in-process fake ---------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
REPO_SMALL = ROOT / "tests" / "fixtures" / "repo_small"
IDENT = "getUserSession"
SESSION = "src/auth/session.py"


def _call(id_: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_calls": [{"id": id_, "name": name, "arguments": arguments}],
        "tokens_in": 100,
        "tokens_out": 10,
    }


def _explore_case(tmp_path: Path, tag: str) -> dict[str, Any]:
    report = {
        "results": [{"path": SESSION, "start_line": 6, "end_line": 12, "why": "session lookup"}],
        "summary": f"{IDENT} lives in {SESSION}.",
        "gaps": [],
    }
    script = [
        _call("c1", "semantic_search", {"query": f"{IDENT} session lookup by token", "k": 5}),
        _call("c2", "read_file", {"path": SESSION, "start_line": 6, "end_line": 12}),
        _call("r", "report", report),
    ]
    return {
        "repo": str(REPO_SMALL),
        "chat_script": script,
        "state_dir": str(tmp_path / tag / "state"),
        "backfill_to": 0,  # fake-store top-k ties differ per process
    }


async def test_semantic_explore_over_http_equals_in_process(tmp_path: Path) -> None:
    """Sync, semantic_search and a scripted explore (chat over SSE) give the same ExploreResult as the
    in-process fake profile."""
    from scout.adapters.fake_chat import ScriptedChat
    from scout.wiring import _turn  # pyright: ignore[reportPrivateUsage]

    case = _explore_case(tmp_path, "http")
    rig = make_rig(chat=ScriptedChat([_turn(t) for t in case["chat_script"]]), keepalive_s=0.05)
    app = build_app(
        Config(
            profile="fake-http",
            case=case,
            transport=rig.transport(),
            key=await rig.key(TENANT),
            gateway_url=URL,
        )
    )
    assert (await app.sync()).state == "ready"
    via_http = await app.explore(ExploreRequest(brief="where is the session lookup implemented?"))

    local = Scout(profile="fake", case=_explore_case(tmp_path, "local"))
    assert (await local.async_sync()).state == "ready"
    in_process = await local.aexplore("where is the session lookup implemented?")

    assert json.loads(via_http.model_dump_json(by_alias=True)) == json.loads(
        in_process.model_dump_json(by_alias=True)
    )
    assert via_http.index.semantic_enabled
    assert [(h.path, h.start_line, h.end_line) for h in via_http.results if h.verified] == [(SESSION, 6, 12)]
    assert via_http.usage.tool_calls == {"semantic_search": 1, "read_file": 1, "report": 1}
    status = await app.index_status(IndexStatusRequest())
    assert status.state == "ready"


# --- AC-3.1: the real HTTP body sizes ------------------------------------------------------------------
class Counting(httpx.AsyncBaseTransport):
    """Counts request and response body bytes of every HTTP call, by path."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self.inner = inner
        self.calls: list[tuple[str, int, int]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        up = len(request.content)
        res = await self.inner.handle_async_request(request)
        body = b"".join([c async for c in cast(Any, res.stream)])
        self.calls.append((request.url.path, up, len(body)))
        return httpx.Response(res.status_code, headers=res.headers, content=body)

    @property
    def total(self) -> int:
        return sum(u + d for _, u, d in self.calls)

    def by_path(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for path, u, d in self.calls:
            out[path] = out.get(path, 0) + u + d
        return out


async def _http_edit_bytes(
    tmp_path: Path, files: int, mp: pytest.MonkeyPatch, edits: int = 1
) -> tuple[int, dict[str, int]]:
    dims = 16
    mp.setattr(
        "scout.core.service.gateway.CHUNK_SCHEMA", NamespaceSchema(dims=dims, attrs=CHUNK_SCHEMA.attrs)
    )
    mp.setattr("scout.core.service.syncsvc.CACHE_SCHEMA", NamespaceSchema(dims=dims, attrs={}))
    rig = make_rig(embedder=CheapEmbedder(dims=dims), store=None, clock=FakeClock(), keepalive_s=0.05)
    counting = Counting(rig.transport())
    gw = HttpGateway(URL, await rig.key(TENANT), transport=counting)
    fs = BranchFS(deep_repo(files), state_dir=str(tmp_path / f"s{files}x{edits}" / "git" / "scout"))
    engine = SyncEngine(fs, cast(Gateway, gw), keys.derive(await gw.team_key()), rig.clock)
    assert (await engine.sync()).state == "ready"
    counting.calls.clear()
    stride = len(fs.files) // (edits + 1)
    for n, target in enumerate(sorted(fs.files)[stride::stride][:edits]):
        fs.write(target, f"def edited_{n}(request):\n    return {7 + n} * request\n")
    status = await engine.sync()
    assert (status.state, status.coverage, status.files) == ("ready", 1.0, files)
    return counting.total, counting.by_path()


async def test_http_wire_bytes_follow_changed_files_not_repo_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-3.1, NFR-5: the same one-file edit in repos of 300 and 1,000 files costs HTTP bytes within 1.5x;
    8 edits in one repo cost more than 1 but at most 8x."""
    small, small_paths = await _http_edit_bytes(tmp_path, 300, monkeypatch)
    big, big_paths = await _http_edit_bytes(tmp_path, 1_000, monkeypatch)
    assert max(small, big) <= 1.5 * min(small, big), (small, big)
    assert "/v1/sync/nodes" not in big_paths  # fast path: no tree walk
    assert big_paths["/v1/embed"] < 3_000
    eight, _ = await _http_edit_bytes(tmp_path, 1_000, monkeypatch, edits=8)
    assert big < eight <= 8 * big


# --- tenant isolation over HTTP (AC-13.1) -----------------------------------------------------------------
async def test_tenant_isolation_matrix_over_http(tmp_path: Path) -> None:
    """B owns a synced branch. A, with its own valid key, attacks with B's exact ids; B's namespaces stay
    byte-identical and every answer is about A's (empty) namespaces."""
    rig = make_rig()
    b_clone = await Clone.create(rig.service, rig.store, "beta", NEW_BR)
    await b_clone.sync(src(3))
    b_names = [n for n in rig.store.names() if n.startswith("scout_beta_")]
    assert len(b_names) >= 3
    before = {n: rig.store.dump(n) for n in b_names}
    a = HttpGateway(URL, await rig.key("acme"), transport=rig.transport())
    b = HttpGateway(URL, await rig.key("beta"), transport=rig.transport())

    assert (await a.whoami()).tenant == "acme"
    assert (await b.whoami()).tenant == "beta"
    assert await a.team_key() != await b.team_key()
    assert (await a.index_status(StatusReq(repo_id=REPO, branch_id=NEW_BR))).state == "absent"
    assert (await b.index_status(StatusReq(repo_id=REPO, branch_id=NEW_BR))).state == "ready"

    node_ids = [i for i, r in rig.store.rows(b_clone.tree_ns).items() if r.attrs.get("kind") == "node"]
    assert node_ids
    assert (await a.sync_nodes(NodesReq(repo_id=REPO, branch_id=NEW_BR, ids=node_ids))).blobs == {}
    assert (await b.sync_nodes(NodesReq(repo_id=REPO, branch_id=NEW_BR, ids=node_ids))).blobs
    chunk_keys = [r.attrs["chunk_key"] for r in rig.store.rows(b_clone.ns).values()]
    plan = await a.sync_plan(PlanReq(repo_id=REPO, branch_id=NEW_BR, chunk_keys=chunk_keys))
    assert set(plan.missing) == set(chunk_keys)  # A's cache holds none of B's embeddings

    for call in (
        lambda: a.search(SearchReq(repo_id=REPO, branch_id=NEW_BR, query="fn")),
        lambda: a.embed(EmbedReq(repo_id=REPO, branch_id=NEW_BR, texts={}, rows=[])),
        lambda: a.sync_stage_nodes(
            StageNodesReq(repo_id=REPO, branch_id=NEW_BR, new_root="n", page=0, pages=1)
        ),
    ):
        with pytest.raises(ScoutError) as ei:
            await call()
        assert ei.value.code is ErrorCode.NOT_FOUND
    with pytest.raises(ScoutError) as ei2:
        await a.sync_commit(CommitReq(repo_id=REPO, branch_id=NEW_BR, new_root="n", written_chunks=0))
    assert ei2.value.code in (ErrorCode.NOT_FOUND, ErrorCode.CONFLICT)
    seeded = await a.seed(
        SeedReq(repo_id=REPO, branch_id=NEW_BR, minhash_b64=b_clone.sketch_b64(b_clone.leaves(src(3))))
    )
    assert seeded.seeded is False  # A has no source namespace to branch from
    await a.delete_branch(REPO, NEW_BR)  # deletes A's (absent) namespaces only
    smuggled = {"repo_id": "scout_beta_x", "branch_id": NEW_BR, "root": "r", "expected_chunks": 0}
    async with httpx.AsyncClient(transport=rig.transport(), base_url=URL) as raw:
        hdr = {"Authorization": f"Bearer {await rig.key('acme')}", "X-Scout-Schema": "1"}
        r = await raw.post("/v1/sync/begin", headers=hdr, json=smuggled)  # a namespace name sent as an id
        assert (r.status_code, r.json()["error"]["code"]) == (400, "INVALID_ARGUMENT")

    stranger = HttpGateway(URL, "sk_scout_" + "B" * 43, transport=rig.transport(), sleep=_nosleep)
    with pytest.raises(ScoutError) as ei4:
        await stranger.whoami()
    assert ei4.value.code is ErrorCode.UNAUTHENTICATED

    assert {n: rig.store.dump(n) for n in b_names} == before
    assert not [n for n in rig.store.names() if n.startswith("scout_acme_") and n != "scout_acme_cache"]


async def _nosleep(_: float) -> None:
    return None
