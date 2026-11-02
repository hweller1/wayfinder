"""Task 3.10: `make_fake_gateway` is the real `GatewayService` over fakes, and every `Gateway` port method
works through it. FR-P1, AC-6.1 (setup).

The method list comes from the `Gateway` Protocol by introspection: a port method added without a smoke
call below fails `test_every_port_method_has_a_smoke_call`.
"""

import base64
import inspect
from typing import Any, cast

import pytest
from svc_helpers import Clone, RecordingStore

from scout import Scout
from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_gateway import make_fake_gateway
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import (
    CommitReq,
    EmbedReq,
    NodesReq,
    NodeUpsert,
    PlanReq,
    ReplaceFile,
    SearchReq,
    SeedReq,
    StageNodesReq,
    StatusReq,
    SyncBeginReq,
)
from scout.core.crypto import ids, keys, minhash, paths
from scout.core.ports.chat import ChatModel, ChatRequest, ChatTurn
from scout.core.ports.gateway import Gateway
from scout.core.service.gateway import GatewayService
from scout.core.sync import merkle
from scout.wiring import build_gateway_service

REPO = "a" * 32
BRANCH = "b" * 16
OTHER = "c" * 16
TURN = ChatTurn(content="hi", tool_calls=(), tokens_in=3, tokens_out=1, finish="stop")
CHAT_REQ = ChatRequest(messages=[{"role": "user", "content": "hello"}])
PATH, TEXT = "src/a.py", "def alpha():\n    return 1\n"


def port_methods() -> set[str]:
    """Public callables of the `Gateway` Protocol, including the inherited `ChatModel` ones."""
    names = {n for klass in inspect.getmro(Gateway) for n in vars(klass) if not n.startswith("_")}
    return {n for n in names if callable(getattr(Gateway, n, None))}


def _sketch(k: keys.Subkeys, tree: merkle.Tree) -> str:
    return base64.b64encode(minhash.sketch(k, [bytes.fromhex(x) for x in tree.leaves.values()])).decode()


async def smoke(gw: Gateway) -> dict[str, Any]:
    """Call every port method once with a valid request; returns method name -> result."""
    out: dict[str, Any] = {}
    out["whoami"] = await gw.whoami()
    out["ping"] = await gw.ping()
    team = await gw.team_key()
    out["team_key"] = team
    k = keys.derive(team)
    tree = merkle.build_tree([(PATH, merkle.leaf_hash(k, TEXT.encode()))], k, "dev", REPO)
    out["sync_begin"] = await gw.sync_begin(
        SyncBeginReq(repo_id=REPO, branch_id=BRANCH, root=tree.root_hash, expected_chunks=1)
    )
    out["sync_nodes"] = await gw.sync_nodes(NodesReq(repo_id=REPO, branch_id=BRANCH, ids=[]))
    ck = "d" * 32
    out["sync_plan"] = await gw.sync_plan(PlanReq(repo_id=REPO, branch_id=BRANCH, chunk_keys=[ck]))
    fid = ids.file_id(k, PATH)
    row = {
        "id": ids.row_id(k, fid, ck, 0),
        "file_id": fid,
        "path_enc": paths.enc_path(k, "dev", REPO, PATH),
        "ext_tag": ids.ext_tag(k, ".py"),
        "dir_tags": [ids.dir_tag(k, "src")],
        "file_hash": merkle.leaf_hash(k, TEXT.encode()),
        "chunk_key": ck,
        "loc_enc": "bG9j",
        "tokens": ["t" * 32],
    }
    out["embed"] = await gw.embed(
        EmbedReq.model_validate(
            {
                "repo_id": REPO,
                "branch_id": BRANCH,
                "texts": {ck: TEXT},
                "rows": [row],
                "replace": [ReplaceFile(file_id=fid, file_hash=row["file_hash"])],
            }
        )
    )
    sealed = merkle.seal_tree(tree, k)
    out["sync_stage_nodes"] = await gw.sync_stage_nodes(
        StageNodesReq(
            repo_id=REPO,
            branch_id=BRANCH,
            new_root=tree.root_hash,
            page=0,
            pages=1,
            node_upserts=[
                NodeUpsert(id=i, node_hash=n.node_hash, children_enc=n.children_enc)
                for i, n in sealed.items()
            ],
        )
    )
    out["sync_commit"] = await gw.sync_commit(
        CommitReq(
            repo_id=REPO,
            branch_id=BRANCH,
            new_root=tree.root_hash,
            staged_pages=1,
            written_chunks=1,
            expected_chunks=1,
            minhash_b64=_sketch(k, tree),
        )
    )
    out["search"] = await gw.search(SearchReq(repo_id=REPO, branch_id=BRANCH, query="alpha"))
    out["seed"] = await gw.seed(
        SeedReq(
            repo_id=REPO,
            branch_id=OTHER,
            minhash_b64=_sketch(k, tree),
            proofs=sorted(tree.leaves.values()),
        )
    )
    out["index_status"] = await gw.index_status(StatusReq(repo_id=REPO, branch_id=BRANCH))
    out["delete_branch"] = await gw.delete_branch(REPO, BRANCH)
    out["complete"] = await gw.complete(CHAT_REQ)
    try:
        gw.stream(CHAT_REQ)
    except NotImplementedError as exc:  # optional in the port; the service does not stream
        out["stream"] = exc
    return out


async def test_every_port_method_has_a_smoke_call() -> None:
    called = await smoke(make_fake_gateway(chat=ScriptedChat([TURN])))
    assert port_methods() == set(called)
    assert port_methods() >= {"whoami", "ping", "team_key", "embed", "complete", "stream", "delete_branch"}


async def test_smoke_results_come_from_the_real_service() -> None:
    out = await smoke(make_fake_gateway(tenant="acme", chat=ScriptedChat([TURN])))
    assert out["whoami"].tenant == "acme"
    assert len(out["team_key"]) == 64
    assert out["sync_begin"].ns_state == "empty"
    assert out["sync_plan"].missing == ["d" * 32]
    assert out["embed"].written == 1
    assert out["embed"].embedded_tokens > 0
    assert out["sync_stage_nodes"].staged == 2  # `src` and the root directory
    assert out["sync_commit"].state == "ready"
    assert out["index_status"].state == "ready"
    assert out["search"].hits  # the committed row is found
    assert out["seed"].seeded is True  # the committed branch is an identical seed source (jaccard 1.0)
    assert out["complete"].content == "hi"
    assert isinstance(out["stream"], NotImplementedError)


async def test_index_is_absent_until_a_sync_runs_and_absent_after_delete() -> None:
    store = FakeStore()
    gw = make_fake_gateway(store=store, chat=ScriptedChat([TURN]))
    status = StatusReq(repo_id=REPO, branch_id=BRANCH)
    assert (await gw.index_status(status)).state == "absent"
    await smoke(gw)  # ends with delete_branch
    assert (await gw.index_status(status)).state == "absent"


async def test_injected_store_embedder_clock_and_chat_are_used() -> None:
    store, emb = FakeStore(), FakeEmbedder()
    chat = ScriptedChat([TURN])
    gw = make_fake_gateway("dev", cast(ChatModel, chat), store, emb, FakeClock())
    await smoke(gw)
    assert await store.list_namespaces("scout_dev_")  # the service wrote into the injected store
    assert emb.calls  # embed and search used the injected embedder
    assert chat.calls == 1


async def test_tenant_is_bound_and_validated() -> None:
    assert (await make_fake_gateway("dev").whoami()).tenant == "dev"
    with pytest.raises(ScoutError) as ei:
        make_fake_gateway("bad_tenant")
    assert ei.value.code is ErrorCode.INVALID_ARGUMENT


async def test_a_real_client_sync_makes_the_index_ready_through_the_fake_gateway() -> None:
    """The service-level `Clone` (real keyed tree, real embed/commit calls) drives the factory's gateway."""

    class Bound:
        def __init__(self, gw: Gateway) -> None:
            self._gw = gw

        def for_tenant(self, tenant: str) -> Gateway:
            return self._gw

    store = RecordingStore()
    gw = make_fake_gateway("dev", store=store)
    clone = await Clone.create(cast(GatewayService, Bound(gw)), store, "dev", BRANCH)
    assert (await gw.index_status(StatusReq(repo_id=REPO, branch_id=BRANCH))).state == "absent"
    await clone.sync({PATH: TEXT})
    status = await gw.index_status(StatusReq(repo_id=REPO, branch_id=BRANCH))
    assert status.state == "ready"
    assert status.semantic_enabled


@pytest.mark.slow
def test_fake_profile_scout_runs_over_the_service_backed_gateway() -> None:
    scout = Scout(profile="fake", case={"repo": ".", "chat_script": []})
    assert scout.index_status().state == "absent"


def test_build_gateway_service_is_a_typed_not_configured_error() -> None:
    with pytest.raises(ScoutError) as ei:
        build_gateway_service({})
    assert ei.value.code is ErrorCode.INTERNAL
    assert "not configured" in ei.value.message
