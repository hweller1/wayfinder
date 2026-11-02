"""Task 8.1: every design-6 route over `TestClient`, bearer auth, header checks, error mapping.

FR-G1, AC-9.1. The service is the real `GatewayService` over fakes; keys come from the real `TenantDirectory`.
"""

import asyncio
import base64
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from gateway_app_helpers import Rig, bearer, body_of, make_rig
from svc_helpers import NEW_BR, REPO, Clone, src, text_key

from scout.adapters.fake_chat import ScriptedChat
from scout.core.contracts.errors import ErrorCode, ScoutError, http_status
from scout.core.contracts.wire import (
    CommitReq,
    EmbedReq,
    NodesReq,
    PlanReq,
    ReplaceFile,
    SearchReq,
    StageNodesReq,
    SyncBeginReq,
)
from scout.core.ports.chat import ChatTurn, ToolCall
from scout.core.service.gateway import GatewayService
from scout.surfaces.gateway_http.app import completion_json

FILES = src(3)


def setup() -> tuple[Rig, TestClient, dict[str, str]]:
    rig = make_rig()
    key = asyncio.run(rig.key("acme"))
    return rig, TestClient(rig.app), bearer(key)


def test_healthz_is_public_and_echoes_request_id() -> None:
    _, c, _ = setup()
    r = c.get("/v1/healthz", headers={"X-Request-Id": "req-123"})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "version": "0", "schema": "1"}
    assert r.headers["x-request-id"] == "req-123"
    assert c.get("/v1/healthz").headers["x-request-id"]  # generated when absent


def test_whoami_and_team_key_stable() -> None:
    _, c, h = setup()
    who = c.get("/v1/whoami", headers=h)
    assert who.status_code == 200
    assert who.json()["tenant"] == "acme"
    assert len(who.json()["key_id"]) == 32
    first = c.post("/v1/keys/team", headers=h).json()["k_team_b64"]
    second = c.post("/v1/keys/team", headers=h).json()["k_team_b64"]
    assert first == second
    assert len(base64.b64decode(first)) == 64


def test_sync_flow_over_http_every_endpoint_200() -> None:
    """begin, nodes, plan, stage, embed, commit, status, search, seed, delete: each answers 200/204."""
    rig, c, h = setup()
    clone = asyncio.run(_clone(rig, "acme", NEW_BR))
    tree = clone.tree(FILES)

    r = c.post("/v1/sync/begin", headers=h, json=body_of(_begin(tree.root_hash, NEW_BR, None)))
    assert r.status_code == 200, r.text
    assert r.json()["ns_state"] == "empty"

    r = c.post("/v1/sync/nodes", headers=h, json=body_of(NodesReq(repo_id=REPO, branch_id=NEW_BR, ids=[])))
    assert (r.status_code, r.json()) == (200, {"blobs": {}})

    ck = text_key("x")
    r = c.post(
        "/v1/sync/plan", headers=h, json=body_of(PlanReq(repo_id=REPO, branch_id=NEW_BR, chunk_keys=[ck]))
    )
    assert r.status_code == 200
    assert r.json() == {"missing": [ck]}

    stage = StageNodesReq(repo_id=REPO, branch_id=NEW_BR, new_root="n" * 8, page=0, pages=1)
    r = c.post("/v1/sync/stage", headers=h, json=body_of(stage))
    assert r.status_code == 200
    assert r.json()["page"] == 0

    path = "src/m0.py"
    row = clone.wire_row(path, FILES[path])
    embed = EmbedReq(
        repo_id=REPO,
        branch_id=NEW_BR,
        texts={text_key(FILES[path]): FILES[path]},
        rows=[row],
        replace=[ReplaceFile(file_id=row.file_id, file_hash=row.file_hash)],
    )
    r = c.post("/v1/embed", headers={**h, "Idempotency-Key": "k1"}, json=body_of(embed))
    assert r.status_code == 200, r.text
    assert r.json()["written"] == 1

    commit = CommitReq(
        repo_id=REPO, branch_id=NEW_BR, new_root=tree.root_hash, written_chunks=1, expected_chunks=1
    )
    r = c.post("/v1/sync/commit", headers={**h, "Idempotency-Key": "k2"}, json=body_of(commit))
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "ready"

    r = c.get("/v1/index/status", headers=h, params={"repo_id": REPO, "branch_id": NEW_BR})
    assert (r.status_code, r.json()["state"]) == (200, "ready")

    r = c.post("/v1/search", headers=h, json=body_of(SearchReq(repo_id=REPO, branch_id=NEW_BR, query="fn")))
    assert r.status_code == 200
    assert len(r.json()["hits"]) == 1

    other = asyncio.run(_clone(rig, "acme", "d" * 16))
    seed = other.seed_req(FILES)
    r = c.post("/v1/seed", headers={**h, "Idempotency-Key": "k3"}, json=body_of(seed))
    assert r.status_code == 200, r.text
    assert set(r.json()) >= {"seeded"}

    r = c.delete(f"/v1/namespaces/{REPO}/{NEW_BR}", headers=h)
    assert (r.status_code, r.content) == (204, b"")
    r = c.get("/v1/index/status", headers=h, params={"repo_id": REPO, "branch_id": NEW_BR})
    assert r.json()["state"] == "absent"


def test_chat_happy_path_is_sse_with_done() -> None:
    rig = make_rig(chat=_chat([ChatTurn("hello", (), 5, 2, "stop")]))
    h = bearer(asyncio.run(rig.key("acme")))
    r = TestClient(rig.app).post(
        "/v1/chat", headers=h, json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [ln[6:] for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert events[-1] == "[DONE]"
    done = json.loads(events[0])
    assert done["choices"][0]["message"]["content"] == "hello"
    assert done["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_completion_json_marks_unparseable_tool_calls() -> None:
    turn = ChatTurn(None, (ToolCall("c1", "grep", {}, parse_error="bad"),), 1, 1, "tool_calls")
    args = completion_json(turn)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    with pytest.raises(ValueError):  # noqa: PT011
        json.loads(args)


ROUTES: list[tuple[str, str, Any]] = [
    ("GET", "/v1/whoami", None),
    ("POST", "/v1/keys/team", None),
    ("POST", "/v1/sync/begin", {"repo_id": REPO, "branch_id": NEW_BR, "root": "r", "expected_chunks": 0}),
    ("POST", "/v1/sync/nodes", {"repo_id": REPO, "branch_id": NEW_BR, "ids": []}),
    ("POST", "/v1/sync/plan", {"repo_id": REPO, "branch_id": NEW_BR, "chunk_keys": []}),
    (
        "POST",
        "/v1/sync/stage",
        {"repo_id": REPO, "branch_id": NEW_BR, "new_root": "n", "page": 0, "pages": 1},
    ),
    ("POST", "/v1/embed", {"repo_id": REPO, "branch_id": NEW_BR, "texts": {}, "rows": []}),
    ("POST", "/v1/sync/commit", {"repo_id": REPO, "branch_id": NEW_BR, "new_root": "n", "written_chunks": 0}),
    ("GET", "/v1/index/status?repo_id=" + REPO + "&branch_id=" + NEW_BR, None),
    ("POST", "/v1/search", {"repo_id": REPO, "branch_id": NEW_BR, "query": "q"}),
    ("POST", "/v1/seed", {"repo_id": REPO, "branch_id": NEW_BR, "minhash_b64": "AA=="}),
    ("POST", "/v1/chat", {"messages": [{"role": "user", "content": "x"}]}),
    ("DELETE", f"/v1/namespaces/{REPO}/{NEW_BR}", None),
    ("POST", "/v1/warm", None),
]


@pytest.mark.parametrize(("verb", "path", "body"), ROUTES, ids=[r[1].split("?")[0] for r in ROUTES])
@pytest.mark.parametrize("auth", [None, "Bearer nope", "Bearer sk_scout_" + "A" * 43, "Basic abc"])
def test_every_route_but_healthz_rejects_bad_credentials(
    verb: str, path: str, body: Any, auth: str | None
) -> None:
    """AC-9.1: 401 UNAUTHENTICATED, one shape, request id present, nothing about why."""
    _, c, _ = setup()
    headers = {"Authorization": auth} if auth else {}
    r = c.request(verb, path, headers=headers, json=body)
    assert r.status_code == 401
    err = r.json()["error"]
    assert err["code"] == "UNAUTHENTICATED"
    assert err["request_id"] == r.headers["x-request-id"]
    assert r.json()["schema_version"] == "1.0"


def test_validation_errors_are_400_without_echoing_values() -> None:
    _, c, h = setup()
    canary = "CANARY-VALUE-X"
    r = c.post(
        "/v1/sync/begin",
        headers=h,
        json={"repo_id": canary, "branch_id": NEW_BR, "root": "r", "expected_chunks": 0},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert canary not in r.text
    r = c.post("/v1/sync/begin", headers=h, content=b"{not json")
    assert r.status_code == 400
    r = c.post(
        "/v1/sync/begin",
        headers=h,
        json={"repo_id": REPO, "branch_id": NEW_BR, "root": "r", "expected_chunks": 0, "x": 1},
    )
    assert r.status_code == 400  # extra fields are forbidden
    r = c.get("/v1/index/status", headers=h, params={"repo_id": "zz", "branch_id": NEW_BR})
    assert r.status_code == 400


def test_unsupported_schema_header_is_422() -> None:
    _, c, h = setup()
    r = c.get("/v1/whoami", headers={**h, "X-Scout-Schema": "2"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNSUPPORTED_SCHEMA"


def test_404_unknown_route_and_missing_namespace() -> None:
    _, c, h = setup()
    r = c.get("/v1/nope", headers=h)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"
    r = c.post("/v1/search", headers=h, json={"repo_id": REPO, "branch_id": NEW_BR, "query": "q"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_403_permission_denied_maps_from_the_service() -> None:
    class Denying(GatewayService):
        async def sync_plan(self, tenant: str, req: PlanReq) -> Any:
            raise ScoutError(ErrorCode.PERMISSION_DENIED, "no")

    rig = make_rig(service_cls=Denying)
    h = bearer(asyncio.run(rig.key("acme")))
    r = TestClient(rig.app).post(
        "/v1/sync/plan", headers=h, json={"repo_id": REPO, "branch_id": NEW_BR, "chunk_keys": []}
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "PERMISSION_DENIED"


@pytest.mark.parametrize("code", list(ErrorCode))
def test_error_table_status_codes(code: ErrorCode) -> None:
    """Every ErrorCode leaves with its design-4 status and the wire body."""

    class Raising(GatewayService):
        async def sync_plan(self, tenant: str, req: PlanReq) -> Any:
            raise ScoutError(code, "x", details={"retry_after": 2, "limit": 10})

    rig = make_rig(service_cls=Raising)
    h = bearer(asyncio.run(rig.key("acme")))
    r = TestClient(rig.app).post(
        "/v1/sync/plan", headers=h, json={"repo_id": REPO, "branch_id": NEW_BR, "chunk_keys": []}
    )
    assert r.status_code == http_status(code)
    assert r.json()["error"]["code"] == code.value
    assert r.json()["error"]["request_id"] == r.headers["x-request-id"]


def test_unhandled_exception_is_500_without_message() -> None:
    class Boom(GatewayService):
        async def sync_plan(self, tenant: str, req: PlanReq) -> Any:
            raise RuntimeError("boom detail")

    rig = make_rig(service_cls=Boom)
    h = bearer(asyncio.run(rig.key("acme")))
    c = TestClient(rig.app, raise_server_exceptions=False)
    r = c.post("/v1/sync/plan", headers=h, json={"repo_id": REPO, "branch_id": NEW_BR, "chunk_keys": []})
    assert r.status_code == 500
    assert "boom detail" not in r.text


def test_tenant_comes_from_the_key_not_the_request() -> None:
    rig = make_rig()
    ha, hb = (bearer(asyncio.run(rig.key(t))) for t in ("acme", "beta"))
    c = TestClient(rig.app)
    assert c.get("/v1/whoami", headers=ha).json()["tenant"] == "acme"
    assert c.get("/v1/whoami", headers={**hb, "X-Tenant": "acme"}).json()["tenant"] == "beta"


# --- helpers ---------------------------------------------------------------------------------------
def _begin(root: str, branch: str, base: str | None) -> SyncBeginReq:
    return SyncBeginReq(repo_id=REPO, branch_id=branch, base_branch_id=base, root=root, expected_chunks=1)


async def _clone(rig: Rig, tenant: str, branch: str) -> Clone:
    return await Clone.create(rig.service, rig.store, tenant, branch)


def _chat(turns: list[ChatTurn]) -> Any:
    from scout.adapters.fake_chat import ScriptedChat

    return ScriptedChat(turns)


class _WarmChat(ScriptedChat):
    def __init__(self, answer: bool) -> None:
        super().__init__([])
        self.answer, self.warmed = answer, 0

    async def warm(self) -> bool:
        self.warmed += 1
        return self.answer


@pytest.mark.parametrize("answer", [True, False])
def test_warm_route_triggers_the_upstream_warm_endpoint_without_a_chat_call(answer: bool) -> None:
    chat = _WarmChat(answer)
    rig = make_rig(chat=chat)
    key = asyncio.run(rig.key("acme"))
    r = TestClient(rig.app).post("/v1/warm", headers=bearer(key))
    assert r.status_code == 200
    assert r.json() == {"warming": answer}
    assert (chat.warmed, chat.calls) == (1, 0)


def test_warm_route_is_false_when_the_chat_model_has_no_warm_endpoint() -> None:
    _, c, h = setup()
    r = c.post("/v1/warm", headers=h)
    assert (r.status_code, r.json()) == (200, {"warming": False})
