"""Task 7.6: `HttpGateway` over `httpx.MockTransport`. FR-G1, FR-G10, AC-9.5 (client side of the mechanism).

Every `Gateway` method hits the design-6 path and verb with the required headers; the error body maps to
`ScoutError`; the `Idempotency-Key` is stable across a retry; an SSE chat with keepalives parses to a turn.
"""

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from scout.adapters.gateway_http import HttpGateway
from scout.core.contracts.errors import ErrorCode, ScoutError, http_status
from scout.core.contracts.wire import (
    CommitReq,
    EmbedReq,
    NodesReq,
    PlanReq,
    SearchReq,
    SeedReq,
    StageNodesReq,
    StatusReq,
    SyncBeginReq,
)
from scout.core.ports.chat import ChatRequest

REPO = "a" * 32
BR = "b" * 16
KEY = "sk_scout_" + "A" * 43
STATUS = {"schema_version": "1.0", "state": "absent", "coverage": 0.0, "semantic_enabled": False}

Handler = Callable[[httpx.Request], httpx.Response]


class Rig:
    def __init__(self, handler: Handler) -> None:
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

        async def sleep(s: float) -> None:
            self.sleeps.append(s)

        def record(req: httpx.Request) -> httpx.Response:
            self.requests.append(req)
            return handler(req)

        self.gw = HttpGateway(
            "https://gw.test/",
            KEY,
            "9.9",
            transport=httpx.MockTransport(record),
            sleep=sleep,
        )


def ok(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body)


def err(code: ErrorCode, message: str = "boom", **extra: Any) -> httpx.Response:
    body = {
        "schema_version": "1.0",
        "error": {
            "code": code.value,
            "message": message,
            "retryable": False,
            "details": extra.pop("details", {}),
            "request_id": "r",
        },
    }
    return httpx.Response(http_status(code), json=body, **extra)


def _begin() -> SyncBeginReq:
    return SyncBeginReq(repo_id=REPO, branch_id=BR, root="r" * 8, expected_chunks=0)


def _embed() -> EmbedReq:
    return EmbedReq(repo_id=REPO, branch_id=BR, texts={}, rows=[])


def _commit() -> CommitReq:
    return CommitReq(repo_id=REPO, branch_id=BR, new_root="x", written_chunks=0)


def _call_table() -> list[tuple[str, str, str, Callable[[HttpGateway], Any], Any]]:
    """(label, verb, path, call, response body)."""
    nodes = {"blobs": {}}
    return [
        ("whoami", "GET", "/v1/whoami", lambda g: g.whoami(), {"tenant": "dev", "key_id": "k"}),
        (
            "team_key",
            "POST",
            "/v1/keys/team",
            lambda g: g.team_key(),
            {"k_team_b64": base64.b64encode(b"k" * 64).decode()},
        ),
        (
            "sync_begin",
            "POST",
            "/v1/sync/begin",
            lambda g: g.sync_begin(_begin()),
            {"ns_state": "empty", "coverage": 0.0},
        ),
        (
            "sync_nodes",
            "POST",
            "/v1/sync/nodes",
            lambda g: g.sync_nodes(NodesReq(repo_id=REPO, branch_id=BR, ids=[])),
            nodes,
        ),
        (
            "sync_plan",
            "POST",
            "/v1/sync/plan",
            lambda g: g.sync_plan(PlanReq(repo_id=REPO, branch_id=BR, chunk_keys=[])),
            {"missing": []},
        ),
        ("embed", "POST", "/v1/embed", lambda g: g.embed(_embed()), {"written": 0, "embedded_tokens": 0}),
        (
            "stage",
            "POST",
            "/v1/sync/stage",
            lambda g: g.sync_stage_nodes(
                StageNodesReq(repo_id=REPO, branch_id=BR, new_root="n", page=0, pages=1)
            ),
            {"page": 0, "staged": 0},
        ),
        ("commit", "POST", "/v1/sync/commit", lambda g: g.sync_commit(_commit()), STATUS),
        (
            "status",
            "GET",
            "/v1/index/status",
            lambda g: g.index_status(StatusReq(repo_id=REPO, branch_id=BR)),
            STATUS,
        ),
        (
            "search",
            "POST",
            "/v1/search",
            lambda g: g.search(SearchReq(repo_id=REPO, branch_id=BR, query="q")),
            {"hits": []},
        ),
        (
            "seed",
            "POST",
            "/v1/seed",
            lambda g: g.seed(SeedReq(repo_id=REPO, branch_id=BR, minhash_b64="AA==")),
            {"seeded": False},
        ),
        ("delete", "DELETE", f"/v1/namespaces/{REPO}/{BR}", lambda g: g.delete_branch(REPO, BR), None),
    ]


@pytest.mark.parametrize("row", _call_table(), ids=lambda r: r[0])
async def test_every_method_hits_its_design6_route(row: tuple[str, str, str, Any, Any]) -> None:
    """FR-G1: verb, path and the standard headers for each endpoint."""
    _, verb, path, call, body = row
    rig = Rig(lambda r: ok(body) if body is not None else httpx.Response(204))
    await call(rig.gw)
    (req,) = rig.requests
    assert (req.method, req.url.path) == (verb, path)
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert req.headers["x-scout-schema"] == "1"
    assert req.headers["x-scout-client"] == "scout/9.9"
    assert req.headers["x-request-id"]
    needs_key = path in ("/v1/embed", "/v1/sync/commit", "/v1/seed")
    assert ("idempotency-key" in req.headers) is needs_key


async def test_index_status_query_and_healthz_ping() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/v1/healthz":
            assert "authorization" not in r.headers
            return ok({"ok": True, "version": "0", "schema": "1"})
        return ok({"tenant": "dev", "key_id": "k"})

    rig = Rig(handler)
    await rig.gw.ping()
    assert [r.url.path for r in rig.requests] == ["/v1/healthz", "/v1/whoami"]
    rig2 = Rig(lambda r: ok(STATUS))
    await rig2.gw.index_status(StatusReq(repo_id=REPO, branch_id=BR))
    assert dict(rig2.requests[0].url.params) == {"repo_id": REPO, "branch_id": BR}


async def test_request_body_is_the_wire_model() -> None:
    rig = Rig(lambda r: ok({"ns_state": "empty", "coverage": 0.0}))
    await rig.gw.sync_begin(_begin())
    assert json.loads(rig.requests[0].content) == _begin().model_dump(mode="json")


@pytest.mark.parametrize(
    "code",
    [c for c in ErrorCode if c is not ErrorCode.RATE_LIMITED and http_status(c) not in (503, 504)],
)
async def test_error_body_maps_to_scout_error(code: ErrorCode) -> None:
    rig = Rig(lambda r: err(code, "nope", details={"x": 1}))
    with pytest.raises(ScoutError) as ei:
        await rig.gw.whoami()
    assert ei.value.code is code
    assert ei.value.message == "nope"
    assert ei.value.details["x"] == 1
    assert len(rig.requests) == 1  # not retried


async def test_non_json_error_maps_from_status() -> None:
    rig = Rig(lambda r: httpx.Response(401, text="<html>nope</html>"))
    with pytest.raises(ScoutError) as ei:
        await rig.gw.whoami()
    assert ei.value.code is ErrorCode.UNAUTHENTICATED


async def test_429_honors_retry_after_then_succeeds() -> None:
    n = {"i": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        n["i"] += 1
        if n["i"] == 1:
            return err(ErrorCode.RATE_LIMITED, "slow", headers={"Retry-After": "7"})
        return ok({"tenant": "dev", "key_id": "k"})

    rig = Rig(handler)
    assert (await rig.gw.whoami()).tenant == "dev"
    assert rig.sleeps == [7.0]


@pytest.mark.parametrize("code", [ErrorCode.UPSTREAM_UNAVAILABLE, ErrorCode.UPSTREAM_TIMEOUT])
async def test_503_504_retry_three_attempts_then_raise(code: ErrorCode) -> None:
    rig = Rig(lambda r: err(code))
    with pytest.raises(ScoutError) as ei:
        await rig.gw.whoami()
    assert ei.value.code is code
    assert len(rig.requests) == 3
    assert len(rig.sleeps) == 2


async def test_transport_error_retries_as_unavailable() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    rig = Rig(handler)
    with pytest.raises(ScoutError) as ei:
        await rig.gw.whoami()
    assert ei.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert len(rig.requests) == 3


@pytest.mark.parametrize("which", ["embed", "commit", "seed"])
async def test_idempotency_key_stable_across_retry_and_fresh_per_call(which: str) -> None:
    n = {"i": 0}
    bodies = {
        "embed": {"written": 0, "embedded_tokens": 0},
        "commit": STATUS,
        "seed": {"seeded": False},
    }

    def handler(r: httpx.Request) -> httpx.Response:
        n["i"] += 1
        return err(ErrorCode.UPSTREAM_UNAVAILABLE) if n["i"] == 1 else ok(bodies[which])

    def call(g: HttpGateway) -> Any:
        if which == "embed":
            return g.embed(_embed())
        if which == "commit":
            return g.sync_commit(_commit())
        return g.seed(SeedReq(repo_id=REPO, branch_id=BR, minhash_b64="AA=="))

    rig = Rig(handler)
    await call(rig.gw)
    first, second = (r.headers["idempotency-key"] for r in rig.requests)
    assert first == second
    assert rig.requests[0].headers["x-request-id"] == rig.requests[1].headers["x-request-id"]
    await call(rig.gw)
    assert rig.requests[2].headers["idempotency-key"] != first


async def test_embed_uses_the_callers_idempotency_key() -> None:
    rig = Rig(lambda r: ok({"written": 0, "embedded_tokens": 0}))
    await rig.gw.embed(_embed(), idempotency_key="engine-key")
    assert rig.requests[0].headers["idempotency-key"] == "engine-key"


def _completion(content: str | None = "hi", calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {
        "choices": [{"message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


def sse(*events: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text="".join(events))


async def test_chat_sse_with_keepalives_parses_to_turn() -> None:
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "grep", "arguments": json.dumps({"pattern": "x"})},
    }
    body = sse(
        ": keepalive\n\n",
        ": keepalive\n\n",
        f"data: {json.dumps(_completion(None, [call]))}\n\n",
        "data: [DONE]\n\n",
    )
    rig = Rig(lambda r: body)
    turn = await rig.gw.complete(ChatRequest(messages=[{"role": "user", "content": "q"}]))
    assert turn.tool_calls[0].name == "grep"
    assert turn.tool_calls[0].arguments == {"pattern": "x"}
    assert (turn.tokens_in, turn.tokens_out, turn.finish) == (11, 3, "tool_calls")
    req = rig.requests[0]
    assert (req.method, req.url.path) == ("POST", "/v1/chat")
    sent = json.loads(req.content)
    assert sent["thinking"] is True
    assert sent["messages"] == [{"role": "user", "content": "q"}]
    assert "idempotency-key" not in req.headers


async def test_chat_bad_tool_arguments_set_parse_error() -> None:
    call = {"id": "c1", "function": {"name": "grep", "arguments": "{not json"}}
    rig = Rig(lambda r: sse(f"data: {json.dumps(_completion(None, [call]))}\n\n", "data: [DONE]\n\n"))
    turn = await rig.gw.complete(ChatRequest(messages=[{"role": "user", "content": "q"}]))
    assert turn.tool_calls[0].parse_error


async def test_chat_error_event_after_200_maps_to_scout_error() -> None:
    body = {
        "schema_version": "1.0",
        "error": {"code": "UPSTREAM_TIMEOUT", "message": "late", "retryable": True},
    }
    rig = Rig(lambda r: sse(": keepalive\n\n", f"data: {json.dumps(body)}\n\n", "data: [DONE]\n\n"))
    with pytest.raises(ScoutError) as ei:
        await rig.gw.complete(ChatRequest(messages=[{"role": "user", "content": "q"}]))
    assert ei.value.code is ErrorCode.UPSTREAM_TIMEOUT


async def test_chat_http_error_and_empty_stream() -> None:
    rig = Rig(lambda r: err(ErrorCode.UNAUTHENTICATED))
    with pytest.raises(ScoutError) as ei:
        await rig.gw.complete(ChatRequest(messages=[{"role": "user", "content": "q"}]))
    assert ei.value.code is ErrorCode.UNAUTHENTICATED
    rig2 = Rig(lambda r: sse(": keepalive\n\n", "data: [DONE]\n\n"))
    with pytest.raises(ScoutError) as ei2:
        await rig2.gw.complete(ChatRequest(messages=[{"role": "user", "content": "q"}]))
    assert ei2.value.code is ErrorCode.UPSTREAM_UNAVAILABLE


async def test_chat_429_retries_with_retry_after() -> None:
    n = {"i": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        n["i"] += 1
        if n["i"] == 1:
            return err(ErrorCode.RATE_LIMITED, headers={"Retry-After": "2"})
        return sse(f"data: {json.dumps(_completion())}\n\n", "data: [DONE]\n\n")

    rig = Rig(handler)
    turn = await rig.gw.complete(ChatRequest(messages=[{"role": "user", "content": "q"}]))
    assert turn.content == "hi"
    assert rig.sleeps == [2.0]


async def test_stream_is_not_supported() -> None:
    rig = Rig(lambda r: ok({}))
    with pytest.raises(NotImplementedError):
        rig.gw.stream(ChatRequest(messages=[]))


async def test_missing_key_is_login_required_without_a_request() -> None:
    n = {"i": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        n["i"] += 1
        return ok({})

    gw = HttpGateway("https://gw.test", lambda: None, transport=httpx.MockTransport(handler))
    with pytest.raises(ScoutError) as ei:
        await gw.whoami()
    assert ei.value.code is ErrorCode.LOGIN_REQUIRED
    assert n["i"] == 0
    box: dict[str, str | None] = {"k": None}
    gw2 = HttpGateway(
        "https://gw.test",
        lambda: box["k"],
        transport=httpx.MockTransport(lambda r: ok({"tenant": "t", "key_id": r.headers["authorization"]})),
    )
    box["k"] = KEY  # a login after construction is picked up
    assert (await gw2.whoami()).key_id == f"Bearer {KEY}"
