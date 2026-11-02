"""Task 8.2: idempotency replay, rate-limit headers, SSE chat with keepalives, no canary in logs.

FR-G10, AC-9.3, AC-9.5 (mechanism), AC-13.3. Real `GatewayService` over fakes behind the real FastAPI app,
driven through `httpx.ASGITransport` so the SSE body streams the way a real client sees it.
"""

import asyncio
import contextlib
import json
import logging
import time
from typing import Any

import httpx
import pytest
from gateway_app_helpers import Rig, bearer, body_of, make_rig
from svc_helpers import NEW_BR, REPO, Clone, src, text_key

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.gateway_http import HttpGateway
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import CommitReq, EmbedReq, ReplaceFile, SeedReq
from scout.core.ports.chat import ChatRequest, ChatTurn
from scout.core.service.ratelimit import Limits, RateLimiter
from scout.surfaces.gateway_http.app import MemoryIdempotencyCache

FILES = src(2)
TURN = ChatTurn("done", (), 4, 2, "stop")


def client(rig: Rig) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=rig.transport(), base_url="http://gw.test")


async def _embed_req(rig: Rig, tenant: str = "acme") -> tuple[Clone, EmbedReq]:
    clone = await Clone.create(rig.service, rig.store, tenant, NEW_BR)
    from scout.core.contracts.wire import SyncBeginReq

    tree = clone.tree(FILES)
    await clone.gw.sync_begin(
        SyncBeginReq(repo_id=REPO, branch_id=NEW_BR, root=tree.root_hash, expected_chunks=2)
    )
    rows = [clone.wire_row(p, t) for p, t in FILES.items()]
    req = EmbedReq(
        repo_id=REPO,
        branch_id=NEW_BR,
        texts={text_key(t): t for t in FILES.values()},
        rows=rows,
        replace=[ReplaceFile(file_id=r.file_id, file_hash=r.file_hash) for r in rows],
    )
    return clone, req


# --- idempotency -----------------------------------------------------------------------------------
async def test_idempotency_key_required_on_embed_commit_seed() -> None:
    """FR-G10: 400 without the header; the service is not called."""
    rig = make_rig()
    h = bearer(await rig.key("acme"))
    _, req = await _embed_req(rig)
    async with client(rig) as c:
        r = await c.post("/v1/embed", headers=h, json=body_of(req))
        assert (r.status_code, r.json()["error"]["code"]) == (400, "INVALID_ARGUMENT")
        assert "Idempotency-Key" in r.json()["error"]["message"]
        commit = CommitReq(repo_id=REPO, branch_id=NEW_BR, new_root="n", written_chunks=0)
        r = await c.post("/v1/sync/commit", headers=h, json=body_of(commit))
        assert r.status_code == 400
        r = await c.post(
            "/v1/seed", headers=h, json=body_of(SeedReq(repo_id=REPO, branch_id=NEW_BR, minhash_b64="AA=="))
        )
        assert r.status_code == 400
        r = await c.post("/v1/embed", headers={**h, "Idempotency-Key": "x" * 129}, json=body_of(req))
        assert r.status_code == 400
    assert rig.embedder.embedded == []


async def test_embed_replay_returns_identical_body_and_no_second_embed_call() -> None:
    """FR-G10: replay within 24 h answers from the store, even after the embed cache is gone."""
    rig = make_rig()
    h = bearer(await rig.key("acme"), **{"Idempotency-Key": "once"})
    _, req = await _embed_req(rig)
    async with client(rig) as c:
        first = await c.post("/v1/embed", headers=h, json=body_of(req))
        calls = len(rig.embedder.embedded)
        assert first.status_code == 200
        assert calls == 2
        second = await c.post("/v1/embed", headers=h, json=body_of(req))
    assert second.content == first.content
    assert second.headers["idempotent-replayed"] == "true"
    assert len(rig.embedder.embedded) == calls


async def test_replay_with_another_body_is_400_and_keys_are_per_tenant() -> None:
    rig = make_rig()
    ha = bearer(await rig.key("acme"), **{"Idempotency-Key": "k"})
    hb = bearer(await rig.key("beta"), **{"Idempotency-Key": "k"})
    _, req = await _embed_req(rig)
    other = req.model_copy(update={"delete_files": []})
    other2 = req.model_copy(update={"rows": req.rows[:1], "texts": {}})
    async with client(rig) as c:
        assert (await c.post("/v1/embed", headers=ha, json=body_of(req))).status_code == 200
        assert (await c.post("/v1/embed", headers=ha, json=body_of(other))).status_code == 200  # same body
        r = await c.post("/v1/embed", headers=ha, json=body_of(other2))
        assert r.status_code == 400
        assert "different request" in r.json()["error"]["message"]
        # beta's identical key is a different slot: it hits its own (missing) namespace, not acme's replay
        r = await c.post("/v1/embed", headers=hb, json=body_of(req))
        assert r.status_code == 404
        assert "idempotent-replayed" not in r.headers


async def test_replay_expires_after_24h_and_errors_are_not_replayed() -> None:
    clock = FakeClock()
    rig = make_rig(clock=clock, cache=MemoryIdempotencyCache(clock))
    h = bearer(await rig.key("acme"), **{"Idempotency-Key": "k"})
    _, req = await _embed_req(rig)
    async with client(rig) as c:
        await c.post("/v1/embed", headers=h, json=body_of(req))
        clock.advance(24 * 3600 + 1)
        r = await c.post("/v1/embed", headers=h, json=body_of(req))
    assert "idempotent-replayed" not in r.headers
    assert r.status_code == 200  # recomputed (row ids and the embed cache make it idempotent anyway)


async def test_embed_replay_survives_the_http_client_retry_path() -> None:
    """The adapter's retry reuses its key: a 503 after the server finished replays instead of re-embedding."""
    rig = make_rig()
    key = await rig.key("acme")
    _, req = await _embed_req(rig)
    state = {"n": 0}

    class FlakyOnce(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            res = await rig.transport().handle_async_request(request)
            state["n"] += 1
            if state["n"] == 1:
                await res.aread()
                return httpx.Response(503, json={"error": {"code": "UPSTREAM_UNAVAILABLE", "message": "x"}})
            return res

    async def nosleep(_: float) -> None:
        return None

    gw = HttpGateway("http://gw.test", key, transport=FlakyOnce(), sleep=nosleep)
    res = await gw.embed(req)
    assert res.written == 2
    assert len(rig.embedder.embedded) == 2  # embedded once; the retry was a replay
    assert state["n"] == 2


# --- rate limits -----------------------------------------------------------------------------------
async def test_429_carries_retry_after_and_ratelimit_headers() -> None:
    """AC-9.3."""
    clock = FakeClock()
    limiter = RateLimiter(clock, Limits(search_per_s=1.0, search_burst=1.0))
    rig = make_rig(clock=clock, limiter=limiter)
    h = bearer(await rig.key("acme"))
    body = {"repo_id": REPO, "branch_id": NEW_BR, "query": "q"}
    async with client(rig) as c:
        first = await c.post("/v1/search", headers=h, json=body)  # 404 (no namespace) but it spent the token
        assert first.status_code == 404
        r = await c.post("/v1/search", headers=h, json=body)
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "RATE_LIMITED"
    assert r.json()["error"]["retryable"] is True
    assert int(r.headers["retry-after"]) >= 1
    assert r.headers["ratelimit-limit"] == "1"
    assert r.headers["ratelimit-remaining"] == "0"
    assert int(r.headers["ratelimit-reset"]) >= 1


async def test_rate_limit_is_per_key() -> None:
    clock = FakeClock()
    rig = make_rig(clock=clock, limiter=RateLimiter(clock, Limits(search_per_s=1.0, search_burst=1.0)))
    ha, hb = bearer(await rig.key("acme")), bearer(await rig.key("beta"))
    body = {"repo_id": REPO, "branch_id": NEW_BR, "query": "q"}
    async with client(rig) as c:
        await c.post("/v1/search", headers=ha, json=body)
        assert (await c.post("/v1/search", headers=ha, json=body)).status_code == 429
        assert (await c.post("/v1/search", headers=hb, json=body)).status_code == 404


# --- SSE chat --------------------------------------------------------------------------------------
async def asgi_stream(
    rig: Rig, headers: dict[str, str], body: dict[str, Any], *, disconnect_after: int | None = None
) -> tuple[int, list[tuple[float, str]]]:
    """Drive the ASGI app directly (httpx's ASGITransport buffers the body): `(status, [(t, line), ...])`.
    With `disconnect_after`, the client goes away after that many body chunks."""
    payload = json.dumps(body).encode()
    raw_headers = [
        (k.lower().encode(), v.encode()) for k, v in {**headers, "content-type": "application/json"}.items()
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat",
        "raw_path": b"/v1/chat",
        "query_string": b"",
        "headers": [*raw_headers, (b"content-length", str(len(payload)).encode())],
        "client": ("127.0.0.1", 1),
        "server": ("gw.test", 80),
    }
    sent_body = False
    gone = asyncio.Event()
    status = 0
    out: list[tuple[float, str]] = []
    t0 = time.monotonic()

    async def receive() -> dict[str, Any]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(msg: dict[str, Any]) -> None:
        nonlocal status
        if msg["type"] == "http.response.start":
            status = msg["status"]
        elif msg["type"] == "http.response.body":
            for line in msg.get("body", b"").decode().splitlines():
                if line:
                    out.append((time.monotonic() - t0, line))
            if disconnect_after is not None and len(out) >= disconnect_after:
                gone.set()
                raise OSError("client disconnected")

    with contextlib.suppress(OSError):
        await rig.app(scope, receive, send)  # type: ignore[arg-type]
    return status, out


class SlowChat:
    """Takes `delay` real seconds, then answers; counts concurrent callers."""

    def __init__(self, delay: float, turn: ChatTurn = TURN) -> None:
        self.delay, self.turn = delay, turn
        self.inflight = 0
        self.peak = 0
        self.two_started = asyncio.Event()

    async def complete(self, req: ChatRequest) -> ChatTurn:
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        if self.inflight >= 2:
            self.two_started.set()
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.inflight -= 1
        return self.turn

    def stream(self, req: ChatRequest) -> Any:
        raise NotImplementedError


CHAT_BODY = {"messages": [{"role": "user", "content": "hi"}]}


async def test_upstream_delay_yields_keepalives_before_the_data_event() -> None:
    """AC-9.5 mechanism: a 3 s upstream call with a 0.5 s keepalive sends >= 3 keepalives first, and the
    first one arrives long before the upstream answers (the connection never idles)."""
    chat = SlowChat(3.0)
    rig = make_rig(chat=chat, keepalive_s=0.5)  # type: ignore[arg-type]
    status, lines = await asgi_stream(rig, bearer(await rig.key("acme")), CHAT_BODY)
    assert status == 200
    kinds = [ln for _, ln in lines]
    first_data = next(i for i, ln in enumerate(kinds) if ln.startswith("data: ") and ln != "data: [DONE]")
    assert kinds[:first_data].count(": keepalive") >= 3
    assert all(k == ": keepalive" for k in kinds[:first_data])
    assert json.loads(kinds[first_data][6:])["choices"][0]["message"]["content"] == "done"
    assert kinds[-1] == "data: [DONE]"
    gaps = [b[0] - a[0] for a, b in zip(lines, lines[1:], strict=False)]
    assert lines[0][0] < 1.0
    assert max(gaps[: first_data - 1] or [0.0]) < 1.0  # keepalives keep coming while the upstream runs
    assert lines[first_data][0] >= 2.9


async def test_http_client_adapter_reads_the_keepalive_stream() -> None:
    chat = SlowChat(0.6)
    rig = make_rig(chat=chat, keepalive_s=0.2)  # type: ignore[arg-type]
    gw = HttpGateway("http://gw.test", await rig.key("acme"), transport=rig.transport())
    turn = await gw.complete(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert (turn.content, turn.tokens_in, turn.tokens_out) == ("done", 4, 2)


async def test_third_concurrent_chat_for_a_key_is_429_before_the_stream() -> None:
    chat = SlowChat(1.0)
    rig = make_rig(chat=chat, keepalive_s=0.2)  # type: ignore[arg-type]
    h = bearer(await rig.key("acme"))
    async with client(rig) as c:

        async def one() -> int:
            r = await c.post("/v1/chat", headers=h, json=CHAT_BODY)
            return r.status_code

        t1, t2 = asyncio.create_task(one()), asyncio.create_task(one())
        await asyncio.wait_for(chat.two_started.wait(), 5)
        third = await c.post("/v1/chat", headers=h, json=CHAT_BODY)
        assert third.status_code == 429
        assert third.json()["error"]["code"] == "RATE_LIMITED"
        assert "retry-after" in third.headers
        assert await t1 == await t2 == 200
    assert chat.peak == 2


async def test_chat_upstream_failure_after_200_is_an_error_event() -> None:
    chat = ScriptedChat([ScoutError(ErrorCode.UPSTREAM_TIMEOUT, "slow")])
    rig = make_rig(chat=chat, keepalive_s=0.2)
    h = bearer(await rig.key("acme"))
    async with client(rig) as c:
        r = await c.post("/v1/chat", headers=h, json=CHAT_BODY)
    # an immediate failure is still reported as a plain error response (known before the 200)
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "UPSTREAM_TIMEOUT"
    gw = HttpGateway("http://gw.test", await rig.key("acme"), transport=rig.transport(), sleep=_nosleep)
    chat2 = ScriptedChat([ScoutError(ErrorCode.UPSTREAM_TIMEOUT, "slow")] * 3)
    rig2 = make_rig(chat=chat2)
    gw2 = HttpGateway("http://gw.test", await rig2.key("acme"), transport=rig2.transport(), sleep=_nosleep)
    with pytest.raises(ScoutError) as ei:
        await gw2.complete(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert ei.value.code is ErrorCode.UPSTREAM_TIMEOUT
    _ = gw


async def _nosleep(_: float) -> None:
    return None


async def test_chat_body_validation_is_400_and_cap_is_enforced() -> None:
    rig = make_rig(chat=SlowChat(0))  # type: ignore[arg-type]
    h = bearer(await rig.key("acme"))
    async with client(rig) as c:
        assert (await c.post("/v1/chat", headers=h, json={"messages": []})).status_code == 400
        assert (
            await c.post("/v1/chat", headers=h, json={**CHAT_BODY, "max_tokens": 10**9})
        ).status_code == 400


async def test_client_disconnect_cancels_the_upstream_call() -> None:
    chat = SlowChat(30.0)
    rig = make_rig(chat=chat, keepalive_s=0.05)  # type: ignore[arg-type]
    t0 = time.monotonic()
    _, lines = await asgi_stream(rig, bearer(await rig.key("acme")), CHAT_BODY, disconnect_after=1)
    assert lines[0][1] == ": keepalive"
    for _ in range(100):
        if chat.inflight == 0:
            break
        await asyncio.sleep(0.02)
    assert chat.inflight == 0
    assert time.monotonic() - t0 < 5  # nowhere near the 30 s upstream


# --- AC-13.3 ---------------------------------------------------------------------------------------
async def test_canary_absent_from_logs_over_http(caplog: pytest.LogCaptureFixture) -> None:
    """AC-13.3: a prompt, a query, chunk text and a path canary never reach any log record."""
    canary = "CANARY-7f3a91-do-not-log"
    chat = ScriptedChat(
        [ChatTurn(f"echo {canary}", (), 3, 1, "stop"), ScoutError(ErrorCode.INTERNAL, canary)]
    )
    rig = make_rig(chat=chat, keepalive_s=0.05)
    h = bearer(await rig.key("acme"))
    caplog.set_level(logging.DEBUG)
    async with client(rig) as c:
        msg = {"messages": [{"role": "user", "content": canary}, {"role": "tool", "content": canary}]}
        assert (await c.post("/v1/chat", headers=h, json=msg)).status_code == 200
        await c.post("/v1/chat", headers=h, json=msg)
        await c.post("/v1/search", headers=h, json={"repo_id": REPO, "branch_id": NEW_BR, "query": canary})
        bad = {"repo_id": canary, "branch_id": NEW_BR, "root": "r", "expected_chunks": 0}
        await c.post("/v1/sync/begin", headers=h, json=bad)
        await c.post("/v1/sync/begin", headers={"Authorization": f"Bearer {canary}"}, json=bad)
    _, req = await _embed_req(rig)
    req = req.model_copy(update={"texts": {k: canary + v for k, v in req.texts.items()}})
    async with client(rig) as c:
        await c.post("/v1/embed", headers={**h, "Idempotency-Key": "k"}, json=body_of(req))
    assert caplog.records
    for rec in caplog.records:
        assert canary not in rec.getMessage()
        assert canary not in str(rec.args)
        assert canary not in (rec.exc_text or "")
