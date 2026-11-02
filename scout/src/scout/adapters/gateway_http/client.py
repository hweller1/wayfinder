"""`HttpGateway`: the client `Gateway` port over the gateway HTTP API (design section 6).

One `httpx.AsyncClient` per running event loop (the SDK runs `asyncio.run` per call, so a client must not
outlive its loop). Every request carries `Authorization: Bearer`, `X-Scout-Schema: 1`, `X-Scout-Client` and a
fresh `X-Request-Id`; `embed`, `sync_commit` and `seed` also carry an `Idempotency-Key` that is generated once
per call and reused by every retry of it. Failures map to `ScoutError` from the JSON error body (design 4);
a body that is not ours (a proxy page) maps from the HTTP status. 429 (honoring `Retry-After`), 503, 504 and
transport errors retry, 3 attempts in all. `/v1/chat` answers with SSE: `: keepalive` comments are ignored,
one `data:` event carries the ChatCompletion (or an error body), `data: [DONE]` ends the stream. Request and
response bodies are never logged.
"""

import asyncio
import base64
import contextlib
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import asdict
from typing import Any

import httpx
from pydantic import BaseModel

from scout.core.contracts.errors import ErrorCode, ScoutError, is_retryable
from scout.core.contracts.types import IndexStatus
from scout.core.contracts.wire import (
    ChatReq,
    CommitReq,
    CommitRes,
    EmbedReq,
    EmbedRes,
    NodesReq,
    NodesRes,
    PlanReq,
    PlanRes,
    SearchReq,
    SearchRes,
    SeedReq,
    SeedRes,
    StageNodesReq,
    StageNodesRes,
    StatusReq,
    SyncBeginReq,
    SyncBeginRes,
    TenantInfo,
)
from scout.core.ports.chat import ChatDelta, ChatRequest, ChatTurn, ToolCall

__all__ = ["HttpGateway"]

SCHEMA_HEADER = "1"
ATTEMPTS = 3
MAX_RETRY_WAIT_S = 30.0
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)
CHAT_TIMEOUT = httpx.Timeout(900.0, connect=15.0, read=120.0)  # keepalives arrive every ~10 s

type Sleep = Callable[[float], Awaitable[None]]

_STATUS_CODE: dict[int, ErrorCode] = {
    400: ErrorCode.INVALID_ARGUMENT,
    401: ErrorCode.UNAUTHENTICATED,
    403: ErrorCode.PERMISSION_DENIED,
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    422: ErrorCode.UNSUPPORTED_SCHEMA,
    429: ErrorCode.RATE_LIMITED,
    503: ErrorCode.UPSTREAM_UNAVAILABLE,
    504: ErrorCode.UPSTREAM_TIMEOUT,
}


class HttpGateway:
    def __init__(
        self,
        base_url: str,
        key: str | Callable[[], str | None],
        version: str = "0",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._key = key  # a callable is read per call, so a `login` after construction is picked up
        self._version = version
        self._transport = transport
        self._sleep: Sleep = sleep or asyncio.sleep
        self._timeout = timeout or DEFAULT_TIMEOUT
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- plumbing ------------------------------------------------------------------------------
    def _http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                transport=self._transport,
                timeout=self._timeout,
                follow_redirects=True,  # Modal answers a long cold start with a 303 to a result URL
            )
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and not client.is_closed:
            await client.aclose()

    def _bearer(self) -> str:
        key = self._key() if callable(self._key) else self._key
        if not key:
            raise ScoutError(ErrorCode.LOGIN_REQUIRED, "no scout key is configured: run `scout login`")
        return key

    def _headers(self, request_id: str, idempotency_key: str | None) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self._bearer()}",
            "X-Scout-Schema": SCHEMA_HEADER,
            "X-Scout-Client": f"scout/{self._version}",
            "X-Request-Id": request_id,
        }
        if idempotency_key is not None:
            h["Idempotency-Key"] = idempotency_key
        return h

    def _anon_headers(self, request_id: str) -> dict[str, str]:
        return {
            "X-Scout-Schema": SCHEMA_HEADER,
            "X-Scout-Client": f"scout/{self._version}",
            "X-Request-Id": request_id,
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: BaseModel | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
        authed: bool = True,
    ) -> httpx.Response:
        """One logical call: up to `ATTEMPTS` tries, the same request id and idempotency key on each."""
        request_id = uuid.uuid4().hex
        headers = self._headers(request_id, idempotency_key) if authed else self._anon_headers(request_id)
        payload = None if body is None else _dump(body)
        last: ScoutError | None = None
        for attempt in range(ATTEMPTS):
            try:
                res = await self._http().request(method, path, headers=headers, json=payload, params=params)
            except httpx.TimeoutException:
                last = ScoutError(ErrorCode.UPSTREAM_TIMEOUT, "the gateway timed out")
            except httpx.HTTPError:
                last = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the gateway is unreachable")
            else:
                if res.status_code < 400:
                    return res
                last = _error_from(res)
                if res.status_code not in (429, 503, 504):
                    raise last
            if attempt + 1 < ATTEMPTS:
                await self._sleep(_wait(last, attempt))
        assert last is not None
        raise last

    async def _json[T: BaseModel](self, model: type[T], method: str, path: str, **kw: Any) -> T:
        res = await self._request(method, path, **kw)
        try:
            return model.model_validate(res.json())
        except ValueError:
            raise ScoutError(
                ErrorCode.UPSTREAM_UNAVAILABLE, "the gateway returned a malformed response"
            ) from None

    # --- Gateway port --------------------------------------------------------------------------
    async def whoami(self) -> TenantInfo:
        return await self._json(TenantInfo, "GET", "/v1/whoami")

    async def ping(self) -> None:
        await self._request("GET", "/v1/healthz", authed=False)
        await self.whoami()

    async def warm(self) -> bool:
        """`POST /v1/warm`: the gateway starts the chat model's cold start and answers at once. False when
        this gateway predates the route (404) or the upstream has no warm endpoint."""
        try:
            res = await self._request("POST", "/v1/warm")
        except ScoutError as e:
            if e.code is ErrorCode.NOT_FOUND:
                return False
            raise
        try:
            return bool(res.json().get("warming"))
        except (ValueError, AttributeError):
            return False

    async def team_key(self) -> bytes:
        res = await self._request("POST", "/v1/keys/team")
        try:
            raw = base64.b64decode(res.json()["k_team_b64"], validate=True)
        except (ValueError, KeyError, TypeError):
            raise ScoutError(
                ErrorCode.UPSTREAM_UNAVAILABLE, "the gateway returned a malformed team key"
            ) from None
        if len(raw) != 64:
            raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the gateway returned a malformed team key")
        return raw

    async def sync_begin(self, req: SyncBeginReq) -> SyncBeginRes:
        return await self._json(SyncBeginRes, "POST", "/v1/sync/begin", body=req)

    async def sync_nodes(self, req: NodesReq) -> NodesRes:
        return await self._json(NodesRes, "POST", "/v1/sync/nodes", body=req)

    async def sync_plan(self, req: PlanReq) -> PlanRes:
        return await self._json(PlanRes, "POST", "/v1/sync/plan", body=req)

    async def embed(self, req: EmbedReq, *, idempotency_key: str | None = None) -> EmbedRes:
        key = idempotency_key or str(uuid.uuid4())
        return await self._json(EmbedRes, "POST", "/v1/embed", body=req, idempotency_key=key)

    async def sync_stage_nodes(self, req: StageNodesReq) -> StageNodesRes:
        return await self._json(StageNodesRes, "POST", "/v1/sync/stage", body=req)

    async def sync_commit(self, req: CommitReq) -> CommitRes:
        return await self._json(
            CommitRes, "POST", "/v1/sync/commit", body=req, idempotency_key=str(uuid.uuid4())
        )

    async def search(self, req: SearchReq) -> SearchRes:
        return await self._json(SearchRes, "POST", "/v1/search", body=req)

    async def seed(self, req: SeedReq) -> SeedRes:
        return await self._json(SeedRes, "POST", "/v1/seed", body=req, idempotency_key=str(uuid.uuid4()))

    async def index_status(self, req: StatusReq) -> IndexStatus:
        return await self._json(
            IndexStatus,
            "GET",
            "/v1/index/status",
            params={"repo_id": req.repo_id, "branch_id": req.branch_id},
        )

    async def delete_branch(self, repo_id: str, branch_id: str) -> None:
        await self._request("DELETE", f"/v1/namespaces/{repo_id}/{branch_id}")

    # --- chat (SSE) ----------------------------------------------------------------------------
    async def complete(self, req: ChatRequest) -> ChatTurn:
        body = ChatReq.model_validate(asdict(req))
        request_id = uuid.uuid4().hex
        headers = {**self._headers(request_id, None), "Accept": "text/event-stream"}
        last: ScoutError | None = None
        for attempt in range(ATTEMPTS):
            try:
                async with self._http().stream(
                    "POST", "/v1/chat", headers=headers, json=_dump(body), timeout=CHAT_TIMEOUT
                ) as res:
                    if res.status_code >= 400:
                        await res.aread()
                        last = _error_from(res)
                        if res.status_code not in (429, 503, 504):
                            raise last
                    else:
                        return await _read_sse(res)
            except httpx.TimeoutException:
                last = ScoutError(ErrorCode.UPSTREAM_TIMEOUT, "the gateway timed out")
            except httpx.HTTPError:
                last = ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the gateway is unreachable")
            if attempt + 1 < ATTEMPTS:
                await self._sleep(_wait(last, attempt))
        assert last is not None
        raise last

    def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        raise NotImplementedError("HttpGateway does not stream; /v1/chat pseudo-streams one completion")


def _dump(body: BaseModel | Mapping[str, Any]) -> Any:
    return body.model_dump(mode="json") if isinstance(body, BaseModel) else dict(body)


def _error_from(res: httpx.Response) -> ScoutError:
    """The typed error of a failed response: our JSON body when present, else mapped from the status."""
    details: dict[str, Any] = {}
    try:
        err = res.json()["error"]
        code = ErrorCode(err["code"])
        message = str(err["message"])
        raw = err.get("details")
        details = dict(raw) if isinstance(raw, dict) else {}  # pyright: ignore[reportUnknownArgumentType]
        retryable = bool(err.get("retryable", is_retryable(code)))
    except (ValueError, KeyError, TypeError, AttributeError):
        code = _STATUS_CODE.get(
            res.status_code, ErrorCode.UPSTREAM_UNAVAILABLE if res.status_code >= 500 else ErrorCode.INTERNAL
        )
        message = f"the gateway answered HTTP {res.status_code}"
        retryable = is_retryable(code)
    with contextlib.suppress(KeyError, ValueError):
        details.setdefault("retry_after", float(res.headers["retry-after"]))
    return ScoutError(code, message, retryable=retryable, details=details)


def _wait(err: ScoutError, attempt: int) -> float:
    """Seconds before the next try: the server's `Retry-After` for 429, else 0.5 * 2**attempt."""
    hinted = err.details.get("retry_after")
    if err.code is ErrorCode.RATE_LIMITED and isinstance(hinted, int | float):
        return min(max(float(hinted), 0.0), MAX_RETRY_WAIT_S)
    return min(0.5 * 2**attempt, MAX_RETRY_WAIT_S)


async def _read_sse(res: httpx.Response) -> ChatTurn:
    """Skip comments (`: keepalive`) and blanks; the first `data:` event is the turn; `[DONE]` ends it."""
    turn: ChatTurn | None = None
    async for line in res.aiter_lines():
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        if turn is None:
            turn = _turn_from(data)
    if turn is None:
        raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the gateway ended the chat stream without a result")
    return turn


def _turn_from(data: str) -> ChatTurn:
    try:
        obj = json.loads(data)
    except ValueError:
        raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the gateway sent a malformed chat event") from None
    if isinstance(obj, dict) and "error" in obj:
        # An error after the 200 was committed (the upstream failed while keepalives ran).
        fake = httpx.Response(500, json=obj)
        raise _error_from(fake)
    try:
        choice = obj["choices"][0]
        msg = choice["message"]
        calls = tuple(_tool_call(i, c) for i, c in enumerate(msg.get("tool_calls") or ()))
        usage = obj.get("usage") or {}
        content = msg.get("content")
        reasoning = msg.get("reasoning_content")
        return ChatTurn(
            content=content if isinstance(content, str) else None,
            tool_calls=calls,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            finish=str(choice.get("finish_reason") or ("tool_calls" if calls else "stop")),
            reasoning=reasoning if isinstance(reasoning, str) and reasoning else None,
        )
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the gateway sent a malformed chat result") from None


def _tool_call(index: int, raw: Mapping[str, Any]) -> ToolCall:
    fn = raw["function"]
    call_id = str(raw.get("id") or f"call_{index}")
    name = str(fn["name"])
    args = fn.get("arguments")
    if isinstance(args, dict):
        return ToolCall(id=call_id, name=name, arguments=dict(args))  # pyright: ignore[reportUnknownArgumentType]
    if args is None or (isinstance(args, str) and not args.strip()):
        return ToolCall(id=call_id, name=name, arguments={})
    try:
        parsed = json.loads(args)
    except (TypeError, ValueError) as e:
        return ToolCall(id=call_id, name=name, arguments={}, parse_error=f"{type(e).__name__}: {e}"[:200])
    if not isinstance(parsed, dict):
        return ToolCall(id=call_id, name=name, arguments={}, parse_error="arguments must be a JSON object")
    return ToolCall(id=call_id, name=name, arguments=dict(parsed))  # pyright: ignore[reportUnknownArgumentType]
