"""Gateway HTTP surface (design sections 4 and 6): a thin FastAPI app over `GatewayService`.

`create_gateway_app(service, auth)` serves every route of the design-6 table. It decides nothing about the
product: it authenticates the bearer key (`AuthService`, `/v1/healthz` exempt), derives the tenant from the
stored record, checks `X-Scout-Schema`, parses the body into the `core.contracts.wire` models, calls the
tenant-bound service, and maps `ScoutError` to the status and JSON body of the design-4 table. Request bodies
are parsed by hand (not bound by FastAPI) so the order is fixed: 401 before 400, and no validation message
ever echoes a value (only field paths and error types). Nothing here logs a body, a path or a key.

Idempotency (FR-G10): `embed`, `sync/commit` and `seed` require an `Idempotency-Key` (1..128 chars, else 400).
A repeat of the same key and body within 24 h returns the stored success response without calling the
service; the same key with another body is 400. The store is the `IdempotencyCache` Protocol (in-memory here,
a Modal Dict in `deploy/modal`); it is a cache only, row ids and the root CAS keep a lost entry correct.

Rate limits are the service's (`core.service.ratelimit`): a `RATE_LIMITED` error leaves as 429 with
`Retry-After` and `RateLimit-Limit`/`RateLimit-Remaining`/`RateLimit-Reset`.

`/v1/chat` pseudo-streams (AC-9.5): the upstream call is non-streaming; while it runs the response is a 200
SSE stream of `: keepalive` comments every `keepalive_s` (`SCOUT_CHAT_KEEPALIVE_S`, default 10), then one
`data: <ChatCompletion JSON>` and `data: [DONE]`. Failures known before the 200 (auth, a bad body, the
2-per-key cap) are ordinary error responses; a failure after it is a `data:` event carrying the error body.
"""

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from scout.core.contracts.errors import ErrorCode, ScoutError, http_status
from scout.core.contracts.wire import (
    ChatReq,
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
from scout.core.ports.chat import ChatRequest, ChatTurn
from scout.core.ports.clock import Clock
from scout.core.service.auth import AuthService
from scout.core.service.gateway import GatewayService, TenantGateway
from scout.core.service.tenants import Tenant

__all__ = ["IdempotencyCache", "MemoryIdempotencyCache", "create_gateway_app"]

LOGGER = logging.getLogger("scout.gateway_http")
SCHEMA_HEADER = "1"
IDEMPOTENCY_TTL_S = 24 * 3600.0
MAX_IDEMPOTENCY_KEY_LEN = 128
MAX_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_KEEPALIVE_S = 10.0
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PUBLIC_VALIDATION = "request body failed validation"


class IdempotencyCache(Protocol):
    """Stored responses by key (a Modal Dict plugs in here). A cache only: never authoritative."""

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_s: float) -> None: ...


class MemoryIdempotencyCache:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._items: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> str | None:
        item = self._items.get(key)
        if item is None or item[0] <= self._clock.now():
            return None
        return item[1]

    async def set(self, key: str, value: str, ttl_s: float) -> None:
        now = self._clock.now()
        self._items = {k: v for k, v in self._items.items() if v[0] > now}
        self._items[key] = (now + ttl_s, value)


@dataclass(frozen=True, slots=True)
class _Call:
    """Everything a route handler needs after authentication."""

    gw: TenantGateway
    tenant: Tenant
    request: Request
    raw: bytes
    idempotency_key: str | None


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if rid is None:
        supplied = request.headers.get("x-request-id", "")
        rid = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        request.state.request_id = rid
    return rid


def _error_response(request: Request, err: ScoutError, status: int | None = None) -> JSONResponse:
    rid = _request_id(request)
    headers = {"X-Request-Id": rid}
    if err.code is ErrorCode.RATE_LIMITED:
        raw = err.details.get("retry_after")
        retry = float(raw) if isinstance(raw, int | float) else 1.0
        limit = err.details.get("limit")
        headers["Retry-After"] = str(max(1, math.ceil(retry)))
        if isinstance(limit, int | float):
            headers["RateLimit-Limit"] = str(int(limit))
        headers["RateLimit-Remaining"] = "0"
        headers["RateLimit-Reset"] = str(max(1, math.ceil(retry)))
    body = err.to_body(rid).model_dump(mode="json")
    return JSONResponse(body, status_code=status or http_status(err.code), headers=headers)


def _validation_error(exc: ValidationError) -> ScoutError:
    found = exc.errors(include_input=False, include_url=False, include_context=False)
    where = ", ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['type']}" for e in found[:5])
    return ScoutError(ErrorCode.INVALID_ARGUMENT, f"{_PUBLIC_VALIDATION} ({where})")


def completion_json(turn: ChatTurn) -> dict[str, Any]:
    """An OpenAI ChatCompletion body for one assistant turn. A tool call whose arguments did not parse is sent
    with an unparseable argument string, so the client flags it as a parse error as well."""
    message: dict[str, Any] = {"role": "assistant", "content": turn.content}
    if turn.reasoning:
        message["reasoning_content"] = turn.reasoning
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": "{unparseable" if c.parse_error else json.dumps(c.arguments),
                },
            }
            for c in turn.tool_calls
        ]
    return {
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": turn.finish}],
        "usage": {
            "prompt_tokens": turn.tokens_in,
            "completion_tokens": turn.tokens_out,
            "total_tokens": turn.tokens_in + turn.tokens_out,
        },
    }


def _chat_request(req: ChatReq) -> ChatRequest:
    return ChatRequest(
        messages=req.messages,
        tools=req.tools,
        tool_choice=req.tool_choice,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        presence_penalty=req.presence_penalty,
        max_tokens=req.max_tokens,
        thinking=req.thinking,
    )


def _sse(event: str) -> bytes:
    return f"data: {event}\n\n".encode()


def create_gateway_app(
    service: GatewayService,
    auth: AuthService,
    *,
    cache: IdempotencyCache | None = None,
    keepalive_s: float | None = None,
    version: str = "0",
) -> FastAPI:
    store: IdempotencyCache = cache if cache is not None else MemoryIdempotencyCache(service.clock)
    keepalive = (
        keepalive_s
        if keepalive_s is not None
        else float(os.environ.get("SCOUT_CHAT_KEEPALIVE_S") or DEFAULT_KEEPALIVE_S)
    )
    app = FastAPI(title="scout-gateway", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request: Request, exc: StarletteHTTPException) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_ARGUMENT
        message = "no such route" if exc.status_code == 404 else "method not allowed"
        return _error_response(request, ScoutError(code, message), exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        LOGGER.error("unhandled %s", type(exc).__name__)  # the class only: a message may quote a body
        return _error_response(request, ScoutError(ErrorCode.INTERNAL, "internal error"))

    async def authenticate(request: Request) -> Tenant:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        tenant = await auth.resolve(token.strip() if scheme.lower() == "bearer" else None)
        schema = request.headers.get("x-scout-schema")
        if schema is not None and schema.strip() != SCHEMA_HEADER:
            raise ScoutError(
                ErrorCode.UNSUPPORTED_SCHEMA, f"unsupported X-Scout-Schema (supported: {SCHEMA_HEADER})"
            )
        return tenant

    async def read_body(request: Request) -> bytes:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "request body is too large")
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "request body is too large")
        return raw

    def idempotency_key(request: Request) -> str:
        key = request.headers.get("idempotency-key", "")
        if not 0 < len(key) <= MAX_IDEMPOTENCY_KEY_LEN:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "Idempotency-Key is required (1..128 characters)")
        return key

    def parse[M: BaseModel](model: type[M], raw: bytes) -> M:
        try:
            return model.model_validate_json(raw)
        except ValidationError as exc:
            raise _validation_error(exc) from None

    def respond(request: Request, result: BaseModel | None, status: int = 200) -> Response:
        if result is None:
            return Response(status_code=204, headers={"X-Request-Id": _request_id(request)})
        return JSONResponse(
            result.model_dump(mode="json"), status_code=status, headers={"X-Request-Id": _request_id(request)}
        )

    def route(
        path: str,
        method: str,
        model: type[BaseModel] | None,
        op: Callable[[_Call, Any], Awaitable[BaseModel | None]],
        *,
        idempotent: bool = False,
    ) -> None:
        async def handler(request: Request) -> Response:
            try:
                tenant = await authenticate(request)
                raw = await read_body(request) if method != "GET" else b""
                body = parse(model, raw) if model is not None else None
                idem = idempotency_key(request) if idempotent else None
                call = _Call(service.for_tenant(tenant), tenant, request, raw, idem)
                if idem is None:
                    return respond(request, await op(call, body))
                return await replayed(call, path, lambda: op(call, body))
            except ScoutError as err:
                return _error_response(request, err)

        app.add_api_route(path, handler, methods=[method], response_model=None)

    async def replayed(call: _Call, path: str, run: Callable[[], Awaitable[BaseModel | None]]) -> Response:
        assert call.idempotency_key is not None
        digest = hashlib.sha256(call.raw).hexdigest()
        slot = hashlib.sha256(f"{call.tenant.tenant}\0{path}\0{call.idempotency_key}".encode()).hexdigest()
        try:
            stored = await store.get(slot)
        except Exception:  # noqa: BLE001
            stored = None  # a cache failure is a miss
        if stored is not None:
            record = json.loads(stored)
            if record["digest"] != digest:
                raise ScoutError(
                    ErrorCode.INVALID_ARGUMENT, "Idempotency-Key was used with a different request"
                )
            return JSONResponse(
                record["body"],
                status_code=record["status"],
                headers={"X-Request-Id": _request_id(call.request), "Idempotent-Replayed": "true"},
            )
        result = await run()
        assert result is not None
        body = result.model_dump(mode="json")
        try:
            await store.set(
                slot, json.dumps({"digest": digest, "status": 200, "body": body}), IDEMPOTENCY_TTL_S
            )
        except Exception:  # noqa: BLE001
            LOGGER.warning("idempotency cache write failed")
        return JSONResponse(body, headers={"X-Request-Id": _request_id(call.request)})

    # --- routes (design section 6) ---------------------------------------------------------------
    @app.get("/v1/healthz")
    async def healthz(request: Request) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        return JSONResponse(
            {"ok": True, "version": version, "schema": SCHEMA_HEADER},
            headers={"X-Request-Id": _request_id(request)},
        )

    async def whoami(c: _Call, _: Any) -> BaseModel | None:
        return await c.gw.whoami()

    async def team_key(c: _Call, _: Any) -> BaseModel | None:
        raw = await c.gw.team_key()
        return _B64Key(k_team_b64=base64.b64encode(raw).decode())

    async def status(c: _Call, _: Any) -> BaseModel | None:
        q = c.request.query_params
        req = parse(
            StatusReq,
            json.dumps({"repo_id": q.get("repo_id", ""), "branch_id": q.get("branch_id", "")}).encode(),
        )
        return await c.gw.index_status(req)

    async def delete_ns(c: _Call, _: Any) -> BaseModel | None:
        params = c.request.path_params
        await c.gw.delete_branch(str(params["repo_id"]), str(params["branch_id"]))
        return None

    async def warm(c: _Call, _: Any) -> BaseModel | None:
        return WarmRes(warming=await c.gw.warm())

    route("/v1/warm", "POST", None, warm)
    route("/v1/whoami", "GET", None, whoami)
    route("/v1/keys/team", "POST", None, team_key)
    route("/v1/sync/begin", "POST", SyncBeginReq, lambda c, b: c.gw.sync_begin(_need(b)))
    route("/v1/sync/nodes", "POST", NodesReq, lambda c, b: c.gw.sync_nodes(_need(b)))
    route("/v1/sync/plan", "POST", PlanReq, lambda c, b: c.gw.sync_plan(_need(b)))
    route("/v1/sync/stage", "POST", StageNodesReq, lambda c, b: c.gw.sync_stage_nodes(_need(b)))
    route(
        "/v1/embed",
        "POST",
        EmbedReq,
        lambda c, b: c.gw.embed(_need(b), idempotency_key=c.idempotency_key),
        idempotent=True,
    )
    route("/v1/sync/commit", "POST", CommitReq, lambda c, b: c.gw.sync_commit(_need(b)), idempotent=True)
    route("/v1/index/status", "GET", None, status)
    route("/v1/search", "POST", SearchReq, lambda c, b: c.gw.search(_need(b)))
    route("/v1/seed", "POST", SeedReq, lambda c, b: c.gw.seed(_need(b)), idempotent=True)

    async def delete_handler(request: Request) -> Response:
        try:
            tenant = await authenticate(request)
            call = _Call(service.for_tenant(tenant), tenant, request, b"", None)
            await delete_ns(call, None)
            return respond(request, None)
        except ScoutError as err:
            return _error_response(request, err)

    app.add_api_route("/v1/namespaces/{repo_id}/{branch_id}", delete_handler, methods=["DELETE"])

    @app.post("/v1/chat")
    async def chat(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        try:
            tenant = await authenticate(request)
            req = parse(ChatReq, await read_body(request))
        except ScoutError as err:
            return _error_response(request, err)
        gw = service.for_tenant(tenant)
        task = asyncio.ensure_future(gw.complete(_chat_request(req)))
        await asyncio.sleep(0)  # the validation and the 2-per-key cap run before the first suspension
        if task.done() and task.exception() is not None:
            exc = task.exception()
            return _error_response(
                request,
                exc if isinstance(exc, ScoutError) else ScoutError(ErrorCode.INTERNAL, "internal error"),
            )
        rid = _request_id(request)

        async def events() -> AsyncIterator[bytes]:
            while not task.done():
                await asyncio.wait({task}, timeout=keepalive)
                if not task.done():
                    yield b": keepalive\n\n"
            try:
                yield _sse(json.dumps(completion_json(task.result())))
            except ScoutError as err:
                yield _sse(json.dumps(err.to_body(rid).model_dump(mode="json")))
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("chat failed %s", type(exc).__name__)
                internal = ScoutError(ErrorCode.INTERNAL, "internal error")
                yield _sse(json.dumps(internal.to_body(rid).model_dump(mode="json")))
            yield b"data: [DONE]\n\n"

        return _CancelOnExit(
            task,
            events(),
            media_type="text/event-stream",
            headers={"X-Request-Id": rid, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


class _CancelOnExit(StreamingResponse):
    """A stream that owns the upstream chat task: however the response ends (finished, client gone, server
    shutdown) the task is cancelled if still running, which frees the per-key chat slot."""

    def __init__(self, task: "asyncio.Future[Any]", content: AsyncIterator[bytes], **kw: Any) -> None:
        super().__init__(content, **kw)
        self._task = task

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if not self._task.done():
                self._task.cancel()


class _B64Key(BaseModel):
    k_team_b64: str


class WarmRes(BaseModel):
    warming: bool


def _need[M: BaseModel](body: M | None) -> M:
    assert body is not None
    return body
