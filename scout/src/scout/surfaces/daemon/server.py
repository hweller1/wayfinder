"""Local HTTP daemon surface (design sections 4 and 14, FR-S5): the five operations of the one `ScoutApp`
facade as `POST /explore /search /grep /read /status`, for hosts that cannot spawn a process per call.

- Loopback only: `check_loopback` refuses any other bind address, and every route needs
  `Authorization: Bearer <token>` where the token lives in a 0600 file (`load_or_create_token`). A 401 never
  reveals whether the header was missing or wrong.
- Bodies are the contract request types (extras forbidden); responses are `ops.result_json` of the result,
  byte for byte what the CLI prints, so conformance compares them as strings. Errors are the section 4 body
  with the HTTP status of the table (`Retry-After` on 429).
"""

import asyncio
import hmac
import ipaddress
import json
import os
import secrets
import stat
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, Response

from scout.core.contracts.errors import ErrorCode, ScoutError, http_status
from scout.surfaces import ops

if TYPE_CHECKING:
    from scout.core.app import ScoutApp

LOOPBACK_HOSTS = ("localhost",)
MAX_BODY_BYTES = 1_000_000
# route -> operation (the `ScoutApp` method, also the `ops` name)
ROUTES: dict[str, str] = {
    "/explore": "explore",
    "/search": "semantic_search",
    "/grep": "grep",
    "/read": "read_range",
    "/status": "index_status",
}


def check_loopback(host: str) -> str:
    """The host if it is a loopback address; anything else is `INVALID_ARGUMENT` (the daemon serves one
    user's checkout and is never reachable from another machine)."""
    if host in LOOPBACK_HOSTS:
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"refusing to bind {host!r}: the daemon is loopback only")


def default_token_path(env: dict[str, str] | os._Environ[str] | None = None) -> Path:
    src = os.environ if env is None else env
    return Path(src.get("SCOUT_DAEMON_TOKEN_FILE") or Path.home() / ".config" / "scout" / "daemon.token")


def load_or_create_token(path: Path) -> str:
    """The bearer token in `path`, created (32 random bytes, hex) with mode 0600 when absent. A token file
    that anyone but the owner can read is refused: it would not be a secret."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ScoutError(
                ErrorCode.PERMISSION_DENIED, f"token file {path} must be mode 0600 (is {mode:04o})"
            ) from None
        token = path.read_text(encoding="utf-8").strip()
        if not token:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"token file {path} is empty") from None
        return token
    token = secrets.token_hex(32)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    return token


def _error(err: ScoutError) -> Response:
    headers: dict[str, str] = {}
    retry_after = err.details.get("retry_after")
    if err.code is ErrorCode.RATE_LIMITED and isinstance(retry_after, int | float):
        headers["Retry-After"] = str(max(1, int(retry_after)))
    return Response(
        err.to_body().model_dump_json(),
        status_code=http_status(err.code),
        media_type="application/json",
        headers=headers,
    )


def create_daemon(app: "ScoutApp | Callable[[], ScoutApp]", token: str) -> FastAPI:
    """The ASGI app. `app` may be a factory: it is called on the first request, so the process starts without
    touching the gateway."""
    if not token:
        raise ValueError("the daemon needs a bearer token")
    expected = token.encode("utf-8")
    holder: list[Any] = [] if callable(app) else [app]
    lock = asyncio.Lock()
    daemon = FastAPI(title="scout daemon", docs_url=None, redoc_url=None, openapi_url=None)

    def authorized(request: Request) -> bool:
        header = request.headers.get("authorization", "")
        scheme, _, presented = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(presented.strip().encode("utf-8"), expected)

    async def get_app() -> "ScoutApp":
        async with lock:
            if not holder:
                holder.append(app())  # type: ignore[operator]
        return holder[0]  # type: ignore[no-any-return]

    async def handle(op: str, request: Request) -> Response:
        if not authorized(request):
            return _error(ScoutError(ErrorCode.UNAUTHENTICATED, "missing or invalid bearer token"))
        try:
            raw = await request.body()
            if len(raw) > MAX_BODY_BYTES:
                raise ScoutError(ErrorCode.INVALID_ARGUMENT, "request body is too large")
            try:
                payload = json.loads(raw) if raw.strip() else {}
            except ValueError:
                raise ScoutError(ErrorCode.INVALID_ARGUMENT, "request body is not JSON") from None
            if not isinstance(payload, dict):
                raise ScoutError(ErrorCode.INVALID_ARGUMENT, "request body must be a JSON object")
            req = ops.parse_request(op, payload)
            result = await getattr(await get_app(), op)(req)
            return Response(ops.result_json(result), media_type="application/json")
        except ScoutError as e:
            return _error(e)
        except Exception as e:  # noqa: BLE001 - a response never carries a trace or a path
            return _error(ScoutError(ErrorCode.INTERNAL, f"internal error: {type(e).__name__}"))

    def register(path: str, op: str) -> None:
        async def route(request: Request) -> Response:
            return await handle(op, request)

        daemon.add_api_route(path, route, methods=["POST"], response_class=Response)

    for path, op in ROUTES.items():
        register(path, op)
    return daemon


def serve(daemon: FastAPI, host: str, port: int) -> None:
    """Run uvicorn on loopback until interrupted."""
    import uvicorn

    uvicorn.run(daemon, host=check_loopback(host), port=port, log_level="warning", access_log=False)
