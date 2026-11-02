"""Conformance drivers (design section 9): one driver per transport over the same fake system.

A driver builds an app for one case (`{"repo": <dir>, "chat_script": [...], ...}`) and answers the contract
requests with contract results. `SdkDriver` is the in-process fake; `SdkOverHttpDriver` is the same SDK with
the gateway behind the real FastAPI app, reached by `HttpGateway` over an ASGI transport (task 8.3). Tasks
6.6 and 6.7 add `CliDriver`, `McpStdioDriver`, `McpHttpDriver` and `DaemonDriver` (second half of this
file): they spawn `scout` with `SCOUT_PROFILE=fake SCOUT_FAKE_CASE=<file>`, so the case syncs inside the
process and every transport answers from an identical system. A typed error raises `ScoutError(code)` from
every driver; the subprocess drivers also assert their transport's own error mapping (CLI exit code and
stderr line, HTTP status and `Retry-After`, MCP `isError` text) while decoding, and keep `raw`, the success
body exactly as delivered.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Protocol

import httpx
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel

from scout.adapters.fake_chat import ScriptedChat
from scout.core.app import ScoutApp
from scout.core.contracts.errors import ErrorBody, ErrorCode, ScoutError, cli_exit, http_status
from scout.core.contracts.types import (
    ExploreRequest,
    ExploreResult,
    GrepRequest,
    GrepResult,
    IndexStatus,
    IndexStatusRequest,
    ReadRangeRequest,
    ReadRangeResult,
    SemanticSearchRequest,
    SemanticSearchResult,
)
from scout.wiring import Config, _turn, build_app  # pyright: ignore[reportPrivateUsage]


class Driver(Protocol):
    name: str

    async def setup(self, case: Mapping[str, Any], tmp: Path) -> None: ...

    async def sync(self) -> IndexStatus: ...

    async def explore(self, req: ExploreRequest) -> ExploreResult: ...

    async def grep(self, req: GrepRequest) -> GrepResult: ...

    async def read_range(self, req: ReadRangeRequest) -> ReadRangeResult: ...

    async def index_status(self, req: IndexStatusRequest) -> IndexStatus: ...

    async def semantic_search(self, req: SemanticSearchRequest) -> SemanticSearchResult: ...


class _AppDriver:
    name = "app"
    app: ScoutApp
    raw: str | None = None  # in-process: there is no transport body

    async def close(self) -> None:
        return None

    async def sync(self) -> IndexStatus:
        return await self.app.sync()

    async def explore(self, req: ExploreRequest) -> ExploreResult:
        return await self.app.explore(req)

    async def grep(self, req: GrepRequest) -> GrepResult:
        return await self.app.grep(req)

    async def read_range(self, req: ReadRangeRequest) -> ReadRangeResult:
        return await self.app.read_range(req)

    async def index_status(self, req: IndexStatusRequest) -> IndexStatus:
        return await self.app.index_status(req)

    async def semantic_search(self, req: SemanticSearchRequest) -> SemanticSearchResult:
        return await self.app.semantic_search(req)


class SdkDriver(_AppDriver):
    name = "sdk"

    async def setup(self, case: Mapping[str, Any], tmp: Path) -> None:
        self.app = build_app(Config(profile="fake", case={**case, "state_dir": str(tmp / "sdk-state")}))


class SdkOverHttpDriver(_AppDriver):
    name = "sdk-over-http"

    async def setup(self, case: Mapping[str, Any], tmp: Path) -> None:
        from gateway_app_helpers import make_rig  # pyright: ignore[reportMissingImports]  # tests/contract: sys.path

        turns = [_turn(t) for t in case.get("chat_script", ())]
        self.rig = make_rig(chat=ScriptedChat(turns), keepalive_s=0.05)
        key = await self.rig.key("dev")
        cfg = Config(
            profile="fake-http",
            case={**case, "state_dir": str(tmp / "http-state")},
            transport=self.rig.transport(),
            key=key,
            gateway_url="http://gateway.test",
        )
        self.app = build_app(cfg)


# --- per-transport drivers (tasks 6.6, 6.7) ---------------------------------------------------------

STARTUP_TIMEOUT_S = 30.0
CONFORMANCE_TOKEN = "conformance-token"  # noqa: S105
OPS = ("explore", "semantic_search", "grep", "read_range", "index_status")
REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "explore": ExploreRequest,
    "semantic_search": SemanticSearchRequest,
    "grep": GrepRequest,
    "read_range": ReadRangeRequest,
    "index_status": IndexStatusRequest,
}
RESULT_MODELS: dict[str, type[BaseModel]] = {
    "explore": ExploreResult,
    "semantic_search": SemanticSearchResult,
    "grep": GrepResult,
    "read_range": ReadRangeResult,
    "index_status": IndexStatus,
}


class TransportDriver(Protocol):
    name: str
    raw: str | None  # the last success body as delivered (None for the in-process SDK)

    async def setup(self, case: Mapping[str, Any], tmp: Path) -> None: ...

    async def explore(self, req: ExploreRequest) -> ExploreResult: ...

    async def grep(self, req: GrepRequest) -> GrepResult: ...

    async def read_range(self, req: ReadRangeRequest) -> ReadRangeResult: ...

    async def index_status(self, req: IndexStatusRequest) -> IndexStatus: ...

    async def semantic_search(self, req: SemanticSearchRequest) -> SemanticSearchResult: ...

    async def close(self) -> None: ...


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def fields(req: BaseModel) -> dict[str, Any]:
    """The request fields a caller set (defaults left to the contract), as JSON-able values."""
    return req.model_dump(mode="json", exclude_none=True, exclude_defaults=True)


def parse_error_text(text: str) -> ScoutError:
    """`scout error <CODE>: <message>` (the MCP error text, design section 4)."""
    head, _, message = text.partition(": ")
    parts = head.split()
    assert parts[:2] == ["scout", "error"], f"not a scout error text: {text!r}"
    return ScoutError(ErrorCode(parts[2]), message)


def _write_case(case: Mapping[str, Any], tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "case.json"
    path.write_text(json.dumps({**case, "state_dir": str(tmp / "state")}), encoding="utf-8")
    return path


class _SubprocessDriver:
    """Shared plumbing: the case file, the five typed operations over `_call(op, req) -> str`."""

    name = "subprocess"

    def __init__(self) -> None:
        self.raw: str | None = None
        self._env: dict[str, str] = {}
        self._stack = AsyncExitStack()

    async def setup(self, case: Mapping[str, Any], tmp: Path) -> None:
        path = await asyncio.to_thread(_write_case, case, tmp)
        self._env = {**os.environ, "SCOUT_PROFILE": "fake", "SCOUT_FAKE_CASE": str(path)}
        self._tmp = tmp
        await self.start()

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        await self._stack.aclose()

    async def _call(self, op: str, req: BaseModel) -> str:
        raise NotImplementedError

    async def _typed(self, op: str, req: BaseModel) -> Any:
        self.raw = await self._call(op, req)
        return RESULT_MODELS[op].model_validate_json(self.raw)

    async def explore(self, req: ExploreRequest) -> ExploreResult:
        return await self._typed("explore", req)  # type: ignore[no-any-return]

    async def grep(self, req: GrepRequest) -> GrepResult:
        return await self._typed("grep", req)  # type: ignore[no-any-return]

    async def read_range(self, req: ReadRangeRequest) -> ReadRangeResult:
        return await self._typed("read_range", req)  # type: ignore[no-any-return]

    async def index_status(self, req: IndexStatusRequest) -> IndexStatus:
        return await self._typed("index_status", req)  # type: ignore[no-any-return]

    async def semantic_search(self, req: SemanticSearchRequest) -> SemanticSearchResult:
        return await self._typed("semantic_search", req)  # type: ignore[no-any-return]


def cli_argv(op: str, req: BaseModel) -> list[str]:
    """The `scout` command line of a request: every request field the cases use has a flag."""
    f = fields(req)
    if op == "explore":
        argv = ["explore", f["brief"]]
        if f.get("paths"):
            argv += ["--paths", *f["paths"]]
        for flag, key in (("--max-results", "max_results"), ("--render-max-tokens", "render_max_tokens")):
            if key in f:
                argv += [flag, str(f[key])]
        for flag, key in (("--max-turns", "max_turns"), ("--max-seconds", "max_seconds")):
            if key in f.get("budget", {}):
                argv += [flag, str(f["budget"][key])]
        return argv
    if op == "semantic_search":
        argv = ["search", f["query"]]
        if "path_glob" in f:
            argv += ["--path-glob", f["path_glob"]]
        if "k" in f:
            argv += ["-k", str(f["k"])]
        return argv
    if op == "grep":
        argv = ["grep", f["pattern"]]
        if "path" in f:
            argv += ["--path", f["path"]]
        if f.get("regex") is False:
            argv.append("--literal")
        if "context" in f:
            argv += ["--context", str(f["context"])]
        return argv
    if op == "read_range":
        end = [str(f["end_line"])] if "end_line" in f else []
        return ["read", f["path"], str(f.get("start_line", 1)), *end]
    return ["status"]


class CliDriver(_SubprocessDriver):
    name = "cli"

    async def _call(self, op: str, req: BaseModel) -> str:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "scout", *cli_argv(op, req),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=self._env,
        )  # fmt: skip
        out, err = await proc.communicate()
        stdout, stderr = out.decode(), err.decode()
        if proc.returncode == 0:
            assert stderr == "", f"CLI wrote to stderr on success: {stderr!r}"
            return stdout.strip()
        error = ErrorBody.model_validate_json(stdout).error  # the error body is on stdout
        assert proc.returncode == cli_exit(error.code), f"exit {proc.returncode} for {error.code}"
        assert stderr.count("\n") == 1, f"expected one human line on stderr, got {stderr!r}"
        assert error.code in stderr
        raise ScoutError(error.code, error.message)


class _Server:
    """A `scout` subprocess serving on a loopback port; stopped on close."""

    def __init__(self, argv: list[str], env: dict[str, str], port: int) -> None:
        self.port = port
        self.proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "scout", *argv], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    async def wait_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                tail = self.proc.stderr.read().decode()[-400:] if self.proc.stderr else ""
                raise AssertionError(f"server exited early: {tail}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                await asyncio.sleep(0.05)
        raise AssertionError("server did not open its port")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        if self.proc.stderr:
            self.proc.stderr.close()


class _McpDriver(_SubprocessDriver):
    structured = True

    def __init__(self, *, structured: bool = True) -> None:
        super().__init__()
        self.structured = structured
        self._client: Client | None = None

    async def connect(self) -> Client:
        raise NotImplementedError

    async def start(self) -> None:
        await self._stack.__aenter__()
        self._client = await asyncio.wait_for(self.connect(), STARTUP_TIMEOUT_S)

    async def tool_names(self) -> list[str]:
        assert self._client is not None
        return [t.name for t in (await self._client.list_tools()).tools]

    async def _call(self, op: str, req: BaseModel) -> str:
        assert self._client is not None
        arguments = fields(req)
        if budget := arguments.pop("budget", None):
            arguments.update(budget)  # the MCP tool takes the budget as flat max_turns / max_seconds
        result = await self._client.call_tool(op, arguments)
        assert len(result.content) == 1, f"expected one content block, got {len(result.content)}"
        text = result.content[0].text  # type: ignore[union-attr]
        if result.is_error:
            raise parse_error_text(text)
        assert result.structured_content is None, "the plain-text surface must not publish structuredContent"
        return text  # type: ignore[no-any-return]

    async def _typed(self, op: str, req: BaseModel) -> Any:
        assert self.structured, "typed results need `mcp --structured`; use `text()` for the default output"
        return await super()._typed(op, req)

    async def text(self, op: str, req: BaseModel) -> str:
        """The plain-text content of a call (the default, non-structured output)."""
        self.raw = await self._call(op, req)
        return self.raw


class McpStdioDriver(_McpDriver):
    name = "mcp-stdio"

    async def connect(self) -> Client:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "scout", "mcp", *(["--structured"] if self.structured else [])],
            env=self._env,
        )
        return await self._stack.enter_async_context(Client(params))


class McpHttpDriver(_McpDriver):
    name = "mcp-http"

    async def connect(self) -> Client:
        port = free_port()
        argv = ["mcp", "--http", str(port), *(["--structured"] if self.structured else [])]
        server = _Server(argv, self._env, port)
        self._stack.callback(server.stop)
        await server.wait_ready()
        url = f"http://127.0.0.1:{port}/mcp"
        return await self._stack.enter_async_context(Client(streamable_http_client(url)))


DAEMON_ROUTES = {
    "explore": "/explore",
    "semantic_search": "/search",
    "grep": "/grep",
    "read_range": "/read",
    "index_status": "/status",
}


def _write_token(tmp: Path) -> Path:
    token_file = tmp / "daemon.token"
    token_file.write_text(CONFORMANCE_TOKEN + "\n")
    token_file.chmod(0o600)
    return token_file


class DaemonDriver(_SubprocessDriver):
    name = "daemon"

    def __init__(self) -> None:
        super().__init__()
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        await self._stack.__aenter__()
        token_file = await asyncio.to_thread(_write_token, self._tmp)
        port = free_port()
        server = _Server(["serve", "--port", str(port), "--token-file", str(token_file)], self._env, port)
        self._stack.callback(server.stop)
        await server.wait_ready()
        self._http = await self._stack.enter_async_context(
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                headers={"Authorization": f"Bearer {CONFORMANCE_TOKEN}"},
                timeout=60,
            )
        )

    async def _call(self, op: str, req: BaseModel) -> str:
        assert self._http is not None
        r = await self._http.post(DAEMON_ROUTES[op], json=fields(req))
        if r.status_code == 200:
            return r.text
        error = ErrorBody.model_validate_json(r.text).error
        assert r.status_code == http_status(error.code), f"HTTP {r.status_code} for {error.code}"
        assert ("retry-after" in r.headers) == (error.code is ErrorCode.RATE_LIMITED)
        raise ScoutError(error.code, error.message)


TRANSPORTS: dict[str, Any] = {
    "sdk": SdkDriver,
    "cli": CliDriver,
    "mcp-stdio": McpStdioDriver,
    "mcp-http": McpHttpDriver,
    "daemon": DaemonDriver,
}
