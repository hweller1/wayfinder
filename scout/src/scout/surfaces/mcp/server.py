"""MCP stdio surface (design sections 4 and 11, PRD 2.1 both contract levels, FR-S3, AC-6.4).

`build_mcp` exposes the delegated level (`explore`) and the primitive level (`semantic_search`, `grep`,
`read_range`, `index_status`) of the one `ScoutApp` facade as MCP tools; `tools=` narrows the set (a Claude
Code shim wants only `explore`). This is the only module of the product that names the `mcp` package.

- Output is ONE plain text block (`structured_output=False`: the SDK's default wraps a returned string as
  `{"result": ...}`, which a host model then reads as JSON noise). For `explore` it is `render_text(result,
  min(requested, ceiling))`; `structured=True` (conformance, tooling) returns the JSON of the result instead.
- Errors are `CallToolResult(isError=True)` with `scout error <CODE>: <message>`; JSON-RPC errors stay
  reserved for malformed protocol. A tool result never carries a stack trace.
- The app is built on the FIRST tool call (`app` may be a factory), so `initialize` costs only the SDK import
  and the server makes no network call at startup. `mcp.pid` (the readiness signal host hooks read, design 11)
  is written into the repository state dir at startup and removed at exit.
- stdout carries protocol frames only: progress and noisy library loggers go to stderr or are muted.
"""

import atexit
import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.surfaces import ops

if TYPE_CHECKING:
    from scout.core.app import ScoutApp

PID_FILE = "mcp.pid"
LOOPBACK = "127.0.0.1"
DEFAULT_RENDER_MAX_TOKENS = 1500
_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
_INSTRUCTIONS = (
    "Scout searches the code of the current repository. `explore(brief)` runs a search subagent over the "
    "repository and returns ranked `path:start-end` ranges (1-based, inclusive) with a short summary and "
    "gaps: prefer it to reading many files. `semantic_search`, `grep`, `read_range` and `index_status` "
    "are the primitives it is built from."
)

type AppSource = "ScoutApp | Callable[[], ScoutApp]"


def error_result(err: ScoutError) -> CallToolResult:
    text = f"scout error {err.code}: {err.message}{ops.error_hint(err)}"
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def select_tools(names: Iterable[str] | None) -> tuple[str, ...]:
    """`--tools a,b` -> validated op names in a stable order; None means all five."""
    if names is None:
        return ops.OPS
    wanted = {n.strip() for chunk in names for n in chunk.split(",") if n.strip()}
    unknown = sorted(wanted - set(ops.OPS))
    if unknown or not wanted:
        raise ScoutError(
            ErrorCode.INVALID_ARGUMENT,
            f"unknown tool {', '.join(unknown) or '(none given)'}: choose from {', '.join(ops.OPS)}",
        )
    return tuple(op for op in ops.OPS if op in wanted)


DEMO_TOOLS: tuple[str, ...] = ("explore", "index_repo", "list_repos")


def select_demo_tools(names: Iterable[str] | None) -> tuple[str, ...]:
    """`--tools` for profile demo, validated against the demo server's own tools (`explore`, `index_repo`,
    `list_repos`); the primitives of the product server do not exist there."""
    if names is None:
        return DEMO_TOOLS
    wanted = {n.strip() for chunk in names for n in chunk.split(",") if n.strip()}
    unknown = sorted(wanted - set(DEMO_TOOLS))
    if unknown or not wanted:
        bad = ", ".join(unknown) or "(none given)"
        raise ScoutError(
            ErrorCode.INVALID_ARGUMENT,
            f"profile demo has no tool {bad}: choose from {', '.join(DEMO_TOOLS)}",
        )
    return tuple(t for t in DEMO_TOOLS if t in wanted)


def restrict_demo_tools(server: MCPServer, keep: Iterable[str]) -> MCPServer:
    """Drop the demo tools not in `keep` (the demo server registers its three itself)."""
    wanted = set(keep)
    for name in DEMO_TOOLS:
        if name not in wanted:
            server.remove_tool(name)
    return server


def build_mcp(
    app: "ScoutApp | Callable[[], ScoutApp]",
    tools: Iterable[str] | None = None,
    render_ceiling: int | None = None,
    *,
    structured: bool = False,
    default_render_tokens: int = DEFAULT_RENDER_MAX_TOKENS,
) -> MCPServer:
    for noisy in ("scout.service.request", "httpx", "httpx2", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    enabled = select_tools(tools)
    holder: list[Any] = [] if callable(app) else [app]

    def get_app() -> "ScoutApp":
        if not holder:
            holder.append(app())  # type: ignore[operator]
        return holder[0]  # type: ignore[no-any-return]

    async def call(op: str, payload: dict[str, Any]) -> str | CallToolResult:
        try:
            req = ops.parse_request(op, {k: v for k, v in payload.items() if v is not None})
            result = await getattr(get_app(), op)(req)
            if structured:
                return ops.result_json(result)
            return ops.render(op, req, result, ceiling=render_ceiling)
        except ScoutError as e:
            return error_result(e)
        except Exception as e:  # noqa: BLE001 - a tool result never carries a stack trace or a path
            return error_result(ScoutError(ErrorCode.INTERNAL, f"internal error: {type(e).__name__}"))

    server = MCPServer("scout", instructions=_INSTRUCTIONS)
    reg = server.tool(structured_output=False, annotations=_READ_ONLY)

    async def explore(
        brief: str,
        paths: list[str] | None = None,
        max_results: int = 12,
        render_max_tokens: int | None = None,
        max_turns: int | None = None,
        max_seconds: int | None = None,
    ) -> str | CallToolResult:
        """Search the repository for what the brief asks about and return the best places to read. The result
        is a short summary, then one `path:start-end  why` line per hit (best first, 1-based inclusive line
        ranges relative to the repository root), then any gaps. Write the brief like a task for a colleague:
        what you need to find and why. `paths` limits the search to those globs."""
        budget = {"max_turns": max_turns, "max_seconds": max_seconds}
        return await call(
            "explore",
            {
                "brief": brief,
                "paths": paths,
                "max_results": max_results,
                "render_max_tokens": render_max_tokens or default_render_tokens,
                "budget": {k: v for k, v in budget.items() if v is not None} or None,
            },
        )

    async def semantic_search(query: str, path_glob: str | None = None, k: int = 10) -> str | CallToolResult:
        """Embedding search over the repository index: ranked code snippets (current file contents, line
        numbered). Needs a ready index; otherwise it says so and `grep` is the fallback."""
        return await call("semantic_search", {"query": query, "path_glob": path_glob, "k": k})

    async def grep(
        pattern: str, path: str | None = None, regex: bool = True, context: int = 0
    ) -> str | CallToolResult:
        """ripgrep over the checkout (respects .gitignore and .scoutignore). Returns `path:line: text`
        lines. `path` limits it to a file or directory; `context` adds 0-5 surrounding lines."""
        return await call("grep", {"pattern": pattern, "path": path, "regex": regex, "context": context})

    async def read_range(path: str, start_line: int = 1, end_line: int | None = None) -> str | CallToolResult:
        """Read a line range (1-based, inclusive, at most 400 lines) of a file in the repository."""
        return await call("read_range", {"path": path, "start_line": start_line, "end_line": end_line})

    async def index_status() -> str | CallToolResult:
        """State of the repository index: ready, partial or absent, coverage, and whether semantic search
        is available."""
        return await call("index_status", {})

    handlers: dict[str, Callable[..., Awaitable[str | CallToolResult]]] = {
        "explore": explore,
        "semantic_search": semantic_search,
        "grep": grep,
        "read_range": read_range,
        "index_status": index_status,
    }
    for op in enabled:
        reg(handlers[op])
    return server


# --- readiness file -------------------------------------------------------------------------------
def write_pid(state_dir: str | os.PathLike[str], *, pid: int | None = None) -> Path:
    """`<state_dir>/mcp.pid` = `{"pid", "started_at"}`, tmp file + rename so a hook never reads half of it."""
    directory = Path(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PID_FILE
    body = {"pid": os.getpid() if pid is None else pid, "started_at": datetime.now(UTC).isoformat()}
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".mcp.pid.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(body, f)
    os.replace(tmp, path)
    return path


def remove_pid(path: Path, *, pid: int | None = None) -> None:
    """Remove the file only if it is still ours (a newer server may have replaced it)."""
    with contextlib.suppress(OSError, ValueError):
        if json.loads(path.read_text(encoding="utf-8")).get("pid") == (os.getpid() if pid is None else pid):
            path.unlink()


def serve(server: MCPServer, *, http_port: int | None = None, state_dir: str | None = None) -> None:
    """Run until the client disconnects. `http_port`: streamable HTTP on loopback instead of stdio."""
    if state_dir:
        try:
            pid_path = write_pid(state_dir)
        except OSError:
            pid_path = None  # an unwritable state dir only costs the readiness signal
        if pid_path is not None:
            atexit.register(remove_pid, pid_path)
    if http_port is None:
        server.run("stdio")
    else:
        server.run("streamable-http", host=LOOPBACK, port=http_port)
