"""MCP stdio server for the demo: `explore(brief, repo)`, `index_repo(source)`, `list_repos()`.

`mcp` is imported only here (the one demo file allowed to, like `surfaces/mcp`). The server makes no network
call at startup; adapters are built on the first tool call. stdout carries only protocol frames, so every
progress line goes to stderr.
"""

import json
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from scout.core.contracts.errors import ScoutError
from scout.core.explore.render import render_text
from scout.demo.runtime import DemoRuntime, resolve_model


def build_server(rt: DemoRuntime) -> MCPServer:
    for noisy in ("scout.service.request", "httpx", "httpx2", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    server = MCPServer(
        "scout-demo",
        instructions=(
            "Scout code search over repos indexed with index_repo. Call list_repos first to see names. "
            "explore(brief, repo) runs a search subagent (grep, glob, read_file and semantic search) and "
            "returns ranked path:start-end lines, a summary and gaps. Prefer it over reading many files."
        ),
    )

    @server.tool()
    async def explore(brief: str, repo: str, max_results: int = 12) -> str:
        """Search an indexed repo for what the brief asks about. Returns a ranked list of
        `path:start-end  why` lines (1-based inclusive line ranges inside the repo checkout), a short summary
        and any gaps. Write the brief like a task for a colleague: what you need to find and why. `repo` is a
        name from list_repos."""
        try:
            cfg = resolve_model(rt.env)
            rep = await rt.explore(brief, repo, cfg, max_results=max_results, warmup=False)
        except ScoutError as e:
            return f"error: {e.code}: {e.message}"
        text = render_text(rep.result, 1500)
        u = rep.result.usage
        root = rt.registry.get(rep.repo)
        where = f"\n(paths are relative to {root.root})" if root else ""
        return f"{text}\n-- turns={u.turns} wall={rep.wall_s:.1f}s model={rep.model}{where}"

    @server.tool()
    async def index_repo(source: str, name: str | None = None) -> str:
        """Index a git URL (shallow clone) or a local checkout path so it can be explored. Embeds with
        voyage-code-4 and writes to turbopuffer; small repos take under a minute and cost cents."""
        try:
            rep = await rt.index(source, name)
        except ScoutError as e:
            return f"error: {e.code}: {e.message}"
        return json.dumps(rep.to_dict(), indent=2)

    @server.tool()
    def list_repos() -> list[dict[str, Any]]:
        """The repos indexed so far: name, checkout path, index state and coverage."""
        return [
            {
                "name": r.name,
                "root": r.root,
                "namespace": r.namespace,
                "state": r.state,
                "coverage": r.coverage,
                "files": r.files,
                "chunks": r.chunks,
            }
            for r in rt.list_repos()
        ]

    return server


def serve(rt: DemoRuntime) -> None:
    build_server(rt).run("stdio")
