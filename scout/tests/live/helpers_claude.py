"""Helpers for the headless Claude Code smoke: run `claude -p` with the plugin, parse stream-json.

Auth: a clean CLAUDE_CONFIG_DIR cannot authenticate without ANTHROPIC_API_KEY (spike 0a-D), so the run uses
the default config dir but isolates it with `--setting-sources project --no-session-persistence`: no user
hooks or settings load and nothing persists. `--strict-mcp-config` is never passed because it blocks the
plugin's own MCP server (the shim would be refused with zero tools).
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "integrations" / "claude-code"
HOOK_LOG = Path.home() / ".claude" / "plugins" / "data" / "scout-inline" / "hook.log"
ARTIFACT = REPO / "artifacts" / "live" / "claude_smoke.json"
MCP_TOOL = "mcp__plugin_scout_scout__explore"
ZERO_TOOLS = "would be spawned with zero tools"


def claude_available() -> bool:
    return shutil.which("claude") is not None


def hook_log_size() -> int:
    return HOOK_LOG.stat().st_size if HOOK_LOG.exists() else 0


def hook_log_since(offset: int) -> list[str]:
    if not HOOK_LOG.exists():
        return []
    with HOOK_LOG.open("rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", "replace").splitlines()


def _flatten(content: object) -> str:
    """tool_result content is a string or a list of text blocks; the JSON-wrapped MCP text is unwrapped."""
    if isinstance(content, list):
        text = "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    else:
        text = str(content)
    try:
        obj = json.loads(text)
    except ValueError:
        return text
    return obj["result"] if isinstance(obj, dict) and isinstance(obj.get("result"), str) else text


@dataclass
class Run:
    events: list[dict] = field(default_factory=list)
    raw: str = ""
    returncode: int = 0

    @property
    def result(self) -> dict:
        return next((e for e in reversed(self.events) if e.get("type") == "result"), {})

    def tool_uses(self) -> list[dict]:
        out: list[dict] = []
        for e in self.events:
            if e.get("type") == "assistant":
                out += [c for c in e["message"]["content"] if c.get("type") == "tool_use"]
        return out

    def tool_results_text(self) -> str:
        parts: list[str] = []
        for e in self.events:
            content = e.get("message", {}).get("content") if e.get("type") == "user" else None
            if isinstance(content, list):
                parts += [_flatten(c.get("content")) for c in content if c.get("type") == "tool_result"]
        return "\n".join(parts)

    def agent_types(self) -> list[str]:
        """subagent_type of every Agent/Task call the main agent made (before any hook rewrite)."""
        return [
            u["input"].get("subagent_type", "") for u in self.tool_uses() if u["name"] in ("Agent", "Task")
        ]

    def mcp_calls(self) -> list[dict]:
        return [u for u in self.tool_uses() if u["name"] == MCP_TOOL]

    def subagent_by_type(self) -> dict[str, int]:
        return self.result.get("subagent_stats", {}).get("by_type", {})


def run_claude(
    prompt: str, cwd: Path, *, timeout: int, budget: float = 0.5, extra_env: dict[str, str] | None = None
) -> Run:
    cmd = [
        "claude",
        "--plugin-dir",
        str(PLUGIN),
        "-p",
        prompt,
        "--model",
        "haiku",
        "--output-format",
        "stream-json",
        "--verbose",
        "--setting-sources",
        "project",
        "--no-session-persistence",
        "--max-budget-usd",
        str(budget),
    ]
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CONFIG_DIR"}
    env.update(extra_env or {})
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    return Run(events=events, raw=proc.stdout, returncode=proc.returncode)
