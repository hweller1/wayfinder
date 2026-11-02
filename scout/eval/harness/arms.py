"""Arm runners: one headless `claude -p` session per (task, arm), parsed from stream-json.

All arms: the same parent model and settings, an isolated `CLAUDE_CONFIG_DIR`, the same fixed parent prompt.
The parent may only call `Agent` (a PreToolUse gate hook denies its other tools; the subagent keeps its own),
so the only retrieval it receives is the Explore handback. `explore_default` loads no plugin. The two scout
arms load the scout plugin (`integrations/claude-code`), whose PreToolUse hook rewrites Explore when scout is
ready: the replace arms pin `SCOUT_EXPLORE_MODE=fast` (the Haiku shim, as in the original run), `hybrid_v3`
pins `hybrid` (the Sonnet scout-explorer: scout explore first, then verified reads).
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from eval.harness.briefs import brief_sha, extract_paths

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "integrations" / "claude-code"
WORK = Path("/tmp/scout-eval")
LLM_GATEWAY_BASE = os.environ.get("LLM_GATEWAY_URL", "").rstrip("/") + "/anthropic"  # your LLM API gateway (Anthropic route)
PARENT_MODEL = "claude-sonnet-5-5"
HAIKU_MODEL = "claude-haiku-5-5"  # what the shim's `model: haiku` resolves to (ANTHROPIC_DEFAULT_HAIKU_MODEL)
SHIM_TYPE = "scout:super-explore"
SHIM_TYPES = ("super-explore:super-explore", SHIM_TYPE)  # plugin name before and after the 0.2.0 rename
HYBRID_TYPE = "scout:scout-explorer"
PLUGIN_DATA = "scout-inline"  # ${CLAUDE_PLUGIN_DATA} of a --plugin-dir plugin named `scout`
PLUGIN_TOOL_PREFIXES = ("mcp__plugin_super-explore_scout__", "mcp__plugin_scout_scout__")
# USD per 1M tokens: input, cache read, cache write (5 min), output. Sonnet from the benchmark's RATES table.
RATES = {"claude-sonnet-5-5": (2.00, 0.20, 2.50, 10.00)}
ARMS = ("explore_default", "super_v3", "super_a1")
AUGMENT = "augment_v3"  # built-in Explore plus scout's MCP tools in its toolbox; nothing rewritten or blocked
A20 = "super_grpo_a20"  # the replace setup of `super_v3`, chat model = Wayfinder GRPO arm a step-20 checkpoint (Addendum B)
HYBRID = "hybrid_v3"  # the plugin's hybrid scout-explorer (Sonnet, scout explore + Read/Grep verification), v3 gateway
HYBRID_A1 = "hybrid_a1"  # the same subagent on the A1 gateway (Addendum C deviation: v3 serving was down)
HYBRID_BF = "hybrid_v3_bf"  # hybrid_v3 with the in-code backfill to 12 files (Addendum E)
RELAY_BF = "super_v3_bf"  # super_v3 (pure Haiku relay) with the in-code backfill (Addendum E)
HYBRIDS = (HYBRID, HYBRID_A1, HYBRID_BF)
CC_EXPLORE, CC_SCOUT, CC_SCOUT_ONLY = "cc_explore", "cc_scout", "cc_scout_only"  # free-parent arms (practical prototype)
CC_EXPLORE_SPEC, CC_SCOUT_SPEC = "cc_explore_spec", "cc_scout_spec"  # retrieval-spec arms (docs/retrieval-spec.md)
SPEC_ARMS = (CC_EXPLORE_SPEC, CC_SCOUT_SPEC)
CC_ARMS = (CC_EXPLORE, CC_SCOUT, CC_SCOUT_ONLY, *SPEC_ARMS)
NO_SCOUT_ARMS = (CC_EXPLORE, CC_EXPLORE_SPEC)  # no scout key, gateway or sync
_PLUGIN_LOCK = threading.Lock()
TRANSCRIPTS: Path | None = None  # set by run_cc --transcripts: dir for raw stream-json per run
SPEC_PLUGIN = "spec"  # the spec-only plugin of cc_explore_spec: hooks only, no MCP server, no agents
SPEC_PLUGIN_DATA = "spec-inline"
ALL_ARMS = (*ARMS, AUGMENT, A20, HYBRID, HYBRID_A1, HYBRID_BF, RELAY_BF, *CC_ARMS)
AUGMENT_TOOLS = "explore,semantic_search"
SCOUT_TOOL_PREFIX = "mcp__scout__"  # server name `scout` in the augment arm's --mcp-config

PROMPT = (
    "You are a coordinator with exactly one tool, Agent. Call Agent exactly once with subagent_type "
    '"Explore", run_in_background false, and the prompt below copied verbatim between the markers. '
    "When it returns, reply with the subagent's complete response verbatim and nothing else. "
    "Do not search the repository yourself.\n\n<<<BRIEF\n{brief}\nBRIEF>>>"
)

CC_PROMPT = (
    "Find the files in this repository that are relevant to the task below, and report them. Use whatever tools "
    "you have.\n\n<<<TASK\n{brief}\nTASK>>>"
)

GATE_HOOK = """#!/usr/bin/env python3
import json, sys
p = json.load(sys.stdin)
if not p.get("agent_id") and p.get("tool_name") != "Agent":
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": "The coordinator may only call Agent."}}))
"""

WATCH_STUB = """#!/bin/sh
# eval: the SessionStart hook must not start a background watcher (the harness syncs explicitly).
if [ "$1" = "watch" ]; then exit 0; fi
exec "{scout}" "$@"
"""


def gateway_key() -> str:
    for f in (REPO / ".env",):
        try:
            for line in f.read_text().splitlines():
                line = line.strip().removeprefix("export ")
                if line.startswith("LLM_GATEWAY_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
        except OSError:
            continue
    return os.environ.get("LLM_GATEWAY_API_KEY", "")


def build_spec_only_plugin() -> Path:
    """`cc_explore_spec`: the spec hooks and nothing else (no scout MCP server, no scout agents, no Explore
    rewrite). Copies the plugin's own spec files, so the two spec arms run the same hook code."""
    dst = WORK / "spec-only-plugin"
    with _PLUGIN_LOCK:
        if (dst / ".built").exists():  # built once per process: concurrent sessions load this directory
            return dst
        return _build_spec_only_plugin(dst)


def _build_spec_only_plugin(dst: Path) -> Path:
    shutil.rmtree(dst, ignore_errors=True)
    for sub in ("hooks", "lib", "spec", ".claude-plugin"):
        (dst / sub).mkdir(parents=True)
    shutil.copy(PLUGIN / "hooks" / "spec_hooks.py", dst / "hooks")
    shutil.copy(PLUGIN / "lib" / "scout_spec.py", dst / "lib")
    shutil.copy(PLUGIN / "lib" / "scout_cc.py", dst / "lib")  # spec_hooks imports it for the tool prefix only
    shutil.copy(PLUGIN / "spec" / "harness_defaults.toml", dst / "spec")
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    keep = {"SessionStart": [], "UserPromptSubmit": [], "PreToolUse": []}
    for event, groups in hooks.items():
        for g in groups:
            hs = [h for h in g["hooks"] if "spec_hooks.py" in h["command"]]
            if hs:
                keep[event].append({**g, "hooks": hs})
    (dst / "hooks" / "hooks.json").write_text(json.dumps({"hooks": keep}, indent=2))
    (dst / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": SPEC_PLUGIN, "version": "0.0.0", "description": "retrieval spec hooks only (eval arm)"})
    )
    (dst / ".built").write_text("1")
    return dst


GUIDANCE = ("When you report, list only files you have evidence for (you read them, or a search returned them), "
            "ranked most relevant first; aim for the 5 to 8 most relevant files and leave out the rest.")
# Hill-climb variants (wayfinder docs/spec_hillclimb_plan.md): name -> (base arm, spec-file overrides by table).
# Every override is a documented option of the retrieval spec (docs/retrieval-spec.md), not an eval hook.
VARIANTS: dict[str, tuple[str, dict[str, dict]]] = {
    "base": (CC_SCOUT_SPEC, {}),
    "h_verbatim": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim"}}),
    "h_direct": (CC_SCOUT_SPEC, {"delivery": {"handoff": "direct"}}),
    "g_agent": (CC_SCOUT_SPEC, {"delivery": {"report_guidance": GUIDANCE}}),
    "g_verbatim": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim", "report_guidance": GUIDANCE}}),
    "g_direct": (CC_SCOUT_SPEC, {"delivery": {"handoff": "direct", "report_guidance": GUIDANCE}}),
    "v_warn": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim", "search_policy": "warn"}}),
    "v_off": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim", "search_policy": "off"}}),
    "v_mine0": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim"}, "examples": {"mine": 0}}),
    "v_mine6": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim"}, "examples": {"mine": 6}}),
    "v_sim3": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim"}, "examples": {"select": "similar", "mine": 3}}),
    "v_sym": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim"}, "index": {"symbols": True}}),
    "v_t6": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim"}, "explore": {"max_turns": 6}}),
    "v_t8": (CC_SCOUT_SPEC, {"delivery": {"handoff": "verbatim"}, "explore": {"max_turns": 8}}),
    "x_base": (CC_EXPLORE_SPEC, {}),
    "x_guid": (CC_EXPLORE_SPEC, {"delivery": {"report_guidance": GUIDANCE}}),
}


def _toml_val(v) -> str:
    return json.dumps(v) if not isinstance(v, bool) else ("true" if v else "false")


def write_spec_file(cfg: Path, task: dict, *, search_first: bool = True, overrides: dict[str, dict] | None = None) -> Path:
    """The per-task user spec of the spec arms (layer 3), written OUTSIDE the checkout and pointed at by
    $SCOUT_SPEC_FILE. Leak guard: examples are mined strictly before `base_commit` and never touch a gold file.
    `overrides` (a variant's spec options, by table) replace the defaults below."""
    cfg.mkdir(parents=True, exist_ok=True)
    tables: dict[str, dict] = {
        "examples": {"mine": 3, "before": task["base_commit"], "avoid": list(task["gold_files"])},
        "delivery": {"search_first": search_first, "workflow_step": True, "session_context": True},
    }
    for t, kv in (overrides or {}).items():
        tables.setdefault(t, {}).update(kv)
    if "search_policy" in tables["delivery"]:
        tables["delivery"].pop("search_first", None)
    top = tables.pop("", {})
    text = "".join(f"{k} = {_toml_val(v)}\n" for k, v in top.items())
    for t, kv in tables.items():
        text += f"\n[{t}]\n" + "".join(f"{k} = {_toml_val(v)}\n" for k, v in kv.items())
    path = cfg / "spec.toml"
    path.write_text(text)
    return path


def prewarm_spec(root: Path, spec_file: Path) -> None:
    """Resolve the task's spec once before the session so the SessionStart hook reads a warm per-commit cache."""
    subprocess.run(  # noqa: S603
        ["python3", str(PLUGIN / "lib" / "scout_spec.py"), "resolve", "--root", str(root), "--spec", str(spec_file), "--no-write"],  # noqa: S607
        capture_output=True, timeout=300, check=False, env={**os.environ, "SCOUT_SPEC_FILE": str(spec_file)},
    )


def ensure_support_files() -> tuple[Path, Path, Path]:
    """`(gate_settings.json, wrapper_scout, empty_demo_home)` under WORK, created once."""
    WORK.mkdir(parents=True, exist_ok=True)
    gate = WORK / "gate.py"
    gate.write_text(GATE_HOOK)
    settings = WORK / "gate_settings.json"
    settings.write_text(
        json.dumps({"hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"type": "command", "command": f"python3 {gate}"}]}]}})
    )
    mcp = WORK / "augment_mcp.json"  # no secret inside: Claude Code expands ${VAR} from the session env
    mcp.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "scout": {
                        "command": str(REPO / ".venv" / "bin" / "scout"),
                        "args": ["mcp", "--tools", AUGMENT_TOOLS],
                        "env": {"SCOUT_KEY": "${SCOUT_KEY}", "SCOUT_GATEWAY_URL": "${SCOUT_GATEWAY_URL}", "SCOUT_KEYSTORE": "file"},
                    }
                }
            }
        )
    )
    wrap = WORK / "scout"
    wrap.write_text(WATCH_STUB.format(scout=REPO / ".venv" / "bin" / "scout"))
    wrap.chmod(0o755)
    demo = WORK / "empty-demo-home"
    demo.mkdir(exist_ok=True)
    return settings, wrap, demo


def build_env(arm: str, config_dir: Path, *, key: str | None, gateway: str | None, llm_key: str) -> dict[str, str]:
    _, wrap, demo = ensure_support_files()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE", "SCOUT_", "GATEWAY_"))}
    env.update(
        CLAUDE_CONFIG_DIR=str(config_dir),
        ANTHROPIC_BASE_URL=LLM_GATEWAY_BASE,
        ANTHROPIC_API_KEY="unused",
        ANTHROPIC_CUSTOM_HEADERS=f"api-key: {llm_key}",
        ANTHROPIC_DEFAULT_HAIKU_MODEL=HAIKU_MODEL,
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        CLAUDE_CODE_DISABLE_BACKGROUND_TASKS="1",
        CLAUDE_CODE_DISABLE_MCP_TASK_BACKGROUND="1",
        DISABLE_TELEMETRY="1",
        DISABLE_AUTOUPDATER="1",
        DISABLE_ERROR_REPORTING="1",
        NO_COLOR="1",
    )
    if arm == AUGMENT:
        assert key and gateway
        env.update(SCOUT_KEY=key, SCOUT_GATEWAY_URL=gateway, SCOUT_KEYSTORE="file")
    elif arm not in ("explore_default", *NO_SCOUT_ARMS):
        assert key and gateway
        env.update(
            SCOUT_KEY=key,
            SCOUT_GATEWAY_URL=gateway,
            SCOUT_BIN=str(wrap),
            SCOUT_KEYSTORE="file",
            SCOUT_DEMO_HOME=str(demo),
            SCOUT_LANE="production",
            SCOUT_EXPLORE_MODE="hybrid" if arm in (*HYBRIDS, CC_SCOUT, CC_SCOUT_ONLY, CC_SCOUT_SPEC) else "fast",
            SCOUT_NO_WARM="1",
            SCOUT_EXPLORE_BACKFILL_TO="12" if arm in (HYBRID_BF, RELAY_BF) else "0",  # the *_bf arms only  # the harness warms the model itself; no per-session warm-up call
        )
    if arm in SPEC_ARMS:
        env["SCOUT_SPEC_FILE"] = str(config_dir / "spec.toml")
        if arm == CC_EXPLORE_SPEC:
            env["SCOUT_SPEC_TARGET"] = "Explore"
    return env


def command(arm: str, brief: str, settings: Path, budget: float) -> list[str]:
    if arm in CC_ARMS:  # free parent: no gate hook, no coordinator prompt; own tools plus Agent
        cmd = [
            "claude", "-p", CC_PROMPT.format(brief=brief), "--model", PARENT_MODEL,
            "--output-format", "stream-json", "--verbose", "--include-hook-events",
            "--setting-sources", "project", "--no-session-persistence", "--max-budget-usd", str(budget),
            "--allowedTools", "Bash", "Read", "Grep", "Glob", "Agent",
            "mcp__plugin_scout_scout__explore", "mcp__plugin_scout_scout__semantic_search", "mcp__plugin_scout_scout__read_range",
            "--disallowedTools", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch",
            "Bash(curl:*)", "Bash(wget:*)", "Bash(git log:*)", "Bash(git show:*)", "Bash(git fetch:*)",
        ]  # fmt: skip
        if arm == CC_SCOUT_ONLY:
            cmd += ["Agent(Explore)"]  # replaces Explore: denied, so the parent must use scout:scout-explorer or search itself
        if arm == CC_EXPLORE_SPEC:
            cmd += ["--plugin-dir", str(build_spec_only_plugin())]
        elif arm != CC_EXPLORE:
            cmd += ["--plugin-dir", str(PLUGIN)]
        return cmd
    cmd = [
        "claude", "-p", PROMPT.format(brief=brief), "--model", PARENT_MODEL,
        "--output-format", "stream-json", "--verbose", "--include-hook-events",
        "--setting-sources", "project", "--settings", str(settings),
        "--no-session-persistence", "--max-budget-usd", str(budget),
        "--allowedTools", "Bash", "Read", "Grep", "Glob",
    ]  # fmt: skip
    if arm == AUGMENT:
        cmd += ["mcp__scout__explore", "mcp__scout__semantic_search"]  # extend --allowedTools; the subagent chooses freely
        cmd += ["--mcp-config", str(WORK / "augment_mcp.json"), "--strict-mcp-config"]
    elif arm != "explore_default":
        cmd += ["--plugin-dir", str(PLUGIN)]
    return cmd


@dataclass
class Parsed:
    handback: str = ""
    final: str = ""
    agent_type: str = ""
    resolved_model: str = ""
    sub_ms: int | None = None
    sub_tokens: int | None = None
    sub_tools: int | None = None
    mcp_calls: int = 0
    agent_calls: int = 0
    subagent_requested: str = ""
    cost_usd: float | None = None
    parent_cost_usd: float | None = None
    subagent_cost_usd: float | None = None
    turns: int | None = None
    duration_ms: int | None = None
    model_usage: dict[str, dict[str, float]] = field(default_factory=dict)
    is_error: bool = False
    denials: list[str] = field(default_factory=list)
    retries: int = 0
    zero_tools: bool = False
    mcp_status: dict[str, str] = field(default_factory=dict)  # init event: server name -> status
    scout_calls: dict[str, int] = field(default_factory=dict)  # augment arm: scout tool -> calls (any agent)
    sub_tool_counts: dict[str, int] = field(default_factory=dict)  # tool name -> calls made inside the subagent
    sub_tool_seq: list[str] = field(default_factory=list)  # subagent tool calls in order (MCP names shortened)
    plugin_tool_counts: dict[str, int] = field(default_factory=dict)  # plugin scout tool -> calls
    parent_agent_types: list[str] = field(default_factory=list)  # subagent_type of each parent Agent call
    parent_tool_counts: dict[str, int] = field(default_factory=dict)  # parent's own tool calls (any name)
    usage_tokens: dict[str, int] = field(default_factory=dict)  # summed billed token fields over all messages


def _text(content: object) -> str:
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return str(content or "")


def _unwrap(text: str) -> str:
    try:
        obj = json.loads(text)
    except ValueError:
        return text
    return obj["result"] if isinstance(obj, dict) and isinstance(obj.get("result"), str) else text


def _est(u: dict[str, int]) -> float:
    r = RATES[PARENT_MODEL]
    cw, cr = u.get("cache_creation_input_tokens", 0), u.get("cache_read_input_tokens", 0)
    return (u.get("input_tokens", 0) * r[0] + cr * r[1] + cw * r[2] + u.get("output_tokens", 0) * r[3]) / 1e6


def split_cost(
    msgs: dict[str, tuple[bool, str, dict[str, int]]], model_usage: dict[str, dict[str, float]]
) -> tuple[float | None, float | None]:
    """`(parent, subagent)` USD. The billed per-model `costUSD` of the result event is authoritative; the
    parent's Sonnet share is split from the subagent's by estimated message cost, and any other model (the
    Haiku shim) is wholly subagent."""
    if not model_usage:
        return None, None
    parent_est = sum(_est(u) for par, m, u in msgs.values() if par and m == PARENT_MODEL)
    sub_est = sum(_est(u) for par, m, u in msgs.values() if not par and m == PARENT_MODEL)
    sonnet = float(model_usage.get(PARENT_MODEL, {}).get("costUSD", 0.0))
    share = parent_est / (parent_est + sub_est) if parent_est + sub_est > 0 else 1.0
    other = sum(float(v.get("costUSD", 0.0)) for k, v in model_usage.items() if k != PARENT_MODEL)
    return round(sonnet * share, 6), round(sonnet * (1 - share) + other, 6)


def parse_stream(raw: str) -> Parsed:
    p = Parsed()
    msgs: dict[str, tuple[bool, str, dict[str, int]]] = {}
    for line in raw.splitlines():
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t = e.get("type")
        if t == "system" and e.get("subtype") == "init":
            p.mcp_status = {str(m.get("name")): str(m.get("status")) for m in e.get("mcp_servers", []) if isinstance(m, dict)}
        if t == "assistant":
            msg = e.get("message", {})
            is_parent = e.get("parent_tool_use_id") is None
            for c in msg.get("content", []):
                if c.get("type") != "tool_use":
                    continue
                if is_parent:
                    p.parent_tool_counts[c["name"]] = p.parent_tool_counts.get(c["name"], 0) + 1
                if c["name"] in ("Agent", "Task") and is_parent:
                    p.agent_calls += 1
                    p.parent_agent_types.append(str(c.get("input", {}).get("subagent_type", "")))
                    p.subagent_requested = str(c.get("input", {}).get("subagent_type", ""))
                if c["name"].startswith(PLUGIN_TOOL_PREFIXES):
                    p.mcp_calls += 1
                    short = c["name"].split("__")[-1]
                    p.plugin_tool_counts[short] = p.plugin_tool_counts.get(short, 0) + 1
                if not is_parent:
                    p.sub_tool_counts[c["name"]] = p.sub_tool_counts.get(c["name"], 0) + 1
                    p.sub_tool_seq.append(c["name"].split("__")[-1] if c["name"].startswith("mcp__") else c["name"])
                if c["name"].startswith(SCOUT_TOOL_PREFIX):
                    short = c["name"].removeprefix(SCOUT_TOOL_PREFIX)
                    p.scout_calls[short] = p.scout_calls.get(short, 0) + 1
            if msg.get("id") and isinstance(msg.get("usage"), dict):
                msgs[msg["id"]] = (is_parent, str(msg.get("model", "")), msg["usage"])
        elif t == "user":
            tur = e.get("tool_use_result")
            if isinstance(tur, dict) and ("agentType" in tur or "totalDurationMs" in tur) and e.get("parent_tool_use_id") is None:
                p.handback = _unwrap(_text(tur.get("content")))
                p.agent_type = str(tur.get("agentType", ""))
                p.resolved_model = str(tur.get("resolvedModel", ""))
                p.sub_ms = tur.get("totalDurationMs")
                p.sub_tokens = tur.get("totalTokens")
                p.sub_tools = tur.get("totalToolUseCount")
            content = e.get("message", {}).get("content")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "tool_result" and "would be spawned with zero tools" in _text(c.get("content")):
                        p.zero_tools = True
        elif t == "system" and e.get("subtype") == "task_started" and e.get("subagent_type"):
            p.agent_type = p.agent_type or str(e.get("subagent_type"))
        elif t == "tool_progress" and e.get("subagent_retry"):
            p.retries += 1
        elif t == "result":
            p.final = str(e.get("result", ""))
            p.cost_usd = e.get("total_cost_usd")
            p.turns = e.get("num_turns")
            p.duration_ms = e.get("duration_ms")
            p.is_error = bool(e.get("is_error"))
            p.denials = [str(d.get("tool_name")) for d in e.get("permission_denials", [])]
            p.model_usage = e.get("modelUsage", {})
            if not p.agent_type and not p.handback:
                pass
    p.parent_cost_usd, p.subagent_cost_usd = split_cost(msgs, p.model_usage)
    for par, _m, u in msgs.values():
        k = "parent" if par else "sub"
        for f in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            p.usage_tokens[f"{k}_{f}"] = p.usage_tokens.get(f"{k}_{f}", 0) + int(u.get(f, 0) or 0)
    return p


def hook_lines_since(config_dir: Path, offset: int) -> tuple[list[str], int]:
    log = config_dir / "plugins" / "data" / PLUGIN_DATA / "hook.log"
    if not log.exists():
        return [], offset
    with log.open("rb") as f:
        f.seek(offset)
        data = f.read()
    return data.decode("utf-8", "replace").splitlines(), offset + len(data)


def spec_layers(config_dir: Path, arm: str) -> dict:
    """Which retrieval-spec layers fired in a session, from the plugin's spec_hooks.jsonl (events only)."""
    name = SPEC_PLUGIN_DATA if arm == CC_EXPLORE_SPEC else PLUGIN_DATA
    log = config_dir / "plugins" / "data" / name / "spec_hooks.jsonl"
    recs = []
    if log.exists():
        for ln in log.read_text().splitlines():
            try:
                recs.append(json.loads(ln))
            except ValueError:
                continue
    ss = [r for r in recs if r.get("hook") == "SessionStart"]
    pre = [r for r in recs if r.get("hook") == "PreToolUse"]
    return {
        "session_context": bool(ss and ss[0].get("injected")),
        "index_description": bool(ss and ss[0].get("has_description")),
        "examples_mined": ss[0].get("n_examples", 0) if ss else 0,
        "workflow_step": any(r.get("hook") == "UserPromptSubmit" and r.get("injected") for r in recs),
        "search_first_denials": sum(1 for r in pre if r.get("denied")),
        "grep_main_calls": sum(1 for r in pre if r.get("event") in ("grep_allowed", "grep_denied")),
        "target": (ss[0].get("target") if ss else None),
        "hook_errors": [r.get("error") for r in recs if r.get("error")][:3],
    }


def hook_log_size(config_dir: Path) -> int:
    log = config_dir / "plugins" / "data" / PLUGIN_DATA / "hook.log"
    return log.stat().st_size if log.exists() else 0


def score_row(
    arm: str, task: dict, brief: str, parsed: Parsed, files: list[str], cwd: str, wall_s: float, hook_lines: list[str]
) -> dict:
    prefixes = {cwd, os.path.realpath(cwd), "/private" + cwd}
    hb_paths = extract_paths(parsed.handback, files, root_prefixes=prefixes)
    final_paths = extract_paths(parsed.final, files, root_prefixes=prefixes)
    gold = set(task["gold_files"])
    rewrote = any(re.search(r"\brewrite lane=production", ln) for ln in hook_lines)
    want = (HYBRID_TYPE,) if arm in (*HYBRIDS, CC_SCOUT, CC_SCOUT_ONLY, CC_SCOUT_SPEC) else SHIM_TYPES
    shim = parsed.agent_type in want or parsed.mcp_calls > 0
    if arm in ("explore_default", AUGMENT, *CC_ARMS):
        fallthrough = False  # cc arms: whether scout was used is a measured outcome (tool-call rate), not a retry trigger
    else:
        fallthrough = not (rewrote and shim)
    if arm in CC_ARMS:  # the parent's reported list is the answer; the subagent handback is kept separately
        row_extra = {"handback_paths": hb_paths}
        hb_paths = final_paths or []
    else:
        row_extra = {}
    top = hb_paths[:12]
    sub_cost = parsed.subagent_cost_usd
    return {
        "task_id": task["native_id"],
        "repo": task["repo"],
        "lang": task["lang"],
        "arm": arm,
        "brief_sha": brief_sha(brief),
        "wall_s": round(wall_s, 2),
        "explore_s": None if parsed.sub_ms is None else round(parsed.sub_ms / 1000, 2),
        "handback_chars": len(parsed.handback),
        "handback_unverified": parsed.handback.count("(unverified)"),
        "paths": hb_paths,
        "final_paths": final_paths,
        "n_gold": len(gold),
        "recall12": len(gold & set(top)) / len(gold),
        "all_gold": gold <= set(top),
        "recall_any": len(gold & set(hb_paths)) / len(gold),
        "final_recall12": len(gold & set(final_paths[:12])) / len(gold),
        "cost_usd": parsed.cost_usd,
        "parent_cost_usd": parsed.parent_cost_usd,
        "subagent_cost_usd": sub_cost,
        "subagent_tokens": parsed.sub_tokens,
        "subagent_tool_calls": parsed.sub_tools,
        "turns": parsed.turns,
        "resolved_subagent_model": parsed.resolved_model,
        "agent_type": parsed.agent_type,
        "models_billed": sorted(parsed.model_usage),
        "mcp_calls": parsed.mcp_calls,
        "scout_calls": sum(parsed.scout_calls.values()),
        "scout_explore_calls": parsed.scout_calls.get("explore", 0),
        "scout_search_calls": parsed.scout_calls.get("semantic_search", 0),
        "scout_mcp_status": parsed.mcp_status.get("scout"),
        "sub_tool_counts": parsed.sub_tool_counts,
        "sub_tool_seq": parsed.sub_tool_seq[:40],
        "plugin_tool_counts": parsed.plugin_tool_counts,
        "explore_first": bool(parsed.sub_tool_seq) and parsed.sub_tool_seq[0] == "explore",
        "reads_after_explore": _reads_after(parsed.sub_tool_seq),
        "rewrite_fired": rewrote,
        "fallthrough": fallthrough,
        "retries": parsed.retries,
        "denials": parsed.denials,
        "is_error": parsed.is_error,
        "zero_tools": parsed.zero_tools,
        "parent_agent_types": parsed.parent_agent_types,
        "parent_tool_counts": parsed.parent_tool_counts,
        "usage_tokens": parsed.usage_tokens,
        "scout_tool_calls_any": sum(parsed.plugin_tool_counts.values()),
        "scout_explore_calls_any": parsed.plugin_tool_counts.get("explore", 0),
        "explore_agent_calls": sum(1 for t in parsed.parent_agent_types if t == "Explore"),
        "scout_agent_calls": sum(1 for t in parsed.parent_agent_types if t.startswith("scout:")),
        "hook_rewrites": sum(1 for ln in hook_lines if re.search(r"\brewrite lane=production", ln)),
        "arm_family": "cc" if arm in CC_ARMS else "coord",
        "n_reported": len(hb_paths),
        "recall5": len(gold & set(hb_paths[:5])) / len(gold),
        "precision12": (len(gold & set(top)) / len(top)) if top else 0.0,
        **row_extra,
    }


def _reads_after(seq: list[str]) -> int:
    """Read-type calls the subagent made after its first scout `explore` (the hybrid arm's verification)."""
    if "explore" not in seq:
        return 0
    return sum(1 for t in seq[seq.index("explore") + 1 :] if t in ("Read", "read_range"))


def valid_row(row: dict) -> str | None:
    """Reason a row is not a healthy measurement, else None (validate-run-outputs)."""
    if row.get("is_error"):
        return "claude reported an error"
    if not row.get("brief_sha") or row.get("wall_s", 0) < 2:
        return "no brief hash or implausible wall time"
    if row.get("arm") in CC_ARMS:
        if not row.get("paths") and not row.get("final_paths"):
            return "no reported paths parsed"
        if row.get("arm") != CC_EXPLORE and row.get("scout_mcp_status") not in (None, "connected", "pending") and False:
            return "scout MCP not connected"
        if row.get("cost_usd") is None or not row.get("turns"):
            return "no cost or turns"
        return None
    if row.get("handback_chars", 0) == 0:
        return "empty handback"
    if row.get("cost_usd") is None:
        return "no cost"
    if not row.get("resolved_subagent_model"):
        return "no resolved subagent model"
    if row.get("zero_tools"):
        return "subagent spawned with zero tools"
    if row.get("arm") == AUGMENT:
        if row.get("scout_mcp_status") != "connected":
            return "scout MCP server not connected (the subagent had no scout tools)"
        if row.get("agent_type") != "Explore":
            return "augment arm did not run the built-in Explore subagent"
    return None


def run_session(
    arm: str, task: dict, brief: str, cwd: Path, config_dir: Path, env: dict[str, str], *, budget: float, timeout: int
) -> tuple[Parsed, float, list[str], str]:
    settings, _, _ = ensure_support_files()
    config_dir.mkdir(parents=True, exist_ok=True)
    off = hook_log_size(config_dir)
    t0 = time.time()
    try:
        proc = subprocess.run(  # noqa: S603
            command(arm, brief, settings, budget), cwd=cwd, env=env, capture_output=True, text=True,
            timeout=timeout, stdin=subprocess.DEVNULL, check=False,
        )  # fmt: skip
        raw, err = proc.stdout, proc.stderr[-300:]
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout
        raw = out.decode("utf-8", "replace") if isinstance(out, bytes) else (out or "")
        err = "timeout"
    wall = time.time() - t0
    lines, _ = hook_lines_since(config_dir, off)
    if TRANSCRIPTS is not None:  # raw stream-json (no secrets: the key lives in env/headers, not the stream)
        TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
        (TRANSCRIPTS / f"{task['native_id'][-10:]}-{config_dir.name.split('-', 1)[1]}.jsonl").write_text(raw)
    return parse_stream(raw), wall, lines, err


def kill_watchers(config_dir: Path) -> None:
    for pidfile in (config_dir / "plugins" / "data" / PLUGIN_DATA).glob("watch-*.pid"):
        try:
            os.kill(int(pidfile.read_text().strip()), 15)
        except (OSError, ValueError):
            pass


def cleanup_config(config_dir: Path) -> None:
    shutil.rmtree(config_dir, ignore_errors=True)
