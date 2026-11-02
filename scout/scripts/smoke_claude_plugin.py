# ruff: noqa: E501, S108, S607  (an ops script: long report lines, fixed scratch dir under /tmp)
"""Headless live smoke of the scout Claude Code plugin (0.2.0) on a public repo, production lane.

    uv run python scripts/smoke_claude_plugin.py [--out artifacts/live/claude_plugin_smoke] [--only CASE,...]

Every `claude -p` runs with its own CLAUDE_CONFIG_DIR under /tmp (the user's ~/.claude is never read or
written), the parent on claude-sonnet-5-5 through an LLM API gateway (the eval harness's route), the plugin from
`--plugin-dir integrations/claude-code`. gorilla/mux is cloned to /tmp and indexed through the hosted gateway
with the dev key (`~/.config/scout/live-smoke-keys.txt`, `dev=` line), passed only as SCOUT_KEY in the child
environment and never printed. Cases:

  routed       a user prompt that uses Explore: the hook rewrites it to scout:scout-explorer, whose first tool
               is scout `explore`, followed by Read verification; the answer carries ranked path:start-end ranges
  fallthrough  SCOUT_GATEWAY_URL points at a closed port: the SessionStart probe records the gateway as
               unreachable, the hook falls through and the built-in Explore answers (no failure shown)
  commands     /scout:status, /scout:model a1 then v3, /scout:mode fast then hybrid, /scout:index, /scout:find

Writes `<out>/summary.json` plus each session's stream-json (`<case>.jsonl`, key-scrubbed) and hook log. The
branch namespaces this run created are deleted at the end.
"""

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.harness.arms import LLM_GATEWAY_BASE, gateway_key, parse_stream  # noqa: E402

PLUGIN = REPO / "integrations" / "claude-code"
SCOUT = REPO / ".venv" / "bin" / "scout"
KEY_FILE = Path.home() / ".config" / "scout" / "live-smoke-keys.txt"
GATEWAY = "https://scout-gateway.example.com"
WORK = Path("/tmp/scout-plugin-smoke")
PARENT = "claude-sonnet-5-5"
RANGE = re.compile(r"\b[\w./-]+\.go:\d+-\d+\b")
HYBRID = "scout:scout-explorer"
MODEL_ENV: dict[
    str, str
] = {}  # --model a1 sets SCOUT_MODEL for the search sessions (the /scout:model switch)


def dev_key() -> str:
    for line in KEY_FILE.read_text().splitlines():
        name, _, value = line.partition("=")
        if name == "dev" and value.strip():
            return value.strip()
    raise SystemExit("no dev= line in live-smoke-keys.txt")


def base_env(cfg: Path, key: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE", "SCOUT_", "GATEWAY_"))
    }
    env.update(
        CLAUDE_CONFIG_DIR=str(cfg),
        ANTHROPIC_BASE_URL=LLM_GATEWAY_BASE,
        ANTHROPIC_API_KEY="unused",
        ANTHROPIC_CUSTOM_HEADERS=f"api-key: {gateway_key()}",
        ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-5-5",
        ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-5-5",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        DISABLE_TELEMETRY="1",
        DISABLE_AUTOUPDATER="1",
        NO_COLOR="1",
        SCOUT_KEY=key,
        SCOUT_BIN=str(SCOUT),
        SCOUT_KEYSTORE="file",
        SCOUT_DEMO_HOME=str(WORK / "empty-demo-home"),
    )
    env.update(extra or {})
    return env


def scrub(text: str, key: str) -> str:
    return text.replace(key, "<scout-key>").replace(gateway_key(), "<llm_key>")


def claude(
    prompt: str, cwd: Path, cfg: Path, key: str, *, extra: dict[str, str] | None = None, timeout: int = 900
) -> tuple[str, float]:
    cfg.mkdir(parents=True, exist_ok=True)
    cmd = [
        "claude", "-p", prompt, "--model", PARENT, "--plugin-dir", str(PLUGIN),
        "--output-format", "stream-json", "--verbose", "--include-hook-events",
        "--setting-sources", "project", "--no-session-persistence", "--max-budget-usd", "2",
    ]  # fmt: skip
    t0 = time.time()
    try:
        proc = subprocess.run(  # noqa: S603
            cmd, cwd=cwd, env=base_env(cfg, key, extra), capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL, check=False,
        )  # fmt: skip
        raw = proc.stdout
    except subprocess.TimeoutExpired as e:
        raw = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    return raw, time.time() - t0


def data_dir(cfg: Path) -> Path:
    return cfg / "plugins" / "data" / "scout-inline"


def hook_log(cfg: Path) -> list[str]:
    p = data_dir(cfg) / "hook.log"
    return p.read_text().splitlines() if p.exists() else []


def events(raw: str) -> list[dict]:
    out = []
    for line in raw.splitlines():
        if line.startswith("{"):
            with contextlib.suppress(ValueError):
                out.append(json.loads(line))
    return out


def result_text(raw: str) -> str:
    return next((str(e.get("result", "")) for e in reversed(events(raw)) if e.get("type") == "result"), "")


def cost(raw: str) -> float:
    return float(
        next((e.get("total_cost_usd") or 0 for e in reversed(events(raw)) if e.get("type") == "result"), 0)
    )


def save(out: Path, name: str, raw: str, cfg: Path, key: str) -> None:
    (out / f"{name}.jsonl").write_text(scrub(raw, key))
    (out / f"{name}.hook.log").write_text("\n".join(hook_log(cfg)) + "\n")


def kill_watchers(cfg: Path) -> None:
    for pidfile in data_dir(cfg).glob("*.pid"):
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(pidfile.read_text().strip()), 15)


def prepare(key: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "empty-demo-home").mkdir(exist_ok=True)
    clone = WORK / "mux"
    if not (clone / ".git").exists():
        subprocess.run(  # noqa: S603
            ["git", "clone", "--depth", "1", "--quiet", "https://github.com/gorilla/mux", str(clone)],  # noqa: S607
            check=True, timeout=180,
        )  # fmt: skip
    env = {**os.environ, "SCOUT_KEY": key, "SCOUT_GATEWAY_URL": GATEWAY, "SCOUT_KEYSTORE": "file"}
    t0 = time.time()
    res = subprocess.run(  # noqa: S603
        [str(SCOUT), "sync", "--wait", "--root", str(clone)], env=env, capture_output=True, text=True,
        timeout=600, check=False, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    status = json.loads(res.stdout)
    assert status.get("state") == "ready", scrub(res.stdout[-300:], key)
    print(f"indexed gorilla/mux: {status.get('files')} files in {time.time() - t0:.0f}s", flush=True)
    return clone


def case_routed(clone: Path, key: str, out: Path) -> dict:
    cfg = WORK / "cfg-routed"
    shutil.rmtree(cfg, ignore_errors=True)
    prompt = (
        "Use the Explore subagent (run_in_background false, wait for it) to find where the router decides to "
        "answer 405 Method Not Allowed instead of 404, and how the method mismatch is detected. Then give me the "
        "ranked file ranges it returned."
    )
    raw, wall = claude(prompt, clone, cfg, key, extra=MODEL_ENV)
    save(out, "routed", raw, cfg, key)
    kill_watchers(cfg)
    p = parse_stream(raw)
    log = hook_log(cfg)
    seq = p.sub_tool_seq
    first_explore = seq[:1] == ["explore"]
    reads_after = sum(1 for t in seq[1:] if t in ("Read", "read_range")) if first_explore else 0
    final = result_text(raw)
    ranges = RANGE.findall(p.handback)
    checks = {
        "hook_rewrote_to_hybrid": any(f"rewrite lane=production target={HYBRID}" in ln for ln in log),
        "parent_asked_for_explore": p.subagent_requested == "Explore",
        "subagent_ran_was_hybrid": p.agent_type == HYBRID,
        "scout_explore_called_first": first_explore,
        "read_verification_after_explore": reads_after >= 1,
        "handback_has_ranked_ranges": len(ranges) >= 2 and bool(re.search(r"^\s*1\.\s", p.handback, re.M)),
        "handback_at_most_12_entries": len(re.findall(r"^\s*\d+\.\s+\S+:\d+-\d+", p.handback, re.M)) <= 12,
        "no_zero_tools": not p.zero_tools,
        "no_scout_tool_denials": not any(d.startswith("mcp__plugin_scout") for d in p.denials),
        "mux_go_found": "mux.go" in p.handback,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "hook_log": log[-4:],
        "subagent_type": p.agent_type,
        "resolved_subagent_model": p.resolved_model,
        "subagent_tool_sequence": seq,
        "plugin_tool_counts": p.plugin_tool_counts,
        "handback_ranges": ranges[:12],
        "handback": p.handback[:3000],
        "final_answer_head": final[:800],
        "wall_s": round(wall, 1),
        "subagent_s": None if p.sub_ms is None else round(p.sub_ms / 1000, 1),
        "cost_usd": p.cost_usd,
    }


def case_fallthrough(clone: Path, key: str, out: Path) -> dict:
    cfg = WORK / "cfg-fallthrough"
    shutil.rmtree(cfg, ignore_errors=True)
    extra = {"SCOUT_GATEWAY_URL": "http://127.0.0.1:9"}  # nothing listens: the gateway is unreachable
    prompt = (
        "Use the Explore subagent (run_in_background false, wait for it) to find which file defines the Router "
        "type and its ServeHTTP method."
    )
    raw, wall = claude(prompt, clone, cfg, key, extra=extra, timeout=600)
    save(out, "fallthrough", raw, cfg, key)
    kill_watchers(cfg)
    p = parse_stream(raw)
    log = hook_log(cfg)
    warm = (
        json.loads((data_dir(cfg) / "warm.json").read_text())
        if (data_dir(cfg) / "warm.json").exists()
        else {}
    )
    final = result_text(raw)
    checks = {
        "session_start_probe_recorded_unreachable": any(
            r.get("state") == "unreachable" for r in warm.values()
        ),
        "hook_fell_through_gateway_unreachable": any(
            "fallthrough" in ln and "gateway-unreachable" in ln for ln in log
        ),
        "no_rewrite": not any(" rewrite " in f" {ln} " for ln in log),
        "builtin_explore_ran": p.agent_type == "Explore",
        "answer_names_mux_go": "mux.go" in (p.handback + final),
        "no_error_result": not p.is_error,
        "no_zero_tools": not p.zero_tools,
    }
    return {"checks": checks, "pass": all(checks.values()), "hook_log": log[-4:], "warm": warm,
            "subagent_type": p.agent_type, "wall_s": round(wall, 1), "cost_usd": p.cost_usd,
            "final_answer_head": final[:400]}  # fmt: skip


def case_commands(clone: Path, key: str, out: Path) -> dict:
    cfg = WORK / "cfg-commands"
    shutil.rmtree(cfg, ignore_errors=True)
    no_warm = {"SCOUT_NO_WARM": "1"}  # /scout:model a1 must not start the A1 GPU for a smoke
    res: dict = {}
    plan = [
        (
            "status",
            "/scout:status",
            no_warm,
            [
                r"lane\s+production",
                r"model\s+v3",
                r"index\s+ready 100%, at HEAD",
                r"endpoint\s+\w",
                r"routing\s+mode hybrid",
            ],
        ),
        ("model_a1", "/scout:model a1", no_warm, [r"a1 - Agents-A1-35B", r"scout-gateway-a1-api"]),
        ("status_after_a1", "/scout:status", no_warm, [r"model\s+a1 - Agents-A1-35B\s+\(model profile\)"]),
        ("model_v3", "/scout:model v3", no_warm, [r"v3 - wayfinder-glimmer"]),
        ("mode_fast", "/scout:mode fast", no_warm, [r"scout mode for this project: fast"]),
        ("mode_hybrid", "/scout:mode hybrid", no_warm, [r"scout mode for this project: hybrid"]),
        (
            "index",
            "/scout:index",
            no_warm,
            [r"estimate \d+ files", r"~\$\d", r"done in \d+s: index ready 100%"],
        ),
    ]
    for name, prompt, extra, pats in plan:
        raw, wall = claude(prompt, clone, cfg, key, extra=extra, timeout=300)
        save(out, f"cmd_{name}", raw, cfg, key)
        text = result_text(raw)
        tool_out = "\n".join(
            str(c.get("content")) for e in events(raw) if e.get("type") == "user"
            for c in (e.get("message", {}).get("content") or []) if isinstance(c, dict)
        )  # fmt: skip
        hay = text + "\n" + tool_out + "\n" + raw
        ok = {p: bool(re.search(p, hay)) for p in pats}
        res[name] = {
            "pass": all(ok.values()),
            "patterns": ok,
            "answer": text[:900],
            "wall_s": round(wall, 1),
            "cost_usd": cost(raw),
        }
        print(f"  {name}: {'ok' if all(ok.values()) else 'FAIL'} ({wall:.0f}s)", flush=True)
    raw, wall = claude(
        "/scout:find where are route variables like {id} parsed and matched against the request path?", clone, cfg, key,
        extra=MODEL_ENV,
    )  # fmt: skip
    save(out, "cmd_find", raw, cfg, key)
    p = parse_stream(raw)
    final = result_text(raw)
    ranges = RANGE.findall(p.handback or final)
    res["find"] = {
        "pass": p.agent_type == HYBRID
        and p.sub_tool_seq[:1] == ["explore"]
        and len(ranges) >= 2
        and "Read" in p.sub_tool_seq,
        "subagent_type": p.agent_type,
        "subagent_tool_sequence": p.sub_tool_seq,
        "handback_ranges": ranges[:12],
        "answer": final[:1500],
        "wall_s": round(wall, 1),
        "cost_usd": p.cost_usd,
    }
    print(f"  find: {'ok' if res['find']['pass'] else 'FAIL'} ({wall:.0f}s)", flush=True)
    kill_watchers(cfg)
    return {"pass": all(v["pass"] for v in res.values()), "commands": res}


async def cleanup(key: str, clone: Path) -> list[str]:
    from scout.adapters.gateway_http import HttpGateway
    from scout.core.sync.state import StateDir

    state = StateDir(str(clone / ".git" / "scout")).read_state()
    gw = HttpGateway(GATEWAY, key)
    deleted = []
    try:
        if state.repo_id and state.last_branch:
            await gw.delete_branch(state.repo_id, state.last_branch)
            deleted.append("branch")
    finally:
        await gw.aclose()
    return deleted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "artifacts" / "live" / "claude_plugin_smoke"))
    ap.add_argument("--only", default="routed,fallthrough,commands")
    ap.add_argument("--keep-index", action="store_true")
    ap.add_argument(
        "--model", choices=["v3", "a1"], default="v3", help="gateway profile of the search sessions"
    )
    args = ap.parse_args()
    if args.model != "v3":
        MODEL_ENV["SCOUT_MODEL"] = args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = dev_key()
    version = subprocess.run(
        ["claude", "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()  # noqa: S607
    clone = prepare(key)
    summary: dict = {"claude_code": version, "plugin": json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())["version"],
                     "repo": "gorilla/mux", "gateway": GATEWAY, "search_model": args.model, "parent_model": PARENT, "config_dirs": str(WORK / "cfg-*"),
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}  # fmt: skip
    cases = {"routed": case_routed, "fallthrough": case_fallthrough, "commands": case_commands}
    try:
        for name in args.only.split(","):
            print(f"case {name} ...", flush=True)
            summary[name] = cases[name](clone, key, out)
            print(f"case {name}: {'PASS' if summary[name]['pass'] else 'FAIL'}", flush=True)
            (out / "summary.json").write_text(
                scrub(json.dumps(summary, indent=2, sort_keys=True), key) + "\n"
            )
    finally:
        if not args.keep_index:
            summary["cleanup"] = asyncio.run(cleanup(key, clone))
        summary["anthropic_usd"] = round(sum(_costs(summary)), 4)
        (out / "summary.json").write_text(scrub(json.dumps(summary, indent=2, sort_keys=True), key) + "\n")
    print(
        json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in summary.items()}, indent=1)
    )


def _costs(obj: object) -> list[float]:
    if isinstance(obj, dict):
        out = [float(obj["cost_usd"])] if isinstance(obj.get("cost_usd"), (int, float)) else []
        for v in obj.values():
            if isinstance(v, dict):
                out += _costs(v)
        return out
    return []


if __name__ == "__main__":
    main()
