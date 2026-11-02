"""Muse Code calibration: 5 public eval tasks, Muse Code with the scout-wayfinder plugin vs Muse Code alone.

A calibration, not a claim (5 tasks, no CI). Each (task, arm) is one headless `muse exec --json` session
with model muse-spark-1.3 at reasoning effort low, in a checkout of the task's repo at `base_commit` (the
eval cache), inside its own isolated XDG_CONFIG_HOME / XDG_DATA_HOME:
  muse_plugin  scout-wayfinder installed from the repo marketplace and approved; production lane (hosted
               gateway, wayfinder-v3) with a dev key read from a file. The repo is synced through the gateway
               before the session.
  muse_plain   nothing installed.
Same prompt in both: a read-only preamble, then the eval brief (question + "End with a JSON list of ...
files ..., best first"). Both arms run with `--disable-write --disable-shell`: the briefs read as feature
requests, and in a first attempt the parent started implementing (edits, `uv run`); this measures retrieval.

Scored: file recall@12 of the parent's final answer (same extraction as the main eval), and for the plugin arm
also of the scout-explorer handback. Wall time, model calls (`run.usage.calls`, parent; children reported
separately from their session logs), tool calls, explorer use, scout tool calls.

    uv run python -m eval.harness.muse_calibration run --key-file KEY [--tasks N] [--arms a,b]
    uv run python -m eval.harness.muse_calibration report

META_API_KEY comes from ../wayfinder/.env (`MODEL_API_KEY`) into the child env only; never printed.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from eval.harness import repos
from eval.harness.briefs import build_brief, extract_paths, recall_at

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "eval" / "eval_tasks.jsonl"
HOST = REPO / "integrations" / "muse-code"
OUT = Path(os.environ.get("MUSE_CAL_OUT") or REPO / "artifacts" / "live" / "muse_calibration")
WORK = Path("/tmp/muse-calibration")  # noqa: S108  scratch XDG dirs, like arms.WORK
WAYFINDER_ENV = REPO.parent / "wayfinder" / ".env"
MUSE = os.environ.get("MUSE_BIN") or str(Path.home() / ".local" / "bin" / "muse")
SCOUT = REPO / ".venv" / "bin" / "scout"
MODEL, EFFORT = "muse-spark-1.3", "low"
ARMS = ("muse_plugin", "muse_plain")
REPOS = (
    "Blaizzy/mlx-vlm",
    "langchain-ai/deepagents",
    "lennix1337/Genexus18MCP",
    "embeddings-benchmark/mteb",
    "skypilot-org/skypilot",
)
EXPLORER = "scout-wayfinder/scout-explorer"
TIMEOUT_S = 1200
PREAMBLE = (
    "Read-only question about this repository: do not modify files and do not run commands. Find the files "
    "that would have to change (or be read) to do the following task, and say why.\n\nTask: "
)
KEYS = ("final_recall12", "handback_recall12", "wall_s", "turns", "scout_calls")


def tasks(n: int, task_set: str = "cal5") -> list[dict]:
    """`cal5`: the first task of each of `REPOS` (the calibration). `ext20`: the 20-task arm, fixed before any
    of its rows: tasks 2 and 3 (file order) of every one of the 10 eval repos, so none overlaps `cal5`."""
    rows = [json.loads(x) for x in TASKS.read_text().splitlines() if x.strip()]
    if task_set == "ext20":
        by_repo: dict[str, list[dict]] = {}
        for r in rows:
            by_repo.setdefault(r["repo"], []).append(r)
        return [t for repo in by_repo for t in by_repo[repo][1:3]][:n]
    return [next(r for r in rows if r["repo"] == repo) for repo in REPOS[:n]]


def meta_key() -> str:
    for line in WAYFINDER_ENV.read_text().splitlines():
        m = re.match(r"^\s*(?:export\s+)?MODEL_API_KEY=(.*)$", line)
        if m:
            return m.group(1).strip().strip("'\"")
    raise SystemExit("MODEL_API_KEY not in wayfinder/.env")


def xdg_env(xdg: Path) -> dict[str, str]:
    return {
        "XDG_CONFIG_HOME": str(xdg / "config"),
        "XDG_DATA_HOME": str(xdg / "data"),
        "XDG_STATE_HOME": str(xdg / "state"),
        "XDG_CACHE_HOME": str(xdg / "cache"),
    }


def setup_plugin(xdg: Path, key_file: Path) -> None:
    env = {**os.environ, **xdg_env(xdg)}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(HOST / "install.py"), "--muse", MUSE, "--lane", "production",
         "--key-file", str(key_file), "--scout-bin", str(SCOUT)],
        env=env, capture_output=True, text=True, check=False, timeout=180,
    )  # fmt: skip
    rep = json.loads(proc.stdout)
    if not rep.get("ok") or set(rep.get("capabilities", {}).values()) != {"trusted_enabled"}:
        raise SystemExit(f"plugin install failed: {proc.stdout[-400:]}")


def sync(root: Path, key_file: Path) -> dict:
    key = key_file.read_text().strip()
    env = {**os.environ, "SCOUT_KEY": key, "SCOUT_KEYSTORE": "file"}
    res = subprocess.run(  # noqa: S603
        [str(SCOUT), "sync", "--wait", "--root", str(root)], env=env, capture_output=True, text=True,
        timeout=3600, check=False, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    out = res.stdout.replace(key, "<key>")
    status = json.loads(out) if out.strip().startswith("{") else {"state": "error", "out": out[-300:]}
    if status.get("state") != "ready":
        raise RuntimeError(f"sync not ready: {json.dumps(status)[:300]}")
    return status


def stop_watchers(root: Path) -> None:
    """The plugin's SessionStart worker starts `scout watch --ensure` on the checkout; stop it after a run."""
    subprocess.run(["pkill", "-f", f"scout watch --ensure --root {root}"], check=False)  # noqa: S603, S607


def session_dirs(xdg: Path) -> list[Path]:
    return sorted(
        (Path(p).parent for p in glob.glob(f"{xdg}/data/muse/sessions/*/*/*/*/session.jsonl")),
        key=lambda p: p.stat().st_mtime,
    )


def _events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        ev = (e.get("payload") or {}).get("event") if isinstance(e.get("payload"), dict) else None
        if isinstance(ev, dict):
            out.append(ev)
    return out


def thread_stats(path: Path) -> dict:
    """Tool calls, model calls and the last assistant text of one session log (main thread or a subagent)."""
    tools: list[str] = []
    calls_args: list[tuple[str, str]] = []
    text = ""
    model_calls = 0
    for ev in _events(path):
        kind = ev.get("kind")
        if kind == "assistant_tool_calls_committed":
            for c in ev.get("tool_calls", []):
                tools.append(c["name"])
                calls_args.append((c["name"], c.get("args", "")))
        elif kind == "assistant_message_committed":
            text = ev.get("text") or text
        elif kind == "model_request_configured":
            model_calls += 1
    return {"tools": tools, "calls_args": calls_args, "text": text, "model_calls": model_calls}


def run_one(task: dict, arm: str, root: Path, key_file: Path | None, meta: str) -> dict:
    slug = f"{task['repo'].split('/')[1]}__{arm}"
    xdg = WORK / "xdg" / slug
    if xdg.exists():
        subprocess.run(["rm", "-rf", str(xdg)], check=True)  # noqa: S603, S607
    xdg.mkdir(parents=True)
    if arm == "muse_plugin":
        assert key_file is not None
        setup_plugin(xdg, key_file)
    brief = PREAMBLE + build_brief(task["question"])
    env = {**os.environ, **xdg_env(xdg), "META_API_KEY": meta}
    stream = OUT / f"{slug}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    proc = subprocess.run(  # noqa: S603
        [MUSE, "exec", "--json", "--trust-workspace", "--no-foreign-personal-context", "--model", MODEL,
         "--reasoning-effort", EFFORT, "--disable-write", "--disable-shell", brief],
        cwd=root, env=env, capture_output=True, text=True, timeout=TIMEOUT_S, check=False,
    )  # fmt: skip
    wall = time.monotonic() - t0
    stop_watchers(root)
    stream.write_text(proc.stdout.replace(meta, "<key>"), encoding="utf-8")
    final, usage = "", {}
    for line in proc.stdout.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("payload_type") == "run.terminal.completed":
            final = e["payload"].get("text", "")
        if e.get("payload_type") == "run.usage":
            usage = e["payload"]
    main_dir = session_dirs(xdg)[-1]
    main = thread_stats(main_dir / "session.jsonl")
    subs = []
    for sub in sorted(main_dir.glob("subagent/*/session.jsonl"), key=lambda p: p.stat().st_mtime):
        st = thread_stats(sub)
        if st["tools"] and not all(t == "submit_reminder_decision" for t in st["tools"]):
            subs.append(st)
    spawns = [a for n, a in main["calls_args"] if n == "subagent_spawn"]
    explorer_used = any(EXPLORER in a for a in spawns)
    route_log = xdg / "data" / "muse" / "plugins" / "data" / "scout-wayfinder" / "route.log"
    rewrote = route_log.exists() and "spawn rewrite" in route_log.read_text()
    files = repos.ls_files(root)
    prefixes = [str(root), str(root.resolve()), "/private" + str(root)]
    final_paths = extract_paths(final, files, root_prefixes=prefixes)
    handback = next((s["text"] for s in subs if any("scout" in t for t in s["tools"])), "")
    hb_paths = extract_paths(handback, files, root_prefixes=prefixes) if handback else []
    scout_calls = sum(1 for s in subs for t in s["tools"] if "scout" in t)
    parent_scout_calls = sum(1 for t in main["tools"] if "scout" in t)  # attempted; guard mode denies explore
    return {
        "task": task["native_id"],
        "repo": task["repo"],
        "lang": task["lang"],
        "arm": arm,
        "rc": proc.returncode,
        "wall_s": round(wall, 1),
        "model": MODEL,
        "parent_model_calls": usage.get("calls"),
        "parent_tokens_in": sum(m.get("input_tokens", 0) for m in usage.get("models", [])),
        "parent_tokens_out": sum(m.get("output_tokens", 0) for m in usage.get("models", [])),
        "parent_tool_calls": len(main["tools"]),
        "parent_tools": main["tools"],
        "subagents": len(subs),
        "subagent_tool_calls": sum(len(s["tools"]) for s in subs),
        "subagent_model_calls": sum(s["model_calls"] for s in subs),
        "turns": len(main["tools"]) + sum(len(s["tools"]) for s in subs),
        "explorer_spawned": explorer_used or rewrote,
        "route_rewrote": rewrote,
        "scout_calls": scout_calls,
        "parent_scout_calls": parent_scout_calls,
        "n_gold": len(task["gold_files"]),
        "final_paths": final_paths[:12],
        "final_recall12": recall_at(final_paths, task["gold_files"]),
        "handback_paths": hb_paths[:12],
        "handback_recall12": recall_at(hb_paths, task["gold_files"]) if hb_paths else None,
        "stderr_tail": proc.stderr.replace(meta, "<key>")[-300:],
    }


def cmd_run(args: argparse.Namespace) -> None:
    meta = meta_key()
    key_file = Path(args.key_file) if args.key_file else None
    arms = args.arms.split(",")
    rows_path = OUT / "rows.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if rows_path.exists():
        prior = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
        done = {(r["task"], r["arm"]) for r in prior if r["rc"] == 0}
    for task in tasks(args.tasks, args.set):
        root = repos.checkout(task["repo"], task["base_commit"])
        if "muse_plugin" in arms and (task["native_id"], "muse_plugin") not in done:
            st = sync(root, key_file) if key_file else {}
            print(f"sync {task['repo']}: {st.get('state')} files={st.get('files')}", flush=True)
        for arm in arms:
            if (task["native_id"], arm) in done:
                continue
            row = run_one(task, arm, root, key_file, meta)
            with rows_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"{task['repo']:32s} {arm:12s} rc={row['rc']} wall={row['wall_s']}s "
                f"r@12 final={row['final_recall12']:.2f} hb={row['handback_recall12']} "
                f"explorer={row['explorer_spawned']} scout={row['scout_calls']} turns={row['turns']}",
                flush=True,
            )


def cmd_report(_: argparse.Namespace) -> None:
    rows = [json.loads(x) for x in (OUT / "rows.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["rc"] == 0]
    by: dict[str, dict[str, dict]] = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["task"]] = r
    common = sorted(set.intersection(*(set(v) for v in by.values()))) if by else []
    summary: dict = {"tasks": len(common), "arms": {}}
    for arm, rs in by.items():
        sel = [rs[t] for t in common]

        def mean(key: str, sel: list[dict] = sel) -> float | None:
            vals = [r[key] for r in sel if r.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        summary["arms"][arm] = {
            "final_recall12": mean("final_recall12"),
            "handback_recall12": mean("handback_recall12"),
            "wall_s_mean": mean("wall_s"),
            "wall_s_median": sorted(r["wall_s"] for r in sel)[len(sel) // 2] if sel else None,
            "turns_mean": mean("turns"),
            "parent_model_calls_mean": mean("parent_model_calls"),
            "explorer_spawned": sum(1 for r in sel if r["explorer_spawned"]),
            "scout_calls": sum(r["scout_calls"] for r in sel),
            "parent_tokens_in_mean": mean("parent_tokens_in"),
        }
    summary["per_task"] = [
        {"task": t, **{arm: {k: by[arm][t][k] for k in KEYS} for arm in by}} for t in common
    ]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Muse Code calibration (scout-wayfinder plugin vs none)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--key-file", default=None)
    r.add_argument("--tasks", type=int, default=5)
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--set", default="cal5", choices=["cal5", "ext20"])
    sub.add_parser("report")
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
