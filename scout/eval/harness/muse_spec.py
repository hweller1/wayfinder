"""Muse Code and the retrieval spec: stock vs spec-only vs spec + Scout, on the cc50 TUNE tasks.

Pre-registered in wayfinder the Muse Scout study plan (not in this release) before any run. Calibration-scale evidence, not a result.

Arms, one headless `muse exec --json` session each, all with the same free-parent prompt (`arms.CC_PROMPT`),
the same brief, the host's default parent model, `--disable-write --disable-shell` (retrieval only), and an
isolated XDG_CONFIG_HOME / XDG_DATA_HOME per (task, arm) so `~/.config/muse` and `~/.local/share/muse` are never
touched:
  muse_stock         nothing installed.
  muse_explore_spec  spec hooks only (SessionStart context, workflow step, search-first), no MCP server, no
                     agent: the delegate is Muse's built-in `general-purpose` agent (built-in search).
  muse_scout_spec    the full scout-wayfinder plugin plus the same spec: delegate `scout-wayfinder/scout-explorer`,
                     Scout `explore` on wayfinder-v3 over the repo's gateway index.
The per-task spec file is the CC spec arms' (`arms.write_spec_file`): 3 mined examples from before base_commit that
never touch a gold file, search-first on, workflow step on.

    uv run python -m eval.harness.muse_spec run --key-file KEY --first 5 --arms muse_stock,muse_explore_spec,muse_scout_spec
    uv run python -m eval.harness.muse_spec report [--out DIR]
    uv run python -m eval.harness.muse_spec sync --key-file KEY --first 1     # index only, prints Voyage tokens

META_API_KEY comes from ../wayfinder/.env (`MODEL_API_KEY`) into the child env only; never printed or stored.
"""

import argparse
import asyncio
import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.harness import arms as cc_arms
from eval.harness import repos, stats
from eval.harness.briefs import brief_sha, build_brief, extract_paths
from eval.harness.muse_calibration import (
    MUSE,
    SCOUT,
    _events,
    meta_key,
    session_dirs,
    stop_watchers,
    thread_stats,
    xdg_env,
)

WORK = Path(os.environ.get("MUSE_SPEC_WORK") or "/tmp/muse-spec")  # noqa: S108  scratch XDG dirs and checkouts
REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "eval" / "cc50_tasks.jsonl"
SPLIT = REPO.parent / "wayfinder" / "data" / "cc_scout_hillclimb_split.json"
HOST = REPO / "integrations" / "muse-code"
BUNDLE = HOST / "scout-wayfinder"
OUT = Path(os.environ.get("MUSE_SPEC_OUT") or REPO / "artifacts" / "live" / "muse_spec")
GATEWAY = "https://scout-gateway.example.com"
ARMS = ("muse_stock", "muse_explore_spec", "muse_scout_spec")
STOCK, EXPL, SCOUT_ARM = ARMS
EXPLORER = "scout-wayfinder/scout-explorer"
TIMEOUT_S = 900
RATES = (1.25, 0.15, 4.25)  # Muse Spark list price per M tokens: uncached input, cached input, output
LOCK = threading.Lock()
SPEC_PLUGIN = "scout-spec"


def use_cache() -> None:
    """Point the checkout cache at this study's own scratch dir (never a cache another run may `git clean`)."""
    repos.CACHE = Path(os.environ.get("SCOUT_EVAL_CACHE") or WORK / "repos")


def tune_tasks() -> list[dict]:
    ids = json.loads(SPLIT.read_text())["tune"]
    rows = {json.loads(x)["native_id"]: json.loads(x) for x in TASKS.read_text().splitlines() if x.strip()}
    return [rows[i] for i in ids]


def build_spec_only_bundle() -> Path:
    """A Muse plugin with only the three spec hooks (no MCP server, agent, skill or route hook), built once."""
    dst = WORK / "spec-only-plugin"
    if (dst / ".built").exists():
        return dst
    shutil.rmtree(dst, ignore_errors=True)
    for rel in (
        "hooks/spec_hooks.py",
        "hooks/spec_session.py",
        "hooks/spec_prompt.py",
        "hooks/spec_pretool.py",
        "lib/lanes.py",
        "lib/scout_spec.py",
        "spec/harness_defaults.toml",
    ):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BUNDLE / rel, dst / rel)
    full = json.loads((BUNDLE / ".muse-plugin" / "plugin.json").read_text())
    keep = {"spec-session", "spec-prompt", "spec-pretool"}
    manifest = {
        **{k: full[k] for k in ("schemaVersion", "version", "compat")},
        "name": SPEC_PLUGIN,
        "description": "Retrieval-spec hooks only (the muse_explore_spec arm): no MCP server, no agent.",
        "capabilities": {
            "skills": [],
            "commands": [],
            "hooks": [h for h in full["capabilities"]["hooks"] if h["id"] in keep],
            "mcpServers": [],
            "reminders": [],
        },
    }
    (dst / ".muse-plugin").mkdir()
    (dst / ".muse-plugin" / "plugin.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dst / ".built").write_text("1")
    return dst


def muse_cmd(env: dict[str, str], *args: str) -> dict:
    proc = subprocess.run(  # noqa: S603
        [MUSE, "plugins", *args, "--json"], env=env, capture_output=True, text=True, timeout=180, check=False
    )
    if proc.returncode != 0:
        raise SystemExit(f"muse plugins {args[0]} failed: {(proc.stdout + proc.stderr)[-400:]}")
    return json.loads(proc.stdout)


def write_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")
    path.chmod(0o600)


def setup_arm(arm: str, xdg: Path, spec_file: Path, key_file: Path | None) -> None:
    env = {**os.environ, **xdg_env(xdg)}
    data = xdg / "data" / "muse" / "plugins" / "data"
    if arm == EXPL:
        muse_cmd(env, "install", str(build_spec_only_bundle()))
        approved = muse_cmd(env, "approve", SPEC_PLUGIN)["runtime_capabilities"]
        assert len(approved) == 3, approved
        write_env(
            data / SPEC_PLUGIN / "scout.env",
            {"SCOUT_SPEC_FILE": str(spec_file), "SCOUT_SPEC_TARGET": "general-purpose"},
        )
    elif arm == SCOUT_ARM:
        assert key_file is not None
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(HOST / "install.py"), "--muse", MUSE, "--lane", "production",
             "--key-file", str(key_file), "--scout-bin", str(SCOUT)],
            env=env, capture_output=True, text=True, check=False, timeout=240,
        )  # fmt: skip
        rep = json.loads(proc.stdout)
        if not rep.get("ok") or set(rep.get("capabilities", {}).values()) != {"trusted_enabled"}:
            raise SystemExit(f"plugin install failed: {proc.stdout[-400:]}")
        env_file = data / "scout-wayfinder" / "scout.env"
        extra = {"SCOUT_SPEC_FILE": str(spec_file), "SCOUT_NO_WARM": "1", "SCOUT_GATEWAY_URL": GATEWAY}
        env_file.write_text(env_file.read_text() + "".join(f"{k}={v}\n" for k, v in extra.items()))


def sync(root: Path, key_file: Path) -> dict:
    """`scout sync --wait` through the gateway, counting the Voyage tokens it embeds (index_cc50's counter)."""
    from eval.harness.index_cc50 import sync_one

    key = key_file.read_text().strip()
    res = asyncio.run(sync_one(root, key, GATEWAY))
    if res.get("state") != "ready":
        raise RuntimeError(f"sync not ready: {json.dumps(res)[:300]}")
    return res


def usage_cost(tokens_in: int, cached: int, out: int) -> float:
    return ((tokens_in - cached) * RATES[0] + cached * RATES[1] + out * RATES[2]) / 1e6


def provider_usage(path: Path) -> dict[str, int]:
    """Provider tokens of one session log (`goal_usage_attribution`, usage_family provider)."""
    tot = {"in": 0, "cached": 0, "out": 0, "calls": 0}
    for ev in _events(path):
        if ev.get("kind") != "goal_usage_attribution":
            continue
        rec = ev.get("record", {})
        q = rec.get("quantity", {})
        if rec.get("usage_family") == "provider" and q.get("reported"):
            tot["in"] += q.get("input_tokens", 0)
            tot["cached"] += q.get("cached_tokens", 0)
            tot["out"] += q.get("output_tokens", 0)
            tot["calls"] += 1
    return tot


def scout_explore_seconds(path: Path) -> list[float]:
    """Wall seconds of each scout `explore` tool call in one thread: commit event to the batch result."""
    out: list[float] = []
    pending: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        ev = (e.get("payload") or {}).get("event") if isinstance(e.get("payload"), dict) else None
        if not isinstance(ev, dict):
            continue
        t = e.get("recorded_at", 0) / 1e6
        if ev.get("kind") == "assistant_tool_calls_committed":
            pending = (
                t
                if any("scout" in c["name"] and c["name"].endswith("explore") for c in ev["tool_calls"])
                else None
            )
        elif ev.get("kind") == "tool_result_batch_committed" and pending is not None:
            out.append(round(t - pending, 1))
            pending = None
    return out


def spawn_types(calls_args: list[tuple[str, str]]) -> list[str]:
    kinds = []
    for name, args in calls_args:
        if name != "subagent_spawn":
            continue
        try:
            kinds.append(str(json.loads(args).get("subagent_type") or "general-purpose"))
        except ValueError:
            kinds.append("?")
    return kinds


def hook_log(xdg: Path, plugin: str) -> list[dict]:
    p = xdg / "data" / "muse" / "plugins" / "data" / plugin / "spec_hooks.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def score(final: str, root: Path, task: dict) -> dict:
    prefixes = [str(root), str(root.resolve()), "/private" + str(root)]
    paths = extract_paths(final, repos.ls_files(root), root_prefixes=prefixes)
    gold = set(task["gold_files"])
    top = paths[:12]
    return {
        "final_paths": paths[:20],
        "n_reported": len(paths),
        "n_gold": len(gold),
        "recall5": len(gold & set(paths[:5])) / len(gold),
        "recall12": len(gold & set(top)) / len(gold),
        "precision12": len(gold & set(top)) / len(top) if top else 0.0,
    }


def validate(row: dict) -> str | None:
    """Why a row is not a healthy measurement (validate-run-outputs), else None."""
    if row["rc"] != 0:
        return f"muse exec rc={row['rc']}"
    if row["wall_s"] < 3 or not row["parent_model_calls"]:
        return "implausible wall time or no parent model call"
    if not row["parent_model"]:
        return "no model recorded"
    if not row["brief_sha"]:
        return "no brief hash"
    if not row["n_reported"]:
        return "no reported paths parsed"
    if row["arm"] != STOCK:
        if not row["spec"].get("session_injected") or not row["spec"].get("prompt_injected"):
            return "spec context or workflow step did not fire"
    if row["arm"] == SCOUT_ARM and not row["scout_explore_calls"]:
        return "scout explore never called"
    return None


def run_one(task: dict, arm: str, root: Path, key_file: Path | None, meta: str) -> dict:
    slug = f"{task['native_id'][-10:]}__{arm}"
    xdg = WORK / "xdg" / slug
    shutil.rmtree(xdg, ignore_errors=True)
    xdg.mkdir(parents=True)
    spec_file = cc_arms.write_spec_file(xdg / "spec", task)
    (repos.state_dir(root) / "spec_resolved.json").unlink(
        missing_ok=True
    )  # one arm's spec must not leak into another
    if arm != STOCK:
        cc_arms.prewarm_spec(root, spec_file)
    setup_arm(arm, xdg, spec_file, key_file)
    brief = build_brief(task["question"])
    prompt = cc_arms.CC_PROMPT.format(brief=brief)
    env = {**os.environ, **xdg_env(xdg), "META_API_KEY": meta}
    t0 = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603
            [MUSE, "exec", "--json", "--trust-workspace", "--no-foreign-personal-context", "--disable-write",
             "--disable-shell", prompt],
            cwd=root, env=env, capture_output=True, text=True, timeout=TIMEOUT_S, check=False, stdin=subprocess.DEVNULL,
        )  # fmt: skip
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        )
        stderr, rc = "timeout", 124
    wall = time.monotonic() - t0
    stop_watchers(root)
    stdout = stdout.replace(meta, "<key>")
    tdir = OUT / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{slug}.jsonl.gz").write_bytes(gzip.compress(stdout.encode()))
    final, usage, model, effort = "", {}, None, None
    for line in stdout.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        pl = e.get("payload") or {}
        if e.get("payload_type") == "run.terminal.completed":
            final = pl.get("text", "")
        elif e.get("payload_type") == "run.usage":
            usage = pl
        elif e.get("payload_type") == "run.model.configured" and model is None:
            model, effort = pl.get("model_id"), pl.get("effort")
    sessions = session_dirs(xdg)
    main_dir = sessions[-1] if sessions else None
    main = (
        thread_stats(main_dir / "session.jsonl")
        if main_dir
        else {"tools": [], "calls_args": [], "model_calls": 0}
    )
    subs, sub_files = [], []
    if main_dir:
        sdir = tdir / f"{slug}.sessions"
        sdir.mkdir(exist_ok=True)
        (sdir / "main.jsonl.gz").write_bytes(gzip.compress((main_dir / "session.jsonl").read_bytes()))
        for i, sub in enumerate(
            sorted(main_dir.glob("subagent/*/session.jsonl"), key=lambda p: p.stat().st_mtime)
        ):
            st = thread_stats(sub)
            if st["tools"] and not all(t == "submit_reminder_decision" for t in st["tools"]):
                subs.append(st)
                sub_files.append(sub)
                (sdir / f"sub{i}.jsonl.gz").write_bytes(gzip.compress(sub.read_bytes()))
    ptok = (
        provider_usage(main_dir / "session.jsonl")
        if main_dir
        else {"in": 0, "cached": 0, "out": 0, "calls": 0}
    )
    stok = {"in": 0, "cached": 0, "out": 0, "calls": 0}
    for f in sub_files:
        for k, v in provider_usage(f).items():
            stok[k] += v
    explore_s = [s for f in sub_files for s in scout_explore_seconds(f)]
    if main_dir:
        explore_s += scout_explore_seconds(main_dir / "session.jsonl")
    sub_tools = [t for s in subs for t in s["tools"]]
    parent_scout = [t for t in main["tools"] if "scout" in t]
    scout_explore_calls = sum(
        1 for t in [*sub_tools, *main["tools"]] if "scout" in t and t.endswith("explore")
    )
    plugin = SPEC_PLUGIN if arm == EXPL else "scout-wayfinder"
    hooks = hook_log(xdg, plugin) if arm != STOCK else []
    row = {
        "task": task["native_id"],
        "repo": task["repo"],
        "lang": task["lang"],
        "repo_type": task.get("repo_type"),
        "arm": arm,
        "rc": rc,
        "wall_s": round(wall, 1),
        "parent_model": model,
        "parent_effort": effort or "host default (high)",
        "brief_sha": brief_sha(brief),
        "prompt_sha": brief_sha(prompt),
        "parent_model_calls": usage.get("calls"),
        "parent_tool_calls": len(main["tools"]),
        "parent_tools": main["tools"],
        "spawned": spawn_types(main["calls_args"]),
        "delegated": bool(spawn_types(main["calls_args"])),
        "delegated_to_scout": any(k == EXPLORER for k in spawn_types(main["calls_args"])),
        "subagents": len(subs),
        "subagent_tool_calls": len(sub_tools),
        "scout_explore_calls": scout_explore_calls,
        "scout_tool_called": scout_explore_calls > 0,
        "parent_scout_attempts": len(parent_scout),
        "scout_explore_s": explore_s,
        "spec": {
            "session_injected": any(h.get("hook") == "SessionStart" and h.get("injected") for h in hooks),
            "prompt_injected": any(h.get("hook") == "UserPromptSubmit" and h.get("injected") for h in hooks),
            "denials": sum(1 for h in hooks if h.get("denied")),
            "targets": sorted({h["target"] for h in hooks if h.get("target")}),
            "hook_errors": [h["error"] for h in hooks if h.get("error")][:3],
        },
        "tokens_parent": ptok,
        "tokens_children": stok,
        "muse_usd_parent": round(usage_cost(ptok["in"], ptok["cached"], ptok["out"]), 4),
        "muse_usd_children": round(usage_cost(stok["in"], stok["cached"], stok["out"]), 4),
        "stderr_tail": stderr.replace(meta, "<key>")[-200:],
        **score(final, root, task),
    }
    row["muse_usd"] = round(row["muse_usd_parent"] + row["muse_usd_children"], 4)
    row["problem"] = validate(row)
    shutil.rmtree(xdg, ignore_errors=True)
    return row


def append(path: Path, row: dict) -> None:
    with LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_rows(out: Path) -> list[dict]:
    p = out / "rows.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def select(args: argparse.Namespace) -> list[dict]:
    tasks = tune_tasks()
    if args.ids:
        tasks = [t for t in tasks if t["native_id"][-10:] in args.ids.split(",")]
    return tasks[: args.first] if args.first else tasks


def cmd_sync(args: argparse.Namespace) -> None:
    use_cache()
    key_file = Path(args.key_file)
    for task in select(args):
        root = repos.checkout(task["repo"], task["base_commit"])
        res = sync(root, key_file)
        print(json.dumps({"repo": task["repo"], "task": task["native_id"][-10:], **res}), flush=True)


def cmd_run(args: argparse.Namespace) -> None:
    use_cache()
    meta = meta_key()
    key_file = Path(args.key_file) if args.key_file else None
    arms = args.arms.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT / "rows.jsonl"
    done = {(r["task"], r["arm"]) for r in load_rows(OUT) if not r["problem"]}
    locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
    tasks = select(args)
    spent = {"voyage_tokens": 0}

    def job(task: dict) -> None:
        with locks[task["repo"]]:
            todo = [a for a in arms if (task["native_id"], a) not in done]
            if not todo:
                return
            root = repos.checkout(task["repo"], task["base_commit"])
            sync_row: dict = {}
            if SCOUT_ARM in todo:
                sync_row = sync(root, key_file) if key_file else {}
                spent["voyage_tokens"] += sync_row.get("tokens", 0)
                print(
                    f"sync {task['repo']}: {sync_row.get('state')} tokens={sync_row.get('tokens')}",
                    flush=True,
                )
                if spent["voyage_tokens"] * 0.18 / 1e6 > args.voyage_cap:
                    raise SystemExit(f"Voyage cap ${args.voyage_cap} reached: stop")
            for arm in todo:
                row = run_one(task, arm, root, key_file, meta)
                row["sync"] = (
                    {k: sync_row.get(k) for k in ("state", "files", "tokens", "wall_s")}
                    if arm == SCOUT_ARM
                    else None
                )
                append(rows_path, row)
                print(
                    f"{task['repo']:30s} {task['native_id'][-6:]} {arm:18s} rc={row['rc']} wall={row['wall_s']}s "
                    f"r12={row['recall12']:.2f} n={row['n_reported']} deleg={row['delegated']} "
                    f"scout={row['scout_explore_calls']} usd={row['muse_usd']} problem={row['problem']}",
                    flush=True,
                )

    with ThreadPoolExecutor(args.width) as ex:
        list(ex.map(job, tasks))


def cmd_report(args: argparse.Namespace) -> None:
    out = Path(args.out) if args.out else OUT
    rows = [r for r in load_rows(out) if not r["problem"]]
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by[r["arm"]][r["task"]] = r  # a later valid row of a (task, arm) replaces an earlier one
    arms_present = [a for a in ARMS if a in by]
    common = sorted(set.intersection(*(set(by[a]) for a in arms_present))) if arms_present else []
    summary: dict = {"n_tasks": len(common), "arms": {}, "contrasts": {}}
    for a in arms_present:
        sel = [by[a][t] for t in common]

        def mean(key: str, sel: list[dict] = sel) -> float:
            vals = [float(r[key]) for r in sel]
            return round(sum(vals) / len(vals), 3)

        walls = [r["wall_s"] for r in sel]
        med, p90 = stats.median_p90(walls)
        ex_s = [s for r in sel for s in r["scout_explore_s"]]
        summary["arms"][a] = {
            "recall5": mean("recall5"),
            "recall12": mean("recall12"),
            "precision12": mean("precision12"),
            "n_reported": mean("n_reported"),
            "delegation_rate": f"{sum(r['delegated'] for r in sel)}/{len(sel)}",
            "scout_tool_rate": f"{sum(r['scout_tool_called'] for r in sel)}/{len(sel)}",
            "wall_median_s": round(med, 1),
            "wall_p90_s": round(p90, 1),
            "scout_explore_median_s": round(stats.median_p90(ex_s)[0], 1) if ex_s else None,
            "muse_usd_per_run": mean("muse_usd"),
            "parent_model": sorted({r["parent_model"] for r in sel}),
        }
    pairs = [(SCOUT_ARM, STOCK), (SCOUT_ARM, EXPL)]
    for a, b in pairs:
        if a in by and b in by:
            c = {}
            for metric in ("recall5", "recall12", "precision12"):
                d = stats.paired_delta(
                    {t: by[a][t][metric] for t in common}, {t: by[b][t][metric] for t in common}
                )
                m, lo, hi = stats.bootstrap_ci(list(d.values()))
                cl = stats.by_cluster(d, {t: by[a][t]["repo"] for t in d})
                cm, clo, chi = stats.cluster_bootstrap_ci(cl)
                c[metric] = {
                    "delta": round(m, 3),
                    "ci": [round(lo, 3), round(hi, 3)],
                    "repo_cluster_ci": [round(clo, 3), round(chi, 3)],
                }
            summary["contrasts"][f"{a} minus {b}"] = c
    summary["per_task"] = [
        {
            "task": t[-10:],
            "repo": by[arms_present[0]][t]["repo"],
            **{
                a: {k: by[a][t][k] for k in ("recall12", "wall_s", "delegated", "scout_explore_calls")}
                for a in arms_present
            },
        }
        for t in common
    ]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))
    bad = [r for r in load_rows(out) if r["problem"]]
    print(f"invalid rows kept out: {len(bad)}", [(r["task"][-6:], r["arm"], r["problem"]) for r in bad][:10])


def main() -> None:
    ap = argparse.ArgumentParser(description="Muse Code retrieval-spec ablation on the cc50 TUNE tasks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "sync"):
        r = sub.add_parser(name)
        r.add_argument("--key-file", default=None)
        r.add_argument("--first", type=int, default=0)
        r.add_argument("--ids", default="", help="comma list of task id suffixes (last 10 chars)")
        r.add_argument("--arms", default=",".join(ARMS))
        r.add_argument("--width", type=int, default=2)
        r.add_argument(
            "--voyage-cap",
            type=float,
            default=8.0,
            help="stop when embedded tokens cost more than this (USD)",
        )
    rep = sub.add_parser("report")
    rep.add_argument("--out", default=None)
    args = ap.parse_args()
    {"run": cmd_run, "sync": cmd_sync, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
    sys.exit(0)
