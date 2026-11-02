"""Eval runner. `index` syncs every repo at its first task's commit; `run` executes (task, arm) sessions.

    uv run python -m eval.harness.run index
    uv run python -m eval.harness.run run --arms explore_default,super_v3,super_a1 --tasks 3 --out artifacts/eval/smoke
    uv run python -m eval.harness.run run --out artifacts/eval            # all 50 tasks, resumable

Repos run in parallel lanes; inside a lane tasks are sequential and the three arms of one task run
concurrently, each in its own checkout (scout's state dir is per checkout, so the two scout arms never share a
readiness file). Rows are appended to `<out>/rows_<arm>.jsonl`; a (task, arm) already holding a valid row is
skipped on resume.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.harness import arms, repos
from eval.harness.briefs import build_brief

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "eval" / "eval_tasks.jsonl"
KEYS = Path.home() / ".config" / "scout" / "eval-keys.txt"
GATEWAYS = Path("/tmp/scout_gateways.json")
SCOUT = REPO / ".venv" / "bin" / "scout"
LOCK = threading.Lock()


def load_tasks(limit: int | None = None, only: list[str] | None = None) -> list[dict]:
    rows = [json.loads(x) for x in TASKS.read_text().splitlines() if x.strip()]
    if only:
        rows = [r for r in rows if r["repo"].split("/")[-1] in only]
    if limit is None:
        return rows
    out: list[dict] = []
    seen: dict[str, int] = defaultdict(int)  # smoke: the first task of the first `limit` repos
    repos_in_order = list(dict.fromkeys(r["repo"] for r in rows))[:limit]
    for r in rows:
        if r["repo"] in repos_in_order and seen[r["repo"]] < 1:
            seen[r["repo"]] += 1
            out.append(r)
    return out


def keys() -> list[str]:
    return [k for k in KEYS.read_text().split() if k]


def gateways() -> dict[str, str]:
    return json.loads(GATEWAYS.read_text())


def sync(root: Path, key: str, gateway: str) -> dict:
    env = {**os.environ, "SCOUT_KEY": key, "SCOUT_GATEWAY_URL": gateway, "SCOUT_KEYSTORE": "file"}
    res = subprocess.run(  # noqa: S603
        [str(SCOUT), "sync", "--wait", "--root", str(root)],
        env=env, capture_output=True, text=True, timeout=3600, check=False, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    out = res.stdout.replace(key, "<key>")
    try:
        status = json.loads(out)
    except ValueError:
        raise RuntimeError(f"sync failed rc={res.returncode}: {out[-300:]} {res.stderr[-200:].replace(key, '<key>')}") from None
    if status.get("state") != "ready":
        raise RuntimeError(f"sync not ready: {out[-300:]}")
    return status


def append(path: Path, row: dict) -> None:
    with LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def done_keys(out: Path, arm_list: list[str]) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    for arm in arm_list:
        p = out / f"rows_{arm}.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                row = json.loads(line)
                if arms.valid_row(row) is None and not row.get("fallthrough"):
                    done.add((row["task_id"], arm))
    return done


def one_session(arm: str, task: dict, root: Path, key: str | None, gw: str | None, out: Path, budget: float, timeout: int, attempts: int) -> dict:
    from eval.harness.briefs import brief_sha

    brief = build_brief(task["question"])
    files = repos.ls_files(root)
    llm_key = arms.gateway_key()
    row: dict = {}
    for attempt in range(1, attempts + 1):
        cfg = Path("/tmp/scout-eval/cfg") / f"{task['native_id'][-10:]}-{arm}-{attempt}"
        env = arms.build_env(arm, cfg, key=key, gateway=gw, llm_key=llm_key)
        parsed, wall, hook_lines, err = arms.run_session(arm, task, brief, root, cfg, env, budget=budget, timeout=timeout)
        row = arms.score_row(arm, task, brief, parsed, files, str(root), wall, hook_lines)
        row["attempt"] = attempt
        row["stderr_tail"] = err.replace(llm_key, "<g>") if llm_key else err
        row["gateway"] = gw
        row["hook_log"] = [ln[:120] for ln in hook_lines][-4:]
        row["problem"] = arms.valid_row(row)
        arms.kill_watchers(cfg)
        arms.cleanup_config(cfg)
        assert row["brief_sha"] == brief_sha(brief)
        if row["problem"] is None and not row["fallthrough"]:
            break
    append(out / f"rows_{arm}.jsonl", row)
    return row


def lane(repo: str, tasks: list[dict], key: str, gw: dict[str, str], out: Path, arm_list: list[str], budget: float, timeout: int, attempts: int, done: set[tuple[str, str]]) -> None:
    for t in tasks:
        todo = [a for a in arm_list if (t["native_id"], a) not in done]
        if not todo:
            continue
        main = repos.checkout(repo, t["base_commit"])
        # the index is shared by every gateway; a run of only the a20 arm syncs through its own gateway so the
        # checkout's readiness state names the gateway the session will use
        # the checkout's readiness state (gateway_ok_at) must come from the gateway the scout arms will use
        scout_arms = [a for a in todo if a != "explore_default"]
        via = {arms.A20: "a20", arms.HYBRID_A1: "a1"}
        sync(main, key, gw[via[scout_arms[0]]] if len(scout_arms) == 1 and scout_arms[0] in via else gw["v3"])
        roots = {"explore_default": main, "super_v3": main, arms.AUGMENT: main, arms.A20: main, arms.HYBRID: main,
                 arms.HYBRID_A1: main, arms.HYBRID_BF: main, arms.RELAY_BF: main}  # fmt: skip
        if "super_a1" in todo:
            second = repos.ensure_second(repo, t["base_commit"])
            repos.copy_state(main, second)
            sync(second, key, gw["a1"])
            roots["super_a1"] = second
        with ThreadPoolExecutor(len(todo)) as ex:
            futs = {a: ex.submit(one_session, a, t, roots[a], None if a == "explore_default" else key, None if a == "explore_default" else gw[{"super_a1": "a1", arms.A20: "a20", arms.HYBRID_A1: "a1"}.get(a, "v3")], out, budget, timeout, attempts) for a in todo}
            for a, f in futs.items():
                try:
                    r = f.result()
                    sys.stderr.write(f"{repo.split('/')[-1]:>14} {t['native_id'][-8:]} {a:>15} recall12={r['recall12']:.2f} wall={r['wall_s']:.0f}s cost={r['cost_usd']} ft={r['fallthrough']} prob={r['problem']}\n")
                except Exception as e:  # noqa: BLE001
                    sys.stderr.write(f"{repo} {t['native_id'][-8:]} {a} FAILED {type(e).__name__}: {str(e)[:200]}\n")


def cmd_index(args: argparse.Namespace) -> None:
    g = gateways()
    ks = keys()
    first: dict[str, dict] = {}
    for t in load_tasks():
        first.setdefault(t["repo"], t)
    if repos.disk_free_gb() < 8:
        raise SystemExit("less than 8 GB free")

    def go(i_repo: tuple[int, str]) -> tuple[str, dict]:
        i, repo = i_repo
        t = first[repo]
        root = repos.checkout(repo, t["base_commit"])
        for attempt in range(3):
            try:
                return repo, sync(root, ks[i % len(ks)], g["v3"])
            except RuntimeError as e:
                sys.stderr.write(f"sync {repo} attempt {attempt + 1}: {str(e)[:260]}\n")
        return repo, {"files": None, "coverage": "FAILED"}

    with ThreadPoolExecutor(10) as ex:
        for repo, st in ex.map(go, enumerate(first)):
            sys.stderr.write(f"indexed {repo}: files={st.get('files')} coverage={st.get('coverage')}\n")


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm_list = args.arms.split(",")
    tasks = load_tasks(args.tasks, args.repos.split(",") if args.repos else None)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_repo[t["repo"]].append(t)
    done = done_keys(out, arm_list) if args.resume else set()
    ks, g = keys(), gateways()
    with ThreadPoolExecutor(len(by_repo)) as ex:
        futs = [ex.submit(lane, repo, ts, ks[i % len(ks)], g, out, arm_list, args.budget, args.timeout, args.attempts, done) for i, (repo, ts) in enumerate(by_repo.items())]
        for f in futs:
            f.result()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    r = sub.add_parser("run")
    r.add_argument("--arms", default=",".join(arms.ARMS), help=f"any of {','.join(arms.ALL_ARMS)}")
    r.add_argument("--tasks", type=int, default=None, help="smoke: first task of the first N repos")
    r.add_argument("--out", default=str(REPO / "artifacts" / "eval"))
    r.add_argument("--repos", default="", help="comma list of repo short names")
    r.add_argument("--budget", type=float, default=1.0)
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--attempts", type=int, default=2)
    r.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()
    {"index": cmd_index, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
