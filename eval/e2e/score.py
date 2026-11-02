"""E2E scorer. No LLM.

  python scripts/e2e/score.py --task <task_id> --diff agent.patch      # score an agent's diff
  python scripts/e2e/score.py --task <task_id> --worktree <dir>        # score an existing working tree
  python scripts/e2e/score.py --self-check [--task <task_id>] [--sandbox]  # --sandbox: tests run in Docker, no network          # real non-test diff must pass, empty must fail

Procedure: fresh worktree at base_commit -> apply the agent's diff (hunks touching held-out test paths are
dropped: those files are overwritten anyway) -> overwrite/add the held-out test files from source_commit ->
run the recorded test units in the recorded env. Output JSON:
  {pass, new_tests_passed, new_tests_total, regressions, files_touched, ...}
pass = every fail-to-pass test passes AND no regression-set test fails/disappears.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gate as g  # noqa: E402

MODAL = False
SANDBOX = None   # set by --sandbox: every test command then runs in a Docker container (scripts/e2e/sandbox.py)


def enable_sandbox():
    global SANDBOX
    import sandbox
    if not sandbox.daemon_up():
        raise SystemExit("--sandbox: Docker daemon is not running; start Docker Desktop and re-run (nothing was executed).")
    SANDBOX = sandbox.Sandbox()
    g.EXEC = SANDBOX.exec


def enable_modal():
    """--modal: every test command runs in a Modal Sandbox (block_network=True) via scripts/e2e/modal_sandbox.py."""
    global SANDBOX, MODAL
    import modal_sandbox
    MODAL = True
    SANDBOX = modal_sandbox.ModalSandbox()
    g.EXEC = SANDBOX.exec


def load_env(task_id):
    return json.load(open(g.EXP / "envs" / f"{task_id}.json"))


def filter_patch(patch: str):
    """Drop per-file sections of a git patch that touch held-out test paths. Returns (patch, files_touched)."""
    sections, cur = [], []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git ") and cur:
            sections.append(cur)
            cur = []
        cur.append(line)
    if cur:
        sections.append(cur)
    keep, touched = [], []
    for s in sections:
        if not s[0].startswith("diff --git "):
            keep += s
            continue
        path = s[0].split(" b/", 1)[-1].strip()
        if g.is_test_path(path):
            continue
        touched.append(path)
        keep += s
    return "".join(keep), touched


def score_worktree(task, env, wd, files_touched):
    cfg = env["config"]
    lang = env["language"]
    if SANDBOX is not None:
        SANDBOX.use(env)
    repo = g.DATA / "repos" / g.repo_slug(task["repo_full"])
    venv = (g.ROOT / env["venv"]) if lang == "Python" else None
    tests = [tuple(x) for x in env["heldout_test_entries"]]
    g.apply_heldout(repo, wd, task, tests)
    res, elapsed, tail = g.run_tests(task, cfg, wd, venv, env["run_units"])
    f2p = env["fail_to_pass"]
    passed = [k for k in f2p if res.get(k, {}).get("status") == "pass"]
    regs = [k for k in env["regression_set"] if res.get(k, {}).get("status") != "pass"]
    return dict(task_id=task["task_id"], **{"pass": len(passed) == len(f2p) and not regs},
                new_tests_passed=len(passed), new_tests_total=len(f2p), regressions=len(regs),
                regression_total=len(env["regression_set"]), regressed_tests=regs[:20],
                files_touched=files_touched, test_seconds=round(elapsed, 1),
                note="" if res else "no results (build/collect error or timeout): " + tail[-300:])


def score_patch(task_id, patch_text):
    tasks = g.load_tasks()
    task = tasks[task_id]
    env = load_env(task_id)
    # Unique per call: concurrent runs of the same task (round-2 screen, 2026-10-24) shared "score_<task>" and
    # deleted each other's checkout mid-write.
    repo, wd = g.make_worktree(task, f"score_{g.short(task)}_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    try:
        patch, touched = filter_patch(patch_text)
        if patch.strip():
            with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
                f.write(patch if patch.endswith("\n") else patch + "\n")
            p = g.git(wd, "apply", "--whitespace=nowarn", f.name)
            Path(f.name).unlink(missing_ok=True)
            if p.returncode:
                return dict(task_id=task_id, **{"pass": False}, new_tests_passed=0, new_tests_total=len(env["fail_to_pass"]),
                            regressions=None, files_touched=touched, note="patch_apply_failed: " + p.stderr[-300:])
        return score_worktree(task, env, wd, touched)
    finally:
        g.drop_worktree(repo, wd)


def score_dir(task_id, src_dir):
    """Score an existing agent working tree: its diff vs base_commit (incl. untracked) is re-applied in a clean worktree."""
    task = g.load_tasks()[task_id]
    g.git(src_dir, "add", "-N", ".")
    d = g.git(src_dir, "diff", "--binary", task["base_commit"]).stdout
    return score_patch(task_id, d)


def real_nontest_patch(task, env):
    repo = g.DATA / "repos" / g.repo_slug(task["repo_full"])
    paths = [p for _, p in env["nontest_entries"]]
    return g.git(repo, "diff", "--binary", "-M", task["base_commit"], task["source_commit"], "--", *paths, check=True).stdout


def self_check(task_ids):
    tasks = g.load_tasks()
    out = []
    for tid in task_ids:
        env = load_env(tid)
        real = score_patch(tid, real_nontest_patch(tasks[tid], env))
        empty = score_patch(tid, "")
        ok = real["pass"] is True and empty["pass"] is False
        out.append(dict(task_id=tid, ok=ok, real=real, empty=empty))
        print(tid[-30:], "SELF-CHECK", "OK" if ok else "FAIL", "real:", real["new_tests_passed"], "/", real["new_tests_total"], "reg", real["regressions"],
              "| empty:", empty["new_tests_passed"], "/", empty["new_tests_total"], flush=True)
    (g.EXP / ("scorer_selfcheck_modal.json" if MODAL else "scorer_selfcheck_sandbox.json" if SANDBOX is not None else "scorer_selfcheck.json")).write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--diff")
    ap.add_argument("--worktree")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--sandbox", action="store_true", help="(superseded) run test execution in local Docker (--network none)")
    ap.add_argument("--modal", action="store_true", help="run all test execution in Modal Sandboxes (network blocked); the supported path")
    a = ap.parse_args()
    if a.modal:
        enable_modal()
    if a.sandbox:
        enable_sandbox()
    if a.self_check:
        ids = [a.task] if a.task else [x["task_id"] for x in json.load(open(g.EXP / "gate_results.json"))["results"] if x.get("gated")]
        out = self_check(ids)
        sys.exit(0 if all(o["ok"] for o in out) else 1)
    if a.worktree:
        r = score_dir(a.task, a.worktree)
    else:
        r = score_patch(a.task, Path(a.diff).read_text())
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
