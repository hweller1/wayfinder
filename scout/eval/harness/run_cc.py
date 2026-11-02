"""Practical-prototype runner: free-parent arms cc_explore / cc_scout / cc_scout_only on the cc50 task set.

    SCOUT_EVAL_CACHE=/tmp/scout-cc50-repos uv run python -m eval.harness.run_cc --arms cc_explore,cc_scout \
        --ids-file ... --out artifacts/eval/cc50/calibration --seed 0

Arms of one task run concurrently in separate checkouts-states (cc_explore needs none). Rows append to
`<out>/rows_<arm>_s<seed>.jsonl`; a valid (task, arm, seed) row is skipped on resume.
"""
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import harness  # noqa: F401
from eval.harness import arms, repos
from eval.harness.briefs import build_brief, brief_sha
from eval.harness.run import REPO, append, gateways, keys, sync

TASKS = REPO / "eval" / "cc50_tasks.jsonl"
SPEND = threading.Lock()
REPO_LOCKS: dict[str, threading.Lock] = {}  # one task at a time per repo checkout (the mbedtls collision)


def run_one(variant, task, root, key, gw, out, seed, budget, timeout):
    arm, overrides = arms.VARIANTS.get(variant, (variant, {}))
    brief = build_brief(task["question"])
    files = repos.ls_files(root)
    llm_key = arms.gateway_key()
    cfg = Path("/tmp/scout-eval/cfg") / f"{task['native_id'][-10:]}-{variant}-s{seed}"
    (rt := repos.state_dir(root) / "spec_resolved.json").unlink(missing_ok=True)  # a spec arm's file must not leak into another arm
    scoutless = arm in arms.NO_SCOUT_ARMS
    if arm in arms.SPEC_ARMS:
        spec_file = arms.write_spec_file(cfg, task, overrides=overrides)
        arms.prewarm_spec(root, spec_file)  # cold resolution of a big repo must not race the hook timeout, in either arm
    env = arms.build_env(arm, cfg, key=None if scoutless else key, gateway=None if scoutless else gw, llm_key=llm_key)
    parsed, wall, hook_lines, err = arms.run_session(arm, task, brief, root, cfg, env, budget=budget, timeout=timeout)
    row = arms.score_row(arm, task, brief, parsed, files, str(root), wall, hook_lines)
    row.update(variant=variant, seed=seed, stderr_tail=err.replace(llm_key, "<g>") if llm_key else err, gateway=gw if arm != arms.CC_EXPLORE else None,
               hook_log=[ln[:140] for ln in hook_lines][-6:], problem=arms.valid_row(row), repo_type=task.get("repo_type"),
               gold=task["gold_files"])
    if arm in arms.SPEC_ARMS:
        row["spec_layers"] = arms.spec_layers(cfg, arm)
        if arm == arms.CC_EXPLORE_SPEC:  # the hooks run inside Claude Code; examples were mined before base_commit (arms.write_spec_file)
            row["problem"] = row["problem"] or (None if row["spec_layers"]["session_context"] else "spec session context did not fire")
    rt.unlink(missing_ok=True)
    arms.kill_watchers(cfg); arms.cleanup_config(cfg)
    append(out / f"rows_{variant}_s{seed}.jsonl", row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="cc_explore,cc_scout")
    ap.add_argument("--ids", default="", help="comma list of task id suffixes (last 10 chars)")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--split", default="", help="TUNE or CONFIRM: only that side of the pre-registered split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--transcripts", action="store_true", help="store raw stream-json per run under <out>/transcripts")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.transcripts: arms.TRANSCRIPTS = out / "transcripts"
    tasks = [json.loads(x) for x in TASKS.read_text().splitlines() if x.strip()]
    if a.ids: tasks = [t for t in tasks if t["native_id"][-10:] in a.ids.split(",")]
    if a.split:
        sp = json.loads((arms.REPO / "eval" / "cc50_split.json").read_text())
        tasks = [t for t in tasks if t["native_id"] in set(sp[a.split.lower()])]
    if a.first: tasks = tasks[: a.first]
    arm_list = a.arms.split(",")
    gw = gateways()["v3"]; ks = keys()
    done = set()
    for arm in arm_list:
        p = out / f"rows_{arm}_s{a.seed}.jsonl"
        if p.exists():
            done |= {(json.loads(l)["task_id"], arm) for l in p.read_text().splitlines() if not json.loads(l).get("problem")}
    jobs = []
    for i, t in enumerate(tasks):
        todo = [x for x in arm_list if (t["native_id"], x) not in done]
        if todo: jobs.append((i, t, todo))

    def task_job(j):
        with REPO_LOCKS.setdefault(j[1]["repo"], threading.Lock()):
            return _task_job(j)

    def _task_job(j):
        i, t, todo = j
        root = repos.checkout(t["repo"], t["base_commit"])  # same checkout cache the index used (clone_id lives in .git/scout)
        key = ks[i % len(ks)]
        if any(arms.VARIANTS.get(x, (x,))[0] not in arms.NO_SCOUT_ARMS for x in todo):
            sync(root, key, gw)  # incremental: refreshes status.json / gateway_ok_at; nothing new to embed
        res = {}
        for x in todo:  # sequential within a task: arms share one checkout and its scout state dir
            r = run_one(x, t, root, key, gw, out, a.seed, a.budget, a.timeout)
            sys.stderr.write(f"{t['repo'].split('/')[-1]:>20} {t['native_id'][-6:]} {x:>14} s{a.seed} r12={r['recall12']:.2f} n={r['n_reported']} wall={r['wall_s']:.0f}s cost={r['cost_usd']} scout={r['scout_tool_calls_any']} expl={r['explore_agent_calls']} sc_agent={r['scout_agent_calls']} prob={r['problem']}\n")
    with ThreadPoolExecutor(a.width) as ex:
        list(ex.map(task_job, jobs))


if __name__ == "__main__":
    main()
