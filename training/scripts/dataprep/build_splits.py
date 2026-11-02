"""data/splits.json: repo-level split of the public tasks.

  * the repos of voyage-scout/eval/eval_tasks.jsonl go to "eval_holdout" entirely (every public task in those repos);
  * the remaining repos are assigned train 80% / dev 10% / test 10% BY REPO (repo counts), deterministic (seed 20261026);
  * stratified: initial assignment cycles through language strata, then a seeded swap search (repo counts fixed) minimises
    the L1 gap between each split's task-level language and category mix and the global mix, plus the gap between the
    split's task share and its target share.

    python build_splits.py --inputs DIR --out data/splits.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import load_public_tasks, task_language

SEED = 20261026
TARGET = {"train": 0.8, "dev": 0.1, "test": 0.1}
SPLITS = ("train", "dev", "test")


def repo_counts(n: int) -> dict[str, int]:
    dev = round(n * TARGET["dev"])
    test = round(n * TARGET["test"])
    return {"train": n - dev - test, "dev": dev, "test": test}


def mix(tasks: list[dict], key) -> dict[str, float]:
    c = Counter(key(t) for t in tasks)
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def l1(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tasks = load_public_tasks(args.inputs / "shards_pub")
    for t in tasks.values():
        t["_lang"] = task_language(t["gold_files"])
    eval_repos = {json.loads(l)["repo"] for l in (args.inputs / "eval_tasks.jsonl").open() if l.strip()}
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in tasks.values():
        by_repo[t["repo_full"]].append(t)
    assert eval_repos <= set(by_repo), f"eval repos missing from public pool: {eval_repos - set(by_repo)}"
    pool = sorted(r for r in by_repo if r not in eval_repos)

    # repo-level language = majority task language (ties: alphabetical)
    repo_lang = {r: sorted(Counter(t["_lang"] for t in by_repo[r]).items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for r in pool}
    quotas = repo_counts(len(pool))
    rng = random.Random(SEED)

    # stratified initial assignment: shuffle within language, deal into a repeating 8/1/1 pattern per stratum offset
    assign: dict[str, str] = {}
    order = sorted(pool, key=lambda r: (repo_lang[r], hashlib.sha256(f"{SEED}:{r}".encode()).hexdigest()))
    remaining = dict(quotas)
    pattern = ["train"] * 8 + ["dev", "test"]
    i = 0
    for r in order:
        for _ in range(len(pattern)):
            s = pattern[i % len(pattern)]
            i += 1
            if remaining[s] > 0:
                assign[r] = s
                remaining[s] -= 1
                break
    assert Counter(assign.values()) == Counter(quotas), (Counter(assign.values()), quotas)

    all_pool_tasks = [t for r in pool for t in by_repo[r]]
    g_lang = mix(all_pool_tasks, lambda t: t["_lang"])
    g_cat = mix(all_pool_tasks, lambda t: t["category"])
    total = len(all_pool_tasks)

    def cost(a: dict[str, str]) -> float:
        c = 0.0
        for s in SPLITS:
            ts = [t for r in pool if a[r] == s for t in by_repo[r]]
            c += l1(mix(ts, lambda t: t["_lang"]), g_lang) + l1(mix(ts, lambda t: t["category"]), g_cat)
            c += 3.0 * abs(len(ts) / total - TARGET[s])
        return c

    best = cost(assign)
    start = best
    for _ in range(6000):
        a, b = rng.sample(pool, 2)
        if assign[a] == assign[b]:
            continue
        assign[a], assign[b] = assign[b], assign[a]
        c = cost(assign)
        if c < best - 1e-12:
            best = c
        else:
            assign[a], assign[b] = assign[b], assign[a]

    split_repos: dict[str, list[str]] = {s: sorted(r for r in pool if assign[r] == s) for s in SPLITS}
    split_repos["eval_holdout"] = sorted(eval_repos)
    split_tasks = {s: sorted(t["task_id"] for r in rs for t in by_repo[r]) for s, rs in split_repos.items()}

    # ---- invariants ----
    seen: dict[str, str] = {}
    for s, rs in split_repos.items():
        for r in rs:
            assert r not in seen, f"repo {r} in two splits: {seen[r]} and {s}"
            seen[r] = s
    assert set(seen) == set(by_repo)
    all_ids = [i for ids in split_tasks.values() for i in ids]
    assert len(all_ids) == len(set(all_ids)) == len(tasks) == 1660, (len(all_ids), len(tasks))
    assert all(t["repo_full"] in eval_repos for i in split_tasks["eval_holdout"] for t in [tasks[i]])

    def stats(s: str) -> dict:
        ts = [tasks[i] for i in split_tasks[s]]
        return {
            "repos": len(split_repos[s]),
            "tasks": len(ts),
            "task_share": round(len(ts) / len(tasks), 4),
            "language_mix": {k: round(v, 3) for k, v in sorted(mix(ts, lambda t: t["_lang"]).items())},
            "category_mix": {k: round(v, 3) for k, v in sorted(mix(ts, lambda t: t["category"]).items())},
        }

    out = {
        "schema": "wayfinder-splits-v1",
        "seed": SEED,
        "method": (
            "Repo-level. eval_holdout = the 10 repos of voyage-scout/eval/eval_tasks.jsonl (all their public tasks). Other repos: "
            "train/dev/test by repo count 80/10/10; language strata dealt round-robin, then a seeded swap search (6000 proposals, "
            "repo counts fixed) minimising L1 distance of each split's task-level language and category mix to the global mix "
            "plus 3x the gap between task share and target share. Language = majority gold-file extension per task, majority "
            "over a repo's tasks for the repo-level stratum."
        ),
        "source": "benchmark-repo/artifacts/shards_pub/pub_s0..7.json (key 'tasks'), 1660 tasks, 316 repos",
        "strat_cost": {"initial": round(start, 4), "final": round(best, 4)},
        "counts": {s: {"repos": len(split_repos[s]), "tasks": len(split_tasks[s])} for s in (*SPLITS, "eval_holdout")},
        "stats": {s: stats(s) for s in (*SPLITS, "eval_holdout")},
        "tasks": {
            i: {
                "task_id": i, "split": seen[tasks[i]["repo_full"]], "repo_full": tasks[i]["repo_full"],
                "base_commit": tasks[i]["base_commit"], "source_commit": tasks[i]["source_commit"],
                "task_text": tasks[i]["task_text"], "gold_files": tasks[i]["gold_files"], "category": tasks[i]["category"],
                "language": tasks[i]["_lang"],
            }
            for ids in split_tasks.values() for i in ids
        },
        "splits": {s: {"repos": split_repos[s], "task_ids": split_tasks[s]} for s in (*SPLITS, "eval_holdout")},
        "repo_to_split": dict(sorted(seen.items())),
        "private_check": "pending: run scripts/dataprep/assert_no_private.py locally (private ids never leave the laptop)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=False))
    print(json.dumps(out["counts"]))
    for s in (*SPLITS, "eval_holdout"):
        print(s, out["stats"][s]["language_mix"])


if __name__ == "__main__":
    main()
