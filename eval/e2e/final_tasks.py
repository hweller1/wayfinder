"""Build the search-task list (question, gold source files) for final_search_app.py from the manifests of one set.
  python3 eval/e2e/final_tasks.py clean|v4 out.json [ids.json]
Same fields as search_app._manifest_tasks. Run with E2E_EXP_DIR etc. set by final_env.py."""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_coder as rc, score as sc, gate as g
which, out = sys.argv[1], sys.argv[2]
keep = set(json.load(open(sys.argv[3]))) if len(sys.argv) > 3 else None
pub = {}
for p in (HERE.parents[1] / "artifacts/e2e_clean/datasets/tasks_1660.json", HERE.parents[1] / "artifacts/e2e_v4/datasets/tasks_1660.json", HERE.parents[2] / "benchmark-repo/datasets/artifacts/public/tasks_1660.json"):
    if p.exists():
        pub.update({t["task_id"]: t for t in json.load(open(p))["tasks"]})
res = []
for t in rc.pilot_tasks():
    if keep is not None and t["task_id"] not in keep:
        continue
    envp = None
    for d in ("e2e_clean", "e2e_v4", "e2e_v3"):
        p = HERE / "experiments" / d / "envs" / f"{t['task_id']}.json"
        if p.exists():
            envp = p; break
    env = json.load(open(envp)); held = [x for _, x in env["heldout_test_entries"]]
    gold_src, _ = rc.gold_source_files(t["gold_files"], held, pub.get(t["task_id"], {}).get("gold_new_at_base"))
    res.append(dict(task_id=t["task_id"], slug=t["slug"], repo=t["repo_full"], base_commit=t["base_commit"], question=t["issue_text"], gold_files=gold_src))
json.dump(res, open(out, "w"), indent=1); print(len(res), "tasks")
