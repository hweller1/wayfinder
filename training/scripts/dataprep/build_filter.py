"""data/task_filter.json: A1 acceptance flags + per-task 35B/Sonnet recall for the 1,660 public tasks.

Runs inside the Modal CPU sandbox (stdlib only). Inputs under --inputs (public files only):
  splits.json, shards_pub/pub_s*.json, modal_pub/v3_pub_{b2,c,g}_sophisticated_agents-a1-35b_merged.jsonl,
  sonnet_pubnew/v3_pubnew_{b2,c}_sophisticated_sonnet-5.5-high_checkpoint.jsonl,
  v3_a_probe_public_checkpoint.jsonl, no_retrieval_required.json

Flag definitions (PRD section 5):
  verifiable        every gold file exists at base_commit (not gold_new_at_base). n_gold_absent is recorded; a task whose
                    gold includes a file the commit creates can never reach all-gold-found, so it is not RL-usable.
  one_shot          PROXY for "a single grep of the commit subject already finds all gold": the 35B b2 run found all gold
                    (scored recall == 1.0) with <= 2 tool calls in total (sum of tool_call_counts). We do not have the
                    subject-grep result itself; this is the closest recorded signal. process_informative = not one_shot.
  contaminated      closed-book probe (arm a, no tools, 421 of 1,660 tasks probed): the answer text names a gold path
                    (full repo-relative path on a path boundary). None = task not probed. probe_basename_hit is a looser
                    secondary signal (basename of a gold file, >= 8 chars, appears in the answer) reported but not used.
                    no_retrieval_required.json has no v3 task ids (erb + external benchmarks only), so it adds no flags.
  verifiable_src    relaxed variant: every NON-test gold file exists at base (the commit may create its own tests, which are
                    the held-out tests in the E2E setup). accepted_relaxed = accepted with verifiable_src in place of verifiable.
  unambiguous       >= 1 gold file that is neither a test file nor a generated/vendored/lock file (path heuristics in
                    common.py: test dirs, *_test.*, *.spec.*, conftest; vendor, node_modules, *.pb.go, *.lock, *.min.js ...).
  accepted          verifiable and not one_shot and contaminated is not True and unambiguous.
                    (Evidence-covering is a trajectory-level criterion, applied when filtering transcripts.)
  mixed_outcome     for RL selection: among the scored arms (35B b2, c, g and Sonnet b2) at least one found all gold
                    (recall == 1.0) and at least one did not. Arms are the proxy for rollout variance.
  rl_eligible       accepted and mixed_outcome.
Recall = scored_recall as in voyage-scout/eval/build.py (score_v4.scored_recall): file-level, None when unscored.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import is_generated_path, is_test_path, leaky, load_public_tasks, load_rows, scored_recall


def path_in_text(path: str, text: str) -> bool:
    return re.search(r"(?<![\w./-])" + re.escape(path) + r"(?![\w-])", text) is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    I = a.inputs

    tasks = load_public_tasks(I / "shards_pub")
    splits = json.load((I / "splits.json").open())
    task_split = {tid: rec["split"] for tid, rec in splits["tasks"].items()}
    assert set(task_split) == set(tasks)

    arms = {
        "b2": load_rows(I / "modal_pub/v3_pub_b2_sophisticated_agents-a1-35b_merged.jsonl"),
        "c": load_rows(I / "modal_pub/v3_pub_c_sophisticated_agents-a1-35b_merged.jsonl"),
        "g": load_rows(I / "modal_pub/v3_pub_g_sophisticated_agents-a1-35b_merged.jsonl"),
    }
    son = {x: load_rows(I / f"sonnet_pubnew/v3_pubnew_{x}_sophisticated_sonnet-5.5-high_checkpoint.jsonl") for x in ("b2", "c")}
    probe = load_rows(I / "v3_a_probe_public_checkpoint.jsonl")
    nrr = json.load((I / "no_retrieval_required.json").open())
    nrr_v3 = [t["sample_id"] for t in nrr["tasks"] if "v3__" in t["sample_id"]]

    out_tasks: dict[str, dict] = {}
    for tid, t in sorted(tasks.items()):
        key = f"cb:{tid}"
        gold = t["gold_files"]
        new = t.get("gold_new_at_base") or {}
        absent = [g for g in gold if new.get(g)]
        b2row = arms["b2"].get(key)
        calls = sum(((b2row or {}).get("record") or {}).get("tool_call_counts", {}).values()) if b2row else None
        r = {k: scored_recall(v.get(key)) for k, v in arms.items()}
        sr = {k: scored_recall(v.get(key)) for k, v in son.items()}
        verifiable = not absent
        verifiable_src = not [g for g in absent if not is_test_path(g)]
        one_shot = None if (b2row is None or r["b2"] is None) else bool(r["b2"] == 1.0 and calls <= 2)
        prow = probe.get(key)
        if prow is None:
            contaminated, base_hit = None, None
        else:
            ans = (prow.get("record") or {}).get("answer") or ""
            contaminated = any(path_in_text(g, ans) for g in gold)
            base_hit = any(len(g.rsplit("/", 1)[-1]) >= 8 and g.rsplit("/", 1)[-1] in ans for g in gold)
        clean_gold = [g for g in gold if not is_test_path(g) and not is_generated_path(g)]
        unambiguous = bool(clean_gold)
        accepted = bool(verifiable and one_shot is not True and contaminated is not True and unambiguous)
        accepted_relaxed = bool(verifiable_src and one_shot is not True and contaminated is not True and unambiguous)
        scored = [x for x in (r["b2"], r["c"], r["g"], sr["b2"]) if x is not None]
        mixed = bool(scored) and any(x == 1.0 for x in scored) and any(x < 1.0 for x in scored)
        out_tasks[tid] = {
            "split": task_split[tid], "repo_full": t["repo_full"], "category": t["category"],
            "n_gold": len(gold), "n_gold_absent_at_base": len(absent), "n_gold_clean": len(clean_gold),
            "verifiable": verifiable, "verifiable_src": verifiable_src, "one_shot": one_shot, "b2_tool_calls": calls,
            "contaminated": contaminated, "probe_basename_hit": base_hit, "unambiguous": unambiguous,
            "accepted": accepted, "accepted_relaxed": accepted_relaxed, "mixed_outcome": mixed,
            "rl_eligible": bool(accepted and mixed), "rl_eligible_relaxed": bool(accepted_relaxed and mixed),
            "leaky_template": bool(leaky(b2row)) if b2row else None,
            "recall_35b_b2": r["b2"], "recall_35b_c": r["c"], "recall_35b_g": r["g"],
            "recall_sonnet_b2": sr["b2"], "recall_sonnet_c": sr["c"],
        }

    def tally(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "tasks": n,
            "verifiable": sum(x["verifiable"] for x in rows),
            "verifiable_src": sum(x["verifiable_src"] for x in rows),
            "one_shot": sum(x["one_shot"] is True for x in rows),
            "not_one_shot_or_unknown": sum(x["one_shot"] is not True for x in rows),
            "probed": sum(x["contaminated"] is not None for x in rows),
            "contaminated": sum(x["contaminated"] is True for x in rows),
            "unambiguous": sum(x["unambiguous"] for x in rows),
            "accepted": sum(x["accepted"] for x in rows),
            "mixed_outcome": sum(x["mixed_outcome"] for x in rows),
            "rl_eligible": sum(x["rl_eligible"] for x in rows),
            "accepted_relaxed": sum(x["accepted_relaxed"] for x in rows),
            "rl_eligible_relaxed": sum(x["rl_eligible_relaxed"] for x in rows),
            "one_shot_note": "0 means no task had 35B b2 recall 1.0 within <= 2 tool calls (min calls among all-gold-found runs: %s)" % min((x["b2_tool_calls"] for x in rows if x["recall_35b_b2"] == 1.0), default=None),
            "sonnet_b2_scored": sum(x["recall_sonnet_b2"] is not None for x in rows),
        }

    by_split = {s: tally([x for x in out_tasks.values() if x["split"] == s]) for s in ("train", "dev", "test", "eval_holdout")}
    doc = {
        "schema": "wayfinder-task-filter-v1",
        "definitions": __doc__,
        "no_retrieval_required_v3_ids": len(nrr_v3),
        "pass_counts": {"all": tally(list(out_tasks.values())), **by_split},
        "tasks": out_tasks,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["pass_counts"], indent=1))
    print(Counter(x["n_gold_absent_at_base"] > 0 for x in out_tasks.values()))


if __name__ == "__main__":
    main()
