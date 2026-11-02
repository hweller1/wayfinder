"""Frontier models (Sonnet 5.5 high, GPT-6.1 Sol) through an LLM API gateway in scout's own explore loop, on data/heldout_hard_v1.json.
Same harness as v3 (test_eval_app.py) and A1 (a1_eval_app.py): product prompt + convergence nudge, grep/glob/read_file/report
(+ semantic_search in the `sem` arm on the same tpuf namespaces), max_turns 40, max_seconds 3600, backfill off, `score_row`, row schema.
Only the model endpoint changes (wayfinder/rollout/frontier_chat.py). Runs on the laptop (rollouts are API-bound; grep is CPU only).

    python -m wayfinder.rollout.frontier_eval --models sonnet,gpt --outdir artifacts/heldout_hard/frontier_cal --limit-tasks 5 \
        --seeds 0 --arms grep,sem --phase calibration
    python -m wayfinder.rollout.frontier_eval --models sonnet,gpt --outdir artifacts/heldout_hard/shards --phase full

Rows: <outdir>/<model>_s0/sft_<model>_<arm>.jsonl (full) or <outdir>/sft_<model>_<arm>.jsonl (calibration, --flat). Resumable: a
(task, seed, arm) with an ok row is skipped; error rows are kept in the file and superseded by a later ok row (as score_heldout_hard).
Spend: every call is appended to --ledger (artifacts/heldout_hard/gateway_spend.jsonl) and a hard cap (--cap-usd, cumulative over the
ledger) stops new calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from wayfinder.rollout.frontier_chat import MAX_TOKENS, RATES, SETTINGS, FrontierChat, SpendCapReached, SpendGuard, _load_gateway_key
from wayfinder.rollout.run import TEACHER_NUDGE, checkout, run_rollout
from wayfinder.rollout.score import score_row

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"sonnet": "claude-sonnet-5-5", "gpt": "gpt-6.1-sol"}
MAX_SECONDS = 3600


def lift_budget_cap(max_seconds: int) -> None:
    """scout's Budget caps max_seconds at 900; lift it at runtime as the v3/A1 apps do (scout source untouched)."""
    from pydantic import Field
    from pydantic.fields import FieldInfo
    from scout.core.contracts.types import Budget, ExploreRequest

    Budget.model_fields["max_seconds"] = FieldInfo.from_annotated_attribute(int, Field(default=120, ge=5, le=100000))
    Budget.model_rebuild(force=True)
    ExploreRequest.model_rebuild(force=True)
    Budget(max_turns=40, max_seconds=max_seconds)


def out_file(outdir: Path, model: str, arm: str, flat: bool) -> Path:
    d = outdir if flat else outdir / f"{model}_s0"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"sft_{model}_{arm}.jsonl"


def load_done(outdir: Path, models: list[str], arms: list[str], flat: bool) -> set[tuple[str, str, int, str]]:
    done = set()
    for m in models:
        for a in arms:
            fp = out_file(outdir, m, a, flat)
            if fp.exists():
                for ln in fp.read_text().splitlines():
                    if ln.strip():
                        r = json.loads(ln)
                        if not r.get("error") and not r.get("infra_error"):
                            done.add((m, r["task_id"], r["seed"], a))
    return done


async def amain(a: argparse.Namespace) -> int:
    tasks = json.loads(Path(a.tasks).read_text())["tasks"]
    if a.task_ids:
        keep = json.loads(Path(a.task_ids).read_text())
        tasks = [t for t in tasks if t["task_id"] in set(keep)]
    if a.limit_tasks:
        tasks = tasks[: a.limit_tasks]
    models, arms, seeds = a.models.split(","), a.arms.split(","), [int(s) for s in a.seeds.split(",")]
    outdir = Path(a.outdir)
    done = load_done(outdir, models, arms, a.flat)
    todo = [(m, t, s, arm) for arm in arms for s in seeds for t in tasks for m in models if (m, t["task_id"], s, arm) not in done]
    print(f"{len(todo)} rollouts to run, {len(done)} done, tasks={len(tasks)}", flush=True)
    key = _load_gateway_key()
    guard = SpendGuard(Path(a.ledger), a.cap_usd, headroom_usd=a.headroom_usd, phase=a.phase)
    print(f"ledger total ${guard.total:.2f} of cap ${a.cap_usd:.0f}", flush=True)
    lift_budget_cap(MAX_SECONDS)
    sems = {m: asyncio.Semaphore(a.concurrency) for m in models}
    clones: dict[str, asyncio.Future] = {}
    files = {(m, arm): out_file(outdir, m, arm, a.flat).open("a") for m in models for arm in arms}
    n = {"done": 0, "err": 0}
    t_start = time.time()

    async def get_root(t: dict) -> tuple[str, float]:
        k = f"{t['repo']}@{t['base_commit']}"
        if k not in clones:
            clones[k] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], a.cache))
        return await clones[k]

    async def one(m: str, t: dict, seed: int, arm: str) -> None:
        async with sems[m]:
            if guard.stopped:
                return
            base = {"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "arm": arm, "model": MODELS[m]}
            try:
                root, clone_s = await get_root(t)
            except Exception as e:  # noqa: BLE001
                row = {**base, "error": f"checkout: {e}"[:300], "infra_error": True}
            else:
                chat = FrontierChat(MODELS[m], guard=guard, key=key, seed=seed, tag={"task_id": t["task_id"], "seed": seed, "arm": arm})
                try:
                    row = await asyncio.wait_for(
                        run_rollout(t, chat, root, seed=seed, max_turns=40, max_seconds=MAX_SECONDS, nudge=TEACHER_NUDGE,
                                    semantic="off" if arm == "grep" else "auto"), timeout=MAX_SECONDS + 600)
                    row.update(score_row(row))
                    row["arm"] = arm
                    row["clone_s"] = round(clone_s, 1)
                    row["meta"] = {"endpoint": "llm_key", "model_id": MODELS[m], "settings": SETTINGS[MODELS[m]], "rates_usd_per_m": dict(zip(
                        ("fresh", "cache_read", "cache_write", "out"), RATES[MODELS[m]])), "max_tokens": MAX_TOKENS, "phase": a.phase,
                        "billed_tokens": chat.buckets, "cost_basis": "billed tokens x rate table (the LLM gateway returns no price)"}
                    row["usage"] = {**row["usage"], "billed": chat.buckets}
                    row["cost_usd"] = round(chat.cost, 6)
                    row["retain_reasoning"] = True
                    row["weights"] = f"llm_key:{MODELS[m]}"
                    row["temperature"] = row["top_p"] = row["top_k"] = row["presence_penalty"] = None
                    row["enable_thinking"] = True
                except SpendCapReached as e:
                    row = {**base, "error": f"spend cap: {e}"[:300], "infra_error": True}
                except Exception as e:  # noqa: BLE001
                    row = {**base, "error": f"harness: {type(e).__name__}: {e}"[:300], "infra_error": True}
            if row.get("error") and "spend cap" in str(row["error"]):
                return
            f = files[(m, arm)]
            f.write(json.dumps(row) + "\n")
            f.flush()
            n["done"] += 1
            n["err"] += bool(row.get("error"))
            print(f"[{n['done']}/{len(todo)}] {m} {arm} {t['task_id'][-30:]} s{seed} turns={row.get('turns')} wall={row.get('wall_s')} "
                  f"recall={row.get('recall_reported')} cost=${row.get('cost_usd')} err={str(row.get('error'))[:80]} | llm_key ${guard.total:.2f} "
                  f"errs={n['err']} t={time.time() - t_start:.0f}s", flush=True)

    jobs = [asyncio.ensure_future(one(*x)) for x in todo]
    try:
        await asyncio.gather(*jobs)
    finally:
        for f in files.values():
            f.close()
    print(f"finished: {n['done']} rows, {n['err']} errors, llm_key total ${guard.total:.2f}, by model {guard.by_model}, stopped={guard.stopped}", flush=True)
    return 3 if guard.stopped else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "data/heldout_hard_v1.json"))
    ap.add_argument("--task-ids", default="", help="JSON list of task ids to restrict to (calibration set)")
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--models", default="sonnet,gpt")
    ap.add_argument("--arms", default="grep,sem")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--flat", action="store_true", help="calibration layout: rows directly in outdir")
    ap.add_argument("--concurrency", type=int, default=20, help="per model")
    ap.add_argument("--cache", default="/tmp/wf_repos")
    ap.add_argument("--ledger", default=str(ROOT / "artifacts/heldout_hard/gateway_spend.jsonl"))
    ap.add_argument("--cap-usd", type=float, default=450.0)
    ap.add_argument("--headroom-usd", type=float, default=8.0)
    ap.add_argument("--phase", default="full")
    a = ap.parse_args()
    sys.exit(asyncio.run(amain(a)))


if __name__ == "__main__":
    main()
