"""Local concurrent runner (same harness as modal_app; used when no Modal secret exists for the endpoint).

    python -m wayfinder.rollout.local_runner --tasks T.json --out O.jsonl --seeds 0,1 --concurrency 16 [--budget-usd 14]

Rollouts are API-bound (grep on a depth-1 checkout is milliseconds), so a laptop at concurrency 16-32 is fine for
the teacher test. Resumable (skips finished task/seed rows), stops launching new rollouts when the running cost
reaches --budget-usd, writes each row as it finishes. Endpoint from env: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY /
LLM_AUTH_SCHEME (defaults: Baseten Spark Open with BASETEN_API_KEY).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from wayfinder.rollout.chat import SPARK_OPEN_BASETEN, CompatChat, Sampling, TokenGate
from wayfinder.rollout.run import TEACHER_NUDGE, checkout, run_rollout
from wayfinder.rollout.score import score_row


GATE: TokenGate | None = None
GATE_UNCACHED = False


def chat_from_env(seed: int) -> CompatChat:
    c = CompatChat(
        os.environ.get("LLM_BASE_URL", SPARK_OPEN_BASETEN["base_url"]),
        os.environ.get("LLM_API_KEY") or os.environ["BASETEN_API_KEY"],
        os.environ.get("LLM_MODEL", SPARK_OPEN_BASETEN["model"]),
        auth_scheme=os.environ.get("LLM_AUTH_SCHEME", SPARK_OPEN_BASETEN["auth_scheme"]),
        sampling=Sampling(0.85, 0.95),
        seed=seed,
        gate=GATE,
        retain_reasoning=False,  # teacher collection: keep the data distribution of the earlier passes
    )
    c.gate_uncached_only = GATE_UNCACHED
    return c


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--cache", default="/tmp/wf_repos")
    ap.add_argument("--budget-usd", type=float, default=1e9)
    ap.add_argument("--tpm", type=int, default=850_000, help="shared prompt-token/min cap (Baseten limit 1M)")
    ap.add_argument("--gate-uncached-only", action="store_true", help="charge the TPM gate only for uncached prompt + completion tokens")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--max-seconds", type=int, default=300)
    ap.add_argument("--semantic", choices=["on", "off", "auto"], default="auto",
                    help="offer semantic_search when repo@commit is indexed (auto: fall back to grep-only, recorded)")
    ap.add_argument("--nudge", action="store_true", help="append run.TEACHER_NUDGE")
    a = ap.parse_args()
    global GATE, GATE_UNCACHED
    GATE_UNCACHED = a.gate_uncached_only
    GATE = TokenGate(a.tpm) if a.tpm > 0 else None
    tasks = json.loads(Path(a.tasks).read_text())
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    spent = 0.0
    if out.exists():
        for ln in out.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                spent += r.get("cost_usd") or 0
                if not r.get("infra_error"):
                    done.add((r["task_id"], r["seed"]))
    jobs = [(t, s) for s in seeds for t in tasks if (t["task_id"], s) not in done]
    print(f"{len(jobs)} rollouts to run, {len(done)} done, ${spent:.2f} already in file", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    state = {"spent": spent, "n": 0, "stopped": False}
    clones: dict[str, asyncio.Future] = {}
    fh = out.open("a")

    async def get_root(t: dict) -> tuple[str, float]:
        key = f"{t['repo']}@{t['base_commit']}"
        if key not in clones:
            clones[key] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], a.cache))
        return await clones[key]

    async def one(t: dict, s: int) -> None:
        async with sem:
            if state["spent"] >= a.budget_usd:
                state["stopped"] = True
                return
            try:
                root, clone_s = await get_root(t)
            except Exception as e:  # noqa: BLE001
                row = {"task_id": t["task_id"], "repo": t["repo"], "seed": s, "error": f"checkout: {e}"[:300], "infra_error": True}
            else:
                chat = chat_from_env(s)
                try:
                    row = await run_rollout(t, chat, root, seed=s, max_turns=a.max_turns, max_seconds=a.max_seconds,
                                        semantic=a.semantic, nudge=TEACHER_NUDGE if a.nudge else None)
                    row.update(score_row(row))
                    row["clone_s"] = round(clone_s, 1)
                except Exception as e:  # noqa: BLE001
                    row = {"task_id": t["task_id"], "repo": t["repo"], "seed": s, "error": f"harness: {type(e).__name__}: {e}"[:300], "infra_error": True}
            async with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                state["spent"] += row.get("cost_usd") or 0
                state["n"] += 1
                print(f"[{state['n']}/{len(jobs)}] {row['task_id'][-40:]} s{s} turns={row.get('turns')} wall={row.get('wall_s')} "
                      f"${row.get('cost_usd')} recall={row.get('recall')} err={row.get('error')} | total ${state['spent']:.2f}", flush=True)

    await asyncio.gather(*[one(t, s) for t, s in jobs])
    fh.close()
    print(f"done n={state['n']} spent=${state['spent']:.3f} budget_stop={state['stopped']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
