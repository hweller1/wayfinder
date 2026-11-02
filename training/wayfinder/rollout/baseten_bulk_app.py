"""Bulk teacher collection against Baseten (`meta-models/muse-spark-open`), run inside Modal.

    modal run --detach wayfinder/rollout/baseten_bulk_app.py::launch --name spark_v1_baseten

Same settings as the self-hosted Stage B / bulk run (`modal_app.py`): scout product loop, grep-only, TEACHER_NUDGE,
reasoning_effort high, temperature 0.85, top_p 0.95, max_turns 40, max_seconds 900 (rate-limit waits excluded from the
clock). Rows have the same schema as /teacher/spark_v1/<task_hash>.jsonl on the Volume `wayfinder-data` and are written
to /teacher/<name>/<task_hash>.jsonl (never into spark_v1), so rows dedupe on (task_id, seed). Pairs already present in
/teacher/spark_v1 are skipped.

Order: phase 1 = RL-eligible accepted train tasks (rl_eligible), seeds 0 and 1; phase 2 = the other accepted train
tasks, seed 0. Tasks are dealt round-robin to W workers (both seeds of a task stay on one worker, so one file has one
writer); each worker runs C rollouts concurrently behind its own adaptive TokenGate holding 1/W of the account-wide
Baseten limit (1M tokens/min, cached prompt tokens count). Failed rollouts (error rows) are not saved as rows, their
cost is logged in /teacher/<name>/_failed/, and a later pass retries them. Spend cap: workers stop launching rollouts
when the run's spend (rows + failures) reaches the cap. The orchestrator writes /teacher/<name>/_progress.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
SCOUT_SRC = REPO.parent / "voyage-scout" / "src"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ripgrep", "ca-certificates")
    .pip_install("httpx==0.28.1", "pydantic>=2.9", "pathspec==1.1.1", "watchfiles==1.3.0",
                 # semantic_search (wayfinder/rollout/semantic.py): scout's real gateway/store/embedder path
                 "cryptography==50.0.2", "keyring==25.7.0", "turbopuffer==3.0.0", "voyageai==0.5.0")
    .env({"SCOUT_SRC": "/scout_src", "PYTHONPATH": "/root", "GIT_TERMINAL_PROMPT": "0"})
    .add_local_dir(str(SCOUT_SRC), "/scout_src", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)
app = modal.App("wayfinder-baseten-bulk", image=image)
data_vol = modal.Volume.from_name("wayfinder-data", create_if_missing=True)


def _load_key() -> str:
    """BASETEN_API_KEY from the environment or ../benchmark-repo/.env (never printed). Attached to the app's functions
    as an inline (non-stored) Modal secret."""
    k = os.environ.get("BASETEN_API_KEY")
    if k:
        return k
    envp = REPO.parent / "benchmark-repo" / ".env"
    if envp.exists():
        for ln in envp.read_text().splitlines():
            ln = ln.strip()
            if ln.startswith("export "):
                ln = ln[7:]
            if ln.startswith("BASETEN_API_KEY="):
                return ln.split("=", 1)[1].strip().strip("'\"")
    return "unset"  # inside the container the inline secret already supplies the real value


SECRET = modal.Secret.from_dict({"BASETEN_API_KEY": _load_key()})

CFG = {
    "model": os.environ.get("WF_BASETEN_MODEL", "meta-models/muse-spark-open-1026"), "base_url": "https://inference.baseten.co/v1", "auth_scheme": "Api-Key",
    "temperature": 0.85, "top_p": 0.95, "max_turns": 40, "max_seconds": 900, "reasoning_effort": "high",
    "prices": [1.25, 4.25, 0.15],  # USD per M tokens: fresh input, output, cached input
}


def _read_rows(out_dir: str, with_messages: bool = False) -> list[dict]:
    rows = []
    d = Path("/data") / out_dir
    if d.exists():
        for fp in d.glob("*.jsonl"):
            if fp.name.startswith("_"):
                continue
            for ln in fp.read_text().splitlines():
                if ln.strip():
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    rows.append(r if with_messages else {k: r.get(k) for k in ("task_id", "seed", "cost_usd", "all_gold_found", "all_gold_reported", "explore_result", "turns", "wall_s", "n429")})
    return rows


def _failed_cost(out_dir: str) -> tuple[float, int]:
    tot, n = 0.0, 0
    d = Path("/data") / out_dir / "_failed"
    if d.exists():
        for fp in d.glob("*.jsonl"):
            for ln in fp.read_text().splitlines():
                if ln.strip():
                    try:
                        tot += json.loads(ln).get("cost_usd") or 0
                        n += 1
                    except ValueError:
                        pass
    return tot, n


def _done_pairs(*dirs: str) -> set[tuple[str, int]]:
    data_vol.reload()
    return {(r["task_id"], int(r["seed"])) for d in dirs for r in _read_rows(d)}


@app.function(
    timeout=24 * 3600, cpu=2, memory=4096, volumes={"/data": data_vol},
    secrets=[SECRET, modal.Secret.from_name("wayfinder-index-keys")],
    retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0),
)
async def worker(wid: int, jobs: list[dict], out_dir: str, tpm: int, tpm_ceiling: int, concurrency: int,
                 cap_usd: float, run_id: str, semantic: str = "auto") -> dict:
    """jobs = [{"task": {...}, "seeds": [..]}] in priority order. Returns a summary."""
    import asyncio
    import hashlib
    import shutil
    import time

    from wayfinder.rollout.chat import CompatChat, Prices, Sampling, TokenGate
    from wayfinder.rollout.run import TEACHER_NUDGE, checkout, run_rollout
    from wayfinder.rollout.score import score_row

    spend = modal.Dict.from_name("wayfinder-baseten-spend", create_if_missing=True)
    key = os.environ["BASETEN_API_KEY"]
    gate = TokenGate(tpm, ceiling=tpm_ceiling)
    sem = asyncio.Semaphore(concurrency)
    wlock = asyncio.Lock()
    state = {"spent": 0.0, "rows": 0, "fails": 0, "stopped": False, "global": 0.0, "global_t": 0.0}
    clones: dict[str, asyncio.Future] = {}
    refs: dict[str, int] = {}
    # pairs already done (a Modal retry of this worker must not repeat finished work)
    done = {(r["task_id"], int(r["seed"])) for r in _read_rows(out_dir)}

    def global_spend() -> float:
        if time.time() - state["global_t"] > 15:
            try:
                state["global"] = sum(v for k, v in list(spend.items()) if k.startswith(run_id + ":"))
            except Exception:  # noqa: BLE001
                pass
            state["global_t"] = time.time()
        return state["global"]

    async def get_root(t: dict) -> tuple[str, float]:
        k = f"{t['repo']}@{t['base_commit']}"
        refs[k] = refs.get(k, 0) + 1
        if k not in clones:
            clones[k] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], "/tmp/repos"))
        return await clones[k]

    def release(t: dict, root: str | None) -> None:
        k = f"{t['repo']}@{t['base_commit']}"
        refs[k] -= 1
        if refs[k] <= 0 and root:
            clones.pop(k, None)
            shutil.rmtree(root, ignore_errors=True)

    async def one(task: dict, seed: int) -> None:
        if (task["task_id"], seed) in done:
            return
        thash = hashlib.sha1(task["task_id"].encode()).hexdigest()[:16]
        async with sem:
            if state["stopped"] or max(state["spent"], global_spend()) >= cap_usd:
                state["stopped"] = True
                return
            root = None
            try:
                root, clone_s = await get_root(task)
            except Exception as e:  # noqa: BLE001
                release(task, None)
                state["fails"] += 1
                await _log_failed(out_dir, wid, task, seed, f"checkout: {e}"[:300], 0.0)
                return
            chat = CompatChat(
                CFG["base_url"], key, CFG["model"], auth_scheme=CFG["auth_scheme"],
                sampling=Sampling(CFG["temperature"], CFG["top_p"]), prices=Prices(*CFG["prices"]), seed=seed,
                vllm_extensions=False, extra_body={"reasoning_effort": CFG["reasoning_effort"]},
                timeout=600.0, max_retries=10, gate=gate,
                retain_reasoning=False,  # teacher collection: keep the data distribution of the earlier passes
            )
            try:
                row = await run_rollout(task, chat, root, seed=seed, max_turns=CFG["max_turns"],
                                        max_seconds=CFG["max_seconds"], nudge=TEACHER_NUDGE, semantic=semantic)
                row.update(score_row(row))
            except Exception as e:  # noqa: BLE001
                row = {"error": f"harness: {type(e).__name__}: {e}"[:300], "cost_usd": chat.cost_usd()}
            finally:
                release(task, root)
            cost = row.get("cost_usd") or 0.0
            state["spent"] += cost
            spend[f"{run_id}:w{wid}"] = state["spent"]
            if row.get("error"):
                state["fails"] += 1
                await _log_failed(out_dir, wid, task, seed, row["error"], cost)
                return
            row["clone_s"] = round(clone_s, 1)
            row["endpoint"] = "baseten"
            row["reasoning_effort"] = CFG["reasoning_effort"]
            row["temperature"], row["top_p"] = CFG["temperature"], CFG["top_p"]
            async with wlock:
                p = Path("/data") / out_dir
                p.mkdir(parents=True, exist_ok=True)
                with (p / f"{thash}.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                await asyncio.to_thread(data_vol.commit)
            state["rows"] += 1
            done.add((task["task_id"], seed))

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(60)
            hb = {"t": time.strftime("%H:%M:%S", time.gmtime()), "wid": wid, "rows": state["rows"], "fails": state["fails"],
                  "spent": round(state["spent"], 2), "gate_limit": int(gate.limit), "gate_n429": gate.n429, "gate_ok": gate.n_ok,
                  "gate_waited_s": int(gate.waited_s), "hdr_remaining": gate._hdr[1] if gate._hdr else None}
            try:
                p = Path("/data") / out_dir / "_heartbeat"
                p.mkdir(parents=True, exist_ok=True)
                (p / f"w{wid}.jsonl").open("a").write(json.dumps(hb) + "\n")
                await asyncio.to_thread(data_vol.commit)
            except Exception:  # noqa: BLE001
                pass

    hb_task = asyncio.ensure_future(heartbeat())
    await asyncio.gather(*[one(j["task"], s) for j in jobs for s in j["seeds"]])
    hb_task.cancel()
    return {"wid": wid, "rows": state["rows"], "fails": state["fails"], "spent": round(state["spent"], 2),
            "stopped_on_cap": state["stopped"], "n429": gate.n429, "gate_limit_end": int(gate.limit)}


async def _log_failed(out_dir: str, wid: int, task: dict, seed: int, err: str, cost: float) -> None:
    import asyncio

    def w() -> None:
        p = Path("/data") / out_dir / "_failed"
        p.mkdir(parents=True, exist_ok=True)
        with (p / f"w{wid}.jsonl").open("a") as f:
            f.write(json.dumps({"task_id": task["task_id"], "seed": seed, "error": err, "cost_usd": cost}) + "\n")
        data_vol.commit()

    await asyncio.to_thread(w)


def _stats(out_dir: str, t0: float, t_start_rows: int) -> dict:
    import time

    rows = _read_rows(out_dir)
    rows_usd = sum(r.get("cost_usd") or 0 for r in rows)
    fcost, nfail = _failed_cost(out_dir)
    conv = [r for r in rows if not (r.get("explore_result") or {}).get("budget_exhausted")]
    el_h = max(time.time() - t0, 1) / 3600
    return {
        "rows": len(rows),
        "all_gold_rows": sum(1 for r in rows if r.get("all_gold_found")),
        "converged_rows": len(conv),
        "converged_all_gold_rows": sum(1 for r in conv if r.get("all_gold_found")),
        "all_gold_reported_rows": sum(1 for r in rows if r.get("all_gold_reported")),
        "usd": round(rows_usd + fcost, 2),
        "usd_rows": round(rows_usd, 2),
        "usd_failed_rollouts": round(fcost, 2),
        "failed_rollouts": nfail,
        "rate_per_hour": round((len(rows) - t_start_rows) / el_h, 1),
        "tasks_touched": len({r["task_id"] for r in rows}),
        "elapsed_min": round(el_h * 60, 1),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }


@app.function(timeout=24 * 3600, cpu=1, memory=2048, volumes={"/data": data_vol})
def orchestrator(jobs_all: list[dict], out_dir: str, n_workers: int, tpm_total: int, tpm_ceiling_total: int,
                 concurrency: int, budget_usd: float, passes: int = 3, semantic: str = "auto") -> dict:
    """Deals jobs to workers, polls them, writes _progress.json every ~60 s, repeats (failed rollouts are retried)."""
    import time

    t0 = time.time()
    data_vol.reload()
    prior_rows = len(_read_rows(out_dir))
    prior_usd = sum(r.get("cost_usd") or 0 for r in _read_rows(out_dir)) + _failed_cost(out_dir)[0]
    prog = Path("/data") / out_dir / "_progress.json"
    prog.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for ps in range(passes):
        done = _done_pairs("teacher/spark_v1", out_dir)
        todo = []
        for j in jobs_all:
            seeds = [s for s in j["seeds"] if (j["task"]["task_id"], s) not in done]
            if seeds:
                todo.append({"task": j["task"], "seeds": seeds})
        spent_now = sum(r.get("cost_usd") or 0 for r in _read_rows(out_dir)) + _failed_cost(out_dir)[0]
        cap = budget_usd - spent_now  # the budget is for the whole track: rows already in out_dir count
        print(f"pass {ps}: {len(todo)} tasks with unfinished seeds, cap remaining ${cap:.2f}", flush=True)
        if not todo or cap <= 5:
            break
        shards: list[list[dict]] = [[] for _ in range(n_workers)]
        for i, j in enumerate(todo):
            shards[i % n_workers].append(j)
        run_id = f"{int(t0)}p{ps}"
        calls = [
            worker.spawn(w, sh, out_dir, tpm_total // n_workers, tpm_ceiling_total // n_workers, concurrency, cap, run_id, semantic)
            for w, sh in enumerate(shards) if sh
        ]
        pending = list(calls)
        while pending:
            time.sleep(60)
            data_vol.reload()
            st = _stats(out_dir, t0, prior_rows)
            st["pass"] = ps
            st["workers_running"] = len(pending)
            prog.write_text(json.dumps(st, indent=1))
            data_vol.commit()
            still = []
            for c in pending:
                try:
                    summaries.append(c.get(timeout=0))
                except TimeoutError:
                    still.append(c)
                except Exception as e:  # noqa: BLE001
                    summaries.append({"error": repr(e)[:200]})
            pending = still
    data_vol.reload()
    st = _stats(out_dir, t0, prior_rows)
    st["finished"] = True
    st["worker_summaries"] = summaries
    prog.write_text(json.dumps(st, indent=1))
    data_vol.commit()
    return st


def _jobs(rl_seeds: list[int], rest_seeds: list[int]) -> list[dict]:
    flt = json.loads((REPO / "data" / "task_filter.json").read_text())["tasks"]
    spl = json.loads((REPO / "data" / "splits.json").read_text())["tasks"]

    def mk(tid: str) -> dict:
        t = spl[tid]
        return {"task_id": tid, "repo": t["repo_full"], "base_commit": t["base_commit"], "question": t["task_text"],
                "gold_files": list(t["gold_files"]), "category": t.get("category")}

    acc = [(tid, f) for tid, f in sorted(flt.items()) if f.get("accepted") and f["split"] == "train"]
    p1 = [{"task": mk(tid), "seeds": rl_seeds} for tid, f in acc if f.get("rl_eligible")]
    p2 = [{"task": mk(tid), "seeds": rest_seeds} for tid, f in acc if not f.get("rl_eligible")]
    return p1 + p2


@app.local_entrypoint()
def launch(name: str = "spark_v1_baseten", workers: int = 8, tpm: int = 900_000, tpm_ceiling: int = 1_000_000,
           concurrency: int = 12, budget_usd: float = 700.0, limit: int = 0, rl_seeds: str = "0,1", rest_seeds: str = "0",
           semantic: str = "auto"):
    """Spawn the orchestrator in Modal; use `modal run --detach`."""
    jobs = _jobs([int(s) for s in rl_seeds.split(",")], [int(s) for s in rest_seeds.split(",")])
    if limit:
        jobs = jobs[:limit]
    n_p1 = sum(1 for j in jobs if len(j["seeds"]) == len(rl_seeds.split(",")) and j["seeds"] == [int(s) for s in rl_seeds.split(",")])
    print(f"{len(jobs)} tasks ({sum(len(j['seeds']) for j in jobs)} rollouts), W={workers} C={concurrency} tpm={tpm}", flush=True)
    res = orchestrator.remote(jobs, f"teacher/{name}", workers, tpm, tpm_ceiling, concurrency, budget_usd, semantic=semantic)
    print(json.dumps({k: v for k, v in res.items() if k != "worker_summaries"}), flush=True)
