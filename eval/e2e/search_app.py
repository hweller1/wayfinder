"""Search subagents for the wayfinder E2E coding track: scout's explore loop (product profile, grep/glob/read_file/report) run on
Modal for the round-3 headroom tasks, one record per (task, seed) in search/issue/<condition>/<slug>_s<seed>.json, the format
run_coder.py reads. The coder (Opus 5.5 medium) later sees only what these records contain, exactly as in round 3.

    modal run eval/e2e/search_app.py --cond a1_35b_scout        # CPU container -> A1-35B endpoint (A1_BASE_URL / A1_API_KEY)
    modal run eval/e2e/search_app.py --cond spark_teacher       # CPU container -> Baseten meta-models/muse-spark-open
    modal run eval/e2e/search_app.py --cond glimmer_base        # one H100 container: vLLM Muse-Glimmer-30B + rollouts (127.0.0.1)

Same loop settings as the Stage B teacher run and the Glimmer baseline: scout product prompt + tools, convergence nudge,
grep-only (`semantic=false`, scout's loop has no semantic backend wired in the harness), T=0.85 top_p=0.95, max_turns 40,
max_seconds 900. The brief is the round-3 issue text. Rollouts run inside Modal (laptop DNS fails on *.modal.host).
GPU safety: ephemeral app `wayfinder-e2e-search*`, hard function timeouts, watcher ops/e2e_watch.py armed at launch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE      # wayfinder repo root; inside the container the file sits in /root
SCOUT_SRC = REPO.parent / "voyage-scout" / "src"
sys.path.insert(0, str(HERE))

APP_NAME = "wayfinder-e2e-search"
H100_USD_PER_S = 0.001097                 # Modal H100 list price (~$3.95/h), same as glimmer_baseline_app
GLIMMER_ID, WEIGHTS_DIR, VLLM_VERSION = "meta-models/Muse-Glimmer-30B", "/weights/Muse-Glimmer-30B", "0.32.0"
GPU_TIMEOUT_S, LAUNCH_DEADLINE_S = 90 * 60, 60 * 60  # was 55/36 min for the base runs; wayfinder_v2* rollouts are slower per turn (3600 s cap)
CPU_TIMEOUT_S = 150 * 60   # Spark via Baseten shares the account-wide 1M TPM with the bulk-collection job, so it queues
NUDGE_ALL = True                           # the Stage B / baseline convergence nudge, applied to all three conditions
MAX_TURNS, MAX_SECONDS = 40, 900
SEARCH_DIR = Path(os.environ["E2E_SEARCH_DIR"]) if os.environ.get("E2E_SEARCH_DIR") else HERE / "search"
RAW_DIR = REPO / "artifacts" / "e2e"

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("wayfinder-weights", create_if_missing=True)
data_vol = modal.Volume.from_name("wayfinder-data", create_if_missing=True)  # merged SFT checkpoints: /data/sft/<variant>_merged

_common_env = {"SCOUT_SRC": "/scout_src", "PYTHONPATH": "/root", "GIT_TERMINAL_PROMPT": "0"}
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ripgrep", "ca-certificates")
    .pip_install("httpx==0.28.1", "pydantic>=2.9", "pathspec==1.1.1", "watchfiles==1.3.0")
    .env(_common_env)
    .add_local_dir(str(SCOUT_SRC), "/scout_src", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub", "jsonschema")
    .apt_install("ripgrep", "ca-certificates")
    .uv_pip_install("httpx==0.28.1", "pydantic>=2.9", "pathspec==1.1.1")
    .env({**_common_env, "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_LOGGING_LEVEL": "INFO"})
    .add_local_dir(str(SCOUT_SRC), "/scout_src", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)
# Inline (non-stored) secrets from the launcher's environment; never printed.
SECRET = modal.Secret.from_dict({k: os.environ.get(k, "") for k in ("A1_API_KEY", "BASETEN_API_KEY")})

BASE_FLAGS = [
    "--served-model-name", "glimmer", "--dtype", "bfloat16", "--gpu-memory-utilization", "0.92",
    "--language-model-only", "--enable-auto-tool-choice", "--enable-prefix-caching", "--max-num-seqs", "64",
    "--tool-call-parser", "muse_glimmer", "--reasoning-parser", "muse_glimmer", "--port", "8000",
]  # fmt: skip

CONDS = {
    "a1_35b_scout": dict(model="a1", key_env="A1_API_KEY", auth="Bearer", base_url_env="A1_BASE_URL", temperature=0.85, top_p=0.95,
                         top_k=20, presence_penalty=1.1, vllm_extensions=True, extra_body=None, tpm=0, concurrency=34,
                         prices=[0.0, 0.0, 0.0], note="Agents-A1-FP8 on the deployed Modal endpoint; A1's recommended sampling"),
    "spark_teacher": dict(model="meta-models/muse-spark-open-1026", key_env="BASETEN_API_KEY", auth="Api-Key",
                          base_url="https://inference.baseten.co/v1", temperature=0.85, top_p=0.95, top_k=0, presence_penalty=0.0,
                          vllm_extensions=False, extra_body={"reasoning_effort": "high"}, retain_reasoning=False, tpm=450_000, tpm_ceiling=750_000,
                          concurrency=17, prices=[1.25, 4.25, 0.15], cap_usd=70.0,
                          note="Muse Spark Open via Baseten, reasoning_effort high, list price $1.25/$4.25/$0.15 per M"),
    **{f"wayfinder_{v}": dict(model="glimmer", key_env=None, auth="Bearer", base_url="http://127.0.0.1:8000/v1", temperature=0.85,
                              top_p=0.95, top_k=0, presence_penalty=0.0, vllm_extensions=False, extra_body=None, tpm=0, concurrency=14,
                              prices=[1.25, 4.25, 0.15], weights_dir=f"/data/sft/{v}_merged", max_seconds=3600,
                              note=f"SFT {v} (Wayfinder, LoRA merged into Muse-Glimmer-30B), vLLM 0.32.0 on 1x H100 in the same container; "
                                   "max_seconds 3600 and concurrency 14 as in the v2 dev eval (the SFT v2 write-up, not in this release)") for v in ("v2a", "v2b", "v3")},
    "glimmer_base": dict(model="glimmer", key_env=None, auth="Bearer", base_url="http://127.0.0.1:8000/v1", temperature=0.85,
                         top_p=0.95, top_k=0, presence_penalty=0.0, vllm_extensions=False, extra_body=None, tpm=0, concurrency=34,
                         prices=[1.25, 4.25, 0.15], note="untrained Muse-Glimmer-30B, vLLM 0.32.0 on 1x H100 in the same container"),
}


def _log(msg: str) -> None:
    print(f"[e2e_search {time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def _run_many(todo: list[tuple[dict, int]], cond: str, cfg: dict, t_start: float, deadline_s: float):
    """Async generator: one row per finished rollout. Shared by the CPU and GPU functions."""
    import asyncio

    from wayfinder.rollout.chat import CompatChat, Prices, Sampling, TokenGate
    from wayfinder.rollout.run import TEACHER_NUDGE, checkout, run_rollout
    from wayfinder.rollout.score import score_row

    import httpx
    # Modal web endpoints answer a request that runs past ~150 s with 303 + a result URL (seen on the A1 endpoint: every long
    # reasoning turn). CompatChat's client does not follow redirects, which scout then maps to INVALID_ARGUMENT. Follow them.
    if not getattr(httpx.AsyncClient, "_wf_follow", False):
        _orig_init = httpx.AsyncClient.__init__

        def _init(self, *a, **kw):  # noqa: ANN001
            kw.setdefault("follow_redirects", True)
            _orig_init(self, *a, **kw)

        httpx.AsyncClient.__init__ = _init
        httpx.AsyncClient._wf_follow = True
    c = {**CONDS[cond], **(cfg.get("override") or {})}   # launcher-side overrides (the container does not see the launcher env)
    if c.get("max_seconds", 0) > 900:  # scout's Budget caps max_seconds at 900; lift it at runtime (scout source untouched), as sft_eval_app does
        from pydantic import Field
        from pydantic.fields import FieldInfo
        from scout.core.contracts.types import Budget, ExploreRequest

        Budget.model_fields["max_seconds"] = FieldInfo.from_annotated_attribute(int, Field(default=120, ge=5, le=100000))
        Budget.model_rebuild(force=True)
        ExploreRequest.model_rebuild(force=True)
        Budget(max_turns=40, max_seconds=c["max_seconds"])  # raises if the lift did not take
    key = os.environ.get(c["key_env"], "") if c["key_env"] else None
    if c["key_env"] and not key:
        raise RuntimeError(f"{c['key_env']} is empty in the container")
    base_url = cfg.get("base_url") or c.get("base_url")
    gate = TokenGate(c["tpm"], ceiling=c.get("tpm_ceiling")) if c.get("tpm") else None
    sem = asyncio.Semaphore(c["concurrency"])
    clones: dict[str, asyncio.Future] = {}
    queue: asyncio.Queue = asyncio.Queue()
    spent = {"usd": 0.0}

    async def get_root(t: dict) -> tuple[str, float]:
        k = f"{t['repo']}@{t['base_commit']}"
        if k not in clones:
            clones[k] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], "/tmp/repos"))
        return await clones[k]

    async def one(t: dict, seed: int) -> None:
        async with sem:
            base = {"task_id": t["task_id"], "repo": t["repo"], "seed": seed, "condition": cond}
            if time.time() - t_start > deadline_s:
                await queue.put({**base, "error": "not launched: function deadline", "infra_error": True})
                return
            if c.get("cap_usd") and spent["usd"] >= c["cap_usd"]:
                await queue.put({**base, "error": "not launched: spend cap", "infra_error": True})
                return
            try:
                root, clone_s = await get_root(t)
            except Exception as e:  # noqa: BLE001
                await queue.put({**base, "error": f"checkout: {e}"[:300], "infra_error": True})
                return
            chat = CompatChat(
                base_url, key, c["model"], auth_scheme=c["auth"],
                sampling=Sampling(c["temperature"], c["top_p"], c["top_k"], c["presence_penalty"]),
                prices=Prices(*c["prices"]), seed=seed, vllm_extensions=c["vllm_extensions"],
                extra_body=c["extra_body"], timeout=600.0, max_retries=(30 if gate else 8), gate=gate,
                # default ON since 2026-10-27 (production scout re-sends earlier reasoning, as the SFT template renders it);
                # the Spark teacher condition stays off: it is the reference the earlier results were collected with
                retain_reasoning=c.get("retain_reasoning", True),
            )
            try:
                row = await asyncio.wait_for(
                    run_rollout(t, chat, root, seed=seed, max_turns=MAX_TURNS, max_seconds=c.get("max_seconds", MAX_SECONDS),
                                nudge=TEACHER_NUDGE if NUDGE_ALL else None),
                    timeout=(3 * 3600 if gate else c.get("max_seconds", MAX_SECONDS) + 600))
                row.update(score_row(row))
                row.update(clone_s=round(clone_s, 1), condition=cond, retain_reasoning=chat.retain_reasoning, temperature=c["temperature"], top_p=c["top_p"])
                spent["usd"] += row.get("cost_usd") or 0.0
            except Exception as e:  # noqa: BLE001
                row = {**base, "error": f"harness: {type(e).__name__}: {e}"[:300], "infra_error": True}
            await queue.put(row)

    tasks = [asyncio.ensure_future(one(t, s)) for t, s in todo]
    try:
        for _ in range(len(tasks)):
            yield await queue.get()
    finally:
        for tk in tasks:
            tk.cancel()


@app.function(image=cpu_image, secrets=[SECRET], timeout=CPU_TIMEOUT_S, cpu=4, memory=8192)
async def cpu_rollouts(todo: list[tuple[dict, int]], cond: str, cfg: dict):
    """A1 / Spark: rollouts in one CPU container calling the model endpoint. Yields rows, then a final {"_meta"}."""
    t0 = time.time()
    async for row in _run_many(todo, cond, cfg, t0, CPU_TIMEOUT_S - 12 * 60):
        yield row
    yield {"_meta": {"cond": cond, "wall_s": round(time.time() - t0, 1)}}


def _start_vllm(weights_dir: str = WEIGHTS_DIR, timeout_s: int = 25 * 60):
    import urllib.request
    log_path = "/tmp/vllm.log"
    proc = subprocess.Popen(["vllm", "serve", weights_dir, *BASE_FLAGS, "--max-model-len", "65536"],
                            stdout=open(log_path, "wb"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited rc={proc.returncode}\n" + "\n".join(Path(log_path).read_text(errors="replace").splitlines()[-30:]))
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                if r.status == 200:
                    return proc, time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    proc.kill()
    raise RuntimeError("vllm not healthy in time")


@app.function(image=gpu_image, gpu="H100", volumes={"/weights": weights, "/data": data_vol}, timeout=GPU_TIMEOUT_S, cpu=8, memory=65536)
async def glimmer_rollouts(todo: list[tuple[dict, int]], cfg: dict, cond: str = "glimmer_base"):
    """Glimmer: vLLM and the rollouts in ONE H100 container (127.0.0.1). Yields rows, then a final {"_meta"}."""
    import asyncio
    t0 = time.time()
    proc, load_s = await asyncio.to_thread(_start_vllm, CONDS[cond].get("weights_dir", WEIGHTS_DIR))
    meta = {"cond": cond, "load_s": round(load_s, 1), "vllm": VLLM_VERSION}
    _log(f"vllm healthy after {load_s:.0f}s")
    try:
        async for row in _run_many(todo, cond, cfg, t0, LAUNCH_DEADLINE_S):
            yield row
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        meta["gpu_function_seconds"] = round(time.time() - t0, 1)
        meta["gpu_cost_estimate_usd"] = round(meta["gpu_function_seconds"] * H100_USD_PER_S, 3)
    yield {"_meta": meta}


# --------------------------------------------------------------------------- launcher side

def sub_record(row: dict, slug: str, extra: dict) -> dict:
    """Search row -> the record shape run_coder.condition_block / allowlist_for read."""
    res = row.get("explore_result") or {}
    hits = res.get("results") or []
    docids = list(dict.fromkeys(h["path"] for h in hits))
    lines = []
    for h in hits:
        lines.append(f"- {h['path']}:{h['start_line']}-{h['end_line']}: {h.get('why', '')}")
    parts = [(res.get("summary") or "").strip()]
    if lines:
        parts.append("Locations it reported:\n" + "\n".join(lines))
    if res.get("gaps"):
        parts.append("Gaps it noted:\n" + "\n".join(f"- {g}" for g in res["gaps"]))
    answer = "\n\n".join(p for p in parts if p)
    usage = row.get("usage") or {}
    return dict(
        task_id=row["task_id"], wording="issue", arm="scout", condition=row["condition"], seed=row["seed"], harness_seed=row["seed"],
        status="ok" if hits else "empty_answer", stop_reason="budget_exhausted" if res.get("budget_exhausted") else "report",
        eval_error=None, answer=answer, retrieved_docids=docids,
        retrieved_spans=[dict(path=h["path"], start_line=h["start_line"], end_line=h["end_line"]) for h in hits],
        tool_call_counts=(res.get("usage") or {}).get("tool_calls"), prompt_tokens=usage.get("in"), completion_tokens=usage.get("out"),
        turns=row.get("turns"), wall_clock_s=row.get("wall_s"), model=row.get("model"), budget_exhausted=bool(res.get("budget_exhausted")),
        n_reported=row.get("n_reported"), recall_reported=row.get("recall_reported"), all_gold_reported=row.get("all_gold_reported"),
        recall_seen_k5=row.get("recall_seen_k5"), all_gold_found=row.get("all_gold_found"), cached_tokens=usage.get("cached"),
        list_cost_usd=row.get("cost_usd"), **extra)


def _manifest_tasks(path: str) -> list[dict]:
    """v4: tasks straight from the committed manifest (the 8 carried v3 tasks are not in the v4 gate dir). Held-out test paths come from the
    task's env record in e2e_v4 or e2e_v3; gold_new_at_base from the public task file."""
    import run_coder as rc
    m = json.load(open(path))
    pub = {t["task_id"]: t for t in json.load(open(HERE.parents[1].parent / "benchmark-repo/datasets/artifacts/public/tasks_1660.json"))["tasks"]}
    pub.update({t["task_id"]: t for t in json.load(open(HERE.parents[1] / "artifacts/e2e_v4/datasets/tasks_1660.json"))["tasks"]})
    out = []
    for t in m["tasks"]:
        env = next(json.load(open(p)) for p in (HERE / "experiments/e2e_v4/envs" / f"{t['task_id']}.json", HERE / "experiments/e2e_v3/envs" / f"{t['task_id']}.json") if p.exists())
        held = [p for _, p in env["heldout_test_entries"]]
        gold_src, _ = rc.gold_source_files(t["gold_files"], held, pub[t["task_id"]].get("gold_new_at_base"))
        out.append(dict(task_id=t["task_id"], slug=t["slug"], repo=t["repo"], base_commit=t["base_commit"], question=t["issue_text"], gold_files=gold_src))
    return out


def _tasks(only: list[str] | None) -> list[dict]:
    if os.environ.get("E2E_TASKS_MANIFEST"):
        ts = _manifest_tasks(os.environ["E2E_TASKS_MANIFEST"])
        return [t for t in ts if not only or any(x.lower() in t["slug"] or x.lower() in t["task_id"].lower() for x in only)]
    import run_coder as rc
    tasks = rc.load_all_tasks()
    out = []
    for t in rc.pilot_tasks(only):
        env = rc.sc.load_env(t["task_id"])
        held = [p for _, p in env["heldout_test_entries"]]
        gold_src, _ = rc.gold_source_files(t["gold_files"], held, tasks[t["task_id"]].get("gold_new_at_base"))
        out.append(dict(task_id=t["task_id"], slug=t["slug"], repo=t["repo_full"], base_commit=t["base_commit"],
                        question=t["issue_text"], gold_files=gold_src))
    return out


@app.local_entrypoint()
def main(cond: str, seeds: str = "1,2", task: str = "", limit: int = 0, shard: str = "", maxruns: int = 0, skip: int = 0):
    import dns_fallback  # noqa: F401
    assert cond in CONDS, cond
    c = CONDS[cond]
    tasks = _tasks([x for x in task.split(",") if x] or None)
    if limit:
        tasks = tasks[:limit]
    slug_of = {t["task_id"]: t["slug"] for t in tasks}
    out_dir = SEARCH_DIR / "issue" / cond
    out_dir.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(t, s) for s in [int(x) for x in seeds.split(",")] for t in tasks if not (out_dir / f"{t['slug']}_s{s}.json").exists()]
    if shard:   # "i/n": this launch runs every n-th pending (task, seed) pair; shards write disjoint record files
        si, sn = (int(x) for x in shard.split("/"))
        todo = todo[si::sn]
    if skip:
        todo = todo[skip:]
    if maxruns:
        todo = todo[:maxruns]
    cfg = {"base_url": os.environ.get(c["base_url_env"], "") if c.get("base_url_env") else None, "override": {}}
    if os.environ.get("E2E_CONC"):
        cfg["override"]["concurrency"] = int(os.environ["E2E_CONC"])
    if os.environ.get("E2E_MAX_SECONDS"):
        cfg["override"]["max_seconds"] = int(os.environ["E2E_MAX_SECONDS"])
    if os.environ.get("E2E_CAP_USD"):
        cfg["override"]["cap_usd"] = float(os.environ["E2E_CAP_USD"])
    _log(f"{cond}: {len(todo)} rollouts to run ({len(tasks)} tasks)")
    if not todo:
        return
    gpu_cond = cond == "glimmer_base" or cond.startswith("wayfinder_")
    fn = glimmer_rollouts if gpu_cond else cpu_rollouts
    args = (todo, cfg, cond) if gpu_cond else (todo, cond, cfg)
    rows, meta, t0 = [], {}, time.time()
    raw = (RAW_DIR / f"search_{cond}.jsonl").open("a")
    for row in fn.remote_gen(*args):
        if "_meta" in row:
            meta = row["_meta"]
            continue
        raw.write(json.dumps(row) + "\n")
        raw.flush()
        rows.append(row)
        ok = not row.get("error")
        _log(f"[{len(rows)}/{len(todo)}] {slug_of.get(row['task_id'], row['task_id'])[:28]} s{row['seed']} turns={row.get('turns')} "
             f"wall={row.get('wall_s')} reported={row.get('n_reported')} recall_rep={row.get('recall_reported')} "
             f"all_gold={row.get('all_gold_found')} list_usd={row.get('cost_usd')} err={row.get('error')}")
        if ok:
            # search cost per run: Spark = Baseten list price of its own tokens; A1 / Glimmer = this launch's GPU seconds
            # (A1 endpoint wall time, Glimmer container time) split equally over the launch's rollouts, filled in below.
            (out_dir / f"{slug_of[row['task_id']]}_s{row['seed']}.json").write_text(
                json.dumps(sub_record(row, slug_of[row["task_id"]], dict(usd=row.get("cost_usd") if cond == "spark_teacher" else None,
                                                                         search_usd_basis="baseten list price" if cond == "spark_teacher" else "pending")), indent=1))
    wall = time.time() - t0
    gpu_usd = meta.get("gpu_cost_estimate_usd") if gpu_cond else (round(wall * H100_USD_PER_S, 3) if cond == "a1_35b_scout" else None)
    n_ok = sum(1 for r in rows if not r.get("error"))
    if gpu_usd is not None and n_ok:
        basis = ("H100 container seconds (vLLM load + rollouts) x list rate, split equally over this launch's rollouts"
                 if gpu_cond else "launch wall seconds x H100 list rate (A1 endpoint is a shared warm H100), split equally over this launch's rollouts")
        for r in rows:
            if not r.get("error"):
                p = out_dir / f"{slug_of[r['task_id']]}_s{r['seed']}.json"
                d = json.loads(p.read_text())
                d.update(usd=round(gpu_usd / n_ok, 5), search_usd_basis=basis)
                p.write_text(json.dumps(d, indent=1))
    summ = dict(cond=cond, n=len(rows), ok=n_ok, errors=[r.get("error") for r in rows if r.get("error")][:10], launcher_wall_s=round(wall, 1),
                search_usd_total=(round(sum(r.get("cost_usd") or 0 for r in rows), 3) if cond == "spark_teacher" else gpu_usd), meta=meta, note=c["note"],
                at=time.strftime("%FT%TZ", time.gmtime()))
    with (SEARCH_DIR / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(summ) + "\n")
    print(json.dumps(summ, indent=1), flush=True)
