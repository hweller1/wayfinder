"""Optional GRPO rollout worker on the GKE G4 pod (docs/grpo_g4_worker.md): the body of `wayfinder/grpo/app.py::worker` minus the Modal container.

vLLM runs in a sibling container (127.0.0.1:8000, deploy/g4/grpo_worker.yaml) with the SAME flags as the Modal workers (`VLLM_FLAGS` below is checked against app.py by a
test), serving the start checkpoint with runtime LoRA and `--logprobs-mode processed_logprobs`. Jobs come from the Modal Queue `wayfinder-grpo-g4q-<run>` (written by
the Modal driver), each job is one task with its seeds, the adapter version and the run's rollout cfg; every rollout goes through the same `grpo_rollout`
(wayfinder/grpo/rollout.py: T 0.85, top_p 0.95, top_k not sent so the checkpoint's generation_config top_k 64 applies, logprobs + token ids captured), so the behaviour
logprobs are the same kind the Modal workers produce and loss.py's ratio stays valid. Chains are uploaded to the Modal volume `wayfinder-grpo` (/<run>/rollouts/<rid>.npz)
BEFORE the summary goes to the results queue `wayfinder-grpo-g4r-<run>`. The adapter v<j> is pulled from the same volume (/<run>/adapters/v<j>) with the modal SDK.
The pod needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (k8s secret wf-modal) and the same index keys as the Modal workers (TURBOPUFFER_API_KEY, VOYAGE_API_KEY,
WF_INDEX_MASTER_KEY_HEX); it never prints them.

    python -m wayfinder.rollout.g4_grpo_worker --run r2_100_hi
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

from wayfinder.grpo import g4_link
from wayfinder.grpo.rollout import TEMPERATURE, TOP_P

BASE = "policy"
START = os.environ.get("GRPO_START", "/data/sft/v3_merged")
VOL = "wayfinder-grpo"
# Must equal wayfinder/grpo/app.py FLAGS except the memory fraction (96 GB card instead of 80 GB): tests/test_grpo_g4.py compares the two lists.
VLLM_FLAGS = ["--served-model-name", BASE, "--dtype", "bfloat16", "--gpu-memory-utilization", "0.92", "--language-model-only",
              "--enable-auto-tool-choice", "--enable-prefix-caching", "--max-num-seqs", "64", "--tool-call-parser", "muse_glimmer",
              "--reasoning-parser", "muse_glimmer", "--port", "8000", "--max-model-len", "65536",
              "--enable-lora", "--max-lora-rank", "64", "--max-loras", "3", "--max-cpu-loras", "6", "--logprobs-mode", "processed_logprobs"]
REQUIRED_ENV = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "TURBOPUFFER_API_KEY", "VOYAGE_API_KEY", "WF_INDEX_MASTER_KEY_HEX")


def _log(msg: str) -> None:
    print(f"[g4_grpo {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_env(env: Any = os.environ) -> list[str]:
    """Names (never values) of the required variables that are missing."""
    return [k for k in REQUIRED_ENV if not env.get(k)]


def check_generation_config(model_dir: str) -> int:
    """The behaviour logprobs are the top-k 64 / top-p 0.95 truncated distribution: top-p comes from the request, top-k from the checkpoint's generation_config.json
    (the harness sends no top_k). Refuse to serve a checkpoint whose default is anything else."""
    gc = json.loads((Path(model_dir) / "generation_config.json").read_text())
    if int(gc.get("top_k", 0)) != 64:
        raise SystemExit(f"{model_dir}/generation_config.json top_k={gc.get('top_k')!r}, expected 64 (loss.py TOP_K): the behaviour logprobs would not match the ratio")
    return 64


def job_usable(job: dict[str, Any], cut: int | None, now: float) -> str | None:
    """None when the job should run, else the reason to skip it (late jobs are never run: their rows would be dropped anyway)."""
    if job.get("expires") is not None and now > float(job["expires"]):
        return "expired"
    if cut is not None and int(job["step"]) <= int(cut):
        return "step already closed"
    if (job.get("cfg") or {}).get("fork_report"):
        return "fork-report not supported on G4"
    return None


def volume_paths(run: str, rid: str) -> tuple[str, str]:
    """(remote path inside the Modal volume, path the trainer reads under /gdata) of a rollout file."""
    return f"/{run}/rollouts/{rid}.npz", f"/gdata/{run}/rollouts/{rid}.npz"


def infra_row(run: str, job: dict[str, Any], seed: int, wid: str, err: str, elapsed: float = 0.0) -> dict[str, Any]:
    from wayfinder.grpo.rollout import rollout_id

    t = job["task"]
    return {"rid": rollout_id(run, job["step"], t["task_id"], seed), "task_id": t["task_id"], "seed": seed, "step": job["step"], "version": job["version"], "worker": wid,
            "infra_error": True, "error": err[:300], "elapsed_s": round(elapsed, 1), "source": "g4"}


# ------------------------------------------------------------------------------------------------ vLLM helpers (as app.py)
def _post(path: str, body: dict, timeout: float = 600.0) -> tuple[int, str]:
    import urllib.error

    req = urllib.request.Request(f"http://127.0.0.1:8000{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]


def pull_adapter(vol: Any, run: str, v: int, dst: Path, tries: int = 120, sleep_s: float = 5.0) -> int:
    """Download /<run>/adapters/v<v> from the Modal volume (the trainer committed it before the driver submitted a job that needs it). Returns bytes."""
    src = f"/{run}/adapters/v{v}"
    for attempt in range(tries):
        try:
            names = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(src)}
            if {"adapter_model.safetensors", "adapter_config.json"} <= names:
                break
        except Exception as e:  # noqa: BLE001
            if attempt % 6 == 0:
                _log(f"adapter v{v} not visible yet ({str(e)[:100]})")
        time.sleep(sleep_s)
    else:
        raise RuntimeError(f"adapter v{v} not on the volume after {tries * sleep_s:.0f}s")
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = 0
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        with open(tmp / name, "wb") as f:
            for chunk in vol.read_file(f"{src}/{name}"):
                f.write(chunk)
                n += len(chunk)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return n


def _lp(model: str) -> list[float]:
    code, txt = _post("/v1/chat/completions", {"model": model, "messages": [{"role": "system", "content": "You are a code search agent."},
                      {"role": "user", "content": "Where is request routing implemented in a typical web framework? Think step by step."}],
                      "max_tokens": 24, "temperature": 0, "logprobs": True}, timeout=300)
    return [x["logprob"] for x in json.loads(txt)["choices"][0]["logprobs"]["content"]] if code == 200 and txt.startswith("{") else []


async def main_async(a: argparse.Namespace) -> None:
    import modal

    from wayfinder.grpo.data import save_rollout
    from wayfinder.grpo.rollout import grpo_rollout, lift_budget_cap, rollout_id
    from wayfinder.rollout.run import checkout

    miss = check_env()
    if miss:
        raise SystemExit(f"missing environment variables: {miss}")
    check_generation_config(START)
    run, wid = a.run, "g4"
    n = g4_link.names(run)
    jobs = modal.Queue.from_name(n["jobs"], create_if_missing=True)
    results = modal.Queue.from_name(n["results"], create_if_missing=True)
    ctl = modal.Dict.from_name(n["ctl"], create_if_missing=True)
    vol = modal.Volume.from_name(VOL)
    for _ in range(600):  # vLLM (sibling container) health
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    else:
        raise SystemExit("vllm not healthy")
    lift_budget_cap(a.max_seconds_cap)
    loaded: dict[int, str] = {}  # version -> served name; v0 is the unadapted start checkpoint unless the run resumed from another run's adapter (then adapters/v0 is that adapter)
    lora_diff: dict[int, Any] = {}
    load_lock = asyncio.Lock()
    sem_slots = asyncio.Semaphore(max(1, a.concurrency // a.group))
    clones: dict[str, asyncio.Future] = {}
    stats = {"rollouts": 0, "errors": 0, "groups": 0, "skipped_jobs": 0, "rollout_wall_s": 0.0}
    cut_cache = {"v": None}
    stop = {"flag": False}
    tasks_by_step: dict[int, set[asyncio.Task]] = {}
    Path(a.adapters).mkdir(parents=True, exist_ok=True)

    async def ensure_version(v: int, resumed: bool) -> str:
        async with load_lock:
            if v == 0 and not resumed:
                return BASE
            if v in loaded:
                return loaded[v]
            t0 = time.time()
            dst = Path(a.adapters) / f"v{v}"
            nbytes = await asyncio.to_thread(pull_adapter, vol, run, v, dst)
            code, txt = await asyncio.to_thread(_post, "/v1/load_lora_adapter", {"lora_name": f"v{v}", "lora_path": str(dst)})
            if code != 200:
                raise RuntimeError(f"load_lora_adapter v{v}: {code} {txt}")
            try:  # the adapter must be live: fixed-prompt logprobs differ between the base and this version
                a_, b_ = await asyncio.to_thread(_lp, BASE), await asyncio.to_thread(_lp, f"v{v}")
                lora_diff[v] = round(sum(abs(x - y) for x, y in zip(a_, b_)), 5) if a_ and b_ else None
            except Exception as e:  # noqa: BLE001
                lora_diff[v] = f"check failed: {e!r}"[:80]
            loaded[v] = f"v{v}"
            for old in sorted(loaded):
                if old != 0 and old < v - 2:
                    await asyncio.to_thread(_post, "/v1/unload_lora_adapter", {"lora_name": loaded.pop(old)})
            _log(f"loaded adapter v{v} ({nbytes / 1e6:.0f} MB) in {time.time() - t0:.1f}s; loaded={sorted(loaded)}; L1 vs base {lora_diff.get(v)}")
            return loaded[v]

    async def get_root(t: dict) -> tuple[str, float]:
        key = f"{t['repo']}@{t['base_commit']}"
        if key not in clones:
            clones[key] = asyncio.ensure_future(asyncio.to_thread(checkout, t["repo"], t["base_commit"], a.repos))
        return await clones[key]

    async def one(job: dict, seed: int, model: str, cfg: dict) -> dict[str, Any]:
        t, step, version = job["task"], job["step"], job["version"]
        rid = rollout_id(run, step, t["task_id"], seed)
        t0 = time.time()
        try:
            root, clone_s = await get_root(t)
            res = await asyncio.wait_for(grpo_rollout(t, "http://127.0.0.1:8000/v1", model, root, seed=seed, cfg=cfg), timeout=cfg["max_seconds"] + 900)
            summ = res["summary"]
            summ.update(rid=rid, step=step, version=version, worker=wid, clone_s=round(clone_s, 1), elapsed_s=round(time.time() - t0, 1), source="g4")
            if summ["error"]:
                summ["infra_error"] = True
            elif summ["n_chains"] and summ["assistant_tokens"]:
                remote, trainer_path = volume_paths(run, rid)
                local = Path(a.scratch) / f"{rid}.npz"
                local.parent.mkdir(parents=True, exist_ok=True)
                summ["n_loss"] = await asyncio.to_thread(save_rollout, str(local), {"rid": rid, "task_id": t["task_id"], "seed": seed, "step": step, "version": version}, res["chains"])
                summ["n_tokens"] = int(sum(len(c["input_ids"]) for c in res["chains"]))

                def _upload() -> None:
                    with vol.batch_upload(force=True) as up:  # committed when the context exits: the file is visible to the trainer before the summary is queued
                        up.put_file(str(local), remote)

                await asyncio.to_thread(_upload)
                local.unlink(missing_ok=True)
                summ["path"] = trainer_path
            else:
                summ["n_loss"], summ["n_tokens"], summ["path"] = 0, 0, None
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            summ = infra_row(run, job, seed, wid, f"{type(e).__name__}: {e}", time.time() - t0)
        stats["errors" if summ.get("infra_error") else "rollouts"] += 1
        stats["rollout_wall_s"] += time.time() - t0
        return summ

    async def run_group(job: dict) -> None:
        async with sem_slots:
            reason = job_usable(job, cut_cache["v"], time.time())
            if reason:
                stats["skipped_jobs"] += 1
                _log(f"skip step {job['step']} {job['task']['task_id'][-30:]}: {reason}")
                return
            cfg = {**job["cfg"]}
            try:
                model = await ensure_version(job["version"], bool(cfg.get("resume_adapter")))
            except Exception as e:  # noqa: BLE001
                await results.put_many.aio([infra_row(run, job, s, wid, f"adapter: {e}") for s in job["seeds"]])
                return
            stats["groups"] += 1
            subs = [asyncio.ensure_future(one(job, s, model, cfg)) for s in job["seeds"]]
            tasks_by_step.setdefault(job["step"], set()).update(subs)
            try:
                rows = await asyncio.gather(*subs)
            except asyncio.CancelledError:
                return  # the driver closed this step while the group was running: nothing to send
            await results.put_many.aio(rows)

    async def control() -> None:
        """Heartbeat for the driver (a stale one makes it submit no G4 tasks) and the cut flag: cancel work of steps the driver already closed."""
        while not stop["flag"]:
            try:
                await ctl.put.aio("hb", {"t": time.time(), "loaded": sorted(loaded), "lora_l1_vs_base": {str(k): x for k, x in lora_diff.items()}, **stats})
                cut = await ctl.get.aio("cut")
                cut_cache["v"] = cut
                if cut is not None:
                    for step in [s for s in tasks_by_step if s <= int(cut)]:
                        for tk in tasks_by_step.pop(step):
                            tk.cancel()
            except Exception as e:  # noqa: BLE001
                _log(f"control error {e!r}")
            await asyncio.sleep(20)

    bg = asyncio.ensure_future(control())
    inflight: set[asyncio.Task] = set()
    _log(f"ready: run {run}, slots {max(1, a.concurrency // a.group)} groups, concurrency {a.concurrency}")
    try:
        while not stop["flag"]:
            inflight = {x for x in inflight if not x.done()}
            if len(inflight) >= 2 * max(1, a.concurrency // a.group):  # a small backlog is fine; more would only hold jobs the driver may cut
                await asyncio.sleep(2)
                continue
            try:
                job = await jobs.get.aio(block=True, timeout=10)
            except Exception:  # noqa: BLE001  (queue empty or a network blip)
                job = None
            if job is None:
                continue
            inflight.add(asyncio.ensure_future(run_group(job)))
    finally:
        stop["flag"] = True
        bg.cancel()
        await asyncio.gather(*inflight, return_exceptions=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--concurrency", type=int, default=64, help="rollouts in flight (= vLLM --max-num-seqs)")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-seconds-cap", type=int, default=1200)
    ap.add_argument("--repos", default="/data/g4repos")
    ap.add_argument("--adapters", default="/adapters")
    ap.add_argument("--scratch", default="/tmp/g4rollouts")
    asyncio.run(main_async(ap.parse_args()))
