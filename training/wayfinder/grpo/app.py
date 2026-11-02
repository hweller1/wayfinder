"""GRPO on Modal, one app per run (`wayfinder-grpo-<run>`, watcher ops/grpo_watch.py): persistent H100 rollout workers and the persistent
H200 trainer. `modal run wayfinder/grpo/app.py --run a ...` is the driver (wayfinder/grpo/driver.py) and keeps the app alive; when it dies,
the app stops with it. With `--remote-driver` the driver runs inside Modal (`driver_remote`, a CPU function; outputs on the volume under
/gdata/<run>/driver/) and the laptop is not needed after the launch.

Rollout worker (`worker`): persistent H100 container = vLLM (SFT checkpoint + runtime LoRA) + the scout harness.

A worker serves the start checkpoint (`/data/sft/v3_merged`, parameter) with `--enable-lora` and loads adapter versions published by the trainer
into the volume `wayfinder-grpo` (`/gdata/<run>/adapters/v<k>`, bf16 peft files) while it keeps serving: `/v1/load_lora_adapter` with a fresh
name per version (`v<k>`), no merge, no restart. Version 0 is the unadapted base. Jobs come from a Modal Queue (`wayfinder-grpo-jobs-<run>`): a
job is one task with its 8 seeds and the adapter version to sample with, so a group shares one repo checkout and the prefix cache. Each rollout
is a full harness run (retain_reasoning on, grep + semantic auto, max_turns 40, T 0.85, top_p 0.95, logprobs + token ids captured), rewarded
and cut into chains (`wayfinder/grpo/rollout.py`); the chains are written to the volume as `<run>/rollouts/<rid>.npz` and committed, and only
then does the rollout's summary row go to the results Queue (`wayfinder-grpo-results-<run>`), so the trainer never sees a missing file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

RUN = os.environ.get("GRPO_RUN", "smoke")
APP_NAME = f"wayfinder-grpo-{RUN}"
START = os.environ.get("GRPO_START", "/data/sft/v3_merged")
GPU = os.environ.get("GRPO_ROLL_GPU", "H100")
USD_PER_S = {"H100": 0.001097, "H200": 0.001261}[GPU]
TIMEOUT_S = int(os.environ.get("GRPO_ROLL_TIMEOUT_H", "14")) * 3600

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
SCOUT_SRC = REPO.parent / "voyage-scout" / "src"

app = modal.App(APP_NAME)
weights = modal.Volume.from_name("wayfinder-weights", create_if_missing=True)
data_vol = modal.Volume.from_name("wayfinder-data", create_if_missing=True)
gvol = modal.Volume.from_name("wayfinder-grpo", create_if_missing=True)
INDEX_SECRET = modal.Secret.from_name("wayfinder-index-keys")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install("vllm==0.32.0", "huggingface_hub", "jsonschema", "numpy")
    .apt_install("ripgrep", "ca-certificates")
    .uv_pip_install("httpx==0.28.1", "pydantic>=2.9", "pathspec==1.1.1")
    .uv_pip_install("watchfiles==1.3.0", "cryptography==50.0.2", "keyring==25.7.0", "turbopuffer==3.0.0", "voyageai==0.5.0")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_LOGGING_LEVEL": "INFO", "SCOUT_SRC": "/scout_src", "PYTHONPATH": "/root", "GIT_TERMINAL_PROMPT": "0",
          "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "GRPO_RUN": RUN, "GRPO_START": START, "GRPO_ROLL_GPU": GPU, "GRPO_ROLL_TIMEOUT_H": str(TIMEOUT_S // 3600)})
    .add_local_dir(str(SCOUT_SRC), "/scout_src", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)

BASE = "policy"
FLAGS = ["--served-model-name", BASE, "--dtype", "bfloat16", "--gpu-memory-utilization", "0.90", "--language-model-only",
         "--enable-auto-tool-choice", "--enable-prefix-caching", "--max-num-seqs", "64", "--tool-call-parser", "muse_glimmer",
         "--reasoning-parser", "muse_glimmer", "--port", "8000", "--max-model-len", "65536",
         "--enable-lora", "--max-lora-rank", "64", "--max-loras", "3", "--max-cpu-loras", "6", "--logprobs-mode", "processed_logprobs"]  # fmt: skip


def _log(msg: str) -> None:
    print(f"[grpo_roll {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _start_vllm(model_dir: str, timeout_s: int = 25 * 60) -> tuple[subprocess.Popen, float, str]:
    import urllib.request

    log_path = "/tmp/vllm_policy.log"
    t0 = time.time()
    proc = subprocess.Popen(["vllm", "serve", model_dir, *FLAGS], stdout=open(log_path, "wb"), stderr=subprocess.STDOUT)
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("vllm exited\n" + "\n".join(Path(log_path).read_text(errors="replace").splitlines()[-40:]))
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
                if r.status == 200:
                    return proc, time.time() - t0, log_path
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    proc.kill()
    raise RuntimeError("vllm not healthy in time")


def _post(path: str, body: dict, timeout: float = 600.0) -> tuple[int, str]:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"http://127.0.0.1:8000{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]


def _evict_repos(cache: str, keep_free_gb: float = 25.0) -> None:
    """Delete the oldest checkouts when the disk gets short (they are re-cloned on demand)."""
    st = os.statvfs(cache)
    if st.f_bavail * st.f_frsize / 1e9 > keep_free_gb:
        return
    dirs = sorted((d for d in Path(cache).iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    for d in dirs[: max(1, len(dirs) // 3)]:
        shutil.rmtree(d, ignore_errors=True)


@app.function(image=image, gpu=GPU, volumes={"/weights": weights, "/data": data_vol, "/gdata": gvol}, secrets=[INDEX_SECRET], timeout=TIMEOUT_S, cpu=8, memory=65536)
async def worker(wid: int, cfg: dict) -> dict:
    import asyncio
    import queue as pyqueue

    from wayfinder.grpo.data import save_rollout
    from wayfinder.grpo.rollout import grpo_rollout, lift_budget_cap, rollout_id
    from wayfinder.rollout.run import checkout

    run = cfg["run"]
    t_start = time.time()
    jobs = modal.Queue.from_name(f"wayfinder-grpo-jobs-{run}", create_if_missing=True)
    results = modal.Queue.from_name(f"wayfinder-grpo-results-{run}", create_if_missing=True)
    ctl = modal.Dict.from_name(f"wayfinder-grpo-ctl-{run}", create_if_missing=True)
    model_dir = cfg.get("start", START)
    proc, load_s, log_path = await asyncio.to_thread(_start_vllm, model_dir)
    _log(f"w{wid} vllm healthy after {load_s:.0f}s ({model_dir})")
    lift_budget_cap(cfg["max_seconds"])
    loaded: dict[int, str] = {0: BASE}  # adapter version -> served name
    lora_diff: dict[int, object] = {}  # version -> L1 distance of 24 greedy-token logprobs vs the base (> 0 = the adapter is applied)
    import threading

    vol_lock = threading.Lock()  # gvol.reload() fails while a file on the volume is open: serialise reload, rollout writes and commit
    load_lock = asyncio.Lock()
    sem = asyncio.Semaphore(cfg["concurrency"])
    clones: dict[str, asyncio.Future] = {}
    pending: list[dict] = []
    plock = asyncio.Lock()
    stats = {"rollouts": 0, "errors": 0, "groups": 0}
    stop = {"flag": False}

    async def ensure_version(v: int) -> str:
        async with load_lock:
            if v in loaded:
                return loaded[v]
            src = f"/gdata/{run}/adapters/v{v}"
            t0 = time.time()
            for attempt in range(40):
                try:
                    def _reload() -> None:
                        with vol_lock:
                            gvol.reload()

                    await asyncio.to_thread(_reload)
                except Exception as e:  # noqa: BLE001
                    _log(f"w{wid} volume reload retry {attempt}: {str(e)[:120]}")
                if Path(src, "adapter_model.safetensors").exists() and Path(src, "adapter_config.json").exists():
                    break
                await asyncio.sleep(5)
            else:
                raise RuntimeError(f"adapter v{v} not visible after reloads")
            dst = f"/tmp/adapters/v{v}"
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
            name = f"v{v}"
            code, txt = await asyncio.to_thread(_post, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": dst})
            if code != 200:
                raise RuntimeError(f"load_lora_adapter v{v}: {code} {txt}")
            # warm the LoRA kernels (the first request after a load paid a JIT spike in the probe) and check the adapter is live: the sampled-token
            # logprobs of a fixed prompt must differ between the base and this version (an adapter vLLM silently ignored would give identical numbers)
            def _lp(model: str) -> list[float]:
                code2, txt2 = _post("/v1/chat/completions", {"model": model, "messages": [{"role": "system", "content": "You are a code search agent."}, {"role": "user", "content": "Where is request routing implemented in a typical web framework? Think step by step."}],
                                                            "max_tokens": 24, "temperature": 0, "logprobs": True}, timeout=300)
                return [x["logprob"] for x in json.loads(txt2)["choices"][0]["logprobs"]["content"]] if code2 == 200 and txt2.startswith("{") else []

            try:
                a_, b_ = await asyncio.to_thread(_lp, BASE), await asyncio.to_thread(_lp, name)
                lora_diff[v] = round(sum(abs(x - y) for x, y in zip(a_, b_)), 5) if a_ and b_ else None
            except Exception as e:  # noqa: BLE001
                lora_diff[v] = f"check failed: {e!r}"[:80]
            loaded[v] = name
            for old in sorted(loaded):  # keep the three newest versions (and the base)
                if old != 0 and old < v - 2:
                    await asyncio.to_thread(_post, "/v1/unload_lora_adapter", {"lora_name": loaded.pop(old)})
            _log(f"w{wid} loaded adapter v{v} in {time.time() - t0:.1f}s; loaded={sorted(loaded)}; L1 vs base {lora_diff.get(v)}")
            return name

    async def get_root(t: dict) -> tuple[str, float]:
        key = f"{t['repo']}@{t['base_commit']}"
        if key not in clones:
            def _co() -> tuple[str, float]:
                try:
                    _evict_repos("/tmp/repos") if Path("/tmp/repos").exists() else None
                except Exception:  # noqa: BLE001
                    pass
                return checkout(t["repo"], t["base_commit"], "/tmp/repos")

            clones[key] = asyncio.ensure_future(asyncio.to_thread(_co))
        return await clones[key]

    def _save_fork(fk: dict, rid: str, t: dict, seed: int, step: int, version: int) -> dict:
        """Forked-report mode: write each trained forked completion as its own rollout file; return the compact record the driver needs (no token arrays)."""
        out = {k: v for k, v in fk.items() if k != "train"}
        out["train"] = []
        for j, ch in fk.get("train") or []:
            fp = f"/gdata/{run}/rollouts/{rid}-f{j}.npz"
            with vol_lock:
                n_loss = save_rollout(fp, {"rid": f"{rid}-f{j}", "task_id": t["task_id"], "seed": seed, "step": step, "version": version, "fork_of": rid}, [ch])
            out["train"].append({"j": j, "path": fp, "n_loss": n_loss, "n_tokens": len(ch["input_ids"])})
        return out

    async def one(t: dict, seed: int, step: int, version: int, model: str) -> None:
        async with sem:
            rid = rollout_id(run, step, t["task_id"], seed)
            t0 = time.time()
            try:
                root, clone_s = await get_root(t)
                res = await asyncio.wait_for(grpo_rollout(t, "http://127.0.0.1:8000/v1", model, root, seed=seed, cfg=cfg), timeout=cfg["max_seconds"] + 900)
                summ = res["summary"]
                summ.update(rid=rid, step=step, version=version, worker=wid, clone_s=round(clone_s, 1), elapsed_s=round(time.time() - t0, 1))
                if summ["error"]:
                    summ["infra_error"] = True
                elif summ["n_chains"] and summ["assistant_tokens"]:
                    path = f"/gdata/{run}/rollouts/{rid}.npz"
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    def _save() -> int:
                        with vol_lock:
                            return save_rollout(path, {"rid": rid, "task_id": t["task_id"], "seed": seed, "step": step, "version": version}, res["chains"])

                    summ["n_loss"] = await asyncio.to_thread(_save)
                    summ["n_tokens"] = int(sum(len(c["input_ids"]) for c in res["chains"]))
                    summ["path"] = path
                    if res.get("fork") is not None:  # forked-report mode only
                        summ["fork"] = await asyncio.to_thread(_save_fork, res["fork"], rid, t, seed, step, version)
                else:
                    summ["n_loss"], summ["n_tokens"], summ["path"] = 0, 0, None
            except Exception as e:  # noqa: BLE001
                summ = {"rid": rid, "task_id": t["task_id"], "seed": seed, "step": step, "version": version, "worker": wid, "infra_error": True,
                        "error": f"{type(e).__name__}: {e}"[:300], "elapsed_s": round(time.time() - t0, 1)}
            stats["errors" if summ.get("infra_error") else "rollouts"] += 1
            async with plock:
                pending.append(summ)

    async def flusher() -> None:
        """Commit the volume, THEN release the summaries: a summary in the results queue means its file is visible to the trainer."""
        while not stop["flag"] or pending:
            await asyncio.sleep(3)
            async with plock:
                batch, pending[:] = pending[:], []
            if not batch:
                continue
            if any(b.get("path") for b in batch):
                for attempt in range(5):
                    try:
                        def _commit() -> None:
                            with vol_lock:
                                gvol.commit()

                        await asyncio.to_thread(_commit)
                        break
                    except Exception as e:  # noqa: BLE001
                        _log(f"w{wid} commit retry {attempt}: {str(e)[:150]}")
                        await asyncio.sleep(3)
            await results.put_many.aio(batch)

    async def run_group(job: dict) -> None:
        t = job["task"]
        try:
            model = await ensure_version(job["version"])
        except Exception as e:  # noqa: BLE001
            async with plock:
                pending.extend({"rid": rollout_id(run, job["step"], t["task_id"], s), "task_id": t["task_id"], "seed": s, "step": job["step"], "version": job["version"],
                                "worker": wid, "infra_error": True, "error": f"adapter: {e}"[:300]} for s in job["seeds"])
            return
        stats["groups"] += 1
        await asyncio.gather(*[one(t, s, job["step"], job["version"], model) for s in job["seeds"]])

    async def heartbeat() -> None:
        while not stop["flag"]:
            try:
                await ctl.put.aio(f"w{wid}", {"t": time.time(), "loaded": sorted(loaded), "lora_l1_vs_base": {str(k): x for k, x in lora_diff.items()}, **stats})
                if await ctl.get.aio("stop"):
                    stop["flag"] = True
            except Exception as e:  # noqa: BLE001
                _log(f"w{wid} heartbeat error {e!r}")
            await asyncio.sleep(30)

    bg = [asyncio.ensure_future(flusher()), asyncio.ensure_future(heartbeat())]
    inflight: set[asyncio.Task] = set()
    slots = max(1, cfg["concurrency"] // cfg["group_size"])
    try:
        while not stop["flag"]:
            inflight = {x for x in inflight if not x.done()}
            if len(inflight) >= slots:
                await asyncio.sleep(2)
                continue
            try:
                job = await jobs.get.aio(block=True, timeout=10)
            except pyqueue.Empty:
                job = None
            if job is None:
                continue
            inflight.add(asyncio.ensure_future(run_group(job)))
        await asyncio.gather(*inflight, return_exceptions=True)
    finally:
        stop["flag"] = True
        await asyncio.gather(*bg, return_exceptions=True)
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {"wid": wid, "load_s": load_s, "seconds": round(time.time() - t_start), "usd": round((time.time() - t_start) * USD_PER_S, 2), **stats}


# ----------------------------------------------------------------------------------------------------------------------------------
# trainer (persistent H200 function) and the CPU merge
# ----------------------------------------------------------------------------------------------------------------------------------
train_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.9.1", "transformers==5.21.0", "peft", "accelerate", "kernels>=0.17.2,<0.18", "huggingface_hub", "numpy", "einops", "safetensors")
    .run_commands("pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/srv"})
    .add_local_file(REPO / "wayfinder" / "sft" / "train.py", "/srv/train.py")
    .add_local_dir(str(REPO / "wayfinder" / "grpo"), "/srv/wayfinder/grpo", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder" / "opd"), "/srv/wayfinder/opd", copy=False, ignore=["**/__pycache__"])
)
H200_USD_PER_S = 0.001261


def _train_body(cfg: dict, gpus: int) -> dict:
    """Parent process of the trainer: bridges the driver's Modal Dict (batch manifests in, step records out) to the files the torchrun
    replicas poll, reloads the volume so they see fresh rollouts, and commits it after each step so rollout workers see the new adapter."""
    import threading

    run = cfg["run"]
    ctl_dir = Path("/tmp/ctl")
    shutil.rmtree(ctl_dir, ignore_errors=True)
    ctl_dir.mkdir(parents=True)
    ctl = modal.Dict.from_name(f"wayfinder-grpo-ctl-{run}", create_if_missing=True)
    t0 = time.time()
    cmd = ["torchrun", "--standalone", f"--nproc-per-node={gpus}", "/srv/wayfinder/grpo/train_grpo.py", "--run-name", run, "--ctl", str(ctl_dir), "--model", cfg["start"],
           "--steps", str(cfg["steps"]), "--start-step", str(cfg.get("start_step", 0)), "--lr", str(cfg["lr"]), "--beta", str(cfg["beta"]), "--clip-lo", str(cfg["clip"]),
           "--clip-hi", str(cfg["clip"]), "--temperature", str(cfg["temperature"]), "--save-every", str(cfg["save_every"]), "--warmup-steps", str(cfg.get("warmup", 2)),
           "--rewarm-updates", str(cfg.get("rewarm", 10)), "--guard-gn-mult", str(cfg.get("guard_gn_mult", 0)), "--guard-kl", str(cfg.get("guard_kl", 0)),
           "--guard-stuck", str(cfg.get("guard_stuck", 3)), "--resume-slot-every", str(cfg.get("resume_slot_every", 1))]
    if cfg.get("resume_adapter"):
        cmd += ["--resume-adapter", cfg["resume_adapter"]]
    if cfg.get("fork_cached"):  # keys exist only in --fork-cached runs
        cmd += ["--fork-cached", "--fork-cached-verify", str(cfg.get("fork_cached_verify", 0))]
        if cfg.get("fork_cached_steps"):
            cmd += ["--fork-cached-steps", str(cfg["fork_cached_steps"])]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout

    def pump() -> None:
        for line in p.stdout:  # type: ignore[union-attr]
            print(line, end="", flush=True)

    threading.Thread(target=pump, daemon=True).start()
    nxt = cfg.get("start_step", 0)
    while nxt < cfg["steps"] and p.poll() is None:
        try:
            if not (ctl_dir / "ready").exists():
                time.sleep(2)
                continue
            batch = ctl.get(f"batch_{nxt}")
            if batch is None:
                if ctl.get("stop"):
                    (ctl_dir / "stop").write_text("1")
                    break
                ctl.put("trainer_hb", {"t": time.time(), "waiting_for": nxt})
                time.sleep(3)
                continue
            gvol.reload()  # fresh rollout files (no file of this volume is open in this process; the replicas close theirs per rollout)
            (ctl_dir / f"batch_{nxt}.json.tmp").write_text(json.dumps(batch))
            (ctl_dir / f"batch_{nxt}.json.tmp").rename(ctl_dir / f"batch_{nxt}.json")
            while not (ctl_dir / f"done_{nxt}.json").exists():
                if p.poll() is not None:
                    raise RuntimeError(f"trainer exited rc={p.returncode} during step {nxt}")
                time.sleep(2)
            rec = json.loads((ctl_dir / f"done_{nxt}.json").read_text())
            gvol.commit()  # publishes adapters/v<nxt+1> (and the fp32 ckpt when saved) to the rollout workers
            rec["trainer_gpu_usd"] = round((time.time() - t0) * gpus * H200_USD_PER_S, 2)
            ctl.put(f"done_{nxt}", rec)
            nxt += 1
        except Exception as e:  # noqa: BLE001
            print(f"[trainer-bridge] error: {e!r}", flush=True)
            ctl.put("trainer_error", {"t": time.time(), "error": repr(e)[:400]})
            time.sleep(5)
    rc = p.wait(timeout=600) if p.poll() is None else p.returncode
    try:
        gvol.commit()
    except Exception:  # noqa: BLE001
        pass
    return {"rc": rc, "seconds": round(time.time() - t0), "gpus": gpus, "usd": round((time.time() - t0) * gpus * H200_USD_PER_S, 2)}


@app.function(image=train_image, gpu="H200:2", volumes={"/weights": weights, "/data": data_vol, "/gdata": gvol}, timeout=TIMEOUT_S)
def trainer_x2(cfg: dict) -> dict:
    return _train_body(cfg, 2)


@app.function(image=train_image, gpu="H200:4", volumes={"/weights": weights, "/data": data_vol, "/gdata": gvol}, timeout=TIMEOUT_S)
def trainer_x4(cfg: dict) -> dict:
    return _train_body(cfg, 4)


@app.function(image=train_image, gpu="H200:8", volumes={"/weights": weights, "/data": data_vol, "/gdata": gvol}, timeout=TIMEOUT_S)
def trainer_x8(cfg: dict) -> dict:
    return _train_body(cfg, 8)


def pick_trainer(train_gpus: int):
    """The trainer function for --train-gpus (2, 4 or 8); anything else is an error rather than a silent fallback to 2 GPUs."""
    table = {2: trainer_x2, 4: trainer_x4, 8: trainer_x8}
    if train_gpus not in table:
        raise SystemExit(f"--train-gpus must be one of {sorted(table)}, got {train_gpus}")
    return table[train_gpus]


merge_image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "safetensors", "numpy")


@app.function(image=merge_image, volumes={"/weights": weights, "/data": data_vol, "/gdata": gvol}, cpu=8, memory=131072, timeout=60 * 60)
def merge(base: str, run: str, step: int) -> dict:
    """W' = W + (alpha/r) B A in fp32, cast to bf16 (the SFT/OPD arithmetic), base = the start checkpoint, adapter = fp32 ckpt/s<step>.
    Writes wayfinder-data:/grpo/<run>/s<step>_merged."""
    import torch
    from safetensors.torch import load_file, save_file

    gvol.reload()
    data_vol.reload()
    src, adp, dst = Path(base), Path(f"/gdata/{run}/ckpt/s{step}"), Path(f"/data/grpo/{run}/s{step}_merged")
    cfg = json.loads((adp / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(adp / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        assert k.startswith("base_model.model."), k
        mod, ab = k[len("base_model.model."):].rsplit(".lora_", 1)
        pairs.setdefault(mod + ".weight", {})[ab[0]] = v
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [n for n in pairs if n not in idx]
    if missing:
        raise RuntimeError(f"{len(missing)} adapter modules not in base, e.g. {missing[:3]}")
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and not f.name.startswith("."):
            shutil.copy(f, dst / f.name)
    merged, max_rel = 0, 0.0
    for shard in sorted(src.glob("model-*.safetensors")):
        tensors = load_file(str(shard))
        for name in list(tensors):
            if name in pairs:
                w, ab = tensors[name], pairs[name]
                delta = scale * (ab["B"].float() @ ab["A"].float())
                assert delta.shape == w.shape, (name, delta.shape, w.shape)
                tensors[name] = (w.float() + delta).to(w.dtype)
                max_rel = max(max_rel, float(delta.norm() / w.float().norm()))
                merged += 1
        save_file(tensors, str(dst / shard.name), metadata={"format": "pt"})
        del tensors
    assert merged == len(pairs), (merged, len(pairs))
    data_vol.commit()
    return {"modules_merged": merged, "scale": scale, "max_rel_delta_norm": max_rel, "dst": str(dst)}


# ----------------------------------------------------------------------------------------------------------------------------------
# cloud driver: the same Driver loop as main(), in a CPU container, so a laptop network drop cannot kill the run
# ----------------------------------------------------------------------------------------------------------------------------------
REMOTE_ROOT = "/root"  # wayfinder/grpo/pool.py's ROOT is parents[2] of /root/wayfinder/grpo/pool.py = /root, so data/ and artifacts/ below land where the Driver looks
_BAKED = ["data/splits.json", "data/task_filter.json", "data/opd_excluded_tasks.json", "data/rl_pool_v0.json", "data/rl_pool_v1.json", "data/rl_pool_v1_hard.json", "data/rl_pool_v2.json", "artifacts/grpo/cal_tasks.json", "artifacts/grpo/fork_smoke_tasks.json",
          "artifacts/sft/e2e_excluded_tasks.json", "artifacts/teacher_test/stageB_tasks.json"]
driver_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")
for _rel in _BAKED:
    if (REPO / _rel).exists():
        driver_image = driver_image.add_local_file(str(REPO / _rel), f"{REMOTE_ROOT}/{_rel}", copy=False)
if (REPO / "artifacts/rl_pool/screen").exists():
    driver_image = driver_image.add_local_dir(str(REPO / "artifacts/rl_pool/screen"), f"{REMOTE_ROOT}/artifacts/rl_pool/screen", copy=False, ignore=["**/__pycache__"])
driver_image = driver_image.add_local_dir(str(REPO / "wayfinder"), f"{REMOTE_ROOT}/wayfinder", copy=False, ignore=["**/__pycache__"])


class _Tee:
    def __init__(self, f, orig):
        self.f, self.orig = f, orig

    def write(self, x):
        self.orig.write(x)
        try:
            self.f.write(x)
        except Exception:  # noqa: BLE001
            pass
        return len(x)

    def flush(self):
        self.orig.flush()
        try:
            self.f.flush()
        except Exception:  # noqa: BLE001
            pass


@app.function(image=driver_image, volumes={"/gdata": gvol}, timeout=24 * 3600, cpu=2, memory=4096)
def driver_remote(cfg_a: dict, cfg: dict) -> dict:
    """The Driver loop of main() in the cloud. cfg_a = the Driver's argument namespace as a dict, cfg = the trainer/worker config."""
    import threading
    import types

    sys.path.insert(0, REMOTE_ROOT)
    from wayfinder.grpo.driver import Driver, remote_paths, retry_transient

    run = cfg["run"]
    a = types.SimpleNamespace(**remote_paths(cfg_a, Path("/nonexistent"), REMOTE_ROOT))
    a.merge_steps = tuple(int(x) for x in str(cfg_a.get("merge_steps", "5,10")).split(",") if x)
    out = Path(f"/gdata/{run}/driver")
    out.mkdir(parents=True, exist_ok=True)
    gvol.reload()
    sys.stdout = _Tee(open(out / "driver.log", "a"), sys.stdout)
    jobs = modal.Queue.from_name(f"wayfinder-grpo-jobs-{run}", create_if_missing=True)
    results = modal.Queue.from_name(f"wayfinder-grpo-results-{run}", create_if_missing=True)
    ctl = modal.Dict.from_name(f"wayfinder-grpo-ctl-{run}", create_if_missing=True)
    start_step = a.start_step
    if start_step == 0:
        jobs.clear(); results.clear(); ctl.clear()
    else:
        for key in ["stop"] + [f"{p}_{k}" for k in range(start_step, start_step + 3) for p in ("batch", "done")]:
            try:
                ctl.pop(key)
            except KeyError:
                pass
        jobs.clear(); results.clear()
    ctl.put("heartbeat", time.time())
    (out / "config.json").write_text(json.dumps({**cfg, "driver": cfg_a, "t_start": time.time()}, indent=1))
    d = Driver(a, ctl, jobs, results, out)
    if getattr(a, "g4_worker", False):  # --g4-worker only
        from wayfinder.grpo import g4_link

        d.g4 = g4_link.from_modal(run)
        d.g4.reset()
    stop_commit = threading.Event()

    def commit() -> None:
        try:
            gvol.commit()
        except Exception as e:  # noqa: BLE001
            print(f"[driver] volume commit failed: {e!r}", flush=True)

    d.on_step = lambda k: commit()

    def periodic() -> None:
        while not stop_commit.wait(120):
            commit()

    threading.Thread(target=periodic, daemon=True).start()
    trainer = pick_trainer(a.train_gpus)
    calls = [trainer.spawn(cfg)] + [worker.spawn(w, cfg) for w in range(a.workers)]
    print(f"[driver] cloud driver up: run {run}, start_step {start_step}, steps {a.steps}, pool {a.pool or 'auto'}, merge_steps {a.merge_steps}", flush=True)

    def merge_fn(base: str, r: str, step: int) -> dict:
        return merge.remote(base, r, step)

    err = None
    try:
        d.run(merge_fn)
        d.finish()
    except BaseException as e:  # noqa: BLE001
        err = repr(e)[:300]
        print(f"[driver] FATAL {err}", flush=True)
    finally:
        d.stop.set()
        try:
            retry_transient(ctl.put, "stop", True, max_s=120, label="ctl.put stop")
        except Exception:  # noqa: BLE001
            pass
        res = []
        for c in calls:
            try:
                res.append(c.get(timeout=300))
            except Exception as e:  # noqa: BLE001
                res.append({"error": repr(e)[:200]})
        (out / "workers_result.json").write_text(json.dumps(res, indent=1))
        print("workers/trainer:", json.dumps(res)[:1500], flush=True)
        stop_commit.set()
        commit()
    return {"run": run, "error": err, "stop_reason": d.stop_reason, "workers": res}


# ----------------------------------------------------------------------------------------------------------------------------------
# driver entrypoint
# ----------------------------------------------------------------------------------------------------------------------------------
@app.local_entrypoint()
def main(run: str = "smoke", steps: int = 30, tasks_per_step: int = 32, group: int = 8, workers: int = 4, train_gpus: int = 2, concurrency: int = 32,
         lr: float = 5e-5, beta: float = 0.02, clip: float = 0.2, precision_coef: float = 0.0, start: str = "/data/sft/v3_merged", eval_every: int = 10, eval_shards: int = 4,
         save_every: int = 10, pool: str = "", no_pool_switch: bool = False, max_seconds: int = 1200, max_turns: int = 40, budget_usd: float = 700.0, seed: int = 0,
         seed_salt: int = 0, straggle_s: int = 600, step_timeout_s: int = 5400, start_step: int = 0, resume_adapter: str = "", warmup: int = 2,
         reward: str = "legacy", reward_k: int = 12, reward_gold_window: int = 8, reward_free_files: int = 8, reward_file_cost: float = 0.03, nudge: str = "teacher",
         papo_lambda: float = 0.5, monitor: bool = False, eval_budget_usd: float = 1e9, guard_gn_mult: float = 0.0, guard_kl: float = 0.0, guard_stuck: int = 3, rewarm: int = 10, resume_slot_every: int = 1,
         remote_driver: bool = False, merge_steps: str = "5,10", fixed_tasks: str = "", fork_report: int = 0, fork_train: int = 2, lambda_fork: float = 1.0, fork_cached: bool = False, fork_cached_verify: int = 0, fork_cached_steps: str = "", g4_worker: bool = False, g4_tasks: int = 8, g4_grace_s: int = 120):
    """reward legacy|f1|rank5|rank5b; nudge teacher|f1 (f1 adds the report-order line to the prompt, the rollouts and every eval of the run); the update guard and
    the padding monitor are off unless set (arms a, b, c ran without them). Run f1: scripts/grpo/launch_f1.sh.
    --fork-report K (default 0 = off, byte-identical to before): after each rollout's report turn sample K more reports from the same prefix, give the K+1 reports an
    intra-prefix advantage (docs/grpo_fork_report.md); --fork-train M of the K forks are also trained (default 2), --lambda-fork weights the intra term (default 1.0).
    --fork-cached (default off; needs --fork-report): train the forked completions over a no-grad KV cache of their shared prefix (one prefix forward per prefix, gradients skip the
    prefix K/V), --fork-cached-verify N logs max|dlogp cached-full| on the first N prefix groups of each step.
    --g4-worker (default off, byte-identical to before): add --g4-tasks extra tasks per step for the optional GKE G4 rollout pod (wayfinder/rollout/g4_grpo_worker.py,
    docs/grpo_g4_worker.md); G4 rows count only when complete and current within --g4-grace-s of the step's Modal cut, otherwise the step runs on the Modal rows alone.
    --remote-driver: spawn `driver_remote` (the Driver in the cloud, evals off, checkpoints merged at --merge-steps) and exit; the laptop is not needed afterwards."""
    import types

    sys.path.insert(0, str(REPO))
    from wayfinder.grpo.driver import Driver, remote_paths

    a = types.SimpleNamespace(run=run, steps=steps, tasks_per_step=tasks_per_step, group=group, workers=workers, train_gpus=train_gpus, precision_coef=precision_coef, start=start,
                              eval_every=eval_every, eval_shards=eval_shards, pool=pool, no_pool_switch=no_pool_switch, budget_usd=budget_usd, seed=seed, seed_salt=seed_salt,
                              straggle_s=straggle_s, step_timeout_s=step_timeout_s, start_step=start_step, papo_lambda=papo_lambda, monitor=monitor, nudge=nudge, eval_budget_usd=eval_budget_usd, fixed_tasks=fixed_tasks)
    if fork_report < 0 or fork_train < 0 or (fork_report and reward != "rank5b"):
        raise SystemExit("--fork-report needs K >= 0 and --reward rank5b (the intra group is scored with rank5b)")
    if fork_cached and fork_report <= 0:
        raise SystemExit("--fork-cached needs --fork-report K > 0")
    if fork_report > 0:  # keys exist only in forked runs: a K=0 run's cfg, namespace and config.json are unchanged
        a.fork_report, a.lambda_fork = fork_report, lambda_fork
    if g4_worker and (g4_tasks <= 0 or g4_grace_s < 0):
        raise SystemExit("--g4-worker needs --g4-tasks > 0 and --g4-grace-s >= 0")
    if reward not in ("legacy", "f1", "rank5", "rank5b") or nudge not in ("teacher", "f1"):
        raise SystemExit(f"reward must be legacy|f1|rank5|rank5b and nudge teacher|f1, got {reward!r} {nudge!r}")
    out = REPO / "artifacts" / "grpo" / run
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"run": run, "start": start, "steps": steps, "lr": lr, "beta": beta, "clip": clip, "temperature": 0.85, "top_p": 0.95, "save_every": save_every, "concurrency": concurrency,
           "group_size": group, "max_seconds": max_seconds, "max_turns": max_turns, "semantic": "auto", "start_step": start_step, "resume_adapter": resume_adapter, "warmup": warmup,
           "precision_coef": precision_coef, "reward_kind": reward, "nudge": nudge, "papo_lambda": papo_lambda, "guard_gn_mult": guard_gn_mult, "guard_kl": guard_kl,
           "guard_stuck": guard_stuck, "rewarm": rewarm, "resume_slot_every": resume_slot_every,
           "reward": {"k": reward_k, "gold_window": reward_gold_window, "free_files": reward_free_files, "file_cost": reward_file_cost}}
    if fork_report > 0:
        cfg.update({"fork_report": fork_report, "fork_train": fork_train, "lambda_fork": lambda_fork})
    if fork_cached:
        cfg.update({"fork_cached": True, "fork_cached_verify": fork_cached_verify, **({"fork_cached_steps": fork_cached_steps} if fork_cached_steps else {})})
    if g4_worker:  # keys exist only in --g4-worker runs: a flag-off run's cfg, namespace and config.json are unchanged
        a.g4_worker, a.g4_tasks, a.g4_grace_s, a.g4_cfg = True, g4_tasks, g4_grace_s, cfg
    (out / "config.json").write_text(json.dumps({**cfg, **({"g4_worker": True, "g4_tasks": g4_tasks, "g4_grace_s": g4_grace_s} if g4_worker else {}), "workers": workers, "train_gpus": train_gpus, "tasks_per_step": tasks_per_step, "eval_every": eval_every, "pool": pool or "auto",
                                                 "seed": seed, "seed_salt": seed_salt, "t_start": time.time(), "remote_driver": remote_driver}, indent=1))
    active = REPO / "artifacts" / "grpo" / "active_runs.txt"
    runs = {x.strip() for x in active.read_text().splitlines() if x.strip()} if active.exists() else set()
    active.write_text("\n".join(sorted(runs | {run})) + "\n")  # arms the watcher's heartbeat rule for this run

    if remote_driver:
        cfg_a = {k: v for k, v in vars(a).items()} | {"merge_steps": merge_steps}
        cfg_a = remote_paths(cfg_a, REPO, REMOTE_ROOT)
        if start_step > 0:  # seed the volume with the earlier attempt's driver outputs so the monitor streak and logs continue (never overwrites)
            try:
                with gvol.batch_upload(force=False) as up:
                    for f in ("steps.jsonl", "rollouts.jsonl", "state.json"):
                        if (out / f).exists():
                            up.put_file(str(out / f), f"/{run}/driver/{f}")
            except Exception as e:  # noqa: BLE001
                print(f"seed upload: {e!r} (continuing; files probably exist already)", flush=True)
        call = driver_remote.spawn(cfg_a, cfg)
        print(f"remote driver spawned: {call.object_id} (run {run}); outputs on volume wayfinder-grpo under /{run}/driver/; heartbeat in wayfinder-grpo-ctl-{run}", flush=True)
        return

    jobs = modal.Queue.from_name(f"wayfinder-grpo-jobs-{run}", create_if_missing=True)
    results = modal.Queue.from_name(f"wayfinder-grpo-results-{run}", create_if_missing=True)
    ctl = modal.Dict.from_name(f"wayfinder-grpo-ctl-{run}", create_if_missing=True)
    if start_step == 0:  # a fresh run: clear control state left by an earlier attempt under this name
        jobs.clear()
        results.clear()
        ctl.clear()
    else:  # a resume: the previous driver set `stop` on exit; leaving it would stop the workers and the trainer at once
        try:
            ctl.pop("stop")
        except KeyError:
            pass
        for k in range(start_step, start_step + 3):
            for key in (f"batch_{k}", f"done_{k}"):  # stale step records of the interrupted attempt would be mistaken for this attempt's
                try:
                    ctl.pop(key)
                except KeyError:
                    pass
        jobs.clear()
        results.clear()
    ctl.put("heartbeat", time.time())

    d = Driver(a, ctl, jobs, results, out)
    if g4_worker:  # --g4-worker only
        from wayfinder.grpo import g4_link

        d.g4 = g4_link.from_modal(run)
        d.g4.reset()
    trainer = pick_trainer(train_gpus)
    calls = [trainer.spawn(cfg)] + [worker.spawn(w, cfg) for w in range(workers)]

    def merge_fn(base: str, r: str, step: int) -> dict:
        return merge.remote(base, r, step)

    try:
        d.run(merge_fn)
        d.finish()
    finally:
        d.stop.set()
        try:
            ctl.put("stop", True)
        except Exception:  # noqa: BLE001
            pass
        res = []
        for c in calls:
            try:
                res.append(c.get(timeout=300))
            except Exception as e:  # noqa: BLE001
                res.append({"error": repr(e)[:200]})
        (out / "workers_result.json").write_text(json.dumps(res, indent=1))
        print("workers/trainer:", json.dumps(res)[:1500], flush=True)
        left = [x.strip() for x in active.read_text().splitlines() if x.strip() and x.strip() != run]
        active.write_text("\n".join(left) + ("\n" if left else ""))
