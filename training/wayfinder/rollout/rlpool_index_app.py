"""RL-pool variant of index_app.py (copy, original untouched): app `wayfinder-rlpool-index`, units come from a JSON file
(`{"repo": [commit,...]}`) instead of data/splits.json tiers, same registry/spend Dicts and the same indexing code path.

    modal run --detach wayfinder/rollout/rlpool_index_app.py::launch_units --units artifacts/rl_pool/index_units.json --cap-usd 150

Original docstring follows.

Bulk-index every rollout repo@commit into turbopuffer through scout's real secure-indexing path, inside Modal.

    modal run --detach wayfinder/rollout/index_app.py::launch            # tiers 1..3, all of train+dev+test
    modal run wayfinder/rollout/index_app.py::launch --tiers dev --limit 3

Tiers (each finishes before the next starts): `rl` = commits of accepted, rl_eligible train tasks; `dev` = commits of dev
tasks (the 49 Stage-B tasks first); `rest` = every other train/dev/test commit in data/splits.json. A unit of work is one
repo with the commits needed in that tier (commits run sequentially, so the embed cache is shared across them).

Namespaces: scout_dev_wf_<repo-slug>_<commit8>(+__tree) per repo@commit; scout_dev_wf_<repo-slug>__{cache,meta,keys} per repo
(see wayfinder/rollout/semantic.py). Keys: Modal secret `wayfinder-index-keys` (TURBOPUFFER_API_KEY, VOYAGE_API_KEY,
WF_INDEX_MASTER_KEY_HEX; built by ops/make_index_secret.py). Registry: Modal Dict `wayfinder-index-registry`
(key `repo@commit` -> index report); spend: Dict `wayfinder-index-spend` (est. Voyage USD, tokens x $0.18/M, the
voyage-code-3 list price that scout's demo uses as its estimate). Workers stop starting commits at `cap_usd`.
Repo@commits whose indexable text exceeds `max_tokens` (estimated) are skipped and recorded as `too_big`.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
SCOUT_SRC = REPO.parent / "voyage-scout" / "src"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ripgrep", "ca-certificates")
    .pip_install(
        "httpx==0.28.1", "pydantic>=2.9", "pathspec==1.1.1", "watchfiles==1.3.0", "cryptography==50.0.2", "keyring==25.7.0",
        "turbopuffer==3.0.0", "voyageai==0.5.0", "tree-sitter==0.26.0", "tree-sitter-python==0.25.0", "tree-sitter-go==0.25.0",
        "tree-sitter-javascript==0.25.0", "tree-sitter-typescript==0.23.2", "tree-sitter-java==0.23.5",
        "tree-sitter-rust==0.24.2", "tree-sitter-c==0.24.2", "tree-sitter-cpp==0.23.4", "tree-sitter-c-sharp==0.23.5",
    )
    .env({"SCOUT_SRC": "/scout_src", "PYTHONPATH": "/root", "GIT_TERMINAL_PROMPT": "0"})
    .add_local_dir(str(SCOUT_SRC), "/scout_src", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(str(REPO / "wayfinder"), "/root/wayfinder", copy=False, ignore=["**/__pycache__"])
)
app = modal.App(os.environ.get("WF_INDEX_APP", "wayfinder-rlpool-index"), image=image)
SECRET = modal.Secret.from_name("wayfinder-index-keys")
USD_PER_M = 0.18
CHARS_PER_TOKEN = 3.5


@app.function(timeout=3 * 3600, cpu=2, memory=6144, secrets=[SECRET], max_containers=int(os.environ.get("WF_INDEX_MAX_CONTAINERS", "128")),
              retries=modal.Retries(max_retries=2, initial_delay=15.0, backoff_coefficient=1.5))
async def index_repo(repo: str, commits: list[str], cap_usd: float, max_tokens: int) -> list[dict]:
    import shutil

    from wayfinder.rollout.run import checkout
    from wayfinder.rollout.semantic import index_checkout, ns_base

    reg = modal.Dict.from_name("wayfinder-index-registry", create_if_missing=True)
    spend = modal.Dict.from_name("wayfinder-index-spend", create_if_missing=True)
    out: list[dict] = []
    for c in commits:
        key = f"{repo}@{c}"
        prev = reg.get(key)
        if prev and prev.get("state") == "ready" and prev.get("coverage", 0) >= 0.8:
            out.append({**prev, "skipped": "already_indexed"})
            continue
        if sum(spend.values()) >= cap_usd:
            out.append({"repo": repo, "base_commit": c, "state": "budget_stop"})
            continue
        root = None
        try:
            root, clone_s = await asyncio.to_thread(checkout, repo, c, "/tmp/repos")
            from scout.adapters.fs_local import LocalFileSystem
            from scout.core.local.ignore import Matcher
            from scout.core.sync.files import list_indexable

            fs = LocalFileSystem(root)
            listing = await asyncio.to_thread(list_indexable, fs, Matcher.from_fs(fs))
            size = 0
            for p in listing.paths:
                try:
                    size += fs.stat(p).size
                except Exception:  # noqa: BLE001
                    pass
            est = int(size / CHARS_PER_TOKEN)
            if est > max_tokens:
                rec = {"repo": repo, "base_commit": c, "namespace": ns_base(repo, c), "state": "too_big",
                       "est_tokens": est, "files": len(listing.paths)}
                reg.put(key, rec)
                out.append(rec)
                continue
            rep = await index_checkout(root, repo, c, "/tmp/wf_home_" + str(abs(hash(key)) % 10**8))
            rep["est_tokens"], rep["clone_s"] = est, round(clone_s, 1)
            rep["est_cost_usd"] = round(rep["voyage_tokens"] * USD_PER_M / 1e6, 5)
            reg.put(key, rep)
            if rep["voyage_tokens"]:
                spend[f"{repo}@{c}"] = rep["est_cost_usd"]
            out.append(rep)
        except Exception as e:  # noqa: BLE001
            out.append({"repo": repo, "base_commit": c, "state": "error", "error": f"{type(e).__name__}: {e}"[:300]})
        finally:
            if root:
                shutil.rmtree(root, ignore_errors=True)
    return out


def _tiers(only: str | None) -> list[tuple[str, dict[str, list[str]]]]:
    spl = json.loads((REPO / "data" / "splits.json").read_text())["tasks"]
    flt = json.loads((REPO / "data" / "task_filter.json").read_text())["tasks"]
    stage_b = {t["task_id"] for t in json.loads((REPO / "artifacts/teacher_test/stageB_tasks.json").read_text())} \
        if (REPO / "artifacts/teacher_test/stageB_tasks.json").exists() else set()
    seen: set[tuple[str, str]] = set()

    def take(pred) -> dict[str, list[str]]:  # type: ignore[no-untyped-def]
        d: dict[str, list[str]] = {}
        for tid, t in sorted(spl.items()):
            k = (t["repo_full"], t["base_commit"])
            if k in seen or t["split"] not in ("train", "dev", "test") or not pred(tid, t):
                continue
            seen.add(k)
            d.setdefault(t["repo_full"], []).append(t["base_commit"])
        return d

    tiers = [
        ("rl", take(lambda tid, t: t["split"] == "train" and flt[tid].get("rl_eligible"))),
        ("dev", take(lambda tid, t: t["split"] == "dev" and tid in stage_b)),
        ("dev_rest", take(lambda tid, t: t["split"] == "dev")),
        ("rest", take(lambda tid, t: True)),
    ]
    return [x for x in tiers if only in (None, "all", x[0]) or (only == "dev" and x[0] == "dev_rest")]


@app.function(timeout=24 * 3600, cpu=1, memory=2048, secrets=[SECRET])
def orchestrator(tiers: list[tuple[str, dict[str, list[str]]]], cap_usd: float, max_tokens: int) -> dict:
    spend = modal.Dict.from_name("wayfinder-index-spend", create_if_missing=True)
    prog = modal.Dict.from_name("wayfinder-index-progress", create_if_missing=True)
    t0 = time.time()
    tot = {"ok": 0, "skipped": 0, "too_big": 0, "error": 0, "budget_stop": 0}
    for name, units in tiers:
        args = [(r, cs, cap_usd, max_tokens) for r, cs in units.items()]
        n_commits = sum(len(cs) for _, cs in units.items())
        print(f"tier {name}: {len(args)} repos, {n_commits} repo@commits", flush=True)
        done = 0
        for res in index_repo.starmap(args, order_outputs=False, return_exceptions=True):
            done += 1
            if isinstance(res, BaseException):
                tot["error"] += 1
                print("unit failed:", repr(res)[:200], flush=True)
                continue
            for r in res:
                st = r.get("state")
                if r.get("skipped"):
                    tot["skipped"] += 1
                elif st in ("ready", "partial"):
                    tot["ok"] += 1
                else:
                    tot[st if st in tot else "error"] += 1
                    if st == "error":
                        print("error:", r.get("repo"), r.get("error"), flush=True)
            prog["state"] = {"tier": name, "units_done": done, "units": len(args), "totals": tot, "elapsed_s": round(time.time() - t0),
                             "est_usd": round(sum(spend.values()), 2)}
        print(f"tier {name} done: {tot} est ${sum(spend.values()):.2f} in {time.time() - t0:.0f}s", flush=True)
    return {"totals": tot, "est_usd": round(sum(spend.values()), 2), "wall_s": round(time.time() - t0)}


@app.local_entrypoint()
def launch_units(units: str, cap_usd: float = 150.0, max_tokens: int = 25_000_000, limit: int = 0, name: str = "rlpool"):
    u = json.loads(Path(units).read_text())
    items = sorted(u.items())
    if limit:
        items = items[:limit]
    ts = [(name, dict(items))]
    print({n: (len(x), sum(len(c) for c in x.values())) for n, x in ts}, flush=True)
    print(json.dumps(orchestrator.remote(ts, cap_usd, max_tokens)), flush=True)
