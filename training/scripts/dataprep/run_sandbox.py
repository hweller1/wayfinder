"""Local driver: run a dataprep script inside a Modal CPU-only Sandbox (owner instruction 2026-10-26).

Only public inputs are staged (shards_pub, eval_tasks.jsonl, public transcripts, probe files). Never uploads
../benchmark-repo/.env or data/pool_internal/*. Prints sandbox wall time so the caller can record spend.

    python run_sandbox.py --script build_filter.py --stage filter --fetch out/task_filter.json=data/task_filter.json -- --inputs /in --out /out/task_filter.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
WF = HERE.parent.parent
AS = WF.parent / "benchmark-repo"
SCOUT = WF.parent / "voyage-scout"

# remote path -> local path. Public files only.
STAGE = {
    "splits": {
        "/in/eval_tasks.jsonl": SCOUT / "eval/eval_tasks.jsonl",
        **{f"/in/shards_pub/pub_s{i}.json": AS / f"artifacts/shards_pub/pub_s{i}.json" for i in range(8)},
    },
    "filter": {
        "/in/splits.json": WF / "data/splits.json",
        **{f"/in/shards_pub/pub_s{i}.json": AS / f"artifacts/shards_pub/pub_s{i}.json" for i in range(8)},
        **{f"/in/modal_pub/v3_pub_{a}_sophisticated_agents-a1-35b_merged.jsonl": AS / f"artifacts/remote/modal_pub/v3_pub_{a}_sophisticated_agents-a1-35b_merged.jsonl" for a in ("b2", "c", "g")},
        **{f"/in/sonnet_pubnew/v3_pubnew_{a}_sophisticated_sonnet-5.5-high_checkpoint.jsonl": AS / f"artifacts/remote/sonnet_pubnew/v3_pubnew_{a}_sophisticated_sonnet-5.5-high_checkpoint.jsonl" for a in ("b2", "c")},
        "/in/v3_a_probe_public_checkpoint.jsonl": AS / "artifacts/v3_a_probe_public_checkpoint.jsonl",
        "/in/no_retrieval_required.json": AS / "artifacts/no_retrieval_required.json",
    },
}
_E2E = AS / "experiments/e2e"
STAGE["rubric"] = {"/in/screen_r3.json": _E2E / "screen_r3.json"}
for _c in ("35b_grep", "35b_index", "sonnet_grep", "sonnet_index"):  # no-search / oracle records carry no retrieval features
    for _f in sorted((_E2E / "runs_r2/issue" / _c).glob("*.json")):
        if not _f.name.endswith(("transcript.json", "error.json")):  # coder transcripts excluded
            STAGE["rubric"][f"/in/runs_r2/issue/{_c}/{_f.name}"] = _f
STAGE["tokenizer"] = {"/in/scout_tool_specs.json": WF / "serve/scout_tool_specs.json"}
FORBIDDEN = (".env", "pool_internal")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(STAGE))
    ap.add_argument("--fetch", action="append", default=[], help="REMOTE=LOCAL (repeatable)")
    ap.add_argument("--pip", default="", help="space-separated pip packages to add to the image")
    ap.add_argument("--secret", default="", help="Modal secret name to attach")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()

    files = STAGE[args.stage]
    for remote, local in files.items():
        assert not any(f in str(local) for f in FORBIDDEN), local
        assert local.exists(), local

    app = modal.App.lookup("wayfinder-dataprep", create_if_missing=True)
    vol = modal.Volume.from_name(f"wayfinder-dataprep-{args.stage}", create_if_missing=True)
    # Sandbox exec / sb.open hung in this workspace on 2026-10-26; entrypoint-command + Volume staging works.
    with vol.batch_upload(force=True) as up:
        for remote, local in files.items():
            up.put_file(str(local), "/in/" + remote.removeprefix("/in/"))
        for f in ("common.py", args.script):
            up.put_file(str(HERE / f), f"/work/{f}")
    print("staged", flush=True)
    image = modal.Image.debian_slim(python_version="3.12")
    if args.pip:
        image = image.pip_install(*args.pip.split())
    cmd = "mkdir -p /vol/out && cd /vol/work && python /vol/work/%s %s> /vol/out/_log.txt 2>&1; echo EXIT=$? > /vol/out/_exit" % (
        args.script, " ".join(a.replace("/in", "/vol/in").replace("/out", "/vol/out") for a in args.rest))
    t0 = time.time()
    sb = modal.Sandbox.create("sh", "-c", cmd, app=app, image=image, secrets=[modal.Secret.from_name(args.secret)] if args.secret else [],
                              cpu=2, memory=4096, timeout=args.timeout, volumes={"/vol": vol})
    try:
        sb.wait()
        # sb.stdout/sb.stderr streams also hang here (DNS to the command router); the log goes through the volume.
        data = {}
        for spec in args.fetch:
            remote, local = spec.split("=", 1)
            blob = b"".join(vol.read_file(remote))
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            Path(local).write_bytes(blob)
            print("fetched", remote, "->", local, len(blob), "bytes")
        print(b"".join(vol.read_file("out/_log.txt")).decode()[-3000:])
        ex = b"".join(vol.read_file("out/_exit")).decode().strip()
        assert ex == "EXIT=0", ex
    finally:
        sb.terminate()
        secs = time.time() - t0
        print(json.dumps({"sandbox_wall_s": round(secs, 1), "est_usd": round(secs * 2 * 0.0000131 + secs * 4 * 0.00000222, 4)}))


if __name__ == "__main__":
    main()
