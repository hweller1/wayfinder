#!/usr/bin/env python3
"""SessionStart hook: report scout's readiness for this checkout to the model, and warm Wayfinder in the
background so the first explore call does not pay the 3-5 minute cold start.

Returns in well under a second: readiness is read from local files (no network), and the warm-up runs in a
detached child (`warm.py`), rate-limited to one per 5 minutes per checkout. Output: one
`hookSpecificOutput.additionalContext` line. Never fails the session: any error prints nothing and exits 0.
`SCOUT_NO_WARM=1` (config or env) skips the warm-up; `SCOUT_NO_WATCH=1` skips starting `scout watch`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, rel: str):  # a sibling by path, so this file imports only the stdlib
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / rel)
    if spec is None or spec.loader is None:
        raise ImportError(rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lanes = _load("lanes", "../lib/lanes.py")

WARM_EVERY_S = 300


def context_line(r: dict) -> str:
    agent = lanes.EXPLORER
    if r.get("ready"):
        cov = r.get("coverage")
        cov_s = f", coverage {cov:.2f}" if isinstance(cov, (int, float)) else ""
        repo = f", demo repo `{r['repo']}`" if r.get("repo") else ""
        return (
            f"scout: index {r.get('state', 'ready')} for this checkout ({r['lane']} lane{repo}{cov_s}). "
            f"For broad where/how questions or before changing unfamiliar code, delegate to the `{agent}` "
            "subagent (see the scout-explore skill) and wait for it (subagent_wait timeout_ms 300000; it "
            "takes 1-3 min); use plain search for exact strings."
        )
    return (
        f"scout: not ready for this checkout ({r['lane']} lane: {r.get('reason') or 'unknown'}). "
        f"`{agent}` still works but falls back to plain search; `/scout-wayfinder:index` builds the index."
    )


def should_warm(data_dir: Path, root: str, now: float) -> bool:
    stamp = data_dir / f"warm-{hashlib.sha256(root.encode()).hexdigest()[:12]}.stamp"
    try:
        if now - stamp.stat().st_mtime < WARM_EVERY_S:
            return False
    except OSError:
        pass
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(int(now)), encoding="utf-8")
    except OSError:
        return False
    return True


def spawn_warm(cwd: str, env: dict[str, str], data_dir: Path) -> None:
    log = data_dir / "warm.log"
    with open(log, "ab") as out:
        subprocess.Popen(  # noqa: S603
            [sys.executable, str(HERE / "warm.py"), cwd],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=out,
            start_new_session=True,  # detached: Muse does not wait for it and it survives the hook
        )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        payload = {}
    try:
        env = dict(os.environ)
        cwd = str(payload.get("cwd") or os.getcwd())
        r = lanes.readiness(cwd, env)
        cfg = lanes.load_config(env)
        data_dir = Path(env.get("MUSE_PLUGIN_DATA_DIR") or env.get("TMPDIR") or tempfile.gettempdir())
        warm = "skipped"
        if r.get("root") and not cfg.get("SCOUT_NO_WARM") and should_warm(data_dir, r["root"], time.time()):
            spawn_warm(r["root"], env, data_dir)
            warm = "started"
        out = {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context_line(r)}}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stderr.write(f"scout-wayfinder: lane={r['lane']} ready={r['ready']} warm={warm}\n")
    except Exception:  # noqa: BLE001  never break the user's session
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
