#!/usr/bin/env python3
# ruff: noqa: S603  (fixed argv: the claude CLI and this plugin's own script)
"""Install and verify the scout plugin in Claude Code; `scout init --claude-code` runs it.

  python3 install.py [--claude PATH] [--dry-run]

Steps, each idempotent: `claude plugin validate <this dir>`, `claude plugin marketplace add <this dir>`
(`update` when it is already known), `claude plugin install scout@voyage-scout`, then verify: the plugin is
listed and enabled, and the plugin's own doctor (python, `scout` binary, key configured, lane, gateway
healthz) passes. `CLAUDE_CONFIG_DIR` scopes everything, so an isolated install is
`CLAUDE_CONFIG_DIR=/tmp/x python3 install.py`. Prints one JSON report on stdout and never a key.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKET, PLUGIN = "voyage-scout", "scout"


def run(argv: list[str], timeout: int = 180) -> tuple[int, str]:
    try:
        p = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, type(e).__name__
    return p.returncode, (p.stdout + p.stderr).strip()


def last(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1][:200] if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", default=shutil.which("claude") or "claude")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    c = args.claude
    steps: list[tuple[str, list[str]]] = [("validate", [c, "plugin", "validate", str(HERE)])]
    rc, out = run([c, "plugin", "marketplace", "list", "--json"])
    known = rc == 0 and any(m.get("name") == MARKET for m in _json_list(out))
    if known:
        steps.append(("marketplace update", [c, "plugin", "marketplace", "update", MARKET]))
    else:
        steps.append(("marketplace add", [c, "plugin", "marketplace", "add", str(HERE)]))
    steps.append(("install", [c, "plugin", "install", f"{PLUGIN}@{MARKET}", "--json"]))
    report: dict = {"config_dir": os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude", "steps": [], "ok": True}
    if args.dry_run:
        report.update(ok=None, steps=[{"step": n, "argv": a} for n, a in steps])
        print(json.dumps(report))
        return 0
    for name, argv in steps:
        rc, out = run(argv)
        report["steps"].append({"step": name, "rc": rc, "tail": last(out)})
        if rc != 0:
            report["ok"] = False
            print(json.dumps(report))
            return 1
    rc, out = run([c, "plugin", "list", "--json"])
    entry = next((p for p in _json_list(out) if p.get("id") == f"{PLUGIN}@{MARKET}"), None)
    report["installed"] = bool(entry and entry.get("enabled"))
    report["version"] = entry.get("version") if entry else None
    rc, out = run([sys.executable, str(HERE / "lib" / "scout_cc.py"), "doctor", "--data", os.devnull])
    try:
        report["doctor"] = json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        report["doctor"] = {"ok": False, "error": last(out)}
    report["ok"] = report["installed"] and bool(report["doctor"].get("scout_binary"))
    report["next"] = (
        "cd <your git checkout> && claude, then /scout:index (once) and /scout:status"
        if report["ok"]
        else "install scout (uv tool install), run `scout login`, re-run scout init --claude-code"
    )
    print(json.dumps(report))
    return 0 if report["ok"] else 1


def _json_list(text: str) -> list[dict]:
    start = text.find("[")
    try:
        data = json.loads(text[start:]) if start >= 0 else []
    except ValueError:
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


if __name__ == "__main__":
    sys.exit(main())
