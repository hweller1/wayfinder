#!/usr/bin/env python3
"""Idempotent installer for the scout-wayfinder Muse plugin; `scout init --muse-code` runs it.

  python3 install.py [--muse PATH] [--demo-project PATH] [--lane production|demo] [--key-file PATH]
                     [--scout-bin PATH]

Steps, each idempotent: `muse plugins marketplace add voyage-scout <this dir>` (`update` when it exists),
`muse plugins install scout-wayfinder@voyage-scout` (`update` when installed), `muse plugins approve
scout-wayfinder` (hooks, MCP server and the scout-explorer Agent Definition), then the plugin's `scout.env` in
Muse's plugin data dir. XDG_CONFIG_HOME / XDG_DATA_HOME scope it, so an isolated install is
`XDG_CONFIG_HOME=... XDG_DATA_HOME=... python3 install.py`. Prints one JSON report and never a key.
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
MARKET, PLUGIN = "voyage-scout", "scout-wayfinder"


def muse(binary: str, *args: str) -> tuple[int, dict | str]:
    proc = subprocess.run(  # noqa: S603
        [binary, "plugins", *args, "--json"], capture_output=True, text=True, timeout=120, check=False
    )
    try:
        return proc.returncode, json.loads(proc.stdout)
    except ValueError:
        return proc.returncode, (proc.stdout + proc.stderr).strip()[-300:]


def data_dir(env: dict[str, str]) -> Path:
    base = env.get("XDG_DATA_HOME") or str(Path(env.get("HOME", str(Path.home()))) / ".local" / "share")
    return Path(base) / "muse" / "plugins" / "data" / PLUGIN


def write_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = []
    if path.exists():
        keep = [x for x in path.read_text(encoding="utf-8").splitlines() if x.split("=", 1)[0] not in values]
    path.write_text("\n".join([*keep, *(f"{k}={v}" for k, v in values.items())]) + "\n", encoding="utf-8")
    path.chmod(0o600)


def main() -> int:
    ap = argparse.ArgumentParser(description="Install and approve the scout-wayfinder Muse plugin")
    ap.add_argument("--muse", default=shutil.which("muse") or str(Path.home() / ".local" / "bin" / "muse"))
    ap.add_argument("--demo-project", default=str(HERE.parents[1]), help="voyage-scout checkout (demo lane)")
    ap.add_argument(
        "--lane", choices=["production", "demo"], default=None, help="force a lane (default: auto)"
    )
    ap.add_argument("--key-file", default=None, help="file holding a scout key (default: `scout login`)")
    ap.add_argument("--scout-bin", default=None, help="the `scout` binary (default: PATH, ~/.local/bin)")
    args = ap.parse_args()
    report: dict = {"plugin": PLUGIN, "steps": []}

    def step(name: str, *a: str, ok_if: str = "") -> bool:
        rc, out = muse(args.muse, *a)
        good = rc == 0 or (ok_if and ok_if in json.dumps(out))
        report["steps"].append({"step": name, "rc": rc, "ok": bool(good)})
        return bool(good)

    listed = muse(args.muse, "marketplace", "list")[1]
    have_market = MARKET in json.dumps(listed)
    ok = (
        step("marketplace update", "marketplace", "update", MARKET)
        if have_market
        else step("marketplace add", "marketplace", "add", MARKET, str(HERE))
    )
    installed = PLUGIN in json.dumps(muse(args.muse, "list")[1])
    if ok:
        ok = (
            step("update", "update", PLUGIN)
            if installed
            else step("install", "install", f"{PLUGIN}@{MARKET}")
        )
    if ok:
        ok = step("approve", "approve", PLUGIN)
    if ok:
        values = {"SCOUT_DEMO_PROJECT": args.demo_project}
        if args.lane:
            values["SCOUT_LANE"] = args.lane
        if args.key_file:
            values["SCOUT_KEY_FILE"] = str(Path(args.key_file).expanduser().resolve())
            values["SCOUT_KEYSTORE"] = "file"  # the key file is the source; do not probe the Keychain
        if args.scout_bin:
            values["SCOUT_BIN"] = str(Path(args.scout_bin).expanduser().resolve())
        target = data_dir(dict(os.environ)) / "scout.env"
        write_env(target, values)
        report["config"] = str(target)
        rc, insp = muse(args.muse, "inspect", PLUGIN)
        if isinstance(insp, dict):
            report["capabilities"] = {
                (c["candidate"].get("stable_id") or c["candidate"].get("scoped_definition_id")): c["status"]
                for c in insp.get("runtime_capabilities", [])
            }
    report["ok"] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
