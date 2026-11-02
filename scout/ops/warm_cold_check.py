"""Confirm the non-blocking warm path on N cold starts of wayfinder-v3-serve.

Per cycle: wait until v3 has scaled to zero by itself (nothing is stopped), run
`scout-demo warm --trigger --preset v3` (the Muse plugin's demo-lane SessionStart path; the production lane
goes through the gateway's `/v1/warm`, same Modal endpoint), record its mode, wall time and exit code, then
poll `GET /v1/models` on the model URL until 200 (time to ready). Keys come from ../wayfinder/.env and are
never printed.

    uv run python ops/warm_cold_check.py --cycles 3 --out artifacts/live/muse_warm_cold3.json
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
APP = "wayfinder-v3-serve"
DEMO = REPO / ".venv" / "bin" / "scout-demo"


def env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def containers() -> list[str]:
    cmd = ["modal", "container", "list", "--json"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603, S607
    try:
        return [c["container_id"] for c in json.loads(res.stdout) if c.get("app_name") == APP]
    except ValueError:
        return []


def to_zero(timeout_s: float = 1800) -> float:
    """Wait (never stop anything) until the app has scaled to zero on its own idle window."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not containers():
            return time.time() - t0
        time.sleep(20)
    raise RuntimeError("v3 did not reach zero containers (someone is using it)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "live" / "muse_warm_cold3.json"))
    args = ap.parse_args()
    wf = env_file(REPO.parent / "wayfinder" / ".env")
    base, key = wf["V3_BASE_URL"].rstrip("/"), wf["V3_API_KEY"]
    rows = []
    for n in range(args.cycles):
        zero_s = to_zero()
        t0 = time.time()
        proc = subprocess.run(  # noqa: S603
            [str(DEMO), "warm", "--trigger", "--preset", "v3"], capture_output=True, text=True, timeout=900,
            check=False, env={**os.environ, "SCOUT_DEMO_PRESET": "v3"},
        )  # fmt: skip
        trigger_s = time.time() - t0
        try:
            doc = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            doc = {}
        ready_s = None
        while time.time() - t0 < 600:
            try:
                r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
                if r.status_code == 200:
                    ready_s = time.time() - t0
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        row = {
            "cycle": n + 1,
            "to_zero_s": round(zero_s, 1),
            "rc": proc.returncode,
            "mode": doc.get("mode"),
            "trigger_wall_s": round(trigger_s, 2),
            "ready_s": round(ready_s, 1) if ready_s else None,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    Path(args.out).write_text(json.dumps({"app": APP, "path": "scout-demo warm --trigger --preset v3",
                                          "rows": rows}, indent=2) + "\n")  # fmt: skip


if __name__ == "__main__":
    main()
