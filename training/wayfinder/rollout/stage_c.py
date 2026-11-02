"""Stage C probes against an OpenAI-compatible endpoint: prefix-cache probe and concurrency latency / 429 counts.
    python -m wayfinder.rollout.stage_c --out artifacts/teacher_test/stageC.json
Requests are real-shaped (scout system prompt + tool schemas + a unique brief), no retries, so 429s are visible."""
from __future__ import annotations

import argparse, asyncio, json, os, statistics as st, time
from pathlib import Path

import httpx

from wayfinder.rollout.chat import SPARK_OPEN_BASETEN

ROOT = Path(__file__).resolve().parents[2]
URL = SPARK_OPEN_BASETEN["base_url"] + "/chat/completions"
H = {"Authorization": f"Api-Key {os.environ.get('BASETEN_API_KEY', '')}"}
MODEL = SPARK_OPEN_BASETEN["model"]
TOOLS = [t for t in json.loads((ROOT / "serve/scout_tool_specs.json").read_text()) if t["function"]["name"] != "semantic_search"]
SYSTEM = (ROOT / "serve/scout_product_prompt.md").read_text()


async def post(c: httpx.AsyncClient, body: dict) -> dict:
    t0 = time.monotonic()
    try:
        r = await c.post(URL, headers=H, json=body)
        d = r.json() if r.status_code == 200 else {}
        u = d.get("usage") or {}
        return {"status": r.status_code, "s": time.monotonic() - t0, "in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
                "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "retry_after": r.headers.get("retry-after")}
    except Exception as e:  # noqa: BLE001
        return {"status": type(e).__name__, "s": time.monotonic() - t0}


async def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out: dict = {}
    async with httpx.AsyncClient(timeout=180) as c:
        prefix = "Repository notes. " + "The quick brown fox jumps over the lazy dog near the riverbank. " * 220
        base = {"model": MODEL, "max_tokens": 8, "temperature": 0, "messages": [{"role": "system", "content": prefix}, {"role": "user", "content": "Say OK."}]}
        out["cache_probe"] = []
        for i in range(4):
            out["cache_probe"].append(await post(c, base)); await asyncio.sleep(2)
        for n in (8, 32, 64):
            async def one(i: int) -> dict:
                body = {"model": MODEL, "max_tokens": 120, "temperature": 0.85, "top_p": 0.95, "tools": TOOLS, "tool_choice": "auto",
                        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Find where request {i} retry backoff is computed in the repo."}]}
                return await post(c, body)
            t0 = time.monotonic()
            res = await asyncio.gather(*[one(i) for i in range(n)])
            ok = sorted(r["s"] for r in res if r["status"] == 200)
            q = lambda p: ok[min(len(ok) - 1, int(p * len(ok)))] if ok else None  # noqa: E731
            out[f"conc_{n}"] = {"n": n, "wall_s": round(time.monotonic() - t0, 1), "ok": len(ok), "n429": sum(r["status"] == 429 for r in res),
                                "other_err": [r["status"] for r in res if r["status"] not in (200, 429)],
                                "p50_s": round(st.median(ok), 2) if ok else None, "p95_s": round(q(0.95), 2) if ok else None,
                                "max_s": round(ok[-1], 2) if ok else None, "cached_in_total": sum(r.get("cached") or 0 for r in res), "in_total": sum(r.get("in") or 0 for r in res)}
            print(n, out[f"conc_{n}"], flush=True)
            await asyncio.sleep(20)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
