"""Bearer-key allow-list reverse proxy in front of vLLM, for the wayfinder-qwen Modal app (serve_qwen.py).

Listens on WAYFINDER_PORT (default 8000) and forwards to vLLM on 127.0.0.1:WAYFINDER_UPSTREAM_PORT (default 8001).

- Enforces `Authorization: Bearer $WAYFINDER_API_KEY` on `/v1/*` (constant-time compare). vLLM itself runs without a key.
- Forwards only `/v1/*`, `/health`, `/version` and `/ping`; every other path is a 404, so vLLM's other endpoints are
  never reachable from outside the container.
- Renames `reasoning_content` to `reasoning` in the assistant messages of `/v1/chat/completions` requests. vLLM 0.32.0
  renders earlier-turn reasoning in the Qwen3.6 chat template only from `reasoning`, and drops `reasoning_content`
  (empty think blocks in the prompt). The Scout gateway sends `reasoning_content`.
"""

from __future__ import annotations

import hmac
import json
import os

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = f"http://127.0.0.1:{os.environ.get('WAYFINDER_UPSTREAM_PORT', '8001')}"
API_KEY = os.environ["WAYFINDER_API_KEY"]
PORT = int(os.environ.get("WAYFINDER_PORT", "8000"))
ALLOWED_EXACT = {"/health", "/version", "/ping"}
HOP = {"host", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "content-length"}

client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(None, connect=10), limits=httpx.Limits(max_connections=256))


async def forward(request: Request) -> Response:
    path = request.url.path
    if not (path.startswith("/v1/") or path in ALLOWED_EXACT):
        return Response("not found", status_code=404)
    if path.startswith("/v1/"):
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth.encode(), f"Bearer {API_KEY}".encode()):
            return Response('{"error":"Unauthorized"}', status_code=401, media_type="application/json")
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    body = await request.body()
    if path == "/v1/chat/completions" and body:
        try:
            j = json.loads(body)
            for m in j.get("messages") or []:
                if isinstance(m, dict) and "reasoning_content" in m:
                    m.setdefault("reasoning", m.pop("reasoning_content"))
            body = json.dumps(j).encode()
        except ValueError:
            pass
    req = client.build_request(request.method, path, params=request.query_params, headers=headers, content=body)
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        return Response(f"upstream error: {e!r}", status_code=502)
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}

    async def body_iter():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(body_iter(), status_code=resp.status_code, headers=out_headers, background=BackgroundTask(resp.aclose))


app = Starlette(routes=[Route("/{path:path}", forward, methods=["GET", "POST", "HEAD", "OPTIONS"])])

if __name__ == "__main__":
    # Two workers: a single Python process is the bottleneck at 32 concurrent token streams.
    uvicorn.run("qwen_proxy:app", host="0.0.0.0", port=PORT, workers=int(os.environ.get("WAYFINDER_PROXY_WORKERS", "2")),
                log_level="warning", timeout_keep_alive=75, http="httptools", loop="uvloop")
