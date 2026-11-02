"""Tiny allow-list reverse proxy in front of vLLM (127.0.0.1:UPSTREAM_PORT -> 0.0.0.0:PORT).

It also enforces the bearer key (env A1_API_KEY) on /v1/*, because in snapshot mode vLLM itself runs without
VLLM_API_KEY (a key must not be baked into a memory snapshot).

Why: the snapshot path runs vLLM with VLLM_SERVER_DEV_MODE=1 (needed for /sleep and /wake_up), which also
exposes unauthenticated dev endpoints (collective_rpc, server_info, ...). vLLM's API-key middleware only
guards /v1. This proxy forwards only /v1/*, /health, /version and /ping, so dev endpoints are reachable
from inside the container (localhost:UPSTREAM_PORT) but never from the public Modal URL.
"""

from __future__ import annotations

import hmac
import os

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = f"http://127.0.0.1:{os.environ.get('A1_UPSTREAM_PORT', '8001')}"
API_KEY = os.environ["A1_API_KEY"]
PORT = int(os.environ.get("A1_PORT", "8000"))
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
    uvicorn.run("a1_proxy:app", host="0.0.0.0", port=PORT, workers=int(os.environ.get("A1_PROXY_WORKERS", "2")),
                log_level="warning", timeout_keep_alive=75, http="httptools", loop="uvloop")
