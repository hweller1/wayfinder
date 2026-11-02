"""Scout gateway as a plain FastAPI/uvicorn service (the VM counterpart of scout/deploy/modal/gateway.py).

    python deploy/gcp/gateway_vm.py serve                 # uvicorn on SCOUT_BIND_HOST:SCOUT_BIND_PORT (127.0.0.1:8080)
    python deploy/gcp/gateway_vm.py issue-key --tenant dev   # prints a scout key once; stores only its HMAC tag
    python deploy/gcp/gateway_vm.py check-config          # validates the environment, makes no network call

The composition is the same as the Modal app's: `scout.wiring.build_gateway_service` (turbopuffer store, Voyage
embedder, OpenAI-compatible chat upstream) and `scout.wiring.build_auth`, wrapped by
`scout.surfaces.gateway_http.app.create_gateway_app`. Two differences, both because there is no Modal Dict here:
key records are cached in process memory for a short time (`MemoryKeyCache`; a revoked key stops working within
`SCOUT_KEY_CACHE_TTL_S`, default 60 s, and the keys namespace in turbopuffer stays authoritative), and
idempotency replays use the app's default in-memory cache.

Environment (from /etc/scout/gateway.env under systemd, mode 600, never committed):
    TURBOPUFFER_API_KEY, VOYAGE_API_KEY            vendor keys
    SCOUT_MASTER_KEY, SCOUT_PEPPER                 base64 secrets, 32 bytes each (reuse the Modal values to keep the
                                                   existing index and keys readable; new values start an empty index)
    SCOUT_LLM_URL                                  http://127.0.0.1:8000/v1 (vLLM on the same VM)
    SCOUT_LLM_API_KEY                              the bearer key vLLM was started with (VLLM_API_KEY)
    SCOUT_LLM_MODEL=wayfinder-glimmer, SCOUT_LLM_SAMPLING=sft
    SCOUT_KEYS_NS (optional), TURBOPUFFER_REGION (optional)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

REQUIRED = (
    "TURBOPUFFER_API_KEY",
    "VOYAGE_API_KEY",
    "SCOUT_MASTER_KEY",
    "SCOUT_PEPPER",
    "SCOUT_LLM_URL",
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_KEY_CACHE_TTL_S = 60.0


class MemoryKeyCache:
    """`scout.core.service.tenants.Cache` over a dict with a TTL. A miss or an expired entry falls through to the
    authoritative keys namespace, so the cache can only delay a revocation, never grant access."""

    def __init__(self, ttl_s: float = DEFAULT_KEY_CACHE_TTL_S, now: Any = time.monotonic) -> None:
        self._ttl = ttl_s
        self._now = now
        self._d: dict[str, tuple[float, dict[str, Any]]] = {}

    async def get(self, key_id: str) -> Mapping[str, Any] | None:
        item = self._d.get(key_id)
        if item is None:
            return None
        if item[0] <= self._now():
            self._d.pop(key_id, None)
            return None
        return item[1]

    async def set(self, key_id: str, record: Mapping[str, Any]) -> None:
        self._d[key_id] = (self._now() + self._ttl, dict(record))

    async def delete(self, key_id: str) -> None:
        self._d.pop(key_id, None)


def missing_env(env: Mapping[str, str]) -> list[str]:
    """Names (never values) of the required variables that are unset or empty."""
    return [n for n in REQUIRED if not env.get(n)]


def _version() -> str:
    from importlib import metadata

    try:
        return metadata.version("scout")
    except metadata.PackageNotFoundError:
        return "0"


def build_app(env: Mapping[str, str] | None = None) -> Any:
    """The gateway ASGI app. Fails closed, naming the missing variables, if the configuration is incomplete."""
    from scout.surfaces.gateway_http.app import create_gateway_app
    from scout.wiring import build_auth, build_gateway_service

    env = dict(os.environ if env is None else env)
    ttl = float(env.get("SCOUT_KEY_CACHE_TTL_S") or DEFAULT_KEY_CACHE_TTL_S)
    service = build_gateway_service(env)
    _, auth = build_auth(service, env, cache=MemoryKeyCache(ttl))
    return create_gateway_app(service, auth, version=_version())


def bind_address(env: Mapping[str, str]) -> tuple[str, int]:
    host = env.get("SCOUT_BIND_HOST") or DEFAULT_HOST
    port = int(env.get("SCOUT_BIND_PORT") or DEFAULT_PORT)
    return host, port


def serve(env: Mapping[str, str] | None = None) -> None:
    import uvicorn

    env = dict(os.environ if env is None else env)
    host, port = bind_address(env)
    # One process: the key cache and idempotency replays are in memory. The gateway is I/O bound (it waits on
    # vLLM, Voyage and turbopuffer), so a single event loop is enough for a demo. A long explore request is one
    # connection held open; the app sends keepalives on /v1/chat.
    uvicorn.run(build_app(env), host=host, port=port, log_level="info", timeout_keep_alive=75, access_log=False)


def issue_key(tenant: str, env: Mapping[str, str] | None = None) -> str:
    from scout.wiring import build_auth, build_gateway_service

    env = dict(os.environ if env is None else env)
    service = build_gateway_service(env)
    directory, _ = build_auth(service, env)
    return asyncio.run(directory.issue_key(tenant))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    sub.add_parser("check-config")
    ik = sub.add_parser("issue-key")
    ik.add_argument("--tenant", required=True, help="[A-Za-z0-9-], at most 24 chars")
    args = ap.parse_args(argv)
    env = os.environ
    if args.cmd == "check-config":
        miss = missing_env(env)
        if miss:
            print("missing: " + ", ".join(miss), file=sys.stderr)
            return 2
        build_app(env)  # builds adapters, makes no network call
        host, port = bind_address(env)
        print(f"ok: gateway configured, would listen on {host}:{port}")
        return 0
    if args.cmd == "issue-key":
        key = issue_key(args.tenant, env)
        sys.stdout.write(f"tenant={args.tenant}\nkey={key}\n")
        return 0
    serve(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
