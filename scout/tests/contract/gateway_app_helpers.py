"""Shared rig for the gateway HTTP tests (tasks 8.1-8.3): the real `GatewayService` over fakes behind the real
FastAPI app, with keys issued through the real `TenantDirectory` (tenants come only from key records)."""

from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI
from svc_helpers import MASTER, RecordingStore

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.core.service.auth import AuthService
from scout.core.service.gateway import GatewayService
from scout.core.service.ratelimit import RateLimiter
from scout.core.service.tenants import TenantDirectory
from scout.surfaces.gateway_http.app import IdempotencyCache, create_gateway_app

PEPPER = b"gateway-http-test-pepper-0123456789"


@dataclass
class Rig:
    store: RecordingStore
    clock: FakeClock
    embedder: FakeEmbedder
    chat: ScriptedChat
    service: GatewayService
    directory: TenantDirectory
    app: FastAPI
    keys: dict[str, str] = field(default_factory=dict[str, str])

    async def key(self, tenant: str) -> str:
        if tenant not in self.keys:
            self.keys[tenant] = await self.directory.issue_key(tenant)
        return self.keys[tenant]

    def transport(self) -> httpx.ASGITransport:
        return httpx.ASGITransport(app=self.app)


def make_rig(
    *,
    chat: ScriptedChat | None = None,
    keepalive_s: float = 10.0,
    cache: IdempotencyCache | None = None,
    limiter: RateLimiter | None = None,
    service_cls: type[GatewayService] = GatewayService,
    store: RecordingStore | None = None,
    clock: FakeClock | None = None,
    embedder: FakeEmbedder | None = None,
) -> Rig:
    store = store or RecordingStore()
    clock = clock or FakeClock()
    embedder = embedder or FakeEmbedder()
    chat = chat or ScriptedChat([])
    service = service_cls(store, clock, MASTER, embedder=embedder, chat=chat, limiter=limiter)
    directory = TenantDirectory(store, clock, PEPPER)
    app = create_gateway_app(service, AuthService(directory, clock), cache=cache, keepalive_s=keepalive_s)
    return Rig(store, clock, embedder, chat, service, directory, app)


def bearer(key: str, **extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "X-Scout-Schema": "1", **extra}


def body_of(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")
