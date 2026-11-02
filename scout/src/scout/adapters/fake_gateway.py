"""`FakeGateway` (design section 3, Testability): the real `GatewayService` over in-memory fakes.

`make_fake_gateway` returns `GatewayService(FakeStore, FakeEmbedder, ScriptedChat, ...).for_tenant(tenant)`,
so every `Gateway` port method runs the production service code. The index stays `absent` (no namespace, no
root row) until a sync runs against it; chat defaults to an empty `ScriptedChat`.
"""

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_store import FakeStore
from scout.core.ports.chat import ChatModel
from scout.core.ports.clock import Clock
from scout.core.ports.embedder import Embedder
from scout.core.ports.gateway import Gateway
from scout.core.ports.store import VectorStore
from scout.core.service.gateway import GatewayService

FAKE_MASTER_KEY = b"fake-gateway-master-key-32-bytes"  # 32 bytes; test-only, protects nothing


def make_fake_gateway(
    tenant: str = "dev",
    chat: ChatModel | None = None,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    clock: Clock | None = None,
) -> Gateway:
    service = GatewayService(
        store if store is not None else FakeStore(),
        clock if clock is not None else FakeClock(),
        FAKE_MASTER_KEY,
        embedder=embedder if embedder is not None else FakeEmbedder(),
        chat=chat if chat is not None else ScriptedChat([]),
    )
    return service.for_tenant(tenant)
