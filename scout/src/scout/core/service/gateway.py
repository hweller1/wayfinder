"""`GatewayService` (design sections 3, 5 and 7): the server-side `Gateway`, one instance per process.

`for_tenant` binds a tenant that came from authentication; clients only ever send hex `repo_id`/`branch_id`,
and every namespace name is built from those plus the tenant (`crypto.ids`), never taken from a request.
The keys namespace is a constructor parameter (core reads no environment variables).

Ordering invariant (5.3, 15): a branch copies `{ns}__tree` BEFORE `{ns}`; delete removes `{ns}` BEFORE
`{ns}__tree`.
The tree can therefore be older than the chunks (the next diff re-syncs the difference), and a crash between
the two steps leaves only an orphan tree, which `sync_begin` (keyed on `{ns}`) deletes and recopies.
"""

import base64
import hashlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict
from typing import Any, NoReturn

from pydantic import ValidationError

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import IndexStatus
from scout.core.contracts.wire import (
    ChatReq,
    CommitReq,
    CommitRes,
    EmbedReq,
    EmbedRes,
    NodesReq,
    NodesRes,
    PlanReq,
    PlanRes,
    SearchReq,
    SearchRes,
    SeedReq,
    SeedRes,
    StageNodesReq,
    StageNodesRes,
    StatusReq,
    SyncBeginReq,
    SyncBeginRes,
    TenantInfo,
)
from scout.core.crypto import ids, keys
from scout.core.ports.chat import ChatDelta, ChatModel, ChatRequest, ChatTurn
from scout.core.ports.clock import Clock
from scout.core.ports.embedder import Embedder
from scout.core.ports.store import Row, VectorStore
from scout.core.service.logging import request_log
from scout.core.service.ratelimit import RateLimiter
from scout.core.service.schema import CHUNK_SCHEMA, ROOT_ID, TREE_SCHEMA
from scout.core.service.searchsvc import SearchService
from scout.core.service.seedsvc import DEFAULT_THRESHOLD, SeedService, branch_pair, repair_orphan_tree
from scout.core.service.syncsvc import MemoryReplays, ReplayStore, SyncService
from scout.core.service.tenants import KEYS_NAMESPACE, KEYS_SCHEMA, Tenant

READY_COVERAGE = 0.8  # semantic search turns on at or above this coverage (PRD 4.1)
_TEAM_ROW_PREFIX = "team-"  # not hex, so it can never equal a key record id (`tag[:32]`)


def _not_yet(what: str) -> NoReturn:
    raise NotImplementedError(f"{what} is implemented by a later task")


def namespaces(tenant: str, repo_id: str, branch_id: str) -> tuple[str, str]:
    """`(ns, ns__tree)` for an authenticated tenant and client-supplied ids; anything else is rejected."""
    try:
        ns = ids.namespace(tenant, repo_id, branch_id)
        return ns, ids.tree_namespace(ns)
    except ValueError as exc:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, str(exc)) from None


class GatewayService:
    def __init__(
        self,
        store: VectorStore,
        clock: Clock,
        master_key: bytes,
        *,
        keys_namespace: str = KEYS_NAMESPACE,
        embedder: Embedder | None = None,
        chat: ChatModel | None = None,
        limiter: RateLimiter | None = None,
        replays: ReplayStore | None = None,
        seed_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if len(master_key) != keys.MASTER_KEY_LEN:
            raise ValueError(f"master key must be {keys.MASTER_KEY_LEN} bytes")
        self.store = store
        self.clock = clock
        self._master = master_key
        self.keys_namespace = keys_namespace
        limiter = limiter or RateLimiter(clock)
        self._limiter = limiter
        self._chat = chat
        self._sync = SyncService(store, embedder, limiter, replays or MemoryReplays(clock))
        self._search = SearchService(store, embedder, limiter)
        self._seed = SeedService(store, limiter, threshold=seed_threshold)

    def for_tenant(self, tenant: Tenant | str) -> "TenantGateway":
        bound = tenant if isinstance(tenant, Tenant) else Tenant(tenant=tenant, key_id="")
        namespaces(bound.tenant, "0" * ids.REPO_ID_LEN, "0" * ids.BRANCH_ID_LEN)  # the one tenant-name rule
        return TenantGateway(self, bound)

    async def ping(self, tenant: str) -> None:
        """Liveness for an authenticated tenant: the tenant name passes the one name rule, no store access."""
        namespaces(tenant, "0" * ids.REPO_ID_LEN, "0" * ids.BRANCH_ID_LEN)

    async def team_key(self, tenant: str) -> bytes:
        """The tenant's 64-byte K_team, unwrapped with AAD=tenant. The first caller creates it: the wrapped
        key is written insert-if-absent, so concurrent first calls converge on whichever write landed."""
        store, ns = self.store, self.keys_namespace
        row_id = _TEAM_ROW_PREFIX + hashlib.sha256(tenant.encode()).hexdigest()[:32]
        await store.ensure_namespace(ns, KEYS_SCHEMA)
        found = await store.get_by_ids(ns, [row_id])
        if not found:
            wrapped = keys.wrap(self._master, keys.new_team_key(), tenant.encode())
            record = {
                "key_tag": "",  # never matches a presented key
                "tenant": tenant,
                "wrapped_k_team": base64.b64encode(wrapped).decode(),
                "created_at": int(self.clock.now()),
                "revoked": True,
            }
            await store.upsert(ns, [Row(id=row_id, attrs=record)], if_absent=True)
            found = await store.get_by_ids(ns, [row_id])
        if not found:
            raise ScoutError(ErrorCode.INTERNAL, "team key record missing after write")
        attrs = found[0].attrs
        blob = attrs.get("wrapped_k_team")
        if attrs.get("tenant") != tenant or not isinstance(blob, str) or not blob:
            raise ScoutError(ErrorCode.INTERNAL, "team key record is not usable")
        try:
            return keys.unwrap(self._master, base64.b64decode(blob), tenant.encode())
        except (keys.CryptoError, ValueError):
            raise ScoutError(ErrorCode.INTERNAL, "team key record is not usable") from None

    async def _root(self, tree: str) -> Row | None:
        if not await self.store.namespace_exists(tree):
            return None
        rows = await self.store.get_by_ids(tree, [ROOT_ID])
        return rows[0] if rows else None

    async def sync_begin(self, tenant: str, req: SyncBeginReq) -> SyncBeginRes:
        store = self.store
        ns, tree = namespaces(tenant, req.repo_id, req.branch_id)
        base = None
        if req.base_branch_id is not None and req.base_branch_id != req.branch_id:
            base = namespaces(tenant, req.repo_id, req.base_branch_id)
        created = False
        if not await store.namespace_exists(ns):
            await repair_orphan_tree(store, ns, tree)  # a crash hit between the two copies
            if (
                base is not None
                and await store.namespace_exists(base[0])
                and await store.namespace_exists(base[1])
            ):
                await branch_pair(store, ns, tree, base[0], base[1])  # tree first, then chunks
                created = True
        if not created:
            await store.ensure_namespace(ns, CHUNK_SCHEMA)
            await store.ensure_namespace(tree, TREE_SCHEMA)
        root = await self._root(tree)
        if root is None:
            return SyncBeginRes(ns_state="empty", server_root=None, coverage=0.0)
        root_hash = root.attrs.get("root_hash")
        return SyncBeginRes(
            ns_state="populated",
            server_root=root_hash if isinstance(root_hash, str) else None,
            coverage=_coverage(root),
        )

    async def index_status(self, tenant: str, req: StatusReq) -> IndexStatus:
        ns, tree = namespaces(tenant, req.repo_id, req.branch_id)
        root = await self._root(tree) if await self.store.namespace_exists(ns) else None
        return _status(root.attrs if root is not None else None)

    async def sync_commit(self, tenant: str, req: CommitReq) -> CommitRes:
        ns, tree = namespaces(tenant, req.repo_id, req.branch_id)
        root = await self._sync.commit(tenant, ns, tree, req)
        return CommitRes.model_validate(_status(root).model_dump())

    async def sync_stage_nodes(self, tenant: str, req: StageNodesReq) -> StageNodesRes:
        ns, tree = namespaces(tenant, req.repo_id, req.branch_id)
        return await self._sync.stage_nodes(ns, tree, req)

    async def sync_nodes(self, tenant: str, req: NodesReq) -> NodesRes:
        _, tree = namespaces(tenant, req.repo_id, req.branch_id)
        return await self._sync.nodes(tree, req.ids)

    async def sync_plan(self, tenant: str, req: PlanReq) -> PlanRes:
        namespaces(tenant, req.repo_id, req.branch_id)  # ids are validated like every other request
        return await self._sync.plan(tenant, req.chunk_keys)

    async def embed(self, tenant: Tenant, req: EmbedReq, idempotency_key: str | None = None) -> EmbedRes:
        ns, _ = namespaces(tenant.tenant, req.repo_id, req.branch_id)
        return await self._sync.embed(
            tenant=tenant.tenant,
            key_id=tenant.key_id or tenant.tenant,  # rate limits are per key
            ns=ns,
            req=req,
            idempotency_key=idempotency_key,
        )

    async def search(self, tenant: Tenant, req: SearchReq) -> SearchRes:
        ns, _ = namespaces(tenant.tenant, req.repo_id, req.branch_id)
        return await self._search.search(
            tenant=tenant.tenant, key_id=tenant.key_id or tenant.tenant, ns=ns, req=req
        )

    async def seed(self, tenant: Tenant, req: SeedReq) -> SeedRes:
        ns, tree = namespaces(tenant.tenant, req.repo_id, req.branch_id)
        return await self._seed.seed(
            tenant=tenant.tenant, key_id=tenant.key_id or tenant.tenant, ns=ns, tree=tree, req=req
        )

    async def warm(self, tenant: Tenant) -> bool:
        """Start the chat model's cold start without waiting (the upstream's `/warm` endpoint, if the
        injected model has one). Never raises for an upstream problem: False means not started."""
        warm = getattr(self._chat, "warm", None)
        with request_log("warm", key_id=tenant.key_id or tenant.tenant, clock=self.clock):
            if warm is None:
                return False
            try:
                return bool(await warm())
            except Exception:  # noqa: BLE001  a warm-up is best effort
                return False

    async def complete(self, tenant: Tenant, req: ChatRequest) -> ChatTurn:
        """Pass one chat turn to the injected `ChatModel` (design 3, 6). The request is checked against the
        wire caps first (oversize never reaches the model), then at most 2 run per key at once. Upstream
        failures leave as typed errors with fixed messages: a prompt or tool result is never echoed back,
        logged, or kept in an exception chain. Only ids, sizes, status, latency and token counts are
        logged."""
        key_id = tenant.key_id or tenant.tenant
        with request_log("chat", key_id=key_id, clock=self.clock) as usage:
            _validate_chat(req)
            usage.messages, usage.tools = len(req.messages), len(req.tools)
            usage.chars = sum(len(str(m.get("content") or "")) for m in req.messages)
            if self._chat is None:
                raise ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "no chat model is configured")
            with self._limiter.chat_slot(key_id):
                try:
                    turn = await self._chat.complete(req)
                except Exception as exc:
                    failure = _chat_failure(exc)
                else:
                    usage.tokens_in, usage.tokens_out = turn.tokens_in, turn.tokens_out
                    return turn
            raise failure  # outside the `except`: no `__context__` carrying the upstream message

    async def delete_branch(self, tenant: str, repo_id: str, branch_id: str) -> None:
        """Chunks first, then the tree: a failure between them leaves an orphan tree, never orphan chunks.
        Both deletes are idempotent, so a retry finishes the job (and a missing branch is a no-op)."""
        ns, tree = namespaces(tenant, repo_id, branch_id)
        await self.store.delete_namespace(ns)
        await self.store.delete_namespace(tree)


def _validate_chat(req: ChatRequest) -> None:
    """Wire caps (`ChatReq`: 1..1000 messages, <= 64 tools, sampling ranges). Pydantic's own message quotes
    the offending input, so the error carries only field paths and error types."""
    try:
        ChatReq.model_validate(asdict(req))
    except ValidationError as exc:
        found = exc.errors(include_input=False, include_url=False, include_context=False)
        where = ", ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['type']}" for e in found[:5])
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"chat request out of bounds ({where})") from None


def _chat_failure(exc: Exception) -> ScoutError:
    """Map any chat-model failure to a typed error whose message is fixed text. A context-length rejection
    keeps `details.reason == "context_length"`, which the explore loop reads to compact its history."""
    if isinstance(exc, ScoutError):
        code, details = exc.code, exc.details
        if code is ErrorCode.INVALID_ARGUMENT:
            if details.get("reason") == "context_length" or "maximum context length" in exc.message.lower():
                return ScoutError(
                    code,
                    "the prompt exceeds the model's context length",
                    details={"reason": "context_length"},
                )
            return ScoutError(code, "the chat request was rejected by the model")
        if code is ErrorCode.RATE_LIMITED:
            kept = {
                k: v
                for k, v in details.items()
                if k in ("retry_after", "limit") and isinstance(v, int | float)
            }
            return ScoutError(code, "the chat model is rate limited", details=kept)
        if code is ErrorCode.UPSTREAM_TIMEOUT:
            return ScoutError(code, "the chat model timed out")
        if code is ErrorCode.UPSTREAM_UNAVAILABLE:
            return ScoutError(code, "the chat model is unavailable")
        return ScoutError(ErrorCode.INTERNAL, "the chat model failed")
    if isinstance(exc, TimeoutError):
        return ScoutError(ErrorCode.UPSTREAM_TIMEOUT, "the chat model timed out")
    return ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the chat model is unavailable")


def _coverage(root: Row) -> float:
    return _coverage_of(root.attrs)


def _coverage_of(attrs: Mapping[str, Any]) -> float:
    expected, written = attrs.get("expected_chunks"), attrs.get("written_chunks")
    if not isinstance(expected, int) or not isinstance(written, int) or expected <= 0 or written <= 0:
        return 0.0
    return min(written / expected, 1.0)


def _status(root: Mapping[str, Any] | None) -> IndexStatus:
    """`absent` without a root row, `partial` below 80% coverage, `ready` at or above it."""
    if root is None:
        return IndexStatus(state="absent", coverage=0.0, semantic_enabled=False)
    coverage = _coverage_of(root)
    commit = root.get("commit")
    return IndexStatus(
        state="ready" if coverage >= READY_COVERAGE else "partial",
        coverage=coverage,
        commit=commit if isinstance(commit, str) and commit else None,
        semantic_enabled=coverage >= READY_COVERAGE,
    )


class TenantGateway:
    """The `Gateway` port bound to one authenticated tenant."""

    def __init__(self, service: GatewayService, tenant: Tenant) -> None:
        self._svc = service
        self._tenant = tenant

    async def whoami(self) -> TenantInfo:
        return TenantInfo(tenant=self._tenant.tenant, key_id=self._tenant.key_id)

    async def ping(self) -> None:
        await self._svc.ping(self._tenant.tenant)

    async def team_key(self) -> bytes:
        return await self._svc.team_key(self._tenant.tenant)

    async def sync_begin(self, req: SyncBeginReq) -> SyncBeginRes:
        return await self._svc.sync_begin(self._tenant.tenant, req)

    async def index_status(self, req: StatusReq) -> IndexStatus:
        return await self._svc.index_status(self._tenant.tenant, req)

    async def delete_branch(self, repo_id: str, branch_id: str) -> None:
        await self._svc.delete_branch(self._tenant.tenant, repo_id, branch_id)

    async def sync_nodes(self, req: NodesReq) -> NodesRes:
        return await self._svc.sync_nodes(self._tenant.tenant, req)

    async def sync_plan(self, req: PlanReq) -> PlanRes:
        return await self._svc.sync_plan(self._tenant.tenant, req)

    async def embed(self, req: EmbedReq, *, idempotency_key: str | None = None) -> EmbedRes:
        return await self._svc.embed(self._tenant, req, idempotency_key)

    async def sync_stage_nodes(self, req: StageNodesReq) -> StageNodesRes:
        return await self._svc.sync_stage_nodes(self._tenant.tenant, req)

    async def sync_commit(self, req: CommitReq) -> CommitRes:
        return await self._svc.sync_commit(self._tenant.tenant, req)

    async def search(self, req: SearchReq) -> SearchRes:
        return await self._svc.search(self._tenant, req)

    async def seed(self, req: SeedReq) -> SeedRes:
        return await self._svc.seed(self._tenant, req)

    async def complete(self, req: ChatRequest) -> ChatTurn:
        return await self._svc.complete(self._tenant, req)

    async def warm(self) -> bool:
        return await self._svc.warm(self._tenant)

    def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        _not_yet("stream")
