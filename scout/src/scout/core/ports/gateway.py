"""Gateway port (design section 3): what the CLIENT calls. GatewayService implements it server-side."""

from typing import Protocol

from scout.core.contracts.types import IndexStatus
from scout.core.contracts.wire import (
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
from scout.core.ports.chat import ChatModel


class Gateway(ChatModel, Protocol):
    async def whoami(self) -> TenantInfo:
        """UNAUTHENTICATED on bad key."""
        ...

    async def ping(self) -> None:
        """Cheap liveness + auth check (`GET /v1/healthz` and `whoami` on the HTTP adapter). Returns when the
        gateway answered for this tenant, raises `ScoutError` otherwise. The watcher calls it every 60 s."""
        ...

    async def team_key(self) -> bytes:
        """Create-if-absent (conditional upsert), returns the 64-byte K_team for the caller's tenant."""
        ...

    async def sync_begin(self, req: SyncBeginReq) -> SyncBeginRes:
        """Creates ns (branch_from base_branch_id if ns absent and base exists), returns server root
        hash + ns_state in {absent, empty, populated}. Idempotent."""
        ...

    async def sync_nodes(self, req: NodesReq) -> NodesRes:
        """Opaque encrypted node blobs by id."""
        ...

    async def sync_plan(self, req: PlanReq) -> PlanRes:
        """Which chunk_keys miss the embed cache."""
        ...

    async def embed(self, req: EmbedReq, *, idempotency_key: str | None = None) -> EmbedRes:
        """Embeds texts for missing keys, writes cache insert-if-absent, upserts rows (vector from
        cache), then delete_by_file(except_file_hash) for each changed file. Idempotent by row ids.
        Missing cache entry for a row without text -> CONFLICT (client replans). `idempotency_key`
        (the HTTP `Idempotency-Key` header) replays a stored result for the same request body."""
        ...

    async def sync_stage_nodes(self, req: StageNodesReq) -> StageNodesRes:
        """Writes one page (<= 500 rows) of the new tree's node rows to the tree namespace, plus a marker.
        Idempotent per page; any order; CONFLICT if the server root is neither base_root nor new_root."""
        ...

    async def sync_commit(self, req: CommitReq) -> CommitRes:
        """Compare-and-set on the root pointer (expected_root -> new_root); same new_root replay is OK.
        Mismatch -> CONFLICT. Refuses (INVALID_ARGUMENT) if a page of `staged_pages` is not staged.
        Writes coverage; drops proofs when the root is committed."""
        ...

    async def search(self, req: SearchReq) -> SearchRes:
        """Embeds query, hybrid, proof filter."""
        ...

    async def seed(self, req: SeedReq) -> SeedRes:
        """MinHash scan + branch."""
        ...

    async def index_status(self, req: StatusReq) -> IndexStatus: ...

    async def delete_branch(self, repo_id: str, branch_id: str) -> None: ...
