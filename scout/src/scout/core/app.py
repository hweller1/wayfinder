"""`ScoutApp`: the one facade every surface (SDK, CLI, MCP, daemon) calls (design section 1, `core.app`).

Methods take and return contract types only; no host or transport type crosses this boundary. It is built
over injected ports (Gateway, FileSystem, Clock, optionally KeyStore) by `wiring`, so the same object runs
against fakes or real adapters.
- Delegated level: `explore` (semantic search is offered to the explorer iff `index.coverage >= 0.8`).
- Primitive level: `grep`, `read_range`, `index_status`, `semantic_search`.
- Lifecycle: `sync`, `watch`, `login`.

Identity (repo id, branch id) is derived per call from the checkout (`FileSystem.git_ids`, the clone id in the
state dir) and the team key, exactly as `SyncEngine` derives it, so a request always addresses the namespace
the last sync wrote. `semantic_search` on an index below 80% answers with no hits and a `note` that says so
(grep-only); a hit whose file changed since it was indexed is dropped and counted in the note, never returned
with stale lines. Every error leaving a method is a `ScoutError`: anything else is wrapped as INTERNAL with
the exception type only (its text may carry a path).
"""

import asyncio
import functools
import threading
from collections.abc import Awaitable, Callable

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import (
    ExploreRequest,
    ExploreResult,
    GrepRequest,
    GrepResult,
    IndexStatus,
    IndexStatusRequest,
    ReadRangeRequest,
    ReadRangeResult,
    SemanticSearchRequest,
    SemanticSearchResult,
)
from scout.core.contracts.wire import StatusReq
from scout.core.crypto import ids
from scout.core.crypto.keys import Subkeys, derive
from scout.core.explore import spec as retrieval_spec
from scout.core.explore.candidates import verify_hits
from scout.core.explore.loop import DEFAULT_BACKFILL_TO, SEMANTIC_MIN_COVERAGE, ExploreContext, explore
from scout.core.explore.profiles import PROFILES, ExploreProfile
from scout.core.explore.semantic import make_backend, spec_search
from scout.core.local.grep import grep
from scout.core.local.ignore import Matcher
from scout.core.local.readfile import read_range
from scout.core.ports.chat import ChatRequest
from scout.core.ports.clock import Clock
from scout.core.ports.fs import FileSystem
from scout.core.ports.gateway import Gateway
from scout.core.ports.keystore import KeyStore
from scout.core.sync import merkle
from scout.core.sync.engine import SyncConfig, SyncEngine
from scout.core.sync.state import StateDir
from scout.core.sync.watch import WatchConfig, WatchResult, ensure_watch

KEY_NAME = "scout_key"


def _typed[**P, T](fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Facade methods raise `ScoutError` only: anything else becomes INTERNAL (type name, no message)."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await fn(*args, **kwargs)
        except ScoutError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ScoutError(ErrorCode.INTERNAL, f"internal error: {type(e).__name__}") from None

    return wrapper


class ScoutApp:
    """One app serves one repo root (`fs.root`); a request `root` must be absent or equal to it."""

    def __init__(
        self,
        gateway: Gateway,
        fs: FileSystem,
        clock: Clock,
        keystore: KeyStore | None = None,
        profile: ExploreProfile | None = None,
        sync_config: SyncConfig | None = None,
        backfill_to: int = DEFAULT_BACKFILL_TO,
    ) -> None:
        self._gateway = gateway
        self._fs = fs
        self._clock = clock
        self._keystore = keystore
        self._profile = profile or PROFILES["product"]
        self._sync_config = sync_config
        self._backfill_to = backfill_to
        self._tenant: str | None = None
        self._keys: Subkeys | None = None
        self._engine: SyncEngine | None = None

    def _check_root(self, root: str | None) -> None:
        if root is not None and root != self._fs.root:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "root does not match the repository this app serves")

    async def _identity(self) -> tuple[str, Subkeys]:
        if self._tenant is None or self._keys is None:
            tenant = (await self._gateway.whoami()).tenant
            self._keys = derive(await self._gateway.team_key())
            self._tenant = tenant
        return self._tenant, self._keys

    async def _status_req(self) -> StatusReq:
        """The namespace ids of the current checkout and branch (`NOT_A_REPO` outside a git checkout)."""
        _, keys = await self._identity()
        git = await asyncio.to_thread(self._fs.git_ids)
        state = await asyncio.to_thread(StateDir.open, self._fs)
        clone_id = await asyncio.to_thread(state.clone_id)
        repo_id = ids.repo_id(keys, git.root_commit or "", git.remote_norm or "")
        return StatusReq(repo_id=repo_id, branch_id=ids.branch_id(keys, repo_id, clone_id, git.branch))

    # --- delegated level --------------------------------------------------------------------
    @_typed
    async def explore(self, req: ExploreRequest) -> ExploreResult:
        self._check_root(req.root)
        status_req = await self._status_req()
        tenant, keys = await self._identity()
        spec = await self._spec()
        backend = make_backend(
            self._gateway,
            keys,
            tenant,
            status_req,
            lambda data: merkle.leaf_hash(keys, data),
            spec,
            self._mtime_ns,
        )
        ctx = ExploreContext(
            self._gateway, self._fs, self._clock, status_req, self._profile, semantic=backend,
            backfill_to=self._backfill_to, spec=spec,
        )  # fmt: skip
        return await explore(req, ctx)

    async def _spec(self) -> "retrieval_spec.RetrievalSpec | None":
        """The resolved retrieval spec of this checkout, read fresh (None: no spec, behave as before)."""
        try:
            state_dir = await asyncio.to_thread(self._fs.state_dir)
        except ScoutError:
            return None
        return await asyncio.to_thread(retrieval_spec.load, state_dir)

    def _mtime_ns(self, path: str) -> int | None:
        try:
            return self._fs.stat(path).mtime_ns
        except ScoutError:
            return None

    # --- primitives -------------------------------------------------------------------------
    @_typed
    async def grep(self, req: GrepRequest) -> GrepResult:
        self._check_root(req.root)
        return grep(self._fs, req, clock=self._clock)

    @_typed
    async def read_range(self, req: ReadRangeRequest) -> ReadRangeResult:
        self._check_root(req.root)
        return read_range(self._fs, req)

    @_typed
    async def warm(self, *, trigger: bool = False) -> dict[str, object]:
        """Wake the explore model (a scale-to-zero model on Modal cold-starts in minutes) so the first
        `explore` does not pay for it. Default: one 1-token completion, returning once it answered
        (`{"mode": "completion", "warm": True}`). `trigger=True`: ask the gateway's non-blocking `warm`
        (about a second; the model is still booting, `{"mode": "endpoint", "warm": False, "warming": True}`),
        falling back to the completion when the gateway or upstream has no warm endpoint. Needs no index."""
        t0 = self._clock.monotonic()
        endpoint = getattr(self._gateway, "warm", None) if trigger else None
        if endpoint is not None and await endpoint():
            return {
                "mode": "endpoint",
                "warm": False,
                "warming": True,
                "seconds": self._clock.monotonic() - t0,
            }
        await self._gateway.complete(
            ChatRequest(messages=[{"role": "user", "content": "ping"}], max_tokens=1, thinking=False)
        )
        return {"mode": "completion", "warm": True, "seconds": self._clock.monotonic() - t0}

    @_typed
    async def index_status(self, req: IndexStatusRequest | None = None) -> IndexStatus:
        self._check_root(req.root if req else None)
        return await self._gateway.index_status(await self._status_req())

    @_typed
    async def semantic_search(self, req: SemanticSearchRequest) -> SemanticSearchResult:
        """Ranked snippets of the CURRENT file contents. Index below 80% (partial or absent): no hits and a
        `note` saying to use grep. Hits whose file changed since indexing are dropped, counted in `note`."""
        self._check_root(req.root)
        status_req = await self._status_req()
        status = await self._gateway.index_status(status_req)
        spec = await self._spec()
        if spec is not None and not spec.index_enabled:
            return SemanticSearchResult(
                hits=[],
                index=status,
                note="semantic search is off in the retrieval spec (index.enabled = false), use grep",
            )
        if status.coverage < SEMANTIC_MIN_COVERAGE:
            return SemanticSearchResult(
                hits=[],
                index=status,
                note=f"index {status.state}: semantic search unavailable, use grep",
            )
        tenant, keys = await self._identity()
        raw = await spec_search(
            self._gateway, keys, tenant, status_req, req.query, req.path_glob, req.k, spec, self._mtime_ns
        )
        hits, dropped = verify_hits(
            self._fs, Matcher.from_fs(self._fs), lambda data: merkle.leaf_hash(keys, data), raw, req.k
        )
        note = f"{dropped} hit(s) dropped: the file changed since it was indexed" if dropped else None
        return SemanticSearchResult(hits=hits, index=status, note=note)

    # --- lifecycle --------------------------------------------------------------------------
    async def _sync_engine(self) -> SyncEngine:
        _, keys = await self._identity()
        if self._engine is None:
            self._engine = SyncEngine(self._fs, self._gateway, keys, self._clock, self._sync_config)
        return self._engine

    @_typed
    async def sync(self, wait: bool = True, *, full: bool = False) -> IndexStatus:
        """Index the checkout (`wait=False` returns once planned; the work continues in the background of
        the running event loop). Another sync holding the lock: its status."""
        return await (await self._sync_engine()).sync(wait, full=full)

    @_typed
    async def watch(self, stop: threading.Event, cfg: WatchConfig | None = None) -> WatchResult:
        """Keep the index fresh until `stop` is set; returns at once (`started=False`) if a watcher runs."""
        engine = await self._sync_engine()
        return await ensure_watch(self._fs, self._gateway, engine, self._clock, stop, cfg)

    @_typed
    async def login(self, key: str) -> None:
        """Store the gateway key (`scout_key`) in the key store; the composition root builds the gateway
        from it. An empty key or an app with no key store is a typed error."""
        if not key.strip():
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "key must not be empty")
        if self._keystore is None:
            raise ScoutError(ErrorCode.INTERNAL, "no key store configured")
        self._keystore.set(KEY_NAME, key.strip().encode("utf-8"))
