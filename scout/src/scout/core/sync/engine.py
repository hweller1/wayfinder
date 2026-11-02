"""`SyncEngine`: the client sync use case, first full sync (design 5.1, 8; PRD 4.3; FR-C8, FR-C9, FR-C11).

Runs entirely client-side through the `Gateway` port: it never sees a `VectorStore` or `Embedder`, so the
whole pipeline is testable against `make_fake_gateway` (the real `GatewayService` over fakes).

Pipeline of `sync()`:
1. Identify: `whoami` (tenant), `git_ids`, state dir, `repo_id`/`branch_id` (clone-scoped). Take the
   `sync.lock` flock; a second sync while it is held returns the gateway's status without doing work.
2. Scan (worker thread): `list_indexable` (one shared matcher, 100k-file cap with a warning), then each file's
   bytes are read ONCE; the leaf hash, the chunks, the chunk keys and the BM25 tokens all come from those same
   bytes, so no edit can land between hashing and chunking. The stat is taken before the read: a file that
   changes afterwards then mismatches its saved stat and is re-hashed by the next sync.
3. Tree: keyed Merkle tree, sealed node rows, MinHash sketch of the leaf set (stored by the gateway for 4.4).
4. `sync_begin`, then `sync_plan` (2000 keys per call) to learn which chunk keys miss the embed cache.
5. Embed: chunks are grouped by chunk key (one text per key, all rows of that key in the same request),
   batched by an estimated token count (<= 100K) and sent by up to 6 workers. Each request carries an
   Idempotency-Key (sha256 of its body), `replace` entries for the files it touches (so rows of an older
   version of a file left by a crashed run are removed) and texts only for keys the cache misses. Rows of
   cached keys are still sent: a crash between the cache write and the row upsert leaves a cache hit with
   no row.
6. Node pages, then commit, strictly after the last embed request has returned (the root row lands after all
   chunk writes AND all node pages): the sealed node rows go out in pages of `MAX_NODE_IDS` through
   `sync_stage_nodes` (idempotent per page, sent in order), then `sync_commit` carries only the new root and
   `staged_pages`. Then `trees/<branch>.json` and `status.json` are written atomically.

A crash anywhere before step 6 leaves no root row, so the index reads `absent`, never `ready`; the re-run
embeds only what the cache lacks.

Incremental sync (task 4.2, design 5.2, 8). When the saved tree's root is the server's root, only what changed
since it travels:
- Stat cache: a file whose `(size, mtime_ns, ino)` equals its saved entry is neither read nor hashed nor
  chunked (its leaf and chunk count come from the saved entry). ACCEPTED RISK (design 8): a file rewritten
  with the same size, the same mtime_ns AND the same inode is not noticed until its stat changes or a sync
  runs with `full=True` (`scout sync --full`), which re-hashes every file. Any change of size, mtime_ns or
  inode is always picked up. A touched file whose bytes hash to the saved leaf is not chunked again.
- Delta: the local leaf maps (saved vs now) give added/changed/removed files with no node fetch (fast path:
  the server root is the saved root). Only added and changed files are chunked and sent; their `replace`
  entries drop the old rows; removed files, and changed files that no longer have any chunk, are sent as
  `delete_files`.
- Tree: only the nodes on the path from each changed file to the root are staged (wire bytes grow with
  changed files x depth x fan-out, not with repo size); directories that vanished are `node_deletes`.
- Commit: `expected_root` is the saved root. The root counters stay repo totals: the saved entries carry each
  file's chunk count, so `expected_chunks` is the sum over all files and `written_chunks` subtracts what this
  sync's embeds failed to write.
- Heal: when the server's coverage is below 1.0 the chunks of unchanged files may be missing, so every file is
  chunked and re-sent (nodes still only the changed path); this also repairs a `partial` index.
Same order as the first sync: chunk writes and deletes, then node pages, then the root CAS.

Branches, conflicts, partial coverage (task 4.3, design 5.3, 15; PRD 4.3 Branches, item 6):
- Branch switch: `branch_id` hashes the branch NAME (case and `/` preserved, a detached HEAD is
  `detached-<commit12>`), so every name is its own namespace. `sync_begin` carries `base_branch_id` (the
  default branch's id, None on the default branch itself); for a new branch namespace the gateway branches
  `{ns}__tree` THEN `{ns}` from it. The first sync on the branch then diffs the local tree against the
  SERVER tree (`sync_nodes`, one round trip per level) and embeds only the files that differ. Switching
  back finds the old namespace populated and, when the files match again, sends nothing.
- Server diff: any populated server whose root is neither this tree's nor the saved tree's (a new branch, a
  lost state dir, another writer that committed) is diffed by `merkle.diff` against the server nodes. The
  stat cache never decides which files travel there: a file the cache skipped but the server names with
  another leaf is re-chunked first. Only the delta files' chunks are sent and only their path nodes staged.
- Tree repair (`MerkleError`: a node missing, unreadable or not matching its parent hash, e.g. node rows
  overwritten by a cross-container race, see syncsvc): none of the server tree is trusted. Every file is
  re-chunked and re-sent (cache hits, so no model call; idempotent row ids and replace-by-hash), every node
  row is rewritten and the root is CAS'd against the current server root. Files only the old local tree
  knew are deleted; chunks of files only the unreadable server tree knew cannot be named and stay
  (residual). `merkle_resyncs` counts these; a warning and a log line are emitted.
- CONFLICT (the server root moved between `sync_begin` and the root CAS): re-prepare and retry, at most
  `conflict_retries` (3) times. The loser's `embed` ran before its CAS and may have overwritten the
  winner's chunk rows, so the retry re-chunks and re-sends EVERY file of its own failed delta (added,
  changed, removed), even those the winner's tree now agrees with (cache hits; replace-by-hash makes the
  rows match the tree). Chunk rows are never assumed to match the tree (a branch copy lands tree-first, so
  the tree may be older than the chunks).
- Partial coverage: a batch whose embed still fails with an upstream error after its retries is skipped, not
  fatal, as long as another batch succeeded. The root is committed with the rows actually written, so
  `coverage` is honest (< 0.8 reads `partial`); the next sync sees coverage < 1 and re-sends every file,
  which the cache makes cheap. If NO batch succeeded the error is raised and nothing is committed.

Teammate seed (task 4.4, design 5.4, 8; PRD 4.4; FR-C16, FR-C17, AC-4.2, AC-4.3). Only a brand-new namespace
is seeded: the first sync of this checkout (no saved tree) whose `sync_begin` says `empty` (no root row).
The client sends the MinHash sketch of its leaf set and the leaf hashes as proofs, in pages of `MAX_PROOFS`
(page 0 first: it decides the match; a repo above 5k files needs several pages, at most `MAX_PROOF_PAGES`).
When the gateway matches (`seeded=true`) it branches a teammate's `{ns}__tree` THEN `{ns}`; the client then
calls `sync_begin` again (now `populated`) and runs the ordinary server-tree diff of 5.3: only files whose
leaf differs from the teammate's are chunk-sent, and the commit drops the proofs. Until that commit,
`search` returns only hits whose `file_hash` is in the proof set, so a file this checkout changed never
shows the teammate's version. The client treats the copied TREE as authoritative and never trusts chunk
presence (the copy order leaves the tree older than the chunks at worst; the diff re-sends those files).
`seeded=false` (no sketch within threshold, or the namespace is already committed) falls through to the full
sync. A seed that fails with a retryable upstream error after its retries is skipped with a warning: a half-
uploaded seed is still fail-closed on the server and the sync converges through the diff.

Still open: a repo emptied of every indexable file is reported `absent` and leaves its server rows untouched.
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import math
from collections.abc import Awaitable, Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.types import IndexStatus
from scout.core.contracts.wire import (
    MAX_CHUNK_KEYS,
    MAX_FILES,
    MAX_NODE_IDS,
    MAX_NODE_PAGES,
    MAX_PROOFS,
    MAX_ROWS,
    CommitReq,
    EmbedReq,
    EmbedRes,
    NodesReq,
    NodeUpsert,
    PlanReq,
    ReplaceFile,
    SeedReq,
    StageNodesReq,
    StatusReq,
    SyncBeginReq,
    WireRow,
)
from scout.core.crypto import ids, minhash, paths, rows
from scout.core.crypto.keys import CryptoError, Subkeys
from scout.core.crypto.tokens import hmac_tokens, identifier_tokens
from scout.core.local.ignore import BINARY_SNIFF_BYTES, MAX_FILE_BYTES
from scout.core.ports.clock import Clock
from scout.core.ports.fs import FileSystem
from scout.core.ports.gateway import Gateway
from scout.core.sync import chunker, chunker_declaration, merkle
from scout.core.sync.files import MAX_FILES as MAX_INDEXED_FILES
from scout.core.sync.files import list_indexable
from scout.core.sync.state import READY_THRESHOLD, FileEntry, StateDir, TreeState, iso_utc, stat_unchanged

BATCH_TOKENS = 100_000
WORKERS = 6
RETRIES = 3
CONFLICT_RETRIES = 3
CHUNK_KEY_LEN = 32
_TEXT_CHARS_PER_TOKEN = 3.5  # same estimate as the fake embedder; the gateway re-batches by real tokens
_State = Literal["ready", "partial", "absent"]
_STRATEGIES = (chunker.TREE_SITTER_STRATEGY, chunker_declaration.DECLARATION_STRATEGY)
_SOFT_EMBED_ERRORS = frozenset(
    {ErrorCode.UPSTREAM_UNAVAILABLE, ErrorCode.UPSTREAM_TIMEOUT, ErrorCode.RATE_LIMITED}
)
LOGGER = logging.getLogger("scout.sync")


@dataclass(frozen=True, slots=True)
class SyncConfig:
    strategy: str = chunker.TREE_SITTER_STRATEGY
    max_files: int = MAX_INDEXED_FILES
    workers: int = WORKERS
    batch_tokens: int = BATCH_TOKENS
    plan_batch: int = MAX_CHUNK_KEYS
    retries: int = RETRIES
    conflict_retries: int = CONFLICT_RETRIES

    def __post_init__(self) -> None:
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"unknown chunk strategy {self.strategy!r}")
        if min(self.max_files, self.workers, self.batch_tokens, self.plan_batch) < 1:
            raise ValueError("sync limits must be positive")
        if self.retries < 0 or self.conflict_retries < 0:
            raise ValueError("sync limits must be positive")
        if self.plan_batch > MAX_CHUNK_KEYS:
            raise ValueError(f"plan_batch is at most {MAX_CHUNK_KEYS}")


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / _TEXT_CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class _Chunk:
    path: str
    key: str
    text: str  # what is embedded: path header, newline, body
    row: WireRow


@dataclass(frozen=True, slots=True)
class _Scan:
    entries: dict[str, FileEntry]
    chunks: list[_Chunk]
    skipped: list[str]
    reused: bool  # some file was not chunked: its entry (and chunk count) came from the saved tree
    chunked: frozenset[str]  # paths read and chunked by this scan (a file with no chunk is still in here)


@dataclass(slots=True)
class _Batch:
    texts: dict[str, str] = field(default_factory=lambda: {})  # fresh keys only
    all_texts: dict[str, str] = field(default_factory=lambda: {})  # every key in the batch (CONFLICT replan)
    rows: list[WireRow] = field(default_factory=lambda: [])
    files: dict[str, str] = field(default_factory=lambda: {})  # file_id -> file_hash
    deletes: list[str] = field(default_factory=lambda: [])  # file ids whose rows are all removed
    tokens: int = 0


def _split_group(group: list[_Chunk]) -> list[list[_Chunk]]:
    """Parts of one chunk key's rows with at most MAX_ROWS rows and MAX_FILES distinct files each (boilerplate
    can repeat in more than MAX_FILES files; a request carries at most MAX_FILES `replace` entries)."""
    parts: list[list[_Chunk]] = []
    cur: list[_Chunk] = []
    files: set[str] = set()
    for c in group:
        if cur and (len(cur) >= MAX_ROWS or (c.row.file_id not in files and len(files) >= MAX_FILES)):
            parts.append(cur)
            cur, files = [], set()
        cur.append(c)
        files.add(c.row.file_id)
    if cur:
        parts.append(cur)
    return parts


@dataclass(slots=True)
class _Run:
    state: StateDir
    tenant: str
    repo_id: str
    branch_id: str
    head: str | None
    prior_state: _State
    prior_coverage: float
    prior_commit: str | None
    prior_last_sync: str | None
    prior_ok_at: str | None
    base_branch_id: str | None = None  # the default branch's id; None on the default branch itself


@dataclass(frozen=True, slots=True)
class _Job:
    tree: merkle.Tree
    entries: dict[str, FileEntry]
    nodes: list[NodeUpsert]
    node_deletes: list[str]
    sketch: str
    batches: list[_Batch]
    expected: int  # chunk rows of the whole tree
    sent: int  # chunk rows this sync sends (all of them for a first sync, the delta for an incremental one)
    expected_root: str | None
    touched: frozenset[str] = frozenset()  # paths of this sync's own delta: a CONFLICT retry re-sends them


class WireMeter:
    """Gateway wrapper that counts the JSON bytes of every request and response (PRD 4.3 item 2: wire bytes
    scale with changed files x depth, not repo size). `calls` lists `(method, request_bytes,
    response_bytes)` in call order; anything it does not know is passed through unmetered by name lookup."""

    def __init__(self, gw: Gateway) -> None:
        self._gw = gw
        self.calls: list[tuple[str, int, int]] = []

    @property
    def request_bytes(self) -> int:
        return sum(c[1] for c in self.calls)

    @property
    def response_bytes(self) -> int:
        return sum(c[2] for c in self.calls)

    @property
    def total_bytes(self) -> int:
        return self.request_bytes + self.response_bytes

    def by_method(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, up, down in self.calls:
            out[name] = out.get(name, 0) + up + down
        return out

    def reset(self) -> None:
        self.calls.clear()

    def __getattr__(self, name: str) -> Callable[..., Awaitable[object]]:
        target = getattr(self._gw, name)

        async def metered(*args: object, **kwargs: object) -> object:
            up = sum(_wire_size(a) for a in args)
            result = await target(*args, **kwargs)
            self.calls.append((name, up, _wire_size(result)))
            return result

        return metered


def _wire_size(obj: object) -> int:
    if isinstance(obj, BaseModel):
        return len(obj.model_dump_json().encode())
    if isinstance(obj, bytes):
        return len(obj)
    return 0


class SyncEngine:
    def __init__(
        self, fs: FileSystem, gw: Gateway, keys: Subkeys, clock: Clock, cfg: SyncConfig | None = None
    ) -> None:
        self._fs, self._gw, self._keys, self._clock = fs, gw, keys, clock
        self._cfg = cfg or SyncConfig()
        self.warnings: list[str] = []
        self.merkle_resyncs = 0  # syncs that distrusted the server tree (MerkleError) and rebuilt it
        self._bg: asyncio.Task[IndexStatus] | None = None

    async def sync(self, wait: bool = True, *, full: bool = False) -> IndexStatus:
        """Index the checkout. `wait=False` returns once the plan is made; embed and commit then run in a
        background task (`join()` awaits it). Another sync holding the lock: its status is returned.
        `full=True` ignores the stat cache and re-hashes every file (`scout sync --full`)."""
        self.warnings = []
        run = await self._identify()
        stack = ExitStack()
        if not stack.enter_context(run.state.lock()):
            stack.close()
            return await self._contended(run)
        try:
            self._write_status(run, run.prior_state, run.prior_coverage, syncing=True)
            job = await self._prepare(run, full)
            if isinstance(job, IndexStatus):
                stack.close()
                return job
        except BaseException:
            self._restore_status(run)
            stack.close()
            raise
        if wait:
            try:
                return await self._execute(run, job, full)
            finally:
                stack.close()
        self._bg = asyncio.create_task(self._execute_released(run, job, stack, full))
        return IndexStatus(
            state=run.prior_state,
            coverage=run.prior_coverage,
            syncing=True,
            files=len(job.entries),
            semantic_enabled=run.prior_coverage >= READY_THRESHOLD,
        )

    async def join(self) -> IndexStatus | None:
        """Result of a `wait=False` sync once it has finished (None if none is running)."""
        bg, self._bg = self._bg, None
        return None if bg is None else await bg

    # --- identify -----------------------------------------------------------------------------
    async def _identify(self) -> _Run:
        tenant = (await self._gw.whoami()).tenant
        git = await asyncio.to_thread(self._fs.git_ids)
        state = await asyncio.to_thread(StateDir.open, self._fs)
        if git.root_commit is None:
            self.warnings.append("no root commit: repo id falls back to the remote only")
        if git.remote_norm is None:
            self.warnings.append("no remote: repo id falls back to the root commit only")
        if git.shallow:
            self.warnings.append("shallow clone: teammate seeding across clone depths may miss")
        repo_id = ids.repo_id(self._keys, git.root_commit or "", git.remote_norm or "")
        clone_id = await asyncio.to_thread(state.clone_id)
        branch_id = ids.branch_id(self._keys, repo_id, clone_id, git.branch)
        base_branch_id = (  # the default branch seeds every other branch (a detached HEAD included)
            ids.branch_id(self._keys, repo_id, clone_id, git.default_branch)
            if git.default_branch and git.default_branch != git.branch
            else None
        )
        await asyncio.to_thread(state.update_state, tenant=tenant, repo_id=repo_id, last_branch=branch_id)
        prior = await asyncio.to_thread(state.read_status)
        return _Run(
            state=state,
            tenant=tenant,
            repo_id=repo_id,
            branch_id=branch_id,
            head=git.head,
            prior_state=prior.state if prior else "absent",
            prior_coverage=prior.coverage if prior else 0.0,
            prior_commit=prior.commit if prior else None,
            prior_last_sync=prior.last_sync_at if prior else None,
            prior_ok_at=prior.gateway_ok_at if prior else None,
            base_branch_id=base_branch_id,
        )

    async def _contended(self, run: _Run) -> IndexStatus:
        status = await self._gw.index_status(StatusReq(repo_id=run.repo_id, branch_id=run.branch_id))
        return status.model_copy(update={"syncing": True})

    # --- prepare: scan, tree, begin, plan -----------------------------------------------------
    async def _prepare(
        self, run: _Run, full: bool, force: frozenset[str] = frozenset()
    ) -> _Job | IndexStatus:
        """`force`: paths whose chunks are re-sent whatever the diff says (the failed delta of a CONFLICT)."""
        listing = await asyncio.to_thread(list_indexable, self._fs, None, max_files=self._cfg.max_files)
        self.warnings.extend(listing.warnings)
        saved = await asyncio.to_thread(run.state.load_tree, run.branch_id)  # None: first sync, or lost state
        return await self._prepare_scan(run, listing.paths, saved, full, False, force)

    async def _prepare_scan(
        self,
        run: _Run,
        paths_: list[str],
        saved: TreeState | None,
        full: bool,
        rechunk: bool,
        force: frozenset[str] = frozenset(),
        *,
        first: bool = True,
        distrust: bool = False,
    ) -> _Job | IndexStatus:
        """`rechunk`: chunk every file. `force`: chunk (and send) these even when unchanged. `distrust`: the
        server tree failed its hash checks, so it is rewritten whole instead of diffed. `first` is False on a
        second pass over the same files (their skip warnings were already given)."""
        scan = await asyncio.to_thread(self._scan, paths_, run, saved, full, rechunk, force)
        if first:
            self.warnings.extend(
                f"skipped {p}: vanished, refused or no longer indexable" for p in scan.skipped
            )
        if not scan.entries:  # empty repo: nothing to index, grep-only
            return self._finish(run, IndexStatus(state="absent", coverage=0.0, semantic_enabled=False))
        tree, nodes, sketch = await asyncio.to_thread(self._build, scan.entries, run)
        if len(nodes) > MAX_NODE_IDS * MAX_NODE_PAGES:  # refused before any embed spend
            raise ScoutError(
                ErrorCode.INVALID_ARGUMENT,
                f"{len(nodes)} directories exceed the {MAX_NODE_IDS * MAX_NODE_PAGES} node rows per sync",
            )
        expected = sum(e.chunks for e in scan.entries.values())
        begin_req = SyncBeginReq(
            repo_id=run.repo_id,
            branch_id=run.branch_id,
            base_branch_id=run.base_branch_id,
            root=tree.root_hash,
            expected_chunks=expected,
        )
        begin = await self._gw.sync_begin(begin_req)
        if first and saved is None and begin.ns_state == "empty" and await self._seed(run, tree, sketch):
            begin = await self._gw.sync_begin(begin_req)  # the namespace is now the teammate's branch
        delta = merkle.Delta()
        node_deletes: list[str] = []
        delta_only = False  # the server holds every chunk of the saved tree: send the delta alone
        if begin.ns_state == "populated":
            if begin.server_root == tree.root_hash:  # the server holds this tree's nodes already
                if begin.coverage >= 1.0 and not force:  # and every chunk: nothing to send
                    await asyncio.to_thread(
                        run.state.save_tree, run.branch_id, TreeState(tree.root_hash, scan.entries)
                    )
                    status = await self._gw.index_status(
                        StatusReq(repo_id=run.repo_id, branch_id=run.branch_id)
                    )
                    return self._finish(run, status, files=len(scan.entries))
                nodes = []  # partial index under an unchanged tree: its node rows are all there already
                delta_only = begin.coverage >= 1.0
            elif saved is not None and begin.server_root == saved.root:  # incremental (5.2)
                delta = merkle.diff_leaves({p: e.leaf for p, e in saved.files.items()}, tree.leaves)
                nodes, node_deletes = self._path_nodes(tree, nodes, delta, run)
                delta_only = begin.coverage >= 1.0
            elif distrust:  # the server tree is rewritten whole; the saved tree only names files that left
                old = {p: e.leaf for p, e in saved.files.items()} if saved is not None else {}
                delta = merkle.diff_leaves(old, tree.leaves)
                node_deletes = self._path_nodes(tree, nodes, delta, run)[1]
            else:  # a new branch, a lost state dir or another writer: diff against the server tree (5.3)
                try:
                    delta = await self._server_diff(run, tree, begin.server_root)
                except merkle.MerkleError as e:
                    return await self._resync_whole(run, paths_, saved, full, force, str(e))
                nodes, node_deletes = self._path_nodes(tree, nodes, delta, run)
                delta_only = begin.coverage >= 1.0
        if delta_only:
            wanted = {*delta.added, *delta.changed, *force} & tree.leaves.keys()
            lacking = wanted - scan.chunked  # the stat cache skipped them but the server lacks them
            if lacking:
                return await self._prepare_scan(
                    run, paths_, saved, full, False, force | lacking, first=False, distrust=distrust
                )
            chunks = [c for c in scan.chunks if c.path in wanted]
        else:
            if scan.reused:
                # Wiped server, partial index to heal or a stale local cache: the chunks of the files the stat
                # cache skipped are not known to be on the server, so every file goes out.
                return await self._prepare_scan(
                    run, paths_, saved, full, True, force, first=False, distrust=distrust
                )
            chunks = scan.chunks
        gone = {
            *delta.removed,
            *(p for p in delta.changed if scan.entries[p].chunks == 0),
            *(p for p in force if p not in scan.entries or scan.entries[p].chunks == 0),
        }
        missing = await self._plan(run, chunks)
        return _Job(
            tree=tree,
            entries=scan.entries,
            nodes=nodes,
            node_deletes=node_deletes,
            sketch=sketch,
            batches=self._batches(chunks, missing, [ids.file_id(self._keys, p) for p in sorted(gone)]),
            expected=expected,
            sent=len(chunks),
            expected_root=begin.server_root,
            touched=frozenset({c.path for c in chunks} | gone),
        )

    async def _seed(self, run: _Run, tree: merkle.Tree, sketch: str) -> bool:
        """Ask the gateway to branch a teammate's index for this brand-new namespace. Proofs (the keyed leaf
        hashes, de-duplicated and sorted) go out in pages of `MAX_PROOFS`, page 0 first. True when seeded and
        every page was accepted; False (full sync) when nothing matched or the seed was skipped."""
        proofs = sorted(set(tree.leaves.values()))
        pages = max(1, math.ceil(len(proofs) / MAX_PROOFS))
        try:
            for page in range(pages):
                res = await self._with_retries(
                    lambda page=page: self._gw.seed(
                        SeedReq(
                            repo_id=run.repo_id,
                            branch_id=run.branch_id,
                            minhash_b64=sketch,
                            proofs=proofs[page * MAX_PROOFS : (page + 1) * MAX_PROOFS],
                            page=page,
                            pages=pages,
                        )
                    )
                )
                if not res.seeded:  # no teammate index within the threshold (only page 0 can say so)
                    return False
            LOGGER.info("sync: seeded from a teammate's index (%d proofs, %d pages)", len(proofs), pages)
            return True
        except ScoutError as e:
            if e.code not in _SOFT_EMBED_ERRORS:
                raise
            self.warnings.append(f"teammate seed skipped ({e.code.value}): full sync")
            LOGGER.warning("sync: teammate seed failed (%s); continuing without it", e.code.value)
            return False

    async def _server_diff(self, run: _Run, tree: merkle.Tree, server_root: str | None) -> merkle.Delta:
        """`merkle.diff` against the server's tree: one `sync_nodes` round trip per tree level. The diff is
        synchronous, so it runs in a worker thread and fetches through the event loop. A node blob that does
        not open raises `MerkleError` like any other inconsistency."""
        loop = asyncio.get_running_loop()

        def fetch(node_ids: Sequence[str]) -> dict[str, merkle.SealedNode]:
            out: dict[str, merkle.SealedNode] = {}
            for i in range(0, len(node_ids), MAX_NODE_IDS):
                part = list(node_ids[i : i + MAX_NODE_IDS])
                res = asyncio.run_coroutine_threadsafe(
                    self._with_retries(
                        lambda part=part: self._gw.sync_nodes(
                            NodesReq(repo_id=run.repo_id, branch_id=run.branch_id, ids=part)
                        )
                    ),
                    loop,
                ).result()
                for nid, blob in res.blobs.items():
                    try:
                        opened = rows.open_node(self._keys, run.tenant, run.repo_id, nid, blob)
                    except CryptoError as e:
                        raise merkle.MerkleError("remote node unreadable") from e
                    out[nid] = merkle.SealedNode(blob, merkle.node_hash(opened))
            return out

        return await asyncio.to_thread(merkle.diff, tree, server_root, fetch, self._keys)

    async def _resync_whole(
        self,
        run: _Run,
        paths_: list[str],
        saved: TreeState | None,
        full: bool,
        force: frozenset[str],
        why: str,
    ) -> _Job | IndexStatus:
        """The server tree is inconsistent (`MerkleError`): do not trust any of it. Every file is chunked and
        sent again (embed cache hits, idempotent row ids), every node row is rewritten and the root is CAS'd
        against the current server root, so a rival's half-written nodes are overwritten, not diffed."""
        self.merkle_resyncs += 1
        self.warnings.append(f"server tree is inconsistent ({why}): full re-sync of the tree")
        LOGGER.warning("sync: server tree inconsistent (%s); full re-sync of the tree", why)
        return await self._prepare_scan(run, paths_, saved, full, True, force, first=False, distrust=True)

    def _path_nodes(
        self, tree: merkle.Tree, nodes: list[NodeUpsert], delta: merkle.Delta, run: _Run
    ) -> tuple[list[NodeUpsert], list[str]]:
        """Only the nodes a delta can have changed: every directory on the path from a touched file to the
        root (upserted, or deleted when no file is left under it)."""
        touched = {*delta.added, *delta.changed, *delta.removed}
        dirs = {""} | {d for p in touched for d in paths.ancestors(p)}
        live = {tree.nodes[d].node_id for d in dirs if d in tree.nodes}
        gone = [ids.node_id(self._keys, run.repo_id, d) for d in sorted(dirs) if d not in tree.nodes]
        return [n for n in nodes if n.id in live], gone

    def _scan(
        self,
        paths: list[str],
        run: _Run,
        saved: TreeState | None,
        full: bool,
        rechunk: bool,
        force: frozenset[str],
    ) -> _Scan:
        """Read each file once; everything derived from a file comes from that one `bytes` object. With a
        saved tree, a file whose `(size, mtime_ns, ino)` matches its entry is not read at all (`full` reads
        every file), and a file whose bytes hash to the saved leaf keeps its saved chunk count and is not
        chunked. `rechunk` chunks every file (healing a partial index or an unknown server); `force` does the
        same for the listed paths."""
        entries: dict[str, FileEntry] = {}
        chunks: list[_Chunk] = []
        skipped: list[str] = []
        reused = False
        chunked: set[str] = set()
        known = saved.files if saved is not None else {}
        for path in paths:
            again = rechunk or path in force
            try:
                st = self._fs.stat(path)
                prev = known.get(path)
                if prev is not None and not again and stat_unchanged(prev, st, full=full):
                    entries[path] = prev
                    reused = True
                    continue
                data = self._fs.read_bytes(path, max_bytes=MAX_FILE_BYTES + 1)  # the one full read
            except ScoutError as e:
                if e.code in (ErrorCode.NOT_FOUND, ErrorCode.PATH_REFUSED):
                    skipped.append(path)
                    continue
                raise
            if len(data) > MAX_FILE_BYTES or b"\x00" in data[:BINARY_SNIFF_BYTES]:
                skipped.append(path)  # grew or became binary after listing
                continue
            leaf = merkle.leaf_hash(self._keys, data)
            if prev is not None and prev.leaf == leaf and not again:  # touched, not edited
                entries[path] = FileEntry(leaf, st.size, st.mtime_ns, st.ino, prev.chunks)
                reused = True
                continue
            chunked.add(path)
            seen: dict[str, int] = {}
            spans = self._chunks(path, data)
            entries[path] = FileEntry(leaf, st.size, st.mtime_ns, st.ino, len(spans))
            for body, text, start, end in spans:
                key = hmac.new(self._keys.chunk, body, hashlib.sha256).hexdigest()[:CHUNK_KEY_LEN]
                occurrence = seen.get(key, 0)
                seen[key] = occurrence + 1
                row = rows.seal_chunk_row(
                    self._keys,
                    run.tenant,
                    run.repo_id,
                    path=path,
                    start=start,
                    end=end,
                    chunk_key=key,
                    occurrence=occurrence,
                    file_hash=leaf,
                    tokens=hmac_tokens(self._keys, identifier_tokens(text)),
                )
                chunks.append(_Chunk(path, key, f"{path}\n{text}", row))
        return _Scan(entries, chunks, skipped, reused, frozenset(chunked))

    def _chunks(self, path: str, data: bytes) -> list[tuple[bytes, str, int, int]]:
        """`(body bytes, body text, start_line, end_line)` per chunk; the key hashes the body bytes."""
        if self._cfg.strategy == chunker_declaration.DECLARATION_STRATEGY:
            return [
                (d.text.encode(), d.text, d.start_line, d.end_line)
                for d in chunker_declaration.chunk_file(path, data, strategy=self._cfg.strategy)
            ]
        out: list[tuple[bytes, str, int, int]] = []
        for c in chunker.chunk_file(path, data, strategy=self._cfg.strategy):
            body = data[c.start_byte : c.end_byte]
            out.append((body, body.decode("utf-8", "replace"), c.start_line, c.end_line))
        return out

    def _build(self, entries: dict[str, FileEntry], run: _Run) -> tuple[merkle.Tree, list[NodeUpsert], str]:
        tree = merkle.build_tree(
            [(p, e.leaf) for p, e in entries.items()], self._keys, run.tenant, run.repo_id
        )
        nodes = [
            NodeUpsert(id=i, node_hash=n.node_hash, children_enc=n.children_enc)
            for i, n in merkle.seal_tree(tree, self._keys).items()
        ]
        leaves = [bytes.fromhex(leaf) for leaf in tree.leaves.values()]
        return tree, nodes, base64.b64encode(minhash.sketch(self._keys, leaves)).decode("ascii")

    async def _plan(self, run: _Run, chunks: list[_Chunk]) -> set[str]:
        unique = list(dict.fromkeys(c.key for c in chunks))
        missing: set[str] = set()
        step = self._cfg.plan_batch
        for i in range(0, len(unique), step):
            res = await self._gw.sync_plan(
                PlanReq(repo_id=run.repo_id, branch_id=run.branch_id, chunk_keys=unique[i : i + step])
            )
            missing.update(res.missing)
        return missing

    def _batches(self, chunks: list[_Chunk], missing: set[str], deleted: list[str]) -> list[_Batch]:
        """Group rows by chunk key (text and all its rows travel together), then pack by token estimate."""
        groups: dict[str, list[_Chunk]] = {}
        for c in chunks:
            groups.setdefault(c.key, []).append(c)
        out: list[_Batch] = []
        cur = _Batch()
        replaced: set[str] = set()  # a file's stale rows are deleted once, by the first batch holding it
        for key, group in groups.items():
            text = group[0].text
            fresh = key in missing
            cost = estimate_tokens(text) if fresh else 0
            for part in _split_group(group):
                new_files = {c.row.file_id: c.row.file_hash for c in part if c.row.file_id not in replaced}
                if cur.rows and (
                    cur.tokens + cost > self._cfg.batch_tokens
                    or len(cur.rows) + len(part) > MAX_ROWS
                    or len(cur.texts) + fresh > MAX_CHUNK_KEYS
                    or len(cur.files) + len(new_files) > MAX_FILES
                ):
                    out.append(cur)
                    cur = _Batch()
                cur.rows.extend(c.row for c in part)
                cur.files.update(new_files)
                replaced.update(new_files)
                cur.all_texts[key] = text
                if fresh:
                    cur.texts[key] = text
                    cur.tokens += cost
        if cur.rows:
            out.append(cur)
        for i in range(0, len(deleted), MAX_FILES):  # files that left the tree: their rows all go
            out.append(_Batch(deletes=deleted[i : i + MAX_FILES]))
        return out

    # --- execute: embed, commit, persist ------------------------------------------------------
    async def _execute_released(self, run: _Run, job: _Job, stack: ExitStack, full: bool) -> IndexStatus:
        try:
            return await self._execute(run, job, full)
        finally:
            stack.close()

    async def _execute(self, run: _Run, job: _Job, full: bool) -> IndexStatus:
        """Embed, stage the nodes, CAS the root. A CONFLICT from the staging or the commit (another writer
        moved the root) re-prepares against the new server root and goes again, at most `conflict_retries`
        times. The retry re-sends every file of this sync's own failed delta (see the module docstring)."""
        try:
            sent: frozenset[str] = frozenset()
            for attempt in range(self._cfg.conflict_retries + 1):
                written = await self._embed_all(run, job.batches)
                written += job.expected - job.sent  # rows of files this sync did not send are on the server
                try:
                    pages = await self._stage_nodes(run, job)  # after the last embed, before the root
                    res = await self._gw.sync_commit(  # strictly after the last embed request and node page
                        CommitReq(
                            repo_id=run.repo_id,
                            branch_id=run.branch_id,
                            expected_root=job.expected_root,
                            new_root=job.tree.root_hash,
                            staged_pages=pages,
                            written_chunks=written,
                            expected_chunks=job.expected,
                            commit=run.head,
                            minhash_b64=job.sketch,
                        )
                    )
                except ScoutError as e:
                    if e.code is not ErrorCode.CONFLICT or attempt == self._cfg.conflict_retries:
                        raise
                    sent |= job.touched
                    LOGGER.info("sync: root moved under us (attempt %d); re-diffing", attempt + 1)
                    again = await self._prepare(run, full, sent)
                    if isinstance(again, IndexStatus):
                        return again
                    job = again
                    continue
                await asyncio.to_thread(
                    run.state.save_tree, run.branch_id, TreeState(job.tree.root_hash, job.entries)
                )
                return self._finish(run, IndexStatus.model_validate(res.model_dump()), files=len(job.entries))
            raise AssertionError("unreachable")  # pragma: no cover
        except BaseException:
            self._restore_status(run)
            raise

    async def _stage_nodes(self, run: _Run, job: _Job) -> int:
        """Send the node rows (and the ids of vanished directories) in pages of `MAX_NODE_IDS`, in order;
        returns the page count. Pages are idempotent, so a crash between them is healed by the next run
        restaging all of them."""
        up, down = job.nodes, job.node_deletes
        count = max(math.ceil(len(up) / MAX_NODE_IDS), math.ceil(len(down) / MAX_NODE_IDS))
        for n in range(count):
            lo, hi = n * MAX_NODE_IDS, (n + 1) * MAX_NODE_IDS
            await self._with_retries(
                lambda lo=lo, hi=hi, n=n: self._gw.sync_stage_nodes(
                    StageNodesReq(
                        repo_id=run.repo_id,
                        branch_id=run.branch_id,
                        base_root=job.expected_root,
                        new_root=job.tree.root_hash,
                        page=n,
                        pages=count,
                        node_upserts=up[lo:hi],
                        node_deletes=down[lo:hi],
                    )
                )
            )
        return count

    async def _with_retries[T](self, call: Callable[[], Awaitable[T]]) -> T:
        """Retryable errors back off like `embed` does."""
        for attempt in range(self._cfg.retries + 1):
            try:
                return await call()
            except ScoutError as e:
                if not e.retryable or e.code is ErrorCode.CONFLICT or attempt == self._cfg.retries:
                    raise  # a CONFLICT is a moved root: the caller re-diffs, resending cannot help
                await self._clock.sleep(min(2.0**attempt, 60.0))
        raise AssertionError("unreachable")  # pragma: no cover

    async def _embed_all(self, run: _Run, batches: list[_Batch]) -> int:
        """Rows written. A batch that still fails with an upstream error after its retries is skipped (its
        rows are simply not written, so the commit reports honest coverage and the next sync heals it); if
        every batch failed the error is raised and nothing is committed. Any other error is fatal."""
        sem = asyncio.Semaphore(self._cfg.workers)
        failures: list[ScoutError] = []

        async def one(batch: _Batch) -> EmbedRes:
            async with sem:
                try:
                    return await self._send(run, batch)
                except ScoutError as e:
                    if e.code not in _SOFT_EMBED_ERRORS:
                        raise
                    failures.append(e)
                    return EmbedRes(written=0, embedded_tokens=0)

        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(one(b)) for b in batches]
        except BaseExceptionGroup as eg:
            raise eg.exceptions[0] from None
        if failures:
            if len(failures) == len(batches):
                raise failures[0]
            self.warnings.append(f"{len(failures)} of {len(batches)} embed batches failed: partial index")
        return sum(t.result().written for t in tasks)

    async def _send(self, run: _Run, batch: _Batch) -> EmbedRes:
        """One embed request. Retryable errors back off; CONFLICT (cache entry missing) resends every text."""
        texts = batch.texts
        replace = [ReplaceFile(file_id=f, file_hash=h) for f, h in batch.files.items()]
        for attempt in range(self._cfg.retries + 1):
            req = EmbedReq(
                repo_id=run.repo_id,
                branch_id=run.branch_id,
                texts=texts,
                rows=batch.rows,
                replace=replace,
                delete_files=batch.deletes,
            )
            key = hashlib.sha256(req.model_dump_json().encode()).hexdigest()
            try:
                return await self._gw.embed(req, idempotency_key=key)
            except ScoutError as e:
                if not e.retryable or attempt == self._cfg.retries:
                    raise
                if e.code is ErrorCode.CONFLICT:
                    texts = batch.all_texts
                else:
                    await self._clock.sleep(min(2.0**attempt, 60.0))
        raise AssertionError("unreachable")  # pragma: no cover

    # --- status.json --------------------------------------------------------------------------
    def _write_status(
        self,
        run: _Run,
        state: str,
        coverage: float,
        *,
        syncing: bool,
        commit: str | None = None,
        last_sync_at: str | None = None,
        ok_at: str | None = None,
    ) -> None:
        run.state.write_status(
            state,
            coverage,
            commit=commit if commit is not None else run.prior_commit,
            syncing=syncing,
            last_sync_at=last_sync_at if last_sync_at is not None else run.prior_last_sync,
            gateway_ok_at=ok_at if ok_at is not None else run.prior_ok_at,
        )

    def _restore_status(self, run: _Run) -> None:
        """A failed sync leaves the previous state (absent for a first sync) with `syncing` off."""
        self._write_status(run, run.prior_state, run.prior_coverage, syncing=False)

    def _finish(self, run: _Run, status: IndexStatus, *, files: int = 0) -> IndexStatus:
        now = iso_utc(self._clock.now())
        synced = status.state != "absent"
        status = status.model_copy(
            update={
                "syncing": False,
                "files": files,
                "last_sync_at": now if synced else None,
                "commit": status.commit or run.head,
            }
        )
        self._write_status(
            run,
            status.state,
            status.coverage,
            syncing=False,
            commit=status.commit,
            last_sync_at=status.last_sync_at,
            ok_at=now,
        )
        return status
