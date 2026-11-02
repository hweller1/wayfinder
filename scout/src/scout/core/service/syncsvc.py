"""`sync_nodes`, `sync_plan` and `embed` (design sections 3, 5.1/5.2, 7; PRD 4.3 items 4 and 5).

The embedding cache is the namespace `scout_{tenant}_cache`: rows `{id: chunk_key, vector}`, written
insert-if-absent, so it is per tenant (no cross-tenant hit is possible) and concurrent syncs of the same
content converge on one vector. Only cache misses reach the `Embedder`; a chunk that is unchanged or moved to
another file keeps its key and is never embedded again. Chunk text is in memory for the length of one `embed`
call and is never written to any store; rows carry only the sealed attributes the client sent plus a vector.

`embed` order (5.1, 14): cache lookup, embed misses in batches sized by `Embedder.count_tokens` (never by
characters, <= 100K tokens), cache write, upsert rows with vectors read back from the cache, THEN delete the
old rows of replaced files, so a search never sees a file with no rows. `embed` never touches `{ns}__tree`.

`stage_nodes` + `commit` (5.1, 5.2, 5.4, 15) are the only writers of node and root rows, and both run
strictly after the client's last `embed` of the sync. NODE PAGING PROTOCOL (task 4.1a; a commit carries <=
500 node rows, a repo has thousands of directories): the client sends the node rows in `pages` pages of <=
`MAX_NODE_IDS` through `stage_nodes(base_root, new_root, page, pages, node_upserts, node_deletes)`, then
`commit(new_root, staged_pages=pages)` carrying no node rows. A page writes its node rows to `{ns}__tree`
and THEN a marker row `nodepage-<sha(new_root)[:16]>-<page>` (`kind=nodepage`, `pages`, the upserted and
deleted ids; no `children_enc`, so `nodes` never returns it). A page is idempotent (same rows, same marker)
and pages may arrive in any order and repeat. It is a no-op when the root already is `new_root` and CONFLICT
(server root in details) when the root is neither `base_root` nor `new_root`. The page's node DELETES are
only recorded in its marker. `commit` refuses with INVALID_ARGUMENT (nothing written) unless markers
`0..staged_pages-1` for this `new_root` all exist with `pages == staged_pages` and no staged or inline
upsert is also a delete. Order inside commit: node rows (inline ones; staged ones are already written), then
the root row (the commit point, so `server_root` never names a node that is missing: the root is after every
chunk write AND every node page), then node deletes (inline plus staged), the marker rows, the proofs drop
and the sketch write. Every step after the root write is idempotent and is re-run by a replay of the same
`new_root`, so a crash between them heals on retry (markers are deleted last, so a replay still sees the
deletes; the markers of an abandoned sync are a few small rows under another `new_root`, left in place).
Crash between pages: the root still names the old tree and no marker set is complete, so the retry restages
(idempotent) and commits. Inline `node_upserts`/`node_deletes` on `CommitReq` stay for small commits (<= 500
each) and combine with staged pages. A replay of the same `new_root` may raise the coverage counters (never
lower them). The CAS is decided by the store, so it holds across processes: the root row is written with
`condition=Eq(root_hash, <base root>)` (turbopuffer `upsert_condition`; the first commit is
insert-if-absent) and zero rows affected is CONFLICT. The per-tree lock is only an optimisation inside one
process: an in-process loser fails on the read-check before it writes any node row. Residual (cross-process
only): two containers racing from one base both pass the read-check and both upsert node rows (ids are
path-keyed, not content-addressed) before the root CAS, so the loser's node rows can overwrite the winner's
while the root stays the winner's; a later diff may then fail its node-hash check (MerkleError). Healing
that needs content-addressed or generation-scoped node ids (follow-up). A clone also syncs under
`sync.lock`.

CONFLICT protocol (client rule, task 4.3). Chunk rows are not versioned by commit, so a loser's `embed`
(which ran before its CAS) may have overwritten the winner's rows for a file. After a CONFLICT the loser
re-diffs against the winner's root AND re-embeds every file of its own failed delta, even those the winner's
tree now agrees with (embed cache hit, no model call; replace-by-hash makes the rows match the tree). A
plain re-diff alone would leave such a file's chunks at the loser's version while the tree names the
winner's.

Staleness window (stale chunks vs the committed tree). For a file F that sync S replaces or removes, chunk
rows disagree with the committed tree from `t_up` to `t_root`: `t_up` is the upsert of F's new rows in S's
`embed` (for a removal, its `delete_files`), `t_root` is the root-row write of S's `commit`. Inside it: (a)
between the upsert and the `delete_by_file(except_file_hash)` of ONE embed call, rows of both hashes exist
(one delete round trip); (b) once that embed returns and until `commit`, only the new hash exists while the
tree still names the old one (tree older than chunks, the safe direction of the ordering invariant); a
removed file has no rows but is still in the tree. After `t_root` nothing disagrees. Stale old-hash rows
never survive a finished embed, and `commit` adds no chunk deletes (the server cannot read the encrypted
tree). If the client dies inside the window it lasts until the next sync, whose diff is unchanged (the
server root is still the old one), replays the same embed (idempotent row ids, cache hits) and commits."""

import asyncio
import base64
import binascii
import hashlib
import weakref
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import (
    MAX_PROOF_PAGES,
    CommitReq,
    EmbedReq,
    EmbedRes,
    NodesRes,
    PlanRes,
    StageNodesReq,
    StageNodesRes,
)
from scout.core.crypto import ids
from scout.core.crypto.minhash import SKETCH_BYTES
from scout.core.ports.clock import Clock
from scout.core.ports.embedder import Embedder
from scout.core.ports.store import Eq, NamespaceSchema, Row, VectorStore
from scout.core.service.ratelimit import RateLimiter
from scout.core.service.schema import CHUNK_DIMS, META_SCHEMA, ROOT_ID, TREE_SCHEMA

CACHE_SCHEMA = NamespaceSchema(dims=CHUNK_DIMS, attrs={})  # vectors only; the row id is the chunk key
MAX_BATCH_TOKENS = 100_000  # per upstream embed request (PRD 4.3 item 4)
REPLAY_TTL_S = 24 * 3600.0  # design 6: Idempotency-Key replays within 24 h return the stored response
MAX_IDEMPOTENCY_KEY_LEN = 128
_DELETE_BATCH = 1000  # ids per `delete_ids` call


def _tenant_namespace(tenant: str, suffix: str) -> str:
    """`scout_{tenant}_{suffix}`; the tenant is validated by the same rule as every other namespace."""
    try:
        ids.namespace(tenant, "0" * ids.REPO_ID_LEN, "0" * ids.BRANCH_ID_LEN)
    except ValueError as exc:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, str(exc)) from None
    return f"scout_{tenant}_{suffix}"


def cache_namespace(tenant: str) -> str:
    return _tenant_namespace(tenant, "cache")


def meta_namespace(tenant: str) -> str:
    """Vector-less: `kind=sketch` rows (one per branch namespace) and `kind=proofs` rows (teammate seed)."""
    return _tenant_namespace(tenant, "meta")


def sketch_row_id(ns: str) -> str:
    return "sketch-" + hashlib.sha256(ns.encode()).hexdigest()[:32]


def proofs_row_id(ns: str) -> str:
    return "proofs-" + hashlib.sha256(ns.encode()).hexdigest()[:32]


def proof_page_row_id(ns: str, page: int) -> str:
    """Staging row for one page of a multi-page seed upload (never read by `search`)."""
    return f"proofpage-{hashlib.sha256(ns.encode()).hexdigest()[:32]}-{page}"


def node_page_row_id(new_root: str, page: int) -> str:
    """Marker row of one staged node page (lives in `{ns}__tree`; the root hash scopes it to one sync)."""
    return f"nodepage-{hashlib.sha256(new_root.encode()).hexdigest()[:16]}-{page}"


def decode_sketch(b64: str | None) -> str | None:
    """The sketch must be exactly 512 bytes of base64; checked before any write."""
    if b64 is None:
        return None
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raw = b""
    if len(raw) != SKETCH_BYTES:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"minhash_b64 must decode to {SKETCH_BYTES} bytes")
    return b64


def _ids(marker: Row, attr: str) -> list[str]:
    value = marker.attrs.get(attr)
    items: list[Any] = value if isinstance(value, list) else []  # pyright: ignore[reportUnknownVariableType]
    return [v for v in items if isinstance(v, str)]


@dataclass(frozen=True, slots=True)
class Replay:
    digest: str  # sha256 of the request body: a reused key with another body is a client bug
    result: EmbedRes


class ReplayStore(Protocol):
    """Idempotency results (Modal Dict plugs in here). A cache only: row ids and the cache make a lost
    replay entry correct, merely slower."""

    async def get(self, key: str) -> Replay | None: ...

    async def put(self, key: str, replay: Replay) -> None: ...


class MemoryReplays:
    def __init__(self, clock: Clock, ttl_s: float = REPLAY_TTL_S) -> None:
        self._clock = clock
        self._ttl = ttl_s
        self._items: dict[str, tuple[float, Replay]] = {}

    async def get(self, key: str) -> Replay | None:
        item = self._items.get(key)
        if item is None or item[0] <= self._clock.now():
            return None
        return item[1]

    async def put(self, key: str, replay: Replay) -> None:
        now = self._clock.now()
        self._items = {k: v for k, v in self._items.items() if v[0] > now}
        self._items[key] = (now + self._ttl, replay)


class SyncService:
    def __init__(
        self, store: VectorStore, embedder: Embedder | None, limiter: RateLimiter, replays: ReplayStore
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._limiter = limiter
        self._replays = replays
        self._tree_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    async def nodes(self, tree: str, node_ids: Sequence[str]) -> NodesRes:
        """Encrypted node blobs by id, read from `{ns}__tree` only. Unknown ids (and a missing tree) are
        simply absent from the result; the root row has no `children_enc`, so it is never returned."""
        if not node_ids or not await self._store.namespace_exists(tree):
            return NodesRes(blobs={})
        rows = await self._store.get_by_ids(tree, list(dict.fromkeys(node_ids)), attrs=["children_enc"])
        blobs: dict[str, str] = {}
        for row in rows:
            blob = row.attrs.get("children_enc")
            if isinstance(blob, str) and blob:
                blobs[row.id] = blob
        return NodesRes(blobs=blobs)

    async def plan(self, tenant: str, chunk_keys: Sequence[str]) -> PlanRes:
        """The keys (deduplicated, in request order) that the tenant's cache does not hold."""
        unique = list(dict.fromkeys(chunk_keys))
        have = await self._cached(cache_namespace(tenant), unique, vectors=False)
        return PlanRes(missing=[k for k in unique if k not in have])

    async def embed(
        self, *, tenant: str, key_id: str, ns: str, req: EmbedReq, idempotency_key: str | None
    ) -> EmbedRes:
        if idempotency_key is None:
            return await self._embed(tenant, key_id, ns, req)
        if not 0 < len(idempotency_key) <= MAX_IDEMPOTENCY_KEY_LEN:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "Idempotency-Key must be 1..128 characters")
        replay_key = f"{tenant}\0embed\0{idempotency_key}"
        digest = hashlib.sha256(req.model_dump_json().encode()).hexdigest()
        prior = await self._replays.get(replay_key)
        if prior is not None:
            if prior.digest != digest:
                raise ScoutError(
                    ErrorCode.INVALID_ARGUMENT, "Idempotency-Key was used with a different request"
                )
            return prior.result
        result = await self._embed(tenant, key_id, ns, req)
        await self._replays.put(replay_key, Replay(digest, result))
        return result

    async def _embed(self, tenant: str, key_id: str, ns: str, req: EmbedReq) -> EmbedRes:
        store = self._store
        if not await store.namespace_exists(ns):
            raise ScoutError(ErrorCode.NOT_FOUND, "branch namespace does not exist: call sync_begin first")
        cache = cache_namespace(tenant)
        needed = {r.chunk_key for r in req.rows}
        vectors = await self._cached(cache, [*needed, *req.texts], vectors=True)
        unresolved = needed - vectors.keys() - req.texts.keys()
        if unresolved:  # checked before any spend: the client replans and sends those texts
            raise ScoutError(
                ErrorCode.CONFLICT,
                "cache entry missing for rows sent without text; replan",
                retryable=True,
                details={"missing": len(unresolved)},
            )
        fresh = {k: t for k, t in req.texts.items() if k not in vectors}
        embedded_tokens = 0
        if fresh:
            embedded_tokens = await self._embed_fresh(cache, key_id, fresh)
            vectors.update(await self._cached(cache, list(fresh), vectors=True))  # the winners' vectors
        rows: list[Row] = []
        for w in req.rows:
            vector = vectors.get(w.chunk_key)
            if vector is None:
                raise ScoutError(ErrorCode.INTERNAL, "cache entry missing after write")
            attrs = {
                "kind": "chunk",
                "file_id": w.file_id,
                "path_enc": w.path_enc,
                "ext_tag": w.ext_tag,
                "dir_tags": list(w.dir_tags),
                "file_hash": w.file_hash,
                "chunk_key": w.chunk_key,
                "loc_enc": w.loc_enc,
                "tokens": list(w.tokens),
            }
            rows.append(Row(id=w.id, attrs=attrs, vector=vector))
        if rows:
            await store.upsert(ns, rows)
        for rep in req.replace:  # strictly after the upsert: no window with a file's rows missing
            await store.delete_by_file(ns, rep.file_id, except_file_hash=rep.file_hash)
        for file_id in req.delete_files:
            await store.delete_by_file(ns, file_id)
        return EmbedRes(written=len(rows), embedded_tokens=embedded_tokens)

    async def stage_nodes(self, ns: str, tree: str, req: StageNodesReq) -> StageNodesRes:
        """Write one page of node rows and its marker (see the module docstring for the protocol)."""
        store = self._store
        if req.page >= req.pages:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "page must be below pages")
        if {n.id for n in req.node_upserts} & set(req.node_deletes):
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "a node cannot be both upserted and deleted")
        if not await store.namespace_exists(ns):
            raise ScoutError(ErrorCode.NOT_FOUND, "branch namespace does not exist: call sync_begin first")
        lock = self._tree_locks.get(tree)
        if lock is None:
            lock = self._tree_locks.setdefault(tree, asyncio.Lock())
        async with lock:
            await store.ensure_namespace(tree, TREE_SCHEMA)
            found = await store.get_by_ids(tree, [ROOT_ID])
            current_root = found[0].attrs.get("root_hash") if found else None
            if current_root == req.new_root:  # already committed: stage nothing, never rewrite its nodes
                return StageNodesRes(page=req.page, staged=0)
            if current_root != req.base_root:
                raise self._conflict(current_root)
            nodes = [
                Row(id=n.id, attrs={"kind": "node", "children_enc": n.children_enc, "node_hash": n.node_hash})
                for n in req.node_upserts
            ]
            if nodes:
                await store.upsert(tree, nodes)
            marker = Row(
                id=node_page_row_id(req.new_root, req.page),
                attrs={
                    "kind": "nodepage",
                    "pages": req.pages,
                    "upserted": [n.id for n in req.node_upserts],
                    "deleted": list(req.node_deletes),
                },
            )
            await store.upsert(tree, [marker])  # after the rows: a marker means its page is fully written
        return StageNodesRes(page=req.page, staged=len(nodes))

    async def _markers(self, tree: str, req: CommitReq) -> list[Row]:
        """The staged-page markers of `req.new_root` that exist (pages `0..staged_pages-1`)."""
        if not req.staged_pages:
            return []
        wanted = [node_page_row_id(req.new_root, p) for p in range(req.staged_pages)]
        return await self._store.get_by_ids(tree, wanted, attrs=["pages", "upserted", "deleted"])

    @staticmethod
    def _check_staged(req: CommitReq, markers: Sequence[Row]) -> None:
        """Every expected page staged with the same page count, and no node both upserted and deleted."""
        if len(markers) != req.staged_pages or any(m.attrs.get("pages") != req.staged_pages for m in markers):
            have = {m.id for m in markers}
            missing = [p for p in range(req.staged_pages) if node_page_row_id(req.new_root, p) not in have]
            raise ScoutError(
                ErrorCode.INVALID_ARGUMENT,
                "node pages are missing or were staged with another page count: stage every page first",
                details={"missing_pages": missing[:20]},
            )
        upserted = {n.id for n in req.node_upserts}
        deleted = set(req.node_deletes)
        for m in markers:
            upserted.update(_ids(m, "upserted"))
            deleted.update(_ids(m, "deleted"))
        if upserted & deleted:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "a node cannot be both upserted and deleted")

    async def commit(self, tenant: str, ns: str, tree: str, req: CommitReq) -> Mapping[str, Any]:
        """CAS the root row `expected_root -> new_root` and return the committed root's attributes.

        A root equal to `new_root` is a replay (OK). Any other mismatch is CONFLICT: the client re-diffs
        against the server root in `details`. See the module docstring for the write order."""
        store = self._store
        sketch = decode_sketch(req.minhash_b64)
        if {n.id for n in req.node_upserts} & set(req.node_deletes):
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "a node cannot be both upserted and deleted")
        if not await store.namespace_exists(ns):
            raise ScoutError(ErrorCode.NOT_FOUND, "branch namespace does not exist: call sync_begin first")
        lock = self._tree_locks.get(tree)
        if lock is None:
            lock = self._tree_locks.setdefault(tree, asyncio.Lock())
        async with lock:
            await store.ensure_namespace(tree, TREE_SCHEMA)
            found = await store.get_by_ids(tree, [ROOT_ID])
            current = dict(found[0].attrs) if found else None
            current_root = current.get("root_hash") if current else None
            if current is not None and current_root == req.new_root:
                root = await self._replay(tree, req, current)  # nodes and root exist; finish the tail
                markers = await self._markers(tree, req)  # whatever the interrupted tail has not cleaned
            else:
                if current_root != req.expected_root:
                    raise self._conflict(current_root)
                markers = await self._markers(tree, req)
                self._check_staged(req, markers)  # before any write
                root = await self._write_commit(tree, req, current)
            doomed = [*req.node_deletes, *(i for m in markers for i in _ids(m, "deleted"))]
            for i in range(0, len(doomed), _DELETE_BATCH):  # after the root: no longer named by it
                await store.delete_ids(tree, doomed[i : i + _DELETE_BATCH])
            if markers:  # last: a replay after a crash above still finds the deletes to redo
                await store.delete_ids(tree, [m.id for m in markers])
            await self._drop_proofs_write_sketch(tenant, ns, sketch)
        return root

    async def _replay(self, tree: str, req: CommitReq, current: Mapping[str, Any]) -> Mapping[str, Any]:
        """Same `new_root` again. The tree is already committed; only coverage may move, and only forward
        (a background embed finishing under an unchanged root raises `written_chunks`; a late duplicate with
        a lower count never lowers it)."""
        written, expected = current.get("written_chunks"), current.get("expected_chunks")
        written = written if isinstance(written, int) else 0
        grown = req.written_chunks > written
        if not grown and (req.expected_chunks is None or req.expected_chunks == expected):
            return current
        gen = current.get("gen")
        root = dict(current) | {
            "written_chunks": max(written, req.written_chunks),
            "gen": (gen if isinstance(gen, int) else 0) + 1,
        }
        if req.expected_chunks is not None:
            root["expected_chunks"] = req.expected_chunks
        await self._cas_root(tree, root, current)
        return root

    async def _cas_root(self, tree: str, root: Mapping[str, Any], current: Mapping[str, Any] | None) -> None:
        """Root write as a store-side compare-and-set on the root row's current `root_hash`: it holds
        across processes (two Modal containers), not just under the per-tree lock. A first commit is
        insert-if-absent; any other write passes `condition=Eq(root_hash, <base root>)`. Zero rows
        affected means another writer moved the root since our read: CONFLICT with the server root."""
        row = Row(id=ROOT_ID, attrs=root)
        base = current.get("root_hash") if current is not None else None
        if isinstance(base, str):
            affected = await self._store.upsert(tree, [row], condition=Eq("root_hash", base))
        else:
            affected = await self._store.upsert(tree, [row], if_absent=True)
        if affected == 0:
            found = await self._store.get_by_ids(tree, [ROOT_ID])
            raise self._conflict(found[0].attrs.get("root_hash") if found else None)

    @staticmethod
    def _conflict(server_root: object) -> ScoutError:
        return ScoutError(
            ErrorCode.CONFLICT,
            "the server root is not the expected base root; re-diff and retry",
            details={"server_root": server_root if isinstance(server_root, str) else None},
        )

    async def _write_commit(
        self, tree: str, req: CommitReq, current: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        store = self._store
        if req.node_upserts:
            nodes = [
                Row(id=n.id, attrs={"kind": "node", "children_enc": n.children_enc, "node_hash": n.node_hash})
                for n in req.node_upserts
            ]
            await store.upsert(tree, nodes)
        gen = current.get("gen") if current else None
        root: dict[str, Any] = {
            "kind": "root",
            "root_hash": req.new_root,
            "expected_chunks": req.written_chunks if req.expected_chunks is None else req.expected_chunks,
            "written_chunks": req.written_chunks,
            "gen": (gen if isinstance(gen, int) else 0) + 1,
        }
        if req.commit:
            root["commit"] = req.commit
        await self._cas_root(tree, root, current)
        return root

    async def _drop_proofs_write_sketch(self, tenant: str, ns: str, sketch: str | None) -> None:
        store, meta = self._store, meta_namespace(tenant)
        if await store.namespace_exists(meta):
            spent = [proofs_row_id(ns), *(proof_page_row_id(ns, i) for i in range(MAX_PROOF_PAGES))]
            await store.delete_ids(meta, spent)  # the roots match now: proofs (and staged pages) are spent
        if sketch is not None:
            await store.ensure_namespace(meta, META_SCHEMA)
            await store.upsert(meta, [Row(id=sketch_row_id(ns), attrs={"kind": "sketch", "sketch": sketch})])

    async def _cached(self, cache: str, keys: Iterable[str], *, vectors: bool) -> dict[str, list[float]]:
        """`{chunk_key: vector}` for the keys the cache holds (vectors empty when not requested)."""
        wanted = list(dict.fromkeys(keys))
        if not wanted or not await self._store.namespace_exists(cache):
            return {}
        rows = await self._store.get_by_ids(cache, wanted, include_vectors=vectors, attrs=[])
        return {r.id: list(r.vector) if r.vector is not None else [] for r in rows}

    async def _embed_fresh(self, cache: str, key_id: str, fresh: dict[str, str]) -> int:
        """Embeds the misses and writes them insert-if-absent; returns the real token count."""
        embedder = self._embedder
        if embedder is None:
            raise ScoutError(ErrorCode.INTERNAL, "embedding is not configured")
        charged = self._limiter.embed(key_id, sum(len(t) for t in fresh.values()))
        real = 0
        try:
            batches = await self._batches(embedder, fresh)
            await self._store.ensure_namespace(cache, CACHE_SCHEMA)
            for batch in batches:
                res = await embedder.embed([t for _, t in batch], input_type="document")
                if len(res.vectors) != len(batch):
                    raise ScoutError(ErrorCode.INTERNAL, "embedder returned the wrong number of vectors")
                real += res.tokens
                rows = [Row(id=k, vector=v) for (k, _), v in zip(batch, res.vectors, strict=True)]
                await self._store.upsert(cache, rows, if_absent=True)
        finally:
            self._limiter.reconcile_embed(key_id, charged, real)  # failures refund what was not embedded
        return real

    @staticmethod
    async def _batches(embedder: Embedder, fresh: dict[str, str]) -> list[list[tuple[str, str]]]:
        """Greedy packing by `count_tokens` (never characters) up to `MAX_BATCH_TOKENS` per request. A single
        chunk above the cap cannot be sent, so it is rejected before any embed call."""
        batches: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        used = 0
        for key, text in fresh.items():
            n = await embedder.count_tokens([text])
            if n > MAX_BATCH_TOKENS:
                raise ScoutError(ErrorCode.INVALID_ARGUMENT, "a chunk exceeds the per-request token cap")
            if current and used + n > MAX_BATCH_TOKENS:
                batches.append(current)
                current, used = [], 0
            current.append((key, text))
            used += n
        if current:
            batches.append(current)
        return batches
