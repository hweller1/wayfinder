"""`seed` (design sections 5.4, 8; PRD 4.4): reuse a teammate's index through a MinHash match.

Match. The tenant's own sketches are scanned linearly: the candidates are the tenant's branch namespaces
(`list_namespaces("scout_{tenant}_")`, filtered to the exact `scout_{tenant}_{repo}_{branch}` shape, so a
stale sketch of a deleted branch and another tenant's rows are never read) and their sketch rows are fetched
by id from `scout_{tenant}_meta`. The best estimated Jaccard (`crypto.minhash.jaccard`, ties by namespace
name) at or above the threshold (default 0.55) and whose tree has a root row wins; below it nothing is
written and the client does a normal full sync. A sketch row holds only the keyed sketch (written by
`sync_commit`), never leaf hashes or paths.

Write order for a match (all idempotent, so a retry of the whole request converges):
1. the proofs row, fail-closed (`proofs_required=true`, no leaves), BEFORE any copy, so the copied namespace
   can never be searched unfiltered, whatever crashes next;
2. `{ns}__tree` copy, THEN the `{ns}` copy (ordering invariant, design 5): a crash in between leaves an
   orphan tree that the next seed or `sync_begin` deletes and recopies; a commit landing on the source
   between the two copies leaves tree-older-than-chunks, which the client's next diff repairs;
3. the proof pages.

Proof paging (eval repos exceed 5k files, `wire.MAX_PROOFS` is per request). `SeedReq` carries `page` of
`pages` (<= 20 x 5k). Page 0 starts the session and must come first: it decides the match, resets the proofs
row to the fail-closed placeholder and clears staged pages. Every page is staged as its own row
(`proofpage-*`, never read by `search`). Only when ALL `pages` staged rows exist does the proofs row get its
`leaves` (the union), so until the final page arrives the proof set is empty and `search` returns nothing from
the seeded namespace: a partial upload can never widen what the client sees. Completion is "every page row
exists", not "the last page arrived", so out-of-order and repeated pages are safe. Page 0 is charged to the
`general` rate limit once per upload; the other pages are bounded (<= 19) cheap row writes.

Idempotency is per (tenant, repo_id, branch_id) (the clone id is inside `branch_id`): with a root row and a
proofs row present the seed is resumed (no re-match, no re-copy); with a root row and no proofs row the
namespace is already committed (or an ordinary one) and the answer is `seeded=false`. An existing empty
namespace pair (the client called `sync_begin` first) is replaced by the branch. `sync_commit` drops the
proofs row and the staged pages (3.4).
"""

import base64
import binascii
import re
from collections.abc import Sequence

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import MAX_PROOF_PAGES, SeedReq, SeedRes
from scout.core.crypto import minhash
from scout.core.ports.store import Row, VectorStore
from scout.core.service.ratelimit import RateLimiter
from scout.core.service.schema import META_SCHEMA, ROOT_ID
from scout.core.service.syncsvc import (
    decode_sketch,
    meta_namespace,
    proof_page_row_id,
    proofs_row_id,
    sketch_row_id,
)

DEFAULT_THRESHOLD = 0.55  # PRD 4.4; tuned by the MinHash sweep (task 13.10)
MAX_PROOF_LEN = 128  # a leaf hash is 64 hex chars; bounds the memory of a request
_SKETCH_BATCH = 1000


async def repair_orphan_tree(store: VectorStore, ns: str, tree: str) -> None:
    """A `{ns}__tree` with no `{ns}` is what a crash between the two branch copies leaves: delete it."""
    if not await store.namespace_exists(ns) and await store.namespace_exists(tree):
        await store.delete_namespace(tree)


async def branch_pair(store: VectorStore, ns: str, tree: str, src_ns: str, src_tree: str) -> None:
    """Copy-on-write branch of a namespace pair: the tree FIRST, then the chunks (design 5)."""
    await repair_orphan_tree(store, ns, tree)
    await store.branch_from(tree, src_tree)
    await store.branch_from(ns, src_ns)


def _proof_ids(ns: str) -> list[str]:
    return [proof_page_row_id(ns, i) for i in range(MAX_PROOF_PAGES)]


class SeedService:
    def __init__(
        self, store: VectorStore, limiter: RateLimiter, *, threshold: float = DEFAULT_THRESHOLD
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("seed threshold must be in (0, 1]")
        self._store = store
        self._limiter = limiter
        self._threshold = threshold

    async def seed(self, *, tenant: str, key_id: str, ns: str, tree: str, req: SeedReq) -> SeedRes:
        store = self._store
        sketch = _sketch_bytes(decode_sketch(req.minhash_b64))
        if req.page >= req.pages:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "page must be below pages")
        if any(len(p) > MAX_PROOF_LEN or not p for p in req.proofs):
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, f"a proof must be 1..{MAX_PROOF_LEN} characters")
        meta = meta_namespace(tenant)
        # "seeded or populated" needs BOTH halves: a lone tree is the orphan of a crash between the copies
        has_root = await store.namespace_exists(ns) and await self._has_root(tree)
        proofs = await self._proofs_row(meta, ns)
        if req.page > 0:
            if not has_root or proofs is None:
                raise ScoutError(ErrorCode.CONFLICT, "no seed in progress for this branch: send page 0 first")
            if proofs.attrs.get("pages") != req.pages:
                raise ScoutError(ErrorCode.INVALID_ARGUMENT, "pages differs from the page-0 request")
            jaccard = _row_jaccard(proofs)
            return await self._stage(meta, ns, req, jaccard)
        self._limiter.general(key_id)
        if has_root and proofs is None:
            return SeedRes(seeded=False)  # already committed, or an ordinary populated namespace
        if has_root and proofs is not None:
            jaccard = _row_jaccard(proofs)  # resume: the branch is already there
            await self._reset(meta, ns, req.pages, jaccard)
        else:
            found, best = await self._match(tenant, ns, sketch)
            if found is None:
                await store.delete_ids(meta, [proofs_row_id(ns), *_proof_ids(ns)])  # stale leftovers
                return SeedRes(seeded=False, jaccard=best)
            src_ns, src_tree, jaccard = found
            await self._reset(meta, ns, req.pages, jaccard)  # fail-closed BEFORE any copy
            if await store.namespace_exists(ns):  # an empty pair left by sync_begin
                await store.delete_namespace(ns)
                await store.delete_namespace(tree)
            await branch_pair(store, ns, tree, src_ns, src_tree)
        return await self._stage(meta, ns, req, jaccard)

    async def _has_root(self, tree: str) -> bool:
        store = self._store
        if not await store.namespace_exists(tree):
            return False
        return bool(await store.get_by_ids(tree, [ROOT_ID], attrs=[]))

    async def _proofs_row(self, meta: str, ns: str) -> Row | None:
        if not await self._store.namespace_exists(meta):
            return None
        rows = await self._store.get_by_ids(meta, [proofs_row_id(ns)])
        return rows[0] if rows and rows[0].attrs.get("proofs_required") is True else None

    async def _match(
        self, tenant: str, ns: str, mine: bytes
    ) -> tuple[tuple[str, str, float] | None, float | None]:
        """`((source ns, source tree, jaccard) | None, best jaccard seen | None)`."""
        store, meta = self._store, meta_namespace(tenant)
        if not await store.namespace_exists(meta):
            return None, None
        shape = re.compile(rf"^scout_{re.escape(tenant)}_[0-9a-f]{{32}}_[0-9a-f]{{16}}$")
        names = [n for n in await store.list_namespaces(f"scout_{tenant}_") if shape.match(n) and n != ns]
        scored: list[tuple[float, str]] = []
        for i in range(0, len(names), _SKETCH_BATCH):
            batch = names[i : i + _SKETCH_BATCH]
            rows = await store.get_by_ids(meta, [sketch_row_id(n) for n in batch], attrs=["sketch"])
            by_id = {r.id: r.attrs.get("sketch") for r in rows}
            for name in batch:
                theirs = _stored_sketch(by_id.get(sketch_row_id(name)))
                if theirs is not None:
                    scored.append((minhash.jaccard(mine, theirs), name))
        scored.sort(key=lambda t: (-t[0], t[1]))
        for jaccard, name in scored:
            if jaccard < self._threshold:
                break
            src_tree = name + "__tree"
            if await store.namespace_exists(src_tree) and await self._has_root(src_tree):
                return (name, src_tree, jaccard), jaccard
        return None, scored[0][0] if scored else None

    async def _reset(self, meta: str, ns: str, pages: int, jaccard: float) -> None:
        """Start (or restart) an upload: the fail-closed proofs row first, then clear staged pages."""
        await self._write_proofs(meta, ns, [], pages, jaccard, complete=False)
        await self._store.delete_ids(meta, _proof_ids(ns))

    async def _write_proofs(
        self, meta: str, ns: str, leaves: Sequence[str], pages: int, jaccard: float, *, complete: bool
    ) -> None:
        store = self._store
        await store.ensure_namespace(meta, META_SCHEMA)
        attrs = {
            "kind": "proofs",
            "proofs_required": True,
            "leaves": list(leaves),
            "pages": pages,
            "jaccard": repr(jaccard),
            "complete": complete,
        }
        await store.upsert(meta, [Row(id=proofs_row_id(ns), attrs=attrs)])

    async def _stage(self, meta: str, ns: str, req: SeedReq, jaccard: float) -> SeedRes:
        store = self._store
        page = Row(
            id=proof_page_row_id(ns, req.page),
            attrs={"kind": "proofpage", "page": req.page, "leaves": list(req.proofs)},
        )
        await store.upsert(meta, [page])
        ids = [proof_page_row_id(ns, i) for i in range(req.pages)]
        staged = await store.get_by_ids(meta, ids, attrs=["leaves"])
        if len(staged) < req.pages:  # outstanding pages: the proofs row stays empty, search stays closed
            return SeedRes(seeded=True, jaccard=jaccard, complete=False)
        union: set[str] = set()
        for row in staged:
            leaves = row.attrs.get("leaves")
            if isinstance(leaves, list | tuple):
                union.update(v for v in leaves if isinstance(v, str))  # pyright: ignore[reportUnknownVariableType]
        await self._write_proofs(meta, ns, sorted(union), req.pages, jaccard, complete=True)
        return SeedRes(seeded=True, jaccard=jaccard, complete=True)


def _sketch_bytes(b64: str | None) -> bytes:
    if b64 is None:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "minhash_b64 is required")
    return base64.b64decode(b64)


def _stored_sketch(value: object) -> bytes | None:
    if not isinstance(value, str):
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == minhash.SKETCH_BYTES else None


def _row_jaccard(row: Row) -> float:
    try:
        return float(str(row.attrs.get("jaccard")))
    except ValueError:
        return 0.0
