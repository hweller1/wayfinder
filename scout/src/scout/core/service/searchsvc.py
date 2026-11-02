"""`search` (design sections 3, 5.6, 6; PRD 4.2 and 4.4): hybrid ANN + BM25 over `{ns}` only.

Scope is a whitelist: the filter is built from the three optional `SearchScope` fields (`path_glob_enc`,
`ext_tag`, `dir_tag`) plus a forced `Eq(kind, "chunk")`. No client-supplied attribute name ever reaches the
store, and the namespace comes from `crypto.ids` (authenticated tenant + hex ids), never from the request,
so `{ns}__tree` and other tenants' namespaces are unreachable. The store sees only the query vector and the
client's pre-hashed BM25 tokens; the query text goes to the `Embedder` and nowhere else.

Fusion is reciprocal rank fusion with k=60 over the two ranked lists (1-based ranks, a row missing from a list
contributes nothing); equal fused scores break by row id, so the result is deterministic. Each list is
fetched at `k * overfetch` (the client post-filters decrypted paths and keeps k). While the namespace's
proofs row says `proofs_required` (teammate seed, until the first sync commit drops it), hits whose
`file_hash` is not in the proof set are dropped here, before the response, and before the cut to
`k * overfetch`, so the response never holds a dropped hit.
"""

import asyncio
from collections.abc import Sequence

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import SearchReq, SearchRes, SearchScope, WireHit
from scout.core.ports.embedder import Embedder
from scout.core.ports.store import And, Contains, Eq, Filter, Glob, Row, VectorStore
from scout.core.service.ratelimit import RateLimiter
from scout.core.service.syncsvc import meta_namespace, proofs_row_id

RRF_K = 60
MAX_FILTER_VALUE_BYTES = 4096  # the store's per-value limit, checked here so the error carries a hint
_HIT_ATTRS = ["path_enc", "loc_enc", "file_hash", "chunk_key"]


def build_filter(scope: SearchScope) -> Filter:
    """`Eq(kind, chunk)` AND the whitelisted scope fields; nothing else can be expressed (`SearchScope`
    forbids extra fields, so an unknown attribute name is rejected when the request is parsed)."""
    parts: list[Filter] = [Eq("kind", "chunk")]
    for name, value in (
        ("path_glob_enc", scope.path_glob_enc),
        ("ext_tag", scope.ext_tag),
        ("dir_tag", scope.dir_tag),
    ):
        if value is None:
            continue
        if len(value.encode()) > MAX_FILTER_VALUE_BYTES:
            raise ScoutError(
                ErrorCode.INVALID_ARGUMENT,
                f"scope.{name} exceeds the 4 KiB filter limit; use a narrower path or dir_tag",
            )
    if scope.path_glob_enc is not None:
        parts.append(Glob("path_enc", scope.path_glob_enc))
    if scope.ext_tag is not None:
        parts.append(Eq("ext_tag", scope.ext_tag))
    if scope.dir_tag is not None:
        parts.append(Contains("dir_tags", scope.dir_tag))
    return And(parts)


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fused `(id, score)`, best first; ties by id."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, row_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))


class SearchService:
    def __init__(self, store: VectorStore, embedder: Embedder | None, limiter: RateLimiter) -> None:
        self._store = store
        self._embedder = embedder
        self._limiter = limiter

    async def search(self, *, tenant: str, key_id: str, ns: str, req: SearchReq) -> SearchRes:
        self._limiter.search(key_id)
        flt = build_filter(req.scope)
        store = self._store
        if not await store.namespace_exists(ns):
            raise ScoutError(ErrorCode.NOT_FOUND, "branch namespace does not exist: call sync_begin first")
        if self._embedder is None:
            raise ScoutError(ErrorCode.INTERNAL, "embedding is not configured")
        embedded = await self._embedder.embed([req.query], input_type="query")
        if len(embedded.vectors) != 1:
            raise ScoutError(ErrorCode.INTERNAL, "embedder returned the wrong number of vectors")
        fetch = req.k * req.overfetch
        ann_coro = store.query_ann(ns, embedded.vectors[0], top_k=fetch, flt=flt, attrs=_HIT_ATTRS)
        if req.tokens:
            ann, bm25 = await asyncio.gather(
                ann_coro, store.query_bm25(ns, "tokens", req.tokens, top_k=fetch, flt=flt, attrs=_HIT_ATTRS)
            )
        else:
            ann, bm25 = await ann_coro, []
        rows: dict[str, Row] = {r.id: r for r in [*bm25, *ann]}
        fused = rrf([[r.id for r in ann], [r.id for r in bm25]])
        allowed = await self._proof_set(tenant, ns)
        hits: list[WireHit] = []
        for row_id, score in fused:
            attrs = rows[row_id].attrs
            file_hash, path_enc, loc_enc, chunk_key = (
                attrs.get("file_hash"),
                attrs.get("path_enc"),
                attrs.get("loc_enc"),
                attrs.get("chunk_key"),
            )
            if not all(isinstance(v, str) for v in (file_hash, path_enc, loc_enc, chunk_key)):
                continue  # not a chunk row; never returned
            if allowed is not None and file_hash not in allowed:
                continue
            hits.append(
                WireHit(
                    path_enc=str(path_enc),
                    loc_enc=str(loc_enc),
                    chunk_key=str(chunk_key),
                    file_hash=str(file_hash),
                    score=score,
                )
            )
            if len(hits) == fetch:
                break
        return SearchRes(hits=hits)

    async def _proof_set(self, tenant: str, ns: str) -> frozenset[str] | None:
        """The seeded namespace's leaf hashes while `proofs_required`, else None (no filtering). A required
        row with a malformed leaf list yields the empty set, so it fails closed."""
        meta = meta_namespace(tenant)
        if not await self._store.namespace_exists(meta):
            return None
        rows = await self._store.get_by_ids(meta, [proofs_row_id(ns)])
        if not rows or rows[0].attrs.get("proofs_required") is not True:
            return None
        leaves = rows[0].attrs.get("leaves")
        if not isinstance(leaves, list | tuple):
            return frozenset()
        return frozenset(v for v in leaves if isinstance(v, str))  # pyright: ignore[reportUnknownVariableType]
