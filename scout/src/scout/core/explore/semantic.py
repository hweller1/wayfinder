"""The raw semantic search behind the `semantic_search` tool and primitive (design 5.6): scope compile, HMAC
query tokens, gateway `search`, decrypt, client post-filter. The result is `RawHit`s that have NOT been
checked against the disk; `candidates.verify_hits` does that.

Only plaintext paths and line ranges leave this module, and only after `open_hit` authenticated them (a hit
that fails to decrypt is skipped, never raised: another tenant's or a tampered row must not abort a search).
"""

from collections.abc import Callable

from scout.core.contracts.wire import SearchReq, StatusReq
from scout.core.crypto import rows
from scout.core.crypto.keys import CryptoError, Subkeys
from scout.core.crypto.scope import compile_scope
from scout.core.crypto.tokens import QUERY_CAP, hmac_tokens, identifier_tokens
from scout.core.explore.candidates import RawHit
from scout.core.explore.spec import RetrievalSpec
from scout.core.explore.tools import SemanticBackend
from scout.core.ports.gateway import Gateway


async def raw_search(
    gateway: Gateway,
    keys: Subkeys,
    tenant: str,
    ids: StatusReq,
    query: str,
    path_glob: str | None,
    k: int,
) -> list[RawHit]:
    """Best first. Raises `ScoutError` (INVALID_ARGUMENT for a glob outside the grammar or an empty query)."""
    plan = compile_scope(path_glob, keys, tenant, ids.repo_id)
    res = await gateway.search(
        SearchReq(
            repo_id=ids.repo_id,
            branch_id=ids.branch_id,
            query=query,
            k=k,
            scope=plan.server,
            tokens=hmac_tokens(keys, identifier_tokens(query, cap=QUERY_CAP)),
            overfetch=plan.overfetch,
        )
    )
    out: list[RawHit] = []
    for hit in res.hits:
        try:
            path, start, end = rows.open_hit(keys, tenant, ids.repo_id, hit)
        except CryptoError:
            continue
        if plan.post(path):
            out.append(RawHit(path, start, end, hit.file_hash, hit.score))
    return out


async def spec_search(
    gateway: Gateway,
    keys: Subkeys,
    tenant: str,
    ids: StatusReq,
    query: str,
    path_glob: str | None,
    k: int,
    spec: RetrievalSpec | None,
    mtime_ns: Callable[[str], int | None] | None = None,
) -> list[RawHit]:
    """`raw_search` under the retrieval spec: `k` is capped at `spec.k`, the gateway is asked for extra hits
    when excludes or boosts can change the list, then excluded paths are dropped and boosts rerank."""
    if spec is None:
        return await raw_search(gateway, keys, tenant, ids, query, path_glob, k)
    want = min(k, spec.k)
    flt = spec.filters
    wide = min(30, want * 3) if (spec.exclude or spec.boost or spec.recency_boost) else want
    hits = await raw_search(gateway, keys, tenant, ids, query, path_glob, wide)
    return flt.apply(hits, mtime_ns)[:want]


def make_backend(
    gateway: Gateway,
    keys: Subkeys,
    tenant: str,
    ids: StatusReq,
    leaf_hash: Callable[[bytes], str],
    spec: RetrievalSpec | None = None,
    mtime_ns: Callable[[str], int | None] | None = None,
) -> SemanticBackend:
    async def search(query: str, path_glob: str | None, k: int) -> list[RawHit]:
        return await spec_search(gateway, keys, tenant, ids, query, path_glob, k, spec, mtime_ns)

    return SemanticBackend(search=search, leaf_hash=leaf_hash)
