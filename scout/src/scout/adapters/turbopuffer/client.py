"""`TurbopufferStore`: the `VectorStore` port over turbopuffer SDK 3.0.0 (design sections 3, 7; research 2).

Region `gcp-us-central1`; `turbopuffer` is imported lazily (the SDK is a gateway-only dependency). The
SDK's own retries are disabled (`max_retries=0`) so backoff lives here, in one testable place: 429, 5xx,
connection errors and timeouts retry with `min(0.5 * 2**n, 8)` s sleeps, then surface as
UPSTREAM_UNAVAILABLE (UPSTREAM_TIMEOUT for timeouts). 400 -> INVALID_ARGUMENT (this includes the vendor's
`each row must have a vector value` / `An empty vector was passed` rejections of vector-less rows in a vector
namespace; vendor text is never echoed), 404 -> NOT_FOUND, 409 -> CONFLICT, 401/403 -> INTERNAL (the
gateway's own credentials, not the caller's). Vendor exceptions never leak.

Mapping notes (spike artifacts/spike/0a_tpuf.json):
- `NamespaceSchema(dims=None)` -> no `vector` attribute (vector-less namespace: tree, meta, keys);
  `dims=int` -> `vector: {type: "[dims]f32", ann: true}`. `fts="pre_tokenized_array"` ->
  `full_text_search: {tokenizer: "pre_tokenized_array"}`.
- `if_absent` -> `upsert_condition=("id", "Eq", None)`; `condition` -> `upsert_condition=<filter>`
  (evaluated against the existing row; a missing row passes).
- `delete_ids` is a `delete_by_filter(id In ids)` so the returned count is rows actually deleted.
- `branch_from` is one namespace per call (`ns.branch_from(source_namespace=...)`); copying a sibling
  `__tree` namespace first is the service's job (design section 5).
- A namespace is a vector namespace or not, decided at creation. Writing a row with a `vector` into a
  vector-less namespace would silently turn it into one on the real service, so this adapter refuses
  (INVALID_ARGUMENT) when the namespace's schema is known to have no vector attribute.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from scout.adapters.turbopuffer.filters import to_tpuf_filter
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.store import And, AttrSpec, Eq, Filter, In, NamespaceSchema, Not, Row

__all__ = ["DEFAULT_REGION", "TurbopufferStore", "schema_payload"]

DEFAULT_REGION = "gcp-us-central1"
GET_BY_IDS_CHUNK = 5000
_DISTANCE = "cosine_distance"
_NOT_ATTRS = frozenset({"id", "vector", "$dist"})

type Sleep = Callable[[float], Awaitable[None]]


def _attr_payload(spec: AttrSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"type": spec.type, "filterable": spec.filterable}
    if spec.glob:
        out["glob"] = True
    if spec.fts == "pre_tokenized_array":
        out["full_text_search"] = {"tokenizer": "pre_tokenized_array"}
    return out


def schema_payload(schema: NamespaceSchema) -> dict[str, Any]:
    """The `schema=` argument of a turbopuffer write for `schema` (no `id`; see `ensure_namespace`)."""
    out: dict[str, Any] = {name: _attr_payload(spec) for name, spec in schema.attrs.items()}
    if schema.dims is not None:
        out["vector"] = {"type": f"[{schema.dims}]f32", "ann": True}
    return out


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def _fts_is_pre_tokenized(cfg: Any) -> bool:
    return cfg is not None and cfg is not False and _get(cfg, "tokenizer") == "pre_tokenized_array"


def _server_dims(server: Mapping[str, Any]) -> int | None:
    vec = server.get("vector")
    typ = _get(vec, "type") if vec is not None else None
    if isinstance(typ, str) and typ.startswith("[") and typ.endswith("]f32"):
        return int(typ[1 : -len("]f32")])
    return None


def _schema_matches(server: Mapping[str, Any], want: NamespaceSchema) -> bool:
    """Tolerant comparison: dims (including vector-less) and, for each declared attribute the server knows,
    type, glob and full-text mode. The server adds `id` and defaults; those are ignored."""
    if _server_dims(server) != want.dims:
        return False
    for name, spec in want.attrs.items():
        cfg = server.get(name)
        if cfg is None:
            continue
        if _get(cfg, "type") != spec.type:
            return False
        if bool(_get(cfg, "glob")) != spec.glob:
            return False
        if _fts_is_pre_tokenized(_get(cfg, "full_text_search")) != (spec.fts == "pre_tokenized_array"):
            return False
    return True


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)  # pyright: ignore[reportUnknownArgumentType]
    out: dict[str, Any] = dict(getattr(row, "model_extra", None) or {})
    out["id"] = row.id
    out["vector"] = getattr(row, "vector", None)
    return out


def _to_row(raw: Any, *, vectors: bool, score: float | None) -> Row:
    d = _row_dict(raw)
    vec = d.get("vector")
    return Row(
        id=str(d["id"]),
        attrs={k: v for k, v in d.items() if k not in _NOT_ATTRS},
        vector=[float(x) for x in vec] if vectors and isinstance(vec, list | tuple) else None,
        score=score,
    )


def _dist(raw: Any) -> float | None:
    v = _row_dict(raw).get("$dist")
    return float(v) if isinstance(v, int | float) else None


class TurbopufferStore:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        region: str = DEFAULT_REGION,
        client: Any = None,
        max_attempts: int = 4,
        sleep: Sleep = asyncio.sleep,
        timeout: float = 60.0,
    ) -> None:
        """`client` injects an `AsyncTurbopuffer`-shaped object (tests); otherwise one is built lazily from
        `api_key`/`region` on first use."""
        self._api_key = api_key
        self._region = region
        self._client: Any = client
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._timeout = timeout
        self._dims: dict[str, int | None] = {}  # known vector-ness per namespace (schema cache)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "TurbopufferStore":
        """`TURBOPUFFER_API_KEY` (required), `TURBOPUFFER_REGION` (default `gcp-us-central1`)."""
        key = env.get("TURBOPUFFER_API_KEY")
        if not key:
            raise ScoutError(ErrorCode.INTERNAL, "TURBOPUFFER_API_KEY is not set")
        return cls(key, region=env.get("TURBOPUFFER_REGION") or DEFAULT_REGION)

    # --- plumbing -----------------------------------------------------------------------------------
    def _sdk(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise ScoutError(ErrorCode.INTERNAL, "turbopuffer API key is not configured")
            import turbopuffer

            self._client = turbopuffer.AsyncTurbopuffer(
                api_key=self._api_key, region=self._region, max_retries=0, timeout=self._timeout
            )
        return self._client

    def _ns(self, name: str) -> Any:
        return self._sdk().namespace(name)

    async def _call[T](self, op: Callable[[], Awaitable[T]]) -> T:
        """Run one vendor call with backoff; vendor exceptions become `ScoutError`."""
        last: ScoutError | None = None
        for attempt in range(self._max_attempts):
            try:
                return await op()
            except ScoutError:
                raise
            except Exception as exc:
                err, retry = _translate(exc)
                if not retry:
                    raise err from None
                last = err
                if attempt + 1 < self._max_attempts:
                    await self._sleep(min(0.5 * 2**attempt, 8.0))
        assert last is not None
        raise last

    async def _schema_of(self, ns: str) -> Mapping[str, Any] | None:
        try:
            return await self._call(lambda: self._ns(ns).schema())
        except ScoutError as e:
            if e.code is ErrorCode.NOT_FOUND:
                return None
            raise

    async def _known_dims(self, ns: str) -> tuple[bool, int | None]:
        if ns not in self._dims:
            server = await self._schema_of(ns)
            if server is None:
                return False, None
            self._dims[ns] = _server_dims(server)
        return True, self._dims[ns]

    # --- namespaces ---------------------------------------------------------------------------------
    async def ensure_namespace(self, ns: str, schema: NamespaceSchema) -> None:
        server = await self._schema_of(ns)
        if server is None:
            # An empty namespace can only be created with `id` declared (SDK docs for create_namespace).
            payload = {"id": "string", **schema_payload(schema)}
            extra: dict[str, Any] = {"distance_metric": _DISTANCE} if schema.dims is not None else {}
            try:
                await self._call(lambda: self._ns(ns).write(create_namespace=True, schema=payload, **extra))
            except ScoutError as e:
                if e.code is not ErrorCode.CONFLICT:
                    raise
            server = await self._schema_of(ns)
        if server is not None and not _schema_matches(server, schema):
            raise ScoutError(ErrorCode.INTERNAL, "namespace exists with a different schema")
        self._dims[ns] = schema.dims

    async def namespace_exists(self, ns: str) -> bool:
        return bool(await self._call(lambda: self._ns(ns).exists()))

    async def list_namespaces(self, prefix: str) -> list[str]:
        async def page() -> list[str]:
            return [str(n.id) async for n in self._sdk().namespaces(prefix=prefix)]

        return sorted(await self._call(page))

    async def delete_namespace(self, ns: str) -> None:
        self._dims.pop(ns, None)
        try:
            await self._call(lambda: self._ns(ns).delete_all())
        except ScoutError as e:
            if e.code is not ErrorCode.NOT_FOUND:
                raise

    async def branch_from(self, ns: str, source: str) -> None:
        if await self.namespace_exists(ns):
            raise ScoutError(ErrorCode.CONFLICT, "destination namespace already exists")
        self._dims.pop(ns, None)
        await self._call(lambda: self._ns(ns).branch_from(source_namespace=source))

    # --- writes -------------------------------------------------------------------------------------
    async def upsert(
        self, ns: str, rows: Sequence[Row], *, if_absent: bool = False, condition: Filter | None = None
    ) -> int:
        if if_absent and condition is not None:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "if_absent and condition are mutually exclusive")
        if not rows:
            return 0
        if any(r.vector is not None for r in rows):
            known, dims = await self._known_dims(ns)
            if known and dims is None:
                raise ScoutError(ErrorCode.INVALID_ARGUMENT, "vector row written to a vector-less namespace")
        body: list[dict[str, Any]] = []
        for r in rows:
            row: dict[str, Any] = {k: _plain(v) for k, v in r.attrs.items() if k not in _NOT_ATTRS}
            row["id"] = r.id
            if r.vector is not None:
                row["vector"] = [float(x) for x in r.vector]
            body.append(row)
        kwargs: dict[str, Any] = {"upsert_rows": body}
        if any(r.vector is not None for r in rows):
            kwargs["distance_metric"] = _DISTANCE
        if if_absent:
            kwargs["upsert_condition"] = ("id", "Eq", None)
        elif condition is not None:
            kwargs["upsert_condition"] = to_tpuf_filter(condition)
        res = await self._call(lambda: self._ns(ns).write(**kwargs))
        return int(res.rows_affected)

    async def _delete_where(self, ns: str, flt: Filter) -> int:
        tpuf_flt = to_tpuf_filter(flt)
        try:
            res = await self._call(lambda: self._ns(ns).write(delete_by_filter=tpuf_flt))
        except ScoutError as e:
            if e.code is ErrorCode.NOT_FOUND:
                return 0
            raise
        return int(res.rows_affected)

    async def delete_ids(self, ns: str, ids: Sequence[str]) -> int:
        unique = list(dict.fromkeys(ids))
        total = 0
        for i in range(0, len(unique), GET_BY_IDS_CHUNK):
            total += await self._delete_where(ns, In("id", unique[i : i + GET_BY_IDS_CHUNK]))
        return total

    async def delete_by_file(self, ns: str, file_id: str, *, except_file_hash: str | None = None) -> int:
        flt: Filter = Eq("file_id", file_id)
        if except_file_hash is not None:
            flt = And([flt, Not(Eq("file_hash", except_file_hash))])
        return await self._delete_where(ns, flt)

    async def delete_by_filter(self, ns: str, flt: Filter) -> int:
        return await self._delete_where(ns, flt)

    # --- reads --------------------------------------------------------------------------------------
    async def get_by_ids(
        self,
        ns: str,
        ids: Sequence[str],
        *,
        include_vectors: bool = False,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return []
        include: Any
        if attrs is not None:
            include = [a for a in attrs if a not in _NOT_ATTRS] + (["vector"] if include_vectors else [])
        elif include_vectors:
            server = await self._schema_of(ns)
            if server is None:
                raise ScoutError(ErrorCode.NOT_FOUND, "namespace does not exist")
            include = [a for a in server if a not in _NOT_ATTRS] + ["vector"]
        else:
            include = True
        found: dict[str, Row] = {}
        for i in range(0, len(wanted), GET_BY_IDS_CHUNK):
            chunk = wanted[i : i + GET_BY_IDS_CHUNK]
            res = await self._call(
                lambda chunk=chunk: self._ns(ns).query(
                    filters=("id", "In", chunk), top_k=len(chunk), include_attributes=include
                )
            )
            for raw in res.rows or ():
                r = _to_row(raw, vectors=include_vectors, score=None)
                found[r.id] = r
        return [found[i] for i in wanted if i in found]

    async def query_ann(
        self,
        ns: str,
        vector: Sequence[float],
        *,
        top_k: int,
        flt: Filter | None = None,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        if top_k < 1:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "top_k must be >= 1")
        rank = ("vector", "ANN", [float(x) for x in vector])
        res = await self._ranked(ns, rank, top_k, flt, attrs)
        # cosine_distance: 0 = identical; the port wants higher = better
        return [_to_row(r, vectors=False, score=1.0 - (_dist(r) or 0.0)) for r in res]

    async def query_bm25(
        self,
        ns: str,
        field: str,
        tokens: Sequence[str],
        *,
        top_k: int,
        flt: Filter | None = None,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        if top_k < 1:
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "top_k must be >= 1")
        if not tokens:
            return []
        rank = (field, "BM25", list(dict.fromkeys(tokens)))  # pre_tokenized_array: operand is list[str]
        res = await self._ranked(ns, rank, top_k, flt, attrs)
        return [_to_row(r, vectors=False, score=_dist(r) or 0.0) for r in res]

    async def _ranked(
        self,
        ns: str,
        rank_by: tuple[Any, ...],
        top_k: int,
        flt: Filter | None,
        attrs: Sequence[str] | None,
    ) -> list[Any]:
        kwargs: dict[str, Any] = {
            "rank_by": rank_by,
            "top_k": top_k,
            "include_attributes": True if attrs is None else [a for a in attrs if a not in _NOT_ATTRS],
        }
        if flt is not None:
            kwargs["filters"] = to_tpuf_filter(flt)
        res = await self._call(lambda: self._ns(ns).query(**kwargs))
        return list(res.rows or ())


def _plain(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def _translate(exc: Exception) -> tuple[ScoutError, bool]:
    """(typed error, retryable?). Fixed messages only: the vendor body is never echoed. Duck-typed on the
    SDK's exception class names and `status_code`: no vendor import, and stubs can raise lookalikes."""
    names = {c.__name__ for c in type(exc).__mro__}
    status = getattr(exc, "status_code", None)
    if "APITimeoutError" in names or isinstance(exc, TimeoutError):
        return ScoutError(ErrorCode.UPSTREAM_TIMEOUT, "the vector store timed out"), True
    if "APIConnectionError" in names:
        return ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the vector store is unreachable"), True
    if "APIStatusError" in names and isinstance(status, int):
        if status == 429 or status >= 500:
            return ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "the vector store is unavailable"), True
        if status == 404:
            return ScoutError(ErrorCode.NOT_FOUND, "namespace does not exist"), False
        if status == 409:
            return ScoutError(ErrorCode.CONFLICT, "the vector store reported a conflict"), False
        if status in (401, 403):
            return ScoutError(
                ErrorCode.INTERNAL, "the vector store rejected the configured credentials"
            ), False
        return ScoutError(ErrorCode.INVALID_ARGUMENT, "the vector store rejected the request"), False
    return ScoutError(ErrorCode.INTERNAL, "the vector store adapter failed"), False
