"""`FakeStore`: in-memory `VectorStore` mirroring the turbopuffer behaviours verified in spike 0a-A.

Part 1 (task 2.3): namespaces, rows, filter evaluation, copy-on-write `branch_from`. Part 2 (task 2.4):
`Glob`, `query_ann` (cosine), `query_bm25` (`pre_tokenized_array`, exact-token OR ranking) and a client-side
4 KiB filter-value guard. Mirrored real behaviour (artifacts/spike/0a_tpuf.json): a vector-less row written
to a namespace that has vectors is rejected as a whole batch (HTTP 400 -> INVALID_ARGUMENT); a vector-less
namespace accepts and copies vector-less rows; `And(Eq, Not(Eq))` and `Contains` on `[]string` filter
correctly; `if_absent` and `condition` upserts report rows actually written (the condition is checked
against the existing row; a missing row passes, as turbopuffer documents).
"""

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.store import (
    And,
    AttrSpec,
    Contains,
    Eq,
    Filter,
    Glob,
    In,
    NamespaceSchema,
    Not,
    Row,
)

_MAX_ID_BYTES = 64
_MAX_FILTER_VALUE_BYTES = 4096  # docs limit; spike 0a-A saw none up to 1 MiB, the guard stays client-side
_BM25_K1 = 1.2
_BM25_B = 0.75


@dataclass(slots=True)
class _Stored:
    attrs: dict[str, Any]
    vector: tuple[float, ...] | None


@dataclass(slots=True)
class _Namespace:
    schema: NamespaceSchema
    has_vector: bool
    rows: dict[str, _Stored] = field(default_factory=dict[str, _Stored])


def _err(code: ErrorCode, message: str) -> ScoutError:
    return ScoutError(code, message)


class FakeStore:
    def __init__(self) -> None:
        self._ns: dict[str, _Namespace] = {}

    # --- namespaces ---------------------------------------------------------------------------------
    async def ensure_namespace(self, ns: str, schema: NamespaceSchema) -> None:
        cur = self._ns.get(ns)
        if cur is None:
            self._ns[ns] = _Namespace(schema=schema, has_vector=schema.dims is not None)
        elif cur.schema != schema:
            raise _err(ErrorCode.INTERNAL, "namespace exists with a different schema")

    async def namespace_exists(self, ns: str) -> bool:
        return ns in self._ns

    async def list_namespaces(self, prefix: str) -> list[str]:
        return sorted(n for n in self._ns if n.startswith(prefix))

    async def delete_namespace(self, ns: str) -> None:
        self._ns.pop(ns, None)

    async def branch_from(self, ns: str, source: str) -> None:
        src = self._ns.get(source)
        if src is None:
            raise _err(ErrorCode.NOT_FOUND, "source namespace does not exist")
        if ns in self._ns:
            raise _err(ErrorCode.CONFLICT, "destination namespace already exists")
        self._ns[ns] = copy.deepcopy(src)  # deep copy stands in for copy-on-write

    # --- writes -------------------------------------------------------------------------------------
    async def upsert(
        self, ns: str, rows: Sequence[Row], *, if_absent: bool = False, condition: Filter | None = None
    ) -> int:
        cur = self._ns.get(ns)
        for row in rows:  # validate the whole batch first: a rejected batch writes nothing
            if not row.id or len(row.id.encode()) > _MAX_ID_BYTES:
                raise _err(ErrorCode.INVALID_ARGUMENT, "row id must be 1..64 bytes")
        if cur is None:
            # turbopuffer creates a namespace on first write; the first batch decides vector-ness.
            has_vec = any(r.vector is not None for r in rows)
            dims = len(next(r.vector for r in rows if r.vector is not None)) if has_vec else None
            cur = _Namespace(schema=NamespaceSchema(dims=dims, attrs={}), has_vector=has_vec)
            created = True
        else:
            created = False
        for row in rows:
            if row.vector is None and cur.has_vector:
                raise _err(ErrorCode.INVALID_ARGUMENT, "each row must have a vector value")
            if row.vector is not None and not cur.has_vector and not created:
                raise _err(ErrorCode.INVALID_ARGUMENT, "vector row written to a vector-less namespace")
            if row.vector is not None and cur.schema.dims is not None and len(row.vector) != cur.schema.dims:
                raise _err(ErrorCode.INVALID_ARGUMENT, "vector length does not match namespace dims")
        _validate_filter(condition, cur.schema.attrs)
        if created:
            self._ns[ns] = cur
        written = 0
        for row in rows:
            old = cur.rows.get(row.id)
            if old is not None and (
                if_absent or (condition is not None and not _matches(condition, old.attrs, cur.schema.attrs))
            ):
                continue  # an absent row always passes the condition (turbopuffer upsert_condition)
            vec = tuple(float(x) for x in row.vector) if row.vector is not None else None
            cur.rows[row.id] = _Stored(attrs=copy.deepcopy(dict(row.attrs)), vector=vec)
            written += 1
        return written

    async def delete_ids(self, ns: str, ids: Sequence[str]) -> int:
        cur = self._ns.get(ns)
        if cur is None:
            return 0
        return sum(1 for i in set(ids) if cur.rows.pop(i, None) is not None)

    async def delete_by_file(self, ns: str, file_id: str, *, except_file_hash: str | None = None) -> int:
        flt: Filter = Eq("file_id", file_id)
        if except_file_hash is not None:
            flt = And([flt, Not(Eq("file_hash", except_file_hash))])
        return await self.delete_by_filter(ns, flt)

    async def delete_by_filter(self, ns: str, flt: Filter) -> int:
        cur = self._ns.get(ns)
        if cur is None:
            return 0
        _validate_filter(flt, cur.schema.attrs)
        doomed = [i for i, r in cur.rows.items() if _matches(flt, r.attrs, cur.schema.attrs)]
        for i in doomed:
            del cur.rows[i]
        return len(doomed)

    # --- reads --------------------------------------------------------------------------------------
    async def get_by_ids(
        self,
        ns: str,
        ids: Sequence[str],
        *,
        include_vectors: bool = False,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        cur = self._require(ns)
        out: list[Row] = []
        for i in ids:
            r = cur.rows.get(i)
            if r is not None:
                out.append(_to_row(i, r, include_vectors=include_vectors, attrs=attrs))
        return out

    async def query_ann(
        self,
        ns: str,
        vector: Sequence[float],
        *,
        top_k: int,
        flt: Filter | None = None,
        attrs: Sequence[str] | None = None,
    ) -> list[Row]:
        cur = self._require(ns)
        if top_k < 1:
            raise _err(ErrorCode.INVALID_ARGUMENT, "top_k must be >= 1")
        if not cur.has_vector:
            raise _err(ErrorCode.INVALID_ARGUMENT, "namespace has no vectors")
        if cur.schema.dims is not None and len(vector) != cur.schema.dims:
            raise _err(ErrorCode.INVALID_ARGUMENT, "query vector length does not match namespace dims")
        _validate_filter(flt, cur.schema.attrs)
        q = [float(x) for x in vector]
        scored: list[tuple[float, str, _Stored]] = []
        for i, r in cur.rows.items():
            if r.vector is None or not _passes(flt, r, cur.schema.attrs):
                continue
            if len(r.vector) != len(q):
                raise _err(ErrorCode.INVALID_ARGUMENT, "query vector length does not match stored vectors")
            scored.append((_cosine(q, r.vector), i, r))
        scored.sort(key=lambda t: (-t[0], t[1]))  # ties by id: real tpuf tie order is unspecified
        return [_to_row(i, r, include_vectors=False, attrs=attrs, score=sc) for sc, i, r in scored[:top_k]]

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
        cur = self._require(ns)
        if top_k < 1:
            raise _err(ErrorCode.INVALID_ARGUMENT, "top_k must be >= 1")
        spec = cur.schema.attrs.get(field)
        if spec is None or spec.fts != "pre_tokenized_array":
            raise _err(ErrorCode.INVALID_ARGUMENT, "field is not a pre_tokenized_array full-text attribute")
        _validate_filter(flt, cur.schema.attrs)
        query = set(tokens)  # exact, case-sensitive; duplicates collapse
        docs = {i: _token_set(r.attrs.get(field)) for i, r in cur.rows.items()}
        docs = {i: t for i, t in docs.items() if t is not None}
        if not docs or not query:
            return []
        n_docs = len(docs)
        avg_len = sum(len(t) for t in docs.values()) / n_docs
        idf = {
            tok: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for tok in query
            if (df := sum(1 for t in docs.values() if tok in t)) > 0
        }
        scored: list[tuple[float, str, _Stored]] = []
        for i, toks in docs.items():
            r = cur.rows[i]
            if not _passes(flt, r, cur.schema.attrs):
                continue
            norm = 1.0 - _BM25_B + _BM25_B * len(toks) / avg_len
            sc = sum(w * (_BM25_K1 + 1.0) / (1.0 + _BM25_K1 * norm) for tok, w in idf.items() if tok in toks)
            if sc > 0.0:
                scored.append((sc, i, r))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [_to_row(i, r, include_vectors=False, attrs=attrs, score=sc) for sc, i, r in scored[:top_k]]

    def _require(self, ns: str) -> _Namespace:
        cur = self._ns.get(ns)
        if cur is None:
            raise _err(ErrorCode.NOT_FOUND, "namespace does not exist")
        return cur


def _to_row(
    i: str, r: _Stored, *, include_vectors: bool, attrs: Sequence[str] | None, score: float | None = None
) -> Row:
    keep = r.attrs if attrs is None else {k: v for k, v in r.attrs.items() if k in attrs}
    return Row(
        id=i,
        attrs=copy.deepcopy(keep),
        vector=list(r.vector) if include_vectors and r.vector is not None else None,
        score=score,
    )


def _passes(flt: Filter | None, r: _Stored, specs: Mapping[str, AttrSpec]) -> bool:
    return flt is None or _matches(flt, r.attrs, specs)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0.0 and nb > 0.0 else 0.0


def _token_set(value: Any) -> set[str] | None:
    if not isinstance(value, list | tuple):
        return None
    return {t for t in value if isinstance(t, str)}  # pyright: ignore[reportUnknownVariableType]


def _validate_filter(flt: Filter | None, specs: Mapping[str, AttrSpec]) -> None:
    """Row-independent checks, so a bad filter fails the same way on an empty namespace."""
    match flt:
        case None:
            return
        case And(parts):
            for p in parts:
                _validate_filter(p, specs)
        case Not(part):
            _validate_filter(part, specs)
        case Eq(attr, value):
            _check_filterable(attr, specs)
            if isinstance(value, str):
                _check_value_size(value)
        case In(attr, values):
            _check_filterable(attr, specs)
            for v in values:
                _check_value_size(v)
        case Contains(attr, value):
            _check_filterable(attr, specs)
            _check_value_size(value)
        case Glob(attr, pattern):
            _check_filterable(attr, specs)
            spec = specs.get(attr)
            if spec is not None and not spec.glob:
                raise _err(ErrorCode.INVALID_ARGUMENT, "Glob needs an attribute declared glob=True")
            _check_value_size(pattern)


def _check_value_size(value: str) -> None:
    if len(value.encode()) > _MAX_FILTER_VALUE_BYTES:
        raise _err(ErrorCode.INVALID_ARGUMENT, "filter value exceeds 4 KiB")


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """`**` crosses `/` (spike 0a-A: `aa11/**` and in-segment `**33` match deep paths); `*` and `?` stay
    inside one `/`-segment; `[..]` classes as in fnmatch."""
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
                while i < n and pattern[i] == "*":
                    i += 1
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[" and (j := pattern.find("]", i + 2)) != -1:
            body = pattern[i + 1 : j]
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append("[" + body.replace("\\", "\\\\") + "]")
            i = j
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out), re.DOTALL)


def _matches(flt: Filter, attrs: Mapping[str, Any], specs: Mapping[str, AttrSpec]) -> bool:
    match flt:
        case And(parts):
            return all(_matches(p, attrs, specs) for p in parts)
        case Not(part):
            return not _matches(part, attrs, specs)
        case Eq(attr, value):
            _check_filterable(attr, specs)
            return attr in attrs and attrs[attr] == value
        case In(attr, values):
            _check_filterable(attr, specs)
            return attr in attrs and attrs[attr] in values
        case Contains(attr, value):
            _check_filterable(attr, specs)
            spec = specs.get(attr)
            if spec is not None and spec.type != "[]string":
                raise _err(ErrorCode.INVALID_ARGUMENT, "Contains needs a []string attribute")
            got = attrs.get(attr)
            return isinstance(got, list | tuple) and value in got  # pyright: ignore[reportUnknownArgumentType]
        case Glob(attr, pattern):
            _validate_filter(flt, specs)
            got = attrs.get(attr)
            return isinstance(got, str) and _glob_regex(pattern).fullmatch(got) is not None


def _check_filterable(attr: str, specs: Mapping[str, AttrSpec]) -> None:
    spec = specs.get(attr)
    if spec is not None and not spec.filterable:
        raise _err(ErrorCode.INVALID_ARGUMENT, "attribute is not filterable")
