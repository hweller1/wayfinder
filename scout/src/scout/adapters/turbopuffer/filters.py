"""`Filter` AST -> turbopuffer filter expressions (design section 3 VectorStore, research section 2.1).

Syntax (SDK 3.0.0): `("attr", "Eq", v)`, `("attr", "In", [..])`, `("attr", "Glob", pattern)`,
`("attr", "Contains", v)` for a `[]string` attribute, `("And", [f, ..])`, `("Not", f)`. Values over 4 KiB are
refused client-side (docs limit; spike 0a-A saw none up to 1 MiB, the design keeps the guard so FakeStore
and this adapter agree). No vendor type appears here: the output is plain tuples and lists.
"""

from typing import Any

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.store import And, Contains, Eq, Filter, Glob, In, Not

__all__ = ["MAX_FILTER_VALUE_BYTES", "to_tpuf_filter"]

MAX_FILTER_VALUE_BYTES = 4096


def _check(value: str) -> str:
    if len(value.encode()) > MAX_FILTER_VALUE_BYTES:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "filter value exceeds 4 KiB")
    return value


def to_tpuf_filter(flt: Filter) -> tuple[Any, ...]:
    match flt:
        case Eq(attr, value):
            if isinstance(value, str):
                _check(value)
            return (attr, "Eq", value)
        case In(attr, values):
            return (attr, "In", [_check(v) for v in values])
        case Glob(attr, pattern):
            return (attr, "Glob", _check(pattern))
        case Contains(attr, value):
            return (attr, "Contains", _check(value))
        case And(parts):
            return ("And", [to_tpuf_filter(p) for p in parts])
        case Not(part):
            return ("Not", to_tpuf_filter(part))
