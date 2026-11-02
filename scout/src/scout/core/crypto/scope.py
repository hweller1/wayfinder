"""Glob scope compiler (design section 7, "Client-side path and BM25 computation"; FR-X4, AC-12.4, AC-12.6).

Turns a user path glob into the server-side `SearchScope` whitelist (`path_glob_enc`, `ext_tag`, `dir_tag`)
plus a client post-filter over decrypted paths. The server only ever sees HMAC/SIV ciphertext, so only
whole-segment prefixes and extension/directory tags can be expressed server-side (research amendment 17).

Exactness contract: the files selected by `server scope AND post-filter` always equal the files the glob
selects on plaintext paths. The server scope is either exactly the glob (`needs_post=False`) or a superset of
it (`needs_post=True`, `overfetch=3`, the caller must apply `post` to every decrypted hit). It is never
narrower and a glob that is outside the grammar below is rejected with `UnsupportedGlobError`, never
silently reinterpreted.

Grammar (matched against the whole repo-relative path, case-sensitive, `/` separated):
- Segment: literal text, or a pattern with `*` (any run inside one segment), `?` (one char) and `[seq]` /
  `[!seq]`. Wildcards never cross `/`. `*` also matches dotfiles.
- A segment of only `**` spans whole directories: zero or more in the middle (`a/**/b` matches `a/b`), one or
  more as the last segment (`src/**` is everything inside `src`, not `src` itself).
- `**` inside a longer segment (`**.py`, `a**b`) equals `*`.
- A segment that is exactly `*.ext` (`*.py`) needs a non-empty stem: it matches `a.py` but not a file named
  `.py`, because dotfiles have no extension (`paths.path_ext`); that is what lets `ext_tag` be exact.
- No `/` in the glob (`*.py`, `Makefile`, `test_*`): matches the basename at any depth, as `**/<glob>`.
- Trailing `/` (`tests/`): a root-anchored directory, as `tests/**`. A leading `./` is dropped.
- Rejected: empty, absolute, `..` or `.` segments, empty segments (`a//b`), backslash, NUL, leading `!`,
  braces (`*.{py,go}`: list the globs separately), unterminated or invalid `[...]`, globs over 4 KiB.

Server shapes (the 4 KiB guard applies to `path_glob_enc`; past it `dir_tag` of the same directory):
- `dir/**`, `dir/sub/*`, a literal path: `path_glob_enc` (`enc(dir)/**`, `enc(dir/sub)/*`, `enc(path)`):
  exact.
- `*.py`, `**/*.py`: `ext_tag`, exact. `dir/*.py`, `dir/**/*.py`: prefix AND `ext_tag`, exact.
- Anything else: the root-anchored literal prefix (`enc(L)/**`, or `dir_tag(L)` when too long) and/or
  `ext_tag` as a superset, plus the post-filter. A literal directory below a wildcard (`**/tests/*`) has no
  usable tag: `dir_tag` hashes the full ancestor path from the root.
- A too-long exact path or `dir/*` falls back to `dir_tag` (descendants, a superset) plus the post-filter.
- Several globs are a union; no single whitelist value can express that, so the server scope is empty and the
  post-filter applies (fewer results per page, hence `overfetch`).

Deviation from the design table: the post-filter is a small regex matcher over the grammar above, not
`pathspec`. gitignore semantics (a directory pattern also selects its contents, `#`, `!`) would not be the
same selection as the server-side glob.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import SearchScope
from scout.core.crypto import ids, paths
from scout.core.crypto.keys import Subkeys

MAX_FILTER_VALUE_BYTES = 4096  # turbopuffer docs limit on a filter value; the fake enforces it too
MAX_GLOB_BYTES = 4096  # PATH_MAX-ish; deep paths trigger the 4 KiB ciphertext guard
OVERFETCH_POST = 3

ScopeFilter = SearchScope


class UnsupportedGlobError(ScoutError):
    """The glob is outside the supported grammar and cannot be expressed exactly."""

    def __init__(self, reason: str) -> None:
        super().__init__(ErrorCode.INVALID_ARGUMENT, f"unsupported path glob: {reason}")


@dataclass(frozen=True, slots=True)
class ScopePlan:
    """`server` goes on the wire; `post(plaintext_path)` is required when `needs_post` (else always true)."""

    server: ScopeFilter
    post: Callable[[str], bool]
    overfetch: int
    needs_post: bool


@dataclass(frozen=True, slots=True)
class _Seg:
    kind: Literal["lit", "pat", "any"]
    text: str


_EXT_GLOB_RE = re.compile(r"\*\.[A-Za-z0-9_+\-]+")
_EXT_TAIL_RE = re.compile(r"\.[A-Za-z0-9_+\-]+\Z")
_ANY = _Seg("any", "**")
_STAR = _Seg("pat", "*")


def _class_end(seg: str, i: int) -> int:
    """Index of the `]` closing the class opened at `seg[i]`, or -1."""
    j = i + 1
    if j < len(seg) and seg[j] == "!":
        j += 1
    if j < len(seg) and seg[j] == "]":
        j += 1
    return seg.find("]", j)


def _segment(raw: str) -> _Seg:
    if raw in ("", ".", ".."):
        raise UnsupportedGlobError("empty, '.' or '..' path segment")
    if set(raw) == {"*"} and len(raw) >= 2:
        return _ANY
    text = re.sub(r"\*+", "*", raw)
    i = 0
    while i < len(text):
        if text[i] == "[":
            end = _class_end(text, i)
            if end == -1:
                raise UnsupportedGlobError("unterminated '[' character class")
            i = end
        i += 1
    if not any(c in text for c in "*?["):
        return _Seg("lit", text)
    return _Seg("pat", text)


def _parse(glob: str) -> list[_Seg]:
    if not glob or len(glob.encode("utf-8")) > MAX_GLOB_BYTES:
        raise UnsupportedGlobError(f"empty or longer than {MAX_GLOB_BYTES} bytes")
    if "\0" in glob or "\\" in glob:
        raise UnsupportedGlobError("NUL or backslash")
    if glob.startswith("!"):
        raise UnsupportedGlobError("negation is not supported")
    if "{" in glob or "}" in glob:
        raise UnsupportedGlobError("braces are not supported; pass the globs separately")
    g = glob
    while g.startswith("./"):
        g = g[2:]
    if g.startswith("/"):
        raise UnsupportedGlobError("absolute path")
    dir_scope = g.endswith("/")
    if dir_scope:
        g = g[:-1]
    raw = g.split("/")
    segs = [_segment(r) for r in raw]
    if dir_scope:
        segs.append(_ANY)
    collapsed: list[_Seg] = []
    for s in segs:
        if not (s is _ANY and collapsed and collapsed[-1] is _ANY):
            collapsed.append(s)
    if len(collapsed) == 1 and collapsed[0] is not _ANY:
        collapsed.insert(0, _ANY)  # no '/': basename at any depth
    if len(collapsed) >= 2 and collapsed[-2] is _ANY and collapsed[-1] == _STAR:
        collapsed.pop()  # `**/*` is everything below, same as a final `**`
    return collapsed


def _class_regex(body: str) -> str:
    neg = body.startswith("!")
    members = body[1:] if neg else body
    if "--" in members:
        raise UnsupportedGlobError("ambiguous '--' inside a character class")
    esc = "".join("\\" + c if c in "\\[]^&~|" else c for c in members)
    return f"[^/{esc}]" if neg else f"[{esc}]"


def _segment_regex(text: str) -> str:
    out: list[str] = []
    i = 0
    if _EXT_GLOB_RE.fullmatch(text):
        out.append("[^/]+")  # a stem of at least one char
        i = 1
    while i < len(text):
        c = text[i]
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            end = _class_end(text, i)
            out.append(_class_regex(text[i + 1 : end]))
            i = end
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def _matcher(segs: Sequence[_Seg]) -> Callable[[str], bool]:
    parts: list[str] = []
    for i, s in enumerate(segs):
        last = i == len(segs) - 1
        if s.kind == "any":
            parts.append("[^/]+(?:/[^/]+)*" if last else "(?:[^/]+/)*")
            continue
        parts.append(re.escape(s.text) if s.kind == "lit" else _segment_regex(s.text))
        if not last:
            parts.append("/")
    try:
        rx = re.compile("".join(parts), re.DOTALL)
    except re.error:
        raise UnsupportedGlobError("invalid '[...]' character class") from None
    return lambda path: rx.fullmatch(path) is not None


def _ext_of(seg: _Seg) -> tuple[str | None, bool]:
    """`(ext, exact)`: `.py` when the segment ends in a literal `.py` (every match then has that extension,
    since the stem is non-empty), and `exact` when the segment is exactly `*.py`."""
    if seg.kind != "pat":
        return None, False
    m = _EXT_TAIL_RE.search(seg.text)
    if m is None or len(seg.text) == len(m.group()):
        return None, False
    return m.group(), seg.text == "*" + m.group()


def _prefix(
    keys: Subkeys, tenant: str, repo_id: str, lits: list[str], *, children: bool
) -> tuple[str | None, str | None, bool]:
    """`(path_glob_enc, dir_tag, exact)` for the literal directory `lits`: `enc(L)/**` or `enc(L)/*`, else
    `dir_tag(L)` (descendants at any depth) when the value would pass 4 KiB. `exact` is False only for the
    `children` form that fell back to the descendants tag."""
    if not lits:
        return None, None, True
    tail = "/*" if children else "/**"
    value = paths.enc_path(keys, tenant, repo_id, "/".join(lits)) + tail
    if len(value.encode("ascii")) <= MAX_FILTER_VALUE_BYTES:
        return value, None, True
    return None, ids.dir_tag(keys, "/".join(lits)), not children


def _server(keys: Subkeys, tenant: str, repo_id: str, segs: list[_Seg]) -> tuple[ScopeFilter, bool]:
    lits: list[str] = []
    for s in segs:
        if s.kind != "lit":
            break
        lits.append(s.text)
    rest = segs[len(lits) :]
    ext, ext_exact = _ext_of(segs[-1])
    ext_tag = ids.ext_tag(keys, ext) if ext else None
    if not rest:  # a literal file path
        value = paths.enc_path(keys, tenant, repo_id, "/".join(lits))
        if len(value.encode("ascii")) <= MAX_FILTER_VALUE_BYTES:
            return ScopeFilter(path_glob_enc=value), True
        parent = "/".join(lits[:-1])
        return ScopeFilter(dir_tag=ids.dir_tag(keys, parent) if parent else None), False
    if rest == [_ANY]:  # dir/** (or everything)
        pg, dt, exact = _prefix(keys, tenant, repo_id, lits, children=False)
        return ScopeFilter(path_glob_enc=pg, dir_tag=dt), exact
    if rest == [_STAR] and lits:  # dir/sub/*
        pg, dt, exact = _prefix(keys, tenant, repo_id, lits, children=True)
        return ScopeFilter(path_glob_enc=pg, dir_tag=dt), exact
    if ext_exact and (rest[:-1] == [_ANY] or (len(rest) == 1 and lits)):
        # dir/*.py (children), dir/**/*.py and **/*.py (descendants): prefix AND ext_tag
        pg, dt, exact = _prefix(keys, tenant, repo_id, lits, children=len(rest) == 1)
        return ScopeFilter(path_glob_enc=pg, dir_tag=dt, ext_tag=ext_tag), exact
    pg, dt, _ = _prefix(keys, tenant, repo_id, lits, children=False)  # superset: everything under L
    return ScopeFilter(path_glob_enc=pg, dir_tag=dt, ext_tag=ext_tag), False


def _compile_one(
    glob: str, keys: Subkeys, tenant: str, repo_id: str
) -> tuple[ScopeFilter, Callable[[str], bool], bool]:
    segs = _parse(glob)
    server, exact = _server(keys, tenant, repo_id, segs)
    return server, _matcher(segs), not exact


def compile_scope(globs: Sequence[str] | str | None, keys: Subkeys, tenant: str, repo_id: str) -> ScopePlan:
    """Compile user path globs (a union when several) into a server scope and a client post-filter.

    Raises `UnsupportedGlobError` (INVALID_ARGUMENT) for a glob outside the grammar in the module docstring.
    """
    items = [globs] if isinstance(globs, str) else list(globs or [])
    distinct = list(dict.fromkeys(items))
    if not distinct:
        return ScopePlan(ScopeFilter(), lambda _p: True, 1, False)
    compiled = [_compile_one(g, keys, tenant, repo_id) for g in distinct]
    if len(compiled) == 1:
        server, post, needs_post = compiled[0]
        return ScopePlan(server, post, OVERFETCH_POST if needs_post else 1, needs_post)
    posts = [c[1] for c in compiled]
    return ScopePlan(ScopeFilter(), lambda p: any(f(p) for f in posts), OVERFETCH_POST, True)
