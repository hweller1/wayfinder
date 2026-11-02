"""BM25 identifier tokens (design section 7, "Client-side path and BM25 computation").

`identifier_tokens` runs on both sides (chunk text at index time, query text at search time), so a query
that names an identifier yields a token set that intersects the chunk's. Tokens are HMAC'd by `hmac_tokens`
and stored/queried as `pre_tokenized_array`: exact, case-sensitive strings, no stemming, no substrings.

Rules (one function, both sides):
- An identifier is `[A-Za-z_][A-Za-z0-9_]*` with optional `-` joins (kebab-case), 2..64 chars. Longer runs
  (hashes, base64, minified blobs) are skipped whole rather than cut into 64-char slices.
- Per identifier emit the lowercase *whole* word with `_` and `-` removed (`get_user_session`,
  `getUserSession`, `GetUserSession`, `get-user-session` all give `getusersession`), then its sub-tokens:
  split on `_`/`-`, camel/Pascal humps (`HTTPServer` -> http, server) and letter/digit boundaries
  (`sha256Hash` -> sha, 256, hash), lowercased.
- Tokens shorter than 2 chars and the 60-word keyword stoplist (incl. 2-letter keywords) are dropped.
- Output is deduplicated in first-occurrence order (whole before its parts), then cut at `cap`, so the
  result is deterministic. Frequency is lost by design.
"""

import hashlib
import hmac
import re

from scout.core.contracts.wire import MAX_QUERY_TOKENS, MAX_TOKENS_PER_ROW
from scout.core.crypto.keys import Subkeys

MIN_TOKEN_LEN = 2  # db, id, io, os, fs, ui are real, specific identifiers
MAX_IDENT_LEN = 64
HMAC_TOKEN_HEX = 32  # 128 bits of the HMAC-SHA256 digest
CHUNK_CAP = MAX_TOKENS_PER_ROW  # 256
QUERY_CAP = MAX_QUERY_TOKENS  # 32

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*")
_PIECE_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")

STOPWORDS = frozenset(
    [
        "if",
        "in",
        "is",
        "or",
        "fn",
        "go",
        "do",
        "as",
        "of",
        "to",
        "self",
        "this",
        "return",
        "import",
        "from",
        "def",
        "class",
        "function",
        "const",
        "var",
        "null",
        "none",
        "true",
        "false",
        "void",
        "public",
        "private",
        "static",
        "new",
        "else",
        "for",
        "while",
        "try",
        "catch",
        "except",
        "finally",
        "throw",
        "raise",
        "with",
        "and",
        "not",
        "the",
        "int",
        "str",
        "bool",
        "async",
        "await",
        "pass",
        "break",
        "continue",
        "package",
        "use",
        "impl",
        "struct",
        "type",
        "interface",
        "export",
        "default",
        "switch",
        "case",
    ]
)


def _pieces(ident: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[_-]+", ident):
        out.extend(p.lower() for p in _PIECE_RE.findall(part))
    return out


def identifier_tokens(text: str, cap: int = CHUNK_CAP) -> list[str]:
    """Normalised plaintext tokens of `text`, at most `cap` (256 for chunks, 32 for queries)."""
    if cap < 1:
        raise ValueError("cap must be >= 1")
    seen: dict[str, None] = {}
    for m in _IDENT_RE.finditer(text):
        ident = m.group(0)
        if len(ident) > MAX_IDENT_LEN:
            continue
        pieces = _pieces(ident)
        whole = "".join(pieces) if pieces else ""
        for tok in (whole, *pieces):
            if len(tok) >= MIN_TOKEN_LEN and tok not in STOPWORDS:
                seen.setdefault(tok)
        if len(seen) >= cap:
            break
    return list(seen)[:cap]


def hmac_tokens(keys: Subkeys, toks: list[str]) -> list[str]:
    """`HMAC-SHA256(K_token, t)[:32 hex]` per token; order kept, duplicates dropped."""
    out: dict[str, None] = {}
    for t in toks:
        out.setdefault(hmac.new(keys.token, t.encode("utf-8"), hashlib.sha256).hexdigest()[:HMAC_TOKEN_HEX])
    return list(out)
