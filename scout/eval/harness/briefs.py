"""Briefs and path extraction. The same brief goes to every arm; paths are validated against `git ls-files`."""

import hashlib
import re
from collections.abc import Collection, Iterable

SUFFIX = "End with a JSON list of the repo-relative files you consider most relevant, best first."
TOP_K = 12
_TOKEN = re.compile(r"[A-Za-z0-9_@.+~\-/\\]+")
_LINES = re.compile(r":\d+(?:[-:,]\d+)*$")


def build_brief(question: str) -> str:
    return f"{question.strip()}\n\n{SUFFIX}"


def brief_sha(brief: str) -> str:
    return hashlib.sha256(brief.encode()).hexdigest()[:16]


def _clean(tok: str, prefixes: Iterable[str]) -> str:
    tok = tok.strip(".,;:()[]{}<>'\"`*")
    for p in prefixes:
        if p and tok.startswith(p):
            tok = tok[len(p) :]
    tok = _LINES.sub("", tok)
    tok = tok.removeprefix("./").lstrip("/")
    return tok.strip(".,;:")


def extract_paths(
    text: str, repo_files: Collection[str], *, root_prefixes: Iterable[str] = (), limit: int | None = None
) -> list[str]:
    """Repo files mentioned in `text`, ranked by first mention, deduplicated. A token counts only when it is
    exactly a tracked path after stripping a checkout-root prefix, `./` and a `:start-end` suffix."""
    prefixes = sorted({p.rstrip("/") + "/" for p in root_prefixes if p}, key=len, reverse=True)
    files = set(repo_files)
    seen: dict[str, int] = {}
    for m in _TOKEN.finditer(text):
        tok = _clean(m.group(0).replace("\\", "/"), prefixes)
        if tok in files and tok not in seen:
            seen[tok] = m.start()
    ranked = sorted(seen, key=lambda p: seen[p])
    return ranked if limit is None else ranked[:limit]


def recall_at(paths: list[str], gold: Collection[str], k: int = TOP_K) -> float:
    g = set(gold)
    return len(g & set(paths[:k])) / len(g) if g else float("nan")
