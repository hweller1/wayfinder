"""Shared helpers for the day-1 data-prep scripts. Stdlib only, so they run in a bare Modal CPU image.

Scoring definitions mirror ../benchmark-repo/scripts/score_v4.py (scored_recall, leaky, MODEL_FAULT_KEYS) and
src/asa/scorers/retrieval.py (document_metrics recall = |retrieved & gold| / |gold|, both deduped). They are copied, not
imported, because the sandbox has no access to the benchmark source tree.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

MODEL_FAULT_KEYS = (
    "regex parse error",
    "PathEscapeError",
    "ToolArgumentError",
    "FileNotFoundError",
    "ripgrep failed",
    "not a file under the corpus root",
)
_SLOT = re.compile(r"(?:Where in this codebase is|What behaviour does) the (.+?) functionality")

LANG_BY_EXT = {
    "py": "Python", "pyi": "Python",
    "go": "Go",
    "rs": "Rust",
    "ts": "TypeScript/JS", "tsx": "TypeScript/JS", "js": "TypeScript/JS", "jsx": "TypeScript/JS", "mjs": "TypeScript/JS",
    "cjs": "TypeScript/JS", "svelte": "TypeScript/JS", "vue": "TypeScript/JS",
    "c": "C/C++", "h": "C/C++", "cc": "C/C++", "cpp": "C/C++", "hpp": "C/C++", "mm": "C/C++", "m": "C/C++",
    "cs": "C#",
    "java": "Java/Kotlin", "kt": "Java/Kotlin", "scala": "Java/Kotlin",
    "rb": "Ruby",
    "swift": "Swift",
}


def ext_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def task_language(gold_files: list[str]) -> str:
    counts: dict[str, int] = {}
    for g in gold_files:
        lang = LANG_BY_EXT.get(ext_of(g), "Other")
        counts[lang] = counts.get(lang, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if counts else "Other"


def load_public_tasks(shards_dir: Path) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for f in sorted(glob.glob(str(shards_dir / "pub_s*.json"))):
        for t in json.load(open(f))["tasks"]:
            tasks[t["task_id"]] = t
    return tasks


def load_rows(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            out[str(row["sample_id"])] = row
    return out


def scored_recall(row: dict | None) -> float | None:
    """score_v4.scored_recall: file-level recall; None when unscored (harness error not authored by the model)."""
    rec = (row or {}).get("record") or {}
    if not rec:
        return None
    if rec.get("status") == "harness_error" and not any(k in (rec.get("harness_reason") or "") for k in MODEL_FAULT_KEYS):
        return None
    gold = list(dict.fromkeys(rec.get("gold_docids") or []))
    if not gold:
        return None
    got = set(rec.get("retrieved_docids") or [])
    return sum(1 for g in gold if g in got) / len(gold)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def leaky(row: dict) -> bool:
    """score_v4.leaky: the generated question names the gold path (template-slot leak)."""
    question = (row.get("record") or {}).get("question") or row.get("task_text") or ""
    m = _SLOT.search(question)
    if not m:
        return False
    slot = _norm(m.group(1))
    return bool(slot) and any(slot in _norm(g) for g in (row.get("gold_files") or []))


# ---- path heuristics (PRD section 5, "Unambiguous") -------------------------------------------------------------
_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "testdata", "testing", "fixtures", "__mocks__", "mocks",
              "e2e", "integration_tests", "unit_tests", "testutil", "testutils", "test_utils", "snapshots"}
_TEST_NAME = re.compile(
    r"(^test_.*\.py$)|(_test\.(py|go|rs|c|cc|cpp|rb|exs?)$)|(\.(test|spec)\.[a-z]+$)|(^conftest\.py$)|"
    r"(tests?\.(rs|go|java|kt|cs|swift)$)|(Tests?\.(java|kt|cs|swift|scala)$)|(_spec\.rb$)|(^test[A-Z_].*\.(js|ts)$)"
)
_GEN_DIRS = {"vendor", "node_modules", "dist", "build", "generated", "__generated__", "gen", ".next", "target"}
_GEN_NAME = re.compile(
    r"(\.pb\.(go|cc|h)$)|(_pb2(_grpc)?\.pyi?$)|(\.pb\.gw\.go$)|(\.generated\.[a-z]+$)|(_generated\.[a-z]+$)|"
    r"(^zz_generated)|(\.gen\.go$)|(_gen\.go$)|(\.min\.(js|css)$)|(\.snap$)|(\.lock$)|(^package-lock\.json$)|"
    r"(^go\.sum$)|(^yarn\.lock$)|(\.d\.ts$)|(_string\.go$)|(\.designer\.cs$)|(\.g\.cs$)|(\.freezed\.dart$)"
)


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return any(p.lower() in _TEST_DIRS for p in parts[:-1]) or bool(_TEST_NAME.search(name))


def is_generated_path(path: str) -> bool:
    parts = path.split("/")
    return any(p.lower() in _GEN_DIRS for p in parts[:-1]) or bool(_GEN_NAME.search(parts[-1]))
