"""Integrations are config and docs plus stdlib hooks: no core imports, no search logic.

AC-8.3, FR-I9.
"""

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INTEGRATIONS = REPO / "integrations"
ALLOWED_SUFFIXES = {".json", ".md", ".sh", ".py", ".toml"}
SEARCH_WORDS = ("embed", "bm25", "vector", "rerank", "tokenize")


def _files() -> list[Path]:
    skip = {"__pycache__", ".git"}
    return sorted(p for p in INTEGRATIONS.rglob("*") if p.is_file() and not skip & set(p.parts))


def test_only_config_docs_and_hooks() -> None:
    bad = [str(p.relative_to(INTEGRATIONS)) for p in _files() if p.suffix not in ALLOWED_SUFFIXES]
    assert bad == []


def _local_modules(path: Path) -> set[str]:
    """Stdlib-only helper modules shipped inside the same host directory (e.g. a plugin's `lib/`); they are
    checked by this same test as files of their own."""
    host = INTEGRATIONS / path.relative_to(INTEGRATIONS).parts[0]
    return {p.stem for p in host.rglob("*.py")}


def test_python_hooks_import_only_stdlib() -> None:
    stdlib = set(sys.stdlib_module_names)
    for path in (p for p in _files() if p.suffix == ".py"):
        local = _local_modules(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name in local and name not in ("scout",):
                    continue
                assert name.split(".")[0] in stdlib, f"{path.name} imports non-stdlib {name}"
                assert not name.startswith("scout"), f"{path.name} imports {name}"


def test_no_search_logic_files() -> None:
    for path in (p for p in _files() if p.suffix == ".py"):
        stem = path.stem.lower()
        assert not any(w in stem for w in SEARCH_WORDS), path.name
        text = path.read_text(encoding="utf-8").lower()
        assert "scout.core" not in text, path.name
