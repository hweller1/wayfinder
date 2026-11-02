"""Pluggability gate: no host names in src/scout, `mcp` only under surfaces/mcp, no core in integrations.

AC-7.8, AC-5.1, FR-P2, FR-I9.

The `Agent|Task` check looks only at string literals of core and surfaces, and skips docstrings
(a docstring may need to name the word); the other literals are checked over raw file text.
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "scout"
INTEGRATIONS = REPO / "integrations"

FORBIDDEN_TEXT = ("plugin_super-explore", "super-explore:", "CLAUDE_")
AGENT_TASK = re.compile(r"\b(Agent|Task)\b")
AGENT_TASK_DIRS = ("core", "surfaces")
MCP_ALLOWED = (("surfaces", "mcp"), ("demo", "mcp_server.py"))  # the demo lane has one MCP entry file


def _py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py")) if root.is_dir() else []


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def scan_src(src: Path) -> list[str]:
    out: list[str] = []
    for path in _py_files(src):
        rel = path.relative_to(src)
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN_TEXT:
            if needle in text:
                out.append(f"{rel}: forbidden string {needle!r}")
        tree = ast.parse(text)
        if rel.parts and rel.parts[0] in AGENT_TASK_DIRS:
            skip = _docstring_nodes(tree)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in skip
                    and AGENT_TASK.search(node.value)
                ):
                    out.append(f"{rel}:{node.lineno}: Agent/Task in string literal")
        if not any(rel.parts[: len(a)] == a for a in MCP_ALLOWED):
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    if name == "mcp" or name.startswith("mcp."):
                        out.append(f"{rel}:{node.lineno}: imports mcp outside surfaces/mcp")
    return out


def scan_integrations(root: Path) -> list[str]:
    out: list[str] = []
    if not root.is_dir():
        return out
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "scout.core" in text:
            out.append(f"{path.relative_to(root)}: references scout.core")
    return out


def test_src_has_no_host_names_and_mcp_is_confined() -> None:
    assert scan_src(SRC) == []


def test_integrations_do_not_reference_core() -> None:
    assert scan_integrations(INTEGRATIONS) == []


def test_scanner_detects_violations(tmp_path: Path) -> None:
    """The gate itself must fail when a forbidden pattern is present."""
    src = tmp_path / "scout"
    (src / "core").mkdir(parents=True)
    (src / "adapters").mkdir()
    (src / "surfaces" / "mcp").mkdir(parents=True)
    (src / "core" / "a.py").write_text('X = "plugin_super-explore"\n')
    (src / "core" / "b.py").write_text('Y = "dispatch an Agent"\n')
    (src / "core" / "c.py").write_text('import os\nZ = os.environ["CLAUDE_CONFIG_DIR"]\n')
    (src / "adapters" / "d.py").write_text("import mcp.types\n")
    (src / "core" / "e.py").write_text('"""An Agent docstring."""\n')
    (src / "surfaces" / "mcp" / "ok.py").write_text("import mcp\n")
    found = "\n".join(scan_src(src))
    assert "a.py: forbidden string 'plugin_super-explore'" in found
    assert "b.py:1: Agent/Task" in found
    assert "c.py: forbidden string 'CLAUDE_'" in found
    assert "d.py:1: imports mcp outside" in found
    assert "e.py" not in found
    assert "ok.py" not in found

    integ = tmp_path / "integrations" / "host"
    integ.mkdir(parents=True)
    (integ / "x.sh").write_text("from scout.core import app\n")
    assert scan_integrations(tmp_path / "integrations") == ["host/x.sh: references scout.core"]
