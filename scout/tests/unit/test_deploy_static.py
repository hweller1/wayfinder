"""Task 8.5: static checks of `deploy/`. FR-G4, AC-9.4, FR-G2.

No Modal call is made: the module is read as text and compiled. `modal` is a deploy-only import, the only
Modal secrets are `scout-gateway` and `scout-llm`, and `asa-run-creds` never appears under `deploy/`.
"""

import ast
import py_compile
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"
SRC = REPO / "src" / "scout"
GATEWAY = DEPLOY / "modal" / "gateway.py"
ALLOWED_SECRETS = {"scout-gateway", "scout-llm", "scout-llm-v3", "scout-llm-a20"}


def _text_files(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]


def test_no_asa_run_creds_anywhere_under_deploy() -> None:
    """AC-9.4."""
    files = _text_files(DEPLOY)
    assert files
    for p in files:
        assert "asa-run-creds" not in p.read_text(encoding="utf-8", errors="ignore"), p


def test_modal_is_imported_only_under_deploy() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            if any(n == "modal" or n.startswith("modal.") for n in names):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == []
    assert re.search(r"^import modal$", GATEWAY.read_text(), re.M)


def test_secret_names_are_the_gateway_llm_and_model_secrets() -> None:
    text = GATEWAY.read_text()
    used = set(re.findall(r'modal\.Secret\.from_name\(\s*"([^"]+)"', text))
    assert used == ALLOWED_SECRETS
    for p in _text_files(DEPLOY):
        assert set(re.findall(r'Secret\.from_name\(\s*"([^"]+)"', p.read_text())) <= ALLOWED_SECRETS


def test_gateway_module_compiles_and_declares_the_app(tmp_path: Path) -> None:
    py_compile.compile(str(GATEWAY), cfile=str(tmp_path / "gateway.pyc"), doraise=True)
    tree = ast.parse(GATEWAY.read_text())
    text = GATEWAY.read_text()
    assert "modal.App(APP_NAME)" in text
    assert 'os.environ.get("SCOUT_GATEWAY_APP", "scout-gateway")' in text  # default app name is unchanged
    assert "def issue_key(" in text
    assert "@app.local_entrypoint()" in text
    assert "@modal.asgi_app()" in text
    assert "modal.Dict.from_name(CACHE_DICT, create_if_missing=True)" in text
    assert 'CACHE_DICT = "scout-gateway-cache"' in text
    assert {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)} >= {
        "api",
        "issue_key",
    }


def _decorator_kwargs(fn_name: str) -> dict[str, object]:
    tree = ast.parse(GATEWAY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "function"
                ):
                    return {
                        k.arg: ast.literal_eval(k.value)
                        for k in dec.keywords
                        if k.arg and isinstance(k.value, ast.Constant)
                    }
    pytest.fail(f"no @app.function on {fn_name}")


def test_api_function_resources_and_scale_to_zero() -> None:
    kw = _decorator_kwargs("api")
    assert kw["min_containers"] == 0  # the idle gateway costs nothing (owner decision for the live deploy)
    assert kw["max_containers"] == 3
    assert (kw["cpu"], kw["memory"], kw["timeout"]) == (1, 1024, 900)
    assert "gpu" not in kw  # CPU only
    assert "@modal.concurrent(max_inputs=50)" in GATEWAY.read_text()


def test_deploy_never_prints_or_logs_secret_values() -> None:
    text = GATEWAY.read_text()
    assert "os.environ[" not in text.replace("dict(os.environ)", "")
    assert "print(" not in text  # only the issued key is written, once, via sys.stdout in issue_key
    assert text.count("sys.stdout.write") == 1
