"""Auto-apply the directory-name marker (tests/unit -> `unit`, ...) to every collected test."""

from pathlib import Path

import pytest

MARKER_DIRS = {"unit", "contract", "conformance", "security", "live", "eval", "spike"}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    tests_root = Path(__file__).parent.resolve()
    for item in items:
        try:
            parts = Path(str(item.path)).resolve().relative_to(tests_root).parts
        except ValueError:
            continue
        if len(parts) > 1 and parts[0] in MARKER_DIRS:
            item.add_marker(getattr(pytest.mark, parts[0]))
