"""Environment and paths for the demo: `.env` loading (never printing values) and the state directory."""

import os
import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Searched in order; the first definition of a variable wins and the process environment beats every file.
DEFAULT_ENV_FILES = (
    REPO_ROOT / ".env",
)
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        if raw.lstrip().startswith("#"):
            continue
        m = _LINE.match(raw)
        if not m:
            continue
        value = m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            out.setdefault(m.group(1), value)
    return out


def load_env(extra_files: tuple[Path, ...] = (), base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The process environment overlaid on the `.env` files (first file wins). Returns a new dict; nothing is
    exported into `os.environ`."""
    merged: dict[str, str] = {}
    files = [*extra_files, *(Path(p) for p in os.environ.get("SCOUT_DEMO_ENV_FILES", "").split(":") if p)]
    for f in (*files, *DEFAULT_ENV_FILES):
        try:
            for k, v in parse_env_text(f.read_text(encoding="utf-8")).items():
                merged.setdefault(k, v)
        except OSError:
            continue
    merged.update(os.environ if base is None else base)
    return merged


def demo_home(env: Mapping[str, str] | None = None) -> Path:
    src: Mapping[str, str] | MutableMapping[str, str] = os.environ if env is None else env
    home = Path(src.get("SCOUT_DEMO_HOME") or Path.home() / ".scout-demo")
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    return home


def sanitize_name(raw: str) -> str:
    """Lowercase `[a-z0-9_]`, at most 48 chars; the namespace is `scout_dev_demo_<this>`."""
    name = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")[:48].strip("_")
    if not name:
        raise ValueError("repo name is empty after sanitising; pass --name")
    return name
