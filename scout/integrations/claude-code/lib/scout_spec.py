#!/usr/bin/env python3
# ruff: noqa: S603, S607, E501, SIM105  (fixed argv: git; no shell)
"""The retrieval spec of the scout Claude Code plugin: three layers, one resolved result.

Layers, later overrides earlier (docs/retrieval-spec.md):
  1 system defaults   built in below (`SYSTEM`): 5 results per search, skip vendored code, lockfiles and
                      minified files at query time, search opens at 80% indexed;
  2 harness defaults  `spec/harness_defaults.toml`, shipped by the plugin: instructions, the delivery toggles,
                      3 mined examples;
  3 user options      `<repo>/.scout/spec.toml` (or the file named by $SCOUT_SPEC_FILE), committed by the
                      developer. `.scoutignore` (gitignore syntax, already read by scout) stays the way to keep
                      files out of the index.

`resolve()` merges the layers, generates the index description (languages, top-level directories with file
counts, main packages, optionally top symbols) and mines examples from git history, caches both per commit
under `<git-path scout>/spec_cache/`, and writes the result to `<git-path scout>/spec_resolved.json`.
That file is the interface to the scout core: the explorer prompt, the path filters and boosts and `k` read
it. Stdlib only, Python 3.9+ (hooks run under whatever `python3` the user has), never imports scout.

Merge rules: scalars, last layer wins. `instructions`: concatenated in layer order (the user's text comes last
and wins any conflict in wording). `paths.boost` / `paths.exclude`: concatenated; a later `!pattern` removes an
earlier `pattern`. Curated `[[example]]` entries: user first, then harness; mined examples fill the rest.
An invalid value is ignored and reported under `warnings`; a spec file never breaks a session.

CLI: `python3 scout_spec.py resolve [--root DIR] [--spec FILE] [--json]` prints the resolved spec.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib as _tomllib
except ImportError:  # pragma: no cover  (3.9/3.10: the subset parser below)
    _tomllib = None

SPEC_REL = os.path.join(".scout", "spec.toml")
RESOLVED = "spec_resolved.json"
CACHE_DIR = "spec_cache"
HARNESS_FILE = Path(__file__).resolve().parent.parent / "spec" / "harness_defaults.toml"
VERSION = 1

# Layer 1. Everything a spec file may set has a value here, so `resolve` always returns a full spec.
SYSTEM: dict[str, Any] = {
    "instructions": "",
    "k": 5,
    "index_enabled": True,
    "index_symbols": False,
    "repos": [],
    "exclude": [
        "vendor/",
        "third_party/",
        "node_modules/",
        "dist/",
        "*.min.js",
        "*.min.css",
        "*.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "go.sum",
        "Cargo.lock",
        "poetry.lock",
    ],
    "boost": [],
    "boost_weight": 0.3,
    "recency_boost": 0.0,
    "mine": 0,
    "mine_max_files": 8,
    "avoid": [],
    "before": "",
    "session_context": True,
    "workflow_step": False,
    "search_first": False,
    "rewrite_explore": True,
    "report_guidance": "",
    "background": False,
    "handoff": "agent",
    "search_policy": "",
    "select": "recent",
    "pool": 40,
    "max_turns": 0,
}
ENUMS = {
    "handoff": ("agent", "verbatim", "direct"),
    "search_policy": ("deny", "warn", "off"),
    "select": ("recent", "similar"),
}
DIRECT_TOOL = "mcp__plugin_scout_scout__explore"
MUSE_DIRECT_TOOL = "mcp__plugin_scout_wayfinder_scout__explore"
# One spec, several hosts: what differs per host is only the vocabulary of its tools and agents. Everything
# above (layers, parsing, description, mining) is shared; the text and the policy below read this table.
HOSTS: dict[str, dict[str, Any]] = {
    "claude-code": {
        "agent": "scout:scout-explorer",
        "relay": "scout:super-explore",
        "direct": DIRECT_TOOL,
        "fallback": "Explore",
        "delegate_tools": ("Agent", "Task"),
        "gated_tools": ("Grep", "Glob"),
        "spawn_hint": False,
    },
    "muse-code": {  # Muse has no built-in Explore: the built-in search agent is `general-purpose`
        "agent": "scout-wayfinder/scout-explorer",
        "relay": None,  # no relay subagent in the Muse plugin: handoff = verbatim falls back to the agent
        "direct": MUSE_DIRECT_TOOL,
        "fallback": "general-purpose",
        "delegate_tools": ("subagent_spawn",),
        "gated_tools": ("search",),
        "spawn_hint": True,
    },
}
SYSTEM_FACTS = {
    "sync_minutes": 5,
    "open_at_coverage": 0.8,
    "chunks": "syntax-aware",
    "scope": "git-tracked files",
}

# Spec-file shape: table -> {key: (flat name, type)}. Top level keys: instructions, k.
SHAPE: dict[str, dict[str, tuple[str, type]]] = {
    "": {"instructions": ("instructions", str), "k": ("k", int)},
    "index": {
        "enabled": ("index_enabled", bool),
        "symbols": ("index_symbols", bool),
        "repos": ("repos", list),
    },
    "paths": {
        "boost": ("boost", list),
        "exclude": ("exclude", list),
        "boost_weight": ("boost_weight", float),
    },
    "explore": {"max_turns": ("max_turns", int)},
    "recency": {"boost": ("recency_boost", float)},
    "examples": {
        "mine": ("mine", int),
        "max_files": ("mine_max_files", int),
        "avoid": ("avoid", list),
        "before": ("before", str),
        "select": ("select", str),
        "pool": ("pool", int),
    },
    "delivery": {
        "session_context": ("session_context", bool),
        "workflow_step": ("workflow_step", bool),
        "search_first": ("search_first", bool),
        "rewrite_explore": ("rewrite_explore", bool),
        "report_guidance": ("report_guidance", str),
        "background": ("background", bool),
        "handoff": ("handoff", str),
        "search_policy": ("search_policy", str),
    },
}
LIST_KEYS = ("boost", "exclude")
MAX_EXAMPLES = 8
CODE_EXT = {
    ".py",
    ".go",
    ".rs",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".sh",
    ".kt",
    ".swift",
    ".php",
    ".scala",
}
LANG = {
    ".py": "Python", ".go": "Go", ".rs": "Rust", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".java": "Java", ".c": "C", ".h": "C", ".cc": "C++", ".cpp": "C++", ".hpp": "C++",
    ".cs": "C#", ".rb": "Ruby", ".sh": "Shell", ".kt": "Kotlin", ".swift": "Swift", ".php": "PHP", ".scala": "Scala",
}  # fmt: skip
SKIP_DIRS = {".git", "node_modules", "__pycache__", "target", "vendor", ".venv", "dist", "build"}


# --- a TOML subset, for Python without tomllib ----------------------------------------------------------
class TomlError(ValueError):
    pass


def _split_top(s: str) -> list[str]:
    out, depth, cur, quote, i = [], 0, "", "", 0
    while i < len(s):
        ch = s[i]
        if quote:
            cur += ch
            if ch == "\\" and quote == '"' and i + 1 < len(s):
                cur += s[i + 1]
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            cur += ch
        elif ch == "[":
            depth += 1
            cur += ch
        elif ch == "]":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    if cur.strip():
        out.append(cur)
    return out


def _strip_comment(line: str) -> str:
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == "\\" and quote == '"':
                continue
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def _value(tok: str) -> Any:
    tok = tok.strip()
    if tok.startswith('"'):
        try:
            return json.loads(tok)
        except ValueError:
            raise TomlError(f"bad string {tok[:30]!r}") from None
    if tok.startswith("'") and tok.endswith("'") and len(tok) >= 2:
        return tok[1:-1]
    if tok in ("true", "false"):
        return tok == "true"
    if tok.startswith("["):
        if not tok.endswith("]"):
            raise TomlError("unterminated array")
        return [_value(p) for p in _split_top(tok[1:-1]) if p.strip()]
    try:
        return int(tok.replace("_", ""))
    except ValueError:
        pass
    try:
        return float(tok.replace("_", ""))
    except ValueError:
        raise TomlError(f"bad value {tok[:30]!r}") from None


def parse_toml(text: str) -> dict[str, Any]:
    """tomllib when available, else a parser for the subset a spec uses: comments, `key = value`, `[table]`,
    `[[array of tables]]`, strings (basic, literal, multi-line `\"\"\"`), numbers, booleans, arrays."""
    if _tomllib is not None:
        try:
            return _tomllib.loads(text)
        except _tomllib.TOMLDecodeError as e:
            raise TomlError(str(e)) from None
    return _parse_subset(text)


def _parse_subset(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    cur = root
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("[["):
            name = line.strip("[] ").strip()
            arr = root.setdefault(name, [])
            cur = {}
            arr.append(cur)
            continue
        if line.startswith("["):
            name = line.strip("[] ").strip()
            cur = root.setdefault(name, {})
            continue
        key, eq, rest = line.partition("=")
        if not eq:
            raise TomlError(f"line {i}: expected key = value")
        key, rest = key.strip().strip("\"'"), rest.strip()
        if rest.startswith('"""'):
            body = rest[3:]
            parts = []
            while '"""' not in body:
                parts.append(body)
                if i >= len(lines):
                    raise TomlError("unterminated multi-line string")
                body = lines[i]
                i += 1
            parts.append(body.split('"""', 1)[0])
            s = "\n".join(parts)
            if s.startswith("\n"):
                s = s[1:]
            s = re.sub(r"\\\n\s*", "", s)
            cur[key] = s
            continue
        while rest.startswith("[") and rest.count("[") > rest.count("]") and i < len(lines):
            rest += " " + _strip_comment(lines[i]).strip()
            i += 1
        cur[key] = _value(rest)
    return root


# --- layers -----------------------------------------------------------------------------------------------
def normalize(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """One layer's parsed TOML -> flat dict of the keys it set (valid ones only) plus warnings."""
    flat: dict[str, Any] = {}
    warns: list[str] = []
    for table, keys in SHAPE.items():
        src = raw if table == "" else raw.get(table)
        if src is None:
            continue
        if not isinstance(src, dict):
            warns.append(f"[{table}] must be a table")
            continue
        for key, val in src.items():
            if table == "" and key in SHAPE or (table == "" and key == "example"):
                continue
            spec = keys.get(key)
            if spec is None:
                warns.append(f"unknown key {(table + '.' if table else '') + key}")
                continue
            name, typ = spec
            ok = isinstance(val, typ) and not (typ is int and isinstance(val, bool))
            if typ is float and isinstance(val, int) and not isinstance(val, bool):
                val, ok = float(val), True
            if typ is list:
                ok = isinstance(val, list) and all(isinstance(x, str) for x in val)
            if not ok:
                warns.append(f"{(table + '.' if table else '') + key} must be {typ.__name__}")
                continue
            if name in ENUMS and val not in ENUMS[name]:
                warns.append(
                    f"{(table + '.' if table else '') + key} must be one of {', '.join(ENUMS[name])}"
                )
                continue
            flat[name] = val
    for table in raw:
        if table not in SHAPE and table != "example" and table not in SHAPE[""]:
            warns.append(f"unknown table {table}")
    ex: list[dict[str, Any]] = []
    for item in raw.get("example", []) if isinstance(raw.get("example"), list) else []:
        q, files = (
            item.get("query") if isinstance(item, dict) else None,
            item.get("files") if isinstance(item, dict) else None,
        )
        if (
            isinstance(q, str)
            and q.strip()
            and isinstance(files, list)
            and files
            and all(isinstance(f, str) for f in files)
        ):
            ex.append({"query": q.strip(), "files": list(files)})
        else:
            warns.append("[[example]] needs query (string) and files (list of strings)")
    if ex:
        flat["examples"] = ex
    if "k" in flat:
        flat["k"] = max(1, min(30, flat["k"]))
    if "boost_weight" in flat:
        flat["boost_weight"] = max(0.0, min(2.0, flat["boost_weight"]))
    if "recency_boost" in flat:
        flat["recency_boost"] = max(0.0, min(1.0, flat["recency_boost"]))
    if "max_turns" in flat:
        flat["max_turns"] = max(0, min(200, flat["max_turns"]))
    if "pool" in flat:
        flat["pool"] = max(5, min(200, flat["pool"]))
    if "mine" in flat:
        flat["mine"] = max(0, min(MAX_EXAMPLES, flat["mine"]))
    return flat, warns


def _concat_negating(lower: list[str], upper: list[str]) -> list[str]:
    out = list(lower)
    for pat in upper:
        if pat.startswith("!") and len(pat) > 1:
            out = [p for p in out if p != pat[1:]]
        elif pat not in out:
            out.append(pat)
    return out


def merge(system: dict[str, Any], *layers: dict[str, Any]) -> dict[str, Any]:
    """Apply flat layers in order over `system` (see the module docstring for the rules)."""
    res = {k: (list(v) if isinstance(v, list) else v) for k, v in system.items()}
    res["examples"] = []
    texts: list[str] = [system.get("instructions", "")] if system.get("instructions") else []
    curated: list[list[dict[str, Any]]] = []
    for layer in layers:
        for key, val in layer.items():
            if key == "instructions":
                if val.strip():
                    texts.append(val.strip())
            elif key in LIST_KEYS:
                res[key] = _concat_negating(res[key], val)
            elif key == "examples":
                curated.append(val)
            else:
                res[key] = list(val) if isinstance(val, list) else val
    res["instructions"] = "\n\n".join(t.strip() for t in texts if t.strip())
    res["examples"] = [e for group in reversed(curated) for e in group]  # user's curated examples first
    return res


def _read_layer(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None or not path.is_file():
        return {}, []
    try:
        return normalize(parse_toml(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, TomlError) as e:
        return {}, [f"{path.name}: not readable TOML ({type(e).__name__}); ignored"]


def spec_path(root: str, env: dict[str, str]) -> Path | None:
    raw = env.get("SCOUT_SPEC_FILE")
    p = Path(raw) if raw else Path(root) / SPEC_REL
    return p if p.is_file() else None


def load_layers(root: str, env: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The merged settings (no description, no mined examples) and a record of which layers contributed."""
    if env.get("SCOUT_SPEC_DISABLE"):
        s = merge(SYSTEM)
        s.update(session_context=False, workflow_step=False, search_first=False, rewrite_explore=False)
        return s, {"harness": False, "user": None, "warnings": [], "disabled": True}
    harness, w1 = _read_layer(HARNESS_FILE)
    sp = spec_path(root, env)
    user, w2 = _read_layer(sp)
    merged = merge(SYSTEM, harness, user)
    return merged, {
        "harness": bool(harness),
        "user": str(sp) if sp else None,
        "warnings": w1 + w2,
        "disabled": False,
    }


# --- git helpers ------------------------------------------------------------------------------------------
def _git(root: str, *args: str, timeout: int = 60, inp: str | None = None) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", root, *args], capture_output=True, text=True, timeout=timeout, check=False,
            input=inp, stdin=None if inp is not None else subprocess.DEVNULL,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout if res.returncode in (0, 1) else None


def git_top_and_state(cwd: str) -> tuple[str, str] | None:
    out = _git(cwd, "rev-parse", "--show-toplevel", "--git-path", "scout", timeout=5)
    lines = (out or "").splitlines()
    if len(lines) < 2:
        return None
    state = lines[1] if os.path.isabs(lines[1]) else os.path.join(cwd, lines[1])
    return lines[0], os.path.normpath(state)


def head_commit(root: str) -> str | None:
    out = _git(root, "rev-parse", "HEAD", timeout=5)
    return out.strip() if out and out.strip() else None


def tracked_files(root: str) -> list[str]:
    out = _git(root, "ls-files", "-z", timeout=60) or ""
    return [p for p in out.split("\0") if p]


def filter_excluded(root: str, files: list[str], patterns: list[str]) -> list[str]:
    """Drop the files matched by gitignore-syntax `patterns` plus the repo's `.scoutignore`, using git's own
    matcher (`check-ignore --no-index` with a scratch excludes file). Any failure keeps all files."""
    lines = [p for p in patterns if p and not p.startswith("!")]
    ign = Path(root) / ".scoutignore"
    try:
        if ign.is_file():
            lines += ign.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        pass
    if not lines or not files:
        return files
    with tempfile.NamedTemporaryFile("w", suffix=".excludes", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        name = f.name
    try:
        out = _git(root, "-c", f"core.excludesFile={name}", "check-ignore", "--no-index", "-z", "--stdin",
                   timeout=120, inp="\0".join(files) + "\0")  # fmt: skip
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass
    if out is None:
        return files
    hit = {p for p in out.split("\0") if p}
    return [f for f in files if f not in hit]


# --- index description ----------------------------------------------------------------------------------
_DOC_EXT = {".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".rst", ".lock"}
_SYMBOL = re.compile(
    r"^\s*(?:export\s+|public\s+|pub\s+|abstract\s+|final\s+|static\s+)*(?:class|struct|interface|trait|enum|type)\s+([A-Z][A-Za-z0-9_]{2,40})",
    re.M,
)


def top_symbols(root: str, files: list[str], n: int = 10) -> list[str]:
    """Type names declared in the largest source files (cheap: at most 300 files, 64 KiB each)."""
    src = [f for f in files if Path(f).suffix in CODE_EXT]
    sized = []
    for f in src:
        try:
            sized.append((os.path.getsize(os.path.join(root, f)), f))
        except OSError:
            continue
    names: list[str] = []
    for _, f in sorted(sized, reverse=True)[:300]:
        try:
            with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
                text = fh.read(65536)
        except OSError:
            continue
        for m in _SYMBOL.finditer(text):
            if m.group(1) not in names:
                names.append(m.group(1))
        if len(names) >= n:
            break
    return names[:n]


def index_description(
    root: str, files: list[str], name: str = "", symbols: bool = False, top_pkgs: int = 8
) -> str:
    """Repo name, languages, top-level directories with file counts, largest packages, optionally top symbols."""
    files = [f for f in files if not (set(Path(f).parts[:-1]) & SKIP_DIRS)]
    src = [f for f in files if Path(f).suffix in LANG]
    langs = Counter(LANG[Path(f).suffix] for f in src)
    top = Counter(f.split("/")[0] if "/" in f else "(root)" for f in files)
    pk = Counter("/".join(Path(f).parts[:-1][:3]) for f in src if "/" in f)
    lines = [
        f"Repository index: {name or os.path.basename(os.path.realpath(root))} ({len(files)} indexed files).",
        "Languages (source files): " + (", ".join(f"{k} {v}" for k, v in langs.most_common(4)) or "none detected"),
        "Top-level directories (files): " + ", ".join(f"{k} {v}" for k, v in sorted(top.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
        "Main packages/modules (source files): " + (", ".join(f"{k} {v}" for k, v in pk.most_common(top_pkgs)) or "none"),
    ]  # fmt: skip
    if symbols:
        syms = top_symbols(root, files)
        if syms:
            lines.append("Top symbols (types in the largest source files): " + ", ".join(syms))
    return "\n".join(lines)


# --- examples mined from git history ------------------------------------------------------------------------
def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(
        path == p or fnmatch.fnmatchcase(path, p) or path.startswith(p.rstrip("/") + "/") for p in patterns
    )


def mine_examples(
    root: str,
    before: str,
    avoid: list[str] | tuple[str, ...] | set[str] = (),
    n: int = 3,
    scan: int = 400,
    max_files: int = 8,
    skip_commits: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Up to n past commits as `{sha, query, files}`: ancestors of `before` EXCLUDING `before` itself (strictly
    before it; nothing after, never a commit in `skip_commits`), non-merge, 1..max_files files, at least one
    source file, none of the files matching `avoid` (exact path, glob or directory prefix: the eval passes
    the gold and held-out files here). Most recent first; deterministic. Leak guard of the eval."""
    if n <= 0 or not before:
        return []
    out = _git(
        root, "log", "--no-merges", f"-n{scan + 1}", "--name-only", "--format=@@%H%x09%s", before, timeout=120
    )
    if out is None:
        return []
    avoid_l = list(avoid)
    commits: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for ln in out.splitlines():
        if ln.startswith("@@"):
            sha, _, subj = ln[2:].partition("\t")
            cur = {"sha": sha, "query": subj.strip(), "files": []}
            commits.append(cur)
        elif ln.strip() and cur is not None:
            cur["files"].append(ln.strip())
    res: list[dict[str, Any]] = []
    for c in commits:
        if (
            c["sha"] == before
            or c["sha"].startswith(before[:12])
            or any(c["sha"].startswith(s[:12]) for s in skip_commits if s)
        ):
            continue
        if not (1 <= len(c["files"]) <= max_files) or len(c["query"]) < 12:
            continue
        if any(_matches_any(f, avoid_l) for f in c["files"]):
            continue
        if not any(Path(f).suffix in CODE_EXT for f in c["files"]):
            continue  # docs- or config-only commits teach nothing about mapping code
        res.append({"sha": c["sha"], "query": c["query"][:140], "files": c["files"]})
        if len(res) >= n:
            break
    return res


# --- resolve ----------------------------------------------------------------------------------------------
def _cache_key(head: str, spec: dict[str, Any]) -> str:
    sig = json.dumps([head, spec["exclude"], spec["index_symbols"], spec["mine"], spec["mine_max_files"],
                      sorted(spec["avoid"]), spec["before"], spec["select"], spec["pool"]], sort_keys=True)  # fmt: skip
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


def resolve(
    root: str, env: dict[str, str] | None = None, state_dir: str | None = None, write: bool = True
) -> dict[str, Any]:
    """The full spec for this checkout: merged layers + description + examples, cached per commit."""
    env = dict(os.environ) if env is None else env
    spec, info = load_layers(root, env)
    head = head_commit(root) or ""
    if state_dir is None:
        got = git_top_and_state(root)
        state_dir = got[1] if got else None
    desc, mined, cached = "", [], False
    if head and (spec["session_context"] or write):
        cdir = os.path.join(state_dir, CACHE_DIR) if state_dir else None
        key = _cache_key(head, spec)
        cpath = os.path.join(cdir, f"{head[:12]}-{key}.json") if cdir else None
        data = _read_json(cpath) if cpath else None
        if data and data.get("version") == VERSION:
            desc, mined, cached = data["description"], data["mined"], True
        else:
            files = filter_excluded(root, tracked_files(root), spec["exclude"])
            if spec["avoid"]:  # the eval's held-out files are not described either
                files = [f for f in files if not _matches_any(f, spec["avoid"])]
            desc = index_description(root, files, symbols=spec["index_symbols"])
            before = spec["before"] or head
            n_mine = (
                spec["pool"] if spec["select"] == "similar" and spec["mine"] > 0 else spec["mine"]
            )  # similar: a pool, ranked per prompt
            mined = mine_examples(root, before, spec["avoid"], n_mine, max_files=spec["mine_max_files"])
            if cdir:
                _write_json(cpath, {"version": VERSION, "commit": head, "description": desc, "mined": mined})
    examples = [{**e, "source": "curated"} for e in spec["examples"]]
    pool: list[dict[str, Any]] = []
    for m in mined:
        if spec["select"] == "similar":
            pool.append({"query": m["query"], "files": m["files"], "source": "mined", "sha": m["sha"][:12]})
            continue
        if len(examples) >= max(MAX_EXAMPLES, spec["mine"]):
            break
        examples.append({"query": m["query"], "files": m["files"], "source": "mined", "sha": m["sha"][:12]})
    resolved = {
        "version": VERSION,
        "commit": head,
        "instructions": spec["instructions"],
        "k": spec["k"],
        "max_turns": spec["max_turns"],
        "index_enabled": spec["index_enabled"],
        "repos": spec["repos"],
        "exclude": spec["exclude"],
        "boost": spec["boost"],
        "boost_weight": spec["boost_weight"],
        "recency_boost": spec["recency_boost"],
        "description": desc,
        "examples": examples,
        "example_pool": pool,
        "example_select": spec["select"],
        "example_n": spec["mine"],
        "delivery": {
            **{
                k: spec[k]
                for k in (
                    "session_context",
                    "workflow_step",
                    "rewrite_explore",
                    "report_guidance",
                    "background",
                    "handoff",
                )
            },
            "search_policy": spec["search_policy"] or ("deny" if spec["search_first"] else "off"),
            "search_first": (spec["search_policy"] or ("deny" if spec["search_first"] else "off")) == "deny",
        },
        "system": SYSTEM_FACTS,
        "layers": {
            "system": True,
            "harness": info["harness"],
            "user": info["user"],
            "disabled": info["disabled"],
        },
        "warnings": info["warnings"]
        + (["index.repos is not supported yet; only this repository is searched"] if spec["repos"] else []),
        "cached": cached,
    }
    if write and state_dir:
        _write_json(os.path.join(state_dir, RESOLVED), resolved)
    return resolved


def _read_json(path: str | None) -> dict[str, Any] | None:
    try:
        with open(path or "", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def _write_json(path: str | None, obj: dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".spec.")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except OSError:
        pass


# --- delivery text ----------------------------------------------------------------------------------------------
def search_target(env: dict[str, str], host: str = "claude-code") -> str:
    """The subagent the workflow step points at: the host's scout explorer (default) or `$SCOUT_SPEC_TARGET`
    (the spec-only arm pins the host's built-in search agent)."""
    return env.get("SCOUT_SPEC_TARGET") or HOSTS[host]["agent"]


def apply_handoff(target: str, handoff: str, host: str = "claude-code") -> str:
    """What the parent is told to call. `agent`: the target subagent (default). `verbatim`: the thin relay
    subagent (`scout:super-explore` on Claude Code; Muse has none and keeps the agent), which returns scout's
    ranked report unedited. `direct`: scout's `explore` MCP tool called by the parent itself, no subagent. A
    built-in target (scout not ready, or the spec-only ablation) is never changed."""
    h = HOSTS[host]
    if target != h["agent"]:
        return target
    return {"verbatim": h["relay"] or target, "direct": h["direct"]}.get(handoff, target)


def _spawn_text(target: str, host: str) -> str:
    """How to start the delegate where the host has no short name for it (Muse: subagent_spawn/subagent_wait)."""
    if not HOSTS[host]["spawn_hint"]:
        return ""
    wait = "timeout_ms 300000; it takes 1-3 minutes" if target == HOSTS[host]["agent"] else "wait for it"
    return f" Start it with subagent_spawn (subagent_type `{target}`, the question as the objective), then subagent_wait ({wait})."


def workflow_text(
    target: str, guidance: str = "", host: str = "claude-code", background: bool = False
) -> str:
    if target == HOSTS[host]["direct"]:
        txt = (f"Before reading or editing code, call the `{target}` tool yourself with a description of what "
               "to find (no subagent needed); work from the files it returns.")  # fmt: skip
    else:
        txt = (f"Before reading or editing code, delegate to the `{target}` subagent with a description of what to "
               "find; work from the files it returns." + _spawn_text(target, host))  # fmt: skip
        if (
            background and HOSTS[host]["spawn_hint"]
        ):  # Muse's spawn is asynchronous: the parent may work meanwhile
            txt += (" Do not wait right away: while it runs, keep going on anything else the task needs (read files "
                    "the user named, plan the change), and call subagent_wait only when you need its list.")  # fmt: skip
    return txt + (" " + guidance.strip() if guidance.strip() else "")


def deny_text(target: str, host: str = "claude-code") -> str:
    names = (
        " and ".join(f"`{t}`" for t in HOSTS[host]["gated_tools"])
        if host != "claude-code"
        else "Grep and Glob"
    )
    verb = "is" if host != "claude-code" and len(HOSTS[host]["gated_tools"]) == 1 else "are"
    if target == HOSTS[host]["direct"]:
        return (f"Map the code first with the `{target}` tool; {names} {verb} for exact identifiers you "
                "already know.")  # fmt: skip
    return (f"Map the code first with the `{target}` subagent; {names} {verb} for exact identifiers you already "
            "know." + _spawn_text(target, host))  # fmt: skip


_STOP = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "when",
    "add",
    "fix",
    "use",
    "all",
    "not",
    "are",
    "new",
    "files",
    "file",
    "find",
    "task",
    "repository",
    "relevant",
    "report",
    "any",
    "you",
    "tools",
    "have",
    "whatever",
}


def _tokens(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t not in _STOP}


def similar_examples(prompt: str, pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The `n` pool examples closest to the prompt by IDF-weighted token overlap of (subject + file paths).
    Deterministic; ties keep pool order (most recent first). The pool was mined strictly before `before` and
    never touches `avoid`, so selection cannot leak the answer."""
    import math

    docs = [_tokens(e["query"] + " " + " ".join(e["files"])) for e in pool]
    df = Counter(t for d in docs for t in d)
    q = _tokens(prompt)
    score = [sum(math.log(1 + len(pool) / df[t]) for t in d & q) for d in docs]
    order = sorted(range(len(pool)), key=lambda i: (-score[i], i))
    return [pool[i] for i in order[:n] if score[i] > 0] or pool[:0]


def examples_text(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return ""
    return (
        "Examples of how past changes in this repository map to files (description -> files touched):\n"
        + "\n".join(f"- {e['query']} -> {', '.join(e['files'])}" for e in examples)
    )


def session_context(resolved: dict[str, Any], target: str) -> str | None:
    """SessionStart additionalContext: instructions, index description, examples, how to search."""
    if not resolved["delivery"].get("session_context"):
        return None
    parts: list[str] = []
    if resolved["instructions"]:
        parts.append("Retrieval instructions for this repository:\n" + resolved["instructions"])
    if resolved["description"]:
        parts.append(resolved["description"])
    ex = examples_text(resolved["examples"])
    if resolved["delivery"].get("report_guidance"):
        parts.append(resolved["delivery"]["report_guidance"].strip())
    if ex:
        parts.append(ex)
    if resolved["boost"]:
        parts.append("Prefer paths: " + ", ".join(resolved["boost"]) + ".")
    return "\n\n".join(parts) or None


def prompt_output(
    prompt: str, resolved: dict[str, Any], target: str, host: str = "claude-code"
) -> dict[str, Any] | None:
    """UserPromptSubmit: the workflow step (plus report guidance, plus the examples closest to the prompt when
    `examples.select = "similar"`), on the user's own prompt (not on background-task notifications)."""
    if prompt.lstrip().startswith("<task-notification>"):
        return None
    parts: list[str] = []
    if resolved["delivery"].get("workflow_step"):
        parts.append(
            workflow_text(
                target,
                resolved["delivery"].get("report_guidance", ""),
                host,
                bool(resolved["delivery"].get("background")),
            )
        )
    if resolved.get("example_select") == "similar" and resolved.get("example_pool"):
        chosen = similar_examples(prompt, resolved["example_pool"], resolved.get("example_n", 3))
        resolved["examples"] = [e for e in resolved["examples"] if e.get("source") == "curated"] + chosen
        if chosen:
            parts.append(examples_text(chosen))
    if not parts:
        return None
    return {
        "hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "\n\n".join(parts)}
    }


def pretool_decision(payload: dict[str, Any], mapped: bool, resolved: dict[str, Any], target: str, scout_prefix: str,
                     host: str = "claude-code", sub: bool | None = None) -> tuple[dict[str, Any] | None, str, bool]:  # fmt: skip
    """`(hook stdout | None, event label, mapped_after)` for the search-first policy. Pure.

    The main agent's first delegation, or its own call of a scout MCP tool, maps the session. Until then its
    search tools (Claude Code: Grep/Glob; Muse: search) are denied. Calls made inside a subagent are never
    gated: Claude Code marks them with `agent_id` in the payload, Muse hosts pass `sub` (thread id != session id)."""
    h = HOSTS[host]
    name = payload.get("tool_name")
    if sub is None:
        sub = bool(payload.get("agent_id"))
    if name in h["delegate_tools"]:
        return None, ("delegation_sub" if sub else "delegation"), mapped or not sub
    if isinstance(name, str) and name.startswith(scout_prefix):
        return None, ("search_sub" if sub else "search_main"), mapped or not sub
    if name in h["gated_tools"] and not sub:
        policy = resolved["delivery"].get("search_policy") or (
            "deny" if resolved["delivery"].get("search_first") else "off"
        )
        if mapped or policy == "off":
            return None, "grep_allowed", mapped
        if policy == "warn":
            warn = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": deny_text(target, host),
                }
            }
            return warn, "grep_warned", mapped
        deny = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": deny_text(target, host)}}  # fmt: skip
        return deny, "grep_denied", mapped
    return None, "other", mapped


# --- CLI -----------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="resolve the scout retrieval spec of a checkout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--root", default=os.getcwd())
    r.add_argument("--spec", help="spec file (default <root>/.scout/spec.toml)")
    r.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    env = dict(os.environ)
    if a.spec:
        env["SCOUT_SPEC_FILE"] = a.spec
    res = resolve(a.root, env, write=not a.no_write)
    sys.stdout.write(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
