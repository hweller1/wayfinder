"""E2E pilot coder: Opus 5.5 (medium) tool-use loop whose tools all execute inside a per-run Modal Sandbox.

  uv run --env-file .env --with modal python scripts/e2e/run_coder.py --wording issue --task mashumaro \
      --conditions none,oracle,sonnet_grep,sonnet_index,35b_grep,35b_index --seeds 0 [--workers 8]

Round 2 (2026-10-23): the coder CANNOT search. Tools: read_file (only allowlisted paths: files the search subagent surfaced, the gold
source files for oracle, or files the coder wrote this run; any other read returns BLOCKED_MSG and is logged), write_file, apply_edit,
run_tests (no args; the recorded test command restricted to units that exist in the sandbox, so held-out tests stay absent).

Per run: sandbox holds the repo at base_commit with the held-out test files REMOVED (asserted); retrieval condition text is
injected into the first user message; the coder edits files (mirrored to a host worktree so the diff is exact); the diff
(test paths excluded) is scored by score.py's Modal path (held-out tests applied, network blocked).
Records: experiments/e2e/runs_r2/<wording>/<condition>/<slug>_s<seed>.json + <slug>_s<seed>.transcript.json (resume-safe).

Interpretation notes (stated in the records): "400k input tokens" = max prompt size of any single call (billed cumulative
input is recorded separately); "content the subagent saw" = the line spans it read/was shown (retrieved_spans), else the head
of the file, capped at 5 files x 512 whitespace tokens like the benchmark's result format.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE  # re-pointed to wayfinder/eval/e2e
sys.path.insert(0, str(HERE))

import gate as g  # noqa: E402
import score as sc  # noqa: E402

import os  # noqa: E402
RUNS = Path(os.environ["E2E_RUNS_DIR"]) if os.environ.get("E2E_RUNS_DIR") else ROOT / "runs"      # wayfinder e2e track: runs/<wording>/<condition>/<slug>_s<seed>.json (round-3 records stay in benchmark-repo)
SUB = Path(os.environ["E2E_SEARCH_DIR"]) if os.environ.get("E2E_SEARCH_DIR") else ROOT / "search"      # search/<wording>/<condition>/<slug>_s<seed>.json, written by search_app.py / glimmer_app.py
CODER_MODEL_ID = "opus-5.5-medium"
PRICE_KEY = "claude-opus-5-5"
CONDITIONS = ("none", "oracle", "glimmer_base", "a1_35b_scout", "spark_teacher", "wayfinder_v2a", "wayfinder_v2b", "wayfinder_v3", "v3", "a1", "sonnet", "gpt")
SUB_ARM = {"glimmer_base": "scout", "a1_35b_scout": "scout", "spark_teacher": "scout", "wayfinder_v2a": "scout", "wayfinder_v2b": "scout", "wayfinder_v3": "scout",
           "v3": "scout", "a1": "scout", "sonnet": "scout", "gpt": "scout"}   # final E2E (the end-to-end final study plan, not in this release) adds v3, a1, sonnet, gpt      # search conditions (scout explore loop, grep-only)
SUB_DIRNAME = {c: c for c in SUB_ARM}  # dir under SUB/<wording>
BLOCKED_MSG = "not available: search is handled by the search assistant"
MAX_TOOL_CALLS = 60
MAX_PROMPT_TOKENS = 400_000
MAX_FILES, MAX_FILE_TOKENS = 5, 512
ORACLE_FILE_CHARS = 200_000  # ~50k tokens per handed-over file; the rest stays reachable via read_file line ranges
OUT_CHARS = 12_000
SNAPSHOT_EXCLUDES = (".git", "__pycache__", "*.pyc", ".pytest_cache", ".e2e_junit.xml", "node_modules", ".mypy_cache", ".ruff_cache")
CMD_TIMEOUT = 600
MODAL_CPU_HR_USD = 0.0000131 * 4 * 3600 + 0.00000222 * 8 * 3600   # 4 cores + 8 GiB list rates (approx), USD/hour

SYSTEM = """You are a software engineer working in a repository checked out at /work. Implement the change described in the \
task so that the project's maintainers' (hidden) tests for it would pass. Match the public names and signatures the task \
describes. You cannot search the repository: there is no grep, no directory listing and no shell. A separate search \
assistant may have supplied files and notes below; work from those. read_file only opens files that were provided to you or \
that you have written yourself, and run_tests runs the project's existing tests (no arguments, no network). Use write_file \
or apply_edit to make focused changes to non-test source files; you do not need to write tests. When the implementation is \
complete, stop calling tools and reply with a short summary."""

TOOLS = [
    dict(name="read_file", description="Read a file you were given or have written (optionally a 1-based inclusive line range). Other paths are not available.",
         parameters=dict(type="object", properties=dict(path={"type": "string"}, start_line={"type": "integer"},
                                                         end_line={"type": "integer"}), required=["path"])),
    dict(name="write_file", description="Create or overwrite a file with the given content.",
         parameters=dict(type="object", properties=dict(path={"type": "string"}, content={"type": "string"}),
                         required=["path", "content"])),
    dict(name="apply_edit", description="Replace one exact occurrence of old_string with new_string in a file.",
         parameters=dict(type="object", properties=dict(path={"type": "string"}, old_string={"type": "string"},
                                                         new_string={"type": "string"}), required=["path", "old_string", "new_string"])),
    dict(name="run_tests", description="Build the whole project (compile/type check), then run the project's existing recorded tests for this task (no arguments). Returns truncated output.",
         parameters=dict(type="object", properties={})),
]


class _Tool:
    def __init__(self, spec):
        self.spec = spec

    def json_schema(self):
        return self.spec


# --------------------------------------------------------------------------- pure logic (unit-tested)

def slug_of(task_id: str) -> str:
    """Repo name, plus the commit prefix when a repo has 2+ gated tasks (they overwrote each other's records under the bare
    repo name: celestia-app, temporal, gh-aw-mcpg, python-genai; found 2026-10-25). Round-3 repos with 2+ tasks are in
    _SHARED_REPOS too. _LEGACY_BARE = the round-2 pilot tasks whose records are already stored under the bare repo name;
    they keep it (so round-2 screen/run records stay valid) and only the OTHER tasks of their repo get the prefix."""
    repo = task_id.split("__")[-2].lower()
    if os.environ.get("E2E_SLUG_ALL"):   # v3 task set: many repos carry several tasks; always repo-commit6 (records for the 7 round-3 tasks are copied under these names)
        return f"{repo}-{task_id.split('__')[-1][:6]}"
    if task_id.split("__")[-1][:6] in _LEGACY_BARE and repo in _SHARED_REPOS - _ROUND2_SHARED:
        return repo
    return f"{repo}-{task_id.split('__')[-1][:6]}" if repo in _SHARED_REPOS else repo


_ROUND2_SHARED = {"celestia-app", "temporal", "gh-aw-mcpg", "python-genai"}
# Round 3 added every repo with 2+ gated tasks (tests/test_e2e_run_coder.py::test_slug_of_unique_across_gated checks this).
_SHARED_REPOS = _ROUND2_SHARED | {"mashumaro", "claude-agent-sdk-python", "reqwest", "github-mcp-server", "linkerd2", "dbtrail", "helm",
                                  "code-graph-rag", "agno", "mcp-client-for-ollama", "openbiliclaw", "cplt", "cert-manager",
                                  "office365-rest-python-client"}
_LEGACY_BARE = {"930843", "79c4bd", "141c37", "57670a", "5d3c7f", "d661bf", "c2ca1b", "febc32", "55ba86", "b8df01", "85bc90", "a623c4", "97c261", "4dfbaa", "10ed5a", "9d61b5", "5e2ef2"}


def load_all_tasks() -> dict:
    d = g.load_tasks()
    if os.environ.get("E2E_TASKS_MANIFEST"):   # final E2E v4 carries one v3 task whose public record is in the benchmark-repo dataset
        p = HERE.parents[2] / "benchmark-repo/datasets/artifacts/public/tasks_1660.json"
        if p.exists():
            for t in json.load(open(p))["tasks"]:
                d.setdefault(t["task_id"], t)
    return d


def pilot_tasks(slugs=None) -> list[dict]:
    if os.environ.get("E2E_TASKS_MANIFEST"):   # final E2E: tasks straight from manifest file(s) (comma separated), optional id filter E2E_TASK_IDS (json list)
        keep = set(json.load(open(os.environ["E2E_TASK_IDS"]))) if os.environ.get("E2E_TASK_IDS") else None
        out = []
        for mf in os.environ["E2E_TASKS_MANIFEST"].split(","):
            for t in json.load(open(mf))["tasks"]:
                if keep is not None and t["task_id"] not in keep:
                    continue
                out.append(dict(task_id=t["task_id"], slug=t["slug"], issue_text=t["issue_text"], subject=t.get("subject", ""),
                                repo_full=t["repo"], base_commit=t["base_commit"], gold_files=list(t["gold_files"])))
        if slugs:
            out = [t for t in out if any(s.lower() in t["slug"] or s.lower() in t["task_id"].lower() for s in slugs)]
        return out
    tasks = load_all_tasks()
    iss = {x["task_id"]: x for x in json.load(open(g.EXP / "issues.json"))["issues"] if x.get("ok")}
    gated = {r["task_id"] for r in json.load(open(g.EXP / "gate_results.json"))["results"] if r.get("gated") and not r.get("dropped")}
    out = []
    for tid in iss:
        if tid not in gated:
            continue
        t = tasks[tid]
        out.append(dict(task_id=tid, slug=slug_of(tid), issue_text=iss[tid]["issue_text"], subject=iss[tid]["subject"],
                        repo_full=t["repo_full"], base_commit=t["base_commit"], gold_files=list(t["gold_files"])))
    if slugs:
        out = [t for t in out if any(s.lower() in t["slug"] or s.lower() in t["task_id"].lower() for s in slugs)]
    return out


def task_text_for(task: dict, wording: str) -> str:
    return task["issue_text"] if wording == "issue" else task["subject"]


def diff_without_tests(patch: str):
    """(patch with test-path sections dropped, files touched). Same rule the scorer applies."""
    return sc.filter_patch(patch)


def head_tokens(text: str, n: int = MAX_FILE_TOKENS) -> str:
    toks = text.split()
    return text if len(toks) <= n else " ".join(toks[:n])


def excerpt(text: str, spans: list[dict] | None, n: int = MAX_FILE_TOKENS) -> str:
    """The lines the subagent saw (union of its spans) capped at n whitespace tokens; head of file if no span."""
    lines = text.splitlines()
    picked = []
    for s in spans or []:
        a, b = s.get("start_line"), s.get("end_line")
        if a and b:
            picked.append("\n".join(lines[max(a - 1, 0):b]))
    return head_tokens("\n...\n".join(picked) if picked else text, n)


def recall_of(retrieved: list[str], gold: list[str], heldout: list[str]) -> float | None:
    gold = [p for p in gold if p not in set(heldout)]
    if not gold:
        return None
    return len(set(retrieved) & set(gold)) / len(gold)


def gold_source_files(gold: list[str], heldout: list[str], new_at_base: dict | None = None) -> tuple[list[str], list[str]]:
    """(retrievable gold source files, gold source files new at the parent commit). Test paths and held-out files are excluded;
    files that do not exist at the parent cannot be retrieved, so they are reported separately and never count as misses."""
    held = set(heldout)
    src = [p for p in dict.fromkeys(gold) if p not in held and not g.is_test_path(p)]
    new = [p for p in src if (new_at_base or {}).get(p) is True]
    return [p for p in src if p not in set(new)], new


def first_seen(retrieved: list[str]) -> list[str]:
    return list(dict.fromkeys(retrieved))


def retrieval_metrics(retrieved: list[str], gold_src: list[str]) -> dict:
    """Metrics of the subagent's output vs the gold source files. `retrieved` is in the order the subagent first surfaced each
    path; duplicates collapse to their first occurrence. Binary relevance. All None when there is no retrievable gold."""
    import math
    seen = first_seen(retrieved)
    gold = set(gold_src)
    if not gold:
        return dict(recall=None, recall_at_5=None, any_hit_at_5=None, ndcg_at_10=None, all_gold_found=None, precision=None,
                    n_gold_src=0, n_retrieved=len(seen))
    hits = [p for p in seen if p in gold]
    top5, top10 = seen[:5], seen[:10]
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(top10) if p in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dict(recall=len(hits) / len(gold), recall_at_5=len(set(top5) & gold) / len(gold), any_hit_at_5=bool(set(top5) & gold),
                ndcg_at_10=dcg / ideal, all_gold_found=len(hits) == len(gold),
                precision=(len(hits) / len(seen)) if seen else 0.0, n_gold_src=len(gold), n_retrieved=len(seen))


class Access:
    """Per-run read allowlist. A path is readable iff it was surfaced to the coder (allowlist) or the coder wrote it this run."""

    def __init__(self, allowlist):
        self.allow = {safe_rel(p) for p in allowlist}
        self.written: set[str] = set()
        self.blocked: list[str] = []
        self.edits_outside_allowlist: list[str] = []

    def can_read(self, rel: str) -> bool:
        return rel in self.allow or rel in self.written

    def read_or_block(self, rel: str) -> str | None:
        """None if allowed; BLOCKED_MSG (and the attempt is logged) otherwise."""
        if self.can_read(rel):
            return None
        self.blocked.append(rel)
        return BLOCKED_MSG

    def note_write(self, rel: str) -> None:
        if rel not in self.allow and rel not in self.written:
            self.edits_outside_allowlist.append(rel)
        self.written.add(rel)

    def summary(self) -> dict:
        return dict(allowlist_size=len(self.allow), blocked_reads=dict(count=len(self.blocked), paths=list(dict.fromkeys(self.blocked))),
                    edits_outside_allowlist=list(dict.fromkeys(self.edits_outside_allowlist)))


def test_command(lang: str, units: list[str], exists: set[str]) -> str | None:
    """The recorded test command restricted to units present in the sandbox (held-out tests are absent). None if nothing to run.
    exists: set of unit paths found: Python test files, Go package dirs, Rust integration-test files tests/<name>.rs."""
    if lang == "Python":
        fs = [u for u in units if u in exists]
        return ("python -m pytest -p no:cacheprovider -q " + " ".join(shlex.quote(f) for f in fs)) if fs else None
    if lang == "Go":
        ds = [u for u in units if u in exists]
        return ("go test -count=1 " + " ".join(shlex.quote(("./" + d) if d else ".") for d in ds)) if ds else None
    if lang == "Rust":
        bs = [u for u in units if f"tests/{u}.rs" in exists]
        return ("cargo test " + " ".join(f"--test {shlex.quote(b)}" for b in bs) + " -- --test-threads=4") if bs else None
    if lang == "TypeScript":
        import ts_support
        fs = [u for u in units if u in exists]
        if not fs:
            return None
        # one command per run dir; no report file (the coder reads the console). Run dirs are resolved in the sandbox from package.json.
        return ("for f in " + " ".join(shlex.quote(f) for f in fs) + "; do d=$(dirname \"$f\"); while [ \"$d\" != . ] && [ ! -f \"$d/vitest.config.ts\" ] "
                "&& [ ! -f \"$d/jest.config.js\" ] && [ ! -f \"$d/jest.config.ts\" ] && [ ! -f \"$d/vitest.config.mts\" ] && [ ! -f \"$d/jest.config.mjs\" ] && [ ! -f \"$d/vitest.config.js\" ] "
                "&& ! grep -qE '\"(vitest|jest)\"' \"$d/package.json\" 2>/dev/null; do d=$(dirname \"$d\"); done; "
                "(cd \"$d\" && rel=${f#$d/}; [ \"$d\" = . ] && rel=$f; "
                "if ls vitest.config.* >/dev/null 2>&1 || grep -q '\"vitest\"' package.json 2>/dev/null; then npx --no-install vitest run \"$rel\"; "
                "else npx --no-install jest --ci --forceExit \"$rel\"; fi); done")
    return None


def build_command(lang: str) -> str:
    """Whole-project compile check the coder always gets from run_tests; exit code is the compiler's, not tail's."""
    cmd = {"Go": "go build ./... && go vet ./...",
           "Rust": "cargo test --no-run --offline --lib --tests",  # not --all-targets: benches/examples pull deps never fetched
           "TypeScript": "true",  # no whole-project build: tests run through vitest/jest transpilers; a tsc pass is too slow and config-dependent
           "Python": "python -m compileall -q -x '(^|/)(\\.git|\\.venv|node_modules)/' ."}.get(lang)
    return f"set -o pipefail; ({cmd}) 2>&1 | tail -200" if cmd else "true"


def unit_probe_paths(lang: str, units: list[str]) -> list[str]:
    return [f"tests/{u}.rs" for u in units] if lang == "Rust" else list(units)


def format_files(files: list[tuple[str, str]]) -> str:
    return "\n".join(f"<file path=\"{p}\">\n{c}\n</file>" for p, c in files)


def condition_block(condition: str, *, sub: dict | None = None, read=None, gold: list[str] | None = None,
                    heldout: list[str] | None = None) -> str:
    """The retrieval text injected after the task. `read(path)` returns the file text at base_commit."""
    heldout = set(heldout or [])
    if condition == "none":
        return ""
    if condition in SUB_ARM:
        assert sub is not None
        seen = [p for p in sub.get("retrieved_docids", []) if p not in heldout][:MAX_FILES]
        spans: dict[str, list] = {}
        for s in sub.get("retrieved_spans", []):
            spans.setdefault(s.get("path"), []).append(s)
        files = []
        for p in seen:
            try:
                files.append((p, excerpt(read(p), spans.get(p))))
            except Exception:  # noqa: BLE001 - file new at a later commit etc.
                continue
        return ("A search assistant explored the repository for this task. Its answer:\n"
                f"{(sub.get('answer') or '(no answer)').strip()}\n\nFiles it surfaced (excerpts, max {MAX_FILES} files):\n"
                + format_files(files))
    if condition == "oracle":
        files, new = [], []
        for p in (gold or []):
            if p in heldout:
                continue
            try:
                text = read(p)
            except FileNotFoundError:  # created by the reference change: name it, there is nothing to show
                new.append(p)
                continue
            if len(text) > ORACLE_FILE_CHARS:  # generated bundles (e.g. centrifugo's 7M-token statik.go) overflow 1M
                text = (text[:ORACLE_FILE_CHARS] + f"\n... [truncated: {len(text):,} chars total; "
                        "use read_file with a line range for the rest]")
            files.append((p, text))
        note = ("\n\nThe reference change also creates these new files: " + ", ".join(new)) if new else ""
        return ("These are the files that the reference change modifies (contents at the base commit; files over "
                f"{ORACLE_FILE_CHARS:,} characters are truncated):\n" + format_files(files) + note)
    raise ValueError(condition)


# claude-opus-5-5 USD per 1M: input, cache read, cache write, output (benchmark-repo scripts/score_frontier.py RATES; list price,
# not confirmed against the the LLM gateway catalog, same as round 3).
OPUS_RATES = (4.00, 0.20, 5.00, 20.00)


def usd_for(usage: dict) -> float:
    i, cr, cw, o = OPUS_RATES
    cached, written = usage.get("cached_input_tokens", 0), usage.get("cache_write_tokens", 0)
    fresh = max(usage.get("input_tokens", 0) - cached - written, 0)
    return (fresh * i + cached * cr + written * cw + usage.get("output_tokens", 0) * o) / 1e6


def modal_cpu_usd(seconds: float) -> float:
    return seconds * MODAL_CPU_HR_USD / 3600


def safe_rel(path: str) -> str:
    """Normalise a model-supplied path to a repo-relative one; reject escapes."""
    p = path.strip()
    if p.startswith("/work/"):
        p = p[len("/work/"):]
    elif p == "/work":
        p = "."
    p = p.lstrip("./") if p.startswith("./") else p
    parts = Path(p).parts
    if p.startswith("/") or ".." in parts:
        raise ValueError(f"path escapes the repository: {path}")
    return p or "."


def assert_no_heldout(sandbox_listing: str, heldout: list[str]) -> None:
    """sandbox_listing: output of `test -e` probes, one 'path:EXISTS|ABSENT' per line."""
    bad = [ln for ln in sandbox_listing.splitlines() if ln.endswith(":EXISTS")]
    if bad:
        raise AssertionError(f"held-out test files present in the coder sandbox: {bad}")
    seen = {ln.rsplit(":", 1)[0] for ln in sandbox_listing.splitlines()}
    miss = [p for p in heldout if p not in seen]
    if miss:
        raise AssertionError(f"held-out leak probe did not cover: {miss}")


def truncate(s: str, n: int = OUT_CHARS) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n...[{len(s) - n} chars truncated]...\n" + s[-n // 2:]


# --------------------------------------------------------------------------- sandbox-backed run

_SB_LOCK = threading.Lock()      # Modal image building / ModalSandbox.use mutate shared state
_SCORE_LOCK = threading.Lock()   # score.py shares gate.EXEC / current_env / worktree names
_SHARED: dict = {}


def shared_sandbox():
    with _SB_LOCK:
        if "sbx" not in _SHARED:
            import modal_sandbox as ms
            _SHARED["sbx"] = ms.ModalSandbox()
            _SHARED["img"] = {}
        return _SHARED["sbx"]


def image_for(env: dict):
    sbx = shared_sandbox()
    with _SB_LOCK:
        tid = env["task_id"]
        if tid not in _SHARED["img"]:
            sbx.use(env)
            _SHARED["img"][tid] = sbx.images[tid].run_commands("chmod 1777 /tmp")
        return _SHARED["img"][tid]


class CoderSandbox:
    """One Modal Sandbox (network blocked) holding the repo; exec() with timing."""

    def __init__(self, env: dict, tarball: bytes):
        import modal
        import modal_sandbox as ms
        self.ms, self.env = ms, env
        sbx = shared_sandbox()
        self.key = ms.task_key(env)
        self.lang = env["language"]
        self.t0 = time.time()
        self.sb = modal.Sandbox.create(app=sbx.app, image=image_for(env), block_network=True, workdir=ms.WORK,
                                       volumes={ms.CACHE: sbx.vol.read_only()}, cpu=ms.CPU, memory=ms.MEM_MB, timeout=3600)
        self.sb.exec("mkdir", "-p", ms.WORK).wait()
        up = self.sb.exec("tar", "xzf", "-", "-C", ms.WORK, text=False)
        ms.write_chunked(up.stdin, tarball)
        up.wait()
        self._rust_ready = False

    def exec(self, argv: list[str], timeout: int = 120, env: dict | None = None):
        p = self.sb.exec(*argv, workdir=self.ms.WORK, timeout=timeout, env=env or {})
        out, err = p.stdout.read(), p.stderr.read()
        return p.wait(), out, err

    def leak_probe(self, heldout: list[str]) -> str:
        script = "; ".join(f"[ -e {shlex.quote(p)} ] && echo {shlex.quote(p)}:EXISTS || echo {shlex.quote(p)}:ABSENT" for p in heldout)
        rc, out, err = self.exec(["bash", "-c", script or "true"])
        return out

    def cmd_env(self) -> dict:
        e = self.ms.command_env(None, self.ms.WORK, self.lang, self.key, network=False)
        if self.lang == "Python":
            e["PYTHONPATH"] = ":".join(f"{self.ms.WORK}/{p}" if p != "." else self.ms.WORK
                                       for p in (self.env.get("config") or {}).get("pythonpath", ["."]))
        if self.lang == "TypeScript":   # node_modules (one tar on the read-only volume) is restored into /work before the first command
            e.update(HOME="/tmp")
        if self.lang == "Rust":
            e.update(CARGO_HOME="/tmp/cargo", CARGO_TARGET_DIR="/tmp/target")
        elif self.lang == "Go":
            e["GOCACHE"] = "/tmp/gocache"
        return e

    def run_command(self, command: str) -> tuple[int, str]:
        if self.lang == "TypeScript" and not getattr(self, "_node_ready", False):
            import ts_support
            self.exec(["bash", "-c", ts_support.restore_script(self.key, self.ms.CACHE)], timeout=600)
            self._node_ready = True
        if self.lang == "Rust" and not self._rust_ready:
            self.exec(["bash", "-c", "mkdir -p /tmp/cargo/registry && cp -a /cache/cargo/registry/index /cache/cargo/registry/cache "
                       "/cache/cargo/registry/src /tmp/cargo/registry/"], timeout=300)
            self._rust_ready = True
        argv = self.ms.wrap_for_lock(["bash", "-c", command], self.lang, self.key)
        try:
            rc, out, err = self.exec(argv, timeout=CMD_TIMEOUT, env=self.cmd_env())
        except Exception as ex:  # noqa: BLE001
            return 124, f"command failed or timed out: {type(ex).__name__}: {str(ex)[:200]}"
        return rc, truncate(out + (("\n[stderr]\n" + err) if err.strip() else ""))

    def run_recorded_tests(self) -> str:
        """The task's recorded test units that exist in the sandbox (held-out tests are absent), nothing else; no free command."""
        units = self.env.get("run_units", [])
        probes = unit_probe_paths(self.lang, units)
        rc, out, err = self.exec(["bash", "-c", "; ".join(f"[ -e {shlex.quote(p)} ] && echo {shlex.quote(p)}" for p in probes) or "true"])
        # Always compile first. Without it, a coder with no shell cannot see compile errors whenever the task's own
        # test files are held out (round-2 screen, 2026-10-24: redb/reqwest oracle failed on unseen build errors
        # that round 1, with a shell, fixed).
        brc, bout = self.run_command(build_command(self.lang))
        parts = [f"build: exit code {brc}\n{bout}"]
        cmd = test_command(self.lang, units, set(out.split()))
        if cmd is None:
            parts.append("tests: none of this task's recorded test units are present in the repository")
        elif brc == 0:
            rc, o = self.run_command(cmd)
            parts.append(f"tests: exit code {rc}\n{o}")
        return "\n\n".join(parts)

    def snapshot(self) -> bytes:
        """tar.gz of /work as the coder left it (junk excluded); the diff is computed from THIS, so shell edits count."""
        ex = " ".join(f"--exclude={shlex.quote(x)}" for x in SNAPSHOT_EXCLUDES)
        p = self.sb.exec("bash", "-c", f"tar czf - {ex} -C {self.ms.WORK} .", workdir=self.ms.WORK, text=False, timeout=300)
        data = p.stdout.read()
        if p.wait() != 0 or not data:
            raise RuntimeError("snapshot of the sandbox failed: " + str(p.stderr.read())[:200])
        return data

    def close(self):
        try:
            self.sb.terminate()
            self.sb.wait()
        except Exception:  # noqa: BLE001
            pass

    @property
    def seconds(self) -> float:
        return time.time() - self.t0


def execute_tool(name: str, args: dict, box: CoderSandbox, host_wd: Path, access: Access) -> str:
    try:
        if name == "read_file":
            rel = safe_rel(args["path"])
            blocked = access.read_or_block(rel)
            if blocked:
                return blocked
            sl, el = args.get("start_line"), args.get("end_line")
            cmd = ["bash", "-c", f"cat -n -- {shlex.quote(rel)}" if not sl else
                   f"sed -n '{int(sl)},{int(el or 10**9)}p' -- {shlex.quote(rel)} | cat -n | sed 's/^ *\\([0-9]*\\)/\\1/'"]
            rc, out, err = box.exec(cmd)
            return truncate(out if rc == 0 else f"error: {err.strip() or 'cannot read'}")
        if name in ("write_file", "apply_edit"):
            rel = safe_rel(args["path"])
            if name == "write_file":
                content = args["content"]
            else:
                rc, cur, err = box.exec(["cat", "--", rel])
                if rc != 0:
                    return f"error: cannot read {rel}: {err.strip()}"
                if cur.count(args["old_string"]) != 1:
                    return f"error: old_string matches {cur.count(args['old_string'])} times in {rel}; it must match exactly once"
                content = cur.replace(args["old_string"], args["new_string"], 1)
            up = box.sb.exec("bash", "-c", f"mkdir -p \"$(dirname {shlex.quote(rel)})\" && cat > {shlex.quote(rel)}",
                             workdir=box.ms.WORK, text=False)
            up.stdin.write(content.encode())
            up.stdin.write_eof()
            up.stdin.drain()
            if up.wait() != 0:
                return "error: write failed"
            access.note_write(rel)
            return f"ok: wrote {rel} ({len(content)} chars)"
        if name == "run_tests":
            return box.run_recorded_tests()
        return f"error: unknown tool {name}"
    except Exception as ex:  # noqa: BLE001
        return f"error: {type(ex).__name__}: {str(ex)[:300]}"


def diff_from_snapshot(snapshot: bytes, repo, wd: Path, base_commit: str) -> str:
    """Replace the host worktree's files with the sandbox snapshot, then diff against base_commit (adds, edits, deletes)."""
    import io
    import os
    import tarfile
    for child in wd.iterdir():
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    def keep_abs_symlinks(member, dest):
        # Repos can track symlinks to absolute paths (helm's testdata/.../null -> /dev/null); the "data" filter
        # rejects them. Recreate such a link verbatim (never followed, only diffed) and filter everything else.
        if member.issym() and member.linkname.startswith("/"):
            target = wd / member.name
            if (wd.resolve() in target.resolve(strict=False).parents) or target.parent.resolve() == wd.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or target.exists():
                    target.unlink()
                os.symlink(member.linkname, target)
            return None
        return tarfile.data_filter(member, dest)

    with tarfile.open(fileobj=io.BytesIO(snapshot), mode="r:gz") as tf:
        tf.extractall(wd, filter=keep_abs_symlinks)
    g.git(wd, "add", "-A")
    out = g.git(wd, "diff", "--cached", "--binary", base_commit).stdout
    g.git(wd, "reset", "-q")
    return out


def tar_without(wd: Path, heldout: list[str]) -> bytes:
    import modal_sandbox as ms
    for p in heldout:
        (wd / p).unlink(missing_ok=True)
    return ms.make_tarball(str(wd))


async def coder_loop(backend, messages: list[dict], box: CoderSandbox, host_wd: Path, transcript: list[dict], access: Access):
    tools = [_Tool(s) for s in TOOLS]
    calls, stop, max_prompt = 0, "done", 0
    while True:
        reply = await backend(messages, tools)
        max_prompt = max(max_prompt, backend._last_prompt)
        transcript.append(dict(role="assistant", text=reply.content, tool_calls=[(c.name, c.arguments) for c in reply.tool_calls],
                               finish=reply.finish_reason))
        if not reply.tool_calls:
            if reply.finish_reason == "length":
                stop = "output_length"
            break
        messages.append(dict(role="assistant", content=reply.content or "",
                             tool_calls=[dict(id=c.id, type="function", function=dict(name=c.name, arguments=c.arguments))
                                         for c in reply.tool_calls]))
        for c in reply.tool_calls:
            calls += 1
            args = c.parsed() or {}
            out = await asyncio.to_thread(execute_tool, c.name, args, box, host_wd, access)
            messages.append(dict(role="tool", tool_call_id=c.id, content=out))
            transcript.append(dict(role="tool", name=c.name, args=args, result=out[:4000]))
        if calls >= MAX_TOOL_CALLS:
            stop = "tool_cap"
            break
        if max_prompt >= MAX_PROMPT_TOKENS:
            stop = "input_cap"
            break
    return calls, stop, max_prompt


# --------------------------------------------------------------------------- one run

def allowlist_for(condition: str, retrieved: list[str], sub: dict | None, gold_src: list[str]) -> list[str]:
    """none: empty. oracle: the gold source files that exist at the parent. subagent conditions: EVERY path the subagent surfaced
    (retrieved_docids, not only the excerpted top-5), so the coder may open anything the search assistant returned."""
    if condition == "none":
        return []
    if condition == "oracle":
        return list(gold_src)
    return list(dict.fromkeys(retrieved))


def run_path(wording, condition, slug, seed) -> Path:
    return RUNS / wording / condition / f"{slug}_s{seed}.json"


def sub_record(wording, condition, slug, seed) -> dict | None:
    p = SUB / wording / SUB_DIRNAME[condition] / f"{slug}_s{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


FINAL_LEDGER = Path(os.environ["E2E_FINAL_LEDGER"]) if os.environ.get("E2E_FINAL_LEDGER") else None
FINAL_CAP = float(os.environ.get("E2E_FINAL_CAP", "350"))
MIN_FREE_GB = 8.0


def ledger_total() -> float:
    if not FINAL_LEDGER or not FINAL_LEDGER.exists():
        return 0.0
    return sum(float(json.loads(l).get("usd") or 0) for l in FINAL_LEDGER.read_text().splitlines() if l.strip())


def ledger_add(row: dict) -> None:
    import fcntl
    if not FINAL_LEDGER:
        return
    with open(FINAL_LEDGER, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(row) + "\n"); f.flush(); fcntl.flock(f, fcntl.LOCK_UN)


def guard_before_run() -> None:
    """the LLM gateway cap and disk preflight. Pauses (a watcher loop, not a debugging sleep) while free disk < 8 GB."""
    if FINAL_LEDGER and ledger_total() + 1.0 > FINAL_CAP:
        raise RuntimeError(f"the LLM gateway cap: ledger ${ledger_total():.2f} + $1 margin exceeds ${FINAL_CAP}")
    while shutil.disk_usage(g.DATA).free / 1e9 < MIN_FREE_GB:
        time.sleep(60)


def run_one(task: dict, wording: str, condition: str, seed: int, score=True) -> dict:
    from asa.gateway_backend import GatewayBackend
    guard_before_run()
    tid, slug = task["task_id"], task["slug"]
    tasks = load_all_tasks()
    t = tasks[tid]
    env = sc.load_env(tid)
    heldout = [p for _, p in env["heldout_test_entries"]]
    rec: dict = dict(task_id=tid, slug=slug, wording=wording, condition=condition, seed=seed, coder=CODER_MODEL_ID,
                     task_text=task_text_for(task, wording), started=time.strftime("%FT%TZ", time.gmtime()))
    t_start = time.time()
    repo, wd = g.make_worktree(t, f"coder_{slug}_{wording}_{condition}_s{seed}")
    box = None
    transcript: list[dict] = []
    try:
        def read(p):
            r = subprocess.run(["git", "-C", str(repo), "show", f"{t['base_commit']}:{p}"], capture_output=True)
            if r.returncode:
                raise FileNotFoundError(p)
            return r.stdout.decode("utf-8", "replace")

        gold_src, new_at_parent = gold_source_files(task["gold_files"], heldout, t.get("gold_new_at_base"))
        sub = None
        retrieved: list[str] = []
        if condition in SUB_ARM:
            sub = sub_record(wording, condition, slug, seed)
            if sub is None:
                raise RuntimeError(f"no subagent record for {wording}/{condition}/{slug}_s{seed}; run modal_subagent.py first")
            retrieved = [p for p in sub["retrieved_docids"] if p not in set(heldout)]
            rec["files_retrieved"] = retrieved
            rec["subagent"] = {k: sub.get(k) for k in ("status", "stop_reason", "prompt_tokens", "completion_tokens", "turns",
                                                       "wall_clock_s", "gpu_seconds", "gpu_usd_repo_basis", "gpu_usd_modal_list",
                                                       "tool_call_counts", "t_first_gold_s", "usd", "grove_usage",
                                                       "model", "search_usd_basis", "budget_exhausted", "n_reported", "recall_reported",
                                                       "recall_seen_k5", "all_gold_found", "all_gold_reported")}
            rec["recall_vs_gold"] = recall_of(retrieved, task["gold_files"], heldout)
        block = condition_block(condition, sub=sub, read=read, gold=task["gold_files"], heldout=heldout)
        if condition == "oracle":
            retrieved = [p for p in task["gold_files"] if p not in set(heldout) and not g.is_test_path(p)]
            rec["files_retrieved"] = retrieved
            rec["recall_vs_gold"] = recall_of(retrieved, task["gold_files"], heldout)
        allow = allowlist_for(condition, retrieved, sub, gold_src)
        access = Access(allow)
        m = retrieval_metrics(retrieved, gold_src)
        if condition == "oracle":
            m.update(recall=1.0, recall_at_5=1.0, any_hit_at_5=True, ndcg_at_10=1.0, all_gold_found=True, precision=1.0)
        elif condition == "none":
            m.update(recall=0.0, recall_at_5=0.0, any_hit_at_5=False, ndcg_at_10=0.0, all_gold_found=False, precision=0.0)
        m.update(gold_src_files=gold_src, gold_new_at_parent=new_at_parent,
                 subagent_t_first_gold_s=(sub or {}).get("t_first_gold_s"),
                 subagent_usd=(sub or {}).get("usd") if sub else None)
        rec["retrieval"] = m
        box = CoderSandbox(env, tar_without(wd, heldout))
        probe = box.leak_probe(heldout)
        assert_no_heldout(probe, heldout)
        rec["heldout_absent_asserted"] = heldout
        user = f"Task:\n{rec['task_text']}\n" + (f"\n{block}\n" if block else "")
        messages = [dict(role="system", content=SYSTEM), dict(role="user", content=user)]
        transcript.append(dict(role="user", text=user))
        backend = GatewayBackend(CODER_MODEL_ID)
        calls, stop, max_prompt = asyncio.run(coder_loop(backend, messages, box, wd, transcript, access))
        rec.update(access.summary())
        usage = backend.usage.as_dict()
        rec.update(tool_calls=calls, stop_reason=stop, max_prompt_tokens=max_prompt, cap_hit=stop in ("tool_cap", "input_cap"),
                   coder_usage=usage, coder_usd=round(usd_for(usage), 4))
        ledger_add(dict(at=time.strftime("%FT%TZ", time.gmtime()), task_id=tid, usd=rec["coder_usd"], purpose="coder", condition=condition, seed=seed,
                        in_tokens=usage.get("input_tokens"), out_tokens=usage.get("output_tokens")))
        raw = diff_from_snapshot(box.snapshot(), repo, wd, t["base_commit"])
        patch, touched = diff_without_tests(raw)
        rec["files_touched"] = touched
        rec["files_touched_in_gold"] = sorted(set(touched) & set(task["gold_files"]))
        rec["diff_bytes"] = len(patch)
        (run_path(wording, condition, slug, seed).with_suffix(".patch")).parent.mkdir(parents=True, exist_ok=True)
        run_path(wording, condition, slug, seed).with_suffix(".patch").write_text(patch)
        rec["sandbox_seconds"] = round(box.seconds, 1)
        box.close()
        box = None
        if score:
            with _SCORE_LOCK:
                if not sc.MODAL:
                    sc.enable_modal()
                t_sc = time.time()
                if patch.strip():
                    res = sc.score_patch(tid, patch)
                else:
                    res = dict(**{"pass": False}, new_tests_passed=0, new_tests_total=len(env["fail_to_pass"]), regressions=None,
                               note="empty diff: not executed (selfcheck shows the empty patch fails)")
                rec["score_seconds"] = round(time.time() - t_sc, 1)
            rec.update({"pass": res["pass"], "new_tests_passed": res["new_tests_passed"], "new_tests_total": res["new_tests_total"],
                        "regressions": res.get("regressions"), "regression_total": res.get("regression_total"),
                        "score_note": res.get("note", "")})
        rec["modal_cpu_usd"] = round(modal_cpu_usd(rec.get("sandbox_seconds", 0) + rec.get("score_seconds", 0)), 4)
        rec["subagent_usd"] = (rec.get("subagent") or {}).get("gpu_usd_modal_list") or (rec.get("subagent") or {}).get("usd") or 0.0
        rec["total_usd"] = round(rec.get("coder_usd", 0) + rec["modal_cpu_usd"] + rec["subagent_usd"], 4)
        rec["error"] = None
    except Exception as ex:  # noqa: BLE001
        rec["error"] = f"{type(ex).__name__}: {str(ex)[:500]}"
        rec.setdefault("stop_reason", "error")
        if box is not None:
            rec["sandbox_seconds"] = round(box.seconds, 1)
            box.close()
    finally:
        rec["wall_s"] = round(time.time() - t_start, 1)
        try:
            g.drop_worktree(repo, wd)
        except Exception:  # noqa: BLE001
            pass
    p = run_path(wording, condition, slug, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.with_suffix(".transcript.json").write_text(json.dumps(transcript, indent=1))
    if rec["error"] is None:    # a failed run leaves no record, so a rerun retries it
        p.write_text(json.dumps(rec, indent=1))
    else:
        p.with_suffix(".error.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wording", default="issue")
    ap.add_argument("--task", action="append")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args()
    jobs = []
    for w in a.wording.split(","):
        for t in pilot_tasks(a.task):
            for c in a.conditions.split(","):
                for s in [int(x) for x in a.seeds.split(",")]:
                    if not run_path(w, c, t["slug"], s).exists():
                        jobs.append((t, w, c, s))
    print(f"{len(jobs)} runs pending", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=min(a.workers, 8)) as ex:
        futs = {ex.submit(run_one, t, w, c, s, not a.no_score): (t["slug"], w, c, s) for t, w, c, s in jobs}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {futs[f]} pass={r.get('pass')} usd={r.get('total_usd')} stop={r.get('stop_reason')} "
                  f"err={r.get('error')}", flush=True)


if __name__ == "__main__":
    main()
