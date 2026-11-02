"""E2E pilot validity gate.

For one mined commit (base_commit..source_commit) split the diff into held-out test changes and the
non-test change the agent must write, identify the new/changed test cases, and prove with real runs that
  (a) at parent + held-out tests the targeted tests FAIL (or error), and
  (b) at source_commit they PASS, in < 5 min.
Only tasks passing both are GATED. Pure stdlib; no LLM calls.

Definitions (documented choices):
  touched tests  = test functions whose line span intersects a line added/changed in the held-out diff
  targeted (F2P) = touched tests that PASS at source_commit and do NOT pass at parent+held-out
  regression set = tests in the touched test files that pass at the parent (original tests) AND at
                   source_commit, minus targeted. Limited to the touched test files / packages / binaries
                   to keep scoring fast; the rest of the repo is not run.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # wayfinder/eval/e2e (copied from benchmark-repo scripts/e2e; ROOT re-pointed)
DATA = ROOT / "data" / "e2e"
EXP = Path(os.environ["E2E_EXP_DIR"]) if os.environ.get("E2E_EXP_DIR") else ROOT / "experiments" / "e2e"   # v3 task set: E2E_EXP_DIR=eval/e2e/experiments/e2e_v3 (default = the 17-task round-3 set)
_DS = Path(os.environ["E2E_DATASETS_DIR"]) if os.environ.get("E2E_DATASETS_DIR") else ROOT / "datasets/artifacts/public"   # v3: full public task file
TASKS = _DS / "tasks_1660.json"
PROV = _DS / "repo_provenance_2447.json"
TIME_LIMIT_S = 300

# --------------------------------------------------------------------------- classifier / splitter

_TEST_DIRS = {"test", "tests", "testing", "__tests__", "testdata", "e2e", "spec", "specs"}
_TEST_FILE = re.compile(
    r"(^|/)(test_[^/]*\.py|[^/]*_tests?\.py|conftest\.py|[^/]*_test\.go|[^/]*_tests?\.rs|[^/]*\.(test|spec)\.[jt]sx?)$"
)


def is_test_path(path: str) -> bool:
    """True for files that hold tests or test support, by common repo conventions.

    Directory rule: any path component in {test,tests,testing,__tests__,testdata,e2e,spec(s)}.
    File rule: test_*.py, *_test.py, conftest.py, *_test.go, *_test(s).rs, *.test|spec.{js,ts,jsx,tsx}.
    Deliberately NOT matched: src/testing-utils-like names (component must equal a test dir name), and
    docs (`*.md`) under test dirs are still held out (they are in a test dir).
    """
    p = path.replace("\\", "/").lstrip("./") if path.startswith("./") else path.replace("\\", "/")
    parts = p.split("/")
    if any(c in _TEST_DIRS for c in parts[:-1]):
        return True
    return bool(_TEST_FILE.search(p))


def split_changes(entries):
    """entries: list of (status, path) from `git diff --name-status`. Returns (tests, nontests) as lists of
    (status, path). Renames/copies (R100\told\tnew) must be passed already collapsed to (status, new_path)."""
    tests, non = [], []
    for st, p in entries:
        (tests if is_test_path(p) else non).append((st, p))
    return tests, non


def parse_name_status(text: str):
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        st = parts[0][0]
        out.append((st, parts[-1]))
        if st in "RC":  # rename: also record the removed old path as D
            out.append(("D", parts[1])) if st == "R" else None
    return out


def parse_added_lines(diff_text: str):
    """unified diff (-U0 ok) -> {path: set(new-side line numbers that were added/changed)}."""
    res: dict[str, set[int]] = {}
    cur = None
    new_ln = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            cur = line[4:].strip()
            cur = None if cur == "/dev/null" else (cur[2:] if cur.startswith("b/") else cur)
            if cur is not None:
                res.setdefault(cur, set())
        elif line.startswith("@@"):
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            new_ln = int(m.group(1)) if m else 0
        elif cur is not None and line.startswith("+") and not line.startswith("+++"):
            res[cur].add(new_ln)
            new_ln += 1
        elif cur is not None and line.startswith(" "):
            new_ln += 1
    return res


# --------------------------------------------------------------------------- test function spans

def py_test_funcs(text: str):
    """[(qualname, start, end)] for test functions/methods (ast)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []

    def visit(node, prefix):
        for ch in getattr(node, "body", []):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) and ch.name.startswith("test"):
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                out.append((prefix + ch.name, start, ch.end_lineno))
            elif isinstance(ch, ast.ClassDef):
                visit(ch, prefix + ch.name + ".")

    visit(tree, "")
    return out


def _brace_spans(text: str, header: re.Pattern, attr_prefix: str):
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        m = header.match(ln)
        if not m:
            continue
        s = i
        while s > 0 and (lines[s - 1].lstrip().startswith(attr_prefix) or lines[s - 1].lstrip().startswith("//")):
            s -= 1
        e = i
        while e < len(lines) and lines[e] != "}":
            e += 1
        out.append((m.group(1), s + 1, e + 1))
    return out


def go_test_funcs(text: str):
    return _brace_spans(text, re.compile(r"^func (Test\w+)\("), "//go:")


def rs_test_funcs(text: str):
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*(?:pub )?(?:async )?fn (\w+)\(", ln)
        if not m:
            continue
        j = i
        is_test = False
        while j > 0 and lines[j - 1].strip().startswith("#["):
            j -= 1
            if re.search(r"#\[(tokio::)?test", lines[j]):
                is_test = True
        if not is_test:
            continue
        e = i
        indent = len(ln) - len(ln.lstrip())
        while e < len(lines) and not (lines[e].startswith(" " * indent + "}") and len(lines[e]) - len(lines[e].lstrip()) == indent):
            e += 1
        out.append((m.group(1), j + 1, e + 1))
    return out


def touched_functions(lang: str, text: str, added: set[int]):
    funcs = {"Python": py_test_funcs, "Go": go_test_funcs, "Rust": rs_test_funcs}[lang](text)
    return sorted({q for q, s, e in funcs if any(s <= n <= e for n in added)})


# --------------------------------------------------------------------------- shell helpers

def sh(cmd, cwd=None, env=None, timeout=1800, check=False):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
                       shell=isinstance(cmd, str))
    p.elapsed = time.time() - t0
    if check and p.returncode:
        raise RuntimeError(f"{cmd} -> {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p


EXEC = None   # set by score.py --sandbox: callable(cmd, cwd=, env=, timeout=) replacing host execution of test commands


def xsh(cmd, **kw):
    return (EXEC or sh)(cmd, **kw)


def git(repo, *args, **kw):
    return sh(["git", "-C", str(repo), *args], **kw)


# --------------------------------------------------------------------------- per-repo environment config

PY_COMMON = ["pytest"]
REPO_CFG = {
    "Fatal1ty/mashumaro": dict(lang="Python", python="3.12", reqs=["typing_extensions>=4.14.0", "msgpack", "pyyaml", "tomli-w", "orjson", "pytest", "pytest-mock", "pendulum", "ciso8601"], pythonpath=["."]),
    "anthropics/claude-agent-sdk-python": dict(lang="Python", python="3.12", reqs=["anyio[trio]>=4", "mcp>=1.23,<2", "pytest", "pytest-asyncio", "pytest-cov"], pythonpath=["src", "."]),
    "google/capirca": dict(lang="Python", python="3.12", reqs=["absl-py", "ply", "pyyaml", "six", "typing_extensions", "pytest", "mock"], pythonpath=["."]),
    "smokin-salmon/smoked-salmon": dict(lang="Python", python="3.12", reqs=None, pythonpath=["src", "."], extra_pyproject=True),
    "xo/dburl": dict(lang="Go"),
    "AgusRdz/chop": dict(lang="Go"),
    "dop251/goja": dict(lang="Go"),
    "cberner/redb": dict(lang="Rust"),
    "seanmonstar/reqwest": dict(lang="Rust"),
}


def tool_versions(lang):
    if lang == "TypeScript":   # host node is irrelevant (tests run in the node:<ver> image); recorded for the env file only
        try:
            return ["node " + sh("node --version").stdout.strip()]
        except Exception:  # noqa: BLE001
            return ["node (host: none; sandbox image decides)"]
    cmds = {"Python": ["uv --version"], "Go": ["go version"], "Rust": ["cargo --version", "rustc --version"]}[lang]
    return [sh(c).stdout.strip() for c in cmds]


def repo_slug(full):
    return full.replace("/", "__")


def short(task):
    return task["task_id"].rsplit("__", 1)[-1][:10] + "_" + task["repo_full"].split("/")[1]


def ensure_python_env(task, cfg, log):
    """uv venv per repo; deps locked with --exclude-newer = source commit date; lock recorded."""
    slug = repo_slug(task["repo_full"])
    venv = DATA / "envs" / slug
    lock = EXP / "envs" / f"{slug}.lock.txt"
    repo = DATA / "repos" / slug
    date = git(repo, "show", "-s", "--format=%cI", task["source_commit"]).stdout.strip()
    cmds = []
    if not (venv / "bin" / "python").exists():
        reqs = cfg["reqs"]
        if reqs is None:  # read runtime deps + dev group from pyproject at base commit
            import tomllib
            py = tomllib.loads(git(repo, "show", f"{task['base_commit']}:pyproject.toml").stdout)
            reqs = list(py["project"]["dependencies"]) + list(py.get("dependency-groups", {}).get("dev", []))
        req_in = EXP / "envs" / f"{slug}.in"
        req_in.write_text("\n".join(reqs) + "\n")
        c1 = f"uv venv --python {cfg['python']} {venv}"
        c2 = f"uv pip compile {req_in} --python {cfg['python']} --exclude-newer {date} -o {lock}"
        c3 = f"uv pip install --python {venv}/bin/python -r {lock}"
        for c in (c1, c2, c3):
            p = sh(c, cwd=ROOT, timeout=1200)
            cmds.append(c)
            if p.returncode:
                raise RuntimeError(f"env setup failed: {c}\n{p.stderr[-1500:]}")
    return venv, cmds


# --------------------------------------------------------------------------- runners

def _env(task, cfg, workdir, venv=None):
    e = dict(os.environ)
    e["CI"] = "1"
    if cfg["lang"] == "Python":
        e["PYTHONPATH"] = os.pathsep.join(str(Path(workdir) / p) for p in cfg["pythonpath"])
        e["PATH"] = f"{venv}/bin:" + e["PATH"]
        e["PYTHONDONTWRITEBYTECODE"] = "1"
    if cfg["lang"] == "Rust":
        e["CARGO_TARGET_DIR"] = str(DATA / "cargo_target" / repo_slug(task["repo_full"]))
    return e


def run_python(workdir, venv, cfg, task, files, timeout=TIME_LIMIT_S):
    xml = Path(workdir) / ".e2e_junit.xml"
    if xml.exists():
        xml.unlink()
    cmd = [str(venv / "bin/python"), "-m", "pytest", "-p", "no:cacheprovider", "-q", "-x" if False else "-q",
           f"--junitxml={xml}", "--rootdir", str(workdir), *files]
    try:
        p = xsh(cmd, cwd=workdir, env=_env(task, cfg, workdir, venv), timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, timeout, "timeout"
    res = {}
    if xml.exists():
        for tc in ET.parse(xml).getroot().iter("testcase"):
            cls, name = tc.get("classname", ""), tc.get("name", "")
            st = "pass"
            for ch in tc:
                if ch.tag in ("failure",):
                    st = "fail"
                elif ch.tag == "error":
                    st = "error"
                elif ch.tag == "skipped":
                    st = "skip"
            res[f"{cls}::{name}"] = dict(unit=cls, func=name.split("[")[0], status=st)
    return res, p.elapsed, p.stdout[-1500:] + p.stderr[-500:]


def run_go(workdir, cfg, task, dirs, timeout=TIME_LIMIT_S):
    if Path(workdir).is_dir():
        dirs = [d for d in dirs if (Path(workdir) / d).is_dir()]   # packages created by the change do not exist at the parent
    if not dirs:
        return {}, 0.0, "no test package present"
    pk = [("./" + d) if d else "." for d in dirs]
    try:
        p = xsh(["go", "test", "-json", "-count=1", *pk], cwd=workdir, env=_env(task, cfg, workdir), timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, timeout, "timeout"
    res = {}
    mod = None
    for line in p.stdout.splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("Action") in ("pass", "fail", "skip") and ev.get("Test"):
            unit = ev["Package"].split("/", 3)[-1] if False else ev["Package"]
            res[f"{ev['Package']}::{ev['Test']}"] = dict(unit=ev["Package"], func=ev["Test"].split("/")[0],
                                                          status={"pass": "pass", "fail": "fail", "skip": "skip"}[ev["Action"]])
    return res, p.elapsed, p.stdout[-800:] + p.stderr[-1500:]


def run_rust(workdir, cfg, task, bins, timeout=TIME_LIMIT_S):
    res = {}
    el = 0.0
    tail = ""
    for b in bins:
        try:
            p = xsh(["cargo", "test", "--test", b, "--", "--test-threads=4"], cwd=workdir, env=_env(task, cfg, workdir), timeout=timeout)
        except subprocess.TimeoutExpired:
            return res, timeout, "timeout"
        el += p.elapsed
        tail = p.stderr[-1500:] + p.stdout[-500:]
        for m in re.finditer(r"^test (\S+) \.\.\. (ok|FAILED|ignored)", p.stdout, re.M):
            st = {"ok": "pass", "FAILED": "fail", "ignored": "skip"}[m.group(2)]
            res[f"{b}::{m.group(1)}"] = dict(unit=b, func=m.group(1).split("::")[-1], status=st)
    return res, el, tail


def run_ts(workdir, cfg, task, units, timeout=TIME_LIMIT_S):
    """TypeScript/JavaScript: vitest or jest per run dir (see ts_support). Reports are JSON files written inside the worktree and copied
    back by the sandbox (modal_sandbox.json_out_path). Units absent from the worktree (files created by the change) are skipped."""
    import ts_support
    present = [u for u in units if (Path(workdir) / u).is_file()] if Path(workdir).is_dir() else list(units)
    if not present:
        return {}, 0.0, "no test file present"
    read, names = ts_support.host_read_pkg(workdir)
    res, el, tail = {}, 0.0, ""
    for i, grp in enumerate(ts_support.group_units(present, read, names, cfg.get("framework", "vitest"))):
        out = Path(workdir) / f".e2e_json_{i}.json"
        if out.exists():
            out.unlink()
        try:
            p = xsh(ts_support.group_command(grp, str(out)), cwd=workdir, env=_env(task, cfg, workdir), timeout=timeout)
        except subprocess.TimeoutExpired:
            return res, timeout, "timeout"
        el += p.elapsed
        tail = (p.stdout[-800:] + p.stderr[-1500:]) if isinstance(p.stdout, str) else str(p.stderr)[-1500:]
        if out.exists():
            res.update(ts_support.parse_report(out.read_text(), [str(workdir), "/work"]))
    return res, el, tail


def units_for(lang, test_files):
    """Map held-out test files to the run targets (files / package dirs / integration test binaries)."""
    if lang == "Python":
        return sorted(f for f in test_files if re.search(r"(test_[^/]*|[^/]*_tests?)\.py$", f))
    if lang == "Go":
        return sorted({os.path.dirname(f) for f in test_files if f.endswith("_test.go")})
    if lang == "Rust":
        return sorted({Path(f).stem for f in test_files if re.match(r"^tests/[^/]+\.rs$", f)})
    if lang == "TypeScript":
        import ts_support
        return ts_support.ts_units(test_files)
    return []


def run_tests(task, cfg, workdir, venv, units, timeout=TIME_LIMIT_S):
    lang = cfg["lang"]
    if lang == "Python":
        return run_python(workdir, venv, cfg, task, units, timeout)
    if lang == "Go":
        return run_go(workdir, cfg, task, units, timeout)
    if lang == "TypeScript":
        return run_ts(workdir, cfg, task, units, timeout)
    return run_rust(workdir, cfg, task, units, timeout)


def passing(res):
    return {k for k, v in res.items() if v["status"] == "pass"}


def match_touched(lang, rec, touched_by_unit):
    """rec: result record; touched_by_unit: {unit_key: set(funcs)} -> bool"""
    for ukey, funcs in touched_by_unit.items():
        if lang == "Python":
            # unit key = dotted module; qualnames may be Class.method
            if rec["unit"] == ukey or rec["unit"].startswith(ukey + "."):
                cls = rec["unit"][len(ukey) + 1:]
                for q in funcs:
                    if q == rec["func"] and not cls:
                        return True
                    if "." in q and cls and q == f"{cls}.{rec['func']}":
                        return True
        elif lang == "TypeScript":
            if rec["unit"] == ukey:   # file-level rule: every test of a changed test file is touched
                return True
        elif lang == "Go":
            if rec["unit"].endswith("/" + ukey) or rec["unit"] == ukey or (ukey == "" and True):
                if rec["func"] in funcs:
                    return True
        else:
            if rec["unit"] == ukey and rec["func"] in funcs:
                return True
    return False


# --------------------------------------------------------------------------- worktrees

def make_worktree(task, name):
    repo = DATA / "repos" / repo_slug(task["repo_full"])
    wd = DATA / "work" / name
    if wd.exists():
        git(repo, "worktree", "remove", "--force", str(wd))
        shutil.rmtree(wd, ignore_errors=True)
    wd.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "--detach", "-f", str(wd), task["base_commit"], check=True)
    return repo, wd


def drop_worktree(repo, wd):
    git(repo, "worktree", "remove", "--force", str(wd))
    shutil.rmtree(wd, ignore_errors=True)


def apply_heldout(repo, wd, task, test_entries):
    """Overwrite/add held-out test files from source_commit; delete those removed by the commit."""
    for st, p in test_entries:
        tgt = Path(wd) / p
        if st == "D":
            tgt.unlink(missing_ok=True)
        else:
            tgt.parent.mkdir(parents=True, exist_ok=True)
            data = subprocess.run(["git", "-C", str(repo), "show", f"{task['source_commit']}:{p}"], capture_output=True, check=True).stdout
            tgt.write_bytes(data)


# --------------------------------------------------------------------------- gate

def analyse_diff(task):
    repo = DATA / "repos" / repo_slug(task["repo_full"])
    ns = git(repo, "diff", "--name-status", "-M", task["base_commit"], task["source_commit"], check=True).stdout
    entries = parse_name_status(ns)
    tests, non = split_changes(entries)
    return repo, tests, non


def gate_task(task, cfg, lang_note=""):
    """Returns (result_dict, env_dict, extra) ; result['gated'] True/False with reason."""
    tid = task["task_id"]
    r = dict(task_id=tid, repo=task["repo_full"], language=cfg["lang"], base=task["base_commit"], source=task["source_commit"], task_text=task["task_text"])
    repo, tests, non = analyse_diff(task)
    r["heldout_test_files"] = [p for _, p in tests]
    r["nontest_files"] = [p for _, p in non]
    if not tests:
        return {**r, "gated": False, "reason": "no_test_files"}, None
    if not non:
        return {**r, "gated": False, "reason": "no_nontest_change"}, None
    lang = cfg["lang"]
    if lang == "Rust" and any(not re.match(r"^tests/", p) for _, p in tests):
        pass
    units = units_for(lang, [p for st, p in tests if st != "D"])
    if not units:
        return {**r, "gated": False, "reason": "no_runnable_test_unit"}, None
    # touched functions
    diff = git(repo, "diff", "-U0", "-M", task["base_commit"], task["source_commit"], "--", *[p for st, p in tests if st != "D"]).stdout
    added = parse_added_lines(diff)
    touched = {}
    for st, p in tests:
        if st == "D" or p not in added:
            continue
        text = git(repo, "show", f"{task['source_commit']}:{p}").stdout
        fs = ["*"] if lang == "TypeScript" and text else touched_functions(lang, text, added[p]) if lang != "TypeScript" else []
        if not fs:
            continue
        if lang == "TypeScript":
            key = p
        elif lang == "Python":
            key = p[:-3].replace("/", ".")
        elif lang == "Go":
            key = os.path.dirname(p)
        else:
            key = Path(p).stem
        touched.setdefault(key, set()).update(fs)
    r["touched_functions"] = {k: sorted(v) for k, v in touched.items()}
    if not touched:
        return {**r, "gated": False, "reason": "no_touched_test_functions"}, None

    env = dict(task_id=tid, repo=task["repo_full"], language=lang, tool_versions=tool_versions(lang), commands=[], run_units=units)
    venv = None
    try:
        if lang == "Python":
            venv, cmds = ensure_python_env(task, cfg, None)
            env["commands"] += cmds + [f"{venv}/bin/python -m pytest -p no:cacheprovider --junitxml=... <files>"]
            env["python"] = sh([str(venv / "bin/python"), "--version"]).stdout.strip()
            env["venv"] = str(venv.relative_to(ROOT))
            env["lockfile"] = str((EXP / "envs" / f"{repo_slug(task['repo_full'])}.lock.txt").relative_to(ROOT))
            env["pythonpath"] = cfg["pythonpath"]
        elif lang == "Go":
            env["commands"] += ["go test -json -count=1 ./<pkgdir>"]
            env["go_env"] = {k: sh(["go", "env", k]).stdout.strip() for k in ("GOVERSION", "GOFLAGS", "GOTOOLCHAIN")}
        elif lang == "TypeScript":
            env["commands"] += ["npx --no-install vitest run|jest --json --outputFile=<f> <files> (per run dir)"]
        else:
            env["commands"] += ["cargo test --test <binary> -- --test-threads=4"]
            env["cargo_target_dir"] = f"data/e2e/cargo_target/{repo_slug(task['repo_full'])}"
    except Exception as ex:  # noqa: BLE001
        return {**r, "gated": False, "reason": "env_setup_failed", "detail": str(ex)[:600]}, env

    name = "gate_" + short(task)
    repo, wd = make_worktree(task, name)
    try:
        # baseline: parent, original tests
        res_p, t_p, tail_p = run_tests(task, cfg, wd, venv, units)
        # parent + held-out tests
        apply_heldout(repo, wd, task, tests)
        res_a, t_a, tail_a = run_tests(task, cfg, wd, venv, units)
        # source commit (full real commit, tests included)
        git(wd, "checkout", "-q", "--detach", "-f", task["source_commit"], check=True)
        git(wd, "clean", "-fdq")
        res_b, t_b, tail_b = run_tests(task, cfg, wd, venv, units)
        res_b2, t_b2, _ = run_tests(task, cfg, wd, venv, units)
    finally:
        drop_worktree(repo, wd)
    r["timings_s"] = dict(parent_original=round(t_p, 1), parent_plus_heldout=round(t_a, 1), source=round(t_b, 1), source_rerun=round(t_b2, 1))
    pb = passing(res_b)
    targeted_all = sorted(k for k in pb if match_touched(lang, res_b[k], touched))
    if not targeted_all:
        return {**r, "gated": False, "reason": "no_touched_test_passes_at_source", "detail": tail_b[-500:]}, env
    f2p = [k for k in targeted_all if res_a.get(k, {}).get("status") != "pass"]
    r["targeted_total"] = len(targeted_all)
    r["fail_to_pass"] = f2p
    r["touched_but_already_passing_at_parent"] = [k for k in targeted_all if k not in f2p]
    r["parent_plus_heldout_status"] = {k: res_a.get(k, {}).get("status", "missing(build/collect error)") for k in targeted_all}
    if not f2p:
        return {**r, "gated": False, "reason": "targeted_tests_pass_at_parent"}, env
    if any(res_b2.get(k, {}).get("status") != "pass" for k in f2p):
        return {**r, "gated": False, "reason": "flaky_at_source"}, env
    warm = min(t_b, t_b2)
    if warm > TIME_LIMIT_S:
        return {**r, "gated": False, "reason": "too_slow"}, env
    base_pass = passing(res_p)
    reg = sorted(k for k in (base_pass & pb) if k not in f2p)
    env["regression_scope"] = "tests in the touched test files/packages/binaries that pass at parent (original tests) and at source_commit"
    r.update(gated=True, reason="ok", regression_set_size=len(reg), n_heldout_test_files=len(tests))
    env["fail_to_pass"] = f2p
    env["regression_set"] = reg
    env["touched_by_unit"] = {k: sorted(v) for k, v in touched.items()}
    env["heldout_test_entries"] = tests
    env["nontest_entries"] = non
    env["base_commit"], env["source_commit"] = task["base_commit"], task["source_commit"]
    env["config"] = {k: v for k, v in cfg.items()}
    env["run_units"] = units
    return r, env


def load_tasks():
    d = {t["task_id"]: t for t in json.load(open(TASKS))["tasks"]}
    if os.environ.get("E2E_TASKS_MANIFEST"):   # final E2E v4 carries one v3 task; its public record lives in the benchmark-repo dataset (read only)
        p = ROOT.parents[1].parent / "benchmark-repo/datasets/artifacts/public/tasks_1660.json"
        if p.exists():
            for t in json.load(open(p))["tasks"]:
                d.setdefault(t["task_id"], t)
    return d


def main(argv):
    tasks = load_tasks()
    cand_file = EXP / "candidates.json"
    cands = json.load(open(cand_file))["candidates"]
    out_path = EXP / "gate_results.json"
    results = json.load(open(out_path)) if out_path.exists() else {"results": []}
    done = {x["task_id"] for x in results["results"]}
    n_gated = sum(1 for x in results["results"] if x.get("gated"))
    for c in cands:
        if n_gated >= 5:
            break
        if c["task_id"] in done:
            continue
        task = tasks[c["task_id"]]
        cfg = REPO_CFG[task["repo_full"]]
        t0 = time.time()
        try:
            r, env = gate_task(task, cfg)
        except Exception as ex:  # noqa: BLE001
            r, env = dict(task_id=c["task_id"], repo=task["repo_full"], language=cfg["lang"], gated=False, reason="exception", detail=str(ex)[:800]), None
        r["gate_wall_s"] = round(time.time() - t0, 1)
        results["results"].append(r)
        if r.get("gated"):
            n_gated += 1
        if env is not None:
            (EXP / "envs" / f"{c['task_id']}.json").write_text(json.dumps(env, indent=1))
        out_path.write_text(json.dumps(results, indent=1))
        print(r["task_id"][-30:], r["gated"], r["reason"], r.get("timings_s"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
