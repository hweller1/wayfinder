"""TypeScript/JavaScript support for the E2E gate/scorer (pure logic, unit-tested in tests/test_e2e_ts.py).

Language label is "TypeScript" for both TS and JS repos. Test frameworks: vitest and jest (both emit a jest-shaped JSON report:
testResults[].name + assertionResults[].{fullName,status}); mocha is not supported.

Deviation notes (documented design choices):
  * Touched-function rule: TS tests are matched at FILE level (every test of a changed test file is "touched"); F2P is then the usual
    "passes at source, does not pass at parent+held-out". Test names are runtime strings (describe/it), so a line-span rule is not reliable.
  * Result key: f"{test_file_rel_to_repo_root}::{fullName}", unit = the test file path.
  * Each test file runs from its run dir (nearest ancestor holding a vitest/jest config or a package.json declaring the framework,
    else the repo root); one command per run dir. Commands use `npx --no-install` (the sandbox has no network).
  * Dependencies are installed once at prepare time (network on, SOURCE commit tree) and tarred into ONE file on the cache volume
    (/cache/node/<task_key>.tar); every no-network sandbox restores it over /work before the timed command.
"""
from __future__ import annotations

import json
import re
import shlex

TS_UNIT_RE = re.compile(r"(\.|-)(test|spec)\.(m|c)?[jt]sx?$")
TS_TESTDIR_RE = re.compile(r"(^|/)__tests__/[^/]+\.(m|c)?[jt]sx?$")
CONFIG_RE = {"vitest": re.compile(r"^vitest\.(config|workspace)\.[cm]?[jt]s$|^vitest\.config\.mts$"),
             "jest": re.compile(r"^jest\.config\.[cm]?[jt]s(on)?$")}
NODE_CHOICES = (18, 20, 22, 24)
REPORT = ".e2e_json"   # per-group report files: <workdir>/.e2e_json_<i>.json


def is_ts_unit(path: str) -> bool:
    return bool(TS_UNIT_RE.search(path) or TS_TESTDIR_RE.search(path))


def ts_units(test_files) -> list[str]:
    return sorted(f for f in test_files if is_ts_unit(f))


def node_major(nvmrc: str | None = None, node_version: str | None = None, engines: str | None = None, default: int = 22) -> int:
    """Node major for the image, clamped to NODE_CHOICES: .nvmrc, then .node-version, then the first number of engines.node."""
    for src in (nvmrc, node_version):
        if src:
            m = re.search(r"(\d+)", src.strip().lstrip("vV").replace("lts/", "").replace("node", ""))
            if m and "lts" not in src.lower():
                return _clamp(int(m.group(1)))
    if engines:
        nums = [int(x) for x in re.findall(r"(\d+)", engines)]
        if nums:
            lo = nums[0]
            if ">=" in engines or ">" in engines:   # open-ended range: newest image that satisfies it (deps often need >=20)
                return default if lo <= default else _clamp(lo)
            return _clamp(lo)
    return default


def _clamp(v: int) -> int:
    if v in NODE_CHOICES:
        return v
    if v < NODE_CHOICES[0]:
        return NODE_CHOICES[0]
    return max(c for c in NODE_CHOICES if c <= v)


def detect_framework(pkg: dict | None, names: set[str] | frozenset[str] = frozenset()) -> str | None:
    """names = file names in the same directory. vitest wins over jest when both are declared (vitest repos often keep a jest shim)."""
    for fw in ("vitest", "jest"):
        if any(CONFIG_RE[fw].match(n) for n in names):
            return fw
    if pkg:
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {}), **pkg.get("peerDependencies", {})}
        scripts = " ".join(str(v) for v in (pkg.get("scripts") or {}).values())
        for fw in ("vitest", "jest"):
            if fw in deps or re.search(rf"\b{fw}\b", scripts) or (fw == "jest" and "jest" in pkg):
                return fw
    return None


def find_run_dir(test_file: str, read_pkg, list_names) -> tuple[str, str | None]:
    """Walk up from the test file's directory; first dir with a framework config or a package.json declaring one wins.
    read_pkg(dir) -> dict|None, list_names(dir) -> set of file names. Falls back to ('', framework of the root or None)."""
    parts = test_file.split("/")[:-1]
    for i in range(len(parts), -1, -1):
        d = "/".join(parts[:i])
        fw = detect_framework(read_pkg(d), list_names(d))
        if fw:
            return d, fw
    return "", None


def group_units(units: list[str], read_pkg, list_names, default_fw: str = "vitest") -> list[dict]:
    """[{dir, framework, files}] one group per run dir, files relative to the run dir, sorted."""
    g: dict[tuple[str, str], list[str]] = {}
    for u in sorted(units):
        d, fw = find_run_dir(u, read_pkg, list_names)
        g.setdefault((d, fw or default_fw), []).append(u)
    return [dict(dir=d, framework=fw, units=fs) for (d, fw), fs in sorted(g.items())]


def group_command(group: dict, out_path: str | None, units_present: list[str] | None = None, report_flag: bool = True) -> list[str]:
    """argv (bash -c) running one group; out_path is the absolute report path (host path; mapped to /work by the sandbox)."""
    d = group["dir"]
    files = [u for u in group["units"] if units_present is None or u in units_present]
    rel = [f[len(d) + 1:] if d and f.startswith(d + "/") else f for f in files]
    fs = " ".join(shlex.quote(f) for f in rel)
    if group["framework"] == "jest":
        core = "npx --no-install jest --ci --forceExit " + (f"--json --outputFile={shlex.quote(out_path)} " if out_path else "") + fs
    else:
        core = "npx --no-install vitest run " + (f"--reporter=json --outputFile={shlex.quote(out_path)} " if out_path else "") + fs
    cd = f"cd {shlex.quote(d)} && " if d else ""
    return ["bash", "-c", f"export CI=1; {cd}{core}"]


_STATUS = {"passed": "pass", "failed": "fail", "pending": "skip", "skipped": "skip", "todo": "skip", "disabled": "skip"}


def parse_report(text: str, prefixes: list[str]) -> dict:
    """jest/vitest JSON report -> {key: dict(unit, func, status)}; prefixes = absolute dirs to strip (sandbox /work and host workdir)."""
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    res = {}
    for tr in data.get("testResults", []):
        name = tr.get("name", "")
        for p in sorted(prefixes, key=len, reverse=True):
            if name.startswith(p.rstrip("/") + "/"):
                name = name[len(p.rstrip("/")) + 1:]
                break
        for a in tr.get("assertionResults", []):
            full = a.get("fullName") or " ".join(list(a.get("ancestorTitles", [])) + [a.get("title", "")])
            res[f"{name}::{full}"] = dict(unit=name, func=full, status=_STATUS.get(a.get("status"), "fail"))
    return res


def prepare_script(key: str, cache: str = "/cache") -> str:
    """Network-on install of the source-commit tree + one tarball of every node_modules dir. pm from the lockfile."""
    return f"""set -e
export CI=1 HOME=/root COREPACK_ENABLE_DOWNLOAD_PROMPT=0
corepack enable >/dev/null 2>&1 || true
if [ -f pnpm-lock.yaml ]; then
  pnpm install --frozen-lockfile || pnpm install --no-frozen-lockfile
elif [ -f yarn.lock ]; then
  yarn install --immutable || yarn install
elif [ -f package-lock.json ]; then
  npm ci || npm install
else
  npm install
fi
mkdir -p {cache}/node
find . -name node_modules -type d -prune -print0 | tar --null -T - -cf {cache}/node/{key}.tar
ls -la {cache}/node/{key}.tar
"""


def restore_script(key: str, cache: str = "/cache") -> str:
    return f"[ -f {cache}/node/{key}.tar ] && tar xf {cache}/node/{key}.tar -C /work || echo NO_NODE_TAR"


OUT_RE = re.compile(r"--outputFile=(\S+)")


def json_out_path(argv: list[str]) -> str | None:
    """The report path of a TS test command (searched in every argv element, the command is inside a bash -c string)."""
    for a in argv:
        m = OUT_RE.search(a)
        if m:
            return m.group(1).strip("'\"")
    return None


def host_read_pkg(workdir):
    from pathlib import Path
    def read(d):
        p = Path(workdir) / d / "package.json"
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return None
    def names(d):
        p = Path(workdir) / d
        return {x.name for x in p.iterdir()} if p.is_dir() else set()
    return read, names
