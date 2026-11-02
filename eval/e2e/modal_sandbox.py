"""Modal Sandboxes for E2E test execution. Everything that runs repo code (gate, scorer self-check, later the agent's
tool calls) runs here, never on the host. Supersedes the local-Docker scripts/e2e/sandbox.py (kept as Dockerfile/toolchain
logic source; `score.py --sandbox` still works if Docker is up, `score.py --modal` is the path to use).

  app      asa-e2e
  volume   asa-e2e-cache-v2  /cache/{gomod,gocache/<task>,cargo,target/<task>,locks/<task>}
  images   python: python:<ver>-slim + uv, task lockfile (macOS lock first; re-resolved for linux/amd64 on failure,
           recorded as experiments/e2e/envs/<task>.linux.lock.txt); go: golang:<go.mod toolchain>;
           rust: rust:<rust-toolchain|rust-version>, falling back to the host gate's rustc when the pin cannot build
           (recorded with the reason in experiments/e2e/modal_toolchains.json)
  prepare  Sandbox WITH network; base-commit checkout only (no agent code); fetches/builds deps into the volume
  exec     Sandbox with block_network=True; uploads the worktree tarball, runs one command (10 min cap)

  uv run --no-project --with modal python scripts/e2e/modal_sandbox.py prepare --task <id>
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE  # re-pointed to wayfinder/eval/e2e
sys.path.insert(0, str(HERE))
import dns_fallback  # noqa: E402,F401  (resolver fallback for *.modal.host; see its docstring)
from sandbox import go_version, rust_version  # noqa: E402  (pure parsing helpers, shared with the Docker version)

APP_NAME, VOL_NAME = os.environ.get("E2E_MODAL_APP", "wayfinder-e2e-sandbox"), "asa-e2e-cache-v2"   # app renamed for the wayfinder track (watcher prefix wayfinder-e2e*); volume is the shared round-3 cache   # v2 volume: no 500K-file cap (v1 asa-e2e-cache filled up at ~25 tasks of Go/Rust deps)
WORK, CACHE = "/work", "/cache"
CMD_TIMEOUT = 600          # 10 min per command
CPU, MEM_MB = 4.0, 8192
import os  # noqa: E402
EXP = Path(os.environ["E2E_EXP_DIR"]) if os.environ.get("E2E_EXP_DIR") else ROOT / "experiments" / "e2e"   # v3 task set override, see gate.py
TOOLCHAINS = EXP / "modal_toolchains.json"
TAR_EXCLUDE = (".git", "__pycache__", ".e2e_junit.xml")


# --------------------------------------------------------------------------- pure logic (unit-tested)

def task_key(env: dict) -> str:
    """Per-task subdir name: repo slug + last 10 chars of the task id (the source-commit prefix)."""
    return env["repo"].replace("/", "__") + "__" + env["task_id"][-10:]


def _rust_pin(show):
    try:
        return rust_version(show("rust-toolchain") or show("rust-toolchain.toml"), show("Cargo.toml"))
    except ValueError:
        return None


def host_rustc_version(env: dict) -> str | None:
    for s in env.get("tool_versions", []):
        m = re.match(r"rustc (\d+\.\d+(?:\.\d+)?)", s)
        if m:
            return m.group(1)
    return None


def exclude_newer_from_lock(text: str) -> str | None:
    m = re.search(r"--exclude-newer\s+(\S+)", text)
    return m.group(1) if m else None


def linux_lock_path(env: dict) -> Path:
    return EXP / "envs" / f"{env['repo'].replace('/', '__')}.linux.lock.txt"


def image_spec(env: dict, show, toolchains: dict | None = None) -> dict:
    """Pure description of the task image: {kind, base, ...}. `show(path)` reads a file at base_commit."""
    lang = env["language"]
    if lang == "Python":
        slug = env["repo"].replace("/", "__")
        lin = linux_lock_path(env)
        use_linux = bool((toolchains or {}).get(env["task_id"], {}).get("python_lock") == "linux")
        return dict(kind="python", base=f"python:{env['config'].get('python', '3.12')}-slim",
                    build_tools=bool((toolchains or {}).get(env["task_id"], {}).get("build_tools")),
                    lock=str((lin if use_linux else ROOT / env["lockfile"]).relative_to(ROOT)), slug=slug)
    if lang == "Go":
        v = go_version(show("go.mod") or "")
        return dict(kind="go", base=f"golang:{v}", version=v)
    if lang == "Rust":
        rec = (toolchains or {}).get(env["task_id"], {})
        pinned = _rust_pin(show)
        v = rec.get("chosen") or pinned or host_rustc_version(env)
        return dict(kind="rust", base=f"rust:{v}", version=v, pinned=pinned, fallback=host_rustc_version(env))
    if lang == "TypeScript":
        import ts_support
        pkg = {}
        try:
            pkg = json.loads(show("package.json") or "{}")
        except Exception:  # noqa: BLE001
            pass
        v = (toolchains or {}).get(env["task_id"], {}).get("node") or ts_support.node_major(
            show(".nvmrc"), show(".node-version"), (pkg.get("engines") or {}).get("node"))
        return dict(kind="node", base=f"node:{v}-bookworm", version=v)
    raise ValueError(lang)


def sandbox_mapped(s: str, host_wd: str) -> str:
    return s.replace(host_wd, WORK)


def command_argv(cmd: list[str], host_wd: str, lang: str) -> list[str]:
    """Host test command -> in-sandbox argv (worktree paths -> /work, host venv python -> image python)."""
    out = [sandbox_mapped(str(a), host_wd) for a in cmd]
    if lang == "Python" and re.search(r"/bin/python\d*$", out[0]):
        out[0] = "python"
    return out


def command_env(host_env: dict | None, host_wd: str, lang: str, key: str, network: bool = False) -> dict:
    e = {"CI": "1"}
    if lang == "Python":
        e["PYTHONDONTWRITEBYTECODE"] = "1"
        if host_env and host_env.get("PYTHONPATH"):
            e["PYTHONPATH"] = sandbox_mapped(host_env["PYTHONPATH"], host_wd)
    elif lang == "Go":
        e.update(GOFLAGS="-mod=mod -buildvcs=false", GOMODCACHE=f"{CACHE}/gomod", GOCACHE="/tmp/gocache",   # build cache on local disk: per-task caches on the volume hit its file-count limit ("no space left")
                 
                 GOTOOLCHAIN="local")
        if not network:
            e["GOPROXY"] = "off"
    elif lang == "TypeScript":
        e.update(HOME="/tmp", NO_COLOR="1", NODE_OPTIONS="--max-old-space-size=6144")
        if network:
            e["HOME"] = "/root"
    elif lang == "Rust":
        e.update(CARGO_HOME=f"{CACHE}/cargo", CARGO_TARGET_DIR=f"{CACHE}/target/{key}")
        if not network:
            e["CARGO_NET_OFFLINE"] = "true"
    return e


def prepare_commands(env: dict, key: str) -> list[list[str]]:
    """Network-on dependency fetch/warm commands (base commit only)."""
    lang = env["language"]
    units = env.get("run_units", [])
    if lang == "Go":
        pk = [("./" + d) if d else "." for d in units] or ["./..."]
        # packages created by the change are absent at the base commit: only warm the ones that exist
        probe = " ".join(_q(p) for p in pk)
        return [["go", "mod", "download"],
                ["bash", "-c", f"pk=''; for d in {probe}; do [ -d \"$d\" ] && pk=\"$pk $d\"; done; "
                               "[ -z \"$pk\" ] && exit 0; go test -count=1 -run '^$' $pk"]]
    if lang == "TypeScript":
        import ts_support
        return [["bash", "-c", ts_support.prepare_script(key, CACHE)]]
    if lang == "Rust":
        return [["cargo", "test", "--no-run", *sum((["--test", u] for u in units), [])],
                ["bash", "-c", f"mkdir -p {CACHE}/locks/{key} && cp -f Cargo.lock {CACHE}/locks/{key}/Cargo.lock"]]
    return [["python", "-c", "import sys; print(sys.version)"]]


def wrap_for_lock(argv: list[str], lang: str, key: str) -> list[str]:
    """Rust has no Cargo.lock in these repos; reuse the one resolved at prepare time so offline resolution matches."""
    if lang != "Rust":
        return argv
    inner = " ".join(_q(a) for a in argv)
    # RUSTUP_TOOLCHAIN pins the image's installed toolchain: a repo `rust-toolchain` file of "1.90" otherwise makes
    # rustup try to download channel-rust-1.90.toml, which fails with the network blocked.
    return ["bash", "-c", "export RUSTUP_TOOLCHAIN=$(rustup toolchain list | head -1 | cut -d' ' -f1); "
            f"[ -f Cargo.lock ] || cp {CACHE}/locks/{key}/Cargo.lock Cargo.lock 2>/dev/null; exec {inner}"]


def _q(a: str) -> str:
    return "'" + a.replace("'", "'\\''") + "'"


def junit_path(argv: list[str]) -> str | None:
    import ts_support
    if ts_support.json_out_path(argv):   # TypeScript reports (jest/vitest --outputFile=)
        return ts_support.json_out_path(argv)
    for a in argv:
        if a.startswith("--junitxml="):
            return a.split("=", 1)[1]
    return None


def write_chunked(stdin, data: bytes, chunk: int = 4 * 1024 * 1024) -> None:
    """Modal's stdin buffer is capped (BufferError 'Buffer size exceed limit' on big repos): drain after every chunk."""
    for i in range(0, len(data), chunk):
        stdin.write(data[i:i + chunk])
        stdin.drain()
    stdin.write_eof()
    stdin.drain()


def make_tarball(wd: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(wd, arcname=".", filter=lambda ti: None if any(
            part in TAR_EXCLUDE for part in Path(ti.name).parts) else ti)
    return buf.getvalue()


def load_toolchains() -> dict:
    return json.loads(TOOLCHAINS.read_text()) if TOOLCHAINS.exists() else {}


def save_toolchain(task_id: str, rec: dict):
    """Read-modify-write under an flock and an atomic rename: v4 gates many tasks in parallel processes sharing this file."""
    import fcntl
    TOOLCHAINS.parent.mkdir(parents=True, exist_ok=True)
    with open(str(TOOLCHAINS) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = load_toolchains()
        d[task_id] = {**d.get(task_id, {}), **rec}
        tmp = TOOLCHAINS.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
        os.replace(tmp, TOOLCHAINS)


# --------------------------------------------------------------------------- Modal side

def _show_factory(env):
    repo = ROOT / "data" / "e2e" / "repos" / env["repo"].replace("/", "__")

    def show(path):
        p = subprocess.run(["git", "-C", str(repo), "show", f"{env['base_commit']}:{path}"], capture_output=True, text=True)
        return p.stdout if p.returncode == 0 else None
    return show


def build_image(spec: dict):
    import modal
    if spec["kind"] == "python":
        img = modal.Image.from_registry(spec["base"])
        if spec.get("build_tools"):   # sdists that compile C extensions (recorded per task in modal_toolchains.json)
            img = img.apt_install("gcc", "g++", "make", "libffi-dev", "libssl-dev", "libpq-dev")
        return (img
                .pip_install("uv")
                .add_local_file(str(ROOT / spec["lock"]), "/tmp/lock.txt", copy=True)
                .run_commands("uv pip install --system --no-cache -r /tmp/lock.txt"))
    if spec["kind"] == "node":
        return modal.Image.from_registry(spec["base"], add_python="3.12")
    # go / rust official images; modal injects its runtime (no python in them -> add_python)
    return modal.Image.from_registry(spec["base"], add_python="3.12")


class ModalSandbox:
    """Callable replacement for gate.sh for test commands: exec(cmd, cwd=, env=, timeout=) -> CompletedProcess-like."""

    def __init__(self, toolchains: dict | None = None):
        import modal
        self.modal = modal
        self.app = modal.App.lookup(APP_NAME, create_if_missing=True)
        self.vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True, version=2)
        self.toolchains = toolchains if toolchains is not None else load_toolchains()
        self.current_env: dict | None = None
        self.specs: dict[str, dict] = {}
        self.images: dict[str, object] = {}
        self.log: list[dict] = []

    def use(self, env: dict, show=None):
        self.current_env = env
        self.toolchains = load_toolchains()
        spec = image_spec(env, show or _show_factory(env), self.toolchains)
        self.specs[env["task_id"]] = spec
        self.images[env["task_id"]] = build_image(spec)

    # -- low level
    def _create(self, env, network: bool):
        tid = env["task_id"]
        return self.modal.Sandbox.create(app=self.app, image=self.images[tid], block_network=not network,
                                         volumes={CACHE: self.vol}, workdir=WORK, cpu=CPU, memory=MEM_MB,
                                         timeout=CMD_TIMEOUT + 300)

    def _run(self, env, tarball: bytes | None, argvs: list[tuple[list[str], dict, str | None]], network: bool,
             timeout: int = CMD_TIMEOUT):
        """One sandbox: extract tarball into /work, run each argv; returns [(rc, out, err, secs, junit_xml_bytes)]."""
        sb = self._create(env, network)
        res = []
        try:
            sb.exec("mkdir", "-p", WORK).wait()
            if tarball is not None:
                up = sb.exec("tar", "xzf", "-", "-C", WORK, text=False)
                write_chunked(up.stdin, tarball)
                up.wait()
            for argv, cenv, xml in argvs:
                t0 = time.time()
                p = sb.exec(*argv, env=cenv, workdir=WORK, timeout=timeout)
                out, err = p.stdout.read(), p.stderr.read()
                rc = p.wait()
                secs = time.time() - t0
                xb = None
                if xml:
                    c = sb.exec("cat", xml, text=False)
                    xb = c.stdout.read()
                    if c.wait() != 0:
                        xb = None
                res.append((rc, out, err, secs, xb))
        finally:
            sb.terminate()
            try:
                sb.wait()
            except Exception:  # noqa: BLE001
                pass
        return res

    # -- public
    def prepare(self, env: dict, worktree_tarball: bytes):
        """Network ON, base-commit checkout only. Returns list of (rc, out, err, secs)."""
        lang, key = env["language"], task_key(env)
        cmds = [(a if lang != "Rust" or a[0] == "bash" else wrap_for_lock(a, lang, key),
                 command_env(None, WORK, lang, key, network=True), None) for a in prepare_commands(env, key)]
        r = self._run(env, worktree_tarball, cmds, network=True, timeout=1800)
        return [x[:4] for x in r]

    def exec(self, cmd, cwd=None, env=None, timeout=None, **_):
        """gate.EXEC adapter: tars `cwd`, runs cmd offline, copies junit xml back, returns CompletedProcess-like."""
        te = self.current_env
        if te is None:
            raise RuntimeError("ModalSandbox.use(env) not called")
        lang, key = te["language"], task_key(te)
        wd = str(cwd)
        argv = wrap_for_lock(command_argv(list(cmd), wd, lang), lang, key)
        cenv = command_env(env, wd, lang, key, network=False)
        xml = junit_path(command_argv(list(cmd), wd, lang))
        pre = []
        if lang == "TypeScript":   # restore node_modules (one tar on the volume) OUTSIDE the timed command
            import ts_support
            pre = [(["bash", "-c", ts_support.restore_script(key, CACHE)], cenv, None)]
        try:
            rc, out, err, secs, xb = self._run(te, make_tarball(wd), pre + [(argv, cenv, xml)], network=False,
                                               timeout=min(timeout or CMD_TIMEOUT, CMD_TIMEOUT))[-1]
        except Exception as ex:  # noqa: BLE001
            if "imeout" in type(ex).__name__ or "imeout" in str(ex):
                raise subprocess.TimeoutExpired(cmd, timeout or CMD_TIMEOUT)
            raise
        if xb:
            Path(xml.replace(WORK, wd)).write_bytes(xb)
        self.log.append(dict(task_id=te["task_id"], argv=argv, rc=rc, seconds=round(secs, 1)))
        return SimpleNamespace(returncode=rc, stdout=out, stderr=err, elapsed=secs)


# --------------------------------------------------------------------------- python lock fallback / rust fallback

def compile_linux_lock(env: dict) -> Path:
    slug = env["repo"].replace("/", "__")
    mac = (ROOT / env["lockfile"]).read_text()
    date = exclude_newer_from_lock(mac)
    out = linux_lock_path(env)
    cmd = ["uv", "pip", "compile", str(EXP / "envs" / f"{slug}.in"), "--python-version", env["config"].get("python", "3.12"),
           "--python-platform", "x86_64-manylinux_2_28", "-o", str(out)] + (["--exclude-newer", date] if date else [])
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def prepare_task(task_id: str, sb: ModalSandbox | None = None) -> dict:
    import gate as g
    env = json.load(open(g.EXP / "envs" / f"{task_id}.json"))
    task = g.load_tasks()[task_id]
    sb = sb or ModalSandbox()
    lang = env["language"]
    t0 = time.time()
    rec: dict = dict(task_id=task_id, language=lang)
    repo, wd = g.make_worktree(task, "prep_" + g.short(task))
    try:
        if lang == "TypeScript":   # deps are installed from the SOURCE commit's manifests/lockfile (the real change may add dependencies)
            g.git(wd, "checkout", "-q", "--detach", "-f", task["source_commit"], check=True)
            g.git(wd, "clean", "-fdq")
        tar = make_tarball(str(wd))
    finally:
        g.drop_worktree(repo, wd)
    show = _show_factory(env)
    if lang == "Python":
        for attempt, lock in (("macos", None), ("linux", compile_linux_lock)):
            if lock:
                lock(env)
                save_toolchain(task_id, dict(python_lock="linux", python_lock_reason=rec["reason"]))
            try:
                sb.use(env, show)
                r = sb.prepare(env, tar)   # forces image build + smoke
                rec.update(python_lock=attempt, prepare=[x[:3] for x in r])
                save_toolchain(task_id, dict(python_lock=attempt))
                break
            except Exception as ex:  # noqa: BLE001
                rec["reason"] = f"macOS lock failed to build on linux/amd64: {str(ex)[-300:]}"
                if attempt == "linux":
                    raise
    elif lang == "TypeScript":
        sb.use(env, show)
        r = sb.prepare(env, tar)
        rec.update(node=sb.specs[task_id]["version"], prepare=[x[:3] for x in r])
    elif lang == "Rust":
        pinned = _rust_pin(show)
        fb = host_rustc_version(env)
        pinned = pinned or fb      # no rust-toolchain / rust-version pin: use the gate host's rustc
        save_toolchain(task_id, dict(chosen=pinned, pinned=pinned, fallback=fb))
        sb.use(env, show)
        r = sb.prepare(env, tar)
        if r[0][0] != 0 and fb and fb != pinned:
            reason = f"pinned rust {pinned} cannot build the dependency set (no Cargo.lock, latest deps): " + (r[0][2] or r[0][1])[-400:]
            save_toolchain(task_id, dict(chosen=fb, reason=reason))
            sb.use(env, show)
            r = sb.prepare(env, tar)
            rec["fallback_reason"] = reason
        rec.update(chosen=load_toolchains()[task_id]["chosen"], prepare=[x[:3] for x in r])
    else:
        sb.use(env, show)
        r = sb.prepare(env, tar)
        # The real commit may add module dependencies (go.mod/go.sum change). Test runs are offline, so also fetch the
        # SOURCE commit's go.mod/go.sum deps (manifests only; no source files) into the same module cache.
        repo_dir = g.DATA / "repos" / env["repo"].replace("/", "__")
        extra = {n: subprocess.run(["git", "-C", str(repo_dir), "show", f"{env['source_commit']}:{n}"], capture_output=True).stdout
                 for n in ("go.mod", "go.sum")}
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for n, data in extra.items():
                if data:
                    ti = tarfile.TarInfo(n); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
        key_ = task_key(env)
        r2 = sb._run(env, buf.getvalue(), [(["go", "mod", "download"], command_env(None, WORK, lang, key_, network=True), None)],
                     network=True, timeout=1800)
        r = list(r) + [x[:4] for x in r2]
        rec.update(go=sb.specs[task_id]["version"], prepare=[x[:3] for x in r])
    rec["ok"] = all(x[0] == 0 for x in r)
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["prepare_tail"] = [(x[2] or x[1])[-300:] for x in r if x[0] != 0]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare"])
    ap.add_argument("--task", action="append", required=True)
    a = ap.parse_args()
    out = []
    for t in a.task:
        r = prepare_task(t)
        print(json.dumps({k: v for k, v in r.items() if k != "prepare"}), flush=True)
        out.append(r)
    path = EXP / "modal_prepare.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    for r in out:
        old[r["task_id"]] = r
    path.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
