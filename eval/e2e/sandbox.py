"""Docker sandbox for E2E test execution. Agent-written code only ever runs inside these containers.

Per task: image pinned to the repo's toolchain (python via uv + the recorded lockfile; go = go.mod toolchain;
rust = rust-toolchain / rust-version). The worktree is bind-mounted at /work; test runs use `--network none`.
Dependencies for Go/Rust are fetched in a separate `prepare` step (network on, NO agent code involved beyond
the module manifests) into named volumes; the test run mounts them read-only-ish and goes offline.

  python scripts/e2e/sandbox.py status                 # is the daemon up?
  python scripts/e2e/sandbox.py build --task <id>      # build the task's image
  python scripts/e2e/sandbox.py prepare --task <id>    # fetch deps into the volumes (base commit manifests)

score.py --sandbox sets gate.EXEC = sandbox.Sandbox().exec, so every test command goes through here.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE  # re-pointed to wayfinder/eval/e2e
sys.path.insert(0, str(HERE))

DOCKER_DIR = "experiments/e2e/docker"
WORK = "/work"
GO_VOL, CARGO_VOL = "e2e-gomod", "e2e-cargo"
MEM, CPUS = "6g", "4"


class SandboxUnavailable(RuntimeError):
    pass


def default_runner(cmd, timeout=None, **kw):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    p.elapsed = time.time() - t0
    return p


def daemon_up(runner=default_runner) -> bool:
    try:
        return runner(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# --------------------------------------------------------------------------- toolchain pinning

def go_version(gomod: str) -> str:
    """Pinned Go version: `toolchain goX.Y.Z` if present, else the `go X.Y[.Z]` directive."""
    m = re.search(r"^toolchain\s+go(\d+\.\d+(?:\.\d+)?)", gomod, re.M) or re.search(r"^go\s+(\d+\.\d+(?:\.\d+)?)", gomod, re.M)
    if not m:
        raise ValueError("no go version in go.mod")
    return m.group(1)


def rust_version(toolchain_file: str | None, cargo_toml: str | None) -> str:
    """rust-toolchain(.toml) channel, else Cargo.toml rust-version (MSRV)."""
    if toolchain_file:
        t = toolchain_file.strip()
        m = re.search(r'channel\s*=\s*"([^"]+)"', t) or re.match(r"^(\d[\w.\-]*|stable|beta|nightly[\w\-]*)\s*$", t)
        if m:
            return m.group(1)
    if cargo_toml:
        m = re.search(r'^rust-version\s*=\s*"(\d+\.\d+(?:\.\d+)?)"', cargo_toml, re.M)
        if m:
            return m.group(1)
    raise ValueError("no rust version pin")


def _show(repo: Path, commit: str, path: str) -> str | None:
    p = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def image_spec(env: dict, show=None) -> dict:
    """{tag, dockerfile, build_args} for a task env record. `show(path)` reads a file at the base commit."""
    lang = env["language"]
    slug = env["repo"].replace("/", "__")
    if show is None:
        repo = ROOT / "data" / "e2e" / "repos" / slug
        show = lambda path: _show(repo, env["base_commit"], path)  # noqa: E731
    if lang == "Python":
        return dict(tag=f"e2e-py-{slug.lower()}", dockerfile=f"{DOCKER_DIR}/Dockerfile.python",
                    build_args={"PYTHON": env["config"].get("python", "3.12"), "LOCK": env["lockfile"]})
    if lang == "Go":
        v = go_version(show("go.mod") or "")
        return dict(tag=f"e2e-go-{v}", dockerfile=f"{DOCKER_DIR}/Dockerfile.go", build_args={"GO_VERSION": v})
    if lang == "Rust":
        v = rust_version(show("rust-toolchain") or show("rust-toolchain.toml"), show("Cargo.toml"))
        return dict(tag=f"e2e-rust-{v}", dockerfile=f"{DOCKER_DIR}/Dockerfile.rust", build_args={"RUST_VERSION": v})
    raise ValueError(lang)


def build_cmd(spec: dict) -> list[str]:
    c = ["docker", "build", "-f", spec["dockerfile"], "-t", spec["tag"]]
    for k, v in spec["build_args"].items():
        c += ["--build-arg", f"{k}={v}"]
    return c + ["."]


# --------------------------------------------------------------------------- running

def _map(s: str, host_wd: str) -> str:
    return s.replace(host_wd, WORK)


def container_argv(cmd: list[str], host_wd: str, lang: str) -> list[str]:
    """Translate a host test command into its in-container form (paths under the worktree -> /work; the host venv
    python -> the image's python)."""
    out = [_map(str(a), host_wd) for a in cmd]
    if lang == "Python" and re.search(r"/bin/python\d*$", out[0]):
        out[0] = "python"
    return out


def container_env(host_env: dict | None, host_wd: str, lang: str, network: bool = False) -> dict:
    e = {"CI": "1"}
    if lang == "Python":
        e["PYTHONDONTWRITEBYTECODE"] = "1"
        if host_env and host_env.get("PYTHONPATH"):
            e["PYTHONPATH"] = _map(host_env["PYTHONPATH"], host_wd)
    if lang == "Go":
        e.update(GOFLAGS="-mod=mod -buildvcs=false", GOMODCACHE="/gomod", GOCACHE="/gomod/.build", GOTOOLCHAIN="local")
        if not network:
            e["GOPROXY"] = "off"
    if lang == "Rust":
        e.update(CARGO_HOME="/cargo", CARGO_TARGET_DIR="/target")
        if not network:
            e["CARGO_NET_OFFLINE"] = "true"
    return e


def run_cmd(spec: dict, cmd: list[str], host_wd: str, lang: str, host_env: dict | None = None, network: bool = False,
            name: str | None = None) -> list[str]:
    c = ["docker", "run", "--rm", "--memory", MEM, "--cpus", CPUS, "--pids-limit", "1024"]
    if not network:
        c += ["--network", "none"]
    if name:
        c += ["--name", name]
    c += ["-v", f"{host_wd}:{WORK}", "-w", WORK]
    if lang == "Go":
        c += ["-v", f"{GO_VOL}:/gomod"]
    if lang == "Rust":
        c += ["-v", f"{CARGO_VOL}:/cargo", "-v", f"e2e-target-{spec['tag']}:/target"]
    for k, v in container_env(host_env, host_wd, lang, network).items():
        c += ["-e", f"{k}={v}"]
    return c + [spec["tag"], *container_argv(cmd, host_wd, lang)]


class Sandbox:
    """Callable replacement for gate.sh for test commands: `exec(cmd, cwd=, env=, timeout=)`."""

    def __init__(self, env_by_cwd=None, runner=default_runner, specs: dict | None = None):
        self.runner = runner
        self.current_env: dict | None = None     # task env record; set by score.py before running
        self.specs = specs or {}
        self.calls: list[list[str]] = []

    def use(self, env: dict, show=None):
        self.current_env = env
        self.specs[env["task_id"]] = image_spec(env, show)

    def exec(self, cmd, cwd=None, env=None, timeout=None, **_):
        if self.current_env is None:
            raise SandboxUnavailable("Sandbox.use(env) not called")
        lang = self.current_env["language"]
        spec = self.specs[self.current_env["task_id"]]
        full = run_cmd(spec, list(cmd), str(cwd), lang, env, network=False)
        self.calls.append(full)
        t0 = time.time()
        try:
            p = self.runner(full, timeout=timeout)
        except subprocess.TimeoutExpired:
            self.runner(["docker", "ps", "-q"], timeout=20)
            raise
        if not hasattr(p, "elapsed"):
            p.elapsed = time.time() - t0
        return p

    # one-time steps (network on; no agent-written code: base-commit manifests only)
    def build(self):
        spec = self.specs[self.current_env["task_id"]]
        return self.runner(build_cmd(spec), timeout=3600, cwd=str(ROOT))

    def prepare(self, host_wd: str):
        lang = self.current_env["language"]
        spec = self.specs[self.current_env["task_id"]]
        if lang == "Go":
            cmd = ["go", "mod", "download"]
            envv = {}
        elif lang == "Rust":
            cmd = ["cargo", "fetch"]
            envv = {}
        else:
            return None
        return self.runner(run_cmd(spec, cmd, host_wd, lang, envv, network=True), timeout=3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["status", "build", "prepare"])
    ap.add_argument("--task")
    a = ap.parse_args()
    if not daemon_up():
        print("DOCKER DAEMON NOT RUNNING: start Docker Desktop, then re-run.")
        sys.exit(2)
    if a.action == "status":
        print("daemon up")
        return
    import gate as g
    env = json.load(open(g.EXP / "envs" / f"{a.task}.json"))
    sb = Sandbox()
    sb.use(env)
    if a.action == "build":
        p = sb.build()
    else:
        task = g.load_tasks()[a.task]
        repo, wd = g.make_worktree(task, "prep_" + g.short(task))
        try:
            p = sb.prepare(str(wd))
        finally:
            g.drop_worktree(repo, wd)
    print((p.stdout or "")[-500:], (p.stderr or "")[-1500:])
    sys.exit(p.returncode if p else 0)


if __name__ == "__main__":
    main()
