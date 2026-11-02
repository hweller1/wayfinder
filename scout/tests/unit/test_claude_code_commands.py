# ruff: noqa: S603, S607, E501  (fixed argv in tests; long fixture lines)
"""lib/scout_cc.py as the slash commands run it: /scout:status, /scout:index, /scout:model, /scout:mode,
doctor, statusline, gateway-url. Stub `scout` binaries, a scratch HOME, no network except a local closed port."""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "integrations" / "claude-code"
LIB = PLUGIN / "lib" / "scout_cc.py"
LAUNCH = PLUGIN / "scripts" / "launch-scout.sh"
V3 = "https://scout-gateway.example.com"
A1 = "https://scout-gateway-a1.example.com"
KEY = "sk_scout_secret_value_123"  # noqa: S105  (a test value)


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Proj:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)  # noqa: S603, S607
        (self.repo / "main.go").write_text("package main\n" * 70)
        (self.repo / "big.bin").write_bytes(b"x" * 2_000_000)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)  # noqa: S603, S607
        subprocess.run(  # noqa: S603
            ["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"],  # noqa: S607
            check=True,
        )
        self.head = subprocess.run(  # noqa: S603
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,  # noqa: S607
        ).stdout.strip()
        self.state = self.repo / ".git" / "scout"
        self.state.mkdir()
        self.home = tmp / "home"
        self.home.mkdir()
        self.data = tmp / "data"

    def status(self, **fields: object) -> None:
        doc = {"schema": "1.0", "state": "ready", "coverage": 1.0, "commit": self.head,
               "last_sync_at": iso(time.time() - 120), "gateway_ok_at": iso(time.time() - 30), **fields}  # fmt: skip
        (self.state / "status.json").write_text(json.dumps(doc))

    def run(self, *argv: str, key: bool = True, extra: dict[str, str] | None = None, stdin: str = "") -> str:
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.home),
            "SCOUT_KEYSTORE": "file",
            "CLAUDE_PROJECT_DIR": str(self.repo),
            **({"SCOUT_KEY": KEY} if key else {}),
            **(extra or {}),
        }
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(LIB), *argv, "--data", str(self.data)],
            input=stdin, env=env, capture_output=True, text=True, timeout=60, check=False,
        )  # fmt: skip
        assert proc.returncode == 0, proc.stderr
        assert KEY not in proc.stdout + proc.stderr
        return proc.stdout


@pytest.fixture
def proj(tmp_path: Path) -> Proj:
    return Proj(tmp_path)


def stub_scout(tmp: Path, sync_out: str, delay: float = 0.0) -> Path:
    s = tmp / "bin" / "scout"
    s.parent.mkdir(exist_ok=True)
    s.write_text(f"#!/bin/sh\necho \"$@\" >> {tmp}/scout_calls.log\nsleep {delay}\necho '{sync_out}'\n")
    s.chmod(0o755)
    return s


# --- /scout:status ---------------------------------------------------------------------------------
def test_status_without_key_or_index(proj: Proj) -> None:
    out = proj.run("status", key=False)
    assert "lane       demo" in out
    assert "not configured" in out
    assert "none for this checkout - run /scout:index" in out
    assert "Explore runs built-in" in out


def test_status_ready_production(proj: Proj) -> None:
    proj.status()
    (proj.state / "mcp.pid").write_text(json.dumps({"pid": os.getpid()}))
    out = proj.run("status")
    assert "lane       production   key configured" in out
    assert "model      v3 - wayfinder-glimmer" in out
    assert f"gateway    {V3}" in out
    assert "index      ready 100%, at HEAD" in out
    assert "mcp server up" in out
    assert "Explore -> scout:scout-explorer (production lane)" in out
    assert "endpoint   unknown" in out


def test_status_reports_stale_commit_and_last_error(proj: Proj) -> None:
    proj.status(commit="0" * 40, last_error="UPSTREAM_UNAVAILABLE: x", syncing=True)
    out = proj.run("status")
    assert "indexed 00000000, HEAD " + proj.head[:8] in out
    assert "(sync running)" in out
    assert "last error UPSTREAM_UNAVAILABLE: x" in out


@pytest.mark.parametrize(
    ("rec", "label"),
    [
        (
            {"state": "warm", "finished_at": "NOW-60", "seconds": 212.0},
            "endpoint   warm - model answered 60s ago, took 212s",
        ),
        ({"state": "warm", "finished_at": "NOW-3600"}, "endpoint   cold?"),
        (
            {"state": "warming", "pid": "SELF", "started_at": "NOW-30"},
            "endpoint   warming - warm-up running for 30s",
        ),
        ({"state": "unreachable", "finished_at": "NOW-5"}, "endpoint   unreachable"),
        (
            {"state": "failed", "finished_at": "NOW-5", "error": "UPSTREAM_TIMEOUT"},
            "failed (UPSTREAM_TIMEOUT)",
        ),
    ],
)
def test_status_endpoint_warm_or_cold(proj: Proj, rec: dict, label: str) -> None:
    now = time.time()
    fixed = {k: (now - float(v[4:]) if isinstance(v, str) and v.startswith("NOW-") else os.getpid() if v == "SELF" else v)
             for k, v in rec.items()}  # fmt: skip
    proj.data.mkdir()
    (proj.data / "warm.json").write_text(json.dumps({V3: fixed}))
    proj.status()
    assert label in proj.run("status")


def test_status_outside_git(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(LIB), "status", "--project", str(plain), "--data", str(tmp_path / "d")],
        capture_output=True, text=True, check=True, env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )  # fmt: skip
    assert "not inside a git checkout" in proc.stdout


# --- /scout:model and /scout:mode ---------------------------------------------------------------------
def test_model_switch_is_per_project_and_feeds_the_launcher(proj: Proj) -> None:
    assert proj.run("gateway-url") == ""  # default v3: the launcher leaves SCOUT_GATEWAY_URL alone
    out = proj.run("model", "a1", extra={"SCOUT_NO_KEYCHAIN_PROBE": "1"})
    assert "a1 - Agents-A1-35B" in out
    assert A1 in out
    assert "Reconnect" in out
    assert proj.run("gateway-url").strip() == A1
    assert "model      a1 - Agents-A1-35B   (model profile)" in proj.run("status")
    proj.run("model", "v3")
    assert proj.run("gateway-url").strip() == V3


def test_glimmer_alias_selects_the_v3_profile(proj: Proj) -> None:
    proj.run("model", "a1")
    out = proj.run("model", "glimmer")
    assert "wayfinder-glimmer" in out
    assert proj.run("gateway-url").strip() == V3
    assert "model      v3 - wayfinder-glimmer" in proj.run("status")
    assert proj.run("gateway-url", extra={"SCOUT_MODEL": "a1"}).strip() == A1
    assert proj.run("gateway-url", extra={"SCOUT_MODEL": "glimmer"}).strip() == V3
    assert "glimmer" in proj.run("model", "gpt")  # help text names the new alias


def test_model_without_argument_and_unknown(proj: Proj) -> None:
    assert "scout model: v3 (default)" in proj.run("model")
    assert "unknown 'gpt'" in proj.run("model", "gpt")
    assert not (proj.data / "settings.json").exists()


def test_explicit_gateway_url_wins_and_is_reported(proj: Proj) -> None:
    out = proj.run("model", "a1", extra={"SCOUT_GATEWAY_URL": V3})
    assert "SCOUT_GATEWAY_URL is set in this session and wins" in out


def test_env_model_overrides_the_setting(proj: Proj) -> None:
    proj.run("model", "v3")
    assert proj.run("gateway-url", extra={"SCOUT_MODEL": "a1"}).strip() == A1


def test_mode_switch(proj: Proj) -> None:
    assert "hybrid  Explore -> scout-explorer" in proj.run("mode")
    assert "scout mode for this project: fast" in proj.run("mode", "fast")
    proj.status()
    (proj.state / "mcp.pid").write_text(json.dumps({"pid": os.getpid()}))
    assert "Explore -> scout:super-explore" in proj.run("status")


def test_launcher_exports_the_profile_gateway(proj: Proj, tmp_path: Path) -> None:
    probe = tmp_path / "probe"
    probe.write_text('#!/bin/sh\necho "url=$SCOUT_GATEWAY_URL"\n')
    probe.chmod(0o755)
    proj.run("model", "a1")
    env = {"PATH": "/usr/bin:/bin:" + str(Path(sys.executable).parent), "HOME": str(proj.home), "SCOUT_KEY": KEY,
           "SCOUT_BIN": str(probe), "SCOUT_KEYSTORE": "file", "CLAUDE_PLUGIN_DATA": str(proj.data)}  # fmt: skip
    out = subprocess.run(
        ["sh", str(LAUNCH), "mcp"], cwd=proj.repo, env=env, capture_output=True, text=True, check=True
    )  # noqa: S603, S607
    assert out.stdout.strip() == f"url={A1}"
    env["SCOUT_GATEWAY_URL"] = "http://explicit"
    out = subprocess.run(
        ["sh", str(LAUNCH), "mcp"], cwd=proj.repo, env=env, capture_output=True, text=True, check=True
    )  # noqa: S603, S607
    assert out.stdout.strip() == "url=http://explicit"


# --- /scout:index -----------------------------------------------------------------------------------
def test_index_without_production_lane_explains(proj: Proj) -> None:
    out = proj.run("index", key=False)
    assert "production lane not available" in out
    assert "scout login" in out


def test_index_runs_sync_and_reports_estimate_and_result(proj: Proj, tmp_path: Path) -> None:
    res = json.dumps({"state": "ready", "coverage": 1.0, "files": 1, "commit": proj.head})
    scout = stub_scout(tmp_path, res, delay=0.3)
    out = proj.run("index", "--wait-s", "20", extra={"SCOUT_BIN": str(scout), "SCOUT_NO_WARM": "1"})
    assert "estimate 1 files" in out  # big.bin (> 1 MiB) is not counted
    assert "~$" in out
    assert "sync started" in out
    assert "done in" in out
    assert "index ready 100%, 1 files, commit " + proj.head[:8] in out
    calls = (tmp_path / "scout_calls.log").read_text().splitlines()
    assert calls[0] == f"sync --wait --root {os.path.realpath(proj.repo)}"
    assert "warming" not in out  # SCOUT_NO_WARM is obeyed by /scout:index too
    assert not (proj.data / "warm.json").exists()


def test_index_reports_a_sync_error(proj: Proj, tmp_path: Path) -> None:
    err = json.dumps({"schema_version": "1.0", "error": {"code": "UNAUTHENTICATED", "message": "bad key"}})
    scout = stub_scout(tmp_path, err)
    out = proj.run("index", "--wait-s", "20", extra={"SCOUT_BIN": str(scout)})
    assert "sync failed: UNAUTHENTICATED: bad key" in out


def test_index_that_outlives_the_wait_continues_in_background(proj: Proj, tmp_path: Path) -> None:
    scout = stub_scout(tmp_path, "{}", delay=5)
    out = proj.run("index", "--wait-s", "1", extra={"SCOUT_BIN": str(scout)})
    assert "continues in the background" in out
    again = proj.run("index", "--wait-s", "0", extra={"SCOUT_BIN": str(scout)})
    assert "a sync is already running" in again
    assert len((tmp_path / "scout_calls.log").read_text().splitlines()) == 1


# --- doctor and statusline ------------------------------------------------------------------------------
def test_doctor_reports_without_the_key(proj: Proj, tmp_path: Path) -> None:
    scout = stub_scout(tmp_path, "{}")
    doc = json.loads(
        proj.run("doctor", extra={"SCOUT_BIN": str(scout), "SCOUT_GATEWAY_URL": "http://127.0.0.1:9"})
    )
    assert doc["key_configured"] is True
    assert doc["scout_binary"] == str(scout)
    assert doc["lane"] == "production"
    assert doc["gateway_healthz"] != "ok"
    assert doc["ok"] is False


def test_statusline_is_compact(proj: Proj) -> None:
    payload = json.dumps({"workspace": {"current_dir": str(proj.repo)}})
    assert proj.run("statusline", stdin=payload).strip() == "scout: not indexed"
    proj.status()
    assert proj.run("statusline", stdin=payload).strip() == "scout v3 ready 100% unknown hybrid"
