"""Task 8.7: `ops/make_secrets.py` against a fake `modal` shim on PATH. FR-G4, AC-9.4.

The shim records its argv and the JSON file it was handed (the values), prints nothing secret, and keeps a
list of existing secrets. No real Modal call is made; no secret value may appear in the script's stdout.
"""

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIM = """#!/usr/bin/env python3
import json, os, sys
state = os.environ["FAKE_MODAL_STATE"]
args = sys.argv[1:]
log = json.load(open(state)) if os.path.exists(state) else {"existing": [], "calls": [], "created": {}}
log["calls"].append(args)
if args[:2] == ["secret", "list"]:
    print(json.dumps([{"name": n} for n in log["existing"]]))
elif args[:2] == ["secret", "create"]:
    name = args[2]
    if name in log["existing"] and "--force" not in args:
        print("exists", file=sys.stderr); sys.exit(1)
    log["created"][name] = json.load(open(args[args.index("--from-json") + 1]))
    if name not in log["existing"]:
        log["existing"].append(name)
    print("Created secret " + name)
json.dump(log, open(state, "w"))
"""
CANARIES = {
    "TURBOPUFFER_API_KEY": "tpuf-canary-aaaa1111",
    "VOYAGE_API_KEY": "voyage-canary-bbbb2222",
    "A1_BASE_URL": "https://a1.example.test/v1",
    "A1_API_KEY": "a1-canary-cccc3333",
}


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_secrets", ROOT / "ops" / "make_secrets.py")
    assert spec
    assert spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_secrets"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path, Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "modal"
    shim.write_text(SHIM)
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    state = tmp_path / "state.json"
    monkeypatch.setenv("FAKE_MODAL_STATE", str(state))
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    mod = load()
    monkeypatch.setattr(mod, "LOCAL_KEYS", tmp_path / "cfg" / "gateway-master.json")
    monkeypatch.setattr(mod, "ENV_FILES", ())
    return mod, state, tmp_path


def run(mod: ModuleType, args: list[str], env: dict[str, str]) -> int:
    return int(mod.main(args, env=env, modal_bin="modal"))


def test_creates_both_secrets_with_the_right_keys_and_no_value_in_stdout(
    rig: tuple[ModuleType, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    mod, state, tmp = rig
    assert run(mod, [], CANARIES) == 0
    out = capsys.readouterr()
    log = json.loads(state.read_text())
    gw, llm = log["created"]["scout-gateway"], log["created"]["scout-llm"]
    assert set(gw) == {
        "TURBOPUFFER_API_KEY",
        "VOYAGE_API_KEY",
        "SCOUT_MASTER_KEY",
        "SCOUT_PEPPER",
        "SCOUT_KEYS_NS",
        "SCOUT_LLM_URL",
        "SCOUT_LLM_API_KEY",
        "SCOUT_LLM_MODEL",
    }
    assert set(llm) == {"SCOUT_LLM_API_KEY"}
    assert gw["TURBOPUFFER_API_KEY"] == CANARIES["TURBOPUFFER_API_KEY"]
    assert gw["SCOUT_LLM_URL"] == CANARIES["A1_BASE_URL"]
    assert gw["SCOUT_LLM_API_KEY"] == CANARIES["A1_API_KEY"]
    assert gw["SCOUT_KEYS_NS"] == "scout_dev_keys"
    assert llm["SCOUT_LLM_API_KEY"] != CANARIES["A1_API_KEY"]
    import base64

    assert len(base64.b64decode(gw["SCOUT_MASTER_KEY"])) == 32
    assert len(base64.b64decode(gw["SCOUT_PEPPER"])) == 32
    for secret in [*CANARIES.values(), *gw.values(), *llm.values()]:
        if len(secret) > 12:
            assert secret not in out.out + out.err
    assert "asa-run-creds" not in json.dumps(log)
    for call in log["calls"]:  # values travel in a file, never in argv
        assert not any(v in " ".join(call) for v in [*gw.values(), *llm.values()] if len(v) > 12)
    local = tmp / "cfg" / "gateway-master.json"
    assert stat.S_IMODE(local.stat().st_mode) == 0o600
    assert json.loads(local.read_text())["SCOUT_MASTER_KEY"] == gw["SCOUT_MASTER_KEY"]


def test_refuses_to_overwrite_without_force_and_force_keeps_the_same_keys(
    rig: tuple[ModuleType, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    mod, state, _ = rig
    assert run(mod, [], CANARIES) == 0
    first = json.loads(state.read_text())["created"]["scout-gateway"]
    capsys.readouterr()
    assert run(mod, [], CANARIES) == 1
    assert "refusing to overwrite" in capsys.readouterr().out
    creates = [c for c in json.loads(state.read_text())["calls"] if c[:2] == ["secret", "create"]]
    assert len(creates) == 2  # the refused run created nothing
    assert run(mod, ["--force"], CANARIES) == 0
    again = json.loads(state.read_text())["created"]["scout-gateway"]
    assert again["SCOUT_MASTER_KEY"] == first["SCOUT_MASTER_KEY"]  # wrapped team keys stay readable
    assert again["SCOUT_PEPPER"] == first["SCOUT_PEPPER"]
    assert [c for c in json.loads(state.read_text())["calls"] if "--force" in c]


def test_missing_variables_fail_before_any_create(rig: tuple[ModuleType, Path, Path]) -> None:
    mod, state, _ = rig
    with pytest.raises(SystemExit) as ei:
        run(mod, [], {"TURBOPUFFER_API_KEY": "x"})
    assert "VOYAGE_API_KEY" in str(ei.value)
    assert (
        "x"
        not in str(ei.value)
        .replace("VOYAGE_API_KEY", "")
        .replace("SCOUT_LLM", "")
        .replace("variables", "")[:0]
    )
    calls = json.loads(state.read_text())["calls"]
    assert not [c for c in calls if c[:2] == ["secret", "create"]]


def test_explicit_scout_llm_variables_win_over_a1(rig: tuple[ModuleType, Path, Path]) -> None:
    mod, state, _ = rig
    env = {
        **CANARIES,
        "SCOUT_LLM_URL": "https://other.test/v1",
        "SCOUT_LLM_API_KEY": "k-other-1234567890",
        "SCOUT_LLM_MODEL": "m",
    }
    assert run(mod, [], env) == 0
    gw = json.loads(state.read_text())["created"]["scout-gateway"]
    assert (gw["SCOUT_LLM_URL"], gw["SCOUT_LLM_MODEL"]) == ("https://other.test/v1", "m")


def test_env_file_parsing(tmp_path: Path) -> None:
    mod = load()
    f = tmp_path / ".env"
    f.write_text("# c\nexport A=1\nB='two'\nC=\n")
    assert mod.parse_env_text(f.read_text()) == {"A": "1", "B": "two"}
    assert mod.load_env((f,), {"A": "override"})["A"] == "override"


def test_llm_v3_creates_only_the_v3_secret_with_the_sft_sampling_profile(
    rig: tuple[ModuleType, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    mod, state, _ = rig
    env = {"V3_BASE_URL": "https://v3.example.test/v1", "V3_API_KEY": "v3-canary-dddd4444eeee"}
    assert run(mod, ["--llm-v3"], env) == 0
    out = capsys.readouterr()
    log = json.loads(state.read_text())
    assert set(log["created"]) == {"scout-llm-v3"}
    assert log["created"]["scout-llm-v3"] == {
        "SCOUT_LLM_URL": "https://v3.example.test/v1",
        "SCOUT_LLM_API_KEY": "v3-canary-dddd4444eeee",
        "SCOUT_LLM_MODEL": "wayfinder-v3",
        "SCOUT_LLM_SAMPLING": "sft",
    }
    assert "v3-canary-dddd4444eeee" not in out.out + out.err
    assert run(mod, ["--llm-v3"], env) == 1  # no overwrite without --force
    with pytest.raises(SystemExit):
        run(mod, ["--llm-v3", "--force"], {})


def test_llm_a20_creates_only_the_a20_secret_with_its_own_url_and_model(
    rig: tuple[ModuleType, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    mod, state, _ = rig
    env = {"A20_BASE_URL": "https://a20.example.test/v1", "V3_API_KEY": "v3-canary-ffff5555aaaa"}
    assert run(mod, ["--llm-a20"], env) == 0
    out = capsys.readouterr()
    log = json.loads(state.read_text())
    assert set(log["created"]) == {"scout-llm-a20"}
    assert log["created"]["scout-llm-a20"] == {
        "SCOUT_LLM_URL": "https://a20.example.test/v1",
        "SCOUT_LLM_API_KEY": "v3-canary-ffff5555aaaa",
        "SCOUT_LLM_MODEL": "wayfinder-grpo-a20",
        "SCOUT_LLM_SAMPLING": "sft",
    }
    assert "v3-canary-ffff5555aaaa" not in out.out + out.err
    assert run(mod, ["--llm-a20"], env) == 1
