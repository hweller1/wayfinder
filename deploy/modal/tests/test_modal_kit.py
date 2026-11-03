"""Offline checks of the Modal kit: no network, no Modal credentials, no GPU, no Modal resource is created.

    cd scout && uv run --extra gateway python -m pytest ../deploy/modal/tests -q -p no:cacheprovider
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest

MODAL_DIR = Path(__file__).resolve().parents[1]
REPO = MODAL_DIR.parents[1]

# Copied from the app that served every published result (wayfinder_v3.py `VLLM_ARGS`), with the served name
# `wayfinder-glimmer`. Do not import it from anywhere: this list is the contract.
REFERENCE_VLLM_ARGS = [
    "--served-model-name", "wayfinder-glimmer",
    "--dtype", "bfloat16",
    "--max-model-len", "65536",
    "--gpu-memory-utilization", "0.92",
    "--enable-prefix-caching",
    "--language-model-only",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "muse_glimmer",
    "--reasoning-parser", "muse_glimmer",
    "--max-num-seqs", "32",
    "--host", "0.0.0.0",
]  # fmt: skip


def _load(name: str, monkeypatch: pytest.MonkeyPatch):
    for k in list(os.environ):
        if k.startswith("WAYFINDER_") or k.startswith("MODAL_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(MODAL_DIR / "no-such-modal.toml"))  # no profile, no token
    spec = importlib.util.spec_from_file_location(name, MODAL_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("modal")
    return _load("serve_glimmer", monkeypatch)


@pytest.fixture
def secrets_mod(monkeypatch: pytest.MonkeyPatch):
    return _load("make_secrets", monkeypatch)


# --- serve_glimmer.py ---------------------------------------------------------------------------------


def test_serve_module_imports_without_modal_credentials(serve) -> None:
    assert serve.app.name == "wayfinder-glimmer-serve"
    assert serve.SECRET_NAME == "wayfinder-glimmer-key"
    assert serve.MODEL_ID == "hweller1/wayfinder-glimmer-30b"


def test_vllm_flags_equal_the_reference_app(serve) -> None:
    assert serve.VLLM_ARGS == REFERENCE_VLLM_ARGS
    assert serve.VLLM_VERSION == "0.32.0"
    assert serve.SERVED_NAME == "wayfinder-glimmer"
    assert "--chat-template" not in serve.VLLM_ARGS  # the checkpoint's own template


def test_vllm_command_serves_the_weights_directory_on_the_public_port(serve) -> None:
    cmd = serve.vllm_command("/models/wayfinder-glimmer-30b")
    assert cmd[:3] == ["vllm", "serve", "/models/wayfinder-glimmer-30b"]
    assert cmd[3:-2] == REFERENCE_VLLM_ARGS
    assert cmd[-2:] == ["--port", "8000"]
    assert "--api-key" not in cmd  # the key travels in VLLM_API_KEY, never on the command line


def test_defaults_scale_to_zero_with_one_gpu(serve) -> None:
    assert (serve.MIN_CONTAINERS, serve.MAX_CONTAINERS, serve.SCALEDOWN_S) == (0, 1, 600)


def test_gpu_image_and_weights_source_in_the_source_text() -> None:
    text = (MODAL_DIR / "serve_glimmer.py").read_text()
    assert 'gpu="H100"' in text
    assert "nvidia/cuda:12.9.1-devel-ubuntu22.04" in text
    assert 'f"vllm=={VLLM_VERSION}"' in text
    assert "snapshot_download(MODEL_ID" in text
    assert 'secrets=[modal.Secret.from_name(SECRET_NAME)]' in text
    assert "hmac.compare_digest" in text  # the warm endpoint checks the same bearer key


def test_serve_app_declares_the_class_warm_endpoint_and_nothing_account_specific(serve) -> None:
    text = (MODAL_DIR / "serve_glimmer.py").read_text()
    assert callable(serve.warm) or hasattr(serve.warm, "get_web_url")  # the /warm ASGI function
    assert "class Glimmer:" in text and "@modal.web_server(port=PORT" in text
    assert 'label=f"{URL_LABEL}-warm"' in text
    for needle in ("/Users/", "wayfinder-v3", "wayfinder-data", "wayfinder-cache", "A1_API_KEY", "V3_"):
        assert needle not in text, needle


def test_env_overrides_apply_at_deploy_time(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("modal")
    _load("serve_glimmer", monkeypatch)  # clears WAYFINDER_* and the Modal config; then set our own
    monkeypatch.setenv("WAYFINDER_MIN_CONTAINERS", "1")
    monkeypatch.setenv("WAYFINDER_MAX_CONTAINERS", "2")
    monkeypatch.setenv("WAYFINDER_APP_NAME", "my-glimmer")
    spec = importlib.util.spec_from_file_location("serve_glimmer_env", MODAL_DIR / "serve_glimmer.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (mod.MIN_CONTAINERS, mod.MAX_CONTAINERS, mod.APP_NAME, mod.URL_LABEL) == (1, 2, "my-glimmer", "my-glimmer")


# --- make_secrets.py ----------------------------------------------------------------------------------

SHIM = """#!/usr/bin/env python3
import json, os, sys
state = os.environ["FAKE_MODAL_STATE"]
args = sys.argv[1:]
log = json.load(open(state)) if os.path.exists(state) else {"existing": [], "calls": [], "created": {}, "modes": {}}
log["calls"].append(args)
if args[:2] == ["secret", "list"]:
    print(json.dumps([{"name": n} for n in log["existing"]]))
elif args[:2] == ["secret", "create"]:
    name = args[2]
    if name in log["existing"] and "--force" not in args:
        print("exists", file=sys.stderr); sys.exit(1)
    path = args[args.index("--from-json") + 1]
    log["created"][name] = json.load(open(path))
    log["modes"][name] = oct(os.stat(path).st_mode & 0o777)
    if name not in log["existing"]:
        log["existing"].append(name)
    print("Created secret " + name)
json.dump(log, open(state, "w"))
"""
FAKE = {
    "TURBOPUFFER_API_KEY": "tpuf-canary-aaaa1111",
    "VOYAGE_API_KEY": "voyage-canary-bbbb2222",
}


@pytest.fixture
def rig(secrets_mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "modal"
    shim.write_text(SHIM)
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    state = tmp_path / "state.json"
    monkeypatch.setenv("FAKE_MODAL_STATE", str(state))
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(secrets_mod, "LOCAL_KEYS", tmp_path / "cfg" / "modal-keys.json")
    return secrets_mod, state, tmp_path


def _run(mod, args: list[str], env: dict[str, str] | None = None) -> int:
    return int(mod.main(args, env=FAKE if env is None else env, modal_bin="modal"))


def test_helper_creates_both_secrets_and_never_writes_a_value_to_stdout(rig, capsys) -> None:
    mod, state, tmp = rig
    assert _run(mod, ["--workspace", "my-ws"]) == 0
    out = capsys.readouterr()
    log = json.loads(state.read_text())
    model, gw = log["created"]["wayfinder-glimmer-key"], log["created"]["scout-gateway"]
    assert set(model) == {"WAYFINDER_API_KEY"}
    assert set(gw) == {
        "TURBOPUFFER_API_KEY", "VOYAGE_API_KEY", "SCOUT_MASTER_KEY", "SCOUT_PEPPER", "SCOUT_KEYS_NS",
        "SCOUT_LLM_URL", "SCOUT_LLM_API_KEY", "SCOUT_LLM_MODEL", "SCOUT_LLM_SAMPLING",
    }  # fmt: skip
    assert gw["SCOUT_LLM_URL"] == "https://my-ws--wayfinder-glimmer-serve.modal.run/v1"
    assert gw["SCOUT_LLM_MODEL"] == "wayfinder-glimmer" and gw["SCOUT_LLM_SAMPLING"] == "sft"
    assert gw["SCOUT_LLM_API_KEY"] == model["WAYFINDER_API_KEY"]
    assert gw["TURBOPUFFER_API_KEY"] == FAKE["TURBOPUFFER_API_KEY"]
    # no value, from the environment, the generated keys or the secrets, reaches stdout or stderr
    shown = out.out + out.err
    for v in [*FAKE.values(), *model.values(), gw["SCOUT_MASTER_KEY"], gw["SCOUT_PEPPER"]]:
        assert v not in shown
    # and none appears on a modal command line
    for call in log["calls"]:
        assert not any(v in " ".join(call) for v in [*FAKE.values(), *model.values()])
    assert log["modes"] == {"wayfinder-glimmer-key": "0o600", "scout-gateway": "0o600"}


def test_generated_keys_are_kept_in_a_0600_file_and_reused(rig) -> None:
    mod, state, tmp = rig
    _run(mod, ["--workspace", "my-ws"])
    first = json.loads(state.read_text())["created"]["scout-gateway"]
    keyfile = tmp / "cfg" / "modal-keys.json"
    assert stat.S_IMODE(keyfile.stat().st_mode) == 0o600
    assert _run(mod, ["--workspace", "my-ws", "--force"]) == 0
    again = json.loads(state.read_text())["created"]["scout-gateway"]
    assert again["SCOUT_MASTER_KEY"] == first["SCOUT_MASTER_KEY"]
    assert again["SCOUT_PEPPER"] == first["SCOUT_PEPPER"]
    assert again["SCOUT_LLM_API_KEY"] == first["SCOUT_LLM_API_KEY"]
    import base64

    assert len(base64.b64decode(first["SCOUT_MASTER_KEY"])) == 32
    assert len(base64.b64decode(first["SCOUT_PEPPER"])) == 32


def test_helper_refuses_to_overwrite_without_force(rig, capsys) -> None:
    mod, state, _ = rig
    assert _run(mod, ["--workspace", "my-ws"]) == 0
    capsys.readouterr()
    assert _run(mod, ["--workspace", "my-ws"]) == 1
    assert "refusing to overwrite" in capsys.readouterr().out


def test_helper_only_gateway_with_explicit_url_and_missing_keys_fail_before_any_create(rig) -> None:
    mod, state, _ = rig
    with pytest.raises(SystemExit, match="VOYAGE_API_KEY"):
        _run(mod, ["--workspace", "my-ws"], env={"TURBOPUFFER_API_KEY": "x"})
    assert not state.exists()  # nothing was listed or created
    assert _run(mod, ["--only", "gateway", "--llm-url", "https://ws--custom.modal.run/"]) == 0
    log = json.loads(state.read_text())
    assert set(log["created"]) == {"scout-gateway"}
    assert log["created"]["scout-gateway"]["SCOUT_LLM_URL"] == "https://ws--custom.modal.run/v1"


def test_helper_needs_a_workspace_or_url_for_the_gateway(rig) -> None:
    mod, state, _ = rig
    with pytest.raises(SystemExit, match="--workspace"):
        _run(mod, [])
    with pytest.raises(SystemExit, match="https://"):
        _run(mod, ["--llm-url", "http://insecure.example"])


def test_modal_failure_message_carries_no_value(rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod, state, _ = rig
    bad = tmp_path / "bin" / "modal"
    bad.write_text("#!/bin/sh\necho 'boom tpuf-canary-aaaa1111' >&2\nexit 1\n")
    with pytest.raises(SystemExit) as e:
        _run(mod, ["--workspace", "my-ws"])
    assert "tpuf-canary" not in str(e.value)


def test_no_owner_names_anywhere_in_the_kit() -> None:
    pattern = re.compile(r"/Users/|wayfinder-v3|wayfinder-data|\.modal\.run/v1/chat|mongodb|10gen|henryweller", re.I)
    for p in sorted(MODAL_DIR.rglob("*.py")):
        if p.name == "test_modal_kit.py":
            continue
        assert not pattern.search(p.read_text()), p
