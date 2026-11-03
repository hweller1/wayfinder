"""Create the Modal secrets for wayfinder-glimmer and the Scout gateway in your own account. Never prints a value.

    export VOYAGE_API_KEY=... TURBOPUFFER_API_KEY=...
    python deploy/modal/make_secrets.py --workspace <your-modal-workspace>        # both secrets
    python deploy/modal/make_secrets.py --llm-url https://<ws>--wayfinder-glimmer-serve.modal.run --only gateway

Two secrets result:

  wayfinder-glimmer-key   WAYFINDER_API_KEY, the bearer key of the model endpoint (serve_glimmer.py).
  scout-gateway           VOYAGE_API_KEY and TURBOPUFFER_API_KEY (from the process environment, or from
                          `--env-file`), SCOUT_MASTER_KEY and SCOUT_PEPPER (32 random bytes each, base64),
                          SCOUT_KEYS_NS, and the chat upstream: SCOUT_LLM_URL (your serve_glimmer endpoint,
                          with /v1), SCOUT_LLM_API_KEY (the same bearer key), SCOUT_LLM_MODEL=wayfinder-glimmer,
                          SCOUT_LLM_SAMPLING=sft.

The generated values (the master key, the pepper and the model API key) are created once and kept in
`~/.config/scout/modal-keys.json`, mode 0600, so a re-run or a `--force` keeps them. Keep that file: the master
key and pepper decide whether a gateway can read an index it wrote earlier, and losing them means re-embedding
every repository. `--workspace` is the workspace name in `modal profile list`; the endpoint URL is derived from
it, or pass the URL the deploy printed with `--llm-url` (needed when the app lives in a non-default Modal
environment). Refuses to overwrite an existing secret without `--force`. Values reach `modal secret create`
through a 0600 temporary JSON file (`--from-json`), never on a command line or on stdout, and the file is removed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

MODEL_SECRET = "wayfinder-glimmer-key"  # noqa: S105  (a Modal secret NAME)
GATEWAY_SECRET = "scout-gateway"  # noqa: S105
LOCAL_KEYS = Path.home() / ".config" / "scout" / "modal-keys.json"
GENERATED = ("SCOUT_MASTER_KEY", "SCOUT_PEPPER", "WAYFINDER_API_KEY")
DEFAULT_KEYS_NS = "scoutsys_keys"
DEFAULT_APP = "wayfinder-glimmer-serve"
SERVED_NAME = "wayfinder-glimmer"
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_WORKSPACE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def parse_env_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        m = _LINE.match(raw)
        if m and not raw.lstrip().startswith("#"):
            value = m.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                out.setdefault(m.group(1), value)
    return out


def load_env(env_file: Path | None = None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """An optional env file, overlaid by the process environment."""
    merged: dict[str, str] = {}
    if env_file is not None:
        try:
            merged.update(parse_env_text(env_file.read_text(encoding="utf-8")))
        except OSError:
            raise SystemExit(f"cannot read --env-file {env_file}") from None
    merged.update(os.environ if base is None else base)
    return merged


def _b64_random(n: int = 32) -> str:
    return base64.b64encode(secrets.token_bytes(n)).decode()


def _fresh(name: str) -> str:
    return "sk-wayfinder-" + secrets.token_urlsafe(32) if name == "WAYFINDER_API_KEY" else _b64_random()


def local_keys(path: Path | None = None) -> dict[str, str]:
    """The generated values, created once and stored 0600. A file that lacks one of them keeps the others."""
    path = LOCAL_KEYS if path is None else path
    data: dict[str, str] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            raise SystemExit(f"{path} is not valid JSON; fix or remove it") from None
        data = {k: v for k, v in loaded.items() if isinstance(v, str) and v}
    if all(k in data for k in GENERATED):
        return {k: data[k] for k in GENERATED}
    for k in GENERATED:
        data.setdefault(k, _fresh(k))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return {k: data[k] for k in GENERATED}


def llm_url(workspace: str | None, explicit: str | None) -> str:
    """The OpenAI-compatible base URL of the serve app, ending in /v1."""
    if explicit:
        url = explicit.strip().rstrip("/")
        if not url.startswith("https://"):
            raise SystemExit("--llm-url must start with https://")
    else:
        if not workspace or not _WORKSPACE.match(workspace):
            raise SystemExit("pass --workspace <modal workspace name> (see `modal profile list`) or --llm-url")
        url = f"https://{workspace}--{DEFAULT_APP}.modal.run"
    return url if url.endswith("/v1") else url + "/v1"


def model_values(keys: Mapping[str, str]) -> dict[str, str]:
    return {"WAYFINDER_API_KEY": keys["WAYFINDER_API_KEY"]}


def gateway_values(env: Mapping[str, str], keys: Mapping[str, str], url: str) -> dict[str, str]:
    missing = [k for k in ("VOYAGE_API_KEY", "TURBOPUFFER_API_KEY") if not env.get(k)]
    if missing:
        raise SystemExit("missing variables: " + ", ".join(missing))
    return {
        "TURBOPUFFER_API_KEY": env["TURBOPUFFER_API_KEY"],
        "VOYAGE_API_KEY": env["VOYAGE_API_KEY"],
        "SCOUT_MASTER_KEY": keys["SCOUT_MASTER_KEY"],
        "SCOUT_PEPPER": keys["SCOUT_PEPPER"],
        "SCOUT_KEYS_NS": env.get("SCOUT_KEYS_NS") or DEFAULT_KEYS_NS,
        "SCOUT_LLM_URL": url,
        "SCOUT_LLM_API_KEY": keys["WAYFINDER_API_KEY"],
        "SCOUT_LLM_MODEL": SERVED_NAME,
        "SCOUT_LLM_SAMPLING": "sft",
    }


def existing_secrets(modal_bin: str) -> set[str]:
    res = subprocess.run(  # noqa: S603
        [modal_bin, "secret", "list", "--json"], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        raise SystemExit(f"`modal secret list` failed (exit {res.returncode}): is modal set up? run `modal setup`")
    try:
        return {str(s["name"]) for s in json.loads(res.stdout)}
    except (ValueError, KeyError, TypeError):
        raise SystemExit("`modal secret list --json` returned something unreadable") from None


def create_secret(modal_bin: str, name: str, values: Mapping[str, str], *, force: bool) -> None:
    fd, tmp = tempfile.mkstemp(prefix="wayfinder-secret-", suffix=".json")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(values), f)
        cmd = [modal_bin, "secret", "create", name, "--from-json", tmp]
        if force:
            cmd.append("--force")
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if res.returncode != 0:
            raise SystemExit(f"`modal secret create {name}` failed (exit {res.returncode})")
    finally:
        os.unlink(tmp)


def main(
    argv: list[str] | None = None, *, env: Mapping[str, str] | None = None, modal_bin: str = "modal"
) -> int:
    ap = argparse.ArgumentParser(description="Create the Modal secrets for wayfinder-glimmer and the Scout gateway.")
    ap.add_argument("--workspace", help="your Modal workspace name (modal profile list); derives the endpoint URL")
    ap.add_argument("--llm-url", help="the endpoint URL `modal deploy` printed for serve_glimmer.py")
    ap.add_argument("--only", choices=("model", "gateway"), help="create just one of the two secrets")
    ap.add_argument("--env-file", type=Path, help="a file with VOYAGE_API_KEY and TURBOPUFFER_API_KEY")
    ap.add_argument("--force", action="store_true", help="overwrite existing secrets")
    args = ap.parse_args(argv)
    want_model = args.only in (None, "model")
    want_gateway = args.only in (None, "gateway")
    environment = load_env(args.env_file, env)
    url = llm_url(args.workspace, args.llm_url) if want_gateway else ""
    wanted = ([MODEL_SECRET] if want_model else []) + ([GATEWAY_SECRET] if want_gateway else [])
    gateway = gateway_values(environment, local_keys(), url) if want_gateway else {}  # fails before any create
    present = existing_secrets(modal_bin)
    clash = sorted(set(wanted) & present)
    if clash and not args.force:
        sys.stdout.write(f"refusing to overwrite existing secret(s) {', '.join(clash)}; pass --force\n")
        return 1
    keys = local_keys()
    if want_model:
        create_secret(modal_bin, MODEL_SECRET, model_values(keys), force=args.force)
        sys.stdout.write(f"{MODEL_SECRET}: WAYFINDER_API_KEY\n")
    if want_gateway:
        create_secret(modal_bin, GATEWAY_SECRET, gateway, force=args.force)
        sys.stdout.write(f"{GATEWAY_SECRET}: {', '.join(sorted(gateway))}\n")
        sys.stdout.write(f"model endpoint: {url}\n")
    sys.stdout.write(f"generated keys are kept in {LOCAL_KEYS}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
