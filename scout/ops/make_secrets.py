"""Create the Modal secrets `scout-gateway` and `scout-llm` (task 8.7). Never prints a secret value.

    set -a; . ./.env; set +a; uv run python ops/make_secrets.py [--force]

`scout-gateway` (the single auth point's credentials): TURBOPUFFER_API_KEY and VOYAGE_API_KEY (process env,
then `./.env`), SCOUT_MASTER_KEY and SCOUT_PEPPER (32 random bytes each, base64; generated once and kept in
`~/.config/scout/gateway-master.json`, mode 0600, so a re-run or a `--force` keeps the same keys and every
wrapped team key stays readable), SCOUT_KEYS_NS (`scout_dev_keys` by default for this deployment), and
the chat upstream SCOUT_LLM_URL / SCOUT_LLM_API_KEY / SCOUT_LLM_MODEL: the deployed A1-35B endpoint
(A1_BASE_URL / A1_API_KEY from the process env, `../wayfinder/.env` or `./.env`) unless SCOUT_LLM_* are
already set.
`--llm-v3` creates only the Modal secret `scout-llm-v3` (the wayfinder-v3 chat upstream: SCOUT_LLM_URL,
SCOUT_LLM_API_KEY, SCOUT_LLM_MODEL=wayfinder-v3, SCOUT_LLM_SAMPLING=sft from V3_BASE_URL / V3_API_KEY,
process env then `../wayfinder/.env`). `deploy/modal/gateway.py` appends it last when SCOUT_LLM_PROFILE=v3
(the default), so it overrides the A1 values in `scout-gateway`; SCOUT_LLM_PROFILE=a1 leaves it out and the
A1 path is unchanged.
`scout-llm` holds SCOUT_LLM_API_KEY (generated) for the future scout-llm GPU service; the gateway mounts both
and `scout-gateway` wins where they overlap. Modal proxy tokens are not used (the A1 endpoint authenticates
with its bearer key). Refuses to overwrite an existing secret without `--force`. Values reach `modal secret
create` through a 0600 temporary JSON file (`--from-json`), never on a command line, and the file is removed.
"""

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

ROOT = Path(__file__).resolve().parents[1]
GATEWAY_SECRET = "scout-gateway"  # noqa: S105  (a Modal secret NAME)
LLM_SECRET = "scout-llm"  # noqa: S105
LLM_V3_SECRET = "scout-llm-v3"  # noqa: S105
LLM_A20_SECRET = "scout-llm-a20"  # noqa: S105
LOCAL_KEYS = Path.home() / ".config" / "scout" / "gateway-master.json"
ENV_FILES = (ROOT / ".env", ROOT.parent / "wayfinder" / ".env")
DEFAULT_KEYS_NS = "scout_dev_keys"
DEFAULT_LLM_MODEL = "a1"
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


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


def load_env(files: tuple[Path, ...] | None = None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Files (first definition wins), overlaid by the process environment."""
    merged: dict[str, str] = {}
    for f in ENV_FILES if files is None else files:
        try:
            for k, v in parse_env_text(f.read_text(encoding="utf-8")).items():
                merged.setdefault(k, v)
        except OSError:
            continue
    merged.update(os.environ if base is None else base)
    return merged


def b64_random(n: int = 32) -> str:
    return base64.b64encode(secrets.token_bytes(n)).decode()


def local_keys(path: Path | None = None) -> dict[str, str]:
    """`SCOUT_MASTER_KEY`, `SCOUT_PEPPER`, `SCOUT_LLM_API_KEY` generated once, stored 0600."""
    path = LOCAL_KEYS if path is None else path
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if all(
            isinstance(data.get(k), str) and data[k]
            for k in ("SCOUT_MASTER_KEY", "SCOUT_PEPPER", "SCOUT_LLM_API_KEY")
        ):
            return {k: data[k] for k in ("SCOUT_MASTER_KEY", "SCOUT_PEPPER", "SCOUT_LLM_API_KEY")}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = {
        "SCOUT_MASTER_KEY": b64_random(),
        "SCOUT_PEPPER": b64_random(),
        "SCOUT_LLM_API_KEY": "sk-scout-llm-" + secrets.token_urlsafe(32),
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def gateway_values(env: Mapping[str, str], keys: Mapping[str, str]) -> dict[str, str]:
    missing = [k for k in ("TURBOPUFFER_API_KEY", "VOYAGE_API_KEY") if not env.get(k)]
    llm_url = env.get("SCOUT_LLM_URL") or env.get("A1_BASE_URL")
    llm_key = env.get("SCOUT_LLM_API_KEY") or env.get("A1_API_KEY")
    if not llm_url:
        missing.append("SCOUT_LLM_URL (or A1_BASE_URL)")
    if not llm_key:
        missing.append("SCOUT_LLM_API_KEY (or A1_API_KEY)")
    if missing:
        raise SystemExit("missing variables: " + ", ".join(missing))
    assert llm_url is not None
    assert llm_key is not None
    return {
        "TURBOPUFFER_API_KEY": env["TURBOPUFFER_API_KEY"],
        "VOYAGE_API_KEY": env["VOYAGE_API_KEY"],
        "SCOUT_MASTER_KEY": keys["SCOUT_MASTER_KEY"],
        "SCOUT_PEPPER": keys["SCOUT_PEPPER"],
        "SCOUT_KEYS_NS": env.get("SCOUT_KEYS_NS") or DEFAULT_KEYS_NS,
        "SCOUT_LLM_URL": llm_url,
        "SCOUT_LLM_API_KEY": llm_key,
        "SCOUT_LLM_MODEL": env.get("SCOUT_LLM_MODEL") or DEFAULT_LLM_MODEL,
    }


def llm_v3_values(env: Mapping[str, str]) -> dict[str, str]:
    missing = [k for k in ("V3_BASE_URL", "V3_API_KEY") if not env.get(k)]
    if missing:
        raise SystemExit("missing variables: " + ", ".join(missing))
    return {
        "SCOUT_LLM_URL": env["V3_BASE_URL"],
        "SCOUT_LLM_API_KEY": env["V3_API_KEY"],
        "SCOUT_LLM_MODEL": env.get("V3_MODEL") or "wayfinder-v3",
        "SCOUT_LLM_SAMPLING": "sft",
    }


def llm_a20_values(env: Mapping[str, str]) -> dict[str, str]:
    """The GRPO arm a step-20 upstream (app wayfinder-a20-serve). It shares the wayfinder-v3 bearer key
    (Modal secret `wayfinder-v3-key`), so V3_API_KEY is the key; the URL and model name are its own."""
    missing = [k for k in ("A20_BASE_URL", "V3_API_KEY") if not env.get(k)]
    if missing:
        raise SystemExit("missing variables: " + ", ".join(missing))
    return {
        "SCOUT_LLM_URL": env["A20_BASE_URL"],
        "SCOUT_LLM_API_KEY": env["V3_API_KEY"],
        "SCOUT_LLM_MODEL": env.get("A20_MODEL") or "wayfinder-grpo-a20",
        "SCOUT_LLM_SAMPLING": "sft",
    }


def existing_secrets(modal_bin: str) -> set[str]:
    res = subprocess.run(  # noqa: S603
        [modal_bin, "secret", "list", "--json"], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        raise SystemExit(f"`modal secret list` failed (exit {res.returncode}): is modal authenticated?")
    try:
        return {str(s["name"]) for s in json.loads(res.stdout)}
    except (ValueError, KeyError, TypeError):
        raise SystemExit("`modal secret list --json` returned something unreadable") from None


def create_secret(modal_bin: str, name: str, values: Mapping[str, str], *, force: bool) -> None:
    fd, tmp = tempfile.mkstemp(prefix="scout-secret-", suffix=".json")
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
    ap = argparse.ArgumentParser(description="Create the Modal secrets scout-gateway and scout-llm.")
    ap.add_argument("--force", action="store_true", help="overwrite existing secrets")
    ap.add_argument(
        "--llm-v3", action="store_true", help="create only scout-llm-v3 (the wayfinder-v3 upstream)"
    )
    ap.add_argument(
        "--llm-a20", action="store_true", help="create only scout-llm-a20 (the GRPO arm a step-20 upstream)"
    )
    args = ap.parse_args(argv)
    present = existing_secrets(modal_bin)
    if args.llm_a20:
        if LLM_A20_SECRET in present and not args.force:
            sys.stdout.write(f"refusing to overwrite existing secret {LLM_A20_SECRET}; pass --force\n")
            return 1
        values = llm_a20_values(load_env(base=env))
        create_secret(modal_bin, LLM_A20_SECRET, values, force=args.force)
        sys.stdout.write(f"{LLM_A20_SECRET}: {', '.join(sorted(values))}\n")
        return 0
    if args.llm_v3:
        if LLM_V3_SECRET in present and not args.force:
            sys.stdout.write(f"refusing to overwrite existing secret {LLM_V3_SECRET}; pass --force\n")
            return 1
        values = llm_v3_values(load_env(base=env))
        create_secret(modal_bin, LLM_V3_SECRET, values, force=args.force)
        sys.stdout.write(f"{LLM_V3_SECRET}: {', '.join(sorted(values))}\n")
        return 0
    clash = sorted({GATEWAY_SECRET, LLM_SECRET} & present)
    if clash and not args.force:
        sys.stdout.write(f"refusing to overwrite existing secret(s) {', '.join(clash)}; pass --force\n")
        return 1
    environment = load_env(base=env)
    keys = local_keys()
    gateway = gateway_values(environment, keys)
    create_secret(modal_bin, GATEWAY_SECRET, gateway, force=args.force)
    create_secret(modal_bin, LLM_SECRET, {"SCOUT_LLM_API_KEY": keys["SCOUT_LLM_API_KEY"]}, force=args.force)
    sys.stdout.write(f"{GATEWAY_SECRET}: {', '.join(sorted(gateway))}\n{LLM_SECRET}: SCOUT_LLM_API_KEY\n")
    sys.stdout.write(f"master key copy: {LOCAL_KEYS}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
