"""Deterministic synthetic repos for sync tests (task 4.1): `{path: text}` dicts, no disk, no randomness."""

from pathlib import Path

SECRET_FILES = {
    ".env": "SYNTHETIC_ENV_SECRET=hunter2-env\n",
    "deploy/server.pem": "-----BEGIN PRIVATE KEY-----\nSYNTHETIC_PEM_SECRET\n-----END PRIVATE KEY-----\n",
    "keys/id_rsa": "SYNTHETIC_IDRSA_SECRET\n",
    "config/credentials.json": '{"token": "SYNTHETIC_CRED_SECRET"}\n',
    "config/signing.key": "SYNTHETIC_KEY_SECRET\n",
}
SECRET_MARKERS = (
    "SYNTHETIC_ENV_SECRET",
    "SYNTHETIC_PEM_SECRET",
    "SYNTHETIC_IDRSA_SECRET",
    "SYNTHETIC_CRED_SECRET",
    "SYNTHETIC_KEY_SECRET",
)


def _module(i: int, lines: int) -> str:
    """A python module with unique identifiers (`handler_i_j`), about `lines` lines long."""
    out = [f'"""Module {i}: synthetic handlers."""', ""]
    for j in range(max(1, lines // 4)):
        out += [
            f"def handler_{i}_{j}(request_{i}):",
            f"    value_{i}_{j} = request_{i}.get('field_{j}', {i * 31 + j})",
            f"    return value_{i}_{j} + {j}",
            "",
        ]
    return "\n".join(out)


def make_repo(files: int = 300, *, dirs: int = 30, lines: int = 12) -> dict[str, str]:
    """`files` python modules spread over `dirs` directories (two levels deep). Every file is unique, so no
    two chunks share a key; `lines` sizes each file (about `lines` lines, one or a few chunks)."""
    out: dict[str, str] = {}
    for i in range(files):
        d = i % dirs
        out[f"pkg{d % 10}/sub{d}/mod_{i}.py"] = _module(i, lines)
    return out


def load_dir(root: Path) -> dict[str, bytes]:
    """Every file under `root` as `{posix relpath: bytes}` (dotfiles included)."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
