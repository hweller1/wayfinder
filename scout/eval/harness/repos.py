"""Public repo checkouts for the eval: blobless clone, one working tree per task, no ref reaching later commits."""

import json
import os
import shutil
import subprocess
from pathlib import Path

# SCOUT_EVAL_CACHE: a separate checkout cache, so two concurrent runs never `git clean` each other's tree
CACHE = Path(os.environ.get("SCOUT_EVAL_CACHE") or Path(__file__).resolve().parents[2] / ".cache" / "eval_repos")


def git(cwd: Path, *args: str, check: bool = True) -> str:
    res = subprocess.run(  # noqa: S603
        ["git", "-C", str(cwd), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"},
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd.name}: {res.stderr.strip()[:300]}")
    return res.stdout


def slug(repo: str) -> str:
    return repo.replace("/", "__")


def main_dir(repo: str) -> Path:
    return CACHE / slug(repo)


def second_dir(repo: str) -> Path:
    return CACHE / (slug(repo) + "__b")


def clone(repo: str) -> Path:
    """Blobless clone, no checkout (idempotent). Only the clone is made here; files appear at `checkout`."""
    d = main_dir(repo)
    if (d / ".git").exists():
        return d
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", f"https://github.com/{repo}.git", str(d)],  # noqa: S607
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    git(d, "config", "gc.auto", "0")
    return d


def checkout(repo: str, commit: str) -> Path:
    """Detach HEAD at `commit` and make sure no ref reaches a later commit (the gold change)."""
    d = clone(repo)
    git(d, "checkout", "--quiet", "--force", "--detach", commit)
    git(d, "clean", "-fdxq", "-e", ".git")
    refs = git(d, "for-each-ref", "--format=delete %(refname)")
    if refs.strip():
        subprocess.run(["git", "-C", str(d), "update-ref", "--stdin"], input=refs, text=True, capture_output=True, check=False)  # noqa: S603, S607
    git(d, "reflog", "expire", "--expire=now", "--all", check=False)
    return d


def ensure_second(repo: str, commit: str) -> Path:
    """A linked worktree at the same commit, for the arm that must not share scout's state dir."""
    d, b = main_dir(repo), second_dir(repo)
    if not b.exists():
        git(d, "worktree", "add", "--quiet", "--detach", str(b), commit)
    else:
        git(b, "checkout", "--quiet", "--force", "--detach", commit)
        git(b, "clean", "-fdxq", "-e", ".git")
    return b


def state_dir(root: Path) -> Path:
    return Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "scout").strip())


def copy_state(src_root: Path, dst_root: Path) -> None:
    src, dst = state_dir(src_root), state_dir(dst_root)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("mcp.pid", "sync.lock", "watch*.pid"))


def ls_files(root: Path) -> list[str]:
    return [p for p in git(root, "ls-files", "-z").split("\0") if p]


def disk_free_gb() -> float:
    return shutil.disk_usage(CACHE.parent).free / 2**30


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
