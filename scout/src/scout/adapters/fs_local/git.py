"""Git plumbing for `LocalFileSystem` (design section 3, section 8 state dir, section 15 edge cases).

Every git call is an argument list (never a shell), runs with a fixed `-c` config that disables the
repo-controlled code paths (fsmonitor hook, pager, hooks, external diff, protocol.ext), a sanitized
environment (no inherited `GIT_*`), and a timeout. A hostile `.git/config` therefore cannot make scout run
a command by merely listing, hashing or watching the repo.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.ports.fs import GitIds

GIT_TIMEOUT_S = 30.0

# Command-line `-c` beats every config file, so these hold even when `.git/config` says otherwise.
FIXED_CONFIG: tuple[str, ...] = (
    "core.fsmonitor=false",  # a path here is an executable git runs on `status`/`ls-files`
    "core.pager=cat",
    f"core.hooksPath={os.devnull}",
    "diff.external=",
    "protocol.ext.allow=never",
    "core.sshCommand=false",
)
_FIXED_ARGS: tuple[str, ...] = ("--no-pager", *(a for c in FIXED_CONFIG for a in ("-c", c)))


@dataclass(frozen=True, slots=True)
class GitResult:
    rc: int
    out: bytes
    err: str


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", GIT_PAGER="cat", LC_ALL="C")
    return env


def run_git(
    root: str, args: list[str], *, ok: tuple[int, ...] = (0,), timeout: float = GIT_TIMEOUT_S
) -> GitResult:
    """Run `git <fixed config> <args>` in `root`. rc outside `ok` raises (NOT_A_REPO or INTERNAL)."""
    exe = shutil.which("git")
    if exe is None:
        raise ScoutError(ErrorCode.INTERNAL, "git executable not found")
    try:
        # list argv, no shell, fixed config, sanitized env, timeout
        p = subprocess.run(  # noqa: S603
            [exe, *_FIXED_ARGS, *args],
            cwd=root,
            env=_env(),
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ScoutError(ErrorCode.INTERNAL, f"git timed out after {timeout:g}s") from None
    except OSError as e:
        raise ScoutError(ErrorCode.INTERNAL, f"git could not run: {e.strerror or 'error'}") from None
    err = p.stderr.decode("utf-8", errors="replace").strip()
    if p.returncode not in ok:
        if "not a git repository" in err:
            raise ScoutError(ErrorCode.NOT_A_REPO, "not a git checkout")
        raise ScoutError(
            ErrorCode.INTERNAL, f"git {args[0]} failed: {err.splitlines()[0][:200] if err else p.returncode}"
        )
    return GitResult(p.returncode, p.stdout, err)


def _text(res: GitResult) -> str:
    return res.out.decode("utf-8", errors="replace").strip()


def list_files(root: str) -> list[str]:
    """Raw `git ls-files -co --exclude-standard -z` names (undecodable names are skipped)."""
    res = run_git(root, ["ls-files", "-co", "--exclude-standard", "-z"])
    names: set[str] = set()
    for raw in res.out.split(b"\0"):
        if not raw:
            continue
        try:
            names.add(raw.decode("utf-8"))
        except UnicodeDecodeError:
            continue
    return sorted(names)


def state_dir(root: str) -> str:
    """`git rev-parse --git-path scout`: per worktree (`.git` is a file in a linked worktree)."""
    out = _text(run_git(root, ["rev-parse", "--git-path", "scout"]))
    return os.path.normpath(os.path.join(root, out))


def git_path(root: str, name: str) -> str:
    """`git rev-parse --git-path <name>` as an absolute path (worktree-safe: `HEAD` is per worktree,
    `refs/heads` lives in the common dir)."""
    out = _text(run_git(root, ["rev-parse", "--git-path", name]))
    return os.path.normpath(os.path.join(root, out))


_SCHEME = re.compile(r"^(?P<scheme>[a-z][a-z0-9+.-]*)://(?P<rest>.*)$", re.IGNORECASE)
_SCP = re.compile(r"^(?:[^@/:]+@)?(?P<host>[^:/@]+):(?P<path>.+)$")


def normalize_remote(url: str) -> str | None:
    """`host/owner/repo` lowercase: no scheme, credentials, port, `.git` or trailing slash. Local paths and
    `file://` remotes carry no shared identity -> None."""
    u = url.strip()
    if not u:
        return None
    m = _SCHEME.match(u)
    if m:
        if m["scheme"].lower() == "file":
            return None
        authority, _, path = m["rest"].partition("/")
        host = authority.rsplit("@", 1)[-1].split(":", 1)[0]
    else:
        s = _SCP.match(u)
        if s is None:
            return None
        host, path = s["host"], s["path"]
    path = path.strip("/")
    path = path.removesuffix(".git").rstrip("/")
    if not host or not path:
        return None
    return f"{host}/{path}".lower()


def _remote_url(root: str) -> str | None:
    names = _text(run_git(root, ["remote"])).splitlines()
    if not names:
        return None
    name = "origin" if "origin" in names else sorted(names)[0]
    res = run_git(root, ["config", "--get", f"remote.{name}.url"], ok=(0, 1))
    return _text(res) or None


def _default_branch(root: str) -> str | None:
    res = run_git(root, ["symbolic-ref", "-q", "refs/remotes/origin/HEAD"], ok=(0, 1))
    ref = _text(res)
    if ref.startswith("refs/remotes/origin/"):
        return ref.removeprefix("refs/remotes/origin/")
    for cand in ("main", "master"):
        if run_git(root, ["rev-parse", "--verify", "-q", f"refs/heads/{cand}"], ok=(0, 1)).rc == 0:
            return cand
    return None


def git_ids(root: str) -> GitIds:
    head_res = run_git(root, ["rev-parse", "--verify", "-q", "HEAD"], ok=(0, 1))
    head = _text(head_res) if head_res.rc == 0 else None
    root_commit: str | None = None
    if head is not None:
        roots = _text(run_git(root, ["rev-list", "--max-parents=0", "HEAD"])).split()
        root_commit = min(roots) if roots else None  # several roots (merged histories): smallest, stable
    ref = _text(run_git(root, ["symbolic-ref", "-q", "HEAD"], ok=(0, 1)))
    detached = not ref.startswith("refs/heads/")
    branch = f"detached-{(head or '0' * 12)[:12]}" if detached else ref.removeprefix("refs/heads/")
    shallow = _text(run_git(root, ["rev-parse", "--is-shallow-repository"])) == "true"
    url = _remote_url(root)
    return GitIds(
        root_commit=root_commit,
        remote_norm=normalize_remote(url) if url else None,
        branch=branch,
        head=head,
        default_branch=_default_branch(root),
        detached=detached,
        shallow=shallow,
    )
