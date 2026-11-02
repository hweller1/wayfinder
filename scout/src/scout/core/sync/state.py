"""Sync state dir (design section 8): everything lives under `FileSystem.state_dir()`.

That is `git rev-parse --git-path scout`: inside the git dir (per worktree), never in the working tree, so
the files are never indexed, grepped or committed. Contents: `state.json` (`clone_id`, tenant, repo id, last
branch), `trees/<branch_id>.json`, `status.json`, `sync.lock`. No key material and no file contents are ever
written here (paths and keyed leaf hashes only).

Every write is tmp file (unique name, same directory) + fsync + `os.replace`, so a reader (the Claude Code
hook reads `status.json` with the stdlib only) sees the old or the new file, never a partial one, even with
concurrent writers. `status.json` is read without taking `sync.lock`, so it stays readable during a sync.
`clone_id` is created once with `os.link` (create-if-absent), so concurrent first runs agree on one value.
"""

import fcntl  # POSIX only; Windows is a v1 non-goal (PRD section 10). Move behind FileSystem for a port.
import json
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import ValidationError

from scout.core.contracts.schema import StatusFile
from scout.core.crypto.ids import new_clone_id
from scout.core.ports.fs import FileStat, FileSystem

STATE_SCHEMA = "1.0"
READY_THRESHOLD = 0.8
GATEWAY_FRESH_S = 600.0
_CLONE_RE = re.compile(r"^[0-9a-f]{12}$")
_BRANCH_RE = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True, slots=True)
class FileEntry:
    leaf: str
    size: int
    mtime_ns: int
    ino: int
    chunks: int = 0  # chunk rows of the file at the last sync (repo totals for an incremental commit)


@dataclass(frozen=True, slots=True)
class TreeState:
    root: str | None
    files: dict[str, FileEntry]


@dataclass(frozen=True, slots=True)
class State:
    clone_id: str
    tenant: str | None = None
    repo_id: str | None = None
    last_branch: str | None = None


def stat_unchanged(saved: FileEntry | None, st: FileStat, *, full: bool = False) -> bool:
    """Stat-cache rule: `(size, mtime_ns, ino)` equal to the saved entry means skip re-hashing."""
    if full or saved is None:
        return False
    return (saved.size, saved.mtime_ns, saved.ino) == (st.size, st.mtime_ns, st.ino)


def classify(written: int, expected: int, *, has_root: bool) -> tuple[str, float]:
    """`(state, coverage)`: no root row -> absent; `coverage = written/expected`, < 0.8 -> partial."""
    if not has_root:
        return "absent", 0.0
    coverage = 1.0 if expected <= 0 else min(max(written / expected, 0.0), 1.0)
    return ("ready" if coverage >= READY_THRESHOLD else "partial"), coverage


def iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_epoch(ts: str) -> float | None:
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()


def readiness(status: StatusFile | None, now: float, creds_present: bool, mcp_alive: bool) -> bool:
    """Hook rule: creds AND state ready/partial AND `now - gateway_ok_at < 600` AND MCP up."""
    if status is None or not creds_present or not mcp_alive:
        return False
    if status.state not in ("ready", "partial") or status.gateway_ok_at is None:
        return False
    ok_at = _parse_epoch(status.gateway_ok_at)
    return ok_at is not None and now - ok_at < GATEWAY_FRESH_S


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "rb") as f:
            data = json.loads(f.read())
    except (OSError, ValueError):
        return None
    return cast("dict[str, Any]", data) if isinstance(data, dict) else None


def _tmp_write(directory: str, data: bytes) -> str:
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        os.unlink(tmp)
        raise
    return tmp


def _atomic_write(path: str, data: bytes) -> None:
    tmp = _tmp_write(os.path.dirname(path), data)
    try:
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _create_once(path: str, data: bytes) -> bool:
    """Atomically create `path` with `data` unless it exists; True when this call created it."""
    tmp = _tmp_write(os.path.dirname(path), data)
    try:
        os.link(tmp, path)
    except FileExistsError:
        return False
    finally:
        os.unlink(tmp)
    return True


def _dump(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


class StateDir:
    def __init__(self, path: str) -> None:
        self.path = path

    @classmethod
    def open(cls, fs: FileSystem) -> "StateDir":
        """Resolve the git-path state dir through the port (NOT_A_REPO if not a checkout) and create it."""
        path = fs.state_dir()
        os.makedirs(os.path.join(path, "trees"), mode=0o700, exist_ok=True)
        return cls(path)

    def _p(self, *parts: str) -> str:
        return os.path.join(self.path, *parts)

    # --- state.json ---------------------------------------------------------------------------
    def read_state(self) -> State:
        """Load `state.json`, creating it (with a fresh `clone_id`) once if absent or unusable."""
        for _ in range(3):
            raw = _read_json(self._p("state.json"))
            cid = raw.get("clone_id") if raw else None
            if raw is not None and isinstance(cid, str) and _CLONE_RE.match(cid):
                return State(
                    cid,
                    _opt_str(raw.get("tenant")),
                    _opt_str(raw.get("repo_id")),
                    _opt_str(raw.get("last_branch")),
                )
            fresh = {"schema": STATE_SCHEMA, "clone_id": new_clone_id()}
            if raw is None and not os.path.exists(self._p("state.json")):
                _create_once(self._p("state.json"), _dump(fresh))
            else:  # unreadable or invalid: replace; a new clone_id only costs a full resync
                _atomic_write(self._p("state.json"), _dump(fresh))
        raise RuntimeError("state.json could not be initialised")

    def clone_id(self) -> str:
        return self.read_state().clone_id

    def update_state(
        self, *, tenant: str | None = None, repo_id: str | None = None, last_branch: str | None = None
    ) -> State:
        cur = self.read_state()
        new = State(
            cur.clone_id,
            tenant if tenant is not None else cur.tenant,
            repo_id if repo_id is not None else cur.repo_id,
            last_branch if last_branch is not None else cur.last_branch,
        )
        doc = {
            "schema": STATE_SCHEMA,
            "clone_id": new.clone_id,
            "tenant": new.tenant,
            "repo_id": new.repo_id,
            "last_branch": new.last_branch,
        }
        _atomic_write(self._p("state.json"), _dump(doc))
        return new

    # --- trees --------------------------------------------------------------------------------
    def _tree_path(self, branch_id: str) -> str:
        if not _BRANCH_RE.match(branch_id):
            raise ValueError("branch_id must be 16 lowercase hex chars")
        return self._p("trees", f"{branch_id}.json")

    def save_tree(self, branch_id: str, tree: TreeState) -> None:
        files = {
            p: {"leaf": e.leaf, "size": e.size, "mtime_ns": e.mtime_ns, "ino": e.ino, "chunks": e.chunks}
            for p, e in tree.files.items()
        }
        _atomic_write(self._tree_path(branch_id), _dump({"root": tree.root, "files": files}))

    def load_tree(self, branch_id: str) -> TreeState | None:
        """The saved tree, or None when absent or unreadable (the caller then does a full resync)."""
        raw = _read_json(self._tree_path(branch_id))
        if raw is None:
            return None
        try:
            root = raw["root"]
            files = {
                str(p): FileEntry(
                    str(e["leaf"]), int(e["size"]), int(e["mtime_ns"]), int(e["ino"]), int(e["chunks"])
                )
                for p, e in cast("dict[str, dict[str, Any]]", raw["files"]).items()
            }
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return TreeState(root if isinstance(root, str) else None, files)

    # --- sync.lock ----------------------------------------------------------------------------
    @contextmanager
    def lock(self, *, wait: bool = False) -> Generator[bool]:
        """Exclusive `flock` on `sync.lock`. Yields True when held. When another process holds it and
        `wait` is False, yields False at once: the caller reports `read_status()` instead of syncing."""
        fd = os.open(self._p("sync.lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            held = True
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:
                held = False
            try:
                yield held
            finally:
                if held:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # --- status.json --------------------------------------------------------------------------
    def write_status(
        self,
        state: str,
        coverage: float,
        *,
        commit: str | None = None,
        syncing: bool = False,
        last_sync_at: str | None = None,
        gateway_ok_at: str | None = None,
        last_error: str | None = None,
    ) -> StatusFile:
        """Validate (pydantic, same shape as `schema/v1/status_file.json`) then write atomically."""
        status = StatusFile.model_validate(
            {
                "state": state,
                "coverage": coverage,
                "commit": commit,
                "syncing": syncing,
                "last_sync_at": last_sync_at,
                "gateway_ok_at": gateway_ok_at,
                "last_error": last_error,
            }
        )
        _atomic_write(self._p("status.json"), status.model_dump_json(by_alias=True).encode("utf-8"))
        return status

    def read_status(self) -> StatusFile | None:
        """Current `status.json` (no lock taken); None when absent or invalid."""
        raw = _read_json(self._p("status.json"))
        if raw is None:
            return None
        try:
            return StatusFile.model_validate(raw)
        except ValidationError:
            return None


def _opt_str(v: object) -> str | None:
    return v if isinstance(v, str) else None
