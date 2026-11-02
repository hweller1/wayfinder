"""Ripgrep stream for `LocalFileSystem.grep_stream` (design section 3, FR-C5, AC-2.1, AC-2.2).

The argv is a fixed list (never a shell). The pattern travels only as the value of `--regexp` and the target
only after `--`, so a pattern or path that starts with `-` can never become an option (the benchmark's leakage
audit L17: a bare `--pre=/path` pattern made rg run an arbitrary program). `--no-config` ignores
`RIPGREP_CONFIG_PATH` and every config file; `--pre` and `--search-zip` are never passed; symlinks are never
followed (`--no-follow`, no `-L`).

The generator owns the process: closing it (or an error) kills and reaps rg, which is how the core grep stops
early. Exit code 1 (no matches) is not an error. Exit code 2 with a summary (rg finished, some file was
unreadable) keeps the partial results; exit code 2 without one (regex parse error, crash) is a typed error.
"""

import base64
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from typing import Any

from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.local.ignore import MAX_FILE_BYTES
from scout.core.ports.fs import RgEvent, RgSpec

__all__ = ["grep_stream"]

_REGEX_ERROR_MARKERS = ("regex parse error", "not allowed in a regex", "look-around", "error parsing")


def _argv(exe: str, spec: RgSpec, ignore_file: str | None, target: str) -> list[str]:
    argv = [
        exe,
        "--json",
        "--no-config",  # no RIPGREP_CONFIG_PATH / config-file options
        "--no-follow",
        "--no-ignore-global",  # the user's global excludes file must not change what the tool can see
        "--hidden",  # dotfiles are searchable; the core matcher drops secrets and `.git/`
        "--glob=!.git",
        f"--max-filesize={MAX_FILE_BYTES}",
        "--sort=path",  # stable file order (port contract); single-threaded
        f"--max-columns={max(spec.max_columns, 1)}",
    ]
    if ignore_file is not None:
        argv.append(f"--ignore-file={ignore_file}")
    if not spec.regex:
        argv.append("--fixed-strings")
    if spec.context > 0:
        argv.append(f"--context={spec.context}")
    argv += ["--regexp", spec.pattern, "--", target]
    return argv


def _field(obj: dict[str, Any]) -> str:
    """rg encodes text as `{"text": ...}` and non-UTF-8 as `{"bytes": base64}`."""
    if "text" in obj:
        return str(obj["text"])
    return base64.b64decode(obj.get("bytes", "")).decode("utf-8", errors="replace")


def _event(data: dict[str, Any], kind: str, max_columns: int) -> RgEvent:
    path = _field(data["path"]).removeprefix("./")
    text = _field(data["lines"]).removesuffix("\n")
    return RgEvent(
        kind="match" if kind == "match" else "context",
        path=path,
        line=int(data["line_number"]),
        text=text[:max_columns],
    )


def _stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
    if proc.stdout is not None:
        proc.stdout.close()
    proc.wait()  # reap: no zombie


def grep_stream(root: str, spec: RgSpec, target: str) -> Iterator[RgEvent]:
    """Stream `RgEvent`s for `spec`. `root` is the real repo root (rg's cwd), `target` the already confined
    repo-relative file or directory. rg starts on the first `next()`; an unstarted generator holds nothing."""
    exe = shutil.which("rg")
    if exe is None:
        raise ScoutError(ErrorCode.INTERNAL, "ripgrep (rg) was not found on PATH")
    try:
        os.lstat(os.path.join(root, target))
    except FileNotFoundError:
        raise ScoutError(ErrorCode.NOT_FOUND, "search target not found") from None
    except OSError:
        raise ScoutError(ErrorCode.INVALID_ARGUMENT, "search target is not accessible") from None
    ignore = spec.ignore_file
    if ignore is not None and not os.path.isfile(os.path.join(root, ignore)):
        ignore = None  # a missing ignore file makes rg warn; absent means "nothing to ignore"
    return _run(exe, root, spec, _argv(exe, spec, ignore, target))


def _run(exe: str, root: str, spec: RgSpec, argv: list[str]) -> Iterator[RgEvent]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("RIPGREP_")}
    with tempfile.TemporaryFile() as err:  # file, not a pipe: no deadlock while we read stdout
        try:
            # list argv, no shell, pattern only after --regexp, target only after --
            proc = subprocess.Popen(  # noqa: S603
                argv, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err
            )
        except (ValueError, UnicodeError):  # NUL or lone surrogate in the pattern
            raise ScoutError(ErrorCode.INVALID_ARGUMENT, "invalid pattern") from None
        except OSError as e:
            raise ScoutError(ErrorCode.INTERNAL, f"ripgrep could not run: {e.strerror or 'error'}") from None
        assert proc.stdout is not None
        finished = False
        try:
            for raw in proc.stdout:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    raise ScoutError(ErrorCode.INTERNAL, "ripgrep produced unreadable output") from None
                kind = msg.get("type")
                if kind == "summary":
                    finished = True
                elif kind in ("match", "context"):
                    yield _event(msg["data"], kind, spec.max_columns)
            rc = proc.wait()
            if rc not in (0, 1) and not finished:
                err.seek(0)
                text = err.read(4096).decode("utf-8", errors="replace").lower()
                if any(m in text for m in _REGEX_ERROR_MARKERS):
                    raise ScoutError(ErrorCode.INVALID_ARGUMENT, "invalid regular expression")
                raise ScoutError(ErrorCode.INTERNAL, f"ripgrep failed (exit {rc})")
        finally:
            _stop(proc)
