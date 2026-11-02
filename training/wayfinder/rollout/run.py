"""One rollout = scout's explore loop, product profile, on a real checkout of {repo, base_commit}.

Nothing of scout is copied: `scout` is imported from ../voyage-scout/src (or the path in SCOUT_SRC; on Modal
it is mounted at /scout_src). grep, glob and read_file are scout's own tools over its real `LocalFileSystem`
adapter; the loop is `scout.core.explore.loop.explore`.

semantic_search: with `semantic="auto"` (default) or `"on"`, the rollout goes through scout's `ScoutApp.explore` over the
real TurbopufferStore + VoyageEmbedder gateway (wayfinder/rollout/semantic.py) whenever an index of repo@base_commit exists
(built by wayfinder/rollout/index_app.py); the loop then offers `semantic_search` (scout offers it at coverage >= 0.8).
`"auto"` falls back to grep-only when there is no index (or scout's semantic modules/keys are unavailable) and records
why in `semantic_reason`; `"on"` raises instead (the row becomes an infra error and is retried); `"off"` is the old
grep-only harness. Every row records `semantic` (bool, semantic_search was offered), `semantic_requested`,
`semantic_reason`, `semantic_index` and `semantic_calls`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scout.adapters.fs_local import LocalFileSystem  # noqa: E402
from scout.core.contracts.errors import ErrorCode, ScoutError  # noqa: E402
from scout.core.contracts.types import Budget, ExploreRequest, IndexStatus  # noqa: E402
from scout.core.contracts.wire import StatusReq  # noqa: E402
from scout.core.explore.loop import ExploreContext, explore  # noqa: E402
from scout.core.explore.profiles import PROFILES, ExploreProfile, load_prompt  # noqa: E402
from scout.core.ports.chat import ChatRequest, ChatTurn  # noqa: E402

from wayfinder.rollout.chat import CompatChat  # noqa: E402

MIN_FREE_BYTES = 3 * 1024**3
MIN_FREE_INODE_FRAC = 0.05


class HarnessFileSystem(LocalFileSystem):
    """scout's real LocalFileSystem with one harness-side fix. scout maps an rg exit 2 whose message is not one
    of its regex-error markers (e.g. the model's pattern contains `\\x00`: 'pattern contains "\\0" but it is
    impossible to match') to INTERNAL, and INTERNAL aborts the whole explore call with no partial result. A bad
    pattern is the model's doing, so it is re-raised as INVALID_ARGUMENT, which the loop feeds back as a tool
    error. Counted in `rg_internal_remapped` so the deviation stays visible. (Reported to the scout owner.)"""

    rg_internal_remapped = 0

    def grep_stream(self, spec):  # type: ignore[no-untyped-def, override]
        def gen():  # type: ignore[no-untyped-def]
            try:
                yield from super(HarnessFileSystem, self).grep_stream(spec)
            except ScoutError as e:
                if e.code is ErrorCode.INTERNAL and "ripgrep failed" in e.message:
                    self.rg_internal_remapped += 1
                    raise ScoutError(ErrorCode.INVALID_ARGUMENT, "invalid regular expression") from None
                raise

        return gen()


# Teacher-run convergence nudge (Stage B onward). Appended to scout's product prompt as its own paragraph; the exact
# text is recorded in every row (`nudge`) and in docs/teacher_test.md. Scout's prompt file is not modified.
from wayfinder.rollout.nudges import TEACHER_NUDGE  # noqa: E402
from wayfinder.rollout.nudges import NUDGES, ORDER_LINE  # noqa: E402,F401  (GRPO run f1: the f1 nudge adds the report-order line; see nudges.py)


@dataclasses.dataclass(frozen=True, slots=True)
class NudgedProfile(ExploreProfile):
    """scout's profile with `nudge` appended to the system prompt (the loop reads `profile.prompt`)."""

    nudge: str = ""

    @property
    def prompt(self) -> str:  # type: ignore[override]
        base = load_prompt(self.prompt_name)
        return base + ("\n\n" + self.nudge if self.nudge else "")


class RealClock:
    """Wall clock for the loop's `max_seconds` budget. Time the harness spent queued behind the provider's rate
    limit (token gate + 429 backoff) is excluded via `exclude()`: it is capacity of the test endpoint, not model
    or tool time, and counting it made 19/20 Stage A rollouts hit the 300 s cap at turn ~10 (see docs)."""

    def __init__(self) -> None:
        self.excluded = 0.0

    def exclude(self, seconds: float) -> None:
        self.excluded += seconds

    def now(self) -> float:
        return time.time() - self.excluded

    def monotonic(self) -> float:
        return time.monotonic() - self.excluded

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class DirectGateway:
    """The two Gateway methods the explore loop calls, without a gateway: chat goes straight to the model
    adapter and the index is reported absent (grep-only)."""

    def __init__(self, chat: CompatChat) -> None:
        self._chat = chat

    async def complete(self, req: ChatRequest) -> ChatTurn:
        return await self._chat.complete(req)

    async def index_status(self, req: StatusReq) -> IndexStatus:
        return IndexStatus(state="absent", coverage=0.0, semantic_enabled=False)


def preflight(path: str) -> None:
    """Refuse to write when disk or inodes are short (never discover it by running out)."""
    os.makedirs(path, exist_ok=True)
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"disk preflight: only {free / 1e9:.1f} GB free under {path}")
    if st.f_files and st.f_favail / st.f_files < MIN_FREE_INODE_FRAC:
        raise RuntimeError(f"inode preflight: {st.f_favail}/{st.f_files} inodes free under {path}")


def _git(args: list[str], cwd: str, timeout: int = 600) -> None:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, timeout=timeout, env=env)  # noqa: S603, S607


def checkout(repo: str, base_commit: str, cache_dir: str, local_mirrors: str | None = None) -> tuple[str, float]:
    """Depth-1 checkout of `base_commit` into cache_dir/<owner>__<name>__<sha12>. Reused if present.
    Returns (path, seconds spent cloning)."""
    dest = Path(cache_dir) / f"{repo.replace('/', '__')}__{base_commit[:12]}"
    if (dest / ".wf_ok").exists():
        return str(dest), 0.0
    preflight(cache_dir)
    t0 = time.monotonic()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    try:
        _git(["init", "-q"], str(dest))
        _git(["remote", "add", "origin", f"https://github.com/{repo}.git"], str(dest))
        last: Exception | None = None
        for attempt in range(3):
            try:
                _git(["fetch", "-q", "--depth", "1", "--no-tags", "origin", base_commit], str(dest))
                last = None
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                last = e
                time.sleep(3 * (attempt + 1))
        if last is not None:
            raise RuntimeError(f"fetch failed for {repo}@{base_commit[:12]}: {_err(last)}")
        _git(["checkout", "-q", "--detach", "FETCH_HEAD"], str(dest))
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    (dest / ".wf_ok").write_text(base_commit)
    return str(dest), time.monotonic() - t0


def _err(e: Exception) -> str:
    out = getattr(e, "stderr", b"") or b""
    return (out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out))[-300:] or type(e).__name__


async def run_rollout(
    task: dict[str, Any],
    chat: CompatChat,
    root: str,
    *,
    seed: int,
    max_turns: int = 40,
    max_seconds: int = 300,
    max_results: int = 12,
    nudge: str | None = None,
    semantic: str = "auto",
) -> dict[str, Any]:
    """Run explore for one task on `root`; return a row (scoring is added by wayfinder.rollout.score)."""
    prof = PROFILES["product"]
    if nudge:
        prof = NudgedProfile(**{f.name: getattr(prof, f.name) for f in dataclasses.fields(ExploreProfile)}, nudge=nudge)
    req = ExploreRequest(
        brief=task["question"], max_results=max_results, budget=Budget(max_turns=max_turns, max_seconds=max_seconds)
    )
    fs = HarnessFileSystem(root)
    clock = RealClock()
    chat.clock = clock
    ctx = ExploreContext(
        gateway=DirectGateway(chat),  # type: ignore[arg-type]
        fs=fs,
        clock=clock,
        status_req=_status_req(),
        profile=prof,
        backfill_to=0,  # scout's code-side backfill to 12 files is OFF in every Wayfinder rollout/eval: v3/A1 baselines have none, and backfilled files must not earn reward
    )
    if semantic not in ("on", "off", "auto"):
        raise ValueError(f"semantic must be on|off|auto, got {semantic!r}")
    sem_app, sem_stack, sem_reason, sem_index = None, None, None, None
    if semantic == "off":
        sem_reason = "flag off"
    else:
        try:
            from wayfinder.rollout.semantic import MIN_COVERAGE, SemanticStack

            sem_stack = SemanticStack(task["repo"], task["base_commit"], root)
            st = await asyncio.wait_for(sem_stack.status(), 90)
            sem_index = {"state": st.state, "coverage": st.coverage}
            if st.coverage >= MIN_COVERAGE:
                sem_app = sem_stack.app(chat, fs, clock, prof)
            else:
                sem_reason = f"no index for repo@commit (state={st.state}, coverage={st.coverage})"
        except Exception as e:  # noqa: BLE001
            sem_reason = f"semantic unavailable: {type(e).__name__}: {e}"[:300]
        if sem_app is None and semantic == "on":
            raise RuntimeError(f"--semantic on but {sem_reason}")
    t0 = time.monotonic()
    error: str | None = None
    result: dict[str, Any] | None = None
    try:
        res = await (sem_app.explore(req) if sem_app is not None else explore(req, ctx))
        result = res.model_dump(mode="json", by_alias=True)
    except ScoutError as e:
        error = f"{e.code}: {e.message}"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"[:300]
    wall = time.monotonic() - t0
    tot = chat.totals()
    return {
        "task_id": task["task_id"],
        "repo": task["repo"],
        "base_commit": task["base_commit"],
        "model": chat.model,
        "seed": seed,
        "semantic": sem_app is not None,
        "semantic_requested": semantic,
        "semantic_reason": sem_reason,
        "semantic_index": sem_index,
        "semantic_query_tokens": sem_stack.query_tokens if (sem_stack and sem_app is not None) else 0,
        "profile": "product",
        "nudge": nudge,
        "retain_reasoning": chat.retain_reasoning,  # default True since 2026-10-27; rows without the key predate it (False)
        "max_turns": max_turns,
        "max_seconds": max_seconds,
        "question": task["question"],
        "gold_files": task["gold_files"],
        "messages": chat.transcript(),
        "explore_result": result,
        "error": error,
        "usage": tot,
        "turns": (result or {}).get("usage", {}).get("turns", len([t for t in chat.turn_log if t.get("ok")])),
        "model_calls": len(chat.turn_log),
        "wall_s": round(wall, 2),
        "wall_budget_s": round(wall - clock.excluded, 2),
        "ratelimit_wait_s": round(clock.excluded, 2),
        "model_latency_s": round(sum(t.get("latency_s", 0) for t in chat.turn_log), 2),
        "cost_usd": round(chat.cost_usd(), 6),
        "rg_internal_remapped": fs.rg_internal_remapped,
        "n429": chat.n429,
        "n_retries": chat.n_retries,
        "turn_log": [{k: v for k, v in t.items() if k != "reasoning_content"} for t in chat.turn_log],
    }


def _status_req() -> StatusReq:
    return StatusReq(repo_id="0" * 32, branch_id="0" * 16)
