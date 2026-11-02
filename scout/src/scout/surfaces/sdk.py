"""Python SDK: a thin sync/async wrapper over `ScoutApp`. No logic here: build the contract request, call
the app, return the contract result. Sync methods use `asyncio.run` (call the `a*` variants from a loop).

A sync wrapper runs its own event loop per call: `sync(wait=False)` is therefore async-only (`await
scout.async_sync(wait=False)`), because its background work needs a loop that outlives the call."""

import asyncio
import os
import threading
from collections.abc import Coroutine
from typing import Any

from scout.core.app import ScoutApp
from scout.core.contracts.types import (
    Budget,
    ExploreRequest,
    ExploreResult,
    GrepRequest,
    GrepResult,
    IndexStatus,
    IndexStatusRequest,
    ReadRangeRequest,
    ReadRangeResult,
    SemanticSearchRequest,
    SemanticSearchResult,
)
from scout.core.sync.watch import WatchConfig, WatchResult
from scout.wiring import CaseSource, Config, build_app


class Scout:
    def __init__(
        self,
        *,
        profile: str = "default",
        case: CaseSource | None = None,
        app: ScoutApp | None = None,
    ) -> None:
        self._app = app or build_app(Config(profile=profile, case=case))

    @staticmethod
    def _run[T](coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run(coro)

    # --- delegated level -------------------------------------------------------------------
    async def aexplore(
        self,
        brief: str,
        *,
        paths: list[str] | None = None,
        max_results: int = 12,
        render_max_tokens: int = 1500,
        budget: Budget | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> ExploreResult:
        req = ExploreRequest(
            brief=brief,
            paths=paths or [],
            max_results=max_results,
            render_max_tokens=render_max_tokens,
            budget=budget or Budget(),
            root=None if root is None else os.fspath(root),
        )
        return await self._app.explore(req)

    def explore(
        self,
        brief: str,
        *,
        paths: list[str] | None = None,
        max_results: int = 12,
        render_max_tokens: int = 1500,
        budget: Budget | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> ExploreResult:
        return self._run(
            self.aexplore(
                brief,
                paths=paths,
                max_results=max_results,
                render_max_tokens=render_max_tokens,
                budget=budget,
                root=root,
            )
        )

    # --- primitives ------------------------------------------------------------------------
    async def agrep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        regex: bool = True,
        context: int = 0,
        root: str | os.PathLike[str] | None = None,
    ) -> GrepResult:
        req = GrepRequest(
            pattern=pattern,
            path=path,
            regex=regex,
            context=context,
            root=None if root is None else os.fspath(root),
        )
        return await self._app.grep(req)

    def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        regex: bool = True,
        context: int = 0,
        root: str | os.PathLike[str] | None = None,
    ) -> GrepResult:
        return self._run(self.agrep(pattern, path=path, regex=regex, context=context, root=root))

    async def aread_range(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> ReadRangeResult:
        req = ReadRangeRequest(
            path=path,
            start_line=start_line,
            end_line=end_line,
            root=None if root is None else os.fspath(root),
        )
        return await self._app.read_range(req)

    def read_range(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> ReadRangeResult:
        return self._run(self.aread_range(path, start_line=start_line, end_line=end_line, root=root))

    async def aindex_status(self, *, root: str | os.PathLike[str] | None = None) -> IndexStatus:
        return await self._app.index_status(
            IndexStatusRequest(root=None if root is None else os.fspath(root))
        )

    def index_status(self, *, root: str | os.PathLike[str] | None = None) -> IndexStatus:
        return self._run(self.aindex_status(root=root))

    async def asemantic_search(
        self,
        query: str,
        *,
        path_glob: str | None = None,
        k: int = 10,
        root: str | os.PathLike[str] | None = None,
    ) -> SemanticSearchResult:
        req = SemanticSearchRequest(
            query=query, path_glob=path_glob, k=k, root=None if root is None else os.fspath(root)
        )
        return await self._app.semantic_search(req)

    def semantic_search(
        self,
        query: str,
        *,
        path_glob: str | None = None,
        k: int = 10,
        root: str | os.PathLike[str] | None = None,
    ) -> SemanticSearchResult:
        return self._run(self.asemantic_search(query, path_glob=path_glob, k=k, root=root))

    # --- lifecycle -------------------------------------------------------------------------
    async def async_sync(self, *, wait: bool = True, full: bool = False) -> IndexStatus:
        return await self._app.sync(wait, full=full)

    def sync(self, *, full: bool = False) -> IndexStatus:
        """Index the checkout and wait for it (the background form is `async_sync(wait=False)`)."""
        return self._run(self.async_sync(wait=True, full=full))

    async def awatch(self, stop: threading.Event, cfg: WatchConfig | None = None) -> WatchResult:
        return await self._app.watch(stop, cfg)

    def watch(self, stop: threading.Event, cfg: WatchConfig | None = None) -> WatchResult:
        """Blocks until `stop` is set."""
        return self._run(self.awatch(stop, cfg))

    async def alogin(self, key: str) -> None:
        await self._app.login(key)

    def login(self, key: str) -> None:
        self._run(self.alogin(key))
