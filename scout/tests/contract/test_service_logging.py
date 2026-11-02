"""Task 3.8: chat passthrough, wire caps, per-key limit, typed upstream errors, and the no-logging guarantee.

Covers FR-X7 and AC-13.3 (neither the gateway nor scout-llm logs prompts or tool results; the canary string in
a prompt, tool result, tool spec, query or chunk text appears in no log record at any level of any logger, on
stdout/stderr, in warnings, or in an exception message), FR-G1 (the gateway implements chat), and the
`details.reason == "context_length"` contract that the explore loop (1.11a) reads.
"""

import asyncio
import base64
import hashlib
import logging
import traceback
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import pytest

from scout.adapters.fake_chat import ScriptedChat
from scout.adapters.fake_clock import FakeClock
from scout.adapters.fake_embedder import FakeEmbedder
from scout.adapters.fake_store import FakeStore
from scout.core.contracts.errors import ErrorCode, ScoutError
from scout.core.contracts.wire import EmbedReq, SearchReq, SeedReq, SyncBeginReq, WireRow
from scout.core.crypto import keys, minhash
from scout.core.ports.chat import ChatRequest, ChatTurn, ToolCall
from scout.core.service.gateway import GatewayService, TenantGateway
from scout.core.service.logging import LOGGER
from scout.core.service.ratelimit import Limits, RateLimiter
from scout.core.service.tenants import Tenant

MASTER = b"m" * 32
REPO = "a" * 32
BRANCH = "c" * 16
CANARY = "CANARY-7f3a9c1e-do-not-log-me"


def turn(content: str | None = "done", tokens_in: int = 11, tokens_out: int = 5) -> ChatTurn:
    return ChatTurn(content=content, tool_calls=(), tokens_in=tokens_in, tokens_out=tokens_out, finish="stop")


class Script(ScriptedChat):
    """`ScriptedChat` that also raises non-`ScoutError` exceptions (what a raw HTTP client would throw)."""

    def __init__(self, script: Sequence[ChatTurn | Exception]) -> None:
        super().__init__(cast("Sequence[ChatTurn | ScoutError]", script))

    async def complete(self, req: ChatRequest) -> ChatTurn:
        nxt = self._script[self._next] if self._next < len(self._script) else None
        if isinstance(nxt, Exception) and not isinstance(nxt, ScoutError):
            self.requests.append(req)
            self._next += 1
            raise nxt
        return await super().complete(req)


def chat_req(**over: Any) -> ChatRequest:
    base: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "You explore code."},
            {"role": "user", "content": f"where is {CANARY} defined?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "t1", "function": {"name": "grep", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "t1", "content": f"src/a.py:3: def {CANARY}(): ..."},
        ],
        "tools": [{"type": "function", "function": {"name": "grep", "description": f"find {CANARY}"}}],
    }
    return ChatRequest(**(base | over))


def make(
    chat: Any = None, limits: Limits | None = None, *, with_chat: bool = True
) -> tuple[GatewayService, ScriptedChat, FakeClock, FakeStore]:
    clock, store = FakeClock(), FakeStore()
    scripted = chat if chat is not None else ScriptedChat([turn()] * 8)
    svc = GatewayService(
        store,
        clock,
        MASTER,
        embedder=FakeEmbedder(),
        chat=scripted if with_chat else None,
        limiter=RateLimiter(clock, limits),
    )
    return svc, scripted, clock, store


def tenant_gw(svc: GatewayService, key_id: str = "k1") -> TenantGateway:
    return svc.for_tenant(Tenant(tenant="acme", key_id=key_id))


# --- passthrough and metering ---------------------------------------------------------------------------


async def test_complete_forwards_the_request_unchanged_and_returns_the_turn():
    """FR-G1: the gateway is a `ChatModel`; the request reaches the injected model as sent."""
    svc, chat, _, _ = make(ScriptedChat([turn("hello", 40, 7)]))
    req = chat_req(temperature=0.2, max_tokens=99, thinking=False, tool_choice="required")
    got = await tenant_gw(svc).complete(req)
    assert got == turn("hello", 40, 7)
    assert chat.calls == 1
    sent = chat.requests[0]
    assert (sent.messages, sent.tools) == (list(req.messages), list(req.tools))
    assert (sent.temperature, sent.max_tokens, sent.thinking, sent.tool_choice) == (
        0.2,
        99,
        False,
        "required",
    )


async def test_gateway_without_a_chat_model_is_a_typed_unavailable_error():
    svc, _, _, _ = make(with_chat=False)
    with pytest.raises(ScoutError) as exc:
        await tenant_gw(svc).complete(chat_req())
    assert exc.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert CANARY not in str(exc.value)


async def test_usage_is_metered_without_content(caplog: pytest.LogCaptureFixture):
    """Tokens in/out, latency, sizes and status are recorded; the content is not."""

    class Slow:
        async def complete(self, req: ChatRequest) -> ChatTurn:
            clock.advance(0.25)
            return turn("x", 321, 45)

    svc, _, clock, _ = make(Slow())
    with caplog.at_level(logging.DEBUG):
        await tenant_gw(svc).complete(chat_req())
    (rec,) = [r for r in caplog.records if r.name == LOGGER.name]
    text = rec.getMessage()
    for part in ("op=chat", "key=k1", "status=ok", "latency_ms=250.0", "messages=4", "tools=1"):
        assert part in text
    assert "tokens_in=321" in text
    assert "tokens_out=45" in text
    assert CANARY not in text


# --- wire caps: oversize is rejected before the model ---------------------------------------------------


@pytest.mark.parametrize(
    "over",
    [
        {"messages": [{"role": "user", "content": CANARY}] * 1001},
        {"messages": []},
        {"tools": [{"type": "function", "function": {"name": CANARY}}] * 65},
        {"max_tokens": 32_769},
        {"max_tokens": 0},
        {"temperature": 2.5},
        {"top_p": 1.5},
    ],
    ids=["1001-messages", "no-messages", "65-tools", "max_tokens-high", "max_tokens-zero", "temp", "top_p"],
)
async def test_oversize_or_out_of_range_requests_never_reach_the_model(over: dict[str, Any]):
    svc, chat, _, _ = make()
    with pytest.raises(ScoutError) as exc:
        await tenant_gw(svc).complete(chat_req(**over))
    assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    assert chat.calls == 0
    assert CANARY not in str(exc.value)
    assert CANARY not in "".join(traceback.format_exception(exc.value))


async def test_the_caps_are_inclusive_at_1000_messages_and_64_tools():
    svc, chat, _, _ = make()
    msgs = [{"role": "user", "content": "m"}] * 1000
    tools = [{"type": "function", "function": {"name": "t"}}] * 64
    await tenant_gw(svc).complete(chat_req(messages=msgs, tools=tools, max_tokens=32_768))
    assert chat.calls == 1


# --- per-key concurrency limit --------------------------------------------------------------------------


class Gate:
    """A chat model that holds every call open until released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.entered = 0

    async def complete(self, req: ChatRequest) -> ChatTurn:
        self.entered += 1
        await self.release.wait()
        return turn()


async def test_two_concurrent_chats_per_key_then_rate_limited_and_other_keys_unaffected():
    gate = Gate()
    svc, _, _, _ = make(gate)
    k1, k2 = tenant_gw(svc, "k1"), tenant_gw(svc, "k2")
    first = [asyncio.create_task(k1.complete(chat_req())) for _ in range(2)]
    other = asyncio.create_task(k2.complete(chat_req()))
    for _ in range(5):
        await asyncio.sleep(0)
    assert gate.entered == 3
    with pytest.raises(ScoutError) as exc:
        await asyncio.wait_for(k1.complete(chat_req()), timeout=2)  # a missing limit would block on the gate
    assert exc.value.code is ErrorCode.RATE_LIMITED
    assert exc.value.retryable
    assert exc.value.details["retry_after"] > 0
    assert gate.entered == 3  # the third call never reached the model
    gate.release.set()
    await asyncio.gather(*first, other)
    await k1.complete(chat_req())  # the slots were released


async def test_the_slot_is_released_when_the_model_fails():
    svc, _, _, _ = make(ScriptedChat([ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, "down")] * 5 + [turn()]))
    gw = tenant_gw(svc)
    for _ in range(5):  # more failures than slots: a leak would turn these into RATE_LIMITED
        with pytest.raises(ScoutError) as exc:
            await gw.complete(chat_req())
        assert exc.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert (await gw.complete(chat_req())).content == "done"


# --- typed upstream errors and the context-length contract ----------------------------------------------


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, f"vLLM 503 for prompt {CANARY}"), "UPSTREAM_UNAVAILABLE"),
        (ScoutError(ErrorCode.UPSTREAM_TIMEOUT, f"timed out on {CANARY}"), "UPSTREAM_TIMEOUT"),
        (TimeoutError(f"read timeout while sending {CANARY}"), "UPSTREAM_TIMEOUT"),
        (ConnectionError(f"connect failed; body was {CANARY}"), "UPSTREAM_UNAVAILABLE"),
        (RuntimeError(f"boom {CANARY}"), "UPSTREAM_UNAVAILABLE"),
        (ScoutError(ErrorCode.INTERNAL, f"adapter bug {CANARY}"), "INTERNAL"),
    ],
    ids=["unavailable", "timeout", "TimeoutError", "ConnectionError", "RuntimeError", "internal"],
)
async def test_upstream_failures_map_to_typed_errors_and_never_echo_the_prompt(raised: Exception, code: str):
    svc, _, _, _ = make(Script([raised]))
    with pytest.raises(ScoutError) as exc:
        await tenant_gw(svc).complete(chat_req())
    assert exc.value.code is ErrorCode(code)
    assert exc.value.retryable is (code != "INTERNAL")
    assert CANARY not in str(exc.value)
    assert CANARY not in repr(exc.value.details)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None  # the upstream message is not reachable through the chain either
    assert CANARY not in "".join(traceback.format_exception(exc.value))


@pytest.mark.parametrize(
    "raised",
    [
        ScoutError(
            ErrorCode.INVALID_ARGUMENT,
            f"bad request: {CANARY}",
            details={"reason": "context_length", "prompt": CANARY},
        ),
        ScoutError(
            ErrorCode.INVALID_ARGUMENT,
            f"This model's maximum context length is 65536 tokens. However, you requested 70000 ({CANARY})",
        ),
    ],
    ids=["reason-detail", "vllm-message-only"],
)
async def test_context_length_errors_keep_the_reason_the_explore_loop_relies_on(raised: ScoutError):
    """1.11a: `_is_context_error` accepts `details.reason == "context_length"`; the gateway always sets it."""
    svc, _, _, _ = make(Script([raised]))
    with pytest.raises(ScoutError) as exc:
        await tenant_gw(svc).complete(chat_req())
    assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    assert exc.value.details == {"reason": "context_length"}  # no prompt echo in details
    assert not exc.value.retryable
    assert CANARY not in str(exc.value)


async def test_an_ordinary_bad_request_is_not_mislabelled_as_context_length():
    svc, _, _, _ = make(ScriptedChat([ScoutError(ErrorCode.INVALID_ARGUMENT, f"unknown field {CANARY}")]))
    with pytest.raises(ScoutError) as exc:
        await tenant_gw(svc).complete(chat_req())
    assert exc.value.code is ErrorCode.INVALID_ARGUMENT
    assert "reason" not in exc.value.details
    assert CANARY not in str(exc.value)


async def test_an_upstream_rate_limit_keeps_only_numeric_retry_hints():
    err = ScoutError(
        ErrorCode.RATE_LIMITED, f"slow down {CANARY}", details={"retry_after": 2.5, "echo": CANARY}
    )
    svc, _, _, _ = make(ScriptedChat([err]))
    with pytest.raises(ScoutError) as exc:
        await tenant_gw(svc).complete(chat_req())
    assert exc.value.code is ErrorCode.RATE_LIMITED
    assert exc.value.details == {"retry_after": 2.5}
    assert CANARY not in str(exc.value)


# --- the no-logging guarantee ---------------------------------------------------------------------------


class Capture:
    """Everything observable that could leak: every log record of every logger at DEBUG (root, named, and
    loggers that do not propagate), warnings, and (via `capfd`) the stdout/stderr file descriptors."""

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []
        self.warnings: list[warnings.WarningMessage] = []

    def text(self) -> str:
        parts = [
            f"{r.name} {r.levelname} {r.getMessage()} {r.args!r} {r.exc_text} {r.stack_info}"
            for r in self.records
        ]
        for r in self.records:
            if r.exc_info and r.exc_info[1] is not None:
                parts.append("".join(traceback.format_exception(r.exc_info[1])))
        parts += [str(w.message) for w in self.warnings]
        return "\n".join(parts)


@contextmanager
def capture_everything() -> Iterator[Capture]:
    cap = Capture()

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if not any(r is record for r in cap.records):  # one record reaches several handlers
                cap.records.append(record)

    handler = Collect(level=logging.NOTSET)
    loggers = [
        logging.getLogger(),
        *[lg for lg in logging.root.manager.loggerDict.values() if isinstance(lg, logging.Logger)],
    ]
    saved = [(lg, lg.level, lg.disabled) for lg in loggers]
    prior_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    for lg in loggers:
        lg.setLevel(logging.DEBUG)
        lg.disabled = False
        lg.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cap.warnings = caught
            yield cap
    finally:
        for lg, level, disabled in saved:
            lg.removeHandler(handler)
            lg.setLevel(level)
            lg.disabled = disabled
        logging.disable(prior_disable)


def test_the_capture_harness_sees_a_deliberate_leak(capfd: pytest.CaptureFixture[str]):
    """Negative control: logging the canary on a named, non-propagating logger, or printing it, is caught."""
    odd = logging.getLogger("scout.service.odd")
    odd.propagate = False
    try:
        with capture_everything() as cap:
            odd.debug("prompt was %s", CANARY)
            warnings.warn(CANARY, stacklevel=1)
            print(CANARY)  # noqa: T201
        out = capfd.readouterr()
    finally:
        odd.propagate = True
    assert CANARY in cap.text()
    assert CANARY in out.out


def wire_row(text: str) -> WireRow:
    ck = hashlib.sha256(text.encode()).hexdigest()[:32]
    return WireRow(
        id=hashlib.sha256(b"row").hexdigest()[:32],
        file_id="f" * 32,
        path_enc="enc/a.py",
        ext_tag="e" * 16,
        dir_tags=["d" * 16],
        file_hash="h-a",
        chunk_key=ck,
        loc_enc="bG9j",
        tokens=["q" * 32],
    )


async def test_canary_in_chat_search_embed_and_seed_inputs_never_reaches_any_output(
    capfd: pytest.CaptureFixture[str],
):
    """AC-13.3 / FR-X7: prompt, tool result, tool spec, model reply, query, chunk text and a bad seed sketch
    all carry the canary; nothing logged, printed, warned or raised contains it."""
    reply = ChatTurn(
        content=f"found {CANARY}",
        tool_calls=(ToolCall(id="c1", name="grep", arguments={"pattern": CANARY}),),
        tokens_in=9,
        tokens_out=4,
        finish="tool_calls",
    )
    svc, _, _, _ = make(
        Script(
            [
                reply,
                ScoutError(ErrorCode.UPSTREAM_UNAVAILABLE, f"echo {CANARY}"),
                RuntimeError(CANARY),
                ScoutError(ErrorCode.INVALID_ARGUMENT, f"maximum context length {CANARY}"),
            ]
        )
    )
    gw = tenant_gw(svc)
    raised: list[BaseException] = []

    async def attempt(coro: Any) -> Any:
        try:
            return await coro
        except Exception as exc:
            raised.append(exc)
            return None

    capfd.readouterr()
    with capture_everything() as cap:
        got = await gw.complete(chat_req())  # success path (model reply carries the canary)
        for _ in range(3):
            await attempt(gw.complete(chat_req()))  # three failure paths
        await attempt(gw.complete(chat_req(messages=[{"role": "user", "content": CANARY}] * 1001)))

        await gw.sync_begin(SyncBeginReq(repo_id=REPO, branch_id=BRANCH, root="0" * 64, expected_chunks=0))
        await gw.embed(
            EmbedReq(
                repo_id=REPO,
                branch_id=BRANCH,
                texts={wire_row(f"def {CANARY}(): pass").chunk_key: f"def {CANARY}(): pass"},
                rows=[wire_row(f"def {CANARY}(): pass")],
            )
        )
        res = await gw.search(SearchReq(repo_id=REPO, branch_id=BRANCH, query=f"find {CANARY}"))
        assert len(res.hits) == 1  # the flow really ran
        subkeys = keys.derive(await gw.team_key())
        good_sketch = base64.b64encode(minhash.sketch(subkeys, [b"x" * 16])).decode()
        await gw.seed(SeedReq(repo_id=REPO, branch_id=BRANCH, minhash_b64=good_sketch, proofs=[CANARY]))

        await attempt(gw.search(SearchReq(repo_id="b" * 32, branch_id=BRANCH, query=f"find {CANARY}")))
        await attempt(
            gw.embed(
                EmbedReq(
                    repo_id="b" * 32,
                    branch_id=BRANCH,
                    texts={wire_row(CANARY).chunk_key: CANARY},
                    rows=[wire_row(CANARY)],
                )
            )
        )
        await attempt(gw.seed(SeedReq(repo_id=REPO, branch_id=BRANCH, minhash_b64=CANARY, proofs=[CANARY])))
    out = capfd.readouterr()

    assert got.content == f"found {CANARY}"  # the caller still gets the model's content
    assert len(raised) >= 7
    assert cap.records, "the harness captured nothing: the guarantee would be vacuous"
    assert any(r.name == LOGGER.name for r in cap.records)  # metering is recorded
    text = cap.text()
    assert CANARY not in text
    assert CANARY not in out.out + out.err
    for exc in raised:
        assert CANARY not in str(exc)
        assert CANARY not in repr(exc)
        assert CANARY not in "".join(traceback.format_exception(exc))
        details = getattr(exc, "details", {})
        assert CANARY not in repr(details)


async def test_logged_records_carry_only_the_fixed_fields():
    """The request log's whole vocabulary: op, key id, status, latency, sizes, token counts, error class."""
    svc, _, _, _ = make(Script([RuntimeError(CANARY), turn()]))
    gw = tenant_gw(svc)
    with capture_everything() as cap:
        with pytest.raises(ScoutError):
            await gw.complete(chat_req())
        await gw.complete(chat_req())
    lines = [r.getMessage() for r in cap.records if r.name == LOGGER.name]
    assert len(lines) == 2
    assert "status=UPSTREAM_UNAVAILABLE" in lines[0]
    assert "err=ScoutError" in lines[0]  # the class we raised, never the upstream message
    assert "RuntimeError" not in lines[0]
    assert "status=ok" in lines[1]
    for r in (r for r in cap.records if r.name == LOGGER.name):
        assert r.exc_info is None
        assert isinstance(r.args, tuple)
        assert len(r.args) == 10  # op, key, status, latency, messages, tools, chars, tokens x2, err
        assert [type(a) for a in r.args[3:9]] in (  # type: ignore[index]
            [float, int, int, int, int, int],
            [float, int, int, int, type(None), type(None)],
        )
