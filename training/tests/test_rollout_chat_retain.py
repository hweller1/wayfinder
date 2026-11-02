"""CPU tests: `CompatChat` re-sends earlier assistant reasoning by default (production scout does; the SFT template renders it)."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

SCOUT_SRC = Path(__file__).resolve().parents[2] / "voyage-scout" / "src"
sys.path.insert(0, str(SCOUT_SRC))

from scout.core.ports.chat import ChatRequest  # noqa: E402

from wayfinder.rollout.chat import CompatChat  # noqa: E402


def _chat(bodies, **kw):
    def handler(req):
        bodies.append(json.loads(req.content))
        msg = {"role": "assistant", "content": "x", "reasoning_content": f"think{len(bodies)}"}
        return httpx.Response(200, json={"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    return CompatChat("https://llm.test/v1", "k", "m", transport=httpx.MockTransport(handler), **kw)


def _two_turns(chat):
    async def go():
        t1 = await chat.complete(ChatRequest(messages=[{"role": "user", "content": "q"}]))
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "go on"}]
        t2 = await chat.complete(ChatRequest(messages=msgs))
        return t1, t2

    return asyncio.run(go())


def test_default_is_on_and_resends_reasoning():
    bodies = []
    chat = _chat(bodies)
    assert chat.retain_reasoning is True
    t1, _ = _two_turns(chat)
    assert t1.reasoning == "think1"
    assert [m.get("reasoning_content") for m in bodies[1]["messages"] if m["role"] == "assistant"] == ["think1"]


def test_off_sends_nothing_and_loop_holds_no_reasoning():
    bodies = []
    chat = _chat(bodies, retain_reasoning=False)
    t1, _ = _two_turns(chat)
    assert t1.reasoning is None
    assert "reasoning_content" not in json.dumps(bodies[1])
    assert chat.turn_log[0]["reasoning_content"] == "think1"  # the transcript still records it
