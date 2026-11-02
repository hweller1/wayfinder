"""Tests for wayfinder.sft.convert_teacher (filters pure; the tokenizer test runs when a local Glimmer tokenizer exists)."""

import copy
import json
import os

import pytest

from wayfinder.sft import convert_teacher as ct


def call(i, name, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def make_row(task="t1", seed=0, turns=3, gold=("a.py",), report_paths=("a.py",), semantic=False, **kw):
    msgs = [
        {"role": "system", "content": "SYS\n\nnudge"},
        {"role": "user", "content": "brief\n\nReport at most 12 results."},
        {"role": "assistant", "content": "chatty", "reasoning_content": "think", "tool_calls": [call(1, "grep", pattern="x")]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.py:1: x"},
        {"role": "assistant", "content": None, "tool_calls": [call(2, "read_file", path="a.py", start_line=1, end_line=5)]},
        {"role": "tool", "tool_call_id": "c2", "content": "a.py lines 1-5\n1\tx"},
        {"role": "assistant", "content": None, "reasoning_content": "done", "tool_calls": [call(3, "report", results=[
            {"path": p, "start_line": 1, "end_line": 2, "why": "w"} for p in report_paths], summary="s", gaps=[])]},
        {"role": "tool", "tool_call_id": "c3", "content": '{"status": "report accepted"}'},
    ]
    row = {"task_id": task, "seed": seed, "repo": "o/r", "model": "spark", "semantic": semantic, "turns": turns, "gold_files": list(gold),
           "messages": msgs, "explore_result": {"budget_exhausted": False}, "error": None, "all_gold_found": True,
           "all_gold_reported": True, "recall": 1.0, "recall_reported": 1.0, "recall_opened": 1.0, "nudge": "nudge"}
    row.update(kw)
    return row


def test_p75():
    assert ct.p75([5]) == 5
    assert ct.p75([1, 2, 3, 4]) == 3
    assert ct.p75([8, 8, 8, 40]) == 8


def test_protocol_ok_and_failures():
    assert ct.protocol_problem(make_row()) is None
    r = make_row(); r["explore_result"]["budget_exhausted"] = True
    assert ct.protocol_problem(r) == "budget_exhausted"
    r = make_row(); r["messages"] = r["messages"][:6]
    assert ct.protocol_problem(r) == "no_report"
    r = make_row(); r["messages"].insert(4, {"role": "user", "content": "Reply with a tool call."})
    assert ct.protocol_problem(r) == "mid_run_user_nudge"
    r = make_row(); r["messages"][2]["reasoning_content"] = "x <|eot|> y"
    assert ct.protocol_problem(r) == "special_token_in_text"
    r = make_row(); r["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{bad"
    assert ct.protocol_problem(r) == "bad_tool_args"


def test_evidence_covered():
    assert ct.evidence_covered(make_row())
    assert not ct.evidence_covered(make_row(report_paths=("a.py", "never_opened.py")))
    r = make_row(); r["messages"][5]["content"] = "error: NOT_FOUND: file not found"
    assert not ct.evidence_covered(r)  # a failed read_file does not count as opened


def test_to_messages_format():
    m = ct.to_messages(make_row())
    assert m[-1]["role"] == "assistant" and m[-1]["tool_calls"][0]["function"]["name"] == "report"  # report-accepted result dropped
    assert isinstance(m[2]["tool_calls"][0]["function"]["arguments"], dict)
    assert m[2]["content"] == "" and m[2]["reasoning_content"] == "think"  # chatty content dropped, reasoning kept
    assert "reasoning_content" not in m[4]


def test_tools_variant():
    tools = [{"function": {"name": n}} for n in ("semantic_search", "grep", "glob", "read_file", "report")]
    assert ct.tools_for(make_row(semantic=True), tools)[1] == "scout5"
    t4, v = ct.tools_for(make_row(semantic=False), tools)
    assert v == "scout4_grep_only" and [t["function"]["name"] for t in t4] == ["grep", "glob", "read_file", "report"]


class FakeTok:
    """Whitespace-free stand-in: one token per character of the rendered text, so offsets are trivial."""

    def apply_chat_template(self, messages, tools=None, tokenize=False):
        out = "<|begin_of_text|>"
        for m in messages:
            if m["role"] == "assistant":
                if m.get("reasoning_content"):
                    out += "<|start|>assistant to=self<|message|>" + m["reasoning_content"] + "<|eom|>"
                for tc in m["tool_calls"]:
                    out += f"<|start|>assistant to={tc['function']['name']}<|message|>{json.dumps(tc['function']['arguments'])}<|eom|>"
                out = out[: -len("<|eom|>")] + "<|eot|>"
            else:
                out += f"<|start|>{m['role']}<|message|>{m['content']}<|eot|>"
        return out

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


SPLITS = {"tasks": {"t1": {"split": "train", "category": "c"}, "t2": {"split": "train"}, "tdev": {"split": "dev"}}}
TOOLS = [{"function": {"name": n}} for n in ("semantic_search", "grep", "glob", "read_file", "report")]


def test_funnel_filters_and_dedup():
    rows = [
        make_row("t1", 0, turns=3),
        make_row("t1", 1, turns=3),
        make_row("t1", 2, turns=30),  # over p75 (cap = 3 of [3,3,30,3])
        make_row("t1", 3, turns=3, all_gold_found=False),  # not all gold
        make_row("t2", 0, report_paths=("a.py", "zzz.py")),  # evidence not covered
        make_row("tdev", 0),  # wrong split
        make_row("t1", 0, turns=5),  # duplicate (t1, 0): the shorter run is kept
    ]
    rows[3]["turns"] = 3
    ex, st = ct.convert_rows(rows, SPLITS, FakeTok(), TOOLS)
    assert [e["id"] for e in ex] == ["t1|teacher|s0", "t1|teacher|s1"]
    assert ex[0]["meta"]["turns"] == 3
    f = st["funnel"]
    assert f["rows_in"] == 7 and f["train_split"] == 6 and f["converted"] == 6 and f["all_gold_found"] == 5
    assert f["evidence_covered"] == 4 and f["turns_le_p75"] == 3 and f["dedup"] == 2
    assert st["drops"]["duplicate_task_seed"] == 1


def test_example_contract_and_spans():
    ex, _ = ct.convert_rows([make_row()], SPLITS, FakeTok(), TOOLS)
    e = ex[0]
    assert set(e) >= {"id", "input_ids", "assistant_spans", "n_tokens", "tools_variant", "messages", "meta"}
    assert e["n_tokens"] == len(e["input_ids"])
    assert all(0 < a < b <= e["n_tokens"] for a, b in e["assistant_spans"])
    assert {"task_id", "semantic", "turns", "recall", "source"} <= set(e["meta"]) and e["meta"]["source"] == "teacher"
    assert len(e["assistant_spans"]) == 3 + 2  # three calls plus two reasoning segments


def test_max_tokens_and_relaxed_recall():
    ex, st = ct.convert_rows([make_row()], SPLITS, FakeTok(), TOOLS, max_tokens=50)
    assert ex == [] and st["drops"]["fits_32k"] == 1
    r = make_row(all_gold_found=False, recall=0.6)
    assert ct.convert_rows([r], SPLITS, FakeTok(), TOOLS)[0] == []
    assert len(ct.convert_rows([r], SPLITS, FakeTok(), TOOLS, min_recall=0.5)[0]) == 1


def test_a1_examples_flagging():
    a = {"id": "x", "input_ids": [1, 2], "assistant_spans": [[1, 2]], "tools_variant": "scout5",
         "meta": {"tier": "strict", "all_gold_found": True}}
    b = copy.deepcopy(a); b["meta"]["all_gold_found"] = False
    c = copy.deepcopy(a); c["meta"]["tier"] = "lossy_render"
    out = ct.a1_examples([a, b, c], TOOLS)
    assert len(out) == 1 and out[0]["meta"]["source"] == "a1" and out[0]["meta"]["semantic"] is True


@pytest.mark.skipif(not os.path.exists("/tmp/wf_tok/glimmer/tokenizer.json"), reason="needs the local Glimmer tokenizer")
def test_real_template_renders_dict_args_and_masks():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/tmp/wf_tok/glimmer")
    tools = json.load(open(os.path.join(os.path.dirname(__file__), "..", "serve", "scout_tool_specs.json")))
    ex, _ = ct.convert_rows([make_row()], SPLITS, tok, tools)
    e = ex[0]
    text = tok.decode(e["input_ids"])
    assert "<|start|>assistant to=self<|message|>think<|eom|>" in text and "chatty" not in text
    assert 'to=read_file' in text and "report accepted" not in text
    spans = [tok.decode(e["input_ids"][a:b]) for a, b in e["assistant_spans"]]
    assert spans[0].startswith(" to=self") and spans[-1].endswith("<|eot|>")
    assert "SYS" not in "".join(spans) and "a.py:1: x" not in "".join(spans)


def test_e2e_exclusion():
    rows = [make_row("t1", 0), make_row("t2", 0)]
    ex, st = ct.convert_rows(rows, SPLITS, FakeTok(), TOOLS, exclude_tasks={"t2"})
    assert [e["id"] for e in ex] == ["t1|teacher|s0"]
    assert st["drops"]["e2e_eval_task"] == 1 and st["funnel"]["train_split"] == 2 and st["funnel"]["not_e2e_task"] == 1
