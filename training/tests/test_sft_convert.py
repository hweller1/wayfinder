"""Unit tests for wayfinder.sft.convert (pure functions; the .eval end-to-end path is exercised by convert_modal)."""

from wayfinder.sft.convert import assistant_char_spans, parse_envelope, render_grep, render_read, synthesize_report

ENV = (
    '<results tool="read_file" backend="filesystem" query="a.py" k="5" returned="1">\n'
    '<result rank="1" id="a.py" source="a.py" score="null" truncated="false">\nx = 1\ny = 2\n</result>\n</results>'
)


def test_parse_and_render_read():
    n, res = parse_envelope(ENV)  # type: ignore[misc]
    assert n == 1 and res[0]["body"] == "x = 1\ny = 2"
    text, lossy, end = render_read(res, 10, None)
    assert text == "a.py lines 10-11\n10\tx = 1\n11\ty = 2" and not lossy and end == 11


def test_render_grep():
    env = (
        '<results tool="grep" backend="ripgrep" query="x" k="5" returned="1">\n'
        '<result rank="1" id="a.py" source="a.py" score="null" truncated="false">\n3:foo x\n9:bar x\n</result>\n</results>'
    )
    _, res = parse_envelope(env)  # type: ignore[misc]
    assert render_grep(res)[0] == "a.py:3: foo x\na.py:9: bar x"


def test_report_requires_a_read_file():
    rep, info = synthesize_report("see `a.py` (lines 3-5): the thing", {"a.py": [(1, 50)]}, {"a.py", "b.py"})
    assert rep and rep["results"][0]["start_line"] == 3 and rep["results"][0]["end_line"] == 5
    none, info = synthesize_report("see b.py", {"a.py": [(1, 50)]}, {"a.py", "b.py"})
    assert none is None and info["cited_unread"] == 1


def test_loss_spans_cover_only_model_output():
    t = (
        "<|begin_of_text|><|start|>system<|message|>S<|eot|><|start|>user<|message|>U<|eot|>"
        "<|start|>assistant to=self<|message|>think<|eom|><|start|>assistant to=grep<|message|>CALL<|eot|>"
        "<|start|>tool grep<|message|>R<|eot|>"
    )
    sp = assistant_char_spans(t)
    got = [t[a:b] for a, b in sp]
    assert got == [" to=self<|message|>think<|eom|>", "<|start|>assistant to=grep<|message|>CALL<|eot|>"]
