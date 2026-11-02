"""Rubric scorer tests: hand-built trajectories with known term values, plus the weights file contract."""

import json
from pathlib import Path

import pytest

from wayfinder import rubric as rb

ROOT = Path(__file__).resolve().parents[1]


def call(i, name, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def asst(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def tool(i, content):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": content}


def hit(path, s=1, e=5, why="relevant"):
    return {"path": path, "start_line": s, "end_line": e, "why": why, "source": "read", "stale": False, "score": None}


def explore(hits, turns=3, out=1000, exhausted=False, gaps=()):
    return {"results": hits, "summary": "s", "gaps": list(gaps), "budget_exhausted": exhausted,
            "usage": {"turns": turns, "tool_calls": {}, "model_tokens": {"in": 10, "out": out}, "wall_ms": 1}}


def good_row():
    msgs = [
        {"role": "system", "content": "x"}, {"role": "user", "content": "brief"},
        asst(call(1, "grep", pattern="validateFact")),
        tool(1, "src/a.py:10: def validateFact(x):\nsrc/b.py:3: validateFact(1)"),
        asst(call(2, "read_file", path="src/a.py", start_line=1, end_line=40)),
        tool(2, "src/a.py lines 1-40\n1\timport x"),
        asst(call(3, "read_file", path="src/b.py")),
        tool(3, "src/b.py lines 1-20\n1\tfoo"),
        asst(call(4, "report", results=[], summary="s")),
        tool(4, '{"status": "report accepted"}'),
    ]
    return {"gold_files": ["src/a.py", "src/b.py"], "messages": msgs, "max_turns": 40,
            "explore_result": explore([hit("src/a.py", 10, 12), hit("src/b.py", 3, 3)], turns=4), "error": None}


def test_perfect_run_scores_high_and_terms_are_exact():
    o = rb.score_rollout(good_row())
    t = o["terms"]
    assert t["recall"] == 1.0 and t["all_gold_found"] == 1.0 and t["evidence"] == 1.0
    assert t["single_report"] == 1.0 and t["stops_in_budget"] == 1.0 and t["no_repeated_queries"] == 1.0
    assert t["routing"] == 1.0 and t["output_valid"] == 1.0 and t["no_rereads"] == 1.0
    assert t["first_gold"] == 1.0 and t["hunk_overlap"] is None
    assert t["turn_eff"] == pytest.approx(0.9) and t["token_eff"] == pytest.approx(0.875)
    assert t["focus"] == pytest.approx(1 - 2 / rb.FOCUS_FILES)
    assert o["score"] > 0.9
    assert sum(o["weights_used"].values()) == pytest.approx(1.0)


def test_missing_gold_lowers_score_monotonically():
    full = rb.score_rollout(good_row())["score"]
    row = good_row()
    row["gold_files"] = ["src/a.py", "src/b.py", "src/never.py"]
    part = rb.score_rollout(row)
    assert part["terms"]["recall"] == pytest.approx(2 / 3) and part["terms"]["all_gold_found"] == 0.0
    assert part["score"] < full


def test_hunk_overlap():
    row = good_row()
    row["gold_hunks"] = {"src/a.py": [[10, 19]], "src/b.py": [[3, 3]]}
    # reported a.py 10-12 (3 of 10 lines) and b.py 3-3 (1 of 1): 4 of 11
    assert rb.score_rollout(row)["terms"]["hunk_overlap"] == pytest.approx(4 / 11)


def test_evidence_penalises_citing_an_unopened_file():
    row = good_row()
    row["explore_result"]["results"].append(hit("src/ghost.py"))
    o = rb.score_rollout(row)
    assert o["detail"]["cited_never_seen"] == ["src/ghost.py"]
    assert o["terms"]["evidence"] == pytest.approx(2 / 3 - 0.5 / 3)
    assert o["terms"]["output_valid"] < 1.0  # ghost.py is not a seen/known file


def test_convergence_terms():
    row = good_row()
    row["messages"] += [asst(call(5, "grep", pattern="validateFact")), tool(5, "no matches"),
                        asst(call(6, "report", results=[], summary="again")), tool(6, "{}")]
    row["explore_result"]["budget_exhausted"] = True
    t = rb.score_rollout(row)["terms"]
    assert t["single_report"] == 0.0 and t["stops_in_budget"] == 0.0
    assert t["no_repeated_queries"] == pytest.approx(1 / 2)
    row2 = good_row()
    row2["explore_result"] = None
    row2["error"] = "UPSTREAM_TIMEOUT: x"
    assert rb.score_rollout(row2)["terms"]["stops_in_budget"] == 0.0


def test_rereads():
    row = good_row()
    row["messages"] += [asst(call(7, "read_file", path="src/a.py", start_line=1, end_line=40)), tool(7, "src/a.py lines 1-40\n1\tx")]
    assert rb.score_rollout(row)["terms"]["no_rereads"] == pytest.approx(2 / 3)


def test_routing_identifier_vs_conceptual_and_missing_path():
    msgs = [
        asst(call(1, "semantic_search", query="validateFact")),  # identifier through semantic: mis-routed
        tool(1, "src/a.py:1-5 (score 0.9)\nsnippet"),
        asst(call(2, "grep", pattern="where is the retry logic for uploads")),  # conceptual through grep: mis-routed
        tool(2, "no matches"),
        asst(call(3, "semantic_search", query="where is the retry logic for uploads")),  # correct
        tool(3, "src/a.py:1-5 (score 0.9)\nsnippet"),
        asst(call(4, "grep", pattern="validateFact")),  # correct
        tool(4, "src/a.py:2: validateFact"),
        asst(call(5, "read_file", path="src/nope.py")),
        tool(5, "error: NOT_FOUND: no such file"),
    ]
    row = {"gold_files": ["src/a.py"], "messages": msgs, "semantic": True, "explore_result": explore([hit("src/a.py")], turns=5)}
    o = rb.score_rollout(row)
    assert o["detail"]["routing"] == {"queries_scored": 4, "routed_ok": 2, "nonexistent_path_calls": 1}
    # 2 of 4 correctly routed, and 1 of the 1 path-bearing calls pointed at a file that does not exist
    assert o["terms"]["routing"] == pytest.approx(0.5 * (1 - 1 / 1))
    # without a semantic tool the conceptual grep is not penalised
    row["semantic"] = False
    assert rb.score_rollout(row)["detail"]["routing"]["queries_scored"] == 2


def test_gaps_stated_when_report_is_incomplete():
    row = good_row()
    row["gold_files"] = ["src/a.py", "src/b.py", "src/never.py"]
    assert rb.score_rollout(row)["terms"]["output_valid"] == pytest.approx(0.8)
    row["explore_result"]["gaps"] = ["could not find never.py"]
    assert rb.score_rollout(row)["terms"]["output_valid"] == pytest.approx(1.0)


def test_classify_query():
    assert rb.classify_query("validateAndNormalizeFact") == "identifier"
    assert rb.classify_query("src/utils/ontology.ts") == "identifier"
    assert rb.classify_query("where is the retry logic") == "conceptual"
    assert rb.classify_query("two words") == "other"
    assert rb.classify_query("") == "other"


def test_chunking_robustness():
    assert rb.chunking_robustness(0.5, 0.5) == 1.0
    assert rb.chunking_robustness(0.4, 0.8) == pytest.approx(0.5)
    assert rb.chunking_robustness(0.0, 0.0) == 1.0


def test_weights_file_contract():
    d = json.loads((ROOT / "data" / "rubric_weights.json").read_text())
    w = rb.load_weights()
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(w) == set(rb.DEFAULT_WEIGHTS) and all(v >= 0 for v in w.values())
    assert d["pooled_r2_r3"] == "not computed by design"
    assert set(d["rounds"]) == {"r3", "r2"}
    assert d["rounds"]["r3"]["full"]["n"] == 136 and d["rounds"]["r2"]["full"]["n"] == 70
    assert w["all_gold_found"] >= 0.1 and w["recall"] >= 0.0


def test_score_entrypoint_is_what_rollout_score_looks_for():
    assert rb.score is rb.score_rollout
    out = rb.score_rollout({"gold_files": [], "messages": [], "explore_result": None, "error": "x"})
    assert 0.0 <= out["score"] <= 1.0 and out["terms"]["recall"] is None
