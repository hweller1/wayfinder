"""Eval harness pieces: path extraction, recall, bootstrap determinism, stream parsing, row validation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # eval/ is a top-level directory, not a package

from eval.harness import arms, report, stats  # noqa: E402
from eval.harness.briefs import SUFFIX, brief_sha, build_brief, extract_paths, recall_at

FILES = ["src/a.py", "src/b/c.py", "README.md", "pkg/x.go"]


def test_brief_has_suffix_and_stable_hash() -> None:
    b = build_brief("Where is X?")
    assert b.endswith(SUFFIX)
    assert brief_sha(b) == brief_sha(build_brief("Where is X?"))


def test_extract_paths_rank_prefix_lines_and_hallucinations() -> None:
    text = (
        "See /work/repo/src/b/c.py:10-20 and `./src/a.py`, also src/ghost.py. Then README.md; src/a.py again."
    )
    got = extract_paths(text, FILES, root_prefixes=["/work/repo"])
    assert got == ["src/b/c.py", "src/a.py", "README.md"]


def test_extract_json_list() -> None:
    assert extract_paths('["pkg/x.go", "src/a.py"]', FILES) == ["pkg/x.go", "src/a.py"]


def test_recall_at_top_k() -> None:
    assert recall_at(["a", "b", "c"], {"a", "c", "z"}, 2) == 1 / 3
    assert recall_at(["a", "c"], {"a", "c"}) == 1.0


def test_bootstrap_deterministic_and_collapses() -> None:
    assert stats.bootstrap_ci([0.5] * 10, n=200) == (0.5, 0.5, 0.5)
    v = [0.1, 0.4, -0.2, 0.3, 0.0, 0.2]
    assert stats.bootstrap_ci(v, n=500) == stats.bootstrap_ci(v, n=500)
    mean, lo, hi = stats.bootstrap_ci(v, n=500)
    assert lo <= mean <= hi


def test_cluster_ci_wider_on_correlated() -> None:
    clusters = {"a": [1.0, 1.0, 1.0], "b": [-1.0, -1.0, -1.0], "c": [1.0, 1.0, 1.0], "d": [-1.0, -1.0, -1.0]}
    flat = [x for v in clusters.values() for x in v]
    _, tlo, thi = stats.bootstrap_ci(flat, n=500)
    _, clo, chi = stats.cluster_bootstrap_ci(clusters, n=500)
    assert chi - clo > thi - tlo


def stream(handback: str, agent_type: str = "Explore") -> str:
    evs = [
        {
            "type": "assistant",
            "parent_tool_use_id": None,
            "message": {
                "id": "m1",
                "model": "claude-sonnet-5-5",
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 1000, "output_tokens": 50},
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Agent", "input": {"subagent_type": "Explore"}}
                ],
            },
        },
        {
            "type": "user",
            "parent_tool_use_id": None,
            "tool_use_result": {
                "agentType": agent_type,
                "resolvedModel": "claude-haiku-5-5",
                "totalDurationMs": 4200,
                "totalTokens": 900,
                "totalToolUseCount": 2,
                "content": [{"type": "text", "text": handback}],
            },
            "message": {"content": [{"type": "tool_result", "content": handback}]},
        },
        {
            "type": "result",
            "result": "done src/a.py",
            "total_cost_usd": 0.05,
            "num_turns": 3,
            "duration_ms": 9000,
            "modelUsage": {"claude-sonnet-5-5": {"costUSD": 0.04}, "claude-haiku-5-5": {"costUSD": 0.01}},
            "permission_denials": [],
        },
    ]
    return "\n".join(json.dumps(e) for e in evs)


def test_parse_stream_and_row() -> None:
    p = arms.parse_stream(stream("found src/a.py:3-9 and pkg/x.go"))
    assert p.agent_type == "Explore"
    assert p.sub_ms == 4200
    assert p.cost_usd == 0.05
    assert p.parent_cost_usd is not None
    assert 0 < p.parent_cost_usd < 0.05
    task = {"native_id": "t", "repo": "o/r", "lang": "Go", "gold_files": ["src/a.py", "pkg/x.go", "zzz.py"]}
    row = arms.score_row("explore_default", task, "brief", p, FILES, "/w", 12.0, [])
    assert row["recall12"] == 2 / 3
    assert not row["all_gold"]
    assert row["fallthrough"] is False
    assert arms.valid_row(row) is None


def test_scout_arm_fallthrough_flag() -> None:
    p = arms.parse_stream(stream("src/a.py", agent_type="Explore"))
    task = {"native_id": "t", "repo": "o/r", "lang": "Go", "gold_files": ["src/a.py"]}
    row = arms.score_row(
        "super_v3", task, "b", p, FILES, "/w", 12.0, ["x rewrite lane=production reason=ready"]
    )
    assert row["fallthrough"] is True  # rewrite logged but the shim never ran
    p2 = arms.parse_stream(stream("src/a.py", agent_type=arms.SHIM_TYPE))
    row2 = arms.score_row(
        "super_v3", task, "b", p2, FILES, "/w", 12.0, ["x rewrite lane=production reason=ready"]
    )
    assert row2["fallthrough"] is False


def test_valid_row_guards() -> None:
    base = {
        "brief_sha": "x",
        "wall_s": 10,
        "handback_chars": 5,
        "cost_usd": 0.1,
        "resolved_subagent_model": "m",
    }
    assert arms.valid_row(base) is None
    assert arms.valid_row({**base, "handback_chars": 0}) == "empty handback"
    assert arms.valid_row({**base, "wall_s": 0.5}) is not None
    assert arms.valid_row({**base, "zero_tools": True}) is not None


@pytest.mark.skipif("not (Path(__file__).resolve().parents[2] / 'eval' / 'eval_tasks.jsonl').exists()", reason="withheld from the public release (derived from a pre-event benchmark); see RELEASE_CHECKLIST.md")
def test_report_pairs_only_complete_tasks(tmp_path: Path) -> None:
    def row(arm: str, task: str, rec: float, repo: str = "o/r") -> dict[str, object]:
        return {
            "task_id": task,
            "repo": repo,
            "lang": "Go",
            "arm": arm,
            "brief_sha": "x",
            "wall_s": 20.0,
            "explore_s": 5.0,
            "handback_chars": 9,
            "cost_usd": 0.1,
            "parent_cost_usd": 0.05,
            "subagent_cost_usd": 0.05,
            "resolved_subagent_model": "m",
            "recall12": rec,
            "all_gold": rec == 1.0,
            "final_recall12": rec,
            "fallthrough": False,
        }

    for arm, shift in zip(arms.ARMS, (0.0, 0.2, 0.1), strict=True):
        with (tmp_path / f"rows_{arm}.jsonl").open("w") as f:
            for i in range(6):
                f.write(
                    json.dumps(row(arm, f"t{i}", min(1.0, 0.3 + shift + i * 0.05), repo=f"o/r{i % 3}")) + "\n"
                )
    with (tmp_path / "rows_super_v3.jsonl").open("a") as f:
        f.write(json.dumps({**row("super_v3", "t9", 1.0), "fallthrough": True}) + "\n")
    good = report.load(tmp_path)
    res = report.summarize(good, report.load.bad)  # type: ignore[attr-defined]
    assert res["n_tasks_complete"] == 6
    assert res["fallthrough"]["super_v3"] == 1
    d = res["deltas"]["super_v3_minus_explore_default"]["recall12"]
    assert abs(d["mean"] - 0.2) < 1e-9
    assert d["ci95"][0] > 0
    assert "super_v3_minus_super_a1" in report.markdown(res)


def test_split_group_caps_rows_and_files() -> None:
    from types import SimpleNamespace

    from scout.core.contracts.wire import MAX_FILES, MAX_ROWS
    from scout.core.sync.engine import _split_group  # pyright: ignore[reportPrivateUsage]

    boiler = [SimpleNamespace(row=SimpleNamespace(file_id=f"f{i}")) for i in range(MAX_FILES + 86)]
    parts = _split_group(boiler)  # type: ignore[arg-type]
    assert sum(len(p) for p in parts) == len(boiler)
    assert all(len({c.row.file_id for c in p}) <= MAX_FILES and len(p) <= MAX_ROWS for p in parts)
    assert len(parts) == 2
    one_file = [SimpleNamespace(row=SimpleNamespace(file_id="f")) for _ in range(MAX_ROWS + 5)]
    assert [len(p) for p in _split_group(one_file)] == [MAX_ROWS, 5]  # type: ignore[arg-type]


def test_augment_arm_wiring_and_validity(tmp_path: Path) -> None:
    settings, _, _ = arms.ensure_support_files()
    cmd = arms.command(arms.AUGMENT, "b", settings, 1.0)
    assert "--plugin-dir" not in cmd  # no plugin, so no Explore rewrite
    assert "--strict-mcp-config" in cmd
    assert "mcp__scout__explore" in cmd
    cfg = json.loads((arms.WORK / "augment_mcp.json").read_text())
    args = cfg["mcpServers"]["scout"]["args"]
    assert args == ["mcp", "--tools", "explore,semantic_search"]
    assert "${SCOUT_KEY}" in json.dumps(cfg)  # expanded from the session environment, never written to disk
    env = arms.build_env(arms.AUGMENT, tmp_path / "cfg", key="k", gateway="https://gw", llm_key="g")
    assert env["SCOUT_KEY"] == "k"
    assert "SCOUT_LANE" not in env
    assert "SCOUT_BIN" not in env

    init = {"type": "system", "subtype": "init", "mcp_servers": [{"name": "scout", "status": "connected"}]}
    sub_call = {
        "type": "assistant",
        "parent_tool_use_id": "t1",
        "message": {
            "id": "m2",
            "model": "claude-sonnet-5-5",
            "usage": {"output_tokens": 5},
            "content": [
                {"type": "tool_use", "id": "s1", "name": "mcp__scout__explore", "input": {}},
                {"type": "tool_use", "id": "s2", "name": "Grep", "input": {}},
            ],
        },
    }
    raw = json.dumps(init) + "\n" + json.dumps(sub_call) + "\n" + stream("src/a.py")
    p = arms.parse_stream(raw)
    assert p.scout_calls == {"explore": 1}
    assert p.sub_tool_counts == {"mcp__scout__explore": 1, "Grep": 1}
    task = {"native_id": "t", "repo": "o/r", "lang": "Go", "gold_files": ["src/a.py"]}
    row = arms.score_row(arms.AUGMENT, task, "b", p, FILES, "/w", 12.0, [])
    assert row["fallthrough"] is False
    assert row["scout_calls"] == 1
    assert arms.valid_row(row) is None
    assert arms.valid_row({**row, "scout_mcp_status": "failed"}) is not None
    assert arms.valid_row({**row, "agent_type": arms.SHIM_TYPE}) is not None


@pytest.mark.skipif("not (Path(__file__).resolve().parents[2] / 'eval' / 'eval_tasks.jsonl').exists()", reason="withheld from the public release (derived from a pre-event benchmark); see RELEASE_CHECKLIST.md")
def test_report_adds_augment_pairs_and_usage(tmp_path: Path) -> None:
    for arm, shift in zip((*arms.ARMS, arms.AUGMENT), (0.0, 0.2, 0.1, 0.05), strict=True):
        rows = []
        for i in range(6):
            row = {
                "task_id": f"t{i}", "repo": f"r{i % 3}", "lang": "Go", "arm": arm, "brief_sha": "x",
                "wall_s": 20.0 + i, "handback_chars": 10, "cost_usd": 0.05, "resolved_subagent_model": "m",
                "zero_tools": False, "is_error": False, "fallthrough": False, "all_gold": False,
                "recall12": min(1.0, 0.3 + shift + 0.05 * i), "final_recall12": 0.3, "explore_s": 5.0,
                "parent_cost_usd": 0.03, "subagent_cost_usd": 0.02, "agent_type": "Explore",
                "scout_mcp_status": "connected", "scout_calls": i % 2, "scout_explore_calls": i % 2,
                "scout_search_calls": 0, "subagent_tool_calls": 4,
            }  # fmt: skip
            rows.append(row)
        (tmp_path / f"rows_{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    good = report.load(tmp_path)
    res = report.summarize(good, report.load.bad)  # type: ignore[attr-defined]
    assert "augment_v3_minus_explore_default" in res["deltas"]
    assert "augment_v3_minus_super_v3" in res["deltas"]
    assert res["augment_usage"]["tasks_with_scout_call"] == 3
    assert "augment_v3 scout-tool use" in report.markdown(res)


@pytest.mark.skipif("not (Path(__file__).resolve().parents[2] / 'eval' / 'eval_tasks.jsonl').exists()", reason="withheld from the public release (derived from a pre-event benchmark); see RELEASE_CHECKLIST.md")
def test_a20_arm_pairs_length_and_precision(tmp_path: Path) -> None:
    gold = report.gold_by_task()
    tasks = sorted(gold)[:6]
    assert arms.A20 in arms.ALL_ARMS
    for arm, shift in zip((*arms.ARMS, arms.A20), (0.0, 0.2, 0.1, 0.15), strict=True):
        rows = []
        for i, t in enumerate(tasks):
            g = sorted(gold[t])
            paths = g[: 1 + i % 2] + [f"x{j}.py" for j in range(2 + i)]
            rows.append({
                "task_id": t, "repo": f"r{i % 3}", "lang": "Go", "arm": arm, "brief_sha": "x",
                "wall_s": 20.0 + i, "handback_chars": 10, "cost_usd": 0.05, "resolved_subagent_model": "m",
                "zero_tools": False, "is_error": False, "fallthrough": False, "all_gold": False,
                "recall12": min(1.0, 0.3 + shift + 0.05 * i), "final_recall12": 0.3, "explore_s": 5.0,
                "parent_cost_usd": 0.03, "subagent_cost_usd": 0.02, "agent_type": "Explore",
                "paths": paths, "n_gold": len(g),
            })  # fmt: skip
        (tmp_path / f"rows_{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = report.summarize(report.load(tmp_path), report.load.bad)  # type: ignore[attr-defined]
    assert "super_grpo_a20_minus_super_v3" in res["deltas"]
    assert "super_grpo_a20_minus_explore_default" in res["deltas"]
    ln = res["arms"]["super_grpo_a20"]["length"]
    assert ln["n_paths"]["n"] == 6
    assert ln["n_paths"]["mean"] > 2
    assert 0.0 < ln["precision12"]["mean"] < 1.0
    assert "precision@5" in report.markdown(res)
    row = {"paths": ["a", "b", "c"], "n_gold": 2}
    ls = report.length_stats(row, {"a", "z"})
    assert ls["n_paths"] == 3
    assert abs(ls["precision5"] - 1 / 3) < 1e-9
    assert report.length_stats({"paths": []}, {"a"})["precision12"] == 0.0
