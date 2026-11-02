# ruff: noqa: E501  (long fixture lines)
"""The Muse spec calibration harness (eval/harness/muse_spec.py): cost arithmetic, usage extraction from a Muse
session log, row validation, scoring and the spec-only plugin bundle. No Muse, no network."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # eval/ is a top-level directory, not a package

from eval.harness import muse_spec as ms  # noqa: E402


def session_line(kind: str, **fields: object) -> str:
    return json.dumps({"recorded_at": 1_000_000_000_000_000, "payload": {"event": {"kind": kind, **fields}}})


def test_usage_cost_prices_cached_input_separately() -> None:
    assert ms.usage_cost(1_000_000, 0, 0) == pytest.approx(1.25)
    assert ms.usage_cost(1_000_000, 1_000_000, 0) == pytest.approx(0.15)
    assert ms.usage_cost(0, 0, 1_000_000) == pytest.approx(4.25)


def test_provider_usage_sums_reported_provider_records_only(tmp_path: Path) -> None:
    def rec(family: str, reported: bool, i: int, c: int, o: int) -> str:
        q = {"reported": reported, "input_tokens": i, "cached_tokens": c, "output_tokens": o}
        return session_line("goal_usage_attribution", record={"usage_family": family, "quantity": q})

    p = tmp_path / "s.jsonl"
    p.write_text(
        "\n".join(
            [
                rec("provider", True, 100, 40, 10),
                rec("provider", True, 50, 0, 5),
                rec("tool", True, 9, 9, 9),
                rec("provider", False, 7, 7, 7),
            ]
        )
    )
    assert ms.provider_usage(p) == {"in": 150, "cached": 40, "out": 15, "calls": 2}


def test_scout_explore_seconds_measures_commit_to_result(tmp_path: Path) -> None:
    def at(t: float, line: str) -> str:
        d = json.loads(line)
        d["recorded_at"] = int(t * 1e6)
        return json.dumps(d)

    calls = [{"name": "mcp__plugin_scout_wayfinder_scout.explore", "args": "{}"}]
    other = [{"name": "read_file", "args": "{}"}]
    p = tmp_path / "s.jsonl"
    p.write_text(
        "\n".join(
            [
                at(10, session_line("assistant_tool_calls_committed", tool_calls=other)),
                at(11, session_line("tool_result_batch_committed")),
                at(20, session_line("assistant_tool_calls_committed", tool_calls=calls)),
                at(95.5, session_line("tool_result_batch_committed")),
            ]
        )
    )
    assert ms.scout_explore_seconds(p) == [75.5]


def test_spawn_types_defaults_to_general_purpose() -> None:
    calls = [
        ("subagent_spawn", json.dumps({"subagent_type": ms.EXPLORER})),
        ("subagent_spawn", "{}"),
        ("search", "{}"),
    ]
    assert ms.spawn_types(calls) == [ms.EXPLORER, "general-purpose"]


def healthy(arm: str) -> dict:
    return {
        "rc": 0, "wall_s": 60.0, "parent_model_calls": 5, "parent_model": "m", "brief_sha": "x", "n_reported": 8,
        "arm": arm, "spec": {"session_injected": True, "prompt_injected": True}, "scout_explore_calls": 1,
    }  # fmt: skip


def test_validate_rejects_rows_that_do_not_carry_their_condition() -> None:
    assert ms.validate(healthy(ms.SCOUT_ARM)) is None
    assert ms.validate(healthy(ms.STOCK)) is None
    assert ms.validate({**healthy(ms.STOCK), "rc": 124}) == "muse exec rc=124"
    assert "implausible" in str(ms.validate({**healthy(ms.STOCK), "wall_s": 1.0}))
    assert "no model" in str(ms.validate({**healthy(ms.STOCK), "parent_model": None}))
    assert "no reported paths" in str(ms.validate({**healthy(ms.STOCK), "n_reported": 0}))
    assert "did not fire" in str(
        ms.validate({**healthy(ms.EXPL), "spec": {"session_injected": True, "prompt_injected": False}})
    )
    assert "never called" in str(ms.validate({**healthy(ms.SCOUT_ARM), "scout_explore_calls": 0}))


def test_score_uses_tracked_paths_in_reported_order(tmp_path: Path) -> None:
    import subprocess

    root = tmp_path / "r"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603, S607
    for f in ("a.py", "b.py", "c.py"):
        (root / f).write_text("x")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)  # noqa: S603, S607
    row = ms.score('["c.py", "ghost.py", "a.py"]', root, {"gold_files": ["a.py", "b.py"]})
    assert row["final_paths"] == ["c.py", "a.py"]  # a path that is not tracked does not count
    assert (row["recall5"], row["recall12"]) == (0.5, 0.5)
    assert row["precision12"] == 0.5


def test_spec_only_bundle_has_the_three_spec_hooks_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ms, "WORK", tmp_path)
    dst = ms.build_spec_only_bundle()
    caps = json.loads((dst / ".muse-plugin" / "plugin.json").read_text())["capabilities"]
    assert {h["id"] for h in caps["hooks"]} == {"spec-session", "spec-prompt", "spec-pretool"}
    assert caps["mcpServers"] == caps["skills"] == caps["commands"] == []
    assert not (dst / "agents").exists()
    for h in caps["hooks"]:
        assert (dst / h["command"][1]).is_file()
    assert ms.build_spec_only_bundle() == dst  # built once


@pytest.mark.skipif("not ms.SPLIT.exists()", reason="withheld from the public release (derived from a pre-event benchmark); see RELEASE_CHECKLIST.md")
def test_the_tune_split_is_26_tasks_and_confirm_is_not_touched() -> None:
    tasks = ms.tune_tasks()
    split = json.loads(ms.SPLIT.read_text())
    assert len(tasks) == 26
    assert not {t["native_id"] for t in tasks} & set(split["confirm"])
