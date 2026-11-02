"""CPU tests for the optional G4 rollout worker (docs/grpo_g4_worker.md): flag-off identity, Modal + G4 merge with groups intact, late-drop, identical sampling parameters.
No Modal, no network, no GPU."""

import asyncio
import json
import re
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voyage-scout" / "src"))

from _g4_harness import FakeCtl, FakeJobs, FakeResults, group_rows, make_row, pool_tasks, run_driver  # noqa: E402

from wayfinder.grpo import g4_link  # noqa: E402
from wayfinder.grpo.driver import Driver, build_batch  # noqa: E402
from wayfinder.grpo.pool import sample_batch  # noqa: E402

GOLDEN = Path(__file__).parent / "data" / "grpo_driver_off_golden.json"
ROOT = Path(__file__).resolve().parents[1]
SB = 100000  # seeds of step 0 as the driver draws them: 100000 * (1 + step) + 17 * seed_salt + s


# ------------------------------------------------------------------------------------------------------------------------------ flag off
def test_flag_off_driver_identical_to_golden(tmp_path, monkeypatch):
    """The golden was produced by Driver.run BEFORE the G4 code existed (same fabricated rows): with no G4 link the steps.jsonl line, the manifest published to the trainer,
    the job puts, the task list and state.json are exactly the same, and no g4 key or file appears."""
    res = run_driver(tmp_path, monkeypatch)
    assert json.loads(json.dumps(res, sort_keys=True)) == json.loads(GOLDEN.read_text())
    assert "g4" not in res["line"] and "g4_tasks_used" not in res["line"]
    assert not any("g4" in f for f in res["files"])


def test_flag_off_app_has_no_g4_state():
    src = (ROOT / "wayfinder/grpo/app.py").read_text()
    assert "g4_worker: bool = False" in src  # default off
    assert "if g4_worker:" in src and "a.g4_worker" in src  # namespace / config keys exist only when on


# ------------------------------------------------------------------------------------------------------------------------------ merge
def g4_driver(tmp_path, monkeypatch, g4_rows_fn, *, g4_tasks=2, group=4, grace=1.0, alive=True, late_rows=None):
    class Link:
        def __init__(self):
            self.jobs, self.cut, self.hb = FakeJobs(), None, {"t": 1e18} if alive else None
            self.results = FakeResults([])
            self.ctl = self

        def get(self, k):
            return self.hb

        def alive(self, max_age_s=300):
            return alive, self.hb

        def submit(self, job):
            self.jobs.put(job)

        def mark_cut(self, step):
            self.cut = step

    link = Link()
    a_extra = {"g4_worker": True, "g4_tasks": g4_tasks, "g4_grace_s": grace, "g4_cfg": {"max_seconds": 1200}}
    pool = pool_tasks()
    modal_tasks = sample_batch(pool, 0, 2, 0)
    g4_tasks_sel = g4_link.sample_g4_tasks(pool, 0, g4_tasks, 0, {t["task_id"] for t in modal_tasks})
    link.results.rows = g4_rows_fn(g4_tasks_sel)
    out = run_driver(tmp_path, monkeypatch, g4=link, extra_args=a_extra, group=group)
    return out, link, g4_tasks_sel, modal_tasks


def test_merge_modal_and_g4_rows_into_one_batch_groups_intact(tmp_path, monkeypatch):
    out, link, g4t, mt = g4_driver(tmp_path, monkeypatch, lambda ts: [r for t in ts for r in group_rows(t["task_id"], 0, 0, 4, "g4", SB)])
    batch = next(v for k, v in out["ctl"] if k == "batch_0")
    tids = {}
    for it in batch["items"]:
        tids.setdefault(it["task_id"] if "task_id" in it else it["rid"].rsplit("-", 2)[0], []).append(it["rid"])
    assert set(tids) == {t["task_id"] for t in mt} | {t["task_id"] for t in g4t}
    assert all(len(v) == 4 for v in tids.values())  # every group has all its seeds, none split across lanes
    assert out["line"]["g4_tasks_used"] == 2 and out["line"]["g4_dropped"] == 0
    assert out["line"]["rollout"]["n_rollouts"] == 16 and out["line"]["g4"]["rollouts_used"] == 8
    assert len(link.jobs.puts) == 2 and all(j["version"] == 0 and j["step"] == 0 for j in link.jobs.puts)
    assert link.cut == 0
    modal_ids = {t["task_id"] for t in mt}
    assert not modal_ids & {t["task_id"] for t in g4t}  # the G4 lane never duplicates a Modal task of the step
    # per-task advantages: the G4 groups get the same group baseline as Modal groups
    man, _, adv = build_batch([dict(r) for t in g4t for r in group_rows(t["task_id"], 0, 0, 4)], 0.0, lam=0.0)
    assert adv["groups"] == 2


def test_merge_g4_unit_incomplete_stale_and_duplicates():
    sub = {"A": [1, 2], "B": [1, 2], "C": [1, 2], "D": [1, 2]}
    mk = lambda t, s, v=0, **kw: {"task_id": t, "seed": s, "version": v, "elapsed_s": 5.0, "rid": f"{t}{s}", **kw}  # noqa: E731
    rows = [mk("A", 1), mk("A", 2), mk("A", 2),  # complete (+ one duplicate)
            mk("B", 1),  # incomplete
            mk("C", 1), mk("C", 2, v=3),  # one stale row
            mk("D", 1, infra_error=True), mk("D", 2, infra_error=True),  # all infra
            mk("Z", 1)]  # unexpected task
    used, info = g4_link.merge_g4(rows, sub, 0)
    assert [r["task_id"] for r in used] == ["A", "A"] and all(r["source"] == "g4" for r in used)
    assert info["g4_tasks_used"] == 1 and info["g4_dropped"] == 3
    assert info["drop_reasons"] == {"incomplete_or_late": 1, "stale_version": 1, "infra_only": 1}
    assert info["rows_duplicate"] == 1 and info["rows_unexpected"] == 1


def test_infra_rows_of_a_partly_failed_group_are_kept_for_the_driver_to_drop():
    used, info = g4_link.merge_g4([{"task_id": "A", "seed": 1, "version": 0, "infra_error": True}, {"task_id": "A", "seed": 2, "version": 0}], {"A": [1, 2]}, 0)
    assert len(used) == 2 and info["rollouts_used"] == 1 and info["rollouts_infra_error"] == 1  # same treatment as Modal infra rows (build_batch skips them)


# ------------------------------------------------------------------------------------------------------------------------------ late / missing
def test_g4_never_answers_modal_only(tmp_path, monkeypatch):
    out, link, g4t, mt = g4_driver(tmp_path, monkeypatch, lambda ts: [], grace=0.2)
    batch = next(v for k, v in out["ctl"] if k == "batch_0")
    assert len(batch["items"]) == 8  # only the 2 Modal groups
    assert out["line"]["g4_tasks_used"] == 0 and out["line"]["g4_dropped"] == 2
    assert out["line"]["rollout"]["n_rollouts"] == 8


def test_partial_g4_group_is_dropped_whole(tmp_path, monkeypatch):
    out, _, g4t, _ = g4_driver(tmp_path, monkeypatch, lambda ts: group_rows(ts[0]["task_id"], 0, 0, 4, "g4", SB) + group_rows(ts[1]["task_id"], 0, 0, 4, "g4", SB)[:3], grace=0.3)
    batch = next(v for k, v in out["ctl"] if k == "batch_0")
    assert len(batch["items"]) == 12 and out["line"]["g4_tasks_used"] == 1 and out["line"]["g4_dropped"] == 1
    assert not any(it["rid"].startswith(g4t[1]["task_id"]) for it in batch["items"])


def test_stale_version_rows_are_dropped_not_used_off_policy(tmp_path, monkeypatch):
    out, _, g4t, _ = g4_driver(tmp_path, monkeypatch, lambda ts: [r for t in ts for r in group_rows(t["task_id"], 0, 7, 4, "g4", SB)], grace=0.3)  # version 7 != the step's v0
    assert out["line"]["g4_tasks_used"] == 0 and out["line"]["g4"]["drop_reasons"] == {"stale_version": 2}
    assert len(next(v for k, v in out["ctl"] if k == "batch_0")["items"]) == 8


def test_dead_pod_submits_nothing_and_logs_skip(tmp_path, monkeypatch):
    out, link, _, _ = g4_driver(tmp_path, monkeypatch, lambda ts: [], alive=False)
    assert link.jobs.puts == [] and out["line"]["g4"]["skipped"] == "no G4 heartbeat" and out["line"]["g4_tasks_used"] == 0


def test_late_rows_after_the_cut_are_dropped_and_counted(tmp_path):
    import types

    a = types.SimpleNamespace(run="t", start_step=0, g4_grace_s=0, step_timeout_s=60)
    d = Driver(a, FakeCtl(), FakeJobs(), FakeResults([]), tmp_path)
    d.g4 = types.SimpleNamespace(results=FakeResults([make_row("A", 1, 0, 0, ["g1"], worker="g4"), make_row("A", 2, 1, 0, ["g1"], worker="g4")]))
    d.g4_sub = {0: {"A": [1]}, 1: {"A": [2]}}
    d.g4_cut = {0}  # step 0 closed: its row is late
    import threading

    th = threading.Thread(target=d.drain_g4, daemon=True)
    th.start()
    import time

    for _ in range(100):
        if d.g4_late and d.g4_rows.get(1):
            break
        time.sleep(0.02)
    d.stop.set()
    th.join(5)
    assert d.g4_late == 1 and 0 not in d.g4_rows and len(d.g4_rows[1]) == 1
    assert len((tmp_path / "g4_late.jsonl").read_text().splitlines()) == 1


def test_sample_g4_tasks_deterministic_disjoint_and_same_pool():
    pool = pool_tasks(40)
    for step in range(6):
        mt = sample_batch(pool, step, 16, 0)
        ex = {t["task_id"] for t in mt}
        g = g4_link.sample_g4_tasks(pool, step, 8, 0, ex)
        assert len(g) == 8 and len({t["task_id"] for t in g}) == 8 and not ({t["task_id"] for t in g} & ex)
        assert g == g4_link.sample_g4_tasks(pool, step, 8, 0, ex)
        assert all(t in pool for t in g)
    assert g4_link.sample_g4_tasks(pool_tasks(5), 0, 8, 0, {"t000", "t001"}) and len(g4_link.sample_g4_tasks(pool_tasks(5), 0, 8, 0, {"t000", "t001"})) == 3


# ------------------------------------------------------------------------------------------------------------------------------ sampling parameters
def _vllm_body(chat_kwargs_fn):
    """The body the real rollout code sends to vLLM for one chat turn, captured with a mock transport."""
    from scout.core.ports.chat import ChatRequest

    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    chat = chat_kwargs_fn(httpx.MockTransport(handler))
    asyncio.run(chat.complete(ChatRequest(messages=[{"role": "user", "content": "q"}])))
    return bodies[0]


def _grpo_chat(cfg, model):
    from wayfinder.rollout.chat import CompatChat, Prices, Sampling

    def mk(transport):  # the construction in wayfinder/grpo/rollout.py::grpo_rollout (checked by test_rollout_construction_is_shared)
        return CompatChat("http://127.0.0.1:8000/v1", "x", model, sampling=Sampling(cfg.get("temperature", 0.85), cfg.get("top_p", 0.95)), prices=Prices(1.25, 4.25, 0.15), seed=3,
                          vllm_extensions=False, timeout=1800.0, max_retries=4, extra_body={"return_token_ids": True, "logprobs": True}, transport=transport)

    return mk


def test_g4_request_params_identical_to_modal():
    """Both lanes call the same grpo_rollout with the job's cfg; the request body differs only in the model name (the adapter's served name is v<j> on both)."""
    cfg = {"max_seconds": 1200}  # cfg as the driver sends it (temperature/top_p default to rollout.TEMPERATURE / TOP_P)
    modal_body = _vllm_body(_grpo_chat(cfg, "v3"))
    g4_body = _vllm_body(_grpo_chat(json.loads(json.dumps(cfg)), "v3"))  # the job went through the queue (pickled/JSON round trip)
    assert modal_body == g4_body
    assert modal_body["temperature"] == 0.85 and modal_body["top_p"] == 0.95 and "top_k" not in modal_body  # top-k 64 comes from the checkpoint's generation_config
    assert modal_body["logprobs"] is True and modal_body["return_token_ids"] is True


def test_rollout_construction_is_shared():
    from wayfinder.grpo import rollout
    from wayfinder.rollout import g4_grpo_worker

    assert g4_grpo_worker.TEMPERATURE == rollout.TEMPERATURE == 0.85 and g4_grpo_worker.TOP_P == rollout.TOP_P == 0.95
    src = (ROOT / "wayfinder/rollout/g4_grpo_worker.py").read_text()
    assert "grpo_rollout(t, \"http://127.0.0.1:8000/v1\", model, root, seed=seed, cfg=cfg)" in src  # same call as app.py's worker
    assert "grpo_rollout(t, \"http://127.0.0.1:8000/v1\", model, root, seed=seed, cfg=cfg)" in (ROOT / "wayfinder/grpo/app.py").read_text()
    rsrc = (ROOT / "wayfinder/grpo/rollout.py").read_text()
    assert 'extra_body={"return_token_ids": True, "logprobs": True}' in rsrc and "vllm_extensions=False" in rsrc


def _flags(text, name):
    m = re.search(rf"^{name} = \[(.*?)\]", text, re.S | re.M)
    return re.findall(r'"([^"]+)"', m.group(1))


def test_vllm_flags_equal_modal_workers_except_memory_fraction():
    modal = [("policy" if x == "BASE" else x) for x in re.findall(r'"([^"]+)"|\b(BASE)\b', (ROOT / "wayfinder/grpo/app.py").read_text().split("FLAGS = [", 1)[1].split("]", 1)[0]) for x in x if x]
    from wayfinder.rollout.g4_grpo_worker import VLLM_FLAGS

    assert "--logprobs-mode" in modal and VLLM_FLAGS[VLLM_FLAGS.index("--logprobs-mode") + 1] == "processed_logprobs"
    diff = {i for i, (x, y) in enumerate(zip(modal, VLLM_FLAGS)) if x != y}
    assert len(modal) == len(VLLM_FLAGS) and {modal[i] for i in diff} == {"0.90"} and {VLLM_FLAGS[i] for i in diff} == {"0.92"}


@pytest.mark.skipif(not (ROOT / "deploy" / "g4" / "grpo_worker.yaml").exists(), reason="needs task manifests or cluster specs withheld from the public release; see RELEASE_CHECKLIST.md")
def test_pod_spec_serves_the_same_flags():
    import yaml

    from wayfinder.rollout.g4_grpo_worker import VLLM_FLAGS

    docs = list(yaml.safe_load_all((ROOT / "deploy/g4/grpo_worker.yaml").read_text()))
    dep = next(d for d in docs if d and d.get("kind") == "Deployment")
    assert dep["metadata"]["namespace"] == "wayfinder-g4-grpo"  # never wayfinder-screen (the eval pod's namespace)
    vllm = next(c for c in dep["spec"]["template"]["spec"]["containers"] if c["name"] == "vllm")
    cmd = vllm["command"]
    assert cmd[:2] == ["vllm", "serve"] and cmd[3:] == VLLM_FLAGS
    sec = [e for c in dep["spec"]["template"]["spec"]["containers"] for e in c.get("env", []) if "secret" in json.dumps(e).lower()]
    assert not any("value" in e and re.search(r"(?i)token|secret|key", e["name"]) and e.get("value") for e in dep["spec"]["template"]["spec"]["containers"][1].get("env", []))
    assert sec is not None


def test_generation_config_top_k_guard(tmp_path):
    from wayfinder.rollout.g4_grpo_worker import check_generation_config

    (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": 64, "top_p": 0.95}))
    assert check_generation_config(str(tmp_path)) == 64
    (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": 40}))
    with pytest.raises(SystemExit):
        check_generation_config(str(tmp_path))
    from wayfinder.grpo.loss import TOP_K, TOP_P

    assert TOP_K == 64 and TOP_P == 0.95


# ------------------------------------------------------------------------------------------------------------------------------ worker logic
def test_job_usable_rules():
    from wayfinder.rollout.g4_grpo_worker import job_usable

    j = g4_link.make_job(5, 4, {"task_id": "t"}, [1], {"max_seconds": 1}, expires=100.0)
    assert job_usable(j, None, 50) is None and job_usable(j, 4, 50) is None
    assert job_usable(j, 5, 50) == "step already closed" and job_usable(j, None, 101) == "expired"
    assert job_usable(g4_link.make_job(5, 4, {}, [1], {"fork_report": 4}, 100.0), None, 50)


def test_check_env_reports_names_only():
    from wayfinder.rollout.g4_grpo_worker import check_env

    assert check_env({"MODAL_TOKEN_ID": "x", "MODAL_TOKEN_SECRET": "y"}) == ["TURBOPUFFER_API_KEY", "VOYAGE_API_KEY", "WF_INDEX_MASTER_KEY_HEX"]
