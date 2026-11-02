"""Contamination probe for the v4 secondary set (docs/e2e_final_plan.md, Addendum 1.2). Copy of contam_probe.py's probe_one, reading issue text
and gold non-test paths from the manifest + env records, ledger eval/e2e/gateway_spend_final.jsonl. Skips tasks that already carry `probe`.
  python3 eval/e2e/final_probe.py <manifest.json> <out.json> [workers]   (run via final_env.py for LLM_GATEWAY_API_KEY)"""
import json, sys, os, fcntl, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contam_probe as cp
from asa.gateway_backend import GatewayBackend, ANTHROPIC_PATH
import re
LEDGER = Path(os.environ["E2E_FINAL_LEDGER"])
man, out_p, workers = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 12
tasks = json.load(open(man))["tasks"]
res = json.load(open(out_p)) if out_p.exists() else {}
lock = threading.Lock()

def env_of(tid):
    for d in ("e2e_clean", "e2e_v4", "e2e_v3"):
        p = HERE / "experiments" / d / "envs" / f"{tid}.json"
        if p.exists(): return json.load(open(p))
    raise FileNotFoundError(tid)

def one(t):
    tid = t["task_id"]
    if tid in res or t.get("probe"): return
    env = env_of(tid)
    gold = [p for st, p in env["nontest_entries"] if st != "D"]
    b = GatewayBackend(model_id="opus-5.5-medium")
    data = b._post(ANTHROPIC_PATH, {"model": b.model, "max_tokens": 800, "messages": [{"role": "user", "content": cp.PROMPT.format(repo=t["repo"], issue=t["issue_text"])}]})
    text = "".join(x.get("text", "") for x in data.get("content", []) if x.get("type") == "text")
    m = re.search(r"\[.*\]", text, re.S)
    try: pred = [str(x) for x in json.loads(m.group(0))] if m else []
    except ValueError: pred = re.findall(r"[\w./\-]+\.\w+", text)
    u = data.get("usage") or {}; c = cp.usd(u)
    row = dict(at=time.strftime("%FT%TZ", time.gmtime()), task_id=tid, usd=c, purpose="contam_probe", in_tokens=u.get("input_tokens"), out_tokens=u.get("output_tokens"))
    r = dict(task_id=tid, repo=t["repo"], predicted=pred, model="claude-opus-5-5 medium", usd=c, n_modified_existing=sum(1 for st, p in env["nontest_entries"] if st in "MRCT"), **cp.score(pred, gold))
    with lock:
        with open(LEDGER, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(row) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
        res[tid] = r; out_p.write_text(json.dumps(res, indent=1))
    print(t["slug"], "FLAGGED" if r["flagged"] else "pass", r["n_hit"], "/", r["n_gold"], flush=True)

with ThreadPoolExecutor(workers) as ex: list(ex.map(one, tasks))
print("done", len(res), "flagged", sum(r["flagged"] for r in res.values()))
