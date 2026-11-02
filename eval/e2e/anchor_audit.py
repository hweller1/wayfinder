"""Round 6: GREP-ANCHOR detector for issue texts, run against the round-5 monorepo workspace (316 repos, ~75M lines).

DEFINITION (plan.md, ROUND 6; fixed before data). A GREP ANCHOR is a token/phrase of the issue that EXISTS in the base-commit task
repo and whose workspace-wide `rg -l -F` returns <= 20 files including >= 1 gold SOURCE file. NEW_API names (absent at base; the
held-out tests call them) are never anchors. ANCHOR-FREE issue = no anchors, verified by running rg on the workspace.

Candidate tokens (extract_tokens, pure): backticked spans (+ their identifier parts), quoted literals, identifiers (snake, camel,
Pascal, SCREAMING, dotted), distinctive lowercase words (>= 5 letters, not a function word), crude stems of those words
(`redaction` -> `redact`: the first thing Claude Code grepped in calibration 2), and adjacent-word joins (`basic decoder` ->
`basic_decoder`, `BasicDecoder`). Words, stems and joins are matched case-insensitively (-i), identifiers/quoted/backtick
case-sensitively. All matching is `rg -l -F` (fixed string, default ignore rules like the agent's Grep tool).

Engine: one Modal CPU sandbox (16 cpu / 32 GiB so the 12 GB tree stays in page cache) holds the workspace (316 tars from volume
asa-e2e-r5-ws, extracted once) plus each task's repo at its BASE commit (held-out test files removed) under /tasks/<key>/. A query
for task X searches /work (minus X's own repo directory, whose generic-commit copy is replaced) + /tasks/<key>. A token is first
looked up in the task's gold source files only (cheap; an anchor must hit one), and only tokens that hit are searched workspace-wide
(stopping after 41 files: anything > 20 is not an anchor). Tokens that miss every gold file are reported as 0 files / no gold hit. Cost: Modal CPU only, $0 gateway.

  uv run --env-file .env --with modal --with dnspython python scripts/e2e/anchor_audit.py [--issues experiments/e2e/issues.json]
      -> experiments/e2e/anchor_audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE   # wayfinder/eval/e2e (copied from benchmark-repo scripts/e2e; paths re-pointed, nothing imported across repos)
sys.path.insert(0, str(HERE))
import os  # noqa: E402

EXP_DIR = Path(os.environ["E2E_EXP_DIR"]) if os.environ.get("E2E_EXP_DIR") else HERE / "experiments" / "e2e"


def ws_dirname(scope: str) -> str:
    """Directory of a repo inside the workspace: the scope without its commit suffix."""
    return scope.rsplit("__", 1)[0]


MAX_FILES = 20            # anchor: <= 20 files in the workspace ...
CAP = MAX_FILES + 21      # ... a search stops after this many files
WS_VOL, WS_MOUNT = "asa-e2e-r5-ws", "/r5ws"   # 316-repo workspace volume, read-only
WS_SCOPES_FILE = HERE / "experiments" / "e2e_clean" / "ws_scopes.json"
AUDIT_OUT = EXP_DIR / "anchor_audit.json"

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STOP = set("""
about above across after again against all also although always among and another any are around because been before being below
between both but can cannot could does doing done down during each either else even ever every expected few for from further had has
have having here how however into its itself just keep like made make many may might more most much must never new next none not now
off once only onto other otherwise our out over own per please rather same should since some such than that the their them then there
these they this those through thus too under until upon use used uses using very via was were what when where whether which while who
whom whose will with within without would yet you your able allow allows already instead currently support supports add adds added
should return returns returned value values call calls called must existing exist exists need needs needed want wants users user
callers caller consumers consumer example examples including include includes whose same behaviour behavior currently being
""".split())
STEM_SUFFIXES = ("ations", "ation", "ions", "ion", "ings", "ing", "ers", "er", "ed", "es", "s", "d", "e")


# --------------------------------------------------------------------------- pure logic (unit-tested)

def _stem(w: str) -> str | None:
    for suf in STEM_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 5:
            return w[: -len(suf)]
    return None


def _ident_shaped(t: str) -> bool:
    s = t.strip("_")
    return bool("_" in s or re.search(r"[a-z][A-Z]", s) or re.match(r"^[A-Z][a-z0-9]+[A-Z]", s) or (re.match(r"^[A-Z0-9_]{3,}$", s)))


def extract_tokens(text: str, new_api: list[str] | None = None) -> list[dict]:
    """Candidate grep tokens of an issue text. Each: {token, kind, ci}. NEW_API names (exact) are dropped here; whether a token
    exists at the base commit is decided by the search (an anchor must hit a gold source file, so absent names cannot anchor)."""
    new_api = set(new_api or [])
    out: dict[tuple[str, bool], dict] = {}

    def add(tok: str, kind: str, ci: bool):
        tok = tok.strip()
        if len(tok) < 3 or tok in new_api:
            return
        out.setdefault((tok, ci), dict(token=tok, kind=kind, ci=ci))

    for m in re.finditer(r"`([^`]+)`", text):
        span = m.group(1)
        add(span, "backtick", False)
        for part in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", span):
            add(part, "backtick", False)
            for sub in TOKEN.findall(part):
                if len(sub) >= 4:
                    add(sub, "backtick", False)
    for m in re.finditer(r"[\"“]([^\"”\n]{4,80})[\"”]", text):
        add(m.group(1), "quoted", False)
    for m in re.finditer(r"(?<![A-Za-z0-9])'([^'\n]{4,80})'(?![A-Za-z0-9])", text):
        add(m.group(1), "quoted", False)
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", text):
        add(m.group(0), "dotted", False)
    for m in TOKEN.finditer(text):
        if _ident_shaped(m.group(0)):
            add(m.group(0), "ident", False)
    # words (lowercase text; capitalised sentence starts are lowered), stems, adjacent joins
    prose = re.sub(r"`[^`]*`", " ", text)
    words = [w for w in re.findall(r"[A-Za-z]+", prose)]
    low = [w.lower() for w in words]
    for w in low:
        if len(w) >= 5 and w not in STOP:
            add(w, "word", True)
            s = _stem(w)
            if s:
                add(s, "stem", True)
    for a, b in zip(low, low[1:]):
        if len(a) >= 4 and len(b) >= 4 and a not in STOP and b not in STOP:
            add(f"{a}_{b}", "join", True)
            add(a.capitalize() + b.capitalize(), "join", True)
    return list(out.values())


def is_anchor(n_files: int, gold_hit: bool) -> bool:
    """The fixed definition: <= 20 workspace files AND >= 1 of them a gold source file."""
    return 0 < n_files <= MAX_FILES and bool(gold_hit)


def is_repo_anchor(n_files: int, task_hit: bool) -> bool:
    """BROADER secondary flag (superset of is_anchor): <= 20 workspace files AND >= 1 of them anywhere in the task repo (a search
    that lands in the right repository even when it misses the gold file). The rewrite guard rejects on this broader flag."""
    return 0 < n_files <= MAX_FILES and bool(task_hit)


def anchors_from(tokens: list[dict], results: dict, broad: bool = False) -> list[dict]:
    """tokens (with ci) + results keyed `<ci>|<token>` -> (n_files, gold_hit[, task_hit]). Returns the anchoring tokens with counts.
    broad=False: the fixed definition (gold source file hit). broad=True: the repo-level superset (any task-repo file hit)."""
    out = []
    for t in tokens:
        r = results.get(f"{int(t['ci'])}|{t['token']}")
        if not r:
            continue
        if (is_repo_anchor(r[0], r[2] if len(r) > 2 else r[1]) if broad else is_anchor(r[0], r[1])):
            out.append(dict(token=t["token"], kind=t["kind"], n_files=r[0], gold_hit=bool(r[1])))
    return sorted(out, key=lambda x: (x["n_files"], x["token"]))


# --------------------------------------------------------------------------- sandbox side

RUNNER = r'''
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
req = json.load(sys.stdin)
CAP = req["cap"]
def one(q):
    qid, tok, ci, own, tkey, gold, full = q
    if not full:
        # cheap pre-check: an anchor (strict or repo-level) must hit the task repo, so a token absent from it cannot anchor
        pre = subprocess.run(["rg", "-l", "-F", "--no-messages"] + (["-i"] if ci else []) + ["-e", tok, "/tasks/%s" % tkey],
                             capture_output=True, text=True)
        if not pre.stdout.strip():
            return qid, 0, False, False
    cmd = ["rg", "-l", "-F", "--no-messages", "--threads", "4"] + (["-i"] if ci else []) + ["-g", "!" + own, "-e", tok, "/work", "/tasks/" + tkey]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    n, g, th, pre = 0, False, False, "/tasks/%s/" % tkey
    gs = set(gold)
    for line in p.stdout:
        n += 1
        line = line.rstrip("\n")
        if line.startswith(pre):
            th = True
            if line[len(pre):] in gs:
                g = True
        if n >= CAP:
            break
    p.kill(); p.wait()
    return qid, n, g, th
with ThreadPoolExecutor(req.get("par", 8)) as ex:
    res = list(ex.map(one, req["queries"]))
print(json.dumps({q: [n, g, th] for q, n, g, th in res}))
'''


class AnchorChecker:
    """Holds one Modal sandbox with the 316-repo workspace; `check` runs the anchor audit of one issue text."""

    def __init__(self, cpu: float = 16.0, memory_mb: int = 32768, timeout_s: int = 6 * 3600):
        import dns_fallback  # noqa: F401
        import modal
        self.modal = modal
        self.app = modal.App.lookup(os.environ.get("E2E_ANCHOR_APP", "wayfinder-e2e-gate-anchor"), create_if_missing=True)
        img = modal.Image.debian_slim().apt_install("ripgrep", "tar")
        self.sb = modal.Sandbox.create(app=self.app, image=img, workdir="/work", cpu=cpu, memory=memory_mb, timeout=timeout_s,
                                       volumes={WS_MOUNT: modal.Volume.from_name(WS_VOL).read_only()})
        self.prepared: dict[str, dict] = {}
        self.build: dict = {}

    def close(self):
        try:
            self.sb.terminate()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        self.build_workspace()
        return self

    def __exit__(self, *a):
        self.close()

    def _sh(self, cmd: str, timeout: int = 900):
        p = self.sb.exec("bash", "-c", cmd, timeout=timeout)
        out, err = p.stdout.read(), p.stderr.read()
        return p.wait(), out, err

    def build_workspace(self):
        t0 = time.time()
        scopes = json.loads(WS_SCOPES_FILE.read_text())
        lines = "\n".join(f"{sc} {ws_dirname(sc)}" for sc in scopes) + "\n"
        self.sb.exec("mkdir", "-p", "/work", "/tasks").wait()
        rc_, out, err = self._sh(
            f"cat > /tmp/ws_list.txt <<'EOF'\n{lines}EOF\n"
            f"cd /work && xargs -P 8 -L 1 sh -c 'tar -xf {WS_MOUNT}/ws/$0.tar -C /work --transform \"s,^$0,$1,\" 2>/dev/null; true' < /tmp/ws_list.txt; "
            "ls -d /work/*/ | wc -l; du -sk /work | cut -f1", timeout=1800)
        nums = out.split()
        self.build = dict(build_s=round(time.time() - t0, 1), n_dirs=int(nums[0]) if nums else None, kb=int(nums[1]) if len(nums) > 1 else None)
        print("[anchor] workspace", self.build, flush=True)

    def prepare(self, spec: dict) -> str:
        """Export the task repo at its base commit (no .git, held-out test files removed) from the LOCAL clone, upload it over stdin
        (chunked) and extract to /tasks/<key>/. spec from task_spec()."""
        key = spec["key"]
        if key in self.prepared:
            return key
        import subprocess
        from modal_sandbox import write_chunked
        arch = subprocess.run(["git", "-C", spec["repo_dir"], "archive", "--format=tar.gz", spec["base_commit"]], capture_output=True, check=True).stdout
        self.sb.exec("mkdir", "-p", f"/tasks/{key}").wait()
        up = self.sb.exec("tar", "xzf", "-", "-C", f"/tasks/{key}", text=False)
        write_chunked(up.stdin, arch)
        up.wait()
        rm = " ".join(shlex.quote(f"/tasks/{key}/{h}") for h in spec["heldout"])
        rc_, out, err = self._sh(f"rm -f {rm}; find /tasks/{key} -type f | wc -l", timeout=600)
        if rc_ != 0:
            raise RuntimeError(f"task tree extract failed for {key}: {err[-300:]}")
        self.prepared[key] = dict(files=int(out.split()[-1]))
        return key

    def search(self, spec: dict, tokens: list[dict], par: int = 8) -> dict[str, tuple[int, bool, bool]]:
        key = self.prepare(spec)
        qs = [[f"{int(t['ci'])}|{t['token']}", t["token"], bool(t["ci"]), spec["owndir"], key, spec["gold"], t.get("kind") == "cal2_probe"]
              for t in tokens]
        payload = json.dumps(dict(queries=qs, cap=CAP, par=par))
        self._sh("cat > /tmp/runner.py <<'EOF'\n" + RUNNER + "\nEOF")
        rc_, out, err = self._sh("cat > /tmp/q.json <<'QEOF'\n" + payload + "\nQEOF\npython3 /tmp/runner.py < /tmp/q.json", timeout=3600)
        if rc_ != 0:
            raise RuntimeError("rg runner failed: " + err[-400:])
        return {k: (int(v[0]), bool(v[1]), bool(v[2])) for k, v in json.loads(out.strip().splitlines()[-1]).items()}

    def check(self, spec: dict, issue_text: str, new_api: list[str], extra: list[dict] | None = None) -> dict:
        toks = extract_tokens(issue_text, new_api) + (extra or [])
        res = self.search(spec, toks)
        strict, broad = anchors_from(toks, res), anchors_from(toks, res, broad=True)
        return dict(anchored=bool(strict), anchors=strict, repo_anchored=bool(broad), repo_anchors=broad, n_tokens=len(toks), counts=res)


def task_spec(task_id: str) -> dict:
    """Everything the checker needs for one gated task: local clone + base commit, workspace dir, gold SOURCE files that exist at base, held-out files."""
    import run_coder as rc   # lazy (heavy imports); gold_source_files is the harness's own definition
    t = rc.load_all_tasks()[task_id]
    env = rc.sc.load_env(task_id)
    heldout = [p for _, p in env["heldout_test_entries"]]
    gold, _new = rc.gold_source_files(list(t["gold_files"]), heldout, t.get("gold_new_at_base"))
    slug = t["repo_full"].replace("/", "__")
    return dict(task_id=task_id, repo_dir=str(HERE / "data" / "e2e" / "repos" / slug), base_commit=t["base_commit"],
                owndir=slug, key="t_" + task_id[-10:], gold=gold, heldout=heldout)


# --------------------------------------------------------------------------- audit driver

# What Claude Code grepped FIRST in the calibration-2 transcripts (experiments/e2e/harness/r5_cal2), as validation probes: added as
# extra tokens for those tasks so the report shows what the same engine says about them.
CAL2_PROBES = {
    "helm": ["already exists, overwriting"],
    "claude-agent-sdk-python": ["task_updated"],
    "gh-aw-mcpg": ["Redact", "Sanitize"],
    "mashumaro": ["standalone", "basic_decoder"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issues", default=str(EXP_DIR / "issues.json"))
    ap.add_argument("--out", default=str(AUDIT_OUT))
    ap.add_argument("--task", action="append")
    a = ap.parse_args()
    issues = [x for x in json.load(open(a.issues))["issues"] if x.get("ok")]
    if a.task:
        issues = [x for x in issues if any(s in x["task_id"] for s in a.task)]
    rows = []
    with AnchorChecker() as ck:
        for i, x in enumerate(issues, 1):
            spec = task_spec(x["task_id"])
            probes = []
            for k, v in CAL2_PROBES.items():
                if k in x["task_id"]:
                    probes = [dict(token=p, kind="cal2_probe", ci=False) for p in v]
            t0 = time.time()
            r = ck.check(spec, x["issue_text"], x.get("new_api") or [], extra=probes)
            counts = r.pop("counts")
            row = dict(task_id=x["task_id"], slug=x["task_id"].split("__")[-2], anchored=r["anchored"], n_tokens=r["n_tokens"],
                       anchors=r["anchors"], repo_anchored=r["repo_anchored"], repo_anchors=r["repo_anchors"], new_api=x.get("new_api"), gold=spec["gold"],
                       probes={p["token"]: dict(zip(("n_files", "gold_hit", "task_repo_hit"), counts.get(f"0|{p['token']}", (None, None, None)))) for p in probes},
                       token_table=sorted([dict(token=k.split("|", 1)[1], ci=k[0] == "1", n_files=v[0], gold_hit=v[1], task_repo_hit=v[2])
                                           for k, v in counts.items() if v[0] > 0], key=lambda d: (d["n_files"], d["token"])))
            rows.append(row)
            print(f"[{i}/{len(issues)}] {row['slug']:28} anchored={row['anchored']} anchors={[(t['token'], t['n_files']) for t in row['anchors'][:4]]} "
                  f"{time.time() - t0:.0f}s", flush=True)
            Path(a.out).write_text(json.dumps(dict(definition=f"<= {MAX_FILES} workspace files incl. >= 1 gold source file", rows=rows,
                                                   build=ck.build), indent=1))
    n_anch = sum(r["anchored"] for r in rows)
    n_rep = sum(r["repo_anchored"] for r in rows)
    summary = dict(n_issues=len(rows), n_anchored=n_anch, n_anchor_free=len(rows) - n_anch,
                   n_repo_anchored_broad=n_rep, n_anchor_free_broad=len(rows) - n_rep)
    Path(a.out).write_text(json.dumps(dict(definition=f"<= {MAX_FILES} workspace files incl. >= 1 gold source file", summary=summary,
                                           rows=rows, build=ck.build), indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
