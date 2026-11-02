"""E2E issue rewriter (Sonnet 5.5 via the the LLM gateway backend that scripts/rewrite_queries.py uses).

For each gated task: write a 2-4 sentence issue a maintainer would file BEFORE the change, naming the NEW public
API the held-out tests call (so the task is solvable) and nothing that already exists in the parent tree (so the
text does not do the retrieval for the agent).

  uv run --env-file .env python scripts/e2e/write_issues.py            # all gated tasks -> experiments/e2e/issues.json
  uv run --env-file .env python scripts/e2e/write_issues.py --task <id> --dry-run   # print NEW_API + prompt only

NEW_API = identifiers that (a) appear on added lines of the held-out test diff, (b) appear on added lines of the
non-test CODE diff (so they are defined by the change, not test-local names), and (c) occur nowhere in the parent
commit's tree (`git grep -w` at base_commit, tests and docs included).

Guards (reject + retry up to 3x, then the task is marked failed; a failed task is dropped, never loosened):
  gold_path        names a gold (non-test changed) file path, basename, or code-shaped stem
  existing_ident   names an identifier that exists at the parent. "Identifier" = backticked span part, code-shaped
                   token, PascalCase token, or capitalised mid-sentence word. Allowlist (never flagged): standard
                   library / builtin names and generic technical acronyms/proper nouns in ALLOW below, plus all
                   plain lowercase words outside backticks, and lowercase ordinary English words even in backticks (dictionary check). NEW_API items are exempt (absent at parent by construction).
  verbatim_diff    >= 8 consecutive word tokens that appear consecutively in the commit diff (one file's +/- lines)
  missing_new_api  omits any NEW_API item
  sentences        not 2-4 sentences
  diff_literal     (round 6 fix, --anchor-guard) the text contains a literal the gold (non-test source) diff INTRODUCES (see DIFF-INTRODUCED
                   LITERALS below). Pure function guard_diff_literal; NEW_API names are exempt.
  anchor           (round 6, --anchor-guard) the text contains a GREP ANCHOR: a token that exists at the base commit and whose
                   workspace-wide `rg -l -F` over the 316-repo monorepo returns <= 20 files including >= 1 gold source file
                   (scripts/e2e/anchor_audit.py; NEW_API names are exempt). The Sonnet call is ledgered (r5_spend.jsonl).

DIFF-INTRODUCED LITERALS (round-6 resume step 2; the strings agents grepped first in round 5 were new in the diff, so the base-tree anchor
definition missed them): from ADDED lines of the non-test source diff only,
  (a) string literals ("..", '..', `..` with >= 4 chars) plus the identifier/flag/key-shaped tokens and non-dictionary words (>= 5 letters,
      not in /usr/share/dict/words, not in anchor_audit.STOP) inside them;
  (b) non-dictionary word tokens (>= 5 letters) of comment text (#, //, /* */, --);
  (c) identifiers DEFINED on added lines (fn/def/func/class/struct/enum/trait/const/type/static/interface) that are not NEW_API.
An issue is rejected if (i) a quoted or backticked phrase in it equals or is a substring (case-insensitive, whitespace-normalised) of
any such string literal, or (ii) it contains an identifier-shaped token (snake/camel/Pascal/kebab/--flag/dotted) or a non-dictionary word
that occurs in those literals / comment words / new identifiers. Ordinary dictionary words never trigger; tokens < 5 chars and numbers are
ignored; NEW_API items are exempt (the tests call them).
the LLM gateway spend: every Sonnet call is appended to eval/e2e/gateway_spend.jsonl (flock) and refused before the call if cumulative + 0.05 > the hard
cap of $40 (E2E_GATEWAY_CAP may only lower it).

Round 6:  uv run --env-file .env --with modal --with dnspython python scripts/e2e/write_issues.py --anchor-guard \
              --out issues.json --tasks-file ids.json --workers 8   (results land under E2E_EXP_DIR)
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import gate as g  # noqa: E402

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CODE_EXT = {".py", ".go", ".rs", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cc", ".cpp", ".rb"}
MAX_ATTEMPTS = 3
NGRAM = 8

# Standard library / builtin names and generic terms that may be named freely (rule documented in the module doc).
ALLOW = {w.lower() for w in """
str int float bool bytes list dict set tuple none true false self cls object type any optional union callable
string error errors result option some ok err vec hashmap hashset box arc rc duration iterator trait struct enum
impl fn pub mod use crate map slice array nil interface func package context bytes reader writer
async await await await yield def class return import from lambda
http https url uri json yaml toml xml html css sql tcp udp tls ssl dns ip api cli sdk sse ssh ftp utf
python go rust golang cargo pip uv git github docker linux macos windows unix posix
get post put delete patch head options
true false null
""".split()}
ALLOW |= {"HTTP", "URL", "JSON", "YAML", "TOML", "API", "CLI", "SDK", "TLS", "SSL", "DNS", "TCP", "UTF", "HTML", "SQL", "XML", "SSE"}
ALLOW |= set("isize usize i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str bool frozenset ids id args kwargs byte rune uint int8 int16 int32 int64 uint8 uint16 uint32 uint64 float32 float64 error cwd dir path data filters command".split())
ALLOW = {a.lower() for a in ALLOW}
ACRONYM = re.compile(r"^[A-Z]{2,}$")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=300).stdout


def exists_at_base(repo: Path, base: str, tok: str, _cache: dict = {}) -> bool:
    key = (str(repo), base, tok)
    if key not in _cache:
        _cache[key] = bool(_git(repo, "grep", "-I", "-l", "-w", "-F", "-m", "1", tok, base, "--").strip())
    return _cache[key]


def added_tokens(diff: str, code_only: bool = False) -> set[str]:
    """Identifier tokens on added lines. code_only keeps only files with a source-code extension."""
    out, keep = set(), True
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            keep = (not code_only) or Path(line.split(" b/", 1)[-1].strip()).suffix in CODE_EXT
        elif keep and line.startswith("+") and not line.startswith("+++"):
            out.update(TOKEN.findall(line[1:]))
    return out


DEF_RE = re.compile(r"\b(?:fn|def|func|class|struct|enum|trait|const|type|static|interface)\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)")


def defined_tokens(diff: str, code_only: bool = True) -> set[tuple[str, str]]:
    """(file, name) pairs introduced by a definition keyword on added lines (fn x, def x, func x, struct X, ...)."""
    out, keep, path = set(), True, ""
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1].strip()
            keep = (not code_only) or Path(path).suffix in CODE_EXT
        elif keep and line.startswith("+") and not line.startswith("+++"):
            out.update((path, n) for n in DEF_RE.findall(line))
    return out


def defined_at_base(repo: Path, base: str, path: str, name: str) -> bool:
    """Is `name` already defined, via a definition keyword, in the SAME file at the parent commit? (Per-file, so a
    new method whose name is common elsewhere in the repo, e.g. Proxy::headers, still counts as new.)"""
    pat = re.compile(rf"(^|[^A-Za-z0-9_])(fn|def|func|class|struct|enum|trait|const|type|static|interface)\s+(\([^)]*\)\s*)?{re.escape(name)}([^A-Za-z0-9_]|$)", re.M)
    return bool(pat.search(_git(repo, "show", f"{base}:{path}")))


def _identifier_shaped(t: str) -> bool:
    """snake_case, camelCase, PascalCase or Capitalised (not lowercase words, not digit-led data like o644)."""
    return bool(re.search(r"_|[a-z][A-Z]", t.strip("_")) or re.match(r"^[A-Z][A-Za-z0-9]+$", t))


def compute_new_api(repo: Path, base: str, test_diff: str, src_diff: str) -> list[str]:
    """Public names the held-out tests reference that the change introduces. Two routes:
    (1) an identifier-shaped token on added test lines AND added source lines that occurs nowhere at the parent
        (`git grep -w`); (2) a name DEFINED (fn/def/func/struct/...) on added source lines, used on added test lines,
        and not defined anywhere at the parent (catches a new method that reuses a common word, e.g. Proxy::headers).
    Unexported Go funcs are dropped (not public API). Test-local names (classes defined only in the test diff) are excluded because route 1 needs the token in the
    source diff and route 2 needs a source definition."""
    in_tests, in_src = added_tokens(test_diff), added_tokens(src_diff, code_only=True)
    out = {t for t in in_tests & in_src
           if len(t) >= 3 and _identifier_shaped(t) and not exists_at_base(repo, base, t)}
    out |= {n for f, n in defined_tokens(src_diff)
            if n in in_tests and len(n) >= 3 and not n.startswith("_") and not defined_at_base(repo, base, f, n)}
    # public only: Go exports start with a capital; Python privates start with "_" (already skipped above)
    lang_go = {n for n in out if re.search(r"^[a-z]", n) and re.search(rf"^\+func\s+(\([^)]*\)\s*)?{n}\(", src_diff, re.M)}
    return sorted(out - lang_go)


# --------------------------------------------------------------------------- guards

def _code_like(stem: str) -> bool:
    return len(stem) >= 5 and bool(re.search(r"_|[a-z][A-Z]|[A-Z]{2,}[a-z]|\d", stem))


def guard_gold_path(text: str, gold: list[str]) -> list[str]:
    low, out = text.lower(), []
    for p in gold:
        base = p.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        if p.lower() in low or base.lower() in low:
            out.append(f"names gold file {base}")
        elif _code_like(stem) and re.search(rf"\b{re.escape(stem.lower())}\b", low):
            out.append(f"names gold stem {stem}")
    return out


def _sentence_start_spans(text: str) -> set[int]:
    return {0} | {m.end() for m in re.finditer(r"[.!?]\s+", text)}


def identifier_candidates(text: str) -> set[str]:
    """Tokens in an issue text that read as code identifiers (see module doc)."""
    cands = set()
    for m in re.finditer(r"`([^`]+)`", text):
        cands.update(TOKEN.findall(m.group(1)))
    starts = _sentence_start_spans(text)
    for m in TOKEN.finditer(text):
        t = m.group(0)
        if len(t) < 3:
            continue
        pascal = bool(re.match(r"^[A-Z][a-z0-9]+[A-Z]", t)) or bool(re.search(r"[a-z][A-Z]", t))
        code = "_" in t.strip("_") or bool(re.search(r"\d", t)) or pascal
        cap_word = t[0].isupper() and m.start() not in starts and not ACRONYM.match(t)
        screaming = ACRONYM.match(t) and len(t) >= 4
        if code or cap_word or screaming:
            cands.add(t)
    return cands


def _plain_word(t: str) -> bool:
    """Lowercase ordinary English word (e.g. `status`, `state`, `command`): generic, never an identifier leak."""
    return t.islower() and t.isalpha() and t in _english()


_EN: set[str] = set()


def _english() -> set[str]:
    if not _EN:
        try:
            _EN.update(w.strip() for w in open("/usr/share/dict/words") if w.strip().islower())
        except OSError:
            pass
    return _EN


def guard_existing_ident(text: str, repo: Path, base: str, new_api: list[str]) -> list[str]:
    exempt = set(new_api)
    bad = [t for t in sorted(identifier_candidates(text))
           if t not in exempt and t.lower() not in ALLOW and not _plain_word(t) and exists_at_base(repo, base, t)]
    return [f"names identifier present at parent: {bad[:5]}"] if bad else []


def _word_tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", s.lower())


def diff_ngrams(diff: str, n: int = NGRAM) -> set[tuple]:
    """n-grams of word tokens over each file's +/- lines concatenated (catches multi-line quotes)."""
    per_file, cur = [], []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            per_file.append(cur); cur = []
        elif line[:1] in "+-" and not line.startswith(("+++", "---")):
            cur += _word_tokens(line[1:])
    per_file.append(cur)
    out = set()
    for toks in per_file:
        out.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return out


def guard_verbatim(text: str, ngrams: set[tuple]) -> list[str]:
    toks = _word_tokens(text)
    for i in range(len(toks) - NGRAM + 1):
        if tuple(toks[i:i + NGRAM]) in ngrams:
            return [f"quotes diff verbatim: '{' '.join(toks[i:i + NGRAM])}'"]
    return []


def guard_new_api(text: str, new_api: list[str]) -> list[str]:
    miss = [a for a in new_api if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(a)}(?![A-Za-z0-9_])", text)]
    return [f"omits NEW_API items: {miss}"] if miss else []


def guard_sentences(text: str) -> list[str]:
    n = len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z`(])", text.strip()) if s.strip()])
    return [] if 2 <= n <= 4 else [f"{n} sentences (need 2-4)"]


def run_guards(text: str, *, repo: Path, base: str, gold: list[str], new_api: list[str], ngrams: set[tuple]) -> dict:
    res = {"gold_path": guard_gold_path(text, gold),
           "existing_ident": guard_existing_ident(text, repo, base, new_api),
           "verbatim_diff": guard_verbatim(text, ngrams),
           "missing_new_api": guard_new_api(text, new_api),
           "sentences": guard_sentences(text)}
    res["ok"] = not any(res[k] for k in list(res))
    return res


# --------------------------------------------------------------------------- generation

PROMPT = """You are helping build a coding-agent benchmark. Below is a real commit (subject, body, diff).
Write the issue a maintainer would have filed BEFORE this change was made: EXACTLY 3 sentences (never more than 4; longer answers are rejected) describing the
problem or missing feature and the expected behaviour, as a person who has NOT seen the diff.

Hard rules:
- You MUST name each of these new names exactly as written, in backticks, with its intended signature or shape
  (arguments, return value, where a user would call it) because the tests call them: {new_api}
- Do NOT name any file, path, module, or directory.
- Do NOT name any existing function, type, class, method, constant, config key or crate/package-internal name
  from the code. Describe existing things in plain words ("the proxy", "the session listing") instead. Standard
  library names and generic words (HTTP, JSON, str, dict, Vec, string) are fine.
- In signatures write types only (no parameter names), and refer to fields of new types in lowercase plain words
  ("keep and drop lists") unless the field itself is in the required list above.
- Do NOT copy code or comment text from the diff. Do not say "commit" or mention the diff.
{retry}
Be terse: at most ~90 words. List the required names compactly in one sentence if there are many.
Reply with the issue text only.

Repository: {repo}
Commit subject: {subject}
Commit body:
{body}

Diff (truncated):
{diff}
"""


ANCHOR_RULE = """
- ROUND 6 ANCHOR RULE: a coding agent will grep the whole workspace for words of this issue, so the text must contain NO searchable
  anchor into the existing code: no quoted error/log/warning message, no distinctive literal, no existing identifier, flag, config key,
  subtype or event name, and no rare domain word that the existing code is likely to spell the same way (e.g. a feature's own name).
  Describe EXISTING code only by its role and observable behaviour in generic words ("the code that parses incoming events",
  "the warning printed when files are overwritten" without quoting it). Technical issue style is kept (problem, expected behaviour,
  exact signatures), and the NEW names listed above must still appear exactly as written.
"""


def task_context(task: dict, env: dict) -> dict:
    repo = g.DATA / "repos" / g.repo_slug(task["repo_full"])
    base, src = task["base_commit"], task["source_commit"]
    tests = [p for _, p in env["heldout_test_entries"]]
    nontests = [p for _, p in env["nontest_entries"]]
    test_diff = _git(repo, "diff", "-M", "-U2", base, src, "--", *tests)
    src_diff = _git(repo, "diff", "-M", "-U2", base, src, "--", *nontests)
    return dict(repo=repo, base=base, gold=nontests, test_diff=test_diff, src_diff=src_diff,
                subject=_git(repo, "log", "-1", "--format=%s", src).strip() or task["task_text"],
                body=_git(repo, "log", "-1", "--format=%b", src).strip()[:1500])


SONNET_RATES = (2.00, 0.20, 2.50, 10.00)   # claude-sonnet-5-5 USD / 1M: fresh input, cache read, cache write, output
GATEWAY_LEDGER = HERE / "gateway_spend.jsonl"
HARD_CAP_USD = 40.00
CAP_MARGIN = 0.05


def gateway_cap() -> float:
    c = os.environ.get("E2E_GATEWAY_CAP")
    return min(HARD_CAP_USD, float(c)) if c else HARD_CAP_USD


def usage_usd(u: dict) -> float:
    """Gateway USD of one Anthropic-route response `usage` block (input_tokens EXCLUDES cache reads/writes on this route)."""
    i, cr, cw, o = SONNET_RATES
    return round(((u.get("input_tokens") or 0) * i + (u.get("cache_read_input_tokens") or 0) * cr
                  + (u.get("cache_creation_input_tokens") or 0) * cw + (u.get("output_tokens") or 0) * o) / 1e6, 6)


def ledger_total(path: Path = GATEWAY_LEDGER) -> float:
    if not path.exists():
        return 0.0
    tot = 0.0
    for ln in path.read_text().splitlines():
        if ln.strip():
            tot += float(json.loads(ln).get("usd") or 0)
    return tot


class the LLM gatewayCapExceeded(RuntimeError):
    pass


def check_cap(path: Path = GATEWAY_LEDGER, cap: float | None = None) -> float:
    """Refuse (raise) when cumulative + margin would exceed the cap. Called BEFORE every the LLM gateway call."""
    cap = gateway_cap() if cap is None else cap
    tot = ledger_total(path)
    if tot + CAP_MARGIN > cap:
        raise the LLM gatewayCapExceeded(f"the LLM gateway ledger ${tot:.2f} + ${CAP_MARGIN} margin would exceed the ${cap:.2f} cap; no call made")
    return tot


def ledger_issue_call(task_id: str, usage: dict, attempt: int, path: Path = GATEWAY_LEDGER) -> float:
    import time
    usd = usage_usd(usage)
    row = dict(at=time.strftime("%FT%TZ", time.gmtime()), task_id=task_id, attempt=attempt, usd=usd, purpose="issue_writer",
               in_tokens=usage.get("input_tokens"), out_tokens=usage.get("output_tokens"),
               cache_read_tokens=usage.get("cache_read_input_tokens"), cache_write_tokens=usage.get("cache_creation_input_tokens"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return usd


# --------------------------------------------------------------------------- round-6 fix: diff-introduced literal guard

_STR_LIT = re.compile(r'"((?:[^"\\\n]|\\.){4,})"|\'((?:[^\'\\\n]|\\.){4,})\'|`([^`\n]{4,})`')
_COMMENT = re.compile(r"(?:^|\s)(?:#|//|--|/\*|\*)\s*(.+)$")
_FLAG = re.compile(r"--[A-Za-z][A-Za-z0-9\-]+")
_KEBAB = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+")
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
MIN_TOK = 5


def _norm(x: str) -> str:
    return " ".join(x.lower().split())


def _stop() -> set[str]:
    try:
        import anchor_audit as aa
        return set(aa.STOP)
    except Exception:  # noqa: BLE001
        return set()


TECH_OK = {"base64", "sha256", "sha1", "utf8", "utf16", "md5", "uuid", "ipv4", "ipv6", "oauth", "oauth2", "json", "yaml", "regex", "unicode",
           "stdin", "stdout", "stderr", "async", "boolean", "timestamp", "metadata", "config", "filesystem", "hostname", "username", "namespace",
           "workspace", "allowlist", "denylist", "backend", "frontend", "runtime", "endpoint", "callback", "checksum", "plugin", "plugins"}


def _is_dict_word(t: str) -> bool:
    """Dictionary word incl. regular inflections (the system dictionary has no plurals / -ed / -ing forms) or a generic tech term."""
    w = t.lower()
    if w in _english() or w in _stop() or w in TECH_OK or w in {a.lower() for a in ALLOW}:
        return True
    for suf, add in (("s", ""), ("es", ""), ("ies", "y"), ("ed", ""), ("ed", "e"), ("d", ""), ("ing", ""), ("ing", "e"), ("er", ""),
                     ("ers", ""), ("ly", ""), ("ions", "ion"), ("ness", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            b = w[: -len(suf)] + add
            if b in _english() or b in _stop() or b in TECH_OK:
                return True
            if len(b) >= 3 and b[-1] == b[-2] and b[:-1] in _english():   # stopped -> stop
                return True
    return False


def _shaped_or_nondict(t: str) -> bool:
    """identifier-shaped (snake/camel/Pascal) or a non-dictionary word; numbers and short tokens never count."""
    if len(t) < MIN_TOK or t.isdigit():
        return False
    if _identifier_shaped(t) and not (t[0].isupper() and t[1:].islower() and t.lower() in _english()):
        return True
    return not _is_dict_word(t)


def _split_ident(t: str) -> list[str]:
    return [w for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", t) if len(w) >= MIN_TOK]


def diff_introduced(src_diff: str, new_api: list[str]) -> dict:
    """{'strings': [literal text], 'tokens': set of grep-able tokens} introduced by ADDED lines of the (non-test source) diff."""
    exempt = set(new_api)
    strings, toks = [], set()

    def add_tok(t: str):
        if t in exempt or len(t) < MIN_TOK or t.isdigit():
            return
        if _shaped_or_nondict(t):
            toks.add(t)

    for line in src_diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:]
        for m in _STR_LIT.finditer(body):
            lit = next(x for x in m.groups() if x is not None)
            strings.append(lit)
            for t in _FLAG.findall(lit) + _KEBAB.findall(lit) + _DOTTED.findall(lit):
                parts = [x for x in re.split(r"[-.]+", t) if x]
                if t not in exempt and (t.startswith("--") or any(not _is_dict_word(x) for x in parts)):
                    toks.add(t)
            for t in TOKEN.findall(lit):
                add_tok(t)
        cm = _COMMENT.search(body)
        if cm:
            for t in TOKEN.findall(cm.group(1)):
                if len(t) >= MIN_TOK and not _is_dict_word(t) and t not in exempt:
                    toks.add(t)
        for n in DEF_RE.findall(body):
            if n not in exempt and len(n) >= MIN_TOK:
                toks.add(n)
    return dict(strings=strings, tokens=toks)


def guard_diff_literal(text: str, src_diff: str, new_api: list[str]) -> list[str]:
    """Reject an issue text that carries a literal the gold source diff introduces (pure; see module docstring)."""
    intro = diff_introduced(src_diff, new_api)
    exempt = set(new_api)
    bad: list[str] = []
    lits = [_norm(s) for s in intro["strings"]]
    phrases = [m.group(1) for m in re.finditer(r"`([^`]+)`", text)]
    phrases += [m.group(1) or m.group(2) for m in re.finditer(r'"([^"\n]+)"|\u201c([^\u201d\n]+)\u201d', text)]
    for ph in phrases:
        n = _norm(ph)
        if len(n) >= 4 and ph not in exempt and any(n == l or n in l for l in lits):
            bad.append(ph)
    flat = set(intro["tokens"])
    cands = set(TOKEN.findall(text)) | set(_FLAG.findall(text)) | set(_KEBAB.findall(text)) | set(_DOTTED.findall(text))
    for t in sorted(cands):
        if t in exempt or len(t) < MIN_TOK or t.isdigit():
            continue
        if t in flat and (_shaped_or_nondict(t) or t.startswith("--") or "-" in t or "." in t):
            bad.append(t)
    bad = list(dict.fromkeys(bad))
    return ([f"contains literals the change introduces (an agent can grep them): {bad[:8]}; describe the behaviour in ordinary words "
             f"instead, do not quote or name them"] if bad else [])


def generate(backend, ctx: dict, task: dict, new_api: list[str], retry_note: str, attempt: int = 1, anchor_rule: bool = False) -> str:
    from asa.gateway_backend import ANTHROPIC_PATH
    diff = ctx["src_diff"][:12000] + "\n" + ctx["test_diff"][:6000]
    msg = PROMPT.format(new_api=", ".join(f"`{a}`" for a in new_api) or "(none)", repo=task["repo_full"],
                        subject=ctx["subject"], body=ctx["body"] or "(none)", diff=diff,
                        retry=(ANCHOR_RULE if anchor_rule else "") + retry_note)
    check_cap()
    data = backend._post(ANTHROPIC_PATH, {"model": backend.model, "max_tokens": 800,
                                          "messages": [{"role": "user", "content": msg}]})
    ledger_issue_call(task["task_id"], data.get("usage") or {}, attempt)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()


def guard_anchor(text: str, checker, spec: dict, new_api: list[str]) -> tuple[list[str], list[dict]]:
    """Round-6 guard: grep anchors of `text` in the monorepo workspace (see anchor_audit). Returns (problems, anchors)."""
    r = checker.check(spec, text, new_api)
    anchors = r["repo_anchors"]   # the broader repo-level flag (superset of the fixed gold-file definition): stricter for the rewrite
    return ([f"contains grep anchors (existing tokens that locate the code with one search): "
             f"{[a['token'] for a in anchors[:8]]}; rephrase WITHOUT these words, describe the code by role/behaviour instead"]
            if anchors else []), anchors


def write_issue(backend, task: dict, env: dict, checker=None, spec: dict | None = None) -> dict:
    ctx = task_context(task, env)
    new_api = compute_new_api(ctx["repo"], ctx["base"], ctx["test_diff"], ctx["src_diff"])
    ngrams = diff_ngrams(ctx["src_diff"] + "\n" + ctx["test_diff"])
    row = dict(task_id=task["task_id"], subject=ctx["subject"], new_api=new_api, issue_text=None,
               guard_results=None, attempts=0, ok=False, rejected=[])
    note = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        text = generate(backend, ctx, task, new_api, note, attempt=attempt, anchor_rule=checker is not None)
        gr = run_guards(text, repo=ctx["repo"], base=ctx["base"], gold=ctx["gold"], new_api=new_api, ngrams=ngrams)
        if checker is not None:
            gr["diff_literal"] = guard_diff_literal(text, ctx["src_diff"], new_api)
            gr["anchor"], anchors = guard_anchor(text, checker, spec, new_api)
            row["anchors_last"] = anchors
            gr["ok"] = not any(gr[k] for k in gr if k != "ok")
        row.update(attempts=attempt, guard_results=gr)
        if gr["ok"]:
            row.update(issue_text=text, ok=True)
            return row
        row["rejected"].append(dict(text=text, guard_results=gr))
        probs = "; ".join(p for k, v in gr.items() if k != "ok" for p in v)
        note = f"\nYour previous attempt was rejected: {probs}. Fix this.\n"
    return row


def main():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--anchor-guard", action="store_true", help="round 6: grep-anchor + diff-literal guards (Modal CPU)")
    ap.add_argument("--out", default="issues.json", help="output file under E2E_EXP_DIR")
    ap.add_argument("--tasks-file", help="JSON list of task ids (default: all gated tasks)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tasks = g.load_tasks()
    gated = [r["task_id"] for r in json.load(open(g.EXP / "gate_results.json"))["results"] if r.get("gated")]
    ids = [a.task] if a.task else (json.load(open(a.tasks_file)) if a.tasks_file else gated)
    envs = {t: json.load(open(g.EXP / "envs" / f"{t}.json")) for t in ids}
    if a.dry_run:
        for t in ids:
            ctx = task_context(tasks[t], envs[t])
            print(t[-45:], compute_new_api(ctx["repo"], ctx["base"], ctx["test_diff"], ctx["src_diff"]))
        return
    from asa.gateway_backend import GatewayBackend
    out_path = g.EXP / a.out
    prior = {r["task_id"]: r for r in json.loads(out_path.read_text())["issues"]} if out_path.exists() else {}
    checker = None
    if a.anchor_guard:
        import anchor_audit as aa
        checker = aa.AnchorChecker()
        checker.build_workspace()
    lock, chk_lock = threading.Lock(), threading.Lock()

    class _LockedChecker:   # one sandbox, serialised checks
        def check(self, spec, text, new_api):
            with chk_lock:
                return checker.check(spec, text, new_api)

    todo = [t for t in ids if not prior.get(t, {}).get("ok") and prior.get(t, {}).get("attempts", 0) < MAX_ATTEMPTS]

    def one(t):
        try:
            spec = aa.task_spec(t) if checker else None
            row = write_issue(GatewayBackend(model_id="sonnet-5.5-high"), tasks[t], envs[t],
                              checker=_LockedChecker() if checker else None, spec=spec)
        except the LLM gatewayCapExceeded as ex:
            print("STOP:", ex, flush=True)
            return
        with lock:
            prior[t] = row
            print(t[-40:], "ok" if row["ok"] else "FAILED", "attempts", row["attempts"], flush=True)
            out_path.write_text(json.dumps({"model": "claude-sonnet-5-5 (sonnet-5.5-high)", "issues": list(prior.values())}, indent=1))
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, todo))
    finally:
        if checker:
            checker.close()


if __name__ == "__main__":
    main()
