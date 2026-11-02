"""BM25 identifier tokens (design section 7): AC-12.5, FR-X5."""

import re

from scout.adapters.fake_store import FakeStore
from scout.core.contracts.wire import MAX_QUERY_TOKENS, MAX_TOKENS_PER_ROW
from scout.core.crypto import keys
from scout.core.crypto.tokens import (
    HMAC_TOKEN_HEX,
    STOPWORDS,
    hmac_tokens,
    identifier_tokens,
)
from scout.core.ports.store import AttrSpec, NamespaceSchema, Row

SK = keys.derive(bytes(range(64)))
OTHER = keys.derive(bytes(range(1, 65)))
SPELLINGS = ["getUserSession", "get_user_session", "GetUserSession", "get-user-session"]
SCHEMA = NamespaceSchema(
    dims=None, attrs={"tokens": AttrSpec("[]string", filterable=False, fts="pre_tokenized_array")}
)


def test_stoplist_has_sixty_words() -> None:
    """Design section 7: a 60-word stoplist."""
    assert len(STOPWORDS) == 60
    assert all(w == w.lower() and len(w) >= 2 for w in STOPWORDS)


def test_getusersession_splits_like_a_developer_searches() -> None:
    """AC-12.5: whole identifier plus parts, lowercased, whole first."""
    assert identifier_tokens("getUserSession") == ["getusersession", "get", "user", "session"]


def test_three_spellings_share_one_token_set() -> None:
    """AC-12.5: camel, snake, Pascal (and kebab) give the identical token set."""
    sets = [identifier_tokens(s) for s in SPELLINGS]
    assert all(s == sets[0] for s in sets)
    hm = [hmac_tokens(SK, s) for s in sets]
    assert all(h == hm[0] for h in hm)


def test_split_rules() -> None:
    assert identifier_tokens("HTTPServer") == ["httpserver", "http", "server"]
    assert identifier_tokens("parseXMLFile") == ["parsexmlfile", "parse", "xml", "file"]
    assert identifier_tokens("sha256Hash") == ["sha256hash", "sha", "256", "hash"]
    assert identifier_tokens("MAX_RETRY_COUNT") == ["maxretrycount", "max", "retry", "count"]
    assert identifier_tokens("__init__") == ["init"]
    assert identifier_tokens("user-agent_v2") == ["useragentv2", "user", "agent"]  # "v" too short
    assert identifier_tokens("session") == ["session"]  # single word: no duplicate


def test_short_tokens_and_stopwords_dropped() -> None:
    """Tokens len >= 2 (ruling at 2.11 review); stoplist dropped; whole word kept if not a stopword."""
    assert identifier_tokens("x = y + z") == []
    assert identifier_tokens("x = ab + id") == ["ab", "id"]
    assert identifier_tokens("def self return import class") == []
    assert identifier_tokens("returnValue") == ["returnvalue", "value"]  # part 'return' is a stopword
    assert identifier_tokens("get_id") == ["getid", "get", "id"]


def test_overlong_identifier_and_non_ascii_skipped() -> None:
    blob = "a" * 65
    assert identifier_tokens(blob) == []
    assert identifier_tokens("a" * 64) == ["a" * 64]
    assert identifier_tokens("café") == ["caf"]  # non-ASCII letter ends the identifier


def test_dedupe_and_deterministic_order() -> None:
    text = "alpha beta alpha gamma_delta beta Alpha"
    assert identifier_tokens(text) == ["alpha", "beta", "gammadelta", "gamma", "delta"]
    assert identifier_tokens(text) == identifier_tokens(text)


def test_caps() -> None:
    """Chunk cap 256 and query cap 32 come from wire.py; first occurrences win."""
    assert MAX_TOKENS_PER_ROW == 256
    assert MAX_QUERY_TOKENS == 32
    text = " ".join(f"word{i:04d}x" for i in range(1000))
    chunk = identifier_tokens(text)
    assert len(chunk) == MAX_TOKENS_PER_ROW
    assert len(set(chunk)) == len(chunk)
    query = identifier_tokens(text, MAX_QUERY_TOKENS)
    assert len(query) == MAX_QUERY_TOKENS
    assert query == chunk[:MAX_QUERY_TOKENS]
    assert len(hmac_tokens(SK, chunk)) == MAX_TOKENS_PER_ROW


def test_hmac_tokens_shape_and_key_separation() -> None:
    toks = identifier_tokens("getUserSession")
    h = hmac_tokens(SK, toks)
    assert len(h) == len(toks)
    assert all(re.fullmatch(r"[0-9a-f]{32}", t) for t in h)
    assert HMAC_TOKEN_HEX == 32
    assert h == hmac_tokens(SK, toks)
    assert set(h).isdisjoint(hmac_tokens(OTHER, toks))
    assert hmac_tokens(SK, ["abc", "abc"]) == hmac_tokens(SK, ["abc"])
    assert hmac_tokens(SK, []) == []


def test_no_plaintext_identifier_in_output() -> None:
    """Byte scan: no identifier or sub-token appears in the HMAC'd output."""
    src = "class UserSessionStore:\n    def get_user_session(self): return parseXMLFile(auth_token)"
    toks = identifier_tokens(src)
    blob = "\n".join(hmac_tokens(SK, toks)).encode()
    needles = set(toks) | set(re.findall(r"[A-Za-z_]{3,}", src))
    assert needles
    assert re.fullmatch(rb"[0-9a-f\n]+", blob)  # hex only, so any needle with a non-hex letter is absent
    for n in needles:
        if re.search(r"[^a-f]", n.lower()):
            assert n.encode() not in blob


async def test_every_spelling_retrieves_the_chunk_via_fakestore_bm25() -> None:
    """AC-12.5: tokens indexed as pre_tokenized_array; a query naming any spelling retrieves the chunk."""
    store = FakeStore()
    await store.ensure_namespace("ns", SCHEMA)
    chunks = {
        "c1": "def getUserSession(token):\n    return cache[token]",
        "c2": "class PaymentGateway:\n    def charge_card(self, amount): ...",
        "c3": "const renderSidebar = () => null",
    }
    await store.upsert(
        "ns", [Row(id=i, attrs={"tokens": hmac_tokens(SK, identifier_tokens(t))}) for i, t in chunks.items()]
    )
    for spelling in SPELLINGS:
        q = hmac_tokens(SK, identifier_tokens(spelling, MAX_QUERY_TOKENS))
        assert set(q) & set(hmac_tokens(SK, identifier_tokens(chunks["c1"])))
        hits = await store.query_bm25("ns", "tokens", q, top_k=5)
        assert [h.id for h in hits] == ["c1"]
    # a natural-language query naming the identifier still ranks its chunk first
    q = hmac_tokens(SK, identifier_tokens("where is get_user_session handled", MAX_QUERY_TOKENS))
    assert (await store.query_bm25("ns", "tokens", q, top_k=5))[0].id == "c1"
    # another key's tokens find nothing
    other = hmac_tokens(OTHER, identifier_tokens("getUserSession"))
    assert await store.query_bm25("ns", "tokens", other, top_k=5) == []


def test_short_specific_identifiers_are_searchable_and_short_keywords_are_not():
    """db/id/io/os/fs/ui are common, specific identifiers; if/in/is/or are keywords and stay out."""
    toks = identifier_tokens("if db.Get(id) in os.environ or fs.Open(): ui.io")
    for want in ("db", "id", "os", "fs", "ui", "io", "get", "open", "environ"):
        assert want in toks
    for kw in ("if", "in", "or"):
        assert kw not in toks
    assert identifier_tokens("x y z") == []
