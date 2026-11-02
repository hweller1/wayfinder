"""Client-gateway wire models (design section 6): requests forbid extras and bound lists."""

from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints

from scout.core.contracts.types import IndexStatus, RequestModel, ResponseModel

ID_PATTERN = r"^[0-9a-f]{16,32}$"
MAX_NODE_IDS = 500
MAX_NODE_PAGES = 400  # 400 pages x 500 rows = 200k node rows per sync (100k files plus directories)
MAX_CHUNK_KEYS = 2000
MAX_ROWS = 2000
MAX_FILES = 500
MAX_PROOFS = 5000  # design 5.4: one seed request carries up to 5k leaf-hash proofs (one page)
MAX_PROOF_PAGES = 20  # 20 pages x 5k = 100k proofs, the file cap of the indexer (design 15)
MAX_TOKENS_PER_ROW = 256
MAX_QUERY_TOKENS = 32

Id = Annotated[str, StringConstraints(pattern=ID_PATTERN)]


class TenantInfo(ResponseModel):
    tenant: str
    key_id: str


class StatusReq(RequestModel):
    repo_id: Id
    branch_id: Id


class SyncBeginReq(RequestModel):
    repo_id: Id
    branch_id: Id
    base_branch_id: Id | None = None
    root: str  # client Merkle root hash (hex)
    expected_chunks: int = Field(ge=0)


class SyncBeginRes(ResponseModel):
    ns_state: Literal["absent", "empty", "populated"]
    server_root: str | None = None
    coverage: float = Field(ge=0.0, le=1.0)


class NodesReq(RequestModel):
    repo_id: Id
    branch_id: Id
    ids: list[Id] = Field(max_length=MAX_NODE_IDS)


class NodesRes(ResponseModel):
    blobs: dict[str, str]  # node id -> base64 encrypted blob


class PlanReq(RequestModel):
    repo_id: Id
    branch_id: Id
    chunk_keys: list[Id] = Field(max_length=MAX_CHUNK_KEYS)


class PlanRes(ResponseModel):
    missing: list[str]


class WireRow(RequestModel):
    """One sealed chunk row (no vector: the gateway fills it from the embed cache)."""

    id: str = Field(min_length=1, max_length=64)
    file_id: Id
    path_enc: str
    ext_tag: str
    dir_tags: list[str] = Field(max_length=64)
    file_hash: str
    chunk_key: Id
    loc_enc: str
    tokens: list[str] = Field(max_length=MAX_TOKENS_PER_ROW)


class ReplaceFile(RequestModel):
    file_id: Id
    file_hash: str


class EmbedReq(RequestModel):
    repo_id: Id
    branch_id: Id
    texts: dict[Id, str] = Field(max_length=MAX_CHUNK_KEYS)  # chunk_key -> text
    rows: list[WireRow] = Field(max_length=MAX_ROWS)
    replace: list[ReplaceFile] = Field(default_factory=list[ReplaceFile], max_length=MAX_FILES)
    delete_files: list[Id] = Field(default_factory=list, max_length=MAX_FILES)


class EmbedRes(ResponseModel):
    written: int
    embedded_tokens: int


class NodeUpsert(RequestModel):
    id: Id
    node_hash: str
    children_enc: str


class StageNodesReq(RequestModel):
    """One page of the node rows (and node deletes) of the tree whose root is `new_root`. Rows are written
    to `{ns}__tree` now; the deletes are only recorded and run by `sync_commit` after the root write."""

    repo_id: Id
    branch_id: Id
    base_root: str | None = None  # the root the client synced from (`CommitReq.expected_root`)
    new_root: str
    page: int = Field(ge=0)  # 0-based, < pages
    pages: int = Field(ge=1, le=MAX_NODE_PAGES)
    node_upserts: list[NodeUpsert] = Field(default_factory=list[NodeUpsert], max_length=MAX_NODE_IDS)
    node_deletes: list[Id] = Field(default_factory=list, max_length=MAX_NODE_IDS)


class StageNodesRes(ResponseModel):
    page: int
    staged: int  # node rows written by this call (0: the tree is already committed)


class CommitReq(RequestModel):
    repo_id: Id
    branch_id: Id
    expected_root: str | None = None
    new_root: str
    node_upserts: list[NodeUpsert] = Field(default_factory=list[NodeUpsert], max_length=MAX_NODE_IDS)
    node_deletes: list[Id] = Field(default_factory=list, max_length=MAX_NODE_IDS)
    staged_pages: int = Field(default=0, ge=0, le=MAX_NODE_PAGES)  # pages 0..n-1 staged for `new_root`
    written_chunks: int = Field(ge=0)
    expected_chunks: int | None = Field(default=None, ge=0)  # absent: the sync wrote everything (coverage 1)
    commit: str | None = Field(default=None, max_length=64)  # git commit of the synced tree
    minhash_b64: str | None = Field(default=None, max_length=1024)  # this namespace's sketch (512 B)


class CommitRes(IndexStatus):
    """The commit response is the resulting `IndexStatus` (design section 6)."""


class SearchScope(RequestModel):
    """Whitelist of server-side filters; no client-supplied attribute names reach the store."""

    path_glob_enc: str | None = None
    ext_tag: str | None = None
    dir_tag: str | None = None


class SearchReq(RequestModel):
    repo_id: Id
    branch_id: Id
    query: str = Field(min_length=1, max_length=100_000)
    k: int = Field(default=10, ge=1, le=30)
    scope: SearchScope = Field(default_factory=SearchScope)
    tokens: list[str] = Field(default_factory=list, max_length=MAX_QUERY_TOKENS)
    overfetch: int = Field(default=1, ge=1, le=5)


class WireHit(ResponseModel):
    path_enc: str
    loc_enc: str
    chunk_key: str
    file_hash: str
    score: float | None = None


class SearchRes(ResponseModel):
    hits: list[WireHit]


class SeedReq(RequestModel):
    """Page `page` (0-based) of `pages` of the client's leaf-hash proof set. Page 0 carries the match
    decision and must come first; the proofs only take effect once every page has arrived (until then the
    seeded namespace returns no search hits at all). Pages may be re-sent and may arrive in any order after
    page 0; a repo with at most 5k files is one request (`page=0, pages=1`)."""

    repo_id: Id
    branch_id: Id
    minhash_b64: str
    proofs: list[str] = Field(default_factory=list, max_length=MAX_PROOFS)
    page: int = Field(default=0, ge=0, lt=MAX_PROOF_PAGES)
    pages: int = Field(default=1, ge=1, le=MAX_PROOF_PAGES)


class SeedRes(ResponseModel):
    seeded: bool
    jaccard: float | None = None
    complete: bool = True  # False while proof pages are outstanding: search returns nothing until True


class ChatReq(RequestModel):
    """OpenAI chat body subset plus `thinking` (-> chat_template_kwargs.enable_thinking)."""

    messages: list[dict[str, Any]] = Field(min_length=1, max_length=1000)
    tools: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]], max_length=64)
    tool_choice: Literal["auto", "required", "none"] = "auto"
    temperature: float = Field(default=0.85, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1)
    presence_penalty: float = Field(default=1.1, ge=-2.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32_768)
    thinking: bool = True


WIRE_MODELS = (
    TenantInfo,
    StatusReq,
    SyncBeginReq,
    SyncBeginRes,
    NodesReq,
    NodesRes,
    PlanReq,
    PlanRes,
    WireRow,
    ReplaceFile,
    EmbedReq,
    EmbedRes,
    NodeUpsert,
    StageNodesReq,
    StageNodesRes,
    CommitReq,
    CommitRes,
    SearchScope,
    SearchReq,
    WireHit,
    SearchRes,
    SeedReq,
    SeedRes,
    ChatReq,
)
