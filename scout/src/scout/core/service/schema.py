"""Row schemas for the gateway's namespaces (design section 7).

Chunk namespaces hold vector rows only. The Merkle sibling `{ns}__tree` and the tenant meta namespace are
vector-less (`dims=None`): turbopuffer rejects vector-less rows in a namespace that has a vector attribute.
"""

from scout.core.ports.store import AttrSpec, NamespaceSchema

CHUNK_DIMS = 1024  # voyage-code-4 output size
ROOT_ID = "root"  # id of the single root row in `{ns}__tree`

CHUNK_SCHEMA = NamespaceSchema(
    dims=CHUNK_DIMS,
    attrs={
        "kind": AttrSpec("string"),
        "file_id": AttrSpec("string"),
        "path_enc": AttrSpec("string", glob=True),
        "ext_tag": AttrSpec("string"),
        "dir_tags": AttrSpec("[]string"),
        "file_hash": AttrSpec("string"),
        "chunk_key": AttrSpec("string", filterable=False),
        "loc_enc": AttrSpec("string", filterable=False),
        "tokens": AttrSpec("[]string", filterable=False, fts="pre_tokenized_array"),
    },
)

TREE_SCHEMA = NamespaceSchema(
    dims=None,
    attrs={
        "kind": AttrSpec("string"),
        "children_enc": AttrSpec("string", filterable=False),
        "node_hash": AttrSpec("string", filterable=False),
        "root_hash": AttrSpec("string"),  # filterable: the root CAS is `upsert_condition=Eq(root_hash, base)`
        "expected_chunks": AttrSpec("int", filterable=False),
        "written_chunks": AttrSpec("int", filterable=False),
        "gen": AttrSpec("int", filterable=False),
        "commit": AttrSpec("string", filterable=False),
        "pages": AttrSpec("int", filterable=False),  # `kind=nodepage` staging marker: pages of that sync
        "upserted": AttrSpec("[]string", filterable=False),  # marker: node ids the page wrote
        "deleted": AttrSpec("[]string", filterable=False),  # marker: node ids `sync_commit` deletes
    },
)

META_SCHEMA = NamespaceSchema(
    dims=None,
    attrs={
        "kind": AttrSpec("string"),
        "sketch": AttrSpec("string", filterable=False),
        "leaves": AttrSpec("[]string", filterable=False),
        "proofs_required": AttrSpec("bool", filterable=False),
        "pages": AttrSpec("int", filterable=False),  # seed upload: number of proof pages
        "page": AttrSpec("int", filterable=False),
        "jaccard": AttrSpec("string", filterable=False),  # repr(float) of the seed match
        "complete": AttrSpec("bool", filterable=False),  # every proof page has arrived
    },
)
