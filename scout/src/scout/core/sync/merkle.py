"""Keyed Merkle tree and BFS diff (design section 8): wire bytes scale with changed files x depth.

A file's leaf is `HMAC-SHA256(K_leaf, bytes)` (hex). A directory node hashes its children sorted by
encrypted name: `SHA256(concat(enc_name_hex || kind_byte || child_hash))`, child hash as raw bytes (names
are hex and hashes fixed width, so the encoding is unambiguous). Names are
per-segment `enc_path` ciphertexts whose AAD includes the parent, so identical files under different parents
share a leaf but not a node hash. Empty directories do not exist (only files are entries). Node ids come from
`crypto.ids.node_id` and blobs are sealed by `crypto.rows`; the server only ever sees ids, hashes and blobs.

`diff` walks level by level, fetching only nodes whose hash differs from the remote's, so a one-file edit at
depth d costs d + 1 fetches whatever the repo size. Remote-only subtrees are enumerated (to name removed
files), so deleting a directory costs one fetch per directory in it.
"""

import hashlib
import hmac
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from scout.core.crypto import ids, paths, rows
from scout.core.crypto.keys import CryptoError, Subkeys

FILE = "f"
DIR = "d"
_KIND_BYTE = {FILE: b"\x00", DIR: b"\x01"}
_LEAF_LEN = 64

Child = tuple[str, str, str]  # (enc_name, kind, hash_hex)


class MerkleError(ValueError):
    """Bad input, or a remote node that is missing, unreadable or inconsistent with its parent."""


class SealedNode(NamedTuple):
    """The server-side node row payload (`children_enc`, `node_hash`)."""

    children_enc: str
    node_hash: str


NodeFetcher = Callable[[Sequence[str]], Mapping[str, SealedNode]]


@dataclass(frozen=True, slots=True, eq=False)
class Node:
    dirpath: str  # plaintext, "" for the root
    enc_dirpath: str
    node_id: str
    node_hash: str
    children: tuple[Child, ...]  # sorted by enc_name
    names: dict[str, str]  # enc_name -> plaintext name (client side only, never sealed)


@dataclass(frozen=True, slots=True, eq=False)
class Tree:
    tenant: str
    repo_id: str
    nodes: dict[str, Node]  # dirpath -> node
    leaves: dict[str, str]  # file path -> leaf hex

    @property
    def root_hash(self) -> str:
        return self.nodes[""].node_hash


@dataclass(frozen=True, slots=True)
class Delta:
    """Files by plaintext path with their leaf (added/changed: local leaf; removed: remote leaf)."""

    added: dict[str, str] = field(default_factory=lambda: {})
    changed: dict[str, str] = field(default_factory=lambda: {})
    removed: dict[str, str] = field(default_factory=lambda: {})
    nodes_fetched: int = 0
    rounds: int = 0

    @property
    def empty(self) -> bool:
        return not (self.added or self.changed or self.removed)


def leaf_hash(keys: Subkeys, data: bytes) -> str:
    return hmac.new(keys.leaf, data, hashlib.sha256).hexdigest()


def node_hash(children: Iterable[Child]) -> str:
    h = hashlib.sha256()
    for enc_name, kind, child_hash in sorted(children):
        h.update(enc_name.encode("ascii"))
        h.update(_KIND_BYTE[kind])
        h.update(bytes.fromhex(child_hash))
    return h.hexdigest()


def _join(parent: str, name: str) -> str:
    return f"{parent}/{name}" if parent else name


def build_tree(entries: Iterable[tuple[str, str]], keys: Subkeys, tenant: str, repo_id: str) -> Tree:
    """`entries` are `(path, leaf_hex)` in any order; the result does not depend on that order."""
    leaves: dict[str, str] = {}
    for path, leaf in entries:
        if len(leaf) != _LEAF_LEN or leaf.strip("0123456789abcdef"):
            raise MerkleError("leaf must be 64 lowercase hex chars")
        if path in leaves:
            raise MerkleError(f"duplicate path {path!r}")
        leaves[path] = leaf

    kids: dict[str, list[tuple[str, str, str, str]]] = {"": []}  # dirpath -> (enc, kind, hash, name)
    dir_enc: dict[str, str] = {"": ""}  # dirpath -> encrypted dirpath
    for path, leaf in leaves.items():
        enc = paths.enc_path(keys, tenant, repo_id, path)
        segs, esegs = path.split("/"), enc.split("/")
        kids.setdefault("/".join(segs[:-1]), []).append((esegs[-1], FILE, leaf, segs[-1]))
        for i in range(len(segs) - 1, 0, -1):
            d = "/".join(segs[:i])
            if d in dir_enc:
                break
            dir_enc[d] = "/".join(esegs[:i])
            kids.setdefault(d, [])

    nodes: dict[str, Node] = {}
    for dirpath in sorted(kids, key=lambda d: -(d.count("/") + 1) if d else 0):  # deepest first, root last
        items = sorted(kids[dirpath])
        if len({e for e, _, _, _ in items}) != len(items):
            raise MerkleError(f"{dirpath or '.'!r} has a file and a directory with one name")
        children = tuple((e, k, h) for e, k, h, _ in items)
        node = Node(
            dirpath=dirpath,
            enc_dirpath=dir_enc[dirpath],
            node_id=ids.node_id(keys, repo_id, dirpath),
            node_hash=node_hash(children),
            children=children,
            names={e: n for e, _, _, n in items},
        )
        nodes[dirpath] = node
        if dirpath:
            parent, _, name = dirpath.rpartition("/")
            kids[parent].append((dir_enc[dirpath].rpartition("/")[2], DIR, node.node_hash, name))
    return Tree(tenant=tenant, repo_id=repo_id, nodes=nodes, leaves=leaves)


def seal_tree(tree: Tree, keys: Subkeys) -> dict[str, SealedNode]:
    """Every node as its server row payload, keyed by node id."""
    return {
        n.node_id: SealedNode(
            rows.seal_node(keys, tree.tenant, tree.repo_id, n.node_id, list(n.children)), n.node_hash
        )
        for n in tree.nodes.values()
    }


def _local_files(tree: Tree, dirpath: str) -> dict[str, str]:
    out: dict[str, str] = {}
    stack = [dirpath]
    while stack:
        node = tree.nodes[stack.pop()]
        for enc_name, kind, h in node.children:
            p = _join(node.dirpath, node.names[enc_name])
            if kind == FILE:
                out[p] = h
            else:
                stack.append(p)
    return out


def diff_leaves(old: Mapping[str, str], new: Mapping[str, str]) -> Delta:
    """Local fast path (server root equals the saved root): diff two path -> leaf maps, zero fetches."""
    return Delta(
        added={p: h for p, h in new.items() if p not in old},
        changed={p: h for p, h in new.items() if p in old and old[p] != h},
        removed={p: h for p, h in old.items() if p not in new},
    )


def diff(local: Tree, remote_root: str | None, fetch: NodeFetcher, keys: Subkeys) -> Delta:
    """Delta that turns the remote tree into `local`; `remote_root` None means no tree on the server."""
    if remote_root is None:
        return Delta(added=dict(local.leaves))
    if remote_root == local.root_hash:
        return Delta()
    tenant, repo_id = local.tenant, local.repo_id
    added: dict[str, str] = {}
    changed: dict[str, str] = {}
    removed: dict[str, str] = {}
    fetched = rounds = 0
    level: list[tuple[str, str, str]] = [("", "", remote_root)]  # (dirpath, enc_dirpath, expected hash)
    while level:
        node_ids = [ids.node_id(keys, repo_id, d) for d, _, _ in level]
        got = fetch(node_ids)
        fetched += len(node_ids)
        rounds += 1
        nxt: list[tuple[str, str, str]] = []
        for (dirpath, enc_dirpath, expected), nid in zip(level, node_ids, strict=True):
            sealed = got.get(nid)
            if sealed is None:
                raise MerkleError("remote node missing")
            try:
                remote = rows.open_node(keys, tenant, repo_id, nid, sealed.children_enc)
            except CryptoError as e:
                raise MerkleError("remote node unreadable") from e
            if node_hash(remote) != expected or sealed.node_hash != expected:
                raise MerkleError("remote node does not match its parent hash")
            rmap = {e: (k, h) for e, k, h in remote}
            lnode = local.nodes.get(dirpath)
            lmap = {e: (k, h) for e, k, h in lnode.children} if lnode else {}
            for enc_name in lmap.keys() | rmap.keys():
                lv, rv = lmap.get(enc_name), rmap.get(enc_name)
                if lv == rv:
                    continue
                enc_child = _join(enc_dirpath, enc_name)
                name = lnode.names.get(enc_name) if lnode else None
                if name is None:
                    try:
                        name = paths.dec_path(keys, tenant, repo_id, enc_child).rpartition("/")[2]
                    except CryptoError as e:
                        raise MerkleError("remote node has an unreadable name") from e
                child = _join(dirpath, name)
                if lv is not None and rv is not None and lv[0] == rv[0]:
                    if lv[0] == FILE:
                        changed[child] = lv[1]
                    else:
                        nxt.append((child, enc_child, rv[1]))
                    continue
                if lv is not None:  # local only, or the kind flipped
                    if lv[0] == FILE:
                        added[child] = lv[1]
                    else:
                        added.update(_local_files(local, child))
                if rv is not None:  # remote only, or the kind flipped
                    if rv[0] == FILE:
                        removed[child] = rv[1]
                    else:
                        nxt.append((child, enc_child, rv[1]))
        level = nxt
    return Delta(added=added, changed=changed, removed=removed, nodes_fetched=fetched, rounds=rounds)
