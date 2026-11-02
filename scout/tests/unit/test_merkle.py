"""Keyed Merkle tree and BFS diff (design section 8): FR-C9, AC-3.1."""

import random
import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scout.core.crypto import keys, rows
from scout.core.sync import merkle
from scout.core.sync.merkle import Delta, MerkleError, SealedNode

SK = keys.derive(bytes(range(64)))
TENANT = "acme"
REPO = "ab" * 16


def leaf(content: str) -> str:
    return merkle.leaf_hash(SK, content.encode())


class Remote:
    """A fake `{ns}__tree`: serves sealed nodes by id and counts every id requested."""

    def __init__(self, tree: merkle.Tree) -> None:
        self.rows = merkle.seal_tree(tree, SK)
        self.root = tree.root_hash
        self.fetched = 0

    def __call__(self, node_ids):
        self.fetched += len(node_ids)
        return {i: self.rows[i] for i in node_ids if i in self.rows}


def tree_of(files: dict[str, str]) -> merkle.Tree:
    return merkle.build_tree(((p, leaf(c)) for p, c in files.items()), SK, TENANT, REPO)


def delta_of(local: dict[str, str], remote: dict[str, str]) -> tuple[Delta, Remote]:
    r = Remote(tree_of(remote))
    return merkle.diff(tree_of(local), r.root, r, SK), r


def expected(local: dict[str, str], remote: dict[str, str]):
    return (
        {p: leaf(c) for p, c in local.items() if p not in remote},
        {p: leaf(c) for p, c in local.items() if p in remote and remote[p] != c},
        {p: leaf(c) for p, c in remote.items() if p not in local},
    )


BASE = {"README.md": "r", "src/a.py": "a", "src/pkg/b.py": "b", "src/pkg/c.py": "c", "docs/x.md": "x"}


def test_identical_trees_empty_delta_zero_fetches():
    d, r = delta_of(BASE, dict(BASE))
    assert d.empty
    assert d.nodes_fetched == 0
    assert r.fetched == 0


def test_no_remote_tree_is_all_added():
    d = merkle.diff(tree_of(BASE), None, Remote(tree_of({"z": "z"})), SK)
    assert d.added == {p: leaf(c) for p, c in BASE.items()}
    assert d.nodes_fetched == 0


def test_one_edit_fetches_at_most_depth_plus_one():
    local = {**BASE, "src/pkg/c.py": "changed"}
    d, r = delta_of(local, BASE)
    assert d.changed == {"src/pkg/c.py": leaf("changed")}
    assert not d.added
    assert not d.removed
    assert d.nodes_fetched == r.fetched == 3
    assert d.rounds == 3


def test_add_modify_delete_and_new_dir():
    local = {**BASE, "src/new.py": "n", "src/a.py": "a2", "fresh/deep/er/f.py": "f"}
    del local["docs/x.md"]
    d, _ = delta_of(local, BASE)
    assert (d.added, d.changed, d.removed) == expected(local, BASE)


def test_remote_dir_removed_lists_every_file():
    local = {p: c for p, c in BASE.items() if not p.startswith("src/")}
    d, _ = delta_of(local, BASE)
    assert d.removed == {p: leaf(c) for p, c in BASE.items() if p.startswith("src/")}
    assert not d.added
    assert not d.changed


def test_file_dir_kind_flip():
    d, _ = delta_of({"a": "file", "z": "z"}, {"a/b.py": "b", "z": "z"})
    assert d.added == {"a": leaf("file")}
    assert d.removed == {"a/b.py": leaf("b")}
    d, _ = delta_of({"a/b.py": "b", "z": "z"}, {"a": "file", "z": "z"})
    assert d.added == {"a/b.py": leaf("b")}
    assert d.removed == {"a": leaf("file")}


def test_rename_keeps_leaf_and_reports_move():
    local = {p: c for p, c in BASE.items() if p != "src/pkg/b.py"} | {"lib/moved/b2.py": "b"}
    d, _ = delta_of(local, BASE)
    assert d.removed == {"src/pkg/b.py": leaf("b")}
    assert d.added == {"lib/moved/b2.py": leaf("b")}
    assert d.removed["src/pkg/b.py"] == d.added["lib/moved/b2.py"]  # same leaf: chunk keys reusable
    assert tree_of(local).leaves["lib/moved/b2.py"] == tree_of(BASE).leaves["src/pkg/b.py"]


def test_identical_files_share_leaf_but_not_node_hash():
    t = tree_of({"a/f.py": "same", "b/f.py": "same", "a2/f.py": "same"})
    assert len({t.leaves[p] for p in t.leaves}) == 1
    assert len({t.nodes[d].node_hash for d in ("a", "b", "a2")}) == 3
    t2 = tree_of({"a/f.py": "same", "a/g.py": "same"})  # same leaf, two names in one dir
    (e1, _, h1), (e2, _, h2) = t2.nodes["a"].children
    assert h1 == h2
    assert e1 != e2


def test_leaf_is_keyed_and_node_hash_covers_names_and_kind():
    other = keys.derive(bytes(range(1, 65)))
    assert merkle.leaf_hash(SK, b"x") != merkle.leaf_hash(other, b"x")
    assert tree_of({"a": "x"}).root_hash != tree_of({"b": "x"}).root_hash
    h = "00" * 32
    assert merkle.node_hash([("aa", merkle.FILE, h)]) != merkle.node_hash([("aa", merkle.DIR, h)])


def test_deterministic_regardless_of_input_order():
    entries = [(p, leaf(c)) for p, c in BASE.items()]
    roots = set()
    for seed in range(8):
        random.Random(seed).shuffle(entries)  # noqa: S311
        t = merkle.build_tree(entries, SK, TENANT, REPO)
        roots.add(t.root_hash)
    assert len(roots) == 1
    assert tree_of(BASE).root_hash == merkle.build_tree(reversed(entries), SK, TENANT, REPO).root_hash


def test_empty_dir_is_omitted():
    t = tree_of({"keep.py": "k", "gone/only.py": "o"})
    after = tree_of({"keep.py": "k"})
    assert "gone" in t.nodes
    assert "gone" not in after.nodes
    assert t.root_hash != after.root_hash
    d, _ = delta_of({"keep.py": "k"}, {"keep.py": "k", "gone/only.py": "o"})
    assert d.removed == {"gone/only.py": leaf("o")}
    assert set(tree_of({}).nodes) == {""}  # empty repo: a lone empty root


def test_local_fast_path_diff_leaves_needs_no_fetch():
    old, new = tree_of(BASE), tree_of({**BASE, "src/a.py": "a2", "n.py": "n"})
    d = merkle.diff_leaves(old.leaves, new.leaves)
    assert d.changed == {"src/a.py": leaf("a2")}
    assert d.added == {"n.py": leaf("n")}
    assert not d.removed
    assert d.nodes_fetched == 0


def test_file_and_dir_with_one_name_rejected():
    with pytest.raises(MerkleError):
        merkle.build_tree([("a", leaf("1")), ("a/b", leaf("2"))], SK, TENANT, REPO)
    with pytest.raises(MerkleError):
        merkle.build_tree([("a", leaf("1")), ("a", leaf("2"))], SK, TENANT, REPO)
    with pytest.raises(MerkleError):
        merkle.build_tree([("a", "nothex")], SK, TENANT, REPO)
    with pytest.raises(ValueError, match="relative"):
        merkle.build_tree([("a//b", leaf("1"))], SK, TENANT, REPO)


def test_sealed_nodes_open_and_are_bound_to_their_id():
    t = tree_of(BASE)
    sealed = merkle.seal_tree(t, SK)
    root = t.nodes[""]
    assert rows.open_node(SK, TENANT, REPO, root.node_id, sealed[root.node_id].children_enc) == list(
        root.children
    )
    assert sealed[root.node_id].node_hash == t.root_hash
    # a node row moved under another node's id (or a lying server) is refused
    swapped = {root.node_id: sealed[t.nodes["src"].node_id]}
    with pytest.raises(MerkleError):
        merkle.diff(tree_of({**BASE, "x": "1"}), t.root_hash, lambda ids_: swapped, SK)
    with pytest.raises(MerkleError, match="missing"):
        merkle.diff(tree_of({**BASE, "x": "1"}), t.root_hash, lambda ids_: {}, SK)
    liar = {root.node_id: SealedNode(sealed[root.node_id].children_enc, "00" * 32)}
    with pytest.raises(MerkleError, match="parent hash"):
        merkle.diff(tree_of({**BASE, "x": "1"}), t.root_hash, lambda ids_: liar, SK)


# --- property test: diff returns exactly the added/modified/deleted files ---------------------------

_NAME = st.sampled_from(["a", "b", "c", "dd", "e.py"])


@st.composite
def file_maps(draw) -> dict[str, str]:
    paths = draw(st.sets(st.lists(_NAME, min_size=1, max_size=4).map("/".join), max_size=14))
    # a path that is also a directory of another path cannot exist on a filesystem
    paths = {p for p in paths if not any(q.startswith(p + "/") for q in paths)}
    return {p: draw(st.sampled_from(["0", "1", "2"])) for p in paths}


@st.composite
def edited(draw, files: dict[str, str]) -> dict[str, str]:
    out = dict(files)
    for _ in range(draw(st.integers(0, 6))):
        op = draw(st.sampled_from(["add", "modify", "delete", "rename", "move"]))
        if op == "add":
            p = "/".join(draw(st.lists(_NAME, min_size=1, max_size=4)))
            if p not in out and not any(q.startswith(p + "/") or p.startswith(q + "/") for q in out):
                out[p] = draw(st.sampled_from(["0", "1", "2"]))
        elif out:
            p = draw(st.sampled_from(sorted(out)))
            if op == "modify":
                out[p] = out[p] + "!"
            elif op == "delete":
                del out[p]
            else:
                head, _, name = p.rpartition("/")
                new_dir = draw(st.sampled_from(["", "a", "b/c", "z/y/x"])) if op == "move" else head
                new = "/".join(s for s in (new_dir, name if op == "move" else name + "_r") if s)
                if new not in out and not any(
                    q.startswith(new + "/") or new.startswith(q + "/") for q in out
                ):
                    out[new] = out.pop(p)
    return out


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_property_diff_is_exact(data):
    remote = data.draw(file_maps())
    local = data.draw(edited(remote))
    d, r = delta_of(local, remote)
    assert (d.added, d.changed, d.removed) == expected(local, remote)
    assert d.nodes_fetched == r.fetched
    assert (d.nodes_fetched == 0) == (local == remote)
    # applying the delta to the remote yields the local tree (same root)
    applied = {p: c for p, c in remote.items() if p not in d.removed}
    applied |= {p: local[p] for p in (*d.added, *d.changed)}
    assert tree_of(applied).root_hash == tree_of(local).root_hash


# --- wire-size and performance ---------------------------------------------------------------------


def synthetic(n: int) -> dict[str, str]:
    """Depth-3 directories (`a/b/c/f`), up to 20 x 20 x 5 dirs, spread evenly; contents are the index."""
    return {f"a{i % 20}/b{(i // 20) % 20}/c{(i // 400) % 5}/f{i}.py": str(i) for i in range(n)}


@pytest.mark.parametrize("n", [10, 1_000, 20_000])
def test_single_edit_fetches_do_not_grow_with_repo_size(n):
    remote = synthetic(n)
    local = dict(remote)
    victim = f"a{(n - 1) % 20}/b{((n - 1) // 20) % 20}/c{((n - 1) // 400) % 5}/f{n - 1}.py"
    local[victim] = "edited"
    t0 = time.perf_counter()
    r = Remote(tree_of(remote))
    local_tree = tree_of(local)
    d = merkle.diff(local_tree, r.root, r, SK)
    assert d.changed == {victim: leaf("edited")}
    assert not d.added
    assert not d.removed
    depth = victim.count("/")
    assert d.nodes_fetched == r.fetched == depth + 1 == 4  # root + 3 dirs, independent of n
    assert d.rounds == depth + 1
    assert time.perf_counter() - t0 < 20


def test_build_20k_files_under_two_seconds():
    entries = [(p, leaf(c)) for p, c in synthetic(20_000).items()]
    t0 = time.perf_counter()
    t = merkle.build_tree(entries, SK, TENANT, REPO)
    assert time.perf_counter() - t0 < 2.0
    assert len(t.leaves) == 20_000
