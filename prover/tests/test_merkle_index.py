"""Merkle openings are bound to the queried index (CPU; the Rust twin is
verifier/src/protocol.rs merkle_tests).

Before this check the verifiers walked a path by the side bits the proof
supplied, so a valid path for ANY committed column answered a query for
another index. protocol.merkle_verify now takes the index and the tree width:
the ordering comes from the index, the path has exactly the tree's depth, the
index is in range, and an odd last node pairs only with itself. core's
merkle_verify is the same function, so the sampled runtime and the benches
get the same check."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import core
import protocol as pr
from layergkr import rs


def _tree(n):
    return rs.build_tree([pr.merkle_leaf([i, 100 + i]) for i in range(n)])


def test_core_and_protocol_share_one_check():
    assert core.merkle_verify is pr.merkle_verify


def test_depth_matches_the_builders():
    for n in range(1, 40):
        assert pr.merkle_depth(n) == len(_tree(n)) - 1


def test_valid_openings_accept_at_their_index():
    for n in (1, 2, 5, 8, 13):
        levels = _tree(n)
        root = levels[-1][0]
        for i in range(n):
            # core.merkle_path and rs.merkle_path share one convention
            assert core.merkle_path(levels, i) == rs.merkle_path(levels, i)
            assert pr.merkle_verify(levels[0][i], core.merkle_path(levels, i),
                                    root, i, n), (n, i)


def test_a_valid_path_for_another_column_rejects():
    levels = _tree(8)
    root = levels[-1][0]
    # column 5 with its own valid path, presented as the answer for index 1
    assert not pr.merkle_verify(levels[0][5], core.merkle_path(levels, 5),
                                root, 1, 8)
    # flipping every side bit does not buy a different ordering
    flipped = [(s, side ^ 1) for s, side in core.merkle_path(levels, 5)]
    assert not pr.merkle_verify(levels[0][5], flipped, root, 5, 8)


def test_truncated_overlong_and_out_of_range_reject():
    levels = _tree(8)
    root = levels[-1][0]
    path = core.merkle_path(levels, 3)
    assert not pr.merkle_verify(levels[0][3], path[:2], root, 3, 8)
    # an interior node offered as a leaf, its path one level short
    assert not pr.merkle_verify(levels[1][1], path[1:], root, 3, 8)
    # one level too many, even where the extra step hashes through
    long_root = rs._b3(root, root)
    assert not pr.merkle_verify(levels[0][3], path + [(root, 1)], long_root, 3, 8)
    assert not pr.merkle_verify(levels[0][3], path, root, 8, 8)
    assert not pr.merkle_verify(levels[0][3], path, root, -1, 8)


def test_a_self_paired_node_takes_no_other_sibling():
    levels = _tree(5)              # leaf 4 pairs with itself twice
    root = levels[-1][0]
    path = core.merkle_path(levels, 4)
    assert pr.merkle_verify(levels[0][4], path, root, 4, 5)
    path[0] = (levels[0][3], path[0][1])
    assert not pr.merkle_verify(levels[0][4], path, root, 4, 5)


def test_rs_commit_openings_are_index_bound():
    cfg = rs.Config(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)
    commit = rs.Commit(cfg, [rs.encode_row(cfg, [r * 8 + j for j in range(8)])
                             for r in range(3)])
    values, path = commit.open(7)
    assert commit.check_open(7, values, path)
    assert not commit.check_open(9, values, path)


def test_bridge_path_is_index_bound_in_its_own_convention():
    import wc_bridge as wc
    leaves = [pr.merkle_leaf([i]) for i in range(16)]
    levels = wc._tree(leaves)
    root = levels[-1][0]
    for i in range(16):
        assert wc._verify_path(leaves[i], wc._path(levels, i), root, i, 16)
    # another column's valid path, relabeled
    assert not wc._verify_path(leaves[13], wc._path(levels, 13), root, 5, 16)
    # the main tree's side bits are the opposite convention: never accepted
    assert not wc._verify_path(leaves[5], core.merkle_path(levels, 5), root, 5, 16)
    path = wc._path(levels, 5)
    assert not wc._verify_path(leaves[5], path[:-1], root, 5, 16)
    assert not wc._verify_path(leaves[5], path + [(root, 0)], root, 5, 16)
    assert not wc._verify_path(leaves[5], path, root, 16, 16)
