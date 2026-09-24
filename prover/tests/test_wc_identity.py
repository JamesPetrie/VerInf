"""The bridge enrollment's trusted identity (CPU; review 2026-09-23 finding 2).

The enrollment root commits the columns, not how they are read: B and lam come
off the wire, so a proof could move the boundary between weights and masks
under the same root. The trusted value (the verifier's argv[4]) is now a
versioned, domain-separated digest of the root, the manifest digest, the
geometry, the claim set's ordered row layout and the padding rule, identical
in wc_bridge.enrollment_identity and the Rust wc_enrollment_identity."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wc_bridge as wc

ROOT = bytes(range(32))
MANIFEST = bytes([3]) * 32
P12 = wc.WcParams(B=12, lam=4, N_w=64, q_w=40)


def test_identity_vectors_shared_with_rust():
    # verify_proof.rs wc_bridge_tests::identity_encoding_matches_the_python_twin
    assert wc.enrollment_identity(ROOT, MANIFEST, P12, {1: [16]}).hex() == \
        "323f385b048c4c26c0e2f30804117a3a2be715a7b9a47e1bd973d1d59be41697"
    assert wc.enrollment_identity(ROOT, MANIFEST, P12,
                                  {4: [100, 7], 6: [52]}).hex() == \
        "b85f6e2f8e4b5b3c07e34708982c5c0feaa1cbd4526fb45fce3d615b398bd238"


def test_identity_binds_every_field():
    base = wc.enrollment_identity(ROOT, MANIFEST, P12, {1: [16]})
    other_root = bytes([1]) + ROOT[1:]
    assert base != wc.enrollment_identity(other_root, MANIFEST, P12, {1: [16]})
    assert base != wc.enrollment_identity(ROOT, bytes([4]) * 32, P12, {1: [16]})
    # the same K_w split elsewhere: the reviewed forgery's geometry
    moved = wc.WcParams(B=10, lam=6, N_w=64, q_w=40)
    assert base != wc.enrollment_identity(ROOT, MANIFEST, moved, {1: [16]})
    assert base != wc.enrollment_identity(
        ROOT, MANIFEST, wc.WcParams(B=12, lam=4, N_w=128, q_w=40), {1: [16]})
    # the same rows in two claims, or in another order, are other layouts
    assert base != wc.enrollment_identity(ROOT, MANIFEST, P12, {1: [8, 8]})
    two = wc.enrollment_identity(ROOT, MANIFEST, P12, {1: [4, 12]})
    assert two != wc.enrollment_identity(ROOT, MANIFEST, P12, {1: [12, 4]})
    # q_w is a per-proof opening count, not part of the enrollment
    assert base == wc.enrollment_identity(
        ROOT, MANIFEST, wc.WcParams(B=12, lam=4, N_w=64, q_w=48), {1: [16]})


def test_claim_layout_follows_the_claim_map_order():
    cmap = [(3, 8, 0, 20), (5, 4, 0, 6), (9, 8, 20, 12)]
    assert wc.claim_layout(cmap) == {8: [20, 12], 4: [6]}


def test_identity_check_reads_the_verifiers_layout():
    trusted = wc.enrollment_identity(ROOT, MANIFEST, P12, {1: [16]})
    ok, _ = wc._identity_ok(ROOT, MANIFEST, {1: (2, 1)}, P12, trusted, {1: [16]})
    assert ok
    ok, why = wc._identity_ok(ROOT, MANIFEST, {1: (2, 1)},
                              wc.WcParams(B=10, lam=6, N_w=64, q_w=40),
                              trusted, {1: [16]})
    assert not ok and "enrollment identity" in why
    ok, why = wc._identity_ok(ROOT, MANIFEST, {1: (3, 1)}, P12, trusted, {1: [16]})
    assert not ok and "group metadata" in why
    # the width in the metadata is the key's (the bridge equation reads it)
    ok, why = wc._identity_ok(ROOT, MANIFEST, {1: (2, 2)}, P12, trusted, {1: [16]})
    assert not ok and "group metadata" in why
    ok, why = wc._identity_ok(ROOT, MANIFEST, {1: (2, 1), 2: (1, 2)}, P12,
                              trusted, {1: [16]})
    assert not ok and "widths" in why


def test_one_manifest_digest_for_both_builds():
    assert wc.manifest_digest_of(MANIFEST) == MANIFEST
    assert wc.manifest_digest_of(b"model|x") == wc.blake3.blake3(b"model|x").digest()
