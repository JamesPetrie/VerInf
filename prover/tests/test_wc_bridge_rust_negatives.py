"""The WC-LCRL-STC bridge through the RUST verifier, negatives included.

The collaborator's bridge suites assert ACCEPT through the binary and run
every tampering case against the Python twin; the binary is the TCB, so
these are the cases it must refuse on its own: a tampered fold, a wrong or
missing trusted enrollment identity, a redeclared weight/mask boundary under
the same root, the production shape (a committed weight
block AND a wc section, each with its own anchor), an opening count below
the floor, the legacy seed path, and a stripped wc section. GPU: the toy
tapes prove on CUDA, as the rest of the bridge suites do."""
import base64
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import core
import protocol as pr
import wc_bridge as wc
from proof_dump import dump_proof
from routed_projected import RoutedProjectedMatmulClaim, routed_projected_matmul
from tape import Tape
from _rust_verify import _verify_proof_bin
from test_routed_projected import _u64, CFG, T, K, J, E
from test_wc_streaming_flag import _prove_with_flag, PARAMS

WRONG = "ab" * 32


def _dump(tape, proof):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    dump_proof(path, pr.claims_to_json(tape.claims, tape.cfg), None, proof, None, None)
    return path


def _run(path, root_w, stmt, wc_identity):
    r = subprocess.run([_verify_proof_bin(), path, root_w, stmt, wc_identity],
                       capture_output=True, text=True)
    return ("rust_verify: ACCEPT" in r.stdout), (r.stdout + r.stderr).strip()


def _rewrite(path, mutate):
    doc = json.load(open(path))
    mutate(doc)
    with open(path, "w") as f:
        json.dump(doc, f)


def _u64_list(v):
    """A wc-section array from the JSON, whichever encoding the dump used."""
    if isinstance(v, str):
        raw = base64.b64decode(v[len("u64le:"):])
        return [int(x) for x in np.frombuffer(raw, dtype="<u8")]
    return list(v)


def _u64_wire(vals, like):
    """The same array back, in the encoding `like` was in."""
    if isinstance(like, str):
        return "u64le:" + base64.b64encode(np.asarray(vals, dtype="<u8").tobytes()).decode("ascii")
    return list(vals)


def _policy(proof):
    return (proof.root_w.hex() if getattr(proof, "root_w", None) else "-",
            proof.statement_digest.hex(), proof.wc_bridge["identity"].hex())


def test_bridged_proof_accepts_with_its_enrollment_identity_and_rejects_without():
    tape, proof, _ = _prove_with_flag()
    path = _dump(tape, proof)
    try:
        root_w, stmt, wc_identity = _policy(proof)
        acc, msg = _run(path, root_w, stmt, wc_identity)
        assert acc, f"honest bridged proof rejected: {msg}"
        acc, msg = _run(path, root_w, stmt, WRONG)
        assert not acc and "enrollment identity" in msg, f"wrong enrollment identity accepted: {msg}"
        acc, msg = _run(path, root_w, stmt, "-")
        assert not acc, "no trusted enrollment identity: expected REJECT"
        acc, msg = _run(path, wc_identity, stmt, "-")
        assert not acc, "enrollment identity in the weight-root slot: expected REJECT"
        print("    enrollment identity: matching ACCEPT / wrong, missing, misplaced REJECT")
    finally:
        os.unlink(path)


def test_tampered_fold_rejects_in_rust():
    tape, proof, _ = _prove_with_flag()
    path = _dump(tape, proof)
    try:
        def flip(doc):
            k = next(iter(doc["wc"]["p_trace"]))
            vals = _u64_list(doc["wc"]["p_trace"][k]); vals[3] ^= 1
            doc["wc"]["p_trace"][k] = _u64_wire(vals, doc["wc"]["p_trace"][k])
        _rewrite(path, flip)
        acc, msg = _run(path, *_policy(proof))
        assert not acc and "wc bridge REJECT" in msg, f"tampered P_trace accepted: {msg}"
        print("    tampered P_trace: REJECT in Rust")
    finally:
        os.unlink(path)


def test_redeclared_block_boundary_rejects():
    """Review 2026-09-23 finding 2: B and lam come off the wire, and the root
    commits the columns, not where the weights end and the masks begin. A
    proof that moves the boundary (same K_w, same root and manifest) names
    another enrollment identity than the trusted one."""
    tape, proof, _ = _prove_with_flag()
    path = _dump(tape, proof)
    try:
        def move_boundary(doc):
            doc["wc"]["params"]["B"] -= 8
            doc["wc"]["params"]["lam"] += 8
        _rewrite(path, move_boundary)
        acc, msg = _run(path, *_policy(proof))
        assert not acc and "enrollment identity" in msg, f"moved boundary accepted: {msg}"
        print("    B-8 / lam+8 under the same root: REJECT (identity)")
    finally:
        os.unlink(path)


def test_opening_count_below_the_floor_rejects():
    """q_w comes off the wire; the verifier holds it to ceil(0.416 * t)."""
    tape, proof, _ = _prove_with_flag()
    path = _dump(tape, proof)
    try:
        def downgrade(doc):
            doc["wc"]["params"]["q_w"] = 1
            for key in ("eta", "v"):
                doc["wc"][key] = _u64_wire(_u64_list(doc["wc"][key])[:1], doc["wc"][key])
        _rewrite(path, downgrade)
        acc, msg = _run(path, *_policy(proof))
        assert not acc and "below the floor" in msg, f"q_w = 1 accepted: {msg}"
        print("    q_w = 1: REJECT (floor)")
    finally:
        os.unlink(path)


def test_legacy_seed_path_never_feeds_the_bridge():
    """A proof without a statement digest takes its seeds from the file; the
    bridge's late coins must not be derivable from a file-supplied s_bind."""
    tape, proof, _ = _prove_with_flag()
    path = _dump(tape, proof)
    try:
        def strip_digest(doc):
            doc.pop("statement_digest", None)
        _rewrite(path, strip_digest)
        root_w, _stmt, wc_identity = _policy(proof)
        acc, msg = _run(path, root_w, "-", wc_identity)
        assert not acc and "s_bind" in msg, f"bridge ran on file-supplied seeds: {msg}"
        print("    legacy seeds: bridge refused")
    finally:
        os.unlink(path)


def test_stripped_wc_section_rejects():
    tape, proof, _ = _prove_with_flag()
    path = _dump(tape, proof)
    try:
        _rewrite(path, lambda doc: doc.pop("wc"))
        root_w, stmt, _ = _policy(proof)
        acc, msg = _run(path, root_w, stmt, "-")
        assert not acc and "no wc section" in msg, f"bridged claim without wc accepted: {msg}"
        print("    stripped wc section: REJECT")
    finally:
        os.unlink(path)


def _build_production_shape():
    """A committed persistent dense weight (X D, the attention/dense case) AND
    bridged expert shards held outside the witness: the proof carries a w
    block and a wc section, two anchors."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    X = torch.arange(1, T * K + 1).reshape(T, K)
    D = (torch.arange(1, K * K + 1) % 11).reshape(K, K)
    W = torch.arange(1, E * K * J + 1).reshape(E, K * J)
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate((0, 1, 0)):
        M[t, e] = 1
    x = tape.commit("X", _u64(X.reshape(-1)), (T, K))
    d = tape.commit_lazy("D", lambda: _u64(D.reshape(-1)), (K, K), K * K, persistent=True)
    xd = tape.matmul(x, d)
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    w = [tape.external(f"W{e}", _u64(W[e]), (K, J)) for e in range(E)]
    routed_projected_matmul(tape, xd, m, w, T=T, K=K, J=J, E=E, use_bridge=True)
    enr = wc.enroll_tape(tape, b"prod-mask", b"prod-manifest", PARAMS)
    wcm = core.WeightCommitment.from_tape(tape, CFG)
    proof = tape.prove(weight_commitment=wcm, weight_enrollment=enr)
    return tape, proof


def test_production_shape_binds_both_anchors():
    """With a weight block present the old verifier checked only the block's
    root and left the enrollment unbound; now each anchor has its slot and a
    wrong or missing enrollment identity rejects beside a correct weight
    root."""
    tape, proof = _build_production_shape()
    assert getattr(proof, "root_w", None) is not None and proof.wc_bridge is not None
    path = _dump(tape, proof)
    try:
        root_w, stmt, wc_identity = _policy(proof)
        acc, msg = _run(path, root_w, stmt, wc_identity)
        assert acc, f"production-shape proof rejected: {msg}"
        acc, msg = _run(path, root_w, stmt, WRONG)
        assert not acc and "enrollment identity" in msg, f"wrong enrollment identity beside a right weight root accepted: {msg}"
        acc, msg = _run(path, root_w, stmt, "-")
        assert not acc, "missing enrollment identity beside a right weight root: expected REJECT"
        acc, msg = _run(path, WRONG, stmt, wc_identity)
        assert not acc, "wrong weight root beside a right enrollment identity: expected REJECT"
        print("    production shape: both anchors bound")
    finally:
        os.unlink(path)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== wc-bridge-rust-negatives: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
