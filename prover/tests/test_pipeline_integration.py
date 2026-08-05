"""The S4 pipeline, end to end: enroll -> admission -> prove -> verify.

Each piece has its own tests; this one checks they compose the way the runbook
says, on a tape that carries a routed MoE matmul with persistent expert shards:

  enrollment writes a commitment handle and a root;
  the admission report binds to THAT root, THAT statement and THAT layout;
  the proof references the commitment and carries the same statement digest;
  the Rust verifier accepts it only when handed both policy values.

And the two ways the composition can silently go wrong:
  a report measured on a different layout must refuse;
  a proof against a different enrolled root must not verify under the policy.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import admission
import core
import claims as _C       # noqa: F401
import packets as _PK     # noqa: F401
import protocol as pr
from tape import Tape
from rescale_claim import rescale
from routed_projected import routed_projected_matmul
from proof_dump import dump_proof
from _rust_verify import _verify_proof_bin

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
T, K, J, E = 3, 4, 4, 4
S, WIDTH = 1 << 4, 16


def _u64(xs):
    return torch.tensor(xs, dtype=torch.int64, device="cuda").to(torch.uint64)


def _build(tokens=T):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    x = tape.commit("X", _u64([1 + i for i in range(tokens * K)]), (tokens, K))
    routes = [0] * (tokens * E)
    for t in range(tokens):
        routes[t * E + t % E] = 1
    m = tape.commit("M", _u64(routes), (tokens, E))
    w = [tape.commit(f"W{e}", _u64([(e + 1 + i) % 29 for i in range(K * J)]),
                     (K, J), persistent=True) for e in range(E)]
    raw = routed_projected_matmul(tape, x, m, w, T=tokens, K=K, J=J, E=E)
    rescale(tape, raw, s_in=S * S, s_out=S, output_width=WIDTH)
    return tape


def _report_for(tape, wc, stmt, at_cap=True):
    return {
        "source_digest": admission.source_digest(),
        "model_root": wc.root.hex(),
        "statement_digest": stmt.hex(),
        "machine": admission.machine_fingerprint(),
        "row_manifest": admission.row_manifest(tape, CFG),
        "runs": 30,
        "bound_kind": "p99_upper",
        "weights": "real_gguf",
        "stages": {k: (v if at_cap else v / 2) for k, v in
                   admission.STAGE_CAPS.items()},
    }


def _enroll(tape, path):
    wc = core.WeightCommitment.from_tape(tape, CFG)
    wc.save(path)
    return core.WeightCommitment.load(path)      # load re-checks the topology


def _verify(path, root_w_hex, stmt_hex):
    r = subprocess.run([_verify_proof_bin(), path, root_w_hex, stmt_hex],
                       capture_output=True, text=True)
    return "rust_verify: ACCEPT" in r.stdout, (r.stdout + r.stderr).strip()


def test_enroll_admit_prove_verify():
    tmp = tempfile.mkdtemp()
    wc_path = os.path.join(tmp, "model.wcommit")
    proof_path = os.path.join(tmp, "proof.json")
    rep_path = os.path.join(tmp, "admission.json")
    try:
        # 1. enrollment — one commitment, one root, kept as policy
        wc = _enroll(_build(), wc_path)
        trusted_root = wc.root

        # 2. the proof tape, and the statement it will carry
        tape = _build()
        claims_bytes, manifest, stmt = admission.prepare(tape, CFG)

        # 3. admission, bound to this root / statement / layout
        json.dump(_report_for(tape, wc, stmt), open(rep_path, "w"))
        admission.check(admission.load_report(rep_path), cfg=CFG,
                        model_root=trusted_root, statement_digest=stmt,
                        manifest=manifest)

        # 4. the proof itself, referencing the enrolled model
        proof = tape.prove(weight_commitment=wc, claims_bytes=claims_bytes)
        assert proof.root_w == trusted_root
        assert proof.statement_digest == stmt, (
            "the statement the gate admitted is not the one the proof carries")
        dump_proof(proof_path, None, None, proof, None, None)

        # 5. the verifier, with both policy values supplied externally
        acc, msg = _verify(proof_path, trusted_root.hex(), stmt.hex())
        assert acc, f"pipeline proof: expected ACCEPT ({msg})"
        print(f"    enroll -> admit -> prove -> verify: ACCEPT "
              f"(root {trusted_root.hex()[:12]}…, stmt {stmt.hex()[:12]}…)")
    finally:
        for p in (wc_path, proof_path, rep_path):
            if os.path.exists(p):
                os.unlink(p)
        os.rmdir(tmp)


def test_report_from_another_layout_refused():
    """A report measured on a shorter context describes different work."""
    tmp = tempfile.mkdtemp()
    wc_path = os.path.join(tmp, "model.wcommit")
    try:
        wc = _enroll(_build(), wc_path)
        tape = _build()
        _cb, manifest, stmt = admission.prepare(tape, CFG)
        other = _build(tokens=4 * T)                    # different geometry
        report = _report_for(other, wc, stmt)
        try:
            admission.check(report, cfg=CFG, model_root=wc.root,
                            statement_digest=stmt, manifest=manifest)
            raise AssertionError("gate admitted a report for another layout")
        except SystemExit as e:
            assert "row manifest mismatch" in str(e), e
        print("    report measured on another layout: refused")
    finally:
        os.unlink(wc_path)
        os.rmdir(tmp)


def test_proof_against_another_model_fails_policy():
    """The claim cannot see swapped weights — the enrolled root is what does."""
    tmp = tempfile.mkdtemp()
    wc_path = os.path.join(tmp, "model.wcommit")
    proof_path = os.path.join(tmp, "proof.json")
    try:
        wc = _enroll(_build(), wc_path)
        tape = _build()
        proof = tape.prove(weight_commitment=wc)
        dump_proof(proof_path, None, None, proof, None, None)
        acc, _ = _verify(proof_path, ("11" * 32), proof.statement_digest.hex())
        assert not acc, "a proof under an untrusted model root must not verify"
        acc, _ = _verify(proof_path, proof.root_w.hex(),
                         proof.statement_digest.hex())
        assert acc, "the same proof under the right root must verify"
        print("    wrong enrolled root: REJECT; right root: ACCEPT")
    finally:
        for p in (wc_path, proof_path):
            if os.path.exists(p):
                os.unlink(p)
        os.rmdir(tmp)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== pipeline: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
