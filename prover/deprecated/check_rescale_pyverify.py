"""Decisive check: does the PROTOCOL.PY verifier (verify.py _checks, which the
Rust verifier mirrors bit-exactly) accept the prover's rescale proof, or reject
it like the Rust verifier does?

If core.verify (prover's own verify) ACCEPTs but verify.py _checks REJECTs, then
the Rust REJECT is FAITHFUL to protocol.py — the divergence is prover-vs-protocol
on the rescale gadget, not a Rust bug."""
import sys
import core, claims as C, packets as PK, protocol as pr, verify as vf  # noqa
import test_rescale as TR


def _ints(t):
    return [int(v) for v in t.cpu().tolist()]


for name in ["build_hadamard", "build_matmul", "build_rmsnorm"]:
    tape, ctype = getattr(TR, name)()
    claims, inputs, cfg = tape.claims, tape.inputs, tape.cfg
    seed = b"chk-rescale"
    proof = core.prove(claims, inputs, cfg, seed=seed)
    acc_core, msg = core.verify(claims, proof, seed, cfg)
    s_op, s_comb, s_col = pr.round_seeds(seed)
    Q = pr.random_columns(s_col, cfg)
    o1 = {j: _ints(proof.opened_p1[j]) for j in Q}
    o2 = {j: _ints(proof.opened_p2[j]) for j in Q}
    acc_vf = vf._checks(claims, cfg, s_op, s_comb, s_col,
                        proof.root_p1, proof.root_p2,
                        _ints(proof.q_irs), _ints(proof.q_lin), _ints(proof.p_0),
                        o1, o2, proof.paths_p1, proof.paths_p2)
    print(f"{name}: core.verify={'ACCEPT' if acc_core else 'REJECT'}  "
          f"verify.py _checks={'ACCEPT' if acc_vf else 'REJECT'}", file=sys.stderr)
