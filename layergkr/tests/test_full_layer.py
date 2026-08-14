"""Gates for the complete layer proof (full_layer.py) over real toy semantics.

Positive: the trace is self-consistent; the proof verifies with masks on and
off; the same weight tensor used by several tokens is enrolled once.

Negative, one per surface: a tampered weight root, a tampered activation root, a
lied gate witness, a lied lookup output, an edited opened column, and an edited
small projected vector are each rejected.

Run:  .venv/bin/python layergkr/tests/run_tests.py test_full_layer
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import full_layer as fl, rs, semantics as sem
from prover.protocol import P as FIELD_P

CFG = rs.Config(ELL=16, K_DEG=32, N_LIG=64, T_QUERIES=4)
TOY = sem.ToyConfig(S=4, d=8, d_ff=16, E=1, table_bits=6, scale_bits=6)
Q = 4


def _build(seed=3, toy=TOY, masks=True):
    rng = random.Random(seed)
    trace = sem.forward(toy, rng)
    enrol = fl.Enrollment(CFG)
    proof = fl.prove_full_layer(trace, CFG, enrol, Q, rng, use_masks=masks)
    return trace, enrol, proof


# ── positive ─────────────────────────────────────────────────────────────────
def test_trace_is_self_consistent():
    rng = random.Random(11)
    trace = sem.forward(TOY, rng)
    ok, why = sem.check_trace(trace)
    assert ok, why


def test_full_layer_verifies_with_masks():
    trace, enrol, proof = _build(masks=True)
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert ok, why
    assert proof.gate_proof.masked, "masks were requested but not applied"


def test_full_layer_verifies_without_masks():
    trace, enrol, proof = _build(masks=False)
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert ok, why


def test_masking_changes_the_transcript_but_not_the_verdict():
    """ZK must not change what is accepted -- only what is revealed."""
    a_trace, _, a = _build(seed=5, masks=False)
    b_trace, _, b = _build(seed=5, masks=True)
    assert a.gate_proof.round_polys != b.gate_proof.round_polys
    for p, t in ((a, a_trace), (b, b_trace)):
        ok, why = fl.verify_full_layer(CFG, p, t.gates)
        assert ok, why


def test_moe_enrolment_is_one_root_per_distinct_tensor():
    """With the routed path, the FFN weights live in MoE nodes: 3 projections
    (gate, up, down) per expert, plus the 5 dense tensors. Enrolment must be one
    root per distinct tensor -- no re-enrolment however the tokens route."""
    for E in (2, 4):
        toy = sem.ToyConfig(S=4, d=8, d_ff=16, E=E, table_bits=6, scale_bits=6)
        trace, enrol, proof = _build(seed=9, toy=toy)
        ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
        assert ok, why
        assert len(enrol.weights) == 5 + 3 * E, (
            f"E={E}: {len(enrol.weights)} enrolments, expected {5 + 3 * E}")


def test_moe_route_is_not_published_by_the_proof():
    """The point of §5: the route must not be readable off the proof. With the
    per-token form it was -- each matmul was named after its expert. Now the
    routed contraction is ONE node whose proof carries no per-token expert id."""
    toy = sem.ToyConfig(S=6, d=8, d_ff=16, E=3, table_bits=6, scale_bits=6)
    trace, enrol, proof = _build(seed=13, toy=toy)
    assert len(trace.moe) == 3, trace.counts()
    for mp in proof.moe:
        fields = vars(mp)
        assert "route" not in fields, "the proof object carries the route"
        # every expert is projected, so the opened set does not single one out
        assert len(mp.p_roots) == 3 and len(mp.w_openings) == 3
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert ok, why


def test_moe_tampered_permutation_is_rejected():
    trace, enrol, proof = _build(seed=4, toy=sem.ToyConfig(
        S=4, d=8, d_ff=16, E=2, table_bits=6, scale_bits=6))
    proof.moe[0].perm_dst = (proof.moe[0].perm_dst + 1) % FIELD_P
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert not ok and "permutation" in why, why


def test_moe_tampered_segment_sums_are_rejected():
    trace, enrol, proof = _build(seed=4, toy=sem.ToyConfig(
        S=4, d=8, d_ff=16, E=2, table_bits=6, scale_bits=6))
    proof.moe[0].a_flat[0] = (proof.moe[0].a_flat[0] + 1) % FIELD_P
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert not ok and "identity" in why, why


def test_lookups_cover_every_nonlinearity():
    rng = random.Random(2)
    trace = sem.forward(TOY, rng)
    names = {l.table.name for l in trace.lookups}
    assert {"exp", "silu", "isqrt", "recip", "range"} <= names, names


# ── negative ─────────────────────────────────────────────────────────────────
def test_wrong_weight_root_is_rejected():
    trace, enrol, proof = _build()
    k = next(iter(proof.weight_roots))
    proof.weight_roots[k] = b"\x77" * 32
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert not ok and "weight seam" in why, why


def test_wrong_activation_root_is_rejected():
    trace, enrol, proof = _build()
    proof.matmuls[0].x_root = b"\x55" * 32
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert not ok and "activation seam" in why, why


def test_lied_gate_witness_is_rejected():
    """Flip one value in a hadamard gate's output: the batched sumcheck's claim
    stops being zero."""
    trace, enrol, proof = _build()
    g = next(g for g in trace.gates if g.kind == "hadamard")
    coeff, factors = g.terms[-1]
    factors[0][0] = (factors[0][0] + 1) % FIELD_P
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert not ok and "gates" in why, why


def test_lied_lookup_output_is_rejected():
    trace, enrol, proof = _build()
    table_id, lp, q_vals, t_vals, mult, alpha = proof.lookups[0]
    bad_q = list(q_vals)
    bad_q[0] = (bad_q[0] + 1) % FIELD_P
    proof.lookups[0] = (table_id, lp, bad_q, t_vals, mult, alpha)
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert not ok and "lookup" in why, why


def test_edited_opened_column_is_rejected():
    trace, enrol, proof = _build()
    mp = proof.matmuls[0]
    mp.w_opening.p_values[0][0] = (mp.w_opening.p_values[0][0] + 1) % FIELD_P
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert not ok, "an edited opened value was accepted"


def test_edited_projected_block_is_rejected():
    """Editing the COMMITTED block breaks the re-encode binding."""
    trace, enrol, proof = _build()
    mp = proof.matmuls[0]
    mp.p_blocks[0][0] = (mp.p_blocks[0][0] + 1) % FIELD_P
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert not ok and "does not match its root" in why, why


def test_edited_flat_projection_is_rejected_by_the_contraction():
    """Editing the FLAT vector the sumcheck runs on is caught there instead --
    both halves of the seam have to agree, so tampering either one fails."""
    trace, enrol, proof = _build()
    mp = proof.matmuls[0]
    mp.p_message[0] = (mp.p_message[0] + 1) % FIELD_P
    ok, why = fl.verify_full_layer(CFG, proof, trace.gates)
    assert not ok and "contraction" in why, why


def test_semantics_check_catches_a_broken_matmul():
    """The trace checker is independent of the prover: corrupt a Y and it fires."""
    rng = random.Random(4)
    trace = sem.forward(TOY, rng)
    trace.matmuls[0].Y[0][0] = (trace.matmuls[0].Y[0][0] + 1) % FIELD_P
    ok, why = sem.check_trace(trace)
    assert not ok and "matmul" in why, why
