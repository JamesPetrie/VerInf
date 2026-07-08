"""
Toy-scale regression suite for the pipeline (claims + framework).

Imports the claim implementations from claims.py and exercises the
end-to-end prove/verify path on small examples + targeted tamper tests.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # tests/ — for test_prover
sys.path.insert(0, str(HERE.parent))     # pipeline/ — core, claims, tape, …

import torch

from core import P, Variable, Table
from test_prover import prove
from _rust_verify import rust_verify as verify
from claims import (
    # Claim types
    AddClaim, HadamardClaim, MatmulClaim, EmbeddingLookupClaim,
    PairedTlookupClaim, RangeWordClaim, RoPEClaim,
    SoftmaxClaim, WordExtractionClaim,
    # Configs
    RoPEConfig, SoftmaxConfig,
    # Shared constants / helpers
    CFG, ELL, M, K, N1, N2,
    matmul_claim, matmul_field,
    _rope_cos_sin, _softmax_exp_tables,
)



# ===========================================================================
# Tests.
# ===========================================================================

SEED_M2 = b"m2-chained-matmul-test"


def _build_m2_claims_and_inputs(seed: int, C1_override=None, C2_override=None):
    A  = Variable("A",  length=M*K)
    B  = Variable("B",  length=K*N1)
    C1 = Variable("C1", length=M*N1)
    D  = Variable("D",  length=N1*N2)
    C2 = Variable("C2", length=M*N2)

    mm1 = matmul_claim("mm1", A, B, C1, m=M, k=K,  n=N1)
    mm2 = matmul_claim("mm2", C1, D, C2, m=M, k=N1, n=N2)

    rng = random.Random(seed)
    A_vals = [rng.randrange(P) for _ in range(M*K)]
    B_vals = [rng.randrange(P) for _ in range(K*N1)]
    D_vals = [rng.randrange(P) for _ in range(N1*N2)]
    C1_vals = list(C1_override) if C1_override is not None else matmul_field(A_vals, B_vals, M, K, N1)
    C2_vals = list(C2_override) if C2_override is not None else matmul_field(C1_vals, D_vals, M, N1, N2)

    inputs = {A: A_vals, B: B_vals, C1: C1_vals, D: D_vals, C2: C2_vals}
    return [mm1, mm2], inputs


def test_honest_chained() -> None:
    print("test_honest_chained")
    claims, inputs = _build_m2_claims_and_inputs(seed=123)
    proof = prove(claims, inputs, seed=SEED_M2, cfg=CFG)
    accepted, msg = verify(claims, proof, seed=SEED_M2, cfg=CFG)
    assert accepted, f"honest chained matmul should accept: {msg}"


def test_wrong_C2() -> None:
    print("test_wrong_C2")
    _, inputs = _build_m2_claims_and_inputs(seed=456)
    C2_var = next(v for v in inputs if v.name == "C2")
    C2_wrong = list(inputs[C2_var])
    C2_wrong[0] = (C2_wrong[0] + 1) % P
    claims, inputs2 = _build_m2_claims_and_inputs(seed=456, C2_override=C2_wrong)
    proof = prove(claims, inputs2, seed=SEED_M2, cfg=CFG)
    accepted, msg = verify(claims, proof, seed=SEED_M2, cfg=CFG)
    assert not accepted, "verifier accepted a wrong C2"
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


def test_wrong_C1() -> None:
    print("test_wrong_C1")
    _, inputs = _build_m2_claims_and_inputs(seed=789)
    C1_var = next(v for v in inputs if v.name == "C1")
    C1_wrong = list(inputs[C1_var])
    C1_wrong[0] = (C1_wrong[0] + 1) % P
    claims, inputs2 = _build_m2_claims_and_inputs(seed=789, C1_override=C1_wrong)
    proof = prove(claims, inputs2, seed=SEED_M2, cfg=CFG)
    accepted, msg = verify(claims, proof, seed=SEED_M2, cfg=CFG)
    assert not accepted, "verifier accepted a wrong C1"
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


def _tamper_uint64_slot(tensor, idx, new_value: int):
    """Workaround: torch chokes on uint64 > 2**63 both reading via .item()
    and writing via `tensor[idx] = python_int`. Bridge through numpy."""
    import numpy as _np
    arr = _np.array(new_value % P, dtype=_np.uint64)
    tensor[idx] = torch.from_numpy(arr).to(tensor.device)


def test_tampered_p1_column() -> None:
    print("test_tampered_p1_column")
    claims, inputs = _build_m2_claims_and_inputs(seed=101)
    proof = prove(claims, inputs, seed=SEED_M2, cfg=CFG)
    j = next(iter(proof.opened_p1))
    cur = proof.opened_p1[j][0:1].cpu().tolist()[0]
    _tamper_uint64_slot(proof.opened_p1[j], 0, cur + 1)
    accepted, msg = verify(claims, proof, seed=SEED_M2, cfg=CFG)
    assert not accepted
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


def test_tampered_p2_column() -> None:
    print("test_tampered_p2_column")
    claims, inputs = _build_m2_claims_and_inputs(seed=202)
    proof = prove(claims, inputs, seed=SEED_M2, cfg=CFG)
    j = next(iter(proof.opened_p2))
    cur = proof.opened_p2[j][0:1].cpu().tolist()[0]
    _tamper_uint64_slot(proof.opened_p2[j], 0, cur + 1)
    accepted, msg = verify(claims, proof, seed=SEED_M2, cfg=CFG)
    assert not accepted
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


def test_tampered_q_irs() -> None:
    print("test_tampered_q_irs")
    claims, inputs = _build_m2_claims_and_inputs(seed=303)
    proof = prove(claims, inputs, seed=SEED_M2, cfg=CFG)
    cur = proof.q_irs[0:1].cpu().tolist()[0]
    _tamper_uint64_slot(proof.q_irs, 0, cur + 1)
    accepted, msg = verify(claims, proof, seed=SEED_M2, cfg=CFG)
    assert not accepted
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


# ---- AddClaim / HadamardClaim regression ----


def _build_add(seed_val: int):
    a = Variable("a_add", length=4)
    b = Variable("b_add", length=4)
    c = Variable("c_add", length=4)
    rng = random.Random(seed_val)
    a_vals = [rng.randrange(P) for _ in range(4)]
    b_vals = [rng.randrange(P) for _ in range(4)]
    c_vals = [(a_vals[i] + b_vals[i]) % P for i in range(4)]
    return [AddClaim(a=a, b=b, c=c, length=4)], {a: a_vals, b: b_vals, c: c_vals}


def test_honest_add():
    print("test_honest_add")
    claims, inputs = _build_add(11)
    proof = prove(claims, inputs, seed=b"add-h", cfg=CFG)
    acc, msg = verify(claims, proof, seed=b"add-h", cfg=CFG)
    assert acc, f"honest add should ACCEPT: {msg}"


def _build_embedding(seed_val: int):
    """Small EmbeddingLookup test fixture. vocab=4, d=4, SEQ=2.
    x[i, j] = E[token_ids[i], j] for token_ids = [0, 2] (deterministic)."""
    vocab, d, SEQ = 4, 4, 2
    token_ids = [0, 2]
    rng = random.Random(seed_val)
    E_vals = [rng.randrange(P) for _ in range(vocab * d)]
    x_vals = [E_vals[token_ids[i] * d + j] for i in range(SEQ) for j in range(d)]
    x = Variable("emb_x", length=SEQ * d)
    E = Variable("emb_E", length=vocab * d)
    return ([EmbeddingLookupClaim(x=x, E=E, token_ids=token_ids, d=d)],
            {x: x_vals, E: E_vals})


def test_honest_embedding():
    print("test_honest_embedding")
    claims, inputs = _build_embedding(21)
    proof = prove(claims, inputs, seed=b"emb-h", cfg=CFG)
    acc, msg = verify(claims, proof, seed=b"emb-h", cfg=CFG)
    assert acc, f"honest embedding should ACCEPT: {msg}"


def test_wrong_add():
    print("test_wrong_add")
    claims, inputs = _build_add(12)
    c_var = next(v for v in inputs if v.name == "c_add")
    inputs[c_var] = list(inputs[c_var]); inputs[c_var][0] = (inputs[c_var][0] + 1) % P
    proof = prove(claims, inputs, seed=b"add-w", cfg=CFG)
    acc, msg = verify(claims, proof, seed=b"add-w", cfg=CFG)
    assert not acc, "wrong add should REJECT"
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


def _build_hadamard(seed_val: int):
    a = Variable("a_h", length=4)
    b = Variable("b_h", length=4)
    c = Variable("c_h", length=4)
    rng = random.Random(seed_val)
    a_vals = [rng.randrange(P) for _ in range(4)]
    b_vals = [rng.randrange(P) for _ in range(4)]
    c_vals = [(a_vals[i] * b_vals[i]) % P for i in range(4)]
    return [HadamardClaim(a=a, b=b, c=c, length=4)], {a: a_vals, b: b_vals, c: c_vals}


def test_honest_hadamard():
    print("test_honest_hadamard")
    claims, inputs = _build_hadamard(21)
    proof = prove(claims, inputs, seed=b"had-h", cfg=CFG)
    acc, msg = verify(claims, proof, seed=b"had-h", cfg=CFG)
    assert acc, f"honest hadamard should ACCEPT: {msg}"


def test_wrong_hadamard():
    print("test_wrong_hadamard")
    claims, inputs = _build_hadamard(22)
    c_var = next(v for v in inputs if v.name == "c_h")
    inputs[c_var] = list(inputs[c_var]); inputs[c_var][0] = (inputs[c_var][0] + 1) % P
    proof = prove(claims, inputs, seed=b"had-w", cfg=CFG)
    acc, msg = verify(claims, proof, seed=b"had-w", cfg=CFG)
    assert not acc, "wrong hadamard should REJECT"
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


# ---- RangeWordClaim / TableSettlement regression (shared-mult LogUp) ----


def _build_range_word_single():
    """One Tape with one Table and one range_word over 8 values."""
    # Inline a tiny Tape stand-in: build claims + inputs by hand so we
    # don't need to import Tape (avoids the demo dependency).
    cfg = CFG
    T_data = list(range(8))     # T = {0..7}
    x_vals = [3, 0, 7, 3, 5, 1, 2, 6]
    return cfg, T_data, x_vals


def _run_range_word(cfg, T_data, list_of_xs, tamper=None, seed=b"rw"):
    """Helper: build Variables, populate inputs (with multiplicities computed
    via lookup_multiplicities), assemble claim list including TableSettlement."""
    T_LEN = len(T_data)
    mult_var = Variable("rw_mult", length=T_LEN, phase=1)
    w_var    = Variable("rw_w",    length=T_LEN, phase=2)
    T_t = torch.tensor(T_data, dtype=torch.uint64, device="cuda")
    table = Table(name="rw", T=T_t, mult_var=mult_var, w_var=w_var)

    mult_data = torch.zeros(T_LEN, dtype=torch.uint64, device="cuda")
    inputs = {mult_var: mult_data}
    claims = []
    for k, xs in enumerate(list_of_xs):
        x_var = Variable(f"x{k}", length=len(xs), phase=1)
        z_var = Variable(f"x{k}_z", length=len(xs), phase=2)
        applied = list(tamper) if (tamper is not None and k == 0) else list(xs)
        inputs[x_var] = applied
        from cuda_primitives import lookup_multiplicities_into
        lookup_multiplicities_into(torch.tensor(xs, dtype=torch.uint64, device="cuda"),
                                    T_t, mult_data)
        table.z_vars.append(z_var)
        claims.append(RangeWordClaim(x=x_var, z=z_var, table=table, length=len(xs)))
    # Op-claims only; prove + the Rust verifier each synthesize the table
    # settlement (settlement-last) — the canonical order both agree on.
    proof = prove(claims, inputs, seed=seed, cfg=cfg)
    return verify(claims, proof, seed=seed, cfg=cfg)


def test_honest_range_word():
    print("test_honest_range_word")
    cfg, T_data, xs = _build_range_word_single()
    acc, msg = _run_range_word(cfg, T_data, [xs], seed=b"rw-h")
    assert acc, f"honest range_word should ACCEPT: {msg}"


def test_out_of_range_word():
    print("test_out_of_range_word")
    cfg, T_data, xs = _build_range_word_single()
    bad = list(xs); bad[2] = 99   # not in T={0..7}
    acc, msg = _run_range_word(cfg, T_data, [xs], tamper=bad, seed=b"rw-w")
    assert not acc, "out-of-range value should REJECT"
    # Tampered z value or unsatisfied sum identity surfaces somewhere in LogUp:
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


def test_shared_range_table():
    """Two range_word calls against the same Table should ACCEPT —
    multiplicities accumulate via lookup_multiplicities_into."""
    print("test_shared_range_table")
    cfg = CFG
    T_data = list(range(8))
    xs_a = [3, 0, 7, 3, 5, 1, 2, 6]
    xs_b = [0, 1, 2, 3, 7, 7, 6, 5]
    acc, msg = _run_range_word(cfg, T_data, [xs_a, xs_b], seed=b"rw-s")
    assert acc, f"shared range_word should ACCEPT: {msg}"


# ---- WordExtractionClaim regression ----


def _build_word_extract(B: int, N: int, x_vals: List[int], word_tamper=None):
    """Build claims + inputs for one WordExtractionClaim + N RangeWordClaim
    (against a shared B-bit Table) + TableSettlement. word_tamper, if set,
    overrides the first word's values."""
    from cuda_primitives import lookup_multiplicities_into
    cfg = CFG
    T_LEN = 1 << B
    L = len(x_vals)

    T = torch.tensor(list(range(T_LEN)), dtype=torch.uint64, device="cuda")
    mult_var = Variable("we_mult", length=T_LEN, phase=1)
    w_var    = Variable("we_w",    length=T_LEN, phase=2)
    table = Table(name="we", T=T, mult_var=mult_var, w_var=w_var)

    mult_data = torch.zeros(T_LEN, dtype=torch.uint64, device="cuda")
    inputs = {mult_var: mult_data}

    x_var = Variable("x", length=L, phase=1)
    inputs[x_var] = x_vals

    mask = (1 << B) - 1
    word_vals_list = [[(v >> (n * B)) & mask for v in x_vals] for n in range(N)]
    if word_tamper is not None:
        word_vals_list[0] = list(word_tamper)

    word_vars, z_vars, range_claims = [], [], []
    for n in range(N):
        wn = Variable(f"w{n}", length=L, phase=1)
        zn = Variable(f"w{n}_z", length=L, phase=2)
        word_vars.append(wn)
        z_vars.append(zn)
        inputs[wn] = word_vals_list[n]
        lookup_multiplicities_into(
            torch.tensor(word_vals_list[n], dtype=torch.uint64, device="cuda"),
            T, mult_data)
        table.z_vars.append(zn)
        range_claims.append(RangeWordClaim(x=wn, z=zn, table=table, length=L))

    we = WordExtractionClaim(x=x_var, words=word_vars,
                              coeffs=[(1 << (n * B)) % P for n in range(N)], length=L)
    full = [we] + range_claims          # settlement auto-synthesized (last)
    return cfg, full, inputs


def test_honest_word_extract():
    print("test_honest_word_extract")
    cfg, claims, inputs = _build_word_extract(B=3, N=2,
        x_vals=[42, 7, 63, 1, 33, 17, 0, 60])
    proof = prove(claims, inputs, seed=b"we-h", cfg=cfg)
    acc, msg = verify(claims, proof, seed=b"we-h", cfg=cfg)
    assert acc, f"honest word_extract should ACCEPT: {msg}"


def test_wrong_word_extract():
    print("test_wrong_word_extract")
    x_vals = [42, 7, 63, 1, 33, 17, 0, 60]
    # Tamper: change low word of x[0] to a wrong value (still in [0, 8)).
    bad_w0 = [v & 0b111 for v in x_vals]
    bad_w0[0] = (bad_w0[0] + 1) % 8     # corrupt one slot
    cfg, claims, inputs = _build_word_extract(B=3, N=2, x_vals=x_vals,
                                              word_tamper=bad_w0)
    proof = prove(claims, inputs, seed=b"we-w", cfg=cfg)
    acc, msg = verify(claims, proof, seed=b"we-w", cfg=cfg)
    assert not acc, "tampered word_extract should REJECT"
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


# ---- PairedTlookupClaim regression ----


def _build_paired_tlookup(T_data, T_Y_data, x_vals, y_tamper=None):
    """Build claims+inputs for one PairedTlookupClaim + TableSettlement
    against a paired (T, T_Y) table."""
    from cuda_primitives import lookup_multiplicities_into
    cfg = CFG
    T_LEN, L = len(T_data), len(x_vals)
    T = torch.tensor(T_data, dtype=torch.uint64, device="cuda")
    T_Y = torch.tensor(T_Y_data, dtype=torch.uint64, device="cuda")
    mult_var = Variable("pt_mult", length=T_LEN, phase=1)
    w_var    = Variable("pt_w",    length=T_LEN, phase=2)
    table = Table(name="pt", T=T, T_Y=T_Y, mult_var=mult_var, w_var=w_var)
    mult_data = torch.zeros(T_LEN, dtype=torch.uint64, device="cuda")
    inputs = {mult_var: mult_data}

    x_var = Variable("x_pt", length=L, phase=1)
    y_var = Variable("y_pt", length=L, phase=1)
    u_var = Variable("u_pt", length=L, phase=2)
    z_var = Variable("z_pt", length=L, phase=2)
    inputs[x_var] = x_vals
    inputs[y_var] = [T_Y_data[v] for v in x_vals] if y_tamper is None else list(y_tamper)
    lookup_multiplicities_into(torch.tensor(x_vals, dtype=torch.uint64, device="cuda"),
                                 T, mult_data)
    table.z_vars.append(z_var)
    pt = PairedTlookupClaim(x=x_var, y=y_var, u=u_var, z=z_var, table=table, length=L)
    full = [pt]                         # settlement auto-synthesized (last)
    return cfg, full, inputs


def test_honest_paired_tlookup():
    print("test_honest_paired_tlookup")
    T = list(range(8))
    T_Y = [(t * t) % 8 for t in T]                 # placeholder f(x) = x² mod 8
    x_vals = [3, 0, 7, 3, 5, 1, 2, 6]
    cfg, claims, inputs = _build_paired_tlookup(T, T_Y, x_vals)
    proof = prove(claims, inputs, seed=b"pt-h", cfg=cfg)
    acc, msg = verify(claims, proof, seed=b"pt-h", cfg=cfg)
    assert acc, f"honest paired_tlookup should ACCEPT: {msg}"


def _build_rope(seed_val: int):
    """Small RoPE fixture: SEQ=2, d_h=4, H=1. Computes x_rot from x using
    the same cos/sin tables _rope_cos_sin uses, so the rotation relation
    holds bit-for-bit (the soundness anchor)."""
    SEQ, d_h, H, s_x = 2, 4, 1, 4
    cfg_rope = RoPEConfig(SEQ=SEQ, d_h=d_h, s_x=s_x, heads=H)
    half = d_h // 2
    rng = random.Random(seed_val)
    L_total = SEQ * H * d_h
    x_vals = [rng.randrange(P) for _ in range(L_total)]
    c_l, s_l = _rope_cos_sin(cfg_rope)
    x_rot_vals = [0] * L_total
    for seq in range(SEQ):
        for h in range(H):
            for k in range(half):
                idx_lo = seq * H * d_h + h * d_h + k
                idx_hi = idx_lo + half
                ci = seq * half + k
                c, s = c_l[ci], s_l[ci]
                x_rot_vals[idx_lo] = (c * x_vals[idx_lo]
                                     + (P - s * x_vals[idx_hi] % P) % P) % P
                x_rot_vals[idx_hi] = (s * x_vals[idx_lo]
                                     + c * x_vals[idx_hi]) % P
    x     = Variable("rope_x",    length=L_total)
    x_rot = Variable("rope_xrot", length=L_total)
    return ([RoPEClaim(x=x, x_rot=x_rot, config=cfg_rope)],
            {x: x_vals, x_rot: x_rot_vals})


def test_honest_rope():
    print("test_honest_rope")
    claims, inputs = _build_rope(41)
    proof = prove(claims, inputs, seed=b"rope-h", cfg=CFG)
    acc, msg = verify(claims, proof, seed=b"rope-h", cfg=CFG)
    assert acc, f"honest rope should ACCEPT: {msg}"


def _build_softmax_basic():
    """Tiny basic softmax (no saturate, no causal). B=1, M=2.

    With s_c=s_y=8, Z_max=8, δ=1, and x=[0,0], the bracket pins c2=5:
      z = [5, 5];  y_A = [T_A[5], T_A[5]] = [4, 4];  s1 = 8 = s_y    ✓
      y_B = [T_B[5], T_B[5]] = [5, 5];  s2 = 10 ≥ s_y + 1            ✓
    """
    from cuda_primitives import lookup_multiplicities_into
    cfg = CFG
    B, M = 1, 2
    L = B * M
    s_x, s_c, s_y, delta, Z_max, aux_w = 8, 8, 8, 1, 8, 16
    sc = SoftmaxConfig(B=B, M=M, s_x=s_x, s_c=s_c, s_y=s_y,
                        delta=delta, Z_max=Z_max, aux_chunk_width=aux_w)

    T_A_data, T_B_data = _softmax_exp_tables(sc)
    Z_tab = Z_max
    # Three tables: exp_A and exp_B (paired), range_aux (range).
    T_idx = torch.tensor(list(range(Z_tab)), dtype=torch.uint64, device="cuda")
    T_A_t = torch.tensor(T_A_data, dtype=torch.uint64, device="cuda")
    T_B_t = torch.tensor(T_B_data, dtype=torch.uint64, device="cuda")
    expA_mult = Variable("sm_eA_mult", length=Z_tab, phase=1)
    expA_w    = Variable("sm_eA_w",    length=Z_tab, phase=2)
    expB_mult = Variable("sm_eB_mult", length=Z_tab, phase=1)
    expB_w    = Variable("sm_eB_w",    length=Z_tab, phase=2)
    tA = Table(name="sm_eA", T=T_idx, T_Y=T_A_t, mult_var=expA_mult, w_var=expA_w)
    tB = Table(name="sm_eB", T=T_idx, T_Y=T_B_t, mult_var=expB_mult, w_var=expB_w)
    T_aux_len = 1 << aux_w
    T_aux = torch.tensor(list(range(T_aux_len)), dtype=torch.uint64, device="cuda")
    aux_mult = Variable("sm_aux_mult", length=T_aux_len, phase=1)
    aux_w_v  = Variable("sm_aux_w",    length=T_aux_len, phase=2)
    tAux = Table(name="sm_aux", T=T_aux, mult_var=aux_mult, w_var=aux_w_v)

    # Witness values (computed honestly for c2=5).
    x_vals  = [0, 0]
    c2_val  = 5
    z_vals  = [c2_val - x for x in x_vals]                     # [5, 5]
    # int() casts: newer torch refuses lists of uint64 scalar tensors in
    # torch.tensor(...), so keep these as plain ints from the start.
    yA_vals = [int(T_A_data[z]) for z in z_vals]               # [4, 4]
    yB_vals = [int(T_B_data[z]) for z in z_vals]               # [5, 5]
    s1_val  = sum(yA_vals) % P
    s2_val  = sum(yB_vals) % P
    r_lo_val = (s_y - s1_val) % P
    r_hi_val = (s2_val - (s_y + 1)) % P
    c2_shift_val = (c2_val + (1 << (aux_w - 1))) % P

    x_v          = Variable("sm_x",          length=L,    phase=1)
    yA_v         = Variable("sm_y_A",        length=L,    phase=1)
    c2_v         = Variable("sm_c2",         length=B,    phase=1)
    z_v          = Variable("sm_z",          length=L,    phase=1)
    yB_v         = Variable("sm_y_B",        length=L,    phase=1)
    s1_v         = Variable("sm_s1",         length=B,    phase=1)
    s2_v         = Variable("sm_s2",         length=B,    phase=1)
    r_lo_v       = Variable("sm_r_lo",       length=B,    phase=1)
    r_hi_v       = Variable("sm_r_hi",       length=B,    phase=1)
    c2_shift_v   = Variable("sm_c2_shifted", length=B,    phase=1)
    pt_u_A_v     = Variable("sm_pt_u_A",     length=L,    phase=2)
    pt_z_A_v     = Variable("sm_pt_z_A",     length=L,    phase=2)
    pt_u_B_v     = Variable("sm_pt_u_B",     length=L,    phase=2)
    pt_z_B_v     = Variable("sm_pt_z_B",     length=L,    phase=2)
    z_c2_v       = Variable("sm_z_c2",       length=B,    phase=2)
    z_r_lo_v     = Variable("sm_z_r_lo",     length=B,    phase=2)
    z_r_hi_v     = Variable("sm_z_r_hi",     length=B,    phase=2)

    # Multiplicities into each table.
    mA_data = torch.zeros(Z_tab, dtype=torch.uint64, device="cuda")
    mB_data = torch.zeros(Z_tab, dtype=torch.uint64, device="cuda")
    z_t = torch.tensor(z_vals, dtype=torch.uint64, device="cuda")
    lookup_multiplicities_into(z_t, T_idx, mA_data)
    lookup_multiplicities_into(z_t, T_idx, mB_data)
    aux_data = torch.zeros(T_aux_len, dtype=torch.uint64, device="cuda")
    aux_keys = torch.tensor([c2_shift_val, r_lo_val, r_hi_val],
                              dtype=torch.uint64, device="cuda")
    lookup_multiplicities_into(aux_keys, T_aux, aux_data)

    # Wire z into each table's z_vars so the sum identity covers it.
    tA.z_vars.append(pt_z_A_v)
    tB.z_vars.append(pt_z_B_v)
    tAux.z_vars.extend([z_c2_v, z_r_lo_v, z_r_hi_v])

    claim = SoftmaxClaim(
        x=x_v, y_A=yA_v, config=sc, length=L,
        c2=c2_v, z=z_v, y_B=yB_v,
        s1=s1_v, s2=s2_v, r_lo=r_lo_v, r_hi=r_hi_v,
        pt_u_A=pt_u_A_v, pt_z_A=pt_z_A_v, pt_u_B=pt_u_B_v, pt_z_B=pt_z_B_v,
        z_c2=z_c2_v, z_r_lo=z_r_lo_v, z_r_hi=z_r_hi_v, c2_shifted=c2_shift_v,
        exp_A=tA, exp_B=tB, range_aux=tAux,
    )
    full = [claim]                      # settlements (tA, tB, tAux) auto-synthesized
    inputs = {
        x_v: x_vals, yA_v: yA_vals,
        c2_v: [c2_val], z_v: z_vals, yB_v: yB_vals,
        s1_v: [s1_val], s2_v: [s2_val], r_lo_v: [r_lo_val], r_hi_v: [r_hi_val],
        c2_shift_v: [c2_shift_val],
        expA_mult: mA_data, expB_mult: mB_data, aux_mult: aux_data,
    }
    return cfg, full, inputs


def test_honest_softmax_basic():
    print("test_honest_softmax_basic")
    cfg, claims, inputs = _build_softmax_basic()
    proof = prove(claims, inputs, seed=b"sm-h", cfg=cfg)
    acc, msg = verify(claims, proof, seed=b"sm-h", cfg=cfg)
    assert acc, f"honest softmax basic should ACCEPT: {msg}"


def test_wrong_paired_tlookup():
    print("test_wrong_paired_tlookup")
    T = list(range(8))
    T_Y = [(t * t) % 8 for t in T]
    x_vals = [3, 0, 7, 3, 5, 1, 2, 6]
    bad_y = [T_Y[v] for v in x_vals]
    bad_y[0] = (bad_y[0] + 1) % 8                    # wrong pair at slot 0
    cfg, claims, inputs = _build_paired_tlookup(T, T_Y, x_vals, y_tamper=bad_y)
    proof = prove(claims, inputs, seed=b"pt-w", cfg=cfg)
    acc, msg = verify(claims, proof, seed=b"pt-w", cfg=cfg)
    assert not acc, "tampered paired_tlookup should REJECT"
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


def test_word_extract_shift_and_coeffs():
    """Exercise the silu-shape decomposition: x + shift = 1·a_0 + b·a_1 + (b·T)·a_high
    with custom coeffs [1, b, b·T] instead of the default 2^(n·B)."""
    print("test_word_extract_shift_and_coeffs")
    cfg = CFG; ell = cfg.ELL
    x_max, b, T_LEN = 4, 2, 4
    bT = b * T_LEN
    x_vals = [-2, -1, 0, 1, 2, 3, -4, 3]      # signed values, all in [-x_max, x_max)
    x_goldi = [(v % P) for v in x_vals]
    L = len(x_vals)

    a0  = [((v + x_max) % bT) %  b     for v in x_vals]
    a1  = [((v + x_max) % bT) // b     for v in x_vals]
    ah  = [ (v + x_max) // bT          for v in x_vals]   # all 0 since in-range

    x_var  = Variable("x_we", length=L)
    a0_var = Variable("a0",   length=L)
    a1_var = Variable("a1",   length=L)
    ah_var = Variable("ah",   length=L)

    claim = WordExtractionClaim(
        x=x_var, words=[a0_var, a1_var, ah_var],
        coeffs=[1, b, bT], shift=x_max, length=L,
    )
    inputs = {x_var: x_goldi, a0_var: a0, a1_var: a1, ah_var: ah}
    proof = prove([claim], inputs, seed=b"we-shift", cfg=cfg)
    acc, msg = verify([claim], proof, seed=b"we-shift", cfg=cfg)
    assert acc, f"honest decomposition w/ shift+coeffs should ACCEPT: {msg}"

    # Tamper a_0[3] (still in range but wrong value).
    a0_bad = list(a0); a0_bad[3] = (a0_bad[3] + 1) % b
    inputs[a0_var] = a0_bad
    proof = prove([claim], inputs, seed=b"we-shift-w", cfg=cfg)
    acc, msg = verify([claim], proof, seed=b"we-shift-w", cfg=cfg)
    assert not acc, "tampered a_0 should REJECT"
    assert "rust_verify: REJECT" in msg, f"got: {msg}"


ALL_TESTS = [
    test_honest_chained,
    test_wrong_C2,
    test_wrong_C1,
    test_tampered_p1_column,
    test_tampered_p2_column,
    test_tampered_q_irs,
    test_honest_add,
    test_wrong_add,
    test_honest_hadamard,
    test_wrong_hadamard,
    test_honest_range_word,
    test_out_of_range_word,
    test_shared_range_table,
    test_honest_word_extract,
    test_wrong_word_extract,
    test_honest_paired_tlookup,
    test_wrong_paired_tlookup,
    test_word_extract_shift_and_coeffs,
]


if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
    print(f"\n{len(ALL_TESTS)} tests passed.")
