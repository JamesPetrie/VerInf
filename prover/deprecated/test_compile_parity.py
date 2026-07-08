"""Parity oracle: protocol.py's independent compile MUST equal the real
prover's compile (claims.py COMPILE_FNS → packets → EXPANDERS), exactly.

For each claim list we expand BOTH sides to a canonical form and assert equal:
  - linear:  A[(row, slot, cid)] = Σ coef           (mod P)
  - quad:    sorted (x_row, y_row, z_row, n, a, b)
  - rhs:     {cid: b}

This is THE check that lets mentees change the prover (claims.py) and have the
behavioral test catch any divergence from the trusted verifier (protocol.py),
without reading their code. Runs on the Spark (needs torch for the real side).

Run:  ~/venv-hf/bin/python test_compile_parity.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))  # deprecated/ (python_verifier_compile)
import torch
import core
import claims as C          # noqa: F401 — registers COMPILE_FNS/SAMPLE_FNS/AUX_FNS
import packets as PK        # noqa: F401 — registers EXPANDERS
from core import EXPANDERS
import protocol as pr
import python_verifier_compile as pvc

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


def V(name, length, phase=1):
    return core.Variable(name, length, phase)


# ---- canonical expansion of the REAL prover's per-row packets ----
def canon_real(per_row, ell):
    A = {}
    for r, pkts in enumerate(per_row):
        by_kind = {}
        for p in pkts:
            by_kind.setdefault(type(p), ([], []))
            by_kind[type(p)][0].append(p)
            by_kind[type(p)][1].append(0)          # single row → local_row 0
        for kind, (ps, lrows) in by_kind.items():
            t, c, v = EXPANDERS[kind](ps, lrows, r, ell)   # chunk_lo=r → target=slot
            t = t.tolist(); c = c.tolist(); v = v.tolist()
            for slot, cid, coef in zip(t, c, v):
                key = (r, int(slot), int(cid))
                A[key] = (A.get(key, 0) + int(coef)) % pr.P
    return {k: x for k, x in A.items() if x}


# ---- canonical expansion of MY (protocol.py) Constraints ----
def canon_mine(cons, cfg):
    A = {}
    for r, exps in enumerate(cons.rows):
        for fn, params in exps:
            for slot, cid, coef in fn(params, cfg):
                key = (r, int(slot), int(cid))
                A[key] = (A.get(key, 0) + int(coef)) % pr.P
    return {k: x for k, x in A.items() if x}


# ORDER-SENSITIVE: the quadratic combiner s_t = challenge(s_comb, t, "quad") is
# indexed by quad POSITION, so the two sides must agree on quad order, not just
# the set. (Sorting here once hid a real bug: protocol emitted rescale range
# quads before the op's own quads while the prover emitted them last.) Also assert
# a/b are constant per quad — protocol.py represents them as SCALARS, so a per-slot
# a_values/b_values would verify-mismatch despite matching at index 0.
def canon_quads_real(quads):
    out = []
    for q in quads:
        av, bv = [int(v) for v in q.a_values], [int(v) for v in q.b_values]
        assert all(v == av[0] for v in av) and all(v == bv[0] for v in bv), \
            f"quad {q.name}: per-slot a/b — protocol.py uses scalar a/b and would mismatch"
        out.append((q.x_row, q.y_row, q.z_row, q.n, av[0], bv[0]))
    return out

def canon_quads_mine(quads):
    return [(q.x_row, q.y_row, q.z_row, q.n, int(q.a), int(q.b)) for q in quads]


def canon_rhs_real(b_chunks):
    d = {}
    for base, chunk in b_chunks:
        for k, val in enumerate(chunk.tolist()):
            if int(val) % pr.P:
                d[base + k] = int(val) % pr.P
    return d

def canon_rhs_mine(rhs):
    return {cid: b % pr.P for cid, b in rhs if b % pr.P}


def compare(tag, claim_list, cfg=CFG):
    # Both sides derive op challenges from the SAME round-1 seed s_op by index.
    # Prover: core._compile_all(settled, s_op). Verifier: compile_claims(ops, s_op)
    # which settles tables itself. Same seed → identical challenges → identical cids.
    s_op = b"parity-s_op"
    cl = core._with_synthesized_settlements(list(claim_list))
    _, _, _, _, m_total = core._layout(cl, cfg)
    per_row, quads, chs, b_chunks, n_lin = core._compile_all(cl, s_op, cfg, m_total)

    cons = pvc.compile_claims(list(claim_list), cfg, s_op)

    rA, mA = canon_real(per_row, cfg.ELL), canon_mine(cons, cfg)
    rQ, mQ = canon_quads_real(quads), canon_quads_mine(cons.quadratic)
    rR, mR = canon_rhs_real(b_chunks), canon_rhs_mine(cons.rhs)

    ok = (rA == mA) and (rQ == mQ) and (rR == mR) and (m_total == cons.m_total)
    print(f"[{'OK ' if ok else 'XX '}] {tag}: "
          f"lin {len(rA)} (mine {len(mA)}), quad {len(rQ)} (mine {len(mQ)}), "
          f"rhs {len(rR)} (mine {len(mR)}), m_total {m_total}/{cons.m_total}")
    if not ok:
        if rA != mA:
            only_r = {k: rA[k] for k in rA if rA.get(k) != mA.get(k)}
            only_m = {k: mA[k] for k in mA if rA.get(k) != mA.get(k)}
            print(f"    LIN MISMATCH: real-only/diff {list(only_r.items())[:6]}")
            print(f"                  mine-only/diff {list(only_m.items())[:6]}")
        if rQ != mQ:
            print(f"    QUAD real {rQ[:4]}\n         mine {mQ[:4]}")
        if rR != mR:
            print(f"    RHS  real {dict(list(rR.items())[:6])}\n         mine {dict(list(mR.items())[:6])}")
    return ok


def cases():
    """Yield (tag, claim_list, cfg) for every op. Shared by the Python-only
    parity test (main) and the cross-language dumper (dump_compile_parity.py),
    so both exercise the IDENTICAL claim constructions."""
    # AddClaim: c = a + b
    a, b, c = V("a", 6), V("b", 6), V("c", 6)
    yield ("add", [C.AddClaim(a, b, c, 6)], CFG)

    # AddClaim reveal pin: a == public_rhs (public-RHS path, no b/c)
    ar = V("ar", 1)
    yield ("add_reveal", [C.AddClaim(a=ar, b=None, c=None, length=1, public_rhs=12345)], CFG)

    # AddClaim multi-row (length 10 > ELL 8)
    a, b, c = V("a", 10), V("b", 10), V("c", 10)
    yield ("add_multirow", [C.AddClaim(a, b, c, 10)], CFG)

    # Hadamard (no rescale): c = a ⊙ b
    a, b, c = V("a", 6), V("b", 6), V("c", 6)
    yield ("hadamard", [C.HadamardClaim(a, b, c, 6)], CFG)

    # Embedding lookup: d | ELL (d=4, ELL=8), SEQ=3, vocab=5
    d = 4; SEQ = 3; vocab = 5
    x = V("x", SEQ * d); E = V("E", vocab * d)
    yield ("embedding", [C.EmbeddingLookupClaim(x, E, [0, 2, 4], d)], CFG)

    # Embedding single-row table with d ∤ ELL (the relaxed-assert path:
    # hadamard_broadcast's vocab=1 gain at Maverick d=5120, ELL=8192)
    d3 = 3; SEQ3 = 2
    x3 = V("x3", SEQ3 * d3); E3 = V("E3", 1 * d3)
    yield ("embedding_singlerow", [C.EmbeddingLookupClaim(x3, E3, [0, 0], d3)], CFG)

    # Concat: dst = a ‖ b ‖ c with uneven segment lengths spanning rows
    ca, cb, cc = V("ca", 5), V("cb", 9), V("cc", 3)
    cd_ = V("cd", 17)
    yield ("concat", [C.ConcatClaim(srcs=[ca, cb, cc], dst=cd_)], CFG)

    # RoPE (no rescale): SEQ=2, d_h=4, heads=1 → L=8
    rc = C.RoPEConfig(SEQ=2, d_h=4, s_x=4096, heads=1)
    x = V("x", 8); xr = V("xr", 8)
    yield ("rope", [C.RoPEClaim(x, xr, rc)], CFG)

    # RoPE multi-head: SEQ=2, d_h=4, heads=2 → L=16 (spans rows at ELL=8)
    rc2 = C.RoPEConfig(SEQ=2, d_h=4, s_x=4096, heads=2)
    x = V("x", 16); xr = V("xr", 16)
    yield ("rope_multihead", [C.RoPEClaim(x, xr, rc2)], CFG)

    # RoPE with a non-default frequency base (Maverick θ=500000) — locks the
    # base through serialization to BOTH verifier twins (the Rust handler used
    # to hardcode 10000.0, and the serializer used to drop float config fields).
    rc3 = C.RoPEConfig(SEQ=2, d_h=4, s_x=4096, heads=1, base=500000.0)
    x = V("x_t5", 8); xr = V("xr_t5", 8)
    yield ("rope_theta500k", [C.RoPEClaim(x, xr, rc3)], CFG)

    # All settlement-free claims together (cid numbering across claims)
    a, b, c = V("a", 6), V("b", 6), V("c", 6)
    a2, b2, c2 = V("a2", 6), V("b2", 6), V("c2", 6)
    yield ("combo_add_hadamard",
           [C.AddClaim(a, b, c, 6), C.HadamardClaim(a2, b2, c2, 6)], CFG)

    # SILU — built via the Tape (it allocates the ~20 aux vars + 5 LogUp tables
    # the way the prover does). This also re-exercises the table-settlement loop
    # (silu folds 4 range checks + 1 paired lookup inline), so _settle_table gets
    # parity coverage again. SILU_TOY config; the tape's own cfg/claims are fed
    # to compare(). No input rescale (s_in unset → rescale_bits = 0).
    from tape import Tape, SILU_TOY
    tape = Tape(CFG, silu_config=SILU_TOY)
    Lsil = 8
    # Mixed-sign inputs spanning the lookup + (positive) saturation range so the
    # decomposition is well-defined; magnitudes small enough for SILU_TOY.
    xs = torch.tensor([0, 1, 2, 3, 5, 7, 6, 4], dtype=torch.int64, device="cuda").to(torch.uint64)
    x = tape.commit("silu_x", xs, (Lsil,))
    tape.silu(x)
    yield ("silu", tape.claims, tape.cfg)

    # RMSNORM — via the Tape. B=2 tokens × d=4, L=8. No rescale (s_in/s_out unset).
    # Exercises rho (the per-claim challenge) in F4/F5, expand_rowsum (stride=d),
    # the rsqrt-bracket quads, and the slack range table (settled by the loop).
    tape_r = Tape(CFG)
    xr = torch.tensor([3, 1, 4, 2, 5, 2, 6, 1], dtype=torch.int64, device="cuda").to(torch.uint64)
    xv = tape_r.commit("rms_x", xr, (8,))
    tape_r.rmsnorm(xv, d=4, s=4, eps_int=1, slack_n_chunks=1)
    yield ("rmsnorm", tape_r.claims, tape_r.cfg)

    # SOFTMAX (non-causal, non-saturating) — reuse the known-good hand-built
    # fixture from test_claims (B=1, M=2, Z_max=8). It returns [3 settlements,
    # claim]; we pass ONLY the SoftmaxClaim as the op (compare() synthesizes the
    # settlements itself). Exercises expand_stride_o2m (z=c2−x), the bracket, the
    # 2 paired-exp lookups + 3 range checks, and 3 tables through _settle_table.
    from test_claims import _build_softmax_basic
    _cfg_sm, sm_full, _sm_inputs = _build_softmax_basic()
    sm_claim = [c for c in sm_full if type(c).__name__ == "SoftmaxClaim"]
    yield ("softmax", sm_claim, _cfg_sm)

    # SOFTMAX causal + saturating — built via the Tape (computes valid LSE +
    # causal-mask + saturation witnesses). Attention shape: B=H·SEQ, M=SEQ.
    # H=1, SEQ=2 ⇒ B=2, M=2. Exercises expand_causal_id / expand_causal_c2 (the
    # filtered z-decomp, L_u constraints) and the saturation mux quads. Parity is
    # witness-independent, so the compile is checked regardless of the witness.
    smx = torch.tensor([0, 0, 0, 0], dtype=torch.int64, device="cuda").to(torch.uint64)
    for tag, kw in [("softmax_causal",     dict(causal=True, heads=1)),
                    ("softmax_causal_sat", dict(causal=True, heads=1, saturate=True))]:
        t = Tape(CFG)
        xv = t.commit(tag + "_x", smx, (4,))
        t.softmax(xv, M=2, s_x=8, s_c=8, s_y=8, Z_max=8, **kw)
        ops = [c for c in t.claims if type(c).__name__ == "SoftmaxClaim"]
        yield (tag, ops, t.cfg)

    # MATMUL, single-head — C = A·B, m=2,k=4,n=2. The compile is witness-
    # independent, so the claim alone suffices (no A/B/C values needed). Exercises
    # the 3 Freivalds expanders (B/A/C) + the p-side stride aggregation + the u·y=p
    # quad. heads=1 ⇒ H=1, K=k. (Multi-head + transpose_b are the next sub-steps.)
    mm = C.matmul_claim("mm", V("mm_A", 2 * 4), V("mm_B", 4 * 2), V("mm_C", 2 * 2),
                        m=2, k=4, n=2)
    yield ("matmul_1head", [mm], CFG)

    # MATMUL multi-head — H=2, head_dim=2 ⇒ k=4, m=2, n=2. Exercises the h-major
    # index decode in all three Freivalds expanders (head = i_k//K ≠ 0) and the
    # per-head LF3 cids [base+2k, base+2k+H). Same expander code as single-head.
    mmh = C.matmul_claim("mmh", V("mmh_A", 2 * 4), V("mmh_B", 4 * 2), V("mmh_C", 2 * 2 * 2),
                         m=2, k=4, n=2, heads=2, head_dim=2)
    yield ("matmul_multihead", [mmh], CFG)

    # MATMUL transpose_b (single-head) — the C = A·B^T decode branch (j=f//k,
    # i_k=f%k in expand_freivalds_b). Q·K^T attention shape.
    mmt = C.matmul_claim("mmt", V("mmt_A", 2 * 4), V("mmt_B", 2 * 4), V("mmt_C", 2 * 2),
                         m=2, k=4, n=2, transpose_b=True)
    yield ("matmul_transpose_b", [mmt], CFG)

    # RESCALE gadget (_emit_rescale) — output rescale on matmul/hadamard/rmsnorm,
    # input rescale on softmax. Used by every value-producing op in the real layer
    # but absent from the cases above. Known-valid small builders from test_rescale.
    import test_rescale as TR
    for tag, builder in [("hadamard_rescale", TR.build_hadamard),
                         ("matmul_rescale", TR.build_matmul),
                         ("softmax_rescale", TR.build_softmax),
                         ("rmsnorm_rescale", TR.build_rmsnorm)]:
        tape, ctype = builder()
        ops = [c for c in tape.claims if type(c).__name__ == ctype]
        yield (tag, ops, tape.cfg)

    # ROUTING + FREIVALDS COMBINE — via the Tape (route_top1 composes the gap
    # word-extraction + range checks as standalone WordExtraction/RangeWord
    # claims, so this case also covers those two handlers + the TransposeO2m
    # expander + table settlement for the shared range table). The combine is the
    # §B.4 projected seam (challenge-bearing: ρ from s_op by claim index, through
    # both verifier twins).
    from tape import Tape as _Tape
    from routing_claim import route_top1, freivalds_combine
    T_rt, E_rt, F_rt = 2, 4, 3
    r_data = torch.tensor([40, 20, 30, 10, 15, 45, 25, 5],
                           dtype=torch.int64, device="cuda").to(torch.uint64)
    tape_fc = _Tape(CFG)
    r2 = tape_fc.commit("fc_r", r_data, (T_rt, E_rt))
    m2, _rc2, _g2 = route_top1(tape_fc, r2, T=T_rt, E=E_rt,
                                B_logit=6, word_bits=4)
    xs2 = [tape_fc.commit(f"fc_x{e}",
                           torch.tensor([10 * (e + 1) + t for t in range(T_rt)
                                          for _ in range(F_rt)],
                                         dtype=torch.int64, device="cuda").to(torch.uint64),
                           (T_rt, F_rt)) for e in range(E_rt)]
    freivalds_combine(tape_fc, m2, xs2, T=T_rt, E=E_rt, F=F_rt)
    yield ("freivalds_combine", tape_fc.claims, tape_fc.cfg)


def main():
    results = [compare(tag, claim_list, cfg) for tag, claim_list, cfg in cases()]
    print(f"\n=== {sum(results)}/{len(results)} parity checks passed ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
