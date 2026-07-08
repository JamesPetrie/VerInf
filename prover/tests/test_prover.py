"""Test harness — flat-witness Ligero prover for the unit + negative tests.

This is NOT the production path. The production prover is core.prove_streaming
(tape-based, streaming, scales to Llama-2-7B at SEQ=1000). This flat
prove(claims, inputs, ...) entry takes an arbitrary witness dict directly, which
is exactly what the per-op and tamper/negative tests need (e.g. test_claims
feeds a deliberately-wrong witness and asserts the verifier REJECTs — something
a tape can't express, since a tape always computes a *valid* witness).

Moved out of core.py so core.py is production-only. Imports engine internals
from core.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # pipeline/ on path

import torch
import protocol as pr
from typing import List
from core import (
    LigeroConfig, NUM_BLINDING_ROWS, challenge_vec, Proof, AUX_FNS,
    encode_messages, merkle_path, compute_p_0_streaming,
    QIrsAccumulator, QLinAccumulator,
    _master_seed_to_cuda, _PhaseLogger, _with_synthesized_settlements, _layout,
    _make_blinding_messages, _encode_2k_blinding_rows, _BLIND_ROW_IRS,
    _make_merkle_acc, _stream_phase, _finalize_merkle_artifact, _LazyResolvingDict,
    _compile_with_chs, _sample_chs, _mix_blinding_into_tests,
)


def _sample_test_challenges(claims: List, cfg: LigeroConfig, seed: bytes):
    """Derive (ch0, ch1, ch2) from a base seed via the shared blake3 PRF —
    test-mode convenience mirroring verify()'s per-round derivation exactly. A
    real interactive protocol instead receives each round's challenges from the
    verifier over the wire (fresh coins after each commitment).

    Round order matches the old single-stream Verifier: ch0 = per-claim op
    challenges (Freivalds ρ,λ / LogUp α,β), ch1 = (r_irs, r_lin, r_quad) test
    combiners, ch2 = opened columns Q."""
    claims = _with_synthesized_settlements(claims)
    _, _, _, _, m_total, _, _, _, _ = _layout(claims, cfg)
    s_op, s_comb, s_col = pr.round_seeds(seed)
    ch0 = _sample_chs(claims, s_op)
    _, quads, _, n_lin = _compile_with_chs(claims, ch0, cfg, m_total)
    # All three test combiners derive from s_comb on-GPU. q_lin gets the raw
    # 32-byte seed and draws its combiner inline (gl_spmv_challenged, never
    # materialized); irs/quad are materialized on-GPU via challenge_vec.
    seed_u8 = torch.tensor(list(s_comb), dtype=torch.uint8, device="cuda")
    ch1 = (challenge_vec(seed_u8, torch.tensor(list(b"irs"), dtype=torch.uint8, device="cuda"),
                         m_total - NUM_BLINDING_ROWS),
           seed_u8,
           challenge_vec(seed_u8, torch.tensor(list(b"quad"), dtype=torch.uint8, device="cuda"),
                         len(quads)))
    ch2 = pr.random_columns(s_col, cfg)
    return ch0, ch1, ch2


def prove(claims, inputs, cfg, ch0=None, ch1=None, ch2=None,
          returnEverything=False, *, seed=None, verbose=False):
    """Ligero prover. Stateless per-round dispatch by which challenges are given.

    Real interactive protocol (caller threads challenges between calls):
        prove(claims, inputs, cfg)                   -> root_p1
        prove(claims, inputs, cfg, ch0)              -> root_p2
        prove(claims, inputs, cfg, ch0, ch1)         -> (q_irs, q_lin, p_0)
        prove(claims, inputs, cfg, ch0, ch1, ch2)    -> (opened_p1, opened_p2,
                                                         paths_p1, paths_p2)

    Test convenience — single fused call returning full Proof:
        prove(claims, inputs, cfg, seed=b"...")
        # equivalent to:
        #   ch0, ch1, ch2 = _sample_test_challenges(claims, cfg, seed)
        #   prove(claims, inputs, cfg, ch0, ch1, ch2, returnEverything=True)

    `returnEverything=True` with all three challenges given also returns a
    full Proof (q-polys, p_0 and mix_blinding computed in the same fused
    pass as columns + paths).
    """
    # TODO: replace constant master_seed with secrets.token_bytes(32) for
    # secure deployment. With a fixed master_seed, ZK blinding is predictable
    # — anyone observing the proof can derive witness values from the column
    # openings. Safe for correctness tests; NOT safe for confidential proofs.
    master_seed = b"\x42" * 32
    master_seed_t = _master_seed_to_cuda(master_seed)
    plog = _PhaseLogger("prove", verbose); plog.log("entry")

    claims = _with_synthesized_settlements(claims)
    _, p1_vars, p2_vars, m_p1_rows, m_total, _, _, _, _ = _layout(claims, cfg)

    # Test-mode shortcut: derive challenges + flip returnEverything.
    if seed is not None:
        ch0, ch1, ch2 = _sample_test_challenges(claims, cfg, seed)
        returnEverything = True

    # Pre-encoded blinding rows (phase-1 merkle prefix + mix_blinding polys).
    u_irs_msg, u_lin_msg, u_quad_msg = _make_blinding_messages(cfg, master_seed)
    u_irs_polys_K, u_irs_codes = encode_messages(
        u_irs_msg.unsqueeze(0), cfg, master_seed=master_seed_t,
        row_offset=_BLIND_ROW_IRS)
    polys_2k, codes_2k = _encode_2k_blinding_rows(
        torch.stack([u_lin_msg, u_quad_msg], dim=0), cfg)
    u_irs_poly, u_lin_poly, u_quad_poly = u_irs_polys_K[0], polys_2k[0], polys_2k[1]
    p1_prefix = torch.cat([u_irs_codes, codes_2k], dim=0)
    n_p1_total = sum(v.n_rows(cfg.ELL) for v in p1_vars) + NUM_BLINDING_ROWS
    n_p2_total = sum(v.n_rows(cfg.ELL) for v in p2_vars)

    # ---------- Round 1: phase-1 commit ----------
    if ch0 is None:
        merkle_p1 = _make_merkle_acc(cfg.N_LIG, n_p1_total)
        _stream_phase(p1_vars, inputs, cfg, master_seed=master_seed_t,
                      abs_row_offset=NUM_BLINDING_ROWS, prefix_codewords=p1_prefix,
                      merkle_acc=merkle_p1)
        plog.log("round 1 done")
        return _finalize_merkle_artifact(merkle_p1).root

    # AUX_FNS -> phase-2 witness (needed by all later rounds).
    witness = dict(inputs)
    for c, ch in zip(claims, ch0):
        witness.update(AUX_FNS[type(c)](c, _LazyResolvingDict(witness), ch))

    # ---------- Round 2: phase-2 commit ----------
    if ch1 is None:
        merkle_p2 = _make_merkle_acc(cfg.N_LIG, n_p2_total)
        _stream_phase(p2_vars, witness, cfg, master_seed=master_seed_t,
                      abs_row_offset=m_p1_rows, merkle_acc=merkle_p2)
        plog.log("round 2 done")
        return _finalize_merkle_artifact(merkle_p2).root

    per_row, quads, _, _ = _compile_with_chs(claims, ch0, cfg, m_total)

    import os as _os
    if _os.environ.get("LIGERO_CV_CHECK"):
        from analysis.cv_check import cv_check
        cv_check(claims, witness, per_row, ch0, cfg, m_total)

    r_irs_t, r_lin_seed, r_quad_t = ch1

    # ---------- Round 3: q polynomials ----------
    if ch2 is None:
        q_irs_acc = QIrsAccumulator(r_irs_t, cfg)
        q_lin_acc = QLinAccumulator(r_lin_seed, per_row, cfg)
        _stream_phase(p1_vars, inputs, cfg, master_seed=master_seed_t,
                      abs_row_offset=NUM_BLINDING_ROWS,
                      q_irs_acc=q_irs_acc, q_lin_acc=q_lin_acc)
        _stream_phase(p2_vars, witness, cfg, master_seed=master_seed_t,
                      abs_row_offset=m_p1_rows,
                      q_irs_acc=q_irs_acc, q_lin_acc=q_lin_acc)
        p_0 = compute_p_0_streaming(p1_vars, p2_vars, witness, m_p1_rows,
                                     r_quad_t, quads, cfg, master_seed_t)
        plog.log("round 3 done")
        return _mix_blinding_into_tests(q_irs_acc.finalize(), q_lin_acc.finalize(),
                                         p_0, u_irs_poly, u_lin_poly, u_quad_poly, cfg)

    # ---------- Round 4: merkle + columns (+ q-polys if returnEverything) ----------
    Q_cols_list = list(ch2)
    merkle_p1 = _make_merkle_acc(cfg.N_LIG, n_p1_total)
    merkle_p2 = _make_merkle_acc(cfg.N_LIG, n_p2_total)
    q_irs_acc = QIrsAccumulator(r_irs_t, cfg) if returnEverything else None
    q_lin_acc = QLinAccumulator(r_lin_seed, per_row, cfg) if returnEverything else None

    p1_res = _stream_phase(p1_vars, inputs, cfg, master_seed=master_seed_t,
                            abs_row_offset=NUM_BLINDING_ROWS, prefix_codewords=p1_prefix,
                            merkle_acc=merkle_p1, q_irs_acc=q_irs_acc, q_lin_acc=q_lin_acc,
                            columns_at=Q_cols_list)
    p2_res = _stream_phase(p2_vars, witness, cfg, master_seed=master_seed_t,
                            abs_row_offset=m_p1_rows,
                            merkle_acc=merkle_p2, q_irs_acc=q_irs_acc, q_lin_acc=q_lin_acc,
                            columns_at=Q_cols_list)
    art_p1 = _finalize_merkle_artifact(merkle_p1)
    art_p2 = _finalize_merkle_artifact(merkle_p2)
    paths_p1 = {j: merkle_path(art_p1.levels, j) for j in ch2}
    paths_p2 = {j: merkle_path(art_p2.levels, j) for j in ch2}

    if not returnEverything:
        plog.log("round 4 done")
        return p1_res['opened_columns'], p2_res['opened_columns'], paths_p1, paths_p2

    # returnEverything: also finalise q-polys + p_0 + assemble full Proof.
    p_0 = compute_p_0_streaming(p1_vars, p2_vars, witness, m_p1_rows,
                                 r_quad_t, quads, cfg, master_seed_t)
    q_irs, q_lin, p_0 = _mix_blinding_into_tests(
        q_irs_acc.finalize(), q_lin_acc.finalize(), p_0,
        u_irs_poly, u_lin_poly, u_quad_poly, cfg)
    plog.log("fused prove done")
    return Proof(root_p1=art_p1.root, root_p2=art_p2.root,
                 q_irs=q_irs, q_lin=q_lin, p_0=p_0,
                 opened_p1=p1_res['opened_columns'], opened_p2=p2_res['opened_columns'],
                 paths_p1=paths_p1, paths_p2=paths_p2)
