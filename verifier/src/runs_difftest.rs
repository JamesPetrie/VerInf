//! Phase 1.2 gate (linear-fold-unification.md): `for_runs ≡ emit` per expander
//! kind, compared as full (col, cid, coef) triple sets per window — not sums —
//! over adversarial window geometries (width 1, widths that straddle every run
//! and segment boundary, partial tails, whole span). `emit` is the untouched
//! per-term oracle, line-for-line with the Python generators.

use super::{Expander, Run};
use crate::field::P;
use std::sync::Arc;

fn emit_triples(e: &Expander, flat_lo: usize, n: usize) -> Vec<(usize, usize, u64)> {
    let mut v = Vec::new();
    e.emit(flat_lo, n, &mut |col, cid, coef| v.push((col, cid, coef)));
    v.sort_unstable();
    v
}

fn run_triples(e: &Expander, flat_lo: usize, n: usize) -> Vec<(usize, usize, u64)> {
    let mut v = Vec::new();
    e.for_runs(flat_lo, n, &mut |r| match r {
        Run::Repeat { slot_lo, len, cid, coef } => {
            for s in 0..len { v.push((slot_lo + s, cid, coef.at(s))); }
        }
        Run::OneToOne { slot_lo, len, cid_lo, cid_step, coef } => {
            for s in 0..len { v.push((slot_lo + s, cid_lo + s * cid_step, coef.at(s))); }
        }
        Run::Fan { slot, cid_lo, len, coef } => {
            for t in 0..len { v.push((slot, cid_lo + t, coef)); }
        }
    });
    v.sort_unstable();
    v
}

/// Slide windows of several awkward widths across the whole span; every window
/// must yield identical triple sets from both paths.
fn check(e: &Expander, length: usize) {
    for w in [1usize, 3, 5, 7, 8, 11, 64, length.max(1)] {
        let mut lo = 0;
        while lo < length {
            let n = w.min(length - lo);
            assert_eq!(emit_triples(e, lo, n), run_triples(e, lo, n),
                       "window (flat_lo={lo}, n_slots={n}) at width {w}");
            lo += n;
        }
    }
}

/// Deterministic pseudo-random field values (LCG) — no rand dependency.
fn vals(n: usize, seed: u64) -> Vec<u64> {
    let mut x = seed | 1;
    (0..n).map(|_| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        x % P
    }).collect()
}

#[test]
fn weighted() {
    check(&Expander::Weighted { cid_base: 11, coefs: vals(53, 2) }, 53);
}

#[test]
fn rowsum() {
    check(&Expander::Rowsum { cid_base: 5, stride: 7, coef_vec: Arc::new(vals(7, 3)) }, 53);
}

#[test]
fn rowsum_const() {
    check(&Expander::RowsumConst { cid_base: 5, stride: 7, coef: 9 }, 53);
    check(&Expander::RowsumConst { cid_base: 5, stride: 1, coef: 9 }, 53); // ≡ Identity
    check(&Expander::RowsumConst { cid_base: 5, stride: 53, coef: 9 }, 53); // one cid
}

#[test]
fn stride_o2m_as_transpose() {   // ex-StrideO2m: cols=1, fan=stride
    check(&Expander::TransposeO2m { cid_base: 3, rows: 1, cols: 1, fan: 5, coef: 4 }, 23);
}

#[test]
fn transpose_o2m() {
    check(&Expander::TransposeO2m { cid_base: 3, rows: 4, cols: 3, fan: 5, coef: 4 }, 12);
    check(&Expander::TransposeO2m { cid_base: 3, rows: 4, cols: 3, fan: 1, coef: 4 }, 12);
}

#[test]
fn causal_id() {
    // seq=4, h=2, m=4 → b ∈ [0,8), length 32; masked tails j > i_qry skipped.
    check(&Expander::CausalId { cid_base: 20, m: 4, h: 2, coef: 6 }, 32);
}

#[test]
fn causal_masked_id() {
    // the complement of causal_id on the same geometry: unmasked heads j ≤ i_qry skipped
    check(&Expander::CausalMaskedId { cid_base: 20, m: 4, h: 2, coef: 6 }, 32);
    check(&Expander::CausalMaskedId { cid_base: 5, m: 3, h: 1, coef: 2 }, 9);
}

#[test]
fn causal_masked_id_is_the_complement_of_causal_id() {
    // seq = m = 4, h = 2: every cell is in exactly one of the two families, and
    // the masked ranks are a permutation of [0, L_m) with L_m = h·m(m−1)/2.
    let (m, h, len) = (4usize, 2usize, 32usize);
    let masked = emit_triples(&Expander::CausalMaskedId { cid_base: 0, m, h, coef: 1 }, 0, len);
    let unmasked = emit_triples(&Expander::CausalId { cid_base: 0, m, h, coef: 1 }, 0, len);
    assert_eq!(masked.len() + unmasked.len(), len);
    let mut cols: Vec<usize> = masked.iter().chain(unmasked.iter()).map(|t| t.0).collect();
    cols.sort_unstable(); cols.dedup();
    assert_eq!(cols.len(), len);
    let mut cids: Vec<usize> = masked.iter().map(|t| t.1).collect();
    cids.sort_unstable();
    assert_eq!(cids, (0..h * m * (m - 1) / 2).collect::<Vec<_>>());
    for &(col, _, _) in &masked {
        let (b, j) = (col / m, col % m);
        assert!(j > b / h, "cell {col} is unmasked");
    }
}
#[test]
fn causal_c2() { check(&Expander::CausalC2 { cid_base: 20, h: 2, coef: 6 }, 8); }

#[test]
fn embed() {
    // tid 3 hits twice (positions 0 and 2) → overlapping slot runs, distinct cids.
    check(&Expander::Embed { cid_base: 10, d: 4, token_ids: vec![3, 1, 3, 0, 4],
                             vocab_lo: 2, rows_per_w: 3 }, 12);
}

#[test]
fn freivalds_b_nontranspose() {
    check(&Expander::FreivaldsB { base: 40, k: 6, n: 4, h: 2, kk: 3,
                                  transpose_b: false, neg_rho: Arc::new(vals(8, 4)) }, 24);
}

#[test]
fn freivalds_b_transpose() {
    check(&Expander::FreivaldsB { base: 40, k: 6, n: 4, h: 2, kk: 3,
                                  transpose_b: true, neg_rho: Arc::new(vals(8, 5)) }, 24);
}

#[test]
fn freivalds_c() {
    check(&Expander::FreivaldsC { base: 40, m: 3, n: 4, h: 2,
                                  lam: Arc::new(vals(6, 7)), rho: Arc::new(vals(8, 8)) }, 24);
}

#[test]
fn rope_xrot() {
    // seq=3, h=2, d_h=6 → length 36; cid_step=2 segments per half.
    check(&Expander::RopeXrot { base: 60, h: 2, d_h: 6 }, 36);
}

#[test]
fn rope_x() {
    check(&Expander::RopeX { base: 60, h: 2, d_h: 6,
                             cos: Arc::new(vals(9, 9)), sin: Arc::new(vals(9, 10)) }, 36);
}

#[test]
fn block_stride() {
    // a head's first part (w1 = 5 of W = 8), its per-head second part
    // (w2 = 3, offset 5), and a shared second part fanned out to H = 4 heads
    check(&Expander::BlockStride { cid_base: 7, inner: 5, outer: 8, fan: 1, fan_stride: 0,
                                   coef: 1 }, 2 * 4 * 5);
    check(&Expander::BlockStride { cid_base: 12, inner: 3, outer: 8, fan: 1, fan_stride: 0,
                                   coef: 1 }, 2 * 4 * 3);
    check(&Expander::BlockStride { cid_base: 12, inner: 3, outer: 32, fan: 4, fan_stride: 8,
                                   coef: 1 }, 2 * 3);
}

