//! Constraint compile — the verifier's own re-derivation of the constraint
//! system from the public claim list (mirrors protocol.compile_claims + the
//! _c_* handlers + expanders). Bit-exact with Python.
//!
//! This file has two parts:
//!   1. the constraint MODEL + the row-expanders (Expander::emit), pure index
//!      math — direct line-for-line ports of the Python generators;
//!   2. the per-op handlers + compile_claims (in handlers.rs), which build the
//!      Constraints from a Claim list.
//!
//! A constraint family is a whole-variable spanning `Expander`. `emit` takes a
//! row window `(flat_lo, n_slots)` and yields that window's terms — so the
//! verifier folds per ROW (window = `[ro·ell, ro·ell+n_slots)`) without ever
//! materializing per-row expanders. The window is a parameter, not state.

use crate::field::{mul, P};
use std::sync::Arc;

/// One quadratic constraint FAMILY (the quad lift, linear-fold-unification.md):
/// for row t ∈ [0, nrows): w[x_row+t][c]·w[y_row+t][c] + a·w[z_row+t][c] = b for
/// c ∈ [0, n_at(t)). Row t's r_quad challenge position is index_base + t —
/// POSITIONAL, identical to the retired per-row Vec's flat order, so the
/// challenge pairing is unchanged. Replaces the O(W/ELL) per-row struct list
/// (a verifier-memory binder at long context) with O(#emit_quad calls).
pub struct QuadFamily {
    pub x_row: usize,
    pub y_row: usize,
    pub z_row: usize,
    pub a: u64,
    pub b: u64,
    pub length: usize,
    pub ell: usize,
    pub index_base: usize,
}

impl QuadFamily {
    pub fn nrows(&self) -> usize { (self.length + self.ell - 1) / self.ell }
    /// Constrained slots on row t (the last row may be partial).
    pub fn n_at(&self, t: usize) -> usize { self.ell.min(self.length - t * self.ell) }
}

/// One constraint family — the (kind, params) pair, mirroring Python's
/// (expand_fn, params) tuples. Vectors (coef_vec / neg_rho / lam / cos …) are
/// owned here; they are O(table)/O(challenge) sized, not O(nnz). The row window
/// is supplied to `emit`, not stored.
pub enum Expander {
    Weighted { cid_base: usize, coefs: Vec<u64> },
    Rowsum { cid_base: usize, stride: usize, coef_vec: Arc<Vec<u64>> },
    /// Rowsum with a single constant coef (all slots collapse onto one cid with
    /// the same coefficient). Parameterized — avoids materializing an O(stride)
    /// coef vector, which for 2^24 LogUp tables would be GBs.
    RowsumConst { cid_base: usize, stride: usize, coef: u64 },
    /// Transposed fan-out (MaskedCombineClaim's replicated-mask pin): source is
    /// a row-major (rows, cols) matrix; slot at flat f = t·cols + e feeds `fan`
    /// consecutive cids in transposed order:
    ///   cid = cid_base + (f % cols)·rows·fan + (f / cols)·fan + k,  k ∈ [0, fan)
    TransposeO2m { cid_base: usize, rows: usize, cols: usize, fan: usize, coef: u64 },
    CausalId { cid_base: usize, m: usize, h: usize, coef: u64 },
    CausalC2 { cid_base: usize, h: usize, coef: u64 },
    Embed { cid_base: usize, d: usize, token_ids: Vec<usize>, vocab_lo: usize, rows_per_w: usize },
    FreivaldsB { base: usize, k: usize, n: usize, h: usize, kk: usize,
                 transpose_b: bool, neg_rho: Arc<Vec<u64>> },
    FreivaldsC { base: usize, m: usize, n: usize, h: usize,
                 lam: Arc<Vec<u64>>, rho: Arc<Vec<u64>> },
    RopeXrot { base: usize, h: usize, d_h: usize },
    RopeX { base: usize, h: usize, d_h: usize, cos: Arc<Vec<u64>>, sin: Arc<Vec<u64>> },
}

/// Coefficient source within one run (linear-fold-unification.md). `at(s)` is
/// the coefficient of the run's s-th slot; all variants are exact field values
/// matching what `emit` yields term-by-term.
pub enum CoefSrc<'a> {
    Const(u64),
    /// coef[s] = v[s] (already-reduced field values)
    Slice(&'a [u64]),
    /// coef[s] = (P − v[s]) % P  (RopeX's −cos/−sin sides)
    NegSlice(&'a [u64]),
    /// coef[s] = (P − a·b[s]) % P  (FreivaldsC's −λ·ρ, λ hoisted per run)
    NegProd { a: u64, b: &'a [u64] },
}

impl CoefSrc<'_> {
    pub fn at(&self, s: usize) -> u64 {
        match self {
            CoefSrc::Const(v) => *v,
            CoefSrc::Slice(v) => v[s],
            CoefSrc::NegSlice(v) => (P - v[s]) % P,
            CoefSrc::NegProd { a, b } => (P - mul(*a, b[s])) % P,
        }
    }
}

/// One maximal homogeneous piece of a family's row window. Slot positions are
/// window-relative (like `emit`'s `col`). The three shapes carry the statically
/// known structure the fold exploits: Repeat = many slots, one cid (one hash);
/// OneToOne = slot i → cid_lo + i·cid_step (distinct cids); Fan = one slot,
/// a contiguous cid range (a challenge range-sum).
pub enum Run<'a> {
    Repeat   { slot_lo: usize, len: usize, cid: usize, coef: CoefSrc<'a> },
    OneToOne { slot_lo: usize, len: usize, cid_lo: usize, cid_step: usize, coef: CoefSrc<'a> },
    Fan      { slot: usize, cid_lo: usize, len: usize, coef: u64 },
}

/// flat index → (pair_t, e_self, coef_idx) for RoPE split-half rotation.
fn rope_decode(f: usize, h: usize, d_h: usize) -> (usize, usize, usize) {
    let half = d_h / 2;
    let seq = f / (h * d_h);
    let hh = (f / d_h) % h;
    let k = f % d_h;
    let e_self = k / half;
    let k_in_pair = k % half;
    let pair_t = seq * h * half + hh * half + k_in_pair;
    (pair_t, e_self, seq * half + k_in_pair)
}

impl Expander {
    /// Emit the terms of the row window `[flat_lo, flat_lo+n_slots)`, calling
    /// `f(col, cid, coef)` where `col` is the slot's position *within the window*
    /// (so a per-row call with `flat_lo = ro·ell` yields col-in-row directly).
    /// Line-for-line with the Python generators; the window replaces the old
    /// per-row expander fields. `emit(0, length, …)` yields the whole variable.
    pub fn emit(&self, flat_lo: usize, n_slots: usize, f: &mut impl FnMut(usize, usize, u64)) {
        match self {
            Expander::Weighted { cid_base, coefs } => {
                for s in 0..n_slots { f(s, cid_base + flat_lo + s, coefs[flat_lo + s]); }
            }
            Expander::Rowsum { cid_base, stride, coef_vec } => {
                for s in 0..n_slots {
                    let flat = flat_lo + s;
                    f(s, cid_base + flat / stride, coef_vec[flat % stride]);
                }
            }
            Expander::RowsumConst { cid_base, stride, coef } => {
                for s in 0..n_slots {
                    let flat = flat_lo + s;
                    f(s, cid_base + flat / stride, *coef);
                }
            }
            Expander::TransposeO2m { cid_base, rows, cols, fan, coef } => {
                for s in 0..n_slots {
                    let flat = flat_lo + s;
                    let cid_lo = cid_base + (flat % cols) * (rows * fan) + (flat / cols) * fan;
                    for k in 0..*fan { f(s, cid_lo + k, *coef); }
                }
            }
            Expander::CausalId { cid_base, m, h, coef } => {
                for s in 0..n_slots {
                    let flat = flat_lo + s;
                    let (b, j) = (flat / m, flat % m);
                    let (i_qry, hh) = (b / h, b % h);
                    if j <= i_qry {
                        let rank = h * i_qry * (i_qry + 1) / 2 + hh * (i_qry + 1) + j;
                        f(s, cid_base + rank, *coef);
                    }
                }
            }
            Expander::CausalC2 { cid_base, h, coef } => {
                for s in 0..n_slots {
                    let b = flat_lo + s;
                    let (i_qry, hh) = (b / h, b % h);
                    let rank_start = h * i_qry * (i_qry + 1) / 2 + hh * (i_qry + 1);
                    for j_off in 0..(i_qry + 1) { f(s, cid_base + rank_start + j_off, *coef); }
                }
            }
            // Embed scatters by token, so the window is a filter (used whole-family
            // only: emit(0, length, …) — see linear_column_test's Embed branch).
            Expander::Embed { cid_base, d, token_ids, vocab_lo, rows_per_w } => {
                let vocab_hi = vocab_lo + rows_per_w;
                for (i, &tid) in token_ids.iter().enumerate() {
                    if *vocab_lo <= tid && tid < vocab_hi {
                        let rel = tid - vocab_lo;
                        for j in 0..*d {
                            let slot = rel * d + j;
                            if flat_lo <= slot && slot < flat_lo + n_slots {
                                f(slot - flat_lo, cid_base + i * d + j, P - 1);
                            }
                        }
                    }
                }
            }
            Expander::FreivaldsB { base, k, n, h, kk, transpose_b, neg_rho } => {
                for s in 0..n_slots {
                    let ff = flat_lo + s;
                    let (j, i_k) = if *transpose_b {
                        (ff / k, ff % k)
                    } else {
                        let (j, rest) = (ff % n, ff / n);
                        (j, (rest % h) * kk + rest / h)
                    };
                    let head = i_k / kk;
                    f(s, base + i_k, neg_rho[head * n + j]);
                }
            }
            Expander::FreivaldsC { base, m, n, h, lam, rho } => {
                for s in 0..n_slots {
                    let ff = flat_lo + s;
                    let (j, rest) = (ff % n, ff / n);
                    let (hh, i_outer) = (rest % h, rest / h);
                    let coef = (P - mul(lam[hh * m + i_outer], rho[hh * n + j])) % P;
                    f(s, base + hh, coef);
                }
            }
            Expander::RopeXrot { base, h, d_h } => {
                for s in 0..n_slots {
                    let (pair_t, e_self, _) = rope_decode(flat_lo + s, *h, *d_h);
                    f(s, base + 2 * pair_t + e_self, 1);
                }
            }
            Expander::RopeX { base, h, d_h, cos, sin } => {
                for s in 0..n_slots {
                    let (pair_t, e_self, ci) = rope_decode(flat_lo + s, *h, *d_h);
                    let (c, sn) = (cos[ci], sin[ci]);
                    if e_self == 0 {
                        f(s, base + 2 * pair_t,     (P - c) % P);
                        f(s, base + 2 * pair_t + 1, (P - sn) % P);
                    } else {
                        f(s, base + 2 * pair_t,     sn);
                        f(s, base + 2 * pair_t + 1, (P - c) % P);
                    }
                }
            }
        }
    }
}

impl Expander {
    /// Yield the window's maximal homogeneous runs — same index math as `emit`,
    /// coarser granularity. `emit` is kept verbatim as the difftest oracle;
    /// `runs_difftest.rs` pins `for_runs ≡ emit` per kind over adversarial
    /// windows. Slot positions are window-relative, like `emit`'s `col`.
    pub fn for_runs(&self, flat_lo: usize, n_slots: usize, f: &mut impl FnMut(Run)) {
        match self {
            Expander::Weighted { cid_base, coefs } => f(Run::OneToOne {
                slot_lo: 0, len: n_slots, cid_lo: cid_base + flat_lo, cid_step: 1,
                coef: CoefSrc::Slice(&coefs[flat_lo..flat_lo + n_slots]) }),
            Expander::Rowsum { cid_base, stride, coef_vec } => {
                let mut s = 0usize;
                while s < n_slots {
                    let flat = flat_lo + s;
                    let q = flat / stride;
                    let s_hi = ((q + 1) * stride - flat_lo).min(n_slots);
                    let off = flat % stride;
                    f(Run::Repeat { slot_lo: s, len: s_hi - s, cid: cid_base + q,
                                    coef: CoefSrc::Slice(&coef_vec[off..off + (s_hi - s)]) });
                    s = s_hi;
                }
            }
            Expander::RowsumConst { cid_base, stride, coef } => {
                let mut s = 0usize;
                while s < n_slots {
                    let flat = flat_lo + s;
                    let q = flat / stride;
                    let s_hi = ((q + 1) * stride - flat_lo).min(n_slots);
                    f(Run::Repeat { slot_lo: s, len: s_hi - s, cid: cid_base + q,
                                    coef: CoefSrc::Const(*coef) });
                    s = s_hi;
                }
            }
            Expander::TransposeO2m { cid_base, rows, cols, fan, coef } => {
                for s in 0..n_slots {
                    let flat = flat_lo + s;
                    let cid_lo = cid_base + (flat % cols) * (rows * fan) + (flat / cols) * fan;
                    f(Run::Fan { slot: s, cid_lo, len: *fan, coef: *coef });
                }
            }
            Expander::CausalId { cid_base, m, h, coef } => {
                let mut s = 0usize;
                while s < n_slots {
                    let flat = flat_lo + s;
                    let (b, j0) = (flat / m, flat % m);
                    let (i_qry, hh) = (b / h, b % h);
                    let b_end = ((b + 1) * m - flat_lo).min(n_slots);
                    if j0 <= i_qry {
                        let len = (i_qry + 1 - j0).min(b_end - s);
                        let rank0 = h * i_qry * (i_qry + 1) / 2 + hh * (i_qry + 1) + j0;
                        f(Run::OneToOne { slot_lo: s, len, cid_lo: cid_base + rank0,
                                          cid_step: 1, coef: CoefSrc::Const(*coef) });
                    }
                    s = b_end;
                }
            }
            Expander::CausalC2 { cid_base, h, coef } => {
                for s in 0..n_slots {
                    let b = flat_lo + s;
                    let (i_qry, hh) = (b / h, b % h);
                    let rank_start = h * i_qry * (i_qry + 1) / 2 + hh * (i_qry + 1);
                    f(Run::Fan { slot: s, cid_lo: cid_base + rank_start,
                                 len: i_qry + 1, coef: *coef });
                }
            }
            Expander::Embed { cid_base, d, token_ids, vocab_lo, rows_per_w } => {
                let vocab_hi = vocab_lo + rows_per_w;
                for (i, &tid) in token_ids.iter().enumerate() {
                    if *vocab_lo <= tid && tid < vocab_hi {
                        let rel = tid - vocab_lo;
                        let lo = (rel * d).max(flat_lo);
                        let hi = (rel * d + d).min(flat_lo + n_slots);
                        if lo < hi {
                            f(Run::OneToOne { slot_lo: lo - flat_lo, len: hi - lo,
                                              cid_lo: cid_base + i * d + (lo - rel * d),
                                              cid_step: 1, coef: CoefSrc::Const(P - 1) });
                        }
                    }
                }
            }
            Expander::FreivaldsB { base, k, n, h, kk, transpose_b: false, neg_rho } => {
                let mut s = 0usize;
                let _ = k;
                while s < n_slots {
                    let ff = flat_lo + s;
                    let (j0, rest) = (ff % n, ff / n);
                    let i_k = (rest % h) * kk + rest / h;
                    let head = i_k / kk;
                    let s_hi = ((rest + 1) * n - flat_lo).min(n_slots);
                    f(Run::Repeat { slot_lo: s, len: s_hi - s, cid: base + i_k,
                        coef: CoefSrc::Slice(&neg_rho[head * n + j0..head * n + j0 + (s_hi - s)]) });
                    s = s_hi;
                }
            }
            // Transposed B and A share the strided-repeat shape: cid advances 1
            // per slot within a k-block; the coef is constant within each
            // head-sized (kk) segment.
            Expander::FreivaldsB { base, k, n, h: _, kk, transpose_b: true, neg_rho } => {
                let mut s = 0usize;
                while s < n_slots {
                    let ff = flat_lo + s;
                    let (j, ik0) = (ff / k, ff % k);
                    let head = ik0 / kk;
                    let blk_end = (ff / k + 1) * k;
                    let head_end = ff - ik0 + (head + 1) * kk;
                    let s_hi = (blk_end.min(head_end) - flat_lo).min(n_slots);
                    f(Run::OneToOne { slot_lo: s, len: s_hi - s, cid_lo: base + ik0,
                        cid_step: 1, coef: CoefSrc::Const(neg_rho[head * n + j]) });
                    s = s_hi;
                }
            }
            Expander::FreivaldsC { base, m, n, h, lam, rho } => {
                let mut s = 0usize;
                while s < n_slots {
                    let ff = flat_lo + s;
                    let (j0, rest) = (ff % n, ff / n);
                    let (hh, i_outer) = (rest % h, rest / h);
                    let s_hi = ((rest + 1) * n - flat_lo).min(n_slots);
                    f(Run::Repeat { slot_lo: s, len: s_hi - s, cid: base + hh,
                        coef: CoefSrc::NegProd { a: lam[hh * m + i_outer],
                                                 b: &rho[hh * n + j0..hh * n + j0 + (s_hi - s)] } });
                    s = s_hi;
                }
            }
            Expander::RopeXrot { base, h, d_h } => {
                let half = d_h / 2;
                let mut s = 0usize;
                while s < n_slots {
                    let ff = flat_lo + s;
                    let (pair_t, e_self, _) = rope_decode(ff, *h, *d_h);
                    let k_in_pair = (ff % d_h) % half;
                    let seg = (half - k_in_pair).min(n_slots - s);
                    f(Run::OneToOne { slot_lo: s, len: seg, cid_lo: base + 2 * pair_t + e_self,
                                      cid_step: 2, coef: CoefSrc::Const(1) });
                    s += seg;
                }
            }
            Expander::RopeX { base, h, d_h, cos, sin } => {
                let half = d_h / 2;
                let mut s = 0usize;
                while s < n_slots {
                    let ff = flat_lo + s;
                    let (pair_t, e_self, ci) = rope_decode(ff, *h, *d_h);
                    let k_in_pair = (ff % d_h) % half;
                    let seg = (half - k_in_pair).min(n_slots - s);
                    let (eq1, eq2): (CoefSrc, CoefSrc) = if e_self == 0 {
                        (CoefSrc::NegSlice(&cos[ci..ci + seg]), CoefSrc::NegSlice(&sin[ci..ci + seg]))
                    } else {
                        (CoefSrc::Slice(&sin[ci..ci + seg]), CoefSrc::NegSlice(&cos[ci..ci + seg]))
                    };
                    f(Run::OneToOne { slot_lo: s, len: seg, cid_lo: base + 2 * pair_t,
                                      cid_step: 2, coef: eq1 });
                    f(Run::OneToOne { slot_lo: s, len: seg, cid_lo: base + 2 * pair_t + 1,
                                      cid_step: 2, coef: eq2 });
                    s += seg;
                }
            }
        }
    }
}

/// One linear family — a whole-variable spanning `Expander` plus the geometry
/// (row_start, length, ell). Row `ro` is the window `[ro·ell, ro·ell+n_slots)`.
pub struct Family {
    pub row_start: usize,
    pub length: usize,
    pub ell: usize,
    pub exp: Expander,
}

impl Family {
    /// Number of witness rows this family spans.
    pub fn nrows(&self) -> usize { (self.length + self.ell - 1) / self.ell }

    /// Yield each (global_row, col_in_row, cid, coef) over the whole variable —
    /// `emit(0, length)` gives the flat slot as `col`, decoded into a global row.
    /// Difftest-only (the compile_difftest oracle); the verdict path never calls
    /// this — it windows per row via `emit`.
    pub fn emit_global(&self, f: &mut impl FnMut(usize, usize, usize, u64)) {
        let (ell, row_start) = (self.ell, self.row_start);
        self.exp.emit(0, self.length, &mut |col, cid, coef| {
            f(row_start + col / ell, col % ell, cid, coef)
        });
    }
}

/// The compiled constraint system (mirrors protocol.Constraints).
pub struct Constraints {
    pub families: Vec<Family>,
    pub rhs: Vec<(usize, usize, u64)>,   // compact runs: (start_cid, length, value)
    pub quadratic: Vec<QuadFamily>,
    /// Difftest-only (compile_difftest checks it against Python); the verdict
    /// path never reads it.
    pub m_total: usize,
}

// compile entry point lives in handlers::compile_claims(&mut ClaimSet, s_op).

// Phase 1.2 gate: for_runs ≡ emit per kind (test-only; not verdict-path TCB).
#[cfg(test)]
#[path = "runs_difftest.rs"]
mod runs_difftest;
