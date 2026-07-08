//! Test-only difftests for the linear-column verifier (verify.rs), included via
//! `#[cfg(test)] #[path = "verify_difftest.rs"] mod lin_contrib_difftest;` — so it
//! is a child module of `verify` (hence `use super::*`). NOT part of the verifier
//! TCB: a reader auditing the verdict path can skip this file.
//!
//! Gates: the generic run fold (`row_contrib`) is compared against the
//! windowed-`emit` per-term oracle, with vs without the family challenge
//! preload, over random params for every expander kind. Also gates the
//! windowing itself (per-row windows tile the whole variable). (The retired
//! fast-path oracle was deleted after the full-scale seq-1000 gates.)

    use super::*;
    use crate::field::P;
    use std::sync::Arc;

    /// tiny deterministic LCG — avoids a `rand` dependency in the TCB crate.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn field(&mut self) -> u64 { self.next() % P }
        fn upto(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    }

    /// Assert the generic run fold equals the windowed-emit oracle for `exp`
    /// over `[flat_lo, flat_lo+n_slots)`, on a random Lagrange table — and,
    /// when `preload_span` is given, that the preloaded path matches too.
    fn assert_matches_oracle(exp: &Expander, flat_lo: usize, n_slots: usize,
                             ncols: usize, t: usize, rng: &mut Lcg,
                             preload_span: Option<(usize, usize)>) {
        let s_comb: &[u8] = b"difftest-seed";
        let chal = Chal::new(s_comb);
        let lag: Vec<u64> = (0..t * ncols).map(|_| rng.field()).collect();
        let pre = prefix_lagrange(&lag, ncols, t);

        let mut generic = vec![0u64; t];
        row_contrib(exp, flat_lo, n_slots, &chal, None, &lag, &pre, ncols, t, &mut generic);

        let mut oracle = vec![0u64; t];
        exp.emit(flat_lo, n_slots, &mut |col, cid, coef| {
            let sc = mul(challenge(s_comb, cid as u64, "lin"), coef);
            for qi in 0..t { oracle[qi] = add(oracle[qi], mul(sc, lag[qi * ncols + col])); }
        });

        assert_eq!(generic, oracle, "generic fold vs emit oracle");

        if let Some((base, span)) = preload_span {
            let tab = chal.preload(base, base + span);
            let mut preloaded = vec![0u64; t];
            row_contrib(exp, flat_lo, n_slots, &chal, Some((base, &tab)),
                        &lag, &pre, ncols, t, &mut preloaded);
            assert_eq!(preloaded, oracle, "preloaded fold vs emit oracle");
        }
    }

    #[test]
    fn rowsumconst_matches_emit() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        for _ in 0..2000 {
            let ncols = 4 + rng.upto(60);          // up to ~64 slots
            let t = 1 + rng.upto(8);
            let n_slots = 1 + rng.upto(ncols);      // slots index lag[..ncols]
            let stride = 1 + rng.upto(80);          // mix of stride < and > n_slots
            let flat_lo = rng.upto(200);            // partial first/last runs
            let cid_base = rng.upto(1000);
            let coef = rng.field();
            let e = Expander::RowsumConst { cid_base, stride, coef };
            assert_matches_oracle(&e, flat_lo, n_slots, ncols, t, &mut rng, None);
        }
        // explicit edges: stride=1 (every slot its own cid), stride ≫ n_slots
        // (single run), run starting mid-stride (flat_lo not a multiple of stride).
        for &(stride, flat_lo, n_slots, ncols) in
            &[(1usize, 0usize, 16usize, 16usize), (1000, 0, 16, 16), (7, 3, 16, 16), (8, 8, 16, 16)] {
            let e = Expander::RowsumConst { cid_base: 10, stride, coef: 0x9e3779b9 };
            assert_matches_oracle(&e, flat_lo, n_slots, ncols, 4, &mut Lcg(99), None);
        }
    }

    #[test]
    fn rowsum_matches_emit() {
        let mut rng = Lcg(0xdead_beef_cafe_1234);
        for _ in 0..2000 {
            let ncols = 4 + rng.upto(60);
            let t = 1 + rng.upto(8);
            let n_slots = 1 + rng.upto(ncols);
            let stride = 1 + rng.upto(80);
            let flat_lo = rng.upto(200);
            let cid_base = rng.upto(1000);
            let coef_vec: Vec<u64> = (0..stride).map(|_| rng.field()).collect();
            let e = Expander::Rowsum { cid_base, stride, coef_vec: Arc::new(coef_vec) };
            assert_matches_oracle(&e, flat_lo, n_slots, ncols, t, &mut rng, None);
        }
    }

    #[test]
    fn freivaldsc_matches_emit() {
        let mut rng = Lcg(0x0bad_f00d_1357_9bdf);
        for _ in 0..2000 {
            let ncols = 4 + rng.upto(60);
            let t = 1 + rng.upto(8);
            let n_slots = 1 + rng.upto(ncols);
            let n = 1 + rng.upto(12);
            let h = 1 + rng.upto(4);
            let flat_lo = rng.upto(50);
            let base = rng.upto(1000);
            // size m / lam / rho so the oracle's indexing stays in bounds
            let m = (flat_lo + n_slots - 1) / n / h + 1;
            let lam: Vec<u64> = (0..h * m).map(|_| rng.field()).collect();
            let rho: Vec<u64> = (0..h * n).map(|_| rng.field()).collect();
            let e = Expander::FreivaldsC { base, m, n, h, lam: Arc::new(lam), rho: Arc::new(rho) };
            assert_matches_oracle(&e, flat_lo, n_slots, ncols, t, &mut rng, Some((base, h)));
        }
    }

    #[test]
    fn freivaldsb_nontranspose_matches_emit() {
        let mut rng = Lcg(0x5151_2727_9393_aeae);
        for _ in 0..2000 {
            let ncols = 4 + rng.upto(60);
            let t = 1 + rng.upto(8);
            let n_slots = 1 + rng.upto(ncols);
            let n = 1 + rng.upto(12);
            let h = 1 + rng.upto(4);
            let kk = 1 + rng.upto(8);
            let flat_lo = rng.upto(50);
            let base = rng.upto(1000);
            let k = 1 + rng.upto(100);   // unused in the non-transpose layout
            // size neg_rho to cover the max head·n+j the oracle indexes; the
            // preload table likewise must cover the max i_k referenced.
            let max_rest = (flat_lo + n_slots - 1) / n;
            let max_ik = (h - 1) * kk + max_rest / h;
            let neg_rho: Vec<u64> = (0..(max_ik / kk + 1) * n).map(|_| rng.field()).collect();
            let e = Expander::FreivaldsB {
                base, k, n, h, kk, transpose_b: false, neg_rho: Arc::new(neg_rho),
            };
            assert_matches_oracle(&e, flat_lo, n_slots, ncols, t, &mut rng,
                                  Some((base, max_ik + 1)));
        }
    }

    #[test]
    fn freivaldsb_transpose_matches_emit() {
        let mut rng = Lcg(0x7777_1111_3333_5555);
        for _ in 0..2000 {
            let ncols = 4 + rng.upto(60);
            let t = 1 + rng.upto(8);
            let n_slots = 1 + rng.upto(ncols);
            let h = 1 + rng.upto(4);
            let kk = 1 + rng.upto(8);
            let k = h * kk;                      // real layout: k = h·kk
            let flat_lo = rng.upto(50);
            let base = rng.upto(1000);
            let max_j = (flat_lo + n_slots - 1) / k;
            let n = max_j + 1;
            let neg_rho: Vec<u64> = (0..h * n).map(|_| rng.field()).collect();
            let e = Expander::FreivaldsB {
                base, k, n, h, kk, transpose_b: true, neg_rho: Arc::new(neg_rho),
            };
            assert_matches_oracle(&e, flat_lo, n_slots, ncols, t, &mut rng, Some((base, k)));
        }
    }

    /// Every remaining kind through the generic fold at value level (the triple-set
    /// equivalence per kind lives in runs_difftest; this exercises the fold itself,
    /// including the Fan range-sum path and the strided rope cids).
    #[test]
    fn generic_fold_all_remaining_kinds() {
        let mut rng = Lcg(0x0102_0304_0506_0708);
        let cases: Vec<(Expander, usize)> = vec![
            (Expander::RowsumConst { cid_base: 100, stride: 1, coef: 7 }, 53),  // ex-Identity
            (Expander::Weighted { cid_base: 11, coefs: (0..53).map(|i| (i as u64 * 7 + 3) % P).collect() }, 53),
            (Expander::TransposeO2m { cid_base: 3, rows: 1, cols: 1, fan: 5, coef: 4 }, 23),  // ex-StrideO2m
            (Expander::TransposeO2m { cid_base: 3, rows: 4, cols: 3, fan: 5, coef: 4 }, 12),
            (Expander::CausalId { cid_base: 20, m: 4, h: 2, coef: 6 }, 32),
            (Expander::CausalC2 { cid_base: 20, h: 2, coef: 6 }, 8),
            (Expander::Embed { cid_base: 10, d: 4, token_ids: vec![3, 1, 3, 0, 4],
                               vocab_lo: 2, rows_per_w: 3 }, 12),
            (Expander::RopeXrot { base: 60, h: 2, d_h: 6 }, 36),
            (Expander::RopeX { base: 60, h: 2, d_h: 6,
                               cos: Arc::new((0..9).map(|i| (i as u64 * 11 + 1) % P).collect()),
                               sin: Arc::new((0..9).map(|i| (i as u64 * 13 + 2) % P).collect()) }, 36),
        ];
        for (e, length) in &cases {
            for w in [1usize, 3, 7, *length] {
                let mut lo = 0;
                while lo < *length {
                    let n = w.min(length - lo);
                    let ncols = n.max(4);
                    assert_matches_oracle(e, lo, n, ncols, 1 + rng.upto(8), &mut rng, None);
                    lo += n;
                }
            }
        }
    }

    fn emit_window(exp: &Expander, flat_lo: usize, n_slots: usize) -> Vec<(usize, usize, u64)> {
        let mut v = Vec::new();
        exp.emit(flat_lo, n_slots, &mut |col, cid, coef| v.push((col, cid, coef)));
        v
    }

    /// The windowing foundation: the per-row windows tile the whole variable.
    /// `emit(0, var_len)` (col = global flat) equals the union over rows of
    /// `emit(ro·ell, n_slots)` (col → ro·ell + col). This is what lets one spanning
    /// Expander stand in for the variable's whole row range in the fold.
    fn check_span(exp: &Expander, var_len: usize, ell: usize) {
        let mut whole = emit_window(exp, 0, var_len);
        let mut rowwise = Vec::new();
        let nrows = (var_len + ell - 1) / ell;
        for ro in 0..nrows {
            let flat_lo = ro * ell;
            let n_slots = ell.min(var_len - flat_lo);
            for (col, cid, coef) in emit_window(exp, flat_lo, n_slots) {
                rowwise.push((flat_lo + col, cid, coef));
            }
        }
        whole.sort_unstable();
        rowwise.sort_unstable();
        assert_eq!(whole, rowwise);
    }

    #[test]
    fn whole_variable_spans_per_row() {
        let mut rng = Lcg(0xfeed_face_0000_1111);
        for _ in 0..1000 {
            let ell = 2 + rng.upto(30);
            let var_len = 1 + rng.upto(8 * ell);     // several rows
            let base = rng.upto(500);
            let coef = rng.field();
            let stride = 1 + rng.upto(40);
            let coef_vec = Arc::new((0..stride).map(|_| rng.field()).collect::<Vec<u64>>());

            check_span(&Expander::RowsumConst { cid_base: base, stride, coef }, var_len, ell);
            check_span(&Expander::Rowsum { cid_base: base, stride, coef_vec: coef_vec.clone() }, var_len, ell);
        }
    }
