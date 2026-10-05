//! Per-op compile handlers + emit helpers + compile_claims — line-for-line port
//! of protocol.py's _c_* / _emit_* / _settle_table / compile_claims. The Rust
//! verifier discovers + settles tables itself (via table_order) and derives all
//! op challenges from s_op by index, so it's a faithful independent twin.

use crate::field::{add, mul, sub, P};
use crate::claim::{Claim, ClaimSet, Table, Var};
use crate::compile::{Constraints, Expander, Family, QuadFamily};
use crate::protocol::{challenge, op_vec, Config, NUM_BLINDING_ROWS};
use std::sync::Arc;

fn nrows(length: usize, ell: usize) -> usize { (length + ell - 1) / ell }

// Mutable compile state threaded through the handlers (families/rhs/quad/nxt).
struct Build {
    families: Vec<Family>,
    rhs: Vec<(usize, usize, u64)>,   // compact runs: (start_cid, length, value)
    quad: Vec<QuadFamily>,
    nxt: usize,
    nq: usize,                       // running quad index (positional r_quad pairing)
    // WC-LCRL-STC: claim_index -> bridge-authenticated P_trace slice,
    // filled by verify_proof.rs BEFORE compile — the public pins for the
    // use_bridge claims' Pj identity rows. Empty when no bridge section.
    wc_pins: std::collections::HashMap<usize, Vec<u64>>,
}

impl Build {
    // Record one whole-variable spanning family for `var`. The Expander carries
    // no row window — the verifier's fold supplies `[ro·ell, ro·ell+n_slots)` to
    // `emit` per row, so one Family stands in for the variable's whole row range.
    fn push_family(&mut self, var: Var, ell: usize, exp: Expander) {
        self.families.push(Family { row_start: var.row_start, length: var.length, ell, exp });
    }

    // _emit_id: identity family on `var`.
    fn emit_id(&mut self, var: Var, cid_base: usize, coef: u64, ell: usize) {
        // Identity ≡ RowsumConst{stride: 1} — merged (Phase 1.4), same cids/coefs.
        self.push_family(var, ell, Expander::RowsumConst { cid_base, stride: 1, coef });
    }

    // _emit_quad: x·y + a·z = b per slot — ONE QuadFamily spanning all row
    // chunks (the quad lift); index_base advances by nrows so the r_quad
    // pairing matches the retired per-row Vec's flat order exactly.
    fn emit_quad(&mut self, x: Var, y: Var, z: Var, a_s: u64, b_s: u64, l: usize, ell: usize) {
        self.quad.push(QuadFamily {
            x_row: x.row_start, y_row: y.row_start, z_row: z.row_start,
            a: a_s, b: b_s, length: l, ell, index_base: self.nq,
        });
        self.nq += nrows(l, ell);
    }

    // _emit_rowsum: strided aggregation on `var`.
    fn emit_rowsum(&mut self, var: Var, cid_base: usize, stride: usize,
                   coef_vec: &[u64], ell: usize) {
        self.push_family(var, ell,
            Expander::Rowsum { cid_base, stride, coef_vec: Arc::new(coef_vec.to_vec()) });
    }

    // StrideO2m ≡ TransposeO2m{cols: 1, fan: stride} — merged (Phase 1.4),
    // same cids/coefs: cid = base + flat·stride + t either way.
    fn emit_stride_o2m(&mut self, var: Var, cid_base: usize, stride: usize, coef: u64, ell: usize) {
        self.push_family(var, ell,
            Expander::TransposeO2m { cid_base, rows: 1, cols: 1, fan: stride, coef });
    }

    // Blocks of `inner` slots at a stride of `outer` cids, fanned out (packets.py
    // L2_BlockStrideScalar).
    fn emit_block_stride(&mut self, var: Var, cid_base: usize, inner: usize, outer: usize,
                         fan: usize, fan_stride: usize, coef: u64, ell: usize) {
        self.push_family(var, ell, Expander::BlockStride { cid_base, inner, outer, fan,
                                                           fan_stride, coef });
    }

    // _emit_transpose_o2m: transposed fan-out on a row-major (rows, cols) var.
    fn emit_transpose_o2m(&mut self, var: Var, cid_base: usize, rows: usize, cols: usize,
                          fan: usize, coef: u64, ell: usize) {
        self.push_family(var, ell, Expander::TransposeO2m { cid_base, rows, cols, fan, coef });
    }

    fn emit_causal_id(&mut self, var: Var, cid_base: usize, m: usize, h: usize,
                      coef: u64, ell: usize) {
        self.push_family(var, ell, Expander::CausalId { cid_base, m, h, coef });
    }

    fn emit_causal_masked_id(&mut self, var: Var, cid_base: usize, m: usize, h: usize,
                             coef: u64, ell: usize) {
        self.push_family(var, ell, Expander::CausalMaskedId { cid_base, m, h, coef });
    }
    fn emit_causal_c2(&mut self, var: Var, cid_base: usize, h: usize, coef: u64, ell: usize) {
        self.push_family(var, ell, Expander::CausalC2 { cid_base, h, coef });
    }

    // _emit_lin_combo: 1·target + Σ (P−coef)·word = const, cids [base, base+L).
    fn emit_lin_combo(&mut self, target: Var, words: &[Var], coeffs: &[u64],
                      base: usize, ell: usize) {
        self.emit_id(target, base, 1, ell);
        for (w, &co) in words.iter().zip(coeffs) {
            self.emit_id(*w, base, (P - co % P) % P, ell);
        }
    }

    // _add_rhs: nz = [(offset, length, value)] kept as compact runs
    // (start_cid, length, value); the lin_sum check expands them on the fly.
    fn add_rhs(&mut self, base: usize, nz: &[(usize, usize, u64)]) {
        for &(off, length, val) in nz {
            if length > 0 && val % P != 0 {
                self.rhs.push((base + off, length, val % P));
            }
        }
    }

    // _emit_rescale: the one rescale gadget. Returns the new cur (cur + 2L).
    #[allow(clippy::too_many_arguments)]
    fn emit_rescale(&mut self, cur: usize, high: Var, low: Var, full: Var, shifted: Var,
                    z_low: Var, z_shifted: Var, rescale_bits: u32, shift_width: u32,
                    tight_alpha: u64, loose_alpha: u64, l: usize, ell: usize) -> usize {
        let offset = 1u64 << (shift_width - 1);
        self.emit_lin_combo(full, &[high, low], &[1u64 << rescale_bits, 1], cur, ell);
        self.emit_lin_combo(shifted, &[high], &[1], cur + l, ell);
        self.add_rhs(cur, &[(l, l, offset)]);
        self.emit_quad(low, z_low, z_low, (P - tight_alpha) % P, P - 1, l, ell);
        self.emit_quad(shifted, z_shifted, z_shifted, (P - loose_alpha) % P, P - 1, l, ell);
        cur + 2 * l
    }
}

// rope_scaled_inv_freq: inverse frequency for dim-pair k under the Llama-3
// wavelength ramp (rope_scaling, rope_type "llama3"). MIRROR of the prover's
// claims.py _rope_scaled_inv_freq — must stay line-for-line identical, in the
// same f64 expression order, or every RoPE constraint diverges. Golden
// vectors: rope_scaling_tests below + prover tests/test_rope_scaling.py.
fn rope_scaled_inv_freq(base: f64, d_h: usize, k: usize, scale_factor: f64,
                        low_freq_factor: f64, high_freq_factor: f64,
                        original_max_pos: f64) -> f64 {
    let inv_freq = 1.0 / base.powf(2.0 * k as f64 / d_h as f64);
    let wavelen = 2.0 * std::f64::consts::PI / inv_freq;
    let low_wl = original_max_pos / low_freq_factor;
    let high_wl = original_max_pos / high_freq_factor;
    if wavelen > low_wl {
        return inv_freq / scale_factor;
    }
    if wavelen < high_wl {
        return inv_freq;
    }
    let smooth = (original_max_pos / wavelen - low_freq_factor)
        / (high_freq_factor - low_freq_factor);
    (1.0 - smooth) * inv_freq / scale_factor + smooth * inv_freq
}

// YaRN rope_scaling (type "yarn": DeepSeek-V3, Kimi K2). Mirrors claims.py
// _yarn_correction_dim / _yarn_get_mscale / _rope_yarn_inv_freq /
// _rope_yarn_mscale line for line, in the same f64 expression order (the
// reference, K2's modeling_deepseek.py at Hugging Face fd1984e2, computes in
// float32; both languages here use f64). Golden vectors: test_rope_yarn.py
// and the yarn tests below.
pub struct Yarn { pub beta_fast: f64, pub beta_slow: f64, pub mscale: f64, pub mscale_all_dim: f64 }

fn yarn_correction_dim(num_rotations: f64, d_h: usize, base: f64, original_max_pos: f64) -> f64 {
    (d_h as f64 * (original_max_pos / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
        / (2.0 * base.ln())
}

fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 { 1.0 } else { 0.1 * mscale * scale.ln() + 1.0 }
}

fn rope_yarn_inv_freq(base: f64, d_h: usize, scale_factor: f64, original_max_pos: f64,
                      y: &Yarn) -> Vec<f64> {
    let low = std::cmp::max(
        yarn_correction_dim(y.beta_fast, d_h, base, original_max_pos).floor() as i64, 0);
    let high = std::cmp::min(
        yarn_correction_dim(y.beta_slow, d_h, base, original_max_pos).ceil() as i64,
        d_h as i64 - 1);
    let hi = if low == high { high as f64 + 0.001 } else { high as f64 };
    (0..d_h / 2).map(|k| {
        let ramp = ((k as i64 - low) as f64 / (hi - low as f64)).max(0.0).min(1.0);
        let mask = 1.0 - ramp;
        let freq_extra = 1.0 / base.powf(2.0 * k as f64 / d_h as f64);
        let freq_inter = 1.0 / (scale_factor * base.powf(2.0 * k as f64 / d_h as f64));
        freq_inter * (1.0 - mask) + freq_extra * mask
    }).collect()
}

fn rope_yarn_mscale(scale_factor: f64, y: &Yarn) -> f64 {
    yarn_get_mscale(scale_factor, y.mscale) / yarn_get_mscale(scale_factor, y.mscale_all_dim)
}

// _rope_cos_sin mirror: c,s integer tables at scale s_x, indexed seq·(d_h/2)+k.
// The unscaled path (original_max_pos == 0) keeps the original p/denominator
// form untouched (NOT p·inv_freq, which rounds differently in the last ulp)
// so tables from pre-scaling dumps stay identical.
fn rope_cos_sin(seq: usize, d_h: usize, s_x: u64, base: f64, pos_off: usize,
                scale_factor: f64, low_freq_factor: f64, high_freq_factor: f64,
                original_max_pos: f64, yarn: Option<&Yarn>) -> (Vec<u64>, Vec<u64>) {
    let half = d_h / 2;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    if let Some(y) = yarn {
        assert!(original_max_pos > 0.0 && scale_factor > 0.0,
                "YaRN RoPE needs its factor and original context");
        let inv = rope_yarn_inv_freq(base, d_h, scale_factor, original_max_pos, y);
        let mfac = rope_yarn_mscale(scale_factor, y);
        // Ties to even, as Python's round(): a cos/sin factor other than 1 can
        // land exactly on a half (the HALF golden vector), where .round()
        // would go away from zero and reject an honest proof. The legacy
        // paths below keep .round(): their values are cos·s_x, which no
        // integer position puts on a half, and their tables stay as dumped.
        for s in 0..seq {
            let p = (s + pos_off) as f64;
            for k in 0..half {
                let theta = p * inv[k];
                cos.push(((theta.cos() * mfac * s_x as f64).round_ties_even() as i128)
                    .rem_euclid(P as i128) as u64);
                sin.push(((theta.sin() * mfac * s_x as f64).round_ties_even() as i128)
                    .rem_euclid(P as i128) as u64);
            }
        }
        return (cos, sin);
    }
    for s in 0..seq {
        let p = (s + pos_off) as f64;
        for k in 0..half {
            let theta = if original_max_pos == 0.0 {
                p / base.powf(2.0 * k as f64 / d_h as f64)
            } else {
                p * rope_scaled_inv_freq(base, d_h, k, scale_factor,
                                         low_freq_factor, high_freq_factor,
                                         original_max_pos)
            };
            cos.push(((theta.cos() * s_x as f64).round() as i128).rem_euclid(P as i128) as u64);
            sin.push(((theta.sin() * s_x as f64).round() as i128).rem_euclid(P as i128) as u64);
        }
    }
    (cos, sin)
}

// ---- the 8 op handlers (dispatch by op name) ----
fn compile_op(cl: &Claim, ci: usize, s_op: &[u8], s_bind: Option<&[u8]>,
              b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    match cl.op.as_str() {
        "AddClaim" => {
            let l = cl.scalar("length") as usize;
            let base = b.nxt; b.nxt += l;
            if let Some(v) = cl.opt_scalar("public_rhs") {
                // REVEAL pin: 1*a = public_rhs (public constant from the claim).
                b.emit_id(cl.var("a"), base, 1, ell);
                b.add_rhs(base, &[(0, l, v % P)]);
            } else {
                b.emit_id(cl.var("a"), base, 1, ell);
                b.emit_id(cl.var("b"), base, 1, ell);
                b.emit_id(cl.var("c"), base, P - 1, ell);
            }
        }
        "LinCombClaim" => {
            // sum_k coefs[k]*xs[k][i] = rhs[i] (public). Mirrors
            // lincomb_compile: one identity band per (var, coef); the RHS
            // rides the b-chunk as (start, len, value) runs, one value for
            // the whole range or one per slot.
            let l = cl.scalar("length") as usize;
            let base = b.nxt; b.nxt += l;
            let xs = cl.var_list("xs");
            let coefs = cl.int_list("coefs");
            assert_eq!(xs.len(), coefs.len(), "LinCombClaim: xs/coefs mismatch");
            for (x, &c) in xs.iter().zip(coefs.iter()) {
                b.emit_id(*x, base, c % P, ell);
            }
            let rhs = cl.int_list("rhs");
            if rhs.len() == 1 {
                b.add_rhs(base, &[(0, l, rhs[0] % P)]);
            } else {
                assert_eq!(rhs.len(), l, "LinCombClaim: rhs length != length");
                let mut runs: Vec<(usize, usize, u64)> = Vec::new();
                let mut i = 0;
                while i < l {
                    let mut j = i;
                    while j < l && rhs[j] == rhs[i] { j += 1; }
                    runs.push((i, j - i, rhs[i] % P));
                    i = j;
                }
                b.add_rhs(base, &runs);
            }
        }
        "HadamardClaim" => {
            let l = cl.scalar("length") as usize;
            if cl.scalar("rescale_bits") > 0 {
                let base = b.nxt; b.nxt += 2 * l;
                // Quad order is load-bearing (combiner indexed by position): the
                // prover emits the product quad FIRST, then the rescale ranges.
                b.emit_quad(cl.var("a"), cl.var("b"), cl.var("c_full"), P - 1, 0, l, ell);
                b.emit_rescale(base, cl.var("c"), cl.var("c_low"), cl.var("c_full"),
                    cl.var("c_shifted"), cl.var("z_c_low"), cl.var("z_c_shifted"),
                    cl.scalar("rescale_bits") as u32, cl.scalar("output_width") as u32,
                    cl.table("range_rescale").alpha, cl.table("range_output").alpha, l, ell);
            } else {
                b.emit_quad(cl.var("a"), cl.var("b"), cl.var("c"), P - 1, 0, l, ell);
            }
        }
        "EmbeddingLookupClaim" => {
            let token_ids = cl.int_list("token_ids");
            let seq = token_ids.len();
            let d = cl.scalar("d") as usize;
            let l = seq * d;
            let base = b.nxt; b.nxt += l;
            b.emit_id(cl.var("x"), base, 1, ell);
            let rows_per_w = ell / d;
            let e = cl.var("E");
            let tok: Vec<usize> = token_ids.iter().map(|&t| t as usize).collect();
            for ro in 0..nrows(e.length, ell) {
                // Embed scatters by token, so each row is its own single-row Family
                // (length = ell, so emit's col == col-in-row, row = e.row_start+ro).
                b.families.push(Family {
                    row_start: e.row_start + ro, length: ell, ell,
                    exp: Expander::Embed { cid_base: base, d, token_ids: tok.clone(),
                                           vocab_lo: ro * rows_per_w, rows_per_w } });
            }
        }
        "RoPEClaim" => {
            let h = cl.cfg_int("config", "heads") as usize;
            let d_h = cl.cfg_int("config", "d_h") as usize;
            let seq = cl.cfg_int("config", "SEQ") as usize;
            let l = seq * h * d_h;
            let rescale_bits = cl.scalar("rescale_bits");
            let target = if rescale_bits > 0 { cl.var("x_rot_full") } else { cl.var("x_rot") };
            let base = b.nxt; b.nxt += l;
            let s_x = cl.cfg_int("config", "s_x");
            let pos_off = cl.cfg_int("config", "position_offset") as usize;
            // θ from the claim (Maverick: 500000); 10000.0 default for old dumps.
            let rope_base = cl.cfg_f64_or("config", "base", 10000.0);
            // Llama-3 rope_scaling ramp; defaults (original_max_pos=0 = OFF)
            // match RoPEConfig's, so pre-scaling dumps verify unchanged.
            // (serde as_f64 reads the int-serialized original_max_pos fine.)
            // YaRN (off unless the claim says so; absent in older dumps)
            let yarn = (cl.cfg_f64_or("config", "yarn", 0.0) != 0.0).then(|| Yarn {
                beta_fast: cl.cfg_f64_or("config", "yarn_beta_fast", 32.0),
                beta_slow: cl.cfg_f64_or("config", "yarn_beta_slow", 1.0),
                mscale: cl.cfg_f64_or("config", "yarn_mscale", 1.0),
                mscale_all_dim: cl.cfg_f64_or("config", "yarn_mscale_all_dim", 0.0),
            });
            let (cos, sin) = rope_cos_sin(
                seq, d_h, s_x, rope_base, pos_off,
                cl.cfg_f64_or("config", "scale_factor", 1.0),
                cl.cfg_f64_or("config", "low_freq_factor", 1.0),
                cl.cfg_f64_or("config", "high_freq_factor", 1.0),
                cl.cfg_f64_or("config", "original_max_pos", 0.0),
                yarn.as_ref());
            let (cos, sin) = (Arc::new(cos), Arc::new(sin));   // share one table across all rows
            b.push_family(target, ell, Expander::RopeXrot { base, h, d_h });
            let x = cl.var("x");
            b.push_family(x, ell, Expander::RopeX { base, h, d_h, cos: cos.clone(), sin: sin.clone() });
            if rescale_bits > 0 {
                let cur = b.nxt;
                b.nxt = b.emit_rescale(cur, cl.var("x_rot"), cl.var("x_rot_low"),
                    cl.var("x_rot_full"), cl.var("x_rot_shifted"),
                    cl.var("z_x_rot_low"), cl.var("z_x_rot_shifted"),
                    rescale_bits as u32, cl.scalar("output_width") as u32,
                    cl.table("range_rescale").alpha, cl.table("range_output").alpha, l, ell);
            }
        }
        "MatmulClaim" => compile_matmul(cl, ci, s_op, b, cfg),
        "RmsNormClaim" => compile_rmsnorm(cl, ci, s_op, b, cfg),
        "SoftmaxClaim" => compile_softmax(cl, b, cfg),
        "SiluClaim" => compile_silu(cl, b, cfg),
        "MaxClaim" => compile_max(cl, b, cfg),
        "InfoFinalizeClaim" => compile_info(cl, b, cfg),
        "PairedTlookupClaim" => compile_paired(cl, b, cfg),
        // x[i] + shift = Σ_n coeffs[n]·words[n][i] — linear-only (standalone
        // composable form; mirrors claims.py word_extract_compile).
        "WordExtractionClaim" => {
            let l = cl.scalar("length") as usize;
            let base = b.nxt; b.nxt += l;
            b.emit_id(cl.var("x"), base, 1, ell);
            let words = cl.var_list("words");
            let coeffs = cl.int_list("coeffs");
            for (w, &co) in words.iter().zip(coeffs.iter()) {
                b.emit_id(*w, base, (P - co % P) % P, ell);
            }
            let shift = cl.scalar("shift") % P;
            if shift != 0 {
                b.add_rhs(base, &[(0, l, (P - shift) % P)]);
            }
        }
        // (α − x)·z = 1 per slot — quad-only, no cids (mirrors range_word_compile).
        "RangeWordClaim" => {
            let table = cl.table("table");
            b.emit_quad(cl.var("x"), cl.var("z"), cl.var("z"),
                        (P - table.alpha) % P, P - 1, cl.scalar("length") as usize, ell);
        }
        "RoutingClaim" => compile_routing(cl, b, cfg),
        "TopkRoutingClaim" => compile_topk_routing(cl, b, cfg),
        "TopkSlotsClaim" => compile_topk_slots(cl, b, cfg),
        "GateBracketClaim" => compile_gate_bracket(cl, b, cfg),
        "SplitClaim" => compile_split(cl, b, cfg),
        "HeadInterleaveClaim" => compile_head_interleave(cl, b, cfg),
        "MaskedCombineClaim" => compile_masked_combine(cl, b, cfg),
        "ConcatClaim" => compile_concat(cl, b, cfg),
        "FreivaldsCombineClaim" => compile_freivalds_combine(cl, ci, s_op, b, cfg),
        "RoutedProjectedMatmulClaim" =>
            compile_routed_projected(cl, ci, s_op, s_bind, b, cfg),
        // Standalone signed-floor rescale — the same two linears and two range
        // LogUps the in-matmul rescale emits, so it reuses emit_rescale.
        "RescaleClaim" => {
            let l = cl.scalar("length") as usize;
            let cur = b.nxt;
            b.nxt = b.emit_rescale(cur, cl.var("x"), cl.var("x_low"),
                cl.var("x_full"), cl.var("x_shifted"), cl.var("z_low"),
                cl.var("z_shifted"), cl.scalar("rescale_bits") as u32,
                cl.scalar("output_width") as u32,
                cl.table("range_rescale").alpha, cl.table("range_output").alpha,
                l, cfg.ell as usize);
        }
        other => panic!("compile_claims: {} not ported", other),
    }
}

fn compile_matmul(cl: &Claim, ci: usize, s_op: &[u8], b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let h = cl.scalar("heads") as usize;
    let kk = cl.scalar("head_dim") as usize;
    let (k, m, n) = (cl.scalar("k") as usize, cl.scalar("m") as usize, cl.scalar("n") as usize);
    let rescale_bits = cl.scalar("rescale_bits");
    let c_fv = if rescale_bits > 0 { cl.var("C_full") } else { cl.var("C") };
    // Built once per matmul; shared (Arc) across this matmul's per-row expanders
    // instead of deep-cloned per row. The projections are seed-derived
    // (op_vec(s_op, ci, ·)), not weight data, but they recur across every row, so
    // one allocation + refcounted handles replaces ~800K copies (~50 GB → tens of MB).
    let rho     = Arc::new(op_vec(s_op, ci, "rho", h * n));
    let lam     = Arc::new(op_vec(s_op, ci, "lam", h * m));
    let neg_rho = Arc::new(rho.iter().map(|&v| (P - v % P) % P).collect::<Vec<u64>>());
    let neg_lam = Arc::new(lam.iter().map(|&v| (P - v % P) % P).collect::<Vec<u64>>());

    let (lf1, lf2, lf3) = (b.nxt, b.nxt + k, b.nxt + 2 * k);
    b.nxt = lf3 + h;

    b.emit_id(cl.var("y"), lf1, 1, ell);
    let bv = cl.var("B");
    b.push_family(bv, ell, Expander::FreivaldsB {
        base: lf1, k, n, h, kk, transpose_b: cl.boolean("transpose_b"), neg_rho: neg_rho.clone() });
    b.emit_id(cl.var("u"), lf2, 1, ell);
    // FreivaldsA ≡ FreivaldsB{transpose_b: true} with λ↔ρ, m↔n — merged
    // (Phase 1.4): both decode i_k = f%k with the coef vector indexed by f/k.
    let av = cl.var("A");
    b.push_family(av, ell, Expander::FreivaldsB {
        base: lf2, k, n: m, h, kk, transpose_b: true, neg_rho: neg_lam.clone() });
    // p rowsum: all-ones coef → RowsumConst (same cids/coefs as the previous
    // Rowsum{ones}; drops the materialized ones-vector). Bit-exact.
    b.push_family(cl.var("p"), ell,
        Expander::RowsumConst { cid_base: lf3, stride: kk, coef: 1 });
    b.push_family(c_fv, ell, Expander::FreivaldsC {
        base: lf3, m, n, h, lam: lam.clone(), rho: rho.clone() });
    b.emit_quad(cl.var("u"), cl.var("y"), cl.var("p"), P - 1, 0, k, ell);

    if rescale_bits > 0 {
        let l_out = m * h * n;
        let cur = b.nxt;
        b.nxt = b.emit_rescale(cur, cl.var("C"), cl.var("C_low"), cl.var("C_full"),
            cl.var("C_shifted"), cl.var("z_C_low"), cl.var("z_C_shifted"),
            rescale_bits as u32, cl.scalar("output_width") as u32,
            cl.table("range_rescale").alpha, cl.table("range_output").alpha, l_out, ell);
    }
}

/// RoutedProjectedMatmulClaim — the routed MoE matmul proved by projection
/// (analysis/routed-projected-protocol.md). The verifier's independent twin of
/// prover/routed_projected.py: same six constraint families, same ids, same
/// two quadratic families, with rho re-derived from s_op and (sigma, lambda)
/// from s_bind — never read from the proof.
///
///   [0,             E*K)             P    = W rho
///   [E*K,           E*K + T)         yr   = Y rho
///   [E*K + T,       E*K + 2T)        sum_k H = yr
///   [E*K + 2T,      E*K + 2T + E)    f_y  = P sigma
///   [E*K + 2T + E,  E*K + 2T + 2E)   f_u  = lambda^T M
///   [E*K + 2T + 2E, +1)              sum_e f_p = sum_{t,k} lambda_t Q sigma_k
fn compile_routed_projected(cl: &Claim, ci: usize, s_op: &[u8],
                            s_bind: Option<&[u8]>, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let (t, k, j, e) = (cl.scalar("T") as usize, cl.scalar("K") as usize,
                        cl.scalar("J") as usize, cl.scalar("E") as usize);
    let s_bind = s_bind.expect(
        "RoutedProjectedMatmulClaim needs the R2 coin s_bind: its late Freivalds \
         challenges must not be derivable before P and Q are committed");

    // Bridged claims share ONE rho per output width (spec 0.2) — mirrors
    // routed_sample; safe because R1 commits every claim's outputs first.
    let rho = if cl.opt_scalar("use_bridge").unwrap_or(0) == 1 {
        op_vec(s_op, 0, &format!("rho-w{j}"), j)
    } else {
        op_vec(s_op, ci, "rho", j)
    };
    let neg_rho = Arc::new(rho.iter().map(|&v| (P - v % P) % P).collect::<Vec<u64>>());
    let sig = Arc::new(op_vec(s_bind, ci, "sig", k));
    let lam = Arc::new(op_vec(s_bind, ci, "lam", t));
    let neg_sig = Arc::new(sig.iter().map(|&v| (P - v % P) % P).collect::<Vec<u64>>());
    let neg_lam = Arc::new(lam.iter().map(|&v| (P - v % P) % P).collect::<Vec<u64>>());

    let b_p = b.nxt;
    let b_yr = b_p + e * k;
    let b_sum = b_yr + t;
    let b_fy = b_sum + t;
    let b_fu = b_fy + e;
    let b_fin = b_fu + e;
    b.nxt = b_fin + 1;

    // P = W rho.  Bridge OFF: one LF1B band per expert shard (expert x owns
    // constraint ids [b_p + x*K, b_p + (x+1)*K)).  Bridge ON (use_bridge,
    // WC-LCRL-STC 0.5): the online fold over the weights is gone — the Pj
    // identity rows are pinned to the bridge-authenticated public P_trace,
    // which verify_proof.rs checked against the enrollment root before
    // handing it to this compile.
    b.emit_id(cl.var("Pj"), b_p, 1, ell);
    if cl.opt_scalar("use_bridge").unwrap_or(0) == 1 {
        let pin = b.wc_pins.remove(&ci).expect(
            "use_bridge claim but no bridge-authenticated P_trace pin — \
             the wc section is missing or its verification did not run");
        assert!(pin.len() >= e * k, "bridge pin shorter than E*K");
        let runs: Vec<(usize, usize, u64)> =
            pin[..e * k].iter().enumerate().map(|(i, &v)| (i, 1, v)).collect();
        b.add_rhs(b_p, &runs);
    } else {
        let w_experts = cl.var_list("W");
        assert_eq!(w_experts.len(), e, "expected one weight variable per expert");
        for (x, wv) in w_experts.iter().enumerate() {
            b.push_family(*wv, ell, Expander::FreivaldsB {
                base: b_p + x * k, k, n: j, h: 1, kk: k,
                transpose_b: false, neg_rho: neg_rho.clone() });
        }
    }
    // yr = Y rho
    b.emit_id(cl.var("yr"), b_yr, 1, ell);
    b.push_family(cl.var("Y"), ell, Expander::FreivaldsB {
        base: b_yr, k: t, n: j, h: 1, kk: t,
        transpose_b: false, neg_rho: neg_rho.clone() });
    // sum_k H[t,k] - yr[t] = 0
    b.push_family(cl.var("Hd"), ell,
        Expander::RowsumConst { cid_base: b_sum, stride: k, coef: 1 });
    b.emit_id(cl.var("yr"), b_sum, P - 1, ell);
    // f_y = P sigma
    b.emit_id(cl.var("f_y"), b_fy, 1, ell);
    b.push_family(cl.var("Pj"), ell, Expander::FreivaldsB {
        base: b_fy, k: e, n: k, h: 1, kk: e,
        transpose_b: false, neg_rho: neg_sig.clone() });
    // f_u = lambda^T M  (FreivaldsB with transpose_b, as compile_matmul's LF2)
    b.emit_id(cl.var("f_u"), b_fu, 1, ell);
    b.push_family(cl.var("M"), ell, Expander::FreivaldsB {
        base: b_fu, k: e, n: t, h: 1, kk: e,
        transpose_b: true, neg_rho: neg_lam.clone() });
    // sum_e f_p = sum_{t,k} lambda_t Q[t,k] sigma_k
    b.push_family(cl.var("f_p"), ell,
        Expander::RowsumConst { cid_base: b_fin, stride: e, coef: 1 });
    b.push_family(cl.var("Qm"), ell, Expander::FreivaldsC {
        base: b_fin, m: t, n: k, h: 1, lam: lam.clone(), rho: sig.clone() });

    // Quadratic, in the prover's order: H = X*Q first, then f_p = f_u*f_y.
    b.emit_quad(cl.var("X"), cl.var("Qm"), cl.var("Hd"), P - 1, 0, t * k, ell);
    b.emit_quad(cl.var("f_u"), cl.var("f_y"), cl.var("f_p"), P - 1, 0, e, ell);
}

fn isqrt_u128(n: u128) -> u128 {
    if n < 2 { return n; }
    let mut x = 1u128 << ((128 - n.leading_zeros() + 1) / 2);
    loop {
        let y = (x + n / x) / 2;
        if y >= x { return x; }
        x = y;
    }
}

fn bitlen_u128(n: u128) -> u32 { 128 - n.leading_zeros() }

// S_total limb width / count — the single S_total-headroom knob (must equal
// claims.RMS_LIMB_W / RMS_N_LIMBS; the compile is positional across sides).
const RMS_LIMB_W: u32 = 18;
const RMS_N_LIMBS: usize = 3;

/// 16-bit chunks plus one narrow top chunk (mirrors claims._chunk_widths).
fn rms_chunk_widths(total_bits: u32) -> Vec<u32> {
    let mut ws = vec![16u32; (total_bits / 16) as usize];
    if total_bits % 16 != 0 { ws.push(total_bits % 16); }
    if ws.is_empty() { ws.push(1); }
    ws
}

/// Derived range-window widths of the wrap-free rmsnorm bracket, computed
/// INDEPENDENTLY from the public config (mirrors RmsNormConfig's properties
/// — the prover cannot widen a window). See rmsnorm-bracket-fix.md.
struct RmsWidths { y: u32, slack: u32, g0h: u32, g1h: u32, g2: u32 }

fn rms_widths(d: u64, eps_int: u64, magic: u64) -> RmsWidths {
    let s_min = d as u128 * eps_int as u128;
    assert!(s_min >= 1, "rmsnorm needs eps_int >= 1");
    let m = magic as u128;
    let mut y = std::cmp::max(1, isqrt_u128(m / s_min));
    while y * y * s_min < m { y += 1; }
    let yw = std::cmp::max(1, bitlen_u128(y - 1));
    assert!(2 * yw + RMS_LIMB_W <= 63, "rmsnorm y_width too wide for wrap-free limbs");
    let cap: u128 = 1 << (RMS_LIMB_W * RMS_N_LIMBS as u32);
    let sw = bitlen_u128(2 * isqrt_u128(m * cap) + cap);
    assert!(m + (1u128 << sw) < P as u128, "rmsnorm slack window admits wrapped negatives");
    let g2 = std::cmp::max(1, bitlen_u128((m + (1u128 << sw)) >> (2 * RMS_LIMB_W)));
    assert!(g2 + 2 * RMS_LIMB_W <= 62, "rmsnorm G2 window would wrap");
    RmsWidths { y: yw, slack: sw, g0h: 2 * yw, g1h: 2 * yw + 1, g2 }
}

fn compile_rmsnorm(cl: &Claim, ci: usize, s_op: &[u8], b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let bsz = cl.cfg_int("config", "B") as usize;
    let d = cl.cfg_int("config", "d") as usize;
    let l = bsz * d;
    let eps_int = cl.cfg_int("config", "eps_int");
    let magic = cl.cfg_int("config", "magic");
    let out_rs = cl.cfg_int("config", "output_rescale_bits");
    let in_rs = cl.cfg_int("config", "rescale_bits");
    let out_target = if out_rs > 0 { cl.var("output_full") } else { cl.var("output") };
    let rho = op_vec(s_op, ci, "rho", d);
    let neg_rho: Vec<u64> = rho.iter().map(|&r| (P - r % P) % P).collect();
    let neg_ones_d = vec![P - 1; d];
    let w = rms_widths(d as u64, eps_int, magic);
    let slack_widths = rms_chunk_widths(w.slack);
    let chunk_strides: Vec<u64> = (0..slack_widths.len()).map(|n| 1u64 << (16 * n)).collect();

    let base = b.nxt;
    b.emit_id(cl.var("S_total"), base, 1, ell); b.emit_id(cl.var("S"), base, P - 1, ell);
    let f2 = base + bsz;
    b.emit_id(cl.var("y_m1"), f2, 1, ell); b.emit_id(cl.var("y"), f2, P - 1, ell);
    let f3 = base + 2 * bsz;
    b.emit_id(cl.var("S"), f3, 1, ell); b.emit_rowsum(cl.var("X_sq"), f3, d, &neg_ones_d, ell);
    let f4 = base + 3 * bsz;
    b.emit_id(cl.var("u"), f4, 1, ell); b.emit_rowsum(cl.var("x"), f4, d, &neg_rho, ell);
    let f5 = base + 4 * bsz;
    b.emit_id(cl.var("p"), f5, 1, ell); b.emit_rowsum(out_target, f5, d, &neg_rho, ell);
    let f6 = base + 5 * bsz;
    b.emit_id(cl.var("s_lo"), f6, 1, ell);
    let lo_chunks = cl.var_list("s_lo_chunks");
    for (n, &chunk) in lo_chunks.iter().enumerate() {
        b.emit_id(chunk, f6, (P - chunk_strides[n] % P) % P, ell);
    }
    let f7 = base + 6 * bsz;
    b.emit_id(cl.var("s_hi"), f7, 1, ell);
    let hi_chunks = cl.var_list("s_hi_chunks");
    for (n, &chunk) in hi_chunks.iter().enumerate() {
        b.emit_id(chunk, f7, (P - chunk_strides[n] % P) % P, ell);
    }

    // F8: y_m1 tight decomposition; F9: S_total limb decomposition.
    let f8 = base + 7 * bsz;
    b.emit_id(cl.var("y_m1"), f8, 1, ell);
    let ym1_chunks = cl.var_list("ym1_chunks");
    for (n, &chunk) in ym1_chunks.iter().enumerate() {
        b.emit_id(chunk, f8, (P - (1u64 << (16 * n)) % P) % P, ell);
    }
    let f9 = base + 8 * bsz;
    b.emit_id(cl.var("S_total"), f9, 1, ell);
    let s_limbs = cl.var_list("S_limbs");
    for (n, &limb) in s_limbs.iter().enumerate() {
        b.emit_id(limb, f9, (P - (1u64 << (RMS_LIMB_W * n as u32)) % P) % P, ell);
    }

    // F10..F13 / F14..F17: wrap-free bracket carry chains
    // (rmsnorm-bracket-fix.md); mirrors claims.rmsnorm_compile.emit_bracket.
    // Limbs + carry lows are LIMB_W-wide (stride 2^Lw); the carry highs
    // g0h/g1h/G2 use 16-bit internal chunk strides.
    let lw = RMS_LIMB_W;
    let emit_bracket = |b: &mut Build, f0: usize, h: &[Var], gl: &[Var],
                            g0h: &[Var], g1h: &[Var], g2: &[Var],
                            slack: Var, slack_coef: u64| {
        b.emit_id(h[0], f0, 1, ell);
        b.emit_id(gl[0], f0, P - 1, ell);
        for (j, &ch) in g0h.iter().enumerate() {
            b.emit_id(ch, f0, (P - (1u64 << (lw + 16 * j as u32)) % P) % P, ell);
        }
        let f1_ = f0 + bsz;
        b.emit_id(h[1], f1_, 1, ell);
        for (j, &ch) in g0h.iter().enumerate() {
            b.emit_id(ch, f1_, (1u64 << (16 * j)) % P, ell);
        }
        b.emit_id(gl[1], f1_, P - 1, ell);
        for (j, &ch) in g1h.iter().enumerate() {
            b.emit_id(ch, f1_, (P - (1u64 << (lw + 16 * j as u32)) % P) % P, ell);
        }
        let f2_ = f0 + 2 * bsz;
        b.emit_id(h[2], f2_, 1, ell);
        for (j, &ch) in g1h.iter().enumerate() {
            b.emit_id(ch, f2_, (1u64 << (16 * j)) % P, ell);
        }
        for (j, &ch) in g2.iter().enumerate() {
            b.emit_id(ch, f2_, (P - (1u64 << (16 * j)) % P) % P, ell);
        }
        let f3_ = f0 + 3 * bsz;
        for (j, &ch) in g2.iter().enumerate() {
            b.emit_id(ch, f3_, (1u64 << (2 * lw + 16 * j as u32)) % P, ell);
        }
        b.emit_id(gl[1], f3_, (1u64 << lw) % P, ell);
        b.emit_id(gl[0], f3_, 1, ell);
        b.emit_id(slack, f3_, slack_coef, ell);
    };
    let (lo_h, lo_gl) = (cl.var_list("lo_H"), cl.var_list("lo_gl"));
    let (lo_g0h, lo_g1h, lo_g2) = (cl.var_list("lo_g0h_chunks"),
        cl.var_list("lo_g1h_chunks"), cl.var_list("lo_G2_chunks"));
    let (hi_h, hi_gl) = (cl.var_list("hi_H"), cl.var_list("hi_gl"));
    let (hi_g0h, hi_g1h, hi_g2) = (cl.var_list("hi_g0h_chunks"),
        cl.var_list("hi_g1h_chunks"), cl.var_list("hi_G2_chunks"));
    emit_bracket(b, base + 9 * bsz, &lo_h, &lo_gl, &lo_g0h, &lo_g1h, &lo_g2,
                 cl.var("s_lo"), P - 1);
    emit_bracket(b, base + 13 * bsz, &hi_h, &hi_gl, &hi_g0h, &hi_g1h, &hi_g2,
                 cl.var("s_hi"), 1);
    let mut cur = base + 17 * bsz;

    // QUAD ORDER load-bearing (combiner indexed by position): op quads (arithmetic
    // + slack range checks + limb range checks) FIRST, rescale range quads LAST —
    // matching the prover.
    b.emit_quad(cl.var("x"), cl.var("x"), cl.var("X_sq"), P - 1, 0, l, ell);
    b.emit_quad(cl.var("y"), cl.var("y"), cl.var("q1"), P - 1, 0, bsz, ell);
    b.emit_quad(cl.var("y_m1"), cl.var("y_m1"), cl.var("q2"), P - 1, 0, bsz, ell);
    for k in 0..RMS_N_LIMBS {
        b.emit_quad(cl.var("q1"), s_limbs[k], lo_h[k], P - 1, 0, bsz, ell);
    }
    for k in 0..RMS_N_LIMBS {
        b.emit_quad(cl.var("q2"), s_limbs[k], hi_h[k], P - 1, 0, bsz, ell);
    }
    b.emit_quad(cl.var("y"), cl.var("u"), cl.var("p"), P - 1, 0, bsz, ell);
    let alpha_t = cl.table("range_slack").alpha;
    let alpha_limb = cl.table("range_limb").alpha;
    let slack_top_alpha = if w.slack % 16 != 0 || w.slack < 16 {
        cl.table("range_slack_top").alpha } else { alpha_t };
    let slack_alpha = |n: usize| if slack_widths[n] == 16 { alpha_t } else { slack_top_alpha };
    let lo_z = cl.var_list("z_lo_chunks");
    for (n, (chunk, z)) in lo_chunks.iter().zip(lo_z.iter()).enumerate() {
        b.emit_quad(*chunk, *z, *z, (P - slack_alpha(n)) % P, P - 1, bsz, ell);
    }
    let hi_z = cl.var_list("z_hi_chunks");
    for (n, (chunk, z)) in hi_chunks.iter().zip(hi_z.iter()).enumerate() {
        b.emit_quad(*chunk, *z, *z, (P - slack_alpha(n)) % P, P - 1, bsz, ell);
    }
    // Limb range checks, in the frozen _rms_limb_range_groups order.
    let top_alpha = |width: u32, tbl: &str| -> u64 {
        if width % 16 != 0 || width < 16 { cl.table(tbl).alpha } else { alpha_t }
    };
    let group_alphas = |widths: &[u32], tbl: &str| -> Vec<u64> {
        widths.iter().map(|&cw| if cw == 16 { alpha_t } else { top_alpha(cw, tbl) }).collect()
    };
    let yw_chunks = rms_chunk_widths(w.y);
    let g0w_chunks = rms_chunk_widths(w.g0h);
    let g1w_chunks = rms_chunk_widths(w.g1h);
    let g2w_chunks = rms_chunk_widths(w.g2);
    let groups: Vec<(Vec<Var>, Vec<Var>, Vec<u64>)> = vec![
        (ym1_chunks.clone(), cl.var_list("z_ym1_chunks"), group_alphas(&yw_chunks, "range_y_top")),
        (s_limbs.clone(), cl.var_list("z_S_limbs"), vec![alpha_limb; RMS_N_LIMBS]),
        (lo_gl.clone(), cl.var_list("z_lo_gl"), vec![alpha_limb; 2]),
        (lo_g0h.clone(), cl.var_list("z_lo_g0h"), group_alphas(&g0w_chunks, "range_g0h_top")),
        (lo_g1h.clone(), cl.var_list("z_lo_g1h"), group_alphas(&g1w_chunks, "range_g1h_top")),
        (lo_g2.clone(), cl.var_list("z_lo_G2"), group_alphas(&g2w_chunks, "range_G2_top")),
        (hi_gl.clone(), cl.var_list("z_hi_gl"), vec![alpha_limb; 2]),
        (hi_g0h.clone(), cl.var_list("z_hi_g0h"), group_alphas(&g0w_chunks, "range_g0h_top")),
        (hi_g1h.clone(), cl.var_list("z_hi_g1h"), group_alphas(&g1w_chunks, "range_g1h_top")),
        (hi_g2.clone(), cl.var_list("z_hi_G2"), group_alphas(&g2w_chunks, "range_G2_top")),
    ];
    for (vars_, zs, alphas) in &groups {
        for ((v, z), a) in vars_.iter().zip(zs.iter()).zip(alphas.iter()) {
            b.emit_quad(*v, *z, *z, (P - a) % P, P - 1, bsz, ell);
        }
    }

    if in_rs > 0 {
        cur = b.emit_rescale(cur, cl.var("x"), cl.var("x_low"), cl.var("x_in"),
            cl.var("x_shifted"), cl.var("z_x_low"), cl.var("z_x_shifted"),
            in_rs as u32, 16u32,
            cl.table("range_rescale").alpha, cl.table("range_slack").alpha, l, ell);
    }
    if out_rs > 0 {
        cur = b.emit_rescale(cur, cl.var("output"), cl.var("output_low"), cl.var("output_full"),
            cl.var("output_shifted"), cl.var("z_output_low"), cl.var("z_output_shifted"),
            out_rs as u32, cl.cfg_int("config", "output_width") as u32,
            cl.table("range_output_rescale").alpha, cl.table("range_output").alpha, l, ell);
    }
    b.nxt = cur;
    b.add_rhs(base, &[(0, bsz, (d as u64 * eps_int) % P), (bsz, bsz, P - 1),
                      (12 * bsz, bsz, magic % P), (16 * bsz, bsz, sub(magic, 1))]);
}

fn compile_silu(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let l = cl.scalar("length") as usize;
    let bb = cl.cfg_int("config", "b");
    let t_len = cl.cfg_int("config", "T_LEN");
    let (b2, b3, b4) = (cl.cfg_int("config", "b_2"), cl.cfg_int("config", "b_3"),
                        cl.cfg_int("config", "b_4"));
    let beta = cl.table("silu_table").beta;
    let base = b.nxt;
    // 7 identity-scalar families at base + i·L.
    b.emit_lin_combo(cl.var("x"), &[cl.var("magnitude"), cl.var("C")], &[1, 2], base, ell);
    b.emit_lin_combo(cl.var("magnitude"),
        &[cl.var("a_0"), cl.var("a_1"), cl.var("a_2"), cl.var("a_3"), cl.var("a_4")],
        &[1, bb, b2, b3, b4], base + l, ell);
    b.emit_lin_combo(cl.var("g"), &[cl.var("a_2"), cl.var("a_3"), cl.var("a_4")],
        &[b2, b3, b4], base + 2 * l, ell);
    b.emit_lin_combo(cl.var("key"), &[cl.var("sign"), cl.var("a_1")], &[t_len, 1], base + 3 * l, ell);
    b.emit_lin_combo(cl.var("x"), &[cl.var("output_sat"), cl.var("C")], &[1, 1], base + 4 * l, ell);
    b.emit_lin_combo(cl.var("y"), &[cl.var("output"), cl.var("mux_a"), cl.var("mux_b")],
        &[1, 1, P - 1], base + 5 * l, ell);
    b.emit_lin_combo(cl.var("pt_u"), &[cl.var("key"), cl.var("y")], &[1, beta], base + 6 * l, ell);
    b.nxt = base + 7 * l;

    b.emit_quad(cl.var("sign"), cl.var("sign"), cl.var("sign"), P - 1, 0, l, ell);
    b.emit_quad(cl.var("sign"), cl.var("x"), cl.var("C"), P - 1, 0, l, ell);
    // sign = C·inv_x: sign = 0 is forced at x = 0 (protocol review F04)
    b.emit_quad(cl.var("C"), cl.var("inv_x"), cl.var("sign"), P - 1, 0, l, ell);
    b.emit_quad(cl.var("g"), cl.var("inv_g"), cl.var("is_high"), P - 1, 0, l, ell);
    b.emit_quad(cl.var("is_high"), cl.var("g"), cl.var("g"), P - 1, 0, l, ell);
    b.emit_quad(cl.var("is_high"), cl.var("is_high"), cl.var("is_high"), P - 1, 0, l, ell);
    b.emit_quad(cl.var("is_high"), cl.var("y"), cl.var("mux_a"), P - 1, 0, l, ell);
    b.emit_quad(cl.var("is_high"), cl.var("output_sat"), cl.var("mux_b"), P - 1, 0, l, ell);
    // every magnitude word is ranged, a_1 included (protocol review F03)
    for (var, z, tbl) in [("a_0", "z_a0", "range_b"), ("a_1", "z_a1", "range_a1"),
                          ("a_2", "z_a2", "range_w2"), ("a_3", "z_a3", "range_w3"),
                          ("a_4", "z_a4", "range_w4")] {
        b.emit_quad(cl.var(var), cl.var(z), cl.var(z),
            (P - cl.table(tbl).alpha) % P, P - 1, l, ell);
    }
    b.emit_quad(cl.var("pt_u"), cl.var("pt_z"), cl.var("pt_z"),
        (P - cl.table("silu_table").alpha) % P, P - 1, l, ell);

    if cl.cfg_int("config", "rescale_bits") > 0 {
        let cur = base + 7 * l;
        b.nxt = b.emit_rescale(cur, cl.var("x"), cl.var("x_low"), cl.var("x_in"),
            cl.var("x_shifted"), cl.var("z_x_low"), cl.var("z_x_shifted"),
            cl.cfg_int("config", "rescale_bits") as u32, cl.cfg_int("config", "width_2") as u32,
            cl.table("range_rescale").alpha, cl.table("range_x").alpha, l, ell);
    }
}

fn compile_softmax(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let bsz = cl.cfg_int("config", "B") as usize;
    let m = cl.cfg_int("config", "M") as usize;
    let h = cl.cfg_int("config", "heads") as usize;
    let l = bsz * m;
    let sat = cl.cfg_int("config", "saturate") != 0;
    let causal = cl.cfg_int("config", "causal") != 0;
    let s_in = cl.cfg_int("config", "s_in");
    let s_x = cl.cfg_int("config", "s_x");
    let rescaling = s_in != 0 && s_in != s_x;
    let z_max = cl.cfg_int("config", "Z_max");
    let s_y = cl.cfg_int("config", "s_y");
    let beta_a = cl.table("exp_A").beta;
    let beta_b = cl.table("exp_B").beta;
    let neg_ones_m = vec![P - 1; m];
    let y_a_look = if sat { cl.var("y_A_raw") } else { cl.var("y_A") };
    let y_b_look = if sat { cl.var("y_B_raw") } else { cl.var("y_B") };
    let seq = bsz / h;
    let l_u = h * seq * (seq + 1) / 2;

    let base = b.nxt;
    let mut cur = base;
    if causal {
        assert!(seq * h == bsz && seq <= m, "causal softmax needs B = SEQ·H with SEQ ≤ M");
        b.emit_causal_id(cl.var("z"), cur, m, h, 1, ell);
        b.emit_causal_c2(cl.var("c2"), cur, h, P - 1, ell);
        b.emit_causal_id(cl.var("x"), cur, m, h, 1, ell);
        if sat { b.emit_causal_id(cl.var("z_high"), cur, m, h, z_max % P, ell); }
        cur += l_u;
        // masked cells: z = 0, so the lookup key is exactly Z_max and the
        // doubled table's zero half answers (protocol review F01).
        let l_m: usize = h * (0..seq).map(|i| m - 1 - i).sum::<usize>();
        b.emit_causal_masked_id(cl.var("z"), cur, m, h, 1, ell);
        cur += l_m;
    } else {
        b.emit_id(cl.var("z"), cur, 1, ell);
        b.emit_stride_o2m(cl.var("c2"), cur, m, P - 1, ell);
        b.emit_id(cl.var("x"), cur, 1, ell);
        if sat { b.emit_id(cl.var("z_high"), cur, z_max % P, ell); }
        cur += l;
    }
    if sat {
        // round_up: y = y_raw - mux + is_high (saturate out-of-table to 1), i.e.
        // y_raw = y + mux - is_high.  Default: y = y_raw - mux (saturate to 0).
        let ru = cl.cfg_int("config", "round_up") != 0;
        let mut wa = vec![cl.var("y_A"), cl.var("mux_y_A")];
        let mut wb = vec![cl.var("y_B"), cl.var("mux_y_B")];
        let mut co = vec![1u64, 1u64];
        if ru { wa.push(cl.var("is_high")); wb.push(cl.var("is_high")); co.push(P - 1); }
        b.emit_lin_combo(cl.var("y_A_raw"), &wa, &co, cur, ell); cur += l;
        b.emit_lin_combo(cl.var("y_B_raw"), &wb, &co, cur, ell); cur += l;
    }
    let (pt_u_a_base, pt_u_b_base) = (cur, cur + l);
    b.emit_lin_combo(cl.var("pt_u_A"), &[cl.var("z"), y_a_look], &[1, beta_a], cur, ell); cur += l;
    b.emit_lin_combo(cl.var("pt_u_B"), &[cl.var("z"), y_b_look], &[1, beta_b], cur, ell); cur += l;
    b.emit_id(cl.var("s1"), cur, 1, ell); b.emit_rowsum(cl.var("y_A"), cur, m, &neg_ones_m, ell); cur += bsz;
    b.emit_id(cl.var("s2"), cur, 1, ell); b.emit_rowsum(cl.var("y_B"), cur, m, &neg_ones_m, ell); cur += bsz;
    // F5/F6/F7 bracket + c2_shifted. Python passes Python ints -1/1 to
    // _emit_lin_combo which does (P-co)%P; mirror that with the field reps.
    let f5 = cur;
    b.emit_lin_combo(cl.var("s1"),         &[cl.var("r_lo")], &[P - 1], cur, ell); cur += bsz;
    b.emit_lin_combo(cl.var("r_hi"),       &[cl.var("s2")],   &[1],     cur, ell); cur += bsz;
    b.emit_lin_combo(cl.var("c2_shifted"), &[cl.var("c2")],   &[1],     cur, ell); cur += bsz;
    // RHS. Causal: per-cell Z_max shift on masked cells (j > i_qry) of pt_u_A/B.
    if causal {
        // Masked cells (j > i_qry) of each row bb2 are the contiguous run
        // [i_qry+1, m); emit one run per row instead of per cell.
        for pt_base in [pt_u_a_base, pt_u_b_base] {
            for bb2 in 0..(l / m) {
                let i_qry = bb2 / h;
                if i_qry + 1 < m {
                    b.add_rhs(pt_base, &[(bb2 * m + i_qry + 1, m - i_qry - 1, z_max)]);
                }
            }
        }
    }
    b.add_rhs(f5, &[(0, bsz, s_y % P),
                    (bsz, bsz, (P - (s_y + 1) % P) % P),
                    (2 * bsz, bsz, (1u64 << (cl.cfg_int("config", "aux_chunk_width") - 1)) % P)]);

    // Quadratics: PT_A/PT_B + 3 bracket range checks (+ 6 sat).
    b.emit_quad(cl.var("pt_u_A"), cl.var("pt_z_A"), cl.var("pt_z_A"),
        (P - cl.table("exp_A").alpha) % P, P - 1, l, ell);
    b.emit_quad(cl.var("pt_u_B"), cl.var("pt_z_B"), cl.var("pt_z_B"),
        (P - cl.table("exp_B").alpha) % P, P - 1, l, ell);
    let alpha_r = cl.table("range_aux").alpha;
    b.emit_quad(cl.var("c2_shifted"), cl.var("z_c2"), cl.var("z_c2"), (P - alpha_r) % P, P - 1, bsz, ell);
    b.emit_quad(cl.var("r_lo"), cl.var("z_r_lo"), cl.var("z_r_lo"), (P - alpha_r) % P, P - 1, bsz, ell);
    b.emit_quad(cl.var("r_hi"), cl.var("z_r_hi"), cl.var("z_r_hi"), (P - alpha_r) % P, P - 1, bsz, ell);
    if sat {
        b.emit_quad(cl.var("z_high"), cl.var("inv_z_high"), cl.var("is_high"), P - 1, 0, l, ell);
        b.emit_quad(cl.var("is_high"), cl.var("z_high"), cl.var("z_high"), P - 1, 0, l, ell);
        b.emit_quad(cl.var("is_high"), cl.var("is_high"), cl.var("is_high"), P - 1, 0, l, ell);
        b.emit_quad(cl.var("is_high"), cl.var("y_A_raw"), cl.var("mux_y_A"), P - 1, 0, l, ell);
        b.emit_quad(cl.var("is_high"), cl.var("y_B_raw"), cl.var("mux_y_B"), P - 1, 0, l, ell);
        b.emit_quad(cl.var("z_high"), cl.var("z_z_high"), cl.var("z_z_high"),
            (P - cl.table("range_z_high").alpha) % P, P - 1, l, ell);
    }

    // Input rescale: its range quads append AFTER the op quads above (prover order).
    if rescaling {
        cur = b.emit_rescale(cur, cl.var("x"), cl.var("x_low"), cl.var("x_in"),
            cl.var("x_shifted"), cl.var("z_x_low"), cl.var("z_x_shifted"),
            cl.cfg_int("config", "rescale_bits") as u32, 16,
            cl.table("range_rescale").alpha, cl.table("range_aux").alpha, l, ell);
    }
    b.nxt = cur;
}

/// Settle one shared table — mirrors _settle_table.
// MaxClaim: v* = max_i l, gap = v* - l >= 0, plus a hidden output select
// gap_o = gap[t, tok_t] over committed/blinded output tokens. Mirrors max_compile.
// Linear families (cids advance b.nxt): ΣA=1 [T], v*=ΣAl [T], gap+l-v* [L],
//   neg_gap+gap [L], ΣO=1 [T], gap_o=ΣOgap [T], tok=Σ i·O [T].
// Quads (no cid): A·A=A, Al=A·l, O·O=O, Ogap=O·gap, (α-gap)·z=1. Tables auto-settled.
fn compile_max(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let t = cl.scalar("T") as usize;
    let v = cl.scalar("V") as usize;
    let l = t * v;
    let (a, lg, al) = (cl.var("A"), cl.var("l"), cl.var("Al"));
    let vs = cl.var("vstar");
    let gap = cl.var("gap");
    let ngap = cl.var("neg_gap");
    let zg = cl.var("z_gap");
    let tok = cl.var("tok");
    let o = cl.var("O");
    let ogap = cl.var("Ogap");
    let gap_o = cl.var("gap_o");
    let table = cl.table("table");
    let neg1 = P - 1;
    let ones = vec![1u64; v];
    let idx: Vec<u64> = (0..v as u64).collect();    // [0,1,..,V-1] for tok = Σ i·O
    let base = b.nxt;
    // Σ_i A[t,i] = 1
    b.emit_rowsum(a, base, v, &ones, ell);
    b.add_rhs(base, &[(0, t, 1)]);
    // v* = Σ_i Al
    b.emit_rowsum(al, base + t, v, &ones, ell);
    b.emit_id(vs, base + t, neg1, ell);
    // gap + l - v*(broadcast) = 0
    let gbase = base + 2 * t;
    b.emit_id(gap, gbase, 1, ell);
    b.emit_id(lg, gbase, 1, ell);
    b.emit_stride_o2m(vs, gbase, v, neg1, ell);
    // neg_gap + gap = 0
    let nbase = gbase + l;
    b.emit_id(ngap, nbase, 1, ell);
    b.emit_id(gap, nbase, 1, ell);
    // Σ_i O[t,i] = 1
    let obase = nbase + l;
    b.emit_rowsum(o, obase, v, &ones, ell);
    b.add_rhs(obase, &[(0, t, 1)]);
    // gap_o = Σ_i Ogap
    let gobase = obase + t;
    b.emit_rowsum(ogap, gobase, v, &ones, ell);
    b.emit_id(gap_o, gobase, neg1, ell);
    // tok_t = Σ_i i·O[t,i]
    let tbase = gobase + t;
    b.emit_rowsum(o, tbase, v, &idx, ell);
    b.emit_id(tok, tbase, neg1, ell);
    b.nxt = tbase + t;
    // quads (no cid advance)
    b.emit_quad(a, a, a, neg1, 0, l, ell);          // A·A = A
    b.emit_quad(a, lg, al, neg1, 0, l, ell);        // Al = A·l
    b.emit_quad(o, o, o, neg1, 0, l, ell);          // O·O = O
    b.emit_quad(o, gap, ogap, neg1, 0, l, ell);     // Ogap = O·gap
    let neg_alpha = (P - table.alpha) % P;
    b.emit_quad(gap, zg, zg, neg_alpha, neg1, l, ell);   // (α - gap)·z = 1
}

// RoutingClaim: top-1 MoE routing — the committed one-hot mask m (T,E) is the
// argmax of tiebroken router logits rt = 2^L·r + (E−1−e). Mirrors
// routing_claim.py routing_compile. Linear families (cids advance b.nxt):
//   F1 rt − 2^L·r = (E−1−e)  [T·E] · F2 Σ_e m = 1  [T] ·
//   F3 Σ_e mrt − rstar  [T] · F4 gap + rt − rstar(bcast)  [T·E] ·
//   F5 2^L·r_chosen + Σ_e (E−1−e)·m − rstar  [T]
// Quads (no cid): m·m = m, m·rt = mrt. The gap range check arrives as separate
// composed WordExtractionClaim + RangeWordClaim claims, not here.
fn compile_routing(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let t = cl.scalar("T") as usize;
    let e = cl.scalar("E") as usize;
    let l_bits = cl.scalar("L_bits") as u32;
    let l = t * e;
    let (r, m, rt) = (cl.var("r"), cl.var("m"), cl.var("rt"));
    let (mrt, rstar) = (cl.var("mrt"), cl.var("rstar"));
    let (gap, r_chosen) = (cl.var("gap"), cl.var("r_chosen"));
    let neg1 = P - 1;
    let two_l = (1u64 << l_bits) % P;
    let ones: Vec<u64> = vec![1; e];
    let bonus: Vec<u64> = (0..e as u64).map(|i| e as u64 - 1 - i).collect();
    let base = b.nxt; b.nxt += 2 * l + 3 * t;
    let mut cur = base;
    // F1: rt − 2^L·r = (E−1−e)
    b.emit_id(rt, cur, 1, ell);
    b.emit_id(r, cur, (P - two_l) % P, ell);
    let bonus_rhs: Vec<(usize, usize, u64)> =
        (0..l).map(|f| (f, 1usize, (e - 1 - (f % e)) as u64)).collect();
    b.add_rhs(cur, &bonus_rhs);
    cur += l;
    // F2: Σ_e m[t,e] = 1
    b.emit_rowsum(m, cur, e, &ones, ell);
    b.add_rhs(cur, &[(0, t, 1)]);
    cur += t;
    // F3: Σ_e mrt[t,e] − rstar[t] = 0
    b.emit_rowsum(mrt, cur, e, &ones, ell);
    b.emit_id(rstar, cur, neg1, ell);
    cur += t;
    // F4: gap + rt − rstar(broadcast over E) = 0
    b.emit_id(gap, cur, 1, ell);
    b.emit_id(rt, cur, 1, ell);
    b.emit_stride_o2m(rstar, cur, e, neg1, ell);
    cur += l;
    // F5: 2^L·r_chosen + Σ_e (E−1−e)·m[t,e] − rstar = 0
    b.emit_id(r_chosen, cur, two_l, ell);
    b.emit_rowsum(m, cur, e, &bonus, ell);
    b.emit_id(rstar, cur, neg1, ell);
    // quads AFTER the linear families, in prover order
    b.emit_quad(m, m, m, neg1, 0, l, ell);
    b.emit_quad(m, rt, mrt, neg1, 0, l, ell);
}

// TopkRoutingClaim: the committed k-hot mask m (T,E) is the top-k of the
// tiebroken selection quantity q̃ = 2^L·(s + b) + (E−1−e), through a free
// per-token threshold τ. Mirrors prover/topk_routing.py topk_compile
// (analysis/topk-routing-design.md §3.1). Linear families (cids from b.nxt):
//   F1 q̃ − 2^L·s − 2^L·b(bcast) = (E−1−e)  [E·T, expert-major e·T + t] ·
//   F2 Σ_e m = k  [T] · Fd d − q̃ + τ(bcast) = 0  [T·E] · Fv v − 2·md + d = 0  [T·E]
// Quads: m·m = m, m·d = md. The range on v is a composed WordExtractionClaim.
// The threshold guard fails closed: τ is free, so a selected and an
// unselected relation sum to q̃_s − q̃_u ≡ a + c < 2^{R+1} − 1, which must not
// reach a wrapped negative difference (> P − 2^width). Top-1's 2^R ≤ P − 2^width
// is not enough (the design's counterexample, R = 63 at width 43).
fn compile_topk_routing(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let t = cl.scalar("T") as usize;
    let e = cl.scalar("E") as usize;
    let k = cl.scalar("k");
    let l_bits = cl.scalar("L_bits") as u32;
    let width = cl.scalar("width") as u32;
    let r = cl.scalar("range_bits") as u32;
    assert!(k >= 1 && (k as usize) < e, "topk: need 1 <= k < E");
    assert!(l_bits < 32 && (e as u64 - 1) < (1u64 << l_bits), "topk: L_bits must hold E - 1");
    assert!(width < 64 && r < 64 && r >= width, "topk: the range must cover the width");
    assert!((1u128 << (r + 1)) - 1 <= (P as u128) - (1u128 << width),
            "topk: threshold guard: a wrapped threshold can split between a selected and an unselected relation");
    let l = t * e;
    let (s, bias, m, qt) = (cl.var("s"), cl.var("b"), cl.var("m"), cl.var("qt"));
    let (tau, d, md, v) = (cl.var("tau"), cl.var("d"), cl.var("md"), cl.var("v"));
    assert!(s.length == l && bias.length == e && m.length == l && tau.length == t,
            "topk: variable lengths");
    let neg1 = P - 1;
    let neg_two_l = P - (1u64 << l_bits) % P;
    let ones: Vec<u64> = vec![1; e];
    let base = b.nxt; b.nxt += 3 * l + t;
    let mut cur = base;
    // F1, expert-major: the bias fans out along its expert's block of T cids
    b.emit_transpose_o2m(qt, cur, t, e, 1, 1, ell);
    b.emit_transpose_o2m(s, cur, t, e, 1, neg_two_l, ell);
    b.emit_stride_o2m(bias, cur, t, neg_two_l, ell);
    let bonus: Vec<(usize, usize, u64)> =
        (0..e - 1).map(|i| (i * t, t, (e - 1 - i) as u64)).collect();
    b.add_rhs(cur, &bonus);
    cur += l;
    // F2: Σ_e m[t,e] = k
    b.emit_rowsum(m, cur, e, &ones, ell);
    b.add_rhs(cur, &[(0, t, k)]);
    cur += t;
    // Fd: d − q̃ + τ(broadcast over E) = 0
    b.emit_id(d, cur, 1, ell);
    b.emit_id(qt, cur, neg1, ell);
    b.emit_stride_o2m(tau, cur, e, 1, ell);
    cur += l;
    // Fv: v − 2·md + d = 0
    b.emit_id(v, cur, 1, ell);
    b.emit_id(md, cur, P - 2, ell);
    b.emit_id(d, cur, 1, ell);
    b.emit_quad(m, m, m, neg1, 0, l, ell);
    b.emit_quad(m, d, md, neg1, 0, l, ell);
}

// TopkSlotsClaim: the k-hot m split into k one-hot slot masks M_i, and the
// slot scores ss[t,i] = Σ_e M_i[t,e]·s[t,e] (token-major). Mirrors
// topk_routing.py slots_compile (design §3.3).
//   Sb Σ_i M_i − m = 0  [T·E] · Sc Σ_e M_i = 1  [k·T, slot-major i·T + t] ·
//   Sv Σ_e MS_i − ss = 0  [k·T, slot-major; ss transposed in]
// Quads: M_i·M_i = M_i for each i, then M_i·s = MS_i for each i.
fn compile_topk_slots(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let t = cl.scalar("T") as usize;
    let e = cl.scalar("E") as usize;
    let k = cl.scalar("k") as usize;
    let (m, s, ss) = (cl.var("m"), cl.var("s"), cl.var("ss"));
    let masks = cl.var_list("M");
    let prods = cl.var_list("MS");
    assert!(k >= 1 && masks.len() == k && prods.len() == k, "slots: k masks and k products");
    assert!(m.length == t * e && s.length == t * e && ss.length == t * k
            && masks.iter().chain(prods.iter()).all(|v| v.length == t * e),
            "slots: variable lengths");
    let neg1 = P - 1;
    let ones: Vec<u64> = vec![1; e];
    let base = b.nxt; b.nxt += t * e + 2 * k * t;
    let mut cur = base;
    for mi in &masks { b.emit_id(*mi, cur, 1, ell); }
    b.emit_id(m, cur, neg1, ell);
    cur += t * e;
    for (i, mi) in masks.iter().enumerate() { b.emit_rowsum(*mi, cur + i * t, e, &ones, ell); }
    b.add_rhs(cur, &[(0, k * t, 1)]);
    cur += k * t;
    for (i, x) in prods.iter().enumerate() { b.emit_rowsum(*x, cur + i * t, e, &ones, ell); }
    b.emit_transpose_o2m(ss, cur, t, k, 1, neg1, ell);
    for mi in &masks { b.emit_quad(*mi, *mi, *mi, neg1, 0, t * e, ell); }
    for (i, mi) in masks.iter().enumerate() { b.emit_quad(*mi, s, prods[i], neg1, 0, t * e, ell); }
}

// GateBracketClaim: w[t,i] = ⌊C·ss[t,i] / Z[t]⌋, Z = Σ_i ss[t,i]. Mirrors
// topk_routing.py bracket_compile (design §3.2).
//   Bz Σ_i ss − Z = 0  [T] · Bb Zb − Z(bcast) = 0  [T·k] ·
//   Bd wZ + rem − C·ss = 0  [T·k] · Bg gr − Zb + rem = −1  [T·k]
// Quad: w·Zb = wZ. rem, gr and w are ranged by composed WordExtractionClaims.
// Guard (fail closed): the division identity stays below P, so it is the
// integer one, and a negative Z − 1 − rem cannot sit in its window; z_bits and
// cs_bits are the statement's bounds on the scores (an upstream hypothesis).
fn compile_gate_bracket(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let t = cl.scalar("T") as usize;
    let k = cl.scalar("k") as usize;
    let c = cl.scalar("C") % P;
    let (zb_bits, cs_bits) = (cl.scalar("z_bits") as u32, cl.scalar("cs_bits") as u32);
    let (rem_bits, w_bits) = (cl.scalar("rem_bits") as u32, cl.scalar("w_bits") as u32);
    assert!(k >= 1 && zb_bits < 63 && cs_bits < 64 && rem_bits < 63 && w_bits < 63,
            "bracket: parameters");
    let pp = P as u128;
    assert!((1u128 << (w_bits + zb_bits)) + (1u128 << rem_bits) < pp && (1u128 << cs_bits) < pp
            && (1u128 << rem_bits) <= pp - (1u128 << rem_bits.max(zb_bits)) - 1,
            "bracket: guard: the division identity must stay below P");
    let (ss, z, zb) = (cl.var("ss"), cl.var("Z"), cl.var("Zb"));
    let (w, wz, rem, gr) = (cl.var("w"), cl.var("wZ"), cl.var("rem"), cl.var("gr"));
    assert!(ss.length == t * k && z.length == t, "bracket: variable lengths");
    let neg1 = P - 1;
    let ones: Vec<u64> = vec![1; k];
    let base = b.nxt; b.nxt += t + 3 * t * k;
    let mut cur = base;
    b.emit_rowsum(ss, cur, k, &ones, ell);
    b.emit_id(z, cur, neg1, ell);
    cur += t;
    b.emit_id(zb, cur, 1, ell);
    b.emit_stride_o2m(z, cur, k, neg1, ell);
    cur += t * k;
    b.emit_id(wz, cur, 1, ell);
    b.emit_id(rem, cur, 1, ell);
    b.emit_id(ss, cur, (P - c) % P, ell);
    cur += t * k;
    b.emit_id(gr, cur, 1, ell);
    b.emit_id(zb, cur, neg1, ell);
    b.emit_id(rem, cur, 1, ell);
    b.add_rhs(cur, &[(0, t * k, neg1)]);
    b.emit_quad(w, zb, wz, neg1, 0, t * k, ell);
}

// HeadInterleaveClaim: dst (T, H, w1 + w2) per head from a (T, H, w1) and b,
// per head (T, H, w2) or shared (T, w2) fanned out to the heads. Mirrors
// prover/head_interleave.py head_interleave_compile: one cid per dst slot,
// base + (t·H + h)·W + c; dst −1, a and b +1 at their positions; no quads.
fn compile_head_interleave(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let t = cl.scalar("T") as usize;
    let h = cl.scalar("H") as usize;
    let (w1, w2) = (cl.scalar("w1") as usize, cl.scalar("w2") as usize);
    let shared = cl.scalar("shared");
    let (a, bv, dst) = (cl.var("a"), cl.var("b"), cl.var("dst"));
    let w = w1 + w2;
    assert!(t >= 1 && h >= 1 && w1 >= 1 && w2 >= 1 && shared <= 1, "head_interleave: shape");
    assert!(a.length == t * h * w1 && dst.length == t * h * w
            && bv.length == if shared == 1 { t * w2 } else { t * h * w2 },
            "head_interleave: variable lengths");
    let base = b.nxt; b.nxt += dst.length;
    b.emit_id(dst, base, P - 1, ell);
    b.emit_block_stride(a, base, w1, w, 1, 0, 1, ell);
    if shared == 1 {
        b.emit_block_stride(bv, base + w1, w2, h * w, h, w, 1, ell);
    } else {
        b.emit_block_stride(bv, base + w1, w2, w, 1, 0, 1, ell);
    }
}

// SplitClaim: the ConcatClaim relation with the parts derived from the whole
// (topk_routing.py split_compile): whole·(−1) + parts at their offsets = 0.
fn compile_split(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let whole = cl.var("whole");
    let parts = cl.var_list("parts");
    let l = whole.length;
    let base = b.nxt; b.nxt += l;
    b.emit_id(whole, base, P - 1, ell);
    let mut off = 0usize;
    for p in parts {
        b.emit_id(p, base + off, 1, ell);
        off += p.length;
    }
    assert_eq!(off, l, "split parts must cover the whole");
}

// MaskedCombineClaim: y[t,:] = Σ_e m[t,e]·X_e[t,:]. Mirrors combine_compile.
//   G1 m_rep_e[t,j] − m[t,e] = 0  [E·T·F] (one TransposeO2m on m covers every
//      expert's pin block at base + e·T·F)
//   G2 Σ_e P_e − y = 0  [T·F]
//   Quads: m_rep_e · X_e = P_e, per expert in order.
fn compile_masked_combine(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let t = cl.scalar("T") as usize;
    let e_n = cl.scalar("E") as usize;
    let f_n = cl.scalar("F") as usize;
    let lf = t * f_n;
    let m = cl.var("m");
    let xs = cl.var_list("xs");
    let m_rep = cl.var_list("m_rep");
    let prods = cl.var_list("prods");
    let y = cl.var("y");
    let neg1 = P - 1;
    let base = b.nxt; b.nxt += e_n * lf + lf;
    for e in 0..e_n {
        b.emit_id(m_rep[e], base + e * lf, 1, ell);
    }
    b.emit_transpose_o2m(m, base, t, e_n, f_n, neg1, ell);
    let cur = base + e_n * lf;
    for e in 0..e_n {
        b.emit_id(prods[e], cur, 1, ell);
    }
    b.emit_id(y, cur, neg1, ell);
    for e in 0..e_n {
        b.emit_quad(m_rep[e], xs[e], prods[e], neg1, 0, lf, ell);
    }
}

// ConcatClaim: dst = srcs concatenated — dst Identity (−1) full-range,
// each src Identity at base + segment offset. Mirrors protocol _c_concat.
fn compile_concat(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let dst = cl.var("dst");
    let srcs = cl.var_list("srcs");
    let l: usize = dst.length;
    let base = b.nxt; b.nxt += l;
    b.emit_id(dst, base, P - 1, ell);
    let mut off = 0usize;
    for v in srcs {
        let vl = v.length;
        b.emit_id(v, base + off, 1, ell);
        off += vl;
    }
    assert_eq!(off, l, "concat segments must cover dst");
}

// FreivaldsCombineClaim: y = Σ_e m[t,e]·X_e[t,:] via the ρ-projected seam.
// Mirrors routing_claim.py fcombine_compile; ρ = op_vec(s_op, ci, "rho", F).
//   C1 s_em − Σ ρ·X_e [E·T] · C2 m_em pin [T·E] · C3 ms_tm pin [T·E] ·
//   C4 yr − Σ ρ·y [T] · C5 Σ ms_tm − yr [T] · Quad ms_em = m_em⊙s_em.
fn compile_freivalds_combine(cl: &Claim, ci: usize, s_op: &[u8], b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let t = cl.scalar("T") as usize;
    let e_n = cl.scalar("E") as usize;
    let f_n = cl.scalar("F") as usize;
    let (m, y) = (cl.var("m"), cl.var("y"));
    let xs = cl.var_list("xs");
    let (m_em, s_em) = (cl.var("m_em"), cl.var("s_em"));
    let (ms_em, ms_tm, yr) = (cl.var("ms_em"), cl.var("ms_tm"), cl.var("yr"));
    let neg1 = P - 1;
    let rho = op_vec(s_op, ci, "rho", f_n);
    let neg_rho: Vec<u64> = rho.iter().map(|&r| (P - r % P) % P).collect();
    let ones: Vec<u64> = vec![1; e_n];
    let base = b.nxt; b.nxt += e_n * t + 2 * t * e_n + 2 * t;
    let mut cur = base;
    // C1
    b.emit_id(s_em, cur, 1, ell);
    for e in 0..e_n {
        b.emit_rowsum(xs[e], cur + e * t, f_n, &neg_rho, ell);
    }
    cur += e_n * t;
    // C2
    b.emit_id(m_em, cur, 1, ell);
    b.emit_transpose_o2m(m, cur, t, e_n, 1, neg1, ell);
    cur += t * e_n;
    // C3
    b.emit_id(ms_tm, cur, 1, ell);
    b.emit_transpose_o2m(ms_em, cur, e_n, t, 1, neg1, ell);
    cur += t * e_n;
    // C4
    b.emit_id(yr, cur, 1, ell);
    b.emit_rowsum(y, cur, f_n, &neg_rho, ell);
    cur += t;
    // C5
    b.emit_rowsum(ms_tm, cur, e_n, &ones, ell);
    b.emit_id(yr, cur, neg1, ell);
    // quad
    b.emit_quad(m_em, s_em, ms_em, neg1, 0, e_n * t, ell);
}

// PairedTlookupClaim: y = T_Y[x+shift].  Mirrors paired_tlookup_compile.
//   Linear: u - x - beta*y = shift   (cids [base, base+L))
//   Quad:   (alpha - u)*z = 1
fn compile_paired(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let l = cl.scalar("length") as usize;
    let x = cl.var("x");
    let y = cl.var("y");
    let u = cl.var("u");
    let z = cl.var("z");
    let table = cl.table("table");
    let shift = cl.scalar("shift") % P;
    let neg1 = P - 1;
    let neg_beta = (P - table.beta % P) % P;
    let base = b.nxt;
    b.emit_id(u, base, 1, ell);
    b.emit_id(x, base, neg1, ell);
    b.emit_id(y, base, neg_beta, ell);
    if shift != 0 {
        b.add_rhs(base, &[(0, l, shift)]);
    }
    b.nxt = base + l;
    let neg_alpha = (P - table.alpha) % P;
    b.emit_quad(u, z, z, neg_alpha, neg1, l, ell);
}

// InfoFinalizeClaim: a=Sum_i e, a<=pw (log-pin), z_o=ceil(gap_o2/k), surprisal=z_o+b.
// Mirrors info_compile in ui_claim.py. Linear families (each advance b.nxt by T):
//   a=Sum e [T], a+d-pw=0 [T], d-Sum 2^(wb j) dw_j=0 [T], k z_o-gap_o2-rem=0 [T],
//   surprisal-z_o-b=0 [T]. Quads: (alpha_wd-dw_j) z_dw_j=1, (alpha_k-rem) z_rem=1.
fn compile_info(cl: &Claim, b: &mut Build, cfg: &Config) {
    let ell = cfg.ell as usize;
    let t = cl.scalar("T") as usize;
    let v = cl.scalar("V") as usize;
    let k = cl.scalar("k");
    let wb = cl.scalar("wb") as usize;
    let e = cl.var("e");
    let pw = cl.var("pw");
    let gap_o2 = cl.var("gap_o2");
    let bb = cl.var("b");
    let a = cl.var("a");
    let d = cl.var("d");
    let dw = cl.var_list("dw");
    let z_o = cl.var("z_o");
    let rem = cl.var("rem");
    let surprisal = cl.var("surprisal");
    let z_dw = cl.var_list("z_dw");
    let z_rem = cl.var("z_rem");
    let zw = cl.var_list("zw");
    let z_zw = cl.var_list("z_zw");
    let range_wd = cl.table("range_wd");
    let range_k = cl.table("range_k");
    // The public size bound (protocol review F05/F06): z_o is ranged below
    // 2^(wb·|zw|) by its words and b below K by the Pow lookup, so the
    // cross-position sum of T positions cannot wrap the field.
    let k_pow = cl.scalar("K") as u128;
    assert_eq!(range_wd.t_len, 1usize << wb, "info: range_wd is not the 2^wb table");
    assert!(!zw.is_empty() && zw.len() == z_zw.len(), "info: z_o words missing");
    let z_o_max = 1u128 << (wb * zw.len());
    assert!((t as u128) * (z_o_max + k_pow) < P as u128,
            "info: surprisal sum over T={t} positions could wrap the field");
    let neg1 = P - 1;
    let ones = vec![1u64; v];
    let base = b.nxt;
    // a = Sum_i e[t,i]
    b.emit_rowsum(e, base, v, &ones, ell);
    b.emit_id(a, base, neg1, ell);
    // a + d - pw = 0
    let pin = base + t;
    b.emit_id(a, pin, 1, ell);
    b.emit_id(d, pin, 1, ell);
    b.emit_id(pw, pin, neg1, ell);
    // d - Sum_j 2^(wb*j) dw_j = 0
    let dec = base + 2 * t;
    b.emit_id(d, dec, 1, ell);
    for (j, dwj) in dw.iter().enumerate() {
        let coef = (P - ((1u64 << (wb * j)) % P)) % P;        // -2^(wb*j)
        b.emit_id(*dwj, dec, coef, ell);
    }
    // k*z_o - gap_o2 - rem = 0
    let ceil = base + 3 * t;
    b.emit_id(z_o, ceil, k % P, ell);
    b.emit_id(gap_o2, ceil, neg1, ell);
    b.emit_id(rem, ceil, neg1, ell);
    // z_o - Sum_j 2^(wb*j) zw_j = 0
    let zdec = base + 4 * t;
    b.emit_id(z_o, zdec, 1, ell);
    for (j, zwj) in zw.iter().enumerate() {
        let coef = (P - ((1u64 << (wb * j)) % P)) % P;        // -2^(wb*j)
        b.emit_id(*zwj, zdec, coef, ell);
    }
    // surprisal - z_o - b = 0
    let sur = base + 5 * t;
    b.emit_id(surprisal, sur, 1, ell);
    b.emit_id(z_o, sur, neg1, ell);
    b.emit_id(bb, sur, neg1, ell);
    b.nxt = base + 6 * t;
    // range LogUp quads: (alpha - x)*z = 1
    let na_wd = (P - range_wd.alpha) % P;
    for (dwj, zj) in dw.iter().zip(z_dw.iter()) {
        b.emit_quad(*dwj, *zj, *zj, na_wd, neg1, t, ell);
    }
    let na_k = (P - range_k.alpha) % P;
    b.emit_quad(rem, z_rem, z_rem, na_k, neg1, t, ell);
    for (zwj, zj) in zw.iter().zip(z_zw.iter()) {
        b.emit_quad(*zwj, *zj, *zj, na_wd, neg1, t, ell);
    }
}

fn settle_table(t: &Table, b: &mut Build, ell: usize) {
    let t_len = t.t_len;
    // T[j] = j for a range table (reconstructed, not transmitted); v = T or T+β·T_Y.
    let v: Vec<u64> = if let Some(ty) = &t.t_y {
        (0..t_len).map(|j| add(t.t_at(j), mul(t.beta, ty[j]))).collect()
    } else {
        (0..t_len).map(|j| t.t_at(j)).collect()
    };
    let w_coef: Vec<u64> = (0..t_len).map(|j| sub(t.alpha, v[j] % P)).collect();
    let base = b.nxt;
    let sum_cid = base + t_len;
    b.nxt += t_len + 1;
    // Spanning Weighted: cid_base = base, full coef vector (length t_len). emit
    // windows it per row (col = col-in-row, cid = base + flat_lo + col).
    b.families.push(Family {
        row_start: t.w_var.row_start, length: t.w_var.length, ell,
        exp: Expander::Weighted { cid_base: base, coefs: w_coef.clone() } });
    b.emit_id(t.mult_var, base, P - 1, ell);
    // Sum identity: each z (+1) and w (−1) collapse onto sum_cid. Both coef
    // vectors are constant, so RowsumConst stores just the scalar — never an
    // O(t_len) vec (which, cloned per row for a 2^24 table, was ~274 GB).
    for z in &t.z_vars {
        b.push_family(*z, ell, Expander::RowsumConst { cid_base: sum_cid, stride: z.length, coef: 1 });
    }
    b.push_family(t.w_var, ell, Expander::RowsumConst { cid_base: sum_cid, stride: t_len, coef: P - 1 });
}

/// m_total: highest witness row + 1, over all vars reachable from claims.
fn m_total(cs: &ClaimSet) -> usize {
    let ell = cs.cfg.ell as usize;
    let mut top = NUM_BLINDING_ROWS - 1;
    let mut bump = |v: Var| {
        if v.row_start >= usize::MAX / 2 { return; }   // external: no rows
        let t = v.row_start + nrows(v.length, ell) - 1; if t > top { top = t; } };
    for cl in &cs.claims {
        for v in cl.all_vars() { bump(v); }
    }
    top + 1
}

/// Compile the public claims into the constraint families (+ the small rhs and
/// quadratic sides). Each (variable, role) becomes one spanning `Family`; the
/// verifier's row fold windows them. Bounded memory — no per-row materialization.
pub fn compile_claims(cs: &mut ClaimSet, s_op: &[u8]) -> Constraints {
    compile_claims_bound(cs, s_op, None)
}

/// `s_bind` is the R2 coin. Claims with a phase-3 stage (RoutedProjected)
/// derive their late challenges from it; passing None is only valid for a
/// claim set that has none, and such a claim panics rather than silently
/// compiling a relation the transcript never bound.
pub fn compile_claims_bound(cs: &mut ClaimSet, s_op: &[u8], s_bind: Option<&[u8]>) -> Constraints {
    compile_claims_bound_pinned(cs, s_op, s_bind, Vec::new())
}

/// compile_claims_bound with the WC-LCRL-STC bridge pins: per use_bridge
/// claim, its bridge-authenticated P_trace slice.
pub fn compile_claims_bound_pinned(cs: &mut ClaimSet, s_op: &[u8], s_bind: Option<&[u8]>,
                                   wc_pins: Vec<(usize, Vec<u64>)>) -> Constraints {
    let cfg = cs.cfg;
    let ell = cfg.ell as usize;
    let mt = m_total(cs);
    let mut b = Build { families: Vec::new(), rhs: Vec::new(), quad: Vec::new(), nxt: 0, nq: 0,
                        wc_pins: wc_pins.into_iter().collect() };
    let n_ops = cs.claims.len();

    // Derive each table's α/β from s_op by settled-list index (after ops), then
    // PATCH every parsed table field so the handlers read the right values
    // (the JSON's α/β are stale). Keyed by table id (= mult_var.row_start).
    let mut ab: std::collections::HashMap<usize, (u64, u64)> = Default::default();
    {
        let by_id = cs.tables_by_id();
        for (k, &tid) in cs.table_order.iter().enumerate() {
            let paired = by_id[&tid].t_y.is_some();
            let alpha = challenge(s_op, 0, &format!("op{}:alpha", n_ops + k));
            let beta = if paired { challenge(s_op, 0, &format!("op{}:beta", n_ops + k)) } else { 0 };
            ab.insert(tid, (alpha, beta));
        }
    }
    for cl in &mut cs.claims { cl.patch_table_ab(&ab); }

    // Pass 1: operations.
    for ci in 0..cs.claims.len() {
        let cl = &cs.claims[ci];
        compile_op(cl, ci, s_op, s_bind, &mut b, &cfg);
    }
    // Pass 2: settle tables in table_order.
    let by_id = cs.tables_by_id();
    for &tid in &cs.table_order {
        settle_table(&by_id[&tid], &mut b, ell);
    }
    Constraints { families: b.families, rhs: b.rhs, quadratic: b.quad, m_total: mt }
}

#[cfg(test)]
mod rope_scaling_tests {
    //! Golden vectors shared bit-for-bit with the prover's
    //! tests/test_rope_scaling.py (generated from the Python implementation).
    //! If either side's table computation drifts, this fails here instead of
    //! rejecting whole proofs with no diagnostic. Config A (d_h=8) exercises
    //! all three ramp branches; config B is the Llama-3.2-1B head dim.
    use super::rope_cos_sin;
    use crate::field::P;

    // factor, low_freq_factor, high_freq_factor, original_max_pos
    const SC: (f64, f64, f64, f64) = (32.0, 1.0, 4.0, 8192.0);

    const A_COS: [u64; 12] = [4096, 4096, 4096, 4096, 2213, 4093, 4096, 4096,
                              18446744069414582616, 4084, 4096, 4096];
    const A_SIN: [u64; 12] = [0, 0, 0, 0, 3447, 154, 2, 0, 3724, 308, 4, 0];
    const B_COS: [u64; 32] = [18446744069414580266, 18446744069414582651, 1012,
                              2620, 3422, 3795, 3962, 4037, 4070, 4085, 4091,
                              4094, 4095, 4096, 4096, 4096, 4096, 4096, 4096,
                              4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
                              4096, 4096, 4096, 4096, 4096];
    const B_SIN: [u64; 32] = [578, 3740, 3969, 3148, 2251, 1542, 1038, 693,
                              461, 306, 203, 135, 90, 59, 39, 16, 5, 1, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    #[test]
    fn golden_vectors_match_python() {
        let (c, s) = rope_cos_sin(3, 8, 4096, 500000.0, 0, SC.0, SC.1, SC.2, SC.3, None);
        assert_eq!(c, A_COS.to_vec(), "config A cos drifted");
        assert_eq!(s, A_SIN.to_vec(), "config A sin drifted");
        let (c, s) = rope_cos_sin(1, 64, 4096, 500000.0, 3, SC.0, SC.1, SC.2, SC.3, None);
        assert_eq!(c, B_COS.to_vec(), "config B cos drifted");
        assert_eq!(s, B_SIN.to_vec(), "config B sin drifted");
    }

    #[test]
    fn unscaled_path_is_legacy() {
        // original_max_pos=0 disables the ramp: must equal the ORIGINAL
        // p/denominator expression byte-for-byte (old dumps verify unchanged).
        let (seq, d_h, s_x, base, off) = (5usize, 8usize, 4096u64, 10000.0f64, 2usize);
        let (c, s) = rope_cos_sin(seq, d_h, s_x, base, off, 1.0, 1.0, 1.0, 0.0, None);
        let (mut ec, mut es) = (Vec::new(), Vec::new());
        for sq in 0..seq {
            let p = (sq + off) as f64;
            for k in 0..d_h / 2 {
                let theta = p / base.powf(2.0 * k as f64 / d_h as f64);
                ec.push(((theta.cos() * s_x as f64).round() as i128).rem_euclid(P as i128) as u64);
                es.push(((theta.sin() * s_x as f64).round() as i128).rem_euclid(P as i128) as u64);
            }
        }
        assert_eq!(c, ec);
        assert_eq!(s, es);
    }
}

#[cfg(test)]
mod topk_tests {
    //! The top-k routing compiles (prover/topk_routing.py twins): the guards
    //! fail closed on the design's counterexample parameters, and the cid
    //! layout, RHS runs and quad count are the ones the prover emits.
    use super::*;
    use crate::claim::parse_claim_set_value;
    use serde_json::{json, Value};

    fn var(row: usize, len: usize) -> Value { json!({"var": [row, len]}) }
    fn sc(x: u64) -> Value { json!({"scalar": x}) }

    fn build() -> Build {
        Build { families: Vec::new(), rhs: Vec::new(), quad: Vec::new(), nxt: 0, nq: 0,
                wc_pins: std::collections::HashMap::new() }
    }

    fn claim_set(op: &str, fields: Value) -> ClaimSet {
        parse_claim_set_value(json!({
            "cfg": {"ELL": 8, "K_DEG": 8, "N_LIG": 32, "T_QUERIES": 4},
            "table_order": [],
            "claims": [{"op": op, "fields": fields}],
        }))
    }

    fn topk(width: u64, range_bits: u64) -> ClaimSet {
        let (t, e) = (2usize, 8usize);
        let l = t * e;
        claim_set("TopkRoutingClaim", json!({
            "s": var(0, l), "b": var(2, e), "m": var(3, l), "qt": var(5, l),
            "tau": var(7, t), "d": var(8, l), "md": var(10, l), "v": var(12, l),
            "T": sc(t as u64), "E": sc(e as u64), "k": sc(3), "L_bits": sc(3),
            "width": sc(width), "range_bits": sc(range_bits),
        }))
    }

    #[test]
    #[should_panic(expected = "threshold guard")]
    fn the_counterexample_parameters_are_refused() {
        // B_q = 40, L = 3: width 43; three 21-bit words: R = 63. Top-1's guard
        // (2^63 <= P - 2^43) would pass; the threshold guard must not.
        let cs = topk(43, 63);
        compile_topk_routing(&cs.claims[0], &mut build(), &cs.cfg);
    }

    #[test]
    fn the_guard_admits_62_bits_and_the_layout_matches_the_prover() {
        let cs = topk(43, 62);
        compile_topk_routing(&cs.claims[0], &mut build(), &cs.cfg);
        let cs = topk(14, 16);
        let mut b = build();
        compile_topk_routing(&cs.claims[0], &mut b, &cs.cfg);
        let (t, l) = (2usize, 16usize);
        assert_eq!(b.nxt, 3 * l + t, "F1 + F2 + Fd + Fv cids");
        assert_eq!(b.families.len(), 3 + 1 + 3 + 3);
        assert_eq!(b.quad.len(), 2);
        // F1's bonus E-1-e over each expert's block of T cids (e = 7 is zero),
        // then F2's k over the T cardinality cids
        let mut want: Vec<(usize, usize, u64)> = (0..7).map(|e| (e * t, t, 7 - e as u64)).collect();
        want.push((l, t, 3));
        assert_eq!(b.rhs, want);
    }

    #[test]
    #[should_panic(expected = "bracket: guard")]
    fn an_unsound_bracket_is_refused() {
        let cs = claim_set("GateBracketClaim", json!({
            "ss": var(0, 6), "Z": var(1, 2), "Zb": var(2, 6), "w": var(3, 6),
            "wZ": var(4, 6), "rem": var(5, 6), "gr": var(6, 6),
            "T": sc(2), "k": sc(3), "C": sc(11579), "z_bits": sc(40), "cs_bits": sc(30),
            "rem_bits": sc(40), "w_bits": sc(32),
        }));
        compile_gate_bracket(&cs.claims[0], &mut build(), &cs.cfg);
    }

    #[test]
    fn the_bracket_slots_and_split_layouts() {
        let cs = claim_set("GateBracketClaim", json!({
            "ss": var(0, 6), "Z": var(1, 2), "Zb": var(2, 6), "w": var(3, 6),
            "wZ": var(4, 6), "rem": var(5, 6), "gr": var(6, 6),
            "T": sc(2), "k": sc(3), "C": sc(11579), "z_bits": sc(14), "cs_bits": sc(26),
            "rem_bits": sc(16), "w_bits": sc(16),
        }));
        let mut b = build();
        compile_gate_bracket(&cs.claims[0], &mut b, &cs.cfg);
        assert_eq!(b.nxt, 2 + 3 * 6);
        assert_eq!(b.rhs, vec![(2 + 6 + 6, 6, P - 1)]);
        assert_eq!(b.quad.len(), 1);

        let cs = claim_set("TopkSlotsClaim", json!({
            "m": var(0, 16), "s": var(2, 16), "ss": var(4, 6),
            "M": {"list": [var(5, 16), var(7, 16), var(9, 16)]},
            "MS": {"list": [var(11, 16), var(13, 16), var(15, 16)]},
            "T": sc(2), "E": sc(8), "k": sc(3),
        }));
        let mut b = build();
        compile_topk_slots(&cs.claims[0], &mut b, &cs.cfg);
        assert_eq!(b.nxt, 16 + 2 * 3 * 2);
        assert_eq!(b.rhs, vec![(16, 6, 1)]);
        assert_eq!(b.quad.len(), 6);

        let cs = claim_set("SplitClaim", json!({
            "whole": var(0, 12), "parts": {"list": [var(2, 4), var(3, 4), var(4, 4)]},
        }));
        let mut b = build();
        compile_split(&cs.claims[0], &mut b, &cs.cfg);
        assert_eq!((b.nxt, b.families.len()), (12, 4));
    }
}

#[cfg(test)]
mod head_interleave_tests {
    //! The compiled HeadInterleaveClaim maps every dst slot to exactly one
    //! source slot, per head and shared (prover/tests/test_head_interleave_layout.py
    //! checks the same map through the prover's descriptor lowering).
    use super::*;
    use crate::claim::parse_claim_set_value;
    use serde_json::json;
    use std::collections::BTreeMap;

    const T: usize = 2; const H: usize = 3; const W1: usize = 2; const W2: usize = 1;

    fn compiled(shared: bool) -> Build {
        let blen = if shared { T * W2 } else { T * H * W2 };
        let cs = parse_claim_set_value(json!({
            "cfg": {"ELL": 4, "K_DEG": 8, "N_LIG": 32, "T_QUERIES": 4},
            "table_order": [],
            "claims": [{"op": "HeadInterleaveClaim", "fields": {
                "a": {"var": [0, T * H * W1]}, "b": {"var": [10, blen]},
                "dst": {"var": [20, T * H * (W1 + W2)]},
                "T": {"scalar": T}, "H": {"scalar": H}, "w1": {"scalar": W1},
                "w2": {"scalar": W2}, "shared": {"scalar": shared as u64}}}]}));
        let mut b = Build { families: Vec::new(), rhs: Vec::new(), quad: Vec::new(), nxt: 0,
                            nq: 0, wc_pins: std::collections::HashMap::new() };
        compile_head_interleave(&cs.claims[0], &mut b, &cs.cfg);
        b
    }

    // cid -> [(variable row_start, slot, coef)]
    fn map(b: &Build) -> BTreeMap<usize, Vec<(usize, usize, u64)>> {
        let mut m: BTreeMap<usize, Vec<(usize, usize, u64)>> = BTreeMap::new();
        for fam in &b.families {
            fam.exp.emit(0, fam.length, &mut |slot, cid, coef| {
                m.entry(cid).or_default().push((fam.row_start, slot, coef));
            });
        }
        m
    }

    fn check(shared: bool) {
        let b = compiled(shared);
        let w = W1 + W2;
        assert_eq!(b.nxt, T * H * w);
        assert!(b.quad.is_empty() && b.rhs.is_empty());
        let m = map(&b);
        assert_eq!(m.len(), T * H * w);
        for t in 0..T { for h in 0..H { for c in 0..w {
            let cid = (t * H + h) * w + c;
            let src = if c < W1 { (0, (t * H + h) * W1 + c) }
                      else if shared { (10, t * W2 + c - W1) }
                      else { (10, (t * H + h) * W2 + c - W1) };
            let mut got = m[&cid].clone(); got.sort();
            let mut want = vec![(20, cid, P - 1), (src.0, src.1, 1)]; want.sort();
            assert_eq!(got, want, "cid {cid}");
        }}}
    }

    #[test] fn per_head_maps_every_dst_slot_to_one_source() { check(false); }
    #[test] fn shared_fans_one_vector_to_every_head() { check(true); }
}

mod rope_yarn_tests {
    //! Golden vectors shared bit-for-bit with prover/tests/test_rope_yarn.py
    //! (generated from the Python implementation): Kimi K2's YaRN config at
    //! positions 1, 4,095 and 131,071, and a smooth-ramp config with a cos/sin
    //! factor other than 1. The compile-path test feeds the config as the
    //! prover serializes it, so the field names and defaults are pinned too.
    use super::*;
    use crate::claim::parse_claim_set_value;
    use serde_json::json;

    const K2_1_COS: [u64; 32] = [
        2213, 3098, 3578, 3830, 3960, 4027, 4061, 4078, 4087, 4091, 4094, 4095,
        4095, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
        4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
    ];
    const K2_1_SIN: [u64; 32] = [
        3447, 2680, 1994, 1453, 1047, 751, 537, 384, 274, 195, 139, 99, 71, 51, 36,
        26, 18, 13, 9, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const K2_4095_COS: [u64; 32] = [
        18446744069414584051, 325, 18446744069414580644, 18446744069414582028,
        18446744069414580353, 1563, 18446744069414583232, 3008,
        18446744069414580788, 3582, 2110, 1395, 18446744069414583784, 3983,
        18446744069414583848, 3496, 3521, 3608, 18446744069414580251, 3836, 4051,
        4073, 4084, 4090, 4093, 4094, 4095, 4096, 4096, 4096, 4096, 4096,
    ];
    const K2_4095_SIN: [u64; 32] = [
        18446744069414580234, 18446744069414580238, 1804, 3394,
        18446744069414583304, 3786, 18446744069414580372, 2780,
        18446744069414582248, 1986, 3511, 18446744069414580470, 4061, 955,
        18446744069414580252, 2135, 18446744069414582229, 1939, 457, 1435, 604, 431,
        308, 220, 157, 112, 80, 57, 41, 29, 21, 15,
    ];
    const K2_131071_COS: [u64; 32] = [
        18446744069414580971, 3630, 1607, 1625, 18446744069414580453, 3536,
        18446744069414583596, 18446744069414584008, 4012, 1578,
        18446744069414580833, 3150, 142, 18446744069414584096, 18446744069414580394,
        1817, 18446744069414583267, 18446744069414580284, 18446744069414580501,
        1993, 101, 18446744069414580339, 18446744069414581276, 18446744069414583721,
        1388, 2630, 3327, 3698, 3892, 3992, 4043, 4069,
    ];
    const K2_131071_SIN: [u64; 32] = [
        18446744069414581965, 18446744069414582424, 3768, 18446744069414580561,
        18446744069414582973, 18446744069414582254, 4031, 4084, 824,
        18446744069414580541, 2148, 18446744069414581703, 18446744069414580227,
        4090, 1166, 18446744069414580650, 3958, 18446744069414583626, 1478,
        18446744069414580743, 18446744069414580226, 18446744069414583362, 2740,
        4052, 3854, 3140, 2390, 1760, 1276, 917, 657, 469,
    ];
    const RAMP_COS: [u64; 12] = [
        5607, 5607, 5607, 5607, 3029, 5579, 5607, 5607, 18446744069414581988, 5495,
        5607, 5607,
    ];
    const RAMP_SIN: [u64; 12] = [
        0, 0, 0, 0, 4718, 560, 29, 0, 5098, 1114, 57, 0,
    ];

    fn k2() -> Yarn { Yarn { beta_fast: 1.0, beta_slow: 1.0, mscale: 1.0, mscale_all_dim: 1.0 } }

    #[test]
    fn golden_vectors_match_python() {
        for (off, c0, s0) in [(1usize, &K2_1_COS[..], &K2_1_SIN[..]),
                              (4095, &K2_4095_COS[..], &K2_4095_SIN[..]),
                              (131071, &K2_131071_COS[..], &K2_131071_SIN[..])] {
            let (c, s) = rope_cos_sin(1, 64, 4096, 50000.0, off, 32.0, 1.0, 1.0, 4096.0, Some(&k2()));
            assert_eq!(c, c0.to_vec(), "K2 cos at {} drifted", off);
            assert_eq!(s, s0.to_vec(), "K2 sin at {} drifted", off);
        }
        let ramp = Yarn { beta_fast: 32.0, beta_slow: 1.0, mscale: 1.0, mscale_all_dim: 0.0 };
        let (c, s) = rope_cos_sin(3, 8, 4096, 10000.0, 0, 40.0, 1.0, 1.0, 4096.0, Some(&ramp));
        assert_eq!(c, RAMP_COS.to_vec());
        assert_eq!(s, RAMP_SIN.to_vec());
    }

    #[test]
    fn a_halfway_entry_rounds_to_even_as_python_does() {
        // cos(0) · m · 4096 is exactly 5606.5 at this mscale; Python's round()
        // gives 5606, and so must the verifier (ties away from zero gave 5607)
        let half = Yarn { beta_fast: 32.0, beta_slow: 1.0, mscale: 0.9996922335080182,
                          mscale_all_dim: 0.0 };
        assert_eq!(1.0 * rope_yarn_mscale(40.0, &half) * 4096.0, 5606.5);
        let (c, s) = rope_cos_sin(1, 8, 4096, 10000.0, 0, 40.0, 1.0, 1.0, 4096.0, Some(&half));
        assert_eq!(c, vec![5606; 4]);
        assert_eq!(s, vec![0; 4]);
    }

    #[test]
    fn the_compile_path_reads_the_serialized_config() {
        let cs = parse_claim_set_value(json!({
            "cfg": {"ELL": 8, "K_DEG": 8, "N_LIG": 32, "T_QUERIES": 4},
            "table_order": [],
            "claims": [{"op": "RoPEClaim", "fields": {
                "x": {"var": [0, 64]}, "x_rot": {"var": [8, 64]},
                "rescale_bits": {"scalar": 0}, "output_width": {"scalar": 16},
                "config": {"config": {"SEQ": 1, "d_h": 64, "s_x": 4096, "base": 50000.0,
                    "position_offset": 131071, "heads": 1, "scale_factor": 32.0,
                    "low_freq_factor": 1.0, "high_freq_factor": 1.0, "original_max_pos": 4096,
                    "yarn": 1, "yarn_beta_fast": 1.0, "yarn_beta_slow": 1.0,
                    "yarn_mscale": 1.0, "yarn_mscale_all_dim": 1.0}}}}],
        }));
        let mut b = Build { families: Vec::new(), rhs: Vec::new(), quad: Vec::new(), nxt: 0,
                            nq: 0, wc_pins: std::collections::HashMap::new() };
        compile_op(&cs.claims[0], 0, &[], None, &mut b, &cs.cfg);
        let tables = b.families.iter().find_map(|f| match &f.exp {
            Expander::RopeX { cos, sin, .. } => Some((cos.to_vec(), sin.to_vec())),
            _ => None,
        }).expect("a RopeX family");
        assert_eq!(tables.0, K2_131071_COS.to_vec());
        assert_eq!(tables.1, K2_131071_SIN.to_vec());
    }
}
