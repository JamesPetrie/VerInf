// Copyright (c) 2026 Levi Neuwirth. Released under the MIT License (see LICENSE).
//! What a proof's shape must be before its six checks mean anything.
//!
//! 1. The test polynomials are bounded. `lin_sum` and `quad_zero` read q_lin
//!    and p_0 at the zeta points, and the column tests read them at the eta
//!    points. Degree bounds prevent distinct allowed polynomials from agreeing
//!    on the entire N-point column domain, so each test polynomial's length is
//!    checked against its limit, which must lie below N.
//!
//! 2. A row is committed before the challenges that depend on it. A Freivalds
//!    or LogUp identity is sound only if its operands were fixed before its
//!    coin was drawn, and the coins are drawn from the roots of the blocks
//!    before them. The statement fixes where each variable's rows are; this
//!    checks that those rows fall inside the blocks committed early enough.
//!
//! 3. A block holds the rows the statement assumes. The statement names rows
//!    by position in the joint column, and block sizes come from the proof, so
//!    each block's row count is checked against the statement, and every block
//!    must have a root.

use std::collections::HashMap;

use crate::claim::{Claim, ClaimSet, VarRole};
use crate::field::P;
use crate::protocol::{Config, NUM_BLINDING_ROWS};

/// The longest test polynomials, by coefficient count: degree below K for
/// q_irs, and below 2K for q_lin and p_0 because the blinding rows u_lin and
/// u_quad have that degree (prover/core.py, `_mix_blinding_into_tests`).
/// analysis/parameter-derivation.md's K + ELL - 2 and 2K - 2 leave blinding out.
pub fn test_poly_limits(cfg: &Config) -> (usize, usize, usize) {
    let k = cfg.k_deg as usize;
    (k, 2 * k, 2 * k)
}

/// Rule 1. A polynomial of degree below 2K can agree with a different one on
/// fewer than 2K of the N columns, so the limits only bite when 2K < N.
pub fn check_test_polys(q_irs: usize, q_lin: usize, p_0: usize, cfg: &Config) -> Result<(), String> {
    if cfg.k_deg >= cfg.n_lig.div_ceil(2) {
        return Err(format!("N = {} must exceed twice K = {}", cfg.n_lig, cfg.k_deg));
    }
    let (irs, lin, quad) = test_poly_limits(cfg);
    if q_irs > irs { return Err(format!("q_irs has {q_irs} coefficients, the limit is {irs}")); }
    if q_lin > lin { return Err(format!("q_lin has {q_lin} coefficients, the limit is {lin}")); }
    if p_0 > quad { return Err(format!("p_0 has {p_0} coefficients, the limit is {quad}")); }
    Ok(())
}

/// The Fiat-Shamir prover hashes a fixed-width coefficient vector for each
/// test polynomial. Shorter encodings of a zero high coefficient change s_col.
pub fn check_transcript_polys(irs: usize, lin: usize, quad: usize, cfg: &Config) -> Result<(), String> {
    check_test_polys(irs, lin, quad, cfg)?;
    if (irs, lin, quad) != test_poly_limits(cfg) {
        return Err("Fiat-Shamir coefficient counts must be K, 2K, 2K".into());
    }
    Ok(())
}

/// A polynomial written with a coefficient at or above P is the same polynomial
/// with other bytes in `s_col`, which would re-draw the columns for nothing.
pub fn is_canonical(coefficients: &[u64]) -> bool {
    coefficients.iter().all(|&c| c < P)
}

/// Each block's row count, from its opened columns. A leaf commits to one column's
/// bytes, but nothing makes a tree's columns agree in length, so that is checked here.
pub fn block_rows(opened: &[HashMap<u64, Vec<u64>>]) -> Result<Vec<usize>, String> {
    opened.iter().enumerate().map(|(b, columns)| {
        let mut lengths = columns.values().map(|c| c.len());
        let first = lengths.next().unwrap_or(0);
        if lengths.all(|n| n == first) { Ok(first) }
        else { Err(format!("block {b}'s opened columns have different lengths")) }
    }).collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Round { R1, R2, R3 }

/// The latest round a variable may be committed in. Everything is round 1 (the
/// blocks before p2, committed before `s_op`) unless it is listed here, so a
/// claim or field this table does not know is held to the strictest rule: a
/// missing entry rejects an honest proof, and never admits a malformed one.
///
/// Round 3 means no limit, since every block precedes `s_comb`. It holds what
/// the prover computes from `s_op`'s coins and what no later coin reads: LogUp's
/// inverses and running sums, a paired lookup's combined key, the Freivalds
/// projections. Each is pinned to its round-1 operands by an identity. The prover
/// keeps them in p2 (its `phase=2` variables in prover/tape.py, claims.py,
/// routing_claim.py, max_claim.py, ui_claim.py); prover/tests/test_phase3_block.py
/// moves a matmul's to p3.
///
/// Round 2 holds the routed claim's P, Q, H and yr, which precede `s_bind`, the
/// source of sigma and lambda. That claim's f_y, f_u and f_p come after it.
fn latest_round(op: &str, role: &VarRole) -> Round {
    let field = match role {
        VarRole::TableMult => return Round::R1,
        VarRole::TableW | VarRole::TableZ => return Round::R3,
        VarRole::Field(f) => *f,
    };
    let auxiliaries: &[&str] = match op {
        "FreivaldsCombineClaim" => &["s_em", "ms_em", "ms_tm", "yr"],
        "HadamardClaim" => &["z_c_low", "z_c_shifted"],
        "InfoFinalizeClaim" => &["z_dw", "z_rem", "z_zw"],
        "MatmulClaim" => &["y", "u", "p", "z_C_low", "z_C_shifted"],
        "MaxClaim" => &["z_gap"],
        "PairedTlookupClaim" => &["u", "z"],
        "RangeWordClaim" => &["z"],
        "RescaleClaim" => &["z_low", "z_shifted"],
        "RmsNormClaim" => &["u", "p", "z_lo_chunks", "z_hi_chunks", "z_ym1_chunks", "z_S_limbs",
                            "z_lo_gl", "z_lo_g0h", "z_lo_g1h", "z_lo_G2",
                            "z_hi_gl", "z_hi_g0h", "z_hi_g1h", "z_hi_G2",
                            "z_output_low", "z_output_shifted", "z_x_low", "z_x_shifted"],
        "RoPEClaim" => &["z_x_rot_low", "z_x_rot_shifted"],
        "RoutedProjectedMatmulClaim" => {
            return match field {
                "Pj" | "Qm" | "Hd" | "yr" => Round::R2,
                "f_y" | "f_u" | "f_p" => Round::R3,
                _ => Round::R1,
            };
        }
        "SiluClaim" => &["pt_u", "pt_z", "z_a0", "z_a1", "z_a2", "z_a3", "z_a4",
                         "z_x_low", "z_x_shifted"],
        "SoftmaxClaim" => &["pt_u_A", "pt_z_A", "pt_u_B", "pt_z_B",
                            "z_c2", "z_r_lo", "z_r_hi", "z_z_high",
                            "z_x_low", "z_x_shifted"],
        _ => &[],
    };
    if auxiliaries.contains(&field) { Round::R3 } else { Round::R1 }
}

/// Linear families walk a variable's declared rows. Quadratic families instead
/// walk the length supplied by their handler (including the rescale gadget).
/// Check those slot counts before trusting the end row computed from a Var.
fn check_operand_span(cl: &Claim, field: &str, available: usize) -> Result<(), String> {
    let dims: &[&str] = match (cl.op.as_str(), field) {
        ("RangeWordClaim", "x" | "z") | ("PairedTlookupClaim", "u" | "z") |
        ("HadamardClaim", "a" | "b" | "c" | "c_full" | "c_low" | "c_shifted" | "z_c_low" | "z_c_shifted") |
        ("RescaleClaim", "x_low" | "x_shifted" | "z_low" | "z_shifted") => &["length"],
        ("SiluClaim", "sign" | "x" | "C" | "inv_x" | "g" | "inv_g" | "is_high" | "y" |
            "mux_a" | "mux_b" | "output_sat" | "a_0" | "a_1" | "a_2" | "a_3" | "a_4" |
            "z_a0" | "z_a1" | "z_a2" | "z_a3" | "z_a4" | "pt_u" | "pt_z" |
            "x_low" | "x_shifted" | "z_x_low" | "z_x_shifted") => &["length"],
        ("MatmulClaim", "u" | "y" | "p") => &["k"],
        ("MatmulClaim", "C_low" | "C_shifted" | "z_C_low" | "z_C_shifted") => &["m", "heads", "n"],
        ("RoutedProjectedMatmulClaim", "X" | "Qm" | "Hd") => &["T", "K"],
        ("RoutedProjectedMatmulClaim", "f_u" | "f_y" | "f_p") => &["E"],
        ("RoPEClaim", "x_rot_low" | "x_rot_shifted" | "z_x_rot_low" | "z_x_rot_shifted") =>
            &["SEQ", "heads", "d_h"],
        ("RmsNormClaim", "x" | "X_sq" | "x_low" | "x_shifted" | "z_x_low" | "z_x_shifted" |
            "output_low" | "output_shifted" | "z_output_low" | "z_output_shifted") => &["B", "d"],
        ("RmsNormClaim", "y" | "u" | "p" | "y_m1" | "q1" | "q2" | "S_limbs" | "lo_H" | "hi_H" |
            "s_lo_chunks" | "s_hi_chunks" | "ym1_chunks" | "lo_gl" | "hi_gl" |
            "lo_g0h_chunks" | "lo_g1h_chunks" | "lo_G2_chunks" | "hi_g0h_chunks" |
            "hi_g1h_chunks" | "hi_G2_chunks" | "z_lo_chunks" | "z_hi_chunks" | "z_ym1_chunks" |
            "z_S_limbs" | "z_lo_gl" | "z_hi_gl" | "z_lo_g0h" | "z_lo_g1h" | "z_lo_G2" |
            "z_hi_g0h" | "z_hi_g1h" | "z_hi_G2") => &["B"],
        ("SoftmaxClaim", "c2_shifted" | "z_c2" | "r_lo" | "z_r_lo" | "r_hi" | "z_r_hi") => &["B"],
        ("SoftmaxClaim", "pt_u_A" | "pt_z_A" | "pt_u_B" | "pt_z_B" | "z_high" |
            "inv_z_high" | "is_high" | "y_A_raw" | "y_B_raw" | "mux_y_A" | "mux_y_B" |
            "z_z_high" | "x_low" | "x_shifted" | "z_x_low" | "z_x_shifted") => &["B", "M"],
        ("MaxClaim", "A" | "l" | "Al" | "O" | "gap" | "Ogap" | "z_gap") => &["T", "V"],
        ("RoutingClaim", "m" | "rt" | "mrt") | ("TopkRoutingClaim", "m" | "d" | "md") |
        ("TopkSlotsClaim", "M" | "s" | "MS") |
        ("FreivaldsCombineClaim", "m_em" | "s_em" | "ms_em") => &["T", "E"],
        ("GateBracketClaim", "w" | "Zb" | "wZ") => &["T", "k"],
        ("MaskedCombineClaim", "m_rep" | "xs" | "prods") => &["T", "F"],
        ("InfoFinalizeClaim", "dw" | "z_dw" | "rem" | "z_rem" | "zw" | "z_zw") => &["T"],
        _ => return Ok(()),
    };
    let mut needed = 1usize;
    for &dim in dims {
        let size = match cl.op.as_str() {
            "RmsNormClaim" | "SoftmaxClaim" | "RoPEClaim" => cl.config_dimension(dim),
            _ => cl.opt_scalar(dim).and_then(|v| usize::try_from(v).ok()),
        }.ok_or_else(|| format!("missing or invalid dimension {dim} for {field}"))?;
        needed = needed.checked_mul(size).ok_or_else(|| format!("span overflows for {field}"))?;
    }
    if needed > available {
        return Err(format!("{field} needs {needed} slots but declares {available}"));
    }
    Ok(())
}

/// Rule 3. The prover's order is blind, w, wnew, p1, p2, p3; a statement with no
/// `blind` holds the blinding rows in p1 (the legacy corpus). The blinding block
/// fixes where `w` starts, the trusted root fixes how long `w` is, and `wnew` must
/// match it. Returns where p2 is.
pub fn check_block_order(block_order: &[String]) -> Result<usize, String> {
    let names: Vec<&str> = block_order.iter().map(String::as_str).collect();
    let weights = names.iter().position(|b| *b == "p1").unwrap_or(names.len());
    let known = matches!(names[..weights], [] | ["blind"] | ["blind", "w"] | ["blind", "w", "wnew"])
        && matches!(names[weights..], ["p1", "p2"] | ["p1", "p2", "p3"]);
    if !known {
        return Err(format!("the block order {} is not the protocol's", names.join(",")));
    }
    Ok(weights + 1)
}

fn check_blocks(block_order: &[String], rows: &[usize]) -> Result<usize, String> {
    if rows.len() != block_order.len() {
        return Err(format!("the proof opens {} blocks and the statement names {}",
                           rows.len(), block_order.len()));
    }
    let p2 = check_block_order(block_order)?;
    if block_order[0] == "blind" && rows[0] != NUM_BLINDING_ROWS {
        return Err(format!("the blind block holds {} rows, not {NUM_BLINDING_ROWS}", rows[0]));
    }
    if block_order.iter().any(|b| b == "wnew") && rows[1] != rows[2] {
        return Err(format!("w holds {} rows and wnew {}", rows[1], rows[2]));
    }
    Ok(p2)
}

/// Rules 2 and 3, for a statement already trusted. `block_order` is the
/// statement's, and `rows` holds each block's row count, in that order.
pub fn check_rows(cs: &ClaimSet, block_order: &[String], rows: &[usize]) -> Result<(), String> {
    let p2 = check_blocks(block_order, rows)?;
    let r1_end: usize = rows[..p2].iter().sum();
    let r2_end = r1_end + rows[p2];
    let total: usize = rows.iter().sum();

    let ell = cs.cfg.ell as usize;
    if ell == 0 { return Err("ELL must be positive".into()); }
    let mut top = NUM_BLINDING_ROWS;
    for (ci, cl) in cs.claims.iter().enumerate() {
        for (role, v) in cl.named_vars() {
            let name = match &role {
                VarRole::Field(f) => (*f).to_string(),
                VarRole::TableMult => "a table's mult_var".to_string(),
                VarRole::TableW => "a table's w_var".to_string(),
                VarRole::TableZ => "a table's z_vars".to_string(),
            };
            if v.row_start >= usize::MAX / 2 {
                // External: held by the bridge, with no rows here. Only a
                // bridged claim's weights may be.
                let bridged_w = cl.op == "RoutedProjectedMatmulClaim"
                    && matches!(role, VarRole::Field("W"))
                    && cl.opt_scalar("use_bridge") == Some(1);
                if !bridged_w {
                    return Err(format!("claim {ci} ({}): {name} has no rows", cl.op));
                }
                continue;
            }
            if let VarRole::Field(field) = role {
                check_operand_span(cl, field, v.length).map_err(|e| format!("claim {ci} ({}): {e}", cl.op))?;
            }
            let end = v.row_start.checked_add(v.length.div_ceil(ell))
                .ok_or_else(|| format!("claim {ci} ({}): {name} end row overflows", cl.op))?;
            top = top.max(end);
            let (limit, round) = match latest_round(&cl.op, &role) {
                Round::R1 => (r1_end, 1),
                Round::R2 => (r2_end, 2),
                Round::R3 => (total, 3),
            };
            if end > limit {
                return Err(format!(
                    "claim {ci} ({}): {name} ends at row {end}, past the end of round {round} at row {limit}",
                    cl.op));
            }
        }
    }
    if total != top {
        return Err(format!("the blocks hold {total} rows and the statement {top}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config { Config { ell: 8, k_deg: 16, n_lig: 64, t_queries: 4 } }
    fn order(names: &[&str]) -> Vec<String> { names.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn test_polynomials_are_held_to_their_limits() {
        let c = cfg();
        assert_eq!(test_poly_limits(&c), (16, 32, 32));
        assert!(check_test_polys(16, 32, 32, &c).is_ok());
        assert!(check_test_polys(17, 32, 32, &c).is_err());
        assert!(check_test_polys(16, 33, 32, &c).is_err());
        assert!(check_test_polys(16, 32, 33, &c).is_err());
        let tight = Config { n_lig: 32, ..cfg() };
        assert!(check_test_polys(1, 1, 1, &tight).is_err(), "2K must stay below N");
    }

    #[test]
    fn coefficients_must_be_canonical() {
        assert!(is_canonical(&[0, 1, P - 1]));
        assert!(!is_canonical(&[0, P]));
    }

    #[test]
    fn a_block_s_opened_columns_agree_in_length() {
        let same: HashMap<u64, Vec<u64>> = [(0, vec![1, 2]), (5, vec![3, 4])].into();
        let ragged: HashMap<u64, Vec<u64>> = [(0, vec![1, 2]), (5, vec![3])].into();
        assert_eq!(block_rows(&[same.clone()]).unwrap(), vec![2]);
        assert!(block_rows(&[same, ragged]).is_err());
    }

    #[test]
    fn only_the_protocol_s_block_orders_and_sizes_are_accepted() {
        let b = NUM_BLINDING_ROWS;
        assert_eq!(check_blocks(&order(&["blind", "p1", "p2"]), &[b, 4, 2]).unwrap(), 2);
        assert_eq!(check_blocks(&order(&["blind", "w", "p1", "p2", "p3"]), &[b, 5, 4, 2, 1]).unwrap(), 3);
        assert_eq!(check_blocks(&order(&["blind", "w", "wnew", "p1", "p2"]), &[b, 5, 5, 4, 2]).unwrap(), 4);
        assert_eq!(check_blocks(&order(&["p1", "p2"]), &[4, 2]).unwrap(), 1);
        assert!(check_blocks(&order(&["blind", "p1", "p2"]), &[b, 4]).is_err(), "count must match");
        assert!(check_blocks(&order(&["blind", "p2", "p1"]), &[b, 4, 2]).is_err(), "order is fixed");
        assert!(check_blocks(&order(&["w", "blind", "p1", "p2"]), &[5, b, 4, 2]).is_err());
        assert!(check_blocks(&order(&["blind", "p1", "p2"]), &[b + 1, 4, 2]).is_err(), "blind size is fixed");
        assert!(check_blocks(&order(&["blind", "w", "wnew", "p1", "p2"]), &[b, 5, 4, 4, 2]).is_err(),
                "wnew must match w");
    }
}
