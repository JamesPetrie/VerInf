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

use crate::claim::{ClaimSet, VarRole};
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
    let (irs, lin, quad) = test_poly_limits(cfg);
    if 2 * (cfg.k_deg as usize) >= cfg.n_lig as usize {
        return Err(format!("N = {} leaves no room above 2K = {}", cfg.n_lig, 2 * cfg.k_deg));
    }
    if q_irs > irs { return Err(format!("q_irs has {q_irs} coefficients, the limit is {irs}")); }
    if q_lin > lin { return Err(format!("q_lin has {q_lin} coefficients, the limit is {lin}")); }
    if p_0 > quad { return Err(format!("p_0 has {p_0} coefficients, the limit is {quad}")); }
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

/// Rule 3. The prover's order is blind, w, wnew, p1, p2, p3; a statement with no
/// `blind` holds the blinding rows in p1 (the legacy corpus). The blinding block
/// fixes where `w` starts, the trusted root fixes how long `w` is, and `wnew` must
/// match it. Returns where p2 is.
fn check_blocks(block_order: &[String], rows: &[usize]) -> Result<usize, String> {
    if rows.len() != block_order.len() {
        return Err(format!("the proof opens {} blocks and the statement names {}",
                           rows.len(), block_order.len()));
    }
    let names: Vec<&str> = block_order.iter().map(String::as_str).collect();
    let weights = names.iter().position(|b| *b == "p1").unwrap_or(names.len());
    let known = matches!(names[..weights], [] | ["blind"] | ["blind", "w"] | ["blind", "w", "wnew"])
        && matches!(names[weights..], ["p1", "p2"] | ["p1", "p2", "p3"]);
    if !known {
        return Err(format!("the block order {} is not the protocol's", names.join(",")));
    }
    if names[0] == "blind" && rows[0] != NUM_BLINDING_ROWS {
        return Err(format!("the blind block holds {} rows, not {NUM_BLINDING_ROWS}", rows[0]));
    }
    if weights == 3 && rows[1] != rows[2] {
        return Err(format!("w holds {} rows and wnew {}", rows[1], rows[2]));
    }
    Ok(weights + 1)
}

/// Rules 2 and 3, for a statement already trusted. `block_order` is the
/// statement's, and `rows` holds each block's row count, in that order.
pub fn check_rows(cs: &ClaimSet, block_order: &[String], rows: &[usize]) -> Result<(), String> {
    let p2 = check_blocks(block_order, rows)?;
    let r1_end: usize = rows[..p2].iter().sum();
    let r2_end = r1_end + rows[p2];
    let total: usize = rows.iter().sum();

    let ell = cs.cfg.ell as usize;
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
            let end = v.row_start + v.length.div_ceil(ell);
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
