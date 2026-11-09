// Copyright (c) 2026 Levi Neuwirth. Released under the MIT License (see LICENSE).
//! The whole round table, as a copy to change on purpose: which round each field
//! of each claim may come in, read from the prover's phases and from the saved
//! full-scale statements.
use ligero_verifier::claim::parse_claim_set;
use ligero_verifier::layout::check_rows;
use serde_json::json;

/// The latest round this field passes in: a statement with the field at row 3
/// (round 1), 4 (round 2) or 5 (round 3).
fn latest(op: &str, field: &str) -> u8 {
    let order: Vec<String> = ["blind", "p1", "p2", "p3"].iter().map(|s| s.to_string()).collect();
    let mut best = 0;
    for (row, round) in [(3u64, 1u8), (4, 2), (5, 3)] {
        let cs = parse_claim_set(&json!({
            "cfg": {"ELL": 8, "K_DEG": 8, "N_LIG": 32, "T_QUERIES": 4},
            "claims": [{"op": op, "fields": {field: {"var": [row, 8]}}},
                       {"op": "RoutedProjectedMatmulClaim", "fields": {"f_p": {"var": [5, 8]}}}],
            "table_order": []}).to_string());
        if check_rows(&cs, &order, &[3, 1, 1, 1]).is_ok() { best = round; }
    }
    best
}

const OPERANDS: &[(&str, &[&str])] = &[
    ("AddClaim", &["a", "b", "c"]),
    ("LinCombClaim", &["xs"]),
    ("ConcatClaim", &["dst", "srcs"]),
    ("EmbeddingLookupClaim", &["E", "x"]),
    ("FreivaldsCombineClaim", &["m", "m_em", "xs", "y"]),
    ("GateBracketClaim", &["Z", "Zb", "gr", "rem", "ss", "w", "wZ"]),
    ("HadamardClaim", &["a", "b", "c", "c_full", "c_low", "c_shifted"]),
    ("HeadInterleaveClaim", &["a", "b", "dst"]),
    ("InfoFinalizeClaim", &["a", "b", "d", "dw", "e", "gap_o2", "pw", "rem", "surprisal", "z_o", "zw"]),
    ("MaskedCombineClaim", &["m", "xs", "m_rep", "prods", "y"]),
    ("MatmulClaim", &["A", "B", "C", "C_full", "C_low", "C_shifted"]),
    ("MaxClaim", &["A", "Al", "O", "Ogap", "gap", "gap_o", "l", "neg_gap", "tok", "vstar"]),
    ("PairedTlookupClaim", &["x", "y"]),
    ("RangeWordClaim", &["x", "local_indices"]),
    ("RescaleClaim", &["x", "x_full", "x_low", "x_shifted"]),
    ("RmsNormClaim", &["S", "S_limbs", "S_total", "X_sq", "hi_G2_chunks", "hi_H", "hi_g0h_chunks",
                       "hi_g1h_chunks", "hi_gl", "lo_G2_chunks", "lo_H", "lo_g0h_chunks", "lo_g1h_chunks",
                       "lo_gl", "output", "output_full", "output_low", "output_shifted", "q1", "q2"]),
    ("RoPEClaim", &["x", "x_rot", "x_rot_full", "x_rot_low", "x_rot_shifted"]),
    ("RoutedProjectedMatmulClaim", &["X", "M", "Y", "W"]),
    ("RoutingClaim", &["gap", "m", "mrt", "r", "r_chosen", "rstar", "rt"]),
    ("SiluClaim", &["C", "a_0", "a_1", "a_2", "a_3", "a_4", "g", "inv_g", "inv_x", "is_high", "key",
                    "magnitude", "mux_a", "mux_b", "output", "output_sat", "sign", "x", "y"]),
    ("SoftmaxClaim", &["c2", "c2_shifted", "inv_z_high", "is_high", "mux_y_A", "mux_y_B", "r_hi", "r_lo",
                       "s1", "s2", "x", "y_A", "y_A_raw", "y_B", "y_B_raw", "z", "z_high"]),
    ("SplitClaim", &["parts", "whole"]),
    ("TopkRoutingClaim", &["b", "d", "m", "md", "qt", "s", "tau", "v"]),
    ("TopkSlotsClaim", &["MS", "M", "m", "s", "ss"]),
    ("WordExtractionClaim", &["words", "x"]),
    ("AClaimThisTableDoesNotKnow", &["z", "u", "p", "f_y"]),
];

const AUXILIARIES: &[(&str, &[&str])] = &[
    ("FreivaldsCombineClaim", &["s_em", "ms_em", "ms_tm", "yr"]),
    ("HadamardClaim", &["z_c_low", "z_c_shifted"]),
    ("InfoFinalizeClaim", &["z_dw", "z_rem", "z_zw"]),
    ("MatmulClaim", &["y", "u", "p", "z_C_low", "z_C_shifted"]),
    ("MaxClaim", &["z_gap"]),
    ("PairedTlookupClaim", &["u", "z"]),
    ("RangeWordClaim", &["z"]),
    ("RescaleClaim", &["z_low", "z_shifted"]),
    ("RmsNormClaim", &["u", "p", "z_lo_chunks", "z_hi_chunks", "z_ym1_chunks", "z_S_limbs", "z_lo_gl",
                       "z_lo_g0h", "z_lo_g1h", "z_lo_G2", "z_hi_gl", "z_hi_g0h", "z_hi_g1h", "z_hi_G2",
                       "z_output_low", "z_output_shifted", "z_x_low", "z_x_shifted"]),
    ("RoPEClaim", &["z_x_rot_low", "z_x_rot_shifted"]),
    ("RoutedProjectedMatmulClaim", &["f_y", "f_u", "f_p"]),
    ("SiluClaim", &["pt_u", "pt_z", "z_a0", "z_a1", "z_a2", "z_a3", "z_a4", "z_x_low", "z_x_shifted"]),
    ("SoftmaxClaim", &["pt_u_A", "pt_z_A", "pt_u_B", "pt_z_B", "z_c2", "z_r_lo", "z_r_hi", "z_z_high",
                       "z_x_low", "z_x_shifted"]),
];

#[test]
fn operands_come_before_s_op() {
    for (op, fields) in OPERANDS {
        for f in *fields { assert_eq!(latest(op, f), 1, "{op}.{f}"); }
    }
}

#[test]
fn the_routed_claims_late_rows_come_before_s_bind() {
    for f in ["Pj", "Qm", "Hd", "yr"] {
        assert_eq!(latest("RoutedProjectedMatmulClaim", f), 2, "{f}");
    }
}

#[test]
fn auxiliaries_may_come_after_s_op() {
    for (op, fields) in AUXILIARIES {
        for f in *fields { assert_eq!(latest(op, f), 3, "{op}.{f}"); }
    }
}

/// Every inverse the compile reads is either an auxiliary above or one of the
/// two z-named fields the prover commits in round 1. A new field fails here.
#[test]
fn every_z_field_the_compile_reads_is_in_the_table() {
    let round_1 = ["z_high", "z_o"];
    let source = include_str!("../src/handlers.rs");
    let mut found = 0;
    for accessor in ["var(\"z_", "var_list(\"z_"] {
        for piece in source.split(accessor).skip(1) {
            let name = format!("z_{}", piece.split('"').next().unwrap());
            found += 1;
            let listed = AUXILIARIES.iter().any(|(_, fields)| fields.contains(&name.as_str()));
            assert!(listed || round_1.contains(&name.as_str()), "{name} is in no list");
        }
    }
    assert!(found > 30, "the scan found only {found} fields");
}
