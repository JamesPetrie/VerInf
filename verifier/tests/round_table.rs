// Copyright (c) 2026 Levi Neuwirth. Released under the MIT License (see LICENSE).
//! The whole round table, as a copy to change on purpose: which round each field
//! of each claim may come in, read from the prover's phases and from the saved
//! full-scale statements.
use ligero_verifier::claim::parse_claim_set;
use ligero_verifier::layout::check_rows;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use syn::visit::{self, Visit};

/// The latest round this field passes in: a statement with the field at row 3
/// (round 1), 4 (round 2) or 5 (round 3).
fn latest(op: &str, field: &str) -> u8 {
    let order: Vec<String> = ["blind", "p1", "p2", "p3"].iter().map(|s| s.to_string()).collect();
    let mut best = 0;
    for (row, round) in [(3u64, 1u8), (4, 2), (5, 3)] {
        // Unit dimensions make each handler's operand span fit in one slot.
        let mut fields = json!({
            "length": {"scalar": 1}, "k": {"scalar": 1}, "T": {"scalar": 1},
            "K": {"scalar": 1}, "E": {"scalar": 1}, "V": {"scalar": 1}, "F": {"scalar": 1},
            "heads": {"scalar": 1}, "m": {"scalar": 1}, "n": {"scalar": 1},
            "config": {"config": {"SEQ": 1, "heads": 1, "d_h": 1, "B": 1, "M": 1, "d": 1}}
        });
        fields[field] = json!({"var": [row, 8]});
        let cs = parse_claim_set(&json!({
            "cfg": {"ELL": 8, "K_DEG": 8, "N_LIG": 32, "T_QUERIES": 4},
            "claims": [{"op": op, "fields": fields},
                       {"op": "RoutedProjectedMatmulClaim", "fields": {"E": {"scalar": 1}, "f_p": {"var": [5, 8]}}}],
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

/// Walk each dispatch arm and the helper functions it calls. Looking at string
/// literals also covers the SiLU inverse names passed through a local variable.
struct HandlerFields<'a> {
    functions: &'a HashMap<String, syn::ItemFn>,
    visited: HashSet<String>,
    fields: HashSet<String>,
}

impl<'ast> Visit<'ast> for HandlerFields<'_> {
    fn visit_lit_str(&mut self, lit: &'ast syn::LitStr) {
        let name = lit.value();
        if name.starts_with("z_") { self.fields.insert(name); }
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        // Scalar parameter names such as GateBracketClaim.z_bits are not rows.
        if call.method != "scalar" && call.method != "opt_scalar" && call.method != "cfg_int" {
            visit::visit_expr_method_call(self, call);
        }
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref() {
            if let Some(id) = path.path.get_ident() {
                let name = id.to_string();
                if let Some(helper) = self.functions.get(&name) {
                    if self.visited.insert(name) { self.visit_block(&helper.block); }
                }
            }
        }
        visit::visit_expr_call(self, call);
    }
}

#[test]
fn every_z_field_the_compile_reads_is_in_the_table() {
    let file = syn::parse_file(include_str!("../src/handlers.rs")).unwrap();
    let functions: HashMap<_, _> = file.items.into_iter().filter_map(|item| match item {
        syn::Item::Fn(f) => Some((f.sig.ident.to_string(), f)),
        _ => None,
    }).collect();
    let dispatcher = &functions["compile_op"];
    let Some(syn::Stmt::Expr(syn::Expr::Match(dispatch), _)) = dispatcher.block.stmts.last() else {
        panic!("expected the claim dispatch match");
    };
    let mut found = HashSet::new();
    for arm in &dispatch.arms {
        let op = match &arm.pat {
            syn::Pat::Lit(syn::ExprLit { lit: syn::Lit::Str(op), .. }) => op.value(),
            syn::Pat::Ident(_) | syn::Pat::Wild(_) => continue, // unsupported claim
            _ => panic!("unrecognized claim dispatch pattern"),
        };
        let mut reads = HandlerFields { functions: &functions, visited: HashSet::new(), fields: HashSet::new() };
        reads.visit_expr(&arm.body);
        for field in reads.fields {
            let listed = AUXILIARIES.iter().chain(OPERANDS).any(|(claim, fields)|
                *claim == op && fields.contains(&field.as_str()));
            assert!(listed, "{op}.{field} is missing from the round table");
            found.insert((op.clone(), field));
        }
    }
    // Exercise both indirect field names and a name shared by two handlers.
    for (op, name) in [("SiluClaim", "z_a1"), ("SiluClaim", "z_x_low"), ("RmsNormClaim", "z_x_low")] {
        assert!(found.contains(&(op.into(), name.into())), "scan missed {op}.{name}");
    }
}
