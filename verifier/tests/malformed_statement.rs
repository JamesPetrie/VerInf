//! A malformed or rewritten statement is a REJECT with its reason, never a
//! panic.
//!
//! The regression: prover/tests/test_fiat_shamir.py's rewritten statement
//! (table_order + [7]) made verify_proof panic in the compile's table lookup
//! (handlers.rs, "no entry found for key", exit 101) instead of rejecting.
//! The proof files here are minimal: they reach the statement's checks, and a
//! verifier that panics on one fails these tests.
use std::process::Command;

use ligero_verifier::fs::statement_digest;

fn table(id: u64) -> serde_json::Value {
    serde_json::json!({"table": {"id": id, "T_len": 4, "T_range": true, "T_Y": null,
                                 "mult_var": [id, 4], "w_var": [id + 1, 1], "z_vars": []}})
}

fn one_table_claim() -> serde_json::Value {
    serde_json::json!([{"op": "RangeWordClaim", "fields": {"table": table(5)}}])
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The claim sub-document's exact bytes, and the digest a policy trusting
/// that statement would carry (blocks p1, p2).
fn statement(claims: &serde_json::Value, table_order: &[u64]) -> (String, String) {
    let doc = serde_json::json!({"cfg": {"ELL": 8, "K_DEG": 8, "N_LIG": 32, "T_QUERIES": 4},
                                 "claims": claims, "table_order": table_order}).to_string();
    let d = statement_digest(doc.as_bytes(), &["p1".to_string(), "p2".to_string()]);
    (doc, hex(&d))
}

/// (exit status, stdout) of verify_proof on a proof carrying `stmt` (its
/// digest in the file) under the trusted statement digest `policy`.
fn run(name: &str, stmt: &(String, String), policy: Option<&str>) -> (Option<i32>, String) {
    run_with_proof(name, stmt, policy, |_| {})
}

fn run_with_proof(name: &str, stmt: &(String, String), policy: Option<&str>,
                  change: impl FnOnce(&mut serde_json::Value)) -> (Option<i32>, String) {
    use serde_json::{json, Value};
    let h = |b: &str| b.repeat(32);
    let mut proof = json!({
        "root_p1": h("66"), "root_p2": h("77"),
        "q_irs": vec![0; 8], "q_lin": vec![0; 16], "p_0": vec![0; 16],
        "opened_p1": {}, "opened_p2": {}, "paths_p1": {}, "paths_p2": {}
    });
    change(&mut proof);
    let file = json!({
        "claims": serde_json::from_str::<Value>(&stmt.0).unwrap(),
        "seeds": {"s_op": h("11"), "s_comb": h("22"), "s_col": h("33"), "s_bind": h("44")},
        "statement_digest": stmt.1, "proof": proof
    }).to_string();
    let path = std::env::temp_dir().join(format!("verinf-malformed-{}-{name}.json", std::process::id()));
    std::fs::write(&path, file).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_verify_proof"))
        .arg(&path).arg("-").arg(policy.unwrap_or("-"))
        .output().unwrap();
    std::fs::remove_file(&path).ok();
    (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned())
}

/// An explicit REJECT after a normal exit, with `line` among the checks.
fn assert_rejects(name: &str, stmt: &(String, String), policy: Option<&str>, line: &str) {
    let (code, out) = run(name, stmt, policy);
    assert_eq!(code, Some(0), "{name}: exit status {code:?}, not a verdict:\n{out}");
    let verdicts: Vec<&str> = out.lines().filter(|l| l.starts_with("rust_verify:")).collect();
    assert_eq!(verdicts, ["rust_verify: REJECT"], "{name}:\n{out}");
    assert!(out.contains(line), "{name}: no '{line}':\n{out}");
}

#[test]
fn a_rewritten_statement_rejects_before_it_is_parsed() {
    // the pod's failure: an honest statement's table_order extended with 7
    let honest = statement(&serde_json::json!([]), &[]);
    let rewritten = statement(&serde_json::json!([]), &[7]);
    assert_rejects("rewritten-no-policy", &rewritten, None,
                   "[XX ] trusted statement digest supplied as policy");
    assert_rejects("rewritten-honest-policy", &rewritten, Some(&honest.1),
                   "[XX ] statement_digest = trusted policy digest");
    // never parsed: no line of the claim set's own checks
    let (_, out) = run("rewritten-unparsed", &rewritten, Some(&honest.1));
    assert!(!out.contains("table_order"), "{out}");
}

#[test]
fn a_trusted_table_order_naming_an_undeclared_table_rejects() {
    let s = statement(&serde_json::json!([]), &[7]);
    assert_rejects("undeclared", &s, Some(&s.1),
                   "[XX ] table_order: table_order names table 7, which no claim declares");
    let s = statement(&one_table_claim(), &[5, 7]);
    assert_rejects("appended", &s, Some(&s.1),
                   "[XX ] table_order: table_order names table 7, which no claim declares");
}

#[test]
fn a_trusted_table_order_settling_a_table_twice_rejects() {
    let s = statement(&one_table_claim(), &[5, 5]);
    assert_rejects("twice", &s, Some(&s.1), "[XX ] table_order: table_order names table 5 twice");
}

#[test]
fn a_trusted_table_order_leaving_a_table_unsettled_rejects() {
    let s = statement(&one_table_claim(), &[]);
    assert_rejects("unsettled", &s, Some(&s.1),
                   "[XX ] table_order: table 5 is declared but not in table_order");
}

#[test]
fn a_well_formed_trusted_statement_passes_both_checks() {
    // no claims, no tables: the statement is trusted and well-formed, and the
    // empty proof is rejected by the checks that follow, still with a verdict
    let s = statement(&serde_json::json!([]), &[]);
    assert_rejects("empty", &s, Some(&s.1), "[OK ] table_order = the claims' tables, each once");
    let (_, out) = run("empty-policy", &s, Some(&s.1));
    assert!(out.contains("[OK ] statement_digest = trusted policy digest"), "{out}");
}

#[test]
fn invalid_labels_get_a_verdict_before_block_decoding() {
    let s = statement(&serde_json::json!([]), &[]);
    for labels in [vec![], vec!["p2", "p1"], vec!["unknown", "p1", "p2"], vec!["p1", "p1", "p2"]] {
        let (code, out) = run_with_proof("block-order", &s, Some(&s.1), |p| {
            p["blocks"] = serde_json::json!(labels);
        });
        assert_eq!(code, Some(0), "{labels:?}");
        assert!(out.contains("[XX ] blocks:") && out.contains("rust_verify: REJECT"), "{out}");
    }
}

#[test]
fn transcript_requires_fixed_width_even_for_zero_polynomials() {
    let s = statement(&serde_json::json!([]), &[]);
    for field in ["q_irs", "q_lin", "p_0"] {
        let (code, out) = run_with_proof(field, &s, Some(&s.1), |p| {
            p[field].as_array_mut().unwrap().pop();
        });
        assert_eq!(code, Some(0));
        assert!(out.contains("[XX ] Fiat-Shamir coefficient counts") && out.contains("rust_verify: REJECT"), "{out}");
    }
}

#[test]
fn row_policy_checks_the_compiled_lookup_span() {
    use serde_json::json;
    let s = statement(&json!([{"op": "RangeWordClaim", "fields": {
        "x": {"var": [3, 8]}, "z": {"var": [4, 9]}, "length": {"scalar": 9},
        "table": table(3)
    }}]), &[3]);
    let (code, out) = run_with_proof("lookup-span", &s, Some(&s.1), |p| {
        p["opened_p1"] = json!({"0": [0, 0, 0, 0]});
        p["opened_p2"] = json!({"0": [0, 0]});
    });
    assert_eq!(code, Some(0));
    assert!(out.contains("x needs 9 slots but declares 8") && out.contains("rust_verify: REJECT"), "{out}");
}

#[test]
fn library_verifier_checks_each_emitted_quadratic_operand() {
    use ligero_verifier::claim::parse_claim_set;
    use ligero_verifier::verify::{verify, Round3, Round4};
    use serde_json::json;
    // The low-level entry point receives no block labels. Its compiler must
    // still check the handler's actual slot count against all three operands.
    for field in ["a", "b", "c"] {
        let mut fields = json!({"length": {"scalar": 9}, "rescale_bits": {"scalar": 0},
            "a": {"var": [3, 9]}, "b": {"var": [5, 9]}, "c": {"var": [7, 9]}});
        fields[field]["var"][1] = json!(8);
        let stmt = statement(&json!([{"op": "HadamardClaim", "fields": fields}]), &[]);
        let mut cs = parse_claim_set(&stmt.0);
        let r3 = Round3 { q_irs: vec![0], q_lin: vec![0], p_0: vec![0] };
        let r4 = Round4 { opened: vec![], paths: vec![] };
        let (ok, checks) = verify(&mut cs, &[], &r3, r4, &[0; 32], &[0; 32], &[0; 32]);
        assert!(!ok && checks.contains(&("compiled operand spans", false)), "{field}: {checks:?}");
    }
}
