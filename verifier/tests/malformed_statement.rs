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
    let h = |b: &str| b.repeat(32);
    let file = format!(
        r#"{{"claims":{},"seeds":{{"s_op":"{}","s_comb":"{}","s_col":"{}","s_bind":"{}"}},"statement_digest":"{}","proof":{{"root_p1":"{}","root_p2":"{}","q_irs":[],"q_lin":[],"p_0":[],"opened_p1":{{}},"opened_p2":{{}},"paths_p1":{{}},"paths_p2":{{}}}}}}"#,
        stmt.0, h("11"), h("22"), h("33"), h("44"), stmt.1, h("66"), h("77"));
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
