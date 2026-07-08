//! Cross-language compile differential test. Reads /tmp/compile_parity.json
//! (per-op claims + s_op + Python's canonical constraint system), runs the Rust
//! compile_claims on each, canonicalizes identically, and asserts equality.
//!
//! This covers every op handler + expander WITHOUT a witness — the only Rust
//! code the matmul proof difftest didn't already exercise. Canonicalization
//! mirrors test_compile_parity.canon_mine / canon_quads_mine / canon_rhs_mine.
use std::collections::HashMap;
use serde_json::Value;
use ligero_verifier::claim::parse_claim_set;
use ligero_verifier::handlers::compile_claims;
use ligero_verifier::field::P;

fn hexbytes(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn exp_lin(v: &Value) -> Vec<[u64; 4]> {
    v.as_array().unwrap().iter().map(|e| {
        let a = e.as_array().unwrap();
        [a[0].as_u64().unwrap(), a[1].as_u64().unwrap(), a[2].as_u64().unwrap(), a[3].as_u64().unwrap()]
    }).collect()
}
fn exp_quad(v: &Value) -> Vec<[u64; 6]> {
    v.as_array().unwrap().iter().map(|e| {
        let a = e.as_array().unwrap();
        [a[0].as_u64().unwrap(), a[1].as_u64().unwrap(), a[2].as_u64().unwrap(),
         a[3].as_u64().unwrap(), a[4].as_u64().unwrap(), a[5].as_u64().unwrap()]
    }).collect()
}
fn exp_rhs(v: &Value) -> Vec<[u64; 2]> {
    v.as_array().unwrap().iter().map(|e| {
        let a = e.as_array().unwrap();
        [a[0].as_u64().unwrap(), a[1].as_u64().unwrap()]
    }).collect()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/compile_parity.json".into());
    let txt = std::fs::read_to_string(&path).expect("read compile_parity.json");
    let cases: Value = serde_json::from_str(&txt).unwrap();

    let mut pass = 0usize;
    let mut total = 0usize;
    for case in cases.as_array().unwrap() {
        total += 1;
        let tag = case["tag"].as_str().unwrap();
        let mut cs = parse_claim_set(&serde_json::to_string(&case["claims"]).unwrap());
        let s_op = hexbytes(case["s_op"].as_str().unwrap());
        let cons = compile_claims(&mut cs, &s_op);

        // canonical lin: accumulate Σcoef per (row,slot,cid), then drop zeros.
        // Each family emits (global_row, col_in_row, cid, coef) over its variable.
        let mut lin: HashMap<(usize, usize, usize), u64> = HashMap::new();
        for fam in &cons.families {
            fam.emit_global(&mut |row, slot, cid, coef| {
                let e = lin.entry((row, slot, cid)).or_insert(0);
                *e = (*e + coef) % P;
            });
        }
        let mut lin_v: Vec<[u64; 4]> = lin.iter().filter(|(_, &v)| v != 0)
            .map(|(&(r, s, c), &v)| [r as u64, s as u64, c as u64, v]).collect();
        lin_v.sort_unstable();

        // canonical quad: sorted (x,y,z,n,a,b) — families expanded to the
        // per-row tuples the Python dump uses (quad lift is representation-only).
        let mut quad_v: Vec<[u64; 6]> = cons.quadratic.iter()
            .flat_map(|qf| (0..qf.nrows()).map(move |tt| {
                [(qf.x_row + tt) as u64, (qf.y_row + tt) as u64, (qf.z_row + tt) as u64,
                 qf.n_at(tt) as u64, qf.a, qf.b]
            }))
            .collect();
        quad_v.sort_unstable();

        // canonical rhs: dict-comp semantics — skip zero BEFORE insert (a later
        // zero must NOT overwrite an earlier nonzero), last nonzero wins.
        let mut rhs: HashMap<usize, u64> = HashMap::new();
        for &(start, length, b) in &cons.rhs {
            let v = b % P;
            if v != 0 { for k in 0..length { rhs.insert(start + k, v); } }
        }
        let mut rhs_v: Vec<[u64; 2]> = rhs.iter().map(|(&c, &b)| [c as u64, b]).collect();
        rhs_v.sort_unstable();

        let e_lin = exp_lin(&case["lin"]);
        let e_quad = exp_quad(&case["quad"]);
        let e_rhs = exp_rhs(&case["rhs"]);
        let e_m = case["m_total"].as_u64().unwrap();

        let ok_lin = lin_v == e_lin;
        let ok_quad = quad_v == e_quad;
        let ok_rhs = rhs_v == e_rhs;
        let ok_m = cons.m_total as u64 == e_m;
        let ok = ok_lin && ok_quad && ok_rhs && ok_m;
        if ok { pass += 1; }

        println!("[{}] {}: lin {} quad {} rhs {} m_total {}",
                 if ok { "OK " } else { "XX " }, tag,
                 lin_v.len(), quad_v.len(), rhs_v.len(), cons.m_total);
        if !ok {
            if !ok_m { println!("    M_TOTAL: rust {} vs py {}", cons.m_total, e_m); }
            if !ok_lin {
                println!("    LIN: rust {} vs py {} entries", lin_v.len(), e_lin.len());
                for r in lin_v.iter().take(2000) {
                    if !e_lin.contains(r) { println!("      rust-only {:?}", r); break; }
                }
                for r in e_lin.iter().take(2000) {
                    if !lin_v.contains(r) { println!("      py-only   {:?}", r); break; }
                }
            }
            if !ok_quad {
                println!("    QUAD: rust {:?}", quad_v.iter().take(3).collect::<Vec<_>>());
                println!("          py   {:?}", e_quad.iter().take(3).collect::<Vec<_>>());
            }
            if !ok_rhs {
                println!("    RHS: rust {} vs py {} entries", rhs_v.len(), e_rhs.len());
                for r in rhs_v.iter().take(2000) {
                    if !e_rhs.contains(r) { println!("      rust-only {:?}", r); break; }
                }
                for r in e_rhs.iter().take(2000) {
                    if !rhs_v.contains(r) { println!("      py-only   {:?}", r); break; }
                }
            }
        }
    }
    println!("\n=== {}/{} compile-parity cases bit-exact ===", pass, total);
    std::process::exit(if pass == total { 0 } else { 1 });
}
