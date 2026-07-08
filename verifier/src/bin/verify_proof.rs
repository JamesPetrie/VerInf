//! Read /tmp/proof.json (claims + proof + seeds dumped by dump_proof.py), run
//! the Rust verifier, print the per-check verdict + overall. The differential
//! driver compares this to Python's verdict and to a tampered-REJECT.
//!
//! Parsing: the big proof arrays (opened columns, merkle paths) are
//! deserialized straight into typed Vec<u64> via serde's streaming from_reader,
//! NOT a serde_json::Value DOM. At full-model scale the proof JSON is ~12 GB;
//! a Value DOM of it peaks >120 GB (each integer a 24-byte enum node) and OOMs.
//! Typed parse keeps peak at a few GB (packed u64). `claims` stays a Value —
//! it is small (~MBs) and parse_claim_set_value already consumes a Value.
use std::collections::HashMap;
use serde::Deserialize;
use serde_json::Value;
use ligero_verifier::claim::parse_claim_set_value;
use ligero_verifier::verify::{Round3, Round4, verify};

#[derive(Deserialize)]
struct RawProof {
    root_p1: String,
    root_p2: String,
    q_irs: Vec<u64>,
    q_lin: Vec<u64>,
    p_0: Vec<u64>,
    opened_p1: HashMap<String, Vec<u64>>,
    opened_p2: HashMap<String, Vec<u64>>,
    paths_p1: HashMap<String, Vec<(String, u8)>>,
    paths_p2: HashMap<String, Vec<(String, u8)>>,
    // Persistent W block (analysis/persistent-weights.md) — present only when
    // the prover split weights into their own root; absent → legacy 2-block
    // proof, parsed and verified byte-identically. Typed (not Value) to keep
    // the streaming parse's memory bound at full-model scale. `blocks` gives
    // the row-block order to join in (default ["p1","p2"]).
    #[serde(default)] blocks: Option<Vec<String>>,
    #[serde(default)] root_w: Option<String>,
    #[serde(default)] opened_w: Option<HashMap<String, Vec<u64>>>,
    #[serde(default)] paths_w: Option<HashMap<String, Vec<(String, u8)>>>,
    // Second weight block of a linking proof (persistent-weights P5): the
    // refreshed commitment's tree. The caller adopts root_wnew as the new
    // trusted R_W' after (a) this proof ACCEPTs and (b) root_w matches the
    // currently-trusted R_W.
    #[serde(default)] root_wnew: Option<String>,
    #[serde(default)] opened_wnew: Option<HashMap<String, Vec<u64>>>,
    #[serde(default)] paths_wnew: Option<HashMap<String, Vec<(String, u8)>>>,
    #[serde(default)] root_blind: Option<String>,
    #[serde(default)] opened_blind: Option<HashMap<String, Vec<u64>>>,
    #[serde(default)] paths_blind: Option<HashMap<String, Vec<(String, u8)>>>,
}

#[derive(Deserialize)]
struct RawSeeds { s_op: String, s_comb: String, s_col: String }

#[derive(Deserialize)]
struct RawTop {
    claims: Value,              // small — keep as Value for parse_claim_set_value
    seeds: RawSeeds,
    proof: RawProof,
    #[serde(default)]
    python_accept: Option<bool>,
}

fn hex32(s: &str) -> [u8; 32] {
    let s = s.strip_prefix("0x").unwrap_or(s);
    assert_eq!(s.len(), 64, "root hex must be 32 bytes");
    let mut b = [0u8; 32];
    for i in 0..32 {
        b[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
    }
    b
}

fn hexbytes(s: &str) -> Vec<u8> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

fn conv_open(m: HashMap<String, Vec<u64>>) -> HashMap<u64, Vec<u64>> {
    // into_iter moves the Vec<u64> — no copy of the (large) column data.
    m.into_iter().map(|(k, v)| (k.parse().unwrap(), v)).collect()
}

fn conv_paths(m: HashMap<String, Vec<(String, u8)>>) -> HashMap<u64, Vec<([u8; 32], u8)>> {
    m.into_iter()
        .map(|(k, steps)| {
            (k.parse().unwrap(),
             steps.into_iter().map(|(h, side)| (hex32(&h), side)).collect())
        })
        .collect()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/proof.json".into());
    let f = std::fs::File::open(&path).expect("open proof.json");
    let top: RawTop = serde_json::from_reader(std::io::BufReader::new(f))
        .expect("parse proof.json");

    let mut cs = parse_claim_set_value(top.claims);
    let mut p = top.proof;
    // Assemble blocks in the ROW-BLOCK ORDER named by `blocks` (default the
    // legacy ["p1","p2"]). Each block's (root, opened, paths) join in that
    // order to form the joint column the compiled row_starts index into.
    let block_order = p.blocks.take().unwrap_or_else(|| vec!["p1".into(), "p2".into()]);
    let mut roots = Vec::new();
    let mut opened = Vec::new();
    let mut paths = Vec::new();
    for b in &block_order {
        let (root, ow, pw) = match b.as_str() {
            "p1" => (Some(std::mem::take(&mut p.root_p1)),
                     Some(std::mem::take(&mut p.opened_p1)), Some(std::mem::take(&mut p.paths_p1))),
            "p2" => (Some(std::mem::take(&mut p.root_p2)),
                     Some(std::mem::take(&mut p.opened_p2)), Some(std::mem::take(&mut p.paths_p2))),
            "w"  => (p.root_w.take(), p.opened_w.take(), p.paths_w.take()),
            "wnew" => (p.root_wnew.take(), p.opened_wnew.take(), p.paths_wnew.take()),
            "blind" => (p.root_blind.take(), p.opened_blind.take(), p.paths_blind.take()),
            other => panic!("unknown proof block '{other}'"),
        };
        roots.push(hex32(&root.expect("missing root for block")));
        opened.push(conv_open(ow.expect("missing opened for block")));
        paths.push(conv_paths(pw.expect("missing paths for block")));
    }
    let r3 = Round3 { q_irs: p.q_irs, q_lin: p.q_lin, p_0: p.p_0 };
    let r4 = Round4 { opened, paths };
    let (s_op, s_comb, s_col) =
        (hexbytes(&top.seeds.s_op), hexbytes(&top.seeds.s_comb), hexbytes(&top.seeds.s_col));

    let t0 = std::time::Instant::now();
    let (ok, per) = verify(&mut cs, &roots, &r3, r4, &s_op, &s_comb, &s_col);
    let elapsed = t0.elapsed();
    for (name, b) in &per {
        println!("  [{}] {}", if *b { "OK " } else { "XX " }, name);
    }
    println!("verify_elapsed_ms: {}  (rayon threads: {})",
             elapsed.as_millis(), rayon::current_num_threads());
    println!("rust_verify: {}", if ok { "ACCEPT" } else { "REJECT" });
    match top.python_accept {
        Some(py) => {
            println!("python_accept: {}", if py { "ACCEPT" } else { "REJECT" });
            println!("match: {}", if ok == py { "YES" } else { "NO" });
        }
        None => println!("python_accept: (none — GPU verify skipped; Rust verdict stands alone)"),
    }
}
