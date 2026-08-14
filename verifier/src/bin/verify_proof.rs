//! Read /tmp/proof.json (claims + proof + seeds dumped by dump_proof.py), run
//! the Rust verifier, print the per-check verdict + overall. The differential
//! driver compares this to Python's verdict and to a tampered-REJECT.
//!
//! Parsing: the big proof arrays (opened columns, merkle paths) are
//! deserialized into typed vectors via serde's streaming from_reader, NOT a
//! serde_json::Value DOM. Production field arrays are u64le/base64 strings;
//! legacy decimal arrays remain accepted. `claims` stays a Value —
//! it is small (~MBs) and parse_claim_set_value already consumes a Value.
use std::collections::HashMap;
use std::convert::TryInto;
use serde::Deserialize;
use serde_json::Value;
use serde_json::value::RawValue;
use ligero_verifier::claim::parse_claim_set_value;
use ligero_verifier::fs;
use ligero_verifier::verify::{Round3, Round4, verify_bound};

/// A field vector on the proof wire.  Legacy proofs use a JSON array; the
/// production writer uses `"u64le:<base64>"`.  Both decode to the identical
/// Vec<u64>, so this is transport-only and does not alter any verifier check or
/// Fiat--Shamir input.
#[derive(Deserialize)]
#[serde(untagged)]
enum WireU64Vec {
    Legacy(Vec<u64>),
    U64Le(String),
}

fn decode_b64(s: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62), b'/' => Some(63), _ => None,
        }
    }
    let b = s.as_bytes();
    if b.len() % 4 != 0 { return Err("base64 length is not a multiple of 4".into()); }
    let mut out = Vec::with_capacity(b.len() / 4 * 3);
    for (qi, q) in b.chunks_exact(4).enumerate() {
        let last = qi + 1 == b.len() / 4;
        let pad2 = q[2] == b'='; let pad3 = q[3] == b'=';
        if (pad2 || pad3) && !last { return Err("interior base64 padding".into()); }
        if pad2 && !pad3 { return Err("invalid base64 padding".into()); }
        let a = val(q[0]).ok_or("bad base64 character")? as u32;
        let c = val(q[1]).ok_or("bad base64 character")? as u32;
        let d = if pad2 { 0 } else { val(q[2]).ok_or("bad base64 character")? as u32 };
        let e = if pad3 { 0 } else { val(q[3]).ok_or("bad base64 character")? as u32 };
        let x = (a << 18) | (c << 12) | (d << 6) | e;
        out.push((x >> 16) as u8);
        if !pad2 { out.push((x >> 8) as u8); }
        if !pad3 { out.push(x as u8); }
    }
    Ok(out)
}

impl WireU64Vec {
    fn into_vec(self) -> Result<Vec<u64>, String> {
        match self {
            Self::Legacy(v) => Ok(v),
            Self::U64Le(s) => {
                let payload = s.strip_prefix("u64le:")
                    .ok_or("unknown string encoding for field vector")?;
                let raw = decode_b64(payload)?;
                if raw.len() % 8 != 0 { return Err("u64le payload is not 8-byte aligned".into()); }
                Ok(raw.chunks_exact(8).map(|c| {
                    u64::from_le_bytes(c.try_into().unwrap())
                }).collect())
            }
        }
    }
}

#[derive(Deserialize)]
struct RawProof {
    root_p1: String,
    root_p2: String,
    q_irs: WireU64Vec,
    q_lin: WireU64Vec,
    p_0: WireU64Vec,
    opened_p1: HashMap<String, WireU64Vec>,
    opened_p2: HashMap<String, WireU64Vec>,
    paths_p1: HashMap<String, Vec<(String, u8)>>,
    paths_p2: HashMap<String, Vec<(String, u8)>>,
    // Persistent W block (analysis/persistent-weights.md) — present only when
    // the prover split weights into their own root; absent → legacy 2-block
    // proof, parsed and verified byte-identically. Typed (not Value) to keep
    // the streaming parse's memory bound at full-model scale. `blocks` gives
    // the row-block order to join in (default ["p1","p2"]).
    #[serde(default)] blocks: Option<Vec<String>>,
    #[serde(default)] root_w: Option<String>,
    #[serde(default)] opened_w: Option<HashMap<String, WireU64Vec>>,
    #[serde(default)] paths_w: Option<HashMap<String, Vec<(String, u8)>>>,
    // Second weight block of a linking proof (persistent-weights P5): the
    // refreshed commitment's tree. The caller adopts root_wnew as the new
    // trusted R_W' after (a) this proof ACCEPTs and (b) root_w matches the
    // currently-trusted R_W.
    #[serde(default)] root_wnew: Option<String>,
    #[serde(default)] opened_wnew: Option<HashMap<String, WireU64Vec>>,
    #[serde(default)] paths_wnew: Option<HashMap<String, Vec<(String, u8)>>>,
    // Phase-3 block: the late auxiliaries committed in R3 (routed-projected
    // Freivalds). Absent on proofs whose tape has no late-stage claim.
    #[serde(default)] root_p3: Option<String>,
    #[serde(default)] opened_p3: Option<HashMap<String, WireU64Vec>>,
    #[serde(default)] paths_p3: Option<HashMap<String, Vec<(String, u8)>>>,
    #[serde(default)] root_blind: Option<String>,
    #[serde(default)] opened_blind: Option<HashMap<String, WireU64Vec>>,
    #[serde(default)] paths_blind: Option<HashMap<String, Vec<(String, u8)>>>,
}

#[derive(Deserialize)]
struct RawSeeds {
    s_op: String,
    s_comb: String,
    s_col: String,
    #[serde(default)] s_bind: Option<String>,
}

#[derive(Deserialize)]
struct RawTop {
    // RAW bytes of the claim sub-document, not a re-encoding: the statement
    // digest is taken over exactly the bytes in the file, so the verifier must
    // hash what it read (a Value round-trip would depend on serde's
    // formatting). Still small (~MBs) — parsed into a Value afterwards.
    claims: Box<RawValue>,
    seeds: RawSeeds,
    proof: RawProof,
    // Present on proofs from the sequential Fiat-Shamir prover. When present,
    // every coin is RECOMPUTED here and the file's `seeds` are only checked
    // for agreement, never trusted.
    #[serde(default)]
    statement_digest: Option<String>,
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

fn conv_open(m: HashMap<String, WireU64Vec>) -> HashMap<u64, Vec<u64>> {
    // into_iter moves the Vec<u64> — no copy of the (large) column data.
    m.into_iter().map(|(k, v)| {
        (k.parse().unwrap(), v.into_vec().expect("decode u64 proof vector"))
    }).collect()
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
    // argv: proof.json [EXPECTED_R_W_HEX] [EXPECTED_STATEMENT_DIGEST_HEX]
    // The policy arguments come from OUTSIDE the proof (the runbook's trusted
    // enrolled weight root and trusted statement digest). They are optional
    // today so the existing test corpus still runs; the driver work (S4) makes
    // them mandatory for a persistent-model proof.
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "/tmp/proof.json".into());
    // "-" means "no policy for this slot" — used to check the statement digest
    // of a proof that has no persistent weight block.
    let opt_hex = |i: usize| args.get(i).filter(|s| s.as_str() != "-").map(|s| hex32(s));
    let policy_root_w = opt_hex(2);
    let policy_stmt = opt_hex(3);
    let f = std::fs::File::open(&path).expect("open proof.json");
    let top: RawTop = serde_json::from_reader(std::io::BufReader::new(f))
        .expect("parse proof.json");

    let claims_bytes = top.claims.get().as_bytes().to_vec();
    let claims_value: Value = serde_json::from_str(top.claims.get())
        .expect("parse claims sub-document");
    let mut cs = parse_claim_set_value(claims_value);
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
            "p3" => (p.root_p3.take(), p.opened_p3.take(), p.paths_p3.take()),
            "w"  => (p.root_w.take(), p.opened_w.take(), p.paths_w.take()),
            "wnew" => (p.root_wnew.take(), p.opened_wnew.take(), p.paths_wnew.take()),
            "blind" => (p.root_blind.take(), p.opened_blind.take(), p.paths_blind.take()),
            other => panic!("unknown proof block '{other}'"),
        };
        roots.push(hex32(&root.expect("missing root for block")));
        opened.push(conv_open(ow.expect("missing opened for block")));
        paths.push(conv_paths(pw.expect("missing paths for block")));
    }
    // The block layout is part of the statement: the digest covers it, so a
    // proof cannot relabel or reorder its own blocks.
    let stmt_recomputed = fs::statement_digest(&claims_bytes, &block_order);
    let r3 = Round3 {
        q_irs: p.q_irs.into_vec().expect("decode q_irs"),
        q_lin: p.q_lin.into_vec().expect("decode q_lin"),
        p_0: p.p_0.into_vec().expect("decode p_0"),
    };
    let r4 = Round4 { opened, paths };

    // ---- transcript + policy -------------------------------------------
    // Fiat-Shamir proofs: recompute every coin here. The file's `seeds` are
    // compared for agreement and otherwise unused, so a prover that wrote
    // itself convenient columns fails the s_col check below.
    let mut policy: Vec<(String, bool)> = Vec::new();
    let mut s_bind_out: Option<Vec<u8>> = None;
    let (s_op, s_comb, s_col) = match &top.statement_digest {
        Some(stmt_hex) => {
            let stmt_claimed = hex32(stmt_hex);
            policy.push(("statement_digest = H(claim bytes, block order)".into(),
                         stmt_claimed == stmt_recomputed));
            // FAIL-CLOSED: the trusted statement digest is not optional. With
            // no external statement the prover picks what it proves, and the
            // verifier is reduced to checking that a proof is internally
            // consistent with itself.
            match policy_stmt {
                Some(exp) => policy.push((
                    "statement_digest = trusted policy digest".into(),
                    exp == stmt_recomputed)),
                None => policy.push((
                    "trusted statement digest supplied as policy".into(), false)),
            }
            // Row-block order is blind|W|Wnew|p1|p2[|p3]: everything up to p2
            // is R1, p2 is R2, and the optional p3 is R3.
            let has_p3 = block_order.last().map(|b| b == "p3").unwrap_or(false);
            let n2 = block_order.len() - 1 - has_p3 as usize;   // index of p2
            assert!(block_order[n2] == "p2", "phase-2 block is misplaced");
            let s_op = fs::s_op(&stmt_recomputed, &block_order[..n2], &roots[..n2]);
            let s_bind = fs::s_bind(&s_op, &roots[n2]);
            s_bind_out = Some(s_bind.to_vec());
            let root_p3 = if has_p3 { roots[n2 + 1] } else { fs::EMPTY_COMMIT_ROOT };
            let s_comb = fs::s_comb(&s_bind, &root_p3);
            let s_col = fs::s_col(&s_comb, &r3.q_irs, &r3.q_lin, &r3.p_0);
            let bind_ok = top.seeds.s_bind.as_ref()
                .map(|h| hexbytes(h) == s_bind).unwrap_or(false);
            policy.push(("seeds in file = recomputed transcript".into(),
                         hexbytes(&top.seeds.s_op) == s_op
                             && bind_ok
                             && hexbytes(&top.seeds.s_comb) == s_comb
                             && hexbytes(&top.seeds.s_col) == s_col));
            (s_op.to_vec(), s_comb.to_vec(), s_col.to_vec())
        }
        // Legacy corpus (the non-streaming test prover): coins were expanded
        // from one base seed, so there is no transcript to recompute.
        None => {
            if policy_stmt.is_some() {
                policy.push(("statement digest required but proof has none".into(), false));
            }
            s_bind_out = top.seeds.s_bind.as_ref().map(|h| hexbytes(h));
            (hexbytes(&top.seeds.s_op), hexbytes(&top.seeds.s_comb),
             hexbytes(&top.seeds.s_col))
        }
    };
    // A proof over an enrolled model must be checked against the enrolled
    // root; without it the prover chooses its own weights.
    let w_idx = block_order.iter().position(|b| b == "w");
    match (w_idx, policy_root_w) {
        (Some(i), Some(exp_w)) => policy.push((
            "weight root = trusted enrolled root".into(), roots[i] == exp_w)),
        (Some(_), None) => policy.push((
            "trusted weight root supplied for a persistent-model proof".into(),
            false)),
        (None, Some(_)) => policy.push((
            "policy names a weight root but the proof has no weight block".into(),
            false)),
        (None, None) => {}
    }

    let t0 = std::time::Instant::now();
    let (ok_checks, per) = verify_bound(&mut cs, &roots, &r3, r4, &s_op,
                                        s_bind_out.as_deref(), &s_comb, &s_col);
    let elapsed = t0.elapsed();
    for (name, b) in &per {
        println!("  [{}] {}", if *b { "OK " } else { "XX " }, name);
    }
    for (name, b) in &policy {
        println!("  [{}] {}", if *b { "OK " } else { "XX " }, name);
    }
    let ok = ok_checks && policy.iter().all(|(_, b)| *b);
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

#[cfg(test)]
mod wire_tests {
    use super::{decode_b64, WireU64Vec};

    #[test]
    fn compact_u64le_roundtrip_and_bad_padding() {
        // little-endian bytes of [1, u64::MAX]
        let got = WireU64Vec::U64Le(
            "u64le:AQAAAAAAAAD//////////w==".to_string())
            .into_vec().unwrap();
        assert_eq!(got, vec![1, u64::MAX]);
        assert!(decode_b64("AA=A").is_err());
        assert!(decode_b64("AAAA=AAA").is_err());
    }
}
