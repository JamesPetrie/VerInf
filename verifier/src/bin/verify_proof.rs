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
use ligero_verifier::verify::{Round3, Round4, verify_bound, verify_bound_pinned};
use ligero_verifier::protocol;

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
            Self::U64Le(s) => Self::U64Le(s).to_vec(),
        }
    }

    fn to_vec(&self) -> Result<Vec<u64>, String> {
        match self {
            Self::Legacy(v) => Ok(v.clone()),
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
    // WC-LCRL-STC bridge materials (analysis/wc-lcrl-stc-spec.md 0.4):
    // verified HERE against the enrollment root before compile consumes the
    // P_trace pin for use_bridge claims.
    #[serde(default)]
    wc: Option<WcSection>,
}

#[derive(Deserialize)]
#[allow(non_snake_case)]
struct WcGeom { B: usize, lam: usize, N_w: usize, q_w: usize }

#[derive(Deserialize)]
struct WcSection {
    root: String,
    manifest_digest: String,
    params: WcGeom,
    claim_index: usize,
    group_meta: HashMap<String, (usize, usize)>,
    // every array on the proof's u64 wire (decimal JSON or "u64le:" base64,
    // like the rest of the proof); pi is flat, row-major (n_blocks x lam)
    p_trace: HashMap<String, WireU64Vec>,
    pi: HashMap<String, WireU64Vec>,
    c: WireU64Vec,
    v: WireU64Vec,
    eta: WireU64Vec,
    opened: HashMap<String, WireU64Vec>,
    paths: HashMap<String, Vec<(String, u8)>>,
}

fn wc_u64le(vals: &[u64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(vals.len() * 8);
    for v in vals { b.extend_from_slice(&v.to_le_bytes()); }
    b
}

fn wc_leaf(col: &[u64]) -> [u8; 32] {
    // leaf = blake3("wc-leaf" || blake3(column bytes)) — the inner hash is
    // the GPU column accumulator's digest, the outer wrap domain-separates
    // the enrollment tree (mirrors wc_bridge._leaf).
    let inner = *blake3::hash(&wc_u64le(col)).as_bytes();
    let mut h = blake3::Hasher::new();
    h.update(b"wc-leaf");
    h.update(&inner);
    *h.finalize().as_bytes()
}

/// The enrollment tree's opening check (python twin: wc_bridge._verify_path),
/// bound like protocol::merkle_verify: the ordering at every level comes from
/// `index`, the path has exactly the tree's depth, the index is in range and
/// an odd last node pairs only with itself. The bridge's side bit is its own
/// convention, the opposite of the main tree's: is_right == 1 ⇔ this node is a
/// right child (sibling ‖ h). A malformed sibling is a REJECT, not a panic.
fn wc_path_ok(leaf: [u8; 32], path: &[(String, u8)], root: [u8; 32],
              index: u64, n_leaves: u64) -> bool {
    if index >= n_leaves || path.len() != protocol::merkle_depth(n_leaves) {
        return false;
    }
    let (mut h, mut idx, mut width) = (leaf, index, n_leaves);
    for (sib_hex, is_right) in path {
        let right = idx & 1 == 1;
        if *is_right != right as u8 { return false; }
        let Some(sib) = hex32_try(sib_hex) else { return false };
        if !right && idx + 1 >= width && sib != h { return false; }
        let mut hh = blake3::Hasher::new();
        if right { hh.update(&sib); hh.update(&h); }
        else { hh.update(&h); hh.update(&sib); }
        h = *hh.finalize().as_bytes();
        idx >>= 1;
        width = (width + 1) / 2;
    }
    h == root
}

// The enrollment's trusted identity (python twin: wc_bridge.enrollment_identity),
// what the policy slot argv[4] carries. The Merkle root commits the columns but
// not how they are read: the geometry, the ordered layout of the claims' rows
// and the padding rule are hashed with it, so a proof declaring another split
// between weights and masks, or another row order, names another identity.
const WC_IDENTITY_DOMAIN: &[u8] = b"verinf/wc-enrollment-identity/v1";
const WC_IDENTITY_RULE: &[u8] =
    b"polys=width-asc,block,output;coeffs=weights[0,B),masks[B,B+lam);\
rows=segments-in-claim-order,zero-padded-to-B;\
domain=omega^i,omega=7^((P-1)/N_w);\
leaf=blake3('wc-leaf'||blake3(column-u64le))";

/// width -> the ordered row counts (E*K) of its bridged claims, width-sorted,
/// from the CANONICAL claim map (python twin: wc_bridge.claim_layout).
fn wc_layout(cmap: &[(usize, usize, usize, usize)]) -> Vec<(usize, Vec<usize>)> {
    let mut widths: Vec<usize> = cmap.iter().map(|m| m.1).collect();
    widths.sort();
    widths.dedup();
    widths.into_iter().map(|w| {
        (w, cmap.iter().filter(|m| m.1 == w).map(|m| m.3).collect())
    }).collect()
}

fn wc_enrollment_identity(root: &[u8; 32], manifest: &[u8; 32], b: usize, lam: usize,
                          n_w: usize, layout: &[(usize, Vec<usize>)]) -> [u8; 32] {
    let u64le = |v: usize| (v as u64).to_le_bytes();
    let mut h = blake3::Hasher::new();
    for tag in [WC_IDENTITY_DOMAIN, WC_IDENTITY_RULE] {
        h.update(&u64le(tag.len()));
        h.update(tag);
    }
    h.update(root);
    h.update(manifest);
    for v in [b, lam, n_w] { h.update(&u64le(v)); }
    h.update(&u64le(layout.len()));
    for (width, segs) in layout {
        let rows: usize = segs.iter().sum();
        let n_blocks = (rows + b - 1) / b;
        h.update(&u64le(*width));
        h.update(&u64le(segs.len()));
        for &s in segs { h.update(&u64le(s)); }
        for v in [rows, n_blocks, n_blocks * b - rows] { h.update(&u64le(v)); }
    }
    *h.finalize().as_bytes()
}

/// The full WC bridge check (python twin: wc_bridge.verify_bridge_hosted +
/// _verify_core), width-general.  Every coin recomputed from (s_op, s_bind)
/// — nothing from the wire is trusted.  `cmap` is the CANONICAL claim map
/// recomputed from the claim set (never from the wire): (claim_index,
/// width, row_off, E*K).  Returns the per-claim P_trace pins on ACCEPT.
fn wc_verify(wc: &WcSection, trusted_identity: &[u8; 32], s_op: &[u8], s_bind: &[u8],
             cmap: &[(usize, usize, usize, usize)], t_cols: usize)
             -> Result<Vec<(usize, Vec<u64>)>, String> {
    use ligero_verifier::field::{add, mul, pow, P};
    let g = &wc.params;
    let k_w = g.B + g.lam;
    // decode the wire arrays once (a bad encoding is a REJECT, not a panic)
    let mut p_trace: HashMap<String, Vec<u64>> = HashMap::new();
    for (k, v) in &wc.p_trace { p_trace.insert(k.clone(), v.to_vec()?); }
    let mut pi: HashMap<String, Vec<u64>> = HashMap::new();
    for (k, v) in &wc.pi { pi.insert(k.clone(), v.to_vec()?); }
    let wc_c = wc.c.to_vec()?;
    let wc_v = wc.v.to_vec()?;
    let wc_eta = wc.eta.to_vec()?;
    let mut opened: HashMap<String, Vec<u64>> = HashMap::new();
    for (k, v) in &wc.opened { opened.insert(k.clone(), v.to_vec()?); }
    // The geometry is the prover's to declare (it is pinned into every coin,
    // so a proof cannot be replayed under other parameters) but it is ALSO
    // validated here: q_w is the whole soundness of the bridge, and a proof
    // declaring q_w = 1 would otherwise pass every check below at a forgery
    // probability near one half. The floor is the review §7 rule the prover
    // carries as WcParams.min_qw_for_tau: the bridge must be at least as
    // strong as the fresh Ligero part, q_w >= ceil(0.416 * t) for t opened
    // columns (t = 54 -> q_w >= 23; the production 40 clears it).
    if k_w == 0 || k_w & (k_w - 1) != 0 {
        return Err(format!("K_w = B + lam = {k_w} is not a power of two"));
    }
    if g.N_w & (g.N_w - 1) != 0 || g.N_w <= k_w {
        return Err(format!("N_w = {} must be a power of two above K_w = {k_w}", g.N_w));
    }
    if g.q_w == 0 || g.q_w > g.N_w {
        return Err(format!("q_w = {} out of range", g.q_w));
    }
    if g.lam == 0 || g.B == 0 {
        return Err("lam and B must be positive: the masks are the hiding".into());
    }
    if g.q_w * 1000 < 416 * t_cols {
        return Err(format!("q_w = {} below the floor ceil(0.416 * {t_cols}) for \
{t_cols} opened Ligero columns", g.q_w));
    }
    // The enrollment this proof names — its root and manifest, read under its
    // declared geometry and the claim set's own layout — must be the trusted
    // one. (The geometry is also pinned into every coin below; that stops a
    // replay under other parameters, not a reinterpretation of the columns.)
    let root = hex32_try(&wc.root).ok_or("enrollment root is not 32 hex bytes")?;
    let manifest = hex32_try(&wc.manifest_digest)
        .ok_or("manifest digest is not 32 hex bytes")?;
    if wc_enrollment_identity(&root, &manifest, g.B, g.lam, g.N_w, &wc_layout(cmap))
        != *trusted_identity {
        return Err("enrollment identity (root, geometry, layout, manifest) is not \
the trusted identity".into());
    }
    if wc_c.len() != k_w { return Err("c length != K_w".into()); }
    let mut widths: Vec<usize> =
        wc.group_meta.keys().map(|w| w.parse().unwrap()).collect();
    widths.sort();
    // the enrollment must cover exactly the claim set's bridged weights
    for &w in &widths {
        let want: usize = cmap.iter().filter(|m| m.1 == w).map(|m| m.3).sum();
        let (n_blocks, _) = wc.group_meta[&w.to_string()];
        if want == 0 { return Err(format!("width {w} has no bridged claim")); }
        if n_blocks != (want + g.B - 1) / g.B {
            return Err(format!("width {w}: {n_blocks} blocks for {want} rows"));
        }
    }
    for m in cmap {
        if !widths.contains(&m.1) {
            return Err(format!("claim {} width {} not enrolled", m.0, m.1));
        }
    }
    for (&ref wkey, pt) in &p_trace {
        let w: usize = wkey.parse().map_err(|_| "width key")?;
        let (n_blocks, _) = *wc.group_meta.get(wkey).ok_or("group missing")?;
        if pt.len() != n_blocks * g.B { return Err("p_trace length".into()); }
        let pim = pi.get(wkey).ok_or("pi group missing")?;
        if pim.len() != n_blocks * g.lam {
            return Err("pi shape".into());
        }
        let _ = w;
    }
    // hosted late coin: s_bind + enrollment identity + geometry + R2 commit
    // (widths iterated SORTED, matching python's _commit_r2)
    let mut geom = b"wc-geom".to_vec();
    for v in [g.B, g.lam, g.N_w, g.q_w] {
        geom.extend_from_slice(&(v as u64).to_le_bytes());
    }
    let mut r2 = blake3::Hasher::new();
    r2.update(b"wc-r2");
    for &w in &widths {
        r2.update(&wc_u64le(&p_trace[&w.to_string()]));
        // the flat row-major vector hashes to the same bytes as its rows in
        // order, and never chunks by a zero width
        r2.update(&wc_u64le(&pi[&w.to_string()]));
    }
    let r2d = *r2.finalize().as_bytes();
    let s_late = fs::fs_seed("wc/hosted-late",
                             &[s_bind, &root, &manifest, &geom, &r2d]);
    // eta: distinct, and exactly the transcript's draw
    let eta = protocol::random_columns_n(
        &fs::fs_seed("wc/eta", &[&s_late]), g.q_w, g.N_w as u64);
    let mut ded = eta.clone(); ded.sort(); ded.dedup();
    if ded.len() != g.q_w { return Err("eta not distinct".into()); }
    if eta != wc_eta { return Err("eta mismatch".into()); }
    // c aggregation: alpha indexed width-major (sorted), then block —
    // exactly bridge_r3's order
    let mut c = vec![0u64; k_w];
    let mut bi: u64 = 0;
    for &w in &widths {
        let (n_blocks, _) = wc.group_meta[&w.to_string()];
        let pt = &p_trace[&w.to_string()];
        let pim = &pi[&w.to_string()];
        for a in 0..n_blocks {
            let alpha = protocol::challenge(&s_late, bi, "alpha");
            for i in 0..g.B {
                c[i] = add(c[i], mul(alpha, pt[a * g.B + i]));
            }
            for h in 0..g.lam {
                c[g.B + h] = add(c[g.B + h], mul(alpha, pim[a * g.lam + h]));
            }
            bi += 1;
        }
    }
    if c != wc_c { return Err("c does not aggregate P_trace/pi".into()); }
    // v = c(eta) on the pinned natural domain omega = 7^((P-1)/N_w)
    let omega = pow(7, (P - 1) / g.N_w as u64);
    for (l, &ei) in eta.iter().enumerate() {
        let x = pow(omega, ei);
        let mut acc = 0u64;
        for k in (0..k_w).rev() { acc = add(mul(acc, x), c[k]); }
        if acc != wc_v[l] { return Err(format!("v[{l}] != c(eta)")); }
    }
    // enrollment side: merkle-verified columns + the bridge equation, under
    // the SHARED per-width rho from the host transcript (spec 0.2)
    for (l, &ei) in eta.iter().enumerate() {
        let col = opened.get(&ei.to_string()).ok_or("column missing")?;
        let total: usize = widths.iter()
            .map(|w| { let (nb, _) = wc.group_meta[&w.to_string()]; nb * w })
            .sum();
        if col.len() != total { return Err("column length".into()); }
        let path = wc.paths.get(&ei.to_string()).ok_or("path missing")?;
        if !wc_path_ok(wc_leaf(col), path, root, ei, g.N_w as u64) {
            return Err(format!("merkle path fails at eta[{l}]"));
        }
        let mut rhs = 0u64;
        let mut bi: u64 = 0;
        let mut off = 0usize;
        for &w in &widths {
            let rho = protocol::op_vec(s_op, 0, &format!("rho-w{w}"), w);
            let (n_blocks, _) = wc.group_meta[&w.to_string()];
            for a in 0..n_blocks {
                let alpha = protocol::challenge(&s_late, bi, "alpha");
                let mut sum = 0u64;
                for j in 0..w {
                    sum = add(sum, mul(rho[j], col[off + a * w + j]));
                }
                rhs = add(rhs, mul(alpha, sum));
                bi += 1;
            }
            off += n_blocks * w;
        }
        if rhs != wc_v[l] { return Err(format!("bridge equation fails at eta[{l}]")); }
    }
    Ok(cmap.iter().map(|&(ci, w, off, ek)| {
        (ci, p_trace[&w.to_string()][off..off + ek].to_vec())
    }).collect())
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

fn hex32_try(s: &str) -> Option<[u8; 32]> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if s.len() != 64 { return None; }
    let mut b = [0u8; 32];
    for i in 0..32 {
        b[i] = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(b)
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
    //       [EXPECTED_WC_ENROLLMENT_ROOT_HEX]
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
    let policy_wc_identity = opt_hex(4);
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
    // t, the number of opened Ligero columns, is what the bridge's q_w floor
    // is measured against (the largest block's opening set).
    let t_cols = opened.iter().map(|m| m.len()).max().unwrap_or(0);
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
            // The bridge's late coins (alpha, eta) derive from s_bind; a
            // file-supplied s_bind would let the prover know the 40 points
            // before committing, and the 1024 free mask coefficients then
            // solve the 40 constraints exactly. Legacy-seed proofs carry no
            // recomputed transcript, so they carry no s_bind for the bridge
            // and a wc section on this path is refused below.
            s_bind_out = None;
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
    // The WC-bridge enrollment is a SECOND trust anchor with its own policy
    // slot (argv[4]). A production proof carries a weight block (the dense
    // weights stay committed rows) AND a wc section (the expert weights), and
    // each must be bound to a value from outside the proof. The bridge's slot
    // carries the enrollment IDENTITY (root, geometry, layout, manifest; see
    // wc_enrollment_identity), checked inside wc_verify below; without one the
    // bridge is not run and the proof is refused.
    match (&top.wc, policy_wc_identity) {
        (Some(_), None) => policy.push((
            "trusted enrollment identity supplied for a wc-bridge proof".into(),
            false)),
        (None, Some(_)) => policy.push((
            "policy names an enrollment identity but the proof has no wc section".into(),
            false)),
        _ => {}
    }

    // ---- WC-LCRL-STC bridge (spec 0.4/0.5) -------------------------------
    // Verified BEFORE compile: on ACCEPT the authenticated P_trace becomes
    // the public pin for the use_bridge claim's Pj rows; a use_bridge claim
    // without a verified bridge fails closed.
    // canonical claim map, recomputed from the claim set (never the wire):
    // (claim_index, width J, row offset within the width group, E*K)
    let mut cmap: Vec<(usize, usize, usize, usize)> = Vec::new();
    {
        let mut off: HashMap<usize, usize> = HashMap::new();
        for (ci, c) in cs.claims.iter().enumerate() {
            if c.opt_scalar("use_bridge").unwrap_or(0) == 1 {
                let (e, k, j) = (c.scalar("E") as usize,
                                 c.scalar("K") as usize,
                                 c.scalar("J") as usize);
                let o = *off.get(&j).unwrap_or(&0);
                cmap.push((ci, j, o, e * k));
                off.insert(j, o + e * k);
            }
        }
    }
    let mut wc_pins: Vec<(usize, Vec<u64>)> = Vec::new();
    match (&top.wc, cmap.len(), s_bind_out.as_deref()) {
        (Some(wcs), n, Some(sb)) if n > 0 => match policy_wc_identity {
            // no trusted identity: already a failed policy line above
            None => {}
            Some(exp) => match wc_verify(wcs, &exp, &s_op, sb, &cmap, t_cols) {
                Ok(pins) => {
                    policy.push(("wc bridge: P_trace authenticated against \
the trusted enrollment".into(), true));
                    wc_pins = pins;
                }
                Err(e) => policy.push((format!("wc bridge REJECT: {e}"), false)),
            },
        },
        (None, n, _) if n > 0 => policy.push((
            "use_bridge claims but no wc section".into(), false)),
        (Some(_), 0, _) => policy.push((
            "wc section but no use_bridge claim".into(), false)),
        (Some(_), _, None) => policy.push((
            "wc bridge needs the transcript's s_bind".into(), false)),
        _ => {}
    }

    if !cmap.is_empty() && wc_pins.len() != cmap.len() {
        // The bridge did not authenticate every bridged claim's fold (a wc
        // section missing, tampered, under the floor, or on the legacy seed
        // path): the compile would ask for a pin that was never produced and
        // abort. Report the policy table and REJECT instead — fail-closed,
        // with the reason on its line rather than in a panic.
        for (name, b) in &policy {
            println!("  [{}] {}", if *b { "OK " } else { "XX " }, name);
        }
        println!("rust_verify: REJECT");
        return;
    }
    let t0 = std::time::Instant::now();
    let (ok_checks, per) = verify_bound_pinned(&mut cs, &roots, &r3, r4, &s_op,
                                        s_bind_out.as_deref(), &s_comb, &s_col,
                                        wc_pins);
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

#[cfg(test)]
mod wc_bridge_tests {
    // A two-block enrollment small enough to build by hand: sixteen scalar
    // weights 1..=16 at B=12, lam=4, N_w=64, q_w=40, one width-1 claim of 16
    // rows. The forgeries keep the enrolled root, the manifest and the claim
    // map, and change what the bridge would authenticate.
    use super::*;
    use ligero_verifier::field::{add, mul, pow, sub, P};

    const N: usize = 64;
    const KW: usize = 16;
    const Q: usize = 40;
    const CMAP: [(usize, usize, usize, usize); 1] = [(0, 1, 0, 16)];

    fn hex(v: &[u8]) -> String { v.iter().map(|b| format!("{b:02x}")).collect() }

    fn evaluate(c: &[u64], x: u64) -> u64 {
        c.iter().rev().fold(0, |v, &a| add(mul(v, x), a))
    }

    struct Enrolled { columns: Vec<Vec<u64>>, levels: Vec<Vec<[u8; 32]>>,
                      coefficients: Vec<Vec<u64>> }

    fn enrolled() -> Enrolled {
        let mut coefficients = vec![vec![0u64; KW]; 2];
        for i in 0..16 { coefficients[i / 12][i % 12] = i as u64 + 1; }
        for a in 0..2 { for h in 0..4 { coefficients[a][12 + h] = 100 + (a * 4 + h) as u64; } }
        let omega = pow(7, (P - 1) / N as u64);
        let columns: Vec<Vec<u64>> = (0..N).map(|i| coefficients.iter()
            .map(|c| evaluate(c, pow(omega, i as u64))).collect()).collect();
        let mut levels = vec![columns.iter().map(|c| wc_leaf(c)).collect::<Vec<_>>()];
        while levels.last().unwrap().len() > 1 {
            let prev = levels.last().unwrap();
            levels.push(prev.chunks_exact(2).map(|pair| {
                let mut h = blake3::Hasher::new();
                h.update(&pair[0]); h.update(&pair[1]);
                *h.finalize().as_bytes()
            }).collect());
        }
        Enrolled { columns, levels, coefficients }
    }

    // the bridge's path convention: is_right == 1 when the node is a right child
    fn path(e: &Enrolled, mut pos: usize) -> Vec<(String, u8)> {
        let mut out = Vec::new();
        for level in &e.levels[..e.levels.len() - 1] {
            out.push((hex(&level[pos ^ 1]), (pos & 1) as u8));
            pos >>= 1;
        }
        out
    }

    /// A wc section for `declared_b`; `negate_odd` proves F(-X) and answers
    /// each queried index i with the enrolled column at i + N/2.
    fn section(e: &Enrolled, declared_b: usize, negate_odd: bool) -> WcSection {
        let root = e.levels.last().unwrap()[0];
        let rho = protocol::op_vec(&[1; 32], 0, "rho-w1", 1)[0];
        let projected: Vec<Vec<u64>> = e.coefficients.iter().map(|c| c.iter().enumerate()
            .map(|(k, &a)| mul(rho, if negate_odd && k % 2 == 1 { sub(0, a) } else { a }))
            .collect()).collect();
        let pt: Vec<u64> = projected.iter().flat_map(|c| c[..declared_b].to_vec()).collect();
        let pi: Vec<u64> = projected.iter().flat_map(|c| c[declared_b..].to_vec()).collect();
        let mut geom = b"wc-geom".to_vec();
        for v in [declared_b, KW - declared_b, N, Q] { geom.extend_from_slice(&(v as u64).to_le_bytes()); }
        let mut r2 = blake3::Hasher::new();
        r2.update(b"wc-r2"); r2.update(&wc_u64le(&pt)); r2.update(&wc_u64le(&pi));
        let r2d = *r2.finalize().as_bytes();
        let late = fs::fs_seed("wc/hosted-late", &[&[2; 32], &root, &[3; 32], &geom, &r2d]);
        let eta = protocol::random_columns_n(&fs::fs_seed("wc/eta", &[&late]), Q, N as u64);
        let mut c = vec![0; KW];
        for a in 0..2 {
            let alpha = protocol::challenge(&late, a as u64, "alpha");
            for k in 0..KW { c[k] = add(c[k], mul(alpha, projected[a][k])); }
        }
        let omega = pow(7, (P - 1) / N as u64);
        let v: Vec<u64> = eta.iter().map(|&i| evaluate(&c, pow(omega, i))).collect();
        let mut opened = HashMap::new();
        let mut paths = HashMap::new();
        for &i in &eta {
            // F(-X) at index i is the enrolled F at i + N/2
            let at = (i as usize + if negate_odd { N / 2 } else { 0 }) % N;
            opened.insert(i.to_string(), WireU64Vec::Legacy(e.columns[at].clone()));
            paths.insert(i.to_string(), path(e, at));
        }
        WcSection {
            root: hex(&root), manifest_digest: hex(&[3; 32]),
            params: WcGeom { B: declared_b, lam: KW - declared_b, N_w: N, q_w: Q },
            claim_index: 0, group_meta: HashMap::from([("1".to_string(), (2, 1))]),
            p_trace: HashMap::from([("1".to_string(), WireU64Vec::Legacy(pt))]),
            pi: HashMap::from([("1".to_string(), WireU64Vec::Legacy(pi))]),
            c: WireU64Vec::Legacy(c), v: WireU64Vec::Legacy(v), eta: WireU64Vec::Legacy(eta),
            opened, paths,
        }
    }

    /// What an auditor certifies for this enrollment: the honest geometry
    /// (B = 12) and the claim set's layout.
    fn trusted(e: &Enrolled) -> [u8; 32] {
        wc_enrollment_identity(&e.levels.last().unwrap()[0], &[3; 32], 12, 4, N,
                               &wc_layout(&CMAP))
    }

    fn verify(wc: &WcSection) -> Result<Vec<(usize, Vec<u64>)>, String> {
        wc_verify(wc, &trusted(&enrolled()), &[1; 32], &[2; 32], &CMAP, 54)
    }

    #[test]
    fn honest_section_accepts_with_the_enrolled_weights() {
        let e = enrolled();
        let pins = verify(&section(&e, 12, false)).unwrap();
        let rho = protocol::op_vec(&[1; 32], 0, "rho-w1", 1)[0];
        assert_eq!(pins[0].1, (1..=16).map(|v| mul(v, rho)).collect::<Vec<_>>());
    }

    #[test]
    fn relabeled_columns_do_not_authenticate_a_changed_projection() {
        // F(-X) under the F(X) root: every bridge equation holds when index i
        // is answered by the column at i + N/2 with that column's valid path
        let e = enrolled();
        let err = verify(&section(&e, 12, true)).unwrap_err();
        assert!(err.contains("merkle path fails"), "{err}");
    }

    #[test]
    fn a_declared_block_boundary_does_not_reinterpret_the_columns() {
        // B = 10, lam = 6 keeps K_w, the root, the manifest and the claim map;
        // under the old check it passed and read weights 11 and 12 as masks
        let e = enrolled();
        let err = verify(&section(&e, 10, false)).unwrap_err();
        assert!(err.contains("enrollment identity"), "{err}");
    }

    #[test]
    fn identity_binds_root_manifest_geometry_and_layout() {
        let e = enrolled();
        let root = e.levels.last().unwrap()[0];
        let base = trusted(&e);
        let other_root = { let mut r = root; r[0] ^= 1; r };
        assert_ne!(base, wc_enrollment_identity(&other_root, &[3; 32], 12, 4, N, &wc_layout(&CMAP)));
        assert_ne!(base, wc_enrollment_identity(&root, &[4; 32], 12, 4, N, &wc_layout(&CMAP)));
        assert_ne!(base, wc_enrollment_identity(&root, &[3; 32], 12, 4, 2 * N, &wc_layout(&CMAP)));
        // the same rows split into two claims is another layout
        assert_ne!(base, wc_enrollment_identity(&root, &[3; 32], 12, 4, N,
                                                &[(1, vec![8, 8])]));
        // an honest section checked against another identity is refused
        let err = wc_verify(&section(&e, 12, false), &[9; 32], &[1; 32], &[2; 32], &CMAP, 54)
            .unwrap_err();
        assert!(err.contains("enrollment identity"), "{err}");
    }

    #[test]
    fn identity_encoding_matches_the_python_twin() {
        // wc_bridge.enrollment_identity on the same inputs (test_merkle_index.py
        // pins the same two vectors)
        let root: [u8; 32] = core::array::from_fn(|i| i as u8);
        assert_eq!(hex(&wc_enrollment_identity(&root, &[3; 32], 12, 4, 64, &[(1, vec![16])])),
                   "323f385b048c4c26c0e2f30804117a3a2be715a7b9a47e1bd973d1d59be41697");
        assert_eq!(hex(&wc_enrollment_identity(&root, &[3; 32], 12, 4, 64,
                                               &[(4, vec![100, 7]), (6, vec![52])])),
                   "b85f6e2f8e4b5b3c07e34708982c5c0feaa1cbd4526fb45fce3d615b398bd238");
    }

    #[test]
    fn path_check_binds_index_depth_and_range() {
        let e = enrolled();
        let root = e.levels.last().unwrap()[0];
        let n = N as u64;
        assert!(wc_path_ok(e.levels[0][9], &path(&e, 9), root, 9, n));
        assert!(!wc_path_ok(e.levels[0][9], &path(&e, 9), root, 41, n));
        let flipped: Vec<_> = path(&e, 9).into_iter().map(|(s, b)| (s, b ^ 1)).collect();
        assert!(!wc_path_ok(e.levels[0][9], &flipped, root, 9, n));
        let p = path(&e, 9);
        assert!(!wc_path_ok(e.levels[0][9], &p[..p.len() - 1], root, 9, n));
        let mut long = p.clone();
        long.push((hex(&root), 0));
        assert!(!wc_path_ok(e.levels[0][9], &long, root, 9, n));
        assert!(!wc_path_ok(e.levels[0][9], &p, root, n, n));
        let mut bad = p.clone();
        bad[0].0 = "zz".repeat(32);
        assert!(!wc_path_ok(e.levels[0][9], &bad, root, 9, n));
    }
}
