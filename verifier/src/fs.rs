//! Sequential Fiat-Shamir transcript — the verifier's own copy.
//!
//! Byte-for-byte mirror of `prover/protocol.py` (`fs_frame`, `fs_seed`,
//! `statement_digest`, `fs_s_op`, `fs_s_comb`, `fs_s_col`). The verifier
//! RECOMPUTES every coin from the transcript it read and never uses a seed
//! that arrived inside the proof: a prover that picks its own coins (in
//! particular its own opened columns) is exactly the attack the staged
//! commit-before-challenge protocol exists to stop.
//!
//! Framing is length-prefixed (u64 little-endian length, then the bytes), so
//! no two different item lists can hash to the same input.

pub const FS_DOMAIN: &[u8] = b"VerInf-FS-v1";

fn update_framed(h: &mut blake3::Hasher, item: &[u8]) {
    h.update(&(item.len() as u64).to_le_bytes());
    h.update(item);
}

/// blake3(FS_DOMAIN || frame(label) || frame(item) ...)
pub fn fs_seed(label: &str, items: &[&[u8]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(FS_DOMAIN);
    update_framed(&mut h, label.as_bytes());
    for it in items {
        update_framed(&mut h, it);
    }
    *h.finalize().as_bytes()
}

/// The trusted static statement digest: a hash of the EXACT claim-set bytes
/// carried by the proof file (the raw `claims` sub-document, not a re-encoding
/// of it — a re-encoding would depend on this verifier's JSON formatting).
pub fn statement_digest(claims_bytes: &[u8]) -> [u8; 32] {
    fs_seed("statement", &[claims_bytes])
}

/// Coin after R1: the statement, the R1 block labels and the R1 roots.
pub fn s_op(stmt_digest: &[u8; 32], block_order_r1: &[String], roots_r1: &[[u8; 32]]) -> [u8; 32] {
    let labels = block_order_r1.join(",");
    let mut items: Vec<&[u8]> = Vec::with_capacity(2 + roots_r1.len());
    items.push(stmt_digest);
    items.push(labels.as_bytes());
    for r in roots_r1 {
        items.push(r);
    }
    fs_seed("s_op", &items)
}

/// Coin after R2 (the phase-2 commitment): the LATE challenges of the
/// routed-projected Freivalds check, which must not exist before P and Q are
/// committed.
pub fn s_bind(s_op: &[u8; 32], root_p2: &[u8; 32]) -> [u8; 32] {
    fs_seed("s_bind", &[s_op, root_p2])
}

/// Coin after R3 (the phase-3 commitment). A proof with no phase-3 block still
/// frames the round with the all-zero empty-commit root, so dropping a message
/// changes the transcript.
pub fn s_comb(s_bind: &[u8; 32], root_p3: &[u8; 32]) -> [u8; 32] {
    fs_seed("s_comb", &[s_bind, root_p3])
}

/// The empty-commitment sentinel (protocol.EMPTY_COMMIT_ROOT).
pub const EMPTY_COMMIT_ROOT: [u8; 32] = [0u8; 32];

/// Coin after the test polynomials — the column challenge. The polynomials are
/// framed as little-endian u64 vectors, the same bytes the prover hashed.
pub fn s_col(s_comb: &[u8; 32], q_irs: &[u64], q_lin: &[u64], p_0: &[u64]) -> [u8; 32] {
    let le = |v: &[u64]| -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 8);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    };
    let (a, b, c) = (le(q_irs), le(q_lin), le(p_0));
    fs_seed("s_col", &[s_comb, &a, &b, &c])
}
