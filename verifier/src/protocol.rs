//! Domains, poly/Lagrange eval, Merkle, and the blake3 challenge PRF —
//! bit-exact with protocol.py. Every byte layout here must match Python exactly
//! (the differential test checks this), so the layouts are spelled out inline.

use crate::field::{add, inv, mul, pow, sub, GLOBAL_G, P};

pub const NUM_BLINDING_ROWS: usize = 3;
pub const BLIND_IRS: usize = 0;
pub const BLIND_LIN: usize = 1;
pub const BLIND_QUAD: usize = 2;

#[derive(Clone, Copy)]
pub struct Config {
    pub ell: u64,       // ELL  — constrained message slots per row
    pub k_deg: u64,     // K_DEG — polynomial degree bound
    pub n_lig: u64,     // N_LIG — codeword length / #columns
    pub t_queries: usize,
}

impl Config {
    pub fn w_k(&self) -> u64 { pow(GLOBAL_G, (P - 1) / self.k_deg) }   // K-th root of unity
    pub fn w_n(&self) -> u64 { pow(GLOBAL_G, (P - 1) / self.n_lig) }   // N-th root of unity
    pub fn zeta(&self, c: u64) -> u64 { pow(self.w_k(), c) }           // ζ_c = ω_K^c
    pub fn eta(&self, j: u64) -> u64 { mul(GLOBAL_G, pow(self.w_n(), j)) } // η_j = γ·ω_N^j
}

/// Horner eval of ascending-coeff polynomial at x (matches poly_eval).
pub fn poly_eval(coeffs: &[u64], x: u64) -> u64 {
    let mut acc = 0u64;
    for &c in coeffs.iter().rev() {
        acc = add(mul(acc, x), c);
    }
    acc
}

/// L_c(η) = ζ_c · (η^K − 1) / (K · (η − ζ_c))   (matches protocol.lagrange).
pub fn lagrange(cfg: &Config, c: u64, eta: u64) -> u64 {
    let zc = cfg.zeta(c);
    let num = mul(zc, sub(pow(eta, cfg.k_deg), 1));
    let den = inv(mul(cfg.k_deg % P, sub(eta, zc)));
    mul(num, den)
}

// ---------------------------------------------------------------------------
// Merkle — blake3 over little-endian-packed columns. Byte layout MUST match
// protocol.pack_column / merkle_leaf / merkle_verify.
// ---------------------------------------------------------------------------
pub const EMPTY_COMMIT_ROOT: [u8; 32] = [0u8; 32];

pub fn pack_column(col: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(col.len() * 8);
    for &v in col {
        out.extend_from_slice(&v.to_le_bytes());     // 8-byte little-endian, as Python
    }
    out
}

pub fn merkle_leaf(col: &[u64]) -> [u8; 32] {
    *blake3::hash(&pack_column(col)).as_bytes()
}

/// path: list of (sibling, side); side==0 ⇒ sibling is the LEFT child (sib‖h).
pub fn merkle_verify(leaf: [u8; 32], path: &[([u8; 32], u8)], root: [u8; 32]) -> bool {
    let mut h = leaf;
    for (sibling, side) in path {
        let mut buf = [0u8; 64];
        if *side == 0 {
            buf[..32].copy_from_slice(sibling);
            buf[32..].copy_from_slice(&h);
        } else {
            buf[..32].copy_from_slice(&h);
            buf[32..].copy_from_slice(sibling);
        }
        h = *blake3::hash(&buf).as_bytes();
    }
    h == root
}

// ---------------------------------------------------------------------------
// Challenge PRF — blake3(seed_32 ‖ label ‖ index_8LE), low 16 bytes mod P.
// MUST match protocol.challenge byte-for-byte.
// ---------------------------------------------------------------------------
/// Seeds are 32 bytes. (Python also accepts ints; the Rust verifier only ever
/// sees byte seeds, so we take &[u8] and pad/truncate to 32 like _seed_bytes.)
fn seed_bytes(seed: &[u8]) -> [u8; 32] {
    let mut s = [0u8; 32];
    let n = seed.len().min(32);
    s[..n].copy_from_slice(&seed[..n]);
    s
}

pub fn challenge(seed: &[u8], index: u64, label: &str) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(&seed_bytes(seed));
    h.update(label.as_bytes());
    h.update(&index.to_le_bytes());                  // 8-byte little-endian index
    let digest = h.finalize();
    let b = digest.as_bytes();
    let lo = u128::from_le_bytes(b[..16].try_into().unwrap());   // low 16 bytes, LE
    (lo % P as u128) as u64
}

/// The linear-fold challenge source (analysis/docs/linear-fold-unification.md):
/// one tiny surface for all r_lin access. Three call patterns, chosen
/// statically per constraint-family kind: `at` (repeat runs — one hash per
/// run), `sum` (fan runs and rhs runs — streamed, O(1) memory), `preload`
/// (strided-repeat families whose cid spans are small and dense, e.g. the
/// Freivalds sides at ≤ H·K cids; callers gate on span size).
pub struct Chal<'a> { s_comb: &'a [u8] }

impl<'a> Chal<'a> {
    pub fn new(s_comb: &'a [u8]) -> Self { Chal { s_comb } }
    /// r_lin[cid] — one BLAKE3.
    pub fn at(&self, cid: usize) -> u64 { challenge(self.s_comb, cid as u64, "lin") }
    /// Σ_{cid ∈ [lo, hi)} r_lin[cid], ascending — streamed, O(1) memory.
    pub fn sum(&self, lo: usize, hi: usize) -> u64 {
        (lo..hi).fold(0u64, |acc, cid| add(acc, self.at(cid)))
    }
    /// Materialize r_lin over [lo, hi) — small dense spans only.
    pub fn preload(&self, lo: usize, hi: usize) -> Vec<u64> {
        (lo..hi).map(|cid| self.at(cid)).collect()
    }
}

/// `count` distinct indices in [0, range_max), sorted — blake3 rejection
/// sampling, bit-exact with random_columns_n.
pub fn random_columns_n(seed: &[u8], count: usize, range_max: u64) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::with_capacity(count);
    let mut k: u64 = 0;
    while out.len() < count {
        let j = challenge(seed, k, "col") % range_max;
        if !out.contains(&j) {
            out.push(j);
        }
        k += 1;
    }
    out.sort_unstable();
    out
}

pub fn random_columns(seed: &[u8], cfg: &Config) -> Vec<u64> {
    random_columns_n(seed, cfg.t_queries, cfg.n_lig)
}

/// op_vec: the per-claim op-challenge vector, v[i] = challenge(s_op, i,
/// "op{ci}:{label}") — matches protocol.op_vec.
pub fn op_vec(s_op: &[u8], claim_index: usize, label: &str, n: usize) -> Vec<u64> {
    let lab = format!("op{}:{}", claim_index, label);
    (0..n as u64).map(|i| challenge(s_op, i, &lab)).collect()
}

#[cfg(test)]
mod chal_tests {
    use super::*;

    // Chal is definitionally a view over challenge(s_comb, ·, "lin"); pin that.
    #[test]
    fn at_sum_preload_match_challenge() {
        let chal = Chal::new(b"unit-seed");
        let serial = (7u64..1000).fold(0u64, |a, c| add(a, challenge(b"unit-seed", c, "lin")));
        assert_eq!(chal.sum(7, 1000), serial);
        let pre = chal.preload(7, 1000);
        assert_eq!(pre.len(), 993);
        for (i, &v) in pre.iter().enumerate() {
            assert_eq!(v, chal.at(7 + i));
            assert_eq!(v, challenge(b"unit-seed", (7 + i) as u64, "lin"));
        }
    }
}
