//! Bit-exactness diff-test for the foundation: prints Rust values for a fixed
//! set of inputs; a Python script prints the same via protocol.py and asserts
//! equality. Catches byte-layout / reduction bugs before the compile is built.
use ligero_verifier::field::{add, mul, sub, inv, pow};
use ligero_verifier::protocol::{challenge, random_columns_n, op_vec, merkle_leaf,
                                merkle_verify, lagrange, poly_eval, Config};

fn main() {
    // field ops on a few edge + random values
    let xs: [u64; 5] = [0, 1, 2, 0xFFFF_FFFF_0000_0000, 123456789012345];
    for &a in &xs { for &b in &xs {
        println!("add {} {} {}", a, b, add(a, b));
        println!("sub {} {} {}", a, b, sub(a, b));
        println!("mul {} {} {}", a, b, mul(a, b));
    }}
    for &a in &xs { if a != 0 { println!("inv {} {}", a, inv(a)); } }
    println!("pow {} {} {}", 7u64, 1000000u64, pow(7, 1000000));

    // challenge PRF (byte layout is the hazard)
    let seed = b"difftest-seed-0";
    for i in 0u64..6 { for lab in ["irs", "lin", "quad", "op0:rho"] {
        println!("challenge {} {} {}", i, lab, challenge(seed, i, lab));
    }}

    // columns + op_vec
    let cfg = Config { ell: 8, k_deg: 8, n_lig: 32, t_queries: 4 };
    println!("cols {:?}", random_columns_n(seed, 4, 32));
    println!("opvec {:?}", op_vec(seed, 3, "rho", 5));

    // domains + lagrange + poly_eval
    for j in 0u64..4 { println!("eta {} {}", j, cfg.eta(j)); }
    for c in 0u64..4 { println!("zeta {} {}", c, cfg.zeta(c)); }
    println!("lagrange {}", lagrange(&cfg, 2, cfg.eta(1)));
    println!("polyeval {}", poly_eval(&[3, 5, 7, 9], 12345));

    // merkle: leaf of a column, and a 2-leaf path verify
    let col = vec![10u64, 20, 30];
    let leaf = merkle_leaf(&col);
    println!("leaf {}", hex(&leaf));
    let sib = merkle_leaf(&vec![40u64, 50]);
    let root = *blake3::hash(&[sib.as_slice(), leaf.as_slice()].concat()).as_bytes();
    // print Python-style bool so the diff is exact
    let ok = merkle_verify(leaf, &[(sib, 0)], root);                  // side 0 = sib left
    println!("mverify {}", if ok { "True" } else { "False" });
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}
