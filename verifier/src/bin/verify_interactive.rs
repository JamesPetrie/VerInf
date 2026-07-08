//! Drive the staged Ligero protocol from Rust against the live Python prover.
//! The Rust verifier draws its own coins (/dev/urandom) and exchanges only seeds
//! + transcript with prover_server.py over a subprocess pipe. The witness never
//! leaves the prover process; the statement (claims) is loaded from a trusted
//! claims.json, never from the prover.
//!
//! Mirrors test_verify_real.py: run both modes against a correct prover (expect
//! ACCEPT) and a tampered prover (expect REJECT).
//!
//! Env:  LIGERO_PYTHON (default python3), LIGERO_PIPELINE (default ../pipeline),
//!       LIGERO_CLAIMS (default /tmp/claims.json).
//! Args: [statement]  (matmul|chain|rmsnorm; default matmul)
use std::fs::File;
use std::io::Read;
use ligero_verifier::claim::{parse_claim_set, ClaimSet};
use ligero_verifier::prover::SubprocessProver;
use ligero_verifier::verify::{run_verification, run_verification_fast};

/// A fresh 32-byte coin from the OS CSPRNG (the verifier's `rand`). No extra deps.
fn fresh_seed() -> Vec<u8> {
    let mut b = [0u8; 32];
    File::open("/dev/urandom").expect("open /dev/urandom")
        .read_exact(&mut b).expect("read /dev/urandom");
    b.to_vec()
}

fn load_claims(path: &str) -> ClaimSet {
    let txt = std::fs::read_to_string(path).expect("read claims.json");
    parse_claim_set(&txt)
}

fn main() {
    let statement = std::env::args().nth(1).unwrap_or_else(|| "matmul".into());
    let python = std::env::var("LIGERO_PYTHON").unwrap_or_else(|_| "python3".into());
    let pipeline = std::env::var("LIGERO_PIPELINE").unwrap_or_else(|_| "../pipeline".into());
    let claims_path = std::env::var("LIGERO_CLAIMS").unwrap_or_else(|_| "/tmp/claims.json".into());

    // Spawn a correct prover, run both modes (fresh claims per run: verify()
    // patches table α/β into the ClaimSet, so each mode gets a clean copy).
    let mut p = SubprocessProver::spawn(&python, &pipeline, &["prover_server.py", &statement])
        .expect("spawn prover");
    let ok_sound = run_verification(&mut p, &mut load_claims(&claims_path), fresh_seed);
    let ok_fast = run_verification_fast(&mut p, &mut load_claims(&claims_path), fresh_seed);
    drop(p);

    // Spawn a tampered prover, run both modes (expect REJECT).
    let mut pt = SubprocessProver::spawn(&python, &pipeline,
        &["prover_server.py", &statement, "--tamper"]).expect("spawn tampered prover");
    let rej_sound = !run_verification(&mut pt, &mut load_claims(&claims_path), fresh_seed);
    let rej_fast = !run_verification_fast(&mut pt, &mut load_claims(&claims_path), fresh_seed);
    drop(pt);

    println!("statement: {}", statement);
    println!("  interactive (sound): {}", if ok_sound { "ACCEPT" } else { "REJECT" });
    println!("  fused (fast):        {}", if ok_fast { "ACCEPT" } else { "REJECT" });
    println!("  tamper interactive:  {}", if rej_sound { "REJECT" } else { "ACCEPT(BUG!)" });
    println!("  tamper fused:        {}", if rej_fast { "REJECT" } else { "ACCEPT(BUG!)" });
    let all = ok_sound && ok_fast && rej_sound && rej_fast;
    println!("result: {}", if all { "PASS" } else { "FAIL" });
    std::process::exit(if all { 0 } else { 1 });
}
