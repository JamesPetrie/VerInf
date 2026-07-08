//! Bit-exact Rust port of the Python standalone Ligero verifier
//! (ligero/pipeline/protocol.py + verify.py). Differential-tested against the
//! Python verifier on identical proofs + seeds.
//!
//! TCB = field + protocol (domains/Merkle/challenge) + compile + the 6 checks.
//! No prover logic; the verifier compiles its own constraints (never trusts
//! prover- or Python-supplied ones).

pub mod field;
pub mod protocol;
pub mod claim;
pub mod compile;
pub mod handlers;
pub mod prover;
pub mod verify;   // the verification side: decode + protocol + the 6 checks
