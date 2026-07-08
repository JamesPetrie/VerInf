//! The prover handle the staged drivers talk to (mirrors verify.py's `prover`
//! callable). The verifier NEVER reads the prover's code — it only exchanges
//! seeds (out) and transcript messages (in). The prover is untrusted.
//!
//! SubprocessProver spawns the Python prover (prover_server.py) and speaks the
//! newline-delimited JSON line protocol over its stdin/stdout. The witness stays
//! entirely in that process; only seeds and the transcript cross the pipe.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use serde_json::{json, Value};

/// One staged round-trip. `stage` is 1..=4 for the interactive protocol or 0 for
/// the fused (one-shot) variant; seeds are supplied as they become available.
/// Returns the prover's response object (the transcript fields for that stage).
pub trait Prover {
    fn request(&mut self, stage: u8, s_op: Option<&[u8]>,
               s_comb: Option<&[u8]>, s_col: Option<&[u8]>) -> Value;
}

fn to_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

pub struct SubprocessProver {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl SubprocessProver {
    /// Spawn `python prover_server.py <args>` in `cwd`, and sync past startup
    /// noise until the {"ready":true} handshake line.
    pub fn spawn(python: &str, cwd: &str, args: &[&str]) -> std::io::Result<Self> {
        let mut child = Command::new(python)
            .args(args)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr inherits the parent's — CUDA/torch diagnostics go to the
            // terminal, never into the response stream.
            .spawn()?;
        let stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        // read until ready (skip any non-JSON / non-ready lines)
        loop {
            let mut line = String::new();
            if stdout.read_line(&mut line)? == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
                                               "prover exited before ready"));
            }
            if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                if v.get("ready").and_then(|r| r.as_bool()) == Some(true) {
                    break;
                }
            }
        }
        Ok(SubprocessProver { child, stdin, stdout })
    }
}

impl Drop for SubprocessProver {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Prover for SubprocessProver {
    fn request(&mut self, stage: u8, s_op: Option<&[u8]>,
               s_comb: Option<&[u8]>, s_col: Option<&[u8]>) -> Value {
        let mut req = json!({ "stage": stage });
        if let Some(s) = s_op   { req["s_op"]   = json!(to_hex(s)); }
        if let Some(s) = s_comb { req["s_comb"] = json!(to_hex(s)); }
        if let Some(s) = s_col  { req["s_col"]  = json!(to_hex(s)); }
        writeln!(self.stdin, "{}", req).expect("write request to prover");
        self.stdin.flush().expect("flush request");
        // read the next JSON object line (skip any stray non-JSON output)
        loop {
            let mut line = String::new();
            let n = self.stdout.read_line(&mut line).expect("read prover response");
            if n == 0 { panic!("prover closed pipe mid-protocol"); }
            if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                return v;
            }
        }
    }
}
