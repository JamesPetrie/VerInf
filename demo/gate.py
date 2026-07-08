"""Prove -> verify gate (and quickstart) on the real Llama-2-7B weights.

Run this after any large code change, or to try the system end to end:

    python demo/gate.py

It proves a small SOUND Llama-2-7B forward on the REAL weights, with the
calibrated unexplained-information bound (fine-scale LM head + matched kernel),
output tokens hidden, then checks the proof with the independent Rust verifier.
It passes only if the verifier ACCEPTs AND the U bound is non-degenerate, so it
catches both prover/verifier regressions and UI-calibration regressions. Exit
code is 0 on pass, 1 otherwise -- drops straight into CI.

Requirements:
  * The Llama-2-7B weights, downloaded (the model is gated):
        huggingface-cli login
        huggingface-cli download meta-llama/Llama-2-7b-hf
    or set VERINF_GATE_MODEL to a local path.
  * A CUDA GPU (the prover JIT-compiles CUDA kernels on first run).
  * Rust/cargo (the verifier is auto-built on first run).

Knobs (env):
  LIGERO_T_QUERIES    opened columns, default 10 here (GATE speed; production
                      soundness is 80 -- this is a regression check, not a
                      soundness-grade proof).
  VERINF_GATE_MODEL HF id or local path (default meta-llama/Llama-2-7b-hf).
  VERINF_GATE_LAYERS  transformer layers, default 32 (the full model).
"""
import os
# Must be set BEFORE importing the demo: the config reads T_QUERIES at import.
os.environ.setdefault("LIGERO_T_QUERIES", "10")
# Use the local HF cache; do not silently auto-download a gated model.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import io
import re
import sys
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prover"))
sys.path.insert(0, str(ROOT / "demo"))

MODEL = os.environ.get("VERINF_GATE_MODEL",
                       os.environ.get("INFPROOF_GATE_MODEL",   # legacy name
                                      "meta-llama/Llama-2-7b-hf"))
LAYERS = int(os.environ.get("VERINF_GATE_LAYERS",
                            os.environ.get("INFPROOF_GATE_LAYERS", "32")))
# "<s> The capital of France is" -- a fixed, deterministic token stream so the
# gate needs no tokenizer call and is reproducible.
TOKENS = [1, 450, 7483, 310, 3444, 338]
U_PER_TOK_MAX = 2.0      # non-degenerate guard: log2(vocab)=14.97 (coarse) / blow-ups fail


class _Tee:
    """Write-through stdout: echo live AND capture, so we can parse the U line."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _check_model():
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(MODEL)         # cheap: config only, no weights
    except Exception as e:
        sys.exit(f"[gate] FAIL: model {MODEL!r} not available ({type(e).__name__}). "
                 "Download it -- the Llama-2 weights are gated: accept the license on HF, "
                 "`huggingface-cli login`, `huggingface-cli download meta-llama/Llama-2-7b-hf` "
                 "-- or set VERINF_GATE_MODEL to a local path.")


def _verifier_bin() -> pathlib.Path:
    crate = ROOT / "verifier"
    binp = crate / "target" / "release" / "verify_proof"
    if binp.exists():
        return binp
    print("[gate] verifier not built -- running `cargo build --release` ...", flush=True)
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=crate, check=True)
    except FileNotFoundError:
        sys.exit("[gate] FAIL: `cargo` not found -- install Rust (https://rustup.rs) "
                 "or build verifier/ manually, then re-run.")
    if not binp.exists():
        sys.exit(f"[gate] FAIL: build completed but {binp} is missing.")
    return binp


def main() -> int:
    _check_model()
    import demo_llama7b
    proof = pathlib.Path(tempfile.gettempdir()) / "infproof_gate.json"

    tq = os.environ["LIGERO_T_QUERIES"]
    print(f"[gate] proving: {LAYERS} layers, SEQ={len(TOKENS)}, T_QUERIES={tq}, real weights "
          f"({MODEL}), calibrated unexplained-info ...", flush=True)

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = _Tee(old, buf)
    try:
        demo_llama7b.main(
            from_hf=MODEL, num_layers=LAYERS, token_ids=TOKENS,
            unexplained_info=True,
            ui_lm_sout=4096, ui_lm_ow=26, ui_s_c=1 << 18,   # fine-scale, calibrated kernel
            engine=True, lazy_weights=True,
            dump_proof=str(proof))
    finally:
        sys.stdout = old
    out = buf.getvalue()

    # Parse the proven U bound.
    m = re.search(r"U = ([\d.]+) bits over (\d+)", out)
    if not m:
        print("[gate] FAIL: no unexplained-information bound found (UI did not run).", flush=True)
        return 1
    u_total, n_tok = float(m.group(1)), int(m.group(2))
    u_per_tok = u_total / max(n_tok, 1)
    u_ok = u_per_tok < U_PER_TOK_MAX
    print(f"[gate] U = {u_total:.4f} bits over {n_tok} tokens = {u_per_tok:.4f} bits/token "
          f"({'non-degenerate' if u_ok else 'DEGENERATE -- calibration regression'})", flush=True)

    vbin = _verifier_bin()
    print(f"[gate] verifying with {vbin.relative_to(ROOT)} ...", flush=True)
    res = subprocess.run([str(vbin), str(proof)], capture_output=True, text=True)
    sys.stdout.write(res.stdout)
    if res.stderr.strip():
        sys.stderr.write(res.stderr)
    accept = res.returncode == 0 and "rust_verify: ACCEPT" in res.stdout

    ok = accept and u_ok
    print(f"\n[gate] {'PASS -- ACCEPT + non-degenerate U' if ok else 'FAIL'} "
          f"(verify={'ACCEPT' if accept else 'REJECT/err'}, U/tok={u_per_tok:.3f})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
