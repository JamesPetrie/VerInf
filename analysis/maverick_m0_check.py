"""M0: Maverick GGUF readability + reference-forward check (no proof involved).

Part A (self-contained, CPU): dequantize one MoE layer from the UD-Q4_K_XL
GGUF, run the layer's MoE module forward in fp32 — replicating the verified
llama.cpp/HF semantics (sigmoid top-1, weight-BEFORE-FFN input scaling, shared
expert on the unscaled stream) — and report routing behavior, router-logit
magnitudes (sizes the proof's sigmoid table), and output stats. Routing is
computed from the (tiny, F32) router first, so only the experts actually
chosen get dequantized.

Part B (cross-check vs llama.cpp itself, same weights): parse a
llama-eval-callback log and (1) verify probs == sigmoid(logits) at the
sampled positions of blk.N's router nodes — runtime confirmation of the
sigmoid semantics; (2) report REAL router-logit samples on real input for the
sigmoid-table range; (3) confirm the weighted-input node precedes the expert
matmuls (weight-before-FFN). NOTE: eval-callback prints only sampled elements
per tensor, so a full input→output replication check is NOT possible from its
log; the bit-level forward comparison happens on the Spark where we control
inference. --capture runs the binary (CPU, 1 short prompt) to produce the log.

Dry-run (H100):
    ~/miniconda3/bin/python maverick_m0_check.py --gguf ~/maverick-gguf/UD-Q4_K_XL
    ~/miniconda3/bin/python maverick_m0_check.py --capture --llama-cpp ~/llama.cpp/build/bin
    ~/miniconda3/bin/python maverick_m0_check.py --eval-log /tmp/m0_eval.log
"""
import argparse
import math
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prover"))

LAYER = 1                 # first MoE layer (MoE on odd indices)
T = 4                     # tokens for the synthetic forward


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def moe_forward(x, router, gate, up, down, gate_sh, up_sh, down_sh):
    """fp32 reference of llama.cpp build_moe_ffn(weight_before_ffn) + shared.
    x: (T, d) POST-ffn_norm input. Weights in (d_out, d_in) numpy layout.
    gate/up/down are dicts {expert_idx: (d_out, d_in)}."""
    logits = x @ router.T                                   # (T, E)
    e_star = logits.argmax(axis=1)                          # top-1 (sigmoid monotone)
    s = _sigmoid(logits[np.arange(len(x)), e_star])         # routing weight
    routed = np.empty_like(x)
    for t in range(len(x)):
        xw = s[t] * x[t]                                    # weight BEFORE the FFN
        e = int(e_star[t])
        h = _silu(xw @ gate[e].T) * (xw @ up[e].T)
        routed[t] = h @ down[e].T
    h_sh = _silu(x @ gate_sh.T) * (x @ up_sh.T)
    return routed + h_sh @ down_sh.T, logits, e_star, s


def part_a(gguf_path):
    from loader import read_maverick_moe_layer
    print(f"── Part A: fp32 reference forward of blk.{LAYER} ──")
    rng = np.random.default_rng(7)
    # Router first (F32, tiny) → routing decides which experts to dequantize.
    # (expert_indices=[0]: gguf-py can't dequantize an empty slice.)
    router = read_maverick_moe_layer(gguf_path, LAYER, expert_indices=[0])["router"]
    E, d = router.shape
    # Synthetic post-norm input at a realistic RMS-normed magnitude (~unit RMS).
    x = rng.standard_normal((T, d)).astype(np.float32)
    logits = x @ router.T
    chosen = sorted(set(int(e) for e in logits.argmax(axis=1)))
    print(f"  router (E={E}, d={d});  chosen experts for {T} synthetic tokens: {chosen}")
    raw = read_maverick_moe_layer(gguf_path, LAYER, expert_indices=chosen)
    gate = {e: raw["gate_exps"][i] for i, e in enumerate(chosen)}
    up = {e: raw["up_exps"][i] for i, e in enumerate(chosen)}
    down = {e: raw["down_exps"][i] for i, e in enumerate(chosen)}
    y, logits, e_star, s = moe_forward(x, router, gate, up, down,
                                        raw["gate_sh"], raw["up_sh"], raw["down_sh"])
    assert np.isfinite(y).all(), "non-finite output"
    lmax = float(np.abs(logits).max())
    print(f"  router logits: max|r| = {lmax:.3f}  (proof sigmoid table covers ±8.0 "
          f"real units at SIG_BITS=16, S=2^12 → {'OK' if lmax < 8 else 'TOO NARROW'})")
    print(f"  sigmoid weights s = {np.round(s, 4).tolist()}")
    print(f"  output: max|y| = {float(np.abs(y).max()):.3f}, "
          f"rms = {float(np.sqrt((y**2).mean())):.4f}")
    print("  Part A OK — layer forward runs end to end on real dequantized weights")
    return True


# ── Part B: llama-eval-callback cross-check ────────────────────────────────

def capture(llama_bin, gguf_path, log_path, prompt="The"):
    """Run llama-eval-callback (CPU, 1 token of context) and save the log."""
    first = sorted(__import__("glob").glob(os.path.join(gguf_path, "*.gguf")))[0] \
        if os.path.isdir(gguf_path) else gguf_path
    cmd = [os.path.join(llama_bin, "llama-eval-callback"), "-m", first,
           "-p", prompt, "-n", "1", "-ngl", "0"]
    print(f"  running: {' '.join(cmd)}  (CPU; mmap pages only what's touched)")
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False,
                       timeout=3600)
    print(f"  log → {log_path}")


def _node_samples(log, name):
    """First sampled float row printed for graph node `name` (the debug-callback
    prefix is `ggml_debug:` in older llama.cpp, `common_debug_cb_eval:` in newer)."""
    m = re.search(rf"(?:ggml_debug|common_debug_cb_eval):\s+{re.escape(name)} = "
                  rf".*?\n(.*?)\n\s*sum\s*=", log, re.S)
    if not m:
        return None
    vals = re.findall(r"-?\d+\.\d+", m.group(1))
    return [float(v) for v in vals[:8]] or None


def part_b(log_path):
    print(f"── Part B: llama-eval-callback cross-check ({log_path}) ──")
    log = open(log_path, errors="replace").read()
    ok = True
    lg = _node_samples(log, f"ffn_moe_logits-{LAYER}")
    pr = _node_samples(log, f"ffn_moe_probs-{LAYER}")
    if lg and pr:
        sig = [_sigmoid(v) for v in lg[:len(pr)]]
        match = all(abs(a - b) < 5e-3 for a, b in zip(sig, pr))
        print(f"  logits[:4] = {lg[:4]}")
        print(f"  sigmoid(logits) vs probs at sampled positions: "
              f"{'MATCH' if match else 'MISMATCH'}")
        print(f"  REAL router-logit samples for the sigmoid-table range: "
              f"max|r| over samples = {max(abs(v) for v in lg):.3f}")
        ok &= match
    else:
        print("  ffn_moe_logits/probs nodes not found — dump node names with: "
              f"grep 'ggml_debug.*-{LAYER} =' {log_path}")
        ok = False
    # weight-before-FFN: the weighted-input node must precede the expert matmul.
    iw = log.find(f"ffn_moe_weighted-{LAYER}")
    ie = log.find(f"ffn_moe_gate-{LAYER}")
    if iw != -1 and ie != -1:
        print(f"  weight-before-FFN node order: "
              f"{'CONFIRMED' if iw < ie else 'UNEXPECTED (weights after experts?)'}")
        ok &= iw < ie
    else:
        print("  weighted/gate nodes not found by those names — inspect the log")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.expanduser("~/maverick-gguf/UD-Q4_K_XL"))
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--llama-cpp", default=os.path.expanduser("~/llama.cpp/build/bin"))
    ap.add_argument("--eval-log", default=None)
    a = ap.parse_args()
    results = []
    if a.capture:
        capture(a.llama_cpp, a.gguf, "/tmp/m0_eval.log")
        results.append(part_b("/tmp/m0_eval.log"))
    elif a.eval_log:
        results.append(part_b(a.eval_log))
    else:
        results.append(part_a(a.gguf))
    raise SystemExit(0 if all(results) else 1)
