"""Soundness gate for the GPU softmax port: proving the SAME toy model with the
GPU path OFF (numpy) vs ON must produce a BYTE-IDENTICAL proof, and the GPU-on
proof must Rust-verify ACCEPT. Single process, fixed seed -> identical weights,
so the ONLY difference is the softmax witness path."""
import sys
from pathlib import Path
R = Path("/home/riftuser/VerInf")
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "prover/tests"))
sys.path.insert(0, str(R / "demo"))
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
import compute_fns as CF
from core import LigeroConfig
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
SEED = b"validate-gpu-softmax"
C._WITNESS_CACHE_ON = True   # keep the applied cache on; isolate the softmax var


def build():
    tape = Tape(CFG, silu_config=dt.SILU_CFG, lazy=True)
    x = dt._rand_signed(dt.SEQ * dt.d, half=dt.HALF_X)
    resid = tape.commit("x_input", x, (dt.SEQ, dt.d))
    w = dt._commit_weights_random(tape, layer_idx=0)
    resid = dt._run_block(tape, resid, w, H=dt.d // dt.d_h)
    vocab = 64
    fn = torch.full((dt.d,), dt.S, dtype=torch.uint64, device="cuda")
    lm = dt._rand_signed(dt.d * vocab, half=dt.HALF)
    fnw = tape.commit("final_norm_w", fn, (dt.d,))
    lmw = tape.commit("W_lm_head", lm, (dt.d, vocab))
    dt._run_tail(tape, resid, fnw, lmw, vocab_size=vocab)
    return tape


torch.manual_seed(1234)
CF._GPU_SOFTMAX_ON = False
p_off = build().prove(seed=SEED)

torch.manual_seed(1234)
CF._GPU_SOFTMAX_ON = True
tape_on = build()
p_on = tape_on.prove(seed=SEED)


def teq(a, b): return torch.equal(a, b)
checks = {
    "root_p1": p_off.root_p1 == p_on.root_p1,
    "root_p2": p_off.root_p2 == p_on.root_p2,
    "q_irs": teq(p_off.q_irs, p_on.q_irs),
    "q_lin": teq(p_off.q_lin, p_on.q_lin),
    "p_0": teq(p_off.p_0, p_on.p_0),
    "opened_p1": all(teq(p_off.opened_p1[j], p_on.opened_p1[j]) for j in p_off.opened_p1),
    "opened_p2": all(teq(p_off.opened_p2[j], p_on.opened_p2[j]) for j in p_off.opened_p2),
}
print("=== GPU softmax soundness: numpy vs gpu, identical toy model ===")
for k, v in checks.items():
    print(f"  {k:12s}: {'OK identical' if v else 'MISMATCH'}")
allok = all(checks.values())
print("byte-identical proof:", "YES" if allok else "NO — UNSOUND, do not apply")

acc, msg = rust_verify_tape(tape_on, p_on, seed=SEED)
print("Rust verify (gpu-on proof):", "ACCEPT" if acc else "REJECT")
if not acc: print(msg)
print("DONE")
sys.exit(0 if (allok and acc) else 1)
