"""Soundness gate for wiring coset-NTT into the real encode path: proving the
SAME model with coset-NTT OFF vs forced ON must give a byte-identical proof
(identical Merkle roots + test polynomials + opened columns). Single process,
fixed seed -> identical model; only the encode path differs. Forces coset ON at
the test K=1024 (where 'auto' would leave it off) purely to exercise the new
code path against the reference; speed there is irrelevant (coset only wins at
K>=2^16 — that win is shown by analysis/bench/coset_ntt_ab.py, not here)."""
import sys
from pathlib import Path
R = Path("/home/riftuser/VerInf")
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
from core import LigeroConfig
from tape import Tape

CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)


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
C._COSET_NTT_MODE = "off"
p_off = build().prove(seed=b"validate-coset")

torch.manual_seed(1234)
C._COSET_NTT_MODE = "on"          # force the coset path at K=1024
p_on = build().prove(seed=b"validate-coset")

teq = torch.equal
checks = {
    "root_p1": p_off.root_p1 == p_on.root_p1,
    "root_p2": p_off.root_p2 == p_on.root_p2,
    "q_irs": teq(p_off.q_irs, p_on.q_irs),
    "q_lin": teq(p_off.q_lin, p_on.q_lin),
    "p_0": teq(p_off.p_0, p_on.p_0),
    "opened_p1": all(teq(p_off.opened_p1[j], p_on.opened_p1[j]) for j in p_off.opened_p1),
    "opened_p2": all(teq(p_off.opened_p2[j], p_on.opened_p2[j]) for j in p_off.opened_p2),
}
print("=== coset-NTT encode soundness: single-N path vs coset path, identical model ===")
for k, v in checks.items():
    print(f"  {k:12s}: {'OK identical' if v else 'MISMATCH'}")
allok = all(checks.values())
print(f"\n{'ALL IDENTICAL — coset encode produces a byte-for-byte identical proof' if allok else 'MISMATCH — coset wiring is UNSOUND, do not keep'}")
sys.exit(0 if allok else 1)
