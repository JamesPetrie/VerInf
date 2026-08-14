"""Soundness gate for the DISK-backed full-witness spill: proving the SAME model
with LIGERO_WITNESS_SPILL_DISK off vs on must give a BYTE-IDENTICAL proof, and the
disk-spilled proof must Rust-verify = ACCEPT. Disk spill re-reads every committed
row from a file instead of recomputing; each value is the identical int64 bit
pattern, so the proof is unchanged. Single process, fixed seed -> identical model;
only the witness backing store (recompute vs disk re-read) differs."""
import sys, os
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "prover/tests"))
sys.path.insert(0, str(R / "demo"))
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core
from core import LigeroConfig
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
SEED = b"validate-disk-spill"
os.environ["LIGERO_WITNESS_SPILL_DIR"] = str(R / "analysis/bench/_spill_tmp")


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


# baseline: recompute (no spill of any kind)
torch.manual_seed(1234)
core._WITNESS_SPILL_ON = False
core._WITNESS_SPILL_DISK = False
p_off = build().prove(seed=SEED)

# disk spill on: full witness re-read from a disk file in rounds 2-4
torch.manual_seed(1234)
core._WITNESS_SPILL_DISK = True
tape_on = build()
p_on = tape_on.prove(seed=SEED)

teq = torch.equal
checks = {
    "root_p1": p_off.root_p1 == p_on.root_p1,
    "root_p2": p_off.root_p2 == p_on.root_p2,
    "q_irs": teq(p_off.q_irs, p_on.q_irs), "q_lin": teq(p_off.q_lin, p_on.q_lin),
    "p_0": teq(p_off.p_0, p_on.p_0),
    "opened_p1": all(teq(p_off.opened_p1[j], p_on.opened_p1[j]) for j in p_off.opened_p1),
    "opened_p2": all(teq(p_off.opened_p2[j], p_on.opened_p2[j]) for j in p_off.opened_p2),
}
print("=== DISK-spill soundness: recompute vs disk re-read, identical model ===")
for k, v in checks.items():
    print(f"  {k:12s}: {'OK identical' if v else 'MISMATCH'}")
identical = all(checks.values())
print(f"  {'ALL IDENTICAL — disk spill produces a byte-for-byte identical proof' if identical else 'MISMATCH — disk spill is UNSOUND'}")

# and the disk-spilled proof must Rust-ACCEPT
acc, msg = rust_verify_tape(tape_on, p_on, seed=SEED)
print(f"  Rust verify_proof (disk spill ON): {'ACCEPT' if acc else 'REJECT'}")
if not acc:
    print(f"    {str(msg)[:400]}")
ok = identical and acc
print(f"\n{'PASS — disk-spill is byte-identical AND ACCEPTs' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
