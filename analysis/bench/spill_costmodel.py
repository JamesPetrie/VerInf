"""Cost model + measurement for the witness SPILL (store-once, re-read in
rounds 2-4 instead of recompute). Answer, before prototyping: does re-reading
the witness from host beat recomputing it on GPU?

Model (per round, per op producing V bytes of committed witness):
    recompute:   T_rc  = GPU forward-pass time for that op's witness
    spill re-read: T_sp = V / BW_hostGPU     (host->device transfer)
Spill wins for an op iff  T_sp < T_rc. Aggregate over the witness, times the
3 rounds (2,3,4) where the witness is re-needed, minus the one-time write.

Key inputs measured on THIS machine:
  - BW_hostGPU: pinned-host <-> device copy bandwidth (PCIe).
  - T_rc: one witness forward pass (run_engine_pass) at a medium config.
  - V_witness: total committed witness bytes (m_total * ELL * 8).
"""
import sys, time
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
from core import LigeroConfig, _with_synthesized_settlements, _layout, _sample_chs, _StreamingPackets
from tape import Tape
import protocol as pr

# ---- 1. host<->GPU bandwidth (pinned) ----
def measure_bw(nbytes=1 << 30, reps=5):
    n = nbytes // 8
    host = torch.empty(n, dtype=torch.int64, pin_memory=True)
    dev = torch.empty(n, dtype=torch.int64, device="cuda")
    for _ in range(2):  # warm
        dev.copy_(host, non_blocking=True); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    h2d = nbytes * reps / (time.time() - t0) / 1e9
    t0 = time.time()
    for _ in range(reps):
        host.copy_(dev, non_blocking=True)
    torch.cuda.synchronize()
    d2h = nbytes * reps / (time.time() - t0) / 1e9
    return h2d, d2h

# ---- 2 & 3. witness recompute time + bytes at a medium config ----
CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
D, DFF, DH, SEQ, NL = 512, 2048, 64, 512, 4
dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
H = D // DH


def build():
    tape = Tape(CFG, silu_config=dt.SILU_CFG, lazy=True)
    x = dt._rand_signed(SEQ * D, half=dt.HALF_X)
    resid = tape.commit("x_input", x, (SEQ, D))
    for L in range(NL):
        resid = dt._run_block(tape, resid, dt._commit_weights_random(tape, L), H=H)
    vocab = 64
    fn = torch.full((D,), dt.S, dtype=torch.uint64, device="cuda")
    lm = dt._rand_signed(D * vocab, half=dt.HALF)
    dt._run_tail(tape, resid, tape.commit("fn", fn, (D,)), tape.commit("lm", lm, (D, vocab)), vocab_size=vocab)
    return tape


print("=== 1. host<->GPU bandwidth (pinned, PCIe) ===")
h2d, d2h = measure_bw()
print(f"  H2D {h2d:.1f} GB/s   D2H {d2h:.1f} GB/s")

print("\n=== 2. witness size + one-forward-pass recompute time ===")
tape = build()
logits = None
for c in tape.claims[::-1]:
    pass
claims = _with_synthesized_settlements(tape.claims)
(_all, p1, p2, m_p1, m_total, *_r) = _layout(claims, CFG)
W_bytes = m_total * CFG.ELL * 8
# time one witness forward pass (engine)
lastvar = tape._deferred[-1][0]
torch.cuda.synchronize(); t0 = time.time()
tape.run_engine_pass(free_intermediates=True, keep=set())
torch.cuda.synchronize()
t_rc = time.time() - t0
print(f"  m_total={m_total:,}  W={m_total*CFG.ELL:,} elems  = {W_bytes/1e9:.2f} GB witness")
print(f"  one forward-pass recompute: {t_rc:.2f}s")

print("\n=== 3. crossover: re-read vs recompute (per round) ===")
t_reread = W_bytes / (h2d * 1e9)
print(f"  spill re-read (W / H2D): {t_reread:.2f}s   vs   recompute: {t_rc:.2f}s")
verdict = "SPILL WINS" if t_reread < t_rc else "RECOMPUTE WINS (spill is slower)"
print(f"  -> per round: {verdict}  (ratio reread/recompute = {t_reread/t_rc:.2f})")
saved = 3 * (t_rc - t_reread)      # 3 rounds re-read instead of recompute
print(f"  over 3 rounds: recompute {3*t_rc:.2f}s vs spill {t_reread + 3*t_reread:.2f}s "
      f"(1 write + 3 reads); net {'save' if saved>0 else 'LOSE'} {abs(saved):.2f}s")

print("\n=== 4. scaling note ===")
print(f"  spill wins only where W/BW < recompute. Post-GPU-softmax the forward")
print(f"  pass is GPU-fast (~{W_bytes/1e9/max(t_rc,1e-9):.0f} GB/s effective compute)")
print(f"  while H2D is {h2d:.0f} GB/s. If compute-throughput >> PCIe, spill loses.")
