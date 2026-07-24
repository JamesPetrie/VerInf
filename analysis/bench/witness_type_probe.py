"""Decompose the REMAINING witness bucket by compute_fn type, cache ON, so we
know which deterministic op to consider caching next (softmax+silu are already
cached -> they'll show ~1x call count; everything else recomputes 4x). Wraps
COMPUTE_FNS with CUDA-synced timers. Reports total s + call count per type."""
import sys, os, time, collections
from pathlib import Path
R = Path("/home/riftuser/VerInf")
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
sys.path.insert(0, str(R / "analysis/bench"))
os.environ["LIGERO_PHASE_TIMING"] = "1"
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
import compute_fns as CF
from core import LigeroConfig
from tape import Tape

CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
D, DFF, DH, SEQ, NL = 512, 2048, 64, 512, 4     # medium, fast (~34s cache on)
dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
H = D // DH

# Wrap every compute fn with a per-type CUDA-synced timer + a bytes-per-call read.
_tsum = collections.defaultdict(float)
_tcnt = collections.defaultdict(int)
_tbytes = collections.defaultdict(int)
_orig = dict(CF.COMPUTE_FNS)
def _wrap(kls, fn):
    name = kls.__name__
    def timed(claim, input_data):
        torch.cuda.synchronize(); t0 = time.time()
        out = fn(claim, input_data)
        torch.cuda.synchronize()
        _tsum[name] += time.time() - t0
        _tcnt[name] += 1
        try:
            _tbytes[name] = max(_tbytes[name],
                                sum(t.numel() * t.element_size() for t in out.values()))
        except Exception:
            pass
        return out
    return timed
for kls, fn in _orig.items():
    CF.COMPUTE_FNS[kls] = _wrap(kls, fn)


def build():
    tape = Tape(CFG, silu_config=dt.SILU_CFG, lazy=True)
    x = dt._rand_signed(SEQ * D, half=dt.HALF_X)
    resid = tape.commit("x_input", x, (SEQ, D))
    for L in range(NL):
        w = dt._commit_weights_random(tape, layer_idx=L)
        resid = dt._run_block(tape, resid, w, H=H)
    vocab = 64
    fn = torch.full((D,), dt.S, dtype=torch.uint64, device="cuda")
    lm = dt._rand_signed(D * vocab, half=dt.HALF)
    fnw = tape.commit("final_norm_w", fn, (D,))
    lmw = tape.commit("W_lm_head", lm, (D, vocab))
    dt._run_tail(tape, resid, fnw, lmw, vocab_size=vocab)
    return tape


torch.manual_seed(1234)
C._WITNESS_CACHE_ON = True
tape = build()
tape.prove(seed=b"type-probe")
print(f"config d{D},ff{DFF},seq{SEQ},L{NL}  cache ON  (softmax+silu cached)")
print(f"{'type':<22}{'total_s':>9}{'calls':>7}{'MB/call':>9}")
for name in sorted(_tsum, key=lambda k: -_tsum[k]):
    print(f"{name:<22}{_tsum[name]:9.2f}{_tcnt[name]:7d}{_tbytes[name]/1e6:9.1f}")
print(f"{'WITNESS COMPUTE TOTAL':<22}{sum(_tsum.values()):9.2f}")
print("DONE")
