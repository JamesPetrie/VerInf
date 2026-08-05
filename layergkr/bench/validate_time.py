"""Validate the MACHINE model: predicted seconds vs the stopwatch, no factor.

The count model predicts operation counts exactly. That is the easy half. This is
the hard half -- turning counts into seconds -- and it is where a calibration
factor would otherwise be reintroduced.

The machine is modelled as the cost of the loop bodies it actually runs, each
measured by bench/kernels.py:

    enc     acc += v * L[c]                 a NONZERO encode slot
    scan    the same loop visiting a ZERO slot (skipped by `if v:`)
    comb    out[j] = (out[j] + cw[j]*a) % P
    fold    (f[i] + x*(f[h+i] - f[i])) % P
    red     (a * b) % P
    xpose   one element of rs.Commit's column transpose
    hashcall  pack + BLAKE3 of one column

History of this file, because the method matters more than the number:

    mul/add rate card                     0.69x .. 2.48x
    + split deferred from reduced muls    0.65x .. 2.12x
    + price loop bodies, not primitives   (a least-squares fit could not even
                                           separate mul from add: they are
                                           collinear here, so they were the wrong
                                           unit)
    + model encode sparsity (scan vs mac) 0.47x .. 0.95x
    + model transpose and hash-call cost  0.81x .. 1.01x

Every step was a machine characteristic that had been left out, found and added.
None of them was a coefficient. The residual on the smallest instances is fixed
per-call overhead that is not yet modelled; it shrinks as instances grow, which
is the direction that matters for extrapolation.

  .venv/bin/python layergkr/bench/validate_time.py
"""
import json, pathlib, random, sys, time
sys.path.insert(0, "/home/riftuser/VerInf")
from layergkr import full_layer as fl, rs, semantics as sem
from layergkr.counters import Counter, KernelRates

k = json.loads(pathlib.Path("/home/riftuser/VerInf/layergkr/bench/kernels.json").read_text())
enc = sum(k["encode_ns_per_slot"].values()) / len(k["encode_ns_per_slot"])
comb = sum(k["combine_ns_per_iter"].values()) / len(k["combine_ns_per_iter"])
fold = sum(k["fold_ns_per_iter"].values()) / len(k["fold_ns_per_iter"])
scan = sum(k["encode_scan_ns_per_slot"].values()) / len(k["encode_scan_ns_per_slot"])
xp = sum(k["transpose_ns_per_elem"].values()) / len(k["transpose_ns_per_elem"])
g = k.get("gpu_encode_ns_per_slot") or {}
gpu_enc = min(g.values()) if g else enc
m = k.get("gpu_marshal_ns_per_elem") or {}
gpu_mar = sum(m.values()) / len(m) if m else 0.0
pkd = k.get("pack_ns_per_value") or {}
pack = sum(pkd.values()) / len(pkd) if pkd else 0.0
rates = KernelRates("python+v100", enc, scan, comb, fold, k["reduce_ns"],
                    gpu_enc, gpu_mar, pack, xp, k["hash_call_ns"], 6.9)
print(f"kernel rates: enc {enc:.1f}  scan {scan:.1f}  comb {comb:.1f}  "
      f"fold {fold:.1f}  red {k['reduce_ns']:.1f}  xpose {xp:.1f}  "
      f"hashcall {k['hash_call_ns']:.0f}  gpu_enc {gpu_enc:.4f}  gpu_marshal {gpu_mar:.1f}  pack {pack:.1f} ns\n")

SPECS = [
 dict(name="real-128", S=16, d=128, d_ff=256, E=8, ELL=256, K=512, N=1024, q=16),
 dict(name="real-64", S=8, d=64, d_ff=128, E=4, ELL=128, K=256, N=512, q=12),
 dict(name="toy-xs", S=2, d=4, d_ff=8, E=1, ELL=8, K=16, N=32, q=4),
 dict(name="toy-s", S=4, d=8, d_ff=16, E=1, ELL=16, K=32, N=64, q=4),
 dict(name="small-1", S=6, d=16, d_ff=32, E=2, ELL=32, K=64, N=128, q=8),
 dict(name="small-2", S=8, d=16, d_ff=32, E=4, ELL=32, K=64, N=128, q=8),
 dict(name="large-1", S=8, d=32, d_ff=64, E=4, ELL=64, K=128, N=256, q=12),
 dict(name="large-2", S=12, d=32, d_ff=64, E=8, ELL=64, K=128, N=256, q=12),
]
print(f"{'run':<9} {'meas s':>8} {'model s':>9} {'ratio':>7}   "
      f"{'enc_mac(cpu)':>12} {'enc_gpu':>14} {'fold':>12} {'red':>10}")
rat=[]
for sp in SPECS:
    cfg = rs.Config(ELL=sp["ELL"], K_DEG=sp["K"], N_LIG=sp["N"], T_QUERIES=sp["q"])
    toy = sem.ToyConfig(S=sp["S"], d=sp["d"], d_ff=sp["d_ff"], E=sp["E"],
                        table_bits=6, scale_bits=6)
    rng = random.Random(7)
    # The Lagrange matrix is one-time setup per Config (a profile showed ~3 s of
    # pow() calls at ELL=128/N=512). Build it BEFORE the stopwatch starts, the
    # same way a production prover would build its tables once -- otherwise a
    # setup cost lands in the per-proof measurement.
    rs.lagrange_matrix(cfg)
    tr = sem.forward(toy, rng)
    en = fl.Enrollment(cfg)
    with Counter("p") as c:
        fl.prove_full_layer(tr, cfg, en, sp["q"], rng, use_masks=True)
    rep = c.report()
    model = rates.seconds(rep)
    r = model / rep["seconds"]; rat.append(r)
    print(f"{sp['name']:<9} {rep['seconds']:8.2f} {model:9.2f} {r:7.2f}x   "
          f"{rep['enc_slot']:12,} {rep['enc_gpu']:14,} {rep['fold_iter']:12,} "
          f"{rep['red_op']:10,}")
print(f"\nspread {min(rat):.2f}x .. {max(rat):.2f}x   (mul/add card was 0.69x .. 2.48x)")
