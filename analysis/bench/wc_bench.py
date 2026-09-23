"""WC-LCRL-STC S1+S2 benchmark: enrollment + bridge at PRODUCTION geometry
(B=15360, lam=1024, K_w=16384, N_w=32768, q_w=40) on a configurable slice of
blocks per width group.

Measures, cuda-synced:
  - coefficient packing (weights -> (n_polys, K_w) rows)
  - coefficient-RS (zero-pad + batched forward NTT to N_w)
  - column-major Merkle (sha256, CPU — the production build would use the
    GPU BLAKE3 accumulator; reported separately so the field-work numbers
    are not polluted)
  - bridge prove (rho, P_trace, pi, alpha-aggregate, v = c(eta))
  - bridge verify (CPU python-int mirror)

Throughput is reported per weight parameter.  The extrapolation lines are
MODELED numbers (linear in parameter count), not measurements; they are
labeled as such.

Usage:  python analysis/bench/wc_bench.py [--widths 4096,11008]
        [--blocks 2] [--toy]  (--toy: shrunk geometry for the dev box)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "prover"))

import torch

import wc_bridge as wc
from cuda_primitives import P


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="4096,11008")
    ap.add_argument("--blocks", type=int, default=2,
                    help="blocks per width group (B input coords each)")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    params = (wc.WcParams(B=48, lam=16, N_w=128, q_w=8) if args.toy
              else wc.WcParams())
    widths = [int(w) for w in args.widths.split(",")]

    torch.manual_seed(0)
    groups = {}
    for n in widths:
        rows = args.blocks * params.B
        groups[n] = (torch.randint(0, 1 << 62, (rows, n), dtype=torch.int64)
                     .to(torch.uint64)).cuda()
    n_params = sum(g.numel() for g in groups.values())
    n_polys = sum(args.blocks * n for n in widths)
    cw_bytes = n_polys * params.N_w * 8
    print(f"geometry: B={params.B} lam={params.lam} K_w={params.K_w} "
          f"N_w={params.N_w} q_w={params.q_w}")
    print(f"slice: widths={widths} blocks/width={args.blocks} "
          f"weights={n_params:,} polys={n_polys:,} "
          f"codeword store={cw_bytes/1e9:.2f} GB")

    sync(); t0 = time.time()
    enr = wc.build_enrollment(groups, b"bench-mask", b"bench-manifest", params)
    sync(); t_enroll = time.time() - t0

    sync(); t0 = time.time()
    proof = wc.prove_bridge(enr, b"\x22" * 32)
    sync(); t_prove = time.time() - t0

    meta = {n: (enr.groups[n].n_blocks, n) for n in enr.groups}
    t0 = time.time()
    ok, why = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                               proof, b"\x22" * 32, params,
                               trusted_identity=enr.identity(), layout=enr.layout)
    t_verify = time.time() - t0
    assert ok, why

    per_param_enroll_ns = t_enroll / n_params * 1e9
    per_param_prove_ns = t_prove / n_params * 1e9
    print(f"enrollment  {t_enroll:8.2f} s   ({per_param_enroll_ns:7.2f} ns/param, one-time)")
    print(f"bridge prove{t_prove:8.2f} s   ({per_param_prove_ns:7.2f} ns/param, per proof)")
    print(f"bridge verify{t_verify:7.2f} s   (CPU python mirror; Rust twin will be ~100x faster)")
    print(f"ACCEPT: {ok}")
    # MODELED extrapolations — linear in parameter count, NOT measured:
    for label, total in (("7B", 6.7e9), ("400B", 4.0e11)):
        print(f"MODELED {label}: enrollment {total*per_param_enroll_ns/1e9:,.0f} s "
              f"(one-time), bridge prove {total*per_param_prove_ns/1e9:,.0f} s/proof")
    if args.json:
        json.dump({"params": vars(args), "n_params": n_params,
                   "t_enroll": t_enroll, "t_prove": t_prove,
                   "t_verify": t_verify,
                   "ns_per_param_enroll": per_param_enroll_ns,
                   "ns_per_param_prove": per_param_prove_ns},
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
