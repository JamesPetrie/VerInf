"""Run VARIANTS of a proof under different field policies and report, per policy:
  - which field each op lands in (per-op),
  - whether the whole proof can use BabyBear (per-proof), and WHY not if not,
  - a first-order payoff projection (BabyBear stores 4B words vs Goldilocks 8B
    and has cheaper 31-bit muls, so its share of the byte-bound work shrinks).

Payoff model (FIRST-ORDER ESTIMATE, clearly not a measured prove — the BabyBear
backend isn't ported; see field_policy.py scope note):
  witness + streaming (encode/hash/merkle/spill) scale ~linearly with witness
  BYTES. BabyBear halves the bytes of its share and lowers mul cost; we credit
  BabyBear-resident work a factor BB_WORK = 0.55x (0.5 bytes, slight mul win).
  Ops staying in Goldilocks are unchanged. quad/lin scale similarly with bytes.

Usage:
  python3 run_field_variants.py --model 400b --policy adaptive-op
  python3 run_field_variants.py --model toy  --policy babybear
  python3 run_field_variants.py --model 400b --all          # compare all policies
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import field_policy as fp
from field_policy import Op, OpKind, Policy, BABYBEAR, GOLDILOCKS
import prod_lens

BB_WORK = 0.55   # first-order cost multiplier for BabyBear-resident byte-bound work


def _layer_ops(prefix: str, d: int, d_ff: int, d_h: int, seq: int, H: int,
               s: int, R: float, w_l1: float) -> list[Op]:
    """The ops of one transformer block, with contraction length k and the
    output element count carried as `.k`-independent weight via witness_elems()."""
    return [
        Op(f"{prefix}.qkv",   OpKind.WEIGHT_MATMUL, k=d,    s_a=s, s_b=s, R=R, w_l1=w_l1),
        Op(f"{prefix}.score", OpKind.ATTENTION_SCORE, k=d_h, s_a=s, s_b=s, R=R),
        # P·V: the softmax operand is a probability (<=1), only V carries R.
        Op(f"{prefix}.av",    OpKind.ATTENTION_SCORE, k=seq, s_a=s, s_b=s, R=1.0, R2=R),
        Op(f"{prefix}.oproj", OpKind.WEIGHT_MATMUL, k=d,    s_a=s, s_b=s, R=R, w_l1=w_l1),
        Op(f"{prefix}.mlp_up",   OpKind.WEIGHT_MATMUL, k=d,    s_a=s, s_b=s, R=R, w_l1=w_l1),
        Op(f"{prefix}.silu",  OpKind.ELEMENTWISE, k=1, s_a=s, R=R),
        Op(f"{prefix}.mlp_dn",   OpKind.WEIGHT_MATMUL, k=d_ff, s_a=s, s_b=s, R=R, w_l1=w_l1),
        Op(f"{prefix}.rms",   OpKind.ELEMENTWISE, k=1, s_a=s, R=R),
    ]


# output element count per op (drives witness bytes). Kept alongside the Op list.
def _op_out_elems(o: Op, d, d_ff, d_h, seq, H) -> int:
    tail = o.name.split(".")[-1]
    return {
        "qkv": seq * 3 * d, "score": H * seq * seq, "av": seq * d,
        "oproj": seq * d, "mlp_up": seq * d_ff, "silu": seq * d_ff,
        "mlp_dn": seq * d, "rms": seq * d,
    }[tail]


def build_model(which: str, s: int, R: float, w_l1: float):
    if which == "toy":
        d, d_ff, d_h, seq, H, L = 512, 1536, 64, 512, 8, 4
    elif which == "400b":
        d, d_ff, d_h, seq, H, L = 16384, 53248, 128, 8192, 128, 126
    else:
        raise SystemExit(f"unknown model {which}")
    ops: list[Op] = []
    for li in range(L):
        ops += _layer_ops(f"L{li}", d, d_ff, d_h, seq, H, s, R, w_l1)
    dims = dict(d=d, d_ff=d_ff, d_h=d_h, seq=seq, H=H, L=L)
    return ops, dims


def _witness_bytes(ops, dims, assign_map) -> dict:
    """Total witness bytes and the fraction that lands in BabyBear (4B) under an
    assignment. Bytes per op = out_elems * field.word_bytes."""
    tot = bb = 0
    for o in ops:
        e = _op_out_elems(o, **{k: dims[k] for k in ("d", "d_ff", "d_h", "seq", "H")})
        f = assign_map[o.name]
        tot += e * f.word_bytes
        if f is BABYBEAR:
            bb += e * f.word_bytes
    # reference: everything Goldilocks (8B)
    gold = sum(_op_out_elems(o, **{k: dims[k] for k in ("d","d_ff","d_h","seq","H")}) * 8
               for o in ops)
    return dict(assigned_bytes=tot, bb_bytes=bb, gold_all_bytes=gold)


def report(which: str, policy: Policy, s: int, R: float, w_l1: float):
    ops, dims = build_model(which, s, R, w_l1)
    amap = fp.assign(ops, policy)
    # per-op field tally (dedup by op TYPE, they repeat per layer)
    by_kind = {}
    for o in ops:
        b = o.acc_bits(); f = amap[o.name]
        key = o.name.split(".")[-1]
        by_kind.setdefault(key, (o.kind, b, f))
    print(f"\n=== model={which}  policy={policy.value}  (s=2^{int(s).bit_length()-1}, R={R}, ||W||1={w_l1}) ===")
    print(f"{'op':10} {'kind':17} {'acc_bits':>8} {'-> field':>12} {'fits BB?':>9}")
    for key, (kind, b, f) in by_kind.items():
        fits = "yes" if b < BABYBEAR.bits else "NO (overflow)"
        print(f"{key:10} {kind.value:17} {b:8.1f} {f.name:>12} {fits:>13}")

    of = fp.overflow_ops(ops, BABYBEAR)
    ofkinds = sorted({o.name.split('.')[-1] for o in of})
    print(f"\n  BabyBear ceiling = {BABYBEAR.bits} bits.  Ops that OVERFLOW BabyBear: "
          f"{ofkinds if ofkinds else 'none'}")

    wb = _witness_bytes(ops, dims, amap)
    bb_frac = wb["bb_bytes"] / wb["assigned_bytes"] if wb["assigned_bytes"] else 0
    # first-order payoff: BabyBear-resident byte-bound work * BB_WORK, rest 1x.
    # fraction of witness bytes in BB (by the Goldilocks-referenced size, since
    # savings are relative to the all-Goldilocks baseline).
    bb_byte_share_vs_gold = wb["bb_bytes"] / (2 if True else 1)  # bb stored at 4B; its gold size is 2x
    bb_gold_equiv = 2 * wb["bb_bytes"]                            # what those elems cost in Goldilocks
    share_of_witness = bb_gold_equiv / wb["gold_all_bytes"] if wb["gold_all_bytes"] else 0
    # prove-time terms that scale with witness bytes: witness(56.8) + streaming(32.3)
    # + quad(10.4) ~ 99.5% of prove. Credit those, in the BB share, factor BB_WORK.
    byte_bound_share = 0.995
    speedup = share_of_witness * byte_bound_share * (1 - BB_WORK)
    print(f"\n  witness BabyBear share (Goldilocks-equiv bytes): {100*share_of_witness:.1f}%")
    print(f"  FIRST-ORDER payoff (BB work x{BB_WORK}, est. NOT measured): "
          f"{100*speedup:+.1f}% faster prove at 400B")
    prod_lens.report(f"field policy {policy.value} [{which}]", 0.0, transfers=True,
                     effect=prod_lens.LeverEffect(term="witness",
                        toy_frac=share_of_witness * (1 - BB_WORK),
                        note=f"(field payoff est.; BabyBear share {100*share_of_witness:.0f}%, "
                             f"crosses streaming+quad too — this projects the witness slice only)"))


def sweep_s(which: str, R: float, w_l1: float):
    """Feasibility map: for each fixed-point scale s=2^b, which op kinds fit
    BabyBear (acc_bits < 30). Shows the tension: small s fits weight matmuls but
    costs precision; attention (score/av) is the binding constraint."""
    ops, dims = build_model(which, 4096, R, w_l1)
    kinds = {}
    for o in ops:
        kinds.setdefault(o.name.split(".")[-1], o)
    print(f"\n=== BabyBear feasibility vs scale s  (model={which}, R={R}, ||W||1={w_l1}) ===")
    print(f"  BabyBear ceiling = {BABYBEAR.bits} bits;  '.' = fits BabyBear, 'X' = overflow\n")
    hdr = "  s     " + "".join(f"{k:>9}" for k in kinds)
    print(hdr)
    for b in (12, 10, 8, 6, 4, 2):
        s = 1 << b
        cells = []
        for k, o in kinds.items():
            o2 = fp.Op(o.name, o.kind, o.k, s_a=s, s_b=s, R=o.R, R2=o.R2, w_l1=o.w_l1)
            bits = o2.acc_bits()
            mark = "." if bits < BABYBEAR.bits else "X"
            cells.append(f"{bits:6.1f}{mark} ")
        print(f"  2^{b:<3} " + "".join(f"{c:>9}" for c in cells))
    print("\n  precision note: each -1 in the exponent of s halves fixed-point")
    print("  resolution — small s that fits BabyBear may be too coarse to prove")
    print("  the model accurately. That accuracy floor is the real limit, not the field.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="400b", choices=["toy", "400b"])
    ap.add_argument("--policy", default="adaptive-op",
                    choices=[p.value for p in Policy])
    ap.add_argument("--all", action="store_true", help="run every policy")
    ap.add_argument("--sweep-s", action="store_true", help="BabyBear feasibility map vs scale")
    ap.add_argument("--s", type=int, default=4096, help="fixed-point scale (default 2^12)")
    ap.add_argument("--R", type=float, default=32.0, help="activation magnitude clip")
    ap.add_argument("--w-l1", type=float, default=20.0, help="max committed weight column L1 norm")
    args = ap.parse_args()
    if args.sweep_s:
        sweep_s(args.model, args.R, args.w_l1)
        return
    pols = list(Policy) if args.all else [Policy(args.policy)]
    for p in pols:
        report(args.model, p, args.s, args.R, args.w_l1)


if __name__ == "__main__":
    main()
