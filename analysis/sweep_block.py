#!/usr/bin/env python
"""Witness-size + prove-time sweep over sequence length for ONE Maverick block.

Reuses demo_maverick_block.build/load_attention. For each --seqs value:
  - builds the tape (claims + Variables; expert weights stay lazy)
  - tallies committed witness slots by (claim_type, phase) and m_total rows,
    walking into LogUp Table objects to capture phase-2 aux z's
  - optionally proves (timed) with --prove   (no verify; this measures cost)

Emits human lines + CSVROW lines + an optional JSON summary. Sequence length
is --seqs; the demo's `S` is the fixed quant scale (2^12), unrelated.

Lives in analysis/; adds repo/pipeline to sys.path to import the demo modules.
"""
import argparse, dataclasses, json, sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))  # analysis/ -> repo/pipeline
import torch
import _uint64_compat  # noqa: F401
from core import Variable
from tape import Tape
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "demo"))
import demo_maverick_block as B
from demo_maverick_moe import (_to_field, _rand_int, CFG, S, SILU_CFG, HALF_X)

SEED = b"maverick-block-sweep"


def _collect(obj, sink, seen_obj, depth=0):
    """Recursively gather distinct Variables reachable from a claim, including
    those held inside LogUp Table objects (mult_var/w_var/z_vars)."""
    if obj is None or depth > 4:
        return
    oid = id(obj)
    if isinstance(obj, Variable):
        sink.append(obj)
        return
    if oid in seen_obj:
        return
    if isinstance(obj, (list, tuple)):
        seen_obj.add(oid)
        for o in obj:
            _collect(o, sink, seen_obj, depth + 1)
    elif isinstance(obj, dict):
        seen_obj.add(oid)
        for o in obj.values():
            _collect(o, sink, seen_obj, depth + 1)
    elif obj.__class__.__name__ == "Table":
        seen_obj.add(oid)
        for attr in ("mult_var", "w_var", "z_vars"):
            _collect(getattr(obj, attr, None), sink, seen_obj, depth + 1)
    elif hasattr(obj, "__dataclass_fields__"):
        seen_obj.add(oid)
        for f in dataclasses.fields(obj):
            _collect(getattr(obj, f.name, None), sink, seen_obj, depth + 1)


def witness_breakdown(tape, ell):
    seen = {}                       # id(var) -> (var, claim_type)
    for c in tape.claims:
        ctype = type(c).__name__
        sink, seen_obj = [], set()
        for f in dataclasses.fields(c):
            _collect(getattr(c, f.name, None), sink, seen_obj)
        for v in sink:
            if id(v) not in seen:
                seen[id(v)] = (v, ctype)
    for v in tape.inputs:           # committed inputs not reached above
        if isinstance(v, Variable) and id(v) not in seen:
            seen[id(v)] = (v, "committed_other")
    by_type, by_phase = {}, {}
    total_slots = m_total = 0
    for v, ctype in seen.values():
        rows = (v.length + ell - 1) // ell
        t = by_type.setdefault(ctype, [0, 0, 0])
        t[0] += v.length; t[1] += 1; t[2] += rows
        by_phase[v.phase] = by_phase.get(v.phase, 0) + v.length
        total_slots += v.length; m_total += rows
    return by_type, by_phase, total_slots, m_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=str, default="512,1024,2048,4096,8192")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--d-ff", type=int, default=8192)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--from-gguf", type=str, required=True)
    ap.add_argument("--prove", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    seqs = [int(x) for x in a.seqs.split(",")]
    theta = 500000.0
    use_rope = (a.layer + 1) % 4 != 0
    ell = CFG.ELL
    print(f"[sweep] E={a.experts} d={a.d} d_ff={a.d_ff} layer={a.layer} rope={use_rope} "
          f"ELL={ell} N_LIG={CFG.N_LIG} T_QUERIES={CFG.T_QUERIES} scale={S} prove={a.prove}",
          flush=True)

    from loader import load_maverick_moe_layer
    print("[sweep] loading layer weights (once, reused across seqs)...", flush=True)
    t0 = time.time()
    real = load_maverick_moe_layer(a.from_gguf, a.layer, S=S,
                                   n_experts=a.experts, skip_experts=True)
    real["_gguf"], real["_layer"] = a.from_gguf, a.layer
    attn_w = B.load_attention(a.from_gguf, a.layer)
    print(f"[sweep] weights loaded ({time.time()-t0:.1f}s)", flush=True)

    results = []
    for seq in seqs:
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(7)
        x_data = _to_field(_rand_int(seq * a.d, half=HALF_X))
        tb = time.time()
        tape = Tape(CFG, silu_config=SILU_CFG, lazy=True)
        B.build(tape, T=seq, E=a.experts, d=a.d, d_ff=a.d_ff, real=real,
                attn_w=attn_w, theta=theta, use_rope=use_rope, x_data=x_data)
        t_build = time.time() - tb
        by_type, by_phase, total_slots, m_total = witness_breakdown(tape, ell)
        t_prove = peak = None
        if a.prove:
            tp = time.time()
            proof = tape.prove(seed=SEED)
            t_prove = round(time.time() - tp, 1)
            peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            del proof
        rec = dict(seq=seq, experts=a.experts, n_claims=len(tape.claims),
                   total_slots=total_slots, m_total=m_total,
                   phase1=by_phase.get(1, 0), phase2=by_phase.get(2, 0),
                   t_build=round(t_build, 2), t_prove=t_prove, peakGPU_GB=peak,
                   by_type={k: v[0] for k, v in by_type.items()})
        results.append(rec)
        print(f"[sweep] seq={seq} claims={rec['n_claims']} slots={total_slots:,} "
              f"m_total={m_total:,} p1={rec['phase1']:,} p2={rec['phase2']:,} "
              f"build={t_build:.1f}s prove={t_prove}s peak={peak}GB", flush=True)
        for ct, (sl, nv, rw) in sorted(by_type.items(), key=lambda kv: -kv[1][0])[:7]:
            print(f"           {ct:30s} slots={sl:>15,} vars={nv:>6} rows={rw:,}", flush=True)
        del tape
        torch.cuda.empty_cache()

    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[sweep] wrote {a.out}", flush=True)
    print("CSV seq,experts,total_slots,m_total,phase1,phase2,t_build,t_prove,peakGPU_GB")
    for r in results:
        print(f"CSVROW {r['seq']},{r['experts']},{r['total_slots']},{r['m_total']},"
              f"{r['phase1']},{r['phase2']},{r['t_build']},{r['t_prove']},{r['peakGPU_GB']}")


if __name__ == "__main__":
    raise SystemExit(main())
