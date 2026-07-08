"""Multi-layer Llama-2-7B proof at T_QUERIES=4, for full-model timing/feasibility.

Builds N transformer blocks on ONE tape (residual chained, like demo_llama7b's
single-tape path), proves, dumps for the Rust verifier. T_QUERIES=4 is for TIMING
and feasibility only — it is NOT cryptographically sound (production T=80); it
makes verify ~20x cheaper and the proof ~20x smaller, but does NOT change the
prover's committed matrix (m_total x N_LIG), so this run also probes the prover's
memory ceiling on this box.

Usage:  python bench_full_model.py <N_layers>
"""
import json, sys, time
import torch
import core
import claims as C          # noqa: F401
import packets as PK        # noqa: F401
import protocol as pr
from demo_llama7b import _run_block, _commit_weights_random, SEQ, d, _rand_signed, HALF_X
from tape import Tape, SILU_14BIT
from core import LigeroConfig

CFG = LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=4)   # 4 columns


def _ints(t):
    return [int(v) for v in t.cpu().tolist()]


def _path_json(p):
    return [[s.hex(), int(sd)] for s, sd in p]


def log(m):
    print(m, file=sys.stderr, flush=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tape = Tape(CFG, silu_config=SILU_14BIT)
    x = _rand_signed(SEQ * d, half=HALF_X)
    resid = tape.commit("x_input", x, (SEQ, d))
    H = d // 128
    for layer in range(n):
        w = _commit_weights_random(tape, layer_idx=layer)
        resid = _run_block(tape, resid, w, H=H)
        del w
        torch.cuda.empty_cache()
    claims = tape.claims
    log(f"N={n} layers  claims={len(claims)}  T_QUERIES={CFG.T_QUERIES}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    proof = core.prove(claims, tape.inputs, CFG, seed=b"full")
    torch.cuda.synchronize(); t_prove = time.time() - t0
    cuda_peak = torch.cuda.max_memory_allocated() / 1e9
    log(f"cuda_peak={cuda_peak:.1f}GB (unified pool)")

    s_op, s_comb, s_col = pr.round_seeds(b"full")
    Q = pr.random_columns(s_col, CFG)
    o1 = {j: _ints(proof.opened_p1[j]) for j in Q}
    o2 = {j: _ints(proof.opened_p2[j]) for j in Q}
    m_total = len(o1[Q[0]]) + len(o2[Q[0]])
    log(f"m_total={m_total}  prove(GPU)={t_prove:.1f}s")

    out = {
        "claims": pr.claims_to_json(claims, CFG),
        "seeds": {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
        "proof": {
            "root_p1": proof.root_p1.hex(), "root_p2": proof.root_p2.hex(),
            "q_irs": _ints(proof.q_irs), "q_lin": _ints(proof.q_lin), "p_0": _ints(proof.p_0),
            "opened_p1": {str(j): o1[j] for j in Q}, "opened_p2": {str(j): o2[j] for j in Q},
            "paths_p1": {str(j): _path_json(proof.paths_p1[j]) for j in Q},
            "paths_p2": {str(j): _path_json(proof.paths_p2[j]) for j in Q},
        },
        "python_accept": True,
    }
    t0 = time.time(); acc, msg = core.verify(claims, proof, b"full", CFG); t_gpu = time.time() - t0
    out["python_accept"] = bool(acc)
    log(f"verify(GPU)={t_gpu:.1f}s -> {'ACCEPT' if acc else 'REJECT'} ({msg})")
    with open("/tmp/proof.json", "w") as f:
        json.dump(out, f)
    log("wrote /tmp/proof.json")
    print(json.dumps({"N": n, "m_total": m_total, "t_prove": t_prove, "t_gpu_verify": t_gpu}))


if __name__ == "__main__":
    main()
