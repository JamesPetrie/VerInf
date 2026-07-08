"""Full Llama-2-7B transformer-layer proof, then time the CPU verifiers.

Builds demo_llama7b._run_block (one full layer, random signed weights, no LM head)
at production config, generates a real fused proof on the GPU, dumps it to
/tmp/proof.json for the Rust verifier, and times the pure-Python verify.py path
(_checks) on the same proof. The GPU core.verify time is printed for reference.

The benchmark of record is Python verify.py vs Rust verify_proof (run separately)
on /tmp/proof.json — both are the minimal-TCB verifier; core.verify (GPU) is not.

Run on the Spark:  PATH=$HOME/venv-hf/bin:$PATH ~/venv-hf/bin/python bench_layer.py
"""
import json, sys, time
import torch
import core
import claims as C          # noqa: F401
import packets as PK        # noqa: F401
import protocol as pr
import verify as vf
from collections import Counter
from demo_llama7b import (_run_block, _commit_weights_random, CFG, SEQ, d,
                          _rand_signed, HALF_X)
from tape import Tape, SILU_14BIT

SEED = b"bench-layer"


def _ints(t):
    return [int(v) for v in t.cpu().tolist()]


def _path_json(path):
    return [[sib.hex(), int(side)] for sib, side in path]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    # ---- build one full transformer layer (random weights, no LM head) ----
    tape = Tape(CFG, silu_config=SILU_14BIT)
    x_data = _rand_signed(SEQ * d, half=HALF_X)
    resid = tape.commit("x_input", x_data, (SEQ, d))
    weights = _commit_weights_random(tape, layer_idx=0)
    H = d // 128
    _run_block(tape, resid, weights, H=H)
    claims, inputs, cfg = tape.claims, tape.inputs, tape.cfg
    counts = Counter(type(c).__name__ for c in claims)
    log(f"claims: {len(claims)}  {dict(sorted(counts.items()))}")

    # ---- generate the proof on the GPU ----
    torch.cuda.synchronize(); t0 = time.time()
    proof = core.prove(claims, inputs, cfg, seed=SEED)
    torch.cuda.synchronize(); t_prove = time.time() - t0

    s_op, s_comb, s_col = pr.round_seeds(SEED)
    Q = pr.random_columns(s_col, cfg)
    opened_p1 = {j: _ints(proof.opened_p1[j]) for j in Q}
    opened_p2 = {j: _ints(proof.opened_p2[j]) for j in Q}
    m_total = len(opened_p1[Q[0]]) + len(opened_p2[Q[0]])
    log(f"m_total (committed rows): {m_total}   T_QUERIES: {cfg.T_QUERIES}")
    log(f"prove (GPU): {t_prove:.2f}s")

    # ---- dump for the Rust verifier ----
    out = {
        "claims": pr.claims_to_json(claims, cfg),
        "seeds": {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
        "proof": {
            "root_p1": proof.root_p1.hex(), "root_p2": proof.root_p2.hex(),
            "q_irs": _ints(proof.q_irs), "q_lin": _ints(proof.q_lin), "p_0": _ints(proof.p_0),
            "opened_p1": {str(j): opened_p1[j] for j in Q},
            "opened_p2": {str(j): opened_p2[j] for j in Q},
            "paths_p1": {str(j): _path_json(proof.paths_p1[j]) for j in Q},
            "paths_p2": {str(j): _path_json(proof.paths_p2[j]) for j in Q},
        },
        "python_accept": True,  # filled below
    }

    # ---- reference: GPU verifier (core.verify) ----
    t0 = time.time(); acc_gpu, msg = core.verify(claims, proof, SEED, cfg); t_gpu = time.time() - t0
    log(f"verify (GPU core.verify): {t_gpu:.2f}s  -> {'ACCEPT' if acc_gpu else 'REJECT'} ({msg})")
    out["python_accept"] = bool(acc_gpu)
    with open("/tmp/proof.json", "w") as f:
        json.dump(out, f)
    log("wrote /tmp/proof.json (for Rust verify_proof)")

    # ---- the benchmark: pure-Python minimal-TCB verifier (verify.py) ----
    ok, t_py = None, None
    if "--no-python" not in sys.argv:
        log("running pure-Python verify.py _checks (this is the slow reference)...")
        t0 = time.time()
        ok = vf._checks(claims, cfg, s_op, s_comb, s_col,
                        proof.root_p1, proof.root_p2,
                        _ints(proof.q_irs), _ints(proof.q_lin), _ints(proof.p_0),
                        opened_p1, opened_p2, proof.paths_p1, proof.paths_p2)
        t_py = time.time() - t0
        log(f"verify (pure-Python verify.py): {t_py:.2f}s  -> {'ACCEPT' if ok else 'REJECT'}")

    print(json.dumps({"m_total": m_total, "t_queries": cfg.T_QUERIES,
                      "n_claims": len(claims), "t_prove_gpu": t_prove,
                      "t_verify_gpu": t_gpu, "t_verify_python": t_py,
                      "python_accept": bool(ok)}))


if __name__ == "__main__":
    main()
