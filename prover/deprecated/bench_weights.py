"""Layer-scale matmul benchmark: the 7 weight matmuls of a Llama-2-7B layer
(real shapes, production Ligero config), proved and verified.

Matmul uses Freivalds with NO LogUp table, so this captures the DOMINANT verifier
cost of a real layer — the ~24.7k committed rows from the 7 weight matrices, which
are 24.7k of a full layer's 33.3k rows — without the pathological 2^24 rescale
range tables that make the full-demo proof.json 2.7 GB. Tractable for both the
pure-Python verifier (verify.py) and the Rust verifier (run separately).

Shapes (one Llama-2-7B layer, SEQ=2): W_Q/K/V/O = d×d (4096²); W_gate/up = d×d_ff
(4096×11008); W_down = d_ff×d. Independent matmuls (A_i·B_i=C_i); the verifier
cost depends only on the committed sizes, so chaining is unnecessary.

Run on the Spark:  PATH=$HOME/venv-hf/bin:$PATH ~/venv-hf/bin/python bench_weights.py
"""
import json, sys, time
import torch
import core
import claims as C          # noqa: F401
import packets as PK        # noqa: F401
import protocol as pr
import verify as vf
from cuda_primitives import gl_matmul

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=80)
SEED = b"bench-weights"
d, d_ff, SEQ = 4096, 11008, 2

# Full Llama-2-7B layer weight shapes (m_total ~24.7k). Used when argv = "layer".
LAYER_SHAPES = [("W_Q", SEQ, d, d), ("W_K", SEQ, d, d), ("W_V", SEQ, d, d),
                ("W_O", SEQ, d, d), ("W_gate", SEQ, d, d_ff), ("W_up", SEQ, d, d_ff),
                ("W_down", SEQ, d_ff, d)]


def shapes_for(arg):
    """argv: 'layer' for the 7 real layer-weight matmuls, or 'D N' for N square
    SEQ×D @ D×D matmuls (a smaller scale where pure-Python verify finishes)."""
    if arg and arg[0] == "layer":
        return LAYER_SHAPES, "layer"
    D = int(arg[0]) if arg else 1024
    N = int(arg[1]) if len(arg) > 1 else 4
    return [(f"mm{i}", SEQ, D, D) for i in range(N)], f"D={D},N={N}"


def _ints(t):
    return [int(v) for v in t.cpu().tolist()]


def _path_json(path):
    return [[sib.hex(), int(side)] for sib, side in path]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def build(shapes):
    claims, inputs = [], {}
    for name, m, k, n in shapes:
        A = core.Variable(f"{name}_A", length=m * k)
        B = core.Variable(f"{name}_B", length=k * n)
        Cv = core.Variable(f"{name}_C", length=m * n)
        claims.append(C.matmul_claim(name, A, B, Cv, m=m, k=k, n=n))
        A_t = torch.randint(0, 1 << 10, (m, k), dtype=torch.int64, device="cuda").to(torch.uint64)
        B_t = torch.randint(0, 1 << 10, (k, n), dtype=torch.int64, device="cuda").to(torch.uint64)
        inputs[A] = A_t.view(-1); inputs[B] = B_t.view(-1)
        inputs[Cv] = gl_matmul(A_t, B_t).view(-1)
    return claims, inputs


def main():
    # argv: [shape-args...] [--no-python]. shape-args = "layer" | "<D> <N>".
    argv = sys.argv[1:]
    do_python = "--no-python" not in argv
    argv = [a for a in argv if a != "--no-python"]
    shapes, tag = shapes_for(argv)
    claims, inputs = build(shapes)
    log(f"scale: {tag}   claims: {len(claims)} matmuls")
    torch.cuda.synchronize(); t0 = time.time()
    proof = core.prove(claims, inputs, CFG, seed=SEED)
    torch.cuda.synchronize(); t_prove = time.time() - t0

    s_op, s_comb, s_col = pr.round_seeds(SEED)
    Q = pr.random_columns(s_col, CFG)
    opened_p1 = {j: _ints(proof.opened_p1[j]) for j in Q}
    opened_p2 = {j: _ints(proof.opened_p2[j]) for j in Q}
    m_total = len(opened_p1[Q[0]]) + len(opened_p2[Q[0]])
    log(f"m_total: {m_total}   T_QUERIES: {CFG.T_QUERIES}   prove(GPU): {t_prove:.2f}s")

    out = {
        "claims": pr.claims_to_json(claims, CFG),
        "seeds": {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
        "proof": {
            "root_p1": proof.root_p1.hex(), "root_p2": proof.root_p2.hex(),
            "q_irs": _ints(proof.q_irs), "q_lin": _ints(proof.q_lin), "p_0": _ints(proof.p_0),
            "opened_p1": {str(j): opened_p1[j] for j in Q},
            "opened_p2": {str(j): opened_p2[j] for j in Q},
            "paths_p1": {str(j): _path_json(proof.paths_p1[j]) for j in Q},
            "paths_p2": {str(j): _path_json(proof.paths_p2[j]) for j in Q},
        },
        "python_accept": True,
    }

    t0 = time.time(); acc_gpu, msg = core.verify(claims, proof, SEED, CFG); t_gpu = time.time() - t0
    log(f"verify(GPU core.verify): {t_gpu:.2f}s -> {'ACCEPT' if acc_gpu else 'REJECT'} ({msg})")
    out["python_accept"] = bool(acc_gpu)
    with open("/tmp/proof.json", "w") as f:
        json.dump(out, f)
    log(f"wrote /tmp/proof.json")

    t_py = None
    if do_python:
        log("running pure-Python verify.py _checks (slow reference)...")
        t0 = time.time()
        ok = vf._checks(claims, CFG, s_op, s_comb, s_col,
                        proof.root_p1, proof.root_p2,
                        _ints(proof.q_irs), _ints(proof.q_lin), _ints(proof.p_0),
                        opened_p1, opened_p2, proof.paths_p1, proof.paths_p2)
        t_py = time.time() - t0
        log(f"verify(pure-Python verify.py): {t_py:.2f}s -> {'ACCEPT' if ok else 'REJECT'}")
    print(json.dumps({"scale": tag, "m_total": m_total, "t_queries": CFG.T_QUERIES,
                      "n_matmuls": len(claims), "t_prove_gpu": t_prove,
                      "t_verify_gpu": t_gpu, "t_verify_python": t_py}))


if __name__ == "__main__":
    main()
