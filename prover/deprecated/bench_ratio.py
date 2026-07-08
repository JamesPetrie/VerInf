"""Rust-vs-Python verifier ratio at a tractable config.

linear_column_test is O(m_total · ELL · T) regardless of op type, so at the
production config (ELL=8192, T=80) pure-Python verify is hours even for one
weight matrix — too slow to time. The Rust/Python ratio, though, is a per-field-op
speedup (compiled+20-core vs interpreted+1-core) and is config-independent. So we
measure the ratio at a SMALL config where Python finishes in seconds, across a
few m_total points (to confirm linear scaling), then extrapolate to layer scale.

Each scale: prove (GPU), dump /tmp/proof_<i>.json, time pure-Python verify.py.
Run the Rust verifier on each dump separately. Matmul-only (no LogUp tables) —
the dominant O(m_total·ELL·T) cost, same shape as a layer's weight matmuls.

Run on the Spark:  ~/venv-hf/bin/python bench_ratio.py
"""
import json, sys, time
import torch
import core
import claims as C          # noqa: F401
import packets as PK        # noqa: F401
import protocol as pr
import verify as vf
from cuda_primitives import gl_matmul

# Small config: rows are 256 wide, 16 queries — Python finishes in seconds.
CFG = core.LigeroConfig(ELL=256, K_DEG=256, N_LIG=1024, T_QUERIES=16)
SEED = b"bench-ratio"
DIM = 512                      # 512x512 matmul -> 512x512/256 = 1024 rows of B each
SCALES = [1, 2, 4]             # number of matmuls -> m_total ~1k, 2k, 4k


def _ints(t):
    return [int(v) for v in t.cpu().tolist()]


def _path_json(path):
    return [[sib.hex(), int(side)] for sib, side in path]


def log(m):
    print(m, file=sys.stderr, flush=True)


def build(n):
    claims, inputs = [], {}
    for i in range(n):
        A = core.Variable(f"A{i}", length=2 * DIM)
        B = core.Variable(f"B{i}", length=DIM * DIM)
        Cv = core.Variable(f"C{i}", length=2 * DIM)
        claims.append(C.matmul_claim(f"mm{i}", A, B, Cv, m=2, k=DIM, n=DIM))
        A_t = torch.randint(0, 1 << 10, (2, DIM), dtype=torch.int64, device="cuda").to(torch.uint64)
        B_t = torch.randint(0, 1 << 10, (DIM, DIM), dtype=torch.int64, device="cuda").to(torch.uint64)
        inputs[A] = A_t.view(-1); inputs[B] = B_t.view(-1); inputs[Cv] = gl_matmul(A_t, B_t).view(-1)
    return claims, inputs


def run(n, idx):
    claims, inputs = build(n)
    proof = core.prove(claims, inputs, CFG, seed=SEED)
    s_op, s_comb, s_col = pr.round_seeds(SEED)
    Q = pr.random_columns(s_col, CFG)
    o1 = {j: _ints(proof.opened_p1[j]) for j in Q}
    o2 = {j: _ints(proof.opened_p2[j]) for j in Q}
    m_total = len(o1[Q[0]]) + len(o2[Q[0]])
    out = {"claims": pr.claims_to_json(claims, CFG),
           "seeds": {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
           "proof": {"root_p1": proof.root_p1.hex(), "root_p2": proof.root_p2.hex(),
                     "q_irs": _ints(proof.q_irs), "q_lin": _ints(proof.q_lin), "p_0": _ints(proof.p_0),
                     "opened_p1": {str(j): o1[j] for j in Q}, "opened_p2": {str(j): o2[j] for j in Q},
                     "paths_p1": {str(j): _path_json(proof.paths_p1[j]) for j in Q},
                     "paths_p2": {str(j): _path_json(proof.paths_p2[j]) for j in Q}},
           "python_accept": True}
    with open(f"/tmp/proof_{idx}.json", "w") as f:
        json.dump(out, f)
    t0 = time.time()
    ok = vf._checks(claims, CFG, s_op, s_comb, s_col, proof.root_p1, proof.root_p2,
                    _ints(proof.q_irs), _ints(proof.q_lin), _ints(proof.p_0),
                    o1, o2, proof.paths_p1, proof.paths_p2)
    t_py = time.time() - t0
    log(f"scale {idx}: n_matmuls={n} m_total={m_total} python_verify={t_py:.2f}s accept={ok}")
    return {"idx": idx, "n": n, "m_total": m_total, "t_py": t_py, "accept": bool(ok)}


def main():
    results = [run(n, i) for i, n in enumerate(SCALES)]
    print(json.dumps(results))


if __name__ == "__main__":
    main()
