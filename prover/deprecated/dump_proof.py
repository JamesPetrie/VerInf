"""Dump a real core.py proof + its claims + seeds to JSON for the Rust verifier
differential test. Public data only (no witness).

Usage:  python dump_proof.py [matmul|rmsnorm|silu]   (default matmul)

matmul exercises the Freivalds path (no LogUp table); rmsnorm/silu exercise the
table-settlement path end-to-end (the code the all-ops compile-parity test
covers statically, here driven through a real proof + the 6 checks)."""
import json, sys
import torch
import core
import claims as C          # noqa: F401 — registers COMPILE/SAMPLE/AUX
import packets as PK        # noqa: F401 — registers EXPANDERS
import protocol as pr
from cuda_primitives import gl_matmul

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


def _ints(t):
    return [int(v) for v in t.cpu().tolist()]


def _path_json(path):
    # path = [(sibling_bytes, side), ...] → [[hex, side], ...]
    return [[sib.hex(), int(side)] for sib, side in path]


def build_matmul():
    m, k, n = 2, 4, 2
    A = core.Variable("A", length=m * k)
    B = core.Variable("B", length=k * n)
    Cv = core.Variable("C", length=m * n)
    claim = C.matmul_claim("mm", A, B, Cv, m=m, k=k, n=n)
    A_t = torch.randint(0, 1 << 20, (m, k), dtype=torch.int64, device="cuda").to(torch.uint64)
    B_t = torch.randint(0, 1 << 20, (k, n), dtype=torch.int64, device="cuda").to(torch.uint64)
    inputs = {A: A_t.view(-1), B: B_t.view(-1), Cv: gl_matmul(A_t, B_t).view(-1)}
    return [claim], inputs, CFG


def build_rmsnorm():
    # Same shape as the compile-parity rmsnorm case (B=2, d=4, L=8); the tape
    # computes a valid witness so core.prove yields a real, verifiable proof.
    from tape import Tape
    t = Tape(CFG)
    xr = torch.tensor([3, 1, 4, 2, 5, 2, 6, 1], dtype=torch.int64, device="cuda").to(torch.uint64)
    xv = t.commit("rms_x", xr, (8,))
    t.rmsnorm(xv, d=4, s=4, eps_int=1, slack_n_chunks=1)
    return t.claims, t.inputs, t.cfg


def build_silu():
    from tape import Tape, SILU_TOY
    t = Tape(CFG, silu_config=SILU_TOY)
    xs = torch.tensor([0, 1, 2, 3, 5, 7, 6, 4], dtype=torch.int64, device="cuda").to(torch.uint64)
    x = t.commit("silu_x", xs, (8,))
    t.silu(x)
    return t.claims, t.inputs, t.cfg


def build_softmax():
    from tape import Tape
    t = Tape(CFG)
    smx = torch.tensor([0, 0, 0, 0], dtype=torch.int64, device="cuda").to(torch.uint64)
    xv = t.commit("sm_x", smx, (4,))
    t.softmax(xv, M=2, s_x=8, s_c=8, s_y=8, Z_max=8, causal=True, heads=1, saturate=True)
    return t.claims, t.inputs, t.cfg


def build_hadamard():
    from tape import Tape
    t = Tape(CFG)
    a = torch.randint(0, 16, (6,), dtype=torch.int64, device="cuda").to(torch.uint64)
    b = torch.randint(0, 16, (6,), dtype=torch.int64, device="cuda").to(torch.uint64)
    av = t.commit("ha", a, (6,)); bv = t.commit("hb", b, (6,))
    t.hadamard(av, bv)
    return t.claims, t.inputs, t.cfg


def _from_test_rescale(fn_name):
    # The rescale gadget (_emit_rescale) — used by every value-producing op in the
    # real layer (output_width=24), but never e2e-tested. test_rescale.py has
    # known-valid small builders.
    import test_rescale as TR
    tape, _ = getattr(TR, fn_name)()
    return tape.claims, tape.inputs, tape.cfg


BUILDERS = {"matmul": build_matmul, "rmsnorm": build_rmsnorm, "silu": build_silu,
            "softmax": build_softmax, "hadamard": build_hadamard,
            "mm_rescale":       lambda: _from_test_rescale("build_matmul"),
            "hadamard_rescale": lambda: _from_test_rescale("build_hadamard"),
            "softmax_rescale":  lambda: _from_test_rescale("build_softmax"),
            "rms_rescale":      lambda: _from_test_rescale("build_rmsnorm")}


def main():
    op = sys.argv[1] if len(sys.argv) > 1 else "matmul"
    claims, inputs, cfg = BUILDERS[op]()
    seed = b"rust-difftest-seed"
    # Real fused proof.
    proof = core.prove(claims, inputs, cfg, seed=seed)
    # Python verify (the reference verdict).
    acc, msg = core.verify(claims, proof, seed, cfg)
    print(f"[{op}] python core.verify: {'ACCEPT' if acc else 'REJECT'} ({msg})", file=sys.stderr)

    s_op, s_comb, s_col = pr.round_seeds(seed)
    Q = pr.random_columns(s_col, cfg)

    out = {
        "claims": pr.claims_to_json(claims, cfg),
        "seeds": {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
        "proof": {
            "root_p1": proof.root_p1.hex(),
            "root_p2": proof.root_p2.hex(),
            "q_irs": _ints(proof.q_irs),
            "q_lin": _ints(proof.q_lin),
            "p_0":   _ints(proof.p_0),
            # opened columns, keyed by query index j (stringified for JSON)
            "opened_p1": {str(j): _ints(proof.opened_p1[j]) for j in Q},
            "opened_p2": {str(j): _ints(proof.opened_p2[j]) for j in Q},
            "paths_p1": {str(j): _path_json(proof.paths_p1[j]) for j in Q},
            "paths_p2": {str(j): _path_json(proof.paths_p2[j]) for j in Q},
        },
        "python_accept": bool(acc),
    }
    with open("/tmp/proof.json", "w") as f:
        json.dump(out, f)
    print("wrote /tmp/proof.json", file=sys.stderr)


if __name__ == "__main__":
    main()
