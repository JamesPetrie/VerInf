"""Shared statement builders — the PUBLIC statement (claim shapes) plus a witness.

setup_statement.py uses these to write the public claims.json (verifier side);
prover_server.py uses them to get the witness (prover side). Layout (row_start)
is value-independent, so the claims.json written from one witness matches the
prover's layout for any witness — the verifier never sees the witness itself."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
import torch
import core
import claims as C          # noqa: F401 — registers COMPILE/SAMPLE/AUX
import packets as PK        # noqa: F401 — registers EXPANDERS
from cuda_primitives import gl_matmul

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


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


def build_chain():
    m, k, n1, n2 = 2, 4, 2, 3
    A = core.Variable("A", length=m * k)
    B = core.Variable("B", length=k * n1)
    C1 = core.Variable("C1", length=m * n1)
    D = core.Variable("D", length=n1 * n2)
    C2 = core.Variable("C2", length=m * n2)
    mm1 = C.matmul_claim("mm1", A, B, C1, m=m, k=k, n=n1)
    mm2 = C.matmul_claim("mm2", C1, D, C2, m=m, k=n1, n=n2)
    A_t = torch.randint(0, 1 << 20, (m, k), dtype=torch.int64, device="cuda").to(torch.uint64)
    B_t = torch.randint(0, 1 << 20, (k, n1), dtype=torch.int64, device="cuda").to(torch.uint64)
    D_t = torch.randint(0, 1 << 20, (n1, n2), dtype=torch.int64, device="cuda").to(torch.uint64)
    C1_t = gl_matmul(A_t, B_t)
    C2_t = gl_matmul(C1_t, D_t)
    inputs = {A: A_t.view(-1), B: B_t.view(-1), C1: C1_t.view(-1),
              D: D_t.view(-1), C2: C2_t.view(-1)}
    return [mm1, mm2], inputs, CFG


def build_rmsnorm():
    from tape import Tape
    t = Tape(CFG)
    xr = torch.tensor([3, 1, 4, 2, 5, 2, 6, 1], dtype=torch.int64, device="cuda").to(torch.uint64)
    xv = t.commit("rms_x", xr, (8,))
    t.rmsnorm(xv, d=4, s=4, eps_int=1, slack_n_chunks=1)
    return t.claims, t.inputs, t.cfg


STATEMENTS = {"matmul": build_matmul, "chain": build_chain, "rmsnorm": build_rmsnorm}
