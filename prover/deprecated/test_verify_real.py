"""END-TO-END: the standalone verifier (verify.py) drives a real core.py prover
through the staged protocol and ACCEPTs — and REJECTs a tampered proof — in BOTH
modes (sound interactive `run_verification` and fast fused `run_verification_fast`).

This is the real protocol loop, not a hand-rolled check sequence: run_verification
calls prover(stage, seeds) round by round, drawing each seed after the prior
commit; the prover (TapeProver over core.prove) owns the witness; the verifier
sees only `claims` and compiles its own constraints. Seeds are the only thing
that crosses between them.

Run on the Spark:  PATH=$HOME/venv-hf/bin:$PATH ~/venv-hf/bin/python test_verify_real.py
"""
import torch
import core
import claims as C          # noqa: F401 — registers COMPILE/SAMPLE/AUX
import packets as PK        # noqa: F401 — registers EXPANDERS
import protocol as pr
import verify as vf
from tape_prover import TapeProver
from cuda_primitives import gl_matmul

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


def make_seq_rand(seeds):
    """A verifier coin source that hands out a fixed sequence of seeds. (A real
    interactive verifier would return secrets.token_bytes(32) each call; we fix
    them so prover and verifier agree in a single-process test.)"""
    it = iter(seeds)
    return lambda: next(it)


def build():
    """One real single-head matmul C = A·B (m=2,k=4,n=2), no rescale → (claims, inputs)."""
    m, k, n = 2, 4, 2
    A = core.Variable("A", length=m * k)
    B = core.Variable("B", length=k * n)
    Cv = core.Variable("C", length=m * n)
    claim = C.matmul_claim("mm", A, B, Cv, m=m, k=k, n=n)
    A_t = torch.randint(0, 1 << 20, (m, k), dtype=torch.int64, device="cuda").to(torch.uint64)
    B_t = torch.randint(0, 1 << 20, (k, n), dtype=torch.int64, device="cuda").to(torch.uint64)
    inputs = {A: A_t.view(-1), B: B_t.view(-1), Cv: gl_matmul(A_t, B_t).view(-1)}
    return [claim], inputs


def build_chain():
    """A MULTI-OP chain: C1 = A·B, then C2 = C1·D (mm2 consumes mm1's output).
    Two ops with op-challenges (ρ,λ) at DIFFERENT settled-list indices — exercises
    that seed-indexed challenges + the driver compose across ops on a real proof.
    Same shape as test_claims' known-good chained-matmul fixture."""
    m, k, n1, n2 = 2, 4, 2, 3
    A  = core.Variable("A",  length=m * k)
    B  = core.Variable("B",  length=k * n1)
    C1 = core.Variable("C1", length=m * n1)
    D  = core.Variable("D",  length=n1 * n2)
    C2 = core.Variable("C2", length=m * n2)
    mm1 = C.matmul_claim("mm1", A, B, C1, m=m, k=k,  n=n1)
    mm2 = C.matmul_claim("mm2", C1, D, C2, m=m, k=n1, n=n2)
    A_t = torch.randint(0, 1 << 20, (m, k),  dtype=torch.int64, device="cuda").to(torch.uint64)
    B_t = torch.randint(0, 1 << 20, (k, n1), dtype=torch.int64, device="cuda").to(torch.uint64)
    D_t = torch.randint(0, 1 << 20, (n1, n2), dtype=torch.int64, device="cuda").to(torch.uint64)
    C1_t = gl_matmul(A_t, B_t)
    C2_t = gl_matmul(C1_t, D_t)
    inputs = {A: A_t.view(-1), B: B_t.view(-1), C1: C1_t.view(-1),
              D: D_t.view(-1), C2: C2_t.view(-1)}
    return [mm1, mm2], inputs


SEEDS = [b"s_op-fixed", b"s_comb-fixed", b"s_col-fixed"]


class TamperedProver(TapeProver):
    """A prover that flips one opened value — slot 0 of opened_p1 (commit 1 holds
    the blinding + first witness rows, so slot 0 always exists)."""
    def __call__(self, stage, *seeds):
        out = super().__call__(stage, *seeds)
        if stage in (4, 0):
            o1 = out[5] if stage == 0 else out[0]       # opened_p1 dict
            j = next(iter(o1)); o1[j][0] = (o1[j][0] + 1) % pr.P
        return out


def run_case(label, claims, inputs):
    """Drive both verifier modes + the tamper check on one (claims, inputs)."""
    prover = TapeProver(claims, inputs, CFG)
    ok_slow = vf.run_verification(prover, claims, CFG, make_seq_rand(list(SEEDS)))
    ok_fast = vf.run_verification_fast(prover, claims, CFG, make_seq_rand(list(SEEDS)))
    tprover = TamperedProver(claims, inputs, CFG)
    rej = (not vf.run_verification(tprover, claims, CFG, make_seq_rand(list(SEEDS)))
           and not vf.run_verification_fast(tprover, claims, CFG, make_seq_rand(list(SEEDS))))
    ok = ok_slow and ok_fast and rej
    print(f"[{'OK ' if ok else 'XX '}] {label}: interactive {'ACCEPT' if ok_slow else 'REJECT'}, "
          f"fused {'ACCEPT' if ok_fast else 'REJECT'}, tamper {'REJECT' if rej else 'ACCEPT(BUG!)'}")
    return ok


def main():
    results = [run_case("single matmul", *build()),
               run_case("chain mm1→mm2", *build_chain())]
    ok = all(results)
    print(f"\n=== standalone verify (both modes): {sum(results)}/{len(results)} "
          f"{'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
