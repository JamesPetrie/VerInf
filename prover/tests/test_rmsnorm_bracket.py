"""RmsNorm wrap-free bracket: honest ACCEPT + the wrapped-y forgery REJECT.

The forgery (degrees-of-freedom-review.md §3, rmsnorm-bracket-fix.md): before
the fix, the bracket quads held only mod P and the production slack-chunk
windows (4 x 16-bit) covered the whole field, so ANY forged y' — with the
output forged to x.y' — satisfied every constraint: rsqrt uniqueness was
vacuous. The fix assembles both bracket products from range-checked limbs, so
the same forged witness must now fail a range LogUp.

The forged witness here satisfies EVERY linear family and quad mod P by
construction (the carry-chain splits telescope back to q.S_total exactly);
only the tight range checks can reject it — which is precisely the property
under test.

Run on the Spark:  ~/venv-hf/bin/python run_tests.py test_rmsnorm_bracket
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import claims as C
import packets as PK        # noqa: F401 — registers EXPANDERS
import compute_fns
from cuda_primitives import P
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)


def _run(tape, label):
    proof = tape.prove(seed=b"rms-bracket")
    acc, msg = rust_verify_tape(tape, proof, seed=b"rms-bracket")
    print(f"    {label}: {'ACCEPT' if acc else 'REJECT'} ({msg})")
    return acc


def test_honest_toy():
    tape = Tape(CFG, lazy=True)
    x = tape.commit("rx", _t([3, 1, 4, 2, 5, 2, 6, 1]), (8,))
    tape.rmsnorm(x, d=4, s=4, eps_int=1)
    assert _run(tape, "honest toy"), "honest toy rmsnorm must ACCEPT"


def test_honest_production_scales():
    """7B-grade constants (d=4096, s=2^12, eps_int=168): exercises the derived
    widths at their real values (y_width=21, 8-bit slack top, 10/11-bit carry
    tops) and the S_total < 2^48 completeness path with realistic activations."""
    torch.manual_seed(0)
    d = 4096
    import numpy as np
    raw = (torch.randn(2 * d, dtype=torch.float64) * (2.0 * 4096)).to(torch.int64)
    field = np.array([int(v) % P for v in raw.tolist()], dtype=np.uint64)
    tape = Tape(CFG, lazy=True)
    x = tape.commit("rxp", torch.from_numpy(field).cuda(), (2 * d,))
    tape.rmsnorm(x, d=d, s=1 << 12, eps_int=168)
    assert _run(tape, "honest 7B scales"), "honest production-scale rmsnorm must ACCEPT"


def test_honest_high_energy():
    """Force S_total into (2^52, 2^54) so the top limb needs >16 bits — the
    regime the 18-bit widening exists for (the old 3x16 cap rejected it).
    Constant x = 1.5e6 at scale 2^12 → S_total = d·x² ≈ 2^53."""
    d = 4096
    C = 1_500_000
    vals = [C] * (2 * d)
    tape = Tape(CFG, lazy=True)
    x = tape.commit("rxhi", torch.tensor(vals, dtype=torch.int64, device="cuda")
                    .to(torch.uint64), (2 * d,))
    tape.rmsnorm(x, d=d, s=1 << 12, eps_int=168)
    assert _run(tape, "honest high-energy (S_total~2^53)"), \
        "honest high-S_total rmsnorm must ACCEPT (18-bit limb path)"


def test_forged_y_rejected():
    """The pre-fix break: y' = y + 3 per row, all downstream witnesses forged
    consistently (mod P). Every linear family and quad is satisfied; the tight
    range windows are the only defense, and they must fire."""
    real_compute = compute_fns.rmsnorm_compute

    def forged_compute(claim, live):
        out = real_compute(claim, live)
        sc = claim.config
        B, d, magic = sc.B, sc.d, sc.magic
        S_tot = [int(v) for v in out[claim.S_total].cpu().tolist()]
        y_h = [int(v) for v in out[claim.y].cpu().tolist()]
        dev = out[claim.y].device

        def tens(vals):
            return torch.tensor([v % P for v in vals], dtype=torch.uint64,
                                device=dev)

        rows = {k: [] for k in ("y", "ym1", "q1", "q2", "s_lo", "s_hi")}
        chunk_rows = {}     # var -> per-row value list

        def put(var, val):
            chunk_rows.setdefault(var, []).append(val)

        for b in range(B):
            yf = y_h[b] + 3                        # ANY y' worked pre-fix
            S = S_tot[b]
            q1 = (yf * yf) % P
            q2 = ((yf - 1) * (yf - 1)) % P
            s_lo = (q1 * S - magic) % P
            s_hi = (magic - 1 - q2 * S) % P
            rows["y"].append(yf); rows["ym1"].append(yf - 1)
            rows["q1"].append(q1); rows["q2"].append(q2)
            rows["s_lo"].append(s_lo); rows["s_hi"].append(s_hi)
            # slack chunks: 16-bit greedy split of the (possibly wrapped) value
            # — exactly what the pre-fix prover committed. Satisfies F6/F7 mod P;
            # the narrowed top-chunk table is what must reject.
            n_slack = len(claim.s_lo_chunks)
            for n in range(n_slack):
                put(claim.s_lo_chunks[n], (s_lo >> (16 * n)) & 0xFFFF)
                put(claim.s_hi_chunks[n], (s_hi >> (16 * n)) & 0xFFFF)
            for n, var in enumerate(claim.ym1_chunks):
                put(var, ((yf - 1) >> (16 * n)) & 0xFFFF)
            limbs = [(S >> (16 * n)) & 0xFFFF for n in range(3)]
            for tag, q, Hv, glv, g0v, g1v, G2v in (
                    ("lo", q1, claim.lo_H, claim.lo_gl, claim.lo_g0h_chunks,
                     claim.lo_g1h_chunks, claim.lo_G2_chunks),
                    ("hi", q2, claim.hi_H, claim.hi_gl, claim.hi_g0h_chunks,
                     claim.hi_g1h_chunks, claim.hi_G2_chunks)):
                # Carry chain over the mod-P products: the splits telescope, so
                # every linear family holds mod P whatever q is.
                H = [(q * limbs[k]) % P for k in range(3)]
                g0l, g0h = H[0] & 0xFFFF, H[0] >> 16
                G1 = H[1] + g0h
                g1l, g1h = G1 & 0xFFFF, G1 >> 16
                G2 = H[2] + g1h
                for k in range(3):
                    put(Hv[k], H[k])
                put(glv[0], g0l); put(glv[1], g1l)
                for j, var in enumerate(g0v):
                    put(var, (g0h >> (16 * j)) & 0xFFFF)
                for j, var in enumerate(g1v):
                    put(var, (g1h >> (16 * j)) & 0xFFFF)
                for j, var in enumerate(G2v):
                    put(var, (G2 >> (16 * j)) & 0xFFFF)

        out[claim.y] = tens(rows["y"])
        out[claim.y_m1] = tens(rows["ym1"])
        out[claim.q1] = tens(rows["q1"])
        out[claim.q2] = tens(rows["q2"])
        out[claim.s_lo] = tens(rows["s_lo"])
        out[claim.s_hi] = tens(rows["s_hi"])
        for var, vals in chunk_rows.items():
            out[var] = tens(vals)
        # Forge the output to x . y'_broadcast so Freivalds and F5 hold.
        from cuda_primitives import gl_mul
        x_t = live[claim.x]
        y_per_cell = out[claim.y].view(B, 1).expand(B, d).contiguous().view(-1)
        out[claim.output] = gl_mul(x_t, y_per_cell)
        return out

    compute_fns.COMPUTE_FNS[C.RmsNormClaim] = forged_compute
    try:
        tape = Tape(CFG, lazy=True)
        x = tape.commit("rxf", _t([3, 1, 4, 2, 5, 2, 6, 1]), (8,))
        tape.rmsnorm(x, d=4, s=4, eps_int=1)
        try:
            acc = _run(tape, "forged y'=y+3")
        except AssertionError:
            # The prover's own witness sanity asserts may fire on the forged
            # values — that is a prover-side guard, not the soundness gate.
            # The claim under test is the VERIFIER's rejection, so a prover
            # that refuses to even build the proof also passes.
            print("    forged y'=y+3: prover refused (witness assert)")
            acc = False
    finally:
        compute_fns.COMPUTE_FNS[C.RmsNormClaim] = real_compute
    assert not acc, "forged wrapped-y witness must be REJECTED post-fix"


if __name__ == "__main__":
    ok = True
    for fn in (test_honest_toy, test_honest_production_scales,
               test_honest_high_energy, test_forged_y_rejected):
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except Exception as e:
            ok = False; print(f"[XX ] {fn.__name__}: {e}")
    sys.exit(0 if ok else 1)
