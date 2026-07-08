"""Find quads whose a_values/b_values are NON-CONSTANT across slots. protocol.py /
verify.py / Rust represent a quad's a,b as SCALARS (= a_values[0]); if the prover
emits a per-slot (varying) a_values or b_values, the scalar verifier mis-evaluates
that quad and rejects an honest proof. The compile-parity test only compared
a_values[0], so it never caught this."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))  # pipeline/ on path
import sys
import core, claims as C, packets as PK, protocol as pr  # noqa
import test_rescale as TR


def check(name, builder):
    tape, _ = builder()
    cfg = tape.cfg
    cl = core._with_synthesized_settlements(list(tape.claims))
    _, _, _, _, m_total = core._layout(cl, cfg)
    _, quads, _, _, _ = core._compile_all(cl, b"chk", cfg, m_total)
    print(f"=== {name}: {len(quads)} quads ===", file=sys.stderr)
    for q in quads:
        av = [int(v) for v in q.a_values]
        bv = [int(v) for v in q.b_values]
        a_const = all(v == av[0] for v in av)
        b_const = all(v == bv[0] for v in bv)
        flag = "" if (a_const and b_const) else "   <<< NON-CONSTANT"
        print(f"  {q.name:28s} n={q.n:3d} a_const={a_const} b_const={b_const}{flag}", file=sys.stderr)
        if not a_const:
            print(f"      a_values={av[:min(len(av),10)]}", file=sys.stderr)
        if not b_const:
            print(f"      b_values={bv[:min(len(bv),10)]}", file=sys.stderr)


for name, b in [("matmul_rescale", TR.build_matmul),
                ("hadamard_rescale", TR.build_hadamard),
                ("rmsnorm_rescale", TR.build_rmsnorm),
                ("softmax_rescale", TR.build_softmax)]:
    check(name, b)
