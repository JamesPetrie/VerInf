"""Compare the ORDER of quadratic constraints between protocol.compile_claims
(what verify.py / Rust use) and the prover's core._compile_all (what p_0 is built
from). The quad combiner s_t = challenge(s_comb, t, 'quad') is indexed by list
POSITION, so a different order makes verification reject even when the quad SET
matches (which is all the parity test checked — it sorted them)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))  # pipeline/ on path
import sys
import core, claims as C, packets as PK, protocol as pr  # noqa
import test_rescale as TR

S_OP = b"chk-order"


def check(name, builder):
    tape, _ = builder()
    cfg = tape.cfg
    claims = list(tape.claims)
    cons = pr.compile_claims(list(claims), cfg, S_OP)
    proto = [(q.x_row, q.y_row, q.z_row, q.n, int(q.a), int(q.b)) for q in cons.quadratic]

    cl = core._with_synthesized_settlements(list(claims))
    _, _, _, _, mt = core._layout(cl, cfg)
    _, pquads, _, _, _ = core._compile_all(cl, S_OP, cfg, mt)
    prover = [(q.x_row, q.y_row, q.z_row, q.n, int(q.a_values[0]), int(q.b_values[0])) for q in pquads]

    same_set = sorted(proto) == sorted(prover)
    same_order = proto == prover
    print(f"=== {name}: proto {len(proto)} quads, prover {len(prover)} quads | "
          f"same_set={same_set} same_order={same_order} ===", file=sys.stderr)
    if not same_order:
        for i in range(max(len(proto), len(prover))):
            p = proto[i] if i < len(proto) else None
            q = prover[i] if i < len(prover) else None
            if p != q:
                print(f"  idx {i}: proto={p}", file=sys.stderr)
                print(f"          prover={q}", file=sys.stderr)


for name, b in [("matmul_rescale", TR.build_matmul),
                ("hadamard_rescale", TR.build_hadamard),
                ("rmsnorm_rescale", TR.build_rmsnorm)]:
    check(name, b)
