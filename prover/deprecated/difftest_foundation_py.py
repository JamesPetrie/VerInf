"""Python reference for the Rust foundation diff-test: prints the SAME lines as
src/bin/difftest_foundation.rs via protocol.py. A driver diffs the two outputs;
any mismatch is a bit-exactness bug (byte layout, reduction, hash framing)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
import protocol as pr

xs = [0, 1, 2, 0xFFFFFFFF00000000, 123456789012345]
for a in xs:
    for b in xs:
        print("add", a, b, pr.add(a, b))
        print("sub", a, b, pr.sub(a, b))
        print("mul", a, b, pr.mul(a, b))
for a in xs:
    if a != 0:
        print("inv", a, pr.inv(a))
print("pow", 7, 1000000, pow(7, 1000000, pr.P))

seed = b"difftest-seed-0"
for i in range(6):
    for lab in ["irs", "lin", "quad", "op0:rho"]:
        print("challenge", i, lab, pr.challenge(seed, i, lab))

cfg = pr.Config(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
print("cols", pr.random_columns_n(seed, 4, 32))
print("opvec", pr.op_vec(seed, 3, "rho", 5))

for j in range(4):
    print("eta", j, cfg.eta(j))
for c in range(4):
    print("zeta", c, cfg.zeta(c))
print("lagrange", pr.lagrange(cfg, 2, cfg.eta(1)))
print("polyeval", pr.poly_eval([3, 5, 7, 9], 12345))

col = [10, 20, 30]
leaf = pr.merkle_leaf(col)
print("leaf", leaf.hex())
import blake3
sib = pr.merkle_leaf([40, 50])
root = blake3.blake3(sib + leaf).digest()        # side 0 = sib left: blake3(sib ‖ leaf)
print("mverify", pr.merkle_verify(leaf, [(sib, 0)], root))
