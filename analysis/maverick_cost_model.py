"""Analytical cost model for the Maverick Ligero proof.

Emits, per claim type, the three cost drivers as polynomials in the context
length S (= sequence length T):
    W      = committed witness slots
    cids   = distinct linear constraint ids
    Q      = quadratic products (slot-products)

Each is c0 + c1*S + c2*S^2. The downstream cost is then
    T_prove ~= A*W + B*cids + C*Q
with machine constants A (bandwidth/NTT per slot), B (hash per cid), C (quad
fold per product) applied separately.

Per-claim (W, cids, Q) forms are the active-in-Maverick variants (output-rescale
ON for every matmul/hadamard/rope/rmsnorm; softmax saturate+causal ON; silu no
rescale), grounded in claims.py / routing_claim.py / CLAIM_SPECS.md. The model
structure (which claims, which shapes) is from demo_maverick_full.py.

Pure Python, no deps. Cross-check: the S^2 coefficient of W must be ~840/layer
(240 scores-matmul + 600 softmax) * 48 = 40320, matching the measured fit in
witness-scaling-measurement.md.
"""
from collections import defaultdict


class Poly:
    """c0 + c1*S + c2*S^2 (degree capped at 2)."""
    __slots__ = ("c",)

    def __init__(self, c0=0.0, c1=0.0, c2=0.0):
        self.c = [float(c0), float(c1), float(c2)]

    @staticmethod
    def lift(x):
        return x if isinstance(x, Poly) else Poly(x)

    def __add__(self, o):
        o = Poly.lift(o)
        return Poly(*(a + b for a, b in zip(self.c, o.c)))
    __radd__ = __add__

    def __mul__(self, o):
        o = Poly.lift(o)
        r = [0.0] * 5
        for i in range(3):
            for j in range(3):
                r[i + j] += self.c[i] * o.c[j]
        assert abs(r[3]) < 1e-6 and abs(r[4]) < 1e-6, f"degree > 2: {r}"
        return Poly(r[0], r[1], r[2])
    __rmul__ = __mul__

    def __repr__(self):
        return f"{self.c[0]:14.6g} {self.c[1]:14.6g} {self.c[2]:14.6g}"


S = Poly(0, 1, 0)

# ---- Maverick config (demo_maverick_full.py / demo_maverick_block.py) ----
d = 5120          # hidden
dff_d = 16384     # dense FFN
dff_e = 8192      # expert FFN
E = 128           # experts (all committed; top-1 active)
H = 40            # query heads
dh = 128          # head dim
V = 202048        # vocab
N_DENSE = 24      # dense layers (even il, all RoPE)
N_MOE_ROPE = 12   # MoE layers with RoPE (il = 1,5,9,...)
N_MOE_NOPE = 12   # MoE layers NoPE  (il = 3,7,11,...)
R_W = 400e9       # committed weights (all experts) -- S-independent floor on W


# ---- per-claim (W, cids, Q), active-in-Maverick forms ----
def matmul(m, k, n, heads=1, rescale=True):
    m, k, n = Poly.lift(m), Poly.lift(k), Poly.lift(n)
    mHn = m * (heads * n)
    if rescale:
        return (6 * mHn + 3 * k, 2 * k + Poly(heads) + 2 * mHn, k + 2 * mHn)
    return (mHn + 3 * k, 2 * k + Poly(heads), k)


def rmsnorm(B, dd):                         # output-rescale ON, K=4 slack chunks
    B, dd = Poly.lift(B), Poly.lift(dd)
    Bd = B * dd
    return (7 * Bd + 26 * B, 7 * B + 2 * Bd, 3 * Bd + 13 * B)


def softmax(B, M):                          # saturate + causal ON
    B, M = Poly.lift(B), Poly.lift(M)
    BM = B * M
    causal_F0 = B * (M + 1) * 0.5           # z-decomp shrinks to the triangle
    return (15 * BM + 9 * B, causal_F0 + 4 * BM + 5 * B, 8 * BM + 3 * B)


def silu(L):       L = Poly.lift(L); return (23 * L, 7 * L, 12 * L)          # no rescale
def hadamard(L):   L = Poly.lift(L); return (6 * L, 2 * L, 3 * L)            # rescale ON
def rope(L):       L = Poly.lift(L); return (6 * L, 3 * L, 2 * L)            # rescale ON
def add_(L):       L = Poly.lift(L); return (L, L, Poly())
def embed_lk(L):   L = Poly.lift(L); return (L, L, Poly())                   # gain bcast / continuation
def ptlookup(L):   L = Poly.lift(L); return (3 * L, L, L)


def routing(T, Ec, nw):
    T = Poly.lift(T); TE = T * Ec
    return ((4 + 2 * nw) * TE + 2 * T, 3 * TE + 3 * T, (2 + nw) * TE)


def masked_combine(T, Ec, F):
    T, F = Poly.lift(T), Poly.lift(F); TF = T * F; ETF = Ec * TF
    return (2 * ETF + TF, ETF + TF, ETF)


def freivalds_combine(T, Ec, F):
    T, F = Poly.lift(T), Poly.lift(F); ET = Ec * T
    return (T * F + 4 * ET + T, 3 * ET + 2 * T, ET)


# ---- blocks: list of (claim_type, (W, cids, Q)) ----
def common(use_rope):
    """Attention + the two RMS norms (rows 1-13), shared by dense & MoE layers."""
    L = [("rmsnorm", rmsnorm(S, d)),
         ("embed_lookup", embed_lk(S * d)), ("hadamard", hadamard(S * d)),     # gain1
         ("matmul", matmul(S, d, d)), ("matmul", matmul(S, d, d)), ("matmul", matmul(S, d, d))]  # Q,K,V
    if use_rope:
        L += [("rope", rope(S * d)), ("rope", rope(S * d))]
    L += [("matmul_scores", matmul(S, d, S, heads=H)),  # scores QK^T  -> 40*S^2 output
          ("softmax", softmax(H * S, S)),              # B = 40S, M = S
          ("matmul", matmul(S, H * S, dh, heads=H)),   # scores@V     -> 5120*S output
          ("matmul", matmul(S, d, d)),                 # O proj
          ("add", add_(S * d)),                        # resid1
          ("rmsnorm", rmsnorm(S, d)),                  # n2
          ("embed_lookup", embed_lk(S * d)), ("hadamard", hadamard(S * d))]    # gain2
    return L


def dense_ffn():
    return [("matmul", matmul(S, d, dff_d)), ("matmul", matmul(S, d, dff_d)),  # gate, up
            ("silu", silu(S * dff_d)), ("hadamard", hadamard(S * dff_d)),
            ("matmul", matmul(S, dff_d, d)),                                    # down
            ("add", add_(S * d))]


def scale3(mult, t):
    return (mult * t[0], mult * t[1], mult * t[2])


def moe_ffn():
    return [("matmul", matmul(S, d, E)),                       # router
            ("routing", routing(S, E, 3)),
            ("ptlookup", ptlookup(S)),                         # sigma(r_chosen)
            ("masked_combine", masked_combine(S, 1, d)),       # s_rep broadcast (E=1)
            ("hadamard", hadamard(S * d)),                     # x_r
            ("matmul_expert", scale3(E, matmul(S, d, dff_e))),  # 128 gate_e
            ("matmul_expert", scale3(E, matmul(S, d, dff_e))),  # 128 up_e
            ("freivalds_combine", freivalds_combine(S, E, dff_e)),   # g_sum
            ("freivalds_combine", freivalds_combine(S, E, dff_e)),   # up_sum
            ("silu", silu(S * dff_e)), ("hadamard", hadamard(S * dff_e)),
            ("matmul_expert", scale3(E, matmul(S, dff_e, d))),  # 128 down_e
            ("freivalds_combine", freivalds_combine(S, E, d)),       # ffn
            # shared expert
            ("matmul", matmul(S, d, dff_e)), ("matmul", matmul(S, d, dff_e)),
            ("silu", silu(S * dff_e)), ("hadamard", hadamard(S * dff_e)),
            ("matmul", matmul(S, dff_e, d)),
            ("add", add_(S * d)), ("add", add_(S * d))]


def io():
    """Embedding select + LM head. Treated over S tokens (the hidden-prompt
    one-hot over V is a separate fixed cost, noted in the doc, not modeled here)."""
    return [("matmul_head", matmul(S, d, V)),                  # LM head, rescale ON
            ("matmul_embed", matmul(S, V, d, rescale=False))]  # token select


# ---- aggregate ----
def tag_scale(mult, block):
    return [(t, scale3(mult, tup)) for (t, tup) in block]


entries = (
    tag_scale(N_DENSE, common(True) + dense_ffn())
    + tag_scale(N_MOE_ROPE, common(True) + moe_ffn())
    + tag_scale(N_MOE_NOPE, common(False) + moe_ffn())
    + tag_scale(1, io())
)

tally = defaultdict(lambda: (Poly(), Poly(), Poly()))
for t, (w, c, q) in entries:
    W, C, Q = tally[t]
    tally[t] = (W + w, C + c, Q + q)

totW, totC, totQ = Poly(), Poly(), Poly()
print(f"{'claim type':18s} {'quantity':5s} | {'c0':>14s} {'c1 (per S)':>14s} {'c2 (per S^2)':>14s}")
print("-" * 86)
for t in sorted(tally):
    W, C, Q = tally[t]
    print(f"{t:18s} {'W':5s} | {W}")
    print(f"{'':18s} {'cids':5s} | {C}")
    print(f"{'':18s} {'Q':5s} | {Q}")
    totW, totC, totQ = totW + W, totC + C, totQ + Q
    print()

totW = totW + Poly(R_W)   # weights floor (S-independent)
print("=" * 86)
print(f"{'TOTAL (+weights)':18s} {'W':5s} | {totW}")
print(f"{'':18s} {'cids':5s} | {totC}")
print(f"{'':18s} {'Q':5s} | {totQ}")
print()
print("cross-check: W S^2 coeff should be ~840*48 = 40320 (measured 840/block)")
print(f"   W   S^2 = {totW.c[2]:.0f}   (per layer = {totW.c[2]/48:.1f})")
print(f"  cids S^2 = {totC.c[2]:.0f}   (per layer = {totC.c[2]/48:.1f})")
print(f"   Q   S^2 = {totQ.c[2]:.0f}   (per layer = {totQ.c[2]/48:.1f})")


# ---- intuitive approximation: one claim dominates each term ----
att = tuple(a + b for a, b in zip(tally["softmax"], tally["matmul_scores"]))  # S^2 source
exp = tally["matmul_expert"]                                                  # S source
nL = 48
print("\nINTUITIVE APPROXIMATION (one cause dominates each term):")
print("  S^2 term = ATTENTION ONLY (softmax + scores matmul are the only S^2 claims):")
print(f"     W    c2 = {att[0].c[2]:.0f}   closed form 21*n_q*n_L = {21*H*nL}")
print(f"     cids c2 = {att[1].c[2]:.0f}   6.5*n_q*n_L = {6.5*H*nL:.0f}")
print(f"     Q    c2 = {att[2].c[2]:.0f}   10*n_q*n_L = {10*H*nL}")
print(f"  S term ~ COMMITTED EXPERTS (W: {exp[0].c[1]:.3g} = {100*exp[0].c[1]/totW.c[1]:.0f}% of the linear term)")
print(f"     closed form 6*E*(2*dff_e+d)*n_moe = {6*E*(2*dff_e+d)*24}")
print(f"  const ~ WEIGHTS (W only): R_W = {R_W:.2g}")


def ev(p, s):
    return p.c[0] + p.c[1] * s + p.c[2] * s * s


print("\n  W(S): which term dominates")
print(f"  {'S':>10s} {'W exact':>12s} {'weights':>9s} {'experts':>9s} {'attention':>9s}")
for s in (1093, 100_000, 1_000_000):
    w = ev(totW, s)
    print(f"  {s:>10d} {w:>12.3g} {R_W / w:>8.0%} {exp[0].c[1] * s / w:>8.0%} {att[0].c[2] * s * s / w:>8.0%}")
