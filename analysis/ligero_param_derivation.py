"""Derive the Ligero geometry (K_DEG, ELL, rho, T_QUERIES) instead of assuming it.

The paper fixes ELL=8192, K_DEG=16384, rho=4, N_LIG=65536 and prices the prover
with the constants of Appendix A.5 (A_c=4.2, A_f=3.4, A_x=4.2, D=0.5, C=15 ns).
Those constants are not independent of the geometry: every one of them is a
transform (or hash) cost per *row*, amortized over ELL message slots per row. So
they all carry a factor

    lambda = K_DEG / ELL          (currently 2.0)

and a factor c(n), the transform cost per element at length n. This script makes
that dependence explicit, prices it with a *measured* c(n) curve, and searches
the geometry. It reproduces the paper's constants and the demonstrated run's
prove time / proof size / verify time as a validation gate before searching.

Two facts the geometry has to respect:

  * Quadratic test. p_0 has degree 2*K_DEG-2, so it needs N_LIG > 2*K_DEG,
    i.e. rho > 2. With power-of-two NTT lengths that forces rho >= 4.
    (prover/core.py asserts 2*K_DEG <= N_LIG.)
  * Zero-knowledge. Each row carries K_DEG - ELL random pad slots, and that pad
    is what hides the opened columns: `pad` random slots perfectly hide any
    <= pad distinct column openings over the lifetime of the commitment
    (analysis/persistent-weights.md:136). So the requirement is

        pad >= T_QUERIES * (number of proofs served by the commitment)

    which is an ABSOLUTE count of columns, not a fraction of K_DEG. The
    per-proof blocks (R_p1, R_p2) are fresh every proof and need only pad >= T.
    Only the persistent weight block R_W needs the long lifetime.

The current design sets pad = K_DEG/2 = 8192, which at K_DEG=16384 forces
ELL=8192 and lambda=2 -- i.e. it pays 2x on every heavy term. That is the choice
this script tests.

c(n) is measured on the local V100 with VerInf's own kernel (prover/kernels/ntt.cuh,
round-trip gated), then rescaled to the GB10 the runs were measured on by
anchoring at the paper's 0.42 ns/element at 2^15.
"""

import math

# ---------------------------------------------------------------------------
# 1. Measured transform primitive: ns per element vs NTT length.
#    Local V100 (Tesla V100-SXM3-32GB), VerInf's gl_ntt kernels, batch sized to
#    2^26 elements per point, iNTT(NTT(x)) == x verified at every length.
#    Note the dip at 2^16: that length takes the Bailey (4-step) path, which is
#    hard-coded for N=65536 in ntt.cuh. Every other length uses the generic
#    fused path, whose DRAM passes grow as ~2 + (log2 n - 12).
# ---------------------------------------------------------------------------
C_V100 = {                 # log2(n): forward ns/element
    12: 0.112, 13: 0.127, 14: 0.149, 15: 0.170,
    16: 0.142,             # <-- Bailey path
    17: 0.214, 18: 0.234, 19: 0.273, 20: 0.381,
    21: 0.523, 22: 0.643, 23: 0.754, 24: 0.889,
}

# Anchor to the machine the runs were measured on: the paper prices the NTT at
# 0.42 ns/element at length 2^15 on the GB10 (Appendix A.5).
GB10_ANCHOR_NS = 0.42
SCALE = GB10_ANCHOR_NS / C_V100[15]        # ~2.47x: GB10 is slower than V100

def c(n):
    """Transform ns/element at length n, on the GB10, from the measured curve."""
    lg = int(math.log2(n))
    assert 2 ** lg == n, "NTT lengths are powers of two"
    if lg not in C_V100:
        # Extrapolate the generic path linearly in log2(n) (it is DRAM-pass bound).
        lg_hi = max(C_V100)
        slope = (C_V100[lg_hi] - C_V100[lg_hi - 1])
        return SCALE * (C_V100[lg_hi] + slope * (lg - lg_hi))
    return SCALE * C_V100[lg]

HASH_NS_PER_COMPRESSION = 0.5              # 2.0 Gcompress/s BLAKE3, Appendix A.5
FIELD_BYTES = 8
JSON_BYTES_PER_ELEM = 20.1                 # measured: 92 GB dump / (40 cols * 1.152e8 rows)

# ---------------------------------------------------------------------------
# 2. The cost drivers of the demonstrated 400B run (paper Appendix A.2).
# ---------------------------------------------------------------------------
def W_of(S):  return 4.00e11 + 4.48e8 * S + 40320 * S ** 2
def L_of(S):  return 1.19e8  + 1.50e8 * S + 12480 * S ** 2
def Q_of(S):  return 5.93e7  + 1.54e8 * S + 19200 * S ** 2

# The verifier's independent compile of the demonstrated 1093-token proof:
W_RUN = 115_235_029 * 8192      # 9.44e11 witness slots  (115,235,029 rows at ELL=8192)
Q_RUN = 23_554_246  * 8192      # 1.93e11 quadratic products
L_RUN = L_of(1093)
W_WEIGHTS = 4.00e11             # the persistent block: committed params

T_WIT_S = 3888.8                # one witness sweep, measured (reveal pass, 64.8 min)
VERIFY_NS_PER_CELL = 14.0 * 3600 * 1e9 / (30 * 115_235_029)   # 14.0 h at T=30 over 1.152e8 rows


# ---------------------------------------------------------------------------
# 3. The cost identity, with the geometry left free.
#    Per ROW of the committed matrix (ELL message slots + pad random slots):
#      encode  A_c : iNTT(K) + NTT(N)           -> K*c(K) + N*c(N)
#      reencode A_x: the round-4 column opening  -> same again
#      lin fold A_f: 2 transforms of length 2K   -> 4K*c(2K)
#      hash    D   : N/8 BLAKE3 compressions     -> N/8 * 0.5 ns
#    Per QUADRATIC row: ~8.9 transforms of length 2K (products + operand re-encode).
#    Per linear constraint: one BLAKE3 challenge hash (B = 0.6 ns).
#    Plus 4 witness sweeps (the only term the four rounds multiply) and the
#    constraint-coefficient arithmetic E (~4.5% of prove, fused into the fold).
# ---------------------------------------------------------------------------
QUAD_TRANSFORMS_PER_ROW = 8.9   # calibrated so C = 15 ns/product at the paper's geometry
B_NS_PER_CID = 0.6
E_FRACTION = 0.045              # constraint-coefficient work, measured 4-5% of prove

def cost(K, ell_w, ell_p, rho, T, W=W_RUN, Q=Q_RUN, L=L_RUN, W_w=W_WEIGHTS,
         weight_commit_lifetime=None, n_wit_sweeps=4):
    """Prove time (s), proof size (bytes), verify time (s) for one geometry.

    ell_w: message slots per row in the persistent weight block
    ell_p: message slots per row in the per-proof blocks (R_p1, R_p2)
    (they may differ: only the weight block needs a long ZK lifetime)
    """
    N = rho * K
    assert 2 * K <= N, "quadratic test needs N_LIG >= 2*K_DEG"
    pad_w, pad_p = K - ell_w, K - ell_p
    assert pad_w >= T and pad_p >= T, "ZK: pad must cover the opened columns"

    W_p = W - W_w                                  # per-proof witness slots
    rows_w, rows_p = W_w / ell_w, W_p / ell_p
    rows = rows_w + rows_p

    def per_row_ns(with_commit_encode=True):
        enc = K * c(K) + N * c(N)                  # A_c
        fold = 4 * K * c(2 * K)                    # A_f
        hsh = (N / 8) * HASH_NS_PER_COMPRESSION    # D
        reenc = enc                                # A_x (round-4 column opening)
        return (enc if with_commit_encode else 0) + fold + hsh + reenc

    # The weight block's commit encode is amortized if the commitment persists.
    lifetime = weight_commit_lifetime or max(1, pad_w // T)
    enc_w = K * c(K) + N * c(N)
    streaming_ns = (
        rows_p * per_row_ns()
        + rows_w * (per_row_ns(with_commit_encode=False) + enc_w / lifetime)
    )

    quad_rows = Q / ell_p
    quad_ns = quad_rows * QUAD_TRANSFORMS_PER_ROW * (2 * K) * c(2 * K)
    lin_ns = B_NS_PER_CID * L

    prove_s = (streaming_ns + quad_ns + lin_ns) / 1e9 + n_wit_sweeps * T_WIT_S
    prove_s /= (1 - E_FRACTION)                    # fold coefficient arithmetic

    proof_b = T * rows * FIELD_BYTES + (4 * K + ell_p) * FIELD_BYTES
    verify_s = T * rows * VERIFY_NS_PER_CELL / 1e9

    return dict(prove_h=prove_s / 3600, proof_gb=proof_b / 1e9,
                proof_json_gb=T * rows * JSON_BYTES_PER_ELEM / 1e9,
                verify_h=verify_s / 3600, rows=rows, lifetime=lifetime,
                lam_w=K / ell_w, lam_p=K / ell_p)


# ---------------------------------------------------------------------------
# 4. Validation gate. Reproduce the paper's per-slot constants and the
#    demonstrated run before trusting any search result.
# ---------------------------------------------------------------------------
def validate():
    K, ELL, rho = 16384, 8192, 4
    N = rho * K
    A_c = (K * c(K) + N * c(N)) / ELL
    A_f = (4 * K * c(2 * K)) / ELL
    D   = (N / 8) * HASH_NS_PER_COMPRESSION / ELL
    C   = QUAD_TRANSFORMS_PER_ROW * (2 * K) * c(2 * K) / ELL
    print("Validation 1 -- reconstruct the paper's Appendix A.5 constants")
    print(f"  A_c (commit encode)   model {A_c:5.2f} ns/slot   paper 4.2")
    print(f"  A_x (column re-encode) model {A_c:5.2f} ns/slot   paper 4.2")
    print(f"  A_f (linear fold)     model {A_f:5.2f} ns/slot   paper 3.4")
    print(f"  D   (column hashing)  model {D:5.2f} ns/slot   paper 0.5")
    print(f"  C   (quadratic fold)  model {C:5.2f} ns/prod   paper 15")

    r = cost(K, ELL, ELL, rho, T=40)
    print("\nValidation 2 -- the demonstrated 1093-token 400B run (T=40)")
    print(f"  prove   model {r['prove_h']:5.1f} h    paper's identity floor 8-10 h; measured 19.3 h (~2x floor)")
    print(f"  proof   model {r['proof_gb']:5.1f} GB binary / {r['proof_json_gb']:.0f} GB as JSON   measured 92 GB JSON (37 GB of columns)")
    print(f"  verify  model {r['verify_h']:5.1f} h    measured 14.0 h at T=30 (calibration anchor)")
    print(f"  rows    model {r['rows']:.3e}          verifier reported 1.152e8")
    print(f"  weight-commit lifetime {r['lifetime']} proofs   paper: 8192/T")


# ---------------------------------------------------------------------------
# 5. The search.
# ---------------------------------------------------------------------------
def search(T=40, pad_w=8192, pad_p=None):
    """Sweep the geometry. pad_w keeps the weight commitment's ZK lifetime;
    pad_p only has to hide one proof's openings."""
    pad_p = pad_p or max(T, 128)
    print(f"\n\nGeometry search  (T={T}, weight-block pad={pad_w} -> lifetime {pad_w//T} proofs,"
          f" per-proof pad={pad_p})")
    print(f"{'K_DEG':>9} {'N_LIG':>9} {'ELL_w':>8} {'ELL_p':>8} {'lam_w':>6} {'prove h':>8}"
          f" {'proof GB':>9} {'verify h':>9}")
    rows_out = []
    for rho in (4,):
        for lg in range(14, 23):
            K = 1 << lg
            ell_w, ell_p = K - pad_w, K - pad_p
            if ell_w <= 0 or ell_p <= 0:
                continue
            r = cost(K, ell_w, ell_p, rho, T)
            rows_out.append((K, rho, r))
            print(f"{K:>9} {rho*K:>9} {ell_w:>8} {ell_p:>8} {r['lam_w']:>6.2f}"
                  f" {r['prove_h']:>8.1f} {r['proof_gb']:>9.2f} {r['verify_h']:>9.2f}")
    return rows_out


if __name__ == "__main__":
    validate()

    print("\n\n=== A. Keep the current NTT lengths, spend the ZK pad correctly ===")
    print("K_DEG=16384, N_LIG=65536 (the Bailey fast path), but stop tying pad to K/2.")
    base = cost(16384, 8192, 8192, 4, T=40)
    for pad in (8192, 4096, 2048, 1024, 512, 256, 128, 64):
        if 16384 - pad <= 0:
            continue
        r = cost(16384, 16384 - pad, 16384 - pad, 4, T=40)
        print(f"  pad={pad:>5}  ELL={16384-pad:>6}  lambda={r['lam_w']:.3f}  "
              f"prove {r['prove_h']:5.1f} h ({base['prove_h']/r['prove_h']:.2f}x)  "
              f"proof {r['proof_gb']:6.2f} GB ({base['proof_gb']/r['proof_gb']:.2f}x)  "
              f"verify {r['verify_h']:5.2f} h  weight-commit lifetime {r['lifetime']:>3} proofs")

    print("\n\n=== B. Split the pad by block (only the weights need a long lifetime) ===")
    for pad_w in (8192, 2048):
        r = cost(16384, 16384 - pad_w, 16384 - 128, 4, T=40)
        print(f"  weight pad={pad_w:>5} (lifetime {r['lifetime']:>3}), per-proof pad=128:  "
              f"prove {r['prove_h']:5.1f} h  proof {r['proof_gb']:6.2f} GB  verify {r['verify_h']:5.2f} h")

    search(T=40, pad_w=8192)
    search(T=80, pad_w=8192)
