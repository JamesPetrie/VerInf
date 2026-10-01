"""Integer reference of the top-k routing claims (analysis/topk-routing-design.md
§3.1–3.3): the honest witness, and every linear, quadratic and range relation
the claims emit, checked over Python integers mod P. No torch: the CPU gate
uses it to test the selection lemma and the review's counterexample, and the
GPU suite uses it as the expected values of the proved claims.

Layouts follow the claims: token-major (T, E) for the routing mask and its
derived values; per-slot masks M_i (T, E); slot scores and gate weights
token-major (T, k)."""
from topk_params import P, l_bits


def f(x):
    """Field representative of an integer."""
    return x % P


# ---- selection (TopkRoutingClaim) ----------------------------------------

def tiebroken(q_row, E):
    """q̃[e] = 2^L·q[e] + (E − 1 − e): distinct, lowest index wins a tie."""
    L = l_bits(E)
    return [(q << L) + (E - 1 - e) for e, q in enumerate(q_row)]


def routing_witness(s, b, k):
    """Honest witness for scores s (T rows of E ints) and bias b (E ints):
    m is the top-k of q̃ = 2^L(s + b) + bonus, τ the k-th largest q̃."""
    T, E = len(s), len(s[0])
    w = {"m": [], "qt": [], "tau": [], "d": [], "md": [], "v": []}
    for t in range(T):
        qt = tiebroken([s[t][e] + b[e] for e in range(E)], E)
        order = sorted(range(E), key=lambda e: -qt[e])
        sel = set(order[:k])
        tau = qt[order[k - 1]]
        m = [1 if e in sel else 0 for e in range(E)]
        d = [qt[e] - tau for e in range(E)]
        md = [m[e] * d[e] for e in range(E)]
        w["m"].append(m)
        w["qt"].append(qt)
        w["tau"].append(tau)
        w["d"].append(d)
        w["md"].append(md)
        w["v"].append([2 * md[e] - d[e] for e in range(E)])
    return w


def routing_failures(w, s, b, k, range_bits):
    """Names of the relations the witness breaks, every value taken mod P:
    F1 q̃ pin, F2 cardinality, Fd d = q̃ − τ, Fv v = 2·md − d, Q1 booleanity,
    Q2 md = m·d, R v ∈ [0, 2^R)."""
    T, E = len(s), len(s[0])
    L = l_bits(E)
    bad = set()
    for t in range(T):
        m, qt, d, md, v = (w[n][t] for n in ("m", "qt", "d", "md", "v"))
        tau = w["tau"][t]
        if sum(f(x) for x in m) % P != k % P:
            bad.add("F2")
        for e in range(E):
            if f(qt[e] - (s[t][e] << L) - (b[e] << L)) != (E - 1 - e):
                bad.add("F1")
            if f(d[e] - qt[e] + tau) != 0:
                bad.add("Fd")
            if f(v[e] - 2 * md[e] + d[e]) != 0:
                bad.add("Fv")
            if f(m[e] * m[e] - m[e]) != 0:
                bad.add("Q1")
            if f(m[e] * d[e] - md[e]) != 0:
                bad.add("Q2")
            if not f(v[e]) < (1 << range_bits):
                bad.add("R")
    return bad


# ---- slot masks (TopkSlotsClaim) -----------------------------------------

def slot_witness(m, s, k):
    """M_i[t,e] = 1 for the i-th selected expert of token t (ascending index,
    the canonical assignment; any assignment of the k experts to the k slots
    gives the same sums), and the slot scores ss[t][i] = s[t, e_i]."""
    T, E = len(m), len(m[0])
    M = [[[0] * E for _ in range(T)] for _ in range(k)]
    ss = [[0] * k for _ in range(T)]
    for t in range(T):
        sel = [e for e in range(E) if m[t][e]]
        for i, e in enumerate(sel):
            M[i][t][e] = 1
            ss[t][i] = s[t][e]
    return M, ss


def slot_failures(M, ss, m, s):
    """B booleanity, C one expert per slot row, S block sum Σ_i M_i = m,
    V slot score ss[t][i] = Σ_e M_i[t,e]·s[t,e]."""
    k, T, E = len(M), len(m), len(m[0])
    bad = set()
    for t in range(T):
        for e in range(E):
            if f(sum(M[i][t][e] for i in range(k)) - m[t][e]) != 0:
                bad.add("S")
        for i in range(k):
            if f(sum(M[i][t])) != 1:
                bad.add("C")
            if any(f(x * x - x) for x in M[i][t]):
                bad.add("B")
            if f(sum(M[i][t][e] * s[t][e] for e in range(E)) - ss[t][i]) != 0:
                bad.add("V")
    return bad


# ---- gate weights (GateBracketClaim) ---------------------------------------

def bracket_witness(ss, C):
    """Z[t] = Σ_i ss[t][i];  w = ⌊C·ss/Z⌋,  rem = C·ss − w·Z,  gr = Z − 1 − rem."""
    out = {"Z": [], "w": [], "rem": [], "gr": []}
    for row in ss:
        Z = sum(row)
        out["Z"].append(Z)
        out["w"].append([C * x // Z for x in row])
        out["rem"].append([C * x - (C * x // Z) * Z for x in row])
        out["gr"].append([Z - 1 - (C * x - (C * x // Z) * Z) for x in row])
    return out


def bracket_failures(br, ss, C, rem_bits, w_bits):
    """Zs the sum, D w·Z + rem = C·ss, G gr = Z − 1 − rem, and the ranges
    Rr (rem), Rg (gr) and Rw (w), every value mod P."""
    bad = set()
    for t, row in enumerate(ss):
        Z = br["Z"][t]
        if f(sum(row) - Z) != 0:
            bad.add("Zs")
        for i, x in enumerate(row):
            w, rem, gr = br["w"][t][i], br["rem"][t][i], br["gr"][t][i]
            if f(w * Z + rem - C * x) != 0:
                bad.add("D")
            if f(gr - Z + 1 + rem) != 0:
                bad.add("G")
            if not f(rem) < (1 << rem_bits):
                bad.add("Rr")
            if not f(gr) < (1 << rem_bits):
                bad.add("Rg")
            if not f(w) < (1 << w_bits):
                bad.add("Rw")
    return bad


# ---- the combine -----------------------------------------------------------

def combine(w, D, k, T):
    """y[t][j] = Σ_i w[t][i]·D[i·T + t][j]: the gate-weighted sum of the k
    slot outputs, before the one rescale."""
    F = len(D[0])
    return [[sum(w[t][i] * D[i * T + t][j] for i in range(k)) for j in range(F)]
            for t in range(T)]
