"""SHA-256 constraint-system reference: witness trace + constraint checker
(token-binding P2, analysis/token-binding.md §12; paper Appendix E.5).

This file freezes the CONSTRAINT LAYOUT of the SHA-256 gadget before the
circuit exists: `trace()` computes every value the gadget commits,
`check_constraints()` evaluates every constraint it emits (exact integer
arithmetic, all residuals must be zero), and the tests difftest the digest
against hashlib and fuzz single-value tampers to show the system has no
unconstrained slots. The tape gadget (token_binding.py) and its claims are
composed against THIS layout; every committed class below maps 1:1 to a
committed variable, and every `req` name to an emitted constraint family.

Single-block scope (the B2 = SHA256(key material) = H2 slice; message length
a multiple of 4 and < 56, so 40-byte key material -> exactly one padded
block). Multi-block chaining lands with the H1 streams in P3.

Arithmetization (bit-level, vectorized over rounds; one block). Everything
is either PUBLIC (the IV prefixes, the padding words, the round constants K,
the digest H2) or a COMMITTED class listed here:

  * State histories A[0..68) and E[0..68): 4 public prefix slots
    (A[0..4) = H0[3..0] reversed, E[0..4) = H0[7..4] reversed), then
    A[4+r]/E[4+r] are round r's outputs. Round r reads a=A[3+r], b=A[2+r],
    c=A[1+r], d=A[r], e=E[3+r], f=E[2+r], g=E[1+r], h=E[r] — index shifts
    realized on the tape as ConcatClaim partition views of the same
    committed spine, never as independent copies.
  * Message: the key-material bytes committed as FOUR byte-position stride
    vectors msg_stride[k][i] = byte 4i+k (range-checked); word i of the
    schedule recomposes big-endian as sum_k stride[k][i]*2^(8(3-k)). The
    padding words W[len/4 .. 16) are pinned to public constants and the
    padding bytes are never witness.
  * Bit decompositions, one class per word ROLE (each: booleanity via the
    [0,2) range table + one recomposition linear): a_bit/b_bit/c_bit of
    A[3+r]/A[2+r]/A[1+r]; e_bit/f_bit/g_bit of E[3+r]/E[2+r]/E[1+r]
    (all [64][32]); w_bit_m of W[1..49) ([48][32], the ssig0 operands);
    w_bit_n of W[14..64) ([50][32], the ssig1 operands PLUS the range pin
    for the schedule outputs W[62], W[63] — every W word is range-pinned by
    exactly one of: byte recomposition (< len/4), public padding
    ([len/4,16)), or an n-class bit recomposition ([16,64) subset of
    [14,64))).
  * XOR intermediates (committed because hadamard operands must be
    committed vars): x12 = rot6(e)^rot11(e), y12 = rot2(a)^rot13(a),
    tbc = b^c (all [64][32]); xm = rot7(w)^rot18(w) over the m range
    ([48][32]); xn = rot17(w)^rot19(w) over the n PRODUCT range
    [14..62) ([48][32]). Each is pinned by the linear x = u + v - 2*prod.
  * Pair products (committed by raw hadamard; products of booleans need no
    range check): p1 = r6*r11, p2 = x12*r25 (Sigma1); q1 = r2*r13,
    q2 = y12*r22 (Sigma0); ef = e*f, eg = e*g (Ch = ef + g - eg, disjoint
    terms); bc = b*c, u = a*tbc (Maj = bc + u) — all [64][32]; m1, m2
    (ssig0) and n1, n2 (ssig1) over their operand ranges ([48][32] each;
    committing products outside the constrained ranges would create free
    slots — the fuzz caught exactly that).
  * Carries: ce, ca in [0,8) per round; cw in [0,4) per schedule round;
    cd in [0,2) per digest word — all range-table checked. Sigma/Ch/Maj
    values are never committed: they fold into the round-add LinComb rows
    as linear expressions over bits and products.
  * The digest is PUBLIC (it is H2): out[i] never exists as witness; the
    digest constraint is 2^32*cd[i] = H0[i] + final_i - H2word_i with only
    cd and the finals committed (finals A[64..68)/E[64..68) are range-pinned
    by the a/e-class bit decompositions of rounds 60..63).
"""
import hashlib

MOD32 = 1 << 32

K = [int(x, 16) for x in (
    "428a2f98 71374491 b5c0fbcf e9b5dba5 3956c25b 59f111f1 923f82a4 ab1c5ed5"
    " d807aa98 12835b01 243185be 550c7dc3 72be5d74 80deb1fe 9bdc06a7 c19bf174"
    " e49b69c1 efbe4786 0fc19dc6 240ca1cc 2de92c6f 4a7484aa 5cb0a9dc 76f988da"
    " 983e5152 a831c66d b00327c8 bf597fc7 c6e00bf3 d5a79147 06ca6351 14292967"
    " 27b70a85 2e1b2138 4d2c6dfc 53380d13 650a7354 766a0abb 81c2c92e 92722c85"
    " a2bfe8a1 a81a664b c24b8b70 c76c51a3 d192e819 d6990624 f40e3585 106aa070"
    " 19a4c116 1e376c08 2748774c 34b0bcb5 391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3"
    " 748f82ee 78a5636f 84c87814 8cc70208 90befffa a4506ceb bef9a3f7 c67178f2"
).split()]

H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

M_LO, M_HI = 1, 49            # ssig0 operand words (m classes)
N_LO, N_HI = 14, 62           # ssig1 PRODUCT words (n products)
NB_HI = 64                    # n-class BIT coverage extends to 64 (range pins)


def _rotr(x, k):
    return ((x >> k) | (x << (32 - k))) & 0xFFFFFFFF


def pad_one_block(msg):
    """Pad a message of < 56 bytes to one 64-byte block."""
    assert len(msg) < 56, "single-block scope: message must fit one block"
    return msg + b"\x80" + bytes(55 - len(msg)) + (8 * len(msg)).to_bytes(8, "big")


def _bits(x):
    return [(x >> j) & 1 for j in range(32)]


def trace(msg):
    """Full witness trace for SHA256(msg), msg length a multiple of 4 and
    < 56 bytes. Every entry is a committed class of the gadget (or a public
    constant recorded for the checker)."""
    assert len(msg) % 4 == 0 and len(msg) < 56
    block = pad_one_block(msg)
    n_words = len(msg) // 4
    t = {"msg_len": len(msg)}
    # message bytes as byte-position strides (the committed form)
    t["msg_stride"] = [[msg[4 * i + k] for i in range(n_words)] for k in range(4)]

    # schedule
    W = [int.from_bytes(block[4 * i:4 * i + 4], "big") for i in range(16)]
    cw = []
    for r in range(16, 64):
        s0 = _rotr(W[r - 15], 7) ^ _rotr(W[r - 15], 18) ^ (W[r - 15] >> 3)
        s1 = _rotr(W[r - 2], 17) ^ _rotr(W[r - 2], 19) ^ (W[r - 2] >> 10)
        full = s1 + W[r - 7] + s0 + W[r - 16]
        cw.append(full // MOD32)
        W.append(full % MOD32)
    t["W"], t["cw"] = W, cw

    # state histories with public 4-slot prefixes
    A = [H0[3], H0[2], H0[1], H0[0]]
    E = [H0[7], H0[6], H0[5], H0[4]]
    ce, ca = [], []
    for r in range(64):
        a, b, c, d = A[3 + r], A[2 + r], A[1 + r], A[r]
        e, f, g, h = E[3 + r], E[2 + r], E[1 + r], E[r]
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & g & 0xFFFFFFFF)
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t1 = h + S1 + ch + K[r] + W[r]
        e_full = d + t1
        ce.append(e_full // MOD32)
        E.append(e_full % MOD32)
        a_full = t1 + S0 + maj
        ca.append(a_full // MOD32)
        A.append(a_full % MOD32)
    t["A"], t["E"], t["ce"], t["ca"] = A, E, ce, ca

    # bit classes, one per word role
    t["a_bit"] = [_bits(A[3 + r]) for r in range(64)]
    t["b_bit"] = [_bits(A[2 + r]) for r in range(64)]
    t["c_bit"] = [_bits(A[1 + r]) for r in range(64)]
    t["e_bit"] = [_bits(E[3 + r]) for r in range(64)]
    t["f_bit"] = [_bits(E[2 + r]) for r in range(64)]
    t["g_bit"] = [_bits(E[1 + r]) for r in range(64)]
    t["w_bit_m"] = [_bits(W[w]) for w in range(M_LO, M_HI)]
    t["w_bit_n"] = [_bits(W[w]) for w in range(N_LO, NB_HI)]

    # xor intermediates + pair products
    cls = {k: [] for k in ("x12", "y12", "tbc", "p1", "p2", "q1", "q2",
                           "ef", "eg", "bc", "u")}
    for r in range(64):
        a_b, b_b, c_b = t["a_bit"][r], t["b_bit"][r], t["c_bit"][r]
        e_b, f_b, g_b = t["e_bit"][r], t["f_bit"][r], t["g_bit"][r]
        row = {k: [] for k in cls}
        for j in range(32):
            r6, r11, r25 = e_b[(j + 6) % 32], e_b[(j + 11) % 32], e_b[(j + 25) % 32]
            row["p1"].append(r6 * r11)
            row["x12"].append(r6 ^ r11)
            row["p2"].append((r6 ^ r11) * r25)
            r2, r13, r22 = a_b[(j + 2) % 32], a_b[(j + 13) % 32], a_b[(j + 22) % 32]
            row["q1"].append(r2 * r13)
            row["y12"].append(r2 ^ r13)
            row["q2"].append((r2 ^ r13) * r22)
            row["ef"].append(e_b[j] * f_b[j])
            row["eg"].append(e_b[j] * g_b[j])
            row["bc"].append(b_b[j] * c_b[j])
            row["tbc"].append(b_b[j] ^ c_b[j])
            row["u"].append(a_b[j] * (b_b[j] ^ c_b[j]))
        for k in cls:
            cls[k].append(row[k])
    t.update(cls)

    sched = {k: [] for k in ("xm", "m1", "m2", "xn", "n1", "n2")}
    for w in range(M_LO, M_HI):
        w_b = t["w_bit_m"][w - M_LO]
        xm, m1, m2 = [], [], []
        for j in range(32):
            s7, s18 = w_b[(j + 7) % 32], w_b[(j + 18) % 32]
            h3 = w_b[j + 3] if j + 3 < 32 else 0
            m1.append(s7 * s18)
            xm.append(s7 ^ s18)
            m2.append((s7 ^ s18) * h3)
        sched["xm"].append(xm)
        sched["m1"].append(m1)
        sched["m2"].append(m2)
    for w in range(N_LO, N_HI):
        w_b = t["w_bit_n"][w - N_LO]
        xn, n1, n2 = [], [], []
        for j in range(32):
            s17, s19 = w_b[(j + 17) % 32], w_b[(j + 19) % 32]
            h10 = w_b[j + 10] if j + 10 < 32 else 0
            n1.append(s17 * s19)
            xn.append(s17 ^ s19)
            n2.append((s17 ^ s19) * h10)
        sched["xn"].append(xn)
        sched["n1"].append(n1)
        sched["n2"].append(n2)
    t.update(sched)

    # digest carries; the digest itself is public
    finals = [A[67], A[66], A[65], A[64], E[67], E[66], E[65], E[64]]
    out, cd = [], []
    for i in range(8):
        s = H0[i] + finals[i]
        cd.append(s // MOD32)
        out.append(s % MOD32)
    t["cd"] = cd
    t["digest"] = b"".join(x.to_bytes(4, "big") for x in out)
    return t


# ------------------------------------------------------------- constraints

def check_constraints(t, digest=None):
    """Evaluate every gadget constraint on a trace. Returns (name, index,
    residual) for every NON-ZERO residual; [] == satisfied. `digest` is the
    PUBLIC digest the gadget pins (H2); defaults to the trace's own."""
    bad = []
    digest = digest if digest is not None else t["digest"]
    dwords = [int.from_bytes(digest[4 * i:4 * i + 4], "big") for i in range(8)]

    def req(name, idx, lhs, rhs=0):
        if lhs != rhs:
            bad.append((name, idx, lhs - rhs))

    A, E, W = t["A"], t["E"], t["W"]
    n_words = t["msg_len"] // 4
    xor2 = lambda x, y, p: x + y - 2 * p

    for i in range(4):
        req("A_prefix", i, A[i], [H0[3], H0[2], H0[1], H0[0]][i])
        req("E_prefix", i, E[i], [H0[7], H0[6], H0[5], H0[4]][i])

    # message: byte range + BE word recomposition from strides; public pad
    for k in range(4):
        for i in range(n_words):
            v = t["msg_stride"][k][i]
            req("msg_byte_range", (k, i), int(not (0 <= v < 256)))
    for i in range(n_words):
        word = sum(t["msg_stride"][k][i] << (8 * (3 - k)) for k in range(4))
        req("W_msg_recompose", i, W[i], word)
    pad_block = pad_one_block(b"\x00" * t["msg_len"])   # padding bytes only
    for i in range(n_words, 16):
        pub = int.from_bytes(pad_block[4 * i:4 * i + 4], "big")
        req("W_pad_public", i, W[i], pub)

    # bit classes: booleanity + recomposition against the sliced words
    roles = [("a_bit", A, 3), ("b_bit", A, 2), ("c_bit", A, 1),
             ("e_bit", E, 3), ("f_bit", E, 2), ("g_bit", E, 1)]
    for name, hist, off in roles:
        for r in range(64):
            bits = t[name][r]
            for j in range(32):
                req(name + "_bool", (r, j), bits[j] * bits[j] - bits[j])
            req(name + "_recompose", r, hist[off + r],
                sum(bits[j] << j for j in range(32)))
    for w in range(M_LO, M_HI):
        bits = t["w_bit_m"][w - M_LO]
        for j in range(32):
            req("w_bit_m_bool", (w, j), bits[j] * bits[j] - bits[j])
        req("w_bit_m_recompose", w, W[w], sum(bits[j] << j for j in range(32)))
    for w in range(N_LO, NB_HI):
        bits = t["w_bit_n"][w - N_LO]
        for j in range(32):
            req("w_bit_n_bool", (w, j), bits[j] * bits[j] - bits[j])
        req("w_bit_n_recompose", w, W[w], sum(bits[j] << j for j in range(32)))

    # per-round products, xor pins, and the two carry-checked adds
    for r in range(64):
        a_b, b_b, c_b = t["a_bit"][r], t["b_bit"][r], t["c_bit"][r]
        e_b, f_b, g_b = t["e_bit"][r], t["f_bit"][r], t["g_bit"][r]
        S1 = S0 = ch = maj = 0
        for j in range(32):
            r6, r11, r25 = e_b[(j + 6) % 32], e_b[(j + 11) % 32], e_b[(j + 25) % 32]
            req("p1_quad", (r, j), t["p1"][r][j], r6 * r11)
            req("x12_pin", (r, j), t["x12"][r][j], xor2(r6, r11, t["p1"][r][j]))
            req("p2_quad", (r, j), t["p2"][r][j], t["x12"][r][j] * r25)
            S1 += xor2(t["x12"][r][j], r25, t["p2"][r][j]) << j
            r2, r13, r22 = a_b[(j + 2) % 32], a_b[(j + 13) % 32], a_b[(j + 22) % 32]
            req("q1_quad", (r, j), t["q1"][r][j], r2 * r13)
            req("y12_pin", (r, j), t["y12"][r][j], xor2(r2, r13, t["q1"][r][j]))
            req("q2_quad", (r, j), t["q2"][r][j], t["y12"][r][j] * r22)
            S0 += xor2(t["y12"][r][j], r22, t["q2"][r][j]) << j
            req("ef_quad", (r, j), t["ef"][r][j], e_b[j] * f_b[j])
            req("eg_quad", (r, j), t["eg"][r][j], e_b[j] * g_b[j])
            ch += (t["ef"][r][j] + g_b[j] - t["eg"][r][j]) << j
            req("bc_quad", (r, j), t["bc"][r][j], b_b[j] * c_b[j])
            req("tbc_pin", (r, j), t["tbc"][r][j], xor2(b_b[j], c_b[j], t["bc"][r][j]))
            req("u_quad", (r, j), t["u"][r][j], a_b[j] * t["tbc"][r][j])
            maj += (t["bc"][r][j] + t["u"][r][j]) << j
        t1 = E[r] + S1 + ch + K[r] + W[r]
        req("e_add", r, E[4 + r] + MOD32 * t["ce"][r], A[r] + t1)
        req("a_add", r, A[4 + r] + MOD32 * t["ca"][r], t1 + S0 + maj)
        req("ce_range", r, int(not (0 <= t["ce"][r] < 8)))
        req("ca_range", r, int(not (0 <= t["ca"][r] < 8)))

    # schedule: products, xor pins, carry-checked add (aligned at r-16)
    for r in range(16, 64):
        i = r - 16
        wbm, wbn = t["w_bit_m"][(r - 15) - M_LO], t["w_bit_n"][(r - 2) - N_LO]
        s0 = s1 = 0
        for j in range(32):
            s7, s18 = wbm[(j + 7) % 32], wbm[(j + 18) % 32]
            h3 = wbm[j + 3] if j + 3 < 32 else 0
            req("m1_quad", (r, j), t["m1"][i][j], s7 * s18)
            req("xm_pin", (r, j), t["xm"][i][j], xor2(s7, s18, t["m1"][i][j]))
            req("m2_quad", (r, j), t["m2"][i][j], t["xm"][i][j] * h3)
            s0 += xor2(t["xm"][i][j], h3, t["m2"][i][j]) << j
            s17, s19 = wbn[(j + 17) % 32], wbn[(j + 19) % 32]
            h10 = wbn[j + 10] if j + 10 < 32 else 0
            req("n1_quad", (r, j), t["n1"][i][j], s17 * s19)
            req("xn_pin", (r, j), t["xn"][i][j], xor2(s17, s19, t["n1"][i][j]))
            req("n2_quad", (r, j), t["n2"][i][j], t["xn"][i][j] * h10)
            s1 += xor2(t["xn"][i][j], h10, t["n2"][i][j]) << j
        req("w_schedule", r, W[r] + MOD32 * t["cw"][i],
            s1 + W[r - 7] + s0 + W[r - 16])
        req("cw_range", r, int(not (0 <= t["cw"][i] < 4)))

    # digest: PUBLIC words; only the carries and finals are witness
    finals = [A[67], A[66], A[65], A[64], E[67], E[66], E[65], E[64]]
    for i in range(8):
        req("digest_add", i, MOD32 * t["cd"][i], H0[i] + finals[i] - dwords[i])
        req("cd_range", i, int(not (0 <= t["cd"][i] < 2)))
    return bad


def sha256_one_block(msg):
    """Reference entry point: trace + checker + hashlib cross-check."""
    t = trace(msg)
    bad = check_constraints(t)
    assert not bad, f"constraint system unsatisfied on honest trace: {bad[:5]}"
    assert t["digest"] == hashlib.sha256(msg).digest(), "digest mismatch"
    return t
