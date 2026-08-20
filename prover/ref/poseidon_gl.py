"""Poseidon over Goldilocks — the reference the circuit is composed against.

Same role for the key commitment that `token_recorder.py` plays for AES and
`sha256_trace.py` for SHA-256: this module IS the spec. It is pure Python with
no torch and no third-party dependency, so both ends of the wire
(`interlock/app/ilk_crypto.py`, including the Raspberry Pi) and the prover can
import the identical implementation. One implementation, or the wire and the
circuit drift.

WHY THIS EXISTS. The key commitment used to be SHA-256, which cost ~950 claims
and ~8.7 s of prove — 92% of the whole crypto binding. SHA-256 is bit-oriented:
proving it in a prime field means decomposing every word into booleans, so it
produces hundreds of tiny committed vectors, each occupying (and paying a full
NTT and four sweeps for) an entire ELL-wide row it barely fills. Poseidon is
built from the operations the proof system already speaks — multiply and add
whole field elements — so the same statement costs tens of claims over full rows.

PARAMETERS (demo-grade — read this before quoting security numbers)
    field    Goldilocks, p = 2^64 - 2^32 + 1
    width    t = 12
    S-box    x^7   (gcd(7, p-1) = 1, since p-1 = 2^32 * 3 * 5 * 17 * 257 * 65537,
                    so x -> x^7 is a bijection)
    rounds   R = 30, ALL FULL. The standard Goldilocks parameter set is 8 full +
             22 partial; partial rounds exist to save constraints in R1CS-style
             systems, where each S-box is its own constraint. Here the whole
             permutation's S-boxes are proven by FOUR hadamard claims over one
             concatenated vector, so a partial round saves nothing and costs
             uniformity. All-full at the same total round count is strictly
             stronger and simpler to constrain.
    MDS      Cauchy: M[i][j] = 1/(x_i + y_j), x_i = i+1, y_j = t+j+1. A Cauchy
             matrix with distinct x_i, distinct y_j and x_i + y_j != 0 is
             provably MDS (every square submatrix is non-singular), which is the
             property the wide-trail argument needs.
    RC       SHA-256(domain || round || pos) mod p — nothing-up-my-sleeve.

    This is a self-consistent instantiation, NOT a published parameter set with
    third-party test vectors. It is appropriate for the demo, where it replaces
    a hash whose only job is to bind a key that both endpoints already share.
    A deployment that needs an auditable hash should swap in a standard
    Goldilocks Poseidon (e.g. Plonky2's) with its published constants and KATs;
    only `RC`, `MDS` and `R` would change, not the gadget or the wire format.

USE. One permutation, fixed-length compression — not a variable-length sponge:
    state = [ m_0 .. m_7 | LEN | 0 | 0 | 0 ]     8 message elements, 4 capacity
    permute
    commitment = state[0:4]                       4 elements = 32 bytes
The 4-element capacity is never touched by input, so collisions cost ~2^128
generically. The input is a fixed 40 bytes packed 5 bytes/element, so there is
no padding ambiguity to get wrong.
"""
import hashlib

P = (1 << 64) - (1 << 32) + 1
T = 12                      # state width
RATE = 8                    # message elements per permutation
CAP = T - RATE              # capacity elements
ROUNDS = 30                 # all full
ALPHA = 7
BYTES_PER_ELEM = 5          # 40 bits < 64, no reduction ambiguity
MSG_BYTES = RATE * BYTES_PER_ELEM      # 40 = key(16) || iv_in(12) || iv_out(12)
OUT_ELEMS = 4
DOMAIN = b"interlock-poseidon-gl-v1"


def _inv(a):
    return pow(a % P, P - 2, P)


def _mds():
    """Cauchy MDS."""
    xs = [i + 1 for i in range(T)]
    ys = [T + j + 1 for j in range(T)]
    return [[_inv(xs[i] + ys[j]) for j in range(T)] for i in range(T)]


MDS = _mds()


def _round_constants():
    rc = []
    for r in range(ROUNDS):
        row = []
        for j in range(T):
            h = hashlib.sha256(DOMAIN + b"rc" + r.to_bytes(4, "big")
                               + j.to_bytes(4, "big")).digest()
            row.append(int.from_bytes(h[:16], "big") % P)
        rc.append(row)
    return rc


RC = _round_constants()


def sbox(x):
    return pow(x % P, ALPHA, P)


def permute(state):
    """One full permutation. Returns the new state (length T)."""
    assert len(state) == T
    s = [v % P for v in state]
    for r in range(ROUNDS):
        a = [(s[j] + RC[r][j]) % P for j in range(T)]
        b = [sbox(v) for v in a]
        s = [sum(MDS[i][j] * b[j] for j in range(T)) % P for i in range(T)]
    return s


def pack(msg: bytes):
    """40 bytes -> 8 field elements, 5 little-endian bytes each."""
    if len(msg) != MSG_BYTES:
        raise ValueError("message must be %d bytes, got %d" % (MSG_BYTES, len(msg)))
    return [int.from_bytes(msg[BYTES_PER_ELEM * k:BYTES_PER_ELEM * (k + 1)], "little")
            for k in range(RATE)]


def hash_bytes(msg: bytes) -> bytes:
    """The key commitment: 40 bytes -> 32 bytes."""
    state = pack(msg) + [MSG_BYTES] + [0] * (CAP - 1)
    out = permute(state)[:OUT_ELEMS]
    return b"".join(v.to_bytes(8, "little") for v in out)


def digest_elems(msg: bytes):
    """The commitment as field elements (what the circuit pins)."""
    state = pack(msg) + [MSG_BYTES] + [0] * (CAP - 1)
    return permute(state)[:OUT_ELEMS]


# ------------------------------------------------------------------ trace

def trace(msg: bytes):
    """Full witness trace, in the layout the gadget commits.

    Position-major vectors of length ROUNDS, one per state position j:
        x[j][r]  state entering round r
        a[j][r]  x + round constant
        s[j][r]  a^7
    plus `xR` — the final state (length T) — because round r's OUTPUT is round
    r+1's input for r < R-1, and the last round's output has nowhere else to
    live. Keeping it separate is what lets the MDS constraint be one aligned
    LinComb per output position instead of a shifted view per round.
    """
    st = pack(msg) + [MSG_BYTES] + [0] * (CAP - 1)
    x = [[0] * ROUNDS for _ in range(T)]
    a = [[0] * ROUNDS for _ in range(T)]
    s = [[0] * ROUNDS for _ in range(T)]
    for r in range(ROUNDS):
        for j in range(T):
            x[j][r] = st[j]
            a[j][r] = (st[j] + RC[r][j]) % P
            s[j][r] = sbox(a[j][r])
        st = [sum(MDS[i][j] * s[j][r] for j in range(T)) % P for i in range(T)]
    return {"x": x, "a": a, "s": s, "xR": st, "msg_elems": pack(msg),
            "digest": st[:OUT_ELEMS]}


def check_constraints(t):
    """Independent re-derivation: every relation the gadget will assert.
    Returns a list of failures (empty == consistent)."""
    bad = []
    x, a, s, xR = t["x"], t["a"], t["s"], t["xR"]
    for r in range(ROUNDS):
        for j in range(T):
            if a[j][r] != (x[j][r] + RC[r][j]) % P:
                bad.append(("rc", r, j))
            if s[j][r] != pow(a[j][r], ALPHA, P):
                bad.append(("sbox", r, j))
        for i in range(T):
            want = sum(MDS[i][j] * s[j][r] for j in range(T)) % P
            got = x[i][r + 1] if r + 1 < ROUNDS else xR[i]
            if got != want:
                bad.append(("mds", r, i))
    for k, v in enumerate(t["msg_elems"]):
        if x[k][0] != v:
            bad.append(("input", k))
    for k in range(CAP):
        want = MSG_BYTES if k == 0 else 0
        if x[RATE + k][0] != want:
            bad.append(("capacity", k))
    return bad
