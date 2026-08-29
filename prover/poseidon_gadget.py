"""In-circuit Poseidon over Goldilocks — the cheap key commitment (B2).

Proves `Poseidon(key_material) == KEY_COMMIT` for committed key bytes and a
public 32-byte digest, composed entirely from claim types the Rust verifier
already handles: ConcatClaim, HadamardClaim, LinCombClaim and
EmbeddingLookupClaim (public-index gathers). No new verifier surface.

THE LAYOUT IS WHAT MAKES IT CHEAP. The naive shape -- one claim per (round,
position) -- would be ~360 S-boxes and hundreds of claims, no better than the
SHA-256 gadget it replaces. Instead the trace is committed POSITION-MAJOR: one
vector per state position, each holding that position's value across all
rounds. Then

  * the entire permutation's S-boxes are FOUR hadamards over one concatenated
    vector (x^7 = ((x^2)*x) squared, times x), not four per round;
  * the round constants are ONE LinComb with a per-slot public RHS;
  * the MDS layer is one LinComb per OUTPUT position (12 of them), each over
    the 12 position vectors -- because the matrix is the same every round, so
    rounds vectorize along the slot axis.

Claim count is therefore independent of the round count: raising ROUNDS makes
the vectors longer, not the circuit wider. That is why the reference uses 30
all-full rounds rather than the usual 8 full + 22 partial -- partial rounds buy
nothing here and cost uniformity.

Round r's MDS output is round r+1's input, so `Xnext` is a shifted view of the
same committed data. It is materialised by a public-index gather from the pool
rather than by committing a second copy, which is the difference between
binding the run and binding a copy of it.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "ref"))

import torch as _torch

import poseidon_gl as pg
from claims import EmbeddingLookupClaim
from tape import WitnessTensor

P = pg.P
T, R, RATE, CAP = pg.T, pg.ROUNDS, pg.RATE, pg.CAP
OUT_ELEMS = pg.OUT_ELEMS
BPE = pg.BYTES_PER_ELEM


def _commit(tape, name, vals):
    t = _torch.tensor([int(v) % P for v in vals], dtype=_torch.uint64, device="cuda")
    return tape.commit(name, t, (len(vals),))


def gather(tape, pool, name, ids):
    """Committed view of `pool` at PUBLIC indices. Every index here is a fixed
    slot of a fixed layout -- nothing about which wire goes where is secret,
    only the values."""
    gv = tape._alloc(name, len(ids))
    claim = EmbeddingLookupClaim(x=gv, E=pool.var, token_ids=list(ids), d=1)
    outs = tape._process_claim(claim, [pool.var])
    tape.claims.append(claim)
    return WitnessTensor(outs[gv] if outs else None, gv, (len(ids),), tape)


def poseidon_commit_gadget(tape, msg, key_bytes_vec, digest, *, name="ps",
                           _tamper=None, _force=False):
    """Prove Poseidon(msg) == `digest` where `msg` is the 40 key-material bytes.

    `msg` is those 40 bytes as a Python value (the prover knows them -- they are
    witness), used to build the trace. `key_bytes_vec` is the committed length-40
    WitnessTensor holding the same bytes in natural order, which the caller
    gathers from the AES gadget's own key and IV wires -- so the bytes hashed
    here are literally the bytes the cipher used, not a second commitment that
    merely happens to agree. `digest` is the public 32-byte commitment; it never
    becomes witness.

    Returns the committed initial-state message elements, for any further
    wiring the caller wants.
    """
    assert len(msg) == pg.MSG_BYTES, (
        "expected %d key-material bytes, got %d" % (pg.MSG_BYTES, len(msg)))
    assert key_bytes_vec.var.length == pg.MSG_BYTES, (
        "committed byte vector is %d long, expected %d"
        % (key_bytes_vec.var.length, pg.MSG_BYTES))
    # Fail fast on the honest path: an unsatisfiable statement should be a
    # clear error here, not a puzzling REJECT ten seconds later. Negative tests
    # opt out -- proving that an unsatisfiable statement really does REJECT is
    # the whole point of them.
    if not (_force or _tamper is not None):
        assert pg.hash_bytes(msg) == digest, (
            "msg does not hash to the public digest -- the prover cannot satisfy this")

    t = pg.trace(msg)
    if _tamper is not None:
        _tamper(t)

    # ---- committed trace, position-major -------------------------------
    X = [_commit(tape, f"{name}_x{j}", t["x"][j]) for j in range(T)]
    A = [_commit(tape, f"{name}_a{j}", t["a"][j]) for j in range(T)]
    S = [_commit(tape, f"{name}_s{j}", t["s"][j]) for j in range(T)]
    XR = _commit(tape, f"{name}_xR", t["xR"])

    XX = tape.concat(X, (T * R,))
    AA = tape.concat(A, (T * R,))
    SS = tape.concat(S, (T * R,))

    # ---- round constants: one LinComb, per-slot public RHS --------------
    rc_flat = [pg.RC[r][j] for j in range(T) for r in range(R)]   # matches concat order
    tape.lincomb([AA, XX], [1, -1], rc_flat)

    # ---- S-box: x^7 over the WHOLE permutation in four hadamards --------
    a2 = tape.hadamard(AA, AA)
    a3 = tape.hadamard(a2, AA)
    a6 = tape.hadamard(a3, a3)
    a7 = tape.hadamard(a6, AA)
    tape.lincomb([SS, a7], [1, -1], 0)

    # ---- MDS: one LinComb per output position ---------------------------
    # Round r's output at position i is round r+1's input, except for the last
    # round, whose output is the final state. Gather that shifted view rather
    # than committing it twice.
    pool = tape.concat([XX, XR], (T * R + T,))
    for i in range(T):
        ids = [i * R + (r + 1) for r in range(R - 1)] + [T * R + i]
        xnext = gather(tape, pool, f"{name}_xn{i}", ids)
        tape.lincomb([xnext] + S, [1] + [-pg.MDS[i][j] for j in range(T)], 0)

    # ---- inputs: message elements and the fixed capacity ----------------
    init = gather(tape, pool, f"{name}_init", [j * R for j in range(T)])
    msg_e = gather(tape, init, f"{name}_msg", list(range(RATE)))
    cap = gather(tape, init, f"{name}_cap", list(range(RATE, T)))
    tape.lincomb([cap], [1], [pg.MSG_BYTES] + [0] * (CAP - 1))

    # bridge: element k = sum_n 256^n * byte[BPE*k + n]
    bn = [gather(tape, key_bytes_vec, f"{name}_b{n}",
                 [BPE * k + n for k in range(RATE)]) for n in range(BPE)]
    tape.lincomb([msg_e] + bn, [1] + [-(256 ** n) for n in range(BPE)], 0)

    # ---- output: the public commitment ----------------------------------
    dig = [int.from_bytes(digest[8 * i:8 * i + 8], "little") for i in range(OUT_ELEMS)]
    out = gather(tape, XR, f"{name}_out", list(range(OUT_ELEMS)))
    tape.lincomb([out], [1], dig)

    return msg_e
