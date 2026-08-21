"""Poseidon commitment gadget, end to end against the Rust verifier.

Positive: the real key material and its real digest ACCEPT. Negatives probe
each committed class and the public digest -- a class the circuit fails to
constrain would let a prover vary it freely while keeping KEY_COMMIT fixed,
which is exactly the freedom the binding exists to remove.

Runs with blinding ON (K_DEG = 2*ELL), the configuration the demo proves under
and the one that caught the SHA gadget's constant-zero bug.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ref"))
import torch  # noqa: F401

import core
import claims as _C        # noqa: F401
import packets as _PK      # noqa: F401
from tape import Tape
from _rust_verify import rust_verify_tape

import poseidon_gl as pg
import poseidon_gadget as pgad

CFG = core.LigeroConfig(ELL=256, K_DEG=512, N_LIG=2048, T_QUERIES=4)
SEED = b"poseidon-gadget"


def _material():
    """40 bytes of key material. Built here rather than imported from the wire
    app -- the gadget is a VerInf capability and its tests stay self-contained."""
    import hashlib
    out, i = b"", 0
    while len(out) < pg.MSG_BYTES:
        out += hashlib.sha256(b"poseidon-gadget-test" + bytes([i])).digest(); i += 1
    return out[:pg.MSG_BYTES]


def _build(km, digest, _tamper=None, _force=False):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    kb = pgad._commit(tape, "kb", list(km))
    pgad.poseidon_commit_gadget(tape, km, kb, digest, _tamper=_tamper, _force=_force)
    return tape


def run(label, km, digest, want_accept, _tamper=None, _force=False):
    t0 = time.time()
    tape = _build(km, digest, _tamper, _force)
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    ok = acc == want_accept
    print("[%s] %s: verify=%s (want %s) [%d claims, %.1fs]"
          % ("OK " if ok else "XX ", label, "ACCEPT" if acc else "REJECT",
             "ACCEPT" if want_accept else "REJECT", len(tape.claims),
             time.time() - t0), flush=True)
    return ok


def main():
    km = _material()
    dig = pg.hash_bytes(km)
    bad_dig = bytes([dig[0] ^ 1]) + dig[1:]

    results = [
        run("positive: real key material + its digest", km, dig, True),
        # public digest lies
        run("cheat: wrong public digest", km, bad_dig, False, _force=True),
        # each committed class
        run("cheat: tampered round input x", km, dig, False,
            _tamper=lambda t: t["x"][3].__setitem__(7, (t["x"][3][7] + 1) % pg.P)),
        run("cheat: tampered post-RC value a", km, dig, False,
            _tamper=lambda t: t["a"][5].__setitem__(2, (t["a"][5][2] + 1) % pg.P)),
        run("cheat: tampered S-box output s", km, dig, False,
            _tamper=lambda t: t["s"][0].__setitem__(11, (t["s"][0][11] + 1) % pg.P)),
        run("cheat: tampered final state", km, dig, False,
            _tamper=lambda t: t["xR"].__setitem__(6, (t["xR"][6] + 1) % pg.P)),
        # the capacity must be pinned: a prover who could vary it would get a
        # free 3-element degree of freedom inside the hash preimage
        run("cheat: nonzero capacity slot", km, dig, False,
            _tamper=lambda t: t["x"][pg.RATE + 1].__setitem__(0, 12345)),
    ]
    fails = results.count(False)
    print("=== poseidon gadget: %d/%d %s ==="
          % (len(results) - fails, len(results), "PASS" if not fails else "FAIL"))
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
