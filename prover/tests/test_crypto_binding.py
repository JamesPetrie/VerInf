"""End-to-end gate for the crypto binding (prover/crypto_binding.py).

Proves, on one tape and against the independent Rust verifier:

    Poseidon(key||iv_in||iv_out) == KEY_COMMIT  and
    AES128-CTR(key, iv_in,  req_tokens) == ct_in  and
    AES128-CTR(key, iv_out, rsp_tokens) == ct_out

over key material and ciphertexts built from VerInf's OWN reference cipher.
These tests deliberately do not import the interlock's wire app: the gadget is a
VerInf capability and must be testable from a bare checkout of this repo. The
wire side closes the loop from its end -- interlock/app/test_ilk_crypto.py gates
its AES against the `cryptography` library and its KEY_COMMIT against this same
Poseidon reference, so a divergence still fails somewhere.

The negative cases are the point. Three of them attack the public inputs
(wrong KEY_COMMIT / wrong ct byte per direction) and three attack the WIRING,
which is where a composed gadget actually goes wrong: a prover who runs a
perfectly valid AES under a key different from the one they hashed, a prover
who swaps a token, and a prover who reuses one direction's IV. Each must be
REJECTed by the verifier, not by a Python assert.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch  # noqa: F401

import core
import claims as _C        # noqa: F401
import packets as _PK      # noqa: F401
from tape import Tape
from _rust_verify import rust_verify_tape

import crypto_binding as cb
import token_binding as tb
import poseidon_gadget as pgad
import poseidon_gl as pg
from ref.token_recorder import serialize_tokens, aes128_ctr_gcm

# K_DEG = 2*ELL: blinding ON, as the demo runs it. The P1-P3 gates use
# K_DEG = ELL, where the slack is empty -- which is why the SHA gadget shipped
# a constraint that only held on the message slots (see shr_opt in
# token_binding.py). Keep this at 2*ELL so that class of bug cannot come back.
CFG = core.LigeroConfig(ELL=256, K_DEG=512, N_LIG=2048, T_QUERIES=4)
SEED = b"crypto-binding"

def _kdf(label, n):
    """Deterministic test key material. Not the wire's HKDF -- these tests are
    about the CIRCUIT, and only need bytes that are fixed run to run."""
    import hashlib
    out, i = b"", 0
    while len(out) < n:
        out += hashlib.sha256(label + bytes([i])).digest(); i += 1
    return out[:n]
REQ_TOKS = [1, 4013, 29871, 3186, 13]          # 5 tokens = 20 B = 2 blocks
RSP_TOKS = [3681, 338, 297, 3444, 29889, 2]    # 6 tokens = 24 B = 2 blocks


def _material():
    """Key material and ciphertexts, built from VerInf's OWN reference cipher.

    This deliberately does not import the interlock's wire crypto: the gadget is
    a VerInf capability and its tests must not depend on a sibling repo. The wire
    side proves the two agree from its end (interlock/app/test_ilk_crypto.py
    checks its AES against the `cryptography` library and its KEY_COMMIT against
    this same Poseidon reference)."""
    km = _kdf(b"crypto-binding-test", cb.KEYMAT_BYTES)
    key, iv_in, iv_out = km[:16], km[16:28], km[28:40]
    assert iv_in != iv_out, "per-direction IVs must differ"
    ct_in = aes128_ctr_gcm(key, iv_in, serialize_tokens(REQ_TOKS))
    ct_out = aes128_ctr_gcm(key, iv_out, serialize_tokens(RSP_TOKS))
    return km, key, iv_in, iv_out, ct_in, ct_out


def _streams(ct_in, ct_out, req=None, rsp=None):
    return [{"name": "in", "tokens": req or REQ_TOKS, "ct": ct_in},
            {"name": "out", "tokens": rsp or RSP_TOKS, "ct": ct_out}]


def _b2_keyview(tape, km, kc, tables, _force=False):
    """B2 over the key material, exactly as bind() builds it, returning the
    committed byte vector. The hand-built negatives below MUST go through this
    rather than a parallel construction -- a negative test that exercises a
    gadget production no longer uses proves nothing about production."""
    km_bytes = tb._commit(tape, "cb_km", list(km))
    tape.range_word(km_bytes, tables["byte"])
    pgad.poseidon_commit_gadget(tape, km, km_bytes, kc, _force=_force)
    return km_bytes


def _verify(tape, label, want_accept):
    t0 = time.time()
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    ok = acc == want_accept
    print(f"[{'OK ' if ok else 'XX '}] {label}: "
          f"verify={'ACCEPT' if acc else 'REJECT'} "
          f"(want {'ACCEPT' if want_accept else 'REJECT'}) "
          f"[{len(tape.claims)} claims, {time.time() - t0:.1f}s]", flush=True)
    return ok


def case_positive():
    km, key, iv_in, iv_out, ct_in, ct_out = _material()
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    cb.bind(tape, keymat=km, key_commit=pg.hash_bytes(km),
            streams=_streams(ct_in, ct_out))
    return _verify(tape, "positive: real wire key material + ciphertexts", True)


def case_public_cheat(label, mutate_kc=False, flip_in=None, flip_out=None):
    km, key, iv_in, iv_out, ct_in, ct_out = _material()
    kc = pg.hash_bytes(km)
    if mutate_kc:
        kc = bytes([kc[0] ^ 1]) + kc[1:]
    if flip_in is not None:
        ct_in = bytes([ct_in[flip_in] ^ 1]) + ct_in[1:] if flip_in == 0 else (
            ct_in[:flip_in] + bytes([ct_in[flip_in] ^ 1]) + ct_in[flip_in + 1:])
    if flip_out is not None:
        ct_out = (ct_out[:flip_out] + bytes([ct_out[flip_out] ^ 1])
                  + ct_out[flip_out + 1:])
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    cb.bind(tape, keymat=km, key_commit=pg.hash_bytes(km) if mutate_kc else kc,
            streams=_streams(ct_in, ct_out))
    # for the KEY_COMMIT cheat the gadget must be handed the LIE as public
    if mutate_kc:
        core._COSET_POWERS_K_CACHE.clear()
        tape = Tape(CFG, lazy=True)
        tables = tb.register_binding_tables(tape, with_xor=True)
        _b2_keyview(tape, km, kc, tables, _force=True)
    return _verify(tape, label, False)


def case_swapped_token():
    """A prover who commits a different request token than the ciphertext holds."""
    km, key, iv_in, iv_out, ct_in, ct_out = _material()
    bad = list(REQ_TOKS); bad[2] += 1
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    cb.bind(tape, keymat=km, key_commit=pg.hash_bytes(km),
            streams=_streams(ct_in, ct_out, req=bad))
    return _verify(tape, "cheat: swapped a request token", False)


def case_desynced_key():
    """THE wiring test. The prover runs a self-consistent AES under key', and
    publishes ct' = AES(key', tokens) so B1 is internally satisfiable -- but
    hashes the REAL key material for B2. Only the key pins catch this."""
    km, key, iv_in, iv_out, _, _ = _material()
    key2 = bytes([key[0] ^ 0xFF]) + key[1:]
    ct_in2 = aes128_ctr_gcm(key2, iv_in, serialize_tokens(REQ_TOKS))
    ct_out2 = aes128_ctr_gcm(key2, iv_out, serialize_tokens(RSP_TOKS))

    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    tables = tb.register_binding_tables(tape, with_xor=True)
    # B2 over the honest key material
    km_bytes = _b2_keyview(tape, km, pg.hash_bytes(km), tables)
    g_key = cb.pool_gather(tape, km_bytes, "cb_km_key", list(range(16)))
    # B1 under key2, then the honest pin -- which must now fail
    for nm, iv, toks, ct in (("in", iv_in, REQ_TOKS, ct_in2),
                             ("out", iv_out, RSP_TOKS, ct_out2)):
        exp = {}
        tb.aes_ctr_gadget(tape, tables, key2, iv, serialize_tokens(toks),
                          ct_public=ct, export=exp)
        a_key = exp["gather"]("cb_%s_key" % nm, [("rk", (0, j)) for j in range(16)])
        cb._pin_equal(tape, g_key, a_key)
    return _verify(tape, "cheat: AES key differs from the hashed key material", False)


def case_iv_reuse():
    """Both directions encrypted under iv_in: keystream reuse, and the iv pin
    for the response direction no longer matches the committed key material."""
    km, key, iv_in, iv_out, ct_in, _ = _material()
    ct_out_bad = aes128_ctr_gcm(key, iv_in, serialize_tokens(RSP_TOKS))
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    tables = tb.register_binding_tables(tape, with_xor=True)
    km_bytes = _b2_keyview(tape, km, pg.hash_bytes(km), tables)
    g_ivout = cb.pool_gather(tape, km_bytes, "cb_km_ivout", list(range(28, 40)))
    exp = {}
    tb.aes_ctr_gadget(tape, tables, key, iv_in, serialize_tokens(RSP_TOKS),
                      ct_public=ct_out_bad, export=exp)
    a_iv = exp["gather"]("cb_out_iv", [("iv_bytes", (j,)) for j in range(12)])
    cb._pin_equal(tape, g_ivout, a_iv)
    return _verify(tape, "cheat: response reuses the request IV", False)


def _weld_case(label, model_tokens, want_accept):
    """The weld, in isolation from the 22-layer model.

    `model_tokens` stands in for MaxClaim's committed output-token vector: the
    real one is length SEQ and holds the token observed at each position, and
    the response occupies a slice of it. The binding gathers that slice and pins
    it to the AES plaintext tokens. If a prover decrypts to one stream and scores
    another, this is the claim that has to notice."""
    km, key, iv_in, iv_out, ct_in, ct_out = _material()
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    res = cb.bind(tape, keymat=km, key_commit=pg.hash_bytes(km),
                  streams=_streams(ct_in, ct_out), reveal_tokens={"in"})
    # SEQ-shaped stand-in for the model's committed tokens; the response sits at
    # positions [2, 2+len(RSP_TOKS)) exactly as sum_positions selects it.
    seq_tok = tb._commit(tape, "fake_ui_tok", [7, 9] + list(model_tokens) + [11])
    g = cb.pool_gather(tape, seq_tok, "weld",
                       list(range(2, 2 + len(RSP_TOKS))))
    cb.bind_tokens_to(tape, res["tok"]["out"], g)
    return _verify(tape, label, want_accept)


def main():
    results = [
        case_positive(),
        case_public_cheat("cheat: wrong KEY_COMMIT", mutate_kc=True),
        case_public_cheat("cheat: flipped request ciphertext byte", flip_in=3),
        case_public_cheat("cheat: flipped response ciphertext byte", flip_out=17),
        case_swapped_token(),
        case_desynced_key(),
        case_iv_reuse(),
        _weld_case("weld: model tokens == decrypted tokens", RSP_TOKS, True),
        _weld_case("cheat: model scored a different response stream",
                   [RSP_TOKS[0], RSP_TOKS[1] + 1] + RSP_TOKS[2:], False),
    ]
    fails = results.count(False)
    print(f"=== crypto binding: {len(results) - fails}/{len(results)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
