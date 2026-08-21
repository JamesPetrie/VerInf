"""Crypto binding: a pre-committed key opens the certified request and response.

Composes the P2 SHA-256 gadget and two P3 AES-CTR gadgets into ONE statement
over a single tape (token-binding.md 2, 9.1):

    (B2)     Poseidon( key || iv_in || iv_out )             == KEY_COMMIT
    (B1_in)  AES128-CTR( key, iv_in,  serialize(req_toks) ) == ct_in
    (B1_out) AES128-CTR( key, iv_out, serialize(rsp_toks) ) == ct_out

KEY_COMMIT, ct_in and ct_out are PUBLIC. KEY_COMMIT is lifted from the request
packet, which the interlock certified INWARD before the response existed; the
two ciphertexts are the certified payload bytes themselves. The key material and
the token ids are committed witness and are never revealed.

WHY BOTH HALVES ARE NEEDED. With B1 alone the statement is vacuous: for ANY key'
the prover picks, tokens' = AES_dec(ct, key') re-encrypts to ct, so they could
grind key' until tokens' said whatever they liked. B2 pins the key to the one
whose commitment was already on the wire, and with key and ct both fixed,
tokens = AES_dec(ct, key) is uniquely the real token stream.

WHY THE CIPHERTEXT IS PUBLIC HERE (and the design doc hashes it instead).
token-binding.md 2 states B1 as SHA256(AES(...)) == H1 because in the general
recorder setting the verifier holds only a digest. On the interlock the verifier
holds the ciphertext itself -- it captured the frames, and commit.audit already
proved that capture IS the certified traffic. So the ciphertext is pinned
directly as a public constant, which is strictly stronger (no hash preimage to
argue about) and much cheaper: it removes a multi-block SHA-256 over the payload
and leaves only the single-block SHA-256 over the 40-byte key material, which is
exactly what the P2 gadget already does. Confidentiality is unaffected: the
ciphertext is on the cable regardless. The tokens stay hidden.

WIRING IS THE WHOLE JOB. Three gadget instances that each independently commit
"a key" prove nothing together. The pins below force:
  * both AES instances to use the SAME committed key bytes;
  * those bytes, and both IVs, to be the SAME bytes SHA-256 hashed for B2;
  * each AES instance's plaintext bytes to be the byte decomposition of the
    committed token integers.
Every pin is a LinComb over gathered wires -- no new verifier surface.

WHAT THIS MODULE DOES NOT DO. The committed token integers here are local to
this tape; welding them to the model proof's tokens (token-binding.md P4/P5) is
the CALLER's job, through `bind_tokens_to`. Both callers now do it, against the
`tok` vector unexplained_info commits: interlock_challenge.py gathers the
original sum_positions, subsample_challenge.py gathers range(_n) because it
slices the residual to the response before the LM head. A caller that skips the
weld gets a sound key binding over tokens that are not tied to any forward pass,
which is why the result line distinguishes OK from OK-NOWELD.
"""
from claims import EmbeddingLookupClaim
from tape import WitnessTensor
from token_binding import (_commit, aes_ctr_gadget, register_binding_tables,
                           token_bytes)
from poseidon_gadget import poseidon_commit_gadget

import poseidon_gl as _pg
import token_recorder as _tr

KEY_BYTES, IV_BYTES = _tr.KEY_BYTES, _tr.IV_BYTES
TOKEN_BYTES = _tr.TOKEN_BYTES
AES_BLOCK = _tr.AES_BLOCK
KEYMAT_BYTES = KEY_BYTES + 2 * IV_BYTES


def pool_gather(tape, pool, name, ids):
    """Committed view of `pool` at PUBLIC indices `ids`, via EmbeddingLookup
    with d=1. The indices are public because every wire move in this binding
    is a fixed byte position -- nothing about the layout is secret, only the
    values are."""
    gv = tape._alloc(name, len(ids))
    claim = EmbeddingLookupClaim(x=gv, E=pool.var, token_ids=list(ids), d=1)
    outs = tape._process_claim(claim, [pool.var])
    tape.claims.append(claim)
    return WitnessTensor(outs[gv] if outs else None, gv, (len(ids),), tape)


def _stride_pool(tape, strides, n_units, name):
    """Concatenate 4 byte-position strides into one gather pool, and return
    (pool, index_of) where index_of(p) locates flat byte position p.

    Both the SHA-256 message and the token serialization are laid out as four
    strides (byte k of every 4-byte unit), so byte p lives at stride p%4,
    offset p//4 -- i.e. pool index (p%4)*n_units + p//4."""
    pool = tape.concat(list(strides), (4 * n_units,))
    return pool, (lambda p: (p % 4) * n_units + p // 4)


def _pin_equal(tape, a, b):
    """a == b elementwise (one LinComb)."""
    tape.lincomb([a, b], [1, -1], 0)


def bind(tape, *, keymat, key_commit, streams, tables=None,
         reveal_tokens=None):
    """Add the full binding to `tape`.

    keymat      40 bytes: key(16) || iv_in(12) || iv_out(12)
    key_commit  32 public bytes = Poseidon(keymat) (the H2 from the request)
    streams     list of dicts, one per direction:
                  {"name": "in"|"out", "tokens": [ids], "ct": bytes, "iv": bytes}
    reveal_tokens  optional set of stream names whose token ids are additionally
                pinned to public constants (the request side is already public
                in today's model proof, so revealing it costs nothing and lets
                the verifier read the binding directly).

    Returns {"tok": {name: WitnessTensor}, "keymat_pool": ..., "n_claims": int}.
    """
    assert len(keymat) == KEYMAT_BYTES, "keymat must be %d bytes" % KEYMAT_BYTES
    assert _pg.hash_bytes(keymat) == key_commit, \
        "keymat does not hash to key_commit -- the prover cannot satisfy B2"
    key, iv_in, iv_out = keymat[:16], keymat[16:28], keymat[28:40]
    assert iv_in != iv_out, "per-direction IVs must differ (CTR keystream reuse)"
    ivs = {"in": iv_in, "out": iv_out}
    reveal_tokens = set(reveal_tokens or ())
    n0 = len(tape.claims)

    tables = tables or register_binding_tables(tape, with_xor=True)

    # ---- B2: Poseidon(keymat) == KEY_COMMIT -----------------------------
    # The key material is committed ONCE, in natural byte order, and both the
    # hash and the two cipher instances are pinned to that one vector. Poseidon
    # rather than SHA-256: the statement is identical, but a field-native sponge
    # costs ~46 claims where the bit-decomposed hash cost ~950 (and 92% of the
    # binding's prove time). See prover/ref/poseidon_gl.py for why, and for the
    # parameter caveat.
    km_bytes = _commit(tape, "cb_km", list(keymat))
    tape.range_word(km_bytes, tables["byte"])   # self-contained: bytes are bytes
    poseidon_commit_gadget(tape, keymat, km_bytes, key_commit)
    g_key = pool_gather(tape, km_bytes, "cb_km_key", list(range(16)))
    g_iv = {"in": pool_gather(tape, km_bytes, "cb_km_ivin", list(range(16, 28))),
            "out": pool_gather(tape, km_bytes, "cb_km_ivout", list(range(28, 40)))}

    # ---- B1 per direction, wired to that same key ------------------------
    out_tok = {}
    for s in streams:
        nm, toks, ct = s["name"], [int(v) for v in s["tokens"]], s["ct"]
        iv = s.get("iv") or ivs[nm]
        assert iv == ivs[nm], "stream %r iv is not the one keymat commits" % nm
        pt = _tr.serialize_tokens(toks)
        assert len(ct) == len(pt), (
            "stream %r: ciphertext %dB but %d tokens serialize to %dB"
            % (nm, len(ct), len(toks), len(pt)))

        exp = {}
        aes_ctr_gadget(tape, tables, key, iv, pt, ct_public=ct, export=exp)
        gather, idx = exp["gather"], exp["idx"]

        # (a) this instance's key bytes ARE the bytes B2 hashed
        a_key = gather("cb_%s_key" % nm, [("rk", (0, j)) for j in range(16)])
        _pin_equal(tape, g_key, a_key)
        # (b) ... and so is its IV
        a_iv = gather("cb_%s_iv" % nm, [("iv_bytes", (j,)) for j in range(IV_BYTES)])
        _pin_equal(tape, g_iv[nm], a_iv)

        # (c) this instance's plaintext bytes ARE the committed tokens' bytes
        tok = _commit(tape, "cb_tok_%s" % nm, toks)
        tb = token_bytes(tape, tok, tables)              # 4 strides, range-checked
        tb_pool, tb_at = _stride_pool(tape, tb, len(toks), "cb_tb_%s" % nm)
        g_pt = pool_gather(tape, tb_pool, "cb_%s_ptg" % nm,
                           [tb_at(p) for p in range(len(pt))])
        a_pt = gather("cb_%s_pt" % nm,
                      [("pt_bytes", (p // AES_BLOCK, p % AES_BLOCK))
                       for p in range(len(pt))])
        _pin_equal(tape, g_pt, a_pt)

        if nm in reveal_tokens:
            tape.lincomb([tok], [1], list(toks))
        out_tok[nm] = tok

    return {"tok": out_tok, "keymat": km_bytes,
            "n_claims": len(tape.claims) - n0}


def bind_tokens_to(tape, bound_tok, other_tok):
    """Pin a bound token vector to token witness that already exists on this
    tape (the P5 seam). `other_tok` must be the SAME WitnessTensor the model
    side consumed, not a fresh commitment of the same values -- otherwise the
    two vectors are only equal because the honest prover made them so."""
    _pin_equal(tape, bound_tok, other_tok)
