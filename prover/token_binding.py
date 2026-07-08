"""Token-binding circuit machinery, P1: lookup tables + token->byte plumbing
(analysis/token-binding.md §11-§12, paper Appendix E).

Provides the three AES lookup tables (S-box, xtime, byte-XOR) and the
token -> 4-little-endian-bytes decomposition that bridges the committed token
integers t_i to the cipher input. Table values are generated from the
reference recorder's algebraic AES (prover/ref/token_recorder.py) — a single
source of truth: the recorder is gated against FIPS-197 and library AES-GCM,
and the circuit difftests against the recorder's vectors.

Everything here reuses existing, Rust-handled claim types (WordExtractionClaim
+ RangeWordClaim for the decomposition, PairedTlookupClaim for the lookups);
no new verifier surface. The tables ride the public claim list like every
other table (`ui_exp`, `silu_table`, ...) and are settled by the generic
TableSettlement on both sides.

POLICY NOTE (for the P5 integration): the verifier checks proof/claim
*consistency* against whatever table data the claim list carries. The
deployment policy check must additionally pin the CONTENTS of `tb_sbox` /
`tb_xtime` / `tb_xor8` to the real AES tables (exactly as it must pin the
model structure) — otherwise a prover could register a self-consistent
non-AES "cipher" and the binding to the recorded digests would simply fail
to reproduce H1 for honest recorders but pass for colluding ones.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "ref"))
from token_recorder import SBOX, TOKEN_BYTES, _gf_mul  # noqa: E402

XTIME = [_gf_mul(x, 2) for x in range(256)]
XOR8_LEN = 1 << 16          # key = a*256 + b  ->  a XOR b


def xor8_values():
    return [((k >> 8) ^ (k & 0xFF)) for k in range(XOR8_LEN)]


def register_binding_tables(tape, *, with_xor=True):
    """Register the token-binding tables on a tape. Returns a dict:
      byte  — range table [0, 256), for the byte decomposition (and P2 limbs)
      sbox  — paired (k, SBOX[k]), 256 entries
      xtime — paired (k, xtime(k) = 2·k in GF(2^8)), 256 entries
      xor8  — paired (a·256 + b, a XOR b), 2^16 entries (skippable while
              unused: a registered table costs committed mult/w columns)
    """
    tables = {
        "byte": tape.register_table("tb_byte", list(range(256))),
        "sbox": tape.register_table("tb_sbox", list(range(256)),
                                    T_Y_data=list(SBOX)),
        "xtime": tape.register_table("tb_xtime", list(range(256)),
                                     T_Y_data=XTIME),
    }
    if with_xor:
        tables["xor8"] = tape.register_table("tb_xor8", list(range(XOR8_LEN)),
                                             T_Y_data=xor8_values())
    return tables


def token_bytes(tape, toks, tables):
    """Decompose committed token ids into TOKEN_BYTES little-endian bytes,
    each range-checked to [0, 256).

    Returns the list [b0, b1, b2, b3] of WitnessTensors, low byte first —
    byte n of token t is exactly serialize_tokens(tokens)[4t + n] in the
    reference recorder. The decomposition itself pins t < 2^32: the
    WordExtraction linear forces t = sum_n 2^(8n)·b_n with every b_n
    range-checked, and the maximum recomposable value 2^32 - 1 is far below
    the field modulus, so no wrapped candidate has a valid decomposition
    (the B.1 width argument).
    """
    return tape.word_extract(toks, tables["byte"], B=8, N=TOKEN_BYTES)


# ===========================================================================
# P2: the SHA-256 gadget — B2 = SHA256(key material) == H2, composed entirely
# from existing Rust-handled claims against the frozen layout of
# ref/sha256_trace.py (see that module's docstring for the layout spec).
# The only claim type added for this gadget anywhere in the system is
# LinCombClaim. Spine index-shifts (a..h as lagged views of the A/E state
# histories) are realized as ConcatClaim partitions: the spine is the dst of
# one natural concat, and each additional partition builds a second spine
# from its parts and pins it to the first with a LinComb equality.
# ===========================================================================

import torch as _torch

from sha256_trace import (K as _SHA_K, H0 as _SHA_H0, MOD32 as _MOD32,
                          M_LO as _M_LO, M_HI as _M_HI, N_LO as _N_LO,
                          N_HI as _N_HI, NB_HI as _NB_HI,
                          pad_one_block as _pad_one_block, trace as _sha_trace)


def _commit(tape, name, vals):
    t = _torch.tensor([int(v) % (2**64 - 2**32 + 1) for v in vals],
                      dtype=_torch.uint64, device="cuda")
    return tape.commit(name, t, (len(vals),))


def register_sha_tables(tape):
    """The gadget's range tables: bits, and the three carry widths."""
    return {
        "bit": tape.register_table("sha_bit", list(range(2))),
        "c8": tape.register_table("sha_c8", list(range(8))),
        "c4": tape.register_table("sha_c4", list(range(4))),
        "c2": tape.register_table("sha_c2", list(range(2))),
    }


def _partition(tape, name, spine, parts):
    """Pin `spine` == concat(parts): build a second spine from the parts and
    equate it to the first (one extra committed vector + one LinComb)."""
    spine2 = tape.concat(parts, (spine.var.length,))
    tape.lincomb([spine, spine2], [1, -1], 0)
    return spine2


def sha256_h2_gadget(tape, sha_tables, byte_table, msg, digest, _tamper=None):
    """Prove SHA256(msg) == digest (public) for a committed msg (length a
    multiple of 4, < 56 bytes — one padded block). `digest` is the public
    32-byte value (H2); it never becomes witness. Returns the four
    byte-position stride WitnessTensors of the committed message, which the
    caller wires to the key-material commitment.

    `_tamper(trace_dict)` is a TEST hook: it mutates the honest trace before
    commitment so negative tests can exercise each constraint family.
    """
    t = _sha_trace(msg)
    if _tamper is not None:
        _tamper(t)
    n_words = len(msg) // 4
    dwords = [int.from_bytes(digest[4 * i:4 * i + 4], "big") for i in range(8)]
    P_ = (1 << 64) - (1 << 32) + 1

    # --- message strides (range-checked bytes) --------------------------
    strides = []
    for k in range(4):
        s = _commit(tape, f"sha_ms{k}", t["msg_stride"][k])
        tape.range_word(s, byte_table)
        strides.append(s)

    # --- spines: W, A, E with their partitions --------------------------
    # W natural partition: [W_m (msg words) | W_pad (public) | W_s (schedule)]
    W_m = _commit(tape, "sha_Wm", t["W"][:n_words])
    W_pad = _commit(tape, "sha_Wpad", t["W"][n_words:16])
    W_s = _commit(tape, "sha_Ws", t["W"][16:])
    W = tape.concat([W_m, W_pad, W_s], (64,))
    # msg words recompose from strides; pad words are public constants
    tape.lincomb([W_m] + strides, [-1] + [1 << (8 * (3 - k)) for k in range(4)], 0)
    pad_block = _pad_one_block(bytes(len(msg)))
    tape.lincomb([W_pad], [1],
                 [int.from_bytes(pad_block[4 * i:4 * i + 4], "big")
                  for i in range(n_words, 16)])

    def slice_view(name, spine, lo, hi, vals):
        """Committed view of spine[lo:hi] via a partition equality."""
        parts, out = [], None
        if lo > 0:
            parts.append(_commit(tape, f"{name}_h", vals[:lo]))
        out = _commit(tape, name, vals[lo:hi])
        parts.append(out)
        if hi < len(vals):
            parts.append(_commit(tape, f"{name}_t", vals[hi:]))
        _partition(tape, name, spine, parts)
        return out

    W_r7 = slice_view("sha_Wr7", W, 9, 57, t["W"])
    W_r16 = slice_view("sha_Wr16", W, 0, 48, t["W"])
    W_mrole = slice_view("sha_Wmr", W, _M_LO, _M_HI, t["W"])
    W_nrole = slice_view("sha_Wnr", W, _N_LO, _NB_HI, t["W"])

    def spine_with_views(nm, hist, h0_prefix):
        # natural partition: 4 public singletons + the 64 round outputs
        prefs = [_commit(tape, f"sha_{nm}p{i}", [hist[i]]) for i in range(4)]
        for i, pv in enumerate(prefs):
            tape.lincomb([pv], [1], h0_prefix[i])
        outs = _commit(tape, f"sha_{nm}o", hist[4:])
        spine = tape.concat(prefs + [outs], (68,))
        views = {
            "a": slice_view(f"sha_{nm}a", spine, 3, 67, hist),
            "b": slice_view(f"sha_{nm}b", spine, 2, 66, hist),
            "c": slice_view(f"sha_{nm}c", spine, 1, 65, hist),
            "d": slice_view(f"sha_{nm}d", spine, 0, 64, hist),
            "o": slice_view(f"sha_{nm}O", spine, 4, 68, hist),
        }
        fins = [slice_view(f"sha_{nm}f{i}", spine, 64 + i, 65 + i, hist)
                for i in range(4)]
        return spine, views, fins

    A, Av, Afin = spine_with_views("A", t["A"], [_SHA_H0[3], _SHA_H0[2], _SHA_H0[1], _SHA_H0[0]])
    E, Ev, Efin = spine_with_views("E", t["E"], [_SHA_H0[7], _SHA_H0[6], _SHA_H0[5], _SHA_H0[4]])

    # --- bit classes: commit, booleanity, recomposition ------------------
    def bit_class(nm, rows, word_view):
        cols = []
        for j in range(32):
            b = _commit(tape, f"sha_{nm}{j}", [row[j] for row in rows])
            tape.range_word(b, sha_tables["bit"])
            cols.append(b)
        tape.lincomb([word_view] + cols, [-1] + [1 << j for j in range(32)], 0)
        return cols

    a_bit = bit_class("ab", t["a_bit"], Av["a"])
    b_bit = bit_class("bb", t["b_bit"], Av["b"])
    c_bit = bit_class("cb", t["c_bit"], Av["c"])
    e_bit = bit_class("eb", t["e_bit"], Ev["a"])
    f_bit = bit_class("fb", t["f_bit"], Ev["b"])
    g_bit = bit_class("gb", t["g_bit"], Ev["c"])
    wm_bit = bit_class("wm", t["w_bit_m"], W_mrole)
    wn_bit = bit_class("wn", t["w_bit_n"], W_nrole)

    # --- xor intermediates + pair products -------------------------------
    def xor_cols(nm, rows):
        cols = []
        for j in range(32):
            x = _commit(tape, f"sha_{nm}{j}", [row[j] for row in rows])
            cols.append(x)
        return cols

    rot = lambda cols, k: [cols[(j + k) % 32] for j in range(32)]
    shr_pad = lambda cols, k, zero: [cols[j + k] if j + k < 32 else zero
                                     for j in range(32)]

    # a committed, publicly-pinned all-zeros vector: the SHR padding operand
    zero48 = _commit(tape, "sha_zero48", [0] * 48)
    tape.lincomb([zero48], [1], 0)

    def xor_gadget(nm, u_cols, v_cols, x_rows):
        """x = u ^ v: p = hadamard(u, v); x committed; pin x = u + v - 2p.
        Returns (x_cols, p_cols)."""
        p_cols = [tape.hadamard(u_cols[j], v_cols[j]) for j in range(32)]
        x_cols = xor_cols(nm, x_rows)
        for j in range(32):
            tape.lincomb([x_cols[j], u_cols[j], v_cols[j], p_cols[j]],
                         [1, -1, -1, 2], 0)
        return x_cols, p_cols

    # Sigma1 pieces: x12 = r6^r11 (p1), p2 = x12*r25
    x12, p1 = xor_gadget("x12_", rot(e_bit, 6), rot(e_bit, 11), t["x12"])
    p2 = [tape.hadamard(x12[j], rot(e_bit, 25)[j]) for j in range(32)]
    # Sigma0: y12 = r2^r13 (q1), q2 = y12*r22
    y12, q1 = xor_gadget("y12_", rot(a_bit, 2), rot(a_bit, 13), t["y12"])
    q2 = [tape.hadamard(y12[j], rot(a_bit, 22)[j]) for j in range(32)]
    # Ch products
    ef = [tape.hadamard(e_bit[j], f_bit[j]) for j in range(32)]
    eg = [tape.hadamard(e_bit[j], g_bit[j]) for j in range(32)]
    # Maj: tbc = b^c (bc), u = a*tbc
    tbc, bc = xor_gadget("tbc_", b_bit, c_bit, t["tbc"])
    u = [tape.hadamard(a_bit[j], tbc[j]) for j in range(32)]
    # schedule: xm = s7^s18 (m1), m2 = xm*h3 ; xn = s17^s19 (n1), n2 = xn*h10
    xm, m1 = xor_gadget("xm_", rot(wm_bit, 7), rot(wm_bit, 18), t["xm"])
    m2 = [tape.hadamard(xm[j], shr_pad(wm_bit, 3, zero48)[j]) for j in range(32)]
    # n products live on words [N_LO, N_HI) = the first 48 of the 50 n rows;
    # slice the n-bit columns down to 48 via partition views
    wn48 = [slice_view(f"sha_wn48_{j}", wn_bit[j], 0, _N_HI - _N_LO,
                       [row[j] for row in t["w_bit_n"]]) for j in range(32)]
    zero48b = zero48
    xn, n1 = xor_gadget("xn_", rot(wn48, 17), rot(wn48, 19), t["xn"])
    n2 = [tape.hadamard(xn[j], shr_pad(wn48, 10, zero48b)[j]) for j in range(32)]

    # --- carries ----------------------------------------------------------
    ce = _commit(tape, "sha_ce", t["ce"]); tape.range_word(ce, sha_tables["c8"])
    ca = _commit(tape, "sha_ca", t["ca"]); tape.range_word(ca, sha_tables["c8"])
    cw = _commit(tape, "sha_cw", t["cw"]); tape.range_word(cw, sha_tables["c4"])
    cd = _commit(tape, "sha_cd", t["cd"]); tape.range_word(cd, sha_tables["c2"])

    # --- the two round adds (length 64, rhs = K[r]) -----------------------
    # e-add: E_o + 2^32 ce - d - h - W - Sigma1 - Ch = K[r]
    def rot_coefs(k_list):
        # coefficient of bit-var k for sum_j 2^j * bit[(j+k) % 32] terms
        return [sum(1 << ((kk - k) % 32) for k in k_list) for kk in range(32)]

    def shr_coefs(k):
        return [(1 << (kk - k)) if kk >= k else 0 for kk in range(32)]

    two_j = [1 << j for j in range(32)]
    e_xs = [Ev["o"], ce, Av["d"], Ev["d"], W]
    e_cf = [1, _MOD32, -1, -1, -1]
    for j in range(32):                      # -Sigma1: bits via x12/p2 form
        e_xs += [x12[j], rot(e_bit, 25)[j], p2[j]]
        e_cf += [-two_j[j], -two_j[j], 2 * two_j[j]]
        e_xs += [ef[j], g_bit[j], eg[j]]     # -Ch
        e_cf += [-two_j[j], -two_j[j], two_j[j]]
    tape.lincomb(e_xs, e_cf, list(_SHA_K))

    a_xs = [Av["o"], ca, Ev["d"], W]
    a_cf = [1, _MOD32, -1, -1]
    for j in range(32):
        a_xs += [x12[j], rot(e_bit, 25)[j], p2[j], ef[j], g_bit[j], eg[j]]
        a_cf += [-two_j[j], -two_j[j], 2 * two_j[j],
                 -two_j[j], -two_j[j], two_j[j]]
        a_xs += [y12[j], rot(a_bit, 22)[j], q2[j]]      # -Sigma0
        a_cf += [-two_j[j], -two_j[j], 2 * two_j[j]]
        a_xs += [bc[j], u[j]]                           # -Maj
        a_cf += [-two_j[j], -two_j[j]]
    tape.lincomb(a_xs, a_cf, list(_SHA_K))

    # --- schedule add (length 48, rhs = 0) --------------------------------
    s_xs = [W_s, cw, W_r7, W_r16]
    s_cf = [1, _MOD32, -1, -1]
    for j in range(32):                      # -ssig0 (m), -ssig1 (n)
        s_xs += [xm[j], shr_pad(wm_bit, 3, zero48)[j], m2[j]]
        s_cf += [-two_j[j], -two_j[j], 2 * two_j[j]]
        s_xs += [xn[j], shr_pad(wn48, 10, zero48b)[j], n2[j]]
        s_cf += [-two_j[j], -two_j[j], 2 * two_j[j]]
    # NOTE alignment: schedule round r=16+i uses m-index (r-15)-M_LO = i and
    # n-index (r-2)-N_LO = i — all length-48 vectors line up at i.
    tape.lincomb(s_xs, s_cf, 0)

    # --- digest (8 singleton pins; H2 words are public) --------------------
    order = [3, 2, 1, 0]         # finals: out[0]=A[67]=Afin[3], out[1]=A[66], ...
    cd_cols = [slice_view(f"sha_cd{i}", cd, i, i + 1, t["cd"]) for i in range(8)]
    for i in range(8):
        fin = (Afin if i < 4 else Efin)[order[i % 4]]
        rhs = (_SHA_H0[i] - dwords[i]) % P_
        tape.lincomb([cd_cols[i], fin], [_MOD32, -1], rhs)

    return strides


# ===========================================================================
# P3: the AES-128-CTR gadget — composed against the frozen layout of
# ref/aes_trace.py, using only existing claims. The workhorse is the gather:
# every wire in the cipher is a byte-move with PUBLIC indices, so the whole
# committed byte layout concatenates into one pool and EmbeddingLookupClaim
# (public token_ids, d=1) gathers lookup operands and outputs from it. The
# cipher is then THREE paired lookups (all XORs against tb_xor8 keyed by
# k = 256a + b, all S-boxes, all xtimes — each one claim over its full site
# vector), a handful of LinComb pins, and per-class byte range checks.
# ===========================================================================

import aes_trace as _at


def aes_ctr_gadget(tape, tables, key, iv, plaintext, ct_public=None,
                   _tamper=None):
    """Prove ct = AES128-CTR(key, iv, plaintext) with the GCM counter layout,
    for committed key/iv/plaintext bytes. `tables` is
    register_binding_tables(tape, with_xor=True) — the P1 tables.

    If `ct_public` is given (bytes), the ciphertext is additionally pinned to
    it as a public constant (the standalone P3 gate). In the full binding the
    ciphertext instead feeds the SHA-256 gadget (H1 = SHA256(ct)), and key,
    iv, and plaintext bytes are wired to the key-material and token-stream
    commitments.

    Returns the dict of committed class WitnessTensors for that wiring.
    `_tamper(trace_dict)` is the TEST hook for negative cases.
    """
    from claims import EmbeddingLookupClaim
    from tape import WitnessTensor

    t = _at.trace(key, iv, plaintext)
    if _tamper is not None:
        _tamper(t)
    idx, pool_vals = _at.pool_layout(t)
    S = _at.sites(t)

    # committed byte classes (flat, range-checked), concatenated into the pool
    parts = {}
    for cls in _at.BYTE_CLASSES:
        vals = [cont[i] for cont, i, _p in _at._slots(t[cls])]
        v = _commit(tape, f"aes_{cls}", vals)
        tape.range_word(v, tables["byte"])
        parts[cls] = v
    pool = tape.concat([parts[c] for c in _at.BYTE_CLASSES], (len(pool_vals),))

    # public pins: counter suffixes and Rcon operands (whole classes public)
    tape.lincomb([parts["ctr_bytes"]], [1],
                 [b for j in range(t["nblocks"])
                  for b in (_at.CTR_START + j).to_bytes(4, "big")])
    tape.lincomb([parts["ks_rcon_b"]], [1], list(_at.RCON))
    if ct_public is not None:
        tape.lincomb([parts["ct_bytes"]], [1], list(ct_public))

    def gather(name, refs):
        ids = [idx[r] for r in refs]
        gv = tape._alloc(name, len(ids))
        claim = EmbeddingLookupClaim(x=gv, E=pool.var, token_ids=ids, d=1)
        outs = tape._process_claim(claim, [pool.var])
        tape.claims.append(claim)
        return WitnessTensor(outs[gv] if outs else None, gv, (len(ids),), tape)

    # all XORs: one key vector, one key linear, one lookup, one output pin
    kv = _commit(tape, "aes_k", _at.key_values(t, S["xor"]))
    a_g = gather("aes_xa", [s["a"] for s in S["xor"]])
    b_g = gather("aes_xb", [s["b"] for s in S["xor"]])
    z_g = gather("aes_xz", [s["z"] for s in S["xor"]])
    tape.lincomb([kv, a_g, b_g], [1, -256, -1], 0)
    z_look = tape.paired_tlookup(kv, tables["xor8"])
    tape.lincomb([z_look, z_g], [1, -1], 0)

    # all S-boxes / all xtimes: gather inputs and outputs, one lookup each
    for nm, tbl, site_list in (("s", tables["sbox"], S["sbox"]),
                               ("t", tables["xtime"], S["xtime"])):
        x_g = gather(f"aes_{nm}x", [s["x"] for s in site_list])
        y_g = gather(f"aes_{nm}y", [s["y"] for s in site_list])
        y_look = tape.paired_tlookup(x_g, tbl)
        tape.lincomb([y_look, y_g], [1, -1], 0)

    return parts
