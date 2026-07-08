"""AES-128-CTR constraint-system reference: witness trace + constraint checker
(token-binding P3; companion to ref/sha256_trace.py, which froze the P2 SHA-256
layout the same way).

This file freezes the CONSTRAINT LAYOUT of the future AES-CTR claim before any
tape or Rust code exists: `trace()` computes every value the claim will commit,
`check_constraints()` evaluates every constraint the claim will emit (exact
integer arithmetic, all residuals must be zero), and the tests difftest the
ciphertext against token_recorder.aes128_ctr_gcm and fuzz single-value tampers
to show the system has no unconstrained slots. The tape compile and the Rust
handler are both written against THIS layout and difftested against it.

Scope: one AES-CTR invocation — one (key, iv) pair, N = ceil(len(pt)/16)
counter blocks with the GCM counter layout iv || BE32(CTR_START + j)
(token_recorder pins CTR_START = 2). The key schedule is committed ONCE per
key and shared by every block. SBOX and the GF(2^8) doubling that derives
XTIME and RCON are imported from token_recorder — single source of truth for
the algebra; nothing here retypes a table.

Arithmetization (byte-level, lookup-based — no bit decompositions):

  * Three public tables: SBOX (rows (x, SBOX[x])), XTIME (rows
    (x, 2·x in GF(2^8))), and XOR8 (rows (k, z) with k = 256a + b and
    z = a XOR b over all byte pairs).
  * Every XOR is one XOR8 lookup plus one linear: the committed key k is
    pinned to its two operand slots by k = 256a + b (the "*_key" residuals);
    the lookup row then fixes the committed output z = hi(k) XOR lo(k).
    Which operand is a is fixed per site — see the class list.
  * An XOR with a PUBLIC operand (Rcon) is still a normal XOR8: the public
    operand is a committed slot pinned by a linear to the constant.
  * Every committed byte is range-checked (tape side: the P1 tb_byte table;
    here the "*_byte_range" residuals). Keys need no separate range check —
    the key linear pins each to two range-checked bytes.
  * Index remaps are free: RotWord and ShiftRows appear only as fixed
    source-index permutations inside other constraints (SHIFT_SRC below),
    never as committed copies.
  * State bytes are flat in FIPS-197 input order: flat[r + 4c] = state row r,
    column c — token_recorder._encrypt_block's convention, whose output byte
    i is flat[i] of the final state. Column c is the contiguous slice
    flat[4c : 4c+4].

Committed classes (dict-of-lists; every int leaf is a committed slot — the
tamper fuzz walks exactly these, via COMMITTED/_slots):

  Per key (the schedule, once per invocation):
    rk[11][16]        round keys; rk[0] is the free key input, every byte of
                      rk[1..10] is pinned as a ks_word_xor output (176 bytes)
    ks_sub[10][4]     SubWord outputs: ks_sub[i-1][j] =
                      SBOX[rk[i-1][12 + (j+1)%4]] (RotWord = remap) (40 bytes)
    ks_rcon_b[10]     the Rcon operand, pinned PUBLIC to RCON[i]    (10 bytes)
    ks_rcon_k/z[10]   key/output of the Rcon XOR: z = ks_sub[i-1][0] ^ rcon
                      (a = subword byte, b = rcon slot)          (10 + 10)
    ks_xor_k[10][16]  keys of the word-chain xors: byte r of word w of rk[i]
                      = rk[i-1][4w+r] XOR src[r], where src is the rcon'd
                      subword [z, sub1, sub2, sub3] for w = 0, else the
                      just-derived previous word of rk[i]
                      (a = rk[i-1] byte, b = src byte)           (160 keys)

  Per invocation:
    iv_bytes[12]      free committed input, shared by all blocks

  Per block j (of N):
    ctr_bytes[j][4]   counter suffix, pinned PUBLIC to BE32(CTR_START + j)
    s_ark[j][11][16]  state after AddRoundKey, rounds 0..10; s_ark[j][10] is
                      the keystream block. ark_xor operands: a = the round
                      input byte (counter block / mc_out / shifted SubBytes),
                      b = the rk[rnd] byte
    ark_k[j][11][16]  the ark_xor keys
    s_sub[j][10][16]  SubBytes outputs: s_sub[r-1][i] = SBOX[s_ark[r-1][i]]
    mc_xt[j][9][16]   XTIME of the SHIFTED SubBytes state, rounds 1..9:
                      mc_xt[r-1][i] = XTIME[s_sub[r-1][SHIFT_SRC[i]]]
    mc_k[j][9][16][4], mc_mid[j][9][16][3], mc_out[j][9][16]
                      MixColumns xor chains. For column c, row r (slot
                      i = 4c + r), with a_* = the shifted column bytes and
                      xt_* their XTIMEs: out_r = 2a_r ^ 3a_{r+1} ^ a_{r+2}
                      ^ a_{r+3} evaluated as the chain
                      xt_r ^ xt_{r+1} ^ a_{r+1} ^ a_{r+2} ^ a_{r+3} —
                      4 XOR8s (a = running accumulator, b = next term),
                      3 committed mids, final z in mc_out, which then feeds
                      the round's ark_xor key linear
    pt_bytes[j][<=16] free committed input (last block may be partial)
    ct_bytes[j][<=16] ciphertext: ct = keystream ^ pt
                      (a = s_ark[j][10][i], b = pt byte); keys in
    ct_k[j][<=16]

Lookup counts: per key 40 sbox + 170 xor (10 rcon + 160 word); per block
160 sbox + 144 xtime + 176 ark + 576 MixColumns + <=16 ct xors. Unused
keystream bytes of a partial final block stay constrained — they are ark_xor
outputs — they simply have no ct_xor consuming them.
"""
from token_recorder import (SBOX, _gf_mul, aes128_ctr_gcm,
                            CTR_START, KEY_BYTES, IV_BYTES, AES_BLOCK)

XTIME = [_gf_mul(x, 2) for x in range(256)]     # public table: y = 2x in GF(2^8)

RCON = [1]                                       # round constants, rounds 1..10
for _ in range(9):
    RCON.append(_gf_mul(RCON[-1], 2))

# ShiftRows as a free source-index remap on the flat state: the byte at
# (row r, col c) after ShiftRows came from (r, (c + r) % 4), so
# shifted[r + 4c] = state[r + 4((c + r) % 4)].
SHIFT_SRC = [(i % 4) + 4 * (((i // 4) + (i % 4)) % 4) for i in range(16)]

# Classes whose leaves are committed BYTES (range-checked) vs XOR8 KEYS
# (pinned by their key linear to two range-checked bytes). Together: every
# committed slot of the layout — the tamper fuzz walks COMMITTED exhaustively.
BYTE_CLASSES = ("rk", "ks_sub", "ks_rcon_b", "ks_rcon_z", "iv_bytes",
                "ctr_bytes", "pt_bytes", "s_ark", "s_sub", "mc_xt", "mc_mid",
                "mc_out", "ct_bytes")
KEY_CLASSES = ("ks_rcon_k", "ks_xor_k", "ark_k", "mc_k", "ct_k")
COMMITTED = BYTE_CLASSES + KEY_CLASSES


def _slots(obj, path=()):
    """Yield (container, index, path) for every int leaf of a nested list —
    the checker range-checks byte leaves through this, and the tamper fuzz
    bumps every leaf through it, so both see exactly the committed slots."""
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, list):
                yield from _slots(v, path + (i,))
            else:
                yield obj, i, path + (i,)


def trace(key, iv, plaintext):
    """Full witness trace for one AES-128-CTR invocation. Every int leaf of
    the COMMITTED classes is a committed variable of the future claim (the
    two PUBLIC classes, ctr_bytes and ks_rcon_b, are pinned to constants)."""
    assert len(key) == KEY_BYTES and len(iv) == IV_BYTES
    assert len(plaintext) > 0, "empty payloads are never recorded"
    nblocks = (len(plaintext) + AES_BLOCK - 1) // AES_BLOCK
    t = {"nblocks": nblocks, "iv_bytes": list(iv)}

    # key schedule — once per key, shared by every block
    rk = [list(key)]
    t.update({k: [] for k in ("ks_sub", "ks_rcon_b", "ks_rcon_k",
                              "ks_rcon_z", "ks_xor_k")})
    for i in range(1, 11):
        prev = rk[-1]
        sub = [SBOX[prev[12 + ((j + 1) % 4)]] for j in range(4)]  # Rot+SubWord
        rb = RCON[i - 1]
        rz = sub[0] ^ rb
        t["ks_sub"].append(sub)
        t["ks_rcon_b"].append(rb)
        t["ks_rcon_k"].append(256 * sub[0] + rb)
        t["ks_rcon_z"].append(rz)
        temp = [rz, sub[1], sub[2], sub[3]]
        cur, keys = [], []
        for w in range(4):
            src = temp if w == 0 else cur[4 * (w - 1): 4 * w]
            for r in range(4):
                a, b = prev[4 * w + r], src[r]
                keys.append(256 * a + b)
                cur.append(a ^ b)
        rk.append(cur)
        t["ks_xor_k"].append(keys)
    t["rk"] = rk

    # per-block CTR keystream + ciphertext
    t.update({k: [] for k in ("ctr_bytes", "pt_bytes", "s_ark", "s_sub",
                              "ark_k", "mc_xt", "mc_k", "mc_mid", "mc_out",
                              "ct_k", "ct_bytes")})
    for j in range(nblocks):
        ctr4 = list((CTR_START + j).to_bytes(4, "big"))
        t["ctr_bytes"].append(ctr4)

        s_ark, ark_k = [], []
        s_sub, mc_xt = [], []
        mc_k, mc_mid, mc_out = [], [], []

        def ark(inp, rnd):
            ark_k.append([256 * inp[i] + rk[rnd][i] for i in range(16)])
            s_ark.append([inp[i] ^ rk[rnd][i] for i in range(16)])

        ark(t["iv_bytes"] + ctr4, 0)
        for rnd in range(1, 11):
            sub = [SBOX[x] for x in s_ark[rnd - 1]]
            s_sub.append(sub)
            shifted = [sub[SHIFT_SRC[i]] for i in range(16)]
            if rnd < 10:
                xt = [XTIME[x] for x in shifted]
                mc_xt.append(xt)
                keys_r, mid_r, out_r = [], [], []
                for c in range(4):
                    for row in range(4):
                        terms = [xt[4 * c + row],
                                 xt[4 * c + (row + 1) % 4],
                                 shifted[4 * c + (row + 1) % 4],
                                 shifted[4 * c + (row + 2) % 4],
                                 shifted[4 * c + (row + 3) % 4]]
                        acc, ks, zs = terms[0], [], []
                        for tm in terms[1:]:
                            ks.append(256 * acc + tm)
                            acc ^= tm
                            zs.append(acc)
                        keys_r.append(ks)
                        mid_r.append(zs[:3])
                        out_r.append(zs[3])
                mc_k.append(keys_r)
                mc_mid.append(mid_r)
                mc_out.append(out_r)
                ark(out_r, rnd)          # out_r index 4c+row IS flat r+4c
            else:
                ark(shifted, rnd)        # final round: no MixColumns

        chunk = list(plaintext[AES_BLOCK * j: AES_BLOCK * (j + 1)])
        ks10 = s_ark[10]                 # keystream block
        t["pt_bytes"].append(chunk)
        t["ct_k"].append([256 * ks10[i] + chunk[i] for i in range(len(chunk))])
        t["ct_bytes"].append([ks10[i] ^ chunk[i] for i in range(len(chunk))])
        t["s_ark"].append(s_ark)
        t["s_sub"].append(s_sub)
        t["ark_k"].append(ark_k)
        t["mc_xt"].append(mc_xt)
        t["mc_k"].append(mc_k)
        t["mc_mid"].append(mc_mid)
        t["mc_out"].append(mc_out)

    t["ciphertext"] = bytes(sum(t["ct_bytes"], []))
    return t


# ------------------------------------------------------------- constraints

def check_constraints(t):
    """Evaluate every constraint of the claim layout on a trace. Returns a
    list of (name, index, residual) for every NON-ZERO residual; empty list
    == satisfied. Exact integer arithmetic. Every equation references only
    committed slots and public constants — never recomputed honest values —
    so a satisfying trace IS a valid witness for the future claim."""
    bad = []

    def req(name, idx, lhs, rhs=0):
        if lhs != rhs:
            bad.append((name, idx, lhs - rhs))

    def look(name, idx, table, x, y):
        # paired lookup: (x, y) must be a row of the public table y = table[x]
        if 0 <= x < 256:
            req(name, idx, y, table[x])
        else:
            bad.append((name + "_in_range", idx, x))

    def xor8(name, idx, a, b, k, z):
        # XOR8 lookup keyed by k: the key linear pins k to the two operand
        # slots; the table row then fixes z = hi(k) XOR lo(k).
        req(name + "_key", idx, k, 256 * a + b)
        if 0 <= k < 65536:
            req(name, idx, z, (k >> 8) ^ (k & 0xFF))
        else:
            bad.append((name + "_key_range", idx, k))

    # byte range on every committed byte slot (tape side: the tb_byte table)
    for cls in BYTE_CLASSES:
        for cont, i, path in _slots(t[cls]):
            req(cls + "_byte_range", path, int(not (0 <= cont[i] < 256)))

    # key schedule
    rk = t["rk"]
    for i in range(1, 11):
        prev, cur, sub = rk[i - 1], rk[i], t["ks_sub"][i - 1]
        for j in range(4):
            look("ks_sbox", (i, j), SBOX, prev[12 + ((j + 1) % 4)], sub[j])
        req("ks_rcon_const", i, t["ks_rcon_b"][i - 1], RCON[i - 1])
        xor8("ks_rcon_xor", i, sub[0], t["ks_rcon_b"][i - 1],
             t["ks_rcon_k"][i - 1], t["ks_rcon_z"][i - 1])
        temp = [t["ks_rcon_z"][i - 1], sub[1], sub[2], sub[3]]
        for w in range(4):
            src = temp if w == 0 else cur[4 * (w - 1): 4 * w]
            for r in range(4):
                xor8("ks_word_xor", (i, w, r), prev[4 * w + r], src[r],
                     t["ks_xor_k"][i - 1][4 * w + r], cur[4 * w + r])

    # blocks
    for b in range(t["nblocks"]):
        ctr4 = (CTR_START + b).to_bytes(4, "big")
        for i in range(4):
            req("ctr_public", (b, i), t["ctr_bytes"][b][i], ctr4[i])
        ark_in = t["iv_bytes"] + t["ctr_bytes"][b]
        for rnd in range(11):
            if rnd > 0:
                sub = t["s_sub"][b][rnd - 1]
                for i in range(16):
                    look("sbox", (b, rnd, i), SBOX,
                         t["s_ark"][b][rnd - 1][i], sub[i])
                shifted = [sub[SHIFT_SRC[i]] for i in range(16)]
                if rnd < 10:
                    xt = t["mc_xt"][b][rnd - 1]
                    for i in range(16):
                        look("xtime", (b, rnd, i), XTIME, shifted[i], xt[i])
                    for c in range(4):
                        for row in range(4):
                            i = 4 * c + row
                            terms = [xt[i],
                                     xt[4 * c + (row + 1) % 4],
                                     shifted[4 * c + (row + 1) % 4],
                                     shifted[4 * c + (row + 2) % 4],
                                     shifted[4 * c + (row + 3) % 4]]
                            chain = (t["mc_mid"][b][rnd - 1][i]
                                     + [t["mc_out"][b][rnd - 1][i]])
                            acc = terms[0]
                            for step in range(4):
                                xor8("mc_xor", (b, rnd, i, step),
                                     acc, terms[step + 1],
                                     t["mc_k"][b][rnd - 1][i][step],
                                     chain[step])
                                acc = chain[step]   # next operand: the
                                                    # committed mid, not a
                                                    # recomputed value
                    ark_in = t["mc_out"][b][rnd - 1]
                else:
                    ark_in = shifted
            for i in range(16):
                xor8("ark_xor", (b, rnd, i), ark_in[i], rk[rnd][i],
                     t["ark_k"][b][rnd][i], t["s_ark"][b][rnd][i])
        ks10 = t["s_ark"][b][10]
        for i in range(len(t["pt_bytes"][b])):
            xor8("ct_xor", (b, i), ks10[i], t["pt_bytes"][b][i],
                 t["ct_k"][b][i], t["ct_bytes"][b][i])
    return bad


def aes_ctr(key, iv, plaintext):
    """Reference entry point: trace + checker + oracle cross-check."""
    t = trace(key, iv, plaintext)
    bad = check_constraints(t)
    assert not bad, f"constraint system unsatisfied on honest trace: {bad[:5]}"
    assert t["ciphertext"] == aes128_ctr_gcm(key, iv, plaintext), \
        "ciphertext mismatch vs token_recorder oracle"
    return t


# ---------------------------------------------------------------- gadget API

def pool_layout(t):
    """Canonical flat ordering of every committed BYTE slot: the gadget
    commits each class as one flat vector and concatenates them (in
    BYTE_CLASSES order) into a single gather pool. Returns
    (index_of: dict (cls, path) -> pool index, values: flat list)."""
    index_of, values = {}, []
    for cls in BYTE_CLASSES:
        for cont, i, path in _slots(t[cls]):
            index_of[(cls, path)] = len(values)
            values.append(cont[i])
    return index_of, values


def key_values(t, xor_sites_list):
    """The committed XOR-key vector, in site order."""
    def deref(ref):
        cls, path = ref
        obj = t[cls]
        for p in path:
            obj = obj[p]
        return obj
    return [deref(s["k"]) for s in xor_sites_list]


def sites(t):
    """The canonical site enumeration the circuit emits — one entry per
    lookup, with operands as (class, path) refs into the committed classes.
    This is the single source of truth the gadget composes from; the tests
    assert its equations coincide with check_constraints on honest and
    tampered traces (same relations, derived independently above)."""
    sbox_sites, xtime_sites, xor_sites_list = [], [], []

    def X(a, b, k, z):
        xor_sites_list.append({"a": a, "b": b, "k": k, "z": z})

    # key schedule
    for i in range(1, 11):
        for j in range(4):
            sbox_sites.append({"x": ("rk", (i - 1, 12 + ((j + 1) % 4))),
                               "y": ("ks_sub", (i - 1, j))})
        X(("ks_sub", (i - 1, 0)), ("ks_rcon_b", (i - 1,)),
          ("ks_rcon_k", (i - 1,)), ("ks_rcon_z", (i - 1,)))
        for w in range(4):
            for r in range(4):
                if w == 0:
                    b_ref = (("ks_rcon_z", (i - 1,)) if r == 0
                             else ("ks_sub", (i - 1, r)))
                else:
                    b_ref = ("rk", (i, 4 * (w - 1) + r))
                X(("rk", (i - 1, 4 * w + r)), b_ref,
                  ("ks_xor_k", (i - 1, 4 * w + r)), ("rk", (i, 4 * w + r)))

    # blocks
    for b in range(t["nblocks"]):
        for rnd in range(11):
            if rnd > 0:
                for i in range(16):
                    sbox_sites.append({"x": ("s_ark", (b, rnd - 1, i)),
                                       "y": ("s_sub", (b, rnd - 1, i))})
                if rnd < 10:
                    for i in range(16):
                        xtime_sites.append(
                            {"x": ("s_sub", (b, rnd - 1, SHIFT_SRC[i])),
                             "y": ("mc_xt", (b, rnd - 1, i))})
            for i in range(16):
                if rnd == 0:
                    a_ref = (("iv_bytes", (i,)) if i < 12
                             else ("ctr_bytes", (b, i - 12)))
                elif rnd < 10:
                    a_ref = ("mc_out", (b, rnd - 1, i))
                else:
                    a_ref = ("s_sub", (b, 9, SHIFT_SRC[i]))
                X(a_ref, ("rk", (rnd, i)),
                  ("ark_k", (b, rnd, i)), ("s_ark", (b, rnd, i)))
            if 0 < rnd < 10:
                for c in range(4):
                    for row in range(4):
                        i = 4 * c + row
                        term_refs = [
                            ("mc_xt", (b, rnd - 1, 4 * c + (row + 1) % 4)),
                            ("s_sub", (b, rnd - 1, SHIFT_SRC[4 * c + (row + 1) % 4])),
                            ("s_sub", (b, rnd - 1, SHIFT_SRC[4 * c + (row + 2) % 4])),
                            ("s_sub", (b, rnd - 1, SHIFT_SRC[4 * c + (row + 3) % 4])),
                        ]
                        acc_ref = ("mc_xt", (b, rnd - 1, i))
                        for step in range(4):
                            z_ref = (("mc_mid", (b, rnd - 1, i, step)) if step < 3
                                     else ("mc_out", (b, rnd - 1, i)))
                            X(acc_ref, term_refs[step],
                              ("mc_k", (b, rnd - 1, i, step)), z_ref)
                            acc_ref = z_ref
        for i in range(len(t["pt_bytes"][b])):
            X(("s_ark", (b, 10, i)), ("pt_bytes", (b, i)),
              ("ct_k", (b, i)), ("ct_bytes", (b, i)))

    publics = ([{"ref": ("ctr_bytes", (b, i)),
                 "value": (CTR_START + b).to_bytes(4, "big")[i]}
                for b in range(t["nblocks"]) for i in range(4)]
               + [{"ref": ("ks_rcon_b", (i,)), "value": RCON[i]}
                  for i in range(10)])
    return {"sbox": sbox_sites, "xtime": xtime_sites,
            "xor": xor_sites_list, "public": publics}
