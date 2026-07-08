# Design: Binding committed input and output tokens to pre-recorded commitments (AES + SHA-256)

**Status:** design, not implemented; the §6 parameter decisions were pinned 2026-07-05
against the interlock protocol v6 (§9), the token-side selection design is §10, and the
implementation roadmap is §12. Binds **both** the committed input
(prompt) tokens and the committed output (decoded) tokens to commitments recorded
independently at generation time, so the proof attests to the *real* input→output
transcript — not to tokens the prover is free to choose. Closes the `tok`-trust-boundary
gap in the unexplained-information proof (`pipeline/max_claim.py:23-25`; the audit note:
the Max/Info claims bound `U` of *whatever* tokens are committed, with nothing tying them
to the run's real tokens).

---

## 1. The gap this closes

The proof keeps **both** input and output tokens committed-and-hidden (README: "reveals
nothing about … even the input and output tokens"). Two distinct problems arise if they
float free of the real run:

- **Outputs.** `U(o) = Σ_t −log₂ q(o_t)` is bounded over the committed output tokens, but
  they're only proven to be *some* valid one-hot selections — not what the model emitted.
  A prover could commit lower-surprisal tokens and **under-claim `U`** (claim less
  exfiltration bandwidth than the truth — the wrong direction for the threat model).
- **Inputs.** The bound is `U(o) = H(o | D(x))` — *conditioned on the input* `x`. If the
  committed input isn't the real prompt, the "declared computation `D(x)`" is on a fake
  `x`, and the bound certifies nothing about the actual run. More broadly, an attestation
  "model `M` ran on `x` and produced `o`" is only meaningful if **both** `x` and `o` are
  pinned to an external record; binding one without the other leaves the transcript
  half-open.

We close both by binding each committed token stream to a commitment recorded
**independently, at generation time**, before/outside the proof.

## 2. Construction

For each token stream `s ∈ {in, out}`, an independent process records at generation time:

```
H1_s = Hash( AES(tokens_s, key_s) )    # hash of the ciphertext of the stream's tokens
H2_s = Hash( key_s )                   # hash of that stream's key
```

`H1_in, H2_in, H1_out, H2_out` are **public** (the root of trust, §3). In the proof the
prover commits `tokens_in`, `tokens_out`, and the key(s) (phase-1, hidden behind the
Merkle root, like the weights), and proves, for each stream:

```
(B1_s)  Hash( AES(committed_tokens_s, committed_key_s) ) == H1_s
(B2_s)  Hash( committed_key_s )                          == H2_s
```

`committed_tokens_out` must be the **same witness variables** as the `tok` consumed by
`MaxClaim`/`InfoFinalizeClaim`, and `committed_tokens_in` the **same** as the prompt
tokens fed to the first `EmbeddingLookupClaim` — wired by copy/equality constraints
(§4.4) — so the binding applies to exactly the tokens used in the bounded computation.

**One record or two.** If the deployment records the whole transcript at once, encrypt
`input ‖ output` under a single key → one `(H1, H2)` pair (cheapest, simplest). If input
and output are recorded at different times (prompt at request, output at generation),
keep them as separate `(H1_in,H2_in)`, `(H1_out,H2_out)` pairs with separate keys. The
circuit is the same gadget instantiated once or twice; pick to match the recorder.

## 3. Soundness

**Both hashes are required (per stream).** With `H1_s` alone the binding is vacuous:
`H1_s = Hash(C)` fixes the ciphertext `C`, but for *any* key' the prover picks,
`tokens' = AES_dec(C, key')` re-encrypts to `C` and satisfies (B1_s). They could grind
key' to make `tokens'` whatever they want. Adding (B2_s) pins `key` to the real key
(collision resistance); with `C` and `key` both fixed, `tokens = AES_dec(C, key)` is
**uniquely** the real tokens.

**Root of trust.** The guarantee reduces to: *the committed tokens equal those that
produced the pre-recorded `H1/H2`*. This is meaningful only if `H1/H2` are recorded by a
party/process the verifier trusts, **independently of and prior to** the proof (an output
log, a notarized record, the deployment's encrypted-I/O store). If the prover controls
the recording, the binding is circular. State this assumption wherever the bound is
reported.

**Confidentiality (why AES).** Tokens are low-entropy (`~18` bits each over a 200k vocab),
so a bare `Hash(tokens)` record is dictionary-attackable and would leak them. Encrypting
under a high-entropy key first makes `H1 = Hash(ciphertext)` a *hiding* commitment,
preserving zero-knowledge (no tokens revealed). If inputs/outputs were public in some
deployment, a plain `Hash(tokens)` binding would suffice and the AES layer could be
dropped for that stream; the default here is hidden, so AES is needed for both.

**Composition.** (B1_s)/(B2_s) add only AES and SHA-256 constraint families (§4); their
soundness error is the usual per-LogUp `(M + T + 1)/|F|` (§5), negligible at these data
sizes. AES/SHA byte/word values are `< 2^{32}` and fit Goldilocks with no wrap risk.

## 4. In-circuit realization (Ligero / LogUp)

This stack is lookup-based (`tape.paired_tlookup`, `tape.range_word`/`word_extract`,
booleanity quadratics, `hadamard`) — the efficient modern way to arithmetize AES/SHA, far
cheaper than bit-by-bit R1CS. We adapt the standard, public constructions onto the
existing gadgets; we do **not** import a circom/gnark/arkworks circuit (those are
R1CS/PLONK, a different proof system). The same gadgets serve both the input and output
streams.

### 4.1 AES (per 16-byte block, AES-128 = 10 rounds)
- **SubBytes:** S-box is a 256-entry byte→byte map — one `paired_tlookup` per byte
  (`T = (k, SBOX[k])`). 16/round.
- **ShiftRows:** fixed permutation — wiring (copy constraints), no cost.
- **MixColumns:** linear over GF(2⁸) (`xtime`) — a small linear family per column (or an
  `xtime` lookup).
- **AddRoundKey:** byte XOR — a 2→1 XOR `paired_tlookup` (`T = (a‖b, a⊕b)`) or bit-decomp
  + booleanity. 16/round.
- **Key schedule:** `RotWord`/`SubWord`/`Rcon` reuse the S-box lookup; once per key.
- New claim type `AesBlockClaim` (or composed S-box/XOR/MixColumns sub-claims) + Rust
  `compile_aes`.

### 4.2 SHA-256 (per 64-byte block)
- `σ0/σ1/Σ0/Σ1` (rotate+shift+XOR), `Ch`/`Maj` (AND/XOR) on 32-bit words → XOR/AND 2→1
  lookup tables on byte/16-bit limbs; rotations/shifts are bit-decomposition rewiring
  (reuse `word_extract` + range checks). 32-bit modular adds use one range-checked carry
  word each (standard limb-addition gadget).
- New claim type `Sha256BlockClaim` + Rust `compile_sha256`.

### 4.3 Padding / packing — must match the external recorder bit-exactly
- **Token→bytes:** bit-decompose each committed token (range-checked) and repack into the
  exact byte layout the recorder used (endianness, packing). A mismatch silently breaks
  (B1_s). Applies identically to input and output streams.
- **SHA-256 padding** (`0x80`, zero pad, 64-bit big-endian length) and **AES mode/padding**
  are fixed public structure compiled into the constraints.

### 4.4 Integration with the rest of the proof
Copy/equality linear constraints link:
- each `EmbeddingLookupClaim` prompt-token input  →  the bytes feeding `AES(·, key_in)`;
- each `MaxClaim`/`InfoFinalizeClaim` output `tok` →  the bytes feeding `AES(·, key_out)`.

Without this wiring the binding would attest to a *different* token set than the one
actually used/bounded — a soundness hole — so each link is mandatory and gets its own
negative test.

## 5. Cost — negligible vs. the LLM proof

The binding runs over the **tokens (~KB), not the GB of activations/weights**. Cost scales
with the **total** token count (input + output); the input often dominates (a long prompt
≫ a short completion), but it is still kilobytes. Rough accounting (order-of-magnitude;
**TODO: validate against an implementation**), for an example transcript of ~few-thousand
tokens (~8 KB):

| Component | Size | Est. constraints/lookups |
|---|---|---|
| AES-128 over input+output tokens (~8 KB → ~500 blocks) | ~160 S-box + ~160 XOR + MixColumns / block | ~few×10⁵ |
| SHA-256 over the ciphertext (~8 KB → ~130 blocks) | ~64 rounds of bit-ops + 32-bit adds / block | ~few×10⁵ |
| SHA-256 of the key(s), key schedule | — | ~10³ |
| **Total** | | **~10⁶** |

Compare the LLM forward proof: `W_R_p1 ≈ 2×10¹¹` slots for Maverick
(`design-feasibility.md §4.5`), ~10⁹–10¹⁰ for the Llama-2-7B 1000-token run. So binding
**both** streams is still **~5 orders of magnitude smaller — well under 0.01%** of the
proof; a 10× error in the estimate keeps it < 0.1%. Both prover work and the `O(√W)`
verifier work move negligibly. The binding is effectively free relative to the forward
pass, because you hash kilobytes of tokens, not the model. (SHA-256/AES are deliberately
"ZK-unfriendly" vs. a ZK-native hash, but at this data size the absolute cost is still
negligible.)

## 6. Open decisions (pin against the external recorder)
Most of these were settled on 2026-07-05 against the interlock protocol (v6) — see §9.
- ~~One combined record vs. separate input/output records~~ — **decided** (§9.1): one key
  per request/response pair (shared `H2` = `KEY_COMMIT`), separate `H1` per direction.
- ~~Hash~~ — **decided** (§9.2): SHA-256 at the packet level.
- ~~AES mode~~ — **decided** (§9.3): CTR, GCM-ciphertext-compatible; GHASH deferred.
- ~~Token packing~~ — **decided pending recomputation-design review** (§9.4): fixed
  4-byte little-endian units.
- **Where `H1/H2` are recorded and by whom** — instantiated by the interlock design
  (§9.1), still the load-bearing trust assumption to state wherever the bound is reported.
- **Per-packet CTR counter-offset rule** for multi-packet streams — open (§9.4), pin
  jointly with the recomputation design.

## 7. Implementation sketch (superseded by the §12 roadmap; kept for the gadget order)
1. `AesSbox`/XOR `paired_tlookup` tables + a `word_extract` byte decomposition of tokens;
   negative tests (bad S-box entry, bad XOR).
2. `AesBlockClaim` + `compile_aes` (one block, then chained); honest-ACCEPT / cheat-REJECT.
3. `Sha256BlockClaim` + `compile_sha256` (message schedule + 64 rounds + carry adds).
4. Wire (B1_s)/(B2_s) for **both** streams: feed committed `tok_in`/`tok_out` + key bytes
   through AES→SHA, equate to public `H1_s`/`H2_s`; add the §4.4 copy constraints and their
   negative tests.
5. Document the new claims in `CLAIM_SPECS.md` (claims-in-sync rule) + Rust-parity coverage.

## 8. Relation to prior work
SHA-256 and AES are among the most-implemented ZK circuits (circom/circomlib, gnark std,
arkworks, halo2, Noir). We reuse the **designs and gate-count intuition**, not the code
(different proof system). S-box-as-lookup and XOR/AND-as-lookup is standard in
lookup-argument systems and is a direct fit for this codebase's `paired_tlookup` +
range-check machinery. The closest external references (R1CS = linear + Hadamard, so
directly comparable): Reclaim Protocol's `zk-symmetric-crypto` (circom AES-128/256-CTR
built for the prove-key-ownership-of-a-transcript setting) and its `gnark-symmetric-crypto`
port, whose AES uses gnark's lookup tables — the same S-box/XOR-table structure as §4.1;
`Electron-Labs/aes-circom` (AES-GCM) and `crema-labs/aes-circom` (FIPS-197) for round
structure and test vectors; circomlib's SHA-256 for the bit-decomposition baseline
(~20-30k R1CS constraints per compression — our LogUp-limb version should land well
under); and Plonky3's `blake3-air`/`keccak-air` for ARX hashing as low-degree AIR
constraints over a small field.

## 9. Decisions (2026-07-05) — alignment with the interlock protocol (v6)

Pinned in discussion against the interlock "Logs, Certificates, and Recomputation" (v6)
design. Direction confirmed: *"the headline run currently should be alright if either the
input or output tokens are revealed. But in the eventual design we want both of them
hidden (and for the hiding to work by checking that Hash(AES(key, tokens)) ==
public_token_hash, and that hash(key) == public_key_hash)."*

### 9.1 Root of trust instantiated
`H1_in`/`H1_out` are the interlock-certified per-packet `pld_digest` values (leaves of
the bucket/certificate digest tree); `H2` is `KEY_COMMIT`, carried in the input packet
header at request time — committed **before the response exists**, which is what stops a
key chosen retroactively to fit a covert payload. One key per request/response pair (v6
spec decision 1): shared `H2`, separate `H1` per direction. This resolves §3's "who
records `H1/H2`" for the interlock deployment; the assumption to state is now "the
interlock certificate chain is sound."

**Key-material layout (pinned at P0, flag for review):** `key (16) || iv_in (12) ||
iv_out (12)`, 40 bytes, hashed as-is for `H2`. The per-direction IVs must be
**distinct**: one key covers both streams (v6 decision 1), and under CTR a shared
`(key, IV)` would reuse keystream across the request and response, leaking the XOR of
the two token streams. The reference recorder rejects `iv_in == iv_out`.

### 9.2 Hash = SHA-256, packet level only
v6 spec decision 2: `pld_digest = SHA-256(PAYLOAD)`, one gateware core. Only this level
enters the circuit: the proof shows `SHA256(ciphertext) = pld_digest` for the challenged
packets. Bucket digests, certificate digests, and the HMAC `AUTH_TAG` are recomputed by
the verifier outside the proof from the opened log slice — they never need gadgets.

### 9.3 AES mode = CTR; GCM-compatible; GHASH deferred
If the recording side runs AES-GCM, the circuit still only needs AES-CTR: GCM's
ciphertext **is** CTR output with counter blocks `inc32^i(J0)`, `J0` derived from the IV
(for a 96-bit IV, `J0 = IV ‖ 0^31 ‖ 1`), so an AES-CTR gadget reproduces GCM ciphertext
exactly given the IV, which rides in the committed key material behind `H2`.

The 16-byte GCM tag needs an explicit convention, because a SHA-256 preimage cannot be
*partially* explained — the `H1` binding is all-or-nothing:

- **Preferred:** define `PAYLOAD` = ciphertext only, tag excluded from `pld_digest` (or
  not transmitted). Nothing in the v6 design ever verifies a GCM tag — integrity comes
  from the certificate chain — so the tag is dead weight in the preimage.
- **Fallback** (if raw GCM records must be hashed as-is): the tag bytes enter the
  preimage as unconstrained witness bytes, charged to `U` at 128 bits/packet — an
  honestly-priced covert channel — until a GHASH gadget (GF(2^128) via 64-bit limbs)
  closes it. Deferred either way.

### 9.4 Token serialization = fixed 4-byte little-endian units
v6 spec decision 3 requires position-addressable fixed-width token units. Unit `i`
occupies bytes `[4i, 4i+4)`; `PLD_LEN ≡ 0 mod 4`. Four bytes rather than three because 4
divides the 16-byte keystream block, so **no unit ever straddles an AES block** — the
same serialization then serves the ZKP's byte decomposition, recomputation Option 1's
per-unit `(ciphertext_unit → probability)` scoring tables, and clean chunk boundaries for
multi-packet outputs. Any vocabulary plus control/EOS ids fits in `2^32` (completion
markers are ordinary payload tokens per v6, so the encoding needs no special cases).
3-byte units (`2^24 > 202,048`) would save 25% of wire bytes but straddle block
boundaries; rejected at these sizes. **Open:** the per-packet CTR counter-offset rule for
streams spanning packets (offset by stream byte position vs. per-packet IV from the
committed key material) — pin jointly with the recomputation design, which is still in
flux.

## 10. Token-side selection: one committed integer per token (2026-07-05)

The committed token integer `t_i` is the single interface between the model side and the
wire side of the proof:

```
embedding select  <--  t_i  -->  4-byte decomposition  -->  AES-CTR  -->  SHA-256  -->  H1
(input stream)         |
MaxClaim select   <----+
(output stream)
```

Every arrow is a copy/equality constraint on shared witness slots; each link gets its own
negative test (§4.4).

### 10.1 Input: token -> one-hot -> `M @ E` (Freivalds)

Decided: keep the one-hot-times-committed-embedding-matrix path (what the headline run
already used for the hidden prompt, `demo/demo_maverick_full.py` `build_inputs`), with
two changes:

1. **A slim one-hot select claim replaces the `RoutingClaim` reuse.** The current hidden
   path enforces one-hot via `route_top1`, dragging in the argmax machinery (word
   decompositions for the gap comparisons — about `10·S·V` witness slots; measured
   ~2-3e9 slots for the 442-token hidden prompt). A bare select needs only:
   - booleanity: `M ∘ M = M` (per-slot quadratic, `S_h·V`),
   - cardinality: `Σ_j M[i,j] = 1` (one linear per token),
   - **index binding:** `t_i = Σ_j j·M[i,j]` (one linear per token).
   This is exactly `MaxClaim`'s output-select gadget `O` (`prover/max_claim.py`: `O·O=O`,
   `Σ O = 1`, `tok_t = Σ_i i·O[t,i]`) instantiated on the input side — booleanity +
   cardinality + the index linear make `M[i]` *uniquely* the one-hot of `t_i`. Cost drops
   to about `2·S_h·V` (~4.4e8 slots at S=1093, V=202k — a few percent of the proof).

2. **The index-binding linear is new on the input side and mandatory.** The current
   hidden path deliberately never represents the token as a number (any one-hot row is
   "some token"). With wire binding, `t_i` must exist as a committed integer so the same
   slot feeds the AES byte decomposition. Without it the binding attests to a different
   token set than the one embedded — the §4.4 soundness hole.

Then `x_h = M @ E` as today: a scale-free `MatmulClaim` (Freivalds — three linears + one
quadratic, projections of length V). **`E` is already committed** — it is a model weight
like the V×d LM head; the one-hot path adds no commitment cost.

Public segments (deployments that reveal one side) keep `EmbeddingLookupClaim` with the
d=1024 sub-row trick, unchanged.

### 10.2 Output: already in place

`MaxClaim`'s `O` gadget already commits `tok_t` with the index-binding linear. The only
change is wiring: the `tok` slots become the same witness variables the output-stream
byte decomposition consumes (§4.4 copy constraints).

### 10.3 Deferred alternative: challenge-compressed LogUp row read

If hidden-input length ever makes `S·V` a leading term (≳10k hidden tokens → ≳4e9
slots), the standard upgrade is: after `E`, `x_h`, and tokens are committed, the verifier
sends ρ; the prover commits `r = E · (1, ρ, ..., ρ^{d-1})` (one Freivalds matvec, V
auxiliary slots); each token proves `(t_i, r_{t_i})` by paired LogUp against the
committed table `{(j, r_j)}`; a public-coefficient linear checks `Σ_d ρ^d·x_h[i,d]`
equals the looked-up value (Schwartz–Zippel error `d/|F| ≈ 2^-51`). ~100× cheaper at
Maverick scale, fits the four-round structure (ρ lands with the Freivalds challenges) —
but requires LogUp over a *committed* table (all current tables are public/static), a
real machinery lift. Not needed at current scale; revisit with long-context work.

## 11. Claim inventory

New claim types (prover `claims.py` + Rust `handlers.rs` + `CLAIM_SPECS.md` + negative
tests, per the claims-in-sync rule):

| claim | scope | constraints | reuses |
|---|---|---|---|
| `OneHotSelectClaim` | per hidden-input stream | booleanity quads, cardinality + index linears | `MaxClaim` O-gadget pattern |
| token-byte decomposition | per token, both streams | `t_i` -> 4 LE bytes, range-checked | `WordExtractionClaim` / `RangeWordClaim` |
| `AesCtrClaim` | per 16-byte block + key schedule once per key | S-box/xtime/XOR paired lookups, §4.1 | `PairedTlookupClaim` |
| `Sha256BlockClaim` | per 64-byte block | limb XOR/AND lookups, carry adds, §4.2 | `PairedTlookupClaim`, range tables |
| stream wiring | per stream | copy/equality: tok slots, ciphertext bytes -> hash input | linear constraints |

New static tables (public, both sides compile them): `SBOX` (256 paired), `XTIME` (256
paired), `XOR8` (2^16 paired, `key = a·256 + b -> a XOR b`), and a 16-bit-limb table for
SHA-256's `Ch`/`Maj`/σ if not decomposed to bytes. All are LogUp tables of the existing
kind; the 2^16 size matches `range_slack`.

Public inputs added to the claim list: `H1_in`, `H1_out`, `H2`, the IV/counter-offset
convention, `PLD_LEN`, and the fixed serialization (§9.4). The verifier compiles all
constraints from these — nothing prover-supplied.

## 12. Roadmap

Ordering principle: SHA-256 lands before AES because `Hash(key) = H2` (B2) is the
smallest end-to-end slice — one hash over 32-48 bytes of committed key material, no
AES, no per-token wiring — and it exercises the whole new pipeline (tables, claim,
handler, negative tests, dump format) at minimum size.

Bit-exactness gates throughout, per the standing discipline: every phase difftests the
Python witness generation against library references (`hashlib.sha256`,
`cryptography`/`pycryptodome` AES-CTR) on random vectors, and the Rust verifier is the
oracle — a phase is done when honest proofs ACCEPT and its targeted tampers REJECT.

- **P0 — freeze the byte-level spec (1-2 days).** Write the reference recorder: a
  standalone Python script that takes `(tokens_in, tokens_out, key material)` and emits
  `(H1_in, H1_out, H2)` exactly as the interlock frontend would — 4-byte LE units, CTR
  counter layout (`inc32^i(J0)`, `J0` from the IV), PAYLOAD = ciphertext-only (§9.3
  preferred branch), SHA-256 padding. Cross-check against `cryptography` +
  `hashlib`. This script *is* the spec the circuit must match and the test-vector
  source for every later phase; the §9.4 open CTR-offset rule gets pinned here (or
  explicitly stubbed single-packet).
  **DONE 2026-07-05:** `prover/ref/token_recorder.py` (pure-Python FIPS-197 AES with a
  generated S-box, GCM counter layout, 4-byte LE units, the §9.1 key-material pin) +
  `prover/tests/test_token_recorder.py` (8 gates: FIPS-197 Appendix C vector,
  byte-identical-to-AES-GCM cross-check against the `cryptography` library,
  keystream-separation, spec-drift gate against checked-in vectors) +
  `prover/tests/vectors/token_binding_v0.json` (4 deterministic vectors with
  ciphertext intermediates for the P1-P5 difftests). Multi-packet CTR offset:
  stubbed single-packet, still open with the recomputation design.
- **P1 — tables and byte plumbing (2 days).** `SBOX`/`XTIME`/`XOR8` tables; token ->
  4-byte decomposition via `WordExtractionClaim`; negative tests (bad S-box entry, bad
  XOR, out-of-range byte).
  **DONE 2026-07-05:** `prover/token_binding.py` (`register_binding_tables` +
  `token_bytes`; table values generated from the P0 recorder — single source of truth;
  no new claim types or Rust surface, tables settle via the generic TableSettlement) +
  `prover/tests/test_token_binding_tables.py`, 5/5 on the Spark with the independent
  Rust verifier: byte decomposition == `serialize_tokens` on the P0 vectors (ACCEPT),
  sbox/xtime/xor8 lookups (ACCEPT), oversized-token / off-table-byte / off-table-key
  cheats all REJECT. Two notes: the paired-lookup compute path was already
  cheat-hardened (clamped gather + in-range-only multiplicity counting); and a POLICY
  obligation is recorded for P5 — the verifier checks consistency against the claim
  list's table data, so the deployment policy must pin the table *contents* to the
  real AES tables, as it pins model structure.
- **P2 — SHA-256 (3-4 days).** Message schedule, 64 rounds, carry adds;
  single block then multi-block + padding; land **B2 (`Hash(key) = H2`)
  end-to-end** in a unit-scale proof; tamper suite (wrong padding, wrong carry,
  wrong limb, wrong `H2`).
  **ARCHITECTURE PINNED 2026-07-05 (TCB-minimal, per maintainer direction):**
  no `compile_sha256` in the verifier. The ONLY new TCB surface is one tiny
  generic claim, `LinCombClaim` (`Σ_k coef_k · x_k[i] = rhs[i]` over aligned
  vars with a public per-slot RHS — `add_compile` generalized, ~20 lines of
  Rust, reusable by the P3 AES key/byte compositions). Everything else
  composes existing Rust-handled claims: `word_extract(B=1, N=32)` for bits
  (booleanity = the [0,2) range check), raw hadamard for the XOR/AND
  products, `ConcatClaim` double-definition for the lag slices (b,c,d,f,g,h
  are shifted views of single 68-slot A/E histories with 4 public IV prefix
  slots — no committed copies), small range tables for carries, reveal pins
  for the prefixes. Σ/σ/Ch/Maj values are never committed — they fold into
  the round-add `LinCombClaim` rows as linear expressions over bits and
  products.
  **Step 1 DONE 2026-07-05:** the constraint layout is frozen and
  machine-checked in `prover/ref/sha256_trace.py` — trace generator +
  exhaustive constraint checker, gated by `tests/test_sha256_trace.py`
  (digest == hashlib on the P0 key materials; reference-level B2: trace
  digest == the recorder's `H2`; and a 25-class tamper fuzz proving no
  unconstrained slots — which already caught one real layout gap: schedule
  XOR products outside the constrained word ranges [1,49)/[14,62) were
  free slots and are now not committed).
  **Step 2 DONE 2026-07-05:** `LinCombClaim` landed end-to-end — the one TCB
  change (~25 lines in the Rust `compile_op` dispatch, mirroring
  `lincomb_compile`), 5/5 gates on the Spark against the independent Rust
  verifier (per-slot and constant RHS ACCEPT; slot-lie and wrong-public-RHS
  cheats REJECT), regressions green (reveal 2/2, test_claims 21/21).
  **P2 COMPLETE 2026-07-05.** Reference layout extended (b/c/f/g bit classes
  with booleanity + recomposition; committed XOR intermediates x12/y12/tbc/
  xm/xn pinned by x=u+v-2p; message as four byte-position strides, padding
  public; public digest) and re-fuzzed to 35 classes. `sha256_h2_gadget`
  (token_binding.py) composes it in 965 claims from existing types only —
  ConcatClaim spine partitions for the A/E index-shifts, the two round adds
  and schedule add as one LinComb each with Σ/Ch/Maj folded into rotation
  coefficients. **B2 end-to-end 6/6 on the Spark** vs the Rust verifier:
  positive (P0 key material + recorded H2) ACCEPT; wrong-H2 / tampered-byte /
  bumped-carry / flipped-bit / tampered-word all REJECT.
- **P3 — AES-128-CTR. COMPLETE 2026-07-05.** No `compile_aes` — same
  TCB-minimal approach (only LinCombClaim). Byte-level, lookup-based:
  `prover/ref/aes_trace.py` freezes the layout (trace + checker + EXHAUSTIVE
  per-slot tamper fuzz, all ~2300 slots/block; no layout gaps found), plus a
  gadget-facing `sites()`/`pool_layout()` API gated to reproduce the checker
  exactly (test_aes_trace 7/7). `aes_ctr_gadget` composes the whole cipher
  into **31 claims**: every committed byte class is one range-checked vector
  concatenated into a gather pool; `EmbeddingLookupClaim(d=1, public ids)`
  gathers operands/outputs; the cipher is three paired lookups (XOR8 keyed by
  k=256a+b, SBOX, XTIME — one claim each over the full site vector) + the key
  linear + output-equality LinCombs + public counter/Rcon pins. Key schedule
  committed once per key; GCM counter layout reproduces GCM ciphertext.
  **P3 end-to-end 7/7 on the Spark** vs the Rust verifier: 1-block and
  3-block-partial positives against the recorder ciphertext; wrong-ct / key /
  plaintext / MixColumns-mid / IV cheats all REJECT.
- **P4 — slim one-hot select (2 days).** `OneHotSelectClaim` per §10.1; swap the hidden
  path in `demo_maverick_full.py` off `route_top1`; add the input-side index-binding
  linear; share the output `tok` slots with `MaxClaim`; negative tests (two-hot,
  index-mismatch, wrong embedding row).
- **P5 — integration (3 days).** B1+B2 for both streams in the 7B pipeline
  (`ui_real_proof.py` + the reference recorder from P0 supplying `H1/H2` as public
  inputs); extend `demo/gate.py` so the regression gate covers the binding; end-to-end
  ACCEPT plus one tamper per §4.4 link (swap a bound token, swap a key, swap a
  ciphertext byte).
- **P6 — audit checkpoint A-TB1 (2 days).** Independent agent per the audit procedure:
  transcript completeness, verifier coverage, emission-order parity, negative-test
  coverage for every new check — plus a degrees-of-freedom pass over the new claims
  under the bracket rule from the 2026-07 review (*every* operand of every new gadget
  independently range-bound; the RMSNorm-rsqrt lesson).
- **P7 — full-model wiring (later, with the interlock demo).** Maverick-scale run with
  the binding on; interlock-challenge integration (the `demo/plaintext-proof` branch's
  public-output mode becomes "reveal one side" per §9); GHASH gadget if the
  tag-in-preimage fallback is ever needed.

Total through P6: ~2.5-3 weeks at recent cadence, matching the earlier estimate; ~1e6
constraint rows at 7B scale (§5), invisible in prover time. Prerequisite carried over
from `full-model-v1-design.md` before *publicly* claiming hiding: the blinding audit
(verify `encode_messages` fills the `K_DEG = 2·ELL` slack with fresh randomness).

When implementation starts, mirror the P0-P7 checklist into the agent-plans repo per the
plan-tracking workflow.
