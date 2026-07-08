# Handoff: porting all packet kinds to CUDA (q_lin fold)

Read this with `qlin-fold-reorg-plan.md` (the why + phases) and
`qlin-fold-kernel-bench-results.md` (the dispatch findings). This doc is the
**cold-start runbook**: where things live, how to run, the exact recipe to add a
family, and what's left.

## Status: 13 of 14 expanders ported (all bit-exact on GB10); EmbedE deferred

| kind | kernel | dispatch | ns/slot (per-slot) |
|---|---|---|---|
| `L2_FreivaldsLF1B` | per-slot + warpcid + **precompute** | precompute | 0.06 (precompute) |
| `L2_IdentityScalar` | per-slot + precompute | per-slot | 0.70 |
| `L2_StrideOneToManyScalar` | per-slot | per-slot | ~0.7·stride |
| `L2_FreivaldsLF2A` | per-slot + **precompute** | precompute | 0.061 (precompute) |
| `L2_FreivaldsLF3C` | per-slot + **precompute** | precompute | 0.061 (precompute) |
| `L2_StrideManyToOneScalar` | per-slot + **precompute** | precompute | 0.14 (precompute) |
| `L2_PerSlotVector` | per-slot | per-slot | 0.70 |
| `L2_RowSumPerSlotVector` | per-slot + **precompute** | precompute | 0.147 (precompute) |
| `L2_RoPEXRot` | per-slot | per-slot | 0.68 |
| `L2_TransposeO2MScalar` | per-slot (fan-sum) | per-slot | 9.74 (~0.6·fan) |
| `L2_CausalFilteredIdScalar` | per-slot + mask | per-slot | 0.39 |
| `L2_CausalFilteredC2Stride` | per-slot (ragged sum) | per-slot | 0.94 † |
| `L2_RoPEX` | per-slot (2 cids) | per-slot | 1.35 |

All five high-reuse families (LF1B / LF2A / LF3C / StrideManyToOne / RSV) now have a
bit-exact **precompute** (gather) variant — the recommended dispatch. The
weight-Freivalds families (LF2A/LF3C, where a cid spans 10^5+ slots) hit the ~0.06
bandwidth floor; the stride-16 families (StrideManyToOne/RSV) reach ~0.14 (~5× over
per-slot). Identity also has a `gather` variant, but 1:1 families don't benefit, so
per-slot stays their dispatch.

The per-slot ns/slot for LF2A/LF3C/StrideManyToOne/PerSlotVector/RowSumPerSlotVector
in the table above were measured this session; `qlin-fold-kernel-bench-results.md`
currently records only the LF1B/Identity/stride numbers.

**Deferred (1 expander):** `L2_EmbedE` is **not** being kernelized. It is cold (one
input-binding claim, `SEQ·d` linear constraints, ~3 ms, <1% of the fold) and routes
to the legacy `expand→sort→spmv` dispatch — bit-exact, and it does not block
end-to-end testing. It must eventually be *replaced* (not just ported): the current
form assumes a publicly-shared / subset-committed embedding, and a deployable proof
can't assume the prover shares the embedding matrix — see the per-kind note. `†` C2's
ns/slot is a correctness gate, not perf-representative (only `B=SEQ·H` slots active).

## Where things live

- **The bench is on branch `feat/qlin-fold-kernel-bench`, NOT main.** Start with
  `git -C /root/infproof checkout feat/qlin-fold-kernel-bench`. The file is
  `pipeline/qlin_fold_bench.py` (~900 lines, self-contained: a separate inline
  CUDA module + a synthetic-chunk + bit-exact driver per family).
- Plan + cost-model docs are on `main` under `analysis/`.
- The packet expanders you transcribe are in `pipeline/packets.py`
  (`EXPANDERS` registry); the challenge device fn is `cuda/gl_spmv.cuh`
  (`gl_sparse::challenge_inline`); field ops in `cuda/goldilocks.cuh`.

## Environment / runbook (there is NO local GPU)

Kernels JIT-compile for `cuda`, so everything runs on the **Spark** (GB10,
sm_121) over Tailscale. From `/root/infproof`:

- **Sync the bench to the Spark** (one line):
  `cat pipeline/qlin_fold_bench.py | tailscale ssh claude@spark-c191 'bash -l -c "cat > ~/ligero/pipeline/qlin_fold_bench.py"'`
  The bench imports `packets` + `cuda_primitives` and `#include`s
  `cuda/{goldilocks,gl_spmv}.cuh`. On a **cold start** — or whenever the expander
  or header you need isn't on the Spark yet — sync those too:
  `tar -C . -cf - pipeline/packets.py pipeline/cuda_primitives.py cuda/*.cuh | tailscale ssh claude@spark-c191 'bash -l -c "tar -C ~/ligero -xf -"'`
- **Run / gate a family** (one line; `ninja` must be on PATH, and it's installed
  in the venv):
  `printf 'cd ~/ligero/pipeline && PATH=$HOME/venv-hf/bin:$PATH ~/venv-hf/bin/python qlin_fold_bench.py --family lf2a --H 8 --variant all\n' | tailscale ssh claude@spark-c191 'bash -l'`
- The Spark layout mirrors the repo: `~/ligero/{pipeline,cuda}`. Python is
  `~/venv-hf/bin/python` (only that interpreter has torch+CUDA). First compile
  per source change is ~60–90 s (cached after).
- The driver prints `OK`/`BAD` per variant (`torch.equal` vs the reference) plus
  ns/slot + GB/s. **`OK` = bit-exact = correct.**

## Recipe: add one family `FOO`

1. **Read its expander** `_expand_<foo>` in `packets.py` — extract the
   closed-form `(target = lr*ELL+s, cid, coef)` for a slot at
   `flat = (chunk_lo+lr - var_row_start)*ELL + s`, valid while `flat < L`/`total`.
2. **CUDA kernel** `k_<foo>_perslot` (insert before `static uint64_t* u64` in
   `_CUDA_SOURCE`): `idx → flat`, guard `flat >= total`, compute `cid`+`coef`,
   `r = gl_sparse::challenge_inline(seed,label,label_len,(uint64_t)cid)`,
   `out[idx] = gl::mul(coef, r)`.
3. **Launcher** `<foo>_perslot(...)` before the closing `"""` of `_CUDA_SOURCE`
   (copy an existing one; `blk=256`, grid over `n_out`).
4. **Decl** in `_CPP_DECLS`; **name** in the `functions=[...]` list in `_module()`.
5. **Python**: import the dataclass + expander; add `<Foo>Chunk(_ChunkBase)`
   (build the packets + any coef tensors via `torch.randint(0, 2**62, …, device="cuda")`);
   `reference_<foo>(c)` = `_spmv_reference(*_expand_<foo>(...), c.n_out, c.seed, c.label)`;
   `v_<foo>_perslot(c)` calls the launcher.
6. **Dispatch**: add `<foo>` to the `--family` `choices=[...]` list in `main()`
   (argparse rejects an unlisted `--family` before the branch ever runs), then add
   the `--family <foo>` branch itself (chunk + ref + variants + header).
7. **Sync + run** `--family <foo>`; confirm `OK`. The reference is the existing
   `expand→argsort→spmv` path, so the gate catches any index-math bug.

## Gotchas (all hit during this work)

- **Negation is `gl::sub((uint64_t)0, x)`** — there is no `gl::neg` device fn.
- `torch.randint(0, 2**62, …, device="cuda")` — `P > int64 max` so `randint(0,P)`
  is invalid; `2**62 < P` is a valid field element. The generator is cuda, so
  `device="cuda"` is required or it errors.
- Pick shapes so `total`/`L ≥ n_chunk*ELL` (default 256·8192 = 2,097,152) so every
  chunk slot is valid; the `flat>=total` guard handles partials anyway.
- `torch::kUInt64` compiles fine in libtorch on this box.
- Keep the per-family `var_row_start`/`chunk_lo` at 0 in the synthetic chunk;
  `flat == idx` then, which makes hand-checking easy.

## Per-kind notes (EmbedE deferred; the rest are DONE, bit-exact)

- **`L2_TransposeO2MScalar`** — **DONE** (bit-exact, 9.74 ns/slot at fan=16,
  `--family transpose`). Fan-out, transposed (used by `masked_combine` G1): like
  `StrideOneToMany` but the `fan` cids are at transposed/strided positions —
  `cid_lo = base + (flat%cols)·rows·fan + (flat/cols)·fan`, output per *source*
  slot summing the `fan` challenges. Per-slot, no atomics.
- **`L2_RoPEXRot`** (`packets.py` 722–776): **DONE** — bit-exact, 0.68 ns/slot,
  `--family ropexrot`. Per-slot, one cid (`base + 2·pair_t + e_self`), coef 1, no
  mask, no atomics. Was the warm-up port.
- **`L2_RoPEX`** (`packets.py` 779–851): **DONE** (bit-exact, 1.35 ns/slot,
  `--family ropex`). The fiddly one (the handoff used to misattribute this to
  RoPEXRot — it's RoPEX). Each x cell contributes to **two** consecutive cids
  (`2·pair_t`, `2·pair_t+1`) summed into the *same* slot, with coefs `±cos`/`±sin`
  selected by `e_self` from the public `cos_t`/`sin_t` tables
  (`coef_idx = seq·half + k_in_pair`). Negate with `gl::sub((uint64_t)0, ·)`.
  Per-slot, 2 hashes/slot, no atomics; the chunk supplies `cos_t`/`sin_t`.
- **`L2_CausalFilteredIdScalar`** (`packets.py` 321–367): **DONE** (bit-exact,
  0.39 ns/slot, `--family causal_id`). Causal identity. Decode
  `b=flat//M, j=flat%M, i_qry=b//H, h=b%H`; **masked iff `j > i_qry`** → `out[idx]=0`.
  Otherwise `rank = H·i_qry·(i_qry+1)/2 + h·(i_qry+1) + j`, `cid = base + rank`,
  scalar coef. Per-slot **+ validity mask**, no atomics. Needs `M`, `H`.
- **`L2_CausalFilteredC2Stride`** (`packets.py` 370–417): **DONE** (bit-exact,
  0.94 ns/slot, `--family causal_c2`). **Not** a masked identity — a *ragged
  fan-out sum*: every c2 slot `b` is active (no mask), summing `fan_out = i_qry+1`
  consecutive challenges from `rank_start = H·i_qry·(i_qry+1)/2 + h·(i_qry+1)`:
  `out = coef · Σ_{k=0}^{i_qry} challenge(base+rank_start+k)`. Per-slot with a
  **variable-length loop** (warp divergence), no atomics. Bench caveat: `B=SEQ·H` is
  small, so only the first `B` chunk slots are active — correctness gate, not a
  perf-representative shape.
- **`L2_EmbedE`** (`packets.py` 420–484): **DEFERRED — do not kernelize.** Cold
  (one input-binding claim, `SEQ·d` linear constraints, ~3 ms, <1% of the fold);
  routed to the legacy `expand→sort→spmv` dispatch (bit-exact). **Eventual plan
  (required):** the current constraint assumes a publicly-shared / subset-committed
  embedding (public `token_ids` + only-the-used-rows commit; `demo_llama7b.py:385`
  flags the missing full-`E` Merkle anchor). A deployable proof can't assume the
  prover shares the embedding matrix, so the lookup must eventually be verified
  against a **fully committed, hidden** `E` — Freivalds `x = S·E` (public selection,
  hidden full `E`) or a committed indexed-gather — which moves cost from the cheap
  `L` term into `W`/`Q` and is a **claim+verifier redesign** (separate from this fold
  port; needs verifier support for a public-selection form). *If* it were kernelized
  as-is, the crux is that **repeated token ids collide** (each position `i` with
  `token_ids[i]=vocab_lo+rel_tid` adds `−challenge(base+i·d+j)` into the same slot
  `rel_tid·d+j`), needing either an inverse index (`vocab_row→[positions]`,
  atomic-free) or a field-safe scatter (`gl` atomic add).

## The precompute follow-up (the actual win for high-reuse families) — DONE

LF2A/LF3C/StrideManyToOne/RSV now have bit-exact **precompute** (gather) variants
(`v_lf2a_precompute` … + `k_*_gather`), the recommended dispatch. Each hashes the
distinct cids once with `challenge_range(seed,label,base,n_distinct)`, then a
`gather` kernel computes the cid-index + coef and reads `table[cid_index]` (no
inline hash). Measured (P2): LF2A/LF3C **0.061 ns/slot** (~11×, bandwidth-bound);
StrideManyToOne/RSV **0.14/0.147** (~5×, stride-16 reuse). LF1B's `precompute` +
`lf1b_gather` was the template (`v_precompute` / `k_lf1b_gather`); the
`n_distinct` per family is `k` (LF2A), `H` (LF3C), `ceil(L/stride)` (s2o/RSV). For
LF3C the gather still computes `coef = -lam[h*m+i]*rho[h*n+j]` per slot — only the
challenge is gathered.

## After all kernels exist (→ P2/P3 in the plan)

Once every family has a bit-exact kernel, move to `qlin-fold-reorg-plan.md`
P2 (the `ConstraintBand` dispatch interface — one band per `(variable, family)`,
`qlin-family-object-reorg.md` §1/§11) and P3 (prover integration — count-pass +
reverse-index, replace `expand→sort→spmv`, delete the lazy/late path, **with the
A-pre/A-post audit**). Field-safe atomics only become relevant in P3 (cross-family
overlap), not for these per-family bench kernels.
