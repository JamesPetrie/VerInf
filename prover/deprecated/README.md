# deprecated/ — archived, not maintained, not tested

Code here is kept for reference and possible future revival. It is **not** part
of the maintained pipeline, is not run by the test suite, and may reference
pre-refactor APIs. After the prover-interface consolidation:

- the flat, non-streaming `prove(claims, inputs, ...)` moved to
  `tests/test_prover.py` — it is the unit/negative-test harness, not production;
- `prove_streaming_sound(...)` was merged into `prove_streaming(..., sound=True)`;
- `Tape.prove` is streaming-only.

So references to `core.prove` / `core.prove_streaming_sound` in these files are
stale. The single production prover is `core.prove_streaming` (see `Tape.prove`).

## Contents

- **Staged/interactive prover stack** — deferred until rebuilt on the streaming
  engine (the current streaming prover is monolithic, not yet stageable):
  `tape_prover.py`, `prover_server.py`, `test_verify_real.py`.
- **Cross-language difftest fixtures:** `dump_proof.py` (Rust foundation difftest);
  the compile-parity difftest was retired here too (P5c) — see below.
- **Benchmarks / one-off checks:** `bench_*.py`, `check_rescale_pyverify.py`.
- **Broken/heavy regression:** `test_pipeline_production.py` (stale
  `from test_pipeline import …`; ran at production scale).
- **The old standalone Python verifier (fully retired, P5).** The Rust `verifier-rs`
  is now the single independent verifier; every test verifies through it
  (`tests/_rust_verify.py` → `verify_proof`), and end-to-end ACCEPT is the
  bit-exactness check. The Python verifier it was bit-exact-ported from lives here:
  - `verify.py` — the six checks (plain-int Ligero verifier) + `test_field.py`,
    `test_column_checks.py`.
  - `python_verifier_compile.py` — the constraint compile (`REAL_COMPILE` /
    `compile_claims` / the `expand_*`/`_c_*`/`_emit_*` machinery), **extracted from
    `protocol.py` in P5c**. `protocol.py` now keeps only the byte-exact
    prover↔verifier contract (field, domains, challenge PRF, `claims_to_json`).
  - the compile-parity difftest that drove it: `test_compile_parity.py`,
    `dump_compile_parity.py`, `test_expanders.py`, `test_rope_straddle.py`.

  This Python oracle had drifted (missing Max / InfoFinalize / PairedTlookup), which
  is partly why it was retired rather than kept in sync. The Rust `compile_difftest`
  bin that consumed it is now dead (no oracle) — to remove with the other dead bins.
  (`core.verify`, the in-process GPU co-simulation, was deleted outright in P5b.)

## Added 2026-10-07 (dead-code sweep)

- `tamper_proof.py` — read `/tmp/proof.json`, which nothing writes any more;
  the negative tests now tamper through `tests/_rust_verify.py` and the
  claim-level tamper hooks.
- `dump_routing_proof.py` — hand-serialized the legacy proof JSON; superseded
  by `proof_dump.dump_proof` and the demos' `--dump-proof`.
- `difftest_lde_skip.py` — a one-off byte-identity check with a hard-coded
  `/home/riftuser` path (routed-projected-status.md records its result).
- `moe_routed.py` (from `demo/`) — the first routed MoE FFN builder; nothing
  imported it. `demo_maverick_full.build_moe_ffn` is the live one (adds
  `use_bridge` and `persistent`).
