# S4f/S5 engineer handoff (2026-08-06)

Read `analysis/routed-projected-status.md` first. It remains the authority.
This patch fixes the two measured S5a failures and two fail-late state bugs,
but S4f is deliberately `[~]` until the CUDA and Rust gates below pass on the
rented machine. Do not start the 400B proof before that and before the real
GGUF stages have an admissible report.

## What changed

1. **Quadratic accumulator (`prover/core.py`).** The quadratic polynomial is
   unchanged. The prover now accumulates the `r_quad`-weighted products in the
   2K evaluation domain and runs one inverse NTT at the end. The old code ran
   an inverse NTT for every constraint row. Repeated public constant rows are
   interpolated once per distinct `(n, coefficient)`. The independent gate is
   `test_quad_eval_accumulator`.
2. **Proof transport (`prover/proof_dump.py`, Rust `verify_proof`).** Production
   vectors are JSON strings `"u64le:<base64>"`; legacy decimal arrays remain
   accepted. Both forms reconstruct the same `Vec<u64>`, so transcript and
   verifier equations are unchanged. The integration gate uses the new form.
3. **Durability.** The output file is reserved with `posix_fallocate` before
   proving. One process holds an exclusive enrollment lock. After proving the
   cumulative opening ledger is atomically saved and fsynced *before* the proof
   is atomically published and the parent directory fsynced. A failed publish
   can conservatively spend columns; it cannot publish unrecorded leakage.
4. **Admission statistics.** NaN, infinity, negative bounds, missing raw
   samples and bounds below their raw observed maxima reject. Thirty samples
   are exploratory only. A distribution-free stagewise-max p99 bound with 99%
   simultaneous confidence over 13 stages requires 714 iid samples under the
   current Bonferroni method. The gate now enforces that fact rather than
   claiming that 30 samples suffice.

The model is now **12,957.864 s = 3.5994 h**, margin **1,442.136 s**. This is
an envelope, not a measured result. The historical A6000 report has a different
source digest and cannot authorize this build.

## 1. Build and correctness gates

Run from the repository root on a CUDA machine:

```bash
cd verifier
cargo build --release
cargo test --release --bin verify_proof
cd ../prover
python tests/run_tests.py test_quad_eval_accumulator
python tests/run_tests.py test_opening_ledger
python tests/run_tests.py test_admission_gate
python tests/run_tests.py test_pipeline_integration
python tests/run_tests.py test_fiat_shamir
python tests/run_tests.py test_phase3_block
python tests/run_tests.py test_routed_projected
python tests/run_tests.py test_rescale_claim
python tests/run_tests.py test_moe_routed
python tests/run_tests.py test_shard_streaming
```

Then run the complete remote gate list:

```bash
PROOF_EGRESS_DIR=/the/production/proof/filesystem \
ADMISSION_RUNS=30 bash analysis/bench/admission_remote.sh
```

The 30-run campaign is intentionally exploratory. It answers whether the two
new implementations clear their caps; it does not create an admission report.

## 2. Required S4f performance outcomes

The campaign must report:

| Stage | Hard cap | Equivalent rate | Old A6000 baseline |
|---|---:|---:|---:|
| `quadratic` | 765.0 s | <=17.0 ns/product over 45B products | 800.2 s / 17.78 ns |
| `proof_egress` | 879.63 s | >=59.11 MB/s for the 52GB cap | 979.8 s / 97.0 MB/s, but decimal wire |

The old egress seconds and rate are not internally comparable to the compact
wire because the modeled byte volume also changed. Trust the new stage-seconds
line. Do not change either cap to make a result green.

For the quadratic gate, `test_quad_eval_accumulator` must pass before timing is
considered. It compares the optimized result with the literal old polynomial,
including a nonzero public `b` row and a partial row. A speedup with a mismatch
is a failed protocol implementation.

## 3. Real-GGUF admission work

After S4f is green, obtain the exact GGUF and generate the trusted enrollment.
Measure the two missing dominant terms on that same source tree, model root,
statement/layout and machine:

- `model_load <= 400 s`;
- all five active-only semantic sweeps together `<= 3609 s`, including actual
  quantized decode/page movement (at least 27.278G decoded MAC/s aggregate).

The final `admission.json` schema additionally requires `stage_samples`: one
raw seconds list per priced stage. `stages[name]` must be at least the observed
maximum. Under the current nonparametric claim every list needs at least 714
iid measurements. If that campaign is impractical, change the *stated
statistical theorem* and document a justified alternative; do not fabricate
714 entries or relabel 30 samples.

The report is also bound to GPU UUID/driver/PCI id/power limit, hostname and
the output filesystem identity. Benchmark egress on the filesystem that will
actually receive `--dump-proof`; `/tmp` is not a valid proxy for another disk.

The kernel-only benchmark output is intentionally incomplete
(`weights=none_kernel_only`, missing model stages) and must remain rejected by
the production driver.

## 4. Production sequence after every gate is green

Use the exact commands in `demo/4h-production-runbook.md`. Additional operational
requirements introduced here:

- `--dump-proof` is mandatory on target runs;
- at least 64GB must be preallocatable in the output filesystem;
- no stale `<proof>.part` may exist;
- only one prover may use a `.wcommit` at a time (the `.lock` file enforces it);
- if proof publication fails after proving, the ledger remains advanced. This
  is deliberate; refresh or accept the conservatively spent columns rather
  than rolling the handle back;
- verify the resulting compact proof with the required external model root and
  statement digest. The Rust verifier must print `ACCEPT`.

## 5. Failure triage

- **Quadratic still over cap:** profile the two forward NTTs, `gl_mul`,
  `gl_matvec`, and the single inverse NTT separately. Do not restore per-row
  inverse transforms. Next safe optimization is fusing product and weighted
  reduction in the evaluation domain.
- **Egress still over cap:** separate base64 CPU time from filesystem write and
  fsync. The field wire must remain canonical u64le; a framed binary container
  is acceptable only with an independently tested Rust decoder and the same
  reconstructed values.
- **ENOSPC at startup:** move output to a filesystem that can genuinely
  `posix_fallocate` 64GB. Sparse `ftruncate` is intentionally rejected.
- **Stale `.part`:** inspect why the previous process died, then remove it
  manually. Never delete or roll back the `.wcommit` ledger to reclaim proof
  budget.
- **Admission digest mismatch:** rebuild/rebenchmark. Never edit the report's
  digest or machine fields.

## Local validation performed in this handoff environment

- all changed Python files compile;
- the 1,000,013-element multi-chunk compact-wire roundtrip is exact;
- preallocation creates the requested real file size;
- the model runs and prints 12,957.864 s;
- `git diff --check` passes.

CUDA, PyTorch, Cargo and the Rust toolchain were unavailable in the handoff
container. Therefore no GPU/Rust gate is claimed as passed here; those commands
above are the load-bearing next action.
