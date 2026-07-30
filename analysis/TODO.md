# VerInf prover optimization — TODO

Pending work, ranked. Context in `analysis/SESSION-HANDOFF.md`, detail in
`analysis/bench/{research_journal.md,optimization_registry.md}`.

## Remote validation campaign (tooling BUILT, not yet run)
- [ ] **Run the vast.ai medium-model validation campaign.** Tooling ready:
      skill `prover-remote-validation`, driver `analysis/bench/remote_suite.py`
      (smoke-tested locally), runner `InitBench/vast_run_verinf.py` (tarball
      upload + VerInf offer filter; NOT run live — needs VAST_AI_KEY, first run
      is a shakedown of the uv/cargo bootstrap). Suggested card: A6000 48GB
      (~$0.5/h) primary; A100 80GB for spill-under-memory-pressure. Budget
      $300/mo = ~20-40 GPU-hours needed; DESTROY instances (runner does).
      Run spill on ≥2 GPU types (its win is hardware-dependent).

## Next experiments (not started)
- [ ] **Accuracy experiment for BabyBear (the real gate).** Prove a small model
      at reduced fixed-point scale s (2^8, then 2^6) and check the inference
      output still matches the s=2^12 reference. This decides whether BabyBear is
      usable AT ALL before any field-backend port — cheaper than the port.
      Tools ready: `bench/field_policy.py`, `bench/run_field_variants.py`.
- [ ] **Disk-backed, full-witness spill (the production +13-34% lever).** Extend
      the validated host-memory spill to (a) disk backing (7.5TB ≫ host RAM),
      (b) spill the FULL witness, not just softmax/silu. Mechanism already sound
      (byte-identical + Rust ACCEPT); work is I/O plumbing + coverage, not
      protocol. Benefit confirmable only on production-scale hardware.

## Blocked / needs a decision
- [ ] **BabyBear field backend port** — only if the accuracy experiment passes.
      Multi-file: Python (protocol.py roots, cuda_primitives.py), CUDA kernels
      (kquant_cuda.py GL_P), Rust (field.rs). Large; soundness-critical.
- [ ] **claim-streaming** — SOUNDNESS-BLOCKED as a round-merge; only the spill
      remnant is sound. Revisit only with a protocol redesign.

## Loose ends
- [ ] Re-measure seq768 prove time post-GPU-softmax (never remeasured; iter8).
- [ ] silu/rmsnorm remaining GPU work (silu ported; rmsnorm needs 2^128 products
      for the isqrt y-search — parked).
- [ ] BabyBear combination-of-fields prototype (mixed-field per-op execution),
      contingent on the field backend.

## Standing discipline (do not drop)
- Every A/B must end with `prod_lens.report(...)` (toy % never stands alone).
- ACCEPT gate: byte-identical proof + Rust verify + fast suite green, or revert.
- No overselling: modeled ≠ measured; test-scale ≠ production.
