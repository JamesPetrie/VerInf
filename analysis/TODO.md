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

## Layer-GKR-LF (proposed replacement protocol) — prototyped, NOT adopted
- [ ] **Decide on `analysis/VerInf_LayerGKR_4h_theorem_ru.md`.** Mechanisms are
      prototyped and gated in `layergkr/` (41 tests, self-contained, existing
      prover untouched); see `layergkr/README.md` and journal iter18. Cost model
      independently reproduces its §9.2 geometry, so the arithmetic holds. The
      open question is kappa: the doc adopts 1.5 (-> 3.95 h, break-even 1.530),
      our prover exhibits 1.83 (-> 4.57 h). Ask the author to (a) restate with
      kappa from the measured point, (b) address the ELL=8192 padding, which is
      worth more than the whole kappa margin. NOTE: this protocol does not use
      the witness spill at all — it replaces four semantic sweeps with one, so
      the item below is about the CURRENT path only.

## Known-suboptimal (measured, deliberately NOT fixed yet)
- [ ] **The disk-spill reader is single-threaded — it is the binding constraint,
      not the disk.** `_disk_spill_load` (prover/core.py:2482) reads with a
      blocking single-thread `os.pread` loop, and the pre-flight gate
      (`analysis/bench/preflight.py:52`) measures with the same pattern. That
      number is a property of OUR READER, not of the hardware. Measured on the
      dev host, same file / same disk: 1 thread 1.46 GB/s vs 8 threads 3.77 GB/s
      (+158%). Consequences:
        - the gate systematically UNDERSTATES a box (~2.6x here, more on real
          NVMe, where one thread caps ~2 GB/s and an array does 7+);
        - the ratio check against vast's advertised `disk_bw` is therefore not
          meaningful as written, and rejects the FASTEST boxes hardest;
        - vast's `disk_bw` itself looks honest — over 1000 sampled offers the
          median is 2.2 GB/s, p90 6.3, max 26.5, none above 20. The earlier
          claim that it "overstates 10-20x" was WRONG; it came from comparing
          our single-thread number against rare tail values (the 36-40 GB/s
          Quebec L40S offers);
        - most important: the production model says spill only beats recompute
          above 1.93 GB/s. At the current single-threaded 1.5-2.2 GB/s the 400B
          disk-spill run would be -28% to -4% vs plain recompute; with 8 threads
          it turns into +15% (this disk) to +28% (real NVMe). The +13-34% lever
          below is NOT reachable with the reader as it stands.
      Fix when the lever is taken up: chunked parallel `pread` (bytes are
      unchanged, so the proof stays byte-identical -> ACCEPT gate applies),
      then align preflight's measurement to the same pattern, restore the
      ratio check to fatal, and raise `MAV_MIN_READ_GBPS` from 1.5 to ~3.0
      (1.5 is below the 1.93 break-even, so it admits boxes where spill loses).

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
