# VerInf validation campaign — NVIDIA A100 80GB PCIe, 81920 MiB
host `76057769fefe` · matrix `phone` · reps 1 · 2026-07-31T13:55:13Z

## Soundness gates (must all PASS or the timings are void)

- ✅ byte-identical: GPU vs numpy softmax — ok
- ✅ byte-identical: GPU vs numpy silu — ok
- ✅ byte-identical: spill vs recompute — ok
- ✅ Rust verify_proof = ACCEPT (spill on) — ok

## A/B levers at medium scale

### GPU softmax (transferable witness-compute win) @ `d896_ff4864_seq256_L24_v151936` — ok (186.9s)
    config                    off_s     on_s   faster  wit_off   wit_on  peakGB
        BUCKETED       85.9s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       78.2s  (vs total prove wall-clock; remainder = setup + un-bucketed)
      PRODUCTION PROJECTION: +34.6%  (28668s -> 18737s)  (transfers; softmax share of witness grows with SEQ, so conservative)

### witness spill (host store-once/re-read) @ `d896_ff4864_seq256_L24_v151936` — ok (275.2s)
        BUCKETED       78.5s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       77.9s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       77.8s  (vs total prove wall-clock; remainder = setup + un-bucketed)
      recompute : prove  81.75s   witness   5.33s
      gpu-cache : prove  81.24s   witness   4.63s   (+0.6% vs recompute)
      host-spill: prove  86.89s   witness   4.65s   (-6.3% vs recompute)
      spill vs gpu-cache: prove 81.24s vs 86.89s (spill trades GPU mem for host mem; slower by 5.66s here)
      PRODUCTION PROJECTION: +27.0%  (28668s -> 20941s)  (NVMe 7GB/s; crossover-type — prod per-term cut, NOT the toy fraction)

### GPU softmax (transferable witness-compute win) @ `d2048_ff8192_seq256_L16_v128256` — ok (218.1s)
    config                    off_s     on_s   faster  wit_off   wit_on  peakGB
        BUCKETED      105.2s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       96.5s  (vs total prove wall-clock; remainder = setup + un-bucketed)
      PRODUCTION PROJECTION: +30.1%  (28668s -> 20040s)  (transfers; softmax share of witness grows with SEQ, so conservative)

### witness spill (host store-once/re-read) @ `d2048_ff8192_seq256_L16_v128256` — ok (325.6s)
        BUCKETED       98.3s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       97.3s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       96.9s  (vs total prove wall-clock; remainder = setup + un-bucketed)
      recompute : prove  98.93s   witness   9.33s
      gpu-cache : prove  97.96s   witness   8.49s   (+1.0% vs recompute)
      host-spill: prove 102.76s   witness   8.58s   (-3.9% vs recompute)
      spill vs gpu-cache: prove 97.96s vs 102.76s (spill trades GPU mem for host mem; slower by 4.80s here)
      PRODUCTION PROJECTION: +27.0%  (28668s -> 20941s)  (NVMe 7GB/s; crossover-type — prod per-term cut, NOT the toy fraction)

> GPU-attached; every % above is paired with its prod_lens 400B projection.
> A failed/OOM/timeout run is recorded as-is, never fabricated.