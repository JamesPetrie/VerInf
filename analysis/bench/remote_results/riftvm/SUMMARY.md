# VerInf validation campaign — Tesla V100-SXM3-32GB, 32768 MiB
host `riftvm` · matrix `phone` · reps 2 · 2026-07-30T11:56:12Z

## Soundness gates (must all PASS or the timings are void)

- ✅ byte-identical: GPU vs numpy softmax — ok
- ✅ byte-identical: GPU vs numpy silu — ok
- ✅ byte-identical: spill vs recompute — ok
- ✅ Rust verify_proof = ACCEPT (spill on) — ok

## A/B levers at medium scale

### GPU softmax (transferable witness-compute win) @ `d896_ff4864_seq256_L24_v151936` — ok (482.8s)
    config                    off_s     on_s   faster  wit_off   wit_on  peakGB
        BUCKETED      112.4s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED      112.0s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       91.1s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       91.2s  (vs total prove wall-clock; remainder = setup + un-bucketed)
      PRODUCTION PROJECTION: +44.1%  (28668s -> 16016s)  (transfers; softmax share of witness grows with SEQ, so conservative)

### witness spill (host store-once/re-read) @ `d896_ff4864_seq256_L24_v151936` — ok (671.9s)
        BUCKETED       92.6s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       91.7s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       91.0s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       90.9s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       90.8s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED       90.6s  (vs total prove wall-clock; remainder = setup + un-bucketed)
      recompute : prove  98.80s   witness   6.84s
      gpu-cache : prove  96.94s   witness   5.88s   (+1.9% vs recompute)
      host-spill: prove 105.44s   witness   5.98s   (-6.7% vs recompute)
      spill vs gpu-cache: prove 96.94s vs 105.44s (spill trades GPU mem for host mem; slower by 8.50s here)
      PRODUCTION PROJECTION: +27.0%  (28668s -> 20941s)  (NVMe 7GB/s; crossover-type — prod per-term cut, NOT the toy fraction)

### GPU softmax (transferable witness-compute win) @ `d2048_ff8192_seq256_L16_v128256` — ok (650.3s)
    config                    off_s     on_s   faster  wit_off   wit_on  peakGB
        BUCKETED      185.3s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED      183.4s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED      113.8s  (vs total prove wall-clock; remainder = setup + un-bucketed)
        BUCKETED      113.6s  (vs total prove wall-clock; remainder = setup + un-bucketed)
      PRODUCTION PROJECTION: +49.7%  (28668s -> 14428s)  (transfers; softmax share of witness grows with SEQ, so conservative)

### witness spill (host store-once/re-read) @ `d2048_ff8192_seq256_L16_v128256` — exit1 (146.3s)
        BUCKETED      116.8s  (vs total prove wall-clock; remainder = setup + un-bucketed)
    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.96 GiB. GPU 0 has a total capacity of 31.73 GiB of which 1.33 GiB is free. Including non-PyTorch memory, this process has 30.40 GiB memory in use. Of the allocated memory 29.53 GiB is allocated by PyTorch, and 378.92 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

> GPU-attached; every % above is paired with its prod_lens 400B projection.
> A failed/OOM/timeout run is recorded as-is, never fabricated.