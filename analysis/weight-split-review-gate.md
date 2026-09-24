# Weight-split review gate — 2026-09-05

The maintainer reported a passing hardware gate on the complete reviewed
working tree based on `e7ff6df`, including audit fixes 1–11, the Wnew totals
follow-up, and all three new regression files. This is the gate result for
the review-fix commits that follow that base.

Environment: NVIDIA A100 SXM, torch 2.8.0, CUDA 12.8, driver 580.126.16.
The A40 pools were empty in both datacenters, so the run used an A100,
also Ampere. Reported cost was approximately $0.50; the pod was terminated
afterward. The earlier launch-shell timeout was the transfer exceeding
two minutes; the test run was already in progress.

| Suite | Result |
|---|---|
| `test_weight_split` | 6/6 |
| `test_shard_worker` | 4/4 |
| `test_weight_provenance` | 2/2 |
| `test_shard_plan` | 3/3 |
| `test_layout_breakdown` | 2/2 |
| `test_persistent_weights_p3` | 3/3, Rust ACCEPT |
| `test_shard_streaming` | 4/4 |
| `test_routed_projected` | 6/6 |
| `test_gguf_loader` | 4/4 |

Proof-byte identity held at every N=2 cut and at N=3 with workers writing
opening chunks directly into the coordinator's sink. The chunk-grid test
remained identical across 1,408 W rows. The quads-on-W test was byte-identical
and Rust accepted its sharded proof. Early device validation did not disturb
the successful proof path.

Local validation before handoff passed 24 profiler tests, 26 calibration
tests, the C++ kernel-indexing check, and 16 CPU prover checks. The hardware
gate above validates the prover execution changes; it does not measure
multi-device speedup or CUDA benchmark timing. M1a still executes every role
sequentially on the coordinator's device.

See the [regression commands](../profiler/README.md#regression-commands) for
the CPU suites, compiler requirement, and weight-split gate recipe.
