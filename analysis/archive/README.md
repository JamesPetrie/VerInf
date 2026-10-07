# analysis/archive

Research scripts that nothing in the tree references any more, kept for the
record rather than deleted. None is run by a gate, a runbook or CI, and the
ones marked broken have not run since the API they call changed. Moved here
on 2026-10-07 by the dead-code sweep; `git log --follow` has each one's history.

- `logits_report.py` (was `analysis/logits_report.py`): reads /tmp/logits_full100.npy from a 2026-08 run; nothing writes it.
- `dump_ui_proof.py` (was `analysis/dump_ui_proof.py`): one-off UI proof dump, superseded by demo --dump-proof.
- `ui_sy_sweep.py` (was `analysis/ui_sy_sweep.py`): one-off s_y sweep for the UI tables.
- `ui_llama_scale.py` (was `analysis/ui_llama_scale.py`): one-off scale probe on Llama.
- `ui_scale_check.py` (was `analysis/ui_scale_check.py`): imports witness_bound/reference_bits, which unexplained_info no longer defines.
- `maverick_noise_sweep.py` (was `analysis/maverick_noise_sweep.py`): hard-codes a /home/amodo path; the sweep's results are in llama-noise-sweep/.
- `unexplained_info_stream.py` (was `analysis/unexplained_info_stream.py`): early streaming UI sketch, superseded by prover/unexplained_info.py.
- `check_quad_avalues.py` (was `analysis/check_quad_avalues.py`): one-off investigation against private core internals; the Rust verifier covers it.
- `check_quad_order.py` (was `analysis/check_quad_order.py`): one-off investigation against private core internals; the Rust verifier covers it.
- `check_tables.py` (was `analysis/check_tables.py`): imports test_compile_parity, retired to prover/deprecated/.
- `compare_accuracy.py` (was `analysis/compare_accuracy.py`): calls demo_llama7b._run_block/_run_tail with pre-ModelConfig signatures; pathlib unimported.
- `per_op_attribution.py` (was `analysis/per_op_attribution.py`): same breakage as compare_accuracy.py.
- `sampled_rs_window_bench.py` (was `analysis/bench/sampled_rs_window_bench.py`): one-off RS window bench, unreferenced.
- `knob_sweep.py` (was `analysis/bench/knob_sweep.py`): needs dukpy, which is not a project dependency; unreferenced.
- `knob_selftest.py` (was `analysis/bench/knob_selftest.py`): only knob_sweep.py used it.
- `confirm_cap_seq1024.py` (was `analysis/bench/confirm_cap_seq1024.py`): confirmed that the LIGERO_WITNESS_CACHE_MAX_ELEMS cap caused the seq1024 collapse; the cap is gone (metric_ledger.md keeps the result).
