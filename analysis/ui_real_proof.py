"""PROVE the unexplained-information bound on the REAL 32-layer Llama-2-7b forward
over the measured 100-token sequence, output tokens HIDDEN, and dump for the Rust
verifier. Full-scale LM head (the 32-layer logits are sane, ~2^17), s_c=2^18
(sigma ~ 0.088, calibrated to the int-vs-fp disagreement from ui_real_measure).

  PYTHONPATH=~/ligero/pipeline LIGERO_SKIP_GPU_VERIFY=1 \\
    ~/venv-hf/bin/python analysis/ui_real_proof.py
  verifier-rs/target/release/verify_proof /tmp/ui_real_proof.json
"""
import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "demo"))
import demo_llama7b

MODEL = str(pathlib.Path("~/models/llama-2-7b-hf").expanduser())
seq = json.load(open("/tmp/ui_real/seq.json"))            # 100-token sequence
N_IN = 50
# Surprisal is computed for all 100 positions (target = next token), but U sums
# ONLY the 50 generated-output positions 49..98 (49+i predicts the output seq[50+i]),
# matching the measurement and excluding the surprising prompt.
out_tokens = seq[1:] + [seq[-1]]                          # next-token target per position
positions = list(range(N_IN - 1, len(seq) - 1))          # 49..98 -> the 50 outputs

demo_llama7b.main(
    from_hf=MODEL, num_layers=32, token_ids=seq,
    unexplained_info=True, ui_positions=positions, ui_output_tokens=out_tokens,
    ui_lm_sout=4096, ui_lm_ow=26, ui_s_c=1 << 18,         # full scale, sigma~0.088
    engine=True, lazy_weights=True,
    dump_proof="/tmp/ui_real_proof.json")
