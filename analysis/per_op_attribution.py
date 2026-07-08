"""Run a single transformer block (or N layers) through ShadowTape and
report per-op quantization-error attribution.

Each Tape op (matmul, hadamard, rmsnorm, silu, softmax, rope, add) is
intercepted by ShadowTape, which computes the FP equivalent given the
quantized inputs the op actually received. The gap between the quantized
output and the FP shadow is recorded per call, then aggregated by op type.

The reported numbers tell you WHICH KIND OF OP introduces the most local
quantization noise, on the real Tape pipeline running on real hardware.
Local error only — for end-to-end ablation (replace one op with FP and
see how much total drift drops), a future mode can use the same
ShadowTape with an `ablate` flag.

Reference baseline is bfloat16 by default (matches Llama-2 native dtype),
not full FP — so the gold-truth shadow we compute IS computed in float64
torch, but the "ground truth" we'd compare to in operational terms is
bfloat16. The deltas reported are vs torch's float64 reference.

Example:
  python3 per_op_attribution.py --from-hf meta-llama/Llama-2-7b-hf \\
      --prompt "The quick brown fox" --seq 16 --num-layers 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prover"))
sys.path.insert(0, str(HERE))

from cuda_primitives import P
from core import LigeroConfig
from tape import SILU_14BIT
from shadow_tape import ShadowTape, gl_to_float

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "demo"))
import demo_llama7b as demo
from loader import (
    tokenize_prompt, load_token_embedding, load_layer_weights,
    free_model_cache,
)


def run_shadow_layer(model_id: str, prompt: str, seq: int,
                      num_layers: int = 1, layer_idx: int = 0) -> ShadowTape:
    """Build N layers through ShadowTape, return the tape with recorded
    deltas. Forward-only (no prove)."""
    # Patch demo SEQ before anything else (other modules in tape.py /
    # claims.py don't reference SEQ; only demo._run_block does, via H,
    # SEQ kwargs that we pass explicitly through tape methods).
    demo.SEQ = seq

    tok_ids = tokenize_prompt(model_id, prompt).cpu().tolist()
    if len(tok_ids) >= seq:
        tok_ids = tok_ids[:seq]
    else:
        tok_ids = tok_ids + [0] * (seq - len(tok_ids))

    E_full = load_token_embedding(model_id, S=demo.S)
    d = demo.d
    unique = sorted(set(tok_ids))
    id_to_subset = {tid: i for i, tid in enumerate(unique)}
    unique_t = torch.tensor(unique, dtype=torch.int64, device="cuda")
    E_subset_data = (E_full.view(-1, d).index_select(0, unique_t)
                     .contiguous().view(-1))
    subset_ids = [id_to_subset[t] for t in tok_ids]
    tok_t = torch.tensor(subset_ids, dtype=torch.int64, device="cuda")
    x_data = (E_subset_data.view(len(unique), d)
              .index_select(0, tok_t).contiguous().view(-1))
    del E_full
    torch.cuda.empty_cache()

    H = d // demo.d_h

    # Per layer: fresh ShadowTape, feed the residual's data + its scale.
    # Local-error model means we don't need to carry a chain FP shadow
    # across layers — each op is measured against an FP recomputation
    # from its quantized inputs, regardless of upstream history.
    input_data = x_data
    input_shape = (seq, d)
    accumulated = []                                    # collect deltas across layers
    last_tape: ShadowTape | None = None

    for L in range(num_layers):
        this_layer = layer_idx + L
        print(f"  [shadow] layer {this_layer} …")
        tape = ShadowTape(demo.CFG, silu_config=SILU_14BIT, default_scale=demo.S)
        x_L = tape.commit(f"x_L{this_layer}", input_data, input_shape,
                            scale=demo.S)
        weights = demo._commit_weights_from_hf(tape, model_id, this_layer)
        resid = demo._run_block(tape, x_L, weights, H=H)
        input_data = resid.data.clone()
        input_shape = resid.shape
        accumulated.extend(tape.deltas)
        last_tape = tape
        # Drop the rest of the tape between layers to bound memory.
        if L < num_layers - 1:
            del tape, weights, resid, x_L

    free_model_cache()
    # Stuff accumulated deltas into a single tape for unified reporting.
    last_tape.deltas = accumulated
    return last_tape


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-hf", type=str, required=True,
                     help="HF model id (e.g. meta-llama/Llama-2-7b-hf).")
    ap.add_argument("--prompt", type=str, required=True,
                     help="Input prompt; will be tokenized + padded to --seq.")
    ap.add_argument("--seq", type=int, default=16,
                     help="Sequence length. Default 16.")
    ap.add_argument("--num-layers", type=int, default=1,
                     help="Number of transformer blocks. Default 1.")
    ap.add_argument("--layer-idx", type=int, default=0)
    args = ap.parse_args()

    print(f"=== Per-op error attribution ===")
    print(f"  model     : {args.from_hf}")
    print(f"  prompt    : {args.prompt!r}")
    print(f"  seq       : {args.seq}")
    print(f"  layers    : [{args.layer_idx}, {args.layer_idx + args.num_layers})")

    t0 = time.time()
    tape = run_shadow_layer(args.from_hf, args.prompt, args.seq,
                              num_layers=args.num_layers,
                              layer_idx=args.layer_idx)
    print(f"  done in {time.time() - t0:.1f}s, {len(tape.deltas)} op calls recorded")
    print()
    tape.print_summary()


if __name__ == "__main__":
    main()
