"""Compare our quantized ZK-pipeline forward pass against an FP/BF16
reference run via Hugging Face.

For a given Llama-2-7B-style model + prompt + sequence length, this
script:

  1. Runs HF inference at the chosen reference dtype (default bfloat16,
     since that's Llama-2's native checkpoint dtype) and captures the
     logits tensor (SEQ, vocab).
  2. Runs our Tape-based forward pass at Q3.12 fixed-point (no prove —
     just builds the witness) and captures the logits, then converts
     them back to float via signed-Goldilocks → divide by S.
  3. Reports per-position and aggregate metrics: argmax agreement,
     top-5 overlap, max |Δ|, mean |Δ|, KL divergence.

Memory: HF model is freed before the pipeline runs, so peak is bounded
by max(HF, pipeline) instead of their sum.

Example:
  python3 compare_accuracy.py --from-hf meta-llama/Llama-2-7b-hf \\
      --prompt "The quick brown fox" --seq 91 --num-layers 32 --per-position
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prover"))

from cuda_primitives import P
from core import LigeroConfig
from tape import Tape, SILU_14BIT
from loader import (
    tokenize_prompt, load_token_embedding, load_layer_weights,
    load_final_weights, free_model_cache,
)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "demo"))
import demo_llama7b as demo


CFG = demo.CFG
S   = demo.S
d   = demo.d
d_h = demo.d_h


# ---------------------------------------------------------------------------
# Goldilocks → signed float conversion.
# ---------------------------------------------------------------------------

def gl_to_signed_float(t_u64: torch.Tensor) -> np.ndarray:
    """Goldilocks uint64 tensor → signed float64 numpy array.

    Values in [0, P/2) are positive; values in [P/2, P) represent the
    corresponding negative integer (v - P)."""
    arr = t_u64.detach().cpu().numpy().astype(np.uint64)
    f = arr.astype(np.float64)
    half_p = P / 2.0
    return np.where(f >= half_p, f - float(P), f)


# ---------------------------------------------------------------------------
# Pipeline forward pass (no prove).
# ---------------------------------------------------------------------------

def run_pipeline(model_id: str, prompt: str, seq: int,
                  num_layers: int, layer_idx: int = 0) -> torch.Tensor:
    """Build the Tape for `num_layers` layers + final RmsNorm + LM head,
    pulling weights from HF. DON'T prove — just compute. Returns the
    final logits as a (seq, vocab) float64 numpy array."""
    print(f"  [pipeline] tokenizing + embedding lookup …")
    tok_ids = tokenize_prompt(model_id, prompt).cpu().tolist()
    if len(tok_ids) >= seq:
        tok_ids = tok_ids[:seq]
    else:
        tok_ids = tok_ids + [0] * (seq - len(tok_ids))

    E_full = load_token_embedding(model_id, S=S)
    unique = sorted(set(tok_ids))
    id_to_subset = {tid: i for i, tid in enumerate(unique)}
    unique_t = torch.tensor(unique, dtype=torch.int64, device="cuda")
    E_subset_data = (E_full.view(-1, d).index_select(0, unique_t)
                     .contiguous().view(-1))
    E_subset_shape = (len(unique), d)
    subset_ids = [id_to_subset[t] for t in tok_ids]
    tok_t = torch.tensor(subset_ids, dtype=torch.int64, device="cuda")
    x_data = (E_subset_data.view(len(unique), d)
              .index_select(0, tok_t).contiguous().view(-1))
    del E_full
    torch.cuda.empty_cache()

    H = d // d_h
    input_data = x_data
    input_shape = (seq, d)

    for L in range(num_layers):
        this_layer = layer_idx + L
        print(f"  [pipeline] forward layer {this_layer} …")
        tape = Tape(CFG, silu_config=SILU_14BIT)
        if L == 0:
            E_local = tape.commit("E_embedding_subset",
                                    E_subset_data, E_subset_shape)
            x_L = tape.embed(E_local, token_ids=subset_ids, d=d)
        else:
            x_L = tape.commit(f"x_L{this_layer}", input_data, input_shape)
        weights = demo._commit_weights_from_hf(tape, model_id, this_layer)
        resid = demo._run_block(tape, x_L, weights, H=H)
        input_data, input_shape = resid.data.clone(), resid.shape
        del tape, weights, resid, x_L
        torch.cuda.empty_cache()

    print(f"  [pipeline] tail: final RmsNorm + LM head …")
    fw = load_final_weights(model_id, S=S)
    vocab = fw["W_lm_head"].numel() // d
    tail_tape = Tape(CFG, silu_config=SILU_14BIT)
    x_tail = tail_tape.commit("x_tail", input_data, input_shape)
    final_norm_w = tail_tape.commit("final_norm_w", fw["final_norm_w"], (d,))
    W_lm = tail_tape.commit("W_lm_head", fw["W_lm_head"], (d, vocab))
    logits = demo._run_tail(tail_tape, x_tail, final_norm_w, W_lm, vocab_size=vocab)
    free_model_cache()

    # `logits.data` is at scale S (matmul s_out=S). Convert to floats.
    logits_signed = gl_to_signed_float(logits.data)
    logits_flat = logits_signed / S
    return logits_flat.reshape(seq, vocab)


# ---------------------------------------------------------------------------
# Reference forward pass via HF.
# ---------------------------------------------------------------------------

def run_reference(model_id: str, prompt: str, seq: int,
                   num_layers: int, layer_idx: int,
                   dtype: torch.dtype) -> torch.Tensor:
    """Run HF inference up to `layer_idx + num_layers` layers + the
    full tail (final RmsNorm + LM head). Returns (seq, vocab) numpy
    float array.

    For the layer-truncation case (num_layers + layer_idx < total), we
    hook the model and stop early — the tail still uses the trimmed
    hidden state, which mirrors what the pipeline does."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  [reference] loading HF model in {dtype} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="cuda")
    model.eval()
    total_layers = model.config.num_hidden_layers
    end_layer = layer_idx + num_layers
    print(f"  [reference] model has {total_layers} layers; using "
          f"[{layer_idx}, {end_layer})")

    tok_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    if len(tok_ids) >= seq:
        tok_ids = tok_ids[:seq]
    else:
        tok_ids = tok_ids + [0] * (seq - len(tok_ids))
    input_ids = torch.tensor([tok_ids], dtype=torch.long, device="cuda")

    with torch.no_grad():
        # If we want fewer than all layers, truncate the model's layer list.
        if end_layer < total_layers:
            orig_layers = model.model.layers
            model.model.layers = torch.nn.ModuleList(
                list(orig_layers)[layer_idx:end_layer])
            try:
                out = model(input_ids, use_cache=False)
            finally:
                model.model.layers = orig_layers
        else:
            out = model(input_ids, use_cache=False)

    logits = out.logits[0].float().cpu().numpy()        # (seq, vocab)
    del model
    torch.cuda.empty_cache()
    return logits


# ---------------------------------------------------------------------------
# Comparison metrics.
# ---------------------------------------------------------------------------

def compare(L_q: np.ndarray, L_f: np.ndarray, per_position: bool = False) -> None:
    """L_q, L_f are (seq, vocab) float arrays at the same scale."""
    assert L_q.shape == L_f.shape, f"shape mismatch: {L_q.shape} vs {L_f.shape}"
    seq, vocab = L_q.shape
    diff = L_q - L_f
    abs_diff = np.abs(diff)

    arg_q = L_q.argmax(axis=-1)
    arg_f = L_f.argmax(axis=-1)
    top5_q = np.argpartition(-L_q, 5, axis=-1)[:, :5]
    top5_f = np.argpartition(-L_f, 5, axis=-1)[:, :5]
    top5_overlap = np.array([
        len(set(top5_q[i].tolist()) & set(top5_f[i].tolist())) for i in range(seq)
    ])

    # Softmax → KL (numerically stable, position-by-position).
    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)
    p_q = softmax(L_q)
    p_f = softmax(L_f)
    eps = 1e-12
    kl_per_pos = (p_q * (np.log(p_q + eps) - np.log(p_f + eps))).sum(axis=-1)

    if per_position:
        print(f"\n  per-position metrics:")
        print(f"  {'pos':>4} {'argmax':>10} {'top-5':>7} "
              f"{'max|Δ|':>10} {'mean|Δ|':>10} {'KL':>10}")
        for i in range(seq):
            agree = "Y" if arg_q[i] == arg_f[i] else "N"
            print(f"  {i:>4} {agree:>10} "
                  f"{top5_overlap[i]:>2}/5    "
                  f"{abs_diff[i].max():>10.4f} "
                  f"{abs_diff[i].mean():>10.4f} "
                  f"{kl_per_pos[i]:>10.4f}")

    print(f"\n  summary across {seq} positions:")
    print(f"    argmax agreement     : {(arg_q == arg_f).sum()}/{seq} "
          f"({100.0 * (arg_q == arg_f).mean():.1f}%)")
    print(f"    top-5 overlap (mean) : {top5_overlap.mean():.2f}/5")
    print(f"    max |Δ| over all pos : {abs_diff.max():.4f}")
    print(f"    mean |Δ|             : {abs_diff.mean():.4f}")
    print(f"    KL (mean)            : {kl_per_pos.mean():.4f}")
    print(f"    KL (max)             : {kl_per_pos.max():.4f}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-hf", type=str, required=True,
                     help="HF model id (e.g. meta-llama/Llama-2-7b-hf).")
    ap.add_argument("--prompt", type=str, required=True,
                     help="Input prompt. Will be tokenized + padded/truncated to --seq.")
    ap.add_argument("--seq", type=int, default=91,
                     help="Sequence length. Default 91.")
    ap.add_argument("--num-layers", type=int, default=32,
                     help="Number of transformer blocks to run. Default 32 (full model).")
    ap.add_argument("--layer-idx", type=int, default=0,
                     help="First layer index to run. Default 0.")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                     choices=["bfloat16", "float16", "float32"],
                     help="Reference dtype for the HF run. Default bfloat16 "
                          "(Llama 2 native). Use float16 if you want strict FP16.")
    ap.add_argument("--per-position", action="store_true",
                     help="Print per-position metrics (one line per token).")
    args = ap.parse_args()

    # Override demo's SEQ — must happen before any tape construction.
    demo.SEQ = args.seq

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                  "float32": torch.float32}
    ref_dtype = dtype_map[args.dtype]

    print(f"=== Accuracy comparison ===")
    print(f"  model     : {args.from_hf}")
    print(f"  prompt    : {args.prompt!r}")
    print(f"  seq       : {args.seq}")
    print(f"  layers    : [{args.layer_idx}, {args.layer_idx + args.num_layers})")
    print(f"  ref dtype : {args.dtype}")

    # 1. Reference first — frees its memory before pipeline runs.
    print(f"\n[1/2] HF reference …")
    t0 = time.time()
    L_ref = run_reference(
        args.from_hf, args.prompt, args.seq, args.num_layers,
        args.layer_idx, ref_dtype)
    print(f"  done in {time.time() - t0:.1f}s, logits shape={L_ref.shape}")

    # 2. Pipeline forward.
    print(f"\n[2/2] Pipeline forward (Q3.12, no prove) …")
    t0 = time.time()
    L_q = run_pipeline(args.from_hf, args.prompt, args.seq,
                        args.num_layers, args.layer_idx)
    print(f"  done in {time.time() - t0:.1f}s, logits shape={L_q.shape}")

    # 3. Compare.
    compare(L_q, L_ref, per_position=args.per_position)


if __name__ == "__main__":
    main()
