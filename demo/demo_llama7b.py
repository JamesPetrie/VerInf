"""Unified Llama 2 7B transformer-block proof at production scale, with
the uniform "input at S, output at S" scale contract enforced end-to-end.

Every value-producing op in the chain (RmsNorm, Matmul, RoPE, Hadamard)
takes inputs at scale S and outputs at scale S via internal output rescale.
The scale and every scale-coupled constant cascade from one knob, SCALE_BITS
(S = 2^SCALE_BITS): EPS_INT, Z_MAX, the silu table scale (s_x = S), and the
rescale range window OUTPUT_WIDTH. Rescaled outputs are range-checked to
±2^(OUTPUT_WIDTH-1) (Llama's massive activations need the headroom).

Full chain at scale S:
  RmsNormClaim          norm1 = rmsnorm(x)           [s_out=S]
  MatmulClaim (×3)      q, k, v = norm1 · W          [s_out=S]
  RoPEClaim (×2)        q, k rotated                  [s_out=S]
  MatmulClaim (×3)      scores = q·k_t; attn_out = scores·v; proj = attn_out·W_O
  AddClaim              resid_1 = x + proj
  RmsNormClaim          norm2 = rmsnorm(resid_1)
  MatmulClaim (×2)      gate, up = norm2 · W
  SiluClaim             silu_gate = silu(gate)
  HadamardClaim         intermediate = silu_gate ⊙ up
  MatmulClaim           ffn_out = intermediate · W_down
  AddClaim              resid_2 = resid_1 + ffn_out

Independent production-shape softmax kept for benchmark purposes (the
chained softmax at SEQ=2 would have M=2, trivially small).

1/√d_h scaling on Q·K^T is folded into W_Q at weight-quantization time
(no new claim — W_Q magnitudes are shrunk by √d_h).

REMAINING GAPS:
  - No causal mask.
  - No HF weight loader; weights are random signed integers (Glorot-ish
    via _rand_signed in [-32, 32)).
  - Challenges are derived from per-round seeds by index (protocol.challenge /
    op_vec); a real run supplies the verifier's fresh per-round coins. This is
    an interactive protocol, not Fiat-Shamir-flattened.

Shapes (Llama 2 7B per-layer; SEQ=2 matches bringup-plan.md two-token target):
  d        = 4096       hidden dim
  d_ff     = 11008      FFN intermediate
  d_h      = 128        head dim (single-head approximation for RoPE)
  SEQ      = 2          two-token target (overridable via --seq)
  S        = 2^SCALE_BITS  Q-format scale (default 2^12); all scale-coupled
                        constants cascade from SCALE_BITS — no hidden magic
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import sys as _s, pathlib as _pl
_R = _pl.Path(__file__).resolve().parents[1]
_s.path.insert(0, str(_R / "prover")); _s.path.insert(0, str(_R / "demo"))
import _uint64_compat  # noqa: F401  — patch uint64 CUDA op gaps (gather/index_select/...) before any prover op runs

from cuda_primitives import P, gl_sub
from core import LigeroConfig
from tape import Tape
from claims import SiluConfig

# T_QUERIES overridable via env (e.g. =4 for a fast non-sound timing/feasibility
# run; production soundness uses 80).
CFG = LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536,
                   T_QUERIES=int(os.environ.get("LIGERO_T_QUERIES", "80")))


SEQ      = 2                    # overridable via --seq
d        = 4096
d_ff     = 11008
d_h      = 128                  # head dim (used for RoPE)
# ── Scale cascade: ONE knob (SCALE_BITS); every scale-coupled constant is
#    derived from it + a named real-world quantity. No hidden magic numbers. ──
SCALE_BITS = 12                       # Q-format fractional bits  (S = 2^SCALE_BITS)
S          = 1 << SCALE_BITS
OUTPUT_WIDTH = 26                     # rescale range-table width; real range ±2^(OUTPUT_WIDTH-1)/S
EPS_REAL   = 1e-5                     # Llama rmsnorm epsilon
EPS_INT    = round(EPS_REAL * S * S)  # ε at scale S²   (168 @ 2^12, 42 @ 2^11)
Z_NONZERO_REAL = 40000 / 4096         # softmax non-zero exp region (~9.77·s_c)
Z_MAX      = round(Z_NONZERO_REAL * S)  # softmax saturation at scale S (40000 @ 2^12)
SILU_CFG   = SiluConfig(b=4, T_LEN=1 << 14, b_2=1 << 16, b_3=1 << 32, b_4=1 << 48,
                        width_2=16, width_3=16, width_4=14, r=SCALE_BITS)  # s_x = S
RMS_SLACK_N_CHUNKS = 4

# Production-shape independent benchmarks (NOT chained from x).
SM_B     = 512
SM_M     = 2048
SM_Z_MAX = 1 << 17
SILU_L   = 4096
RMS_PROD_B = 2048


def _rand_small(*shape, lo=0, hi=1000):
    return torch.randint(lo, hi, shape, dtype=torch.int64,
                         device="cuda").to(torch.uint64)


def _rand_signed(*shape, half=32):
    """Signed integers in [-half, half), mapped to Goldilocks field rep
    (P − |v| for negative v) via gl_sub for the mod-P subtraction."""
    v = torch.randint(-half, half, shape, dtype=torch.int64, device="cuda")
    v_abs = v.abs().to(torch.uint64)
    P_t   = torch.full_like(v_abs, P)
    neg_field = gl_sub(P_t, v_abs)                     # = P − |v| (mod P)
    # int64-view select: torch CUDA has no `where` for uint64; bits are identical.
    return torch.where(v >= 0, v, neg_field.view(torch.int64)).view(torch.uint64)


# Random-weight HALF (signed range for non-W_Q weights); W_Q is shrunk by √d_h.
HALF = 32
# `x` magnitude bound. Honest post-embedding values in Llama 2 7B at Q3.12
# have integer magnitudes ≲ 64 (real BF16 magnitudes ~0.01–0.02, ×S=4096
# gives ~40–80). Using `HALF_X = 4` gives even more headroom for the
# chain to stay within the matmul rescale's 24-bit output window when
# combined with real-Llama weight magnitudes (W_K/V/O reach ~200 Q3.12).
HALF_X = 4


def _commit_weights_random(tape, layer_idx: int = 0) -> Dict[str, object]:
    """Random signed Glorot-ish weights, with 1/√d_h folded into W_Q.
    rms_pre_*_w are committed as identity (all = S) so the per-channel
    gain hadamard is a no-op for the random-weights smoke test.
    Layer-suffixed names mirror the HF loader."""
    HALF_Q = max(1, HALF // int(round(math.sqrt(d_h))))   # ≈ 3 for d_h=128
    identity_gain = torch.full((d,), S, dtype=torch.uint64, device="cuda")
    sfx = f"_L{layer_idx}"
    return {
        "W_Q":    tape.commit(f"W_Q{sfx}",    _rand_signed(d, d,    half=HALF_Q), (d, d)),
        "W_K":    tape.commit(f"W_K{sfx}",    _rand_signed(d, d,    half=HALF),   (d, d)),
        "W_V":    tape.commit(f"W_V{sfx}",    _rand_signed(d, d,    half=HALF),   (d, d)),
        "W_O":    tape.commit(f"W_O{sfx}",    _rand_signed(d, d,    half=HALF),   (d, d)),
        "W_gate": tape.commit(f"W_gate{sfx}", _rand_signed(d, d_ff, half=HALF),   (d, d_ff)),
        "W_up":   tape.commit(f"W_up{sfx}",   _rand_signed(d, d_ff, half=HALF),   (d, d_ff)),
        "W_down": tape.commit(f"W_down{sfx}", _rand_signed(d_ff, d, half=HALF),   (d_ff, d)),
        "rms_pre_attn_w": tape.commit(f"rms_pre_attn_w{sfx}", identity_gain.clone(), (d,)),
        "rms_pre_ffn_w":  tape.commit(f"rms_pre_ffn_w{sfx}",  identity_gain.clone(), (d,)),
    }


def _commit_weights_from_hf(tape, model_id: str, layer_idx: int) -> Dict[str, object]:
    """Quantize Llama 2 7B layer weights from HF format and commit them.
    Weight Variable names are prefixed `_L{layer_idx}` so multi-layer
    demos can commit distinct weight sets without name collisions.

    Requires the `transformers` library + a downloaded checkpoint or HF auth."""
    from loader import load_layer_weights
    # No extra_q_k_shrink: proper multi-head attention (H=32 heads, d_h=128)
    # reduces the q·k^T contraction from d=4096 to d_h=128, and the
    # per-channel RmsNorm gain (rms_pre_attn_w) dampens norm1 outliers
    # before W_Q/K — together these bring scores into a range the
    # saturating softmax handles natively.
    w = load_layer_weights(model_id, layer_idx, S=S, d_h=d_h)
    sfx = f"_L{layer_idx}"
    return {
        "W_Q":    tape.commit(f"W_Q{sfx}",    w["W_Q"],    (d, d)),
        "W_K":    tape.commit(f"W_K{sfx}",    w["W_K"],    (d, d)),
        "W_V":    tape.commit(f"W_V{sfx}",    w["W_V"],    (d, d)),
        "W_O":    tape.commit(f"W_O{sfx}",    w["W_O"],    (d, d)),
        "W_gate": tape.commit(f"W_gate{sfx}", w["W_gate"], (d, d_ff)),
        "W_up":   tape.commit(f"W_up{sfx}",   w["W_up"],   (d, d_ff)),
        "W_down": tape.commit(f"W_down{sfx}", w["W_down"], (d_ff, d)),
        "rms_pre_attn_w": tape.commit(
            f"rms_pre_attn_w{sfx}", w["rms_pre_attn_w"], (d,)),
        "rms_pre_ffn_w":  tape.commit(
            f"rms_pre_ffn_w{sfx}",  w["rms_pre_ffn_w"],  (d,)),
    }


def _commit_weights_from_hf_lazy(tape, lazy_loader, layer_idx: int) -> Dict[str, object]:
    """Lazy variant of _commit_weights_from_hf: registers per-weight
    LOADERS via tape.commit_lazy instead of materializing tensors. Each
    weight loads from disk on demand (during compute_fn / commit / etc.)
    and is freed before the next loader fires. Used when 32 × ~1.6 GB of
    weights can't coexist in unified memory (DGX Spark GB10 driver bug)."""
    sfx = f"_L{layer_idx}"
    shapes = lazy_loader.layer_shapes(d, d_ff)
    specs = lazy_loader.layer_specs(layer_idx)
    out = {}
    for short, (hf_name, transpose, divide_by) in specs.items():
        shape = shapes[short]
        length = shape[0] * (shape[1] if len(shape) > 1 else 1)
        loader = lazy_loader.make_loader(
            hf_name, transpose=transpose, divide_by=divide_by)
        out[short] = tape.commit_lazy(f"{short}{sfx}", loader, shape, length)
    return out


def _run_block(tape, x, weights, *, H: int):
    """One transformer block: x → norm1 → attn → resid_1 → norm2 → FFN → resid_2.
    Returns the output residual, ready to feed into the next block's input."""
    W_Q    = weights["W_Q"]
    W_K    = weights["W_K"]
    W_V    = weights["W_V"]
    W_O    = weights["W_O"]
    W_gate = weights["W_gate"]
    W_up   = weights["W_up"]
    W_down = weights["W_down"]
    rms_pre_attn_w = weights["rms_pre_attn_w"]
    rms_pre_ffn_w  = weights["rms_pre_ffn_w"]

    # ----- Attention -----
    norm1 = tape.rmsnorm(x, d=d, s=S, eps_int=EPS_INT,
                         slack_n_chunks=RMS_SLACK_N_CHUNKS,
                         s_out=S, output_width=OUTPUT_WIDTH)
    norm1_g = tape.hadamard_broadcast(norm1, rms_pre_attn_w, SEQ=SEQ, d=d,
                                       s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)

    q = tape.matmul(norm1_g, W_Q, s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    k = tape.matmul(norm1_g, W_K, s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    v = tape.matmul(norm1_g, W_V, s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)

    q_rope = tape.rope(q, SEQ=SEQ, d_h=d_h, heads=H,
                        s_x=S, s_out=S, output_width=OUTPUT_WIDTH)
    k_rope = tape.rope(k, SEQ=SEQ, d_h=d_h, heads=H,
                        s_x=S, s_out=S, output_width=OUTPUT_WIDTH)

    scores = tape.matmul(q_rope, k_rope, transpose_b=True,
                          heads=H, head_dim=d_h,
                          s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    sm_scores = tape.softmax(scores, M=SEQ, s_x=S, s_c=S, s_y=S,
                              Z_max=Z_MAX, saturate=True,
                              Z_high_width=16, aux_chunk_width=24,
                              causal=True, heads=H)
    attn_out = tape.matmul(sm_scores, v, heads=H, head_dim=SEQ,
                            s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    proj    = tape.matmul(attn_out, W_O, s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    resid_1 = x + proj

    # ----- FFN -----
    norm2 = tape.rmsnorm(resid_1, d=d, s=S, eps_int=EPS_INT,
                          slack_n_chunks=RMS_SLACK_N_CHUNKS,
                          s_out=S, output_width=OUTPUT_WIDTH)
    norm2_g = tape.hadamard_broadcast(norm2, rms_pre_ffn_w, SEQ=SEQ, d=d,
                                       s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    gate = tape.matmul(norm2_g, W_gate, s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    up   = tape.matmul(norm2_g, W_up,   s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    silu_gate = tape.silu(gate)
    intermediate = tape.hadamard(silu_gate, up, s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    ffn_out      = tape.matmul(intermediate, W_down, s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    resid_2      = resid_1 + ffn_out
    return resid_2


def _run_tail(tape, x, final_norm_w, W_lm_head, *,
               vocab_size: int, lm_s_out: int = S, lm_ow: int = OUTPUT_WIDTH):
    """Post-layer-stack tail: final RmsNorm + per-channel gain + LM head.
    Returns the logits WitnessTensor of shape (SEQ, vocab_size).

    `lm_s_out` sets the LM-head output scale: the default S keeps the uniform
    scale contract; a COARSER scale (e.g. 1) is used for the unexplained-info
    bound so the bound-relevant gaps land in a feasible range, BOUND by the
    matmul's own rescale (no separate coarsening claim)."""
    final_norm = tape.rmsnorm(x, d=d, s=S, eps_int=EPS_INT,
                               slack_n_chunks=RMS_SLACK_N_CHUNKS,
                               s_out=S, output_width=OUTPUT_WIDTH)
    final_norm_g = tape.hadamard_broadcast(
        final_norm, final_norm_w, SEQ=SEQ, d=d,
        s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    logits = tape.matmul(final_norm_g, W_lm_head,
                          s_a=S, s_b=S, s_out=lm_s_out, output_width=lm_ow)
    return logits


# Unexplained-info scales for the folded run. The LM head is run COARSE (s_out=1)
# so the bound-relevant gaps land in a feasible range; s_c keeps k=s_c/s_b a small
# power of two (so the rem range table [0,k) is tiny). With 1-layer garbage logits
# the kernel is effectively degenerate (far tokens floor to 1) and U is dominated
# by the partition self-info -- the point here is the end-to-end fold, not a
# calibrated U (that needs a real multi-layer model + sigma calibration).
UI_COARSE_S, UI_COARSE_OW = 1, 24
# s_y=2^28 >> V keeps the floor-1 partition over-count negligible (~0.02 bits over
# 50 tokens) so the proven U tracks the float to ~0.02 bits; table counts are set by
# gap_max / ln(V)*s_b, not s_y, so a large s_y is free (values stay < P).
UI_S_C, UI_S_Y_BITS, UI_S_B_BITS = 1 << 28, 28, 12


def _run_unexplained_info(tape, logits, *, vocab_size, seq, output_tokens=None,
                           s_c=UI_S_C, sum_positions=None):
    """Append the unexplained-information bound over the (bound) LM-head logits,
    output tokens HIDDEN. `s_c = 2*sigma^2` sets the kernel width. Returns
    (Sz_wt, info)."""
    from unexplained_info import prove_unexplained_info
    from max_claim import to_signed

    ls = to_signed(logits.data.view(seq, vocab_size))             # signed logits
    vstar = ls.max(dim=1).values
    gap_max = int((vstar.view(seq, 1) - ls).max().item()) + 2
    if output_tokens is None:
        tokens = [int(ls[t].argmax()) for t in range(seq)]        # greedy -> gap_o = 0
    else:
        assert len(output_tokens) == seq, "ui-output-tokens must have SEQ entries"
        tokens = [int(v) for v in output_tokens]

    s_y, s_b = 1 << UI_S_Y_BITS, 1 << UI_S_B_BITS
    k = s_c // s_b
    print(f"  unexplained-info (folded): gap_max={gap_max:,} (exp+gap tables), "
          f"s_c={s_c}, s_y=2^{UI_S_Y_BITS}, s_b=2^{UI_S_B_BITS}, k={k} "
          f"(output tokens hidden)")
    if gap_max.bit_length() > 26:
        print(f"    [skip] gap_max=2^{gap_max.bit_length()} too large for the exp table; "
              f"use a coarser LM-head scale.")
        return None, None

    Sz, h = prove_unexplained_info(tape, logits, tokens, T=seq, V=vocab_size,
                                   s_c=s_c, s_y=s_y, s_b=s_b, gap_max=gap_max,
                                   sum_positions=sum_positions, reveal=True)
    n = len(sum_positions) if sum_positions is not None else seq
    return Sz, dict(s_b=s_b, tokens=tokens, n=n, reveal_pin=h.get('reveal_pin'))


def _report_unexplained_info(tape, Sz, info, *, seq, vocab_size):
    """Read the committed Σ surprisal and print the U bound (eager: value live)."""
    from unexplained_info import bound_bits
    Sz_val = int(tape.inputs[Sz.var].cpu().item())
    U = bound_bits(Sz_val, s_b=info["s_b"])
    n = info.get("n", seq)
    print(f"  unexplained-information bound: U = {U:.4f} bits "
          f"over {n} hidden output token(s)")
    return U


def main(*, from_hf: Optional[str] = None, layer_idx: int = 0,
          save_logits: Optional[str] = None,
          num_layers: int = 1,
          prompt: Optional[str] = None,
          with_lm_head: bool = True,
          verbose: bool = False,
          lazy_weights: bool = False,
          engine: bool = False,
          time_ops: bool = False,
          forward_only: bool = False,
          dump_proof: Optional[str] = None,
          unexplained_info: bool = False,
          ui_sigma: int = 256,
          ui_sy_bits: int = 16,
          ui_output_tokens: Optional[list] = None,
          ui_lm_sout: int = UI_COARSE_S,
          ui_lm_ow: int = UI_COARSE_OW,
          ui_s_c: int = UI_S_C,
          ui_positions: Optional[list] = None,
          token_ids: Optional[list] = None):
    global SEQ
    if token_ids is not None:
        SEQ = len(token_ids)        # forward over an exact id stream
    print(f"=== Llama 2 7B transformer block, production scale ===")
    print(f"  cfg: ELL={CFG.ELL}, K_DEG={CFG.K_DEG}, "
          f"N_LIG={CFG.N_LIG}, T_QUERIES={CFG.T_QUERIES}")
    print(f"  shapes: d={d}, d_ff={d_ff}, SEQ={SEQ}, d_h={d_h}; scale s=2^{SCALE_BITS} "
          f"(EPS_INT={EPS_INT}, Z_MAX={Z_MAX}, output_width={OUTPUT_WIDTH}, silu r={SCALE_BITS})")
    if from_hf is not None:
        print(f"  weights: HF model {from_hf!r}, layers "
              f"[{layer_idx}, {layer_idx + num_layers})")
        if prompt is not None:
            print(f"  prompt:  {prompt!r}")
    else:
        print(f"  weights: random signed (Glorot-ish in [-32, 32); W_Q/√d_h), "
              f"{num_layers} layers")

    # ---------- Prepare the input residual data (no commits yet) ----------
    # Each layer creates its own Tape and commits fresh, so we just need
    # the raw data here. For the prompt path we also save E_subset and
    # the re-indexed token_ids so layer 0 can bind via EmbeddingLookupClaim.
    E_subset_data = None
    E_subset_shape = None
    subset_token_ids_saved = None
    if from_hf is not None and (prompt is not None or token_ids is not None):
        from loader import tokenize_prompt, load_token_embedding, free_model_cache
        tok_ids = (list(token_ids) if token_ids is not None
                   else tokenize_prompt(from_hf, prompt).cpu().tolist())
        if len(tok_ids) >= SEQ:
            tok_ids = tok_ids[:SEQ]
            print(f"  prompt tokens (first {SEQ}): {tok_ids}")
        else:
            tok_ids = tok_ids + [0] * (SEQ - len(tok_ids))
            print(f"  prompt tokens (padded to {SEQ}): {tok_ids}")
        # SOUNDNESS NOTE: committing the full Llama 2 7B embedding table
        # (vocab_size=32000 × d=4096 ≈ 131M cells) inflates m_total to
        # ~46K rows. For the demo we commit only the SUBSET of rows used by
        # this prompt; a deployable version would commit the full E once
        # with a Merkle anchor.
        E_full = load_token_embedding(from_hf, S=S)
        unique_ids = sorted(set(tok_ids))
        id_to_subset_idx = {tid: i for i, tid in enumerate(unique_ids)}
        unique_t = torch.tensor(unique_ids, dtype=torch.int64, device="cuda")
        E_subset_data = E_full.view(-1, d).index_select(
            0, unique_t).contiguous().view(-1)
        E_subset_shape = (len(unique_ids), d)
        subset_token_ids_saved = [id_to_subset_idx[t] for t in tok_ids]
        del E_full
        torch.cuda.empty_cache()
        # The actual x data for layer 0's input: just look up the rows now
        # so we can pass it through layer 0 (which RE-binds via embed in
        # its own tape, but we also need the .data for downstream layers).
        tok_t = torch.tensor(subset_token_ids_saved, dtype=torch.int64, device="cuda")
        x_data_init = E_subset_data.view(len(unique_ids), d).index_select(
            0, tok_t).contiguous().view(-1)
        print(f"  E subset: {len(unique_ids)} rows (full E would be "
              f"vocab_size × d ≈ 131M cells — see SOUNDNESS NOTE)")
        free_model_cache()
        torch.cuda.synchronize()
    else:
        x_data_init = _rand_signed(SEQ * d, half=HALF_X)

    H = d // d_h                       # Llama 2 7B: 4096 / 128 = 32 heads

    # ---------- Single-tape proof ----------
    # All layers (+ optional tail) record claims on ONE Tape. The residual
    # flows between layers as a WitnessTensor directly — no per-layer
    # fingerprint chain needed since everything is bound in one proof.
    # Lookup tables (SiLU, RmsNorm range, softmax exp) are shared across
    # all layers via Tape's register_table cache, so their TableSettlement
    # constraints are emitted once instead of N_LAYERS times.
    tape = Tape(CFG, silu_config=SILU_CFG, lazy=engine, time_ops=time_ops)

    # Initial residual: prompt-bound via EmbeddingLookupClaim, or random.
    if from_hf is not None and prompt is not None:
        E_wt = tape.commit(
            "E_embedding_subset", E_subset_data, E_subset_shape)
        resid = tape.embed(E_wt, token_ids=subset_token_ids_saved, d=d)
    else:
        resid = tape.commit("x_input", x_data_init, (SEQ, d))

    # Optional lazy HF loader: one instance shared across all layers; each
    # weight loads from disk on demand. Avoids holding 32×~1.6 GB of
    # weights in tape.inputs simultaneously, which would trip the DGX
    # Spark GB10 unified-memory driver bug (pytorch issue #174358).
    lazy_loader = None
    if lazy_weights:
        if from_hf is None:
            raise ValueError("--lazy-weights requires --from-hf "
                             "(random weights are generated, not loaded)")
        from loader import LazyHFLoader
        lazy_loader = LazyHFLoader(from_hf, S=S, d_h=d_h)
        print(f"  lazy weights: enabled (per-weight on-demand load from "
              f"{lazy_loader.model_dir})")

    # Record each transformer block on the same tape. resid flows through.
    for L in range(num_layers):
        this_layer_idx = layer_idx + L
        if lazy_loader is not None:
            layer_weights = _commit_weights_from_hf_lazy(
                tape, lazy_loader, this_layer_idx)
        elif from_hf is not None:
            layer_weights = _commit_weights_from_hf(
                tape, from_hf, this_layer_idx)
        else:
            layer_weights = _commit_weights_random(
                tape, layer_idx=this_layer_idx)
        resid = _run_block(tape, resid, layer_weights, H=H)
        del layer_weights
        torch.cuda.empty_cache()

    # Optional tail: final RmsNorm + LM head, on the same tape.
    logits = None
    vocab_size = None
    if with_lm_head:
        if from_hf is not None:
            from loader import load_final_weights, free_model_cache
            fw = load_final_weights(from_hf, S=S)
            final_norm_data = fw["final_norm_w"]
            lm_head_data    = fw["W_lm_head"]
            vocab_size = lm_head_data.numel() // d
            free_model_cache()
        else:
            vocab_size = 32000              # Llama-2-7B vocab; matches HF
            final_norm_data = torch.full((d,), S, dtype=torch.uint64, device="cuda")
            lm_head_data    = _rand_signed(d * vocab_size, half=HALF)
        final_norm_w_wt = tape.commit("final_norm_w", final_norm_data, (d,))
        W_lm_head_wt    = tape.commit("W_lm_head", lm_head_data, (d, vocab_size))
        # For --unexplained-info, run the LM head COARSE (s_out=1) so the
        # bound-relevant gaps fit a feasible exp/gap table -- bound by the
        # matmul's own rescale (no separate coarsening claim).
        _lm_s_out = ui_lm_sout if unexplained_info else S
        _lm_ow = ui_lm_ow if unexplained_info else OUTPUT_WIDTH
        logits = _run_tail(tape, resid, final_norm_w_wt, W_lm_head_wt,
                            vocab_size=vocab_size, lm_s_out=_lm_s_out, lm_ow=_lm_ow)

    # Optional: unexplained-information bound over HIDDEN, committed output tokens.
    ui_Sz = ui_info = None
    if unexplained_info:
        assert with_lm_head and logits is not None, "--unexplained-info needs the LM head"
        # The U-bound's argmax (A) and log-pin (b) are committed from the LM-head
        # logits, so they must be materialized before the build. In lazy/engine
        # mode the forward is deferred -> run an engine pass to compute the logits,
        # then zero the mult tables so the streaming prover's replay re-accumulates
        # them from scratch (no double count). The forward is recomputed in prove.
        if logits.data is None:
            _live = tape.run_engine_pass(free_intermediates=True, keep={logits.var})
            logits._data = _live[logits.var]
            for _v in list(tape.inputs):
                if _v.name.endswith("_mult"):
                    tape.inputs[_v].zero_()
        # Optionally bound U over only a SUBSET of positions (e.g. the generated
        # output positions, excluding the prompt). Public position select via embed.
        # `ui_positions` bounds U over a SUBSET of positions (e.g. only the generated
        # output positions): surprisal is computed for all SEQ positions, but U sums
        # only those (the Σ chain selects them via embed(surprisal,[i],d=1), 1|ELL ok).
        ui_Sz, ui_info = _run_unexplained_info(
            tape, logits, vocab_size=vocab_size, seq=SEQ,
            output_tokens=ui_output_tokens, s_c=ui_s_c, sum_positions=ui_positions)

    # ---------- Forward-only fast path (accuracy checks; no proof) ----------
    # Logits are primary witnesses, so the engine/forward pass produces them
    # without any of the proof machinery. Reuses the whole model build above.
    if forward_only:
        torch.cuda.synchronize(); _t = time.time()
        if engine:
            # Free each layer's witnesses as the residual moves on — only the
            # logits are kept — so peak memory is O(one layer), not O(all 32).
            keep = {logits.var} if (with_lm_head and logits is not None) else set()
            tape.run_engine_pass(free_intermediates=True, keep=keep)
        torch.cuda.synchronize()
        print(f"  forward-only (no proof): witness in {time.time() - _t:.2f}s")
        if with_lm_head:
            ls = logits.data.view(SEQ, vocab_size).to(torch.int64).clone()
            print(f"  argmax(logits[{SEQ-1}]) = token {int(ls[SEQ-1].argmax())}")
            if save_logits is not None:
                import numpy as np
                np.save(save_logits, ls.cpu().numpy())
                print(f"  saved logits ({SEQ}, {vocab_size}) int64 → {save_logits}")
        if ui_Sz is not None:
            _report_unexplained_info(tape, ui_Sz, ui_info, seq=SEQ, vocab_size=vocab_size)
        return logits

    # ---------- Single prove + verify ----------
    # Reveal the bound: compute Sz (engine pass), pin it as the public value,
    # then re-zero LogUp mult tables so the prove sweep re-accumulates cleanly.
    if ui_Sz is not None and ui_info is not None and ui_info.get("reveal_pin") is not None:
        from unexplained_info import bound_bits
        _live = tape.run_engine_pass(free_intermediates=True, keep={ui_Sz.var})
        _sz = int(_live[ui_Sz.var].cpu().item())
        ui_info["reveal_pin"].public_rhs = _sz
        for _v in list(tape.inputs):
            if _v.name.endswith("_mult"):
                tape.inputs[_v].zero_()
        print(f"  reveal: Sz = {_sz} pinned as PUBLIC bound = "
              f"{bound_bits(_sz, s_b=ui_info['s_b']):.4f} bits "
              f"(verifier reads this from the claim)")

    from collections import Counter
    counts = Counter(type(c).__name__ for c in tape.claims)
    tail_str = " + tail" if with_lm_head else ""
    print(f"\n--- single-tape proof: {num_layers} layer(s){tail_str} ---")
    print(f"  claims: {len(tape.claims)}  ({dict(sorted(counts.items()))})")
    torch.cuda.synchronize()
    t0 = time.time()
    proof = tape.prove(seed=b"xformer-single-tape", verbose=verbose)
    torch.cuda.synchronize()
    t_prove = time.time() - t0
    # Verification is the standalone Rust verifier's job (verifier-rs/verify_proof
    # on the dumped proof); the demo only proves + dumps.

    # Optional: dump the proof + claims + seeds for the standalone Rust verifier
    # (verifier-rs/verify_proof). Public data only — no witness.
    _dump = dump_proof or os.environ.get("LIGERO_DUMP_PROOF")
    if _dump:
        import protocol as pr
        from proof_dump import dump_proof as _write_proof   # single block-driven writer
        _t_dump = time.time()
        s_op, s_comb, s_col = pr.round_seeds(b"xformer-single-tape")
        Q = list(pr.random_columns(s_col, CFG))
        _write_proof(_dump, pr.claims_to_json(tape.claims, CFG),
                     {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
                     proof, Q, None)
        print(f"  dumped proof for Rust verifier → {_dump}  (dump {time.time()-_t_dump:.1f}s)")
    print(f"  prove: {t_prove:.2f}s   (verify the dump with verify_proof)")

    if with_lm_head:
        # argmax(logits[last position]) is the predicted next token. The
        # verifier could compute the same from the committed `logits`.
        logits_signed = logits.data.view(SEQ, vocab_size).to(torch.int64).clone()
        next_tok = int(logits_signed[SEQ - 1].argmax().item())
        print(f"  argmax(logits[{SEQ-1}]) = token {next_tok}")
        if save_logits is not None:
            import numpy as np
            np.save(save_logits, logits_signed.cpu().numpy())
            print(f"  saved logits ({SEQ}, {vocab_size}) int64 → {save_logits}")
        if from_hf is not None:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(from_hf)
                print(f"  decoded: {tok.decode([next_tok])!r}")
            except Exception as e:
                print(f"  (couldn't decode token: {e})")

    if ui_Sz is not None:
        _report_unexplained_info(tape, ui_Sz, ui_info, seq=SEQ, vocab_size=vocab_size)

    print(f"\n=== total ===")
    print(f"  claims:  {len(tape.claims)}")
    print(f"  prove:   {t_prove:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-hf", type=str, default=None,
                     help="HF model id (e.g. 'meta-llama/Llama-2-7b-hf') or "
                          "path. Loads + quantizes weights from BF16; needs "
                          "the `transformers` library and either a local "
                          "checkpoint or HF auth.")
    ap.add_argument("--layer-idx", type=int, default=0,
                     help="First transformer layer to load when --from-hf is set. "
                          "Layers [layer_idx, layer_idx + num_layers) run.")
    ap.add_argument("--num-layers", type=int, default=1,
                     help="Number of consecutive transformer blocks to chain "
                          "in the proof. Each block commits its own weight set.")
    ap.add_argument("--no-lm-head", action="store_true",
                     help="Skip the final RmsNorm + LM head tail proof "
                          "(default is to run it; produces committed logits "
                          "and prints argmax for the next-token prediction).")
    ap.add_argument("--prompt", type=str, default=None,
                     help="Prompt to tokenize and embed (uses --from-hf's "
                          "tokenizer + embedding table). When set, x is "
                          "produced via EmbeddingLookupClaim instead of "
                          "random. Truncated/padded to SEQ tokens.")
    ap.add_argument("--prompt-file", type=str, default=None,
                     help="Read the prompt from a file instead of --prompt "
                          "(avoids shell-quoting a long prompt). Overrides "
                          "--prompt when set. E.g. --prompt-file demo_prompt.txt "
                          "for the SEQ=1000 demo run.")
    ap.add_argument("--seq", type=int, default=None,
                     help="Override sequence length (default 2). Larger SEQ "
                          "scales activation cost linearly; attention "
                          "contributions scale as SEQ². Prompt is padded with "
                          "token 0 if shorter than SEQ.")
    ap.add_argument("--save-logits", type=str, default=None,
                     help="If set, save committed logits (int64, shape SEQ×vocab) "
                          "as .npy to this path for post-hoc comparison vs HF.")
    ap.add_argument("--time-ops", action="store_true",
                     help="Print cuda-synchronised wall time for each per-claim "
                          "compute_fn + side-effects dispatch (one line per claim). "
                          "Useful to find hot ops; grep by claim type to aggregate.")
    ap.add_argument("--verbose", action="store_true",
                     help="Per-phase wall time + GPU memory snapshots for "
                          "prove/verify. Useful for diagnosing where memory "
                          "peaks at large SEQ.")
    ap.add_argument("--lazy-weights", action="store_true",
                     help="Load HF weights on-demand from .safetensors shards "
                          "instead of eagerly into tape.inputs. Required for "
                          "multi-layer single-tape on unified-memory systems "
                          "where holding all weights simultaneously trips the "
                          "DGX Spark GB10 driver bug (pytorch #174358).")
    ap.add_argument("--engine", action="store_true",
                     help="Build the Tape in lazy mode: compute_fn dispatch "
                          "and per-claim side effects defer to a single engine "
                          "pass right before prove. tape.inputs only holds "
                          "externally-committed values; intermediates flow "
                          "through a live dict. Required to scale single-tape "
                          "Llama-2-7B past ~7 layers — combined with "
                          "--lazy-weights, never holds the whole layer set in "
                          "memory at once.")
    ap.add_argument("--forward-only", action="store_true",
                     help="Skip the proof: run only the witness/forward pass "
                          "and emit logits. For accuracy/quantization checks "
                          "(pairs with --save-logits + the fp16 comparison) "
                          "without paying for prove + verify.")
    ap.add_argument("--dump-proof", type=str, default=None,
                     help="Write the proof + claims + seeds as JSON to this path "
                          "for the standalone Rust verifier (verifier-rs/verify_proof). "
                          "Public data only, no witness. Falls back to the "
                          "LIGERO_DUMP_PROOF env var.")
    ap.add_argument("--unexplained-info", action="store_true",
                     help="After the LM head, append the unexplained-information "
                          "bound U(o) over the output tokens. The output tokens are "
                          "HIDDEN: committed + blinded exactly like the model weights "
                          "(never public). Requires the LM head + eager mode (no "
                          "--engine).")
    ap.add_argument("--ui-sigma", type=int, default=256,
                     help="Gaussian-kernel width sigma_g (int-logit units at scale S) "
                          "for --unexplained-info. Larger sigma_g -> larger exp table "
                          "(Z_max ~ sigma_g^2 * ln s_y). Default 256.")
    ap.add_argument("--ui-sy-bits", type=int, default=16,
                     help="Entropy-table output scale s_y = 2^bits for "
                          "--unexplained-info (default 16). s_y >> V keeps the "
                          "dropped-tail correction negligible.")
    ap.add_argument("--ui-output-tokens", type=str, default=None,
                     help="Comma-separated output token ids (length SEQ) for "
                          "--unexplained-info, committed + hidden. Default: the "
                          "greedy argmax of the int logits.")
    args = ap.parse_args()
    if args.seq is not None:
        SEQ = args.seq
    _prompt = args.prompt
    if args.prompt_file:
        _prompt = open(args.prompt_file).read()
    _ui_tokens = ([int(v) for v in args.ui_output_tokens.split(",")]
                  if args.ui_output_tokens else None)
    main(from_hf=args.from_hf, layer_idx=args.layer_idx,
         num_layers=args.num_layers, prompt=_prompt,
         with_lm_head=not args.no_lm_head, verbose=args.verbose,
         lazy_weights=args.lazy_weights, engine=args.engine,
         save_logits=args.save_logits, time_ops=args.time_ops,
         forward_only=args.forward_only,
         dump_proof=args.dump_proof,
         unexplained_info=args.unexplained_info, ui_sigma=args.ui_sigma,
         ui_sy_bits=args.ui_sy_bits, ui_output_tokens=_ui_tokens)
