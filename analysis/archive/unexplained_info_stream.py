"""Unexplained information U(o) for a real token stream -- the faithful version.

  float inference (fp16)        -> o, the output tokens (what the prover emits)
  int recomputation (Ligero/field forward, demo_llama7b) -> per-position logits
  U(o) = Sum_t -log2 q(o_t)     scored against the INT recomputation's logits

U(o) is large exactly where the int recomputation's argmax differs from the token
the float run emitted -- the int-vs-float inference disagreement that the noise
model sigma is meant to absorb. There is no "float U": float's only job is to
produce o. This is a MEASUREMENT (it proves nothing); it is what a sound proof
would over-estimate.

Run on the Spark:  ~/venv-hf/bin/python analysis/unexplained_info_stream.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))  # pipeline/

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "demo"))
import demo_llama7b as demo
from unexplained_info import stream_unexplained_information

MODEL = "meta-llama/Llama-2-7b-hf"
S = 1 << 12   # demo's logit scale (SCALE_BITS = 12)
P = (1 << 64) - (1 << 32) + 1


def _field_to_real(fld_u64):
    """Goldilocks field uint64 -> real logit. Negative reals are stored as P-|v|
    (>= 2^63), so their int64 reinterpretation is off by 2^64 - P = 2^32 - 1."""
    s = fld_u64.to(torch.int64)
    s = torch.where(s < 0, s + ((1 << 32) - 1), s)
    return s.double() / S


def float_output(prompt, max_new_tokens):
    """fp16 greedy decode -> (full_ids = input+output, n_in, output text)."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    inp = tok(prompt, return_tensors="pt").input_ids.cuda()
    gen = model.generate(inp, max_new_tokens=max_new_tokens, do_sample=False)
    n_in = inp.shape[1]
    text = tok.decode(gen[0, n_in:], skip_special_tokens=True)
    full_ids = gen[0].cpu().tolist()
    del model
    torch.cuda.empty_cache()
    return full_ids, n_in, text


def int_recompute_logits(full_ids):
    """Ligero/field forward over the exact id stream -> (L, V) real logits."""
    logits_wt = demo.main(from_hf=MODEL, num_layers=32, token_ids=full_ids,
                          with_lm_head=True, lazy_weights=True, engine=True,
                          forward_only=True)
    L = len(full_ids)
    V = logits_wt.var.length // L
    return _field_to_real(logits_wt.data.view(L, V))


def main():
    prompt = "The capital of France is"
    full_ids, n_in, out_text = float_output(prompt, max_new_tokens=24)
    print(f"prompt : {prompt!r}")
    print(f"greedy : {out_text!r}\n")

    logits = int_recompute_logits(full_ids).cuda()         # int recomputation, (L, V)
    o = torch.tensor(full_ids[n_in:], device="cuda")
    pred = logits[n_in - 1: len(full_ids) - 1]             # logits predicting each output token

    n_dis = int((pred.argmax(dim=1) != o).sum().item())
    print(f"\nint recompute argmax disagrees with the float output at "
          f"{n_dis}/{len(o)} positions")
    for sigma in (0.5, 1.0, 2.0):
        U, _ = stream_unexplained_information(pred, o, sigma)
        print(f"  sigma={sigma:4.1f}: U(o) = {U:8.3f} bits  ({U / len(o):.3f}/tok)")


if __name__ == "__main__":
    main()
