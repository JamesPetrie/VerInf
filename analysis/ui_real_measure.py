"""Real-model unexplained-information MEASUREMENT (forward-only, no proof).

fp16 Llama-2-7b greedy-generates 50 tokens from a ~50-token prompt; the quantized
INT 32-layer forward recomputes logits over the same 100-token sequence; U(o) is
the surprise of the fp16 outputs under the INT model. Saves both (100,32000) logit
sets for the fp-vs-int comparison (which is what calibrates sigma).

  PYTHONPATH=~/ligero/pipeline ~/venv-hf/bin/python analysis/ui_real_measure.py
"""
import sys
import pathlib
import json
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))
import torch
import numpy as np

MODEL = str(pathlib.Path("~/models/llama-2-7b-hf").expanduser())
PROMPT = ("The history of the Roman Empire covers the period from 27 BC, when "
          "Augustus became the first emperor, through the gradual decline of the "
          "Western Empire in the fifth century. At its height the empire controlled "
          "territory across three continents and tens of millions of people, with a "
          "professional army, paved roads, aqueducts, and a common legal code.")
N_IN, N_OUT, S = 50, 50, 4096
OUT = pathlib.Path("/tmp/ui_real")


def _unsigned_to_signed(a):
    """Undo the demo's uint64-field-rep -> int64 wrap: a negative logit was stored
    as P-|v| then .to(int64)'d to -(2^32-1)-|v|; add (2^32-1) back on the negatives."""
    a = a.astype(np.int64)
    return a + (a < 0) * ((1 << 32) - 1)


def fp16_generate():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).cuda().eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids[0][:N_IN].cuda()
    with torch.no_grad():
        gen = model.generate(ids.unsqueeze(0), max_new_tokens=N_OUT,
                             do_sample=False)[0]
    seq = gen[:N_IN + N_OUT]
    with torch.no_grad():
        fp = model(seq.unsqueeze(0)).logits[0].float().cpu().numpy()   # (100, 32000)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "fp_logits.npy", fp)
    seq_list = [int(x) for x in seq.cpu().tolist()]
    json.dump(seq_list, open(OUT / "seq.json", "w"))
    print(f"  fp16: {N_IN} prompt tok, greedy-generated {N_OUT}; seq len {len(seq_list)}")
    print(f"  continuation: {tok.decode(seq_list[N_IN:])!r}")
    del model
    torch.cuda.empty_cache()
    return seq_list


def int_forward(seq_list):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "demo"))
    import demo_llama7b
    demo_llama7b.main(from_hf=MODEL, num_layers=32, token_ids=seq_list,
                      forward_only=True, lazy_weights=True, engine=True,
                      with_lm_head=True, save_logits=str(OUT / "int_logits.npy"))
    return _unsigned_to_signed(np.load(OUT / "int_logits.npy"))        # (100,32000)


def compute_U(fp, int_signed, seq_list):
    int_real = int_signed.astype(np.float64) / S                       # real logit units
    pos = list(range(N_IN - 1, N_IN + N_OUT - 1))                      # 49..98 predict 50..99
    out_ids = seq_list[N_IN:N_IN + N_OUT]

    fp_arg = fp[pos].argmax(1)
    int_arg = int_real[pos].argmax(1)
    out = np.array(out_ids)
    print(f"  sanity  fp argmax == generated token: {(fp_arg == out).mean()*100:.0f}%")
    print(f"  int argmax == fp output token:        {(int_arg == out).mean()*100:.1f}%  "
          f"(int agreeing with fp's choice)")
    gaps_o = np.array([int_real[p].max() - int_real[p][o] for p, o in zip(pos, out_ids)])
    print(f"  gap_o (int's gap to the fp output, real logit units): "
          f"mean {gaps_o.mean():.3f}, median {np.median(gaps_o):.3f}, max {gaps_o.max():.3f}")
    dperp = np.abs(fp[pos] - int_real[pos])
    print(f"  |fp - int| logit (real): mean {dperp.mean():.4f}, "
          f"p99 {np.percentile(dperp, 99):.3f}, max {dperp.max():.3f}")

    print("  U(sigma) over the 50 hidden output tokens:")
    for sigma in [0.1, 0.3, 1.0, 3.0, 10.0]:
        s_c = 2.0 * sigma * sigma
        U = 0.0
        for p, o in zip(pos, out_ids):
            l = int_real[p]
            g2 = (l.max() - l) ** 2
            w = np.exp(-g2 / s_c)
            U += -math.log2(w[o] / w.sum())
        print(f"    sigma={sigma:>5}: U = {U:8.2f} bits  ({U/N_OUT:.3f} bits/token)")


def main():
    seq = fp16_generate()
    intl = int_forward(seq)
    compute_U(np.load(OUT / "fp_logits.npy"), intl, seq)
    print(f"  saved {OUT}/fp_logits.npy (100,32000 fp32), "
          f"{OUT}/int_logits.npy (int64), {OUT}/seq.json")


if __name__ == "__main__":
    main()
