"""Reusable accuracy check: compare a forward-only quantized run's logits to
base fp16 Llama-2-7B on the same prompt.

Usage:  python compare_to_fp16.py [quant_logits.npy] [tokens_src.txt]
  quant_logits : raw int64 logits saved by `demo_llama7b.py --forward-only
                 --save-logits` (field-cast; converted here). Default
                 /tmp/logits_silufix.npy.
  tokens_src   : a demo stdout log containing the "prompt tokens" line.
                 Default /tmp/silufix_out.txt.

fp16 logits are cached in /tmp/fp16_logits.npy (delete it if the prompt
changes). Only the quantized run needs re-running between experiments."""
import sys, os, re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-2-7b-hf"
MASK32 = (1 << 32) - 1
quant_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/logits_silufix.npy"
toks_src   = sys.argv[2] if len(sys.argv) > 2 else "/tmp/silufix_out.txt"
FP16_CACHE = "/tmp/fp16_logits.npy"

_log = open(toks_src).read()
toks = eval(re.search(r"prompt tokens \(first \d+\): (\[[0-9,\s]+\])", _log).group(1))
# LM-head output scale tracks the run's scale knob; auto-detect from the log.
_sm = re.search(r"scale s=2\^(\d+)", _log)
S = float(1 << int(_sm.group(1))) if _sm else 4096.0
print(f"(logit scale S = {int(S)})")
n = len(toks)
tok = AutoTokenizer.from_pretrained(MODEL)

if os.path.exists(FP16_CACHE):
    fp16 = np.load(FP16_CACHE)
else:
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map="cuda")
    with torch.no_grad():
        fp16 = m(torch.tensor([toks], device="cuda")).logits[0].float().cpu().numpy()
    np.save(FP16_CACHE, fp16)

arr = np.load(quant_path).astype(np.int64)
q = np.where(arr < 0, arr + MASK32, arr).astype(np.float64) / S      # field -> real
print(f"quant={quant_path}  fp16 range [{fp16.min():.1f},{fp16.max():.1f}]  "
      f"quant range [{q.min():.1f},{q.max():.1f}]")

agree = (fp16.argmax(1) == q.argmax(1)).mean()
def ppl(L):
    lp = torch.log_softmax(torch.from_numpy(L), -1).numpy()
    return float(np.exp(np.mean([-lp[i, toks[i + 1]] for i in range(n - 1)])))
corrs = [np.corrcoef(fp16[i], q[i])[0, 1] for i in range(n)]
def top5(l): return set(np.argsort(l)[-5:])
ov = [len(top5(fp16[i]) & top5(q[i])) / 5 for i in range(n)]
def kl(p, qq):
    P = torch.softmax(torch.from_numpy(p), -1)
    return float((P * (torch.log_softmax(torch.from_numpy(p), -1)
                       - torch.log_softmax(torch.from_numpy(qq), -1))).sum())
kls = [kl(fp16[i], q[i]) for i in range(n)]

print(f"  argmax agreement : {agree*100:.1f}%    top-5 overlap : {np.mean(ov)*100:.1f}%")
print(f"  logit Pearson r  : {np.mean(corrs):.4f} (min {np.min(corrs):.4f})")
print(f"  KL(fp16||quant)  : {np.mean(kls):.3f} mean / {np.max(kls):.3f} max nats")
print(f"  perplexity       : fp16 {ppl(fp16):.2f}  |  quant {ppl(q):.2f}")
print(f"  next-tok@{n-1}: fp16 {[tok.decode([t]) for t in np.argsort(fp16[n-1])[-5:][::-1]]}")
print(f"             quant {[tok.decode([t]) for t in np.argsort(q[n-1])[-5:][::-1]]}")
