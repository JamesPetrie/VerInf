#!/usr/bin/env python3
"""Convert the proven LM-head logits (/tmp/logits_full100.npy, int64 field-cast)
to real-valued logits and report per-position top-1 predictions.

The saved int64 is the committed field element bit-cast: field-positives are
small >=0; a real-negative logit -|x| is stored as P-|x| and bit-casts to
~-(2^32-1)-|x|. Recover signed, then divide by the logit scale S=4096 (the
LM-head matmul emits s_out=S)."""
import re, numpy as np, torch
from transformers import AutoTokenizer

MASK32 = (1 << 32) - 1
S = 4096.0
MODEL = "meta-llama/Llama-2-7b-hf"

txt = open("/tmp/full100_out.txt").read()
m = re.search(r"prompt tokens \(first \d+\): (\[[0-9,\s]+\])", txt)
toks = eval(m.group(1)); n = len(toks)

arr = np.load("/tmp/logits_full100.npy").astype(np.int64)        # [SEQ, vocab]
print(f"raw logits npy: shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}")
signed = np.where(arr < 0, arr + MASK32, arr)                    # field -> signed int
real = signed.astype(np.float64) / S                            # real-valued logits
np.save("/tmp/logits_full100_real.npy", real)
print(f"real logits: shape={real.shape}  range [{real.min():.2f}, {real.max():.2f}]  -> /tmp/logits_full100_real.npy")

tok = AutoTokenizer.from_pretrained(MODEL)
argmax = real.argmax(axis=1)

# Cross-check the last-position argmax against the demo's printed value.
mdemo = re.search(r"argmax\(logits\[\d+\]\) = token (\d+)", txt)
if mdemo:
    print(f"cross-check pos {n-1}: this={int(argmax[n-1])}  demo={mdemo.group(1)}  "
          f"{'OK' if int(argmax[n-1])==int(mdemo.group(1)) else 'MISMATCH'}")

print("\n  pos | context token        argmax (top-1 next)     logit    actual-next logp")
print("  " + "-"*78)
logp = torch.log_softmax(torch.from_numpy(real), dim=-1).numpy()
nll = []
for i in range(n):
    pred = int(argmax[i]); plog = float(logp[i, pred])
    nxt = int(toks[i+1]) if i+1 < n else None
    nxt_lp = float(logp[i, nxt]) if nxt is not None else float("nan")
    if nxt is not None: nll.append(-nxt_lp)
    hit = " <hit" if (nxt is not None and pred == nxt) else ""
    if i < 12 or i >= n-4:
        print(f"  {i:3d} | {tok.decode([toks[i]])!r:14} {tok.decode([pred])!r:16} {real[i,pred]:8.2f}   "
              f"{(tok.decode([nxt]) if nxt is not None else '<END>')!r:12} {nxt_lp:7.3f}{hit}")
acc = sum(1 for i in range(n-1) if int(argmax[i])==int(toks[i+1]))/(n-1)
print("  " + "-"*78)
print(f"\nnext-token prediction after the 100-token context (pos {n-1}): "
      f"{tok.decode([int(argmax[n-1])])!r} (token {int(argmax[n-1])})")
print(f"teacher-forced mean NLL = {np.mean(nll):.4f} -> perplexity {np.exp(np.mean(nll)):.2f}; "
      f"top-1 next-token acc on the 99 transitions = {acc*100:.1f}%")
