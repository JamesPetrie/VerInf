"""Profile WHERE _softmax_witness_vec's ~2s/call goes: CPU numpy math (which
section?) vs host<->device transfer. Capture the real kwargs from a prove at
seq512, then replay a section-timed copy of the body and ASSERT it is
byte-identical to the original (so the timings are trustworthy and the copy is a
correctness baseline for a future GPU port)."""
import sys, os, time, copy
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
sys.path.insert(0, str(R / "analysis/bench"))
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
import compute_fns as CF
from core import LigeroConfig
from tape import Tape, _to_signed_np, _to_field_np

CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
D, DFF, DH, SEQ, NL = 512, 2048, 64, 512, 4
dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
H = D // DH

_orig = CF._softmax_witness_vec
_cap = {}
def _capture(x_in_uint64, **kw):
    n = kw['B'] * kw['M']
    if n >= _cap.get('_n', -1):          # keep the largest call's args
        _cap.clear(); _cap['_n'] = n
        _cap['x'] = x_in_uint64.copy()
        _cap['kw'] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
    return _orig(x_in_uint64, **kw)
CF._softmax_witness_vec = _capture


def build():
    tape = Tape(CFG, silu_config=dt.SILU_CFG, lazy=True)
    x = dt._rand_signed(SEQ * D, half=dt.HALF_X)
    resid = tape.commit("x_input", x, (SEQ, D))
    for L in range(NL):
        w = dt._commit_weights_random(tape, layer_idx=L)
        resid = dt._run_block(tape, resid, w, H=H)
    vocab = 64
    fn = torch.full((D,), dt.S, dtype=torch.uint64, device="cuda")
    lm = dt._rand_signed(D * vocab, half=dt.HALF)
    fnw = tape.commit("final_norm_w", fn, (D,))
    lmw = tape.commit("W_lm_head", lm, (D, vocab))
    dt._run_tail(tape, resid, fnw, lmw, vocab_size=vocab)
    return tape


torch.manual_seed(1234)
C._WITNESS_CACHE_ON = False   # ensure softmax actually computes (capture args)
build().prove(seed=b"sm-prof")
CF._softmax_witness_vec = _orig
x_in = _cap['x']; kw = _cap['kw']
B, M = kw['B'], kw['M']
print(f"captured softmax call: B={B} M={M}  (B*M={B*M:,})  saturate={kw['saturate']} causal={kw['causal']}")


def timed_body(x_in_uint64, *, B, M, s_x, s_c, s_y, T_A_np, T_B_np, Z_max,
               aux_chunk_width, saturate, Z_high_width, causal, heads, round_up=False):
    t = {}
    def mark(k, t0): t[k] = t.get(k, 0.0) + time.perf_counter() - t0
    s = time.perf_counter()
    x_signed = _to_signed_np(x_in_uint64).reshape(B, M)
    if causal:
        i_qry = (np.arange(B) // heads)[:, None]; j_idx = np.arange(M)[None, :]
        mask_2d = (j_idx > i_qry)
    else:
        mask_2d = np.zeros((B, M), dtype=bool)
    unmasked_2d = ~mask_2d
    INT_MIN = np.iinfo(np.int64).min; INT_MAX = np.iinfo(np.int64).max
    x_for_max = np.where(unmasked_2d, x_signed, INT_MIN)
    x_for_min = np.where(unmasked_2d, x_signed, INT_MAX)
    max_x = x_for_max.max(axis=1); min_x = x_for_min.min(axis=1)
    mark('1_input+mask+minmax', s)

    def s1_at(c2_b):
        z = c2_b[:, None] - x_signed
        in_range = (z >= 0) & (z < Z_max) & unmasked_2d
        z_clamped = np.where(in_range, z, 0)
        T_A_vals = T_A_np[z_clamped]
        ss = np.where(in_range, T_A_vals.astype(np.int64), 0)
        if round_up: ss = ss + (unmasked_2d & (z >= Z_max)).astype(np.int64)
        return ss.sum(axis=1)

    s = time.perf_counter()
    c2_lo = max_x.astype(np.int64).copy()
    c2_hi = (max_x + Z_max).astype(np.int64) if saturate else (min_x + Z_max - 1).astype(np.int64)
    s1_lo = s1_at(c2_lo); skip_search = s1_lo <= s_y
    if not saturate:
        s1_hi = s1_at(c2_hi)
    n_iter = max(1, (int(Z_max) - 1).bit_length()) + 2
    n_ran = 0
    for _ in range(n_iter):
        active = (c2_lo + 1 < c2_hi) & ~skip_search
        if not active.any(): break
        n_ran += 1
        c2_mid = (c2_lo + c2_hi) // 2
        s1_mid = s1_at(c2_mid)
        update_hi = (s1_mid <= s_y) & active; update_lo = (~(s1_mid <= s_y)) & active
        c2_hi = np.where(update_hi, c2_mid, c2_hi); c2_lo = np.where(update_lo, c2_mid, c2_lo)
    c2 = np.where(skip_search, c2_lo, c2_hi); s1 = s1_at(c2)
    mark(f'2_bsearch({n_ran}it,{n_iter}cap)', s)

    s = time.perf_counter()
    z_2d = c2[:, None] - x_signed
    in_range_2d = (z_2d >= 0) & (z_2d < Z_max) & unmasked_2d
    z_clamped_2d = np.where(in_range_2d, z_2d, 0)
    y_A_2d = np.where(in_range_2d, T_A_np[z_clamped_2d], np.uint64(0))
    y_B_2d = np.where(in_range_2d, T_B_np[z_clamped_2d], np.uint64(0))
    if round_up:
        far_2d = unmasked_2d & (z_2d >= Z_max)
        y_A_2d = np.where(far_2d, np.uint64(1), y_A_2d); y_B_2d = np.where(far_2d, np.uint64(1), y_B_2d)
    s2 = y_B_2d.astype(np.int64).sum(axis=1)
    r_lo = s_y - s1; r_hi = s2 - s_y - 1
    mark('3_ycell+s2', s)

    s = time.perf_counter()
    out = {"c2": _to_field_np(c2), "c2_shifted": _to_field_np(c2 + (1 << (aux_chunk_width - 1))),
           "z": None, "y_A": y_A_2d.reshape(-1).astype(np.uint64),
           "y_B": y_B_2d.reshape(-1).astype(np.uint64), "s1": _to_field_np(s1),
           "s2": _to_field_np(s2), "r_lo": _to_field_np(r_lo), "r_hi": _to_field_np(r_hi)}
    if saturate:
        z_low_unmasked = z_2d % np.int64(Z_max); z_high_unmasked = z_2d // np.int64(Z_max)
        if causal:
            z_low_2d = np.where(mask_2d, np.int64(0), z_low_unmasked)
            z_high_2d = np.where(mask_2d, np.int64(0), z_high_unmasked)
        else:
            z_low_2d, z_high_2d = z_low_unmasked, z_high_unmasked
        is_high_2d = (z_high_2d != 0).astype(np.int64)
        zl_clamped = np.where(mask_2d, 0, z_low_2d).astype(np.intp)
        yA_raw_2d = np.where(mask_2d, np.uint64(0), T_A_np[zl_clamped])
        yB_raw_2d = np.where(mask_2d, np.uint64(0), T_B_np[zl_clamped])
        is_high_bool = is_high_2d.astype(bool)
        mux_yA_2d = np.where(is_high_bool, yA_raw_2d, np.uint64(0))
        mux_yB_2d = np.where(is_high_bool, yB_raw_2d, np.uint64(0))
        out["z"] = z_low_2d.reshape(-1).astype(np.uint64); out["z_high"] = z_high_2d.reshape(-1).astype(np.uint64)
        out["is_high"] = is_high_2d.reshape(-1).astype(np.uint64)
        out["y_A_raw"] = yA_raw_2d.reshape(-1).astype(np.uint64); out["y_B_raw"] = yB_raw_2d.reshape(-1).astype(np.uint64)
        out["mux_y_A"] = mux_yA_2d.reshape(-1).astype(np.uint64); out["mux_y_B"] = mux_yB_2d.reshape(-1).astype(np.uint64)
    else:
        out["z"] = _to_field_np(z_2d.reshape(-1))
    mark('4_output+saturate', s)
    return out, t


REPS = 7
def med(xs): xs = sorted(xs); return xs[len(xs)//2]

# warm up (page in T_A/T_B, x_in) so cold-cache doesn't skew rep 1
_orig(x_in, **kw); timed_body(x_in, **kw)

# time _orig and the section-timed copy's TOTAL identically
tf, tb = [], []
for _ in range(REPS):
    t0 = time.perf_counter(); _orig(x_in, **kw); tf.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); out_t, t = timed_body(x_in, **kw); tb.append(time.perf_counter() - t0)
full = med(tf); copy_total = med(tb)

# section breakdown: sum the per-section marks from one representative rep,
# aggregated by median across reps
agg = {}
for _ in range(REPS):
    _o, t = timed_body(x_in, **kw)
    for k, v in t.items(): agg.setdefault(k, []).append(v)
agg = {k: med(v) for k, v in agg.items()}
sect_sum = sum(agg.values())

ref = _orig(x_in, **kw)
ok = all(np.array_equal(np.asarray(ref[k]), np.asarray(out_t[k])) for k in ref)
print(f"byte-identical copy vs original: {'YES' if ok else 'NO — timings suspect'}")
print(f"_orig full call   (median of {REPS}): {full*1e3:8.1f} ms")
print(f"copy total        (median of {REPS}): {copy_total*1e3:8.1f} ms")
print(f"section sum       (median):           {sect_sum*1e3:8.1f} ms  "
      f"({100*sect_sum/copy_total:.0f}% of copy total)")
for k in sorted(agg, key=lambda k: -agg[k]):
    print(f"  {k:<26} {agg[k]*1e3:8.1f} ms  ({100*agg[k]/copy_total:4.1f}% of copy)")
print("DONE")
