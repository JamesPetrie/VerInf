"""Isolated prototype + A/B: port _softmax_witness_vec (numpy/CPU) to GPU torch
int64, diff-test BIT-EXACT against the numpy path on REAL captured softmax
inputs (+ random variants), and time GPU vs numpy. No prover code touched yet;
this de-risks the port and predicts the win before wiring it in.

Field math: Goldilocks P = 2^64-2^32+1 > int64 max, so we replicate numpy's
uint64 field<->signed conversions exactly using order-preserving int64 tricks;
table entries (<2^62) and activations are int64-safe."""
import sys, os, time
from pathlib import Path
import numpy as np
R = Path("/home/riftuser/VerInf")
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
sys.path.insert(0, str(R / "analysis/bench"))
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
import compute_fns as CF
from core import LigeroConfig
from tape import Tape, _softmax_witness_vec

P = (1 << 64) - (1 << 32) + 1
P_HALF = (P - 1) // 2
FIELD_GAP = (1 << 64) - P                 # 2^32 - 1
INT64_MIN = -(1 << 63)
DEV = "cuda"


def to_signed_gpu(u_u64):
    """uint64 cuda tensor (values in [0,P)) -> signed int64 cuda tensor. Exact
    replica of numpy _to_signed_np: where(u <= P_HALF, u, u-P).view(int64)."""
    ui = u_u64.view(torch.int64)
    # order-preserving uint64 compare via sign-bit flip
    key_u = ui ^ INT64_MIN
    key_h = (torch.tensor(P_HALF, dtype=torch.uint64, device=DEV).view(torch.int64)) ^ INT64_MIN
    le = key_u <= key_h
    minus_P = ui + FIELD_GAP                # int64 wrap == uint64 (u-P) bits
    return torch.where(le, ui, minus_P)


def to_field_gpu(s_i64):
    """signed int64 cuda tensor -> uint64 field-rep cuda tensor. Replica of
    numpy _to_field_np: where(s>=0, s, s-FIELD_GAP).view(uint64)."""
    out = torch.where(s_i64 >= 0, s_i64, s_i64 - FIELD_GAP)
    return out.view(torch.uint64)


def gpu_softmax_witness(x_in_u64, *, B, M, s_x, s_c, s_y, T_A_np, T_B_np, Z_max,
                        aux_chunk_width, saturate, Z_high_width, causal, heads,
                        round_up=False):
    x_signed = to_signed_gpu(x_in_u64).reshape(B, M)                        # (B,M) int64
    T_A = torch.from_numpy(T_A_np.astype(np.int64)).to(DEV)                 # <2^62 fits int64
    T_B = torch.from_numpy(T_B_np.astype(np.int64)).to(DEV)
    if causal:
        i_qry = (torch.arange(B, device=DEV) // heads).unsqueeze(1)
        j_idx = torch.arange(M, device=DEV).unsqueeze(0)
        mask_2d = j_idx > i_qry
    else:
        mask_2d = torch.zeros((B, M), dtype=torch.bool, device=DEV)
    unmasked = ~mask_2d
    imin = torch.iinfo(torch.int64).min; imax = torch.iinfo(torch.int64).max
    max_x = torch.where(unmasked, x_signed, torch.full_like(x_signed, imin)).amax(dim=1)
    min_x = torch.where(unmasked, x_signed, torch.full_like(x_signed, imax)).amin(dim=1)

    Zt = int(Z_max)

    def s1_at(c2_b):
        z = c2_b.unsqueeze(1) - x_signed
        in_range = (z >= 0) & (z < Zt) & unmasked
        z_cl = torch.where(in_range, z, torch.zeros_like(z))
        TA = T_A[z_cl]
        s = torch.where(in_range, TA, torch.zeros_like(TA))
        if round_up:
            s = s + (unmasked & (z >= Zt)).to(torch.int64)
        return s.sum(dim=1)

    c2_lo = max_x.clone()
    c2_hi = (max_x + Zt) if saturate else (min_x + Zt - 1)
    s1_lo = s1_at(c2_lo)
    skip = s1_lo <= s_y
    n_iter = max(1, (Zt - 1).bit_length()) + 2
    for _ in range(n_iter):
        active = (c2_lo + 1 < c2_hi) & ~skip
        if not bool(active.any()):
            break
        c2_mid = (c2_lo + c2_hi) // 2
        s1_mid = s1_at(c2_mid)
        up_hi = (s1_mid <= s_y) & active
        up_lo = (~(s1_mid <= s_y)) & active
        c2_hi = torch.where(up_hi, c2_mid, c2_hi)
        c2_lo = torch.where(up_lo, c2_mid, c2_lo)
    c2 = torch.where(skip, c2_lo, c2_hi)
    s1 = s1_at(c2)

    z_2d = c2.unsqueeze(1) - x_signed
    in_range_2d = (z_2d >= 0) & (z_2d < Zt) & unmasked
    z_cl2 = torch.where(in_range_2d, z_2d, torch.zeros_like(z_2d))
    z0 = torch.zeros_like(z_2d)
    y_A_2d = torch.where(in_range_2d, T_A[z_cl2], z0)
    y_B_2d = torch.where(in_range_2d, T_B[z_cl2], z0)
    if round_up:
        far = unmasked & (z_2d >= Zt)
        one = torch.ones_like(z_2d)
        y_A_2d = torch.where(far, one, y_A_2d)
        y_B_2d = torch.where(far, one, y_B_2d)
    s2 = y_B_2d.sum(dim=1)
    r_lo = s_y - s1
    r_hi = s2 - s_y - 1

    out = {
        "c2": to_field_gpu(c2), "c2_shifted": to_field_gpu(c2 + (1 << (aux_chunk_width - 1))),
        "z": None, "y_A": to_field_gpu(y_A_2d.reshape(-1)),
        "y_B": to_field_gpu(y_B_2d.reshape(-1)), "s1": to_field_gpu(s1),
        "s2": to_field_gpu(s2), "r_lo": to_field_gpu(r_lo), "r_hi": to_field_gpu(r_hi),
    }
    if saturate:
        z_low_un = torch.remainder(z_2d, Zt)
        z_high_un = torch.div(z_2d, Zt, rounding_mode='floor')
        if causal:
            z_low = torch.where(mask_2d, z0, z_low_un)
            z_high = torch.where(mask_2d, z0, z_high_un)
        else:
            z_low, z_high = z_low_un, z_high_un
        is_high = (z_high != 0).to(torch.int64)
        zl_cl = torch.where(mask_2d, z0, z_low)
        yA_raw = torch.where(mask_2d, z0, T_A[zl_cl])
        yB_raw = torch.where(mask_2d, z0, T_B[zl_cl])
        is_high_b = is_high.to(torch.bool)
        mux_yA = torch.where(is_high_b, yA_raw, z0)
        mux_yB = torch.where(is_high_b, yB_raw, z0)
        out["z"] = z_low.reshape(-1).view(torch.uint64)
        out["z_high"] = z_high.reshape(-1).view(torch.uint64)
        out["is_high"] = is_high.reshape(-1).view(torch.uint64)
        out["y_A_raw"] = yA_raw.reshape(-1).view(torch.uint64)
        out["y_B_raw"] = yB_raw.reshape(-1).view(torch.uint64)
        out["mux_y_A"] = mux_yA.reshape(-1).view(torch.uint64)
        out["mux_y_B"] = mux_yB.reshape(-1).view(torch.uint64)
    else:
        out["z"] = to_field_gpu(z_2d.reshape(-1))
    return out


# ---- capture real args from a seq512 prove ----
CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
D, DFF, DH, SEQ, NL = 512, 2048, 64, 512, 4
dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
H = D // DH
_orig = CF._softmax_witness_vec
_cap = {}
def _capture(x_in_uint64, **kw):
    n = kw['B'] * kw['M']
    if n >= _cap.get('_n', -1):
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
C._WITNESS_CACHE_ON = False
build().prove(seed=b"gpu-sm")
CF._softmax_witness_vec = _orig
x_in = _cap['x']; kw = _cap['kw']
print(f"captured B={kw['B']} M={kw['M']} saturate={kw['saturate']} causal={kw['causal']} round_up={kw.get('round_up')}")

# ---- bit-exact check on the REAL captured inputs ----
ref = _orig(x_in, **kw)
x_gpu = torch.from_numpy(x_in).to(DEV)
got = gpu_softmax_witness(x_gpu, **kw)
allok = True
for k in ref:
    a = np.asarray(ref[k])
    b = got[k].cpu().numpy()
    ok = (a.shape == b.shape) and np.array_equal(a.view(np.uint64), b.view(np.uint64))
    if not ok:
        allok = False
        ndiff = int((a.view(np.uint64) != b.view(np.uint64)).sum()) if a.shape == b.shape else -1
        print(f"  MISMATCH {k}: shapes {a.shape} vs {b.shape}, ndiff={ndiff}")
print(f"REAL-input bit-exact: {'YES' if allok else 'NO'}")

# ---- timing A/B (matched inputs) ----
def time_numpy(reps=7):
    _orig(x_in, **kw)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); _orig(x_in, **kw); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts)//2]

def time_gpu(reps=20):
    gpu_softmax_witness(x_gpu, **kw); torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True); ts = []
    for _ in range(reps):
        st.record(); gpu_softmax_witness(x_gpu, **kw); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts.sort(); return ts[len(ts)//2]

np_ms = time_numpy() * 1e3
gpu_ms = time_gpu()
print(f"numpy witness call (warm median): {np_ms:8.1f} ms")
print(f"gpu   witness call (cuda-event) : {gpu_ms:8.1f} ms   -> {np_ms/gpu_ms:.1f}x")
print("DONE")
