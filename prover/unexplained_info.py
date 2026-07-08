"""Unexplained-information UPPER bound, Gaussian-kernel model, explicit a + log.

  q(o_t) = exp(-gap_o^2 / s_c) / Sum_i exp(-gap_i^2 / s_c),
  gap_i  = v*[t] - l[t,i] >= 0,  v*[t] = max_i l,  s_c = 2*sigma^2.

surprisal_t (nats) = gap_o^2 / s_c + ln(a_t / s_y),  a_t = Sum_i exp(-gap_i^2/s_c).

Composed from tested gadgets + ONE new claim (InfoFinalizeClaim):
  * MaxClaim (max_gap)        -> gap (>=0), and the HIDDEN output select gap_o
  * tape.paired_tlookup(EXP)  -> e_i = EXP[gap_i]            (the exp-kernel table)
  * tape.hadamard             -> gap_o^2
  * tape.paired_tlookup(POW)  -> pw_t = POW[b_t]             (the log table)
  * InfoFinalizeClaim         -> a = Sum e, a <= pw (log-pin), z_o = ceil(gap_o^2/k),
                                  surprisal = z_o + b
  * U = Sum_t surprisal / (s_b * ln2) bits, rounded up OUTSIDE the proof.

EVERY rounding is pushed so U_proved >= U_true (a sound UPPER bound):
  EXP[g] = max(1, ceil(exp(-g^2/s_c)*s_y))   (each e_i >= true  -> a up)
  b = smallest with POW[b] >= a   (POW[b]=floor(s_y*e^(b/s_b)); b >= s_b*ln(a/s_y))
  z_o = ceil(gap_o^2 / k)         (numerator up)
The far-token over-count is <= V (in s_y units) per row, i.e. <= V/(s_y*ln2) bits;
keep it tight with s_y >> V.
"""
import math

import numpy as np
import torch

from tape import WitnessTensor
from max_claim import max_gap
from ui_claim import info_finalize


def exp_table_values(gap_max, s_c, s_y):
    """EXP[g] = max(1, ceil(exp(-g^2/s_c) * s_y)), g in [0, gap_max).  np.uint64."""
    g = np.arange(gap_max, dtype=np.float64)
    v = np.ceil(np.exp(-(g * g) / s_c) * s_y)
    v = np.maximum(1.0, v)
    return v.astype(np.uint64)


def pow_table_size(V, s_b):
    return int(math.ceil(math.log(V) * s_b)) + 4


def pow_table_values(K, s_y, s_b):
    """POW[k] = floor(s_y * e^(k/s_b)), k in [0, K).  np.uint64 (<= V*s_y)."""
    k = np.arange(K, dtype=np.float64)
    return np.floor(s_y * np.exp(k / s_b)).astype(np.uint64)


def prove_unexplained_info(tape, logits, tokens, *, T, V, s_c, s_y, s_b, gap_max,
                           force_argmax=None, sum_positions=None, reveal=False,
                           O_ext=None):
    """Append the circuit. Returns (Sz, handles); U_bits = bound_bits(Sz, ...).

    `O_ext`: optional existing (T,V) one-hot to use as the output select
    (see max_gap) — ties the scored tokens to the tokens the model consumed."""
    assert len(tokens) == T
    assert s_c % s_b == 0 and (s_c // s_b) & (s_c // s_b - 1) == 0, \
        "need s_c/s_b a power of two (clean ceil divisor k)"
    k = s_c // s_b

    # ONE: exact max -> gap (>=0), hidden output select gap_o.
    gap, gap_o, neg_gap, vstar = max_gap(tape, logits, tokens, T=T, V=V,
                                          gap_max=gap_max, force_argmax=force_argmax,
                                          O_ext=O_ext)

    # exp-kernel table: e_i = EXP[gap_i] (also range-proves gap_i in [0,gap_max)).
    EXP = exp_table_values(gap_max, s_c, s_y)
    exp_tbl = tape.register_table("ui_exp", T_data=np.arange(gap_max, dtype=np.uint64),
                                  T_Y_data=EXP)
    e = tape.paired_tlookup(gap, exp_tbl)

    # output token's squared gap (gap_o in [0,gap_max) -> gap_o^2 < P).
    gap_o2 = tape.hadamard(gap_o, gap_o)

    # log-pin: b = smallest with POW[b] >= a, DERIVED by InfoFinalize at
    # witness time (lazy-safe); the tlookup AFTER it binds pw to ui_pow.
    K = pow_table_size(V, s_b)
    POW = pow_table_values(K, s_y, s_b)
    pow_tbl = tape.register_table("ui_pow", T_data=np.arange(K, dtype=np.uint64),
                                  T_Y_data=POW)

    # finalize: a = Sum e, a <= pw, z_o = ceil(gap_o^2/k), surprisal = z_o + b.
    surprisal, b, pw = info_finalize(tape, e, gap_o2, T=T, V=V, k=k,
                                      d_max=V * s_y, s_y=s_y, s_b=s_b, K=K)
    tape.paired_tlookup(b, pow_tbl, y_var=pw.var)   # binds pw = POW[b]
    Sz = _chain_sum(tape, surprisal, T, positions=sum_positions)
    handles = dict(gap=gap, gap_o=gap_o, e=e, b=b, pw=pw, surprisal=surprisal, Sz=Sz)
    if reveal:
        # Expose the bound: pin committed Sz to a public value (filled post-witness).
        handles['reveal_pin'] = tape.reveal(Sz, value=None)
    return Sz, handles


def _chain_sum(tape, surprisal, T, positions=None):
    """Sum surprisal over `positions` (default all 0..T) via Add + Embed (d=1).
    A subset bounds U over only those positions (e.g. the generated outputs); the
    other positions' surprisal is still computed but excluded from U."""
    positions = list(range(T)) if positions is None else list(positions)
    s = tape.embed(surprisal, token_ids=[positions[0]], d=1)
    for i in positions[1:]:
        s = tape.add(s, tape.embed(surprisal, token_ids=[i], d=1))
    return s


def bound_bits(Sz_val, *, s_b):
    """U bits = (Sum_t surprisal) / (s_b * ln2), rounded up (surprisal at scale s_b nats)."""
    return Sz_val / (s_b * math.log(2.0))


def unexplained_info_reference(logits_int, tokens, s_c):
    """Float reference U for the kernel exp(-gap^2/s_c) (base-e, no rounding)."""
    U = 0.0
    for t, o in enumerate(tokens):
        row = list(logits_int[t])
        vstar = max(row)
        w = [math.exp(-((vstar - x) ** 2) / s_c) for x in row]
        U += -math.log2(w[o] / sum(w))
    return U


def stream_unexplained_information(logits, tokens, s_c):
    """Float U(o) for a real token stream under exp(-gap^2/s_c)."""
    vstar = logits.max(dim=1, keepdim=True).values
    w = torch.exp(-((vstar - logits).double() ** 2) / s_c)
    W = w.sum(dim=1)
    w_o = w[torch.arange(logits.shape[0], device=logits.device), tokens]
    per = -torch.log2(w_o / W)
    return float(per.sum().item()), per
