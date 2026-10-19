"""Fidelity metrics and the pre-registered acceptance of
analysis/k2-fidelity-study-design.md (§4, §6), independent of any one model:
per-position KL against the high-precision reference p64 and the
continuation's NLL, the routing quantities that explain them, and the paired
passage bootstrap that turns them into verdicts.

Every distribution is a float64 log-softmax over the full vocabulary,
computed in row chunks so a 1,000 × 163,840 logit block never exists
twice. KL is D_KL(p64 ‖ p_arm) in nats; NLL is in bits. The tolerances, the
bootstrap's size and seed and the percentile convention are the note's §6,
frozen 2026-10-19: change them there first, by a dated amendment."""
import math

import numpy as np

# ---- §6.2 and §6.3, frozen ------------------------------------------------------------

RATIO = 1.1                 # mean and p99 KL: integer <= RATIO x engine + ABS
ABS = 1e-6                  # nats, the allowance near zero
NLL_MARGIN = 0.01           # bits/token: integer - engine <= NLL_MARGIN
MAX_RATIO = 2.0             # maximum KL: integer <= MAX_RATIO x engine maximum + ABS
PERCENTILE = 99.0
PERCENTILE_METHOD = "linear"
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20261019
UPPER_Q, LOWER_Q = 95.0, 5.0    # the one-sided 95% bounds on the excess
CHUNK = 64                      # rows per log-softmax chunk

# ---- per position ------------------------------------------------------------------------

def log_softmax64(logits):
    """Rows of float64 log-probabilities from logits of any float dtype."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def position_metrics(ref_logits, arm_logits, targets, *, chunk=CHUNK):
    """Per scored position t: KL_t = Σ_v p64(v)(log p64(v) − log p_arm(v))
    in nats, and the arm's NLL_t = −log2 p_arm(targets[t]) in bits. Row t of
    the logits is the distribution that scores targets[t]."""
    ref = np.asarray(ref_logits)
    arm = np.asarray(arm_logits)
    tgt = np.asarray(targets, dtype=np.int64)
    if ref.shape != arm.shape or ref.ndim != 2 or len(tgt) != ref.shape[0]:
        raise ValueError(f"shapes {ref.shape}, {arm.shape}, {tgt.shape} do not pair")
    T = ref.shape[0]
    kl, nll = np.empty(T), np.empty(T)
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        lr, la = log_softmax64(ref[a:b]), log_softmax64(arm[a:b])
        kl[a:b] = (np.exp(lr) * (lr - la)).sum(axis=1)
        nll[a:b] = -la[np.arange(b - a), tgt[a:b]] / math.log(2)
    return {"kl": kl, "nll": nll}


def nll_bits(logits, targets, *, chunk=CHUNK):
    """The arm's own NLL per position, without a reference (p64's for §6.5)."""
    lg = np.asarray(logits)
    tgt = np.asarray(targets, dtype=np.int64)
    out = np.empty(lg.shape[0])
    for a in range(0, lg.shape[0], chunk):
        b = min(lg.shape[0], a + chunk)
        out[a:b] = -log_softmax64(lg[a:b])[np.arange(b - a), tgt[a:b]] / math.log(2)
    return out


# ---- routing and residuals: explanation only (§4) --------------------------------------

def gate_matrix(top, w, E):
    """(T, E) gate weights, zero on unselected experts."""
    top = np.asarray(top)
    g = np.zeros((top.shape[0], E))
    np.put_along_axis(g, top, np.asarray(w, dtype=np.float64), axis=1)
    return g


def gate_distance(g_ref, g_arm):
    """½ Σ_e |g_ref − g_arm| per token: support changes and reweighting."""
    return 0.5 * np.abs(np.asarray(g_ref) - np.asarray(g_arm)).sum(axis=1)


def swapped_weight(g_ref, g_arm):
    """Per token, the reference's weight on experts the arm drops and the
    arm's weight on experts the reference does not select."""
    g_ref, g_arm = np.asarray(g_ref), np.asarray(g_arm)
    dropped = ((g_ref > 0) & (g_arm == 0)) * g_ref
    added = ((g_arm > 0) & (g_ref == 0)) * g_arm
    return {"dropped": dropped.sum(axis=1), "added": added.sum(axis=1)}


def support_differences(top_ref, top_arm):
    """Per token, how many of the arm's selected experts the reference does
    not select."""
    a, b = np.asarray(top_ref), np.asarray(top_arm)
    return np.asarray([len(set(b[t]) - set(a[t])) for t in range(a.shape[0])])


def selection_margins(q, k):
    """The k-th minus the (k+1)-th of q per token, for every token."""
    srt = -np.sort(-np.asarray(q, dtype=np.float64), axis=1)
    return srt[:, k - 1] - srt[:, k]


def residual_error(x_ref, x_arm):
    """‖x_arm − x_ref‖ / ‖x_ref‖ per position."""
    x_ref = np.asarray(x_ref, dtype=np.float64)
    d = np.asarray(x_arm, dtype=np.float64) - x_ref
    return np.linalg.norm(d, axis=-1) / np.maximum(np.linalg.norm(x_ref, axis=-1), 1e-300)


# ---- §6: the corpus, the bootstrap and the verdicts ----------------------------------

def p99(x):
    return float(np.percentile(x, PERCENTILE, method=PERCENTILE_METHOD))


def _excess(int_kl, int_nll, eng_kl, eng_nll):
    """The three bootstrapped rows' excess over their bounds (<= 0 passes)."""
    return {"mean_kl": float(int_kl.mean() - (RATIO * eng_kl.mean() + ABS)),
            "p99_kl": p99(int_kl) - (RATIO * p99(eng_kl) + ABS),
            "mean_nll": float(int_nll.mean() - eng_nll.mean() - NLL_MARGIN)}


def _verdict(lo, hi):
    return "pass" if hi <= 0 else ("fail" if lo > 0 else "inconclusive")


def acceptance(integer, engine, *, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, worst=10):
    """§6 for one integer configuration against the engine baseline.

    `integer` and `engine` are lists, one entry per frozen passage in the
    same order, each {"kl": per-position KL against p64, "nll": per-position
    NLL in bits} over that passage's scored positions. Positions pool with
    equal weight; the bootstrap resamples passages, both arms together."""
    if len(integer) != len(engine) or not integer:
        raise ValueError("one entry per passage, the same passages for both arms")
    for i, (a, b) in enumerate(zip(integer, engine)):
        if len(a["kl"]) != len(b["kl"]) or len(a["nll"]) != len(a["kl"]) \
                or len(b["nll"]) != len(b["kl"]):
            raise ValueError(f"passage {i}: the arms score different positions")
    cat = lambda rows, key, idx: np.concatenate([np.asarray(rows[j][key], dtype=np.float64)
                                                 for j in idx])
    N = len(integer)
    every = range(N)
    ik, inl, ek, enl = (cat(integer, "kl", every), cat(integer, "nll", every),
                        cat(engine, "kl", every), cat(engine, "nll", every))
    point = _excess(ik, inl, ek, enl)
    rng = np.random.default_rng(seed)
    boot = {k: np.empty(B) for k in point}
    for r in range(B):
        idx = rng.integers(0, N, N)
        ex = _excess(cat(integer, "kl", idx), cat(integer, "nll", idx),
                     cat(engine, "kl", idx), cat(engine, "nll", idx))
        for k, v in ex.items():
            boot[k][r] = v
    rows = {}
    for k, v in point.items():
        lo = float(np.percentile(boot[k], LOWER_Q, method=PERCENTILE_METHOD))
        hi = float(np.percentile(boot[k], UPPER_Q, method=PERCENTILE_METHOD))
        rows[k] = dict(excess=v, lower95=lo, upper95=hi, verdict=_verdict(lo, hi))
    max_excess = float(ik.max() - (MAX_RATIO * ek.max() + ABS))
    rows["max_kl"] = dict(excess=max_excess, verdict="pass" if max_excess <= 0 else "fail")
    verdicts = [r["verdict"] for r in rows.values()]
    overall = ("accept" if all(v == "pass" for v in verdicts)
               else "fail" if "fail" in verdicts else "inconclusive")
    where = [(j, t) for j in every for t in range(len(integer[j]["kl"]))]

    def worst_of(key_arm):
        flat = ik if key_arm == "integer" else ek
        order = np.argsort(-flat, kind="stable")[:worst]
        return [dict(passage=where[i][0], position=where[i][1], integer_kl=float(ik[i]),
                     engine_kl=float(ek[i])) for i in order]
    stats = {arm: dict(mean_kl=float(k.mean()), p99_kl=p99(k), max_kl=float(k.max()),
                       mean_nll=float(n.mean()))
             for arm, k, n in (("integer", ik, inl), ("engine", ek, enl))}
    return dict(overall=overall, rows=rows, stats=stats, passages=N, positions=len(ik),
                worst_integer=worst_of("integer"), worst_engine=worst_of("engine"),
                bootstrap=dict(B=B, seed=seed, method=PERCENTILE_METHOD))


def nll_calibration(reference_nll, engine_nll, *, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """§6.5: p64's own NLL excess over the engine on the engine's continuation,
    NLL_64 − NLL_engine − NLL_MARGIN, with the same paired bootstrap. Reported
    beside the criterion, never part of it."""
    N = len(reference_nll)
    if N != len(engine_nll) or not N:
        raise ValueError("one entry per passage for both")
    cat = lambda rows, idx: np.concatenate([np.asarray(rows[j], dtype=np.float64) for j in idx])
    ex = lambda idx: float(cat(reference_nll, idx).mean() - cat(engine_nll, idx).mean()
                           - NLL_MARGIN)
    rng = np.random.default_rng(seed)
    boot = np.asarray([ex(rng.integers(0, N, N)) for _ in range(B)])
    lo = float(np.percentile(boot, LOWER_Q, method=PERCENTILE_METHOD))
    hi = float(np.percentile(boot, UPPER_Q, method=PERCENTILE_METHOD))
    return dict(excess=ex(range(N)), lower95=lo, upper95=hi, verdict=_verdict(lo, hi))
