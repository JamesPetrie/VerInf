"""Diagnostic: find the linear constraints the witness violates (A·w != b).

Gated behind LIGERO_CV_CHECK in prove(). Computes cv[cid] = A_cid·w for every
linear constraint directly from the full witness (which the prover has), then
reports the cids where cv != b and maps them back to the originating claim.
Used to localize the scale-dependent lin_sum failure at multi-layer SEQ=100
(lin_col passes, so A and q_lin are correct; this isolates which constraint's
A·w != b).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))  # pipeline/ on path
import torch
from collections import Counter
from cuda_primitives import P, gl_mul, gl_add
from packets import EXPANDERS

TWO32 = (1 << 32) % P


def cv_check(claims, witness, per_row_packets, ch0, cfg, m_total):
    import core
    ELL = cfg.ELL

    all_vars = core._layout(claims, cfg)[0]
    _, _, b_chunks, n_lin = core._compile_with_chs(claims, ch0, cfg, m_total)

    # Per-claim cid ranges, mirroring _compile_with_chs's sequential counting.
    ranges = []                                   # (claim_idx, type_name, base, n)
    nc = 0
    for ci, (c, ch) in enumerate(zip(claims, ch0)):
        _, _, n_added, _ = core.COMPILE_FNS[type(c)](c, ch, cfg, nc)
        ranges.append((ci, type(c).__name__, nc, n_added))
        nc += n_added
    assert nc == n_lin, f"{nc} != {n_lin}"

    # Full (m_total, ELL) message matrix: each variable at its absolute row_start.
    msgs = torch.zeros((m_total, ELL), dtype=torch.uint64, device="cuda")
    for v in all_vars:
        val = witness[v]
        val = val() if callable(val) else val
        data = core._to_device_u64(val).reshape(-1)
        rs, full = v.row_start, data.numel()
        rf = full // ELL
        if rf:
            msgs[rs:rs + rf] = data[:rf * ELL].view(rf, ELL)
        rem = full - rf * ELL
        if rem:
            msgs[rs + rf, :rem] = data[rf * ELL:]
    w_flat = msgs.reshape(-1)

    # cv[cid] = Σ coef·w[target], mod P via lo/hi 32-bit int64 index_add
    # (each cid's term count fits int64 at this scale).
    cv_lo = torch.zeros(n_lin, dtype=torch.int64, device="cuda")
    cv_hi = torch.zeros(n_lin, dtype=torch.int64, device="cuda")
    CH = 2048
    for lo in range(0, m_total, CH):
        hi = min(lo + CH, m_total)
        by_kind = {}
        for r in range(lo, hi):
            for pkt in per_row_packets[r]:
                by_kind.setdefault(type(pkt), ([], []))
                by_kind[type(pkt)][0].append(pkt)
                by_kind[type(pkt)][1].append(r - lo)
        for kind, (pkts, lrows) in by_kind.items():
            t, c, v = EXPANDERS[kind](pkts, lrows, lo, ELL)
            tgt = (t.to(torch.int64) + lo * ELL)
            w_g = w_flat.view(torch.int64).index_select(0, tgt).view(torch.uint64)
            prod = gl_mul(v.contiguous(), w_g)              # uint64 < P
            vi = prod.view(torch.int64)
            cid_i = c.to(torch.int64)
            cv_lo.index_add_(0, cid_i, vi & 0xFFFFFFFF)
            cv_hi.index_add_(0, cid_i, (vi >> 32) & 0xFFFFFFFF)

    # Recombine cv mod P on GPU (cv_lo, cv_hi >= 0 and < 2^63 < P here).
    cv = gl_add(gl_mul(cv_hi.view(torch.uint64),
                       torch.full((n_lin,), TWO32, dtype=torch.uint64, device="cuda")),
                cv_lo.view(torch.uint64))

    b = torch.zeros(n_lin, dtype=torch.uint64, device="cuda")
    for base, bc in b_chunks:
        b[base:base + bc.numel()] = bc

    bad_t = (cv != b).nonzero(as_tuple=True)[0]
    bad = bad_t.cpu().tolist()
    print(f"[cv_check] n_lin={n_lin}  violated linear constraints (cv != b): {len(bad)}")

    def claim_of(cid):
        for ci, tn, base, n in ranges:
            if base <= cid < base + n:
                return ci, tn, base, n
        return None

    tally = Counter()
    for cid in bad:
        info = claim_of(cid)
        tally[(info[1], info[0]) if info else ("<none>", -1)] += 1
    for (tn, ci), cnt in sorted(tally.items(), key=lambda x: -x[1])[:25]:
        print(f"   claim #{ci} {tn}: {cnt} violated cids")
    show = bad_t[:8]
    cv_s = cv.view(torch.int64).index_select(0, show).view(torch.uint64).cpu().tolist()
    b_s  = b.view(torch.int64).index_select(0, show).view(torch.uint64).cpu().tolist()
    for k, cid in enumerate(bad[:8]):
        info = claim_of(cid)
        print(f"   cid={cid} cv={cv_s[k]} b={b_s[k]} "
              f"rel={cid - info[2] if info else '?'} claim={info}")
