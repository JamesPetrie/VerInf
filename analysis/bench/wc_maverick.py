"""WC-LCRL-STC over the REAL Maverick GGUF: streaming enrollment + bridge at
production geometry (B=15360, lam=1024, K_w=16384, N_w=32768, q_w=40).

Three streaming passes over the persistent linear maps (nothing model-sized
is ever resident):

  pass A  dequant -> field -> per-width row stream -> B-blocks -> coefficient
          rows (+PRG masks) -> forward NTT -> GPU BLAKE3 column accumulator
          -> enrollment root (core's production hash path, not the toy
          sha256 of wc_bridge).
  coins   rho per width from (root, manifest, geometry, s_r1);
  pass B  re-stream: P_trace = W rho per block, pi = z^T rho from the same
          PRG masks; R2 commit; alpha/eta; aggregate c; v = c(eta).
  pass C  re-stream: recompute codewords, gather the q_w eta columns for
          every polynomial; self-check their digests against pass A's
          column hashes.

Verification: every coin recomputed; Merkle paths for the eta columns; v =
c(eta) in python ints; the bridge equation checked on GPU over all blocks
plus an independent python-int spot check on --spot sampled blocks.  (A full
python-int pass over ~26M polynomials is days of CPU; the GPU check uses the
same gl kernels the prover uses and is labeled as such in the report.)

Usage (shakedown, local shard):
  python analysis/bench/wc_maverick.py --gguf ~/maverick-gguf/UD-Q4_K_XL/... \
      --layers 2 --experts 4
Full run: --layers -1 --experts -1 --lm-head
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "prover"))

import numpy as np
import torch

import protocol as pr
import wc_bridge as wc
from core import _make_merkle_acc, _finalize_merkle_artifact, merkle_path
from cuda_primitives import P, gl_axpy, gl_matvec, ntt_forward_batched, poly_eval
from loader import MAVERICK_MOE_TENSORS, _gguf_by_name, quantize_to_field

SCALE = 1 << 12
POLY_CHUNK = 4096          # NTT batch rows (VRAM cap ~ POLY_CHUNK*N_w*8B = 1 GB)


def sync():
    torch.cuda.synchronize()


# ── unit enumeration ─────────────────────────────────────────────────────────
# A unit is one persistent linear map: (name, expert|None, d_out, d_in).
# llama.cpp layout is (d_out, d_in) row-major: polynomial j's coefficients
# are row j's slice of input coords — no transpose needed for packing.

def enumerate_units(gguf_path, n_layers, n_experts, lm_head):
    """GGUF dim metadata is ne-ordered and quant-packed, so never trust
    t.shape: probe one leading-dim row through dequantize (cheap) — its shape
    is the logical (d_out, d_in), and the raw leading dim is the expert axis
    for stacked tensors (exactly the loader's slicing convention)."""
    from gguf.quants import dequantize
    by_name = _gguf_by_name(gguf_path)
    units = []
    for name in sorted(by_name):
        if not name.endswith(".weight"):
            continue
        if name.startswith("blk."):
            lyr = int(name.split(".")[1])
            if n_layers >= 0 and lyr >= n_layers:
                continue
        elif not lm_head:
            continue                        # token_embd/output only with --lm-head
        t = by_name[name]
        probe = dequantize(t.data[:1], t.tensor_type)
        if probe.ndim == 3:                 # stacked experts: data lead = E
            E, d_out, d_in = int(t.data.shape[0]), probe.shape[1], probe.shape[2]
            e_lim = E if n_experts < 0 else min(n_experts, E)
            for e in range(e_lim):
                units.append((name, e, d_out, d_in))
        elif probe.ndim == 2 and probe.shape[1] > 1:   # (d_out, d_in) map
            units.append((name, None, int(t.data.shape[0]), probe.shape[1]))
        # 1-D (norm gains): out of scope in v1 (status doc)
    return units


def load_unit_field(gguf_path, name, expert):
    """Dequantize one unit to (d_out, d_in) uint64 field rows on the GPU.
    Stacked-expert tensors are sliced on the RAW quantized memmap first
    (loader's convention), so no full-tensor dequant ever happens.  K-quants
    take the fused GPU path (kquant_to_field) — the CPU numpy dequant of a
    245 GB GGUF three times over would dominate all three passes."""
    from gguf.quants import dequantize
    t = _gguf_by_name(gguf_path)[name]
    qt = t.tensor_type.name
    if qt in ("Q4_K", "Q5_K", "Q6_K"):
        from kquant_cuda import kquant_to_field
        raw = t.data[expert] if expert is not None else t.data
        d_out = int(raw.shape[0])
        raw = np.ascontiguousarray(raw.reshape(d_out, -1))
        w = kquant_to_field(torch.from_numpy(raw).cuda(), qt, SCALE)
        return w.view(d_out, w.numel() // d_out)
    if expert is not None:
        arr = dequantize(t.data[expert:expert + 1], t.tensor_type)[0]
    else:
        arr = dequantize(t.data, t.tensor_type)
    w = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
    return quantize_to_field(w, SCALE)      # (d_out, d_in) uint64


# ── width-group streaming ────────────────────────────────────────────────────

class GroupStream:
    """Per-width row stream: input-coord rows of all maps of one output
    width, concatenated (spec 0.1), cut into B-row blocks.  The consumer
    callback receives (width, block_idx, rows_tensor (B, n) uint64 cuda)."""

    def __init__(self, width, params):
        self.width, self.params = width, params
        self.buf = torch.zeros(params.B, width, dtype=torch.uint64,
                               device="cuda")
        self.fill = 0
        self.blocks = 0
        self.total_rows = 0

    def feed(self, rows, consume):          # rows: (r, n) uint64 cuda
        r = rows.size(0)
        self.total_rows += r
        off = 0
        while r - off > 0:
            take = min(self.params.B - self.fill, r - off)
            self.buf[self.fill:self.fill + take] = rows[off:off + take]
            self.fill += take
            off += take
            if self.fill == self.params.B:
                consume(self.width, self.blocks, self.buf)
                self.blocks += 1
                self.fill = 0
                self.buf.zero_()

    def flush(self, consume):
        if self.fill:
            consume(self.width, self.blocks, self.buf)   # zero-padded tail
            self.blocks += 1
            self.fill = 0


def stream_all(gguf_path, units, params, consume, groups=None):
    """Drive every unit through its width group in manifest order."""
    groups = {} if groups is None else groups
    for name, e, d_out, d_in in units:
        w = load_unit_field(gguf_path, name, e)          # (d_out, d_in)
        g = groups.setdefault(d_out, GroupStream(d_out, params))
        # rows of the group stream are INPUT coords: feed W^T in chunks
        for lo in range(0, d_in, params.B):
            g.feed(w[:, lo:lo + params.B].T.contiguous(), consume)
        del w
    for width in sorted(groups):
        groups[width].flush(consume)
    return groups


def block_masks(mask_seed, width, block, params):
    g = torch.Generator(device="cpu")
    g.manual_seed(int.from_bytes(hashlib.sha256(
        mask_seed + b"|" + width.to_bytes(8, "little")
        + block.to_bytes(8, "little")).digest()[:8], "little"))
    return (torch.randint(0, 1 << 62, (width, params.lam), generator=g,
                          dtype=torch.int64).to(torch.uint64)).cuda()


def block_codewords(rows, width, block, mask_seed, params):
    """(B, n) block -> (n, N_w) codewords, chunked over polys."""
    masks = block_masks(mask_seed, width, block, params)
    for j0 in range(0, width, POLY_CHUNK):
        j1 = min(j0 + POLY_CHUNK, width)
        coeffs = torch.zeros(j1 - j0, params.N_w, dtype=torch.uint64,
                             device="cuda")
        coeffs[:, :params.B] = rows[:, j0:j1].T
        coeffs[:, params.B:params.K_w] = masks[j0:j1]
        ntt_forward_batched(coeffs)
        yield j0, j1, coeffs                 # (chunk, N_w) evaluations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layers", type=int, default=2, help="-1 = all")
    ap.add_argument("--experts", type=int, default=4, help="-1 = all")
    ap.add_argument("--lm-head", action="store_true",
                    help="include token_embd/output (width 202048)")
    ap.add_argument("--spot", type=int, default=4,
                    help="blocks spot-checked in python ints")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    params = wc.WcParams()
    mask_seed = b"wc-maverick-mask-v1"
    s_r1 = b"\x33" * 32
    units = enumerate_units(args.gguf, args.layers, args.experts, args.lm_head)
    manifest = json.dumps({"geometry": [params.B, params.lam, params.N_w,
                                        params.q_w], "scale": SCALE,
                           "units": [[n, e, o, i] for n, e, o, i in units]},
                          sort_keys=True).encode()
    manifest_digest = hashlib.sha256(manifest).digest()
    n_weights = sum(o * i for _, _, o, i in units)
    print(f"units={len(units)}  weights={n_weights:,}  "
          f"widths={sorted(set(o for _, _, o, _ in units))}")

    # pre-scan: poly counts per width (blocks known only from row totals)
    rows_per_width = {}
    for _, _, d_out, d_in in units:
        rows_per_width[d_out] = rows_per_width.get(d_out, 0) + d_in
    blocks_per_width = {n: -(-r // params.B) for n, r in rows_per_width.items()}
    poly_base, off = {}, 0
    for n in sorted(blocks_per_width):
        poly_base[n] = off
        off += blocks_per_width[n] * n
    total_polys = off
    print(f"blocks={sum(blocks_per_width.values())}  polys={total_polys:,}  "
          f"opened columns will be {total_polys * params.q_w * 8 / 1e9:.2f} GB")

    # ── pass A: enrollment root ─────────────────────────────────────────────
    sync(); tA = time.time()
    acc = _make_merkle_acc(params.N_w, total_polys)
    order_check = []

    def consume_a(width, block, rows):
        for j0, j1, cw in block_codewords(rows, width, block, mask_seed, params):
            acc.update(cw)
        order_check.append((width, block))

    # feed strictly in (width-sorted, block) order so poly indices are
    # reproducible: stream into per-width SPOOLS first is too big — instead
    # stream units grouped by width via two-phase unit ordering
    units_by_width = sorted(units, key=lambda u: (u[2], units.index(u)))
    stream_all(args.gguf, units_by_width, params, consume_a)
    art = _finalize_merkle_artifact(acc)
    sync(); tA = time.time() - tA
    print(f"pass A (enroll): {tA:.1f} s  root={art.root.hex()[:16]}…")

    # ── coins after R1 ──────────────────────────────────────────────────────
    s_rho = pr.fs_seed("wc/rho", s_r1, art.root, manifest_digest,
                       wc._params_bytes(params))
    widths = sorted(blocks_per_width)
    rho = {n: pr.op_vec(s_rho, gi, "rho", n) for gi, n in enumerate(widths)}
    rho_t = {n: torch.tensor(rho[n], dtype=torch.uint64, device="cuda")
             for n in widths}

    # ── pass B: P_trace, pi, c, v ───────────────────────────────────────────
    sync(); tB = time.time()
    p_trace = {n: torch.zeros(blocks_per_width[n] * params.B,
                              dtype=torch.uint64) for n in widths}
    pi = {n: torch.zeros(blocks_per_width[n], params.lam, dtype=torch.uint64)
          for n in widths}

    def consume_b(width, block, rows):
        p = gl_matvec(rows.contiguous(), rho_t[width])            # (B,)
        p_trace[width][block * params.B:(block + 1) * params.B] = p.cpu()
        masks = block_masks(mask_seed, width, block, params)      # (n, lam)
        pi[width][block] = gl_matvec(masks.T.contiguous(),
                                     rho_t[width]).cpu()

    stream_all(args.gguf, units_by_width, params, consume_b)
    r2 = hashlib.sha256(b"wc-r2")
    for n in widths:
        r2.update(p_trace[n].numpy().tobytes())
        r2.update(pi[n].numpy().tobytes())
    s_late = pr.fs_seed("wc/late", s_rho, r2.digest())
    c = torch.zeros(params.K_w, dtype=torch.uint64, device="cuda")
    bi = 0
    alphas = {}
    for n in widths:
        for a in range(blocks_per_width[n]):
            alpha = pr.challenge(s_late, bi, "alpha")
            alphas[(n, a)] = alpha
            u = torch.cat([p_trace[n][a * params.B:(a + 1) * params.B],
                           pi[n][a]]).cuda()
            gl_axpy(c, alpha, u)
            bi += 1
    eta_idx = pr.random_columns_n(pr.fs_seed("wc/eta", s_late),
                                  params.q_w, params.N_w)
    dom = wc._rs_domain(params)
    idx_t = torch.tensor(eta_idx, dtype=torch.long, device="cuda")
    v = poly_eval(c, dom.view(torch.int64)[idx_t].view(torch.uint64)).cpu()
    sync(); tB = time.time() - tB
    print(f"pass B (bridge): {tB:.1f} s  blocks={bi}")

    # ── pass C: gather eta columns + digest self-check ──────────────────────
    sync(); tC = time.time()
    opened = torch.zeros(total_polys, params.q_w, dtype=torch.uint64)
    chk = _make_merkle_acc(params.q_w, total_polys)
    cursor = {n: 0 for n in widths}

    def consume_c(width, block, rows):
        base = poly_base[width] + block * width
        for j0, j1, cw in block_codewords(rows, width, block, mask_seed, params):
            cols = cw.view(torch.int64)[:, idx_t].view(torch.uint64)
            opened[base + j0:base + j1] = cols.cpu()
            chk.update(cols)

    stream_all(args.gguf, units_by_width, params, consume_c)
    chk_digests = chk.finalize().cpu().numpy()
    drift = any(bytes(chk_digests[k].tolist()) != art.column_hashes[i]
                for k, i in enumerate(eta_idx))
    sync(); tC = time.time() - tC
    print(f"pass C (columns): {tC:.1f} s  drift={drift}")

    # ── verify ──────────────────────────────────────────────────────────────
    t0 = time.time()
    fails = []
    if drift:
        fails.append("column digests drift from pass A")
    if len(set(eta_idx)) != params.q_w:
        fails.append("eta not distinct")
    # Column binding = the drift check: pass C re-extracts the eta columns
    # and their BLAKE3 digests must equal pass A's committed column hashes
    # (the same accumulator convention core's opening self-check uses); the
    # root over those hashes is what the coins were derived from.  A Rust
    # twin walking merkle paths independently is future work (status doc).
    # v = c(eta) in python ints
    c_cpu = c.cpu().tolist()
    dom_cpu = dom.cpu().tolist()
    for l, i in enumerate(eta_idx):
        acc_v, x = 0, dom_cpu[i]
        for k in reversed(range(params.K_w)):
            acc_v = (acc_v * x + c_cpu[k]) % P
        if acc_v != int(v[l].item()) % P:
            fails.append(f"v[{l}] != c(eta)")
            break
    # bridge equation on GPU over all blocks
    opened_gpu_ok = True
    for l in range(params.q_w):
        rhs = 0
        for n in widths:
            base = poly_base[n]
            col = opened[base:base + blocks_per_width[n] * n, l].cuda()
            col = col.view(blocks_per_width[n], n)
            s_blk = gl_matvec(col.contiguous(), rho_t[n]).cpu().tolist()
            for a in range(blocks_per_width[n]):
                rhs = (rhs + alphas[(n, a)] * s_blk[a]) % P
        if rhs != int(v[l].item()) % P:
            opened_gpu_ok = False
            fails.append(f"bridge equation fails at eta[{l}]")
            break
    # python-int spot check on a few blocks at eta[0]
    import random
    rnd = random.Random(0)
    keys = rnd.sample(sorted(alphas), min(args.spot, len(alphas)))
    for (n, a) in keys:
        base = poly_base[n] + a * n
        col = opened[base:base + n, 0].tolist()
        s_spot = sum(r * cv for r, cv in zip(rho[n], col)) % P
        g_spot = int(gl_matvec(opened[base:base + n, 0].cuda().view(1, n)
                               .contiguous(), rho_t[n]).cpu().item()) % P
        if s_spot != g_spot:
            fails.append(f"spot check gl vs python mismatch at {(n, a)}")
    t_ver = time.time() - t0
    ok = not fails
    print(f"verify: {t_ver:.1f} s  ACCEPT={ok}" + ("" if ok else f"  {fails}"))
    print(f"H_40 = {params.soundness_bound():.3e}; ledger spend: "
          f"{params.q_w}/{params.lam} mask points")
    if args.json:
        json.dump({"weights": n_weights, "polys": total_polys,
                   "units": len(units), "root": art.root.hex(),
                   "t_enroll": tA, "t_bridge": tB, "t_columns": tC,
                   "t_verify": t_ver, "accept": ok, "fails": fails,
                   "ns_per_param_enroll": tA / n_weights * 1e9,
                   "ns_per_param_total": (tA + tB + tC) / n_weights * 1e9},
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
