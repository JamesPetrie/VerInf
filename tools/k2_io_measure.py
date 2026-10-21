"""What a full-depth pass spends reading and decoding K2's weights
(analysis/k2-fidelity-study-design.md §7.1, review of 2026-10-21): measured
per tensor on representative MoE layers, not as one sequential throughput.

For each sampled layer, its three routed-expert tensors:
- **read cold**: the shard's pages dropped from the page cache
  (posix_fadvise DONTNEED), then each expert's bytes copied out of the map;
- **read warm**: the same, again, from the page cache;
- **float64 decode** (p64, k2_float_torch.DeviceWeights): each expert's
  blocks decoded on the device, transposed and widened;
- **field decode** (the run-only pass, loader.maverick_lazy_expert): each
  expert's K-quant blocks to field integers at the profile's weight scale.

Beside them, one sequential read of the shard (the disk's own throughput) and
the extrapolation to a pass over 60 MoE layers, by each layer's tensor types
as the GGUF lists them. Every pass of a full-depth run, the reference's and
each configuration's, reads and decodes each expert once per passage unless
passages share the decode, which is what the numbers are for.

    python tools/k2_io_measure.py --gguf SHARD1.gguf --layers 1 [--device cuda] [--S 65536] \\
        [--types-from STANDIN-00001-of-00013.gguf] --record OUT.json"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "prover"))

KINDS = ("gate", "up", "down")


def _drop(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def sequential(path, nbytes=4 << 30, chunk=64 << 20):
    """One sequential read of a shard's first `nbytes`, its pages dropped first."""
    _drop([path])
    t0, n = time.time(), 0
    with open(path, "rb", buffering=0) as f:
        while n < nbytes:
            b = f.read(min(chunk, nbytes - n))
            if not b:
                break
            n += len(b)
    return dict(bytes=n, seconds=time.time() - t0, gb_per_s=n / (time.time() - t0) / 1e9)


def layer(gguf, il, *, device, S, experts=None):
    import torch
    import gguf_dequant_torch as gd
    import gguf_truncate as gt
    from loader import _gguf_by_name
    by = _gguf_by_name(str(gguf))
    rows = {}
    for kind in KINDS:
        t = by[f"blk.{il}.ffn_{kind}_exps.weight"]
        E = int(t.data.shape[0]) if experts is None else experts
        qt = t.tensor_type.name
        nbytes = int(t.data[:E].nbytes)
        r = dict(type=qt, experts=E, bytes=nbytes)
        _drop(gt.shard_paths(str(gguf)))
        t0 = time.time()
        raw = [np.ascontiguousarray(t.data[e]) for e in range(E)]
        r["read_cold_s"] = time.time() - t0
        t0 = time.time()
        raw = [np.ascontiguousarray(t.data[e]) for e in range(E)]
        r["read_warm_s"] = time.time() - t0
        _sync(device)
        t0 = time.time()
        for e in range(E):
            w = gd.dequantize(raw[e][None], t.tensor_type, device=device)[0].T.to(torch.float64)
        _sync(device)
        r["decode_float64_s"] = time.time() - t0
        del w
        if qt in ("Q4_K", "Q5_K", "Q6_K") and str(device).startswith("cuda"):
            from kquant_cuda import kquant_to_field
            _sync(device)
            t0 = time.time()
            for e in range(E):
                f = kquant_to_field(torch.from_numpy(raw[e]).to(device), qt, S)
            _sync(device)
            r["decode_field_s"] = time.time() - t0
            del f
        for k in ("read_cold", "read_warm"):
            r[f"{k}_gb_per_s"] = nbytes / r[f"{k}_s"] / 1e9
        rows[kind] = r
    return rows


def extrapolate(gguf, sampled):
    """Seconds per full pass (60 MoE layers), each layer priced by its tensors'
    types from the sampled layer's per-byte rates for the same type."""
    from loader import _gguf_by_name
    by = _gguf_by_name(str(gguf))
    rate = {}
    for rows in sampled.values():
        for r in rows.values():
            for k in ("read_cold_s", "read_warm_s", "decode_float64_s", "decode_field_s"):
                if k in r:
                    rate.setdefault((r["type"], k), []).append(r[k] / r["bytes"])
    total, missing = {}, set()
    for il in range(1, 61):
        for kind in KINDS:
            t = by.get(f"blk.{il}.ffn_{kind}_exps.weight")
            if t is None:
                missing.add(il)
                continue
            for (qt, k), rs in rate.items():
                if qt == t.tensor_type.name:
                    total[k] = total.get(k, 0.0) + float(np.mean(rs)) * int(t.data.nbytes)
    return dict(seconds_per_pass=total, layers_missing=sorted(missing))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layers", default="1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--S", type=int, default=1 << 16, help="the field decode's weight scale (a16w16: 2^16)")
    ap.add_argument("--experts", type=int, default=None, help="fewer experts per tensor, for a quick run")
    ap.add_argument("--types-from", help="a GGUF listing every layer's tensors (the full model, or its stand-in "
                    "of headers): the pass is priced over its layers; default --gguf")
    ap.add_argument("--record", required=True)
    a = ap.parse_args(argv)
    import gguf_truncate as gt
    rec = dict(what=__doc__.split("\n\n")[0], gguf=pathlib.Path(a.gguf).name, device=a.device, S=a.S,
               sequential=sequential(gt.shard_paths(a.gguf)[0]), layers={})
    print(f"sequential read: {rec['sequential']['gb_per_s']:.2f} GB/s", flush=True)
    for il in (int(x) for x in a.layers.split(",")):
        rec["layers"][il] = layer(a.gguf, il, device=a.device, S=a.S, experts=a.experts)
        for kind, r in rec["layers"][il].items():
            print(f"L{il} {kind} {r['type']} {r['bytes'] / 1e9:.2f} GB: cold {r['read_cold_gb_per_s']:.2f} GB/s, "
                  f"warm {r['read_warm_gb_per_s']:.2f} GB/s, float64 {r['decode_float64_s']:.1f} s"
                  + (f", field {r['decode_field_s']:.1f} s" if "decode_field_s" in r else ""), flush=True)
    if a.experts is None:
        rec["full_pass"] = extrapolate(a.types_from or a.gguf, rec["layers"])
        print("per full pass:", {k: round(v) for k, v in rec["full_pass"]["seconds_per_pass"].items()}, flush=True)
    pathlib.Path(a.record).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
