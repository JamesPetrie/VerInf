"""One operation of llama.cpp's graph, on identical inputs, through each
path's kernels (analysis/k2-fidelity-study-design.md §7.1, review of
2026-10-21): after prefill_localize.py names the first intermediate where the
batched and one-token paths part, its operation runs alone here
(tools/llama_runner/verinf_optest), all tokens in one graph and one token per
graph, on the GPU and on the CPU, against float64 on the same inputs. The
inputs are what the batched run itself fed the operation (its captures); a
matrix held in the KV cache is rounded to f16 as the cache holds it, so the
comparison isolates the kernels' arithmetic.

Operations (OPS): the attention scores ("kq-<layer>", the cached keys against
the queries, accumulated in f32 as llama.cpp asks), the attention's value
product ("kqv-<layer>"), a weight's matmul ("weight:<tensor>:<input capture>")
and the routed experts' ("experts:<tensor>:<input capture>:<ids capture>").

    python prover/engine_optest.py --optest BIN --gguf TWO_LAYER.gguf --captures LOCALIZE/batch-captured \\
        --op kq-1 --work DIR [--device CUDA0]"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

import llama_runner as lr


def optest(binary, out, *, op="mul_mat", device="CUDA0", split="all", a=None, a_type="f16", a_gguf=None,
           a_tensor=None, b=None, b_permute=None, token_dim=1, ids=None, prec=None):
    """Run verinf_optest; returns the result as (tokens, everything else flattened in
    ggml's order) and the run's meta. `a`, `b`: float32 arrays in numpy order (ggml's
    ne reversed); `ids`: int32 (tokens, n_used)."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--op", op, "--device", device, "--split", split, "--out", str(out)]
    if a is not None:
        a = np.ascontiguousarray(a, dtype=np.float32)
        a.tofile(out / "a.f32")
        cmd += ["--a", str(out / "a.f32"), "--a-ne", ",".join(map(str, a.shape[::-1])), "--a-type", a_type]
    else:
        cmd += ["--a-gguf", str(a_gguf), "--a-tensor", a_tensor]
    b = np.ascontiguousarray(b, dtype=np.float32)
    b.tofile(out / "b.f32")
    cmd += ["--b", str(out / "b.f32"), "--b-ne", ",".join(map(str, b.shape[::-1])), "--token-dim", str(token_dim)]
    if b_permute:
        cmd += ["--b-permute", ",".join(map(str, b_permute))]
    if ids is not None:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        ids.tofile(out / "ids.i32")
        cmd += ["--ids", str(out / "ids.i32"), "--ids-ne", f"{ids.shape[1]},{ids.shape[0]}"]
    if prec:
        cmd += ["--prec", prec]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    m = json.loads((out / "meta.json").read_text())
    ne = m["chunk_ne"]
    chunks = np.fromfile(out / "out.f32", dtype="<f4").reshape(m["chunks"], ne[3], ne[2], ne[1], ne[0])
    ax = 3 - (2 if op == "mul_mat_id" else token_dim)            # ggml's token dim as a numpy axis of a chunk
    per = [np.moveaxis(c, ax, 0) for c in chunks]
    return np.concatenate([x.reshape(x.shape[0], -1) for x in per], axis=0), m


def _rel(x, ref, mask=None):
    ref = np.asarray(ref, np.float64).reshape(len(ref), -1)
    d = np.asarray(x, np.float64).reshape(len(x), -1) - ref
    if mask is not None:
        mask = np.asarray(mask).reshape(len(mask), -1)
    if mask is not None:
        d, ref = d[mask], ref[mask]
    return dict(rel_l2=float(np.sqrt((d ** 2).sum() / max((ref ** 2).sum(), 1e-300))), max_abs=float(np.abs(d).max()))


def compare(binary, work, *, ref, inputs, devices=("CUDA0", "CPU"), mask=None):
    """Each device, all tokens in one graph and one per graph, against `ref`
    (float64, tokens first) and against each other."""
    got, meta = {}, {}
    for dev in devices:
        for split in ("all", "single"):
            key = f"{dev}-{split}"
            got[key], meta[key] = optest(binary, pathlib.Path(work) / key, device=dev, split=split, **inputs)
    rows = {k: _rel(v, ref, mask) for k, v in got.items()}
    pairs = {f"{dev}: all vs single": _rel(got[f"{dev}-all"], np.asarray(got[f"{dev}-single"], np.float64), mask)
             for dev in devices}
    return dict(against_float64=rows, between=pairs, src0_type=meta[next(iter(meta))]["src0_type"],
                seconds={k: m["compute_s"] for k, m in meta.items()})


def _calls(res, key):
    return lr.capture_calls(res, key)


def _rows(res, key):
    """A 2-D capture's rows over every position (tokens, width)."""
    return np.concatenate([c.reshape(-1, c.shape[-1]) for c in _calls(res, key)], axis=0)


def _f16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float64)


def _dequant(gguf, name):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    t = next(t for t in GGUFReader(str(gguf)).tensors if t.name == name)
    return dequantize(t.data, t.tensor_type).astype(np.float64).reshape(tuple(int(x) for x in t.shape[::-1]))


def op_kq(res, il):
    """kq = mul_mat(K cache, Q): the cached keys (f16) against every position's
    queries, permuted as llama.cpp permutes them; f32 accumulation."""
    K = np.concatenate([c.reshape(-1, c.shape[-1]) for c in _calls(res, f"Kcur-{il}/CONCAT")], axis=0)  # (T, 576)
    Q = np.concatenate([c.reshape(-1, *c.shape[-2:]) for c in _calls(res, f"Qcur-{il}/CONCAT")], axis=0)  # (T, H, 576)
    T = K.shape[0]
    ref = np.einsum("thc,nc->htn", Q.astype(np.float64), _f16(K))          # (H, T, n_kv)
    ref = np.moveaxis(ref, 1, 0)                                           # tokens first
    mask = np.broadcast_to(np.tril(np.ones((T, T), bool))[:, None, :], ref.shape)
    return dict(inputs=dict(a=K[None], a_type="f16", b=Q, b_permute=(0, 2, 1, 3), token_dim=1, prec="f32"),
                ref=ref, mask=mask)


def op_kqv(res, il):
    """kqv = mul_mat(V cache made contiguous, the probabilities), from the batched call
    (the one with every position)."""
    v = max(_calls(res, f"v_cont-{il}/CONT"), key=lambda c: c.size)[0]          # (1, 512, n_kv)
    p = max(_calls(res, f"kq_soft_max-{il}/SOFT_MAX"), key=lambda c: c.size)[0]  # (H, tokens, n_kv)
    n = min(v.shape[-1], p.shape[-1])
    v, p = v[..., :n], p[..., :n]
    ref = np.einsum("hts,ms->thm", p.astype(np.float64), _f16(v[0]))     # (tokens, H, 512)
    return dict(inputs=dict(a=v, a_type="f16", b=p, token_dim=1), ref=ref)


def op_weight(res, gguf, tensor, inp):
    """A weight's matmul: the GGUF's own tensor (its own type) against the captured input."""
    X = _rows(res, inp)
    W = _dequant(gguf, tensor)                                              # (N, K)
    return dict(inputs=dict(a_gguf=gguf, a_tensor=tensor, b=X, token_dim=1), ref=X.astype(np.float64) @ W.T)


def op_experts(res, gguf, tensor, inp, ids_key, k=8):
    """The routed experts' matmul (mul_mat_id): each position's selected experts' weights
    against its captured input."""
    X = _rows(res, inp)
    ids = _rows(res, ids_key)[:, :k].astype(np.int32)
    W = _dequant(gguf, tensor)                                              # (E, N, K)
    ref = np.einsum("tk,tjnk->tjn", X.astype(np.float64), W[ids])
    return dict(inputs=dict(op="mul_mat_id", a_gguf=gguf, a_tensor=tensor, b=X[:, None, :], ids=ids, token_dim=2),
                ref=ref)


def build(op, res, gguf):
    kind, _, rest = op.partition(":")
    if kind.startswith("kq-"):
        return op_kq(res, int(kind[3:]))
    if kind.startswith("kqv-"):
        return op_kqv(res, int(kind[4:]))
    if kind == "weight":
        tensor, inp = rest.split(":")
        return op_weight(res, gguf, tensor, inp)
    if kind == "experts":
        tensor, inp, ids = rest.split(":")
        return op_experts(res, gguf, tensor, inp, ids)
    raise ValueError(f"unknown operation {op!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--optest", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--captures", required=True, help="a captured run's directory (prefill_localize's batch)")
    ap.add_argument("--op", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--device", default="CUDA0")
    a = ap.parse_args(argv)
    res = lr.load(a.captures)
    spec = build(a.op, res, a.gguf)
    safe = a.op.replace(":", "_").replace("/", "@")
    rec = compare(a.optest, pathlib.Path(a.work) / safe, ref=spec["ref"], inputs=spec["inputs"],
                  devices=(a.device, "CPU") if a.device != "CPU" else ("CPU",), mask=spec.get("mask"))
    rec["op"] = a.op
    dst = pathlib.Path(a.work) / f"optest-{safe}.json"
    dst.write_text(json.dumps(rec, indent=1))
    for k, v in rec["against_float64"].items():
        print(f"{a.op} {k:14s} against float64: rel {v['rel_l2']:.2e} max {v['max_abs']:.2e}")
    for k, v in rec["between"].items():
        print(f"{a.op} {k}: rel {v['rel_l2']:.2e}")
    print("record:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
