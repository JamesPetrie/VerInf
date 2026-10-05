"""Kimi K2's GGUF weights as the prover commits them (the two-layer gate and
beyond; analysis/k2-session-sizing.md §5).

Two faces of one mapping:

* integer weights on the CPU (`*_ints`, `GgufWeights(ints=True)`): gguf-py's
  dequantize, then np.rint(f64(v)·mult·S), as signed int64 in the orientation
  tape.matmul takes (in, out). The integer reference
  (prover/k2_int_reference.py) uses these. With mult = 1 this is exactly what
  the prover's loaders commit: the K-quant kernel is bit-exact with
  dequantize then round-half-to-even of v·S (kquant_cuda.py), and
  loader.quantize_to_field rounds f64(v)·S the same way (torch.round).
* lazy field loaders for the driver (`attention_loader`, `ints_loader`),
  with provenance: the source tensor and the share of its packed bytes the
  variable holds.

`GgufWeights(ints=False)` gives the float64 weights of the same tensors,
dequantized and nothing else (σ not folded, nothing rounded), for the float
reference.

Attention goes through k2_attention.mla_weights_from_gguf (#29): splits,
per-head reassembly, the rotary de-interleave; σ, the softmax scale, is
folded into W_q_nope and W_q_pe here, at quantization (mult = σ). Large dense
tensors (embedding, head, dense FFN, shared expert) and the expert slices go
through the fused kernel in the driver; their CPU twins here give the same
integers. Tensor names are llama.cpp's for DEEPSEEK2 (llama-arch.cpp).

Only the shards on disk are indexed (loader._gguf_by_name), so a pod holding
shard 1 of UD-Q4_K_XL builds the two-layer gate."""
import numpy as np

from k2_attention import K2_MLA, SIGMA_FOLDED, mla_weights_from_gguf, softmax_scale

S_ACT = 1 << 12                       # activation and weight scale, as Maverick's
S_SEL = 1 << 16                       # router scores and selection bias

# The integer model's public constants, one source for the driver and the
# integer reference. Z_max is Maverick's (~9.77 real units: T_A is 0 beyond
# ~8.3 at s_y = 2^12); eps is K2's rms_norm_eps; C is routed_scaling_factor
# at the gate scale S_w; the sigmoid table spans ±64 real logit units.
K2_INT = dict(S=S_ACT, S_sel=S_SEL, output_width=26, eps=1e-6,
              eps_int=round(1e-6 * S_ACT * S_ACT), z_max=40000, sig_bits=19,
              S_w=1 << 12, C=round(2.827 * (1 << 12)), select_word_bits=13,
              bracket_word_bits=8)

ATTN_TENSORS = ("attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv_a_mqa",
                "attn_kv_a_norm", "attn_k_b", "attn_v_b", "attn_kv_b", "attn_output")


def _attn_sources(cfg):
    """committed attention variable -> (GGUF tensor, share of its packed bytes)"""
    dn, dr, kr = cfg["d_nope"], cfg["d_rope"], cfg["kv_rank"]
    return {"g_attn": ("attn_norm", 1.0), "g_q_a": ("attn_q_a_norm", 1.0),
            "g_kv_a": ("attn_kv_a_norm", 1.0), "g_ffn": ("ffn_norm", 1.0),
            "W_qa": ("attn_q_a", 1.0),
            "W_q_nope": ("attn_q_b", dn / (dn + dr)), "W_q_pe": ("attn_q_b", dr / (dn + dr)),
            "W_ckv": ("attn_kv_a_mqa", kr / (kr + dr)), "W_k_pe": ("attn_kv_a_mqa", dr / (kr + dr)),
            "W_k_nope": ("attn_k_b", 1.0), "W_v": ("attn_v_b", 1.0),
            "W_o": ("attn_output", 1.0)}


def dequant(t) -> np.ndarray:
    """A gguf-py ReaderTensor as float32, in gguf-py's (reversed-dimension) shape."""
    from gguf.quants import dequantize
    return dequantize(np.ascontiguousarray(t.data), t.tensor_type)


def to_ints(w: np.ndarray, S: int, mult: float = 1.0) -> np.ndarray:
    """round-half-to-even of f64(w)·mult·S, as signed int64. With mult = 1
    the product f64(w)·1.0 is exact, so this is the prover's rint(v·S)."""
    return np.rint((np.asarray(w, dtype=np.float64) * mult) * S).astype(np.int64)


def by_name(gguf_path):
    from loader import _gguf_by_name
    return _gguf_by_name(gguf_path)


# ---- the CPU integer weights ---------------------------------------------------

def _attn_tensors(by, layer):
    return {k: dequant(by[f"blk.{layer}.{k}.weight"]) for k in ATTN_TENSORS
            if f"blk.{layer}.{k}.weight" in by}


def attention_ints(by, layer: int, S: int = S_ACT, cfg=K2_MLA) -> dict:
    """One layer's committed attention variables (and the post-attention gain
    g_ffn) as signed int64, σ folded into the two query weights."""
    w = mla_weights_from_gguf(_attn_tensors(by, layer), cfg)
    w["g_ffn"] = dequant(by[f"blk.{layer}.ffn_norm.weight"])
    sigma = softmax_scale(cfg)
    return {k: to_ints(v, S, sigma if k in SIGMA_FOLDED else 1.0) for k, v in w.items()}


def dense_ints(by, name: str, S: int = S_ACT, transpose: bool = True) -> np.ndarray:
    """A 2-D tensor (out, in) as committed: transposed to (in, out) for the
    projections, kept (V, d) for the embedding."""
    w = dequant(by[name])
    return to_ints(w.T if transpose else w, S)


def router_ints(by, layer: int, S: int = S_ACT) -> np.ndarray:
    """ffn_gate_inp, F32 (E, d) in gguf-py's shape, committed (d, E)."""
    return to_ints(dequant(by[f"blk.{layer}.ffn_gate_inp.weight"]).T, S)


def bias_ints(by, layer: int, S_sel: int = S_SEL) -> np.ndarray:
    """exp_probs_b, the selection bias (E,), at the selection scale."""
    return to_ints(dequant(by[f"blk.{layer}.exp_probs_b.bias"]), S_sel)


def expert_float(by, layer: int, kind: str, e: int) -> np.ndarray:
    """Expert e of ffn_{kind}_exps as float32 (in, out): the raw rows of the
    stacked tensor that loader.maverick_lazy_expert decodes, transposed."""
    from gguf.quants import dequantize
    t = by[f"blk.{layer}.ffn_{kind}_exps.weight"]
    return dequantize(np.ascontiguousarray(t.data[e:e + 1]), t.tensor_type)[0].T


def expert_ints(by, layer: int, kind: str, e: int, S: int = S_ACT) -> np.ndarray:
    return to_ints(expert_float(by, layer, kind, e), S)


def sigmoid_table(S_in: int, S_out: int, bits: int = 19):
    """Paired table (k, max(1, round(sigmoid((k − 2^(bits-1)) / S_in) · S_out)))
    for k in [0, 2^bits): router logits at S_in within ±2^(bits-1)/S_in real
    units, scores at S_out. The floor at 1 keeps every slot score positive,
    so the gate bracket's Z >= 1 (analysis/topk-routing-design.md §3.2)."""
    shift = 1 << (bits - 1)
    k = np.arange(1 << bits, dtype=np.float64)
    y = np.rint(1.0 / (1.0 + np.exp(-(k - shift) / S_in)) * S_out)
    y = np.maximum(y, 1).astype(np.int64)
    return list(range(1 << bits)), [int(v) for v in y], shift


class GgufWeights:
    """Both references' weights from one GGUF, a layer at a time and the
    experts on demand: integers exactly as the driver commits them
    (`ints=True`), or float64 dequantized weights with nothing folded or
    rounded (`ints=False`). Shapes are the committed (in, out)."""

    def __init__(self, gguf_path, *, ints: bool = True, S: int = S_ACT,
                 S_sel: int = S_SEL, cfg=K2_MLA):
        self.path, self.ints, self.S, self.S_sel, self.cfg = gguf_path, ints, S, S_sel, cfg
        self.by = by_name(gguf_path)
        self.d = cfg["d"]
        head = "output.weight" if "output.weight" in self.by else "token_embd.weight"
        self.head_name = head
        self.V = int(self.by[head].n_elements) // self.d

    def _q(self, w, S=None):
        return to_ints(w, S or self.S) if self.ints else np.asarray(w, dtype=np.float64)

    def _mat(self, name):
        return self._q(dequant(self.by[name]).T)

    def kind(self, il: int) -> str:
        return "moe" if f"blk.{il}.ffn_gate_inp.weight" in self.by else "dense"

    def n_experts(self, il: int) -> int:
        return int(self.by[f"blk.{il}.ffn_gate_inp.weight"].n_elements) // self.d

    def attn(self, il: int) -> dict:
        if self.ints:
            return attention_ints(self.by, il, self.S, self.cfg)
        w = mla_weights_from_gguf(_attn_tensors(self.by, il), self.cfg)
        w["g_ffn"] = dequant(self.by[f"blk.{il}.ffn_norm.weight"])
        return {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}

    def dense(self, il: int) -> dict:
        return {k: self._mat(f"blk.{il}.ffn_{n}.weight")
                for k, n in (("W_gate", "gate"), ("W_up", "up"), ("W_down", "down"))}

    def shared(self, il: int) -> dict:
        return {k: self._mat(f"blk.{il}.ffn_{n}_shexp.weight")
                for k, n in (("W_gate", "gate"), ("W_up", "up"), ("W_down", "down"))}

    def router(self, il: int) -> np.ndarray:
        return self._mat(f"blk.{il}.ffn_gate_inp.weight")

    def bias(self, il: int) -> np.ndarray:
        return self._q(dequant(self.by[f"blk.{il}.exp_probs_b.bias"]), self.S_sel)

    def expert(self, il: int, kind: str, e: int) -> np.ndarray:
        return self._q(expert_float(self.by, il, kind, e))

    def embed_rows(self, ids) -> np.ndarray:
        """The embedding rows of `ids` (T, d): the rows the one-hot select picks."""
        from gguf.quants import dequantize
        t = self.by["token_embd.weight"]
        rows = dequantize(np.ascontiguousarray(t.data[np.asarray(ids)]), t.tensor_type)
        return self._q(rows.reshape(len(ids), self.d))

    def g_out(self) -> np.ndarray:
        return self._q(dequant(self.by["output_norm.weight"]).reshape(-1))

    def head(self, v0: int, v1: int) -> np.ndarray:
        """Columns v0..v1 of the LM head, committed (d, V)."""
        from gguf.quants import dequantize
        t = self.by[self.head_name]
        rows = dequantize(np.ascontiguousarray(t.data[v0:v1]), t.tensor_type)
        return self._q(rows.reshape(v1 - v0, self.d).T)


# ---- lazy field loaders for the driver ------------------------------------------

def _field(a: np.ndarray):
    import torch
    from loader import _signed_to_field
    return _signed_to_field(torch.from_numpy(np.ascontiguousarray(a)).cuda()).reshape(-1)


_ATTN_MEMO = {}


def attention_loader(gguf_path: str, layer: int, key: str, S: int = S_ACT, cfg=K2_MLA):
    """Zero-arg closure for tape.commit_lazy: one committed attention variable,
    decoded with the layer's group (memoized while it is the last asked for,
    as Maverick's attention groups are)."""
    def load():
        memo_key = (gguf_path, layer, S)
        if memo_key not in _ATTN_MEMO:
            _ATTN_MEMO.clear()
            _ATTN_MEMO[memo_key] = attention_ints(by_name(gguf_path), layer, S, cfg)
        return _field(_ATTN_MEMO[memo_key][key])
    from loader import gguf_provenance
    src, share = _attn_sources(cfg)[key]
    prov = gguf_provenance(gguf_path, f"blk.{layer}.{src}.weight")
    prov["packed_bytes"] = int(round(prov["packed_bytes"] * share))
    prov["transform"] = key
    load.provenance = prov
    return load


def ints_loader(fn, provenance=None):
    """A closure committing fn()'s int64 array as field elements."""
    def load():
        return _field(fn())
    if provenance is not None:
        load.provenance = provenance
    return load
