"""Model architecture read from an HF checkpoint's own config.json.

One frozen dataclass is the single source of truth for every dimension the
prover and demos need (hidden size, head counts, vocab, RoPE parameters,
tied-embedding flag). Nothing here touches weights. Building it from the
checkpoint's config.json means demo/loader code never hand-types dims, so a
new dense-Llama checkpoint (e.g. Llama-3.2-1B-Instruct) is just a new path.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


def find_model_dir(model_id_or_path: str) -> str:
    """Local checkpoint directory for a filesystem path or HF model id
    (the id resolves through the HF cache; only config.json is fetched)."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    from transformers.utils import cached_file
    return os.path.dirname(cached_file(model_id_or_path, "config.json"))


@dataclass(frozen=True)
class RopeScaling:
    """Llama-3-style RoPE frequency scaling (config.json `rope_scaling`).

    Only rope_type="llama3" (the piecewise wavelength ramp) is representable;
    other types are rejected at parse time — silently ignoring an unknown
    scaling would commit a different model than the released checkpoint."""
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True)
class ModelConfig:
    """Dense-Llama architecture parameters, as read from config.json."""
    d: int                      # hidden_size
    d_ff: int                   # intermediate_size (FFN)
    n_layers: int               # num_hidden_layers
    n_heads: int                # num_attention_heads (query heads)
    n_kv_heads: int             # num_key_value_heads; < n_heads ⇒ GQA
    d_h: int                    # head dim (config head_dim, else d/n_heads)
    vocab: int                  # vocab_size
    eps_real: float             # rms_norm_eps
    rope_theta: float
    rope_scaling: Optional[RopeScaling]
    tied_embeddings: bool       # tie_word_embeddings: LM head = embedding^T

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads={self.n_heads} not a multiple of n_kv_heads={self.n_kv_heads}")
        assert self.d_h % 2 == 0, f"d_h={self.d_h} must be even (RoPE pairs)"

    @property
    def kv_groups(self) -> int:
        """Column-replication factor for the GQA public weight transform
        (1 ⇒ plain MHA, replication is the identity)."""
        return self.n_heads // self.n_kv_heads

    @property
    def q_width(self) -> int:
        """Second dim of the committed Q/K/V projections (K/V post-
        replication): n_heads·d_h. Equals d for every Llama so far."""
        return self.n_heads * self.d_h

    @classmethod
    def from_hf(cls, model_id_or_path: str) -> "ModelConfig":
        with open(os.path.join(find_model_dir(model_id_or_path),
                               "config.json")) as f:
            cfg = json.load(f)
        n_heads = cfg["num_attention_heads"]
        d = cfg["hidden_size"]
        rs = cfg.get("rope_scaling")
        scaling = None
        if rs is not None:
            rtype = rs.get("rope_type", rs.get("type"))
            if rtype != "llama3":
                raise ValueError(
                    f"unsupported rope_scaling type {rtype!r} in "
                    f"{model_id_or_path} — only 'llama3' is implemented")
            scaling = RopeScaling(
                factor=float(rs["factor"]),
                low_freq_factor=float(rs["low_freq_factor"]),
                high_freq_factor=float(rs["high_freq_factor"]),
                original_max_position_embeddings=int(
                    rs["original_max_position_embeddings"]))
        return cls(
            d=d,
            d_ff=cfg["intermediate_size"],
            n_layers=cfg["num_hidden_layers"],
            n_heads=n_heads,
            n_kv_heads=cfg.get("num_key_value_heads", n_heads),
            d_h=cfg.get("head_dim") or d // n_heads,
            vocab=cfg["vocab_size"],
            eps_real=float(cfg.get("rms_norm_eps", 1e-5)),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rope_scaling=scaling,
            tied_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )
