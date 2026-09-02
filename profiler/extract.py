"""Tape -> manifest extractor. Runs where the prover runs (needs torch).

This is the model-agnostic ground truth: build any model's Tape in lazy mode
(claim recording defers witness compute, and commit_lazy defers weight loads;
note LogUp tables still build eagerly on the GPU, so a tape build needs CUDA
and real table memory) and walk it into a manifest. The 1T-parameter model
needs nothing here — if it builds a Tape, it extracts.

Usage on the Spark (library-first, mirroring how demos build tapes):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("profiler").resolve()))
    from extract import extract_tape
    # ... build `tape` exactly as demo_maverick_full/demo_llama7b do,
    #     with Tape(cfg, lazy=True) and commit_lazy for weights ...
    extract_tape(tape, model=dict(name="maverick"), seq=1000).save("manifest.json")

`python profiler/extract.py --selftest` builds a two-matmul toy tape on the
GPU and round-trips it, as a smoke test of the walking logic.

Cross-checks once on hardware: totals here vs LIGERO_LAYOUT_BREAKDOWN=1 (rows
and elements per claim type), and vs the synth builders for the two standard
models. Divergence means a formula or a builder is stale — trust the tape.
"""
from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # manifest, claimcosts
sys.path.insert(0, str(_HERE.parent / "prover"))     # tape/core (torch side)

from manifest import Manifest, ClaimRecord, VariableRecord   # noqa: E402

# Best-effort layer parse from variable-name labels. Both team demos tag
# weights per layer ("L17_Wg", "W_Q_L0"); derived names inherit the tag via
# the "a@b" prefix convention. Rightmost tag wins (the weight operand).
_LAYER_RES = (re.compile(r"(?:^|[_@.])L(\d+)(?:[_.#]|$)"),
              re.compile(r"blk\.(\d+)"))


# Cost-relevant flags some configs expose as PROPERTIES, invisible to
# dataclasses.fields(): RmsNormConfig/SiluConfig/SoftmaxConfig derive
# rescale_bits (and rmsnorm's output_rescale_bits) from their scale fields.
_PROP_PARAMS = ("rescale_bits", "output_rescale_bits")


def _claim_params(claim) -> dict:
    """Scalar fields become manifest params. Prover claims keep shapes in two
    places: top-level fields (m/k/n/heads, length, T/E) and a nested `config`
    dataclass (B/M/causal, B/d, SEQ/d_h/heads) — flatten both, config last.
    claimcosts accepts these spellings directly (length, SEQ*heads*d_h, ...)."""
    out = {}

    def take(obj):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name, None)
            if isinstance(v, (int, float, bool, str)):
                out[f.name] = v
        for name in _PROP_PARAMS:      # properties, not fields
            if name not in out:
                v = getattr(obj, name, None)
                if isinstance(v, int) and not isinstance(v, bool):
                    out[name] = v

    if dataclasses.is_dataclass(claim):
        take(claim)
        cfg = getattr(claim, "config", None)
        if dataclasses.is_dataclass(cfg):
            take(cfg)
    tok = getattr(claim, "token_ids", None)
    if isinstance(tok, (list, tuple)):
        out["n_tokens"] = len(tok)   # EmbeddingLookup length = n_tokens * d
    words = getattr(claim, "words", None)
    if isinstance(words, (list, tuple)):
        out["n_words"] = len(words)  # WordExtraction W = n_words * length
    # Rescale presence: prover matmul/hadamard claims carry rescale_bits.
    if "rescale_bits" in out:
        out["rescale"] = bool(out.get("rescale_bits"))
    return out


def _claim_variables(claim):
    """Every Variable the claim references, in field order (lists included)."""
    from core import Variable   # prover import, deferred to call time
    seen = []
    def visit(v):
        if isinstance(v, Variable):
            seen.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                visit(x)
    if dataclasses.is_dataclass(claim):
        for f in dataclasses.fields(claim):
            visit(getattr(claim, f.name, None))
    return seen


def _claim_tables(claim):
    """Table objects referenced from claim fields (mirrors core._collect_tables)."""
    try:
        from core import Table
    except ImportError:
        return []
    if not dataclasses.is_dataclass(claim):
        return []
    return [v for f in dataclasses.fields(claim)
            if isinstance(v := getattr(claim, f.name, None), Table)]


def _layer_of(label: str):
    hits = [m.group(1) for rx in _LAYER_RES for m in rx.finditer(label)]
    return int(hits[-1]) if hits else None


def extract_tape(tape, *, model: dict, seq: int) -> Manifest:
    """Walk a lazy Tape into a manifest. Exact W per claim comes from Variable
    lengths at first sight, deduped by object identity — the same rule
    _layout uses (duplicate *names* are legal prover-side; colliding names
    get a `~n` suffix in the manifest so records stay addressable). Shared
    LogUp tables settle through TableSettlement claims (explicit ones are
    reused, the rest synthesized, mirroring core._collect_tables): mult is a
    producer-less commitment consumed by its lookups and the settlement, w
    is the settlement's phase-2 output."""
    if len(getattr(tape, "_deferred", ())) != len(tape.claims):
        raise ValueError(
            "extract_tape needs a lazy tape (every claim recorded in "
            "_deferred). Build with Tape(cfg, lazy=True); an eager tape "
            "has no input lists and would misclassify inputs as outputs.")
    cfg = tape.cfg
    inputs_by_claim = {id(e[0]): e[1] for e in tape._deferred}

    man = Manifest(
        source=dict(kind="tape", generator="extract.extract_tape"),
        model=model,
        run=dict(seq=seq, ligero=dict(
            ELL=cfg.ELL, K_DEG=cfg.K_DEG, N_LIG=cfg.N_LIG,
            T_QUERIES=cfg.T_QUERIES)))

    seen = {}        # id(Variable) -> VariableRecord
    names_taken = set()
    tables = {}      # id(Table) -> (Table, [referencing claim idxs])

    def record(v, producer):
        name, i = v.name, 1
        while name in names_taken:         # distinct object, colliding name
            name = f"{v.name}~{i}"         # (a literal x~1 may exist: loop)
            i += 1
        names_taken.add(name)
        rec = VariableRecord(
            name=name, length=int(v.length), phase=int(v.phase),
            persistent=bool(getattr(v, "persistent", False)),
            producer=producer,
            w_new=bool(getattr(v, "w_new", False)))
        if rec.persistent:
            # Source provenance for the storage models (weightsplit): lazy
            # weight loaders may carry a `provenance` dict — the GGUF/
            # safetensors quant type and the exact PACKED source bytes
            # attributable to this variable (a K/V logical variable is
            # several times its packed source; quant alone cannot size it).
            # Optional by design: eager tensors and plain loaders record
            # nothing and the profiler falls back to its quant table.
            src = getattr(tape, "inputs", {}).get(v)
            prov = getattr(src, "provenance", None) if callable(src) else None
            if isinstance(prov, dict):
                q = prov.get("quant")
                pb = prov.get("packed_bytes")
                rec.quant = str(q) if q is not None else None
                rec.packed_bytes = float(pb) if pb is not None else None
        seen[id(v)] = rec
        return rec

    explicit = {}    # id(Table) -> idx of an explicit TableSettlement claim

    for idx, claim in enumerate(tape.claims):
        input_vars = inputs_by_claim[id(claim)]
        input_ids = {id(v) for v in input_vars}
        outputs, w_slots, extra_inputs = [], 0.0, []
        for v in _claim_variables(claim):
            rec = seen.get(id(v))
            if rec is None:
                # Persistent variables are committed run inputs by
                # construction — never claim outputs, even if a claim's
                # _deferred input list omits them (e.g. lookup tables).
                is_input = (id(v) in input_ids
                            or bool(getattr(v, "persistent", False)))
                rec = record(v, producer=None if is_input else idx)
                if is_input:
                    rec.consumers.append(idx)
                    if id(v) not in input_ids:
                        extra_inputs.append(rec.name)
                else:
                    outputs.append(rec.name)
                    w_slots += v.length
            elif rec.producer != idx and idx not in rec.consumers:
                # Includes reused-output Variables (paired_tlookup y-reuse):
                # the schema has one producer, later sites count as consumers.
                rec.consumers.append(idx)
                if id(v) not in input_ids:
                    extra_inputs.append(rec.name)
        for v in input_vars:            # inputs that aren't claim fields
            rec = seen.get(id(v)) or record(v, producer=None)
            if rec.producer != idx and idx not in rec.consumers:
                rec.consumers.append(idx)
        if type(claim).__name__ == "TableSettlement":
            tab = getattr(claim, "table", None)
            if tab is not None:         # reused (not re-synthesized) below
                explicit[id(tab)] = idx
                tables.setdefault(id(tab), (tab, []))
        else:
            for t in _claim_tables(claim):
                tables.setdefault(id(t), (t, []))[1].append(idx)
        inputs = [seen[id(v)].name for v in input_vars] + extra_inputs
        label = outputs[0] if outputs else (inputs[0] if inputs else "")
        man.claims.append(ClaimRecord(
            idx=idx, type=type(claim).__name__, label=label,
            layer=_layer_of(label), params=_claim_params(claim),
            inputs=inputs, outputs=outputs, w_slots=w_slots or None))

    # Shared tables settle after all ops in the real layout: one
    # TableSettlement claim per table — an explicit one from the tape when
    # present (wired retroactively), else synthesized at the end. Roles
    # differ per variable — mult is a producer-less phase-1 commitment
    # incremented by every lookup claim and read at settlement; each z is
    # consumed by the settlement ONLY (its producer is its own lookup
    # claim); w is the settlement's phase-2 OUTPUT (w[j] =
    # mult[j]/(alpha - v[j]), computed at settle time).
    for t, ref_idxs in tables.values():
        idx = explicit.get(id(t), len(man.claims))
        inputs, outputs, w_slots = [], [], 0.0
        mult = getattr(t, "mult_var", None)
        if mult is not None:
            rec = seen.get(id(mult)) or record(mult, producer=None)
            inputs.append(rec.name)
            for ci in (*ref_idxs, idx):
                if ci not in rec.consumers:
                    rec.consumers.append(ci)
                # consumer <=> listed input, both directions (lookups
                # increment mult, so it is an input of theirs too)
                if ci != idx and rec.name not in man.claims[ci].inputs:
                    man.claims[ci].inputs.append(rec.name)
        for z in getattr(t, "z_vars", ()):
            rec = seen.get(id(z)) or record(z, producer=None)
            inputs.append(rec.name)
            if idx not in rec.consumers:
                rec.consumers.append(idx)
        w = getattr(t, "w_var", None)
        if w is not None:
            rec = seen.get(id(w)) or record(w, producer=idx)
            if rec.producer is None:
                rec.producer = idx
            outputs.append(rec.name)
            if not rec.persistent:
                w_slots += rec.length
        label = f"settle.{getattr(t, 'name', '?')}"
        params = dict(T_LEN=int(mult.length) if mult else 0)
        if id(t) in explicit:
            crec = man.claims[idx]      # fill the walked explicit record
            crec.params.update(params)
            crec.label = crec.label or label
            crec.inputs, crec.outputs = inputs, outputs
            crec.w_slots = w_slots or None
        else:
            man.claims.append(ClaimRecord(
                idx=idx, type="TableSettlement", label=label, params=params,
                inputs=inputs, outputs=outputs, w_slots=w_slots or None))

    man.variables = list(seen.values())
    return man


def _selftest():
    import torch
    from core import LigeroConfig
    from tape import Tape
    cfg = LigeroConfig(ELL=64, K_DEG=128, N_LIG=256, T_QUERIES=4)
    tape = Tape(cfg, lazy=True)
    a = tape.commit("a", torch.zeros(8 * 16, dtype=torch.uint64, device="cuda"), (8, 16))
    w = tape.commit("w", torch.zeros(16 * 4, dtype=torch.uint64, device="cuda"), (16, 4),
                    persistent=True)
    c = tape.matmul(a, w)
    tape.matmul(c, c, transpose_b=True)
    man = extract_tape(tape, model=dict(name="selftest"), seq=8)
    assert len(man.claims) == 2, man.claims
    assert any(v.persistent for v in man.variables)
    print(f"selftest OK: {len(man.claims)} claims, {len(man.variables)} variables")
    for c in man.claims:
        print(f"  {c.type} params={c.params} in={c.inputs} out={c.outputs} W={c.w_slots}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
