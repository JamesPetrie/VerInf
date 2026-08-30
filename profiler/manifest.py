"""Workload manifest: the contract between extraction and analysis.

A manifest is one JSON document describing everything cost-relevant about a
proving run *before it happens*: the Ligero config, the model, and one record
per claim (type, shape params, produced/consumed variables, weight bytes).
It is produced either by the tape extractor (extract.py, exact, model-agnostic
— any model that builds a Tape) or by a synthetic builder (synth.py, closed
form, for boxes without torch and as a cross-check).

Everything downstream — cost totals, time/memory prediction, DAG export, and
eventually the multi-GPU scheduler's work list — consumes this format only,
never the tape directly. Extending the schema: add fields, bump
SCHEMA_VERSION, keep old readers working (readers ignore unknown fields).
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

SCHEMA_VERSION = 1


@dataclass
class ClaimRecord:
    idx: int
    type: str                      # canonical name or prover dataclass name
    label: str = ""                # human label, e.g. "L17.moe.expert42.gate"
    layer: Optional[int] = None
    params: Dict = field(default_factory=dict)   # shape params for claimcosts
    inputs: List[str] = field(default_factory=list)   # variable names consumed
    outputs: List[str] = field(default_factory=list)  # variable names produced
    w_slots: Optional[float] = None  # exact witness slots when known (extract)


@dataclass
class VariableRecord:
    name: str
    length: int                    # field slots
    phase: int = 1
    persistent: bool = False       # model weight (streamed, own Merkle block)
    producer: Optional[int] = None  # claim idx, None = run input / weight
    consumers: List[int] = field(default_factory=list)
    # persistent weights only, optional: on-disk provenance for storage
    # models (weightsplit). `packed_bytes` is the exact packed source size
    # when extraction knows it (a logical K/V variable can be several
    # times its packed GGUF source); `quant` the GGUF block type.
    quant: Optional[str] = None
    packed_bytes: Optional[float] = None


@dataclass
class Manifest:
    schema_version: int = SCHEMA_VERSION
    source: Dict = field(default_factory=dict)   # kind: synth|tape, generator
    model: Dict = field(default_factory=dict)    # name + dims
    run: Dict = field(default_factory=dict)      # seq, ligero cfg (ELL/K_DEG/N_LIG/T_QUERIES)
    claims: List[ClaimRecord] = field(default_factory=list)
    variables: List[VariableRecord] = field(default_factory=list)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @staticmethod
    def load(path: str) -> "Manifest":
        # archived extractions ship gzipped (a Maverick S=1000 manifest is
        # ~35 MB raw, ~1.7 MB compressed); read either form
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"manifest root must be a JSON object, got {type(raw).__name__}")
        ver = raw.get("schema_version")
        if isinstance(ver, bool) or not isinstance(ver, int) \
                or not 1 <= ver <= SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {ver!r} (this reader knows "
                f"1..{SCHEMA_VERSION}; readers ignore unknown fields, but a "
                f"bumped version signals a semantic change)")
        m = Manifest(
            schema_version=ver,
            source=raw.get("source", {}),
            model=raw.get("model", {}),
            run=raw.get("run", {}),
        )
        try:
            for c in raw.get("claims", []):
                m.claims.append(ClaimRecord(
                    idx=c["idx"], type=c["type"], label=c.get("label", ""),
                    layer=c.get("layer"), params=c.get("params", {}),
                    inputs=c.get("inputs", []), outputs=c.get("outputs", []),
                    w_slots=c.get("w_slots")))
            for v in raw.get("variables", []):
                m.variables.append(VariableRecord(
                    name=v["name"], length=v["length"], phase=v.get("phase", 1),
                    persistent=v.get("persistent", False),
                    producer=v.get("producer"), consumers=v.get("consumers", []),
                    quant=v.get("quant"), packed_bytes=v.get("packed_bytes")))
        except (KeyError, TypeError, AttributeError) as e:
            # structural failures (non-dict records, missing required keys)
            # normalize to ValueError: the CLI boundary catches that
            raise ValueError(f"malformed manifest record: {e!r}") from e
        if not m.claims:
            raise ValueError("structurally empty manifest: no claims")
        return m

    def var_by_name(self) -> Dict[str, VariableRecord]:
        return {v.name: v for v in self.variables}
