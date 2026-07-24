"""The production lens — structural guard against reading a TOY measurement as a
production result (the error that cost me a whole wrong conclusion: "witness is
6%" from a toy where it is actually 57% at 400B).

The insight: a toy A/B correctly measures a lever's MECHANISM — the fractional
effect it has on a specific phase/term (e.g. "spill cuts the witness term by
21%"). What does NOT transfer is that term's SHARE of prove. So: measure the
per-term fraction on toy, then apply the AUTHORITATIVE production share from the
cost model. Never eyeball a toy percentage.

Every A/B in this bench should end by calling `report(...)` so the toy number is
ALWAYS printed next to its production projection, clearly labeled.

Production breakdown (cost_calculator --S 1093 --witness-mode notebook):
    witness_recompute ~57%, identity floor ~43% (streaming/quad/lin).
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_calculator as cc


def prod_terms(S: int = 1093) -> dict:
    """Authoritative 400B per-term seconds + shares, from the cost model in
    NOTEBOOK witness mode (the only mode valid at production; the `measured`
    mode is a toy fit). Terms: witness, streaming(encode+fold+hash), quad, lin."""
    cfg = cc.Config(S=S)
    r = cc.predict(cfg, witness_mode="notebook")
    ft = r["floor_terms"]
    terms = {
        "witness": r["witness_recompute_s"],
        "streaming": ft["streaming_s"],   # encode + reencode + fold + hashing
        "quad": ft["quadratic_s"],
        "lin": ft["lin_s"],
    }
    total = sum(terms.values())
    return {"terms": terms, "total": total,
            "shares": {k: v / total for k, v in terms.items()}}


# which measurable phase buckets roll up into which production term. Extend as
# the phase instrumentation grows; unknown buckets are flagged, not silently
# dropped, so a lever targeting an unmapped phase can't be mis-projected.
_PHASE_TO_TERM = {
    "witness": "witness",
    "encode": "streaming", "merkle": "streaming", "fold_qirs": "streaming",
    "fold_qlin": "streaming", "cols": "streaming", "compile": "streaming",
    "quad": "quad",
    "qlin_polymul": "lin", "qlin_rTA": "lin", "qlin_rowsum": "lin",
    "qlin_interp": "lin", "aux": "lin",
}


@dataclass
class LeverEffect:
    """A lever that reduces `term` by `toy_frac` (measured on toy) and optionally
    adds `added_prod_s` of new cost at production (e.g. spill's disk I/O). The
    net production speedup is computed against the authoritative prod shares."""
    term: str
    toy_frac: float          # fractional reduction of that TERM, measured on toy
    added_prod_s: float = 0.0  # production-only added cost (I/O, transfer, ...)
    note: str = ""


def project(effect: LeverEffect, S: int = 1093) -> dict:
    pt = prod_terms(S)
    assert effect.term in pt["terms"], f"unknown term {effect.term!r}"
    base_total = pt["total"]
    term_s = pt["terms"][effect.term]
    saved = term_s * effect.toy_frac - effect.added_prod_s
    new_total = base_total - saved
    return {"prod_total_s": base_total, "prod_new_s": new_total,
            "prod_speedup_pct": 100 * saved / base_total,
            "term_s": term_s, "term_share": pt["shares"][effect.term],
            "saved_s": saved}


def report(title: str, toy_pct: float, effect: LeverEffect | None = None,
           S: int = 1093, transfers: bool = True) -> None:
    """Print the toy number and, if a mechanism effect is given, its PRODUCTION
    projection. Use at the end of every A/B. Set transfers=False for levers whose
    MECHANISM does not exist at 400B (e.g. a GPU-memory cache: the witness is 7TB,
    it can't be held) — then the honest production number is 0, not the toy win."""
    pt = prod_terms(S)
    print(f"\n--- production lens ({title}) ---")
    print(f"  TOY end-to-end: {toy_pct:+.1f}%  <-- do NOT read as production")
    print(f"  400B term shares (cost model, notebook mode):")
    for k, s in pt["terms"].items():
        print(f"     {k:10s} {s:8.0f}s  {100*pt['shares'][k]:5.1f}%")
    if not transfers:
        print("  PRODUCTION PROJECTION: 0.0%  <-- mechanism does NOT transfer to "
              "400B (test-scale only); the toy win is an artifact")
        return
    if effect is None:
        print("  (no mechanism->term mapping given; production effect unprojected)")
        return
    p = project(effect, S)
    print(f"  lever: cuts '{effect.term}' by {100*effect.toy_frac:.0f}% "
          f"(measured on toy){'; +%.0fs prod I/O' % effect.added_prod_s if effect.added_prod_s else ''}")
    print(f"  PRODUCTION PROJECTION: {p['prod_speedup_pct']:+.1f}%  "
          f"({p['prod_total_s']:.0f}s -> {p['prod_new_s']:.0f}s)"
          f"{'  ' + effect.note if effect.note else ''}")


if __name__ == "__main__":
    pt = prod_terms()
    print("=== 400B production breakdown (the authoritative shares) ===")
    for k, s in pt["terms"].items():
        print(f"  {k:10s} {s:9.0f}s  {100*pt['shares'][k]:5.1f}%")
    print(f"  {'TOTAL':10s} {pt['total']:9.0f}s")
    # self-check: the witness share must be ~57%, NOT ~6%
    ws = pt["shares"]["witness"]
    print(f"\n  witness share = {100*ws:.1f}%  "
          f"({'OK ~57% as expected' if ws > 0.4 else 'BUG: toy-like share, lens is wrong'})")
