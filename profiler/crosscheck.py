"""Extraction cross-check: profiler roadmap item 1, one command on a GPU box.

Builds a real demo tape (lazy — claim recording only, no witness compute, no
prove), extracts a manifest, and diffs it against the closed-form synth
builder three ways:

  1. claim counts per type (extracted may exceed synth by the UI chain and
     LogUp settlements — those are known-unmodeled and labeled, not flagged);
  2. cost totals — W/cids/Q per type and overall, persistent (weight) slots
     compared EXACTLY;
  3. with --layout: the prover's own LIGERO_LAYOUT_BREAKDOWN table, captured
     from a subprocess run of the demo, against the same aggregation
     recomputed from the extracted manifest — row for row.

Divergence = stale formula; trust the tape (synth.py docstring). A FLAG in
the output is a finding, not necessarily a bug here.

Requirements: CUDA (tape construction materializes LogUp tables eagerly).
llama7b runs weight-free — random weights, identical tape structure.
maverick needs a GGUF on disk (metadata is read eagerly; payloads stay lazy).

    python3 profiler/crosscheck.py llama7b  --seq 100 --layout
    python3 profiler/crosscheck.py maverick --from-gguf ~/maverick-gguf/UD-Q4_K_XL \
        --prompt-n 2 --cont-n 2 --layout
    python3 profiler/crosscheck.py maverick --from-gguf ... --seq 1000   # big-S extract only

Exit 0: no FLAG lines. Exit 1: at least one FLAG — eyeball before trusting
extracted manifests (README roadmap 1 is the gate).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_PROFILER = Path(__file__).resolve().parent
_REPO = _PROFILER.parent
for p in (_PROFILER, _REPO / "prover", _REPO / "demo"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import claimcosts                              # noqa: E402
import synth                                   # noqa: E402
from manifest import Manifest                  # noqa: E402

# Claim types a real tape carries that synth deliberately does not model
# (README caveats: UI/MaxClaim + hidden-prompt one-hot chain, LogUp table
# commitments and their settlements). Their presence in the extracted-only
# section is expected; their absence would be the surprise.
KNOWN_EXTRACT_ONLY = {
    "table_settle": "LogUp settlement — synth doesn't model tables",
    "MaxClaim": "UI chain — synth omits it (README caveat)",
    "ConcatClaim": "UI chain — synth omits it (README caveat)",
    "InfoFinalizeClaim": "UI chain — synth omits it (README caveat)",
}

# The routing bundle: synth emits one `routing` row; a real tape emits the
# pieces. claimcosts guarantees bundle == core + word + n_words*range, so the
# compare groups them under one key on both sides.
_ROUTING_GROUP = {"routing", "routing_core", "word_extract", "range_word"}


def _canon_group(claim_type: str) -> str:
    c = claimcosts.canonical(claim_type)
    return "routing[+aux]" if c in _ROUTING_GROUP else c


def per_type(man: Manifest) -> dict:
    """canonical type-group -> [count, W, cids, Q]. Extracted W prefers the
    tape's exact w_slots; mode rejections surface as their own bucket."""
    agg: dict = {}
    for c in man.claims:
        try:
            w, cids, q = claimcosts.cost(c.type, c.params,
                                         w_hint=c.w_slots or 0.0)
        except ValueError as e:
            key = f"REJECTED:{_canon_group(c.type)}"
            agg.setdefault(key, [0, 0.0, 0.0, 0.0, str(e)])[0] += 1
            continue
        if c.w_slots is not None:
            w = c.w_slots
        row = agg.setdefault(_canon_group(c.type), [0, 0.0, 0.0, 0.0])
        row[0] += 1
        row[1] += w
        row[2] += cids
        row[3] += q
    return agg


def persistent_slots(man: Manifest) -> int:
    return sum(v.length for v in man.variables if v.persistent)


def rows_total(man: Manifest) -> int:
    ell = man.run["ligero"]["ELL"]
    return sum((v.length + ell - 1) // ell for v in man.variables)


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


# Per-type EXPECTED extra claims of modeled types on a real Maverick tape
# with C continuation positions, enumerated from the emitting code (the
# extract-only Max/Concat/InfoFinalize and table settlements are handled
# separately). prove_unexplained_info: one hadamard (gap_o^2), two
# paired_tlookups (exp kernel, pow pin), C embed(d=1) position selects and
# C-1 chain adds plus reveal() which REUSES AddClaim. build_inputs: one
# route_top1 on the token indicator (core + word-extract + range words —
# 4 claims at V=202048, capped at 6 for parameter headroom); its select
# matmul corresponds 1:1 to synth's embed.select and is NOT an extra —
# matmul deliberately has no allowance (review finding: a duplicated
# expert matmul must flag, not hide under a UI excuse).
#
# On top of that sits a REPRESENTATION delta, not a UI one: every synth
# `routing` claim is one bundled record where a real tape emits the pieces
# (core + word-extract + n_words range words = 2+n_words records, cost-sum
# identical by the claimcosts bundle identity), so each bundle contributes
# 1+n_words extra records to the group count. diff_report derives that
# exactly from synth's own bundles (24 x 4 = 96 for standard Maverick) —
# second-pass review finding: with only the input-route cap, real Maverick
# flags by construction. Cross-check vs the measured archive delta:
# 96 + 4 routing + UI's fixed handful matches the ~107 intercept, and the
# slope (1149-207)/450 = 2.09/position matches embed+add = 2/position.
def _ui_expected_extra(positions: int, synth_man: Manifest) -> dict:
    bundle_extra = sum(
        1 + int(c.params.get("n_words", 3)) for c in synth_man.claims
        if claimcosts.canonical(c.type) == "routing")
    return {
        "hadamard": 1,
        "ptlookup": 2,
        "embed_lookup": positions,
        "add": positions,               # (C-1) chain adds + 1 reveal pin
        "routing[+aux]": bundle_extra + 6,
    }


_REL_TOL = 0.001          # per-type W/cids/Q at equal claim counts
_UI_TOTAL_TOL = 0.02      # shared-type totals backstop when UI is present


def diff_report(sy: Manifest, ex: Manifest, ui_positions=None) -> list:
    """Print the synth-vs-extracted comparison; return FLAG strings.

    ui_positions: continuation-position count when the extracted tape
    carries a UI chain (maverick build_model always does) — each modeled
    type then tolerates at most _ui_expected_extra() additional claims,
    with shared-type W/cids/Q totals still bounded at _UI_TOTAL_TOL as a
    backstop (excess-count types can't be W-checked per claim). None =
    strict: every modeled type must match synth's count exactly, and all
    of W, cids, and Q per type within _REL_TOL."""
    flags = []
    a, b = per_type(sy), per_type(ex)
    ui_caps = _ui_expected_extra(ui_positions, sy) \
        if ui_positions is not None else {}
    print(f"\n{'type':24s} {'synth n':>8s} {'extr n':>8s} "
          f"{'synth W':>16s} {'extr W':>16s}  note")
    for t in sorted(set(a) | set(b), key=lambda t: -(b.get(t) or a[t])[1]):
        sa, sb = a.get(t), b.get(t)
        if t.startswith("REJECTED:"):
            print(f"{t:24s} {'-':>8s} {sb[0]:>8,d} {'':>16s} {'':>16s}  "
                  f"FLAG mode rejected by claimcosts: {sb[4]}")
            flags.append(f"claim mode rejected during costing: {t} ({sb[4]})")
            continue
        if sa and sb:
            note = ""
            if sb[0] < sa[0]:
                note = "FLAG extracted has FEWER than synth models"
                flags.append(f"{t}: extracted count {sb[0]} < synth {sa[0]}")
            elif sb[0] > sa[0]:
                extra = sb[0] - sa[0]
                cap = ui_caps.get(t, 0)
                if 0 < extra <= cap:
                    note = f"+{extra} extracted (UI chain, expected <= {cap})"
                elif cap:
                    note = f"FLAG +{extra} exceeds UI expectation {cap}"
                    flags.append(
                        f"{t}: {extra} extracted claim(s) beyond synth, but "
                        f"the UI chain accounts for at most {cap}")
                else:
                    note = f"FLAG +{extra} modeled claims"
                    flags.append(
                        f"{t}: {extra} extracted claim(s) beyond synth with "
                        f"no UI chain to attribute them to")
            else:
                drift = [(nm, sa[i], sb[i])
                         for i, nm in ((1, "W"), (2, "cids"), (3, "Q"))
                         if abs(sb[i] - sa[i]) > _REL_TOL * max(abs(sa[i]), 1.0)]
                if drift:
                    note = "FLAG " + ", ".join(
                        f"{nm} {_fmt(x)}->{_fmt(y)}" for nm, x, y in drift)
                    flags.append(
                        f"{t}: cost drift at equal claim count — " + ", ".join(
                            f"{nm} synth {_fmt(x)} vs tape {_fmt(y)}"
                            for nm, x, y in drift))
            print(f"{t:24s} {sa[0]:>8,d} {sb[0]:>8,d} "
                  f"{_fmt(sa[1]):>16s} {_fmt(sb[1]):>16s}  {note}")
        elif sb:
            why = KNOWN_EXTRACT_ONLY.get(t)
            note = f"extract-only: {why}" if why \
                else "FLAG extract-only type synth doesn't know"
            if not why:
                flags.append(f"unexpected extract-only claim type {t}")
            print(f"{t:24s} {'-':>8s} {sb[0]:>8,d} "
                  f"{'-':>16s} {_fmt(sb[1]):>16s}  {note}")
        else:
            print(f"{t:24s} {sa[0]:>8,d} {'-':>8s} "
                  f"{_fmt(sa[1]):>16s} {'-':>16s}  "
                  f"FLAG synth-only type missing from tape")
            flags.append(f"synth models {t} but the tape never emitted it")

    ps, pe = persistent_slots(sy), persistent_slots(ex)
    verdict = "exact match" if ps == pe else "FLAG MISMATCH"
    if ps != pe:
        flags.append(f"persistent (weight) slots: synth {ps:,} != tape {pe:,}")
    print(f"\npersistent (weight) slots: synth {ps:,}  tape {pe:,}  "
          f"[{verdict}]")

    for name, fn in [("claims", lambda m: len(m.claims)),
                     ("rows (per-var ceil)", rows_total)]:
        va, vb = fn(sy), fn(ex)
        print(f"{name:24s} synth {va:>14,}   tape {vb:>14,}   "
              f"delta {vb - va:+,}")
    # totals over SHARED modeled types: extract-only types (settlements, UI
    # Max/Concat/Info) are additive by design and already itemized above
    tol = _UI_TOTAL_TOL if ui_positions is not None else _REL_TOL
    for i, nm in ((1, "W"), (2, "cids"), (3, "Q")):
        ta = sum(r[i] for t, r in a.items() if not t.startswith("REJECTED:"))
        tb = sum(r[i] for t, r in b.items()
                 if t in a and not t.startswith("REJECTED:"))
        d = (tb - ta) / max(abs(ta), 1.0)
        over = abs(d) > tol
        if over:
            flags.append(f"total {nm} over shared types drifts {100 * d:+.2f}% "
                         f"(tolerance {100 * tol:.1f}%)")
        print(f"{nm + ' (shared types)':24s} synth {_fmt(ta):>14s}   "
              f"tape {_fmt(tb):>14s}   delta {100 * d:+.3f}%"
              + ("  FLAG" if over else ""))
    return flags


# ---------------------------------------------------------------- builders

def _placeholder(name: str):
    def load():
        raise RuntimeError(
            f"placeholder weight loader for {name} fired — crosscheck never "
            f"runs the engine pass, so nothing should load weight data")
    return load


def build_llama7b(seq: int, layers: int):
    """Mirror demo_llama7b.main's tape structure using the demo's own
    helpers, stopping at the built tape — main() itself has no build-only
    exit and would run the engine pass, which consumes the lazy claim list
    extract_tape requires intact.

    Weight matrices are committed as LAZY PLACEHOLDERS (commit_lazy defaults
    persistent=True, matching the real --lazy-weights path and synth's
    persistent flags) rather than via _commit_weights_random, whose eager
    tape.commit defaults persistent=False — which would zero the persistent
    slot count AND materialize ~53 GB of random tensors. The loaders never
    fire: extraction reads shape metadata only. Gains and the tiny norm
    weights stay eager non-persistent, matching both the demo and synth.

    Shapes come from the demo's RANDOM_WEIGHTS_CFG ModelConfig (the
    historical MHA Llama-2-7B — n_kv_heads == n_heads, so kv_cols == d;
    the KV shapes are written GQA-ready anyway). NOTE: re-mirrored for
    the ModelConfig demo rework on main 8314878; not yet re-validated on
    GPU hardware — the first crosscheck run of the next rental session
    is the gate, same as ever."""
    import torch
    import demo_llama7b as dl
    dl.SEQ = seq                      # module global the helpers shape against
    mcfg = dl.RANDOM_WEIGHTS_CFG      # historical Llama-2-7B shapes (MHA)
    d, d_ff = mcfg.d, mcfg.d_ff
    kv_cols = mcfg.n_kv_heads * mcfg.d_h   # == d for the MHA random config
    tape = dl.Tape(dl.CFG, silu_config=dl.SILU_CFG, lazy=True)
    resid = tape.commit("x_input",
                        dl._rand_signed(seq * d, half=dl.HALF_X),
                        (seq, d))
    identity_gain = torch.full((d,), dl.S, dtype=torch.uint64, device="cuda")

    def lazy_w(name, shape):
        return tape.commit_lazy(name, _placeholder(name), shape,
                                shape[0] * shape[1])

    for il in range(layers):
        sfx = f"_L{il}"               # names/shapes per _commit_weights_random
        weights = {
            "W_Q": lazy_w(f"W_Q{sfx}", (d, d)),
            "W_K": lazy_w(f"W_K{sfx}", (d, kv_cols)),
            "W_V": lazy_w(f"W_V{sfx}", (d, kv_cols)),
            "W_O": lazy_w(f"W_O{sfx}", (d, d)),
            "W_gate": lazy_w(f"W_gate{sfx}", (d, d_ff)),
            "W_up": lazy_w(f"W_up{sfx}", (d, d_ff)),
            "W_down": lazy_w(f"W_down{sfx}", (d_ff, d)),
            "rms_pre_attn_w": tape.commit(f"rms_pre_attn_w{sfx}",
                                          identity_gain.clone(), (d,)),
            "rms_pre_ffn_w": tape.commit(f"rms_pre_ffn_w{sfx}",
                                         identity_gain.clone(), (d,)),
        }
        resid = dl._run_block(tape, resid, weights, mcfg=mcfg)
        del weights
    vocab = mcfg.vocab
    final_norm = tape.commit("final_norm_w", identity_gain.clone(), (d,))
    w_lm = lazy_w("W_lm_head", (d, vocab))
    dl._run_tail(tape, resid, final_norm, w_lm, mcfg=mcfg,
                 vocab_size=vocab)
    model = dict(name="llama2-7b", d=d, heads=mcfg.n_heads,
                 head_dim=mcfg.d_h, vocab=vocab, layers=layers)
    return tape, model


def build_maverick(gguf: str, prompt_n: int, cont_n: int, *, layers: int,
                   experts: int, d: int, d_ff: int, vocab: int):
    """Mirror demo_maverick_full.main up to the built tape: same seeds, same
    synthetic-id generator, then build_model — which returns before any
    engine pass runs."""
    import torch
    import demo_maverick_full as dm
    torch.manual_seed(7)
    g = torch.Generator().manual_seed(11)
    prompt_ids = torch.randint(0, vocab, (prompt_n,), generator=g).tolist()
    cont_ids = torch.randint(0, vocab, (cont_n,), generator=g).tolist()
    tape = dm.Tape(dm.CFG, silu_config=dm.SILU_CFG, lazy=True)
    dm.build_model(tape, gguf, prompt_ids, cont_ids, V=vocab, d=d,
                   n_layers=layers, E=experts, d_ff=d_ff)
    model = dict(name="llama4-maverick", d=d, d_ff_expert=d_ff,
                 experts=experts, vocab=vocab, layers=layers)
    return tape, model


# ------------------------------------------------- layout breakdown probe

_LAYOUT_HEAD = re.compile(
    r"=== witness layout by claim type \(m_total=([\d,]+), W=([\d,]+)")
_LAYOUT_ROW = re.compile(
    r"^\s+(\S+)\s+rows=\s*([\d,]+)\s+elements=\s*([\d,]+)")


def parse_layout(text: str):
    """-> (m_total, {prover type name: (rows, elements)}) or (None, {})."""
    m = _LAYOUT_HEAD.search(text)
    if not m:
        return None, {}
    table = {}
    for line in text[m.start():].splitlines():
        r = _LAYOUT_ROW.match(line)
        if r:
            table[r.group(1)] = (int(r.group(2).replace(",", "")),
                                 int(r.group(3).replace(",", "")))
    return int(m.group(1).replace(",", "")), table


def layout_from_manifest(man: Manifest) -> dict:
    """Reproduce core's LIGERO_LAYOUT_BREAKDOWN aggregation from a manifest:
    every variable attributed to the first claim that touches it, in tape
    order — except the table commitments mult/w, which core books under
    TableSettlement (it reaches them only via the Table object on the
    settlement; the lookup claims that increment mult hold the Table, not
    the Variable). The lookup z variables are ALSO settlement inputs in the
    manifest, but each z is a direct field of its own lookup claim, which
    core encounters first — so z stays with the lookup, and the mirror
    separates the cases by producer: mult/w-style table vars are
    producer-less, every z's producer is its lookup claim."""
    ell = man.run["ligero"]["ELL"]
    by_name = man.var_by_name()
    owner: dict = {}
    for c in man.claims:              # producer-less table vars first
        if c.type == "TableSettlement":
            for name in c.inputs:
                v = by_name.get(name)
                if v is not None and v.producer is None:
                    owner.setdefault(name, c.type)
    for c in man.claims:                          # then first-touch
        for name in list(c.outputs) + list(c.inputs):
            owner.setdefault(name, c.type)
    agg: dict = {}
    for name, t in owner.items():
        v = by_name.get(name)
        if v is None:
            continue
        row = agg.setdefault(t, [0, 0])
        row[0] += (v.length + ell - 1) // ell
        row[1] += v.length
    return {t: tuple(r) for t, r in agg.items()}


def diff_layout(probe_m_total, probe: dict, man: Manifest) -> list:
    flags = []
    ours = layout_from_manifest(man)
    print(f"\n{'type':24s} {'prover rows':>12s} {'manifest rows':>14s} "
          f"{'prover elems':>16s} {'manifest elems':>16s}")
    for t in sorted(set(probe) | set(ours),
                    key=lambda t: -(probe.get(t) or ours[t])[1]):
        pr, ov = probe.get(t), ours.get(t)
        note = ""
        if pr is None:
            note = "FLAG manifest-only"
            flags.append(f"layout: manifest books {t} but prover doesn't")
        elif ov is None:
            note = "FLAG prover-only"
            flags.append(f"layout: prover books {t} but the manifest "
                         f"never extracted it")
        elif pr != ov:
            note = "FLAG differs"
            flags.append(f"layout: {t} prover rows/elems {pr} != "
                         f"manifest {ov}")
        cell = lambda pair, i: f"{pair[i]:,}" if pair else "-"  # noqa: E731
        print(f"{t:24s} {cell(pr, 0):>12} {cell(ov, 0):>14} "
              f"{cell(pr, 1):>16} {cell(ov, 1):>16}  {note}")
    mrows = rows_total(man)
    if probe_m_total is not None:
        gap = probe_m_total - mrows
        print(f"\nprover m_total {probe_m_total:,} vs manifest rows {mrows:,} "
              f"(gap {gap:+,} = blinding/protocol rows — small and "
              f"T_QUERIES-shaped is expected; negative is a FLAG)")
        if gap < 0:
            flags.append(f"layout: manifest rows {mrows:,} EXCEED prover "
                         f"m_total {probe_m_total:,}")
    return flags


def run_layout_probe(cmd: list, timeout: int) -> str:
    env = dict(os.environ, LIGERO_LAYOUT_BREAKDOWN="1",
               PYTHONPATH=os.pathsep.join(
                   [str(_REPO / "prover"), str(_REPO / "demo"),
                    os.environ.get("PYTHONPATH", "")]))
    print(f"\n[layout probe] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env=env, cwd=str(_REPO))
    if not _LAYOUT_HEAD.search(r.stdout):
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        raise RuntimeError("layout probe produced no breakdown table "
                           "(see output above)")
    return r.stdout


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="extract a real demo tape and diff it against synth")
    ap.add_argument("model", choices=["llama7b", "maverick"])
    ap.add_argument("--seq", type=int, default=None,
                    help="llama7b context length (default 100)")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--from-gguf", default=None, help="maverick GGUF path")
    ap.add_argument("--prompt-n", type=int, default=2)
    ap.add_argument("--cont-n", type=int, default=2)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--d-ff", type=int, default=8192)
    ap.add_argument("--vocab", type=int, default=202048)
    ap.add_argument("--t-queries", type=int, default=None,
                    help="sets LIGERO_T_QUERIES before the demo imports read it")
    ap.add_argument("--layout", action="store_true",
                    help="also run the prover's LIGERO_LAYOUT_BREAKDOWN probe "
                         "(subprocess; runs the demo's pre-prove passes, so "
                         "keep the context small) and diff row layouts")
    ap.add_argument("--layout-timeout", type=int, default=3600)
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("-o", "--out-dir", default="crosscheck-out")
    a = ap.parse_args(argv)

    if a.t_queries is not None:      # must precede any demo import
        os.environ["LIGERO_T_QUERIES"] = str(a.t_queries)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    flags = []

    if not a.skip_selftest:
        print("[1] extract.py --selftest")
        r = subprocess.run([sys.executable, str(_PROFILER / "extract.py"),
                            "--selftest"], cwd=str(_REPO))
        if r.returncode != 0:
            print("FLAG selftest failed — stopping")
            return 1

    print("[2] building lazy tape (claim recording only — no witness "
          "compute, no weights load beyond metadata)")
    if a.model == "llama7b":
        seq = a.seq or 100
        layers = a.layers if a.layers is not None else 32
        tape, model = build_llama7b(seq, layers)
        sy = synth.BUILDERS["llama7b"](seq, layers=layers,
                                       t_queries=tape.cfg.T_QUERIES)
        probe_cmd = [sys.executable, str(_REPO / "demo" / "demo_llama7b.py"),
                     "--seq", str(seq), "--num-layers", str(layers),
                     "--engine"]
    else:
        if not a.from_gguf:
            ap.error("maverick needs --from-gguf (metadata is read eagerly)")
        if a.seq is not None:
            a.prompt_n, a.cont_n = 2, a.seq - 2
        seq = a.prompt_n + a.cont_n
        layers = a.layers if a.layers is not None else 48
        tape, model = build_maverick(a.from_gguf, a.prompt_n, a.cont_n,
                                     layers=layers, experts=a.experts,
                                     d=a.d, d_ff=a.d_ff, vocab=a.vocab)
        sy = None
        if layers == 48 and a.experts == 128 and a.d == 5120:
            sy = synth.BUILDERS["maverick"](seq,
                                            t_queries=tape.cfg.T_QUERIES)
        else:
            print("note: non-standard shape — synth models the full 48x128 "
                  "Maverick only; skipping the synth diff (layout diff "
                  "still runs)")
        probe_cmd = [sys.executable,
                     str(_REPO / "demo" / "demo_maverick_full.py"),
                     "--from-gguf", a.from_gguf,
                     "--prompt-n", str(a.prompt_n), "--cont-n", str(a.cont_n),
                     "--layers", str(layers), "--experts", str(a.experts),
                     "--d", str(a.d), "--d-ff", str(a.d_ff),
                     "--vocab", str(a.vocab)]

    print(f"    built: {len(tape.claims):,} claims")
    print("[3] extracting manifest")
    from extract import extract_tape
    man = extract_tape(tape, model=model, seq=seq)
    man_path = out / f"{a.model}-s{seq}-extracted.json"
    man.save(str(man_path))
    print(f"    saved {man_path} ({len(man.claims):,} claims, "
          f"{len(man.variables):,} variables)")

    if sy is not None:
        print("[4] diff vs synth")
        ui_positions = a.cont_n if a.model == "maverick" else None
        flags += diff_report(sy, man, ui_positions=ui_positions)

    if a.layout:
        print("[5] prover layout probe (this runs the demo up to the start "
              "of prove, including any engine pass — minutes at small "
              "context)")
        text = run_layout_probe(probe_cmd, a.layout_timeout)
        (out / f"{a.model}-s{seq}-layout.txt").write_text(text)
        m_total, table = parse_layout(text)
        flags += diff_layout(m_total, table, man)

    print("\n" + "=" * 70)
    if flags:
        print(f"RESULT: {len(flags)} FLAG(s) — eyeball before trusting "
              f"extracted manifests:")
        for f in flags:
            print(f"  - {f}")
        return 1
    print("RESULT: clean — no flags. Extracted manifest agrees with synth "
          "within the documented unmodeled set"
          + (" and with the prover's own layout" if a.layout else "")
          + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
