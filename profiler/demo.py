"""A guided tour of the profiler. Pure Python, no torch — runs anywhere:

    python3 profiler/demo.py

Builds a small synthetic Maverick manifest in memory, then walks the pieces
in dependency order: the manifest contract, per-claim costs, the whole-run
prediction, the dependency DAG, and the partition scorecard. Each section
prints a short narration and then real output. Read top to bottom alongside
README.md; runtime is a few seconds. The one piece this tour can't run is
extract.py (building a real tape needs torch/CUDA) — see the last section.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import claimcosts                            # noqa: E402
import dag as dagmod                         # noqa: E402
import partition                             # noqa: E402
import predict                               # noqa: E402
import synth                                 # noqa: E402
from machine import MachineProfile           # noqa: E402

SEQ = 100          # small context: builds fast, structure identical to S=1000


def section(title: str, blurb: str) -> None:
    print("\n" + "=" * 78)
    print(f"== {title}")
    print("=" * 78)
    print(textwrap.fill(textwrap.dedent(blurb).strip(), width=78))
    print()


def main() -> None:
    section("1. The manifest: one contract, two producers", f"""
        Everything downstream consumes a manifest — one record per proving
        claim (including table settlements) plus one per variable — and never
        the tape itself. synth.py builds one in closed form (below); extract.py
        walks a real Tape into the same format on a GPU box. We build synthetic
        Maverick at S={SEQ}: claim COUNT is context-independent, only the shape
        params change.
    """)
    man = synth.BUILDERS["maverick"](SEQ)
    print(f"built: {len(man.claims):,} claims, {len(man.variables):,} variables "
          f"(model={man.model['name']}, seq={man.run['seq']})")

    section("2. The records: claims and variables", """
        A ClaimRecord is a typed proving claim with shape params and its
        variable wiring; a VariableRecord knows its producer claim and every
        consumer, which is what the DAG, live-set replay, and traffic model all
        run on. Weights are persistent producer-less inputs. Here is one of the
        expert matmuls of MoE layer 1 and its weight:
    """)
    expert_claim = next(c for c in man.claims if ".e3.up" in (c.label or ""))
    print(f"  claim {expert_claim.idx}: type={expert_claim.type} "
          f"label={expert_claim.label} layer={expert_claim.layer}")
    print(f"    params={expert_claim.params}")
    print(f"    inputs={expert_claim.inputs}")
    print(f"    outputs={expert_claim.outputs}")
    weight = next(v for v in man.variables
                  if v.name == expert_claim.inputs[1])
    print(f"  variable {weight.name}: length={weight.length:,} "
          f"persistent={weight.persistent} producer={weight.producer} "
          f"consumers={weight.consumers}")

    section("3. claimcosts: (W, #cids, Q) per claim", """
        Three cost drivers per claim — witness slots, linear constraint ids,
        quadratic products — using the active-in-production formulas, verified
        against the prover's compile functions. Two properties worth knowing:
        the synth `routing` bundle equals the separate pieces a real tape emits
        (so both producers sum identically), and a claim carrying a mode flag
        we have no formula for is REJECTED, never mispriced.
    """)
    print(f"  {expert_claim.type}({expert_claim.params}):  W, cids, Q = "
          f"{claimcosts.cost(expert_claim.type, expert_claim.params)}")
    T, E, nw = man.run["seq"], man.model["experts"], 3
    bundle = claimcosts.cost("routing", dict(T=T, E=E, n_words=nw))
    pieces = [claimcosts.cost("RoutingClaim", dict(T=T, E=E)),
              claimcosts.cost("WordExtractionClaim",
                              dict(length=T * E, n_words=nw)),
              tuple(nw * x for x in claimcosts.cost(
                  "RangeWordClaim", dict(length=T * E)))]
    summed = tuple(sum(t[i] for t in pieces) for i in range(3))
    print(f"  routing bundle == core + word + {nw}*range:  "
          f"{bundle == summed}")
    try:
        claimcosts.cost("RmsNormClaim",
                        dict(B=T, d=man.model["d"], rescale=True))
    except ValueError as e:
        print(f"  non-production mode -> rejected: {e}")

    section("4. predict: the dry-run cost report", """
        Totals x a measured machine profile. Prove time is a BRACKET on
        purpose: the floor (A*W + B*cids + C*Q) is the NTT-bound target where
        A/C ride memory bandwidth and B rides compute — they scale to new
        hardware by different ratios; the aggregate (~40 ns/slot) is calibrated
        on today's code. Their ratio bounds the remaining implementation
        overhead; per-phase validation is what will attribute the gap. Memory is
        reported term by term with the unattributed remainder stated explicitly.
    """)
    mp = MachineProfile.load("gb10-spark")
    print(predict.report(man, mp))

    section("5. dag: what a dispatcher will schedule against", """
        Claim-level dependency graph (weights excluded — they stream, they are
        not dataflow edges): critical path via longest weighted path, width
        profile, and the ideal witness-parallelism bound. The four protocol
        rounds are hard barriers on top of this.
    """)
    print(dagmod.summary_text(dagmod.build(man)))

    section("6. partition: the strategy scorecard", """
        Maps claims to N shards three ways (contiguous rows, layer pipeline,
        expert round-robin), then scores per-shard work, imbalance, weight
        streaming, opened-column memory, and reduction-aware cross-shard
        traffic: Freivalds combines and LogUp multiplicities ship one partial
        per remote shard, not their inputs. Interconnect topology is unknown,
        so comms are priced across a swept bandwidth: conclusions stable
        across the sweep are actionable now; ones that flip wait for topology.
    """)
    print(partition.compare(man, 4, mp, weight_bytes_per_param=0.7))

    section("7. Where to go next", """
        The exact path: on a GPU box, build any model's Tape(cfg, lazy=True)
        exactly as the demos do and extract_tape(tape, ...) it. Lazy recording
        defers per-claim witness computation, and commit_lazy defers weight
        loads; LogUp tables are the exception and still materialize eagerly, so
        tape construction requires CUDA and table memory. `python3
        profiler/extract.py --selftest` smoke-tests the walker; diffing a real
        extraction against synth and LIGERO_LAYOUT_BREAKDOWN=1 is roadmap item 1
        and the gate for trusting extracted manifests. `python3
        profiler/test_profiler.py` runs the no-torch regression suite. README.md
        has the validation table, caveats, and the roadmap.
    """)
    print("tour complete.")


if __name__ == "__main__":
    main()
