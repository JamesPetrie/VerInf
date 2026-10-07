"""Active-only MoE FFN block (S3).

Replaces the builder that materializes 128 gate + 128 up + 128 down expert
outputs per layer and folds them with three `freivalds_combine` calls — the
first forbidden regression in demo/4h-production-runbook.md, and the reason the
ordinary tape carries 396 G witness slots at S=1000.

Here each of the three expert matrices becomes ONE `RoutedProjectedMatmulClaim`
followed by its `RescaleClaim`:

    x_r --gate--> raw --rescale--> g   \\
                                        silu(g) * u --down--> raw --rescale--> ffn
    x_r --up----> raw --rescale--> u   /

The expert shards stay one variable each, so the witness pass loads a single
expert at a time and computes only the tokens routed to it, and the projection
P = W*rho streams the same way (cached per rho for the proof's lifetime).

Routing, the sigmoid gate on the chosen logit, and the shared expert are
unchanged from the old builder — this stage changes how the expert matmuls are
proved, nothing else.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))

from rescale_claim import rescale
from routed_projected import routed_projected_matmul
from routing_claim import route_top1, freivalds_combine


def routed_expert_matmul(tape, x, m, w_experts, *, T, K, J, E,
                         s_in, s_out, output_width):
    """One routed expert matmul plus the standalone rescale that must follow
    its raw accumulator."""
    raw = routed_projected_matmul(tape, x, m, w_experts, T=T, K=K, J=J, E=E)
    return rescale(tape, raw, s_in=s_in, s_out=s_out, output_width=output_width)


def build_moe_ffn_routed(tape, n2g, w, sig_tbl, ones_bc, *, T, E, d, d_ff,
                         S, output_width, sig_shift):
    """The MoE FFN of one layer, active-only.

    `w` supplies the layer's weights:
      w["router"]                  (d, E)
      w["gate"], w["up"]           lists of E shards, each (d, d_ff)
      w["down"]                    list of E shards, each (d_ff, d)
      w["gate_sh"], w["up_sh"], w["down_sh"]   the shared expert
    Values are WitnessTensors already committed on the tape (lazily, in
    production: one GGUF shard per variable).
    """
    mm = dict(s_a=S, s_b=S, s_out=S, output_width=output_width)
    r = tape.matmul(n2g, w["router"], **mm)
    m, r_chosen, _gap = route_top1(tape, r, T=T, E=E, B_logit=output_width,
                                   word_bits=11)
    s_val = tape.paired_tlookup(r_chosen, sig_tbl, shift=sig_shift)
    s_rep = freivalds_combine(tape, s_val, [ones_bc], T=T, E=1, F=d)
    x_r = tape.hadamard(s_rep, n2g, **mm)

    rp = dict(s_in=S * S, s_out=S, output_width=output_width)
    g = routed_expert_matmul(tape, x_r, m, w["gate"], T=T, K=d, J=d_ff, E=E, **rp)
    u = routed_expert_matmul(tape, x_r, m, w["up"], T=T, K=d, J=d_ff, E=E, **rp)
    hidden = tape.hadamard(tape.silu(g), u, **mm)
    ffn = routed_expert_matmul(tape, hidden, m, w["down"], T=T, K=d_ff, J=d,
                               E=E, **rp)

    h_s = tape.hadamard(tape.silu(tape.matmul(n2g, w["gate_sh"], **mm)),
                        tape.matmul(n2g, w["up_sh"], **mm), **mm)
    return ffn + tape.matmul(h_s, w["down_sh"], **mm)
