"""WC-LCRL-STC integration brick 1 (spec §0.5/§0.8 precondition): the
bridge's P_trace must BE the tape's Pj variable — same rho, same values,
same layout — not an unrelated copy.

Builds the routed toy tape from test_routed_projected, proves it, recovers
the claim's actual rho from the proof's own s_op coin, enrolls THE SAME
expert weight tensors, and checks byte equality between bridge_r2's
P_trace and the P the prover's _project_weights computed.  This is the
wire the transcript integration will solder: rho is injected from the host
transcript, not derived by the bridge."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import protocol as pr
import wc_bridge as wc
from routed_projected import (RoutedProjectedMatmulClaim, _project_weights,
                              routed_sample, clear_p_cache)
from tests.test_routed_projected import _build, T, K, J, E


def test_bridge_p_trace_is_the_tape_pj():
    tape, y, X, W, M = _build()
    proof = tape.prove()
    s_op = proof.seeds["s_op"]
    claim = next(c for c in tape.claims
                 if isinstance(c, RoutedProjectedMatmulClaim))
    ci = tape.claims.index(claim)
    rho = routed_sample(claim, ci, s_op)          # the claim's real R1 coin

    # enroll THE SAME expert tensors, expert-major (spec §0.5: all E expert
    # enrollments), one width-J group whose row order is exactly the claim's
    # P layout [e*K + k].  tape.inputs holds the committed flats (tape.py).
    def _flat(wv):
        val = tape.inputs[wv]
        return (val() if callable(val) else val)
    w_rows = torch.cat([_flat(wv).reshape(K, J) for wv in claim.W])
    params = wc.WcParams(B=48, lam=16, N_w=128, q_w=8)
    enr = wc.build_enrollment({J: w_rows.cuda()}, b"link-mask",
                              b"link-manifest", params)

    p_trace, pi = wc.bridge_r2(enr, {J: rho})

    # the prover's own projection of the same weights under the same rho
    clear_p_cache()
    live_vals = {wv: w_rows[e * K:(e + 1) * K].reshape(-1)
                 for e, wv in enumerate(claim.W)}
    P_prover = _project_weights(claim, live_vals, rho).reshape(-1)

    got = p_trace[J][:E * K].cpu()
    want = P_prover.cpu()
    assert torch.equal(got.view(torch.int64), want.view(torch.int64)), \
        "bridge P_trace != the tape's Pj — the §0.5 substitution would be unsound"
    # and the padded tail past E*K stays zero-projected (bound by the bridge)
    tail = p_trace[J][E * K:].cpu().view(torch.int64)
    assert int(tail.abs().sum()) == 0
