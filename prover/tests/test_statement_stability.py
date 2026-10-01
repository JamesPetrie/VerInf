"""A tape's statement bytes do not change when it is proved (CPU; review
2026-09-23 finding 8).

The bridge attaches its per-proof P_trace pin to the routed claim as
_bridge_pin, and claims_to_json serialized every attribute, so the same tape
had different statement bytes, and a different statement digest, before and
after its first bridged proof. Underscore attributes are prover-side state and
are never serialized. The repeated-proof check on a card is
test_wc_streaming_flag.test_repeated_bridged_proofs_keep_one_statement."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

import protocol as pr
from routed_projected import RoutedProjectedMatmulClaim, routed_projected_matmul
from tape import Tape
from tests.test_routed_projected import CFG, T, K, J, E


def _u64(t):                       # on the host: no card needed to build
    return t.to(torch.int64).to(torch.uint64)


def _bridged_tape():
    tape = Tape(CFG, lazy=True)
    x = tape.commit("X", _u64(torch.arange(1, T * K + 1)), (T, K))
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate((0, 1, 0)):
        M[t, e] = 1
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    w = [tape.commit(f"W{e}", _u64(torch.arange(1, K * J + 1) + e), (K, J))
         for e in range(E)]
    routed_projected_matmul(tape, x, m, w, T=T, K=K, J=J, E=E, use_bridge=True)
    return tape


def _statement(tape):
    return json.dumps(pr.claims_to_json(tape.claims, tape.cfg), sort_keys=True)


def test_bridge_pin_is_not_statement():
    tape = _bridged_tape()
    before = _statement(tape)
    claim = next(c for c in tape.claims if isinstance(c, RoutedProjectedMatmulClaim))
    claim._bridge_pin = torch.arange(E * K, dtype=torch.int64)
    assert _statement(tape) == before
    assert "_bridge_pin" not in before
