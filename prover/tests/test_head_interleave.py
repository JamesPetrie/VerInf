"""HeadInterleaveClaim on a toy tape: honest proofs the Rust verifier ACCEPTs
and targeted tampers it must REJECT (analysis/mla-attention-design.md §4.3).

Honest: a toy latent-attention assembly — each head's query from its own
two parts, each head's key from its own first part and ONE shared second
part — feeding the multi-head scores matmul at head width w1 + w2, values
checked against a direct reference, then proved and verified.
Tampers, each on a tape with one interleave claim so only its constraints can
catch it: a dst slot in the first part, a dst slot in the per-head second
part, and a fanned-out slot changed in one head only. Every tamper is
counted and must have been applied.

Needs a card. Run:  python prover/tests/run_tests.py test_head_interleave
The constraint map itself is checked on the CPU (test_head_interleave_layout.py)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import core
import claims as _C          # noqa: F401
import packets as _PK        # noqa: F401
import head_interleave as hi
from _rust_verify import rust_verify_tape
from compute_fns import WITNESS_TAMPER
from tape import Tape

P = core.P
CFG = core.LigeroConfig(ELL=16, K_DEG=16, N_LIG=64, T_QUERIES=4)
SEED = b"head-interleave"
T, H, W1, W2 = 3, 4, 4, 2
S = 1 << 4


def _u64(t):
    """Signed integers to Goldilocks field elements: v mod P, not the 2^64
    wrap a uint64 cast gives a negative."""
    return torch.from_numpy(np.array([int(v) % P for v in t.tolist()],
                                     dtype=np.uint64)).cuda()


def _signed(t):
    """Field elements back to signed integers: above P/2 means negative."""
    vals = [v % (1 << 64) for v in t.contiguous().view(torch.int64).cpu().tolist()]
    return [v - P if v > P // 2 else v for v in vals]


_HITS = {}


def _tamper(field_idx):
    WITNESS_TAMPER.clear()
    _HITS.clear()
    key = ("HeadInterleaveClaim", "dst")

    def fn(t):
        _HITS[key] = _HITS.get(key, 0) + 1
        t = t.clone().contiguous()
        t.view(torch.int64)[field_idx] += 1
        return t
    WITNESS_TAMPER[key] = fn


def _verdict(tape):
    try:
        proof = tape.prove(seed=SEED)
        result = rust_verify_tape(tape, proof, seed=SEED)
        unapplied = [k for k in WITNESS_TAMPER if not _HITS.get(k)]
    finally:
        WITNESS_TAMPER.clear()
        _HITS.clear()
    assert not unapplied, f"tamper(s) never applied: {unapplied}"
    return result


def _parts(seed=3):
    g = torch.Generator().manual_seed(seed)
    r = lambda n: torch.randint(-8, 8, (n,), generator=g)
    return dict(q_nope=r(T * H * W1), q_pe=r(T * H * W2), k_nope=r(T * H * W1), k_pe=r(T * W2))


def _attention_tape():
    core._COSET_POWERS_K_CACHE.clear()
    p = _parts()
    tape = Tape(CFG, lazy=True)
    c = {n: tape.commit(n, _u64(v), (T, v.numel() // T)) for n, v in p.items()}
    q = hi.head_interleave(tape, c["q_nope"], c["q_pe"], T=T, H=H, w1=W1, w2=W2)
    k = hi.head_interleave(tape, c["k_nope"], c["k_pe"], T=T, H=H, w1=W1, w2=W2, shared=True)
    sc = tape.matmul(q, k, transpose_b=True, heads=H, head_dim=W1 + W2,
                     s_a=S, s_b=S, s_out=S, output_width=16)
    return tape, p, q, k, sc


def test_honest_latent_attention_assembly_accepts():
    tape, p, q, k, sc = _attention_tape()
    live = tape.run_engine_pass()
    qv = torch.tensor(_signed(live[q.var])).view(T, H, W1 + W2)
    kv = torch.tensor(_signed(live[k.var])).view(T, H, W1 + W2)
    want_q = torch.cat([p["q_nope"].view(T, H, W1), p["q_pe"].view(T, H, W2)], dim=2)
    want_k = torch.cat([p["k_nope"].view(T, H, W1),
                        p["k_pe"].view(T, 1, W2).expand(T, H, W2)], dim=2)
    assert torch.equal(qv, want_q) and torch.equal(kv, want_k)
    # the scores directly: per head, query t against key u over all w1 + w2
    # dimensions, rescaled by the signed floor of S·S -> S; layout (t, h, u)
    raw = torch.einsum("thc,uhc->thu", want_q, want_k)
    want_sc = torch.div(raw, S, rounding_mode="floor")
    got_sc = torch.tensor(_signed(live[sc.var])).view(T, H, T)
    assert torch.equal(got_sc, want_sc), (got_sc, want_sc)
    assert want_sc.abs().max() < (1 << 15), "fixture scores outside the 16-bit output"
    acc, msg = _verdict(_attention_tape()[0])
    assert acc, f"honest assembly + scores: expected ACCEPT ({msg})"
    print("    per-head query, shared-key assembly, scores at head width 6: ACCEPT")


def _single(shared):
    core._COSET_POWERS_K_CACHE.clear()
    p = _parts(seed=7)
    tape = Tape(CFG, lazy=True)
    a = tape.commit("a", _u64(p["k_nope"]), (T, H * W1))
    b = tape.commit("b", _u64(p["k_pe"] if shared else p["q_pe"]),
                    (T, W2 if shared else H * W2))
    hi.head_interleave(tape, a, b, T=T, H=H, w1=W1, w2=W2, shared=shared)
    return tape


def _dst_index(t, h, col):
    return (t * H + h) * (W1 + W2) + col


def test_tampered_first_part_rejects():
    tape = _single(False)
    _tamper(_dst_index(1, 2, 1))
    acc, _ = _verdict(tape)
    assert not acc, "a tampered first-part slot accepted"
    print("    tampered first-part slot: REJECT")


def test_tampered_second_part_rejects():
    tape = _single(False)
    _tamper(_dst_index(2, 0, W1 + 1))
    acc, _ = _verdict(tape)
    assert not acc, "a tampered per-head second-part slot accepted"
    print("    tampered per-head second-part slot: REJECT")


def test_a_shared_slot_changed_in_one_head_rejects():
    tape = _single(True)
    _tamper(_dst_index(1, 3, W1))       # head 3 only; heads 0-2 keep the shared value
    acc, _ = _verdict(tape)
    assert not acc, "a fanned-out slot that differs in one head accepted"
    print("    shared key slot changed in one head only: REJECT")


def test_honest_single_claims_accept():
    for shared in (False, True):
        acc, msg = _verdict(_single(shared))
        assert acc, f"honest single interleave (shared={shared}): expected ACCEPT ({msg})"
    print("    honest single interleave, per head and shared: ACCEPT")
