"""TapeProver — the reference prover behind the verifier's staged interface.

verify.run_verification / run_verification_fast call a `prover(stage, *seeds)`
callable. This is that callable for core.py's GPU prover. It owns the witness
(claims + inputs, including lazy loaders) — the verifier never sees it — and
maps the verifier's per-round SEEDS to the expanded challenge forms core.prove
consumes, and core's tensor outputs back to the int lists the verifier checks.

A new prover (different backend, optimizations, new claim types) just implements
the same `prove(stage, *seeds)` shape; the verifier never reads its code.

The seed→challenge expansion here MUST match protocol/compile_claims exactly
(it does — both go through core._sample_chs / protocol.op_vec / _combiner_vec /
random_columns by index), so the verifier and prover agree with no values sent.
"""
import torch
import core
import protocol as pr


def _ints(t):
    return [int(v) for v in t.cpu().tolist()]


class TapeProver:
    """Wraps core.prove's staged signature as prover(stage, s_op, s_comb, s_col).
    Holds claims+inputs (the witness); forwards lazy inputs untouched so the
    streaming/lazy memory behavior of core.prove is preserved."""

    def __init__(self, claims, inputs, cfg):
        self.claims = claims
        self.inputs = inputs
        self.cfg = cfg
        # The settled list (ops + appended table settlements) — the order both
        # sides index challenges by. Needed to expand s_op into ch0.
        self._settled = core._with_synthesized_settlements(claims)
        _, _, _, _, self._m_total = core._layout(self._settled, cfg)

    # ---- seed → the expanded challenge forms core.prove consumes ----
    def _ch0(self, s_op):
        return core._sample_chs(self._settled, s_op)         # per-claim op challenges

    def _ch1(self, s_op, s_comb):
        # combiner tensors (r_irs, r_lin, r_quad). Lengths need n_lin / n_quad,
        # which come from compiling with ch0 (same as core.verify does).
        _, quads, _, _, n_lin = core._compile_all(self._settled, s_op, self.cfg, self._m_total)
        mk = lambda lbl, n: torch.tensor(core._combiner_vec(s_comb, lbl, n),
                                         dtype=torch.uint64, device="cuda")
        return (mk("irs", self._m_total - core.NUM_BLINDING_ROWS),
                mk("lin", n_lin),
                mk("quad", len(quads)))

    def _ch2(self, s_col):
        return pr.random_columns(s_col, self.cfg)            # opened columns Q

    # ---- the staged callable the verifier drives ----
    def __call__(self, stage, s_op=None, s_comb=None, s_col=None):
        cl, inp, cfg = self.claims, self.inputs, self.cfg
        if stage == 1:
            return core.prove(cl, inp, cfg)                              # root_p1
        if stage == 2:
            return core.prove(cl, inp, cfg, self._ch0(s_op))            # root_p2
        if stage == 3:
            q_irs, q_lin, p_0 = core.prove(cl, inp, cfg,
                                           self._ch0(s_op), self._ch1(s_op, s_comb))
            return _ints(q_irs), _ints(q_lin), _ints(p_0)
        if stage == 4:
            o1, o2, p1, p2 = core.prove(cl, inp, cfg,
                                        self._ch0(s_op), self._ch1(s_op, s_comb),
                                        self._ch2(s_col))
            return ({j: _ints(o1[j]) for j in o1}, {j: _ints(o2[j]) for j in o2}, p1, p2)
        if stage == 0:   # fused: everything in one streaming pass (~4× faster)
            pf = core.prove(cl, inp, cfg, self._ch0(s_op), self._ch1(s_op, s_comb),
                            self._ch2(s_col), returnEverything=True)
            return (pf.root_p1, pf.root_p2, _ints(pf.q_irs), _ints(pf.q_lin), _ints(pf.p_0),
                    {j: _ints(pf.opened_p1[j]) for j in pf.opened_p1},
                    {j: _ints(pf.opened_p2[j]) for j in pf.opened_p2},
                    pf.paths_p1, pf.paths_p2)
        raise ValueError(f"bad stage {stage}")
