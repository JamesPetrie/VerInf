"""Does Maverick's enrollment bind its RMSNorm gains and the gate's ones
operand? (GPU; a probe, run outside the gates.)

demo_maverick_full.build_model commits each RMSNorm gain and g_out with
tape.commit (not persistent: outside the enrolled weight block), and the MoE's
`bc_ones`, the operand freivalds_combine uses to broadcast the sigmoid gate
over the routed input, the same way; no claim pins their values. This probe
builds a toy tape from the driver's own pieces (build_inputs, the g_out
pattern, build_moe_ffn with its sigmoid table and ones operand), with no
public output, so a tamper changes neither the statement nor the enrolled
weights. Each tampered tape is proved against the honest run's
WeightCommitment and checked by the Rust verifier under the honest weight
root and statement digest.

- control, honest: accepted under the honest anchors;
- control, an enrolled weight (the router) altered: must be rejected;
- bc_ones = 2, and g_out raised by about 0.1 in one channel: SHOULD be
  rejected. If they are accepted while the layer's output differs from the
  honest run's, the enrollment does not bind them.

The assertions state the property (rejection), so until Maverick's driver
binds these inputs the last two are expected to FAIL; that failure is the
confirmation. Plain test functions, for tests/run_tests.py."""
import pathlib
import sys
import tempfile

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE, _HERE.parents[1] / "demo"):
    sys.path.insert(0, str(_p))

import _uint64_compat  # noqa: E402,F401
import torch           # noqa: E402

import core                               # noqa: E402
import demo_maverick_full as dm           # noqa: E402
from _rust_verify import rust_verify_anchored  # noqa: E402
from max_claim import to_signed           # noqa: E402
from tape import Tape                     # noqa: E402

CFG = core.LigeroConfig(ELL=16, K_DEG=16, N_LIG=64, T_QUERIES=4)
V, D, DFF, E, IL = 32, 64, 64, 4, 1
PROMPT, CONT = [3, 17], [9, 22]
T = len(PROMPT) + len(CONT)
EPS_INT = round(1e-5 * dm.S * dm.S)
_DIR = []


def _gguf():
    """Maverick's names; Q8_0 experts, F32 elsewhere (test_gguf_loader's mix)."""
    if _DIR:
        return _DIR[1]
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType as Q
    from gguf.quants import quantize
    rng = np.random.default_rng(3)
    n = lambda std, *s: rng.normal(0.0, std, s).astype(np.float32)
    t = {"token_embd.weight": n(0.5, V, D),
         "output_norm.weight": (1.0 + rng.normal(0.0, 0.1, D)).astype(np.float32),
         f"blk.{IL}.ffn_gate_inp.weight": n(D ** -0.5, E, D),
         f"blk.{IL}.ffn_gate_exps.weight": n(D ** -0.5, E, DFF, D),
         f"blk.{IL}.ffn_up_exps.weight": n(D ** -0.5, E, DFF, D),
         f"blk.{IL}.ffn_down_exps.weight": n(DFF ** -0.5, E, D, DFF),
         f"blk.{IL}.ffn_gate_shexp.weight": n(D ** -0.5, DFF, D),
         f"blk.{IL}.ffn_up_shexp.weight": n(D ** -0.5, DFF, D),
         f"blk.{IL}.ffn_down_shexp.weight": n(DFF ** -0.5, D, DFF)}
    td = tempfile.TemporaryDirectory()
    path = str(pathlib.Path(td.name) / "maverick-toy.gguf")
    w = GGUFWriter(path, "llama4")
    for name, a in t.items():
        if "_exps" in name:
            qd = quantize(a, Q.Q8_0)
            w.add_tensor(name, qd, raw_shape=qd.shape, raw_dtype=Q.Q8_0)
        else:
            w.add_tensor(name, a, raw_dtype=Q.F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file()
    w.close()
    _DIR.extend([td, path])
    return path


def _bump(t, i=3, by=410):
    """One entry raised by `by` (~0.1 at scale 4096; the field element of a
    small positive value stays canonical)."""
    u = t.contiguous().view(torch.int64).clone()
    u[i] = u[i] + by
    return u.view(torch.uint64)


def _build(*, ones=1, gain_bump=False, router_bump=False):
    """Embedding select, RMSNorm with g_out committed as Maverick commits it,
    the MoE FFN with its sigmoid table and bc_ones, the residual. Returns
    (tape, output WitnessTensor)."""
    gguf = _gguf()
    tape = Tape(CFG, silu_config=dm.SILU_CFG, lazy=True)
    E_wt = tape.commit_lazy("token_embd", dm._field_loader(gguf, "token_embd.weight"),
                            (V, D), V * D)
    x, _ind_mid, _o_last = dm.build_inputs(tape, gguf, E_wt, PROMPT, CONT, V=V, d=D)
    g = dm._field_loader(gguf, "output_norm.weight")()
    g_out = tape.commit("g_out", _bump(g) if gain_bump else g, (D,))      # build_model's g_out
    n = tape.rmsnorm(x, d=D, s=dm.S, eps_int=EPS_INT, s_out=dm.S, output_width=dm.OUTPUT_WIDTH)
    n2g = tape.hadamard_broadcast(n, g_out, SEQ=T, d=D, s_a=dm.S, s_b=dm.S, s_out=dm.S,
                                  output_width=dm.OUTPUT_WIDTH)
    t_in, t_out = dm._sigmoid_table()
    sig_tbl = tape.register_table("sigmoid", T_data=t_in, T_Y_data=t_out)
    ones_bc = tape.commit("bc_ones", torch.full((T * D,), ones, dtype=torch.uint64,
                                                device="cuda"), (T, D))    # build_model's
    ffn = dm.build_moe_ffn(tape, n2g, gguf, IL, sig_tbl, ones_bc, T=T, E=E, d=D, d_ff=DFF)
    out = x + ffn
    if router_bump:
        var = next(v for v in tape.inputs if getattr(v, "name", "") == f"L{IL}_Wr")
        honest = tape.inputs[var]
        tape.inputs[var] = lambda: _bump(honest())
    return tape, out


def _output(tape, out):
    live = tape.run_engine_pass(free_intermediates=True, keep={out.var})
    vals = to_signed(live[out.var].reshape(-1)).cpu().numpy()
    for v in list(tape.inputs):
        if getattr(v, "name", "").endswith("_mult"):
            tape.inputs[v].zero_()
    return vals


_HONEST = {}


def _honest():
    """The honest run: its enrollment, its proof's anchors, its output."""
    if not _HONEST:
        tape, out = _build()
        wc = core.WeightCommitment.from_tape(tape, CFG)
        y = _output(tape, out)
        proof = tape.prove(weight_commitment=wc)
        assert proof.root_w == wc.root
        _HONEST.update(wc=wc, root_w=proof.root_w, stmt=proof.statement_digest, y=y)
    return _HONEST


def _against_the_honest_anchors(**tamper):
    h = _honest()
    tape, out = _build(**tamper)
    y = _output(tape, out)
    proof = tape.prove(weight_commitment=h["wc"])
    acc, msg = rust_verify_anchored(tape.claims, proof, CFG, root_w=h["root_w"],
                                    stmt=h["stmt"], wc_identity=None)
    same_stmt = proof.statement_digest == h["stmt"]
    changed = int((y != h["y"]).sum())
    print(f"  [binding] {tamper or 'honest'}: Rust {'ACCEPT' if acc else 'REJECT'} under the "
          f"honest anchors; statement {'unchanged' if same_stmt else 'CHANGED'}; "
          f"{changed}/{y.size} output values differ from the honest run", flush=True)
    return acc, same_stmt, changed, msg


def test_control_the_honest_tape_accepts_under_the_honest_anchors():
    acc, same, changed, msg = _against_the_honest_anchors()
    assert acc and same and changed == 0, msg[-1500:]


def test_control_an_altered_enrolled_weight_rejects():
    acc, same, changed, _ = _against_the_honest_anchors(router_bump=True)
    assert same, "the statement must not change"
    assert not acc, "an enrolled weight altered after enrollment was ACCEPTED"


def test_bc_ones_two_rejects_under_the_honest_anchors():
    acc, same, changed, _ = _against_the_honest_anchors(ones=2)
    assert same and changed > 0, "the tamper changed nothing; the probe is void"
    assert not acc, ("bc_ones = 2 ACCEPTED under the honest weight root and statement "
                     f"digest while {changed} output values differ: bc_ones is unbound")


def test_an_altered_gain_rejects_under_the_honest_anchors():
    acc, same, changed, _ = _against_the_honest_anchors(gain_bump=True)
    assert same and changed > 0, "the tamper changed nothing; the probe is void"
    assert not acc, ("g_out altered in one channel ACCEPTED under the honest weight root and "
                     f"statement digest while {changed} output values differ: the gain is "
                     "unbound")
