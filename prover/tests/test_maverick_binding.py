"""Maverick's model binding through the driver's own construction (GPU):
demo_maverick_full.build_model on a toy GGUF with Maverick's tensor names and
attention shapes (40 query heads, 8 KV heads, 128 dimensions) and a small d.

Before the repair (2026-10-06) build_model committed every RMSNorm gain and
g_out as a plain input, outside the enrolled weight block, and committed
`bc_ones` (the operand freivalds_combine uses to spread the sigmoid gate
over the routed input) with no constraint on its values; a proof with an
altered gain or bc_ones = 2 was ACCEPTED under the honest weight root
(analysis/b200-session-10-archive.md, the binding probe). Now every gain is
a weight and every bc_ones entry is pinned to 1, outside the weight root.

Each case builds the tape with build_model, enrolls it with the honest run's
seed (so equal weight blocks give equal roots), proves it, and asks the Rust
verifier under the HONEST weight root; the statement digest is the run's
own, since a tampered witness moves the public bound Sz. The verdict must be
explicit (tests/_rust_verify.run_verify_proof). Plain test functions, for
tests/run_tests.py."""
import pathlib
import sys
import tempfile

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE, _HERE.parents[1] / "demo"):
    sys.path.insert(0, str(_p))

import _uint64_compat  # noqa: E402,F401
import torch           # noqa: E402

import core                                   # noqa: E402
import demo_maverick_full as dm               # noqa: E402
from _rust_verify import rust_verify_anchored  # noqa: E402
from claims import LinCombClaim               # noqa: E402

CFG = core.LigeroConfig(ELL=64, K_DEG=64, N_LIG=256, T_QUERIES=4)   # ELL >= d: one-row gain tables
V, D, DFF_DENSE, DFF, E = 32, 64, 64, 64, 4
PROMPT, CONT = [3, 17], [9, 22]
SEED = b"maverick-binding-test-seed-32byt"
H, HKV, DH = dm.H, dm.HKV, dm.DH
_DIR = []


def _gguf():
    """Two layers as Maverick's first two (dense, then MoE), Q8_0 experts,
    F32 elsewhere (test_gguf_loader's mix)."""
    if _DIR:
        return _DIR[1]
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType as Q
    from gguf.quants import quantize
    rng = np.random.default_rng(9)
    n = lambda std, *s: rng.normal(0.0, std, s).astype(np.float32)
    gain = lambda: (1.0 + rng.normal(0.0, 0.1, D)).astype(np.float32)
    t = {"token_embd.weight": n(0.5, V, D), "output_norm.weight": gain(),
         "output.weight": n(D ** -0.5, V, D)}
    for il in range(2):
        p = f"blk.{il}."
        t.update({p + "attn_q.weight": n(D ** -0.5, H * DH, D),
                  p + "attn_k.weight": n(D ** -0.5, HKV * DH, D),
                  p + "attn_v.weight": n(D ** -0.5, HKV * DH, D),
                  p + "attn_output.weight": n((H * DH) ** -0.5, D, H * DH),
                  p + "attn_norm.weight": gain(), p + "ffn_norm.weight": gain()})
    t.update({"blk.0.ffn_gate.weight": n(D ** -0.5, DFF_DENSE, D),
              "blk.0.ffn_up.weight": n(D ** -0.5, DFF_DENSE, D),
              "blk.0.ffn_down.weight": n(DFF_DENSE ** -0.5, D, DFF_DENSE),
              "blk.1.ffn_gate_inp.weight": n(D ** -0.5, E, D),
              "blk.1.ffn_gate_exps.weight": n(D ** -0.5, E, DFF, D),
              "blk.1.ffn_up_exps.weight": n(D ** -0.5, E, DFF, D),
              "blk.1.ffn_down_exps.weight": n(DFF ** -0.5, E, D, DFF),
              "blk.1.ffn_gate_shexp.weight": n(D ** -0.5, DFF, D),
              "blk.1.ffn_up_shexp.weight": n(D ** -0.5, DFF, D),
              "blk.1.ffn_down_shexp.weight": n(DFF ** -0.5, D, DFF)})
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


def _var(tape, name):
    return next(v for v in tape.inputs if getattr(v, "name", "") == name)


def _bumped(load, by=410):
    """The loader's tensor with entry 3 raised by `by` (~0.1 at scale 4096)."""
    def run():
        u = load().contiguous().view(torch.int64).clone()
        u[3] = u[3] + by
        return u.view(torch.uint64)
    run.provenance = getattr(load, "provenance", None)
    return run


def _build(tamper=None):
    tape = dm.Tape(CFG, silu_config=dm.SILU_CFG, lazy=True)
    logits, Sz, handles, sum_pos = dm.build_model(tape, _gguf(), PROMPT, CONT, V=V, d=D,
                                                  n_layers=2, E=E, d_ff=DFF)
    if tamper == "bc_ones":
        v = _var(tape, "bc_ones")
        tape.inputs[v] = torch.full((v.length,), 2, dtype=torch.uint64, device="cuda")
    elif tamper in ("g_out", "L1_gF"):
        v = _var(tape, tamper)
        tape.inputs[v] = _bumped(tape.inputs[v])
    return tape, Sz, handles


def _prove(tape, Sz, handles):
    """Enroll with the honest seed, discover Sz, prove."""
    wc = core.WeightCommitment.from_tape(tape, CFG, master_seed=SEED)
    live = tape.run_engine_pass(free_intermediates=True, keep={Sz.var})
    handles["reveal_pin"].public_rhs = int(live[Sz.var].cpu().reshape(-1)[0])
    for v in list(tape.inputs):
        if getattr(v, "name", "").endswith("_mult"):
            tape.inputs[v].zero_()
    del live
    return wc, tape.prove(weight_commitment=wc)


_HONEST = {}


def _honest():
    if not _HONEST:
        tape, Sz, handles = _build()
        wc, proof = _prove(tape, Sz, handles)
        assert proof.root_w == wc.root
        _HONEST.update(root_w=wc.root, stmt=proof.statement_digest,
                       Sz=handles["reveal_pin"].public_rhs)
    return _HONEST


def _against_the_honest_root(tamper):
    h = _honest()
    tape, Sz, handles = _build(tamper)
    wc, proof = _prove(tape, Sz, handles)
    v = rust_verify_anchored(tape.claims, proof, CFG, root_w=h["root_w"],
                             stmt=proof.statement_digest, wc_identity=None)
    same_root = wc.root == h["root_w"]
    print(f"  [binding] {tamper or 'honest rebuild'}: Rust {v['verdict']} (exit "
          f"{v['returncode']}) under the honest weight root; its own root "
          f"{'equals' if same_root else 'differs from'} the honest one; Sz "
          f"{handles['reveal_pin'].public_rhs} vs honest {h['Sz']}; failed checks "
          f"{[c for c in v['checks'] if c.startswith('[XX ]')][:3]}", flush=True)
    return v, same_root


def test_build_model_enrolls_every_gain_and_pins_bc_ones():
    tape, _, _ = _build()
    gains = [v for v in tape.inputs if getattr(v, "name", "") in
             ("L0_gA", "L0_gF", "L1_gA", "L1_gF", "g_out")]
    assert len(gains) == 5 and all(v.persistent for v in gains), \
        [(v.name, v.persistent) for v in gains]
    ones = _var(tape, "bc_ones")
    assert not ones.persistent, "bc_ones must stay out of the weight root"
    pins = [c for c in tape.claims if isinstance(c, LinCombClaim) and ones in c.xs]
    assert len(pins) == 1 and pins[0].xs == [ones] and pins[0].coefs == [1] \
        and pins[0].rhs == [1], pins


def test_an_honest_rebuild_accepts_under_the_honest_root():
    v, same_root = _against_the_honest_root(None)
    assert same_root and v["verdict"] == "ACCEPT", v["output"][-1500:]


def test_bc_ones_two_rejects_under_the_honest_root():
    v, same_root = _against_the_honest_root("bc_ones")
    assert same_root, "bc_ones is not a weight: the root must not move"
    assert v["verdict"] == "REJECT", "bc_ones = 2 ACCEPTED: the pin does not bind it"


def test_an_altered_g_out_rejects_under_the_honest_root():
    v, same_root = _against_the_honest_root("g_out")
    assert not same_root, "g_out altered and the weight root did not move: not enrolled"
    assert v["verdict"] == "REJECT", "an altered g_out ACCEPTED under the honest root"


def test_an_altered_layer_gain_rejects_under_the_honest_root():
    v, same_root = _against_the_honest_root("L1_gF")
    assert not same_root, "L1_gF altered and the weight root did not move: not enrolled"
    assert v["verdict"] == "REJECT", "an altered L1_gF ACCEPTED under the honest root"
