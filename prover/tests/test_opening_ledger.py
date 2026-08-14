"""The enrolled model's opening ledger, and atomic proof output (S4e).

Every proof that references one enrollment opens columns of the SAME weight
rows under the SAME padding, so what leaks is the CUMULATIVE set of distinct
columns, not the per-proof count. Each row carries K_DEG-ELL random pad
coefficients; once the union approaches that, the padding no longer hides and
the enrollment has to be refreshed. Without a ledger nothing in the system
notices — the proofs keep verifying.

The write side: a proof that runs out of disk after four hours must not leave
a truncated file that looks like a proof.
"""
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import claims as _C       # noqa: F401
import packets as _PK     # noqa: F401
from tape import Tape
from proof_dump import dump_proof, estimated_bytes, reserve_output

# A config WITH slack, so the ledger is live: K_DEG = 2*ELL.
CFG = core.LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)


def _build():
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    w1 = tape.commit("W1", _t(list(range(64))), (64,), persistent=True)
    w2 = tape.commit("W2", _t([v * 2 for v in range(64)]), (64,), persistent=True)
    tape.add(w1, w2)
    return tape


def test_ledger_records_and_persists():
    wc = core.WeightCommitment.from_tape(_build(), CFG)
    assert wc.opened_columns == []
    proof = _build().prove(weight_commitment=wc)
    wc.record_openings(proof.Q_cols)
    assert wc.opened_columns == sorted(proof.Q_cols)
    fd, path = tempfile.mkstemp(suffix=".wc")
    os.close(fd)
    try:
        wc.save(path)
        again = core.WeightCommitment.load(path)
        assert again.opened_columns == wc.opened_columns, \
            "the ledger did not survive save/load — an unsaved ledger is a " \
            "silently reusable pad"
        print(f"    ledger persists: {len(wc.opened_columns)} columns spent "
              f"of {wc.opening_budget(CFG)}")
    finally:
        os.unlink(path)


def test_exhausted_budget_refuses_before_opening():
    """The refusal must land before any weight column is produced."""
    wc = core.WeightCommitment.from_tape(_build(), CFG)
    wc.record_openings(range(wc.opening_budget(CFG) + 1))   # pretend it is spent
    try:
        _build().prove(weight_commitment=wc)
        raise AssertionError("an exhausted enrollment still produced a proof")
    except AssertionError as e:
        assert "opening budget exhausted" in str(e), e
        assert "REFRESH" in str(e), "the refusal must say what to do next"
    print(f"    budget {wc.opening_budget(CFG)} spent: proof refused, refresh "
          f"demanded")


def test_budget_scales_with_the_pad():
    class Big:
        ELL, K_DEG, N_LIG, T_QUERIES = 8192, 16384, 65536, 54
    wc = core.WeightCommitment(root=b"\x00" * 32, levels=[], m_w=1, n_lig=65536)
    budget = wc.opening_budget(Big)
    assert budget == (Big.K_DEG - Big.ELL) // 2 == 4096
    print(f"    at the production geometry: {budget} columns "
          f"(~{budget // Big.T_QUERIES} proofs at {Big.T_QUERIES} per proof)")


def test_proof_write_is_atomic_and_space_checked():
    proof = _build().prove()
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "proof.json")
    try:
        need = estimated_bytes(proof, proof.Q_cols, 0)
        assert need > 0
        dump_proof(path, None, None, proof, None, None)
        assert os.path.exists(path) and not os.path.exists(path + ".part"), \
            "the .part file must be renamed away, not left behind"
        size = os.path.getsize(path)
        assert size <= need, f"estimate {need} was below the real size {size}"
        print(f"    atomic write ok: {size} bytes, estimate {need}, no .part left")
    finally:
        for f in os.listdir(tmp):
            os.unlink(os.path.join(tmp, f))
        os.rmdir(tmp)


def test_write_refused_when_the_disk_cannot_hold_it():
    proof = _build().prove()
    import proof_dump
    real = proof_dump.estimated_bytes
    proof_dump.estimated_bytes = lambda *a, **k: 1 << 60      # ~1 EB
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "proof.json")
    try:
        try:
            dump_proof(path, None, None, proof, None, None)
            raise AssertionError("wrote a proof that cannot fit on the disk")
        except OSError as e:
            assert "refusing to write" in str(e), e
        assert not os.path.exists(path) and not os.path.exists(path + ".part")
        print("    proof larger than the free space: refused, nothing written")
    finally:
        proof_dump.estimated_bytes = real
        for f in os.listdir(tmp):
            os.unlink(os.path.join(tmp, f))
        os.rmdir(tmp)


def test_compact_wire_and_preproof_reservation():
    """Production encoding is smaller and consumes a pre-reserved inode."""
    import base64, json
    proof = _build().prove()
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "proof.json")
    try:
        compact_need = estimated_bytes(
            proof, proof.Q_cols, 0, u64_encoding="u64le-base64")
        decimal_need = estimated_bytes(proof, proof.Q_cols, 0)
        assert compact_need < decimal_need
        part = reserve_output(path, compact_need + (1 << 20))
        dump_proof(path, None, None, proof, None, None,
                   u64_encoding="u64le-base64", reserved_part=part)
        doc = json.load(open(path))
        wire = doc["proof"]["q_irs"]
        assert wire.startswith("u64le:")
        raw = base64.b64decode(wire[6:])
        got = [int.from_bytes(raw[i:i + 8], "little")
               for i in range(0, len(raw), 8)]
        want = [int(x) for x in proof.q_irs.cpu().tolist()]
        assert got == want, "compact wire changed field elements"
        assert not os.path.exists(part)
        print(f"    compact wire: {os.path.getsize(path)} bytes, exact u64 roundtrip")
    finally:
        for f in os.listdir(tmp):
            os.unlink(os.path.join(tmp, f))
        os.rmdir(tmp)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== opening-ledger: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
