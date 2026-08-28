"""The real Tape -> 5-of-49 runtime adapter uses one engine pass."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import admission
import compute_fns
import core
import pytest
import torch
from cuda_primitives import hash_columns_streamed
from rescale_claim import rescale
from routing_claim import freivalds_combine
from sampled_claim_runtime import ClaimWindowAudit, _finish_tensor_digest, _tensor_digest_parts
from tape import Tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


def _case():
    tape = Tape(CFG, lazy=True)
    a = tape.commit(
        "a", torch.arange(5, dtype=torch.int64, device="cuda").to(torch.uint64),
        (5,))
    b = tape.commit(
        "b", (10 + torch.arange(3, dtype=torch.int64, device="cuda")).to(
            torch.uint64), (3,))
    dst = tape.concat([a, b], (8,))
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1)
    return tape, dst, audit


def _matmul_case(secret=b"v" * 32):
    tape = Tape(CFG, lazy=True)

    def u64(values):
        return torch.tensor(
            values, dtype=torch.int64, device="cuda").to(torch.uint64)

    a = tape.commit("mm_a", u64([1, 2, 3, 4, 5, 6]), (2, 3))
    b = tape.commit("mm_b", u64([7, 8, 9, 10, 11, 12]), (3, 2))
    dst = tape.matmul(a, b)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, secret, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000)
    return tape, dst, audit


def _multihead_matmul_case(*, transpose_b):
    tape = Tape(CFG, lazy=True)

    def sequence(name, length, shape, offset):
        values = torch.arange(
            offset, offset + length, dtype=torch.int64,
            device="cuda").to(torch.uint64)
        return tape.commit(name, values, shape)

    # m=2, H=2, K=2, n=3. Tape layouts match production attention:
    # A=(m,H*K); B=(n,H*K) for QK^T or (K,H*n) for AV.
    a = sequence("mh_a", 8, (2, 4), 1)
    b_shape = (3, 4) if transpose_b else (2, 6)
    b = sequence("mh_b", 12, b_shape, 11)
    dst = tape.matmul(
        a, b, transpose_b=transpose_b, heads=2, head_dim=2)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000)
    return tape, dst, audit


def _relation_case(kind, length=8):
    tape = Tape(CFG, lazy=True)
    a = tape.commit(
        f"{kind}_a", torch.arange(
            1, length + 1, dtype=torch.int64,
            device="cuda").to(torch.uint64), (length,))
    b = tape.commit(
        f"{kind}_b", torch.arange(
            11, length + 11, dtype=torch.int64,
            device="cuda").to(torch.uint64), (length,))
    dst = tape.add(a, b) if kind == "add" else tape.hadamard(a, b)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000)
    return tape, dst, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_real_tape_observer_accepts_without_replay():
    tape, dst, audit = _case()
    live = tape.run_engine_pass(
        free_intermediates=True, keep={dst.var}, observer=audit)
    result = audit.finish()

    assert live[dst.var].to(torch.int64).cpu().tolist() == [0, 1, 2, 3, 4, 10, 11, 12]
    assert result["accepted"] is True
    assert result["claims"] == result["selected"] == 1
    assert len(result["c0_root"]) == 64
    assert result["local_argument"] == "sumcheck"
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0
    assert result["cryptographic_local_proofs"] is False
    assert result["rs_openings_materialized"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_c0_is_independent_of_window_batching():
    tape, dst, per_claim = _case()
    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=per_claim)
    first = per_claim.finish()

    one_window = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=len(tape.claims),
        sample_per_window=len(tape.claims))
    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=one_window)
    second = one_window.finish()

    assert first["c0_root"] == second["c0_root"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_window_rs_commit_opens_and_binds_selected_wire():
    tape, dst, _raw = _case()
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        enable_rs_binding=True, rs_columns=4,
        rs_ell=3, rs_k_deg=8, rs_n_lig=32,
        heartbeat_every=1000)

    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["rs_openings_materialized"] is True
    assert result["binding"] == "rs-window+striped-blake3 sumcheck runtime"
    assert result["rs_commit_s"] > 0
    assert result["rs_open_s"] > 0
    assert result["rs_verify_s"] > 0
    assert result["rs_geometry"] == {"ELL": 3, "K_DEG": 8, "N_LIG": 32}
    assert result["rs_opened_values"] == result["rs_rows"] * 4
    assert len(audit.rs_roots) == 1
    assert len(result["local_receipts"]) == 1
    assert len(result["rs_column_samples"]) == 1
    assert result["rs_column_samples"][0]["local_receipts"] == 1
    assert len(set(result["rs_column_samples"][0]["columns"])) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_window_rs_rejects_invalid_merkle_opening(monkeypatch):
    tape, dst, _raw = _case()
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        enable_rs_binding=True, rs_columns=4,
        rs_ell=3, rs_k_deg=8, rs_n_lig=32,
        heartbeat_every=1000)
    monkeypatch.setattr(core, "merkle_verify",
                        lambda _leaf, _path, _root: False)

    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("RS Merkle opening failed" in failure
               for failure in result["failures"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_selected_matmul_materializes_freivalds_proof():
    tape, dst, audit = _matmul_case()
    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "freivalds"
    assert result["selected_proof_family_counts"] == {"freivalds": 1}
    assert result["materialized_local_proof_counts"] == {"freivalds": 1}
    assert result["materialized_local_proofs"] == 1
    assert result["exact_fallbacks"] == 0
    assert result["local_proof_bytes"] == (3 + 2) * 8
    assert len(result["local_proof_digests"]) == 1
    assert len(result["local_proof_digests"][0]["digest"]) == 64
    assert result["local_receipts"] == [{
        "claim": 0,
        "proof_digest": result["local_proof_digests"][0]["digest"],
        "digest": result["local_receipts"][0]["digest"],
    }]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_selected_bad_matmul_fails_freivalds_contraction():
    tape, dst, audit = _matmul_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = outs[dst.var].clone()
        bad[0] = bad[0].view(torch.int64).add(1).view(torch.uint64)
        outs[dst.var] = bad
        live[dst.var] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert result["materialized_local_proofs"] == 1
    assert any("Freivalds contraction failed" in failure
               for failure in result["failures"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_freivalds_proof_challenge_depends_on_verifier_secret():
    tape_a, dst_a, audit_a = _matmul_case(b"a" * 32)
    tape_a.run_engine_pass(free_intermediates=True, keep={dst_a.var},
                           observer=audit_a)
    first = audit_a.finish()

    tape_b, dst_b, audit_b = _matmul_case(b"b" * 32)
    tape_b.run_engine_pass(free_intermediates=True, keep={dst_b.var},
                           observer=audit_b)
    second = audit_b.finish()

    assert first["c0_root"] == second["c0_root"]
    assert (first["local_proof_digests"][0]["digest"]
            != second["local_proof_digests"][0]["digest"])


@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_multihead_attention_matmul_freivalds(transpose_b):
    tape, dst, audit = _multihead_matmul_case(transpose_b=transpose_b)
    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "freivalds"
    # H*K + H*m field elements.
    assert result["local_proof_bytes"] == (4 + 4) * 8


@pytest.mark.parametrize("kind", ["add", "hadamard"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_selected_relation_materializes_sumcheck(kind):
    tape, dst, audit = _relation_case(kind)
    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "sumcheck"
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0
    assert result["local_proof_bytes"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_eq_weighted_sumcheck_detects_cancelling_add_errors():
    tape, dst, audit = _relation_case("add")

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = outs[dst.var].clone().view(torch.int64)
        # The two errors sum to zero. An unweighted aggregate relation would
        # accept; the post-commit random eq polynomial must reject it.
        bad[0] += 1
        bad[1] -= 1
        bad = bad.view(torch.uint64)
        outs[dst.var] = bad
        live[dst.var] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert result["materialized_local_proofs"] == 1
    assert any("sumcheck relation claim is not zero" in failure
               for failure in result["failures"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_large_relation_uses_gpu_sumcheck_fast_path():
    from layergkr import sumcheck as sc

    tape, dst, audit = _relation_case("add", length=1 << 15)
    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert sc._SC_GPU.get("ok") is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_lincomb_materializes_sumcheck():
    tape = Tape(CFG, lazy=True)
    values_i64 = torch.arange(1, 9, dtype=torch.int64, device="cuda")
    values = values_i64.to(torch.uint64)
    a = tape.commit("lc_a", values, (8,))
    b = tape.commit("lc_b", (values_i64 + 3).to(torch.uint64), (8,))
    rhs = (2 * values.view(torch.int64)
           + 3 * b.data.view(torch.int64)).tolist()
    tape.lincomb([a, b], [2, 3], rhs)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000)
    tape.run_engine_pass(observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_word_extraction_sumcheck_with_explicit_lookup_fallbacks():
    tape = Tape(CFG, lazy=True)
    x = tape.commit(
        "word_x", torch.tensor(
            [0, 1, 5, 10, 15], dtype=torch.uint64, device="cuda"), (5,))
    table = tape.register_table("word2", range(4))
    words = tape.word_extract(x, table, B=2, N=2)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000)
    tape.run_engine_pass(free_intermediates=True,
                         keep={word.var for word in words}, observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["claims"] == result["selected"] == 3
    assert result["materialized_local_proof_counts"] == {
        "product-tree": 2, "sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_rescale_linears_use_sumcheck_and_ranges_stay_exact():
    tape = Tape(CFG, lazy=True)
    raw = tape.commit(
        "raw", torch.tensor(
            [0, 1, 17, 31, 63], dtype=torch.uint64, device="cuda"), (5,))
    out = rescale(tape, raw, s_in=16, s_out=4, output_width=8)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000)
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "sumcheck+exact-recomputation"
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallback_counts"] == {"RescaleClaim": 1}


def _range_product_case():
    tape = Tape(CFG, lazy=True)
    x = tape.commit(
        "range_x", torch.tensor(
            [0, 3, 7, 3, 1], dtype=torch.uint64, device="cuda"), (5,))
    table = tape.register_table("range8", range(8))
    tape.range_word(x, table)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000)
    return tape, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_range_word_materializes_compact_product_tree():
    tape, audit = _range_product_case()
    claim = tape.claims[0]
    assert claim.local_indices is claim.x
    tape.run_engine_pass(observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "product-tree"
    assert result["materialized_local_proof_counts"] == {"product-tree": 1}
    assert result["local_proof_bytes"] == 16
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_range_product_tree_rejects_out_of_range_query_index():
    tape, audit = _range_product_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = live[claim.local_indices].clone()
        bad[0] = 99
        input_data[claim.local_indices] = bad
        live[claim.local_indices] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert result["materialized_local_proofs"] == 1
    assert any("product-tree" in failure or "lookup" in failure
               for failure in result["failures"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_real_tape_observer_rejects_tampered_selected_output():
    tape, dst, audit = _case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = outs[dst.var].clone()
        bad[0] = 99
        outs[dst.var] = bad
        live[dst.var] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert len(result["failures"]) == 1
    assert result["failures"][0].startswith("claim 0: sumcheck")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_foldable_boundary_wires_are_retained_without_disabling_fold(
        tmp_path, monkeypatch):
    monkeypatch.delenv("LIGERO_NO_FOLD", raising=False)
    tape = Tape(CFG, lazy=True)
    mask = tape.commit(
        "mask", torch.tensor([1, 0, 0, 0, 0, 1, 0, 0],
                             dtype=torch.uint64, device="cuda"), (2, 4))
    xs = []
    for expert in range(4):
        a = tape.commit(
            f"x{expert}a", torch.full((3,), 10 + expert,
                                      dtype=torch.uint64, device="cuda"), (3,))
        b = tape.commit(
            f"x{expert}b", torch.full((3,), 20 + expert,
                                      dtype=torch.uint64, device="cuda"), (3,))
        xs.append(tape.concat([a, b], (6,)))
    out = freivalds_combine(tape, mask, xs, T=2, E=4, F=3)
    admission.prepare(tape, CFG)
    assert compute_fns.FoldRunner(tape._deferred).is_fold(tape.claims[-1])
    progress = tmp_path / "progress.jsonl"
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=len(tape.claims),
        sample_per_window=len(tape.claims), progress_path=str(progress),
        heartbeat_every=1000)

    live = tape.run_engine_pass(
        free_intermediates=True, keep={out.var}, observer=audit)
    result = audit.finish()
    events = [json.loads(line) for line in progress.read_text().splitlines()]

    assert result["accepted"] is True
    assert result["fold_retained_peak_bytes"] > 0
    assert audit.fold_values == {}
    assert live[out.var].numel() == 6
    assert any(e["event"] == "window_complete" for e in events)
    assert events[-1]["event"] == "audit_complete"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_striped_wire_digest_matches_portable_cpu_and_detects_tamper():
    value = torch.arange(10003, dtype=torch.int64, device="cuda").to(torch.uint64)

    def finish(tensor):
        n, columns, payload = _tensor_digest_parts(tensor)
        if isinstance(payload, torch.Tensor):
            payload = hash_columns_streamed(payload).cpu().numpy().tobytes()
        return _finish_tensor_digest(n, columns, payload)

    gpu_digest = finish(value)
    assert gpu_digest == finish(value.cpu())
    tampered = value.clone()
    tampered.view(torch.int64)[917] += 1
    assert finish(tampered) != gpu_digest
