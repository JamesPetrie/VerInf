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
from claims import SILU_TOY
from cuda_primitives import gl_add, gl_sub, hash_columns_streamed
from max_claim import max_gap
from rescale_claim import rescale
from routed_projected import routed_projected_matmul
from routing_claim import RoutingClaim, freivalds_combine
from sampled_claim_runtime import ClaimWindowAudit, _finish_tensor_digest, _tensor_digest_parts
from tape import Tape
from ui_claim import info_finalize

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
        expected_claims=0, window_size=1, sample_per_window=1, prototype=True)
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
        heartbeat_every=1000, prototype=True)
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
        heartbeat_every=1000, prototype=True)
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
        heartbeat_every=1000, prototype=True)
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
    assert result["cryptographic_local_proofs"] is True
    assert result["rs_openings_materialized"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_progress_records_each_local_proof_timing(tmp_path):
    tape = Tape(CFG, lazy=True)
    a = tape.commit(
        "logged_a", torch.arange(
            8, dtype=torch.int64, device="cuda").to(torch.uint64), (8,))
    b = tape.commit(
        "logged_b", torch.arange(
            8, 16, dtype=torch.int64, device="cuda").to(torch.uint64), (8,))
    out = tape.add(a, b)
    admission.prepare(tape, CFG)
    progress = tmp_path / "progress.jsonl"
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, progress_path=progress, prototype=True)
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    events = [json.loads(line) for line in progress.read_text().splitlines()]
    starts = [event for event in events
              if event["event"] == "local_proof_start"]
    completes = [event for event in events
                 if event["event"] == "local_proof_complete"]
    assert result["accepted"] is True
    assert len(starts) == len(completes) == 1
    assert starts[0]["claim_type"] == completes[0]["claim_type"] == "AddClaim"
    assert completes[0]["family"] == "sumcheck"
    assert completes[0]["proof_s"] >= 0
    assert completes[0]["proof_bytes"] > 0
    assert completes[0]["exact_fallbacks"] == 0
    assert completes[0]["accepted"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_c0_is_independent_of_window_batching():
    tape, dst, per_claim = _case()
    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=per_claim)
    first = per_claim.finish()

    one_window = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=len(tape.claims),
        sample_per_window=len(tape.claims), prototype=True)
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
        heartbeat_every=1000, prototype=True)

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
        heartbeat_every=1000, prototype=True)
    monkeypatch.setattr(core, "merkle_verify",
                        lambda _leaf, _path, _root, _index, _n_leaves: False)

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
        heartbeat_every=1000, prototype=True)
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
        heartbeat_every=1000, prototype=True)
    tape.run_engine_pass(free_intermediates=True,
                         keep={word.var for word in words}, observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["claims"] == result["selected"] == 3
    assert result["materialized_local_proof_counts"] == {
        "product-tree": 2, "sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_rescale_materializes_sumcheck_and_range_products():
    tape = Tape(CFG, lazy=True)
    raw = tape.commit(
        "raw", torch.tensor(
            [0, 1, 17, 31, 63], dtype=torch.uint64, device="cuda"), (5,))
    out = rescale(tape, raw, s_in=16, s_out=4, output_width=8)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "sumcheck"
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0


def _fused_rounding_case(kind):
    tape = Tape(CFG, lazy=True)

    def u64(values):
        return torch.tensor(
            values, dtype=torch.int64, device="cuda").to(torch.uint64)

    if kind == "matmul":
        a = tape.commit("rounded_mm_a", u64([1, 2, 3, 4, 5, 6]), (2, 3))
        b = tape.commit("rounded_mm_b", u64([7, 8, 9, 10, 11, 12]), (3, 2))
        out = tape.matmul(
            a, b, s_a=2, s_b=2, s_out=1, output_width=8)
    else:
        a = tape.commit("rounded_h_a", u64([1, 2, 3, 4]), (4,))
        b = tape.commit("rounded_h_b", u64([5, 6, 7, 8]), (4,))
        out = tape.hadamard(
            a, b, s_a=2, s_b=2, s_out=1, output_width=8)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, out, audit


@pytest.mark.parametrize("kind", ["matmul", "hadamard"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_fused_rounding_has_no_exact_fallback(kind):
    tape, out, audit = _fused_rounding_case(kind)
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["exact_fallbacks"] == 0
    assert result["materialized_local_proofs"] == 1


@pytest.mark.parametrize("kind", ["matmul", "hadamard"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_fused_rounding_rejects_out_of_range_low_with_linears_intact(kind):
    tape, _out, audit = _fused_rounding_case(kind)

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        low_var = claim.C_low if kind == "matmul" else claim.c_low
        out_var = claim.C if kind == "matmul" else claim.c
        shifted_var = (claim.C_shifted if kind == "matmul"
                       else claim.c_shifted)
        one = torch.ones(1, dtype=torch.uint64, device="cuda")
        step = torch.full((1,), 1 << claim.rescale_bits,
                          dtype=torch.uint64, device="cuda")
        low = outs[low_var].clone()
        out = outs[out_var].clone()
        shifted = outs[shifted_var].clone()
        low[:1] = gl_add(low[:1], step)
        out[:1] = gl_sub(out[:1], one)
        shifted[:1] = gl_sub(shifted[:1], one)
        for var, value in ((low_var, low), (out_var, out),
                           (shifted_var, shifted)):
            outs[var] = value
            live[var] = value
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("rounding low value" in failure
               for failure in result["failures"])


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
        heartbeat_every=1000, prototype=True)
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


def _embedding_product_case():
    tape = Tape(CFG, lazy=True)
    embedding = tape.commit(
        "embedding", torch.arange(
            1, 13, dtype=torch.int64,
            device="cuda").to(torch.uint64), (4, 3))
    out = tape.embed(embedding, token_ids=[2, 0, 2], d=3)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, out, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_embedding_lookup_materializes_product_tree():
    tape, out, audit = _embedding_product_case()
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "product-tree"
    assert result["materialized_local_proof_counts"] == {"product-tree": 1}
    assert result["local_proof_bytes"] == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_embedding_product_tree_rejects_positional_permutation():
    tape, out, audit = _embedding_product_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = outs[out.var].clone()
        first = bad[0].clone()
        bad[0] = bad[1]
        bad[1] = first
        outs[out.var] = bad
        live[out.var] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("embedding product-tree roots differ" in failure
               for failure in result["failures"])


def _paired_product_case():
    tape = Tape(CFG, lazy=True)
    x = tape.commit(
        "paired_x", torch.tensor(
            [2, 0, 3, 2], dtype=torch.uint64, device="cuda"), (4,))
    table = tape.register_table(
        "paired4", T_data=range(4), T_Y_data=[11, 22, 33, 44])
    y = tape.paired_tlookup(x, table)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, y, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_paired_lookup_materializes_product_tree():
    tape, y, audit = _paired_product_case()
    tape.run_engine_pass(free_intermediates=True, keep={y.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "product-tree"
    assert result["materialized_local_proof_counts"] == {"product-tree": 1}
    assert result["local_proof_bytes"] == 16
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_paired_lookup_product_tree_rejects_wrong_value():
    tape, y, audit = _paired_product_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = outs[y.var].clone()
        bad[0] = 99
        outs[y.var] = bad
        live[y.var] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("paired lookup product-tree roots differ" in failure
               for failure in result["failures"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_paired_lookup_product_tree_rejects_out_of_range_key():
    tape, _y, audit = _paired_product_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = live[claim.x].clone()
        bad[0] = 99
        input_data[claim.x] = bad
        live[claim.x] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("outside the public table" in failure
               for failure in result["failures"])


def _freivalds_combine_case():
    tape = Tape(CFG, lazy=True)
    mask = tape.commit(
        "fc_mask", torch.tensor(
            [1, 0, 0, 0, 0, 1], dtype=torch.uint64,
            device="cuda"), (2, 3))
    xs = []
    for expert in range(3):
        values = torch.tensor(
            [10 + expert, 20 + expert, 30 + expert, 40 + expert],
            dtype=torch.uint64, device="cuda")
        xs.append(tape.commit(f"fc_x{expert}", values, (2, 2)))
    out = freivalds_combine(tape, mask, xs, T=2, E=3, F=2)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, out, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_freivalds_combine_materializes_projection_proof():
    tape, out, audit = _freivalds_combine_case()
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "freivalds"
    assert result["materialized_local_proof_counts"] == {"freivalds": 1}
    assert result["local_proof_bytes"] == (3 * 2 + 2) * 8
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_freivalds_combine_rejects_wrong_output():
    tape, out, audit = _freivalds_combine_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = outs[out.var].clone()
        bad[0] = 99
        outs[out.var] = bad
        live[out.var] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("combine Freivalds contraction failed" in failure
               for failure in result["failures"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_freivalds_combine_rejects_wrong_route_mask():
    tape, _out, audit = _freivalds_combine_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = live[claim.m].clone()
        bad[0] = 0
        bad[1] = 1
        input_data[claim.m] = bad
        live[claim.m] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("combine Freivalds contraction failed" in failure
               for failure in result["failures"])


def _routed_matmul_case():
    tape = Tape(CFG, lazy=True)
    x = tape.commit(
        "routed_x", torch.tensor(
            [1, 2, 3, 4, 5, 6], dtype=torch.uint64,
            device="cuda"), (3, 2))
    mask = tape.commit(
        "routed_mask", torch.tensor(
            [1, 0, 0, 1, 1, 0], dtype=torch.uint64,
            device="cuda"), (3, 2))
    weights = []
    for expert in range(2):
        values = torch.arange(
            1 + expert * 4, 5 + expert * 4,
            dtype=torch.int64, device="cuda").to(torch.uint64)
        weights.append(tape.commit(f"routed_w{expert}", values, (2, 2)))
    out = routed_projected_matmul(
        tape, x, mask, weights, T=3, K=2, J=2, E=2)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, out, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_routed_matmul_materializes_freivalds_seam():
    tape, out, audit = _routed_matmul_case()
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "freivalds"
    assert result["materialized_local_proof_counts"] == {"freivalds": 1}
    assert result["local_proof_bytes"] == (2 * 2 + 3) * 8
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_routed_matmul_freivalds_rejects_wrong_output():
    tape, out, audit = _routed_matmul_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = outs[out.var].clone()
        bad[0] = 99
        outs[out.var] = bad
        live[out.var] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("routed Freivalds contraction failed" in failure
               for failure in result["failures"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_routed_matmul_freivalds_rejects_wrong_route():
    tape, _out, audit = _routed_matmul_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = live[claim.M].clone()
        bad[0] = 0
        bad[1] = 1
        input_data[claim.M] = bad
        live[claim.M] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("routed Freivalds contraction failed" in failure
               for failure in result["failures"])


def _rope_sumcheck_case(heads=1):
    tape = Tape(CFG, lazy=True)
    length = 2 * heads * 4
    x = tape.commit(
        "rope_x", torch.arange(
            1, length + 1, dtype=torch.int64,
            device="cuda").to(torch.uint64), (2, heads * 4))
    out = tape.rope(
        x, SEQ=2, d_h=4, heads=heads, s_x=4, s_out=4,
        output_width=8)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, out, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_rope_materializes_rotation_rescale_and_ranges():
    tape, out, audit = _rope_sumcheck_case()
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "sumcheck"
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_rope_multihead_expanded_coefficients_are_contiguous():
    tape, out, audit = _rope_sumcheck_case(heads=3)
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_rope_sumcheck_rejects_wrong_rotated_output():
    tape, out, audit = _rope_sumcheck_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = outs[out.var].clone()
        bad[0] = 99
        outs[out.var] = bad
        live[out.var] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("sumcheck" in failure for failure in result["failures"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_rope_product_tree_rejects_out_of_range_low_with_linears_intact():
    tape, _out, audit = _rope_sumcheck_case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        one = torch.ones(1, dtype=torch.uint64, device="cuda")
        step = torch.full((1,), 1 << claim.rescale_bits,
                          dtype=torch.uint64, device="cuda")
        low = outs[claim.x_rot_low].clone()
        rotated = outs[claim.x_rot].clone()
        shifted = outs[claim.x_rot_shifted].clone()
        low[:1] = gl_add(low[:1], step)
        rotated[:1] = gl_sub(rotated[:1], one)
        shifted[:1] = gl_sub(shifted[:1], one)
        for var, value in ((claim.x_rot_low, low),
                           (claim.x_rot, rotated),
                           (claim.x_rot_shifted, shifted)):
            outs[var] = value
            live[var] = value
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("RoPE low range index" in failure
               for failure in result["failures"])


def _routing_sumcheck_case():
    tape = Tape(CFG, lazy=True)
    r = tape.commit(
        "routing_r", torch.tensor(
            [40, 20, 30, 10, 15, 45, 25, 5], dtype=torch.uint64,
            device="cuda"), (2, 4))
    m = tape._alloc("routing_m", 8)
    rt = tape._alloc("routing_rt", 8)
    mrt = tape._alloc("routing_mrt", 8)
    rstar = tape._alloc("routing_rstar", 2)
    gap = tape._alloc("routing_gap", 8)
    r_chosen = tape._alloc("routing_chosen", 2)
    claim = RoutingClaim(
        r=r.var, m=m, rt=rt, mrt=mrt, rstar=rstar, gap=gap,
        r_chosen=r_chosen, T=2, E=4, L_bits=2)
    tape._process_claim(claim, [r.var])
    tape.claims.append(claim)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, claim, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_routing_materializes_all_relation_sumcheck():
    tape, _claim, audit = _routing_sumcheck_case()
    tape.run_engine_pass(observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "sumcheck"
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
@pytest.mark.parametrize("field", ["m", "rt", "mrt", "rstar", "gap",
                                    "r_chosen"])
def test_routing_sumcheck_rejects_tampered_relation(field):
    tape, claim, audit = _routing_sumcheck_case()

    def tamper_then_observe(index, op, input_vars, input_data, outs, live):
        var = getattr(claim, field)
        bad = outs[var].clone()
        bad[0] = 99
        outs[var] = bad
        live[var] = bad
        audit(index, op, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("sumcheck" in failure for failure in result["failures"])


def _silu_sumcheck_case():
    tape = Tape(CFG, silu_config=SILU_TOY, lazy=True)
    x = tape.commit(
        "silu_x", torch.tensor(
            [0, 1, 2, 3], dtype=torch.uint64, device="cuda"), (4,))
    out = tape.silu(x)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, out, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_silu_materializes_all_relations_and_lookups():
    tape, out, audit = _silu_sumcheck_case()
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "sumcheck"
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
@pytest.mark.parametrize("field", ["output", "sign", "magnitude", "is_high",
                                    "key", "y"])
def test_silu_sumcheck_rejects_tampered_relation(field):
    tape, _out, audit = _silu_sumcheck_case()
    claim = tape.claims[0]

    def tamper_then_observe(index, op, input_vars, input_data, outs, live):
        var = getattr(claim, field)
        bad = outs[var].clone()
        bad[0] = 99
        outs[var] = bad
        live[var] = bad
        audit(index, op, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("sumcheck" in failure or "SiLU" in failure
               for failure in result["failures"])


def _rmsnorm_sumcheck_case():
    tape = Tape(CFG, lazy=True)
    x = tape.commit(
        "rms_x", torch.tensor(
            [1, 2, 3, 4, 2, 3, 4, 5], dtype=torch.uint64,
            device="cuda"), (2, 4))
    out = tape.rmsnorm(
        x, d=4, s=4, eps_int=1, s_out=4, output_width=8)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, out, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_rmsnorm_materializes_bracket_output_and_ranges():
    tape, out, audit = _rmsnorm_sumcheck_case()
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "sumcheck"
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
@pytest.mark.parametrize("field", ["output", "X_sq", "S_total", "y", "q1",
                                    "s_lo", "lo_H"])
def test_rmsnorm_sumcheck_rejects_tampered_relation(field):
    tape, _out, audit = _rmsnorm_sumcheck_case()
    claim = tape.claims[0]

    def tamper_then_observe(index, op, input_vars, input_data, outs, live):
        target = getattr(claim, field)
        var = target[0] if isinstance(target, list) else target
        bad = outs[var].clone()
        bad[0] = 99
        outs[var] = bad
        live[var] = bad
        audit(index, op, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("sumcheck" in failure or "RMSNorm" in failure
               for failure in result["failures"])


def _softmax_sumcheck_case():
    tape = Tape(CFG, lazy=True)
    x = tape.commit(
        "softmax_x", torch.tensor(
            [2, 1, 3, 1], dtype=torch.uint64, device="cuda"), (2, 2))
    out = tape.softmax(
        x, M=2, s_x=4, s_c=4, s_y=4, Z_max=16,
        saturate=True, Z_high_width=4, aux_chunk_width=8,
        causal=True, heads=1)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, out, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_softmax_materializes_causal_saturating_relations_and_lookups():
    tape, out, audit = _softmax_sumcheck_case()
    tape.run_engine_pass(free_intermediates=True, keep={out.var},
                         observer=audit)
    result = audit.finish()

    assert result["accepted"] is True
    assert result["local_argument"] == "sumcheck"
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
@pytest.mark.parametrize("field", ["y_A", "c2", "z", "s1", "r_lo",
                                    "is_high", "y_A_raw"])
def test_softmax_sumcheck_rejects_tampered_relation(field):
    tape, _out, audit = _softmax_sumcheck_case()
    claim = tape.claims[0]

    def tamper_then_observe(index, op, input_vars, input_data, outs, live):
        var = getattr(claim, field)
        bad = outs[var].clone()
        bad[0] = 99
        outs[var] = bad
        live[var] = bad
        audit(index, op, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert any("sumcheck" in failure or "Softmax" in failure
               for failure in result["failures"])


def _max_sumcheck_case():
    tape = Tape(CFG, lazy=True)
    logits = tape.commit(
        "max_logits", torch.tensor(
            [40, 20, 30, 10, 15, 45, 25, 5], dtype=torch.uint64,
            device="cuda"), (2, 4))
    max_gap(tape, logits, [2, 0], T=2, V=4, gap_max=64)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, tape.claims[0], audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_max_materializes_all_relations_and_gap_range():
    tape, _claim, audit = _max_sumcheck_case()
    tape.run_engine_pass(observer=audit)
    result = audit.finish()
    assert result["accepted"] is True
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
@pytest.mark.parametrize("field", ["A", "Al", "vstar", "gap", "Ogap"])
def test_max_sumcheck_rejects_tampered_relation(field):
    tape, claim, audit = _max_sumcheck_case()

    def tamper_then_observe(index, op, input_vars, input_data, outs, live):
        var = getattr(claim, field)
        bad = outs[var].clone()
        bad[0] = 99
        outs[var] = bad
        live[var] = bad
        audit(index, op, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()
    assert result["accepted"] is False


def _info_sumcheck_case():
    tape = Tape(CFG, lazy=True)
    e = tape.commit(
        "info_e", torch.tensor(
            [4, 3, 2, 3, 2, 1], dtype=torch.uint64,
            device="cuda"), (2, 3))
    gap_o2 = tape.commit(
        "info_gap2", torch.tensor(
            [7, 13], dtype=torch.uint64, device="cuda"), (2,))
    info_finalize(
        tape, e, gap_o2, T=2, V=3, k=16, d_max=3 * 4096,
        s_y=4096, s_b=16, K=128)
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    return tape, tape.claims[0], audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_info_finalize_materializes_relations_and_ranges():
    tape, _claim, audit = _info_sumcheck_case()
    tape.run_engine_pass(observer=audit)
    result = audit.finish()
    assert result["accepted"] is True
    assert result["materialized_local_proof_counts"] == {"sumcheck": 1}
    assert result["exact_fallbacks"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
@pytest.mark.parametrize("field", ["a", "d", "dw", "z_o", "rem",
                                    "surprisal"])
def test_info_sumcheck_rejects_tampered_relation(field):
    tape, claim, audit = _info_sumcheck_case()

    def tamper_then_observe(index, op, input_vars, input_data, outs, live):
        target = getattr(claim, field)
        var = target[0] if isinstance(target, list) else target
        bad = outs[var].clone()
        bad[0] = 99
        outs[var] = bad
        live[var] = bad
        audit(index, op, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()
    assert result["accepted"] is False


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
        heartbeat_every=1000, prototype=True)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_result_is_labeled_a_prototype():
    """Review 2026-09-23 findings 5 and 6: an ACCEPT from the runtime is a
    prototype's, and its result says which properties were not checked."""
    tape, dst, _raw = _case()
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1,
        heartbeat_every=1000, prototype=True)
    tape.run_engine_pass(free_intermediates=True, keep={dst.var}, observer=audit)
    result = audit.finish()
    assert result["accepted"] is True
    assert result["prototype"] is True and result["verified_inference"] is False
    assert len(result["unchecked"]) == 2
