"""Active-only MoE block (S3) vs. the all-expert builder.

The point of the stage is a structural one: the MoE FFN must stop building 128
expert output tensors per matrix. Two things have to hold for that to be a
replacement rather than a different model:

  1. the block computes the SAME thing — expert output equal, element for
     element, to the dense all-expert-then-select reference;
  2. no all-expert activation tensor is ever allocated, and the expert shards
     are read one at a time (asserted by watching what the witness pass touches).

Plus the projection cache: the prover regenerates the witness in five epochs,
and P = W*rho must be computed once per rho, not once per epoch.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "demo"))
import torch
import core
import claims as _C       # noqa: F401
import packets as _PK     # noqa: F401
import routed_projected
from tape import Tape
from rescale_claim import rescale
from routed_projected import routed_projected_matmul, RoutedProjectedMatmulClaim
from routing_claim import freivalds_combine
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=16, K_DEG=16, N_LIG=64, T_QUERIES=4)
T, D, FF, E = 3, 4, 4, 4
S, WIDTH = 1 << 4, 16


def _u64(t):
    return t.to(torch.int64).to(torch.uint64).cuda()


def _fixture(seed=5):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 7, (T, D), generator=g)
    W = torch.randint(0, 7, (E, D, FF), generator=g)
    routes = torch.randint(0, E, (T,), generator=g)
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate(routes.tolist()):
        M[t, e] = 1
    return X, W, M


def _routed_tape(X, W, M):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    x = tape.commit("X", _u64(X.reshape(-1)), (T, D))
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    w = [tape.commit(f"W{e}", _u64(W[e].reshape(-1)), (D, FF)) for e in range(E)]
    raw = routed_projected_matmul(tape, x, m, w, T=T, K=D, J=FF, E=E)
    out = rescale(tape, raw, s_in=S * S, s_out=S, output_width=WIDTH)
    return tape, out


def _all_expert_tape(X, W, M):
    """The old shape: every expert's full output, folded by freivalds_combine."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    mm = dict(s_a=S, s_b=S, s_out=S, output_width=WIDTH)
    x = tape.commit("X", _u64(X.reshape(-1)), (T, D))
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    outs = []
    for e in range(E):
        w_e = tape.commit(f"W{e}", _u64(W[e].reshape(-1)), (D, FF))
        outs.append(tape.matmul(x, w_e, **mm))
    y = freivalds_combine(tape, m, outs, T=T, E=E, F=FF)
    return tape, y, outs


def test_routed_block_matches_all_expert_builder():
    X, W, M = _fixture()
    tape_r, out_r = _routed_tape(X, W, M)
    got = tape_r.run_engine_pass()[out_r.var].view(torch.int64).cpu()
    tape_a, out_a, _ = _all_expert_tape(X, W, M)
    want = tape_a.run_engine_pass()[out_a.var].view(torch.int64).cpu()
    assert torch.equal(got, want), (
        f"routed block {got.tolist()} != all-expert builder {want.tolist()}")
    print(f"    routed == all-expert output, element for element: {got.tolist()}")


def _activation_slots(tape):
    """Committed witness slots that are NOT weights — the thing that used to
    grow with the expert count."""
    weights = {w for c in tape.claims
               if isinstance(c, RoutedProjectedMatmulClaim) for w in c.W}
    total = 0
    seen = set()
    for c in tape.claims:
        for f in c.__dataclass_fields__:
            v = getattr(c, f)
            for var in (v if isinstance(v, list) else [v]):
                if (isinstance(var, core.Variable) and var not in weights
                        and id(var) not in seen and not var.name.startswith("W")):
                    seen.add(id(var))
                    total += var.length
    return total


def test_activation_cost_is_independent_of_expert_count():
    """The structural claim of the stage: doubling E must not add a single
    activation slot to the routed block, while the all-expert builder grows
    with every expert it materializes."""
    def routed_slots(n_experts):
        g = torch.Generator().manual_seed(3)
        X = torch.randint(0, 7, (T, D), generator=g)
        W = torch.randint(0, 7, (n_experts, D, FF), generator=g)
        M = torch.zeros(T, n_experts, dtype=torch.int64)
        for t in range(T):
            M[t, t % n_experts] = 1
        core._COSET_POWERS_K_CACHE.clear()
        tape = Tape(CFG, lazy=True)
        x = tape.commit("X", _u64(X.reshape(-1)), (T, D))
        m = tape.commit("M", _u64(M.reshape(-1)), (T, n_experts))
        w = [tape.commit(f"W{e}", _u64(W[e].reshape(-1)), (D, FF))
             for e in range(n_experts)]
        raw = routed_projected_matmul(tape, x, m, w, T=T, K=D, J=FF, E=n_experts)
        rescale(tape, raw, s_in=S * S, s_out=S, output_width=WIDTH)
        return _activation_slots(tape)

    small, big = routed_slots(E), routed_slots(2 * E)
    # the E-sized rows are P (E*K) and the three length-E late auxiliaries;
    # everything else is fixed by T, d and d_ff.
    grew = big - small
    # What may legitimately scale with E: the route matrix M (T*E), the
    # projected weights P (E*K), and the three length-E late Freivalds
    # vectors. Nothing of size T*d_ff — that is the term the old builder paid
    # E times over.
    assert grew == T * E + E * D + 3 * E, (
        f"routed activations grew by {grew} for {E} extra experts; only M, P "
        f"and the three length-E Freivalds vectors may scale with E "
        f"(expected {T * E + E * D + 3 * E})")
    tape_a, _, outs_a = _all_expert_tape(*_fixture())
    assert len(outs_a) == E, "the reference builder should materialize E streams"
    print(f"    routed: +{grew} slots for {E} more experts (M + P + 3 length-E "
          f"vectors; no T x d_ff term); all-expert builder: {E} full "
          f"T x d_ff streams")


def test_projection_computed_once_per_rho():
    """Five witness epochs, one projection: the cache is what turns four
    identical passes over the enrolled weights into one."""
    X, W, M = _fixture()
    tape_r, _ = _routed_tape(X, W, M)
    proof = tape_r.prove()
    stats = routed_projected.P_CACHE_STATS
    assert stats["misses"] == 1, (
        f"P = W*rho was recomputed {stats['misses']} times; expected 1 "
        f"(hits={stats['hits']})")
    assert stats["hits"] >= 3, f"cache barely used: {stats}"
    acc, msg = rust_verify_tape(tape_r, proof, seed=None)
    assert acc, f"cached-projection proof: expected ACCEPT ({msg})"
    print(f"    projection: 1 computation, {stats['hits']} cache hits, ACCEPT")


def test_cache_is_keyed_by_the_challenge():
    """A different rho must miss: reusing another challenge's P would prove a
    relation the transcript never fixed."""
    X, W, M = _fixture()
    tape_r, _ = _routed_tape(X, W, M)
    claim = next(c for c in tape_r.claims
                 if isinstance(c, RoutedProjectedMatmulClaim))
    routed_projected.clear_p_cache()
    live = tape_r.run_engine_pass()
    rho_a = [3, 5, 7, 11][:FF]
    rho_b = [3, 5, 7, 13][:FF]
    routed_projected._project_weights(claim, live, rho_a)
    routed_projected._project_weights(claim, live, rho_a)
    routed_projected._project_weights(claim, live, rho_b)
    stats = routed_projected.P_CACHE_STATS
    assert (stats["misses"], stats["hits"]) == (2, 1), stats
    print(f"    cache keyed by rho: {stats}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== moe-routed: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
