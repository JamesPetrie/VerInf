"""CPU-runnable bookkeeping checks for the prover's instruments: the per-sweep
table (LIGERO_SWEEP_TIMING) and the decoded-weight cache's accounting
(LIGERO_WEIGHT_CACHE). No proof is built and no CUDA is needed: the sync and
the pinned allocator are stubbed when CUDA is absent, and the pinned-memory
spill helpers are stubbed always, so this locks the counters, the per-kind
timing, the budget and the centered-int32 packing arithmetic — not the
proof bytes, which prover/tests/test_weight_cache.py gates on a GPU.
Runs under prover/tests/run_tests.py like the rest, or standalone."""
import contextlib
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import core

V = core.Variable


@contextlib.contextmanager
def _instrumented(weight_cache=None):
    """Sweep + phase timing on, the spill helpers stubbed to CPU clones (no
    pinned allocator without CUDA), and every touched global restored."""
    saved = {k: getattr(core, k) for k in (
        "_SWEEP_ON", "_PHASE_ON", "_SWEEP_CUR", "_SWEEP_RECS", "_SWEEP_OUTSIDE",
        "_PHASE_TIMES", "_PHASE_STACK", "_WEIGHT_CACHE", "_spill_store",
        "_spill_load", "_pinned_empty")}
    saved_sync = torch.cuda.synchronize
    try:
        core._SWEEP_ON = True; core._PHASE_ON = True
        core._SWEEP_CUR = None; core._SWEEP_RECS = []; core._SWEEP_OUTSIDE = {}
        core._PHASE_TIMES = {}; core._PHASE_STACK = []
        core._WEIGHT_KIND.clear(); core._WEIGHT_CACHE = weight_cache
        core._spill_store = lambda t: ("cpu", t.clone())
        core._spill_load = lambda e: e[1].clone()
        if not torch.cuda.is_available():
            torch.cuda.synchronize = lambda *a, **k: None
            core._pinned_empty = lambda shape, dtype: torch.empty(shape, dtype=dtype)
        yield
    finally:
        for k, v in saved.items():
            setattr(core, k, v)
        torch.cuda.synchronize = saved_sync
        core._WEIGHT_KIND.clear()


def _loader(calls, name, value, n=8, sleep=0.01):
    def load():
        calls[name] = calls.get(name, 0) + 1
        time.sleep(sleep)
        return torch.full((n,), value, dtype=torch.int64).to(torch.uint64)
    return load


def test_sweep_table_fetch_is_exclusive_and_counts_by_kind():
    """A loader call nested in `witness` is booked to `fetch` for the sweep
    row (witness stays exclusive of it), nested buckets fold into the top
    bucket, calls outside a sweep are counted apart, and the off path just
    calls the loader."""
    with _instrumented():
        calls = {}
        ld = _loader(calls, "x", 1, n=1000, sleep=0.05)
        core._sweep_begin("R1")
        with core._phase('witness'):
            core._resolve_loader(ld); time.sleep(0.02)
        with core._phase('aux'):
            d = core._LazyResolvingDict({'w': ld, 'x': 1}); d['w']; d['w']
            assert d['x'] == 1
        with core._sphase('cache_w'):
            time.sleep(0.01)
        core._sweep_count('cache_wr')
        core._sweep_end()
        rec = core._SWEEP_RECS[-1]
        assert rec['label'] == "R1"
        assert rec['n'] == {'loads': 2, 'loads_input': 2, 'load_bytes': 16000, 'cache_wr': 1}, rec['n']
        assert rec['k']['fetch_input'] >= 0.09, rec['k']
        assert 0.09 <= rec['t']['fetch'] <= 0.5 and 0.015 <= rec['t']['witness'] <= 0.1, rec['t']
        assert rec['t']['aux'] < 0.05 and 0.008 <= rec['t']['cache_w'] <= 0.1, rec['t']
        assert 0.015 <= core._PHASE_TIMES['witness'] <= 0.1 and core._PHASE_TIMES['fetch'] >= 0.09
        assert core._PHASE_STACK == []
        core._resolve_loader(ld)
        assert core._SWEEP_OUTSIDE == {'loads': 1, 'loads_input': 1, 'load_bytes': 8000}, core._SWEEP_OUTSIDE
        core._sweep_begin("fold")
        with core._phase('fold_qlin'):
            with core._phase('qlin_interp'):
                core._resolve_loader(ld)
            time.sleep(0.01)
        core._sweep_end()
        rec = core._SWEEP_RECS[-1]
        assert 'qlin_interp' not in rec['t'] and rec['t']['fetch'] >= 0.045, rec['t']
        assert rec['n'] == {'loads': 1, 'loads_input': 1, 'load_bytes': 8000}, rec['n']
        core._sweep_report(1.0)
        core._SWEEP_ON = False
        before = dict(core._SWEEP_OUTSIDE)
        with core._sphase('cache_r'):
            pass
        assert core._resolve_loader(lambda: 5) == 5 and core._SWEEP_OUTSIDE == before
        core._sweep_begin("R1"); assert core._SWEEP_CUR is None; core._sweep_end()


def test_weight_cache_bookkeeping_kinds_budget_and_release():
    """A dense weight is decoded once and served afterwards; shards and
    inputs are never cached; kinds are timed per sweep; the budget refuses
    cleanly; the cache is released when the prove raises."""
    wc = core._weight_cache_new(); wc['_budget'] = 1000; wc['_gpu_budget'] = 0
    with _instrumented(weight_cache=wc):
        dense = V("L0_W_Q", length=8, phase=1, persistent=True)
        shard = V("L0_Wg3", length=8, phase=1, persistent=True)
        act = V("x", length=8, phase=1)
        calls = {}
        core._WEIGHT_KIND[id(shard)] = 'shard'
        core._sweep_begin("R1")
        a = core._resolve_loader(_loader(calls, "dense", 7), dense)
        b = core._resolve_loader(_loader(calls, "dense", 7), dense)
        assert calls["dense"] == 1 and torch.equal(a.view(torch.int64), b.view(torch.int64))
        assert a.data_ptr() != b.data_ptr()
        for _ in range(2):
            core._resolve_loader(_loader(calls, "shard", 3), shard)
            core._resolve_loader(_loader(calls, "act", 1), act)
        assert calls["shard"] == 2 and calls["act"] == 2, "shards and inputs must not be cached"
        core._sweep_end()
        n = core._SWEEP_RECS[-1]['n']; k = core._SWEEP_RECS[-1]['k']
        assert n['loads'] == 6 and n['weight_hits'] == 1 and n['weight_stores'] == 1, n
        assert n['loads_weight'] == 1 and n['loads_shard'] == 2 and n['loads_input'] == 2, n
        assert k['fetch_weight'] >= 0.009 and k['fetch_shard'] >= 0.018, k
        assert wc['_hits'] == 1 and wc['_misses'] == 1 and wc['_refused'] == 0, wc
        assert wc[id(dense)][0] == 'i32' and wc['_bytes'] == 32 and wc['_packed'] == 32
        big = V("L1_W_K", length=8, phase=1, persistent=True)
        wc['_budget'] = 40                      # 32 used + 32 > 40 -> refused, served by the loader
        core._sweep_begin("R2")
        core._resolve_loader(_loader(calls, "dense", 9), big)
        core._resolve_loader(_loader(calls, "dense", 9), big)
        assert calls["dense"] == 3 and wc['_refused'] == 2 and id(big) not in wc
        core._sweep_end()
        core._sweep_report(1.0)
        core._WEIGHT_CACHE = None
        core._sweep_begin("R3")
        core._resolve_loader(_loader(calls, "dense", 7), dense)
        core._resolve_loader(_loader(calls, "dense", 7), dense)
        core._sweep_end()
        assert calls["dense"] == 5 and core._SWEEP_RECS[-1]['n']['loads_weight'] == 2
    # release however the prove ends (prove_streaming's finally)
    saved_on, saved_body = core._WEIGHT_CACHE_ON, core._prove_streaming_body
    try:
        core._WEIGHT_CACHE_ON = True
        def boom(*a, **k):
            core._WEIGHT_CACHE = core._weight_cache_new(); raise RuntimeError("mid-prove")
        core._prove_streaming_body = boom
        try:
            core.prove_streaming(None, None)
        except RuntimeError:
            pass
        assert core._WEIGHT_CACHE is None
    finally:
        core._WEIGHT_CACHE_ON, core._prove_streaming_body = saved_on, saved_body


def test_centered_int32_packing_roundtrip_and_charged_sizes():
    """Exact over the whole int32 domain, the int64 fallbacks exact, the
    budget charged the allocator's power-of-two ceiling, the GPU tier
    preferred when it fits (on a CUDA box)."""
    P = core.P
    def u64(vals):
        return torch.tensor([v - (1 << 64) if v >= (1 << 63) else v for v in vals],
                            dtype=torch.int64).view(torch.uint64)
    with _instrumented():
        small = u64([0, 7, P - 7, (1 << 31) - 1, P - ((1 << 31) - 1)])
        e = core._weight_pack(small, 10 ** 9); assert e[0] == 'i32' and e[2] == 32, e[:3]
        back = core._weight_unpack(e)
        assert back.dtype == torch.uint64 and torch.equal(back.view(torch.int64), small.view(torch.int64))
        for bad in ([1 << 31], [P - (1 << 31)], [P], [P + 5], [1 << 40]):
            e = core._weight_pack(u64(bad), 10 ** 9); assert e[0] == 'i64', (bad, e[0])
            assert torch.equal(core._weight_unpack(e).view(torch.int64), u64(bad).view(torch.int64))
        assert core._weight_pack(torch.zeros(3, dtype=torch.int64), 10 ** 9)[0] == 'i64'
        assert core._weight_pack(u64([1] * 9), 10 ** 9)[2] == 64
        assert core._weight_pack(small, 31) is None and core._weight_pack(u64([1 << 40]), 7) is None
        assert core._pinned_bytes(105 * 2 ** 20) == 128 * 2 ** 20 and core._pinned_bytes(0) == 0
        if torch.cuda.is_available():
            g = core._weight_pack(small.cuda(), 0, 10 ** 9)
            assert g[0] == 'g32' and g[1].is_cuda and g[2] == 20
            assert torch.equal(core._weight_unpack(g).view(torch.int64), small.cuda().view(torch.int64))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== instrument-bookkeeping: {len(tests)-fails}/{len(tests)} {'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
