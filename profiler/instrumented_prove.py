"""Instrumented RESEARCH prove of the Maverick demo tape — timing, not policy.

Why this exists: demo_maverick_full.py is fail-closed on current main. Its
proof path refuses to start without an enrolled commitment, the trusted
root, the public Sz, an admission report with the 714-run kernel bound AND
measured five-sweep semantics on real GGUF, and the target T_QUERIES — the
production contract in demo/4h-production-runbook.md. A rented-box timing
session has none of that (the admission bench emits the semantic stages as
null on purpose, so no report can pass), which is why the runbook's old
"just run the demo with --dump-proof" step cannot execute any more.

This driver runs the SAME tape through the SAME prover (build_model ->
reveal engine pass -> enrollment -> tape.prove with the enrollment) the way
the pre-policy demo did, with LIGERO_PHASE_TIMING on, and prints the wall,
the per-phase breakdown, the enrollment time and the proof size. It is a
measurement harness, deliberately NOT a production proof:

  * no admission gate, no statement digest, no verifier policy binding;
  * the enrollment is made in-process from the same tape (its secret seed is
    fresh, as WeightCommitment.from_tape defaults) and discarded — nothing
    is written to a ledger, so the run is not reusable as an enrollment;
  * the public Sz is DISCOVERED by a reveal engine pass (the extra sweep the
    production driver refuses to pay), exactly as the old demo did.

Do not cite its proof as a production artifact. Cite its timings.

    LIGERO_PHASE_TIMING=1 tools/spark_run.sh mavp-s1000 \\
        python3 profiler/instrumented_prove.py --from-gguf <local> \\
        --t-queries 54 --prompt-n 500 --cont-n 500 --dump-proof /workspace/mavp.bin

`--t-queries` is set before any demo import (the demo configs read
LIGERO_T_QUERIES at import time); the target geometry is 54. `--dump-proof`
writes the production u64le/base64 wire so the dump rate is a real
io.proof_dump_compact_MBps data point; omit it to measure the prove alone.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_PROFILER = Path(__file__).resolve().parent
_REPO = _PROFILER.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="instrumented research prove of the Maverick demo tape "
                    "(no admission gate; timings only)")
    ap.add_argument("--from-gguf", required=True)
    ap.add_argument("--t-queries", type=int, default=54,
                    help="LIGERO_T_QUERIES for the demo configs (target 54)")
    ap.add_argument("--prompt-n", type=int, default=2)
    ap.add_argument("--cont-n", type=int, default=2)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--d-ff", type=int, default=8192)
    ap.add_argument("--vocab", type=int, default=202048)
    ap.add_argument("--dump-proof", default=None,
                    help="also write the proof (u64le/base64 wire) here and "
                         "time the dump")
    ap.add_argument("--skip-reveal", action="store_true",
                    help="do not run the reveal engine pass; the UI bound "
                         "stays unpinned (only if the tape has no reveal pin)")
    ap.add_argument("--sweep-timing", action="store_true",
                    help="LIGERO_SWEEP_TIMING=1: one table per proof with a "
                         "row per sweep (buckets, loader calls and bytes, "
                         "cache reads/writes, projections)")
    ap.add_argument("--routed-cache", action="store_true",
                    help="LIGERO_ROUTED_Y_CACHE=1: reuse the routed claims' "
                         "first-sweep outputs (the A/B arm; proof bytes are "
                         "unchanged, gated in tests/test_shard_streaming.py)")
    ap.add_argument("--weight-cache", action="store_true",
                    help="LIGERO_WEIGHT_CACHE=1: decode each dense weight once "
                         "per proof and serve later resolutions from pinned "
                         "host memory (the A/B arm for the session-2 loader "
                         "finding; proof bytes unchanged)")
    ap.add_argument("--wc-bridge", action="store_true",
                    help="the collaborator's WC-LCRL-STC bridge: the expert "
                         "shards leave the witness (tape.external, use_bridge) "
                         "and a streaming coefficient-RS enrollment, built "
                         "in-process and thrown away like the weight "
                         "commitment, authenticates P = W rho at 40 points; "
                         "the dense weights stay in the enrolled block")
    ap.add_argument("--verify", action="store_true",
                    help="after proving, dump the proof and check it with the "
                         "Rust verifier (small runs only: the dump is the "
                         "legacy JSON and the check is single-threaded)")
    ap.add_argument("--dump-min-free-gb", type=float, default=60.0,
                    help="with --dump-proof, refuse to start unless the dump "
                         "directory has this much free space (an S=1000 "
                         "projected proof is about 36 GB)")
    ap.add_argument("--allow-env-geometry", action="store_true",
                    help="let LIGERO_ELL/K_DEG/N_LIG from the environment "
                         "override the target geometry (dev boxes only)")
    a = ap.parse_args(argv)

    # must precede any demo import: the demo configs read the geometry at
    # import time. The target geometry is pinned here so the shell's
    # environment cannot change the tape between two arms of an A/B; the
    # routed-cache flag is set both ways for the same reason.
    os.environ["LIGERO_T_QUERIES"] = str(a.t_queries)
    for k, v in (("LIGERO_ELL", "8192"), ("LIGERO_K_DEG", "16384"),
                 ("LIGERO_N_LIG", "65536")):
        if a.allow_env_geometry:
            os.environ.setdefault(k, v)
        else:
            os.environ[k] = v
    os.environ.setdefault("LIGERO_PHASE_TIMING", "1")
    if a.sweep_timing:                 # unset when off: core treats any value as on
        os.environ["LIGERO_SWEEP_TIMING"] = "1"
    else:
        os.environ.pop("LIGERO_SWEEP_TIMING", None)
    os.environ["LIGERO_ROUTED_Y_CACHE"] = "1" if a.routed_cache else "0"
    os.environ["LIGERO_WEIGHT_CACHE"] = "1" if a.weight_cache else "0"
    for k in ("LIGERO_LAYOUT_BREAKDOWN", "LIGERO_COMPILE_PROFILE_EXIT"):
        if os.environ.get(k):
            raise SystemExit(f"{k} is set: that diagnostic exits the prover "
                             "early; unset it for a timing run")
    if a.dump_proof:
        import shutil
        d = os.path.dirname(os.path.abspath(a.dump_proof)) or "."
        if not os.path.isdir(d):
            raise SystemExit(f"{d}: the proof dump directory does not exist; "
                             "create it on the verified local filesystem first")
        free_gb = shutil.disk_usage(d).free / 1e9
        if free_gb < a.dump_min_free_gb:
            raise SystemExit(f"{d}: {free_gb:.1f} GB free, below the "
                             f"--dump-min-free-gb {a.dump_min_free_gb:g} the "
                             "proof dump needs; not starting")
        if os.path.exists(a.dump_proof):
            raise SystemExit(f"{a.dump_proof} exists; not overwriting a proof — "
                             "name a new path")
        # The writer creates <path>.part and renames it; prove that a file
        # can be created, written and fsynced there BEFORE the hours of work.
        # Exclusive creation: an existing .part is a dump in progress or an
        # interrupted one, never something to truncate (review, 2026-09-13).
        probe = a.dump_proof + ".part"
        try:
            with open(probe, "xb") as fh:
                fh.write(b"preflight"); fh.flush(); os.fsync(fh.fileno())
        except FileExistsError:
            raise SystemExit(f"{probe} exists: a dump may be in progress or was "
                             "interrupted; remove it deliberately or name a "
                             "new path; not starting")
        except OSError as e:
            raise SystemExit(f"{d}: cannot create the proof's temporary file "
                             f"({e}); not starting")
        os.unlink(probe)                      # only the file this run created
        print(f"[instrumented_prove] proof dump dir {d}: {free_gb:.1f} GB free, "
              "writable", flush=True)
    for p in (_REPO / "prover", _REPO / "demo"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import torch
    import core
    import demo_maverick_block as _dmb
    import demo_maverick_full as dm

    _t_start = time.time()

    def log(msg):        # elapsed seconds since the driver started, so phases can be placed in the sampler's timeline
        print(f"[instrumented_prove +{time.time() - _t_start:.1f}s] {msg}", flush=True)

    log("RESEARCH TIMING RUN — not a production proof: no admission gate, "
        "in-process throwaway enrollment, Sz discovered by a reveal pass")
    log(f"geometry ELL={dm.CFG.ELL} K_DEG={dm.CFG.K_DEG} N_LIG={dm.CFG.N_LIG} "
        f"T_QUERIES={dm.CFG.T_QUERIES}"
        + (" (from the environment)" if a.allow_env_geometry else " (pinned)"))
    log(f"flags: witness cache {'on' if core._WITNESS_CACHE_ON else 'off'}, "
        f"host spill {'on' if core._WITNESS_SPILL_ON else 'off'}, "
        f"disk spill {'on' if core._WITNESS_SPILL_DISK else 'off'}, "
        f"routed-output cache {'ON' if core._ROUTED_Y_CACHE_ON else 'off'}, "
        f"decoded-weight cache {'ON' if core._WEIGHT_CACHE_ON else 'off'}, "
        f"group memo {'on' if _dmb._group_memo_on() else 'OFF'}, "
        f"wc bridge {'ON' if a.wc_bridge else 'off'}, "
        f"per-sweep table {'ON' if core._SWEEP_ON else 'off'}, "
        f"phase timing {'on' if core._PHASE_ON else 'off'}")
    torch.manual_seed(7)
    g = torch.Generator().manual_seed(11)
    prompt_ids = torch.randint(0, a.vocab, (a.prompt_n,), generator=g).tolist()
    cont_ids = torch.randint(0, a.vocab, (a.cont_n,), generator=g).tolist()
    T = len(prompt_ids) + len(cont_ids)
    log(f"layers={a.layers} E={a.experts} T={T} V={a.vocab} "
        f"T_QUERIES={dm.CFG.T_QUERIES} (ELL={dm.CFG.ELL} K_DEG={dm.CFG.K_DEG} "
        f"N_LIG={dm.CFG.N_LIG})")

    tape = dm.Tape(dm.CFG, silu_config=dm.SILU_CFG, lazy=True)
    dm.WC_BRIDGE = bool(a.wc_bridge)      # the demo's builder reads it per expert shard
    t0 = time.time()
    logits, Sz, handles, sum_pos = dm.build_model(
        tape, a.from_gguf, prompt_ids, cont_ids, V=a.vocab, d=a.d,
        n_layers=a.layers, E=a.experts, d_ff=a.d_ff)
    t_build = time.time() - t0
    log(f"build {t_build:.1f}s, {len(tape.claims)} claims")

    # Enrollment (P3 kept trees): the per-proof passes on the W block are
    # what the weight-split milestone splits, so the timing must be taken
    # against an enrolled block, as production runs are. Fresh secret seed,
    # never persisted.
    t0 = time.time()
    wc = core.WeightCommitment.from_tape(tape, dm.CFG)
    t_enroll = time.time() - t0
    log(f"enrolled {wc.m_w} weight rows in {t_enroll:.1f}s "
        f"(root {wc.root.hex()[:16]}…, throwaway)")
    # Under the bridge the expert shards are external inputs, so the block
    # above holds only the dense weights; the shards are enrolled by the
    # streaming coefficient-RS pass (throwaway too), timed on its own.
    wc_enr, t_wc_enroll = None, 0.0
    if a.wc_bridge:
        import gc
        import wc_bridge as _wcb
        gc.collect(); torch.cuda.empty_cache()
        t0 = time.time()
        wc_enr = _wcb.lazy_enroll_tape(
            tape, b"wc-instrumented-mask",
            f"maverick|{a.from_gguf}|S={1 << 12}".encode(), _wcb.WcParams())
        t_wc_enroll = time.time() - t0
        log(f"wc enrollment (streaming coefficient-RS, throwaway) {t_wc_enroll:.1f}s "
            f"root {wc_enr.root.hex()[:16]}…")

    # Reveal pass: discover Sz and pin it as the public bound, then re-zero
    # the LogUp multiplicities so the prove sweeps re-accumulate cleanly —
    # the pre-policy demo's exact sequence (530283a demo_maverick_full.main).
    t_reveal = 0.0
    if handles.get("reveal_pin") is not None and not a.skip_reveal:
        t0 = time.time()
        live = tape.run_engine_pass(free_intermediates=True, keep={Sz.var})
        sz = int(live[Sz.var].cpu().item())
        t_reveal = time.time() - t0
        bits = dm.bound_bits(sz, s_b=dm.UI["s_b"])
        log(f"reveal engine pass {t_reveal:.1f}s: Sz={sz} -> "
            f"{bits / max(len(sum_pos), 1):.4f} bits/token over "
            f"{len(sum_pos)} positions (pinned as the public bound)")
        handles["reveal_pin"].public_rhs = sz
        for v in list(tape.inputs):
            if getattr(v, "name", "").endswith("_mult"):
                tape.inputs[v].zero_()
        del live
        torch.cuda.empty_cache()
    elif handles.get("reveal_pin") is not None:
        raise SystemExit("this tape has a reveal pin; drop --skip-reveal so "
                         "the public bound can be discovered and pinned")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    proof = tape.prove(weight_commitment=wc, weight_enrollment=wc_enr)
    t_prove = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    log(f"PROVE WALL {t_prove:.1f}s ({t_prove / 60:.1f} min) peakGPU={peak:.2f}GiB "
        f"— the per-phase table above is LIGERO_PHASE_TIMING's")
    log(f"opened columns: {len(proof.Q_cols)} (T_QUERIES={dm.CFG.T_QUERIES})")

    if a.dump_proof:
        from proof_dump import dump_proof
        t0 = time.time()
        dump_proof(a.dump_proof, None, None, proof, None, None,
                   u64_encoding="u64le-base64")
        t_dump = time.time() - t0
        size = os.path.getsize(a.dump_proof)
        log(f"proof dumped: {size / 1e9:.2f} GB in {t_dump:.1f}s = "
            f"{size / t_dump / 1e6:.1f} MB/s (u64le/base64 wire) -> "
            f"io.proof_dump_compact_MBps candidate for the machine profile")

    if a.verify:
        # The independent check of a small proof through this very driver:
        # the prover's leaf check is fail-closed, but only the verifier
        # establishes the algebraic constraints.
        sys.path.insert(0, str(_REPO / "prover" / "tests"))
        from _rust_verify import rust_verify
        t0 = time.time()
        acc, msg = rust_verify(tape.claims, proof, None, dm.CFG)
        log(f"rust verify_proof: {'ACCEPT' if acc else 'REJECT'} in "
            f"{time.time() - t0:.1f}s")
        if not acc:
            log(msg[-2000:])
            return 1

    log(f"SUMMARY build={t_build:.1f}s enroll={t_enroll:.1f}s "
        + (f"wc_enroll={t_wc_enroll:.1f}s " if a.wc_bridge else "")
        + f"reveal={t_reveal:.1f}s prove={t_prove:.1f}s T={T} "
        f"T_QUERIES={dm.CFG.T_QUERIES} claims={len(tape.claims)}"
        + (" verify=ACCEPT" if a.verify else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
