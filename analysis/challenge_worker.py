"""Resident prover: pay torch + CUDA import once, not once per challenge.

`model_backend.py` used to spawn CHALLENGE_PY fresh for every challenge, so each
one paid ~1.2 s of interpreter start, torch import and CUDA context creation
before any work began. This process stays alive across challenges and runs them
in-process.

PROTOCOL (newline JSON in, line-oriented text out -- deliberately the same shape
as the old subprocess, so the caller's streaming code is unchanged):

    in   {"argv": ["--request", "1,2", ...]}
    out  ... the challenge's own stdout, streamed live ...
         WORKER_DONE <rc>

It prints WORKER_READY once warm. The caller treats a dead worker as a fallback
condition and reverts to a one-shot subprocess, so this can only make things
faster, never break them.

WHAT IS AND IS NOT PRE-WARMED. torch, the CUDA context and the prover core are
imported up front -- none of them read per-run configuration. `demo_llama7b` is
NOT: it reads LIGERO_T_QUERIES at import time, so pre-importing it would freeze
the query count at whatever the worker started with and silently ignore a later
change. It is imported on the first job instead, and the caller respawns the
worker if the query count ever changes (see model_backend._challenge_worker).
"""
import importlib.util
import json
import os
import sys
import time


def _load(script_path):
    spec = importlib.util.spec_from_file_location("_challenge_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) < 2:
        print("usage: challenge_worker.py <challenge_script.py>", file=sys.stderr)
        return 2
    script = sys.argv[1]

    t0 = time.time()
    import torch
    torch.zeros(1).cuda()                       # force CUDA context creation now
    root = os.path.dirname(os.path.dirname(os.path.abspath(script)))
    ilk_app = os.environ.get("ILK_APP") or os.path.join(
        os.path.dirname(root), "interlock", "app")
    for p in (root + "/prover", root + "/demo", ilk_app):
        if p not in sys.path:
            sys.path.insert(0, p)
    import core, claims, packets, tape          # noqa: F401  (prover core, config-free)
    print("[worker] warm in %.1fs (%s)" % (time.time() - t0, os.path.basename(script)),
          flush=True)
    print("WORKER_READY", flush=True)

    mod = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except ValueError:
            print("WORKER_DONE 2", flush=True)
            continue
        if mod is None:                          # first job binds LIGERO_T_QUERIES
            mod = _load(script)
        old_argv = sys.argv
        sys.argv = ["challenge"] + list(job.get("argv", []))
        rc = 1
        try:
            rc = mod.main() or 0
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
        except Exception as e:                   # never take the worker down
            import traceback
            traceback.print_exc()
            print("[worker] error: %s: %s" % (type(e).__name__, e), flush=True)
        finally:
            sys.argv = old_argv
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print("WORKER_DONE %d" % rc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
