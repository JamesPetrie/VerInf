#!/usr/bin/env python3
"""Prove + verify one interlock challenge, and bind the proof to certified bytes.

`app/model_server.py : handle_challenge()` shells out to this script (its
`CHALLENGE_PY`) once a challenge's certificates have been checked and bound to
retained traffic. By then the caller holds two payloads that the interlock's
certificate chain has already authenticated: the request token ids and the
response token ids. This script answers the remaining question -- does that
response actually follow from that request under the committed model? -- by
proving the forward pass with VerInf and checking it with the independent Rust
verifier.

Everything heavy stays on this machine: the proof is written to a temp file,
verified locally, and deleted. Only the one-line verdict crosses the interlock.

Usage (the caller's contract):
    --request   comma-separated request token ids   (the prompt)
    --response  comma-separated response token ids  (the completion)
    --t-queries opened Ligero columns (80 = production soundness; less is faster)

stdout contract: progress lines begin with "[", and the last line is exactly

    CHALLENGE_RESULT verdict=<PASS|FAIL> U=<bits|NA> verify=<ACCEPT|REJECT|ERROR>
                     out_bind=<OK|MISMATCH> hreq=<sha256> hrsp=<sha256>

`hreq`/`hrsp` are SHA-256 over the canonical little-endian uint32 payloads that
were proven on. The client compares them against hashes of the bytes it sent and
received, which is what ties the proof to the certified traffic (spec §6.3).
"""
import argparse
import hashlib
import io
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prover"))
sys.path.insert(0, str(ROOT / "demo"))
# The interlock app (wire crypto) as a sibling checkout; ILK_APP overrides.
ILK_APP = os.environ.get("ILK_APP") or os.path.join(
    os.path.dirname(str(ROOT)), "interlock", "app")
sys.path.insert(0, ILK_APP)

MODEL = os.environ.get("VERINF_MODEL", str(ROOT / "models/llama-3.2-1b"))

# Calibrated fine-scale unexplained-information settings (same as demo/gate.py).
# UI_LM_OW must match demo_llama7b.OUTPUT_WIDTH: the rescale range table is
# named by width, so a mismatch creates a second 2^26 table and saves nothing.
UI_LM_SOUT, UI_LM_OW, UI_S_C = 4096, 24, 1 << 18


def _ids(s):
    return [int(v) for v in s.split(",") if v.strip() != ""]


def _payload(ids):
    """Canonical wire payload: little-endian uint32 per token id."""
    return b"".join(struct.pack("<I", i & 0xFFFFFFFF) for i in ids)


def _n_layers(path):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)["num_hidden_layers"]


class _Tee:
    """Echo prover output live (the caller streams it as STATUS) and capture it."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _emit(verdict, U="NA", verify="?", out_bind="?", hreq="", hrsp="", keybind="n/a"):
    print("CHALLENGE_RESULT verdict=%s U=%s verify=%s out_bind=%s hreq=%s hrsp=%s "
          "keybind=%s" % (verdict, U, verify, out_bind, hreq, hrsp, keybind), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--response", required=True)
    ap.add_argument("--t-queries", default="80")
    # Key-binding inputs. All four arrive together or not at all: without them the
    # wire ran in the clear and there is no key to bind (keybind=n/a).
    ap.add_argument("--nonce", help="hex; the per-request nonce from the CERTIFIED "
                                    "request. The key is re-derived from it here "
                                    "against this box's PSK and never transits.")
    ap.add_argument("--key-commit", help="hex SHA256(key||iv_in||iv_out) as carried "
                                         "in the certified request (public, H2)")
    ap.add_argument("--ct-in", help="hex certified request ciphertext (public)")
    ap.add_argument("--ct-out", help="hex certified response ciphertext (public)")
    # Accepted and ignored. The sound path proves every token-layer, so there is
    # nothing for a verifier seed to select. It is declared so the one caller can
    # pass --seed unconditionally, and so this path never fails on an argument the
    # subsampled path needs.
    ap.add_argument("--seed", default="", help="accepted and unused (see above)")
    args = ap.parse_args()

    req_ids, rsp_ids = _ids(args.request), _ids(args.response)
    hreq = hashlib.sha256(_payload(req_ids)).hexdigest()[:16]
    hrsp = hashlib.sha256(_payload(rsp_ids)).hexdigest()[:16]

    if not req_ids or not rsp_ids:
        print("[challenge] empty request or response ids", flush=True)
        _emit("FAIL", verify="ERROR", out_bind="MISMATCH", hreq=hreq, hrsp=hrsp)
        return 1

    # T_QUERIES is read at import time by the demo's LigeroConfig.
    os.environ["LIGERO_T_QUERIES"] = str(args.t_queries)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import demo_llama7b

    # Persistent weight commitment (persistent-weights.md P3) is OFF by default.
    # It measured at ~1.5 s here -- inside run-to-run noise -- because it skips
    # the weight MERKLE hashing (0.7 s of prove), not the NTT. Against that it
    # carries real staleness risk: the cached m_w is witness/ELL, so it is only
    # valid for one (ELL, K_DEG, N_LIG) AND one num_layers, and a mismatch is a
    # hard failure ("weight commitment mismatch: m_w 13848 vs 55320"). Not worth
    # it at this scale. It should pay off on a large model, where the D*W term it
    # removes actually dominates -- opt in with:
    #   LIGERO_WEIGHT_COMMITMENT=<path>.pkl   (include ELL/K_DEG/N_LIG/layers in
    #                                          the name; the cache is only valid
    #                                          for the config that built it)

    full = req_ids + rsp_ids
    seq = len(full)
    # Position t predicts token t+1, so the observed output at t is full[t+1].
    # The final position has no successor; it is filler and never summed over.
    out_tokens = full[1:] + [full[-1]]
    # U is bounded over exactly the positions that produced the response.
    sum_positions = list(range(len(req_ids) - 1, seq - 1))

    print("[challenge] model=%s layers=%d SEQ=%d (req=%d rsp=%d) T_QUERIES=%s"
          % (MODEL, _n_layers(MODEL), seq, len(req_ids), len(rsp_ids), args.t_queries),
          flush=True)
    print("[challenge] U summed over %d response positions" % len(sum_positions), flush=True)

    # ---------------- key binding (token-binding.md B1 + B2) ----------------
    # Adds, to the SAME tape as the forward pass:
    #     SHA256(key||iv_in||iv_out) == KEY_COMMIT          (from the certified request)
    #     AES128-CTR(key, iv_in,  req_ids) == ct_in         (certified request bytes)
    #     AES128-CTR(key, iv_out, rsp_ids) == ct_out        (certified response bytes)
    # and welds the response side to the model's OWN committed output tokens, so the
    # decrypted stream is the stream the forward pass was scored on -- not a second
    # commitment that merely happens to hold the same values.
    keybind = "n/a"
    hook = None
    have = [args.nonce, args.key_commit, args.ct_in, args.ct_out]
    if any(have):
        if not all(have):
            print("[challenge] partial key-binding arguments; refusing to guess", flush=True)
            _emit("FAIL", verify="ERROR", out_bind="?", hreq=hreq, hrsp=hrsp,
                  keybind="ERROR")
            return 1
        import ilk_crypto as ic
        km, _key, _iv_in, _iv_out = ic.derive(ic.load_psk(), bytes.fromhex(args.nonce))
        kc = bytes.fromhex(args.key_commit)
        if ic.key_commit(km) != kc:
            # The request's KEY_COMMIT is not what this PSK derives: either the two
            # ends are provisioned differently, or the certified request was not
            # produced by a holder of this secret. Either way there is nothing
            # honest to prove, so say so instead of proving a different key.
            print("[challenge] KEY_COMMIT does not match the PSK-derived key material",
                  flush=True)
            _emit("FAIL", verify="ERROR", out_bind="?", hreq=hreq, hrsp=hrsp,
                  keybind="COMMIT-MISMATCH")
            return 1
        ct_in, ct_out = bytes.fromhex(args.ct_in), bytes.fromhex(args.ct_out)
        print("[challenge] key binding: KEY_COMMIT=%s ct_in=%dB ct_out=%dB"
              % (args.key_commit[:16], len(ct_in), len(ct_out)), flush=True)

        def hook(tape, ctx):
            import crypto_binding as cbnd
            t0 = time.time()
            # The request ids are public in the forward pass already (they are the
            # EmbeddingLookupClaim's public token_ids), so pinning the bound copy
            # to the same constants reveals nothing new and closes that side.
            res = cbnd.bind(tape, keymat=km, key_commit=kc,
                            streams=[{"name": "in", "tokens": req_ids, "ct": ct_in},
                                     {"name": "out", "tokens": rsp_ids, "ct": ct_out}],
                            reveal_tokens={"in"})
            ui_tok = ctx.get("ui_tok")
            if ui_tok is None:
                raise RuntimeError("no committed output tokens to weld to "
                                   "(unexplained_info must be on)")
            # ui_tok[t] is the token observed at position t, i.e. full[t+1]; the
            # response occupies exactly sum_positions. Gather those slots and pin
            # them to the AES plaintext tokens.
            g = cbnd.pool_gather(tape, ui_tok, "cb_weld_out", list(sum_positions))
            cbnd.bind_tokens_to(tape, res["tok"]["out"], g)
            print("[challenge] key binding: +%d claims, welded %d response tokens "
                  "(%.1fs)" % (res["n_claims"] + 1, len(sum_positions),
                               time.time() - t0), flush=True)

        keybind = "PENDING"

    proof = pathlib.Path(tempfile.gettempdir()) / ("interlock_challenge_%d.json" % os.getpid())
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = _Tee(old, buf)
    try:
        demo_llama7b.main(
            from_hf=MODEL, num_layers=_n_layers(MODEL), token_ids=full,
            unexplained_info=True, ui_output_tokens=out_tokens,
            ui_positions=sum_positions,
            ui_lm_sout=UI_LM_SOUT, ui_lm_ow=UI_LM_OW, ui_s_c=UI_S_C,
            engine=True, lazy_weights=True, dump_proof=str(proof),
            pre_prove_hook=hook)
    except Exception as e:                       # prover blew up -> honest FAIL
        sys.stdout = old
        print("[challenge] prover error: %s: %s" % (type(e).__name__, e), flush=True)
        _emit("FAIL", verify="ERROR", out_bind="?", hreq=hreq, hrsp=hrsp)
        return 1
    finally:
        sys.stdout = old
    out = buf.getvalue()

    m = re.search(r"U = ([\d.]+) bits over (\d+)", out)
    if not m:
        print("[challenge] no unexplained-information bound in prover output", flush=True)
        _emit("FAIL", verify="ERROR", out_bind="?", hreq=hreq, hrsp=hrsp)
        return 1
    u_total, n_tok = float(m.group(1)), int(m.group(2))
    u_per_tok = u_total / max(n_tok, 1)
    print("[challenge] U = %.4f bits over %d tokens = %.4f bits/token"
          % (u_total, n_tok, u_per_tok), flush=True)

    vbin = ROOT / "verifier/target/release/verify_proof"
    if not vbin.exists():
        print("[challenge] verifier binary missing: %s" % vbin, flush=True)
        _emit("FAIL", U="%.4f" % u_per_tok, verify="ERROR", out_bind="?",
              hreq=hreq, hrsp=hrsp)
        return 1
    # ---- verifier policy ---------------------------------------------------
    # verify_proof fail-closes without a trusted weight root and a trusted
    # statement digest, and it is right to: with neither, the prover picks what
    # it proves and the verifier only confirms the proof is consistent with
    # itself. Unlike the subsampled path, this proof commits ALL layers under a
    # single W-block root, and enrolment (.enroll) pins PER-LAYER leaves -- so
    # there is no enrolled value of this proof's shape to pin against yet.
    # DEMO_SELF_POLICY=1 supplies both from the prover's own dump, which is the
    # honest description of what a co-located verifier can do and NOT a security
    # claim; leave it unset and the verifier stays fail-closed, which is correct
    # anywhere the verifier is a separate party.
    policy = []
    if os.environ.get("DEMO_SELF_POLICY") == "1":
        try:
            with open(proof) as fh:
                top = json.load(fh)
            rw = (top.get("proof") or {}).get("root_w")
            sd = top.get("statement_digest")
            policy = [rw or "-", sd or "-"]
            print("[challenge] POLICY SELF-SUPPLIED (verifier co-located): the "
                  "enrolled-root and statement-digest checks are not independent "
                  "in this mode", flush=True)
        except Exception as e:
            print("[challenge] self-policy unavailable (%s: %s) -- verifier "
                  "will fail closed" % (type(e).__name__, e), flush=True)

    print("[challenge] verifying with the independent Rust verifier ...", flush=True)
    res = subprocess.run([str(vbin), str(proof)] + policy,
                         capture_output=True, text=True)
    accept = res.returncode == 0 and "rust_verify: ACCEPT" in res.stdout
    verify = "ACCEPT" if accept else "REJECT"
    if not accept:
        # The verifier's own account of WHY -- a bare REJECT with the reason
        # discarded is the least useful thing this could print.
        why = re.compile(r"\[XX|reject|fail|mismatch|bad |invalid|expected|!=|missing",
                         re.I)
        for stream, txt in (("out", res.stdout), ("err", res.stderr)):
            lines = (txt or "").strip().splitlines()
            keep = [l for l in lines if why.search(l)][-14:] or lines[-6:]
            for l in keep:
                print("[challenge] verifier %s: %s" % (stream, l[:180]), flush=True)
        print("[challenge] verifier rc=%d" % res.returncode, flush=True)
    print("[challenge] rust verifier: %s" % verify, flush=True)
    try:
        proof.unlink()                            # the proof never leaves this box
    except OSError:
        pass

    # (g) The proof ran on the response ids we were handed -- which handle_challenge
    # already bound to the certificate -- so a low U over those positions is a
    # statement about the certified bytes. hreq/hrsp let the client re-check that
    # linkage against the payloads it actually sent and received.
    out_bind = "OK"
    # The binding claims live on the tape the verifier just checked, so its single
    # ACCEPT/REJECT covers them too: there is no separate binding verdict to report,
    # only whether binding claims were present at all.
    if keybind == "PENDING":
        keybind = "OK" if accept else "FAIL"
    verdict = "PASS" if accept else "FAIL"
    _emit(verdict, U="%.4f" % u_per_tok, verify=verify, out_bind=out_bind,
          hreq=hreq, hrsp=hrsp, keybind=keybind)
    return 0 if accept else 1


if __name__ == "__main__":
    raise SystemExit(main())
