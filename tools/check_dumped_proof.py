#!/usr/bin/env python3
"""Check one saved bridged proof with Rust after its prover process exits.

The companion policy file is written by instrumented_prove from the same
proof. It tests the proof's mechanics, not an independently enrolled model.
On ACCEPT, write a receipt with the exact proof's SHA-256, byte count,
the prover's and the verifier's revisions (recorded separately: the verifier
may be rebuilt from a later tree than the one that proved), the verifier
binary's SHA-256 and its wall time, for the run archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _hex32(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"policy has no lowercase 32-byte {name}")
    return value


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(proof: Path, verifier: Path, revision_file: Path | None = None,
          verifier_revision_file: Path | None = None) -> dict:
    proof = proof.resolve(strict=True)
    verifier = verifier.resolve(strict=True)
    receipt_path = Path(str(proof) + ".verify.json")
    part = Path(str(receipt_path) + ".part")
    if receipt_path.exists() or part.exists():
        raise FileExistsError(f"refusing to overwrite {receipt_path} or its .part")
    proof_bytes = proof.stat().st_size
    if proof_bytes <= 0:
        raise ValueError("proof file is empty")
    revision = revision_file.read_text().strip() if revision_file else None
    if revision_file and not revision:
        raise ValueError("revision stamp is empty")
    # the verifier's own revision: an explicit file, else <verifier>.revision
    # written beside the binary when it was built
    if verifier_revision_file is None:
        beside = Path(str(verifier) + ".revision")
        verifier_revision_file = beside if beside.exists() else None
    verifier_revision = (verifier_revision_file.read_text().strip()
                         if verifier_revision_file else None)
    if verifier_revision_file and not verifier_revision:
        raise ValueError("verifier revision stamp is empty")
    policy_path = Path(str(proof) + ".policy.json")
    with policy_path.open() as fh:
        policy = json.load(fh)
    if policy.get("format") != "verinf-dump-policy-v1":
        raise ValueError("unknown same-run policy format")
    if policy.get("policy_source") != "same-run prover (mechanism check only)":
        raise ValueError("policy source is not the instrumented prover")
    if policy.get("proof_file") != proof.name or policy.get("proof_bytes") != proof_bytes:
        raise ValueError("policy sidecar does not name this proof and byte count")
    weight_root = _hex32(policy.get("weight_root"), "weight root")
    statement_digest = _hex32(policy.get("statement_digest"), "statement digest")
    wc_identity = _hex32(policy.get("wc_identity"), "wc enrollment identity")
    argv = [str(verifier), str(proof), weight_root, statement_digest, wc_identity]
    print(f"Rust check of {proof} ({proof_bytes} bytes)", flush=True)
    print("Policy roots from the same prover run: proof mechanics check only", flush=True)
    t0 = time.monotonic()
    # stdout carries the verdict; stderr (the verifier's per-check progress
    # marks) passes straight through, so an hours-long check shows it is alive
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=None, text=True,
                            errors="replace")
    elapsed = time.monotonic() - t0
    sys.stdout.write(result.stdout)
    if result.returncode != 0 or "rust_verify: ACCEPT" not in result.stdout.splitlines():
        raise RuntimeError(f"Rust verifier did not ACCEPT (rc={result.returncode})")
    digest = _sha256_file(proof)
    receipt = {
        "format": "verinf-rust-check-v1",
        "proof_file": proof.name,
        "proof_bytes": proof_bytes,
        "proof_sha256": digest,
        "revision": revision,
        "prover_revision": revision,
        "verifier_revision": verifier_revision,
        "verifier_sha256": _sha256_file(verifier),
        "verifier": "verify_proof",
        "verdict": "ACCEPT",
        "verify_seconds": round(elapsed, 3),
        "policy_source": policy["policy_source"],
        "weight_root": weight_root,
        "statement_digest": statement_digest,
        "wc_identity": wc_identity,
    }
    with part.open("x") as fh:
        json.dump(receipt, fh, sort_keys=True, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(part, receipt_path)
    dfd = os.open(receipt_path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    print(f"S=1000 PROOF CHECKED: rust_verify: ACCEPT; "
          f"sha256={digest}; bytes={receipt['proof_bytes']}; "
          f"prover={revision}; verifier={verifier_revision}; "
          f"verify_s={elapsed:.1f}; receipt={receipt_path}", flush=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("proof", type=Path)
    ap.add_argument("--verifier", type=Path,
                    default=Path(os.environ.get("LIGERO_VERIFY_PROOF", "verifier/target/release/verify_proof")))
    ap.add_argument("--revision-file", type=Path,
                    help="the prover's revision stamp")
    ap.add_argument("--verifier-revision-file", type=Path,
                    help="the verifier's revision stamp (default: <verifier>.revision)")
    args = ap.parse_args(argv)
    try:
        check(args.proof, args.verifier, args.revision_file,
              args.verifier_revision_file)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"S=1000 PROOF CHECK FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
