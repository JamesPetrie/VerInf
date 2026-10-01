"""Fail-closed consistency checks for a downloaded sampled-audit campaign.

Two artifact formats, told apart by what the result carries and checked by
their own rules; anything else fails:

  august-2026    the August runs, before the prototype labels: exact
                 recomputation or partial cryptographic local proofs, and the
                 log line "sampled audit: ACCEPT; 265/2596";
  prototype-v1   the runtime since it runs only as a labeled prototype:
                 prototype true, verified_inference false, the unchecked
                 list, every selected claim a materialized local proof (the
                 same rules sampled_audit_vast.sh applies after a run), and
                 the log line "sampled audit [PROTOTYPE, not verified
                 inference]: local checks ACCEPT; 265/2596".
"""
from __future__ import annotations

import argparse
import json
import pathlib


FORMATS = ("august-2026", "prototype-v1")
FAMILIES = {"freivalds", "product-tree", "sumcheck"}
LOG_LINES = {
    "august-2026": "sampled audit: ACCEPT; 265/2596",
    "prototype-v1": ("sampled audit [PROTOTYPE, not verified inference]: "
                     "local checks ACCEPT; 265/2596"),
}


# the timed audit's cap: the August runs' 600 s, and for prototype-v1 the
# campaign script's default SAMPLED_AUDIT_TIMEOUT_S (sampled_audit_vast.sh);
# a campaign run under another cap passes it with --cap-s
DEFAULT_CAP_S = {"august-2026": 600, "prototype-v1": 1740}


def artifact_format(result: dict) -> str:
    if "prototype" not in result:
        return "august-2026"
    assert result["prototype"] is True, "a labeled result that is not a prototype"
    return "prototype-v1"


def validate(root: pathlib.Path, cap_s: float | None = None) -> dict:
    result = json.loads((root / "campaign_results.json").read_text())
    progress = [json.loads(line) for line in
                (root / "progress.jsonl").read_text().splitlines() if line]
    full_log = (root / "full.log").read_text()
    enroll_log = (root / "enroll.log").read_text()

    fmt = artifact_format(result)
    assert result["accepted"] is True
    assert result["failures"] == []
    assert result["claims"] == 2596 and result["selected"] == 265
    assert result["selected"] == len(result["selected_indices"])
    assert len(set(result["selected_indices"])) == result["selected"]
    assert abs(result["fraction"] - 265 / 2596) < 1e-12
    cap = DEFAULT_CAP_S[fmt] if cap_s is None else cap_s
    assert result["wall_s"] <= cap, f"timed audit {result['wall_s']:.1f}s exceeds the {cap}s cap"
    if result["local_argument"] == "exact-recomputation":
        expected_binding = "rs-window+striped-blake3 exact-local runtime"
    else:
        families = set(result["local_argument"].split("+"))
        assert families <= {
            "freivalds", "product-tree", "sumcheck", "exact-recomputation"}
        expected_binding = ("rs-window+striped-blake3 "
                            + result["local_argument"] + " runtime")
    assert result["binding"] == expected_binding
    assert result["rs_openings_materialized"] is True
    if fmt == "august-2026":
        assert result["cryptographic_local_proofs"] is False
        if result["local_argument"] != "exact-recomputation":
            _check_materialized(result)
            assert 0 < result["cryptographic_local_proof_coverage"] < 1
    else:
        # a prototype's ACCEPT: its local checks passed, nothing more
        assert result["verified_inference"] is False
        assert [u.split(":")[0] for u in result["unchecked"]] == [
            "weight-to-enrollment binding", "cross-window value consistency"]
        assert set(result["local_argument"].split("+")) == FAMILIES
        assert result["cryptographic_local_proofs"] is True
        assert result["manifest_materialized_local_proofs"] == result["claims"] == 2596
        assert set(result["materialized_local_proof_counts"]) == FAMILIES
        assert result["materialized_local_proofs"] == result["selected"] == 265
        assert result["exact_fallbacks"] == 0
        _check_materialized(result)
        assert result["cryptographic_local_proof_coverage"] == 1
    assert result["rs_columns"] == 61
    assert result["rs_geometry"] == {
        "ELL": 16322, "K_DEG": 16384, "N_LIG": 32768}
    assert result["rs_rows"] > 0
    assert result["rs_opened_values"] == result["rs_rows"] * 61
    assert result["rs_commit_s"] > 0
    assert result["rs_open_s"] > 0
    assert result["rs_verify_s"] > 0
    components = (result["forward_s"] + result["commit_s"]
                  + result["local_checks_s"] + result["rs_open_s"]
                  + result["rs_verify_s"])
    assert abs(components - result["wall_s"]) < 0.05

    starts = [event for event in progress if event.get("event") == "audit_start"]
    windows = [event for event in progress
               if event.get("event") == "window_complete"]
    completes = [event for event in progress
                 if event.get("event") == "audit_complete"]
    assert len(starts) == 1 and len(windows) == 53 and len(completes) == 1
    assert [event["window"] for event in windows] == list(range(53))
    assert all(a["elapsed_s"] < b["elapsed_s"]
               for a, b in zip(windows, windows[1:]))
    last = windows[-1]
    assert last["failures"] == 0
    assert last["rs_rows_total"] == result["rs_rows"]
    assert last["rs_opened_values"] == result["rs_opened_values"]
    assert abs(last["rs_commit_total_s"] - result["rs_commit_s"]) < 1e-6
    assert abs(last["rs_open_total_s"] - result["rs_open_s"]) < 1e-6
    assert abs(last["rs_verify_total_s"] - result["rs_verify_s"]) < 1e-6
    assert completes[0]["accepted"] is True
    assert completes[0]["selected"] == 265
    assert LOG_LINES[fmt] in full_log
    assert "2596 claims total" in enroll_log
    assert "enrolled 49160720 weight rows" in enroll_log

    return {
        "validated": True,
        "format": fmt,
        "artifact": str(root),
        "wall_s": result["wall_s"],
        "claims": result["claims"],
        "selected": result["selected"],
        "windows": len(windows),
        "rs_rows": result["rs_rows"],
        "rs_opened_values": result["rs_opened_values"],
        "c0_root": result["c0_root"],
        "cryptographic_local_proofs": result["cryptographic_local_proofs"],
        "local_argument": result["local_argument"],
        "materialized_local_proofs": result.get(
            "materialized_local_proofs", 0),
    }


def _check_materialized(result: dict) -> None:
    """The materialized local proofs' bookkeeping, common to both formats."""
    assert result["manifest_proof_family_counts"] == {
        "freivalds": 554, "product-tree": 755, "sumcheck": 1287}
    assert result["materialized_local_proofs"] > 0
    assert result["materialized_local_proofs"] == len(result["local_proof_digests"])
    assert len(result["local_receipts"]) == result["selected"]
    assert len(result["rs_column_samples"]) == 53
    assert sum(sample["local_receipts"] for sample in
               result["rs_column_samples"]) == result["selected"]
    assert all(len(sample["columns"]) == len(set(sample["columns"])) == 61
               for sample in result["rs_column_samples"])
    assert result["local_proof_bytes"] > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--cap-s", type=float, default=None,
                    help="the campaign's SAMPLED_AUDIT_TIMEOUT_S when it was not "
                         "the default for its format")
    args = ap.parse_args()
    report = validate(args.artifact, args.cap_s)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.out:
        args.out.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
