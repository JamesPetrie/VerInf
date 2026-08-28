"""Fail-closed consistency checks for a downloaded sampled-audit campaign."""
from __future__ import annotations

import argparse
import json
import pathlib


def validate(root: pathlib.Path) -> dict:
    result = json.loads((root / "campaign_results.json").read_text())
    progress = [json.loads(line) for line in
                (root / "progress.jsonl").read_text().splitlines() if line]
    full_log = (root / "full.log").read_text()
    enroll_log = (root / "enroll.log").read_text()

    assert result["accepted"] is True
    assert result["failures"] == []
    assert result["claims"] == 2596 and result["selected"] == 265
    assert result["selected"] == len(result["selected_indices"])
    assert len(set(result["selected_indices"])) == result["selected"]
    assert abs(result["fraction"] - 265 / 2596) < 1e-12
    assert result["wall_s"] <= 600
    assert result["binding"] == "rs-window+striped-blake3 exact-local runtime"
    assert result["rs_openings_materialized"] is True
    assert result["cryptographic_local_proofs"] is False
    assert result["local_argument"] == "exact-recomputation"
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
    assert "sampled audit: ACCEPT; 265/2596" in full_log
    assert "2596 claims total" in enroll_log
    assert "enrolled 49160720 weight rows" in enroll_log

    return {
        "validated": True,
        "artifact": str(root),
        "wall_s": result["wall_s"],
        "claims": result["claims"],
        "selected": result["selected"],
        "windows": len(windows),
        "rs_rows": result["rs_rows"],
        "rs_opened_values": result["rs_opened_values"],
        "c0_root": result["c0_root"],
        "cryptographic_local_proofs": result["cryptographic_local_proofs"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    report = validate(args.artifact)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.out:
        args.out.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
