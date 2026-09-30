"""The sampled-audit campaign validator knows both artifact formats (CPU;
review of 62c1cf4, 2026-09-30).

It was written for the August runs (no prototype labels, partial or no
cryptographic local proofs, the old acceptance line) and so rejected every
current run, which the campaign script requires to be a labeled prototype
with full coverage. The archived August campaign still validates as
august-2026; a current-format artifact, derived from it with the fields the
campaign script checks, validates as prototype-v1; and a current artifact
that is unlabeled, claims verified inference or carries the old line fails."""
import json
import os
import shutil
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO, "analysis", "bench"))

import validate_sampled_campaign as v

AUGUST = os.path.join(REPO, "analysis", "bench", "remote_results", "dc3f672fd559")
FAMILIES = "freivalds+product-tree+sumcheck"


def test_the_august_campaign_still_validates():
    report = v.validate(v.pathlib.Path(AUGUST))
    assert report["validated"] and report["format"] == "august-2026"


def _current(tmp_path, **override):
    """The August artifact rewritten as a prototype-v1 run would record it."""
    root = tmp_path / "campaign"
    shutil.copytree(AUGUST, root)
    r = json.loads((root / "campaign_results.json").read_text())
    r.update({
        "prototype": True, "verified_inference": False,
        "notice": "PROTOTYPE sampled audit: its result is not verified model inference",
        "unchecked": ["weight-to-enrollment binding: ...",
                      "cross-window value consistency: ..."],
        "local_argument": FAMILIES,
        "binding": "rs-window+striped-blake3 " + FAMILIES + " runtime",
        "cryptographic_local_proofs": True,
        "cryptographic_local_proof_coverage": 1.0,
        "manifest_proof_family_counts": {"freivalds": 554, "product-tree": 755,
                                         "sumcheck": 1287},
        "manifest_materialized_local_proofs": 2596,
        "materialized_local_proof_counts": {"freivalds": 60, "product-tree": 80,
                                            "sumcheck": 125},
        "materialized_local_proofs": 265,
        "local_proof_digests": ["00" * 32] * 265,
        "local_receipts": [{}] * 265,
        "exact_fallbacks": 0,
        "local_proof_bytes": 1,
        "rs_column_samples": [{"local_receipts": 5, "columns": list(range(61))}] * 53,
    })
    r.update(override)
    (root / "campaign_results.json").write_text(json.dumps(r))
    log = (root / "full.log").read_text().replace(
        v.LOG_LINES["august-2026"], v.LOG_LINES["prototype-v1"])
    (root / "full.log").write_text(log)
    return root


def test_a_current_prototype_campaign_validates(tmp_path):
    report = v.validate(_current(tmp_path))
    assert report["validated"] and report["format"] == "prototype-v1"


@pytest.mark.parametrize("override", [
    {"prototype": False},                       # labeled but not a prototype
    {"verified_inference": True},               # claims more than it checked
    {"cryptographic_local_proof_coverage": 0.5},
    {"exact_fallbacks": 1},
    {"unchecked": []},
])
def test_a_current_campaign_that_overclaims_fails(tmp_path, override):
    with pytest.raises(AssertionError):
        v.validate(_current(tmp_path, **override))


def test_the_old_acceptance_line_does_not_pass_a_current_campaign(tmp_path):
    root = _current(tmp_path)
    log = (root / "full.log").read_text().replace(
        v.LOG_LINES["prototype-v1"], v.LOG_LINES["august-2026"])
    (root / "full.log").write_text(log)
    with pytest.raises(AssertionError):
        v.validate(root)
