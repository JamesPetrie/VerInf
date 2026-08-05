"""Machine profiles: measured hardware constants a prediction is priced with.

A profile is a JSON file in profiler/machines/. Fields are null until
calibrated on the target — predictions report which constants they used and
which were missing. The GB10 profile is seeded from measured numbers in
analysis/ (provenance strings say exactly where each came from); Blackwell
profiles get filled by running the calibration suite when cluster access
lands (see README: calibration).

The `interconnect` section is deliberately a free-form dict and null for now:
cluster topology is unknown as of 2026-07. When it lands, the schema grows
(link bandwidth, domain size, hierarchy) without touching existing profiles.
"""
from __future__ import annotations

import json
import os
from typing import Optional

MACHINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "machines")


class MachineProfile:
    def __init__(self, raw: dict):
        self.raw = raw
        self.name = raw.get("name", "?")

    def get(self, *path, default=None):
        cur = self.raw
        for p in path:
            if not isinstance(cur, dict) or p not in cur or cur[p] is None:
                return default
            cur = cur[p]
        return cur

    @staticmethod
    def load(name_or_path: str) -> "MachineProfile":
        path = name_or_path
        if not os.path.exists(path):
            path = os.path.join(MACHINES_DIR, name_or_path + ".json")
        with open(path) as f:
            return MachineProfile(json.load(f))


def list_machines() -> list:
    return sorted(f[:-5] for f in os.listdir(MACHINES_DIR)
                  if f.endswith(".json") and not f.startswith("_"))
