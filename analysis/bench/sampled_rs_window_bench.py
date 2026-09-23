"""Measure the exact production RS/Merkle sweeps used by sampled audit.

This is a preflight, not a substitute for the real Maverick campaign.  It uses
the production Ligero geometry and reports commit/open throughput separately,
because opening queried columns currently rebuilds the codewords from messages.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prover"))

import admission  # noqa: E402
import core  # noqa: E402
import torch  # noqa: E402


class HashingColumnSink(core.ColumnSink):
    def __init__(self, n_rows, columns, row_base=0):
        super().__init__(n_rows, columns, row_base)
        self.digest_acc = core._make_merkle_acc(len(columns), n_rows)

    def write(self, abs_row, chunk, columns):
        self.digest_acc.update(chunk.contiguous())
        super().write(abs_row, chunk, columns)

    def finish_with_digests(self):
        opened = super().finish()
        raw = self.digest_acc.finalize().cpu().numpy()
        return opened, [bytes(row.tolist()) for row in raw]


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--columns", type=int, default=61)
    ap.add_argument("--ell", type=int, default=8192)
    ap.add_argument("--k-deg", type=int, default=16384)
    ap.add_argument("--n-lig", type=int, default=65536)
    ap.add_argument("--witness-gib", type=float, default=508.5406)
    ap.add_argument("--out")
    args = ap.parse_args()
    target = dict(admission.TARGET)
    target["ELL"] = args.ell
    target["K_DEG"] = args.k_deg
    target["N_LIG"] = args.n_lig
    target["T_QUERIES"] = args.columns
    cfg = core.LigeroConfig(**target)
    if args.rows <= 0 or not 0 < args.columns <= cfg.N_LIG:
        ap.error("rows must be positive and columns must be in [1, N_LIG]")

    seed = core._master_seed_to_cuda(b"sampled-rs-preflight-seed-v1!!!!")
    values = torch.randint(
        0, 1 << 62, (args.rows * cfg.ELL,), dtype=torch.int64,
        device="cuda").view(torch.uint64)
    var = core.Variable("sampled_rs_preflight", length=values.numel())
    columns = list(range(0, cfg.N_LIG,
                         max(1, cfg.N_LIG // args.columns)))[:args.columns]

    # Compile/warm the exact encode path outside measurement.
    core.encode_messages(values[:cfg.ELL].view(1, cfg.ELL), cfg,
                         master_seed=seed, row_offset=0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def commit():
        acc = core._make_merkle_acc(cfg.N_LIG, args.rows)
        core._stream_phase([var], {var: values}, cfg, master_seed=seed,
                           abs_row_offset=0, merkle_acc=acc)
        return core._finalize_merkle_artifact(acc)

    commit_s, artifact = timed(commit)

    def open_columns():
        sink = HashingColumnSink(args.rows, columns, 0)
        core._stream_phase(
            [var], {var: values}, cfg, master_seed=seed, abs_row_offset=0,
            columns_at=columns, column_sink=sink)
        return sink.finish_with_digests()

    open_s, (opened, opened_digests) = timed(open_columns)

    def verify():
        for k, column in enumerate(columns):
            path = core.merkle_path(artifact.levels, column)
            if not core.merkle_verify(opened_digests[k], path, artifact.root,
                                      column, cfg.N_LIG):
                raise RuntimeError(f"Merkle opening failed at column {column}")

    verify_s, _ = timed(verify)
    witness_rows = args.witness_gib * (2 ** 30) / (cfg.ELL * 8)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "geometry": {
            "ELL": cfg.ELL, "K_DEG": cfg.K_DEG, "N_LIG": cfg.N_LIG,
            "rows": args.rows, "columns": len(columns),
        },
        "commit_s": commit_s,
        "open_s": open_s,
        "verify_s": verify_s,
        "commit_rows_s": args.rows / commit_s,
        "open_rows_s": args.rows / open_s,
        "projected_witness_rows": witness_rows,
        "projected_full_commit_s": witness_rows / (args.rows / commit_s),
        "projected_full_open_s": witness_rows / (args.rows / open_s),
        "second_full_lde_for_opening": True,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "root": artifact.root.hex(),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        part = pathlib.Path(str(out) + ".part")
        part.write_text(payload)
        part.replace(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
