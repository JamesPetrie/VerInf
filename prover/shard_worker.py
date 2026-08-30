"""Weight-split worker passes (multi-GPU milestone 1).

The two per-proof passes the protocol leaves on the ENROLLED weight block,
executed for one contiguous run of weight variables on one device:

  fold_run  — the test-polynomial sweep's share: interpolate each row
              (no codeword) and fold it into q_irs and q_lin. Returns the
              UN-finalised partials; the coordinator merges them into its
              own accumulators (exact field sums; the fused q_lin path sums
              eval-domain partials before its single inverse NTT).
  open_run  — the openings sweep's share: encode each row and extract the
              challenged columns into a ColumnSink for exactly this run's
              rows. Returns host tensors keyed by absolute row; the
              coordinator scatters them into the full W sink.

Both are thin wrappers over core._stream_phase with the same padding rule
the coordinator's sweep applies to a weight group (w_pad: the enrollment
seed at the block's logical offset), the run's true absolute row, and the
SAME stream_pk (the band index is immutable and complete after
_build_stream_packets, so a worker needs no sweep state). Nothing here
runs a tape op: weight variables resolve from `inputs` (tensors or lazy
loaders) independently of any activation.

`device` selects the CUDA device for the pass (torch.cuda.device context);
None runs on the current device — the single-GPU byte-identity gate runs
every role sequentially on cuda:0. Results are returned on the device they
were produced on; the coordinator's merge moves them.
"""
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch

import core


def _pad_for(w_pad, row_start: int):
    if w_pad is None:
        return None, None
    return w_pad[0], w_pad[1] + (row_start - w_pad[2])


def _ctx(device: Optional[str]):
    return torch.cuda.device(device) if device is not None else nullcontext()


def fold_run(weight_vars: List, inputs: Dict, cfg, master_seed_t: torch.Tensor,
             w_pad, lo: int, hi: int, r_irs_t: torch.Tensor,
             r_lin_seed: torch.Tensor, stream_pk, *,
             device: Optional[str] = None) -> Dict[str, Any]:
    """Fold weight_vars[lo:hi] into fresh q_irs / q_lin accumulators and
    return their partials: {'q_irs', 'q_eval', 'q_coeff'} (the q_lin pair
    holds whichever representations the fuse mode keeps)."""
    if hi <= lo:
        return dict(q_irs=None, q_eval=None, q_coeff=None)
    run = weight_vars[lo:hi]
    with _ctx(device):
        q_irs = core.QIrsAccumulator(r_irs_t, cfg)
        q_lin = core.QLinAccumulator(r_lin_seed, stream_pk, cfg)
        pad_seed, pad_off = _pad_for(w_pad, run[0].row_start)
        core._stream_phase(run, inputs, cfg, master_seed=master_seed_t,
                           abs_row_offset=run[0].row_start,
                           pad_seed=pad_seed, pad_row_offset=pad_off,
                           q_irs_acc=q_irs, q_lin_acc=q_lin)
        q_eval, q_coeff = q_lin.partials()
        return dict(q_irs=q_irs.q, q_eval=q_eval, q_coeff=q_coeff)


def open_run(weight_vars: List, inputs: Dict, cfg, master_seed_t: torch.Tensor,
             w_pad, lo: int, hi: int, Q_cols: List[int], *,
             device: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Encode weight_vars[lo:hi] and extract the challenged columns.
    Returns {'abs_row', 'n_rows', 'cols': {j: host tensor}} or None for an
    empty run."""
    if hi <= lo:
        return None
    run = weight_vars[lo:hi]
    n_rows = sum(v.n_rows(cfg.ELL) for v in run)
    with _ctx(device):
        sink = core.ColumnSink(n_rows, Q_cols, run[0].row_start)
        pad_seed, pad_off = _pad_for(w_pad, run[0].row_start)
        core._stream_phase(run, inputs, cfg, master_seed=master_seed_t,
                           abs_row_offset=run[0].row_start,
                           pad_seed=pad_seed, pad_row_offset=pad_off,
                           columns_at=Q_cols, column_sink=sink)
        cols = sink.finish()
    return dict(abs_row=run[0].row_start, n_rows=n_rows, cols=cols)
