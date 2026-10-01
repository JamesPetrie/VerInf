"""Weight-split worker passes (multi-GPU milestone 1).

The two per-proof passes the protocol leaves on the ENROLLED weight block,
executed for one contiguous run of weight variables on one device:

  fold_run  — the test-polynomial sweep's share: interpolate each row
              (no codeword) and fold it into q_irs and q_lin. Returns the
              UN-finalised partials; the coordinator merges them into its
              own accumulators (exact field sums; the fused q_lin path sums
              eval-domain partials before its single inverse NTT).
  open_run  — the openings sweep's share: encode each row and extract the
              challenged columns directly into the coordinator's W sink,
              one encode chunk at a time at its absolute row. The current
              in-process workers need no separate full-run host buffer.

Both are thin wrappers over core._stream_phase with the same padding rule
the coordinator's sweep applies to a weight group (w_pad: the enrollment
seed at the block's logical offset), the run's true absolute row, and the
SAME stream_pk (the band index is immutable and complete after
_build_stream_packets, so a worker needs no sweep state). Nothing here
runs a tape op: weight variables resolve from `inputs` (tensors or lazy
loaders) independently of any activation.

`device` selects the CUDA device for the pass (torch.cuda.device context);
None runs on the current device — the single-GPU byte-identity gate runs
every role sequentially on cuda:0. Fold partials are returned on the device
they were produced on; the coordinator's merge moves them. Opening chunks
write through to the shared host sink, whose final coverage check includes
every role's rows.
"""
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch

import core


def _pad_for(w_pad, row_start: int):
    if w_pad is None:
        return None, None
    return w_pad[0], w_pad[1] + (row_start - w_pad[2])


def validate_device(device: Optional[str]) -> None:
    """M1a runs every role on the coordinator's device. A different device
    is M1b work: the coins (r_irs, r_lin seed), master seed, w_pad seed,
    band-template tensors and the module caches all live on the
    coordinator's device and nothing moves them, so switching the current
    device here would either fault or silently read across devices (P2P).
    Refuse loudly instead of masking that."""
    if device is None:
        return
    want = torch.device(device)
    cur = torch.device("cuda", torch.cuda.current_device())
    if want.type != "cuda" or (want.index is not None and want.index != cur.index):
        raise NotImplementedError(
            f"weight-split worker on {want} while the coordinator runs on "
            f"{cur}: per-device coins/inputs are milestone M1b; M1a executes "
            f"every role on the coordinator's device (pass device=None)")


def _ctx(device: Optional[str]):
    """Use the same device check during preflight and worker execution."""
    validate_device(device)
    return nullcontext()


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
             column_sink: core.ColumnSink,
             device: Optional[str] = None) -> None:
    """Write weight_vars[lo:hi]'s challenged columns to the shared W sink.

    _stream_phase writes each encode chunk at its absolute row, so workers
    can arrive in any order without allocating or copying a full run's
    openings. The coordinator finishes the sink after every run completes.
    """
    if hi <= lo:
        return
    run = weight_vars[lo:hi]
    with _ctx(device):
        pad_seed, pad_off = _pad_for(w_pad, run[0].row_start)
        core._stream_phase(run, inputs, cfg, master_seed=master_seed_t,
                           abs_row_offset=run[0].row_start,
                           pad_seed=pad_seed, pad_row_offset=pad_off,
                           columns_at=Q_cols, column_sink=column_sink)
