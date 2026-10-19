#!/usr/bin/env python3
"""Dispatch explicit metal mode from the existing MPRAGE normal/retro launchers."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parser():
    """Return the standard preparation CLI extended with explicit metal/profile options."""
    from scripts.prepare_mprage_normal import _parser

    result = _parser()
    result.description = __doc__
    result.add_argument("--mode", choices=("normal", "retro", "r3x3"), required=True)
    result.add_argument("--metal-csm-extrapolation", action="store_true")
    result.add_argument("--metal-csm-config", type=Path,
                        help="Optional reviewed manual ROI; absent default OUTPUT_ROOT/metal_csm.local.json selects automatic mode")
    profiles = result.add_mutually_exclusive_group()
    for name in ("reg-full", "wavelet-only", "fista-only"):
        profiles.add_argument("--"+name, dest="profile", action="store_const", const=name)
    result.add_argument("--ecalib-crop", type=float, default=.6)
    result.add_argument("--r3-lambda", type=float)
    result.add_argument("--cpu-threads", type=int, choices=range(1, 9), default=8)
    result.add_argument("--metal-prepare-only", action="store_true")
    result.add_argument("-g", action="store_true", help="Use GPU for BART Wave; CSM preparation remains on CPU")
    result.add_argument("--rovir", action="store_true")
    result.add_argument("--include-legacy-lr", action="store_true")
    result.add_argument("--n4-bias-correction", action="store_true")
    return result


def main(argv=None) -> int:
    """Validate a metal request, prepare on CPU, then use the selected BART backend."""
    # Cap preparation threads without hiding the caller's CUDA selection for GPU BART.
    os.environ.update(OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    args = parser().parse_args(argv)
    from wave_retro_lr.metal_pipeline import run_pipeline, validate_options
    from wave_retro_lr.metal_runner import configure_idle_process
    from scripts.mprage_metal_recon import interrupted

    validate_options(args)
    if not args.g:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configure_idle_process(gpu=args.g)
    signal.signal(signal.SIGTERM, interrupted)
    result = run_pipeline(args)
    print(f"Metal MPRAGE {args.mode} complete: {result}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
