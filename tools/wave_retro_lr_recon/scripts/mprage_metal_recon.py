#!/usr/bin/env python3
"""Opt-in CPU-only metal CSM continuation and native R3x1 MPRAGE reconstruction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parser() -> argparse.ArgumentParser:
    """Return the explicit-opt-in CLI parser; no experiment paths or spatial defaults are inferred."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--metal-csm-extrapolation", action="store_true", help="Explicit metal-only experimental opt-in")
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--reference-nifti", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True, help="Reviewed case-specific parameter JSON")
    result.add_argument("--cpu-threads", type=int, choices=range(1, 9), default=8)
    result.add_argument("--prepare-only", action="store_true", help="Prepare maps without launching BART")
    return result


def interrupted(signum: int, frame: object) -> None:
    """Raise on SIGTERM so the runner terminates only its own BART child and records failure."""
    raise InterruptedError(f"Metal diagnostic driver received signal {signum}")


def main(argv: list[str] | None = None) -> int:
    """Run one explicit user request; return zero only after validated preparation or full completion."""
    args = parser().parse_args(argv)
    if not args.metal_csm_extrapolation:
        raise ValueError("Disabled by default: supply --metal-csm-extrapolation for reviewed metal cases only")
    # Import numerical libraries only after explicit opt-in and CPU environment setup.
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    from wave_retro_lr.metal_csm import MetalCSMParameters
    from wave_retro_lr.metal_runner import configure_idle_process, run_metal_workflow
    from wave_retro_lr.metal_workflow import read_json

    settings = MetalCSMParameters.from_dict(read_json(args.config))
    configure_idle_process()
    signal.signal(signal.SIGTERM, interrupted)
    result = run_metal_workflow(args.source_manifest, args.reference_nifti, args.output, settings,
                               enabled=True, prepare_only=args.prepare_only, threads=args.cpu_threads)
    print(f"{result['status']}: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
