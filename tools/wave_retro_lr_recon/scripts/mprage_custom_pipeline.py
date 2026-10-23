#!/usr/bin/env python3
"""Explicit custom-CSM dispatch for existing MPRAGE normal/retro launchers."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parser() -> argparse.ArgumentParser:
    """Return a strict MPRAGE custom-CSM parser with mutually exclusive profiles."""
    from scripts.prepare_mprage_normal import _parser
    result = _parser()
    result.description = __doc__
    result.set_defaults(virtual_coils=None)
    result.add_argument("--mode", choices=("normal", "retro", "r3x3"), required=True)
    result.add_argument("--custom-csm", required=True, type=Path, help="Completed hash-bound custom CSM JSON")
    result.add_argument("--custom-prepare-only", action="store_true", help="Prepare isolated cases/maps, never run BART")
    result.add_argument("--custom-check-only", action="store_true", help="Validate sources and maps read-only; create nothing")
    result.add_argument("--rovir", action="store_true")
    result.add_argument("-g", action="store_true", help="GPU BART only; map processing stays on CPU")
    result.add_argument("--cpu-threads", type=int, choices=range(1, 9), default=8)
    result.add_argument("--r3-lambda", type=float)
    profiles = result.add_mutually_exclusive_group()
    for name in ("reg-full", "wavelet-only", "fista-only"):
        profiles.add_argument("--"+name, dest="profile", action="store_const", const=name)
    return result


def main(argv=None) -> int:
    """Validate custom-mode options and run only the explicitly requested workflow.

    The root-only ROVir compatibility launcher derives acquisition paths from the
    supplied bound manifest. Unknown flags (including N4, metal and new ROVir ROI
    selection) fail rather than silently changing the selected forward model.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--root-only" in argv:
        index = argv.index("--root-only")
        root = argv[index+1]
        del argv[index:index+2]
        from wave_retro_lr.csm_cache import read_json
        option = next((v.split("=", 1)[1] for v in argv if v.startswith("--custom-csm=")), None)
        config = Path(option if option is not None else argv[argv.index("--custom-csm")+1])
        acquisition = read_json(config)["source"]["manifest_content"]["source"]
        argv = [acquisition["twix"]["path"], root, acquisition["sequence"]["path"], *argv]
    args = parser().parse_args(argv)
    if args.rovir and (args.psf_coefficient_processing != "sine-line" or any(
            getattr(args, f"psf_fit_{axis}_{edge}") is not None
            for axis in ("kx", "y", "z") for edge in ("min", "max"))):
        raise ValueError("Custom ROVir reuses its bound PSF; fitting overrides are not accepted")
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if not args.g:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from wave_retro_lr.metal_runner import configure_idle_process
    from wave_retro_lr.custom_pipeline import run_custom_pipeline
    configure_idle_process(gpu=args.g)
    print(f"Custom MPRAGE {args.mode}: {run_custom_pipeline(args)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, MemoryError, IndexError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
