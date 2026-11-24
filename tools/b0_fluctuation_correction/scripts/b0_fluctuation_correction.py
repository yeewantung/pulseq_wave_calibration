#!/usr/bin/env python3
"""B0-fluctuation correction for measured Wave-CAIPI GRE (consistency navigator).

Subcommands:

  estimate     Estimate per-line B0 fluctuations on a standard GRE tree
               (tools/wave_retro_lr_recon layout, e.g. <recon_gre>/vcc24) and
               write a corrected tree with the same layout plus estimation
               arrays and figures. GPU recommended.
  reconstruct  Run BART wave (FISTA-r0 and Wavelet 0.015) on the corrected
               tree, export NIfTI with the standard converter, and write
               standard-vs-corrected QC figures.
  simulate     Inject a known respiration-like fluctuation into data
               synthesized from the scan and report how well each method
               recovers it (validation; writes only to --output-dir).

Methods (--method):
  linear_field  uniform + linear gradients inside the Wave forward model (default)
  global        one phase per line for all coils
  coil_affine   per-coil phases constrained by coil centres of mass
  per_coil      free per-coil phases (Meineke & Nielsen, MRM 2019)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from b0_fluctuation.acquisition import timing_from_loop_order, timing_from_twix  # noqa: E402
from b0_fluctuation.pipeline import METHODS, RETRO_STRATEGIES, estimate_and_correct  # noqa: E402


def _add_estimation_options(parser: argparse.ArgumentParser) -> None:
    """Add options shared by ``estimate`` and ``simulate``.

    Args:
        parser: Subcommand parser.
    """
    parser.add_argument("--standard-root", type=Path, required=True, help="Standard GRE vccN tree.")
    parser.add_argument("--twix", type=Path, help="TWIX file (per-line MDH timestamps; preferred).")
    parser.add_argument("--repetition-time-s", type=float, help="TR for loop-order timing without --twix.")
    parser.add_argument("--axes", default="z", help="Gradient axes for linear_field/coil_affine: z, xyz, or none.")
    parser.add_argument("--iterations", type=int, default=20, help="Maximum navigator cycles.")
    parser.add_argument("--cg-iterations", type=int, default=30)
    parser.add_argument("--fwhm-ms", type=float, default=300.0, help="Temporal smoothing FWHM.")
    parser.add_argument("--gradient-fwhm-ms", type=float, default=600.0, help="Extra gradient smoothing (linear_field).")
    parser.add_argument("--relaxation", type=float, help="Update relaxation (default 2; use 1 for per_coil).")
    parser.add_argument("--device", default="cuda")


def _options(args: argparse.Namespace) -> dict[str, object]:
    """Translate CLI arguments into settings overrides.

    Args:
        args: Parsed arguments.

    Returns:
        Keyword overrides for ``build_settings``.
    """
    axes = () if args.axes.lower() == "none" else tuple(args.axes.lower().replace(",", ""))
    relaxation = args.relaxation
    if relaxation is None:
        relaxation = 1.0 if getattr(args, "method", None) == "per_coil" else 2.0
    return {
        "axes": axes,
        "outer_iterations": args.iterations,
        "cg_iterations": args.cg_iterations,
        "temporal_fwhm_s": args.fwhm_ms / 1000.0,
        "gradient_fwhm_s": args.gradient_fwhm_ms / 1000.0 if args.gradient_fwhm_ms > 0 else None,
        "relaxation": relaxation,
    }


def _parser() -> argparse.ArgumentParser:
    """Build the CLI.

    Returns:
        Argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    estimate = sub.add_parser("estimate", help="Estimate and write corrected inputs.")
    _add_estimation_options(estimate)
    estimate.add_argument("--output-root", type=Path, required=True)
    estimate.add_argument("--method", choices=METHODS, default="linear_field")
    estimate.add_argument("--reference-echo", type=int, help="One-based estimated echo (default: last).")
    estimate.add_argument("--retro-case", action="append", help="Retrospective case to correct (default native_r3x3).")
    estimate.add_argument("--no-retro", action="store_true")
    estimate.add_argument("--retro-strategy", choices=RETRO_STRATEGIES, default="from-normal")

    reconstruct = sub.add_parser("reconstruct", help="BART wave + NIfTI + QC on a corrected tree.")
    reconstruct.add_argument("--output-root", type=Path, required=True)
    reconstruct.add_argument("--twix", type=Path, required=True)
    reconstruct.add_argument("--sequence", type=Path, required=True)
    reconstruct.add_argument("--branch", action="append", choices=("fista_r0", "wavelet"))
    reconstruct.add_argument("-g", "--gpu", action="store_true")
    reconstruct.add_argument("--bart", default="bart")

    simulate = sub.add_parser("simulate", help="Injected-fluctuation validation on the real geometry.")
    _add_estimation_options(simulate)
    simulate.add_argument("--output-dir", type=Path, required=True)
    simulate.add_argument("--methods", default="linear_field,global")
    simulate.add_argument("--uniform-hz", type=float, default=1.5)
    simulate.add_argument("--gradient-hz-per-mm", type=float, default=0.01)
    simulate.add_argument("--frequency-hz", type=float, default=0.35)
    simulate.add_argument("--noise-scale", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the requested subcommand.

    Args:
        argv: Optional argument vector.

    Returns:
        Zero on success.
    """
    args = _parser().parse_args(argv)
    if args.command == "reconstruct":
        from b0_fluctuation.bart_recon import reconstruct_tree

        result = reconstruct_tree(args.output_root, args.twix, args.sequence,
                                  branches=tuple(args.branch or ("fista_r0", "wavelet")), gpu=args.gpu, bart=args.bart)
        for record in result["comparisons"]:
            print(f"QC {record['case_id']}/{record['branch']}: {record['figure']}")
        return 0
    if args.command == "estimate":
        retro = () if args.no_retro else tuple(args.retro_case or ("native_r3x3",))
        manifest = estimate_and_correct(
            args.standard_root, args.output_root, method=args.method, twix=args.twix,
            repetition_time_s=args.repetition_time_s, reference_echo=args.reference_echo,
            retro_cases=retro, retro_strategy=args.retro_strategy, device=args.device, **_options(args),
        )
        for case_id, summary in manifest["cases"].items():
            if "uniform_hz_std" in summary:
                print(f"{case_id}: uniform B0 fluctuation std {summary['uniform_hz_std']:.3f} Hz, "
                      f"peak-to-peak {summary['uniform_hz_peak_to_peak']:.3f} Hz")
        print(f"Corrected tree: {args.output_root}")
        return 0
    from b0_fluctuation.simulation import FluctuationModel
    from b0_fluctuation.standard_tree import load_case
    from b0_fluctuation.validation import run_simulation
    import numpy as np

    if args.twix is not None:
        timing = timing_from_twix(args.twix)
    else:
        mask = np.load(load_case(args.standard_root).inputs / "sampling_mask.npy")
        timing = timing_from_loop_order(mask, args.repetition_time_s)
    model = FluctuationModel(uniform_hz=args.uniform_hz, gradient_hz_per_mm=args.gradient_hz_per_mm, frequency_hz=args.frequency_hz)
    options = _options(args)
    run_simulation(args.standard_root, args.output_dir, timing, methods=tuple(args.methods.split(",")),
                   model=model, noise_scale=args.noise_scale, device=args.device, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
