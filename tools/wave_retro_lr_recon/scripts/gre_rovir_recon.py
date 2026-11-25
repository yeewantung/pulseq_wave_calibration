#!/usr/bin/env python3
"""Experimental Wave-GRE ROVir reconstruction with box and ellipsoid null regions.

Subcommands (all resumable; completed stages are validated and reused):

  inspect  Export physical-coil ACS/Wave k-space from TWIX and write ACS RSS
           views with native index ticks for placing the null region.
  roi      inspect + null-region masks, ``bart rovir`` transform, overlays,
           and the signal/null-region energy curve (no reconstruction).
  run      roi + ROVir projection, ecalib, BART Wave for the normal case and
           retrospective cases, NIfTI export, and standard-vs-ROVir QC.

Null regions are given in native BART indices (RO, LIN with 0 = anterior, PAR
with 0 = inferior). Ellipsoid radii are in millimetres:

  --null-ellipsoid "ro=125,lin=-20,par=-10,radius_mm=90"
  --null-ellipsoid "ro=125,lin=-20,par=-10,ro_radius_mm=80,lin_radius_mm=100,par_radius_mm=70"
  --null-box "ro=0:249,lin=0:30,par=0:10"

Several boxes and ellipsoids may be combined; their union is nulled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.gre_rovir import (  # noqa: E402
    BRANCH_LAMBDAS,
    DEFAULT_BRANCHES,
    DEFAULT_RETRO_CASES,
    DEFAULT_ROVIR_VIRTUAL_COILS,
    SUPPORTED_RETRO_CASES,
    export_physical_inputs,
    prepare_roi,
    prepare_variant,
    reconstruct_variant,
    write_comparison_qc,
)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line interface.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("inspect", "roi", "run"))
    parser.add_argument("twix", type=Path, help="Measured Wave-GRE TWIX file.")
    parser.add_argument("output_root", type=Path, help="User-approved GRE ROVir output root.")
    parser.add_argument("sequence", type=Path, help="Matching Wave-GRE Pulseq sequence.")
    parser.add_argument(
        "--standard-root", type=Path, required=True,
        help="Completed standard-PCA tree (for example .../recon_gre/vcc24) of the same scan.",
    )
    parser.add_argument("--null-box", action="append", default=[], help="Inclusive native-index box; repeatable.")
    parser.add_argument("--null-ellipsoid", action="append", default=[], help="Sphere/ellipsoid; repeatable.")
    parser.add_argument("--null-region-file", type=Path, help='JSON {"boxes": [...], "ellipsoids": [...]}.')
    parser.add_argument("--virtual-coils", type=int, default=DEFAULT_ROVIR_VIRTUAL_COILS)
    parser.add_argument(
        "--retro-case", action="append", choices=SUPPORTED_RETRO_CASES,
        help=f"Retrospective same-grid case; repeatable (default: {', '.join(DEFAULT_RETRO_CASES)}).",
    )
    parser.add_argument("--no-retro", action="store_true", help="Reconstruct only the normal (prospective) case.")
    parser.add_argument(
        "--branch", action="append", choices=tuple(BRANCH_LAMBDAS),
        help=f"Reconstruction branch; repeatable (default: {', '.join(DEFAULT_BRANCHES)}).",
    )
    parser.add_argument("--ecalib-crop", type=float, help="ESPIRiT crop; default inherits the standard tree.")
    parser.add_argument("-g", "--gpu", action="store_true", help="Use BART's GPU Wave path.")
    parser.add_argument("--bart", default="bart", help="BART executable (default: bart from PATH).")
    return parser


def _null_regions(args: argparse.Namespace) -> tuple[list[object], list[object]]:
    """Collect null boxes and ellipsoids from inline options and a JSON file.

    Args:
        args: Parsed arguments.

    Returns:
        Box and ellipsoid specifications.

    Raises:
        SystemExit: If no region is given for ``roi``/``run``.
    """
    boxes: list[object] = list(args.null_box)
    ellipsoids: list[object] = list(args.null_ellipsoid)
    if args.null_region_file is not None:
        payload = json.loads(args.null_region_file.read_text(encoding="utf-8"))
        boxes.extend(payload.get("boxes", []))
        ellipsoids.extend(payload.get("ellipsoids", []))
    if not boxes and not ellipsoids:
        raise SystemExit("Error: roi/run require --null-box, --null-ellipsoid, or --null-region-file.")
    return boxes, ellipsoids


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested GRE ROVir stage.

    Args:
        argv: Optional argument vector.

    Returns:
        Zero on success.
    """
    args = _parser().parse_args(argv)
    export_physical_inputs(args.twix, args.sequence, args.standard_root, args.output_root, bart=args.bart)
    print(f"Physical inputs ready: {args.output_root.resolve() / 'physical_inputs'}", flush=True)
    if args.stage == "inspect":
        print("Review physical_inputs/figures/acs_rss_*.png to place the null region.")
        return 0
    boxes, ellipsoids = _null_regions(args)
    roi = prepare_roi(args.output_root, boxes=boxes, ellipsoids=ellipsoids, bart=args.bart)
    region_id = roi["region"]["region_id"]
    print(f"ROI {region_id}: {roi['region']['union_voxel_count']} null voxels", flush=True)
    for entry in roi["energy_curve"]["channel_counts"]:
        print(
            f"  Ncc {entry['virtual_coils']:2d}: kept outside {entry['signal_retention_fraction']:.4f}, "
            f"remaining in null region {entry['interference_remaining_fraction']:.4f}"
        )
    if args.stage == "roi":
        return 0
    retro = () if args.no_retro else tuple(args.retro_case or DEFAULT_RETRO_CASES)
    prepare_variant(
        args.output_root, region_id, virtual_coils=args.virtual_coils,
        retro_cases=retro, ecalib_crop=args.ecalib_crop, bart=args.bart,
    )
    results = reconstruct_variant(
        args.output_root, region_id, args.twix, args.sequence,
        virtual_coils=args.virtual_coils, branches=tuple(args.branch or DEFAULT_BRANCHES),
        gpu=args.gpu, bart=args.bart,
    )
    for result in results:
        print(f"Reconstructed {result['case_id']}/{result['branch']}: {result['nifti']}")
    qc = write_comparison_qc(args.output_root, region_id, virtual_coils=args.virtual_coils)
    for pair in qc["pairs"]:
        print(f"QC {pair['case_id']}/{pair['rovir_branch']}: {json.dumps(pair['metrics'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
