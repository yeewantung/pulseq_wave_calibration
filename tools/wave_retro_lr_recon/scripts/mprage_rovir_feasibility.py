#!/usr/bin/env python3
"""Run one non-BART stage of the measured MPRAGE ROVir feasibility workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.rovir_feasibility import (  # noqa: E402
    approve_region_mask_candidate,
    derive_region_mask_candidates,
    export_manual_roi_annotation_nifti,
    export_mprage_physical_calibration,
    preflight_mprage_rovir_sources,
    prepare_masked_rovir_inputs,
    record_calibration_images,
    validate_manual_roi_annotation,
    write_rovir_transform_qc,
)


def _parser() -> argparse.ArgumentParser:
    """Build the explicit staged feasibility command interface.

    Returns:
        Configured top-level argument parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    commands = parser.add_subparsers(dest="stage", required=True)

    for name in ("preflight", "export-calibration"):
        stage = commands.add_parser(name)
        stage.add_argument("twix", type=Path)
        stage.add_argument("sequence", type=Path)
        stage.add_argument("normal_output_root", type=Path)
        stage.add_argument("feasibility_output_root", type=Path)

    images = commands.add_parser("record-images")
    images.add_argument("feasibility_output_root", type=Path)
    images.add_argument("bart_version_file", type=Path)

    manual_export = commands.add_parser("export-manual-roi")
    manual_export.add_argument("feasibility_output_root", type=Path)

    manual_validate = commands.add_parser("validate-manual-roi")
    manual_validate.add_argument("feasibility_output_root", type=Path)
    manual_validate.add_argument("--reviewed-labels", type=Path, default=None)

    masks = commands.add_parser("derive-masks")
    masks.add_argument("feasibility_output_root", type=Path)
    masks.add_argument("candidate_config", type=Path)

    approve = commands.add_parser("approve-mask")
    approve.add_argument("feasibility_output_root", type=Path)
    approve.add_argument("candidate_id")

    prepare = commands.add_parser("prepare-rovir-inputs")
    prepare.add_argument("feasibility_output_root", type=Path)

    qc = commands.add_parser("qc-transform")
    qc.add_argument("feasibility_output_root", type=Path)
    qc.add_argument("bart_version_file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute exactly one requested non-BART feasibility stage.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after the requested stage completes.
    """
    args = _parser().parse_args(argv)
    if args.stage == "preflight":
        result = preflight_mprage_rovir_sources(
            args.twix,
            args.sequence,
            args.normal_output_root,
            args.feasibility_output_root,
        )
    elif args.stage == "export-calibration":
        print(
            "Hashing the large TWIX source, then reading integrated refscan set 4; "
            "this stage can be quiet for several minutes.",
            file=sys.stderr,
            flush=True,
        )
        result = export_mprage_physical_calibration(
            args.twix,
            args.sequence,
            args.normal_output_root,
            args.feasibility_output_root,
        )
    elif args.stage == "record-images":
        result = record_calibration_images(
            args.feasibility_output_root, args.bart_version_file
        )
    elif args.stage == "export-manual-roi":
        result = export_manual_roi_annotation_nifti(args.feasibility_output_root)
    elif args.stage == "validate-manual-roi":
        result = validate_manual_roi_annotation(
            args.feasibility_output_root, args.reviewed_labels
        )
    elif args.stage == "derive-masks":
        result = derive_region_mask_candidates(
            args.feasibility_output_root, args.candidate_config
        )
    elif args.stage == "approve-mask":
        result = approve_region_mask_candidate(
            args.feasibility_output_root, args.candidate_id
        )
    elif args.stage == "prepare-rovir-inputs":
        result = prepare_masked_rovir_inputs(args.feasibility_output_root)
    elif args.stage == "qc-transform":
        result = write_rovir_transform_qc(
            args.feasibility_output_root, args.bart_version_file
        )
    else:  # pragma: no cover - argparse enforces the choices.
        raise RuntimeError(f"Unhandled stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
