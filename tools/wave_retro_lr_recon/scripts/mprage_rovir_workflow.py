#!/usr/bin/env python3
"""Run one validated Python stage of the public MPRAGE ROVir workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.rovir_workflow import (  # noqa: E402
    approve_and_prepare_solver,
    finalize_normal_rovir,
    finish_inspection,
    load_recommended_boxes,
    normal_rovir_context,
    prepare_inspection,
    prepare_reviewed_candidate,
    record_transform_and_prepare_reconstruction,
)


def _parser() -> argparse.ArgumentParser:
    """Build the internal manifest-backed workflow parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("context", "inspect-prepare", "candidate-id"):
        command = commands.add_parser(stage)
        command.add_argument("output_root", type=Path)
        if stage == "context":
            command.add_argument("--field", default=None)
    finish = commands.add_parser("inspect-finish")
    finish.add_argument("output_root", type=Path)
    finish.add_argument("bart_version_file", type=Path)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("output_root", type=Path)
    candidate.add_argument("--null-box", action="append")
    candidate.add_argument("--null-box-file", type=Path)
    candidate.add_argument("--use-recommended", action="store_true")
    candidate.add_argument("--id-only", action="store_true")
    approve = commands.add_parser("approve")
    approve.add_argument("output_root", type=Path)
    approve.add_argument("candidate_id")
    transform = commands.add_parser("transform-prepare")
    transform.add_argument("output_root", type=Path)
    transform.add_argument("bart_version_file", type=Path)
    transform.add_argument("virtual_coils", type=int)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("output_root", type=Path)
    finalize.add_argument("virtual_coils", type=int)
    return parser


def _boxes(args: argparse.Namespace) -> list[Any]:
    """Resolve exactly one ROI input mode from parsed arguments.

    Args:
        args: Candidate-stage argument namespace.

    Returns:
        Box specifications accepted by the workflow module.

    Raises:
        ValueError: If modes conflict or a file has invalid structure.
    """
    modes = int(bool(args.null_box)) + int(args.null_box_file is not None) + int(args.use_recommended)
    if modes != 1:
        raise ValueError(
            "Choose exactly one of repeatable --null-box, --null-box-file, or --use-recommended."
        )
    if args.null_box:
        return list(args.null_box)
    if args.use_recommended:
        return load_recommended_boxes(args.output_root)
    payload = json.loads(args.null_box_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("null_boxes")
    if not isinstance(payload, list) or not payload:
        raise ValueError("Null-box file must be a nonempty list or {'null_boxes': [...] }.")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one internal workflow stage and print JSON.

    Args:
        argv: Optional arguments; ``None`` reads process arguments.

    Returns:
        Zero after successful completion.
    """
    args = _parser().parse_args(argv)
    if args.stage == "context":
        result = normal_rovir_context(args.output_root)
    elif args.stage == "inspect-prepare":
        result = prepare_inspection(args.output_root)
    elif args.stage == "inspect-finish":
        result = finish_inspection(args.output_root, args.bart_version_file)
    elif args.stage == "candidate":
        result = prepare_reviewed_candidate(args.output_root, _boxes(args))
    elif args.stage == "candidate-id":
        context = normal_rovir_context(args.output_root)
        candidate_manifest = json.loads(
            (
                Path(context["feasibility_root"])
                / "masks"
                / "candidates"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        candidates = candidate_manifest.get("candidates", [])
        if len(candidates) != 1:
            raise ValueError("Expected exactly one current ROVir candidate.")
        print(candidates[0]["candidate_id"])
        return 0
    elif args.stage == "approve":
        result = approve_and_prepare_solver(args.output_root, args.candidate_id)
    elif args.stage == "transform-prepare":
        result = record_transform_and_prepare_reconstruction(
            args.output_root, args.bart_version_file, args.virtual_coils
        )
    elif args.stage == "finalize":
        result = finalize_normal_rovir(args.output_root, args.virtual_coils)
    else:  # pragma: no cover
        raise RuntimeError(f"Unhandled stage: {args.stage}")
    if args.stage == "candidate" and args.id_only:
        print(result["candidate_id"])
    elif args.stage == "context" and args.field is not None:
        if args.field not in result or isinstance(result[args.field], (dict, list)):
            raise ValueError(f"Context has no scalar field {args.field!r}.")
        print(result[args.field])
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
