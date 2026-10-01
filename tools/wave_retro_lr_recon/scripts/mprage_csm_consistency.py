#!/usr/bin/env python3
"""Run one Python stage of the set-4 coil-sensitivity consistency diagnostics.

The stages validate, record, and analyse calibration data for one accepted
Wave-MPRAGE normal root. This command never launches BART: the reviewed shell
entry point ``sample_mprage_csm_consistency.sh`` runs the only BART command,
the diagnostic ``bart ecalib -m 2 -c 0``, between ``validate`` and
``record-calibration``. Implementation: ``wave_retro_lr.csm_consistency``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr import csm_consistency  # noqa: E402

STAGE_HELP = {
    "prepare": "validate sources, export the physical set-4 ACS, and record noise and channel identity",
    "validate": "revalidate every prepared input without writing anything",
    "record-calibration": "validate and record the shell-run diagnostic two-map ecalib outputs",
    "roi-template": "export the canonical-RAS reference and an empty five-label template",
    "diagnose": "compute diagnostics, figures, and the report from reviewed ROIs",
    "print-calibration-command": "print the exact diagnostic BART command without running it",
}
# Stages that can write a manifest and therefore record the shell environment log.
ENVIRONMENT_LOG_STAGES = ("prepare", "record-calibration", "roi-template", "diagnose")


def _parser() -> argparse.ArgumentParser:
    """Build the stage-based command interface.

    Returns:
        Configured argument parser with one subcommand per stage.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, help_text in STAGE_HELP.items():
        command = stages.add_parser(stage, help=help_text, description=help_text)
        command.add_argument("twix", type=Path, help="Measured Wave-MPRAGE TWIX file.")
        command.add_argument("sequence", type=Path, help="Matching Pulseq sequence file.")
        command.add_argument(
            "accepted_normal_root",
            type=Path,
            help="Accepted normal root holding the one-map c=0 CSM.",
        )
        command.add_argument(
            "output_root", type=Path, help="User-approved diagnostic output root."
        )
        if stage in ENVIRONMENT_LOG_STAGES:
            command.add_argument(
                "--environment-log",
                type=Path,
                help="Shell log of this invocation under OUTPUT_ROOT/logs/environment; "
                "recorded only when the stage writes a new manifest.",
            )
        if stage == "diagnose":
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument(
                "--roi-labels",
                type=Path,
                help="Reviewed five-label NIfTI drawn on rois/template.",
            )
            source.add_argument(
                "--roi-box",
                action="append",
                help="Inclusive LABEL=ro=a:b,lin=c:d,par=e:f box; repeatable.",
            )
    return parser


def _summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a stage manifest to a short printable summary.

    Args:
        manifest: Stage manifest returned by the library.

    Returns:
        Status, format version, and the most relevant review fields.
    """
    summary: dict[str, Any] = {
        "status": manifest.get("status"),
        "format_version": manifest.get("format_version"),
    }
    for key in ("report", "reviewed_label_destination", "map_count", "diagnostic_only"):
        if key in manifest:
            summary[key] = manifest[key]
    if "noise_model" in manifest:
        summary["rnr_status"] = manifest["noise_model"].get("rnr_status")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run one diagnostics stage.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after the stage completes.
    """
    args = _parser().parse_args(argv)
    positional = (args.twix, args.sequence, args.accepted_normal_root, args.output_root)
    if args.stage == "print-calibration-command":
        # The TWIX and sequence arguments are accepted for a uniform interface.
        print(
            csm_consistency.format_command(
                csm_consistency.two_map_ecalib_argv(
                    args.accepted_normal_root, args.output_root
                )
            )
        )
        return 0
    log = getattr(args, "environment_log", None)
    if args.stage == "prepare":
        manifest = csm_consistency.prepare_csm_consistency(*positional, environment_log=log)
    elif args.stage == "validate":
        manifest = csm_consistency.load_prepared_inputs(*positional)
    elif args.stage == "record-calibration":
        manifest = csm_consistency.record_two_map_calibration(*positional, environment_log=log)
    elif args.stage == "roi-template":
        manifest = csm_consistency.write_roi_template(*positional, environment_log=log)
    else:
        manifest = csm_consistency.diagnose_csm_consistency(
            *positional,
            labels_path=args.roi_labels,
            boxes=tuple(args.roi_box or ()),
            environment_log=log,
        )
    print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
