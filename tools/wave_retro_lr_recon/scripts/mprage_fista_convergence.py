#!/usr/bin/env python3
"""Validate and record the logged FISTA-r0 convergence control.

The control repeats the accepted PCA-12 one-map FISTA-r0 reconstruction with a
larger iteration cap. This command never launches BART: the reviewed shell
entry point ``sample_mprage_fista_convergence.sh`` runs the single ``bart wave``
command between ``validate`` and ``record``. Implementation:
``wave_retro_lr.fista_convergence``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr import csm_consistency, fista_convergence, rovir_feasibility  # noqa: E402

STAGE_HELP = {
    "print-command": "print the command derived from the accepted record, without reading data",
    "validate": "bind the accepted baseline, check the shell command, and classify existing outputs",
    "record": "verify the finished run and write the convergence manifest",
    "summary": "print the convergence and resource summary of a recorded control",
    "residual-figure": "plot the recorded residual trace into qc/figures",
}


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Parser with one subcommand per stage.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, text in STAGE_HELP.items():
        command = stages.add_parser(stage, help=text)
        command.add_argument("accepted_normal_root", type=Path, help="Accepted normal root (read only).")
        command.add_argument("output_root", type=Path, help="Convergence-control output root.")
        command.add_argument("--iterations", type=int, required=True, help="FISTA iteration cap of the control.")
        if stage != "print-command":
            command.add_argument(
                "--protected-root",
                type=Path,
                action="append",
                default=[],
                help="Root that the output must not overlap; repeatable.",
            )
        if stage == "validate":
            command.add_argument("--command-text", required=True, help="Command the shell will run.")
        if stage == "record":
            command.add_argument("--environment-log", type=Path, help="Shell log of this invocation.")
    return parser


def _summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the convergence and resource summary of a manifest.

    Args:
        manifest: Convergence manifest.

    Returns:
        Final count, tolerance stop, last residual, BART timing, and resources.
    """
    convergence = manifest["convergence"]
    keys = ("final_iteration_count", "iteration_cap", "stopped_by_tolerance", "last_residual",
            "max_eigenvalue", "reconstruction_time_s", "total_time_s")
    return {
        "status": manifest["status"],
        **{key: convergence[key] for key in keys},
        "resources": manifest["resources"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one convergence-control stage.

    Args:
        argv: Optional command arguments; ``None`` reads process arguments.

    Returns:
        Zero after the stage completes.
    """
    args = _parser().parse_args(argv)
    if args.stage == "print-command":
        print(
            csm_consistency.format_command(
                fista_convergence.convergence_argv(args.accepted_normal_root, args.output_root, args.iterations)
            )
        )
        return 0
    if args.stage == "validate":
        result = fista_convergence.validate_control(
            args.accepted_normal_root,
            args.output_root,
            args.iterations,
            args.command_text,
            args.protected_root,
        )
        # The shell reads this single line to decide whether to run BART.
        print(f"STATE={result['state']}")
        return 0
    paths = fista_convergence.layout(args.output_root, args.iterations)
    if args.stage == "record":
        manifest = fista_convergence.record_control(
            args.accepted_normal_root,
            args.output_root,
            args.iterations,
            environment_log=args.environment_log,
            protected_roots=args.protected_root,
        )
        print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
        return 0
    manifest = rovir_feasibility._read_json(paths["manifest"])
    # Reports are given only for a control that still meets its full contract.
    fista_convergence.verify_control(
        manifest, args.accepted_normal_root, args.output_root, args.iterations, args.protected_root
    )
    if args.stage == "summary":
        print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
        return 0
    record = fista_convergence.write_residual_figure(manifest, paths["residual_figure"])
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
