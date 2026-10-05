#!/usr/bin/env python3
"""Validate, resume, and finalize one standard-PCA reconstruction branch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.reconstruction_state import (  # noqa: E402
    record_completed_reconstruction,
    reconstruction_status,
)


def _parser() -> argparse.ArgumentParser:
    """Build the reconstruction-state command-line parser.

    Returns:
        Parser for status inspection and completion recording.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "record"))
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--prepared-manifest", required=True, type=Path)
    parser.add_argument("--normal-manifest", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--method", required=True, choices=("fista", "wavelet"))
    parser.add_argument("--regularization", required=True, type=float)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--psf", required=True, action="append", type=Path)
    parser.add_argument("--kspace", required=True, action="append", type=Path)
    parser.add_argument("--image", required=True, action="append", type=Path)
    parser.add_argument("--command-record", required=True, action="append", type=Path)
    parser.add_argument("--expected-command", required=True, action="append")
    parser.add_argument("--nifti-directory", required=True, type=Path)
    return parser


def _request(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed options into a reconstruction-state request.

    Args:
        args: Parsed command-line namespace.

    Returns:
        Keyword mapping for the reconstruction-state API.
    """

    return {
        "prepared_manifest": args.prepared_manifest.expanduser().resolve(),
        "normal_manifest": args.normal_manifest.expanduser().resolve(),
        "profile": args.profile,
        "case": args.case,
        "branch": args.branch,
        "method": args.method,
        "regularization": args.regularization,
        "maps": args.maps.expanduser().resolve(),
        "psfs": [path.expanduser().resolve() for path in args.psf],
        "kspaces": [path.expanduser().resolve() for path in args.kspace],
        "images": [path.expanduser().resolve() for path in args.image],
        "command_records": [
            path.expanduser().resolve() for path in args.command_record
        ],
        "expected_commands": args.expected_command,
        "nifti_directory": args.nifti_directory.expanduser().resolve(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect or record one reconstruction branch.

    Args:
        argv: Optional argument vector.

    Returns:
        Zero after printing status or recording completion.
    """

    args = _parser().parse_args(argv)
    request = _request(args)
    if args.action == "status":
        print(reconstruction_status(args.run_manifest, **request))
    else:
        record_completed_reconstruction(args.run_manifest, **request)
        print("complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
