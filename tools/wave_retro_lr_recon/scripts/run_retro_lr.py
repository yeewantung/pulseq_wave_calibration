#!/usr/bin/env python3
"""Run config-driven retrospective PE low-resolution Wave-MPRAGE cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_ROOT.parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.pipeline import run_config  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path, help="Dataset and case JSON.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only", action="store_true", help="Check paths, headers, geometry, masks, and cases."
    )
    mode.add_argument(
        "--prepare-only", action="store_true", help="Create case BART inputs but do not reconstruct."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse matching prepared or completed case manifests."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_config(
        args.config,
        repo_root=REPO_ROOT,
        validate_only=args.validate_only,
        prepare_only=args.prepare_only,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
