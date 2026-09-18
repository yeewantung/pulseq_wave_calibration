#!/usr/bin/env python3
"""Prepare manifest-bound ROVir inputs for all MPRAGE retrospective cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.rovir_retro import prepare_mprage_rovir_retro  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare all ROVir retro inputs and print their manifests.

    Args:
        argv: Optional arguments; ``None`` reads process arguments.

    Returns:
        Zero on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_mprage_rovir_retro(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
