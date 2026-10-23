#!/usr/bin/env python3
"""Prepare an isolated uncropped CSM cache and single-map crop comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wave_retro_lr.csm_cache import run_diagnostics  # noqa: E402


def parser() -> argparse.ArgumentParser:
    """Return the diagnostic-only CLI; no production reconstruction options are exposed."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepared-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True,
                        help="User-confirmed named directory beneath diagnostics/")
    result.add_argument("--hard-crops", type=float, nargs="+", default=[.6])
    result.add_argument("--soft-crops", type=float, nargs="*", default=[],
                        help="Reviewed BART 1.0 single-map soft cropping (not multi-map Soft-SENSE)")
    result.add_argument("--materialize", action="store_true",
                        help="Also write full coil_sens for every candidate; costs one CSM volume each")
    result.add_argument("--slices", type=int, nargs=3, metavar=("RO", "LIN", "PAR"),
                        help="Zero-based logical slices; defaults to native-grid centers")
    result.add_argument("--cpu-threads", type=int, default=1)
    result.add_argument("--check-only", action="store_true", help="Read-only validation; no BART execution")
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the requested cache diagnostics, print concise JSON, and return a shell status."""
    args = parser().parse_args(argv)
    try:
        variants = [("hard", x) for x in args.hard_crops] + [("soft_bart_1_0", x) for x in args.soft_crops]
        result = run_diagnostics(args.prepared_manifest, args.output, variants,
                                 threads=args.cpu_threads, materialize=args.materialize,
                                 check_only=args.check_only, slices=args.slices)
        if args.check_only:
            result = {"check_only": True, "output": result["output"],
                      "source_kind": result["source"]["kind"],
                      "virtual_coils": result["source"]["virtual_coils"],
                      "acs_sha256": result["source"]["acs"]["payload_sha256"],
                      "estimated_additional_payload_GiB": result["estimated_additional_payload_bytes"]/1024**3,
                      "variants": variants, "materialize": args.materialize}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, MemoryError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
