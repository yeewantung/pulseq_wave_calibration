#!/usr/bin/env python3
"""Validate and display a resolved synthetic-Wave dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from dataset_manifest import DatasetManifestError, load_dataset_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Dataset manifest JSON to validate.")
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Also require the TWIX, Wave sequence, and enabled DICOM input to exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_dataset_manifest(args.manifest)
    if args.check_inputs:
        required = {
            "TWIX": (manifest.input_path("twix"), "file"),
            "Wave sequence": (manifest.input_path("wave_sequence"), "file"),
        }
        if manifest.dicom_enabled:
            required["DICOM directory"] = (manifest.dicom_directory, "directory")
        missing = [
            f"{label} not found: {path}"
            for label, (path, kind) in required.items()
            if (kind == "file" and not path.is_file())
            or (kind == "directory" and not path.is_dir())
        ]
        if missing:
            raise FileNotFoundError("\n".join(missing))

    print(json.dumps(manifest.resolved_contract(), indent=2))
    print(f"Manifest SHA-256: {manifest.sha256}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DatasetManifestError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
