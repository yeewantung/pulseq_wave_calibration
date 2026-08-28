#!/usr/bin/env python3
"""Bind explicit visual approval to one completed Wave lambda-zero run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from bart_cfl import sha256_file
from checkpoint_io import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-zero-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confirm-reconstruction-and-maps-reviewed",
        required=True,
        action="store_true",
    )
    parser.add_argument("--notes", default="Lambda-zero reconstruction and maps look good.")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _reviewed_artifacts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    paths = [Path(manifest["nifti"]["central_slice_quicklook"])]
    paths.extend(Path(value) for value in manifest["ecalib"]["diagnostic_montages"])
    if len(paths) != 3:
        raise ValueError("Expected one reconstruction quicklook and two map montages.")
    records = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        records.append(
            {
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_reconstruction_and_maps_reviewed:
        raise ValueError("Explicit visual-review confirmation is required.")
    manifest_path = args.lambda_zero_manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest = _read_json(manifest_path)
    config = manifest.get("config", {})
    command = manifest.get("wave_lambda0", {}).get("command", [])
    if manifest.get("status") != "lambda0_complete_awaiting_visual_review":
        raise ValueError("Lambda-zero reconstruction is not complete.")
    if float(config.get("ecalib_crop", -1)) != 0.6:
        raise ValueError("Visual approval requires the crop-0.6 ESPIRiT maps.")
    if config.get("gpu_wave_reconstruction") is not True or "-g" not in command:
        raise ValueError("Lambda zero did not record GPU BART wave -g.")

    manifest_record = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    approval = {
        "format_version": 1,
        "status": "approved_for_regularization_sweep",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "lambda_zero_manifest": manifest_record,
        "reviewed_artifacts": _reviewed_artifacts(manifest),
        "decision_fields": {
            "lambda_zero_reconstruction_visually_approved": True,
            "espirit_map_magnitude_visually_approved": True,
            "espirit_map_phase_visually_approved": True,
            "notes": args.notes,
        },
    }
    if output_path.is_file():
        existing = _read_json(output_path)
        if (
            existing.get("status") == approval["status"]
            and existing.get("lambda_zero_manifest") == manifest_record
            and existing.get("reviewed_artifacts") == approval["reviewed_artifacts"]
            and existing.get("decision_fields") == approval["decision_fields"]
        ):
            print(f"Reusing lambda-zero visual approval: {output_path}")
            return existing
        raise FileExistsError(f"Refusing to overwrite a different approval: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, approval)
    print(f"Recorded lambda-zero visual approval: {output_path}")
    return approval


def main(argv: Sequence[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
