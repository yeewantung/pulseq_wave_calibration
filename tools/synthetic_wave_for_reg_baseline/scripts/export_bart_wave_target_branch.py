#!/usr/bin/env python3
"""Branch a new post-Wave sampling target from accepted full-Wave inputs.

This script does no Wave synthesis or reconstruction. It calls the existing
sampling-mask/BART export helpers, then links the already validated PSF and
measured ACS so the standard BART lambda-zero runner can own reconstruction.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bart_cfl import sha256_file
from checkpoint_io import write_json_atomic
from export_bart_wave_inputs import link_bart_pair
from sampling_mask import (
    historical_cartesian_with_full_pe1_acs_mask,
    write_masked_bart_kspace,
)
from wave_synthesis import logical_array_sha256, logical_bart_cfl_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-synthesis-manifest", required=True, type=Path)
    parser.add_argument("--source-bart-input-manifest", required=True, type=Path)
    parser.add_argument("--operator-validation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--pe1-acceleration", required=True, type=int)
    parser.add_argument("--pe2-acceleration", required=True, type=int)
    parser.add_argument("--pe1-residue", required=True, type=int)
    parser.add_argument("--pe2-residue", required=True, type=int)
    parser.add_argument("--acs-pe1-start", required=True, type=int)
    parser.add_argument("--acs-pe1-stop-exclusive", required=True, type=int)
    parser.add_argument("--pe2-chunk", type=int, default=8)
    parser.add_argument("--confirm-full-wave-reviewed", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate provenance and target-mask settings without writing outputs.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _manifest_record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _same_dataset(*manifests: dict[str, Any]) -> dict[str, Any]:
    records = [manifest.get("dataset_manifest", {}) for manifest in manifests]
    hashes = {record.get("sha256") for record in records}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("Source synthesis, BART inputs, and operator gate differ by dataset.")
    source = records[0]
    dataset_path = Path(source.get("path", "")).expanduser().resolve()
    if not dataset_path.is_file() or sha256_file(dataset_path) != source["sha256"]:
        raise ValueError("The shared source dataset manifest is missing or changed.")
    return source


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_id": args.target_id,
        "mask_kind": "cartesian_with_full_pe1_acs",
        "pe1_acceleration": args.pe1_acceleration,
        "pe2_acceleration": args.pe2_acceleration,
        "pe1_residue": args.pe1_residue,
        "pe2_residue": args.pe2_residue,
        "acs_pe1_start": args.acs_pe1_start,
        "acs_pe1_stop_exclusive": args.acs_pe1_stop_exclusive,
    }


def _complete_output_reusable(
    manifest_path: Path,
    *,
    config: dict[str, Any],
    source_records: dict[str, dict[str, str]],
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = _load_json(manifest_path, "target BART-input manifest")
        if manifest.get("status") != "calibration_kspace_ready_for_ecalib":
            return False
        if manifest.get("target_config") != config:
            return False
        if any(manifest.get(name) != record for name, record in source_records.items()):
            return False
        for key in ("masked_wave_kspace", "psf", "kspace_calib"):
            record = manifest[key]
            if sha256_file(Path(record["cfl"])) != record["cfl_sha256"]:
                return False
        mask = np.load(Path(manifest["sampling_mask"]["path"]), mmap_mode="r")
        return (
            logical_array_sha256(mask)
            == manifest["sampling_mask"]["logical_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_full_wave_reviewed:
        raise ValueError("Explicit --confirm-full-wave-reviewed is required.")
    if args.acs_pe1_stop_exclusive <= args.acs_pe1_start:
        raise ValueError("The ACS half-open interval is empty.")

    synthesis_path = args.source_synthesis_manifest.expanduser().resolve()
    source_bart_path = args.source_bart_input_manifest.expanduser().resolve()
    operator_path = args.operator_validation_manifest.expanduser().resolve()
    synthesis = _load_json(synthesis_path, "full-Wave synthesis manifest")
    source_bart = _load_json(source_bart_path, "source BART-input manifest")
    operator = _load_json(operator_path, "operator-validation manifest")
    if synthesis.get("status") != "awaiting_visual_review_before_mask_and_bart":
        raise ValueError("Full-Wave synthesis has not reached its review gate.")
    if source_bart.get("status") != "calibration_kspace_ready_for_ecalib":
        raise ValueError("Source BART inputs do not contain validated measured ACS.")
    if operator.get("status") != "passed":
        raise ValueError("The full-sampling Wave operator validation has not passed.")
    dataset_record = _same_dataset(synthesis, source_bart, operator)

    source_records = {
        "source_synthesis_manifest": _manifest_record(synthesis_path),
        "source_bart_input_manifest": _manifest_record(source_bart_path),
        "operator_validation_manifest": _manifest_record(operator_path),
    }
    if operator.get("synthesis_manifest") != source_records["source_synthesis_manifest"]:
        raise ValueError("Operator validation does not bind the selected synthesis manifest.")
    if source_bart.get("source_synthesis_manifest") != source_records[
        "source_synthesis_manifest"
    ]:
        raise ValueError("Source BART inputs do not bind the selected synthesis manifest.")

    full_wave = synthesis.get("full_wave_kspace", {})
    full_wave_path = Path(full_wave.get("path", "")).expanduser().resolve()
    full_wave_shape = tuple(int(value) for value in full_wave.get("shape", ()))
    if (
        full_wave.get("sampling_mask_applied") is not False
        or len(full_wave_shape) != 4
        or full_wave.get("dtype") != "complex64"
        or not full_wave_path.is_file()
    ):
        raise ValueError("The selected source is not valid unmasked full-Wave k-space.")

    source_psf = synthesis.get("psf", {})
    source_psf_base = Path(source_psf.get("bart_base", "")).expanduser().resolve()
    psf_shape = tuple(int(value) for value in source_psf.get("bart_shape", ()))
    if psf_shape != (*full_wave_shape[:3], 1, 1):
        raise ValueError("The source PSF and full-Wave k-space shapes disagree.")
    if logical_bart_cfl_sha256(source_psf_base, psf_shape) != source_psf.get(
        "logical_sha256"
    ):
        raise ValueError("The source PSF differs from its synthesis manifest.")

    source_calibration = source_bart.get("kspace_calib", {})
    source_calibration_base = Path(
        source_calibration.get("bart_base", "")
    ).expanduser().resolve()
    if sha256_file(source_calibration_base.with_suffix(".cfl")) != source_calibration.get(
        "bart_cfl_sha256"
    ):
        raise ValueError("The measured ACS differs from its source BART-input manifest.")
    requested_acs = [args.acs_pe1_start, args.acs_pe1_stop_exclusive]
    if source_calibration.get("pe1_lines_half_open") != requested_acs:
        raise ValueError("The requested target ACS differs from the validated measured ACS.")

    mask, mask_info = historical_cartesian_with_full_pe1_acs_mask(
        full_wave_shape[1:3],
        accelerations=(args.pe1_acceleration, args.pe2_acceleration),
        residues=(args.pe1_residue, args.pe2_residue),
        fully_sampled_pe1_lines=np.arange(
            args.acs_pe1_start, args.acs_pe1_stop_exclusive
        ),
    )
    config = _config(args)
    if args.check_only:
        preflight = {
            "status": "passed",
            "target_config": config,
            "dataset_manifest": dataset_record,
            **source_records,
            "sampling_mask": mask_info,
            "writes_performed": False,
        }
        print(json.dumps(preflight, indent=2))
        return preflight

    output_dir = args.output_dir.expanduser().resolve()
    bart_dir = output_dir / "bart_inputs"
    manifest_path = bart_dir / "manifest.json"
    if args.resume and _complete_output_reusable(
        manifest_path, config=config, source_records=source_records
    ):
        print(f"Reusing validated target branch: {output_dir}")
        return _load_json(manifest_path, "target BART-input manifest")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Target output is not safely reusable: {output_dir}")
    bart_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / "sampling_mask.npy"
    np.save(mask_path, mask)
    mask_info.update(
        {
            "path": str(mask_path),
            "dtype": str(mask.dtype),
            "logical_sha256": logical_array_sha256(mask),
            "applied_after_wave_encoding": True,
        }
    )
    masked = write_masked_bart_kspace(
        full_wave_path,
        mask,
        bart_dir / "wave_kspace",
        pe2_chunk=args.pe2_chunk,
    )
    masked["cfl_sha256"] = sha256_file(Path(masked["cfl"]))

    psf_base = bart_dir / "psf"
    calibration_base = bart_dir / "kspace_calib"
    link_bart_pair(source_psf_base, psf_base, replace=False)
    link_bart_pair(source_calibration_base, calibration_base, replace=False)
    psf_record = {
        "base": str(psf_base),
        "header": str(psf_base.with_suffix(".hdr")),
        "cfl": str(psf_base.with_suffix(".cfl")),
        "shape": list(psf_shape),
        "logical_sha256": logical_bart_cfl_sha256(psf_base, psf_shape),
        "cfl_sha256": sha256_file(psf_base.with_suffix(".cfl")),
        "symlink_source_base": str(source_psf_base),
    }
    calibration_record = {
        **source_calibration,
        "base": str(calibration_base),
        "header": str(calibration_base.with_suffix(".hdr")),
        "cfl": str(calibration_base.with_suffix(".cfl")),
        "cfl_sha256": sha256_file(calibration_base.with_suffix(".cfl")),
        "symlink_source_base": str(source_calibration_base),
    }
    calibration_record.pop("bart_base", None)
    calibration_record.pop("bart_cfl_sha256", None)

    manifest = {
        "format_version": 1,
        "format": "BART CFL",
        "status": "calibration_kspace_ready_for_ecalib",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "post-Wave target branch without repeated Wave synthesis",
        "target_config": config,
        "mask_application_order": "accepted full Wave encoding -> target mask",
        "full_wave_visual_review_confirmed": True,
        "dataset_manifest": dataset_record,
        **source_records,
        "sampling_mask": mask_info,
        "full_wave_kspace": full_wave,
        "masked_wave_kspace": masked,
        "psf": psf_record,
        "kspace_calib": calibration_record,
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "echoes": [
            {
                "echo": 1,
                "wave_kspace": "wave_kspace",
                "wave_kspace_shape": masked["shape"],
                "wave_kspace_norm": masked["norm"],
                "psf": "psf",
                "psf_shape": list(psf_shape),
            }
        ],
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        output_dir / "target_manifest.json",
        {
            "format_version": 1,
            "status": "target_bart_inputs_ready",
            "target_config": config,
            "bart_input_manifest": _manifest_record(manifest_path),
            **source_records,
        },
    )
    print(f"Target BART inputs: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
