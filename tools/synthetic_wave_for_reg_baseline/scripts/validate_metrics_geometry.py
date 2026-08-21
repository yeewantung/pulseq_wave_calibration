#!/usr/bin/env python3
"""Validate metric-reference provenance and exact candidate-grid geometry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from bart_cfl import sha256_file
from checkpoint_io import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-reference-manifest", required=True, type=Path)
    parser.add_argument(
        "--sweep-root",
        required=True,
        action="append",
        type=Path,
        help="Sweep directory containing one manifest-backed directory per case; repeatable.",
    )
    parser.add_argument(
        "--expected-case",
        required=True,
        action="append",
        help="Required case ID: wavelet:<lambda-label> or llr:block-<size>:<lambda-label>.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _verified_file(record: dict[str, Any], path_key: str, hash_key: str, label: str) -> Path:
    path = Path(record[path_key]).expanduser().resolve()
    expected = record[hash_key]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}: {path}")
    return path


def _geometry(image: nib.spatialimages.SpatialImage) -> dict[str, Any]:
    return {
        "shape": list(image.shape),
        "voxel_size_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "affine": np.asarray(image.affine, dtype=float).tolist(),
    }


def _require_same_geometry(
    candidate: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    label: str,
) -> float:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"{label} shape differs from reference: {candidate.shape} != {reference.shape}"
        )
    candidate_zooms = np.asarray(candidate.header.get_zooms()[:3], dtype=float)
    reference_zooms = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    if not np.allclose(candidate_zooms, reference_zooms, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{label} voxel size differs from reference: "
            f"{candidate_zooms.tolist()} != {reference_zooms.tolist()}"
        )
    orientation = tuple(nib.aff2axcodes(candidate.affine))
    reference_orientation = tuple(nib.aff2axcodes(reference.affine))
    if orientation != reference_orientation or orientation != ("R", "A", "S"):
        raise ValueError(
            f"{label} orientation differs or is not RAS: {orientation} != "
            f"{reference_orientation}"
        )
    difference = float(
        np.max(
            np.abs(
                np.asarray(candidate.affine, dtype=float)
                - np.asarray(reference.affine, dtype=float)
            )
        )
    )
    if difference > 1e-6:
        raise ValueError(f"{label} affine differs from reference by {difference:.9g}")
    return difference


def _finite_summary(image: nib.spatialimages.SpatialImage, label: str) -> dict[str, Any]:
    data = np.asarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f"{label} contains non-finite voxels")
    if not np.any(data != 0):
        raise ValueError(f"{label} is identically zero")
    return {
        "all_voxels_finite": True,
        "nonzero_voxel_count": int(np.count_nonzero(data)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def _case_id(manifest: dict[str, Any]) -> str:
    config = manifest.get("config", {})
    regularizer = config.get("regularizer")
    label = config.get("lambda_label")
    if regularizer == "wavelet" and isinstance(label, str):
        return f"wavelet:{label}"
    if regularizer == "llr" and isinstance(label, str):
        block_size = config.get("block_size")
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError(f"Invalid LLR block size: {block_size}")
        return f"llr:block-{block_size}:{label}"
    raise ValueError(f"Invalid regularization config: {config}")


def _magnitude_record(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    records = [
        record
        for record in manifest.get("nifti_outputs", [])
        if record.get("part") == "mag"
    ]
    if len(records) != 1:
        raise ValueError(
            f"Expected one magnitude NIfTI record, found {len(records)}: {manifest_path}"
        )
    return records[0]


def _discover_case_manifests(sweep_roots: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for root_input in sweep_roots:
        root = root_input.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Missing sweep root: {root}")
        paths.extend(sorted(root.glob("*/manifest.json")))
    if not paths:
        raise ValueError("No regularization case manifests were discovered")
    return paths


def _validate_reference(manifest_path: Path) -> tuple[dict[str, Any], Any, Any]:
    metrics = _load_json(manifest_path, "metrics-reference manifest")
    if metrics.get("status") != "approved_for_metrics":
        raise ValueError("Metrics-reference manifest is not approved_for_metrics")

    dataset_path = _verified_file(
        metrics["dataset"], "manifest", "manifest_sha256", "dataset manifest"
    )
    reference_record = metrics["ranking_reference"]
    if reference_record.get("kind") != "direct_fft_rss":
        raise ValueError("Ranking reference must be direct_fft_rss")
    reference_path = _verified_file(
        reference_record, "path", "sha256", "direct FFT RSS reference"
    )
    _verified_file(
        reference_record,
        "source_kspace",
        "source_kspace_sha256",
        "fully sampled source k-space",
    )

    mask_record = metrics["brain_mask"]
    if mask_record.get("usage") != "metrics_only":
        raise ValueError("Approved brain mask must be restricted to metrics_only")
    mask_path = _verified_file(mask_record, "path", "sha256", "brain mask")
    mask_manifest_path = _verified_file(
        mask_record, "manifest", "manifest_sha256", "brain-mask manifest"
    )
    mask_manifest = _load_json(mask_manifest_path, "brain-mask manifest")
    approval = mask_manifest.get("approval", {})
    if mask_manifest.get("status") != "approved_for_metrics" or not all(
        (
            approval.get("mask_boundary_visually_approved") is True,
            approval.get("left_right_orientation_visually_approved") is True,
        )
    ):
        raise ValueError("Brain-mask visual approval gate has not passed")

    reference = nib.load(str(reference_path))
    if tuple(nib.aff2axcodes(reference.affine)) != ("R", "A", "S"):
        raise ValueError("Direct FFT RSS reference is not canonical RAS")
    reference_summary = _finite_summary(reference, "direct FFT RSS reference")
    mask = nib.load(str(mask_path))
    mask_affine_difference = _require_same_geometry(mask, reference, "brain mask")
    mask_data = np.asarray(mask.dataobj)
    if not np.isfinite(mask_data).all() or not np.all(
        np.logical_or(mask_data == 0, mask_data == 1)
    ):
        raise ValueError("Brain mask must be finite and binary")
    if int(np.count_nonzero(mask_data)) != int(mask_record["voxel_count"]):
        raise ValueError("Brain-mask voxel count differs from approved manifest")

    return metrics, reference, {
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": metrics["dataset"]["manifest_sha256"],
        "reference": {
            "path": str(reference_path),
            "sha256": reference_record["sha256"],
            "geometry": _geometry(reference),
            **reference_summary,
        },
        "brain_mask": {
            "path": str(mask_path),
            "sha256": mask_record["sha256"],
            "manifest": str(mask_manifest_path),
            "manifest_sha256": mask_record["manifest_sha256"],
            "voxel_count": int(np.count_nonzero(mask_data)),
            "affine_max_abs_difference_from_reference": mask_affine_difference,
            "visual_approval": approval,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    metrics_path = args.metrics_reference_manifest.expanduser().resolve()
    metrics, reference, reference_report = _validate_reference(metrics_path)
    expected_ids = set(args.expected_case)
    if len(expected_ids) != len(args.expected_case):
        raise ValueError("Duplicate --expected-case values are not allowed")

    manifests = _discover_case_manifests(args.sweep_root)
    loaded = [(path, _load_json(path, "case manifest")) for path in manifests]
    discovered_ids = [_case_id(manifest) for _, manifest in loaded]
    if len(set(discovered_ids)) != len(discovered_ids):
        raise ValueError(f"Duplicate discovered case IDs: {discovered_ids}")
    if set(discovered_ids) != expected_ids:
        raise ValueError(
            "Discovered cases differ from the required grid; "
            f"missing={sorted(expected_ids - set(discovered_ids))}, "
            f"unexpected={sorted(set(discovered_ids) - expected_ids)}"
        )

    dataset_hash = metrics["dataset"]["manifest_sha256"]
    cases = []
    lambda_zero_hashes: set[str] = set()
    bart_input_hashes: set[str] = set()
    map_hashes: set[str] = set()
    for manifest_path, manifest in loaded:
        case_id = _case_id(manifest)
        if manifest.get("status") != "complete":
            raise ValueError(f"Case is not complete: {manifest_path}")
        config = manifest.get("config", {})
        if config.get("backend") != "gpu" or "-g" not in manifest.get(
            "effective_bart_command", []
        ):
            raise ValueError(f"Case did not record GPU BART -g: {manifest_path}")
        if config.get("dataset_manifest_sha256") != dataset_hash:
            raise ValueError(f"Dataset provenance mismatch: {manifest_path}")
        provenance = manifest.get("source_provenance", {})
        dataset_provenance = provenance.get("dataset_manifest", {})
        if dataset_provenance.get("sha256") != dataset_hash:
            raise ValueError(f"Source dataset provenance mismatch: {manifest_path}")
        lambda_zero_hashes.add(config.get("lambda_zero_manifest_sha256", ""))
        bart_input_hashes.add(config.get("bart_input_manifest_sha256", ""))
        map_hashes.add(manifest.get("maps", {}).get("cfl_sha256", ""))

        manifest_sha256 = sha256_file(manifest_path)
        magnitude = _magnitude_record(manifest, manifest_path)
        nifti_path = _verified_file(
            magnitude, "nifti", "nifti_sha256", f"{case_id} magnitude NIfTI"
        )
        sidecar_path = _verified_file(
            magnitude, "json", "json_sha256", f"{case_id} magnitude sidecar"
        )
        image = nib.load(str(nifti_path))
        affine_difference = _require_same_geometry(image, reference, case_id)
        finite_summary = _finite_summary(image, case_id)
        if list(config.get("matrix_rolinpar", [])) != list(reference.shape):
            raise ValueError(f"Configured matrix differs from reference: {manifest_path}")

        cases.append(
            {
                "case_id": case_id,
                "regularizer": config["regularizer"],
                "lambda": float(config["lambda"]),
                "lambda_label": config["lambda_label"],
                "block_size": config.get("block_size"),
                "run_manifest": str(manifest_path),
                "run_manifest_sha256": manifest_sha256,
                "magnitude_nifti": str(nifti_path),
                "magnitude_nifti_sha256": magnitude["nifti_sha256"],
                "magnitude_sidecar": str(sidecar_path),
                "magnitude_sidecar_sha256": magnitude["json_sha256"],
                "geometry": _geometry(image),
                "affine_max_abs_difference_from_reference": affine_difference,
                **finite_summary,
                "gpu_bart": True,
            }
        )

    for label, values in (
        ("lambda-zero manifest", lambda_zero_hashes),
        ("BART-input manifest", bart_input_hashes),
        ("ESPIRiT maps", map_hashes),
    ):
        if len(values) != 1 or "" in values:
            raise ValueError(f"Cases do not share one valid {label} hash: {values}")

    cases.sort(
        key=lambda item: (
            item["regularizer"],
            item["block_size"] or 0,
            item["lambda"],
        )
    )
    report = {
        "format_version": 1,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "exact-grid geometry and provenance gate before R1 metrics",
        "metrics_reference_manifest": {
            "path": str(metrics_path),
            "sha256": sha256_file(metrics_path),
        },
        **reference_report,
        "geometry_policy": {
            "required_orientation": ["R", "A", "S"],
            "voxel_size_absolute_tolerance_mm": 1e-6,
            "affine_max_absolute_tolerance": 1e-6,
            "registration_performed": False,
            "interpolation_performed": False,
        },
        "shared_reconstruction_provenance": {
            "lambda_zero_manifest_sha256": next(iter(lambda_zero_hashes)),
            "bart_input_manifest_sha256": next(iter(bart_input_hashes)),
            "espirit_maps_cfl_sha256": next(iter(map_hashes)),
        },
        "case_count": len(cases),
        "cases": cases,
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite geometry report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, report)
    print(f"Geometry/provenance gate: PASSED ({len(cases)} cases)")
    print(f"Report: {output}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
