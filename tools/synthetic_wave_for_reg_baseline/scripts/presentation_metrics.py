"""Exact-grid presentation metrics against the approved direct-FFT R1 reference."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from bart_cfl import sha256_file
from evaluate_regularization_volume import (
    _gradient_magnitude,
    build_fixed_masks,
    compute_metrics,
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _record_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _verify_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}: {path}")


def nifti_sidecar_path(nifti_path: Path) -> Path:
    """Return the BIDS-like JSON sidecar path for one ``.nii.gz`` output."""
    if not nifti_path.name.endswith(".nii.gz"):
        raise ValueError(f"Expected a .nii.gz magnitude output: {nifti_path}")
    return nifti_path.with_name(nifti_path.name[:-7] + ".json")


def _finite_magnitude(
    path: Path, label: str
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all() or not np.any(data > 0):
        raise ValueError(f"{label} is non-finite or has no positive voxels: {path}")
    if float(np.min(data)) < -1e-6:
        raise ValueError(f"{label} contains negative magnitude values: {path}")
    return image, np.abs(data)


def validate_metrics_reference_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate approval, hashes, and exact reference/mask geometry."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path, "metrics-reference manifest")
    if manifest.get("status") != "approved_for_metrics":
        raise ValueError("Metrics-reference manifest is not approved")

    reference_record = manifest["ranking_reference"]
    reference_path = _record_path(reference_record["path"], manifest_path)
    _verify_hash(
        reference_path,
        reference_record["sha256"],
        "direct FFT RSS reference",
    )
    mask_record = manifest["brain_mask"]
    mask_path = _record_path(mask_record["path"], manifest_path)
    _verify_hash(mask_path, mask_record["sha256"], "approved brain mask")

    reference_image = nib.load(str(reference_path))
    mask_image = nib.load(str(mask_path))
    if reference_image.shape != mask_image.shape or not np.array_equal(
        reference_image.affine, mask_image.affine
    ):
        raise ValueError("Approved brain mask is not on the exact reference grid")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "reference_path": reference_path,
        "mask_path": mask_path,
        "shape": tuple(int(value) for value in reference_image.shape),
        "affine": np.asarray(reference_image.affine),
    }


def evaluate_against_direct_fft(
    candidate_nifti: Path,
    metrics_reference_manifest: Path,
) -> dict[str, Any]:
    """Evaluate one exported magnitude on the unchanged direct-FFT R1 grid."""
    candidate_nifti = candidate_nifti.expanduser().resolve()
    context = validate_metrics_reference_manifest(metrics_reference_manifest)
    manifest = context["manifest"]
    reference_path = context["reference_path"]
    mask_path = context["mask_path"]

    reference_image, reference = _finite_magnitude(
        reference_path, "direct FFT RSS reference"
    )
    mask_image = nib.load(str(mask_path))
    brain = np.asarray(mask_image.dataobj) > 0

    candidate_image, normalized = _finite_magnitude(candidate_nifti, "candidate")
    if candidate_image.shape != reference_image.shape or not np.array_equal(
        candidate_image.affine, reference_image.affine
    ):
        raise ValueError(
            "Candidate is not on the exact direct-FFT reference grid; registration "
            "and interpolation are forbidden"
        )

    sidecar_path = nifti_sidecar_path(candidate_nifti)
    sidecar = _load_json(sidecar_path, "candidate magnitude sidecar")
    normalization = sidecar.get("MagnitudeNormalization", {})
    if normalization.get("Method") != "positive-finite-percentile":
        raise ValueError("Unsupported candidate NIfTI magnitude normalization")
    input_percentile = float(normalization.get("InputPercentileValue", 0.0))
    output_percentile = float(normalization.get("OutputPercentileValue", 0.0))
    if input_percentile <= 0 or output_percentile <= 0:
        raise ValueError("Candidate NIfTI magnitude normalization values are invalid")
    restoration_multiplier = input_percentile / output_percentile
    candidate = normalized * np.float32(restoration_multiplier)
    if not np.isfinite(candidate).all() or not np.any(candidate > 0):
        raise ValueError("Restored candidate magnitude is invalid")

    masks, mask_metadata = build_fixed_masks(reference, brain)
    mask_metadata["reference_kind"] = "direct_fft_rss"
    metrics, _ = compute_metrics(
        reference,
        candidate,
        masks,
        mask_metadata,
        reference_gradient=_gradient_magnitude(reference),
    )
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Direct-FFT metric calculation produced a non-finite value")

    reference_record = manifest["ranking_reference"]
    mask_record = manifest["brain_mask"]
    return {
        "status": "complete",
        "reference_kind": "direct_fft_rss",
        "metrics_reference_manifest": {
            "path": str(context["manifest_path"]),
            "sha256": context["manifest_sha256"],
            "approval_status": manifest["status"],
        },
        "reference": {
            "path": str(reference_path),
            "sha256": reference_record["sha256"],
        },
        "approved_brain_mask": {
            "path": str(mask_path),
            "sha256": mask_record["sha256"],
            "voxel_count": int(brain.sum()),
        },
        "candidate": {
            "magnitude_nifti": str(candidate_nifti),
            "magnitude_nifti_sha256": sha256_file(candidate_nifti),
            "magnitude_sidecar": str(sidecar_path),
            "magnitude_sidecar_sha256": sha256_file(sidecar_path),
        },
        "geometry_policy": {
            "exact_shape_required": True,
            "exact_affine_required": True,
            "registration_performed": False,
            "interpolation_performed": False,
        },
        "intensity_policy": {
            "export_normalization_restored": True,
            "export_normalization_method": normalization["Method"],
            "export_normalization_percentile": float(normalization["Percentile"]),
            "restoration_multiplier": restoration_multiplier,
            "one_unconstrained_lsq_scale_inside_approved_brain_mask": True,
        },
        "mask_metadata": mask_metadata,
        "metrics": metrics,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
