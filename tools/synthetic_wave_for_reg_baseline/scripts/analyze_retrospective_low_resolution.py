#!/usr/bin/env python3
"""Quantify configured retrospective-resolution fidelity and sharpness tradeoffs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
from nibabel.processing import resample_from_to
from scipy.ndimage import (
    affine_transform,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
)
from skimage import __version__ as skimage_version
from skimage.metrics import structural_similarity


@dataclass
class AnalysisVolume:
    key: str
    title: str
    path: Path
    image: nib.spatialimages.SpatialImage
    data: np.ndarray
    geometry: dict[str, Any]
    case: dict[str, Any] | None


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_image(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) != 3 or nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise ValueError(f"Expected a canonical RAS 3D image: {path}")
    data = np.abs(np.asarray(image.dataobj, dtype=np.float32))
    if not np.isfinite(data).all() or float(data.max()) <= 0:
        raise ValueError(f"Image must be finite and nonzero: {path}")
    return image, data


def _rotation_matrix_xyz_degrees(angles_degrees: Sequence[float]) -> np.ndarray:
    x, y, z = np.deg2rad(np.asarray(angles_degrees, dtype=float))
    cx, cy, cz = np.cos((x, y, z))
    sx, sy, sz = np.sin((x, y, z))
    rotation_x = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rotation_z @ rotation_y @ rotation_x


def map_fixed_mask_to_reconstruction(
    fixed_mask_image: nib.spatialimages.SpatialImage,
    target_image: nib.spatialimages.SpatialImage,
    registration: dict[str, Any],
) -> np.ndarray:
    """Map the approved fixed mask to an untouched reconstruction grid.

    The frozen registration maps reconstruction data into the fixed DICOM
    grid. Sampling the fixed mask at those forward-mapped coordinates applies
    the inverse operation to the mask without interpolating the reconstruction.
    """
    fixed_zooms = np.asarray(fixed_mask_image.header.get_zooms()[:3], dtype=float)
    if not np.allclose(fixed_zooms, 1.0, atol=1e-5):
        raise ValueError("Frozen registration translations require a 1 mm fixed mask grid.")
    rigid = registration["rigid"]
    parameters = rigid["parameters"]
    rotation = _rotation_matrix_xyz_degrees(parameters["rotation_degrees_ras_xyz"])
    recorded_rotation = np.asarray(rigid["rotation_matrix"], dtype=float)
    if not np.allclose(rotation, recorded_rotation, atol=1e-8):
        raise ValueError("Recorded rigid rotation matrix does not match its angles.")
    translation = np.asarray(parameters["translation_mm_ras_xyz"], dtype=float)
    center = (np.asarray(fixed_mask_image.shape, dtype=float) - 1.0) / 2.0
    rigid_index = np.eye(4)
    rigid_index[:3, :3] = rotation
    rigid_index[:3, 3] = center + translation - rotation @ center
    target_to_fixed = (
        rigid_index @ np.linalg.inv(fixed_mask_image.affine) @ target_image.affine
    )
    fixed = np.asarray(fixed_mask_image.dataobj) > 0
    mapped = affine_transform(
        fixed.astype(np.uint8),
        target_to_fixed[:3, :3],
        offset=target_to_fixed[:3, 3],
        output_shape=target_image.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    mask = mapped > 0
    fraction = float(mask.mean())
    if not 0.01 < fraction < 0.8:
        raise ValueError(f"Mapped BET mask has implausible volume fraction {fraction:g}.")
    return mask


def gradient_components_per_mm(
    data: np.ndarray,
    voxel_size_mm_xyz: Sequence[float],
    *,
    smoothing_sigma_mm: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X/Y/Z and total gradients using physical-mm spacing."""
    zooms = np.asarray(voxel_size_mm_xyz, dtype=float)
    if zooms.shape != (3,) or np.any(~np.isfinite(zooms)) or np.any(zooms <= 0):
        raise ValueError(f"Invalid voxel sizes: {voxel_size_mm_xyz}")
    smoothed = gaussian_filter(
        np.asarray(data, dtype=np.float32),
        sigma=tuple(float(smoothing_sigma_mm / value) for value in zooms),
    )
    components = np.gradient(smoothed, *zooms, edge_order=1)
    total = np.sqrt(sum(component * component for component in components))
    return (*components, total)


def _ncc(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    first_values = first[mask].astype(np.float64)
    second_values = second[mask].astype(np.float64)
    first_values -= first_values.mean()
    second_values -= second_values.mean()
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    return float(np.dot(first_values, second_values) / denominator) if denominator else float("nan")


def _lsq_scale(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    reference_values = reference[mask].astype(np.float64)
    candidate_values = candidate[mask].astype(np.float64)
    denominator = float(np.dot(candidate_values, candidate_values))
    if denominator <= 0:
        raise ValueError("Candidate is zero inside the fixed brain mask.")
    return float(np.dot(reference_values, candidate_values) / denominator)


def _axial_ssim_mean(
    reference: np.ndarray, candidate: np.ndarray, brain: np.ndarray
) -> tuple[float, int]:
    coordinates = np.argwhere(brain)
    low_xyz = coordinates.min(axis=0)
    high_xyz = coordinates.max(axis=0) + 1
    reference_values = reference[brain]
    low_intensity, high_intensity = np.percentile(reference_values, [1.0, 99.0])
    data_range = float(high_intensity - low_intensity)
    if data_range <= 0:
        raise ValueError("Reference has no SSIM intensity range inside the brain mask.")
    scores = []
    x_slice = slice(int(low_xyz[0]), int(high_xyz[0]))
    y_slice = slice(int(low_xyz[1]), int(high_xyz[1]))
    minimum_mask_voxels = min(100, max(10, int(0.005 * brain.shape[0] * brain.shape[1])))
    for z_index in range(int(low_xyz[2]), int(high_xyz[2])):
        if int(brain[:, :, z_index].sum()) < minimum_mask_voxels:
            continue
        reference_slice = np.clip(
            reference[x_slice, y_slice, z_index], low_intensity, high_intensity
        )
        candidate_slice = np.clip(
            candidate[x_slice, y_slice, z_index], low_intensity, high_intensity
        )
        scores.append(
            structural_similarity(reference_slice, candidate_slice, data_range=data_range)
        )
    if not scores:
        raise ValueError("No axial brain slices were available for SSIM.")
    return float(np.mean(scores)), len(scores)


def matched_fidelity_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    brain: np.ndarray,
    edge: np.ndarray,
    voxel_sizes: Sequence[float],
) -> dict[str, float]:
    scale = _lsq_scale(reference, candidate, brain)
    scaled = candidate * scale
    residual = scaled[brain].astype(np.float64) - reference[brain].astype(np.float64)
    reference_values = reference[brain].astype(np.float64)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    ssim, slice_count = _axial_ssim_mean(reference, scaled, brain)
    *_, reference_gradient = gradient_components_per_mm(reference, voxel_sizes)
    *_, candidate_gradient = gradient_components_per_mm(scaled, voxel_sizes)
    return {
        "intensity_scale_lsq": scale,
        "nrmse_brain": rmse
        / max(float(np.sqrt(np.mean(reference_values * reference_values))), 1e-12),
        "nmae_brain": mae / max(float(np.mean(np.abs(reference_values))), 1e-12),
        "ncc_brain": _ncc(reference, scaled, brain),
        "ssim_axial_brain_bbox_mean": ssim,
        "ssim_axial_slice_count": float(slice_count),
        "gradient_ncc_fixed_edge": _ncc(reference_gradient, candidate_gradient, edge),
        "edge_gradient_preservation_ratio": float(np.mean(candidate_gradient[edge]))
        / max(float(np.mean(reference_gradient[edge])), 1e-12),
    }


def native_resolution_metrics(
    volume: AnalysisVolume,
    brain: np.ndarray,
    background: np.ndarray,
    edge: np.ndarray,
    smooth_region: np.ndarray,
) -> dict[str, Any]:
    zooms = np.asarray(volume.image.header.get_zooms()[:3], dtype=float)
    brain_values = volume.data[brain]
    positive = brain_values[brain_values > 0]
    if positive.size < 100:
        raise ValueError(f"Too few positive brain voxels in {volume.key}.")
    normalization = float(np.percentile(positive, 99.0))
    normalized = volume.data / normalization
    gx, gy, gz, total = gradient_components_per_mm(normalized, zooms)
    background_values = normalized[background].astype(np.float64)
    background_std = float(np.std(background_values))
    smooth_values = normalized[smooth_region].astype(np.float64)
    local_mean = gaussian_filter(
        normalized,
        sigma=tuple(float(2.0 / value) for value in zooms),
    )
    smooth_residual = (normalized - local_mean)[smooth_region].astype(np.float64)
    residual_median = float(np.median(smooth_residual))
    residual_robust_sigma = float(
        np.median(np.abs(smooth_residual - residual_median)) / 0.6744897501960817
    )
    smooth_signal = float(np.median(smooth_values))
    return {
        "case": volume.key,
        "title": volume.title.replace("\n", " "),
        "shape_x": int(volume.image.shape[0]),
        "shape_y": int(volume.image.shape[1]),
        "shape_z": int(volume.image.shape[2]),
        "voxel_size_x_mm": float(zooms[0]),
        "voxel_size_y_mm": float(zooms[1]),
        "voxel_size_z_mm": float(zooms[2]),
        "voxel_volume_mm3": float(np.prod(zooms)),
        "brain_voxel_count": int(brain.sum()),
        "background_voxel_count": int(background.sum()),
        "edge_voxel_count": int(edge.sum()),
        "smooth_region_voxel_count": int(smooth_region.sum()),
        "brain_positive_p99_normalization": normalization,
        "brain_median_normalized": float(np.median(normalized[brain])),
        "background_mean_normalized": float(np.mean(background_values)),
        "background_std_normalized": background_std,
        "background_rms_normalized": float(np.sqrt(np.mean(background_values**2))),
        "background_p95_normalized": float(np.percentile(background_values, 95.0)),
        "smooth_region_median_normalized": smooth_signal,
        "smooth_region_residual_robust_sigma": residual_robust_sigma,
        "smooth_region_signal_to_residual_proxy": smooth_signal
        / max(residual_robust_sigma, 1e-12),
        "edge_gradient_mean_per_mm": float(np.mean(total[edge])),
        "edge_gradient_p90_per_mm": float(np.percentile(total[edge], 90.0)),
        "edge_abs_gradient_x_mean_per_mm": float(np.mean(np.abs(gx[edge]))),
        "edge_abs_gradient_y_mean_per_mm": float(np.mean(np.abs(gy[edge]))),
        "edge_abs_gradient_z_mean_per_mm": float(np.mean(np.abs(gz[edge]))),
    }


def _resample_binary_mask(
    mask: np.ndarray,
    source_image: nib.spatialimages.SpatialImage,
    target_image: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    mask_image = nib.Nifti1Image(mask.astype(np.uint8), source_image.affine)
    return np.asarray(resample_from_to(mask_image, target_image, order=0).dataobj) > 0


def _build_reference_masks(
    reference: np.ndarray, brain: np.ndarray, voxel_sizes: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    normalized = reference / float(np.percentile(reference[brain], 99.0))
    *_, gradient = gradient_components_per_mm(normalized, voxel_sizes)
    interior = binary_erosion(brain, iterations=2)
    threshold = float(np.percentile(gradient[interior], 80.0))
    edge = interior & (gradient >= threshold)

    anatomy_support = normalized >= 0.02
    excluded = binary_dilation(anatomy_support, iterations=6)
    distance_from_brain = distance_transform_edt(~brain, sampling=voxel_sizes)
    border_safe = np.ones(brain.shape, dtype=bool)
    border_voxels = np.maximum(np.ceil(4.0 / np.asarray(voxel_sizes)).astype(int), 1)
    for axis, width in enumerate(border_voxels):
        slices = [slice(None)] * 3
        slices[axis] = slice(0, int(width))
        border_safe[tuple(slices)] = False
        slices[axis] = slice(-int(width), None)
        border_safe[tuple(slices)] = False
    background = ~excluded & (distance_from_brain >= 12.0) & border_safe
    smooth_interior = binary_erosion(brain, iterations=3)
    smooth_intensities = normalized[smooth_interior]
    intensity_low, intensity_high = np.percentile(smooth_intensities, [20.0, 80.0])
    smooth_gradient_threshold = float(np.percentile(gradient[smooth_interior], 30.0))
    smooth_region = (
        smooth_interior
        & (normalized >= intensity_low)
        & (normalized <= intensity_high)
        & (gradient <= smooth_gradient_threshold)
    )
    if any(int(mask.sum()) < 1000 for mask in (edge, background, smooth_region)):
        raise ValueError("A reference-derived metric mask is unexpectedly small.")
    metadata = {
        "edge_rule": "top 20% of sigma-0.7-mm full-resolution gradient inside 2-mm-eroded BET brain",
        "edge_gradient_threshold_per_mm": threshold,
        "edge_voxel_count": int(edge.sum()),
        "background_rule": (
            "fixed full-resolution grid: >=12 mm from BET brain, outside 6-mm dilation "
            "of full-resolution intensity >=2% brain p99, excluding 4-mm FOV border"
        ),
        "background_voxel_count": int(background.sum()),
        "smooth_region_rule": (
            "fixed full-resolution grid: inside 3-mm-eroded BET brain, reference "
            "intensity p20-p80, and bottom 30% of sigma-0.7-mm gradient"
        ),
        "smooth_region_gradient_threshold_per_mm": smooth_gradient_threshold,
        "smooth_region_intensity_range_normalized": [
            float(intensity_low),
            float(intensity_high),
        ],
        "smooth_region_voxel_count": int(smooth_region.sum()),
    }
    return edge, background, smooth_region, metadata


def _save_mask(
    path: Path, mask: np.ndarray, reference_image: nib.spatialimages.SpatialImage, description: str
) -> None:
    header = reference_image.header.copy()
    header.set_data_dtype(np.uint8)
    header["descrip"] = description.encode("ascii", errors="ignore")[:79]
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference_image.affine, header), path)


def _load_review_volumes(
    review: dict[str, Any], required_reference_keys: Sequence[str]
) -> list[AnalysisVolume]:
    volumes = []
    for record in review["inputs"]:
        path = Path(record["path"]).expanduser().resolve()
        expected_hash = record.get("sha256")
        if expected_hash and sha256_file(path) != expected_hash:
            raise ValueError(f"Review input hash changed: {path}")
        image, data = _canonical_image(path)
        case = None
        case_record = record.get("case_manifest")
        if isinstance(case_record, dict):
            case_manifest = Path(case_record["path"])
            if sha256_file(case_manifest) != case_record["sha256"]:
                raise ValueError(f"Case manifest hash changed: {case_manifest}")
            case = _load_json(case_manifest)["case"]
        volumes.append(
            AnalysisVolume(
                key=str(record["key"]),
                title=str(record["title"]),
                path=path,
                image=image,
                data=data,
                geometry=dict(record["geometry"]),
                case=case,
            )
        )
    required = set(required_reference_keys)
    if not required.issubset(volume.key for volume in volumes) or len(volumes) != 5:
        raise ValueError(
            "Review manifest must contain the configured references and three LR cases."
        )
    return volumes


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty metrics table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    native_rows: Sequence[dict[str, Any]],
    matched_rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    full_resolution_key: str,
    excluded_resolution_keys: Sequence[str],
    figure_title: str,
    fidelity_title: str,
) -> None:
    excluded = set(excluded_resolution_keys)
    resolution_rows = [row for row in native_rows if row["case"] not in excluded]

    def display_label(row: dict[str, Any]) -> str:
        return " x ".join(
            f"{float(row[f'voxel_size_{axis}_mm']):.2f}".rstrip("0").rstrip(".")
            for axis in ("x", "y", "z")
        ) + " mm"

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    markers = ["o", "s", "^", "D"]
    for row, marker in zip(resolution_rows, markers, strict=True):
        label = display_label(row)
        axes[0, 0].scatter(
            row["voxel_volume_mm3"],
            row["smooth_region_signal_to_residual_proxy"],
            marker=marker,
            s=70,
            label=label,
        )
        axes[0, 1].scatter(
            row["voxel_volume_mm3"],
            row["edge_gradient_mean_per_mm_ratio_to_full"],
            marker=marker,
            s=70,
            label=label,
        )
    axes[0, 0].set(xlabel="Voxel volume (mm³)", ylabel="Signal/local-residual proxy")
    axes[0, 0].set_title("Fixed smooth-brain proxy (not true SNR)")
    axes[0, 1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 1].set(xlabel="Voxel volume (mm³)", ylabel="Edge-gradient ratio to full 1 mm")
    axes[0, 1].set_title("Native-grid sharpness in physical units")
    axes[0, 0].legend(fontsize=7)

    x = np.arange(len(resolution_rows))
    width = 0.25
    hatches = ("//", "..", "xx")
    for offset, (axis_name, hatch) in enumerate(zip(("x", "y", "z"), hatches, strict=True)):
        values = [
            row[f"edge_abs_gradient_{axis_name}_mean_per_mm_ratio_to_full"]
            for row in resolution_rows
        ]
        axes[1, 0].bar(x + (offset - 1) * width, values, width, label=axis_name.upper(), hatch=hatch)
    axes[1, 0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_xticks(
        x, [display_label(row) for row in resolution_rows], rotation=18, ha="right"
    )
    axes[1, 0].set_ylabel("Directional edge-gradient ratio")
    axes[1, 0].set_title("Native directional sharpness")
    axes[1, 0].legend(title="RAS axis")

    full_reference = {
        row["candidate"]: row
        for row in matched_rows
        if row["reference"] == full_resolution_key
        and row["candidate"] not in excluded
    }
    values = [full_reference[row["case"]]["nrmse_brain"] for row in resolution_rows]
    axes[1, 1].bar(x, values, color="0.45", hatch="//")
    axes[1, 1].set_xticks(
        x, [display_label(row) for row in resolution_rows], rotation=18, ha="right"
    )
    axes[1, 1].set_ylabel("Brain NRMSE")
    axes[1, 1].set_title(fidelity_title)
    figure.suptitle(figure_title, fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--review-manifest", type=Path, help="Legacy product-analysis input.")
    parser.add_argument("--approved-bet-mask", type=Path, help="Legacy product-analysis input.")
    parser.add_argument("--shared-registration", type=Path, help="Legacy product-analysis input.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    legacy = (
        args.review_manifest,
        args.approved_bet_mask,
        args.shared_registration,
        args.output_dir,
    )
    if args.config is not None:
        if any(value is not None for value in legacy):
            parser.error("--config cannot be mixed with legacy arguments")
    elif any(value is None for value in legacy):
        parser.error("either --config or all legacy analysis arguments are required")
    return args


def _legacy_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "review_path": args.review_manifest.expanduser().resolve(),
        "mask_path": args.approved_bet_mask.expanduser().resolve(),
        "registration_path": args.shared_registration.expanduser().resolve(),
        "output_dir": args.output_dir.expanduser().resolve(),
        "full_resolution_key": "full_resolution_llr",
        "anatomical_reference_key": "grappa",
        "metric_mask_reference_key": "full_resolution_llr",
        "matched_reference_keys": ["full_resolution_llr", "grappa"],
        "excluded_resolution_keys": ["grappa"],
        "figure_title": (
            "Retrospective low-resolution tradeoffs — descriptive, "
            "no automatic selection"
        ),
        "fidelity_title": "Matched 1 mm fidelity to full-resolution LLR",
        "scientific_scope": {
            "full_resolution_reference": "same corrected LLR regularization",
            "grappa_reference": "temporary secondary anatomical comparison",
            "dicom_intensities_used": False,
            "approved_bet_used_for_metrics_only": True,
            "candidate_registration_performed": False,
            "true_snr_or_cnr_claimed": False,
            "automatic_selection_performed": False,
        },
        "approval_path": None,
        "config_path": None,
    }


def _configured_settings(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _load_json(config_path)
    if config.get("format_version") != 1:
        raise ValueError("Analysis config format_version must be 1.")
    mask_alignment = config.get("mask_alignment")
    if mask_alignment not in ("exact_reference_grid", "shared_rigid_registration"):
        raise ValueError("Unsupported mask_alignment.")
    registration = config.get("shared_registration")
    if mask_alignment == "shared_rigid_registration" and not registration:
        raise ValueError("shared_registration is required for shared rigid mask alignment.")
    if mask_alignment == "exact_reference_grid" and registration is not None:
        raise ValueError("Exact-grid mask alignment must not specify registration.")
    matched_keys = [str(value) for value in config["matched_reference_keys"]]
    if not matched_keys:
        raise ValueError("matched_reference_keys must not be empty.")
    scope = config.get("scientific_scope", {})
    if not isinstance(scope, dict):
        raise ValueError("scientific_scope must be an object.")
    return {
        "review_path": _resolve_config_path(config_path, str(config["review_manifest"])),
        "mask_path": _resolve_config_path(config_path, str(config["approved_bet_mask"])),
        "registration_path": (
            None
            if registration is None
            else _resolve_config_path(config_path, str(registration))
        ),
        "output_dir": _resolve_config_path(config_path, str(config["output_dir"])),
        "full_resolution_key": str(config["full_resolution_key"]),
        "anatomical_reference_key": str(config["anatomical_reference_key"]),
        "metric_mask_reference_key": str(config["metric_mask_reference_key"]),
        "matched_reference_keys": matched_keys,
        "excluded_resolution_keys": [
            str(value) for value in config.get("excluded_resolution_keys", [])
        ],
        "figure_title": str(
            config.get(
                "figure_title",
                "Retrospective low-resolution tradeoffs — descriptive, "
                "no automatic selection",
            )
        ).replace("\\n", "\n"),
        "fidelity_title": str(
            config.get("fidelity_title", "Matched-grid fidelity to full resolution")
        ),
        "scientific_scope": scope,
        "approval_path": _resolve_config_path(
            config_path, str(config["visual_approval_record"])
        ),
        "config_path": config_path,
    }


def _validate_visual_approval(approval_path: Path, review_path: Path) -> dict[str, Any]:
    approval = _load_json(approval_path)
    if approval.get("status") != "approved":
        raise ValueError("Visual approval record is not approved.")
    review_record = approval.get("review_manifest", {})
    if Path(str(review_record.get("path", ""))).expanduser().resolve() != review_path:
        raise ValueError("Visual approval record names a different review manifest.")
    if review_record.get("sha256") != sha256_file(review_path):
        raise ValueError("Visual approval record does not match the review manifest hash.")
    if set(approval.get("approved_outputs", [])) != {
        "native_grid_comparison",
        "matched_grid_comparison",
    }:
        raise ValueError("Both native-grid and matched-grid outputs require approval.")
    return approval


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = (
        _legacy_settings(args)
        if getattr(args, "config", None) is None
        else _configured_settings(args.config)
    )
    review_path = settings["review_path"]
    mask_path = settings["mask_path"]
    registration_path = settings["registration_path"]
    output_dir = settings["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Analysis output directory is not empty: {output_dir}")

    review = _load_json(review_path)
    if review.get("status") != "complete":
        raise ValueError(f"Visual review is not complete: {review_path}")
    approval = (
        None
        if settings["approval_path"] is None
        else _validate_visual_approval(settings["approval_path"], review_path)
    )
    required_keys = {
        settings["full_resolution_key"],
        settings["anatomical_reference_key"],
        settings["metric_mask_reference_key"],
        *settings["matched_reference_keys"],
    }
    volumes = _load_review_volumes(review, sorted(required_keys))
    by_key = {volume.key: volume for volume in volumes}
    full = by_key[settings["full_resolution_key"]]
    anatomical_reference = by_key[settings["anatomical_reference_key"]]
    metric_mask_reference = by_key[settings["metric_mask_reference_key"]]
    if full.image.shape != anatomical_reference.image.shape or not np.allclose(
        full.image.affine, anatomical_reference.image.affine, atol=1e-4
    ):
        raise ValueError("Configured full-resolution references do not share a grid.")
    if full.image.shape != metric_mask_reference.image.shape or not np.allclose(
        full.image.affine, metric_mask_reference.image.affine, atol=1e-4
    ):
        raise ValueError("Metric-mask reference does not share the full-resolution grid.")

    mask_image = nib.as_closest_canonical(nib.load(str(mask_path)))
    if registration_path is None:
        if mask_image.shape != full.image.shape or not np.allclose(
            mask_image.affine, full.image.affine, atol=1e-5
        ):
            raise ValueError("Approved BET mask does not match the exact reference grid.")
        brain = np.asarray(mask_image.dataobj) > 0
        if not 0.01 < float(brain.mean()) < 0.8:
            raise ValueError("Approved BET mask has an implausible volume fraction.")
    else:
        registration = _load_json(registration_path)
        brain = map_fixed_mask_to_reconstruction(mask_image, full.image, registration)
    zooms = full.image.header.get_zooms()[:3]
    edge, background, smooth_region, mask_metadata = _build_reference_masks(
        metric_mask_reference.data, brain, zooms
    )

    native_rows = []
    for volume in volumes:
        native_brain = _resample_binary_mask(brain, full.image, volume.image)
        native_background = _resample_binary_mask(background, full.image, volume.image)
        native_edge = _resample_binary_mask(edge, full.image, volume.image)
        native_smooth = _resample_binary_mask(smooth_region, full.image, volume.image)
        row = native_resolution_metrics(
            volume, native_brain, native_background, native_edge, native_smooth
        )
        native_rows.append(row)
    excluded = set(settings["excluded_resolution_keys"])
    native_rows.sort(
        key=lambda row: (
            row["case"] in excluded,
            row["voxel_volume_mm3"],
            row["case"],
        )
    )
    full_native = next(
        row for row in native_rows if row["case"] == settings["full_resolution_key"]
    )
    ratio_metrics = (
        "edge_gradient_mean_per_mm",
        "edge_abs_gradient_x_mean_per_mm",
        "edge_abs_gradient_y_mean_per_mm",
        "edge_abs_gradient_z_mean_per_mm",
    )
    for row in native_rows:
        for key in ratio_metrics:
            row[f"{key}_ratio_to_full"] = (
                float("nan")
                if row["case"] in excluded
                else row[key] / max(full_native[key], 1e-12)
            )

    matched_data = {}
    for volume in volumes:
        if volume.image.shape == full.image.shape and np.allclose(
            volume.image.affine, full.image.affine, atol=1e-5
        ):
            matched_data[volume.key] = volume.data
        else:
            data = np.abs(np.asarray(resample_from_to(volume.image, full.image, order=1).dataobj))
            if not np.isfinite(data).all():
                raise ValueError(f"Non-finite matched-grid data: {volume.key}")
            matched_data[volume.key] = data.astype(np.float32, copy=False)

    matched_rows = []
    for reference_key in settings["matched_reference_keys"]:
        reference = matched_data[reference_key]
        for volume in volumes:
            metrics = matched_fidelity_metrics(
                reference, matched_data[volume.key], brain, edge, zooms
            )
            matched_rows.append(
                {
                    "reference": reference_key,
                    "candidate": volume.key,
                    "candidate_voxel_volume_mm3": float(
                        np.prod(volume.image.header.get_zooms()[:3])
                    ),
                    **metrics,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "metrics_masks"
    mask_dir.mkdir()
    brain_path = mask_dir / "approved_bet_on_reconstruction_grid.nii.gz"
    edge_path = mask_dir / "fixed_reference_edge_mask.nii.gz"
    background_path = mask_dir / "fixed_reference_background_mask.nii.gz"
    smooth_path = mask_dir / "fixed_reference_smooth_brain_mask.nii.gz"
    _save_mask(brain_path, brain, full.image, "Approved BET mapped with frozen rigid")
    _save_mask(edge_path, edge, full.image, "Fixed full-resolution edge mask")
    _save_mask(background_path, background, full.image, "Fixed background proxy mask")
    _save_mask(smooth_path, smooth_region, full.image, "Fixed smooth-brain proxy mask")

    native_path = output_dir / "native_resolution_metrics.csv"
    matched_path = output_dir / "matched_fidelity_metrics.csv"
    figure_path = output_dir / "resolution_tradeoff_summary.png"
    _plot_summary(
        native_rows,
        matched_rows,
        figure_path,
        full_resolution_key=settings["full_resolution_key"],
        excluded_resolution_keys=settings["excluded_resolution_keys"],
        figure_title=settings["figure_title"],
        fidelity_title=settings["fidelity_title"],
    )
    _write_csv(native_path, native_rows)
    _write_csv(matched_path, matched_rows)

    metric_definitions = {
        "native_grid": (
            "Each image remains on its acquired/reconstructed grid. Gradients use mm spacing "
            "after fixed 0.7-mm Gaussian smoothing. Intensities are divided by brain-positive p99."
        ),
        "smooth_region_signal_to_residual_proxy": (
            "median signal divided by robust sigma of data minus a 2-mm Gaussian local mean "
            "inside one fixed low-gradient brain region; mixes noise and residual anatomy and is not true SNR"
        ),
        "background_metrics": (
            "fixed-air-region descriptive QC only; excluded from the signal/residual summary "
            "because BART reconstruction support makes background noise nearly zero"
        ),
        "matched_grid": "linear interpolation to the full-resolution 1 mm RAS grid; no rigid registration of candidates",
        "fidelity_scaling": "one least-squares scalar inside the fixed BET mask for each reference/candidate pair",
        "nrmse_brain": "brain-mask RMSE divided by brain-mask RMS of the named reference",
        "ssim_axial_brain_bbox_mean": "mean SSIM over axial slices cropped to the fixed brain bounding box",
        "edge_metrics": mask_metadata["edge_rule"],
        "selection": "descriptive tradeoffs only; no composite rank and no automatic selected resolution",
    }
    manifest = {
        "format_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "retrospective low-resolution fidelity, sharpness, and noise-proxy analysis",
        "scientific_scope": settings["scientific_scope"],
        "analysis_config": (
            None
            if settings["config_path"] is None
            else {
                "path": str(settings["config_path"]),
                "sha256": sha256_file(settings["config_path"]),
                "snapshot": _load_json(settings["config_path"]),
            }
        ),
        "inputs": {
            "review_manifest": {"path": str(review_path), "sha256": sha256_file(review_path)},
            "approved_bet_mask": {"path": str(mask_path), "sha256": sha256_file(mask_path)},
            "visual_approval": (
                None
                if settings["approval_path"] is None
                else {
                    "path": str(settings["approval_path"]),
                    "sha256": sha256_file(settings["approval_path"]),
                    "record": approval,
                }
            ),
            "mask_alignment": (
                {
                    "mode": "exact_reference_grid",
                    "reference_key": settings["metric_mask_reference_key"],
                }
                if registration_path is None
                else {
                    "mode": "shared_rigid_registration",
                    "path": str(registration_path),
                    "sha256": sha256_file(registration_path),
                    "use": "transfer the approved fixed mask into untouched reconstruction space",
                }
            ),
        },
        "fixed_masks": {
            **mask_metadata,
            "brain_voxel_count": int(brain.sum()),
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in (brain_path, edge_path, background_path, smooth_path)
            ],
        },
        "metric_definitions": metric_definitions,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
            "scikit_image": skimage_version,
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (native_path, matched_path, figure_path)
        ],
    }
    manifest_path = output_dir / "analysis_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Retrospective low-resolution analysis manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
