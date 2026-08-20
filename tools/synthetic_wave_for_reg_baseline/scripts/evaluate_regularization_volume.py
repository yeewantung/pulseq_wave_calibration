#!/usr/bin/env python3
"""Register and evaluate a complete Wave regularization sweep against DICOM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
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
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    label,
)
from scipy.optimize import minimize
from skimage import __version__ as skimage_version
from skimage.metrics import structural_similarity


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Convert scientific scalar/path values while rejecting unknown objects."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_magnitude(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = np.asarray(image.dataobj, dtype=np.float32)
    data = np.abs(data)
    data[~np.isfinite(data)] = 0.0
    return data


def signed_axis_transform(
    data: np.ndarray, permutation: Sequence[int], flips: Sequence[bool]
) -> np.ndarray:
    transformed = np.transpose(data, tuple(permutation))
    for axis, flip in enumerate(flips):
        if flip:
            transformed = np.flip(transformed, axis=axis)
    return transformed


def rotation_matrix_xyz_degrees(angles_degrees: Sequence[float]) -> np.ndarray:
    """Return Rz @ Ry @ Rx for rotations about canonical RAS axes."""
    x, y, z = np.deg2rad(np.asarray(angles_degrees, dtype=float))
    cx, cy, cz = np.cos((x, y, z))
    sx, sy, sz = np.sin((x, y, z))
    rotation_x = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rotation_z @ rotation_y @ rotation_x


def rigid_resample(
    data: np.ndarray,
    parameters: Sequence[float],
    *,
    translation_scale: float = 1.0,
    order: int = 1,
) -> np.ndarray:
    """Apply moving-to-fixed rigid parameters using inverse output sampling."""
    parameters = np.asarray(parameters, dtype=float)
    rotation = rotation_matrix_xyz_degrees(parameters[:3])
    translation = parameters[3:] / float(translation_scale)
    center = (np.asarray(data.shape, dtype=float) - 1.0) / 2.0
    inverse = rotation.T
    offset = center - inverse @ (center + translation)
    return affine_transform(
        data,
        inverse,
        offset=offset,
        output_shape=data.shape,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    ).astype(np.float32, copy=False)


def normalized_cross_correlation(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> float:
    reference_values = reference[mask].astype(np.float64)
    candidate_values = candidate[mask].astype(np.float64)
    reference_values -= reference_values.mean()
    candidate_values -= candidate_values.mean()
    denominator = float(
        np.linalg.norm(reference_values) * np.linalg.norm(candidate_values)
    )
    return (
        float(np.dot(reference_values, candidate_values) / denominator)
        if denominator
        else float("nan")
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = label(mask)
    if not count:
        raise ValueError("Reference foreground mask is empty")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def load_brain_mask(
    path: Path, reference_image: nib.spatialimages.SpatialImage
) -> np.ndarray:
    """Load an approved mask only when it shares the canonical reference grid."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    if image.shape != reference_image.shape or not np.allclose(
        image.affine, reference_image.affine, atol=1e-5
    ):
        raise ValueError("Fixed BET mask does not share the exact DICOM reference grid")
    mask = np.asarray(image.dataobj) > 0
    if int(mask.sum()) < 1_000_000:
        raise ValueError("Fixed BET brain mask is unexpectedly small")
    return mask


def build_fixed_masks(
    reference: np.ndarray, brain: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference_values = reference[brain]
    positive = reference_values[reference_values > 0]
    if not positive.size:
        raise ValueError("DICOM reference has no positive voxels inside the BET mask")
    reference_p99 = float(np.percentile(positive, 99.0))
    anatomy_threshold = 0.05 * reference_p99

    border_safe = np.ones(reference.shape, dtype=bool)
    border_safe[:4] = False
    border_safe[-4:] = False
    border_safe[:, :4] = False
    border_safe[:, -4:] = False
    border_safe[:, :, :4] = False
    border_safe[:, :, -4:] = False
    background = border_safe & ~binary_dilation(brain, iterations=8)

    gradients = np.gradient(gaussian_filter(reference, sigma=0.7))
    gradient_magnitude = np.sqrt(sum(component * component for component in gradients))
    edge_threshold = float(np.percentile(gradient_magnitude[brain], 80.0))
    edge = brain & (gradient_magnitude >= edge_threshold)
    masks = {"brain": brain, "background": background, "edge": edge}
    metadata = {
        "reference_brain_positive_p99": reference_p99,
        "brain_rule": "externally generated and visually approved fixed FSL BET mask",
        "brain_voxel_count": int(brain.sum()),
        "anatomy_missed_intensity_threshold": anatomy_threshold,
        "background_rule": "fixed FOV excluding 4-voxel border and 8-voxel dilation of BET brain mask",
        "background_voxel_count": int(background.sum()),
        "edge_rule": "top 20% of sigma-0.7 reference gradient magnitude inside BET brain mask",
        "edge_gradient_threshold": edge_threshold,
        "edge_voxel_count": int(edge.sum()),
    }
    return masks, metadata


def estimate_shared_rigid(
    reference: np.ndarray,
    candidate: np.ndarray,
    foreground: np.ndarray,
    downsample_factor: int = 4,
) -> dict[str, Any]:
    """Estimate one bounded six-DOF proper rigid transform using smoothed NCC."""
    factor = int(downsample_factor)
    stride = (slice(None, None, factor),) * 3
    reference_small = gaussian_filter(reference, sigma=1.5)[stride]
    candidate_small = gaussian_filter(candidate, sigma=1.5)[stride]
    mask_small = foreground[stride]
    evaluations = 0

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        transformed = rigid_resample(
            candidate_small,
            parameters,
            translation_scale=factor,
            order=1,
        )
        return -normalized_cross_correlation(reference_small, transformed, mask_small)

    initial = np.zeros(6, dtype=float)
    initial_ncc = -objective(initial)
    bounds = [(-3.0, 3.0)] * 3 + [(-5.0, 5.0)] * 3
    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={"xtol": 1e-3, "ftol": 1e-6, "maxiter": 80},
    )
    if not result.success:
        raise RuntimeError(f"Rigid registration failed: {result.message}")
    parameters = np.asarray(result.x, dtype=float)
    if any(
        abs(value - bound) < 0.01
        for value, bounds_pair in zip(parameters, bounds)
        for bound in bounds_pair
    ):
        raise ValueError(f"Rigid registration reached a search bound: {parameters}")
    final_ncc = -float(result.fun)
    if final_ncc <= initial_ncc:
        raise ValueError(
            f"Rigid registration did not improve NCC: {initial_ncc} -> {final_ncc}"
        )
    rotation = rotation_matrix_xyz_degrees(parameters[:3])
    return {
        "algorithm": "bounded Powell maximization of smoothed foreground NCC",
        "parameter_convention": (
            "moving-to-fixed: Rz@Ry@Rx about RAS-grid center, then RAS translation"
        ),
        "parameters": {
            "rotation_degrees_ras_xyz": parameters[:3].tolist(),
            "translation_mm_ras_xyz": parameters[3:].tolist(),
        },
        "rotation_matrix": rotation.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "downsample_factor": factor,
        "bounds": {
            "rotation_degrees_each_axis": [-3.0, 3.0],
            "translation_mm_each_axis": [-5.0, 5.0],
        },
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "objective_evaluations": evaluations,
        "downsampled_ncc_before": initial_ncc,
        "downsampled_ncc_after": final_ncc,
    }


def _load_on_reference_grid(
    path: Path,
    reference_image: nib.spatialimages.SpatialImage,
    permutation: Sequence[int],
    flips: Sequence[bool],
) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(str(path)))
    resampled = resample_from_to(image, reference_image, order=1)
    data = _finite_magnitude(resampled)
    transformed = signed_axis_transform(data, permutation, flips)
    if transformed.shape != reference_image.shape:
        raise ValueError(f"Orientation mapping changed shape for {path}: {transformed.shape}")
    return np.asarray(transformed, dtype=np.float32)


def _lsq_scale(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    reference_values = reference[mask].astype(np.float64)
    candidate_values = candidate[mask].astype(np.float64)
    denominator = float(np.dot(candidate_values, candidate_values))
    if denominator <= 0:
        raise ValueError("Candidate is zero inside foreground mask")
    return float(np.dot(reference_values, candidate_values) / denominator)


def _bounding_box(mask: np.ndarray, padding: int = 2) -> tuple[slice, ...]:
    coordinates = np.argwhere(mask)
    low = np.maximum(coordinates.min(axis=0) - padding, 0)
    high = np.minimum(coordinates.max(axis=0) + padding + 1, mask.shape)
    return tuple(slice(int(start), int(stop)) for start, stop in zip(low, high))


def _gradient_magnitude(data: np.ndarray) -> np.ndarray:
    gradients = np.gradient(gaussian_filter(data, sigma=0.7))
    return np.sqrt(sum(component * component for component in gradients))


def compute_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    masks: dict[str, np.ndarray],
    mask_metadata: dict[str, Any],
    reference_gradient: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    brain = masks["brain"]
    background = masks["background"]
    edge = masks["edge"]
    scale = _lsq_scale(reference, candidate, brain)
    scaled = (candidate * scale).astype(np.float32, copy=False)
    reference_values = reference[brain].astype(np.float64)
    candidate_values = scaled[brain].astype(np.float64)
    residual = candidate_values - reference_values
    rmse = float(np.sqrt(np.mean(residual * residual)))
    rms_reference = float(np.sqrt(np.mean(reference_values * reference_values)))
    mae = float(np.mean(np.abs(residual)))
    mean_reference = float(np.mean(np.abs(reference_values)))
    reference_p99 = mask_metadata["reference_brain_positive_p99"]

    bbox = _bounding_box(brain)
    low, high = np.percentile(reference_values, [1.0, 99.0])
    data_range = float(high - low)
    reference_crop = np.clip(reference[bbox], low, high).astype(np.float32)
    candidate_crop = np.clip(scaled[bbox], low, high).astype(np.float32)
    ssim_3d = float(
        structural_similarity(reference_crop, candidate_crop, data_range=data_range)
    )
    axial_ssim = []
    for index in range(reference.shape[2]):
        if int(brain[:, :, index].sum()) < 500:
            continue
        axial_ssim.append(
            structural_similarity(
                np.clip(reference[:, :, index], low, high),
                np.clip(scaled[:, :, index], low, high),
                data_range=data_range,
            )
        )

    if reference_gradient is None:
        reference_gradient = _gradient_magnitude(reference)
    candidate_gradient = _gradient_magnitude(scaled)
    edge_preservation = float(
        np.mean(candidate_gradient[edge])
        / max(float(np.mean(reference_gradient[edge])), 1e-12)
    )
    valid_fov = np.ones(reference.shape, dtype=bool)
    metrics = {
        "intensity_scale_lsq": scale,
        "ncc_brain": normalized_cross_correlation(reference, scaled, brain),
        "ncc_full_fov": normalized_cross_correlation(reference, scaled, valid_fov),
        "nrmse_brain": rmse / max(rms_reference, 1e-12),
        "mae_brain": mae,
        "nmae_brain": mae / max(mean_reference, 1e-12),
        "psnr_p99_db": 20.0 * math.log10(reference_p99 / max(rmse, 1e-12)),
        "ssim_3d_brain_bbox": ssim_3d,
        "ssim_axial_brain_mean": float(np.mean(axial_ssim)),
        "ssim_axial_brain_slice_count": float(len(axial_ssim)),
        "gradient_ncc_brain_edge": normalized_cross_correlation(
            reference_gradient, candidate_gradient, edge
        ),
        "edge_preservation_ratio": edge_preservation,
        "background_std_normalized_p99": float(np.std(scaled[background]))
        / reference_p99,
        "background_mean_abs_normalized_p99": float(np.mean(np.abs(scaled[background])))
        / reference_p99,
        "background_p95_abs_normalized_p99": float(
            np.percentile(np.abs(scaled[background]), 95.0)
        )
        / reference_p99,
        "anatomy_missed_brain_fraction": float(
            np.mean(
                scaled[brain] < mask_metadata["anatomy_missed_intensity_threshold"]
            )
        ),
    }
    return metrics, scaled


def _case_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    kind_order = {"lambda0": 0, "wavelet": 1, "llr": 2}
    return (
        kind_order[case["kind"]],
        case.get("block_size") or 0,
        float(case["lambda"]),
    )


def load_cases(input_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for record in input_manifest["reconstruction_cases"]:
        magnitude = [item for item in record["outputs"] if item["part"] == "mag"]
        if len(magnitude) != 1:
            raise ValueError(f"Expected one magnitude output for {record['case']}")
        path = Path(magnitude[0]["nifti"]["destination"])
        if not path.is_file():
            raise FileNotFoundError(path)
        cases.append(
            {
                "case": record["case"],
                "kind": record["kind"],
                "lambda": float(record["lambda"]),
                "lambda_label": record["lambda_label"],
                "block_size": record["block_size"],
                "backend": record["backend"],
                "source_nifti": path,
                "source_sha256": magnitude[0]["nifti"]["sha256"],
            }
        )
    return sorted(cases, key=_case_sort_key)


def save_registered_case(
    data: np.ndarray,
    reference_image: nib.spatialimages.SpatialImage,
    case: dict[str, Any],
    output_root: Path,
    registration_signature: str,
) -> tuple[Path, Path, str]:
    output_dir = output_root / case["case"]
    output_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = output_dir / "magnitude_registered_to_dicom.nii.gz"
    json_path = output_dir / "magnitude_registered_to_dicom.json"
    expected = {
        "source_nifti": str(case["source_nifti"]),
        "source_sha256": case["source_sha256"],
        "registration_signature": registration_signature,
    }
    if nifti_path.is_file() and json_path.is_file():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            return nifti_path, json_path, existing["output_sha256"]
        raise FileExistsError(f"Existing registered output has different provenance: {nifti_path}")
    if nifti_path.exists() or json_path.exists():
        raise FileExistsError(f"Incomplete registered output pair: {output_dir}")
    header = reference_image.header.copy()
    header.set_data_dtype(np.float32)
    header["descrip"] = b"Shared rigid registration to unfiltered ND DICOM"
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference_image.affine, header)
    nib.save(image, str(nifti_path))
    output_hash = sha256_file(nifti_path)
    _write_json(
        json_path,
        {
            **expected,
            "output_sha256": output_hash,
            "shape": [int(value) for value in data.shape],
            "voxel_size_mm": [float(value) for value in reference_image.header.get_zooms()[:3]],
            "orientation": list(nib.aff2axcodes(reference_image.affine)),
            "intensity_scaling": "none; per-case LSQ scale is recorded only in metrics",
        },
    )
    return nifti_path, json_path, output_hash


def save_mask(
    mask: np.ndarray, name: str, reference_image: nib.spatialimages.SpatialImage, output_dir: Path
) -> Path:
    path = output_dir / f"mask_{name}.nii.gz"
    header = reference_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference_image.affine, header), str(path))
    return path


def _plane(data: np.ndarray, plane: str, center: Sequence[int]) -> np.ndarray:
    x, y, z = center
    if plane == "sagittal":
        return data[x, :, :].T
    if plane == "coronal":
        return data[:, y, :].T
    if plane == "axial":
        return data[:, :, z].T
    raise ValueError(plane)


def _directions(axis: plt.Axes, plane: str) -> None:
    left, right = ("L", "R") if plane != "sagittal" else ("P", "A")
    bottom, top = ("P", "A") if plane == "axial" else ("I", "S")
    style = dict(color="white", fontsize=8, weight="bold")
    axis.text(0.02, 0.5, left, transform=axis.transAxes, **style)
    axis.text(0.98, 0.5, right, ha="right", transform=axis.transAxes, **style)
    axis.text(0.5, 0.02, bottom, ha="center", transform=axis.transAxes, **style)
    axis.text(0.5, 0.98, top, ha="center", va="top", transform=axis.transAxes, **style)


def _rgb(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref_max = float(np.percentile(reference[reference > 0], 99.5))
    cand_max = float(np.percentile(candidate[candidate > 0], 99.5))
    return np.stack(
        (
            np.clip(reference / ref_max, 0, 1),
            np.clip(candidate / cand_max, 0, 1),
            np.zeros_like(reference),
        ),
        axis=-1,
    )


def plot_registration_qc(
    reference: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    center: Sequence[int],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    vmax = float(np.percentile(reference[reference > 0], 99.5))
    for row, plane in enumerate(("sagittal", "coronal", "axial")):
        reference_slice = _plane(reference, plane, center)
        before_slice = _plane(before, plane, center)
        after_slice = _plane(after, plane, center)
        panels = [
            (reference_slice, "gray", "DICOM"),
            (_rgb(reference_slice, before_slice), None, "Before rigid"),
            (_rgb(reference_slice, after_slice), None, "After shared rigid"),
            (np.abs(reference_slice / vmax - after_slice / max(np.percentile(after[after > 0], 99.5), 1e-12)), "magma", "Normalized |difference|"),
        ]
        for column, (data, cmap, title) in enumerate(panels):
            axes[row, column].imshow(data, cmap=cmap, origin="lower")
            axes[row, column].set_title(f"{title}\n{plane}", fontsize=9)
            axes[row, column].set_axis_off()
            _directions(axes[row, column], plane)
    figure.suptitle("λ=0 shared registration QC — overlays: red=DICOM, green=recon", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_mask_qc(
    reference: np.ndarray,
    masks: dict[str, np.ndarray],
    center: Sequence[int],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    vmax = float(np.percentile(reference[reference > 0], 99.5))
    for axis, plane in zip(axes, ("sagittal", "coronal", "axial")):
        axis.imshow(_plane(reference, plane, center), cmap="gray", origin="lower", vmin=0, vmax=vmax)
        axis.contour(_plane(masks["brain"], plane, center), levels=[0.5], colors="lime", linewidths=0.6)
        axis.contour(_plane(masks["edge"], plane, center), levels=[0.5], colors="yellow", linewidths=0.35)
        axis.contour(_plane(masks["background"], plane, center), levels=[0.5], colors="cyan", linewidths=0.35)
        axis.set_title(f"{plane}: BET brain=green, edge=yellow, background=cyan", fontsize=8)
        axis.set_axis_off()
        _directions(axis, plane)
    figure.suptitle("Fixed DICOM-space masks used for every case", fontsize=13)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _rank_records(records: list[dict[str, Any]]) -> None:
    regularized = [record for record in records if record["kind"] != "lambda0"]
    specifications = [
        ("nrmse_brain", False),
        ("ssim_3d_brain_bbox", True),
        ("ncc_brain", True),
        ("gradient_ncc_brain_edge", True),
        ("edge_preservation_ratio", True),
    ]
    rank_sums = {record["case"]: 0.0 for record in regularized}
    for key, higher_is_better in specifications:
        ordered = sorted(regularized, key=lambda record: record[key], reverse=higher_is_better)
        for rank_value, record in enumerate(ordered, start=1):
            rank_sums[record["case"]] += rank_value
    for record in records:
        record["composite_mean_rank"] = (
            rank_sums[record["case"]] / len(specifications)
            if record["case"] in rank_sums
            else float("nan")
        )


def write_metrics_csv(records: list[dict[str, Any]], path: Path) -> None:
    columns = list(records[0].keys())
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def plot_wavelet(records: list[dict[str, Any]], path: Path) -> None:
    rows = sorted((r for r in records if r["kind"] == "wavelet"), key=lambda r: r["lambda"])
    metrics = [
        ("nrmse_brain", "Brain NRMSE ↓"),
        ("ssim_3d_brain_bbox", "Brain 3D SSIM ↑"),
        ("gradient_ncc_brain_edge", "Brain-edge gradient NCC ↑"),
        ("background_std_normalized_p99", "Background SD / DICOM p99 ↓"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    lambdas = [r["lambda"] for r in rows]
    for axis, (key, label_text) in zip(axes.ravel(), metrics):
        axis.semilogx(lambdas, [r[key] for r in rows], marker="o")
        axis.set_xlabel("Wavelet λ")
        axis.set_ylabel(label_text)
        axis.grid(alpha=0.3)
    figure.suptitle("Wavelet whole-volume metrics", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_llr_heatmaps(records: list[dict[str, Any]], path: Path) -> None:
    rows = [r for r in records if r["kind"] == "llr"]
    blocks = sorted({int(r["block_size"]) for r in rows})
    lambdas = sorted({float(r["lambda"]) for r in rows})
    metrics = [
        ("nrmse_brain", "Brain NRMSE ↓", "viridis_r"),
        ("ssim_3d_brain_bbox", "Brain 3D SSIM ↑", "viridis"),
        ("gradient_ncc_brain_edge", "Brain-edge gradient NCC ↑", "viridis"),
        ("background_std_normalized_p99", "Background SD / p99 ↓", "viridis_r"),
    ]
    lookup = {(int(r["block_size"]), float(r["lambda"])): r for r in rows}
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, (key, title, cmap) in zip(axes.ravel(), metrics):
        matrix = np.asarray([[lookup[(block, value)][key] for block in blocks] for value in lambdas])
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_xticks(range(len(blocks)), labels=blocks)
        axis.set_yticks(range(len(lambdas)), labels=[f"{value:g}" for value in lambdas])
        axis.set_xlabel("LLR block size")
        axis.set_ylabel("λ")
        axis.set_title(title)
        for row_index in range(len(lambdas)):
            for column_index in range(len(blocks)):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.3g}", ha="center", va="center", color="white", fontsize=7)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("LLR whole-volume metric heatmaps", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_summary(records: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    styles = {"lambda0": ("black", "*"), "wavelet": ("tab:blue", "o"), "llr": ("tab:orange", "s")}
    for kind in ("lambda0", "wavelet", "llr"):
        rows = [record for record in records if record["kind"] == kind]
        color, marker = styles[kind]
        axes[0].scatter([r["nrmse_brain"] for r in rows], [r["ssim_3d_brain_bbox"] for r in rows], label=kind, color=color, marker=marker)
        axes[1].scatter([r["background_std_normalized_p99"] for r in rows], [r["gradient_ncc_brain_edge"] for r in rows], label=kind, color=color, marker=marker)
    axes[0].set_xlabel("Brain NRMSE ↓")
    axes[0].set_ylabel("Brain 3D SSIM ↑")
    axes[1].set_xlabel("Background SD / DICOM p99 ↓")
    axes[1].set_ylabel("Brain-edge gradient NCC ↑")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Regularization trade-offs across the entire 3D volume", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_representatives(
    reference: np.ndarray,
    records: list[dict[str, Any]],
    center: Sequence[int],
    path: Path,
) -> list[str]:
    lambda0 = next(record for record in records if record["kind"] == "lambda0")
    best_wavelet = min(
        (record for record in records if record["kind"] == "wavelet"),
        key=lambda record: record["composite_mean_rank"],
    )
    best_llr = min(
        (record for record in records if record["kind"] == "llr"),
        key=lambda record: record["composite_mean_rank"],
    )
    selected = [lambda0, best_wavelet, best_llr]
    volumes = [reference]
    for record in selected:
        volumes.append(_finite_magnitude(nib.load(record["registered_nifti"])))
    titles = ["DICOM", "λ=0", f"Wavelet composite leader\n{best_wavelet['case']}", f"LLR composite leader\n{best_llr['case']}"]
    vmaxes = [float(np.percentile(volume[volume > 0], 99.5)) for volume in volumes]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for row, plane in enumerate(("coronal", "axial")):
        for column, (volume, title, vmax) in enumerate(zip(volumes, titles, vmaxes)):
            axes[row, column].imshow(_plane(volume, plane, center), cmap="gray", origin="lower", vmin=0, vmax=vmax)
            axes[row, column].set_title(f"{title}\n{plane}", fontsize=8)
            axes[row, column].set_axis_off()
            _directions(axes[row, column], plane)
    figure.suptitle("DICOM and representative registered reconstructions", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return [record["case"] for record in selected]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--orientation-report", required=True, type=Path)
    parser.add_argument("--brain-mask", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest_path = args.input_manifest.expanduser().resolve()
    orientation_report_path = args.orientation_report.expanduser().resolve()
    brain_mask_path = args.brain_mask.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    orientation_report = json.loads(orientation_report_path.read_text(encoding="utf-8"))
    if orientation_report.get("status") != "orientation_approved":
        raise ValueError("Orientation report does not contain explicit user approval")
    decision = orientation_report["decision_fields"]
    if decision.get("user_approved_best_signed_axis_mapping") is not True:
        raise ValueError("Best signed-axis mapping was not approved")
    mapping = decision["approved_mapping"]
    permutation = mapping["permutation"]
    flips = mapping["flips_ras_grid_axes"]
    if np.linalg.det(np.diag([-1.0 if flip else 1.0 for flip in flips])) < 0:
        raise ValueError("Approved signed-axis mapping is a reflection, not a proper rotation")

    reference_path = Path(input_manifest["dicom_reference_nifti"]["path"])
    reference_image = nib.as_closest_canonical(nib.load(str(reference_path)))
    if tuple(nib.aff2axcodes(reference_image.affine)) != ("R", "A", "S"):
        raise ValueError("DICOM reference did not canonicalize to RAS")
    reference = _finite_magnitude(reference_image)
    cases = load_cases(input_manifest)
    if len(cases) != int(input_manifest["case_count"]):
        raise ValueError(
            f"Manifest declares {input_manifest['case_count']} cases, found {len(cases)}"
        )
    if not any(case["kind"] == "wavelet" for case in cases) or not any(
        case["kind"] == "llr" for case in cases
    ):
        raise ValueError("Evaluation requires lambda zero plus Wavelet and corrected LLR cases")

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    registered_root = output_dir / "registered"
    brain_mask = load_brain_mask(brain_mask_path, reference_image)
    masks, mask_metadata = build_fixed_masks(reference, brain_mask)
    mask_metadata.update(
        {
            "approved_brain_mask_path": str(brain_mask_path),
            "approved_brain_mask_sha256": sha256_file(brain_mask_path),
        }
    )
    mask_records = {}
    for name, mask in masks.items():
        mask_path = save_mask(mask, name, reference_image, masks_dir)
        mask_records[name] = {"path": str(mask_path), "sha256": sha256_file(mask_path)}

    lambda0_case = next(case for case in cases if case["kind"] == "lambda0")
    lambda0_oriented = _load_on_reference_grid(
        lambda0_case["source_nifti"], reference_image, permutation, flips
    )
    rigid = estimate_shared_rigid(reference, lambda0_oriented, masks["brain"])
    parameters = [
        *rigid["parameters"]["rotation_degrees_ras_xyz"],
        *rigid["parameters"]["translation_mm_ras_xyz"],
    ]
    lambda0_registered = rigid_resample(lambda0_oriented, parameters)
    rigid["full_resolution_ncc_before"] = normalized_cross_correlation(
        reference, lambda0_oriented, masks["brain"]
    )
    rigid["full_resolution_ncc_after"] = normalized_cross_correlation(
        reference, lambda0_registered, masks["brain"]
    )
    if rigid["full_resolution_ncc_after"] <= rigid["full_resolution_ncc_before"]:
        raise ValueError("Shared rigid registration reduced full-resolution NCC")
    registration_configuration = {
        "approved_orientation_mapping": {"permutation": permutation, "flips_ras_grid_axes": flips},
        "rigid": rigid,
        "reference_nifti": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
    }
    registration_signature = hashlib.sha256(
        json.dumps(registration_configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    registration_configuration["registration_signature"] = registration_signature
    _write_json(output_dir / "shared_registration.json", registration_configuration)

    brain_coordinates = np.argwhere(masks["brain"])
    center = [int(round(value)) for value in brain_coordinates.mean(axis=0)]
    plot_registration_qc(
        reference,
        lambda0_oriented,
        lambda0_registered,
        center,
        plots_dir / "registration_qc_lambda0.png",
    )
    plot_mask_qc(reference, masks, center, plots_dir / "fixed_masks_qc.png")

    records = []
    reference_gradient = _gradient_magnitude(reference)
    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] {case['case']}", flush=True)
        if case["kind"] == "lambda0":
            registered = lambda0_registered
        else:
            oriented = _load_on_reference_grid(
                case["source_nifti"], reference_image, permutation, flips
            )
            registered = rigid_resample(oriented, parameters)
        registered_path, registered_json, registered_hash = save_registered_case(
            registered,
            reference_image,
            case,
            registered_root,
            registration_signature,
        )
        metrics, _ = compute_metrics(
            reference,
            registered,
            masks,
            mask_metadata,
            reference_gradient=reference_gradient,
        )
        records.append(
            {
                "case": case["case"],
                "kind": case["kind"],
                "lambda": case["lambda"],
                "lambda_label": case["lambda_label"],
                "block_size": "" if case["block_size"] is None else case["block_size"],
                "backend": case["backend"],
                "source_nifti": str(case["source_nifti"]),
                "source_sha256": case["source_sha256"],
                "registered_nifti": str(registered_path),
                "registered_json": str(registered_json),
                "registered_sha256": registered_hash,
                **metrics,
            }
        )
        del registered
    _rank_records(records)
    metrics_path = output_dir / "metrics.csv"
    write_metrics_csv(records, metrics_path)
    plot_wavelet(records, plots_dir / "wavelet_metrics.png")
    plot_llr_heatmaps(records, plots_dir / "llr_metric_heatmaps.png")
    plot_summary(records, plots_dir / "metric_tradeoffs.png")
    representative_cases = plot_representatives(
        reference,
        records,
        center,
        plots_dir / "dicom_representative_comparison.png",
    )

    provenance = {
        "format_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "orientation_report": str(orientation_report_path),
        "orientation_report_sha256": sha256_file(orientation_report_path),
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "shape": list(reference.shape),
            "voxel_size_mm": list(reference_image.header.get_zooms()[:3]),
            "orientation": list(nib.aff2axcodes(reference_image.affine)),
            "dicom_series": input_manifest["selected_dicom_series"],
        },
        "approved_orientation_mapping": registration_configuration["approved_orientation_mapping"],
        "shared_registration": registration_configuration,
        "fixed_masks": {"metadata": mask_metadata, "files": mask_records},
        "metric_definitions": {
            "intensity_scaling": "one positive LSQ scale per case inside fixed BET brain mask",
            "nrmse_brain": "RMSE divided by RMS(DICOM) inside fixed BET brain mask",
            "psnr_p99_db": "20*log10(DICOM brain-positive p99 / brain-mask RMSE)",
            "ssim_3d_brain_bbox": "3D SSIM on clipped fixed-BET-brain bounding box",
            "gradient_ncc_brain_edge": "NCC of sigma-0.7 gradient magnitudes in fixed brain-edge mask",
            "background_metrics": "scaled reconstruction signal in fixed DICOM-space background mask",
            "anatomy_missed_brain_fraction": "fixed BET brain fraction below 0.05 of DICOM brain-positive p99",
            "composite_mean_rank": "mean rank over brain NRMSE, brain 3D SSIM, brain NCC, brain-edge gradient NCC, and edge preservation; regularized cases only",
        },
        "metrics_csv": {"path": str(metrics_path), "sha256": sha256_file(metrics_path), "row_count": len(records)},
        "representative_cases": representative_cases,
        "plots": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(plots_dir.glob("*.png"))
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
            "scikit_image": skimage_version,
            "matplotlib": matplotlib.__version__,
        },
        "case_records": records,
        "notes": [
            "All metrics use the entire 3D volume or fixed 3D masks; slice SSIM is secondary.",
            "No independent per-case registration was performed.",
            "WM/GM CoV and CNR were omitted because no validated tissue segmentation was available.",
        ],
    }
    provenance_path = output_dir / "metrics_provenance.json"
    _write_json(provenance_path, provenance)
    print(f"Metrics CSV: {metrics_path}")
    print(f"Provenance: {provenance_path}")
    return provenance


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
