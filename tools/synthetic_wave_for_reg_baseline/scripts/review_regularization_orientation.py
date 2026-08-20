#!/usr/bin/env python3
"""Generate a no-registration, explicitly labeled L/R orientation review package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import permutations, product
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _corners(shape: Sequence[int], edges: bool) -> np.ndarray:
    if edges:
        limits = [(-0.5, float(size) - 0.5) for size in shape[:3]]
    else:
        limits = [(0.0, float(size) - 1.0) for size in shape[:3]]
    return np.asarray(list(product(*limits)), dtype=float)


def image_geometry(path: Path) -> dict[str, Any]:
    image = nib.load(str(path))
    affine = np.asarray(image.affine, dtype=float)
    center_world = nib.affines.apply_affine(
        affine, (np.asarray(image.shape[:3], dtype=float) - 1.0) / 2.0
    )
    center_corners = nib.affines.apply_affine(affine, _corners(image.shape, False))
    edge_corners = nib.affines.apply_affine(affine, _corners(image.shape, True))
    canonical = nib.as_closest_canonical(image)
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": [int(value) for value in image.shape],
        "storage_dtype": str(image.get_data_dtype()),
        "voxel_size_mm": [float(value) for value in image.header.get_zooms()[:3]],
        "axis_codes": list(nib.aff2axcodes(affine)),
        "affine": affine.tolist(),
        "affine_determinant": float(np.linalg.det(affine[:3, :3])),
        "qform_code": int(qform_code),
        "qform": None if qform is None else np.asarray(qform).tolist(),
        "sform_code": int(sform_code),
        "sform": None if sform is None else np.asarray(sform).tolist(),
        "world_center_mm": center_world.tolist(),
        "voxel_center_world_bounds_mm": {
            "minimum": center_corners.min(axis=0).tolist(),
            "maximum": center_corners.max(axis=0).tolist(),
        },
        "voxel_edge_world_bounds_mm": {
            "minimum": edge_corners.min(axis=0).tolist(),
            "maximum": edge_corners.max(axis=0).tolist(),
        },
        "canonical_axis_codes": list(nib.aff2axcodes(canonical.affine)),
        "canonical_shape": [int(value) for value in canonical.shape],
        "canonical_affine": np.asarray(canonical.affine).tolist(),
    }


def _finite_magnitude(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = np.asarray(image.dataobj, dtype=np.float32)
    data = np.abs(data)
    data[~np.isfinite(data)] = 0.0
    return data


def _foreground_mask(reference: np.ndarray) -> tuple[np.ndarray, float]:
    positive = reference[reference > 0]
    if not positive.size:
        raise ValueError("DICOM reference has no positive voxels")
    threshold = 0.05 * float(np.percentile(positive, 99.0))
    mask = reference > threshold
    if int(mask.sum()) < 1000:
        raise ValueError("DICOM foreground mask is unexpectedly small")
    return mask, threshold


def _weighted_center(reference: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    weights = np.where(mask, reference, 0.0).astype(np.float64, copy=False)
    total = float(weights.sum())
    coordinates = np.indices(reference.shape, dtype=np.float64)
    center = [int(round(float((coordinates[axis] * weights).sum()) / total)) for axis in range(3)]
    return tuple(max(0, min(reference.shape[axis] - 1, value)) for axis, value in enumerate(center))


def _lsq_scale(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    ref_values = reference[mask].astype(np.float64)
    candidate_values = candidate[mask].astype(np.float64)
    denominator = float(np.dot(candidate_values, candidate_values))
    if denominator <= 0.0:
        raise ValueError("Candidate is zero inside the DICOM foreground mask")
    return float(np.dot(ref_values, candidate_values) / denominator)


def _ncc(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    ref_values = reference[mask].astype(np.float64)
    candidate_values = candidate[mask].astype(np.float64)
    ref_values -= ref_values.mean()
    candidate_values -= candidate_values.mean()
    denominator = float(np.linalg.norm(ref_values) * np.linalg.norm(candidate_values))
    return float(np.dot(ref_values, candidate_values) / denominator) if denominator else float("nan")


def _signed_axis_transform(
    data: np.ndarray, permutation: Sequence[int], flips: Sequence[bool]
) -> np.ndarray:
    transformed = np.transpose(data, tuple(permutation))
    for axis, flip in enumerate(flips):
        if flip:
            transformed = np.flip(transformed, axis=axis)
    return transformed


def _signed_axis_search(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    downsample_factor: int = 4,
) -> list[dict[str, Any]]:
    """Rank all 48 signed axis mappings as orientation diagnostics, not registration."""
    stride = (slice(None, None, downsample_factor),) * 3
    reference_small = reference[stride]
    mask_small = mask[stride]
    results = []
    for permutation in permutations(range(3)):
        for flips in product((False, True), repeat=3):
            transformed = _signed_axis_transform(candidate, permutation, flips)
            results.append(
                {
                    "permutation": list(permutation),
                    "flips_ras_grid_axes": list(flips),
                    "ncc": _ncc(reference_small, transformed[stride], mask_small),
                }
            )
    return sorted(results, key=lambda item: item["ncc"], reverse=True)


def _plane_slice(data: np.ndarray, plane: str, indices: tuple[int, int, int]) -> np.ndarray:
    x, y, z = indices
    if plane == "sagittal":
        return data[x, :, :].T
    if plane == "coronal":
        return data[:, y, :].T
    if plane == "axial":
        return data[:, :, z].T
    raise ValueError(f"Unsupported plane: {plane}")


def _plane_indices(
    center: tuple[int, int, int], plane: str, offset: int = 0
) -> tuple[int, int, int]:
    values = list(center)
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
    values[axis] += offset
    return tuple(values)


def _annotate_directions(axis: plt.Axes, plane: str) -> None:
    if plane in {"axial", "coronal"}:
        left, right = "L", "R"
    else:
        left, right = "P", "A"
    bottom, top = ("P", "A") if plane == "axial" else ("I", "S")
    style = dict(color="white", fontsize=10, weight="bold",
                 bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5))
    axis.text(0.02, 0.50, left, ha="left", va="center", transform=axis.transAxes, **style)
    axis.text(0.98, 0.50, right, ha="right", va="center", transform=axis.transAxes, **style)
    axis.text(0.50, 0.02, bottom, ha="center", va="bottom", transform=axis.transAxes, **style)
    axis.text(0.50, 0.98, top, ha="center", va="top", transform=axis.transAxes, **style)


def _show_gray(
    axis: plt.Axes, data: np.ndarray, plane: str, title: str, vmax: float
) -> None:
    axis.imshow(data, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()
    _annotate_directions(axis, plane)


def _rgb_overlay(reference: np.ndarray, candidate: np.ndarray, ref_vmax: float, cand_vmax: float) -> np.ndarray:
    red = np.clip(reference / max(ref_vmax, 1e-12), 0.0, 1.0)
    green = np.clip(candidate / max(cand_vmax, 1e-12), 0.0, 1.0)
    return np.stack((red, green, np.zeros_like(red)), axis=-1)


def _show_rgb(axis: plt.Axes, rgb: np.ndarray, plane: str, title: str) -> None:
    axis.imshow(rgb, origin="lower")
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()
    _annotate_directions(axis, plane)


def make_lr_choice_figure(
    reference: np.ndarray,
    current: np.ndarray,
    mirrored: np.ndarray,
    center: tuple[int, int, int],
    output_path: Path,
) -> None:
    planes = ("coronal", "axial")
    ref_vmax = float(np.percentile(reference[reference > 0], 99.5))
    current_vmax = float(np.percentile(current[current > 0], 99.5))
    mirrored_vmax = float(np.percentile(mirrored[mirrored > 0], 99.5))
    figure, axes = plt.subplots(len(planes), 5, figsize=(17, 7.2), constrained_layout=True)
    for row, plane in enumerate(planes):
        indices = _plane_indices(center, plane)
        ref_slice = _plane_slice(reference, plane, indices)
        current_slice = _plane_slice(current, plane, indices)
        mirrored_slice = _plane_slice(mirrored, plane, indices)
        _show_gray(axes[row, 0], ref_slice, plane, f"DICOM reference\n{plane}", ref_vmax)
        _show_gray(axes[row, 1], current_slice, plane, "Recon: current physical mapping", current_vmax)
        _show_rgb(
            axes[row, 2],
            _rgb_overlay(ref_slice, current_slice, ref_vmax, current_vmax),
            plane,
            "Current overlay\nred=DICOM, green=recon",
        )
        _show_gray(axes[row, 3], mirrored_slice, plane, "Diagnostic L/R mirror", mirrored_vmax)
        _show_rgb(
            axes[row, 4],
            _rgb_overlay(ref_slice, mirrored_slice, ref_vmax, mirrored_vmax),
            plane,
            "Mirror overlay\nred=DICOM, green=recon",
        )
    figure.suptitle(
        "L/R orientation choice before registration — diagnostic mirror is not an accepted transform",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_triplanar_figure(
    reference: np.ndarray,
    current: np.ndarray,
    center: tuple[int, int, int],
    output_path: Path,
) -> None:
    planes = ("sagittal", "coronal", "axial")
    ref_vmax = float(np.percentile(reference[reference > 0], 99.5))
    current_vmax = float(np.percentile(current[current > 0], 99.5))
    figure, axes = plt.subplots(3, 4, figsize=(13.5, 10), constrained_layout=True)
    for row, plane in enumerate(planes):
        indices = _plane_indices(center, plane)
        ref_slice = _plane_slice(reference, plane, indices)
        current_slice = _plane_slice(current, plane, indices)
        checker = ref_slice.copy()
        tile = 16
        yy, xx = np.indices(ref_slice.shape)
        use_candidate = ((xx // tile) + (yy // tile)) % 2 == 1
        checker[use_candidate] = current_slice[use_candidate]
        difference = np.abs(
            np.clip(ref_slice / max(ref_vmax, 1e-12), 0, 1)
            - np.clip(current_slice / max(current_vmax, 1e-12), 0, 1)
        )
        _show_gray(axes[row, 0], ref_slice, plane, f"DICOM reference\n{plane}", ref_vmax)
        _show_gray(axes[row, 1], current_slice, plane, "Recon, no registration", current_vmax)
        _show_gray(axes[row, 2], checker, plane, "16-voxel checkerboard", ref_vmax)
        axes[row, 3].imshow(difference, cmap="magma", origin="lower", vmin=0, vmax=0.5)
        axes[row, 3].set_title("Normalized |difference|", fontsize=9)
        axes[row, 3].set_axis_off()
        _annotate_directions(axes[row, 3], plane)
    figure.suptitle("Initial physical-space agreement (no registration)", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_multislice_lr_figure(
    reference: np.ndarray,
    current: np.ndarray,
    mirrored: np.ndarray,
    center: tuple[int, int, int],
    output_path: Path,
) -> None:
    selections = [("coronal", offset) for offset in (-32, 0, 32)] + [
        ("axial", offset) for offset in (-32, 0, 32)
    ]
    ref_vmax = float(np.percentile(reference[reference > 0], 99.5))
    current_vmax = float(np.percentile(current[current > 0], 99.5))
    mirror_vmax = float(np.percentile(mirrored[mirrored > 0], 99.5))
    figure, axes = plt.subplots(len(selections), 3, figsize=(10.5, 18), constrained_layout=True)
    for row, (plane, offset) in enumerate(selections):
        indices = _plane_indices(center, plane, offset)
        ref_slice = _plane_slice(reference, plane, indices)
        current_slice = _plane_slice(current, plane, indices)
        mirror_slice = _plane_slice(mirrored, plane, indices)
        _show_gray(
            axes[row, 0], ref_slice, plane, f"DICOM {plane}, offset {offset:+d}", ref_vmax
        )
        _show_rgb(
            axes[row, 1],
            _rgb_overlay(ref_slice, current_slice, ref_vmax, current_vmax),
            plane,
            "Current mapping overlay",
        )
        _show_rgb(
            axes[row, 2],
            _rgb_overlay(ref_slice, mirror_slice, ref_vmax, mirror_vmax),
            plane,
            "Diagnostic L/R mirror overlay",
        )
    figure.suptitle("Multi-slice L/R review — red=DICOM, green=reconstruction", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_signed_axis_choice_figure(
    reference: np.ndarray,
    current: np.ndarray,
    lr_only: np.ndarray,
    best: np.ndarray,
    best_label: str,
    center: tuple[int, int, int],
    output_path: Path,
) -> None:
    """Show the header mapping, L/R-only test, and best signed-axis hypothesis."""
    planes = ("sagittal", "coronal", "axial")
    ref_vmax = float(np.percentile(reference[reference > 0], 99.5))
    maxima = [
        float(np.percentile(volume[volume > 0], 99.5))
        for volume in (current, lr_only, best)
    ]
    figure, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for row, plane in enumerate(planes):
        indices = _plane_indices(center, plane)
        ref_slice = _plane_slice(reference, plane, indices)
        current_slice = _plane_slice(current, plane, indices)
        lr_slice = _plane_slice(lr_only, plane, indices)
        best_slice = _plane_slice(best, plane, indices)
        _show_gray(axes[row, 0], ref_slice, plane, f"DICOM reference\n{plane}", ref_vmax)
        _show_rgb(
            axes[row, 1],
            _rgb_overlay(ref_slice, current_slice, ref_vmax, maxima[0]),
            plane,
            "Header-based mapping",
        )
        _show_rgb(
            axes[row, 2],
            _rgb_overlay(ref_slice, lr_slice, ref_vmax, maxima[1]),
            plane,
            "L/R flip only",
        )
        _show_rgb(
            axes[row, 3],
            _rgb_overlay(ref_slice, best_slice, ref_vmax, maxima[2]),
            plane,
            best_label,
        )
    figure.suptitle(
        "Signed-axis orientation hypotheses — red=DICOM, green=reconstruction; no registration",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-nifti", required=True, type=Path)
    parser.add_argument("--lambda0-nifti", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--accept-best-signed-axis",
        action="store_true",
        help="Record explicit user approval of the highest-NCC signed-axis candidate.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dicom_path = args.dicom_nifti.expanduser().resolve()
    lambda0_path = args.lambda0_nifti.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dicom_path.is_file() or not lambda0_path.is_file():
        raise FileNotFoundError("DICOM or lambda-zero NIfTI is missing")
    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_native = nib.load(str(dicom_path))
    lambda0_native = nib.load(str(lambda0_path))
    dicom_canonical = nib.as_closest_canonical(dicom_native)
    lambda0_canonical = nib.as_closest_canonical(lambda0_native)
    if tuple(nib.aff2axcodes(dicom_canonical.affine)) != ("R", "A", "S"):
        raise ValueError("Could not canonicalize DICOM reference to RAS")
    if tuple(nib.aff2axcodes(lambda0_canonical.affine)) != ("R", "A", "S"):
        raise ValueError("Could not canonicalize lambda-zero reconstruction to RAS")

    reference = _finite_magnitude(dicom_canonical)
    resampled_image = resample_from_to(lambda0_canonical, dicom_canonical, order=1)
    current_unscaled = _finite_magnitude(resampled_image)
    mask, threshold = _foreground_mask(reference)
    current_scale = _lsq_scale(reference, current_unscaled, mask)
    current = current_unscaled * current_scale
    mirrored_unscaled = current_unscaled[::-1, :, :]
    mirror_scale = _lsq_scale(reference, mirrored_unscaled, mask)
    mirrored = mirrored_unscaled * mirror_scale
    signed_axis_results = _signed_axis_search(reference, current_unscaled, mask)
    best_mapping = signed_axis_results[0]
    best_unscaled = _signed_axis_transform(
        current_unscaled,
        best_mapping["permutation"],
        best_mapping["flips_ras_grid_axes"],
    )
    best_scale = _lsq_scale(reference, best_unscaled, mask)
    best = best_unscaled * best_scale
    center = _weighted_center(reference, mask)

    lr_path = output_dir / "orientation_lr_choice.png"
    triplanar_path = output_dir / "orientation_triplanar_current.png"
    multislice_path = output_dir / "orientation_lr_multislice.png"
    signed_axis_path = output_dir / "orientation_signed_axis_choice.png"
    make_lr_choice_figure(reference, current, mirrored, center, lr_path)
    make_triplanar_figure(reference, current, center, triplanar_path)
    make_multislice_lr_figure(reference, current, mirrored, center, multislice_path)
    best_label = (
        f"Best: perm={best_mapping['permutation']}\n"
        f"flips RAS={best_mapping['flips_ras_grid_axes']}"
    )
    make_signed_axis_choice_figure(
        reference,
        current,
        mirrored,
        best,
        best_label,
        center,
        signed_axis_path,
    )

    dicom_geometry = image_geometry(dicom_path)
    lambda0_geometry = image_geometry(lambda0_path)
    dicom_affine = np.asarray(dicom_geometry["affine"])
    lambda0_affine = np.asarray(lambda0_geometry["affine"])
    report = {
        "format_version": 1,
        "status": (
            "orientation_approved"
            if args.accept_best_signed_axis
            else "user_lr_assessment_required"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "No registration or orientation correction has been accepted or applied. All signed-"
            "axis variants are diagnostic display candidates only."
        ),
        "dicom_reference": dicom_geometry,
        "lambda0_reconstruction": lambda0_geometry,
        "geometry_comparison": {
            "world_center_difference_lambda0_minus_dicom_mm": (
                np.asarray(lambda0_geometry["world_center_mm"])
                - np.asarray(dicom_geometry["world_center_mm"])
            ).tolist(),
            "lambda0_voxel_to_dicom_voxel": (
                np.linalg.inv(dicom_affine) @ lambda0_affine
            ).tolist(),
            "same_shape": dicom_native.shape == lambda0_native.shape,
            "same_affine_atol_1e-5": bool(
                np.allclose(dicom_native.affine, lambda0_native.affine, atol=1e-5)
            ),
        },
        "initial_no_registration_comparison": {
            "reference_foreground_rule": "DICOM > 0.05 * p99(DICOM positive voxels)",
            "reference_foreground_threshold": threshold,
            "reference_foreground_voxel_count": int(mask.sum()),
            "display_and_diagnostic_scale_current_lsq": current_scale,
            "display_and_diagnostic_scale_lr_mirror_lsq": mirror_scale,
            "display_and_diagnostic_scale_best_signed_axis_lsq": best_scale,
            "whole_foreground_ncc_current_physical_mapping": _ncc(reference, current, mask),
            "whole_foreground_ncc_diagnostic_lr_mirror": _ncc(reference, mirrored, mask),
            "whole_foreground_ncc_best_signed_axis_candidate": _ncc(reference, best, mask),
            "weighted_reference_center_voxel_ras_grid": list(center),
            "signed_axis_search": {
                "purpose": "orientation diagnostic only; not an accepted transform",
                "downsample_factor": 4,
                "axis_order": ["R/L", "A/P", "S/I"],
                "candidate_count": len(signed_axis_results),
                "top_candidates": signed_axis_results[:10],
            },
        },
        "figures": [
            {"path": str(lr_path), "sha256": sha256_file(lr_path)},
            {"path": str(triplanar_path), "sha256": sha256_file(triplanar_path)},
            {"path": str(multislice_path), "sha256": sha256_file(multislice_path)},
            {"path": str(signed_axis_path), "sha256": sha256_file(signed_axis_path)},
        ],
        "decision_fields": {
            "user_approved_best_signed_axis_mapping": (
                True if args.accept_best_signed_axis else None
            ),
            "approved_mapping": best_mapping if args.accept_best_signed_axis else None,
            "reviewed_at_utc": (
                datetime.now(timezone.utc).isoformat()
                if args.accept_best_signed_axis
                else None
            ),
            "notes": (
                "User confirmed the last column of orientation_signed_axis_choice.png."
                if args.accept_best_signed_axis
                else None
            ),
        },
    }
    report_path = output_dir / "orientation_report.json"
    _write_json(report_path, report)
    print(f"Orientation report: {report_path}")
    for figure in report["figures"]:
        print(f"Review figure: {figure['path']}")
    if args.accept_best_signed_axis:
        print(f"Accepted signed-axis mapping: {best_mapping}")
    else:
        print("STOP: user L/R assessment is required before registration or metrics.")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
