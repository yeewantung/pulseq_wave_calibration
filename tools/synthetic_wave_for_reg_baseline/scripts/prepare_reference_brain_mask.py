#!/usr/bin/env python3
"""Create and document one fixed FSL BET mask from a canonical RAS reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_bet_command(
    bet: Path,
    reference: Path,
    output_base: Path,
    fractional_threshold: float,
    robust_center: bool = False,
) -> list[str]:
    if not 0.0 < fractional_threshold < 1.0:
        raise ValueError("BET fractional threshold must lie strictly between zero and one")
    command = [
        str(bet),
        str(reference),
        str(output_base),
    ]
    if robust_center:
        # Full-head references with substantial neck benefit from BET's repeated
        # center-of-gravity estimation before the final surface extraction.
        command.append("-R")
    return command + ["-m", "-f", f"{fractional_threshold:.6g}", "-g", "0"]


def _plane(data: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "sagittal":
        return data[index, :, :].T
    if plane == "coronal":
        return data[:, index, :].T
    if plane == "axial":
        return data[:, :, index].T
    raise ValueError(plane)


def expand_mask(mask: np.ndarray, dilation_voxels: int) -> np.ndarray:
    """Expand a 3D mask by a reproducible face-connected voxel radius."""
    if mask.ndim != 3:
        raise ValueError(f"Brain mask must be 3D, got shape {mask.shape}")
    if dilation_voxels < 0:
        raise ValueError("Mask dilation must be non-negative")
    if dilation_voxels == 0:
        return mask.astype(bool, copy=True)
    return binary_dilation(mask, iterations=dilation_voxels)


def _annotate_directions(axis: plt.Axes, plane: str) -> None:
    left, right = (("P", "A") if plane == "sagittal" else ("L", "R"))
    bottom, top = (("P", "A") if plane == "axial" else ("I", "S"))
    style = dict(
        color="white",
        fontsize=8,
        weight="bold",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1),
    )
    axis.text(0.02, 0.5, left, transform=axis.transAxes, va="center", **style)
    axis.text(0.98, 0.5, right, transform=axis.transAxes, ha="right", va="center", **style)
    axis.text(0.5, 0.02, bottom, transform=axis.transAxes, ha="center", **style)
    axis.text(0.5, 0.98, top, transform=axis.transAxes, ha="center", va="top", **style)


def make_mask_qc(
    reference: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    native_bet_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    coordinates = np.argwhere(mask)
    center = np.rint(coordinates.mean(axis=0)).astype(int)
    offsets = (-32, 0, 32)
    planes = ("sagittal", "coronal", "axial")
    vmax = float(np.percentile(reference[mask], 99.5))
    figure, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
    for row, plane in enumerate(planes):
        axis_index = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
        for column, offset in enumerate(offsets):
            index = int(np.clip(center[axis_index] + offset, 0, mask.shape[axis_index] - 1))
            reference_slice = _plane(reference, plane, index)
            mask_slice = _plane(mask, plane, index)
            axes[row, column].imshow(
                reference_slice, cmap="gray", origin="lower", vmin=0.0, vmax=vmax
            )
            if np.any(mask_slice) and not np.all(mask_slice):
                axes[row, column].contour(
                    mask_slice,
                    levels=[0.5],
                    colors="#00ff66",
                    linewidths=0.8,
                )
            if native_bet_mask is not None:
                native_slice = _plane(native_bet_mask, plane, index)
                if np.any(native_slice) and not np.all(native_slice):
                    axes[row, column].contour(
                        native_slice,
                        levels=[0.5],
                        colors="#ffb000",
                        linewidths=0.7,
                        linestyles="--",
                    )
            axes[row, column].set_title(
                (
                    f"{plane}, index {index}; expanded=green, native BET=orange"
                    if native_bet_mask is not None
                    else f"{plane}, index {index}; BET boundary=green"
                ),
                fontsize=9,
            )
            axes[row, column].set_axis_off()
            _annotate_directions(axes[row, column], plane)
    figure.suptitle("Candidate fixed brain-mask boundary on reference", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {"center_voxel_ras_grid": center.tolist(), "slice_offsets_voxels": list(offsets)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bet", type=Path, default=Path("/path/to/software/packages/fsl/6.0.6/share/fsl/bin/bet"))
    parser.add_argument("--fractional-threshold", type=float, default=0.25)
    parser.add_argument(
        "--robust-center",
        action="store_true",
        help="Use BET's repeated center-of-gravity estimation for full-head inputs.",
    )
    parser.add_argument(
        "--mask-dilation-voxels",
        type=int,
        default=0,
        help=(
            "Expand the native BET mask outward by this many face-connected voxels; "
            "the native mask is preserved for QC (default: 0)."
        ),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference_path = args.reference.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    bet = args.bet.expanduser().resolve()
    if not reference_path.is_file() or not bet.is_file():
        raise FileNotFoundError("Reference NIfTI or FSL BET executable is missing")
    if args.mask_dilation_voxels < 0:
        raise ValueError("Mask dilation must be non-negative")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Brain-mask output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_image = nib.load(str(reference_path))
    if tuple(nib.aff2axcodes(reference_image.affine)) != ("R", "A", "S"):
        raise ValueError("BET reference must already be canonical RAS")
    if len(reference_image.shape) != 3:
        raise ValueError(f"BET reference must be 3D, got {reference_image.shape}")

    output_base = output_dir / "reference_brain"
    command = build_bet_command(
        bet,
        reference_path,
        output_base,
        args.fractional_threshold,
        args.robust_center,
    )
    environment = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"}
    process = subprocess.run(
        command, text=True, capture_output=True, check=False, env=environment
    )
    log_path = output_dir / "bet.log"
    log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"FSL BET failed with status {process.returncode}")

    brain_path = output_base.with_suffix(".nii.gz")
    mask_path = output_dir / "reference_brain_mask.nii.gz"
    for path in (brain_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    mask_image = nib.load(str(mask_path))
    if mask_image.shape != reference_image.shape or not np.allclose(
        mask_image.affine, reference_image.affine, atol=1e-5
    ):
        raise ValueError("BET mask does not share the exact reference grid")
    native_bet_mask = np.asarray(mask_image.dataobj) > 0
    mask = expand_mask(native_bet_mask, args.mask_dilation_voxels)
    native_mask_path = None
    native_brain_path = None
    if args.mask_dilation_voxels:
        # Preserve BET's direct outputs and make the canonical filenames reflect
        # the final expanded mask that downstream metric code will consume.
        native_mask_path = output_dir / "reference_brain_mask_native_bet.nii.gz"
        native_brain_path = output_dir / "reference_brain_native_bet.nii.gz"
        mask_path.replace(native_mask_path)
        brain_path.replace(native_brain_path)

        mask_header = reference_image.header.copy()
        mask_header.set_data_dtype(np.uint8)
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), reference_image.affine, mask_header),
            str(mask_path),
        )
        reference_for_mask = np.asarray(reference_image.dataobj, dtype=np.float32)
        brain_header = reference_image.header.copy()
        brain_header.set_data_dtype(np.float32)
        nib.save(
            nib.Nifti1Image(
                np.where(mask, reference_for_mask, 0.0).astype(np.float32),
                reference_image.affine,
                brain_header,
            ),
            str(brain_path),
        )
    mask_voxels = int(mask.sum())
    if not 1_000_000 < mask_voxels < int(0.8 * mask.size):
        raise ValueError(f"BET mask voxel count is implausible: {mask_voxels}")
    reference = np.asarray(reference_image.dataobj, dtype=np.float32)
    qc_path = output_dir / "reference_brain_mask_qc.png"
    qc = make_mask_qc(
        reference,
        mask,
        qc_path,
        native_bet_mask if args.mask_dilation_voxels else None,
    )

    fsldir = environment.get("FSLDIR")
    fsl_version_path = Path(fsldir) / "etc" / "fslversion" if fsldir else None
    manifest = {
        "format_version": 1,
        "status": "visual_review_required",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "shape": list(reference_image.shape),
            "voxel_size_mm": [
                float(value) for value in reference_image.header.get_zooms()[:3]
            ],
            "orientation": list(nib.aff2axcodes(reference_image.affine)),
        },
        "bet": {
            "executable": str(bet),
            "command": command,
            "fractional_threshold": args.fractional_threshold,
            "robust_center": args.robust_center,
            "vertical_gradient": 0.0,
            "fsl_version": (
                fsl_version_path.read_text(encoding="utf-8").strip()
                if fsl_version_path and fsl_version_path.is_file()
                else None
            ),
            "log": str(log_path),
            "log_sha256": sha256_file(log_path),
        },
        "brain_extracted": {"path": str(brain_path), "sha256": sha256_file(brain_path)},
        "brain_mask": {
            "path": str(mask_path),
            "sha256": sha256_file(mask_path),
            "voxel_count": mask_voxels,
            "volume_ml": float(
                mask_voxels
                * float(np.prod(reference_image.header.get_zooms()[:3]))
                / 1000.0
            ),
        },
        "mask_postprocessing": {
            "dilation_voxels": args.mask_dilation_voxels,
            "connectivity": "3D face-connected",
            "native_bet_voxel_count": int(native_bet_mask.sum()),
            "added_voxel_count": int(mask.sum() - native_bet_mask.sum()),
            "native_bet_mask": (
                {
                    "path": str(native_mask_path),
                    "sha256": sha256_file(native_mask_path),
                }
                if native_mask_path is not None
                else None
            ),
            "native_bet_brain_extracted": (
                {
                    "path": str(native_brain_path),
                    "sha256": sha256_file(native_brain_path),
                }
                if native_brain_path is not None
                else None
            ),
        },
        "qc_figure": {"path": str(qc_path), "sha256": sha256_file(qc_path), **qc},
        "approval": {
            "mask_boundary_visually_approved": None,
            "left_right_orientation_visually_approved": None,
            "reviewed_at_utc": None,
            "notes": None,
        },
    }
    manifest_path = output_dir / "brain_mask_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Brain mask: {mask_path}")
    print(f"Review figure: {qc_path}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
