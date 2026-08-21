#!/usr/bin/env python3
"""Compare an intensity-corrected ESPIRiT pilot without selecting regularization."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt

from evaluate_regularization_volume import (
    _finite_magnitude,
    _load_on_reference_grid,
    normalized_cross_correlation,
    rigid_resample,
    sha256_file,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--metrics-provenance", required=True, type=Path)
    parser.add_argument("--raw-dicom-nifti", required=True, type=Path)
    parser.add_argument("--grappa-nifti", required=True, type=Path)
    parser.add_argument("--sense-nifti", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _magnitude_output(manifest: dict[str, Any]) -> Path:
    outputs = [record for record in manifest["nifti"]["outputs"] if record["part"] == "mag"]
    if len(outputs) != 1:
        raise ValueError("Pilot manifest must contain exactly one magnitude NIfTI")
    return Path(outputs[0]["nifti"]).expanduser().resolve()


def _load_external(
    path: Path,
    reference_image: nib.spatialimages.SpatialImage,
    rigid_parameters: Sequence[float],
) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = _finite_magnitude(resample_from_to(image, reference_image, order=1))
    return rigid_resample(data, rigid_parameters)


def _plane(data: np.ndarray, plane: str, center: Sequence[int]) -> np.ndarray:
    x, y, z = center
    if plane == "coronal":
        return data[:, y, :].T
    if plane == "axial":
        return data[:, :, z].T
    raise ValueError(plane)


def _directions(axis: plt.Axes, plane: str) -> None:
    bottom, top = (("P", "A") if plane == "axial" else ("I", "S"))
    style = dict(
        color="white",
        fontsize=8,
        weight="bold",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1),
    )
    axis.text(0.02, 0.5, "L", transform=axis.transAxes, va="center", **style)
    axis.text(0.98, 0.5, "R", transform=axis.transAxes, ha="right", va="center", **style)
    axis.text(0.5, 0.02, bottom, transform=axis.transAxes, ha="center", **style)
    axis.text(0.5, 0.98, top, transform=axis.transAxes, ha="center", va="top", **style)


def _positive_lsq_scale(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    reference_values = reference[mask].astype(np.float64)
    candidate_values = candidate[mask].astype(np.float64)
    denominator = float(np.dot(candidate_values, candidate_values))
    if denominator <= 0:
        raise ValueError("Candidate is zero inside the brain mask")
    return float(np.dot(reference_values, candidate_values) / denominator)


def _agreement(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    scale = _positive_lsq_scale(reference, candidate, mask)
    scaled = candidate * scale
    residual = (scaled - reference)[mask].astype(np.float64)
    reference_values = reference[mask].astype(np.float64)
    return {
        "scale": scale,
        "ncc": normalized_cross_correlation(reference, scaled, mask),
        "nrmse": float(
            np.sqrt(np.mean(residual * residual))
            / max(np.sqrt(np.mean(reference_values * reference_values)), 1e-12)
        ),
    }


def _profile_record(
    key: str,
    title: str,
    source: Path,
    data: np.ndarray,
    brain: np.ndarray,
    core: np.ndarray,
    shell: np.ndarray,
    normalized_dicom: np.ndarray,
    raw_dicom: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    positive_brain = brain & (data > 0)
    median = float(np.median(data[positive_brain]))
    if median <= 0:
        raise ValueError(f"Non-positive brain median for {key}")
    normalized = data / median
    core_median = float(np.median(normalized[core]))
    shell_median = float(np.median(normalized[shell]))
    return (
        {
            "case": key,
            "role": title,
            "source": str(source),
            "source_sha256": sha256_file(source),
            "brain_median_native_units": median,
            "core_median_brain_units": core_median,
            "shell_median_brain_units": shell_median,
            "core_to_shell_ratio": core_median / shell_median,
            **{
                f"normalized_dicom_{name}": value
                for name, value in _agreement(normalized_dicom, data, brain).items()
            },
            **{
                f"raw_dicom_{name}": value
                for name, value in _agreement(raw_dicom, data, brain).items()
            },
        },
        normalized,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    pilot_manifest_path = args.pilot_manifest.expanduser().resolve()
    provenance_path = args.metrics_provenance.expanduser().resolve()
    raw_dicom_path = args.raw_dicom_nifti.expanduser().resolve()
    grappa_path = args.grappa_nifti.expanduser().resolve()
    sense_path = args.sense_nifti.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (pilot_manifest_path, provenance_path, raw_dicom_path, grappa_path, sense_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Comparison output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    if pilot_manifest.get("ecalib", {}).get("intensity_correction") is not True:
        raise ValueError("Pilot did not record BART ecalib intensity correction")
    if pilot_manifest.get("wave_lambda0", {}).get("backend") != "gpu":
        raise ValueError("Pilot Wave reconstruction was not recorded as GPU")
    pilot_path = _magnitude_output(pilot_manifest)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    reference_path = Path(provenance["reference"]["path"])
    reference_image = nib.as_closest_canonical(nib.load(str(reference_path)))
    normalized_dicom = _finite_magnitude(reference_image)
    raw_dicom_image = nib.as_closest_canonical(nib.load(str(raw_dicom_path)))
    raw_dicom = _finite_magnitude(resample_from_to(raw_dicom_image, reference_image, order=1))
    brain = np.asarray(
        nib.load(provenance["fixed_masks"]["files"]["brain"]["path"]).dataobj
    ) > 0
    distance = distance_transform_edt(brain)
    core = brain & (distance >= 35.0)
    shell = brain & (distance >= 5.0) & (distance < 15.0)
    if int(core.sum()) < 10_000 or int(shell.sum()) < 10_000:
        raise ValueError("Core or shell mask is unexpectedly small")

    rigid = provenance["shared_registration"]["rigid"]["parameters"]
    rigid_parameters = [
        *rigid["rotation_degrees_ras_xyz"],
        *rigid["translation_mm_ras_xyz"],
    ]
    orientation = provenance["approved_orientation_mapping"]
    pilot_oriented = _load_on_reference_grid(
        pilot_path,
        reference_image,
        orientation["permutation"],
        orientation["flips_ras_grid_axes"],
    )
    pilot_registered = rigid_resample(pilot_oriented, rigid_parameters)
    current_record = next(
        record for record in provenance["case_records"] if record["kind"] == "lambda0"
    )
    current_path = Path(current_record["registered_nifti"])
    current_wave = _finite_magnitude(nib.load(str(current_path)))

    sources = [
        ("raw_dicom", "IDEA ACC, no normalization", raw_dicom_path, raw_dicom),
        ("normalized_dicom", "IDEA ACC + Prescan Normalize", reference_path, normalized_dicom),
        (
            "grappa",
            "No-wave GRAPPA comparison",
            grappa_path,
            _load_external(grappa_path, reference_image, rigid_parameters),
        ),
        (
            "sense",
            "No-wave SENSE comparison",
            sense_path,
            _load_external(sense_path, reference_image, rigid_parameters),
        ),
        ("current_wave", "Wave lambda zero, current maps", current_path, current_wave),
        ("intensity_wave", "Wave lambda zero, ecalib -I", pilot_path, pilot_registered),
    ]
    records = []
    display_volumes = []
    for key, title, path, data in sources:
        record, display = _profile_record(
            key,
            title,
            path,
            data,
            brain,
            core,
            shell,
            normalized_dicom,
            raw_dicom,
        )
        records.append(record)
        display_volumes.append((title, display))

    csv_path = output_dir / "intensity_profile_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    center = np.rint(np.argwhere(brain).mean(axis=0)).astype(int).tolist()
    figure_path = output_dir / "intensity_profile_comparison.png"
    figure, axes = plt.subplots(
        2,
        len(display_volumes),
        figsize=(3 * len(display_volumes), 6.5),
        constrained_layout=True,
    )
    for row, plane in enumerate(("coronal", "axial")):
        for column, (title, volume) in enumerate(display_volumes):
            axes[row, column].imshow(
                _plane(volume, plane, center),
                cmap="gray",
                origin="lower",
                vmin=0.0,
                vmax=1.8,
            )
            axes[row, column].set_title(f"{title}\n{plane}", fontsize=8)
            axes[row, column].set_axis_off()
            _directions(axes[row, column], plane)
    figure.suptitle(
        "ESPIRiT intensity-correction pilot — each volume divided by its brain median",
        fontsize=14,
    )
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    manifest = {
        "format_version": 1,
        "status": "comparison_complete_no_regularization_selection",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "GRAPPA and SENSE are comparison references only; neither is treated as truth, "
            "and this pilot does not select a regularization parameter."
        ),
        "pilot_manifest": {
            "path": str(pilot_manifest_path),
            "sha256": sha256_file(pilot_manifest_path),
        },
        "metrics_provenance": {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
        },
        "profile_masks": {
            "brain_voxels": int(brain.sum()),
            "core_rule": "distance at least 35 voxels inside approved BET mask",
            "core_voxels": int(core.sum()),
            "shell_rule": "distance 5 to less than 15 voxels inside approved BET mask",
            "shell_voxels": int(shell.sum()),
        },
        "display": {
            "scaling": "each volume divided by its own positive approved-brain median",
            "window_brain_median_units": [0.0, 1.8],
            "center_voxel_ras_grid": center,
        },
        "records": records,
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (csv_path, figure_path)
        ],
    }
    manifest_path = output_dir / "comparison_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Comparison figure: {figure_path}")
    print(f"Comparison metrics: {csv_path}")
    print(f"Comparison manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
