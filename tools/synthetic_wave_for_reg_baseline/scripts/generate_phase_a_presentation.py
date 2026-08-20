#!/usr/bin/env python3
"""Generate the compact Phase-A R3 presentation package with fixed display settings."""

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

from evaluate_regularization_volume import (
    _finite_magnitude,
    compute_metrics,
    rigid_resample,
    sha256_file,
)


def select_representatives(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected = {"lambda0": next(record for record in records if record["kind"] == "lambda0")}
    for kind in ("wavelet", "llr"):
        candidates = [record for record in records if record["kind"] == kind]
        if not candidates:
            raise ValueError(f"No {kind} cases are available for presentation")
        selected[kind] = min(candidates, key=lambda record: record["composite_mean_rank"])
    return selected


def _load_external_on_reference_grid(
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-provenance", required=True, type=Path)
    parser.add_argument("--grappa-nifti", required=True, type=Path)
    parser.add_argument("--sense-nifti", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    provenance_path = args.metrics_provenance.expanduser().resolve()
    grappa_path = args.grappa_nifti.expanduser().resolve()
    sense_path = args.sense_nifti.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Presentation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    selected = select_representatives(provenance["case_records"])

    reference_path = Path(provenance["reference"]["path"])
    reference_image = nib.as_closest_canonical(nib.load(str(reference_path)))
    reference = _finite_magnitude(reference_image)
    mask_files = provenance["fixed_masks"]["files"]
    masks = {
        name: np.asarray(nib.load(record["path"]).dataobj) > 0
        for name, record in mask_files.items()
    }
    mask_metadata = provenance["fixed_masks"]["metadata"]
    rigid = provenance["shared_registration"]["rigid"]
    parameters = [
        *rigid["parameters"]["rotation_degrees_ras_xyz"],
        *rigid["parameters"]["translation_mm_ras_xyz"],
    ]

    volumes: list[tuple[str, str, np.ndarray]] = [("dicom", "Normalized DICOM", reference)]
    metric_rows = []
    for key, title, path in (
        ("grappa", "No-Wave GRAPPA", grappa_path),
        ("sense", "No-Wave SENSE", sense_path),
    ):
        registered = _load_external_on_reference_grid(path, reference_image, parameters)
        metrics, scaled = compute_metrics(reference, registered, masks, mask_metadata)
        volumes.append((key, title, scaled))
        metric_rows.append({"case": key, "kind": "no_wave", "source": str(path), **metrics})
    for kind, title in (
        ("lambda0", "Wave R3x2 λ=0"),
        ("wavelet", "Selected Wavelet"),
        ("llr", "Selected corrected LLR"),
    ):
        record = selected[kind]
        registered = _finite_magnitude(nib.load(record["registered_nifti"]))
        metrics, scaled = compute_metrics(reference, registered, masks, mask_metadata)
        volumes.append((record["case"], f"{title}\n{record['case']}", scaled))
        metric_rows.append(
            {"case": record["case"], "kind": kind, "source": record["registered_nifti"], **metrics}
        )

    center = np.rint(np.argwhere(masks["brain"]).mean(axis=0)).astype(int).tolist()
    display_vmax = float(np.percentile(reference[masks["brain"]], 99.5))
    comparison_path = output_dir / "phase_a_method_comparison.png"
    figure, axes = plt.subplots(2, len(volumes), figsize=(3 * len(volumes), 6.5), constrained_layout=True)
    for row, plane in enumerate(("coronal", "axial")):
        for column, (_, title, volume) in enumerate(volumes):
            axes[row, column].imshow(
                _plane(volume, plane, center),
                cmap="gray",
                origin="lower",
                vmin=0.0,
                vmax=display_vmax,
            )
            axes[row, column].set_title(f"{title}\n{plane}", fontsize=8)
            axes[row, column].set_axis_off()
            _directions(axes[row, column], plane)
    figure.suptitle(
        "Phase A R3 comparison — identical DICOM-derived intensity window",
        fontsize=14,
    )
    figure.savefig(comparison_path, dpi=180)
    plt.close(figure)

    difference_path = output_dir / "phase_a_selected_difference_maps.png"
    difference_volumes = [entry for entry in volumes if entry[0] != "dicom"]
    figure, axes = plt.subplots(2, len(difference_volumes), figsize=(3 * len(difference_volumes), 6.5), constrained_layout=True)
    for row, plane in enumerate(("coronal", "axial")):
        for column, (_, title, volume) in enumerate(difference_volumes):
            difference = np.abs(volume - reference) / max(display_vmax, 1e-12)
            image = axes[row, column].imshow(
                _plane(difference, plane, center),
                cmap="magma",
                origin="lower",
                vmin=0.0,
                vmax=0.30,
            )
            axes[row, column].set_title(f"{title}\n|Δ| / DICOM brain p99.5", fontsize=8)
            axes[row, column].set_axis_off()
            _directions(axes[row, column], plane)
    figure.colorbar(image, ax=axes, shrink=0.65, label="Normalized absolute difference")
    figure.suptitle("Selected Phase A difference maps — fixed 0–0.30 scale", fontsize=14)
    figure.savefig(difference_path, dpi=180)
    plt.close(figure)

    metrics_path = output_dir / "selected_method_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_label": "provisional R3-dataset-specific Phase A optimization",
        "metrics_provenance": {"path": str(provenance_path), "sha256": sha256_file(provenance_path)},
        "selected_cases": {kind: record["case"] for kind, record in selected.items()},
        "external_methods": {
            "grappa": {"path": str(grappa_path), "sha256": sha256_file(grappa_path)},
            "sense": {"path": str(sense_path), "sha256": sha256_file(sense_path)},
            "orientation_rule": "canonical RAS headers; shared lambda-zero rigid transform; no additional signed-axis flips",
        },
        "display": {
            "center_voxel_ras_grid": center,
            "planes": ["coronal", "axial"],
            "intensity_window": [0.0, display_vmax],
            "difference_window_normalized": [0.0, 0.30],
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (comparison_path, difference_path, metrics_path)
        ],
    }
    manifest_path = output_dir / "presentation_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Phase A presentation manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
