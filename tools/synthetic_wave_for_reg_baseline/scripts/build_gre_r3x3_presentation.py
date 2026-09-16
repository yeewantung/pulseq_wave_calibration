#!/usr/bin/env python3
"""Build the selected native-R3x3 GRE presentation artifact collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from PIL import Image

CASE_ID = "native_r3x3"
ECHO_IDS = ("echo-01", "echo-02")
ORIENTATION_AXES = {"sagittal": 0, "coronal": 1, "axial": 2}


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Compute a streaming SHA-256 digest for one file.

    Args:
        path: Existing file to hash.
        chunk_bytes: Maximum bytes read per iteration.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    """Load one required JSON object.

    Args:
        path: Existing JSON path.
        label: Human-readable input name for validation errors.

    Returns:
        Parsed JSON object.
    """

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return document


def file_identity(path: Path) -> dict[str, Any]:
    """Return the stable identity of one existing file.

    Args:
        path: Existing file to identify.

    Returns:
        Absolute path, size, and SHA-256 digest.
    """

    return {
        "path": str(path.absolute()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Write one JSON object by atomic replacement.

    Args:
        path: Destination JSON path.
        document: JSON-compatible mapping to serialize.

    Returns:
        None. The destination appears only after serialization completes.
    """

    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> dict[str, Any]:
    """Copy one source byte-for-byte after verifying its pinned hash.

    Args:
        source: Existing source artifact.
        destination: New presentation artifact path.
        expected_sha256: Manifest-pinned source digest.

    Returns:
        Source and destination identities with byte-equality status.

    Raises:
        ValueError: If either source or copied payload differs from the digest.
    """

    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError(f"Manifest-bound source changed or disappeared: {source}")
    if destination.is_file():
        if sha256_file(destination) != expected_sha256:
            raise FileExistsError(f"Refusing to overwrite a changed presentation file: {destination}")
        return {
            "source": file_identity(source),
            "presentation": file_identity(destination),
            "byte_identical": True,
        }
    temporary = Path(str(destination) + ".tmp")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Copied payload differs from its source: {source}")
    os.replace(temporary, destination)
    return {
        "source": file_identity(source),
        "presentation": file_identity(destination),
        "byte_identical": True,
    }


def validate_nifti(path: Path, *, phase: bool) -> nib.spatialimages.SpatialImage:
    """Validate one selected NIfTI's native R3x3 geometry and finite values.

    Args:
        path: Copied presentation NIfTI.
        phase: Whether wrapped phase rather than magnitude is expected.

    Returns:
        Loaded canonical-RAS NIfTI image.

    Raises:
        ValueError: If geometry, orientation, values, or component semantics differ.
    """

    image = nib.load(str(path))
    values = np.asarray(image.dataobj)
    if image.shape != (250, 250, 72) or nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise ValueError(f"Selected NIfTI geometry is not canonical native GRE R3x3: {path}")
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError(f"Selected NIfTI must be finite and real: {path}")
    if phase:
        tolerance = 5e-6
        if float(values.min()) < -math.pi - tolerance or float(values.max()) > math.pi + tolerance:
            raise ValueError(f"Selected phase NIfTI is outside [-pi, pi]: {path}")
    elif np.any(values < 0) or not np.any(values > 0):
        raise ValueError(f"Selected magnitude NIfTI is empty or negative: {path}")
    return image


def orientation_slice(volume: np.ndarray, orientation: str, index: int) -> np.ndarray:
    """Extract one neurological display slice from a canonical-RAS volume.

    Args:
        volume: Three-dimensional canonical-RAS magnitude array.
        orientation: Sagittal, coronal, or axial plane name.
        index: Source-array index along the corresponding physical axis.

    Returns:
        Two-dimensional slice with superior or anterior displayed upward.
    """

    if orientation == "sagittal":
        return np.flip(volume[index, :, :].T, axis=0)
    if orientation == "coronal":
        return np.flip(volume[:, index, :].T, axis=0)
    if orientation == "axial":
        return np.flip(volume[:, :, index].T, axis=0)
    raise ValueError(f"Unsupported orientation: {orientation}")


def export_center_tiffs(
    magnitude_path: Path, output_dir: Path, *, echo_id: str, reconstruction_id: str
) -> list[dict[str, Any]]:
    """Export three center slices from one selected magnitude NIfTI.

    Args:
        magnitude_path: Copied canonical-RAS magnitude NIfTI.
        output_dir: Confirmed presentation TIFF directory.
        echo_id: Echo identifier included in stable filenames.
        reconstruction_id: Stable FISTA or selected-Wavelet identifier.

    Returns:
        Hash-bound TIFF records for sagittal, coronal, and axial centers.
    """

    image = validate_nifti(magnitude_path, phase=False)
    volume = np.asarray(image.dataobj, dtype=np.float32)
    positive = volume[volume > 0]
    display_max = float(np.percentile(positive, 99.5))
    if not math.isfinite(display_max) or display_max <= 0:
        raise ValueError(f"Invalid display maximum for {magnitude_path}")
    records = []
    for orientation, axis in ORIENTATION_AXES.items():
        index = int(volume.shape[axis] // 2)
        values = orientation_slice(volume, orientation, index)
        pixels = np.rint(np.clip(values, 0, display_max) / display_max * 65535).astype(np.uint16)
        destination = output_dir / (
            f"{CASE_ID}_{echo_id}_{reconstruction_id}_{orientation}_center.tiff"
        )
        temporary = destination.with_name(f".{destination.name}.tmp")
        Image.fromarray(pixels).save(temporary, format="TIFF", compression="tiff_lzw")
        with Image.open(temporary) as saved:
            if saved.size != (pixels.shape[1], pixels.shape[0]):
                raise ValueError(f"TIFF shape validation failed: {destination}")
        if destination.is_file():
            if sha256_file(destination) != sha256_file(temporary):
                temporary.unlink(missing_ok=True)
                raise FileExistsError(f"Refusing to overwrite a changed TIFF: {destination}")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        records.append(
            {
                "echo_id": echo_id,
                "reconstruction_id": reconstruction_id,
                "orientation": orientation,
                "source_axis": axis,
                "source_index": index,
                "pixel_shape": list(pixels.shape),
                "display_percentile": 99.5,
                "display_max": display_max,
                "file": file_identity(destination),
            }
        )
    return records


def validate_metrics_csv(path: Path) -> int:
    """Require every metric row to describe only native R3x3.

    Args:
        path: Copied per-echo or shared-echo metrics CSV.

    Returns:
        Number of validated data rows.
    """

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or {row.get("case_id") for row in rows} != {CASE_ID}:
        raise ValueError(f"Metrics CSV contains a non-R3x3 case or no rows: {path}")
    return len(rows)


def build_presentation(
    run_root: Path, output_dir: Path, *, selected_lambda: float, refresh: bool
) -> dict[str, Any]:
    """Build the confirmed native-R3x3 presentation folder.

    Args:
        run_root: Completed native-R3x3 GRE sweep root.
        output_dir: Exact user-confirmed presentation directory.
        selected_lambda: Shared Wavelet lambda chosen by manual review.
        refresh: Extend an exporter-owned presentation folder when true.

    Returns:
        Completed presentation manifest.

    Side Effects:
        Creates a new presentation tree and byte-copies selected artifacts.
    """

    run_root = run_root.expanduser().absolute()
    output_dir = output_dir.expanduser().absolute()
    if output_dir != run_root / "presentation":
        raise ValueError(f"Output must be the confirmed directory: {run_root / 'presentation'}")
    presentation_manifest_path = output_dir / "manifest.json"
    prior: dict[str, Any] | None = None
    if output_dir.exists() and any(output_dir.iterdir()):
        if not refresh or not presentation_manifest_path.is_file():
            raise FileExistsError(f"Presentation directory is not an owned refresh target: {output_dir}")
        prior = load_json(presentation_manifest_path, "existing presentation manifest")
        if prior.get("status") != "complete" or prior.get("case_id") != CASE_ID:
            raise ValueError("Existing presentation manifest is not a completed native-R3x3 collection.")
    subdirectories = {
        name: output_dir / name for name in ("niftis", "tiff_images", "metric_curves", "metrics")
    }
    for path in subdirectories.values():
        path.mkdir(parents=True, exist_ok=True)

    selection_specs = (
        {
            "reconstruction_id": "fista_lambda-0",
            "sweep": "coarse",
            "method": "fista_lambda0",
            "lambda": 0.0,
        },
        {
            "reconstruction_id": "wavelet_lambda-1.5e-2",
            "sweep": "fine",
            "method": "wavelet",
            "lambda": selected_lambda,
        },
    )
    sweep_records: dict[str, dict[str, Any]] = {}
    selected: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for spec in selection_specs:
        sweep_name = str(spec["sweep"])
        sweep_path = run_root / "reconstructions" / sweep_name / "sweep_manifest.json"
        sweep = load_json(sweep_path, f"{sweep_name} sweep manifest")
        if sweep.get("status") != "complete" or sweep.get("sweep") != sweep_name:
            raise ValueError(f"{sweep_name} sweep is not complete.")
        sweep_records[sweep_name] = file_identity(sweep_path)
        for binding in sweep["candidate_manifests"]:
            path = Path(binding["path"])
            if sha256_file(path) != binding["sha256"]:
                raise ValueError(f"{sweep_name} candidate manifest changed: {path}")
            candidate = load_json(path, f"{sweep_name} candidate manifest")
            setting = candidate.get("setting", {})
            if (
                candidate.get("case_id") == CASE_ID
                and candidate.get("echo_id") in ECHO_IDS
                and setting.get("method") == spec["method"]
                and math.isclose(
                    float(setting.get("lambda", math.nan)),
                    float(spec["lambda"]),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                selected[(str(spec["reconstruction_id"]), str(candidate["echo_id"]))] = (
                    path,
                    candidate,
                )
    expected_selection_keys = {
        (str(spec["reconstruction_id"]), echo_id)
        for spec in selection_specs
        for echo_id in ECHO_IDS
    }
    if set(selected) != expected_selection_keys:
        raise ValueError("Sweeps do not contain exactly one FISTA and selected Wavelet candidate per echo.")

    nifti_records = []
    tiff_records = []
    for spec in selection_specs:
        reconstruction_id = str(spec["reconstruction_id"])
        for echo_id in ECHO_IDS:
            candidate_path, _ = selected[(reconstruction_id, echo_id)]
            export_path = candidate_path.parent / "nifti_export_manifest.json"
            export = load_json(export_path, f"{reconstruction_id}/{echo_id} NIfTI export manifest")
            if (
                export.get("status") != "complete"
                or export.get("case_id") != CASE_ID
                or export.get("echo_id") != echo_id
                or export.get("setting", {}).get("method") != spec["method"]
                or not math.isclose(
                    float(export.get("setting", {}).get("lambda", math.nan)),
                    float(spec["lambda"]),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"Selected NIfTI export contract changed for {reconstruction_id}/{echo_id}.")
            copied: dict[str, Any] = {}
            for component, phase in (("magnitude", False), ("phase", True)):
                source_record = export[f"{component}_nifti"]
                source = Path(source_record["path"])
                destination = subdirectories["niftis"] / (
                    f"{CASE_ID}_{echo_id}_{reconstruction_id}_{component}_ras.nii.gz"
                )
                copied[component] = copy_verified(source, destination, source_record["sha256"])
                validate_nifti(destination, phase=phase)
            nifti_records.append(
                {
                    "reconstruction_id": reconstruction_id,
                    "echo_id": echo_id,
                    "candidate_manifest": file_identity(candidate_path),
                    "nifti_export_manifest": file_identity(export_path),
                    "components": copied,
                }
            )
            magnitude_path = Path(copied["magnitude"]["presentation"]["path"])
            tiff_records.extend(
                export_center_tiffs(
                    magnitude_path,
                    subdirectories["tiff_images"],
                    echo_id=echo_id,
                    reconstruction_id=reconstruction_id,
                )
            )

    curve_records = []
    curve_sources = (
        run_root / "figures" / "fine" / "figure_manifest.json",
        run_root / "figures" / "fine_shared_lambda" / "figure_manifest.json",
    )
    for manifest_path in curve_sources:
        manifest = load_json(manifest_path, "metric-curve figure manifest")
        if manifest.get("status") != "complete":
            raise ValueError(f"Metric-curve figure manifest is incomplete: {manifest_path}")
        for source_record in manifest["figures"]:
            source = Path(source_record["path"])
            if CASE_ID not in source.name or (
                "curves" not in source.name and source.parent.name != "metric_curves"
            ):
                continue
            destination = subdirectories["metric_curves"] / source.name
            curve_records.append(copy_verified(source, destination, source_record["sha256"]))
    if len(curve_records) != 3:
        raise ValueError(f"Expected exactly three R3x3 metric curves, found {len(curve_records)}.")

    metric_records = []
    evaluation_manifests = (
        run_root / "evaluation" / "fine" / "evaluation_manifest.json",
        run_root / "evaluation" / "fine_shared_lambda" / "evaluation_manifest.json",
    )
    metric_names = ("native_r3x3_fine_per_echo_metrics.csv", "native_r3x3_fine_shared_echo_metrics.csv")
    for manifest_path, name in zip(evaluation_manifests, metric_names, strict=True):
        manifest = load_json(manifest_path, "metrics evaluation manifest")
        source_record = manifest["metrics_csv"]
        destination = subdirectories["metrics"] / name
        copied = copy_verified(Path(source_record["path"]), destination, source_record["sha256"])
        copied["row_count"] = validate_metrics_csv(destination)
        copied["evaluation_manifest"] = file_identity(manifest_path)
        metric_records.append(copied)

    manifest = {
        "format_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": CASE_ID,
        "echo_ids": list(ECHO_IDS),
        "manual_selection": {
            "method": "wavelet",
            "shared_lambda_across_echoes": selected_lambda,
            "automatic_selection_performed": False,
        },
        "included_reconstructions": list(selection_specs),
        "run_root": str(run_root),
        "sweep_manifests": sweep_records,
        "niftis": nifti_records,
        "center_slice_tiffs": tiff_records,
        "metric_curves": curve_records,
        "metrics_csv": metric_records,
        "artifact_counts": {
            "niftis": 8,
            "center_slice_tiffs": len(tiff_records),
            "metric_curves": len(curve_records),
            "metrics_csv": len(metric_records),
        },
        "spatial_resampling_performed": False,
        "nifti_payloads_copied_byte_for_byte": True,
    }
    current_tiff_paths = {
        Path(record["file"]["path"]).absolute() for record in tiff_records
    }
    if prior is not None:
        for record in prior.get("center_slice_tiffs", []):
            previous_file = record.get("file", {})
            stale = Path(str(previous_file.get("path", ""))).absolute()
            if stale in current_tiff_paths:
                continue
            if stale.parent != subdirectories["tiff_images"] or not stale.is_file():
                raise ValueError(f"Prior manifest contains an unsafe or missing TIFF: {stale}")
            if sha256_file(stale) != previous_file.get("sha256"):
                raise FileExistsError(f"Refusing to remove a changed prior TIFF: {stale}")
            stale.unlink()
    write_json_atomic(presentation_manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the presentation export command-line parser.

    Returns:
        Parser requiring a completed run root and confirmed presentation path.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selected-lambda", required=True, type=float)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one native-R3x3 presentation collection.

    Args:
        argv: Optional command-line arguments; ``None`` uses process arguments.

    Returns:
        Zero after successful manifested export.
    """

    args = build_parser().parse_args(argv)
    manifest = build_presentation(
        args.run_root,
        args.output_dir,
        selected_lambda=float(args.selected_lambda),
        refresh=bool(args.refresh),
    )
    print(json.dumps({"status": manifest["status"], **manifest["artifact_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
