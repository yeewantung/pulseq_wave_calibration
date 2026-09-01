#!/usr/bin/env python3
"""Build a manifested FISTA-versus-selected-Wavelet presentation package."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from evaluate_pure_mask_sweeps import (
    _canonical_mask,
    _target_affine,
    logical_to_physical_xyz,
    scale_candidate_for_display,
)
from export_presentation_orientation_tiffs import (
    ORIENTATIONS,
    _save_tiff_atomic,
    _to_uint16,
    orientation_slices,
    slice_indices,
)
from pure_mask_rerun import (
    CASE_IDS,
    load_json,
    logical_array_sha256,
    sha256_file,
    validate_config,
    write_json_atomic,
)


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp.

    Returns:
        ISO-8601 UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    """Build the presentation-package command interface.

    Returns:
        Argument parser for validation or package creation.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--confirm-output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _setting_key(setting: Mapping[str, Any]) -> tuple[str, int | None, float]:
    """Normalize a reconstruction setting for exact candidate matching.

    Args:
        setting: Method, block-size, and lambda mapping.

    Returns:
        Normalized method, optional block size, and lambda tuple.
    """
    return (
        str(setting["method"]),
        None if setting.get("block_size") is None else int(setting["block_size"]),
        float(setting["lambda"]),
    )


def _lambda_token(value: float) -> str:
    """Convert a positive lambda into a stable filename token.

    Args:
        value: Positive regularization value.

    Returns:
        Decimal lambda token with ``p`` replacing the decimal point.
    """
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError("Selected Wavelet lambda must be finite and positive.")
    return f"{numeric:.12g}".replace(".", "p")


def _candidate_record(
    sweep: Mapping[str, Any], case_id: str, setting: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Find one unique hash-bound sweep candidate.

    Args:
        sweep: Completed fine sweep manifest.
        case_id: Pure-mask case identifier.
        setting: Exact reconstruction setting.

    Returns:
        Unique candidate-manifest record.

    Raises:
        ValueError: If zero or multiple candidates match.
    """
    key = _setting_key(setting)
    matches = [
        record
        for record in sweep["candidate_manifests"]
        if record["case_id"] == case_id and _setting_key(record["setting"]) == key
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {case_id} candidate for {key}; found {len(matches)}.")
    return matches[0]


def _metric_row(
    evaluation: Mapping[str, Any], case_id: str, setting: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Find one unique evaluation row for a candidate setting.

    Args:
        evaluation: Completed fine evaluation manifest.
        case_id: Pure-mask case identifier.
        setting: Exact reconstruction setting.

    Returns:
        Unique metric row.

    Raises:
        ValueError: If zero or multiple rows match.
    """
    key = _setting_key(setting)
    matches = [
        row
        for row in evaluation["rows"]
        if row["case_id"] == case_id and _setting_key(row) == key
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {case_id} metric row for {key}; found {len(matches)}.")
    return matches[0]


def _validate_candidate(
    record: Mapping[str, Any], expected_shape: tuple[int, int, int]
) -> tuple[Path, np.ndarray]:
    """Validate and load one candidate magnitude array.

    Args:
        record: Sweep candidate-manifest record.
        expected_shape: Required logical RO/LIN/PAR matrix.

    Returns:
        Candidate manifest path and memory-mapped magnitude array.

    Raises:
        ValueError: If manifest, hash, shape, type, or finite-value checks fail.
    """
    manifest_path = Path(record["manifest"]).resolve()
    if sha256_file(manifest_path) != record["manifest_sha256"]:
        raise ValueError(f"Candidate manifest changed: {manifest_path}")
    candidate = load_json(manifest_path, "presentation candidate")
    magnitude_record = candidate["outputs"]["magnitude"]
    magnitude_path = Path(magnitude_record["path"]).resolve()
    if sha256_file(magnitude_path) != magnitude_record["sha256"]:
        raise ValueError(f"Candidate magnitude changed: {magnitude_path}")
    magnitude = np.load(magnitude_path, mmap_mode="r", allow_pickle=False)
    if (
        magnitude.shape != expected_shape
        or magnitude.dtype != np.float32
        or not np.isfinite(magnitude).all()
    ):
        raise ValueError(f"Candidate magnitude validation failed: {magnitude_path}")
    return manifest_path, magnitude


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    """Save and validate one raw-magnitude canonical-RAS NIfTI atomically.

    Args:
        path: Destination ``.nii.gz`` file.
        data: Physical XYZ float32 magnitude array.
        affine: Canonical-RAS voxel-to-world affine.

    Returns:
        Hash, geometry, type, and intensity-scaling record.
    """
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    temporary = path.with_name(f".{path.name}.tmp.nii.gz")
    nib.save(image, temporary)
    saved = nib.load(str(temporary))
    values = np.asarray(saved.dataobj)
    if (
        saved.shape != data.shape
        or saved.get_data_dtype() != np.dtype(np.float32)
        or nib.aff2axcodes(saved.affine) != ("R", "A", "S")
        or not np.isfinite(values).all()
    ):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Saved presentation NIfTI validation failed: {path}")
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape_xyz": list(data.shape),
        "voxel_size_mm_xyz": [float(value) for value in saved.header.get_zooms()[:3]],
        "axis_codes": ["R", "A", "S"],
        "dtype": "float32",
        "intensity_scaling": "none; raw reconstruction magnitude",
    }


def _write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write the ten presentation metric rows atomically.

    Args:
        path: Destination CSV.
        rows: Ordered presentation metric mappings.

    Returns:
        Output path, hash, and row-count record.
    """
    if len(rows) != 10:
        raise ValueError("Presentation metrics must contain exactly ten rows.")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256_file(path), "row_count": len(rows)}


def _prior_reusable(
    manifest_path: Path, expected_selection_hash: str
) -> dict[str, Any] | None:
    """Return a complete unchanged presentation manifest when safely reusable.

    Args:
        manifest_path: Existing presentation manifest.
        expected_selection_hash: Current approved selection-manifest hash.

    Returns:
        Parsed reusable manifest, or ``None`` when reuse is unsafe.
    """
    if not manifest_path.is_file():
        return None
    prior = load_json(manifest_path, "presentation manifest")
    if (
        prior.get("status") != "complete"
        or prior.get("inputs", {}).get("selection_manifest", {}).get("sha256")
        != expected_selection_hash
    ):
        return None
    records = [prior["metrics_csv"], prior["center_slices_manifest"]]
    for entry in prior.get("entries", []):
        records.append(entry["nifti"])
        records.extend(entry["center_slices"].values())
    if all(Path(record["path"]).is_file() and sha256_file(record["path"]) == record["sha256"] for record in records):
        return prior
    return None


def build(
    config_path: Path,
    *,
    confirmed_output_dir: Path | None,
    validate_only: bool,
    resume: bool,
) -> dict[str, Any]:
    """Validate or build the ten-reconstruction presentation package.

    Args:
        config_path: Ignored pure-mask rerun configuration.
        confirmed_output_dir: Exact user-confirmed presentation directory.
        validate_only: Perform no writes when true.
        resume: Reuse a complete hash-identical package when true.

    Returns:
        Validation summary or completed presentation manifest.
    """
    validated = validate_config(config_path)
    root = Path(validated["layout"]["root"])
    expected_output_dir = root / "presentation"
    selection_path = root / "evaluation" / "review" / "selection_manifest.json"
    selection = load_json(selection_path, "pure-mask selection manifest")
    selection_hash = sha256_file(selection_path)
    if (
        selection.get("status") != "complete"
        or selection.get("automatic_selection_performed") is not False
        or selection.get("composite_score_used") is not False
        or set(selection.get("selections", {})) != set(CASE_IDS)
    ):
        raise ValueError("Presentation requires a complete five-case selection manifest.")
    shortlist_record = selection["shortlist_manifest"]
    shortlist_path = Path(shortlist_record["path"]).resolve()
    if sha256_file(shortlist_path) != shortlist_record["sha256"]:
        raise ValueError("Selection shortlist manifest changed.")
    shortlist = load_json(shortlist_path, "pure-mask shortlist manifest")
    if shortlist.get("automatic_shortlist_selection_performed") is not False:
        raise ValueError("Presentation shortlist must be explicitly manual.")
    sweep_record = shortlist["inputs"]["sweep_manifest"]
    evaluation_record = shortlist["inputs"]["evaluation_manifest"]
    sweep_path = Path(sweep_record["path"]).resolve()
    evaluation_path = Path(evaluation_record["path"]).resolve()
    if sha256_file(sweep_path) != sweep_record["sha256"]:
        raise ValueError("Presentation fine sweep manifest changed.")
    if sha256_file(evaluation_path) != evaluation_record["sha256"]:
        raise ValueError("Presentation fine evaluation manifest changed.")
    sweep = load_json(sweep_path, "presentation fine sweep")
    evaluation = load_json(evaluation_path, "presentation fine evaluation")
    if (
        sweep.get("status") != "complete"
        or sweep.get("stage") != "fine"
        or evaluation.get("status") != "complete"
        or evaluation.get("stage") != "fine"
        or evaluation.get("scientific_scope", {}).get(
            "automatic_composite_selection_performed"
        )
        is not False
    ):
        raise ValueError("Presentation sweep and evaluation must be complete.")
    if evaluation.get("sweep_manifest") != sweep_record:
        raise ValueError("Presentation evaluation is not bound to the selected sweep.")
    preparation_record = sweep["preparation_manifest"]
    preparation_path = Path(preparation_record["path"]).resolve()
    if sha256_file(preparation_path) != preparation_record["sha256"]:
        raise ValueError("Presentation preparation manifest changed.")
    preparation = load_json(preparation_path, "presentation preparation manifest")

    config = validated["config"]["snapshot"]
    axis_order = config["evaluation"]["logical_to_canonical_axis_order"]
    axis_flips = config["evaluation"]["logical_to_canonical_axis_flips"]
    native_geometry = validated["geometry"]
    native_ro, native_lin, native_par = (
        int(value) for value in native_geometry["logical_matrix_ro_lin_par"]
    )
    approved_image, _approved_mask = _canonical_mask(
        Path(validated["source"]["approved_bet_mask"]["path"]),
        expected_shape_xyz=(native_par, native_lin, native_ro),
        expected_fov_mm_xyz=tuple(
            float(value) for value in native_geometry["physical_fov_mm_xyz"]
        ),
    )

    entries: list[dict[str, Any]] = []
    for case_id in CASE_IDS:
        selected = selection["selections"][case_id]
        selected_setting = selected["setting"]
        if selected_setting["method"] != "wavelet":
            raise ValueError(f"{case_id} presentation selection must be Wavelet.")
        control_setting = {"method": "fista_lambda0", "block_size": None, "lambda": 0.0}
        for role, setting in (("fista_control", control_setting), ("approved_wavelet", selected_setting)):
            record = _candidate_record(sweep, case_id, setting)
            metric = _metric_row(evaluation, case_id, setting)
            if role == "approved_wavelet":
                selected_manifest = selected["candidate_manifest"]
                if (
                    Path(selected_manifest["path"]).resolve() != Path(record["manifest"]).resolve()
                    or selected_manifest["sha256"] != record["manifest_sha256"]
                ):
                    raise ValueError(f"{case_id} selected candidate differs from the fine sweep.")
                key = f"{case_id}__wavelet_lambda-{_lambda_token(setting['lambda'])}"
            else:
                key = f"{case_id}__fista_lambda0"
            entries.append(
                {
                    "display_order": len(entries) + 1,
                    "key": key,
                    "case_id": case_id,
                    "role": role,
                    "setting": dict(setting),
                    "candidate_record": record,
                    "metric_row": metric,
                }
            )
    if len(entries) != 10:
        raise AssertionError("Presentation entry count differs from ten.")

    # Validate all candidate arrays and direct-FFT references before any output write.
    for entry in entries:
        case_id = entry["case_id"]
        case_record = preparation["cases"][case_id]
        case_path = Path(case_record["case_manifest"]).resolve()
        if sha256_file(case_path) != case_record["case_manifest_sha256"]:
            raise ValueError(f"{case_id} preparation case manifest changed.")
        case = load_json(case_path, f"{case_id} presentation preflight case")
        logical_shape = tuple(
            int(value) for value in case["case"]["target_logical_matrix_ro_lin_par"]
        )
        _validate_candidate(entry["candidate_record"], logical_shape)
        reference_record = case["direct_fft_reference"]
        if sha256_file(reference_record["path"]) != reference_record["sha256"]:
            raise ValueError(f"{case_id} direct-FFT reference changed.")
        reference = np.load(reference_record["path"], mmap_mode="r", allow_pickle=False)
        if (
            reference.shape != logical_shape
            or reference.dtype != np.float32
            or logical_array_sha256(reference) != reference_record["logical_sha256"]
            or not np.isfinite(reference).all()
        ):
            raise ValueError(f"{case_id} direct-FFT reference validation failed.")
    if validate_only:
        if confirmed_output_dir is not None:
            raise ValueError("--confirm-output-dir is not used with --validate-only.")
        return {"status": "validated", "entry_count": len(entries)}
    if (
        confirmed_output_dir is None
        or confirmed_output_dir.expanduser().resolve() != expected_output_dir
    ):
        raise ValueError("Presentation requires the exact user-confirmed output directory.")

    manifest_path = expected_output_dir / "presentation_manifest.json"
    if resume:
        prior = _prior_reusable(manifest_path, selection_hash)
        if prior is not None:
            print(f"Reusing presentation package: {manifest_path}")
            return prior
    if expected_output_dir.exists() and any(expected_output_dir.iterdir()):
        raise FileExistsError(f"Presentation output is not safely reusable: {expected_output_dir}")
    nifti_dir = expected_output_dir / "niftis"
    slice_dir = expected_output_dir / "center_slices"
    nifti_dir.mkdir(parents=True)
    slice_dir.mkdir()

    metrics_rows: list[dict[str, Any]] = []
    output_entries: list[dict[str, Any]] = []
    for entry in entries:
        case_id = entry["case_id"]
        case_record = preparation["cases"][case_id]
        case_path = Path(case_record["case_manifest"]).resolve()
        if sha256_file(case_path) != case_record["case_manifest_sha256"]:
            raise ValueError(f"{case_id} preparation case manifest changed.")
        case = load_json(case_path, f"{case_id} presentation case")
        geometry = case["case"]
        logical_shape = tuple(
            int(value) for value in geometry["target_logical_matrix_ro_lin_par"]
        )
        physical_shape = tuple(
            int(value) for value in geometry["target_physical_matrix_xyz"]
        )
        zooms = tuple(float(value) for value in geometry["achieved_resolution_mm_xyz"])
        candidate_path, magnitude = _validate_candidate(
            entry["candidate_record"], logical_shape
        )
        physical = logical_to_physical_xyz(
            magnitude, axis_order=axis_order, axis_flips=axis_flips
        )
        if physical.shape != physical_shape:
            raise ValueError(f"{entry['key']} physical shape differs from case geometry.")
        affine = _target_affine(approved_image, physical_shape, zooms)
        nifti_record = _save_nifti(nifti_dir / f"{entry['key']}.nii.gz", physical, affine)

        reference_record = case["direct_fft_reference"]
        if sha256_file(reference_record["path"]) != reference_record["sha256"]:
            raise ValueError(f"{case_id} direct-FFT reference changed.")
        reference_logical = np.load(reference_record["path"], allow_pickle=False)
        if logical_array_sha256(reference_logical) != reference_record["logical_sha256"]:
            raise ValueError(f"{case_id} direct-FFT logical hash changed.")
        reference = logical_to_physical_xyz(
            reference_logical, axis_order=axis_order, axis_flips=axis_flips
        )
        positive = reference[reference > 0]
        display_max = float(np.percentile(positive, 99.5))
        if not np.isfinite(display_max) or display_max <= 0:
            raise ValueError(f"{case_id} direct-FFT display window is invalid.")
        scale = float(entry["metric_row"]["intensity_scale_lsq"])
        display_volume = scale_candidate_for_display(physical, scale)
        indices = slice_indices(display_volume.shape, 0, use_center=True)
        slices = orientation_slices(display_volume, indices)
        slice_records: dict[str, Any] = {}
        for orientation in ORIENTATIONS:
            slice_path = slice_dir / f"{entry['key']}__{orientation}_center.tiff"
            pixels = _to_uint16(slices[orientation], display_max)
            _save_tiff_atomic(slice_path, pixels)
            slice_records[orientation] = {
                "path": str(slice_path),
                "sha256": sha256_file(slice_path),
                "source_axis_index": indices[orientation],
                "pixel_shape": list(pixels.shape),
                "dtype": "uint16",
            }

        metric_row = {
            "display_order": entry["display_order"],
            "presentation_key": entry["key"],
            "selection_role": entry["role"],
            **entry["metric_row"],
        }
        metrics_rows.append(metric_row)
        output_entries.append(
            {
                "display_order": entry["display_order"],
                "key": entry["key"],
                "case_id": case_id,
                "role": entry["role"],
                "setting": entry["setting"],
                "candidate_manifest": {
                    "path": str(candidate_path),
                    "sha256": sha256_file(candidate_path),
                },
                "nifti": nifti_record,
                "center_slices": slice_records,
                "tiff_display": {
                    "intensity_scale_lsq": scale,
                    "shared_case_reference_window": [0.0, display_max],
                    "window_rule": "resolution-matched direct-FFT positive-value p99.5",
                },
            }
        )
        print(f"Exported {entry['key']}.", flush=True)

    metrics_record = _write_metrics(expected_output_dir / "metrics.csv", metrics_rows)
    slices_manifest_path = slice_dir / "center_slices_manifest.json"
    slices_manifest = {
        "format_version": 1,
        "status": "complete",
        "selection_manifest": {"path": str(selection_path), "sha256": selection_hash},
        "entry_count": len(output_entries),
        "tiff_count": sum(len(entry["center_slices"]) for entry in output_entries),
        "entries": [
            {"key": entry["key"], "outputs": entry["center_slices"]}
            for entry in output_entries
        ],
    }
    write_json_atomic(slices_manifest_path, slices_manifest)
    completed = {
        "format_version": 1,
        "status": "complete",
        "purpose": "FISTA controls and user-approved Wavelet reconstructions",
        "inputs": {
            "selection_manifest": {"path": str(selection_path), "sha256": selection_hash},
            "shortlist_manifest": {"path": str(shortlist_path), "sha256": sha256_file(shortlist_path)},
            "evaluation_manifest": {"path": str(evaluation_path), "sha256": sha256_file(evaluation_path)},
            "sweep_manifest": {"path": str(sweep_path), "sha256": sha256_file(sweep_path)},
            "preparation_manifest": {"path": str(preparation_path), "sha256": sha256_file(preparation_path)},
        },
        "scientific_scope": {
            "magnitude_niftis": 10,
            "center_slice_tiffs": 30,
            "phase_exported": False,
            "nifti_spatial_resampling_performed": False,
            "nifti_intensity_scaling_performed": False,
            "tiff_intensity_scaling": "BET-restricted candidate LSQ scale from fine evaluation",
            "tiff_window": "shared within each case using resolution-matched direct-FFT p99.5",
        },
        "entries": output_entries,
        "metrics_csv": metrics_record,
        "center_slices_manifest": {
            "path": str(slices_manifest_path),
            "sha256": sha256_file(slices_manifest_path),
        },
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(manifest_path, completed)
    print(f"Pure-mask presentation manifest: {manifest_path}")
    return completed


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and validate or build the presentation package.

    Args:
        argv: Optional argument vector; ``None`` reads process arguments.

    Returns:
        Zero after successful validation or package creation.
    """
    args = _parser().parse_args(argv)
    result = build(
        args.config,
        confirmed_output_dir=args.confirm_output_dir,
        validate_only=args.validate_only,
        resume=args.resume,
    )
    if args.validate_only:
        print(
            f"Validated {result['entry_count']} presentation entries; no output was written."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
