#!/usr/bin/env python3
"""Evaluate a completed MPRAGE VCC24 Wavelet sweep without selecting a winner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from checkpoint_io import write_json_atomic
from dataset_manifest import (
    DatasetManifest,
    DatasetManifestError,
    load_dataset_manifest,
    load_passed_inspection,
    sha256_file,
)
from evaluate_direct_fft_regularization import run as evaluate_regularization
from export_grappa_rss import run as export_multicoil_rss
from run_bart_regularization import canonical_lambda
from validate_metrics_geometry import run as validate_metrics_geometry


EXPECTED_LAMBDAS = (
    0.0,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.035,
    0.04,
    0.045,
    0.05,
)


def _parser() -> argparse.ArgumentParser:
    """Build the post-sweep evaluation command-line interface.

    Returns:
        Parser for dataset, approved-mask, and resume settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument(
        "--approved-brain-mask-manifest",
        required=True,
        type=Path,
        help="Previously approved same-subject metrics-only brain-mask manifest.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete stages whose exact input and output hashes still match.",
    )
    return parser


def _load_json(path: Path, label: str) -> dict[str, Any]:
    """Load one JSON object with a labelled validation error.

    Args:
        path: JSON file to read.
        label: Human-readable artifact name used in errors.

    Returns:
        Parsed JSON object.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def validate_one_shot_contract(payload: Mapping[str, Any]) -> None:
    """Require the fixed native-R3x1 standard-PCA VCC24 MPRAGE contract.

    Args:
        payload: Loaded dataset-manifest object.

    Raises:
        ValueError: If geometry, sampling, or compression differs from the sweep.
    """
    reconstruction = payload.get("reconstruction", {})
    sampling = payload.get("sampling", {})
    geometry = payload.get("geometry", {})
    if (
        reconstruction.get("physical_coils") != 64
        or reconstruction.get("virtual_coils") != 24
    ):
        raise ValueError("MPRAGE evaluation requires standard PCA 64-to-24 compression.")
    if reconstruction.get("coil_compression_source") != "image":
        raise ValueError("MPRAGE evaluation requires the declared image-derived PCA basis.")
    if geometry.get("matrix") != [256, 256, 256]:
        raise ValueError("MPRAGE evaluation requires the native 256-cubed grid.")
    if sampling.get("synthetic_wave_mask_kind") != "pure_cartesian_image_lattice":
        raise ValueError("MPRAGE evaluation requires a pure Cartesian image lattice.")
    if sampling.get("synthetic_wave_acceleration_pe1_pe2") != [3, 1]:
        raise ValueError("MPRAGE evaluation requires native R3x1 acceleration [3, 1].")
    if sampling.get("synthetic_wave_residue_pe1_pe2") != [1, 0]:
        raise ValueError("MPRAGE evaluation requires native R3x1 residue [1, 0].")


def _validate_approved_mask(
    manifest_path: Path,
) -> tuple[dict[str, Any], Path, str, int]:
    """Validate a visually approved metrics-only brain mask.

    Args:
        manifest_path: Approved brain-mask manifest.

    Returns:
        Manifest, mask path, manifest hash, and nonzero voxel count.
    """
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path, "approved brain-mask manifest")
    approval = manifest.get("approval", {})
    if manifest.get("status") != "approved_for_metrics" or not all(
        (
            approval.get("mask_boundary_visually_approved") is True,
            approval.get("left_right_orientation_visually_approved") is True,
        )
    ):
        raise ValueError("Brain-mask visual approval is incomplete.")
    record = manifest.get("brain_mask", {})
    mask_path = Path(record.get("path", "")).expanduser().resolve()
    expected_hash = record.get("sha256")
    if not mask_path.is_file() or sha256_file(mask_path) != expected_hash:
        raise ValueError("Approved brain-mask payload is missing or changed.")
    image = nib.load(str(mask_path))
    data = np.asarray(image.dataobj)
    if tuple(nib.aff2axcodes(image.affine)) != ("R", "A", "S"):
        raise ValueError("Approved brain mask must be canonical RAS.")
    if not np.isfinite(data).all() or not np.all(np.logical_or(data == 0, data == 1)):
        raise ValueError("Approved brain mask must be finite and binary.")
    voxel_count = int(np.count_nonzero(data))
    if voxel_count != int(record.get("voxel_count", -1)):
        raise ValueError("Approved brain-mask voxel count differs from its manifest.")
    return manifest, mask_path, sha256_file(manifest_path), voxel_count


def _source_contract(dataset: DatasetManifest) -> tuple[Path, Path, str, str]:
    """Validate and hash the fully sampled VCC24 no-Wave source.

    Args:
        dataset: Validated dataset contract.

    Returns:
        Source path, report path, source hash, and report hash.
    """
    prefix = dataset.output_path("source_reconstruction_prefix")
    source_path = prefix.with_name(prefix.name + "_full_ncc24.npy")
    report_path = prefix.with_name(prefix.name + "_report.json")
    report = _load_json(report_path, "direct-source report")
    if report.get("dataset_manifest", {}).get("sha256") != dataset.sha256:
        raise ValueError("Direct-source report uses a different dataset manifest.")
    assembly = report.get("assembly", {})
    if (
        Path(assembly.get("output", "")).resolve() != source_path.resolve()
        or assembly.get("shape") != [256, 256, 256, 24]
        or assembly.get("dtype") != "complex64"
        or assembly.get("finite") is not True
        or assembly.get("interpolation") != "none"
        or assembly.get("grappa_applied") is not False
    ):
        raise ValueError("Direct-source report does not certify exact finite VCC24 k-space.")
    source = np.load(source_path, mmap_mode="r")
    if source.shape != (256, 256, 256, 24) or source.dtype != np.complex64:
        raise ValueError("Direct-source checkpoint shape or dtype changed.")
    return source_path, report_path, sha256_file(source_path), sha256_file(report_path)


def _same_geometry(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
) -> bool:
    """Compare two NIfTI geometries.

    Args:
        first: First NIfTI image.
        second: Second NIfTI image.

    Returns:
        ``True`` when shapes and affines match within the fixed tolerance.
    """
    return first.shape == second.shape and bool(
        np.allclose(first.affine, second.affine, rtol=0.0, atol=1e-6)
    )


def _reference_is_reusable(
    metrics_path: Path,
    *,
    dataset: DatasetManifest,
    source_path: Path,
    source_sha256: str,
    report_path: Path,
    report_sha256: str,
    mask_path: Path,
    mask_manifest_path: Path,
    mask_manifest_sha256: str,
) -> bool:
    """Check whether a VCC24 reference package matches all current inputs.

    Args:
        metrics_path: Existing metrics-reference manifest.
        dataset: Current dataset contract.
        source_path: Fully sampled VCC24 k-space.
        source_sha256: Exact source payload hash.
        report_path: Direct-source report.
        report_sha256: Exact report hash.
        mask_path: Approved binary mask.
        mask_manifest_path: Approved mask manifest.
        mask_manifest_sha256: Exact mask-manifest hash.

    Returns:
        ``True`` only when all paths, hashes, and statuses match.
    """
    try:
        manifest = _load_json(metrics_path, "metrics-reference manifest")
        reference = manifest["ranking_reference"]
        brain = manifest["brain_mask"]
        reference_path = Path(reference["path"])
        return all(
            (
                manifest.get("status") == "approved_for_metrics",
                manifest.get("dataset", {}).get("manifest_sha256") == dataset.sha256,
                Path(reference.get("source_kspace", "")).resolve()
                == source_path.resolve(),
                reference.get("source_kspace_sha256") == source_sha256,
                Path(reference.get("source_report", "")).resolve()
                == report_path.resolve(),
                reference.get("source_report_sha256") == report_sha256,
                reference_path.is_file(),
                sha256_file(reference_path) == reference.get("sha256"),
                Path(reference.get("sidecar", "")).is_file(),
                sha256_file(Path(reference.get("sidecar", "")))
                == reference.get("sidecar_sha256"),
                Path(brain.get("path", "")).resolve() == mask_path.resolve(),
                brain.get("sha256") == sha256_file(mask_path),
                Path(brain.get("manifest", "")).resolve()
                == mask_manifest_path.resolve(),
                brain.get("manifest_sha256") == mask_manifest_sha256,
            )
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _prepare_metrics_reference(
    dataset: DatasetManifest,
    *,
    source_path: Path,
    source_sha256: str,
    report_path: Path,
    report_sha256: str,
    mask_path: Path,
    mask_manifest_path: Path,
    mask_manifest_sha256: str,
    mask_voxel_count: int,
    resume: bool,
) -> Path:
    """Create or strictly reuse the VCC24 direct-FFT RSS reference.

    Args:
        dataset: Current dataset contract.
        source_path: Fully sampled VCC24 k-space.
        source_sha256: Exact source payload hash.
        report_path: Direct-source report.
        report_sha256: Exact report hash.
        mask_path: Approved same-subject binary mask.
        mask_manifest_path: Approved mask manifest.
        mask_manifest_sha256: Exact mask-manifest hash.
        mask_voxel_count: Validated nonzero mask count.
        resume: Whether exact complete output may be reused.

    Returns:
        Path to the VCC24 metrics-reference manifest.
    """
    reference_dir = (
        dataset.output_root
        / "evaluation"
        / "native_r3x1_wavelet_sweep"
        / "reference"
    )
    metrics_path = reference_dir / "metrics_reference_manifest.json"
    if metrics_path.exists():
        if resume and _reference_is_reusable(
            metrics_path,
            dataset=dataset,
            source_path=source_path,
            source_sha256=source_sha256,
            report_path=report_path,
            report_sha256=report_sha256,
            mask_path=mask_path,
            mask_manifest_path=mask_manifest_path,
            mask_manifest_sha256=mask_manifest_sha256,
        ):
            print(f"Reusing validated VCC24 metrics reference: {metrics_path}")
            return metrics_path
        raise ValueError("Existing VCC24 metrics reference does not exactly match current inputs.")
    reference_dir.mkdir(parents=True, exist_ok=True)
    final_nifti = reference_dir / "direct_fft_rss_ncc24_ras.nii.gz"
    final_sidecar = reference_dir / "direct_fft_rss_ncc24_ras.json"
    partial_nifti = reference_dir / "direct_fft_rss_ncc24_ras.partial.nii.gz"
    partial_sidecar = reference_dir / "direct_fft_rss_ncc24_ras.partial.json"
    for path in (partial_nifti, partial_sidecar):
        if path.exists():
            path.unlink()
    export_multicoil_rss(
        Namespace(
            kspace=source_path,
            twix=dataset.input_path("twix"),
            output=partial_nifti,
            measurement_index=int(
                _load_json(report_path, "direct-source report")["measurement_index"]
            ),
            canonical_ras=True,
            reference_recon=(
                Path(__file__).resolve().parents[3]
                / "external"
                / "wave-mprage"
                / "recon"
            ),
        )
    )
    reference_image = nib.load(str(partial_nifti))
    mask_image = nib.load(str(mask_path))
    reference_data = np.asarray(reference_image.dataobj)
    if (
        tuple(nib.aff2axcodes(reference_image.affine)) != ("R", "A", "S")
        or not _same_geometry(reference_image, mask_image)
        or not np.isfinite(reference_data).all()
        or not np.any(reference_data > 0)
    ):
        raise ValueError("New VCC24 reference is invalid or differs from approved-mask geometry.")
    os.replace(partial_nifti, final_nifti)
    os.replace(partial_sidecar, final_sidecar)
    manifest = {
        "format_version": 1,
        "status": "approved_for_metrics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "VCC24 MPRAGE Wavelet sweep metrics against fully sampled direct FFT RSS",
        "selection_status": "no_parameter_selected",
        "dataset": {
            "manifest": str(dataset.path),
            "manifest_sha256": dataset.sha256,
        },
        "ranking_reference": {
            "kind": "direct_fft_rss",
            "path": str(final_nifti),
            "sha256": sha256_file(final_nifti),
            "sidecar": str(final_sidecar),
            "sidecar_sha256": sha256_file(final_sidecar),
            "source_kspace": str(source_path),
            "source_kspace_sha256": source_sha256,
            "source_report": str(report_path),
            "source_report_sha256": report_sha256,
            "virtual_coils": 24,
        },
        "brain_mask": {
            "usage": "metrics_only",
            "path": str(mask_path),
            "sha256": sha256_file(mask_path),
            "manifest": str(mask_manifest_path),
            "manifest_sha256": mask_manifest_sha256,
            "voxel_count": mask_voxel_count,
            "reuse_policy": "same-subject approved binary mask; intensities are not reused",
        },
        "scientific_policy": {
            "reference_recomputed_for_ncc24": True,
            "historical_ncc12_reference_intensities_reused": False,
            "automatic_winner_selection": False,
        },
    }
    write_json_atomic(metrics_path, manifest)
    return metrics_path


def _geometry_is_reusable(path: Path, metrics_path: Path) -> bool:
    """Check whether a geometry report still binds every case manifest.

    Args:
        path: Exact-grid geometry report.
        metrics_path: Current metrics-reference manifest.

    Returns:
        ``True`` when reference and candidate hashes remain exact.
    """
    try:
        report = _load_json(path, "geometry report")
        if (
            report.get("status") != "passed"
            or report.get("metrics_reference_manifest", {}).get("sha256")
            != sha256_file(metrics_path)
            or report.get("case_count") != len(EXPECTED_LAMBDAS)
        ):
            return False
        return all(
            Path(case["run_manifest"]).is_file()
            and sha256_file(Path(case["run_manifest"])) == case["run_manifest_sha256"]
            for case in report["cases"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _metrics_are_reusable(
    metrics_dir: Path, metrics_path: Path, geometry_path: Path
) -> bool:
    """Check whether completed metric tables and plots retain exact hashes.

    Args:
        metrics_dir: Completed metrics directory.
        metrics_path: Current metrics-reference manifest.
        geometry_path: Current exact-grid report.

    Returns:
        ``True`` when every recorded artifact and input remains exact.
    """
    provenance_path = metrics_dir / "metrics_provenance.json"
    try:
        report = _load_json(provenance_path, "metrics provenance")
        if (
            report.get("status") != "complete"
            or report.get("selection_status") != "no_parameter_selected"
            or report.get("metrics_reference_manifest", {}).get("sha256")
            != sha256_file(metrics_path)
            or report.get("geometry_report", {}).get("sha256")
            != sha256_file(geometry_path)
        ):
            return False
        csv_record = report["metrics_csv"]
        if sha256_file(Path(csv_record["path"])) != csv_record["sha256"]:
            return False
        derived = report["fixed_masks"]["derived"]
        if any(
            sha256_file(Path(record["path"])) != record["sha256"]
            for record in derived.values()
        ):
            return False
        return all(
            sha256_file(Path(record["path"])) == record["sha256"]
            for record in report["plots"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _replace_path_prefix(value: Any, old: Path, new: Path) -> Any:
    """Replace staging-path strings with their final canonical prefix.

    Args:
        value: Nested provenance value.
        old: Staging directory prefix.
        new: Final directory prefix.

    Returns:
        Nested value with owned staging paths rewritten.
    """
    if isinstance(value, dict):
        return {key: _replace_path_prefix(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, str) and value.startswith(str(old) + os.sep):
        return str(new) + value[len(str(old)) :]
    return value


def _remove_owned_staging(path: Path) -> None:
    """Remove the evaluator's named incomplete staging directory.

    Args:
        path: Exact owned staging directory.
    """
    if path.exists():
        shutil.rmtree(path)


def _repair_owned_staging_paths(metrics_dir: Path, staging: Path) -> None:
    """Canonicalize provenance after interruption immediately following rename.

    Args:
        metrics_dir: Final metrics directory.
        staging: Former staging-directory prefix embedded in provenance.
    """
    provenance_path = metrics_dir / "metrics_provenance.json"
    if not provenance_path.is_file():
        return
    report = _load_json(provenance_path, "metrics provenance")
    repaired = _replace_path_prefix(report, staging, metrics_dir)
    if repaired != report:
        write_json_atomic(provenance_path, repaired)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare the VCC24 reference and write non-ranking curves.

    Args:
        args: Parsed dataset, mask, and resume settings.

    Returns:
        Completed metrics-provenance object.
    """
    dataset = load_dataset_manifest(args.dataset_manifest)
    load_passed_inspection(dataset)
    validate_one_shot_contract(dataset.payload)
    mask_manifest_path = args.approved_brain_mask_manifest.expanduser().resolve()
    _, mask_path, mask_manifest_sha256, mask_voxel_count = _validate_approved_mask(
        mask_manifest_path
    )
    source_path, report_path, source_sha256, report_sha256 = _source_contract(dataset)
    metrics_reference = _prepare_metrics_reference(
        dataset,
        source_path=source_path,
        source_sha256=source_sha256,
        report_path=report_path,
        report_sha256=report_sha256,
        mask_path=mask_path,
        mask_manifest_path=mask_manifest_path,
        mask_manifest_sha256=mask_manifest_sha256,
        mask_voxel_count=mask_voxel_count,
        resume=args.resume,
    )

    evaluation_root = (
        dataset.output_root / "evaluation" / "native_r3x1_wavelet_sweep"
    )
    sweep_root = (
        dataset.output_root / "reconstructions" / "native_r3x1_wavelet_sweep"
    )
    geometry_path = evaluation_root / "exact_grid_geometry.json"
    expected_cases = [
        f"wavelet:{canonical_lambda(value)}" for value in EXPECTED_LAMBDAS
    ]
    if geometry_path.exists():
        if not args.resume or not _geometry_is_reusable(geometry_path, metrics_reference):
            raise ValueError("Existing geometry report is stale or mismatched.")
        print(f"Reusing validated exact-grid report: {geometry_path}")
    else:
        validate_metrics_geometry(
            Namespace(
                metrics_reference_manifest=metrics_reference,
                sweep_root=[sweep_root],
                expected_case=expected_cases,
                output=geometry_path,
            )
        )

    metrics_dir = evaluation_root / "metrics"
    staging = evaluation_root / "metrics.incomplete"
    if metrics_dir.exists():
        if args.resume:
            _repair_owned_staging_paths(metrics_dir, staging)
        if args.resume and _metrics_are_reusable(
            metrics_dir, metrics_reference, geometry_path
        ):
            print(f"Reusing validated MPRAGE lambda curves: {metrics_dir}")
            return _load_json(
                metrics_dir / "metrics_provenance.json", "metrics provenance"
            )
        raise ValueError("Existing MPRAGE metric output is incomplete or mismatched.")

    _remove_owned_staging(staging)
    evaluate_regularization(
        Namespace(
            metrics_reference_manifest=metrics_reference,
            geometry_report=geometry_path,
            output_dir=staging,
        )
    )
    os.replace(staging, metrics_dir)
    provenance_path = metrics_dir / "metrics_provenance.json"
    staged_result = _load_json(provenance_path, "metrics provenance")
    result = _replace_path_prefix(staged_result, staging, metrics_dir)
    write_json_atomic(provenance_path, result)
    print(f"MPRAGE lambda curves: {metrics_dir / 'plots' / 'wavelet_metrics.png'}")
    print("No parameter winner was selected.")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MPRAGE VCC24 Wavelet post-sweep evaluator.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process status code.
    """
    try:
        run(_parser().parse_args(argv))
    except (
        DatasetManifestError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
