"""Orchestrate the public two-command Wave-MPRAGE ROVir workflow.

Python performs validation, provenance, and bounded preparation. The public
shell entry point retains every BART command explicitly.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bart_io import cfl_record, sha256_file
from .rovir_control import (
    prepare_mprage_rovir_comparison,
    write_mprage_rovir_mask_series_qc,
)
from .rovir_feasibility import (
    APPROVED_MASK_MANIFEST,
    ROVIR_QC_MANIFEST,
    _validated_source_contract,
    approve_region_mask_candidate,
    derive_box_union_mask_candidate,
    export_mprage_physical_calibration,
    prepare_masked_rovir_inputs,
    recommend_null_boxes,
    record_calibration_images,
    write_rovir_transform_qc,
)

CANONICAL_ROVIR_MANIFEST = Path("normal") / "rovir" / "manifest.json"


def validate_normal_rovir_invocation(
    twix: str | Path, output_root: str | Path, sequence: str | Path
) -> dict[str, Any]:
    """Validate public CLI sources against accepted prepared MPRAGE inputs.

    Args:
        twix: Measured Wave-MPRAGE TWIX supplied on the command line.
        output_root: Existing reconstruction root containing prepared inputs.
        sequence: Matching Pulseq sequence supplied on the command line.

    Returns:
        Validated canonical workflow context.

    Raises:
        FileNotFoundError: If a source or required prepared artifact is absent.
        ValueError: If either source differs from the normal manifest or the
            recorded normal provenance is incomplete.
    """
    _validated_source_contract(
        twix,
        sequence,
        output_root,
        include_twix_hash=False,
    )
    return normal_rovir_context(output_root)


def normal_rovir_context(output_root: str | Path) -> dict[str, Any]:
    """Resolve sources and optional comparison settings from prepared inputs.

    Args:
        output_root: Existing reconstruction root.

    Returns:
        Source paths, ROVir paths, ecalib crop policy, and optional standard
        reconstruction comparison.

    Raises:
        FileNotFoundError: If required prepared inputs or sources are absent.
        ValueError: If manifests, source paths, or command records are invalid.
    """
    root = Path(output_root).expanduser().resolve()
    manifest_path = root / "normal" / "bart_inputs" / "manifest.json"
    manifest = _read_json(manifest_path)
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Normal manifest lacks source provenance.")
    twix = _source_path(source.get("twix"), "TWIX")
    sequence = _source_path(source.get("sequence"), "sequence")
    normal_output = root / "normal" / "bart_output"
    normal_magnitudes = _magnitude_candidates(
        root / "normal" / "nifti" / "fista_r0", recursive=True
    )
    normal_nifti = normal_magnitudes[0] if len(normal_magnitudes) == 1 else None
    comparison_status = (
        "available"
        if normal_nifti is not None
        else "not_available"
        if not normal_magnitudes
        else "ambiguous_multiple_magnitude_niftis"
    )
    ecalib_record = normal_output / "ecalib_command.txt"
    if ecalib_record.is_file():
        crop = _ecalib_crop(ecalib_record, "Standard normal")
        crop_source = "standard_normal_ecalib_command"
    else:
        crop = 0.6
        crop_source = "mprage_launcher_default"
    rovir_root = root / "normal" / "rovir"
    return {
        "output_root": str(root),
        "normal_manifest": str(manifest_path),
        "normal_manifest_sha256": sha256_file(manifest_path),
        "twix": str(twix),
        "sequence": str(sequence),
        "normal_reconstruction_required": False,
        "normal_magnitude_nifti": None if normal_nifti is None else str(normal_nifti),
        "normal_magnitude_candidate_count": len(normal_magnitudes),
        "standard_comparison_status": comparison_status,
        "ecalib_crop": crop,
        "ecalib_crop_source": crop_source,
        "rovir_root": str(rovir_root),
        "feasibility_root": str(rovir_root / "feasibility"),
    }


def prepare_inspection(output_root: str | Path) -> dict[str, Any]:
    """Validate prepared inputs and export corrected physical ACS.

    Args:
        output_root: Existing reconstruction root.

    Returns:
        Public workflow context and physical-calibration manifest.

    Side Effects:
        Writes only under ``normal/rovir/feasibility``.
    """
    context = normal_rovir_context(output_root)
    physical = export_mprage_physical_calibration(
        context["twix"],
        context["sequence"],
        context["output_root"],
        context["feasibility_root"],
    )
    return {"status": "mprage_rovir_inspection_prepared", "context": context, "physical": physical}


def finish_inspection(output_root: str | Path, bart_version_file: str | Path) -> dict[str, Any]:
    """Record explicit BART ACS images and write an ROI recommendation.

    Args:
        output_root: Existing reconstruction root.
        bart_version_file: Text file generated by ``bart version``.

    Returns:
        Image and recommendation manifests.

    Side Effects:
        Writes review figures and recommendation diagnostics.
    """
    context = normal_rovir_context(output_root)
    images = record_calibration_images(context["feasibility_root"], bart_version_file)
    recommendation = recommend_null_boxes(context["feasibility_root"])
    return {"status": "mprage_rovir_inspection_ready", "context": context, "images": images, "recommendation": recommendation}


def prepare_reviewed_candidate(
    output_root: str | Path,
    null_boxes: Sequence[str | Mapping[str, Sequence[int]]],
) -> dict[str, Any]:
    """Create but do not approve one canonical null-box union.

    Args:
        output_root: Existing reconstruction root.
        null_boxes: User-supplied inclusive boxes.

    Returns:
        Candidate manifest and exact candidate identifier.
    """
    context = normal_rovir_context(output_root)
    manifest = derive_box_union_mask_candidate(context["feasibility_root"], null_boxes)
    candidate_id = manifest.get("active_candidate_id")
    candidates = [
        record
        for record in manifest.get("candidates", [])
        if record.get("candidate_id") == candidate_id
    ]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one active canonical null-box candidate.")
    return {
        "status": "mprage_rovir_candidate_ready_for_review",
        "candidate_id": candidate_id,
        "review_overlay": candidates[0]["review_overlay"]["path"],
        "manifest": manifest,
    }


def approve_and_prepare_solver(output_root: str | Path, candidate_id: str) -> dict[str, Any]:
    """Approve the exact reviewed ID and prepare native BART ROVir inputs.

    Args:
        output_root: Existing reconstruction root.
        candidate_id: Exact hash-derived candidate identifier.

    Returns:
        Approval and solver-input manifests.

    Side Effects:
        Installs immutable approved masks and masked coil images.
    """
    context = normal_rovir_context(output_root)
    feasibility = Path(context["feasibility_root"])
    approved_path = feasibility / APPROVED_MASK_MANIFEST
    if approved_path.is_file():
        approved = _read_json(approved_path)
        if approved.get("candidate_id") != candidate_id:
            raise ValueError("Existing approved ROI has a different candidate ID.")
    else:
        approved = approve_region_mask_candidate(feasibility, candidate_id)
    solver = prepare_masked_rovir_inputs(feasibility)
    return {"status": "mprage_rovir_solver_inputs_ready", "approved": approved, "solver": solver}


def record_transform_and_prepare_reconstruction(
    output_root: str | Path,
    bart_version_file: str | Path,
    virtual_coils: int,
) -> dict[str, Any]:
    """Validate the BART transform and project one selected ROVir branch.

    Args:
        output_root: Existing reconstruction root.
        bart_version_file: Exact BART version output.
        virtual_coils: Explicit retained leading ROVir coil count.

    Returns:
        Transform QC and canonical prepared-input manifests.

    Side Effects:
        Writes transform diagnostics and ``normal/rovir/bart_inputs``.
    """
    if isinstance(virtual_coils, bool) or int(virtual_coils) < 1:
        raise ValueError("Virtual-coil count must be a positive integer.")
    context = normal_rovir_context(output_root)
    qc = write_rovir_transform_qc(context["feasibility_root"], bart_version_file)
    prepared = prepare_mprage_rovir_comparison(
        context["twix"],
        context["sequence"],
        context["output_root"],
        context["feasibility_root"],
        context["rovir_root"],
        channel_counts=(int(virtual_coils),),
        canonical_single_branch=True,
    )
    return {"status": "mprage_rovir_reconstruction_inputs_ready", "context": context, "transform_qc": qc, "prepared": prepared}


def finalize_normal_rovir(output_root: str | Path, virtual_coils: int) -> dict[str, Any]:
    """Write fixed-window QC and the canonical completed ROVir contract.

    Args:
        output_root: Existing reconstruction root.
        virtual_coils: Explicit retained coil count used by reconstruction.

    Returns:
        Canonical completed ROVir manifest.

    Raises:
        FileNotFoundError: If reconstruction or NIfTI artifacts are missing.
        ValueError: If the selected count or provenance is inconsistent.

    Side Effects:
        Writes shared-window QC and ``normal/rovir/manifest.json``.
    """
    context = normal_rovir_context(output_root)
    root = Path(context["output_root"])
    rovir_root = Path(context["rovir_root"])
    feasibility = Path(context["feasibility_root"])
    inputs_manifest_path = rovir_root / "bart_inputs" / "manifest.json"
    inputs_manifest = _read_json(inputs_manifest_path)
    rovir = inputs_manifest.get("rovir", {})
    if not isinstance(rovir, Mapping) or int(rovir.get("virtual_coils", -1)) != int(virtual_coils):
        raise ValueError("Prepared ROVir coil count differs from the requested count.")
    image_base = rovir_root / "bart_output" / "fista_r0" / "image_wave"
    _require_cfl(image_base)
    rovir_magnitude = _single_magnitude(
        rovir_root / "nifti" / "fista_r0", recursive=True
    )
    series: list[tuple[str, str | Path]] = []
    figure_filename = "rovir_fista_r0_fixed_window.png"
    comparison_status = context["standard_comparison_status"]
    comparison_error: str | None = None
    if context["normal_magnitude_nifti"] is not None:
        series.append(
            ("standard normal FISTA lambda=0", context["normal_magnitude_nifti"])
        )
        figure_filename = "standard_vs_rovir_fista_r0_fixed_window.png"
    series.append((f"ROVir-{virtual_coils} FISTA lambda=0", rovir_magnitude))
    try:
        qc = write_mprage_rovir_mask_series_qc(
            tuple(series),
            rovir_root / "qc",
            figure_filename=figure_filename,
        )
    except Exception as exc:
        if context["normal_magnitude_nifti"] is None:
            raise
        # A stale optional standard NIfTI must not block the independent branch.
        comparison_status = "skipped_invalid_standard_qc_source"
        comparison_error = f"{type(exc).__name__}: {exc}"
        qc = write_mprage_rovir_mask_series_qc(
            ((f"ROVir-{virtual_coils} FISTA lambda=0", rovir_magnitude),),
            rovir_root / "qc",
            figure_filename="rovir_fista_r0_fixed_window.png",
        )
    approved_path = feasibility / APPROVED_MASK_MANIFEST
    approved = _read_json(approved_path)
    transform_qc_path = feasibility / ROVIR_QC_MANIFEST
    transform_qc = _read_json(transform_qc_path)
    transform = feasibility / "transforms" / "rovir_full" / "transform"
    rovir_ecalib_record = rovir_root / "bart_output" / "ecalib_command.txt"
    rovir_ecalib_command = rovir_ecalib_record.read_text(encoding="utf-8").strip()
    crop_match = re.search(r"(?:^|\s)-c\s+([^\s]+)", rovir_ecalib_command)
    if crop_match is None:
        raise ValueError("ROVir ecalib command does not record a -c crop value.")
    rovir_crop = float(crop_match.group(1))
    standard_magnitude_record = None
    if context["normal_magnitude_nifti"] is not None:
        standard_path = Path(context["normal_magnitude_nifti"])
        if standard_path.is_file():
            standard_magnitude_record = _file_record(standard_path)
    contract = {
        "format_version": 1,
        "status": "mprage_normal_rovir_complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root),
        "normal_source_manifest": {
            "path": context["normal_manifest"],
            "sha256": context["normal_manifest_sha256"],
        },
        "source": inputs_manifest["source"],
        "coil_processing": {
            "label": "bart_rovir",
            "candidate_id": approved["candidate_id"],
            "virtual_coils": int(virtual_coils),
            "approved_mask_manifest": _file_record(approved_path),
            "transform": cfl_record(transform),
            "transform_qc_manifest": _file_record(transform_qc_path),
            "basis_applied_identically_to_image_and_acs": True,
        },
        "ecalib": {
            "crop": rovir_crop,
            "reference_crop": context["ecalib_crop"],
            "reference_crop_source": context["ecalib_crop_source"],
            "normal_crop": (
                context["ecalib_crop"]
                if context["ecalib_crop_source"] == "standard_normal_ecalib_command"
                else None
            ),
            "inherited_from_normal": (
                context["ecalib_crop_source"] == "standard_normal_ecalib_command"
                and rovir_crop == context["ecalib_crop"]
            ),
            "deliberate_override": rovir_crop != context["ecalib_crop"],
            "command_record": _file_record(rovir_ecalib_record),
            "coil_sens": cfl_record(rovir_root / "bart_output" / "coil_sens"),
        },
        "reconstruction": {
            "method": "fista_r0",
            "lambda": 0.0,
            "regularization_optimized_for_rovir": False,
            "command_record": _file_record(image_base.parent / "wave_command.txt"),
            "image": cfl_record(image_base),
        },
        "artifacts": {
            "prepared_inputs_manifest": _file_record(inputs_manifest_path),
            "magnitude_nifti": _file_record(rovir_magnitude),
            "qc_manifest": _file_record(rovir_root / "qc" / "manifest.json"),
            "standard_comparison_status": comparison_status,
            "standard_comparison_error": comparison_error,
            "standard_magnitude_nifti": standard_magnitude_record,
        },
        "normal_reconstruction_required": False,
        "psf_recalibrated": False,
        "roi_automatically_approved": False,
        "retro_consumable": True,
        "transform_validation": transform_qc["transform_validation"],
    }
    _write_json(rovir_root / "manifest.json", contract)
    return contract


def finalize_existing_normal_rovir(output_root: str | Path) -> dict[str, Any]:
    """Finalize or validate an already reconstructed normal ROVir branch.

    Args:
        output_root: Existing reconstruction root containing ROVir artifacts.

    Returns:
        Existing or newly written canonical completed ROVir contract.

    Raises:
        FileNotFoundError: If prepared or reconstructed ROVir artifacts are absent.
        ValueError: If the existing contract or selected coil count is invalid.

    Side Effects:
        When the canonical contract is absent, writes normal ROVir QC and the
        completion manifest after validating all already generated artifacts.
        It does not run BART or modify scientific arrays.
    """
    context = normal_rovir_context(output_root)
    canonical_path = Path(context["output_root"]) / CANONICAL_ROVIR_MANIFEST
    if canonical_path.is_file():
        existing = _read_json(canonical_path)
        if (
            existing.get("status") != "mprage_normal_rovir_complete"
            or existing.get("retro_consumable") is not True
            or existing.get("normal_source_manifest", {}).get("sha256")
            != context["normal_manifest_sha256"]
        ):
            raise ValueError("Existing canonical normal ROVir contract is incompatible.")
        return existing
    inputs_manifest = _read_json(
        Path(context["rovir_root"]) / "bart_inputs" / "manifest.json"
    )
    rovir = inputs_manifest.get("rovir")
    if not isinstance(rovir, Mapping):
        raise ValueError("Prepared normal ROVir inputs lack coil-processing metadata.")
    try:
        virtual_coils = int(rovir["virtual_coils"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Prepared normal ROVir virtual-coil count is invalid.") from exc
    if virtual_coils < 1:
        raise ValueError("Prepared normal ROVir virtual-coil count must be positive.")
    return finalize_normal_rovir(output_root, virtual_coils)


def load_recommended_boxes(output_root: str | Path) -> list[dict[str, list[int]]]:
    """Load inspect-stage recommended boxes without approving them.

    Args:
        output_root: Existing reconstruction root.

    Returns:
        Nonempty recommended box list.

    Raises:
        ValueError: If inspect produced no safe recommendation.
    """
    context = normal_rovir_context(output_root)
    path = Path(context["feasibility_root"]) / "diagnostics" / "roi_recommendation" / "roi_recommendation.json"
    manifest = _read_json(path)
    boxes = manifest.get("recommended_boxes")
    if manifest.get("status") != "safe_conservative_recommendation_available" or not isinstance(boxes, list) or not boxes:
        raise ValueError("Inspect produced no safe automatic null-box recommendation.")
    return boxes


def _source_path(record: object, label: str) -> Path:
    """Validate and return one recorded source path.

    Args:
        record: Source manifest mapping.
        label: Human-readable source name.

    Returns:
        Existing resolved path.
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise ValueError(f"Normal manifest lacks a valid {label} path.")
    path = Path(record["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Recorded {label} is unavailable: {path}")
    return path


def _single_magnitude(directory: Path, *, recursive: bool) -> Path:
    """Find exactly one magnitude NIfTI below a reconstruction directory.

    Args:
        directory: Directory to search.
        recursive: Whether to search child method directories.

    Returns:
        Unique magnitude NIfTI path.
    """
    pattern = "**/*_part-mag_*.nii.gz" if recursive else "*_part-mag_*.nii.gz"
    matches = sorted(directory.glob(pattern)) if directory.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one magnitude NIfTI below {directory}; found {len(matches)}."
        )
    return matches[0].resolve()


def _magnitude_candidates(directory: Path, *, recursive: bool) -> list[Path]:
    """Return all candidate magnitude NIfTIs without making them a hard gate.

    Args:
        directory: Directory containing an optional standard reconstruction.
        recursive: Whether to include nested BIDS subject directories.

    Returns:
        Sorted resolved candidate paths; the list may be empty or ambiguous.
    """
    pattern = "**/*_part-mag_*.nii.gz" if recursive else "*_part-mag_*.nii.gz"
    return (
        [path.resolve() for path in sorted(directory.glob(pattern))]
        if directory.is_dir()
        else []
    )


def _ecalib_crop(record: Path, label: str) -> float:
    """Read and validate the crop value from one ecalib command record.

    Args:
        record: Existing ecalib command text file.
        label: Human-readable reconstruction label for errors.

    Returns:
        Validated crop in ``(0, 1]``.

    Raises:
        ValueError: If the command does not contain a valid crop value.
    """
    command = record.read_text(encoding="utf-8").strip()
    match = re.search(r"(?:^|\s)-c\s+([^\s]+)", command)
    if match is None:
        raise ValueError(f"{label} ecalib command does not record a -c crop value.")
    try:
        crop = float(match.group(1))
    except ValueError as exc:
        raise ValueError(f"{label} ecalib crop is not numeric.") from exc
    if not 0 < crop <= 1:
        raise ValueError(f"{label} ecalib crop must lie in (0, 1].")
    return crop


def _require_cfl(base: Path) -> None:
    """Require both files of one BART CFL pair.

    Args:
        base: BART basename.

    Raises:
        FileNotFoundError: If either pair member is absent.
    """
    if not base.with_suffix(".hdr").is_file() or not base.with_suffix(".cfl").is_file():
        raise FileNotFoundError(f"Missing BART CFL pair: {base}.{{hdr,cfl}}")


def _file_record(path: str | Path) -> dict[str, Any]:
    """Return path, size, and SHA-256 for one file.

    Args:
        path: Existing file.

    Returns:
        JSON-native immutable file record.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read one required JSON object.

    Args:
        path: JSON file path.

    Returns:
        Parsed mapping.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON after creating its parent directory.

    Args:
        path: Destination JSON file.
        payload: JSON-native mapping.

    Side Effects:
        Replaces the destination text file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
