"""Record explicit, hash-bound Wave-MPRAGE ROVir reconstruction choices."""

from __future__ import annotations

import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np

from .bart_io import cfl_record, sha256_file
from .pca_control import _file_record, _read_json, _same_cfl_content, _write_json


def record_mprage_rovir_selection(
    feasibility_root: str | Path,
    reconstruction_root: str | Path,
    comparison_qc_manifest: str | Path,
    *,
    selection_scope: str,
    candidate_id: str,
    virtual_coils: int,
    ecalib_crop: float,
    reviewer_note: str,
    confirm_user_selection: bool,
) -> dict[str, Any]:
    """Validate and record one user-selected MPRAGE ROVir reconstruction.

    Args:
        feasibility_root: Reviewed ROVir estimation root containing the
            approved masks and native BART transform.
        reconstruction_root: Reconstruction root containing the selected
            retained-coil branch.
        comparison_qc_manifest: Multi-candidate fixed-window QC manifest used
            for visual review.
        selection_scope: Nonempty dataset or subject label to which the choice
            applies.
        candidate_id: Explicit approved mask candidate identifier.
        virtual_coils: Selected positive retained ROVir coil count.
        ecalib_crop: Selected finite ecalib crop threshold.
        reviewer_note: Nonempty human rationale stored verbatim.
        confirm_user_selection: Explicit acknowledgement that a person chose
            this candidate after review.

    Returns:
        Completed selection manifest.

    Raises:
        FileExistsError: If the immutable selection record already exists.
        FileNotFoundError: If a required manifest or output is absent.
        ValueError: If confirmation, hashes, provenance, commands, geometry,
            or finite-value validation fails.

    Side Effects:
        Atomically writes ``selection/selection_manifest.json`` below the
        reconstruction root. It does not run BART or alter reconstructions.
    """
    if not confirm_user_selection:
        raise ValueError("Explicit user-selection confirmation is required.")
    if not selection_scope.strip() or not reviewer_note.strip():
        raise ValueError("Selection scope and reviewer note must be nonempty.")
    ncc = int(virtual_coils)
    crop = float(ecalib_crop)
    if ncc < 1 or not np.isfinite(crop) or crop <= 0:
        raise ValueError("Virtual coils and ecalib crop must be positive and finite.")

    feasibility = Path(feasibility_root).expanduser().resolve()
    reconstruction = Path(reconstruction_root).expanduser().resolve()
    qc_path = Path(comparison_qc_manifest).expanduser().resolve()
    output_path = reconstruction / "selection" / "selection_manifest.json"
    if output_path.exists():
        raise FileExistsError(f"Selection record already exists: {output_path}")

    approved_path = feasibility / "masks" / "approved" / "manifest.json"
    candidates_path = feasibility / "masks" / "candidates" / "manifest.json"
    transform_qc_path = feasibility / "manifests" / "rovir_transform_qc.json"
    shared_path = reconstruction / "shared" / "manifest.json"
    approved = _read_json(approved_path)
    candidates = _read_json(candidates_path)
    transform_qc = _read_json(transform_qc_path)
    shared = _read_json(shared_path)
    qc = _read_json(qc_path)

    if approved.get("status") != "mprage_rovir_masks_approved":
        raise ValueError("ROVir mask approval is incomplete.")
    if approved.get("candidate_id") != candidate_id:
        raise ValueError("Requested candidate differs from the approved ROVir mask.")
    if sha256_file(candidates_path) != approved.get("candidate_manifest_sha256"):
        raise ValueError("Approved ROVir candidate manifest changed after approval.")
    candidate_entries = [
        item for item in candidates.get("candidates", []) if item.get("candidate_id") == candidate_id
    ]
    if len(candidate_entries) != 1:
        raise ValueError("Approved candidate must occur exactly once in its manifest.")
    for name in ("positive_estimation_mask", "negative_estimation_mask"):
        current = cfl_record(feasibility / "masks" / "approved" / name)
        if not _same_cfl_content(approved[name], current):
            raise ValueError(f"Approved {name} changed after mask approval.")

    if (
        transform_qc.get("status") != "mprage_bart_rovir_transform_qc_ready"
        or transform_qc.get("automatic_selection") is not False
        or transform_qc.get("solver_backend") != "bart rovir only"
    ):
        raise ValueError("ROVir transform QC contract is incomplete or automatic.")
    if transform_qc.get("selected_virtual_coils") is not None:
        raise ValueError("Feasibility QC must not preselect a virtual-coil count.")
    transform = cfl_record(feasibility / "transforms" / "rovir_full" / "transform")
    if not _same_cfl_content(transform_qc["transform"], transform):
        raise ValueError("ROVir transform changed after transform QC.")

    branch_id = f"rovir_ncc{ncc}"
    if (
        shared.get("status") != "measured_wave_mprage_rovir_coil_count_comparison_ready"
        or shared.get("automatic_winner_selected") is not False
        or str(feasibility) != str(Path(shared.get("feasibility_root", "")).resolve())
        or ncc not in shared.get("channel_counts", [])
    ):
        raise ValueError("Selected reconstruction is not bound to this feasibility root and Ncc.")
    branch_path = reconstruction / branch_id / "bart_inputs" / "manifest.json"
    _validate_file_record(shared.get("branches", {}).get(branch_id), branch_path, "branch manifest")
    branch = _read_json(branch_path)
    if (
        branch.get("status") != "measured_wave_mprage_rovir_control_ready"
        or branch.get("rovir", {}).get("virtual_coils") != ncc
        or branch.get("rovir", {}).get("solver_backend") != "bart rovir only"
        or branch.get("rovir", {}).get("basis_applied_identically_to_image_and_acs") is not True
        or branch.get("sampling_validation", {}).get("zero_outside_sampling_mask") is not True
        or branch.get("sampling_validation", {}).get("acs_merged_into_wave_image_kspace") is not False
        or branch.get("psf_calibration", {}).get("reused_without_recalibration") is not True
    ):
        raise ValueError("Selected branch violates the reviewed ROVir reconstruction contract.")
    _validate_file_record(
        branch["rovir"]["transform_qc_manifest"], transform_qc_path, "transform QC manifest"
    )
    if not _same_cfl_content(branch["rovir"]["transform_source"], transform):
        raise ValueError("Selected branch uses a different ROVir transform.")

    outputs = _validated_nifti_outputs(reconstruction / branch_id / "nifti" / "fista_r0", branch_path)
    magnitude_record = outputs["magnitude"]["nifti"]
    if (
        qc.get("status") != "mprage_rovir_negative_roi_series_qc_ready"
        or qc.get("automatic_winner_selected") is not False
        or qc.get("display_window", {}).get("shared_between_rows") is not True
    ):
        raise ValueError("A complete manual, shared-window comparison QC is required.")
    selected_qc_entries = [
        item
        for item in qc.get("candidates", [])
        if Path(item.get("nifti", {}).get("path", "")).resolve()
        == Path(magnitude_record["path"]).resolve()
    ]
    if len(selected_qc_entries) != 1:
        raise ValueError("Selected magnitude must occur exactly once in comparison QC.")
    _validate_file_record(selected_qc_entries[0]["nifti"], Path(magnitude_record["path"]), "QC magnitude")

    ecalib_command_path = reconstruction / branch_id / "bart_output" / "ecalib_command.txt"
    wave_command_path = reconstruction / branch_id / "bart_output" / "fista_r0" / "wave_command.txt"
    ecalib_command = ecalib_command_path.read_text(encoding="utf-8").strip()
    wave_command = wave_command_path.read_text(encoding="utf-8").strip()
    _validate_commands(ecalib_command, wave_command, crop)

    comparison_only = []
    for item in qc["candidates"]:
        if item is selected_qc_entries[0]:
            continue
        path = Path(item["nifti"]["path"]).resolve()
        _validate_file_record(item["nifti"], path, "comparison-only magnitude")
        comparison_only.append({"label": item["label"], "magnitude_nifti": item["nifti"]})

    selection = {
        "format_version": 1,
        "status": "complete",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": selection_scope.strip(),
        "selection": {
            "candidate_id": candidate_id,
            "candidate_parameters": candidate_entries[0]["parameters"],
            "solver_backend": "bart rovir only",
            "virtual_coils": ncc,
            "ecalib_crop": crop,
            "reconstruction": "FISTA lambda zero control",
            "lambda": 0.0,
        },
        "selected_artifacts": {
            "approved_mask_manifest": _file_record(approved_path),
            "transform_qc_manifest": _file_record(transform_qc_path),
            "transform": transform,
            "shared_reconstruction_manifest": _file_record(shared_path),
            "branch_manifest": _file_record(branch_path),
            "outputs": outputs,
            "ecalib_command": _file_record(ecalib_command_path),
            "wave_command": _file_record(wave_command_path),
        },
        "evidence": {
            "comparison_qc_manifest": _file_record(qc_path),
            "selected_qc_entry": selected_qc_entries[0],
            "comparison_only_candidates": comparison_only,
        },
        "decision": {
            "method": "explicit user decision after fixed-window visual review",
            "reviewer_note": reviewer_note.strip(),
            "automatic_selection_performed": False,
            "generalizes_to_other_datasets": False,
        },
    }
    _write_json(output_path, selection)
    return selection


def _validate_file_record(
    record: Mapping[str, Any] | None, expected_path: Path, label: str
) -> None:
    """Validate one ordinary-file record against an expected path.

    Args:
        record: Mapping containing path, size, and SHA-256 fields.
        expected_path: Exact existing file expected by the caller.
        label: Human-readable artifact name used in errors.

    Returns:
        None.
    """
    if not isinstance(record, Mapping):
        raise ValueError(f"Missing {label} record.")
    current = _file_record(expected_path)
    if (
        Path(str(record.get("path", ""))).resolve() != expected_path.resolve()
        or record.get("size_bytes") != current["size_bytes"]
        or record.get("sha256") != current["sha256"]
    ):
        raise ValueError(f"{label} path, size, or hash changed.")


def _validated_nifti_outputs(nifti_root: Path, branch_path: Path) -> dict[str, Any]:
    """Validate the selected magnitude and phase NIfTI pair.

    Args:
        nifti_root: FISTA-r0 NIfTI directory for one ROVir branch.
        branch_path: Prepared-input manifest that both sidecars must reference.

    Returns:
        Magnitude and phase file/sidecar records.
    """
    result: dict[str, Any] = {}
    images = []
    for part, token in (("magnitude", "_part-mag_"), ("phase", "_part-phase_")):
        matches = sorted(nifti_root.rglob(f"*{token}*.nii.gz"))
        if len(matches) != 1:
            raise ValueError(f"Expected one selected {part} NIfTI, found {len(matches)}.")
        nifti_path = matches[0].resolve()
        sidecar_path = nifti_path.with_suffix("").with_suffix(".json")
        sidecar = _read_json(sidecar_path)
        if Path(sidecar.get("PreparedInputManifest", "")).resolve() != branch_path.resolve():
            raise ValueError(f"Selected {part} sidecar references another prepared input.")
        image = nib.load(nifti_path)
        values = np.asanyarray(image.dataobj)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError(f"Selected {part} NIfTI must be finite and three-dimensional.")
        images.append(image)
        result[part] = {
            "nifti": _file_record(nifti_path),
            "sidecar": _file_record(sidecar_path),
        }
    if images[0].shape != images[1].shape or not np.allclose(
        images[0].affine, images[1].affine, atol=1e-5
    ):
        raise ValueError("Selected magnitude and phase NIfTIs have different geometry.")
    return result


def _validate_commands(ecalib_command: str, wave_command: str, crop: float) -> None:
    """Validate selected ecalib and BART Wave command parameters.

    Args:
        ecalib_command: Recorded complete ecalib command.
        wave_command: Recorded complete BART Wave command.
        crop: Expected ecalib crop value.

    Returns:
        None.
    """
    ecalib = shlex.split(ecalib_command)
    wave = shlex.split(wave_command)
    if ecalib[:2] != ["bart", "ecalib"] or "-c" not in ecalib:
        raise ValueError("Recorded ecalib command is incomplete.")
    if not np.isclose(float(ecalib[ecalib.index("-c") + 1]), crop):
        raise ValueError("Recorded ecalib crop differs from the selected value.")
    if wave[:2] != ["bart", "wave"] or not all(flag in wave for flag in ("-w", "-f", "-r")):
        raise ValueError("Recorded command is not BART Wave FISTA.")
    if float(wave[wave.index("-r") + 1]) != 0.0:
        raise ValueError("Selected ROVir reconstruction must be the FISTA lambda-zero control.")
