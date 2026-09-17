"""Staged ROVir feasibility preparation for measured Wave-MPRAGE data.

This module exports physical-coil calibration data, builds explicitly reviewed
region-mask candidates, prepares masked inputs, and writes transform QC. BART
execution remains in a readable shell script; the only ROVir eigensolver is
the native ``bart rovir`` command.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bart_io import cfl_record, create_cfl, open_cfl, read_shape, sha256_file
from .rovir import (
    region_correlation_diagnostics,
    region_energy_curve,
    validate_region_masks,
    validate_rovir_transform,
)

PHYSICAL_CALIBRATION_MANIFEST = Path("manifests") / "physical_calibration.json"
CALIBRATION_IMAGES_MANIFEST = Path("manifests") / "calibration_images.json"
MANUAL_ANNOTATION_MANIFEST = (
    Path("masks") / "manual_annotation" / "manifest.json"
)
MASK_CANDIDATES_MANIFEST = Path("masks") / "candidates" / "manifest.json"
APPROVED_MASK_MANIFEST = Path("masks") / "approved" / "manifest.json"
ROVIR_INPUT_MANIFEST = Path("manifests") / "rovir_inputs.json"
ROVIR_QC_MANIFEST = Path("manifests") / "rovir_transform_qc.json"
REGION_MASK_NAMES = (
    "preservation_mask",
    "positive_estimation_mask",
    "negative_estimation_mask",
    "contaminated_holdout_mask",
)
MPRAGE_NIFTI_ARRAY_AXIS_FLIPS = (False, False, True)
MPRAGE_NIFTI_AFFINE_AXIS_FLIPS = (True, False, True)
MPRAGE_NIFTI_AXIS_ROLES = ("phase", "readout", "slice")
MANUAL_ROI_LABELS = {
    0: "unassigned_or_mixed",
    1: "clean_desired_signal",
    2: "pure_shoulder_interference",
}


def preflight_mprage_rovir_sources(
    twix: str | Path,
    sequence: str | Path,
    normal_output_root: str | Path,
    feasibility_output_root: str | Path,
) -> dict[str, Any]:
    """Validate source identities and report the non-destructive output state.

    Args:
        twix: Measured Wave-MPRAGE TWIX file.
        sequence: Matching Pulseq sequence file.
        normal_output_root: Existing normal reconstruction root.
        feasibility_output_root: User-approved independent diagnostic root.

    Returns:
        JSON-compatible source, geometry, and output-state report.

    Raises:
        FileNotFoundError: If a required source or normal manifest is absent.
        ValueError: If the supplied sources do not match the normal manifest.
    """
    source = _validated_source_contract(
        twix, sequence, normal_output_root, include_twix_hash=False
    )
    output = Path(feasibility_output_root).expanduser().resolve()
    return {
        "status": "mprage_rovir_preflight_passed",
        "source": source,
        "output_root": str(output),
        "output_exists": output.exists(),
        "output_entries": (
            sorted(path.name for path in output.iterdir()) if output.is_dir() else []
        ),
        "production_work_executed": False,
    }


def export_mprage_physical_calibration(
    twix: str | Path,
    sequence: str | Path,
    normal_output_root: str | Path,
    feasibility_output_root: str | Path,
    *,
    acs_set_index: int = 4,
) -> dict[str, Any]:
    """Export the physical-coil set-4 refscan on its calibration grid.

    Args:
        twix: Measured Wave-MPRAGE TWIX file.
        sequence: Matching Pulseq sequence file.
        normal_output_root: Existing reconstruction root whose manifest binds
            the source and accepted acquisition geometry.
        feasibility_output_root: User-approved independent diagnostic root.
        acs_set_index: Zero-based integrated-refscan set index; set 4 is the
            required Wave-MPRAGE sensitivity-calibration stream.

    Returns:
        Manifest describing the source and exported BART CFL pair.

    Raises:
        FileExistsError: If incompatible diagnostic inputs already exist.
        ValueError: If source provenance, geometry, refscan set, or samples are
            incompatible with the accepted normal acquisition.

    Side Effects:
        Reads the TWIX refscan and atomically installs a physical-coil BART CFL
        below the user-approved feasibility root. It does not launch BART.
    """
    import torch

    source = _validated_source_contract(
        twix, sequence, normal_output_root, include_twix_hash=True
    )
    root = Path(feasibility_output_root).expanduser().resolve()
    manifest_path = root / PHYSICAL_CALIBRATION_MANIFEST
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        _validate_export_reuse(existing, source, root)
        return existing

    destination = root / "inputs" / "physical_calibration"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Physical-calibration directory is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".physical_calibration-", dir=destination.parent)
    )
    try:
        from .mprage import load_wave_mprage_helpers

        native = load_wave_mprage_helpers()
        reference = native.load_ref(os.fspath(Path(twix).expanduser().resolve()))
        geometry = source["geometry"]
        ro_oversampled = int(geometry["readout_oversampled"])
        readout_oversampling = int(geometry["readout_oversampling_factor"])
        logical_readout = int(geometry["logical_matrix_ro_lin_par"][0])
        ncalib = int(source["psf_calibration"]["ncalib"])
        nacs = int(source["psf_calibration"]["nacs"])
        physical_coils = int(source["coil_compression"]["physical_coils"])
        if reference.ndim != 5:
            raise ValueError(
                "Integrated refscan must expose an explicit set dimension; "
                f"found shape {tuple(reference.shape)}."
            )
        expected_prefix = (ro_oversampled, ncalib, ncalib)
        if tuple(reference.shape[:3]) != expected_prefix:
            raise ValueError(
                f"Integrated refscan shape {tuple(reference.shape)} disagrees with "
                f"the accepted calibration grid {expected_prefix}."
            )
        if reference.shape[-1] != physical_coils:
            raise ValueError("Refscan physical-coil count disagrees with normal provenance.")
        if reference.shape[3] <= acs_set_index or acs_set_index != 4:
            raise ValueError(
                "ROVir feasibility requires zero-based integrated refscan set 4; "
                f"found {reference.shape[3]} sets and requested {acs_set_index}."
            )
        packed_acs = reference[
            ::readout_oversampling, :nacs, :nacs, acs_set_index, :
        ].contiguous()
        del reference
        expected_shape = (logical_readout, ncalib, ncalib, physical_coils)
        expected_packed_shape = (logical_readout, nacs, nacs, physical_coils)
        if tuple(packed_acs.shape) != expected_packed_shape:
            raise ValueError(
                f"Packed physical ACS shape {tuple(packed_acs.shape)} is not "
                f"the required {expected_packed_shape}."
            )
        if not torch.isfinite(packed_acs).all():
            raise ValueError("Physical set-4 calibration contains nonfinite samples.")
        acquired_nonzero = int(torch.count_nonzero(packed_acs).item())
        if acquired_nonzero == 0:
            raise ValueError("Physical set-4 calibration contains no acquired samples.")
        calibration = torch.zeros(expected_shape, dtype=torch.complex64)
        lin_start = ncalib // 2 - nacs // 2
        par_start = ncalib // 2 - nacs // 2
        calibration[
            :, lin_start : lin_start + nacs, par_start : par_start + nacs, :
        ] = packed_acs
        embedded_nonzero = int(torch.count_nonzero(calibration).item())
        if embedded_nonzero != acquired_nonzero:
            raise ValueError("Center embedding changed the acquired set-4 ACS samples.")
        del packed_acs

        output = create_cfl(staging / "physical_set4_kspace", expected_shape)
        output[...] = calibration.cpu().numpy()
        output.flush()
        del output, calibration
        installed_record = cfl_record(staging / "physical_set4_kspace")
        if destination.exists():
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"Physical-calibration directory became nonempty: {destination}"
                )
            destination.rmdir()
        staging.replace(destination)

        manifest = {
            "format_version": 1,
            "status": "mprage_rovir_physical_calibration_ready",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "refscan_contract": {
                "set_index_zero_based": acs_set_index,
                "set_name": "integrated refscan set 4",
                "readout_oversampling_removed_by_stride": readout_oversampling,
                "calibration_matrix_ro_lin_par": [logical_readout, ncalib, ncalib],
                "central_acs_width": nacs,
                "packed_acs_shape_ro_lin_par_coil": list(expected_packed_shape),
                "center_embedding_start_lin_par": [lin_start, par_start],
                "acquired_nonzero_samples_before_embedding": acquired_nonzero,
                "acquired_nonzero_samples_after_embedding": embedded_nonzero,
                "acquired_sample_equality": True,
                "zero_outside_centered_acs": True,
                "physical_coils": physical_coils,
                "acs_kept_separate_from_wave_image_kspace": True,
            },
            "physical_set4_kspace": _relocate_cfl_record(
                installed_record, staging, destination
            ),
            "bart_launched": False,
            "psf_recalibrated": False,
            "ecalib_launched": False,
            "wave_reconstruction_launched": False,
        }
        _write_json(manifest_path, manifest)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def record_calibration_images(
    feasibility_output_root: str | Path,
    bart_version_file: str | Path,
) -> dict[str, Any]:
    """Validate and record BART IFFT/RSS calibration-image outputs.

    Args:
        feasibility_output_root: Approved diagnostic root containing the
            physical calibration and BART-generated images.
        bart_version_file: Text file containing the exact ``bart version``
            output from the image-generation stage.

    Returns:
        Manifest with geometry, finite-value, hash, and BART provenance checks.

    Raises:
        FileNotFoundError: If an expected CFL pair or version record is absent.
        ValueError: If BART outputs have incompatible geometry or values.

    Side Effects:
        Writes review PNGs and a JSON manifest; image calculation remains in
        the shell.
    """
    root = Path(feasibility_output_root).expanduser().resolve()
    source_manifest = _read_json(root / PHYSICAL_CALIBRATION_MANIFEST)
    physical = root / "inputs" / "physical_calibration" / "physical_set4_kspace"
    images = root / "inputs" / "physical_calibration" / "physical_set4_coil_images"
    rss = root / "inputs" / "physical_calibration" / "physical_set4_rss"
    physical_shape = _active_coil_shape(physical)
    image_shape = _active_coil_shape(images)
    if image_shape != physical_shape:
        raise ValueError(
            f"Calibration-image shape {image_shape} differs from k-space {physical_shape}."
        )
    rss_array = _spatial_cfl_view(rss)
    if rss_array.shape != physical_shape[:3]:
        raise ValueError("Calibration RSS geometry differs from physical-coil images.")
    if not np.isfinite(open_cfl(images)).all() or not np.isfinite(rss_array).all():
        raise ValueError("BART calibration images contain nonfinite samples.")
    if np.any(rss_array.real < 0) or not np.any(rss_array.real > 0):
        raise ValueError("Calibration RSS must be finite, nonnegative, and nonempty.")
    version = _version_record(bart_version_file)
    overview = root / "diagnostics" / "calibration_views" / "physical_set4_rss.png"
    _write_rss_overview(np.asarray(rss_array.real), overview)
    slice_montages = _write_rss_slice_montages(
        np.asarray(rss_array.real), root / "diagnostics" / "calibration_views"
    )
    manifest = {
        "format_version": 1,
        "status": "mprage_rovir_calibration_images_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_calibration_manifest_sha256": sha256_file(
            root / PHYSICAL_CALIBRATION_MANIFEST
        ),
        "physical_set4_kspace": cfl_record(physical),
        "physical_set4_coil_images": cfl_record(images),
        "physical_set4_rss": cfl_record(rss),
        "physical_set4_rss_review_figure": _file_record(overview),
        "physical_set4_rss_slice_montages": [
            _file_record(path) for path in slice_montages
        ],
        "commands": {
            "ifft": "bart fft -iu 7 physical_set4_kspace physical_set4_coil_images",
            "rss": "bart rss 8 physical_set4_coil_images physical_set4_rss",
        },
        "bart": version,
        "source_status": source_manifest["status"],
    }
    _write_json(root / CALIBRATION_IMAGES_MANIFEST, manifest)
    return manifest


def export_manual_roi_annotation_nifti(
    feasibility_output_root: str | Path,
) -> dict[str, Any]:
    """Export calibration RSS and an empty manual-ROI label template as NIfTI.

    Args:
        feasibility_output_root: User-approved diagnostic root containing the
            physical calibration RSS and its immutable manifests.

    Returns:
        Manifest describing the canonical-RAS annotation geometry, reference,
        empty template, and label contract.

    Raises:
        FileExistsError: If an unrecognized manual-annotation directory exists.
        FileNotFoundError: If required calibration inputs or source TWIX are absent.
        ValueError: If provenance, geometry, affine, or RSS values are invalid.

    Side Effects:
        Atomically writes one normalized RSS NIfTI, one empty label NIfTI,
        JSON sidecars, instructions, and a manifest. It does not launch BART.
    """
    import nibabel as nib

    from .mprage import load_wave_mprage_helpers

    root = Path(feasibility_output_root).expanduser().resolve()
    calibration_manifest_path = root / PHYSICAL_CALIBRATION_MANIFEST
    images_manifest_path = root / CALIBRATION_IMAGES_MANIFEST
    calibration_manifest = _read_json(calibration_manifest_path)
    _read_json(images_manifest_path)
    destination = root / "masks" / "manual_annotation"
    manifest_path = root / MANUAL_ANNOTATION_MANIFEST
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("physical_calibration_manifest_sha256") != sha256_file(
            calibration_manifest_path
        ):
            raise FileExistsError(
                "Existing manual-annotation export uses different calibration data."
            )
        if existing.get("calibration_images_manifest_sha256") != sha256_file(
            images_manifest_path
        ):
            raise FileExistsError(
                "Existing manual-annotation export uses different calibration images."
            )
        for key in (
            "reference_nifti",
            "reference_sidecar",
            "label_template_nifti",
            "label_template_sidecar",
            "instructions",
        ):
            _validate_file_record(existing[key])
        return existing
    if destination.exists():
        raise FileExistsError(
            f"Unrecognized manual-annotation directory already exists: {destination}"
        )

    rss_path = root / "inputs" / "physical_calibration" / "physical_set4_rss"
    rss = np.asarray(_spatial_cfl_view(rss_path).real, dtype=np.float32)
    if not np.isfinite(rss).all() or np.any(rss < 0) or not np.any(rss > 0):
        raise ValueError("Calibration RSS must be finite, nonnegative, and nonempty.")
    source = calibration_manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Physical-calibration manifest lacks source provenance.")
    twix_record = source.get("twix")
    geometry = source.get("geometry")
    if not isinstance(twix_record, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("Physical-calibration manifest lacks TWIX or geometry data.")
    twix_path = Path(str(twix_record.get("path", ""))).expanduser().resolve()
    if not twix_path.is_file():
        raise FileNotFoundError(f"Source TWIX is unavailable: {twix_path}")
    voxel_size_logical = _calibration_voxel_size_logical_mm(geometry, rss.shape)

    helpers = load_wave_mprage_helpers()
    source_affine, _, twix_info = helpers.make_nifti_affine_from_twix(
        twix_file=str(twix_path),
        npy_shape=rss.shape,
        twix_array_axis_roles=MPRAGE_NIFTI_AXIS_ROLES,
        twix_array_axis_flips=MPRAGE_NIFTI_AFFINE_AXIS_FLIPS,
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=voxel_size_logical,
    )
    corrected_rss = helpers.apply_array_axis_flips(
        (rss,), MPRAGE_NIFTI_ARRAY_AXIS_FLIPS
    )[0]
    canonical_arrays, canonical_affine, orientation_transform = (
        helpers.canonicalize_arrays_to_ras((corrected_rss,), source_affine)
    )
    canonical_rss = np.asarray(canonical_arrays[0], dtype=np.float32)
    positive = canonical_rss[canonical_rss > 0]
    scale = float(np.percentile(positive, 99.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration RSS annotation scale is invalid.")
    display_rss = canonical_rss / scale

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".manual-annotation-", dir=destination.parent))
    try:
        reference_directory = staging / "reference"
        template_directory = staging / "template"
        reference_directory.mkdir()
        template_directory.mkdir()
        reference_nifti = reference_directory / "physical_set4_rss_ras.nii.gz"
        reference_sidecar = reference_directory / "physical_set4_rss_ras.json"
        template_nifti = template_directory / "rovir_roi_labels_template.nii.gz"
        template_sidecar = template_directory / "rovir_roi_labels_template.json"
        _write_geometry_bound_nifti(
            reference_nifti, display_rss, canonical_affine, label_image=False
        )
        _write_geometry_bound_nifti(
            template_nifti,
            np.zeros(display_rss.shape, dtype=np.uint8),
            canonical_affine,
            label_image=True,
        )
        saved_reference = nib.load(str(reference_nifti))
        saved_template = nib.load(str(template_nifti))
        if (
            saved_reference.shape != saved_template.shape
            or not np.allclose(
                saved_reference.affine,
                saved_template.affine,
                rtol=0.0,
                atol=1e-6,
            )
            or tuple(nib.aff2axcodes(saved_reference.affine)) != ("R", "A", "S")
        ):
            raise ValueError("Manual-annotation NIfTI geometries are inconsistent.")
        geometry_payload = {
            "stored_shape": [int(value) for value in saved_reference.shape],
            "stored_affine": np.asarray(saved_reference.affine).tolist(),
            "stored_axis_codes": ["R", "A", "S"],
            "source_bart_shape_ro_lin_par": [int(value) for value in rss.shape],
            "source_affine_before_ras": np.asarray(source_affine).tolist(),
            "logical_voxel_size_mm_ro_lin_par": list(voxel_size_logical),
            "physical_array_flips_before_ras": list(MPRAGE_NIFTI_ARRAY_AXIS_FLIPS),
            "affine_axis_flips": list(MPRAGE_NIFTI_AFFINE_AXIS_FLIPS),
            "twix_array_axis_roles": list(MPRAGE_NIFTI_AXIS_ROLES),
            "orientation_transform_to_ras": orientation_transform,
            "canonicalization_used_resampling": False,
        }
        _write_json(
            reference_sidecar,
            {
                "ImageRole": "ROVir manual-annotation reference",
                "Units": "relative",
                "MagnitudeNormalization": {
                    "method": "positive-voxel percentile scaling without clipping",
                    "percentile": 99.0,
                    "input_percentile_value": scale,
                },
                "Geometry": geometry_payload,
                "TwixOrientation": twix_info,
            },
        )
        _write_json(
            template_sidecar,
            {
                "ImageRole": "ROVir manual sparse-seed label template",
                "LabelDefinitions": {
                    str(key): value for key, value in MANUAL_ROI_LABELS.items()
                },
                "Instructions": (
                    "Paint only unequivocal clean desired-signal voxels as 1 and "
                    "pure shoulder-interference voxels as 2. Leave mixed or uncertain "
                    "voxels at 0. Do not resample, crop, or change the affine."
                ),
                "Geometry": geometry_payload,
            },
        )
        instructions = staging / "README.txt"
        instructions.write_text(
            "ROVir manual sparse-seed annotation\n\n"
            "Open reference/physical_set4_rss_ras.nii.gz and "
            "template/rovir_roi_labels_template.nii.gz together.\n"
            "Label 0: unassigned, mixed, or uncertain.\n"
            "Label 1: unequivocal clean desired signal (head, brain, scalp, face).\n"
            "Label 2: unequivocal pure shoulder interference.\n"
            "Sparse pure regions are preferred over complete anatomical segmentation.\n"
            "Do not resample, crop, smooth, reorient, or change the affine.\n"
            "Save the completed label map as "
            "reviewed/rovir_roi_labels_reviewed.nii.gz.\n",
            encoding="utf-8",
        )
        final_root = destination
        manifest = {
            "format_version": 1,
            "status": "mprage_rovir_manual_annotation_export_ready",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "physical_calibration_manifest_sha256": sha256_file(
                calibration_manifest_path
            ),
            "calibration_images_manifest_sha256": sha256_file(images_manifest_path),
            "source_rss": cfl_record(rss_path),
            "geometry": geometry_payload,
            "label_definitions": {
                str(key): value for key, value in MANUAL_ROI_LABELS.items()
            },
            "reference_nifti": _relocate_file_record(
                _file_record(reference_nifti), staging, final_root
            ),
            "reference_sidecar": _relocate_file_record(
                _file_record(reference_sidecar), staging, final_root
            ),
            "label_template_nifti": _relocate_file_record(
                _file_record(template_nifti), staging, final_root
            ),
            "label_template_sidecar": _relocate_file_record(
                _file_record(template_sidecar), staging, final_root
            ),
            "instructions": _relocate_file_record(
                _file_record(instructions), staging, final_root
            ),
            "reviewed_label_destination": str(
                final_root / "reviewed" / "rovir_roi_labels_reviewed.nii.gz"
            ),
            "automatic_roi_detection": False,
            "bart_launched": False,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def validate_manual_roi_annotation(
    feasibility_output_root: str | Path,
    reviewed_labels: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a reviewed manual label map and invert it to BART geometry.

    Args:
        feasibility_output_root: User-approved diagnostic root containing the
            immutable manual-annotation export.
        reviewed_labels: Optional reviewed NIfTI path; ``None`` uses the
            documented destination below the manual-annotation directory.

    Returns:
        JSON-compatible geometry, label-count, hash, and round-trip report.
        No BART masks are written or approved by this validation function.

    Raises:
        FileNotFoundError: If the export or reviewed label map is absent.
        ValueError: If immutable files, affine, shape, values, labels, or the
            inverse orientation mapping are incompatible.
    """
    import nibabel as nib

    root = Path(feasibility_output_root).expanduser().resolve()
    manifest = _read_json(root / MANUAL_ANNOTATION_MANIFEST)
    for key in (
        "reference_nifti",
        "reference_sidecar",
        "label_template_nifti",
        "label_template_sidecar",
        "instructions",
    ):
        _validate_file_record(manifest[key])
    labels_path = (
        Path(reviewed_labels).expanduser().resolve()
        if reviewed_labels is not None
        else root
        / "masks"
        / "manual_annotation"
        / "reviewed"
        / "rovir_roi_labels_reviewed.nii.gz"
    )
    if not labels_path.is_file():
        raise FileNotFoundError(f"Reviewed manual ROI label map is absent: {labels_path}")
    image = nib.load(str(labels_path))
    values = np.asanyarray(image.dataobj)
    geometry = manifest["geometry"]
    expected_shape = tuple(int(value) for value in geometry["stored_shape"])
    expected_affine = np.asarray(geometry["stored_affine"], dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(
            f"Reviewed label shape {values.shape} differs from {expected_shape}."
        )
    if not np.isfinite(image.affine).all() or not np.allclose(
        image.affine, expected_affine, rtol=0.0, atol=1e-5
    ):
        raise ValueError("Reviewed label affine differs from the exported template.")
    if not np.isfinite(values).all() or not np.allclose(
        values, np.rint(values), rtol=0.0, atol=0.0
    ):
        raise ValueError("Reviewed ROI labels must be finite integer values.")
    labels = np.rint(values).astype(np.uint8)
    observed = set(int(value) for value in np.unique(labels))
    allowed = set(MANUAL_ROI_LABELS)
    if not observed.issubset(allowed):
        raise ValueError(f"Reviewed ROI contains unsupported labels: {sorted(observed)}")
    counts = {str(label): int(np.count_nonzero(labels == label)) for label in allowed}
    if counts["1"] == 0 or counts["2"] == 0:
        raise ValueError("Reviewed ROI requires nonempty label-1 and label-2 seeds.")

    source_affine = np.asarray(
        geometry["source_affine_before_ras"], dtype=np.float64
    )
    source_orientation = nib.orientations.io_orientation(source_affine)
    ras_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    reverse = nib.orientations.ornt_transform(ras_orientation, source_orientation)
    corrected = np.ascontiguousarray(
        nib.orientations.apply_orientation(labels, reverse)
    )
    bart_labels = corrected
    for axis, should_flip in enumerate(
        geometry["physical_array_flips_before_ras"]
    ):
        if bool(should_flip):
            bart_labels = np.flip(bart_labels, axis=axis)
    bart_labels = np.ascontiguousarray(bart_labels)
    expected_bart_shape = tuple(
        int(value) for value in geometry["source_bart_shape_ro_lin_par"]
    )
    if bart_labels.shape != expected_bart_shape:
        raise ValueError(
            "Reviewed ROI inverse orientation produced incompatible BART geometry: "
            f"{bart_labels.shape} versus {expected_bart_shape}."
        )
    return {
        "status": "mprage_rovir_manual_annotation_validated_not_approved",
        "reviewed_labels": _file_record(labels_path),
        "stored_shape": list(expected_shape),
        "stored_affine": expected_affine.tolist(),
        "stored_axis_codes": list(nib.aff2axcodes(image.affine)),
        "label_definitions": manifest["label_definitions"],
        "label_counts": counts,
        "source_bart_shape_ro_lin_par": list(expected_bart_shape),
        "inverse_orientation_used_resampling": False,
        "positive_and_negative_disjoint": True,
        "automatic_selection": False,
        "approved": False,
        "bart_launched": False,
    }


def derive_region_mask_candidates(
    feasibility_output_root: str | Path,
    candidate_config: str | Path,
) -> dict[str, Any]:
    """Create non-ranking four-region ROI candidates and review overlays.

    Args:
        feasibility_output_root: Approved diagnostic root containing RSS data.
        candidate_config: Reviewed JSON describing normalized ellipsoidal ROIs.

    Returns:
        Candidate manifest; no candidate is selected or ranked.

    Raises:
        FileExistsError: If candidates already exist.
        ValueError: If configuration, masks, safety gap, or RSS is invalid.

    Side Effects:
        Atomically writes BART masks, review PNGs, and a candidate manifest.
    """
    root = Path(feasibility_output_root).expanduser().resolve()
    _read_json(root / CALIBRATION_IMAGES_MANIFEST)
    config_path = Path(candidate_config).expanduser().resolve()
    config = _read_json(config_path)
    if config.get("scientific_status") != "ready_for_visual_review":
        raise ValueError(
            "Mask config scientific_status must be 'ready_for_visual_review'; "
            "the tracked template is intentionally not executable."
        )
    records = config.get("candidates")
    if not isinstance(records, list) or not records:
        raise ValueError("Mask config must contain a nonempty candidates list.")
    rss_path = root / "inputs" / "physical_calibration" / "physical_set4_rss"
    rss = np.asarray(_spatial_cfl_view(rss_path).real, dtype=np.float64)
    destination = root / "masks" / "candidates"
    if destination.exists():
        raise FileExistsError(f"ROVir mask candidates already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".candidates-", dir=destination.parent))
    try:
        candidate_records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in records:
            if not isinstance(raw, Mapping):
                raise ValueError("Each mask candidate must be a JSON object.")
            candidate_id = str(raw.get("candidate_id", ""))
            if not candidate_id or not candidate_id.replace("_", "").isalnum():
                raise ValueError(f"Invalid candidate_id: {candidate_id!r}.")
            if candidate_id in seen:
                raise ValueError(f"Duplicate mask candidate_id: {candidate_id}.")
            seen.add(candidate_id)
            preservation = _region_from_spec(
                rss, raw.get("preservation"), "preservation"
            )
            positive = _region_from_spec(
                rss, raw.get("positive_estimation"), "positive estimation"
            )
            negative = _region_from_spec(
                rss, raw.get("negative_estimation"), "negative estimation"
            )
            holdout = _region_from_spec(
                rss, raw.get("contaminated_holdout"), "contaminated holdout"
            )
            validation = _validate_four_region_masks(
                preservation,
                positive,
                negative,
                holdout,
            )
            required_gap = int(raw.get("minimum_safety_gap_voxels", 2))
            measured_gap = _minimum_mask_distance(positive, negative)
            if required_gap < 1 or measured_gap < required_gap:
                raise ValueError(
                    f"Candidate {candidate_id} positive/negative gap {measured_gap:g} "
                    f"is below {required_gap} voxels."
                )
            candidate_directory = staging / candidate_id
            candidate_directory.mkdir()
            masks = {
                "preservation_mask": preservation,
                "positive_estimation_mask": positive,
                "negative_estimation_mask": negative,
                "contaminated_holdout_mask": holdout,
            }
            for name, mask in masks.items():
                _write_real_cfl(candidate_directory / name, mask)
            overlay_path = candidate_directory / "review_overlay.png"
            _write_region_overlay(
                rss,
                preservation,
                positive,
                negative,
                holdout,
                overlay_path,
                candidate_id,
            )
            overlay_montages = _write_region_slice_montages(
                rss,
                preservation,
                positive,
                negative,
                holdout,
                candidate_directory,
                candidate_id,
            )
            candidate_record: dict[str, Any] = {
                "candidate_id": candidate_id,
                "status": "ready_for_visual_review",
                "parameters": dict(raw),
                "validation": validation,
                "minimum_positive_to_negative_distance_voxels": measured_gap,
                "review_overlay": _relocate_file_record(
                    _file_record(overlay_path),
                    candidate_directory,
                    destination / candidate_id,
                ),
                "review_slice_montages": [
                    _relocate_file_record(
                        _file_record(path),
                        candidate_directory,
                        destination / candidate_id,
                    )
                    for path in overlay_montages
                ],
            }
            for name in REGION_MASK_NAMES:
                candidate_record[name] = _relocate_cfl_record(
                    cfl_record(candidate_directory / name),
                    candidate_directory,
                    destination / candidate_id,
                )
            candidate_records.append(candidate_record)
        manifest = {
            "format_version": 1,
            "status": "mprage_rovir_mask_candidates_ready",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_images_manifest_sha256": sha256_file(
                root / CALIBRATION_IMAGES_MANIFEST
            ),
            "candidate_config": _file_record(config_path),
            "rss": cfl_record(rss_path),
            "automatic_ranking": False,
            "automatic_selection": False,
            "selection_status": "not_selected",
            "candidates": candidate_records,
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "REVIEW_INSTRUCTIONS.txt").write_text(
            "Review every candidate review_overlay.png and all three indexed "
            "review_slices_*.png montages.\n"
            "Yellow is the desired whole-head preservation region, blue is the clean "
            "positive estimation subset, red is pure external shoulder interference, "
            "and magenta is the contaminated in-head holdout.\n"
            "Confirm that red remains outside yellow, and neither blue nor red "
            "includes the ambiguous magenta region.\n"
            "No candidate is ranked or selected automatically.\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def approve_region_mask_candidate(
    feasibility_output_root: str | Path,
    candidate_id: str,
) -> dict[str, Any]:
    """Install one explicitly selected candidate as immutable approved masks.

    Args:
        feasibility_output_root: Approved diagnostic root.
        candidate_id: Exact visually reviewed candidate identifier.

    Returns:
        Approval manifest binding copied masks to the candidate manifest.

    Raises:
        FileExistsError: If an approved-mask directory already exists.
        ValueError: If the candidate is absent or hashes fail to match.

    Side Effects:
        Copies four reviewed masks into a new approved directory. It never
        chooses the candidate itself.
    """
    root = Path(feasibility_output_root).expanduser().resolve()
    candidates_path = root / MASK_CANDIDATES_MANIFEST
    candidates = _read_json(candidates_path)
    matches = [
        record
        for record in candidates.get("candidates", [])
        if record.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one reviewed mask candidate {candidate_id!r}.")
    selected = matches[0]
    approved = root / "masks" / "approved"
    if approved.exists():
        raise FileExistsError(f"Approved ROVir masks already exist: {approved}")
    source = root / "masks" / "candidates" / candidate_id
    staging = Path(tempfile.mkdtemp(prefix=".approved-", dir=approved.parent))
    try:
        for name in REGION_MASK_NAMES:
            _validate_cfl_against_record(source / name, selected.get(name))
            for suffix in (".hdr", ".cfl"):
                shutil.copy2(source / f"{name}{suffix}", staging / f"{name}{suffix}")
        preservation = _spatial_cfl_view(staging / "preservation_mask").real
        positive = _spatial_cfl_view(staging / "positive_estimation_mask").real
        negative = _spatial_cfl_view(staging / "negative_estimation_mask").real
        holdout = _spatial_cfl_view(staging / "contaminated_holdout_mask").real
        validation = _validate_four_region_masks(
            preservation, positive, negative, holdout
        )
        manifest = {
            "format_version": 1,
            "status": "mprage_rovir_masks_approved",
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "approval_method": "explicit candidate_id supplied by user",
            "candidate_id": candidate_id,
            "candidate_manifest_sha256": sha256_file(candidates_path),
            "validation": validation,
        }
        for name in REGION_MASK_NAMES:
            manifest[name] = _relocate_cfl_record(
                cfl_record(staging / name), staging, approved
            )
        _write_json(staging / "manifest.json", manifest)
        staging.replace(approved)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def prepare_masked_rovir_inputs(
    feasibility_output_root: str | Path,
    *,
    readout_chunk: int = 8,
) -> dict[str, Any]:
    """Apply approved masks to physical-coil images in bounded memory.

    Args:
        feasibility_output_root: Approved diagnostic root.
        readout_chunk: Maximum leading-axis planes processed at once.

    Returns:
        Manifest describing positive and negative BART ROVir inputs.

    Raises:
        FileExistsError: If incompatible masked inputs already exist.
        ValueError: If geometry, masks, samples, or conditioning are invalid.

    Side Effects:
        Writes two masked physical-coil CFL pairs. It does not launch BART.
    """
    if readout_chunk < 1:
        raise ValueError("ROVir input readout chunk must be positive.")
    root = Path(feasibility_output_root).expanduser().resolve()
    approved_manifest_path = root / APPROVED_MASK_MANIFEST
    _read_json(approved_manifest_path)
    images_path = root / "inputs" / "physical_calibration" / "physical_set4_coil_images"
    images = _coil_cfl_view(images_path)
    positive_mask = np.asarray(
        _spatial_cfl_view(
            root / "masks" / "approved" / "positive_estimation_mask"
        ).real
    )
    negative_mask = np.asarray(
        _spatial_cfl_view(
            root / "masks" / "approved" / "negative_estimation_mask"
        ).real
    )
    preservation = np.asarray(
        _spatial_cfl_view(root / "masks" / "approved" / "preservation_mask").real
    )
    holdout = np.asarray(
        _spatial_cfl_view(
            root / "masks" / "approved" / "contaminated_holdout_mask"
        ).real
    )
    four_region_validation = _validate_four_region_masks(
        preservation, positive_mask, negative_mask, holdout
    )
    correlation = region_correlation_diagnostics(
        images,
        positive_mask,
        negative_mask,
        coil_axis=3,
        voxel_chunk=65536,
    )
    destination = root / "inputs" / "rovir"
    manifest_path = root / ROVIR_INPUT_MANIFEST
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("approved_mask_manifest_sha256") != sha256_file(
            approved_manifest_path
        ):
            raise FileExistsError("Existing ROVir inputs use different approved masks.")
        if existing.get("calibration_images_manifest_sha256") != sha256_file(
            root / CALIBRATION_IMAGES_MANIFEST
        ):
            raise FileExistsError("Existing ROVir inputs use different calibration images.")
        read_shape(destination / "positive_signal_images")
        read_shape(destination / "negative_interference_images")
        return existing
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"ROVir input directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    positive_output = create_cfl(
        destination / "positive_signal_images", images.shape
    )
    negative_output = create_cfl(
        destination / "negative_interference_images", images.shape
    )
    for start in range(0, images.shape[0], readout_chunk):
        stop = min(start + readout_chunk, images.shape[0])
        block = images[start:stop]
        positive_output[start:stop] = block * positive_mask[
            start:stop, ..., np.newaxis
        ]
        negative_output[start:stop] = block * negative_mask[
            start:stop, ..., np.newaxis
        ]
    positive_output.flush()
    negative_output.flush()
    del positive_output, negative_output
    manifest = {
        "format_version": 1,
        "status": "mprage_bart_rovir_inputs_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved_mask_manifest_sha256": sha256_file(approved_manifest_path),
        "calibration_images_manifest_sha256": sha256_file(
            root / CALIBRATION_IMAGES_MANIFEST
        ),
        "physical_coil_images": cfl_record(images_path),
        "positive_signal_images": cfl_record(destination / "positive_signal_images"),
        "negative_interference_images": cfl_record(
            destination / "negative_interference_images"
        ),
        "correlation_diagnostics": correlation,
        "four_region_validation": four_region_validation,
        "solver_backend": "bart rovir only",
        "bart_launched": False,
    }
    _write_json(manifest_path, manifest)
    return manifest


def write_rovir_transform_qc(
    feasibility_output_root: str | Path,
    bart_version_file: str | Path,
) -> dict[str, Any]:
    """Validate a BART ROVir transform and write its complete region curve.

    Args:
        feasibility_output_root: Approved diagnostic root containing the BART
            transform and its immutable inputs.
        bart_version_file: Exact BART version record from transform estimation.

    Returns:
        QC manifest containing transform and all-channel region metrics.

    Raises:
        FileNotFoundError: If required inputs or transform are absent.
        ValueError: If transform, geometry, conditioning, or energies are invalid.

    Side Effects:
        Writes JSON, CSV, and PNG QC artifacts without selecting a coil count.
    """
    root = Path(feasibility_output_root).expanduser().resolve()
    input_manifest_path = root / ROVIR_INPUT_MANIFEST
    _read_json(input_manifest_path)
    transform_path = root / "transforms" / "rovir_full" / "transform"
    transform = np.asarray(open_cfl(transform_path))
    validation = validate_rovir_transform(transform, orthogonality_tolerance=1e-4)
    images = _coil_cfl_view(
        root / "inputs" / "physical_calibration" / "physical_set4_coil_images"
    )
    mask_root = root / "masks" / "approved"
    preservation = _spatial_cfl_view(mask_root / "preservation_mask").real
    positive = _spatial_cfl_view(mask_root / "positive_estimation_mask").real
    negative = _spatial_cfl_view(mask_root / "negative_estimation_mask").real
    holdout = _spatial_cfl_view(mask_root / "contaminated_holdout_mask").real
    _validate_four_region_masks(preservation, positive, negative, holdout)
    counts = tuple(range(1, int(validation["virtual_coils_available"]) + 1))
    solver_curve = region_energy_curve(
        images,
        transform,
        positive,
        negative,
        counts,
        coil_axis=3,
        voxel_chunk=65536,
    )
    preservation_curve = region_energy_curve(
        images,
        transform,
        preservation,
        negative,
        counts,
        coil_axis=3,
        voxel_chunk=65536,
    )
    holdout_curve = region_energy_curve(
        images,
        transform,
        holdout,
        negative,
        counts,
        coil_axis=3,
        voxel_chunk=65536,
    )
    diagnostics = root / "diagnostics" / "region_curves"
    diagnostics.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics / "rovir_four_region_curves.csv"
    _write_four_region_curve_csv(
        csv_path,
        solver_curve["channel_counts"],
        preservation_curve["channel_counts"],
        holdout_curve["channel_counts"],
    )
    plot_path = diagnostics / "rovir_four_region_curves.png"
    _write_four_region_curve_plot(
        plot_path,
        solver_curve["channel_counts"],
        preservation_curve["channel_counts"],
        holdout_curve["channel_counts"],
    )
    manifest = {
        "format_version": 1,
        "status": "mprage_bart_rovir_transform_qc_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rovir_input_manifest_sha256": sha256_file(input_manifest_path),
        "solver_backend": "bart rovir only",
        "command": "bart rovir positive_signal_images negative_interference_images transform",
        "bart": _version_record(bart_version_file),
        "transform": cfl_record(transform_path),
        "transform_validation": validation,
        "region_curves": {
            "solver_clean_positive_vs_pure_negative": solver_curve,
            "whole_head_preservation_vs_pure_negative": preservation_curve,
            "contaminated_holdout_vs_pure_negative": holdout_curve,
            "holdout_energy_is_not_anatomy_specific": True,
        },
        "region_curve_csv": _file_record(csv_path),
        "region_curve_plot": _file_record(plot_path),
        "automatic_ranking": False,
        "automatic_selection": False,
        "selected_virtual_coils": None,
        "ecalib_launched": False,
        "wave_reconstruction_launched": False,
    }
    _write_json(root / ROVIR_QC_MANIFEST, manifest)
    return manifest


def _validated_source_contract(
    twix: str | Path,
    sequence: str | Path,
    normal_output_root: str | Path,
    *,
    include_twix_hash: bool,
) -> dict[str, Any]:
    """Bind supplied sources to the accepted normal-input manifest.

    Args:
        twix: Candidate TWIX path.
        sequence: Candidate sequence path.
        normal_output_root: Existing normal reconstruction root.
        include_twix_hash: Whether to hash the large TWIX payload for an
            immutable production export record.

    Returns:
        Source and acquisition fields required by feasibility preparation.

    Raises:
        FileNotFoundError: If a source or manifest is absent.
        ValueError: If source identity or required provenance is incompatible.
    """
    twix_path = Path(twix).expanduser().resolve()
    sequence_path = Path(sequence).expanduser().resolve()
    normal_root = Path(normal_output_root).expanduser().resolve()
    manifest_path = normal_root / "normal" / "bart_inputs" / "manifest.json"
    manifest = _read_json(manifest_path)
    for path in (twix_path, sequence_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Normal manifest has no source provenance.")
    _require_same_file(twix_path, source.get("twix"), "TWIX")
    _require_same_file(sequence_path, source.get("sequence"), "sequence")
    expected_sequence_hash = source["sequence"].get("sha256")
    if expected_sequence_hash != sha256_file(sequence_path):
        raise ValueError("Sequence hash differs from the accepted normal manifest.")
    geometry = manifest.get("geometry")
    compression = manifest.get("coil_compression")
    psf = manifest.get("psf_calibration")
    if not all(isinstance(value, Mapping) for value in (geometry, compression, psf)):
        raise ValueError("Normal manifest lacks geometry, compression, or PSF provenance.")
    matrix = [int(value) for value in geometry["logical_matrix_ro_lin_par"]]
    oversampling = int(geometry["readout_oversampling_factor"])
    if len(matrix) != 3 or min(matrix) < 1 or oversampling < 1:
        raise ValueError("Normal manifest contains invalid MPRAGE geometry.")
    return {
        "normal_manifest": _file_record(manifest_path),
        "twix": _file_identity(twix_path, include_hash=include_twix_hash),
        "sequence": _file_identity(sequence_path, include_hash=True),
        "geometry": {
            **dict(geometry),
            "readout_oversampled": matrix[0] * oversampling,
        },
        "coil_compression": dict(compression),
        "psf_calibration": {
            "ncalib": int(psf["ncalib"]),
            "nacs": int(psf["nacs"]),
            "reused_without_recalibration": True,
        },
    }


def _require_same_file(path: Path, record: object, label: str) -> None:
    """Require a supplied file to match one manifest identity record.

    Args:
        path: Existing supplied file.
        record: Manifest record with path, size, and modification time.
        label: Human-readable source label.

    Raises:
        ValueError: If identity metadata or filesystem identity differs.
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise ValueError(f"Normal manifest has no valid {label} source record.")
    recorded = Path(str(record["path"])).expanduser()
    try:
        same = recorded.is_file() and os.path.samefile(path, recorded)
    except OSError:
        same = path == recorded.resolve()
    stat = path.stat()
    if (
        not same
        or int(record.get("size_bytes", -1)) != stat.st_size
        or int(record.get("mtime_ns", -1)) != stat.st_mtime_ns
    ):
        raise ValueError(f"Supplied {label} differs from the accepted normal manifest.")


def _validate_export_reuse(
    manifest: Mapping[str, Any], source: Mapping[str, Any], root: Path
) -> None:
    """Validate an existing physical-calibration export for exact reuse.

    Args:
        manifest: Existing export manifest.
        source: Current validated source contract.
        root: Feasibility output root.

    Raises:
        ValueError: If provenance or payload records do not match.
    """
    if manifest.get("source") != source:
        raise ValueError("Existing physical calibration uses different sources.")
    current = cfl_record(
        root / "inputs" / "physical_calibration" / "physical_set4_kspace"
    )
    recorded = manifest.get("physical_set4_kspace")
    for key in ("shape", "payload_bytes", "header_sha256", "payload_sha256"):
        if not isinstance(recorded, Mapping) or recorded.get(key) != current.get(key):
            raise ValueError("Existing physical calibration failed exact hash reuse.")


def _validate_cfl_against_record(path: Path, record: object) -> None:
    """Require one CFL pair to match its manifest dimensions and hashes.

    Args:
        path: Existing BART basename.
        record: Candidate manifest CFL record.

    Raises:
        ValueError: If the record is absent or the current CFL pair differs.
    """
    current = cfl_record(path)
    if not isinstance(record, Mapping):
        raise ValueError(f"Missing CFL provenance record for {path}.")
    for key in ("shape", "payload_bytes", "header_sha256", "payload_sha256"):
        if record.get(key) != current.get(key):
            raise ValueError(f"CFL pair differs from candidate manifest: {path}.")


def _region_from_spec(rss: np.ndarray, spec: object, label: str) -> np.ndarray:
    """Rasterize one union of normalized ellipsoids with optional RSS floor.

    Args:
        rss: Finite nonnegative three-dimensional calibration RSS.
        spec: JSON mapping containing ellipsoids and relative RSS threshold.
        label: Region name used in validation errors.

    Returns:
        Float32 binary region mask.

    Raises:
        ValueError: If region parameters are absent or invalid.
    """
    if not isinstance(spec, Mapping):
        raise ValueError(f"Candidate {label} region must be a JSON object.")
    ellipsoids = spec.get("ellipsoids")
    if not isinstance(ellipsoids, list) or not ellipsoids:
        raise ValueError(f"Candidate {label} requires at least one ellipsoid.")
    axes = [np.linspace(-1.0, 1.0, size, dtype=np.float64) for size in rss.shape]
    coordinates = np.meshgrid(*axes, indexing="ij", sparse=True)
    mask = np.zeros(rss.shape, dtype=bool)
    for ellipsoid in ellipsoids:
        if not isinstance(ellipsoid, Mapping):
            raise ValueError(f"Candidate {label} ellipsoid must be a JSON object.")
        center = np.asarray(ellipsoid.get("center_normalized"), dtype=np.float64)
        radii = np.asarray(ellipsoid.get("radii_normalized"), dtype=np.float64)
        if (
            center.shape != (3,)
            or radii.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(radii).all()
            or np.any(np.abs(center) > 1)
            or np.any(radii <= 0)
            or np.any(radii > 2)
        ):
            raise ValueError(
                f"Candidate {label} ellipsoid center/radii must be finite "
                "three-vectors in normalized RO/LIN/PAR coordinates."
            )
        distance = sum(
            ((coordinate - center[index]) / radii[index]) ** 2
            for index, coordinate in enumerate(coordinates)
        )
        mask |= distance <= 1.0
    relative_floor = float(spec.get("minimum_relative_rss", 0.0))
    if not np.isfinite(relative_floor) or relative_floor < 0 or relative_floor >= 1:
        raise ValueError(f"Candidate {label} minimum_relative_rss must lie in [0, 1).")
    positive = rss[rss > 0]
    if positive.size == 0:
        raise ValueError("Calibration RSS contains no positive samples.")
    scale = float(np.percentile(positive, 99.0))
    if relative_floor > 0:
        mask &= rss >= relative_floor * scale
    if not np.any(mask):
        raise ValueError(f"Candidate {label} region is empty after RSS thresholding.")
    return mask.astype(np.float32)


def _minimum_mask_distance(signal: np.ndarray, interference: np.ndarray) -> float:
    """Measure the nearest voxel-center distance between two binary masks.

    Args:
        signal: Positive signal-region mask.
        interference: Positive interference-region mask.

    Returns:
        Minimum Euclidean distance in voxel units.
    """
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~(np.asarray(signal) > 0))
    return float(np.min(distance[np.asarray(interference) > 0]))


def _validate_four_region_masks(
    preservation: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    holdout: np.ndarray,
) -> dict[str, Any]:
    """Validate estimation masks separately from preservation and holdout ROIs.

    Args:
        preservation: Desired whole-head preservation region.
        positive: Clean head subset used in the ROVir numerator.
        negative: Pure external shoulder region used in the denominator.
        holdout: Contaminated in-head region excluded from estimation.

    Returns:
        JSON-compatible support and overlap diagnostics.

    Raises:
        ValueError: If masks are nonbinary, geometrically incompatible, empty,
            or violate subset and disjointness requirements.
    """
    masks = {
        "preservation": np.asarray(preservation),
        "positive_estimation": np.asarray(positive),
        "negative_estimation": np.asarray(negative),
        "contaminated_holdout": np.asarray(holdout),
    }
    shape = masks["preservation"].shape
    if len(shape) != 3:
        raise ValueError(f"ROVir region masks must be three-dimensional: {shape}.")
    supports: dict[str, np.ndarray] = {}
    for name, values in masks.items():
        if values.shape != shape or np.iscomplexobj(values):
            raise ValueError(f"ROVir {name} mask geometry is incompatible.")
        if not np.isfinite(values).all() or np.any((values != 0) & (values != 1)):
            raise ValueError(f"ROVir {name} mask must be finite and binary.")
        support = values > 0
        if not np.any(support):
            raise ValueError(f"ROVir {name} mask is empty.")
        supports[name] = support
    if np.any(supports["positive_estimation"] & ~supports["preservation"]):
        raise ValueError("Positive estimation must be contained in preservation.")
    if np.any(supports["contaminated_holdout"] & ~supports["preservation"]):
        raise ValueError("Contaminated holdout must be contained in preservation.")
    if np.any(supports["positive_estimation"] & supports["contaminated_holdout"]):
        raise ValueError(
            "Positive estimation and contaminated holdout must be disjoint."
        )
    if np.any(supports["negative_estimation"] & supports["preservation"]):
        raise ValueError(
            "Pure shoulder negative estimation must remain outside preservation."
        )
    clean_preservation = supports["preservation"] & ~supports["contaminated_holdout"]
    if not np.any(clean_preservation):
        raise ValueError(
            "Contaminated holdout cannot consume the whole preservation ROI."
        )
    estimation = validate_region_masks(positive, negative, shape)
    return {
        "spatial_shape": list(shape),
        "preservation_voxels": int(np.count_nonzero(supports["preservation"])),
        "clean_preservation_voxels": int(np.count_nonzero(clean_preservation)),
        "positive_estimation_voxels": int(
            np.count_nonzero(supports["positive_estimation"])
        ),
        "negative_estimation_voxels": int(
            np.count_nonzero(supports["negative_estimation"])
        ),
        "contaminated_holdout_voxels": int(
            np.count_nonzero(supports["contaminated_holdout"])
        ),
        "positive_and_negative_disjoint": True,
        "negative_outside_preservation": True,
        "holdout_excluded_from_estimation": True,
        "estimation_mask_validation": estimation,
    }


def _write_region_overlay(
    rss: np.ndarray,
    preservation: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    holdout: np.ndarray,
    output: Path,
    candidate_id: str,
) -> None:
    """Write center-slice and maximum-projection mask review panels.

    Args:
        rss: Calibration root-sum-of-squares magnitude.
        preservation: Desired whole-head preservation region.
        positive: Clean positive estimation subset.
        negative: Pure shoulder negative estimation region.
        holdout: Contaminated in-head region excluded from estimation.
        output: Destination PNG path.
        candidate_id: Candidate label shown in the title.

    Side Effects:
        Writes one PNG using a fixed intensity window across all panels.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    display_maximum = float(np.percentile(rss[rss > 0], 99.5))
    figure = Figure(figsize=(12, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    preservation_bool = preservation > 0
    positive_bool = positive > 0
    negative_bool = negative > 0
    holdout_bool = holdout > 0
    centers = tuple(
        int(round(value)) for value in np.argwhere(preservation_bool).mean(axis=0)
    )
    axis_names = ("RO", "LIN", "PAR")
    for axis in range(3):
        image_slice = np.take(rss, centers[axis], axis=axis)
        preservation_slice = np.take(preservation_bool, centers[axis], axis=axis)
        positive_slice = np.take(positive_bool, centers[axis], axis=axis)
        negative_slice = np.take(negative_bool, centers[axis], axis=axis)
        holdout_slice = np.take(holdout_bool, centers[axis], axis=axis)
        _draw_mask_panel(
            figure.add_subplot(2, 3, axis + 1),
            image_slice,
            preservation_slice,
            positive_slice,
            negative_slice,
            holdout_slice,
            display_maximum,
            f"{axis_names[axis]} center {centers[axis]}",
        )
        _draw_mask_panel(
            figure.add_subplot(2, 3, axis + 4),
            np.max(rss, axis=axis),
            np.any(preservation_bool, axis=axis),
            np.any(positive_bool, axis=axis),
            np.any(negative_bool, axis=axis),
            np.any(holdout_bool, axis=axis),
            display_maximum,
            f"{axis_names[axis]} maximum projection",
        )
    figure.suptitle(
        f"ROVir {candidate_id}: yellow=preserve, blue=positive, "
        "red=shoulder, magenta=holdout"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)


def _write_rss_overview(rss: np.ndarray, output: Path) -> None:
    """Write center-slice and maximum-projection calibration RSS panels.

    Args:
        rss: Finite nonnegative calibration RSS in RO/LIN/PAR order.
        output: Destination PNG path.

    Side Effects:
        Writes a six-panel fixed-window review figure.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    display_maximum = float(np.percentile(rss[rss > 0], 99.5))
    centers = tuple(size // 2 for size in rss.shape)
    axis_names = ("RO", "LIN", "PAR")
    figure = Figure(figsize=(12, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    for axis_index in range(3):
        center_axis = figure.add_subplot(2, 3, axis_index + 1)
        center_axis.imshow(
            np.rot90(np.take(rss, centers[axis_index], axis=axis_index)),
            cmap="gray",
            vmin=0,
            vmax=display_maximum,
        )
        center_axis.set_title(f"{axis_names[axis_index]} center {centers[axis_index]}")
        center_axis.axis("off")
        projection_axis = figure.add_subplot(2, 3, axis_index + 4)
        projection_axis.imshow(
            np.rot90(np.max(rss, axis=axis_index)),
            cmap="gray",
            vmin=0,
            vmax=display_maximum,
        )
        projection_axis.set_title(f"{axis_names[axis_index]} maximum projection")
        projection_axis.axis("off")
    figure.suptitle("Physical-coil set-4 calibration RSS; shared fixed window")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)


def _write_rss_slice_montages(rss: np.ndarray, output_directory: Path) -> list[Path]:
    """Write indexed slice montages along every logical calibration axis.

    Args:
        rss: Finite nonnegative calibration RSS in RO/LIN/PAR order.
        output_directory: Destination directory for three montage PNGs.

    Returns:
        Paths to RO-, LIN-, and PAR-slice montage figures.

    Side Effects:
        Writes three fixed-window PNGs without changing the source CFL.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    display_maximum = float(np.percentile(rss[rss > 0], 99.5))
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for axis_index, axis_name in enumerate(("RO", "LIN", "PAR")):
        grid = 5 if axis_index == 0 else 4
        indices = np.linspace(0, rss.shape[axis_index] - 1, grid * grid, dtype=int)
        figure = Figure(
            figsize=(3.0 * grid, 3.0 * grid), constrained_layout=True
        )
        FigureCanvasAgg(figure)
        for panel, index in enumerate(indices, start=1):
            axis = figure.add_subplot(grid, grid, panel)
            axis.imshow(
                np.rot90(np.take(rss, int(index), axis=axis_index)),
                cmap="gray",
                vmin=0,
                vmax=display_maximum,
            )
            axis.set_title(f"{axis_name} {int(index)}")
            axis.axis("off")
        figure.suptitle(
            f"Physical-coil set-4 calibration RSS: indexed {axis_name} slices"
        )
        output = output_directory / f"physical_set4_rss_slices_{axis_name.lower()}.png"
        figure.savefig(output, dpi=140)
        outputs.append(output)
    return outputs


def _write_region_slice_montages(
    rss: np.ndarray,
    preservation: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    holdout: np.ndarray,
    output_directory: Path,
    candidate_id: str,
) -> list[Path]:
    """Write indexed four-region overlay montages in all logical orientations.

    Args:
        rss: Finite nonnegative calibration RSS in RO/LIN/PAR order.
        preservation: Desired whole-head preservation region.
        positive: Clean positive estimation subset.
        negative: Pure external shoulder estimation region.
        holdout: Contaminated in-head region excluded from estimation.
        output_directory: Destination directory for three montage PNGs.
        candidate_id: Candidate label shown in figure titles.

    Returns:
        Paths to RO-, LIN-, and PAR-slice overlay montage figures.

    Side Effects:
        Writes three fixed-window review PNGs.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    display_maximum = float(np.percentile(rss[rss > 0], 99.5))
    masks = tuple(
        np.asarray(mask) > 0
        for mask in (preservation, positive, negative, holdout)
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for axis_index, axis_name in enumerate(("RO", "LIN", "PAR")):
        grid = 5 if axis_index == 0 else 4
        indices = np.linspace(0, rss.shape[axis_index] - 1, grid * grid, dtype=int)
        figure = Figure(
            figsize=(3.0 * grid, 3.0 * grid), constrained_layout=True
        )
        FigureCanvasAgg(figure)
        for panel, index in enumerate(indices, start=1):
            _draw_mask_panel(
                figure.add_subplot(grid, grid, panel),
                np.take(rss, int(index), axis=axis_index),
                *(np.take(mask, int(index), axis=axis_index) for mask in masks),
                display_maximum,
                f"{axis_name} {int(index)}",
            )
        figure.suptitle(
            f"ROVir {candidate_id}: indexed {axis_name} slices; "
            "yellow=preserve, blue=positive, red=shoulder, magenta=holdout"
        )
        output = output_directory / f"review_slices_{axis_name.lower()}.png"
        figure.savefig(output, dpi=140)
        outputs.append(output)
    return outputs


def _draw_mask_panel(
    axis: Any,
    magnitude: np.ndarray,
    preservation: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    holdout: np.ndarray,
    display_maximum: float,
    title: str,
) -> None:
    """Draw one grayscale image with four region overlays.

    Args:
        axis: Matplotlib axis receiving the image.
        magnitude: Two-dimensional magnitude panel.
        preservation: Two-dimensional whole-head preservation support.
        positive: Two-dimensional clean-positive support.
        negative: Two-dimensional pure-shoulder support.
        holdout: Two-dimensional contaminated holdout support.
        display_maximum: Shared upper grayscale window.
        title: Panel title.
    """
    axis.imshow(np.rot90(magnitude), cmap="gray", vmin=0, vmax=display_maximum)
    yellow = np.ma.masked_where(~np.rot90(preservation), np.rot90(preservation))
    blue = np.ma.masked_where(~np.rot90(positive), np.rot90(positive))
    red = np.ma.masked_where(~np.rot90(negative), np.rot90(negative))
    magenta = np.ma.masked_where(~np.rot90(holdout), np.rot90(holdout))
    axis.imshow(yellow, cmap="YlOrBr", alpha=0.18, vmin=0, vmax=1)
    axis.imshow(blue, cmap="Blues", alpha=0.35, vmin=0, vmax=1)
    axis.imshow(red, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
    axis.imshow(magenta, cmap="RdPu", alpha=0.40, vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")


def _write_four_region_curve_csv(
    path: Path,
    solver_entries: Sequence[Mapping[str, Any]],
    preservation_entries: Sequence[Mapping[str, Any]],
    holdout_entries: Sequence[Mapping[str, Any]],
) -> None:
    """Write aligned estimation, preservation, and holdout metrics.

    Args:
        path: Destination CSV path.
        solver_entries: Clean-positive versus pure-negative metrics.
        preservation_entries: Whole-head versus pure-negative metrics.
        holdout_entries: Contaminated-holdout versus pure-negative metrics.

    Side Effects:
        Writes a UTF-8 CSV file.
    """
    fields = (
        "virtual_coils",
        "clean_positive_retention_fraction",
        "whole_head_mixed_energy_retention_fraction",
        "contaminated_holdout_mixed_energy_retention_fraction",
        "pure_shoulder_remaining_fraction",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for solver, preservation, holdout in zip(
            solver_entries, preservation_entries, holdout_entries, strict=True
        ):
            count = int(solver["virtual_coils"])
            if (
                int(preservation["virtual_coils"]) != count
                or int(holdout["virtual_coils"]) != count
            ):
                raise ValueError("Four-region curve channel counts are misaligned.")
            writer.writerow(
                {
                    "virtual_coils": count,
                    "clean_positive_retention_fraction": solver[
                        "signal_retention_fraction"
                    ],
                    "whole_head_mixed_energy_retention_fraction": preservation[
                        "signal_retention_fraction"
                    ],
                    "contaminated_holdout_mixed_energy_retention_fraction": holdout[
                        "signal_retention_fraction"
                    ],
                    "pure_shoulder_remaining_fraction": solver[
                        "interference_remaining_fraction"
                    ],
                }
            )


def _write_four_region_curve_plot(
    path: Path,
    solver_entries: Sequence[Mapping[str, Any]],
    preservation_entries: Sequence[Mapping[str, Any]],
    holdout_entries: Sequence[Mapping[str, Any]],
) -> None:
    """Plot estimation, preservation, holdout, and shoulder energy curves.

    Args:
        path: Destination PNG path.
        solver_entries: Clean-positive versus pure-negative metrics.
        preservation_entries: Whole-head versus pure-negative metrics.
        holdout_entries: Contaminated-holdout versus pure-negative metrics.

    Side Effects:
        Writes one fixed-axis PNG and does not mark a preferred channel count.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    counts = [int(entry["virtual_coils"]) for entry in solver_entries]
    positive = [
        float(entry["signal_retention_fraction"]) for entry in solver_entries
    ]
    preservation = [
        float(entry["signal_retention_fraction"]) for entry in preservation_entries
    ]
    holdout = [
        float(entry["signal_retention_fraction"]) for entry in holdout_entries
    ]
    negative = [
        float(entry["interference_remaining_fraction"]) for entry in solver_entries
    ]
    figure = Figure(figsize=(8, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(counts, positive, marker="o", markersize=3, label="clean positive")
    axis.plot(counts, preservation, label="whole-head mixed energy")
    axis.plot(counts, holdout, label="contaminated holdout mixed energy")
    axis.plot(
        counts,
        negative,
        marker="o",
        markersize=3,
        label="pure shoulder remaining",
    )
    axis.set_xlabel("Leading ordered ROVir virtual coils")
    axis.set_ylabel("Fraction of physical-coil region energy")
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.set_title("ROVir four-region tradeoff; no automatic selection")
    figure.savefig(path, dpi=180)


def _active_coil_shape(path: str | Path) -> tuple[int, int, int, int]:
    """Return a BART array's first four active dimensions.

    Args:
        path: BART basename containing spatial dimensions followed by coils.

    Returns:
        Four positive dimensions in RO, LIN, PAR, coil order.

    Raises:
        ValueError: If later dimensions are nonsingleton or coils are absent.
    """
    shape = read_shape(path)
    padded = shape + (1,) * max(0, 4 - len(shape))
    if padded[3] < 2 or any(value != 1 for value in padded[4:]):
        raise ValueError(f"Expected only spatial and physical-coil dimensions: {shape}.")
    return tuple(int(value) for value in padded[:4])


def _coil_cfl_view(path: str | Path) -> np.ndarray:
    """View one BART CFL as an RO/LIN/PAR/coil array.

    Args:
        path: BART basename with physical coils in dimension 3.

    Returns:
        Memory-mapped four-dimensional array view.
    """
    shape = _active_coil_shape(path)
    return np.reshape(open_cfl(path), shape, order="F")


def _spatial_cfl_view(path: str | Path) -> np.ndarray:
    """View a real-valued BART scalar CFL as three spatial dimensions.

    Args:
        path: BART basename with only RO/LIN/PAR nonsingleton dimensions.

    Returns:
        Three-dimensional complex64 memory-map view.

    Raises:
        ValueError: If later dimensions are nonsingleton or values are complex.
    """
    shape = read_shape(path)
    padded = shape + (1,) * max(0, 3 - len(shape))
    if any(value != 1 for value in padded[3:]):
        raise ValueError(f"Expected a scalar spatial BART array: {shape}.")
    view = np.reshape(open_cfl(path), padded[:3], order="F")
    maximum_imaginary = float(np.max(np.abs(view.imag)))
    if maximum_imaginary > 1e-6 * max(1.0, float(np.max(np.abs(view.real)))):
        raise ValueError(f"Spatial BART scalar contains complex residual: {path}.")
    return view


def _write_real_cfl(path: Path, values: np.ndarray) -> None:
    """Write one finite real array as a complex64 BART CFL pair.

    Args:
        path: Destination BART basename.
        values: Finite real spatial array.

    Side Effects:
        Creates one CFL/HDR pair.
    """
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.isfinite(array).all():
        raise ValueError("Real BART mask values must be finite and noncomplex.")
    output = create_cfl(path, array.shape)
    output[...] = array.astype(np.complex64)
    output.flush()
    del output


def _calibration_voxel_size_logical_mm(
    geometry: Mapping[str, Any], calibration_shape: Sequence[int]
) -> tuple[float, float, float]:
    """Derive calibration-grid RO/LIN/PAR spacing from the full physical FOV.

    Args:
        geometry: Source geometry containing physical XYZ FOV and full logical
            RO/LIN/PAR matrix records.
        calibration_shape: Zero-filled calibration matrix in RO/LIN/PAR order.

    Returns:
        Voxel sizes in logical RO/LIN/PAR order, in millimeters.

    Raises:
        ValueError: If geometry or resulting spacing is invalid.
    """
    try:
        fov_xyz = tuple(float(value) for value in geometry["physical_fov_mm_xyz"])
        full_logical = tuple(
            int(value) for value in geometry["logical_matrix_ro_lin_par"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Source geometry lacks valid FOV or logical matrix data.") from exc
    shape = tuple(int(value) for value in calibration_shape)
    if (
        len(fov_xyz) != 3
        or len(full_logical) != 3
        or len(shape) != 3
        or any(not np.isfinite(value) or value <= 0 for value in fov_xyz)
        or any(value < 1 for value in full_logical + shape)
    ):
        raise ValueError("Source or calibration geometry is invalid.")
    # Sagittal MPRAGE logical (RO, LIN, PAR) maps to physical (Z, Y, X).
    spacing = (fov_xyz[2] / shape[0], fov_xyz[1] / shape[1], fov_xyz[0] / shape[2])
    if any(not np.isfinite(value) or value <= 0 or value > 20 for value in spacing):
        raise ValueError(f"Calibration voxel sizes are implausible: {spacing}.")
    return spacing


def _write_geometry_bound_nifti(
    path: Path,
    values: np.ndarray,
    affine: np.ndarray,
    *,
    label_image: bool,
) -> None:
    """Write one finite 3D NIfTI with explicit matching qform and sform.

    Args:
        path: Destination ``.nii`` or ``.nii.gz`` path.
        values: Finite three-dimensional reference or label array.
        affine: Finite nonsingular voxel-to-RAS affine.
        label_image: Whether to encode the image as a uint8 label template.

    Raises:
        ValueError: If array or affine geometry is invalid.

    Side Effects:
        Writes one NIfTI file with millimeter spatial units.
    """
    import nibabel as nib

    array = np.asarray(values)
    matrix = np.asarray(affine, dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("Geometry-bound NIfTI data must be finite and 3D.")
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-8
    ):
        raise ValueError("Geometry-bound NIfTI affine must be finite and nonsingular.")
    stored = array.astype(np.uint8 if label_image else np.float32, copy=False)
    image = nib.Nifti1Image(stored, matrix)
    image.set_qform(matrix, code=1)
    image.set_sform(matrix, code=1)
    image.header.set_xyzt_units("mm")
    if label_image:
        image.header.set_intent("label", name="ROVirROI")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _validate_file_record(record: object) -> dict[str, Any]:
    """Verify an ordinary file against a recorded path, size, and SHA-256.

    Args:
        record: Manifest mapping produced by :func:`_file_record`.

    Returns:
        Fresh strict file record after successful validation.

    Raises:
        FileNotFoundError: If the recorded file is absent.
        ValueError: If the record is incomplete or its identity changed.
    """
    if not isinstance(record, Mapping):
        raise ValueError("Expected a recorded file mapping.")
    if not all(key in record for key in ("path", "size_bytes", "sha256")):
        raise ValueError("Recorded file mapping is incomplete.")
    actual = _file_record(str(record["path"]))
    if (
        actual["size_bytes"] != int(record["size_bytes"])
        or actual["sha256"] != str(record["sha256"])
    ):
        raise ValueError(f"Recorded immutable file identity changed: {actual['path']}")
    return actual


def _file_identity(path: Path, *, include_hash: bool) -> dict[str, Any]:
    """Record one source file's path, stat identity, and optional digest.

    Args:
        path: Existing file.
        include_hash: Whether to compute SHA-256.

    Returns:
        JSON-compatible file identity.
    """
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


def _file_record(path: str | Path) -> dict[str, Any]:
    """Return a strict SHA-256 record for one ordinary file.

    Args:
        path: Existing file path.

    Returns:
        Resolved path, byte count, and digest.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _version_record(path: str | Path) -> dict[str, Any]:
    """Read and hash a nonempty BART version text record.

    Args:
        path: Text file written by the reviewed shell stage.

    Returns:
        File identity plus exact stripped version output.

    Raises:
        ValueError: If the record is empty.
    """
    resolved = Path(path).expanduser().resolve()
    text = resolved.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"BART version record is empty: {resolved}")
    return {**_file_record(resolved), "output": text}


def _relocate_cfl_record(
    record: Mapping[str, Any], old_root: Path, new_root: Path
) -> dict[str, Any]:
    """Update only a staged CFL record's basename after atomic installation.

    Args:
        record: CFL record calculated in a staging directory.
        old_root: Former staging root.
        new_root: Installed destination root.

    Returns:
        Record with the installed basename and unchanged hashes.
    """
    updated = dict(record)
    updated["base"] = str(new_root / Path(str(record["base"])).relative_to(old_root))
    return updated


def _relocate_file_record(
    record: Mapping[str, Any], old_root: Path, new_root: Path
) -> dict[str, Any]:
    """Update a staged ordinary-file record after atomic installation.

    Args:
        record: File record calculated in a staging directory.
        old_root: Former staging root.
        new_root: Installed destination root.

    Returns:
        Record with the installed path and unchanged size and digest.
    """
    updated = dict(record)
    updated["path"] = str(new_root / Path(str(record["path"])).relative_to(old_root))
    return updated


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read one required JSON object.

    Args:
        path: JSON file path.

    Returns:
        Decoded JSON object.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If the top-level value is not an object.
    """
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one indented JSON object.

    Args:
        path: Destination file path.
        payload: JSON-compatible mapping.

    Side Effects:
        Creates parent directories and atomically replaces the destination.
    """
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
