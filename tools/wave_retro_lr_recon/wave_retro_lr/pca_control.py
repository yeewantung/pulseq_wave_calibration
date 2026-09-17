"""Prepare and review higher-channel PCA controls for Wave-MPRAGE."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bart_io import cfl_record, create_cfl, open_cfl, sha256_file
from .mprage import _embed_image_stream, load_wave_mprage_helpers
from .sampling import SamplingPattern, inspect_twix_sampling

PHYSICAL_CALIBRATION_MANIFEST = Path("manifests") / "physical_calibration.json"


def prepare_mprage_pca_control(
    twix: str | Path,
    sequence: str | Path,
    accepted_normal_root: str | Path,
    feasibility_root: str | Path,
    output_root: str | Path,
    *,
    virtual_coils: int = 24,
) -> dict[str, Any]:
    """Prepare a standard ACS-PCA control while reusing the accepted Wave PSF.

    Args:
        twix: Measured Wave-MPRAGE TWIX file.
        sequence: Matching Pulseq sequence file.
        accepted_normal_root: Existing reconstruction root containing the
            accepted normal-input manifest and calibrated PSF.
        feasibility_root: Existing ROVir feasibility root containing the
            physical-coil set-4 calibration export.
        output_root: User-approved independent PCA-control output root.
        virtual_coils: Number of leading standard PCA coils to retain.

    Returns:
        Manifest describing the transformed image k-space, ACS, basis, and
        exactly reused PSF.

    Raises:
        FileExistsError: If an incompatible or partial destination exists.
        FileNotFoundError: If a required source artifact is absent.
        ValueError: If provenance, dimensions, samples, hashes, or values fail
            validation.

    Side Effects:
        Reads the source TWIX image stream and writes BART inputs below
        ``output_root/normal/bart_inputs``. It does not launch BART, ecalib, or
        PSF calibration.
    """
    import torch

    twix_path = Path(twix).expanduser().resolve()
    sequence_path = Path(sequence).expanduser().resolve()
    accepted_root = Path(accepted_normal_root).expanduser().resolve()
    feasibility = Path(feasibility_root).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    destination = root / "normal" / "bart_inputs"
    manifest_path = destination / "manifest.json"

    print(
        "Validating the accepted normal manifest and hashing the source TWIX.",
        file=sys.stderr,
        flush=True,
    )
    source = _validated_source_contract(
        twix_path, sequence_path, accepted_root, include_twix_hash=True
    )
    physical_coils = int(source["coil_compression"]["physical_coils"])
    ncc = int(virtual_coils)
    if not 1 <= ncc <= physical_coils:
        raise ValueError(
            f"Virtual-coil count must be within [1, {physical_coils}], found {ncc}."
        )
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        _validate_pca_control_reuse(existing, source, feasibility, destination, ncc)
        return existing
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"PCA-control BART input directory is not empty: {destination}")

    print(
        "Validating the exported physical set-4 ACS and its hashes.",
        file=sys.stderr,
        flush=True,
    )
    physical_manifest = _read_json(feasibility / PHYSICAL_CALIBRATION_MANIFEST)
    _validate_physical_calibration_export(physical_manifest, source, feasibility)
    physical_base = (
        feasibility / "inputs" / "physical_calibration" / "physical_set4_kspace"
    )
    physical_record = cfl_record(physical_base)
    physical = open_cfl(physical_base)

    geometry = source["geometry"]
    nro, nlin, npar = (
        int(value) for value in geometry["logical_matrix_ro_lin_par"]
    )
    readout_oversampling = int(geometry["readout_oversampling_factor"])
    ro_oversampled = nro * readout_oversampling
    ncalib = int(source["psf_calibration"]["ncalib"])
    nacs = int(source["psf_calibration"]["nacs"])
    if tuple(physical.shape) != (nro, ncalib, ncalib, physical_coils):
        raise ValueError("Physical set-4 calibration geometry is incompatible.")
    calib_start = ncalib // 2 - nacs // 2
    packed = np.array(
        physical[
            :, calib_start : calib_start + nacs, calib_start : calib_start + nacs, :
        ],
        dtype=np.complex64,
        copy=True,
    )
    if not np.isfinite(packed).all() or not np.any(packed):
        raise ValueError("Physical set-4 ACS must be finite and nonempty.")
    calib_stop = calib_start + nacs
    outside_nonzero = sum(
        int(np.count_nonzero(region))
        for region in (
            physical[:, :calib_start, :, :],
            physical[:, calib_stop:, :, :],
            physical[:, calib_start:calib_stop, :calib_start, :],
            physical[:, calib_start:calib_stop, calib_stop:, :],
        )
    )
    if outside_nonzero:
        raise ValueError("Physical set-4 calibration is not zero outside its ACS.")
    del physical

    print(
        f"Estimating the standard ACS-PCA basis at Ncc={ncc}.",
        file=sys.stderr,
        flush=True,
    )
    native = load_wave_mprage_helpers()
    basis, singular_values, retained_energy = native.estimate_cc_matrix_coillast(
        packed, ncc=ncc, acs=nacs, x_step=1
    )
    basis = np.asarray(basis, dtype=np.complex64)
    if basis.shape != (physical_coils, ncc) or not np.isfinite(basis).all():
        raise ValueError("Estimated PCA basis has invalid shape or values.")
    recorded_singular_values = np.asarray(
        source["coil_compression"].get("leading_singular_values", []),
        dtype=np.float64,
    )
    if (
        recorded_singular_values.size == 0
        or recorded_singular_values.size > singular_values.size
        or not np.allclose(
            singular_values[: recorded_singular_values.size],
            recorded_singular_values,
            rtol=1e-6,
            atol=1e-8,
        )
    ):
        raise ValueError(
            "Physical set-4 ACS does not reproduce the accepted PCA spectrum."
        )
    gram = basis.conj().T @ basis
    orthogonality_error = float(
        np.linalg.norm(gram - np.eye(ncc), ord="fro") / np.sqrt(ncc)
    )
    if orthogonality_error > 1e-5:
        raise ValueError("Estimated PCA basis is not orthonormal.")

    packed_tensor = torch.from_numpy(packed)
    compressed_acs = native.apply_cc_coillast_torch(
        packed_tensor, basis, x_chunk=8
    ).contiguous()
    del packed_tensor, packed
    if tuple(compressed_acs.shape) != (nro, nacs, nacs, ncc):
        raise ValueError("Compressed ACS has unexpected geometry.")

    sampling, _ = inspect_twix_sampling(twix_path, matrix_lin_par=(nlin, npar))
    accepted_manifest = _read_json(
        accepted_root / "normal" / "bart_inputs" / "manifest.json"
    )
    if sampling.to_json() != accepted_manifest.get("sampling"):
        raise ValueError("Current TWIX sampling differs from the accepted normal manifest.")

    print(
        "Loading the measured image stream and applying the shared PCA basis.",
        file=sys.stderr,
        flush=True,
    )
    image = native.load_img(os.fspath(twix_path))
    if image.ndim != 4 or image.shape[-1] != physical_coils:
        raise ValueError("TWIX image stream has incompatible physical-coil geometry.")
    compressed_loaded = native.apply_cc_coillast_torch(image, basis, x_chunk=8)
    del image
    compressed = _embed_image_stream(
        compressed_loaded,
        _sampling_from_json(accepted_manifest["sampling"]),
        readout_oversampled=ro_oversampled,
        physical_coils=ncc,
    )
    del compressed_loaded
    for start in range(0, ro_oversampled, 8):
        if not torch.isfinite(compressed[start : start + 8]).all():
            raise ValueError("Compressed image k-space contains nonfinite values.")
    kspace_norm = float(torch.linalg.vector_norm(compressed).item())
    if not np.isfinite(kspace_norm) or kspace_norm <= 0:
        raise ValueError("Compressed image k-space norm is invalid.")

    source_inputs = accepted_root / "normal" / "bart_inputs"
    source_psf = source_inputs / "psf"
    source_psf_record = cfl_record(source_psf)
    expected_psf_shape = (ro_oversampled, nlin, npar, 1, 1)
    if tuple(source_psf_record["shape"]) != expected_psf_shape:
        raise ValueError("Accepted PSF geometry differs from the PCA-control image grid.")
    psf_values = open_cfl(source_psf)
    for start in range(0, ro_oversampled, 16):
        if not np.isfinite(psf_values[start : start + 16]).all():
            raise ValueError("Accepted PSF contains nonfinite values.")
    del psf_values

    print(
        "Writing transformed image/ACS inputs, copying the accepted PSF, and hashing outputs.",
        file=sys.stderr,
        flush=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bart_inputs_pca_control-", dir=destination.parent))
    try:
        wave_output = create_cfl(
            staging / "wave_kspace", (ro_oversampled, nlin, npar, ncc, 1)
        )
        wave_output[:, :, :, :, 0] = compressed.cpu().numpy()
        wave_output.flush()
        del wave_output, compressed

        calibration_output = create_cfl(
            staging / "kspace_calib", (nro, nlin, npar, ncc)
        )
        calibration_output[:] = 0
        lin_start = nlin // 2 - nacs // 2
        par_start = npar // 2 - nacs // 2
        calibration_output[
            :, lin_start : lin_start + nacs, par_start : par_start + nacs, :
        ] = compressed_acs.cpu().numpy()
        calibration_output.flush()
        del calibration_output, compressed_acs

        np.save(staging / "coil_compression_basis.npy", basis, allow_pickle=False)
        _copy_cfl_exact(source_psf, staging / "psf")
        destination_psf_record = cfl_record(staging / "psf")
        if not _same_cfl_content(source_psf_record, destination_psf_record):
            raise RuntimeError("Copied PSF differs from the accepted PSF.")

        wave_record = _relocate_cfl_record(
            cfl_record(staging / "wave_kspace"), staging, destination
        )
        calibration_record = _relocate_cfl_record(
            cfl_record(staging / "kspace_calib"), staging, destination
        )
        destination_psf_record = _relocate_cfl_record(
            destination_psf_record, staging, destination
        )
        basis_path = staging / "coil_compression_basis.npy"
        manifest = {
            "format_version": 1,
            "status": "measured_wave_mprage_standard_pca_control_ready",
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "physical_calibration": {
                "manifest_path": str(feasibility / PHYSICAL_CALIBRATION_MANIFEST),
                "manifest_sha256": sha256_file(
                    feasibility / PHYSICAL_CALIBRATION_MANIFEST
                ),
                "cfl": physical_record,
                "set_index_zero_based": 4,
                "acs_kept_separate_from_wave_image_kspace": True,
            },
            "geometry": {
                "physical_fov_mm_xyz": list(geometry["physical_fov_mm_xyz"]),
                "logical_matrix_ro_lin_par": [nro, nlin, npar],
                "readout_oversampling_factor": readout_oversampling,
            },
            "sampling": sampling.to_json(),
            "sampling_validation": {
                "acquired_pe_coordinate_count": int(np.count_nonzero(sampling.mask())),
                "same_mdh_sampling_before_and_after_compression": True,
                "zero_outside_sampling_mask": True,
                "acs_merged_into_wave_image_kspace": False,
            },
            "coil_compression": {
                "physical_coils": physical_coils,
                "virtual_coils": ncc,
                "method": "integrated set-4 ACS covariance eigendecomposition",
                "basis_applied_identically_to_image_and_acs": True,
                "basis_file": "coil_compression_basis.npy",
                "basis_sha256": sha256_file(basis_path),
                "basis_shape": list(basis.shape),
                "orthogonality_relative_frobenius_error": orthogonality_error,
                "retained_energy": float(retained_energy[ncc - 1]),
                "leading_singular_values": [
                    float(value) for value in singular_values[:ncc]
                ],
            },
            "psf_calibration": {
                "reused_without_recalibration": True,
                "source": source_psf_record,
                "copied": destination_psf_record,
                "source_and_copy_hashes_equal": True,
            },
            "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
            "kspace_calib": "kspace_calib",
            "kspace_calib_shape": list(calibration_record["shape"]),
            "artifacts": {
                "wave_kspace": wave_record,
                "kspace_calib": calibration_record,
            },
            "echoes": [
                {
                    "echo": 1,
                    "wave_kspace": "wave_kspace",
                    "wave_kspace_shape": list(wave_record["shape"]),
                    "wave_kspace_norm": kspace_norm,
                    "psf": "psf",
                    "psf_shape": list(destination_psf_record["shape"]),
                }
            ],
            "scientific_actions": {
                "twix_image_imported": True,
                "psf_recalibrated": False,
                "ecalib_launched": False,
                "wave_reconstruction_launched": False,
            },
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "sampling_class.txt").write_text(
            sampling.name + "\n", encoding="utf-8"
        )
        if destination.exists():
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"PCA-control BART input directory became nonempty: {destination}"
                )
            destination.rmdir()
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def write_mprage_pca_control_qc(
    baseline_nifti: str | Path,
    control_nifti: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write a shared-window center-slice comparison for two magnitude NIfTIs.

    Args:
        baseline_nifti: Accepted Ncc=12 FISTA-r0 magnitude NIfTI.
        control_nifti: New higher-Ncc FISTA-r0 magnitude NIfTI.
        output_directory: User-approved QC output directory.

    Returns:
        Manifest recording input hashes, geometry, window, and figure.

    Raises:
        ValueError: If images differ in geometry or contain invalid values.

    Side Effects:
        Writes one PNG and one JSON manifest below ``output_directory``.
    """
    import matplotlib.pyplot as plt
    import nibabel as nib

    baseline_path = Path(baseline_nifti).expanduser().resolve()
    control_path = Path(control_nifti).expanduser().resolve()
    baseline_image = nib.load(str(baseline_path))
    control_image = nib.load(str(control_path))
    baseline = np.asarray(baseline_image.dataobj, dtype=np.float32)
    control = np.asarray(control_image.dataobj, dtype=np.float32)
    if (
        baseline.shape != control.shape
        or baseline.ndim != 3
        or not np.allclose(baseline_image.affine, control_image.affine, atol=1e-5)
    ):
        raise ValueError("Baseline and PCA-control NIfTIs have different geometry.")
    if (
        not np.isfinite(baseline).all()
        or not np.isfinite(control).all()
        or np.any(baseline < 0)
        or np.any(control < 0)
    ):
        raise ValueError("Magnitude NIfTIs must be finite and nonnegative.")
    positive = baseline[baseline > 0]
    if positive.size == 0:
        raise ValueError("Baseline magnitude NIfTI is empty.")
    vmax = float(np.percentile(positive, 99.5))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Baseline-anchored display window is invalid.")

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "fixed_window_ncc12_vs_ncc24.png"
    labels = ("sagittal", "coronal", "axial")
    indices = tuple(size // 2 for size in baseline.shape)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, (name, values) in enumerate(
        (("Ncc=12 FISTA λ=0", baseline), ("Ncc=24 FISTA λ=0", control))
    ):
        slices = (
            np.rot90(values[indices[0], :, :]),
            np.rot90(values[:, indices[1], :]),
            np.rot90(values[:, :, indices[2]]),
        )
        for column, (orientation, plane) in enumerate(zip(labels, slices, strict=True)):
            axes[row, column].imshow(plane, cmap="gray", vmin=0, vmax=vmax)
            axes[row, column].set_title(f"{name}\n{orientation} center")
            axes[row, column].axis("off")
    figure.suptitle("Standard ACS-PCA coil-count control; shared Ncc=12 p99.5 window")
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    manifest = {
        "format_version": 1,
        "status": "mprage_pca_control_qc_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": _file_record(baseline_path),
        "control": _file_record(control_path),
        "shape": list(baseline.shape),
        "affine": np.asarray(baseline_image.affine).tolist(),
        "center_indices": list(indices),
        "display_window": {
            "vmin": 0.0,
            "vmax": vmax,
            "anchor": "Ncc=12 positive-voxel percentile",
            "percentile": 99.5,
            "shared_between_rows": True,
        },
        "figure": _file_record(figure_path),
        "automatic_winner_selected": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _sampling_from_json(payload: Mapping[str, Any]) -> SamplingPattern:
    """Construct a sampling pattern from an accepted manifest object.

    Args:
        payload: JSON sampling record.

    Returns:
        Validated sampling-pattern value object.
    """
    return SamplingPattern(
        name=str(payload["name"]),
        acceleration_lin_par=tuple(
            int(value) for value in payload["acceleration_lin_par"]
        ),
        lin_residue=(
            None if payload["lin_residue"] is None else int(payload["lin_residue"])
        ),
        matrix_lin_par=tuple(int(value) for value in payload["matrix_lin_par"]),
        acquired_lin=tuple(int(value) for value in payload["acquired_lin"]),
        acquired_par=tuple(int(value) for value in payload["acquired_par"]),
        measurement_index=payload.get("measurement_index"),
        skip_lin_par=tuple(int(value) for value in payload["skip_lin_par"]),
    )


def _validated_source_contract(
    twix: str | Path,
    sequence: str | Path,
    normal_output_root: str | Path,
    *,
    include_twix_hash: bool,
) -> dict[str, Any]:
    """Bind supplied sources to an accepted normal-input manifest.

    Args:
        twix: Candidate TWIX path.
        sequence: Candidate sequence path.
        normal_output_root: Existing accepted reconstruction root.
        include_twix_hash: Whether to hash the large TWIX payload.

    Returns:
        Source, geometry, coil, and PSF fields required by PCA preparation.

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
    if source["sequence"].get("sha256") != sha256_file(sequence_path):
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


def _validate_physical_calibration_export(
    manifest: Mapping[str, Any], source: Mapping[str, Any], root: Path
) -> None:
    """Validate an existing physical set-4 calibration export for exact reuse.

    Args:
        manifest: Physical-calibration manifest.
        source: Current source contract.
        root: Root containing the exported physical calibration.

    Returns:
        None.

    Raises:
        ValueError: If provenance, dimensions, or stored hashes differ.
    """
    if manifest.get("source") != source:
        raise ValueError("Physical calibration uses different sources.")
    current = cfl_record(
        root / "inputs" / "physical_calibration" / "physical_set4_kspace"
    )
    recorded = manifest.get("physical_set4_kspace")
    for key in ("shape", "payload_bytes", "header_sha256", "payload_sha256"):
        if not isinstance(recorded, Mapping) or recorded.get(key) != current.get(key):
            raise ValueError("Physical calibration failed exact hash reuse.")


def _require_same_file(path: Path, record: object, label: str) -> None:
    """Require a supplied file to match a manifest identity record.

    Args:
        path: Existing supplied file.
        record: Manifest record with path, size, and modification time.
        label: Human-readable source label.

    Returns:
        None.

    Raises:
        ValueError: If metadata or filesystem identity differs.
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


def _file_identity(path: Path, *, include_hash: bool) -> dict[str, Any]:
    """Return path, size, timestamp, and optional SHA-256 for one source file.

    Args:
        path: Existing source file.
        include_hash: Whether to include a SHA-256 digest.

    Returns:
        JSON-compatible source identity.
    """
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


def _copy_cfl_exact(source: Path, destination: Path) -> None:
    """Copy one BART CFL pair and require byte-identical files.

    Args:
        source: Existing source BART basename.
        destination: Destination BART basename.

    Returns:
        None.

    Raises:
        RuntimeError: If either copied file differs from its source.
    """
    for suffix in (".hdr", ".cfl"):
        source_file = source.with_suffix(suffix)
        destination_file = destination.with_suffix(suffix)
        shutil.copy2(source_file, destination_file)
        if sha256_file(source_file) != sha256_file(destination_file):
            raise RuntimeError(f"CFL copy hash mismatch: {destination_file}")


def _same_cfl_content(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Compare the shape, size, and stored-file hashes of two CFL records.

    Args:
        first: First ``cfl_record`` mapping.
        second: Second ``cfl_record`` mapping.

    Returns:
        ``True`` when both records describe byte-identical BART pairs.
    """
    keys = ("shape", "payload_bytes", "header_sha256", "payload_sha256")
    return all(first.get(key) == second.get(key) for key in keys)


def _relocate_cfl_record(
    record: Mapping[str, Any], old_root: Path, new_root: Path
) -> dict[str, Any]:
    """Replace a staged CFL basename with its installed destination.

    Args:
        record: CFL identity created below ``old_root``.
        old_root: Temporary staging directory.
        new_root: Final installed BART-input directory.

    Returns:
        Copy of the record with the final basename.
    """
    updated = dict(record)
    updated["base"] = str(
        new_root / Path(str(record["base"])).relative_to(old_root)
    )
    return updated


def _validate_pca_control_reuse(
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    feasibility_root: Path,
    destination: Path,
    virtual_coils: int,
) -> None:
    """Validate a complete existing PCA-control input directory for reuse.

    Args:
        manifest: Existing PCA-control manifest.
        source: Current source contract.
        feasibility_root: Current physical-calibration feasibility root.
        destination: Existing BART-input directory.
        virtual_coils: Requested output coil count.

    Returns:
        None.

    Raises:
        ValueError: If provenance, coil count, or any artifact differs.
    """
    if (
        manifest.get("status") != "measured_wave_mprage_standard_pca_control_ready"
        or manifest.get("source") != source
        or manifest.get("coil_compression", {}).get("virtual_coils") != virtual_coils
        or manifest.get("physical_calibration", {}).get("manifest_sha256")
        != sha256_file(feasibility_root / PHYSICAL_CALIBRATION_MANIFEST)
    ):
        raise ValueError("Existing PCA-control inputs use a different contract.")
    for name in ("wave_kspace", "kspace_calib"):
        current = cfl_record(destination / name)
        recorded = manifest.get("artifacts", {}).get(name)
        if not isinstance(recorded, Mapping) or not _same_cfl_content(recorded, current):
            raise ValueError(f"Existing PCA-control {name} failed exact hash reuse.")
    current_psf = cfl_record(destination / "psf")
    psf_contract = manifest.get("psf_calibration", {})
    recorded_psf = psf_contract.get("copied")
    if not isinstance(recorded_psf, Mapping) or not _same_cfl_content(
        recorded_psf, current_psf
    ):
        raise ValueError("Existing PCA-control PSF failed exact hash reuse.")
    recorded_source_psf = psf_contract.get("source")
    if not isinstance(recorded_source_psf, Mapping):
        raise ValueError("Existing PCA-control manifest lacks its source PSF record.")
    current_source_psf = cfl_record(Path(str(recorded_source_psf.get("base", ""))))
    if not _same_cfl_content(recorded_source_psf, current_source_psf):
        raise ValueError("Accepted source PSF changed after PCA-control preparation.")
    basis_path = destination / "coil_compression_basis.npy"
    if sha256_file(basis_path) != manifest["coil_compression"]["basis_sha256"]:
        raise ValueError("Existing PCA-control basis failed exact hash reuse.")


def _file_record(path: str | Path) -> dict[str, Any]:
    """Return the path, size, and SHA-256 identity of an ordinary file.

    Args:
        path: Existing file.

    Returns:
        Strict file identity record.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON object.

    Args:
        path: JSON file.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If the JSON root is not an object.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one formatted JSON object.

    Args:
        path: Destination JSON file.
        payload: JSON-compatible mapping.

    Returns:
        None.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
