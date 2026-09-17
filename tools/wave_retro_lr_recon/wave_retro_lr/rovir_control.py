"""Prepare and review matched ROVir coil-count controls for Wave-MPRAGE."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bart_io import cfl_record, create_cfl, open_cfl, sha256_file
from .mprage import _embed_image_stream, load_wave_mprage_helpers
from .pca_control import (
    PHYSICAL_CALIBRATION_MANIFEST,
    _copy_cfl_exact,
    _file_record,
    _read_json,
    _relocate_cfl_record,
    _same_cfl_content,
    _sampling_from_json,
    _validate_physical_calibration_export,
    _validated_source_contract,
    _write_json,
)
from .rovir import rovir_matrix_view, validate_rovir_transform
from .sampling import inspect_twix_sampling

ROVIR_INPUT_MANIFEST = Path("manifests") / "rovir_inputs.json"
ROVIR_QC_MANIFEST = Path("manifests") / "rovir_transform_qc.json"
ROVIR_TRANSFORM = Path("transforms") / "rovir_full" / "transform"


def prepare_mprage_rovir_comparison(
    twix: str | Path,
    sequence: str | Path,
    accepted_normal_root: str | Path,
    feasibility_root: str | Path,
    output_root: str | Path,
    *,
    channel_counts: Sequence[int] = (24, 48),
    partition_progress_interval: int = 12,
) -> dict[str, Any]:
    """Prepare matched ROVir-projected inputs for several retained coil counts.

    Args:
        twix: Measured Wave-MPRAGE TWIX file.
        sequence: Matching Pulseq sequence file.
        accepted_normal_root: Existing reconstruction root supplying the
            accepted calibrated PSF and source contract.
        feasibility_root: Reviewed feasibility root containing the approved
            BART ROVir transform and alias-free physical set-4 ACS.
        output_root: User-approved comparison output root.
        channel_counts: Unique retained ROVir coil counts to prepare.
        partition_progress_interval: Number of acquired partition planes
            between preparation progress messages.

    Returns:
        Shared manifest binding every prepared branch to one source and ROVir
        transform.

    Raises:
        FileExistsError: If an incompatible or partial destination exists.
        FileNotFoundError: If a required source artifact is absent.
        ValueError: If provenance, hashes, geometry, sampling, or finite-value
            checks fail.

    Side Effects:
        Reads the TWIX image stream once and writes BART inputs for every
        requested ROVir count. It does not launch ecalib or reconstruction.
    """
    import torch

    counts = tuple(int(value) for value in channel_counts)
    if not counts or len(set(counts)) != len(counts) or any(value < 1 for value in counts):
        raise ValueError("ROVir channel counts must be a nonempty unique sequence.")
    counts = tuple(sorted(counts))
    if partition_progress_interval < 1:
        raise ValueError("Partition progress interval must be positive.")

    twix_path = Path(twix).expanduser().resolve()
    sequence_path = Path(sequence).expanduser().resolve()
    accepted_root = Path(accepted_normal_root).expanduser().resolve()
    feasibility = Path(feasibility_root).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()

    print("Validating accepted sources, PSF, ACS, and ROVir transform.", file=sys.stderr)
    source = _validated_source_contract(
        twix_path, sequence_path, accepted_root, include_twix_hash=True
    )
    physical_manifest_path = feasibility / PHYSICAL_CALIBRATION_MANIFEST
    physical_manifest = _read_json(physical_manifest_path)
    _validate_physical_calibration_export(physical_manifest, source, feasibility)
    rovir_input_manifest_path = feasibility / ROVIR_INPUT_MANIFEST
    rovir_qc_manifest_path = feasibility / ROVIR_QC_MANIFEST
    rovir_input_manifest = _read_json(rovir_input_manifest_path)
    rovir_qc_manifest = _read_json(rovir_qc_manifest_path)
    if rovir_qc_manifest.get("status") != "mprage_bart_rovir_transform_qc_ready":
        raise ValueError("ROVir transform has not passed the feasibility QC gate.")
    if rovir_qc_manifest.get("rovir_input_manifest_sha256") != sha256_file(
        rovir_input_manifest_path
    ):
        raise ValueError("ROVir transform QC does not match the current solver inputs.")
    if rovir_qc_manifest.get("selected_virtual_coils") is not None:
        raise ValueError("Feasibility QC unexpectedly contains an automatic coil selection.")

    transform_base = feasibility / ROVIR_TRANSFORM
    transform_record = cfl_record(transform_base)
    if not _same_cfl_content(rovir_qc_manifest.get("transform", {}), transform_record):
        raise ValueError("Current ROVir transform differs from its QC manifest.")
    transform = open_cfl(transform_base)
    transform_validation = validate_rovir_transform(
        transform,
        orthogonality_tolerance=float(
            rovir_qc_manifest["transform_validation"]["orthogonality_tolerance"]
        ),
    )
    matrix = np.asarray(rovir_matrix_view(transform), dtype=np.complex64)
    del transform
    physical_coils = int(source["coil_compression"]["physical_coils"])
    if matrix.shape != (physical_coils, physical_coils):
        raise ValueError("ROVir transform and physical receive-coil dimensions disagree.")
    if counts[-1] > physical_coils:
        raise ValueError(f"ROVir channel counts exceed {physical_coils}: {counts}.")

    destinations = {
        count: root / f"rovir_ncc{count}" / "bart_inputs" for count in counts
    }
    existing: dict[int, dict[str, Any]] = {}
    for count, destination in destinations.items():
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            _validate_rovir_branch_reuse(
                manifest, source, feasibility, destination, transform_record, count
            )
            existing[count] = manifest
        elif destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"ROVir BART input directory is not empty: {destination}")
    if len(existing) == len(counts):
        return _write_shared_manifest(
            root, source, feasibility, transform_record, transform_validation, existing
        )

    geometry = source["geometry"]
    nro, nlin, npar = (int(value) for value in geometry["logical_matrix_ro_lin_par"])
    oversampling = int(geometry["readout_oversampling_factor"])
    ro_oversampled = nro * oversampling
    ncalib = int(source["psf_calibration"]["ncalib"])
    nacs = int(source["psf_calibration"]["nacs"])
    physical_base = feasibility / "inputs" / "physical_calibration" / "physical_set4_kspace"
    physical_record = cfl_record(physical_base)
    physical = open_cfl(physical_base)
    if tuple(physical.shape) != (nro, ncalib, ncalib, physical_coils):
        raise ValueError("Physical set-4 calibration geometry is incompatible.")
    calib_start = ncalib // 2 - nacs // 2
    calib_stop = calib_start + nacs
    packed = np.array(
        physical[:, calib_start:calib_stop, calib_start:calib_stop, :],
        dtype=np.complex64,
        copy=True,
    )
    if not np.isfinite(packed).all() or not np.any(packed):
        raise ValueError("Physical set-4 ACS must be finite and nonempty.")
    outside_nonzero = sum(
        int(np.count_nonzero(region))
        for region in (
            physical[:, :calib_start, :, :],
            physical[:, calib_stop:, :, :],
            physical[:, calib_start:calib_stop, :calib_start, :],
            physical[:, calib_start:calib_stop, calib_stop:, :],
        )
    )
    del physical
    if outside_nonzero:
        raise ValueError("Physical set-4 calibration is not zero outside its ACS.")

    # BART forward ccapply conjugates its stored ROVir matrix.
    projection_basis = np.ascontiguousarray(matrix[:, : counts[-1]].conj())
    projected_acs = packed.reshape(-1, physical_coils) @ projection_basis
    projected_acs = projected_acs.reshape(nro, nacs, nacs, counts[-1])
    del packed
    if not np.isfinite(projected_acs).all():
        raise ValueError("ROVir-projected ACS contains nonfinite values.")

    sampling, _ = inspect_twix_sampling(twix_path, matrix_lin_par=(nlin, npar))
    accepted_manifest = _read_json(accepted_root / "normal" / "bart_inputs" / "manifest.json")
    if sampling.to_json() != accepted_manifest.get("sampling"):
        raise ValueError("Current TWIX sampling differs from the accepted normal manifest.")
    sampling_value = _sampling_from_json(accepted_manifest["sampling"])
    mask_numpy = sampling_value.mask()

    missing = tuple(count for count in counts if count not in existing)
    stagings: dict[int, Path] = {}
    wave_outputs: dict[int, np.memmap] = {}
    norms_squared = {count: 0.0 for count in missing}
    source_inputs = accepted_root / "normal" / "bart_inputs"
    source_psf = source_inputs / "psf"
    source_psf_record = cfl_record(source_psf)
    expected_psf_shape = (ro_oversampled, nlin, npar, 1, 1)
    if tuple(source_psf_record["shape"]) != expected_psf_shape:
        raise ValueError("Accepted PSF geometry differs from the ROVir image grid.")
    psf_values = open_cfl(source_psf)
    for start in range(0, ro_oversampled, 16):
        if not np.isfinite(psf_values[start : start + 16]).all():
            raise ValueError("Accepted PSF contains nonfinite values.")
    del psf_values

    try:
        for count in missing:
            destination = destinations[count]
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".bart_inputs_rovir_ncc{count}-", dir=destination.parent)
            )
            stagings[count] = staging
            wave_outputs[count] = create_cfl(
                staging / "wave_kspace", (ro_oversampled, nlin, npar, count, 1)
            )
            calibration_output = create_cfl(
                staging / "kspace_calib", (nro, nlin, npar, count)
            )
            calibration_output[:] = 0
            lin_start = nlin // 2 - nacs // 2
            par_start = npar // 2 - nacs // 2
            calibration_output[
                :, lin_start : lin_start + nacs, par_start : par_start + nacs, :
            ] = projected_acs[..., :count]
            calibration_output.flush()
            del calibration_output
            np.save(
                staging / "rovir_projection_basis.npy",
                projection_basis[:, :count],
                allow_pickle=False,
            )
            _copy_cfl_exact(source_psf, staging / "psf")

        print(
            f"Loading measured image k-space once and projecting to ROVir-{counts[-1]}.",
            file=sys.stderr,
            flush=True,
        )
        native = load_wave_mprage_helpers()
        image = native.load_img(os.fspath(twix_path))
        if image.ndim != 4 or image.shape[0] != ro_oversampled or image.shape[-1] != physical_coils:
            raise ValueError("TWIX image stream has incompatible physical-coil geometry.")
        image = _embed_image_stream(
            image,
            sampling_value,
            readout_oversampled=ro_oversampled,
            physical_coils=physical_coils,
        )
        basis_tensor = torch.as_tensor(projection_basis, dtype=torch.complex64)
        acquired_lin = np.asarray(sampling_value.acquired_lin, dtype=np.int64)
        acquired_par = tuple(int(value) for value in sampling_value.acquired_par)
        for position, par in enumerate(acquired_par, start=1):
            # Project only acquired PE coordinates and leave the new CFL's
            # POSIX-zero-filled exterior sparse. Each write retains complete,
            # contiguous RO vectors instead of small strided RO fragments.
            physical_block = image[:, acquired_lin, par, :]
            projected = (
                physical_block.reshape(-1, physical_coils) @ basis_tensor
            ).reshape(ro_oversampled, acquired_lin.size, counts[-1])
            if not torch.isfinite(projected).all():
                raise ValueError("ROVir-projected image k-space contains nonfinite values.")
            projected_numpy = projected.numpy()
            for count in missing:
                values = projected_numpy[..., :count]
                # NumPy moves the advanced LIN index ahead of the basic RO
                # axis, so present values as [acquired LIN, RO, coil].
                wave_outputs[count][:, acquired_lin, par, :, 0] = np.transpose(
                    values, (1, 0, 2)
                )
                norms_squared[count] += float(np.sum(np.abs(values) ** 2, dtype=np.float64))
            if (
                position == 1
                or position == len(acquired_par)
                or position % partition_progress_interval == 0
            ):
                print(
                    f"Projected acquired PAR plane {position}/{len(acquired_par)}.",
                    file=sys.stderr,
                    flush=True,
                )
            del physical_block, projected, projected_numpy
        del image

        branches = dict(existing)
        for count in missing:
            wave_outputs[count].flush()
            del wave_outputs[count]
            staging = stagings[count]
            destination = destinations[count]
            wave_record = _relocate_cfl_record(cfl_record(staging / "wave_kspace"), staging, destination)
            calibration_record = _relocate_cfl_record(
                cfl_record(staging / "kspace_calib"), staging, destination
            )
            destination_psf_record = _relocate_cfl_record(
                cfl_record(staging / "psf"), staging, destination
            )
            if not _same_cfl_content(source_psf_record, destination_psf_record):
                raise RuntimeError("Copied PSF differs from the accepted PSF.")
            basis_path = staging / "rovir_projection_basis.npy"
            kspace_norm = float(np.sqrt(norms_squared[count]))
            if not np.isfinite(kspace_norm) or kspace_norm <= 0:
                raise ValueError("ROVir-projected image k-space norm is invalid.")
            branch_manifest = {
                "format_version": 1,
                "status": "measured_wave_mprage_rovir_control_ready",
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "physical_calibration": {
                    "manifest_path": str(physical_manifest_path),
                    "manifest_sha256": sha256_file(physical_manifest_path),
                    "cfl": physical_record,
                    "set_index_zero_based": 4,
                    "acs_kept_separate_from_wave_image_kspace": True,
                },
                "rovir": {
                    "solver_backend": "bart rovir only",
                    "transform_source": transform_record,
                    "transform_qc_manifest": _file_record(rovir_qc_manifest_path),
                    "solver_input_manifest": _file_record(rovir_input_manifest_path),
                    "physical_coils": physical_coils,
                    "virtual_coils": count,
                    "projection_convention": "BART forward ccapply: data @ stored_transform.conj()",
                    "basis_applied_identically_to_image_and_acs": True,
                    "projection_basis_file": "rovir_projection_basis.npy",
                    "projection_basis_sha256": sha256_file(basis_path),
                    "projection_basis_shape": [physical_coils, count],
                },
                "geometry": {
                    "physical_fov_mm_xyz": list(geometry["physical_fov_mm_xyz"]),
                    "logical_matrix_ro_lin_par": [nro, nlin, npar],
                    "readout_oversampling_factor": oversampling,
                },
                "sampling": sampling.to_json(),
                "sampling_validation": {
                    "acquired_pe_coordinate_count": int(np.count_nonzero(mask_numpy)),
                    "same_mdh_sampling_before_and_after_projection": True,
                    "zero_outside_sampling_mask": True,
                    "acs_merged_into_wave_image_kspace": False,
                },
                "psf_calibration": {
                    "reused_without_recalibration": True,
                    "source": source_psf_record,
                    "copied": destination_psf_record,
                    "source_and_copy_hashes_equal": True,
                },
                "reconstruction_layout": {
                    "bart_output_relative_to_inputs": "../bart_output",
                },
                "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
                "kspace_calib": "kspace_calib",
                "kspace_calib_shape": list(calibration_record["shape"]),
                "artifacts": {"wave_kspace": wave_record, "kspace_calib": calibration_record},
                "echoes": [{
                    "echo": 1,
                    "wave_kspace": "wave_kspace",
                    "wave_kspace_shape": list(wave_record["shape"]),
                    "wave_kspace_norm": kspace_norm,
                    "psf": "psf",
                    "psf_shape": list(destination_psf_record["shape"]),
                }],
                "scientific_actions": {
                    "twix_image_imported": True,
                    "psf_recalibrated": False,
                    "ecalib_launched": False,
                    "wave_reconstruction_launched": False,
                },
            }
            _write_json(staging / "manifest.json", branch_manifest)
            (staging / "sampling_class.txt").write_text(sampling.name + "\n", encoding="utf-8")
            if destination.exists():
                if any(destination.iterdir()):
                    raise FileExistsError(f"ROVir destination became nonempty: {destination}")
                destination.rmdir()
            staging.replace(destination)
            branches[count] = branch_manifest
        return _write_shared_manifest(
            root, source, feasibility, transform_record, transform_validation, branches
        )
    except Exception:
        for output in wave_outputs.values():
            try:
                output.flush()
            except Exception:
                pass
        for staging in stagings.values():
            if staging.exists():
                shutil.rmtree(staging)
        raise


def write_mprage_rovir_comparison_qc(
    rovir24_nifti: str | Path,
    rovir48_nifti: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write shared-window center-slice QC for ROVir-24 and ROVir-48.

    Args:
        rovir24_nifti: ROVir-24 FISTA-r0 magnitude NIfTI.
        rovir48_nifti: ROVir-48 FISTA-r0 magnitude NIfTI.
        output_directory: User-approved QC output directory.

    Returns:
        Manifest recording hashes, geometry, shared window, and figure.

    Raises:
        ValueError: If images differ in geometry or contain invalid values.

    Side Effects:
        Writes one PNG and one JSON manifest below ``output_directory``.
    """
    import matplotlib.pyplot as plt
    import nibabel as nib

    paths = [Path(rovir24_nifti).resolve(), Path(rovir48_nifti).resolve()]
    images, arrays, normalization = _load_restored_magnitudes(paths)
    if arrays[0].ndim != 3 or arrays[0].shape != arrays[1].shape or not np.allclose(
        images[0].affine, images[1].affine, atol=1e-5
    ):
        raise ValueError("ROVir-24 and ROVir-48 NIfTIs have different geometry.")
    percentiles = []
    for values in arrays:
        positive = values[values > 0]
        if positive.size == 0:
            raise ValueError("ROVir magnitude NIfTI is empty.")
        percentiles.append(float(np.percentile(positive, 99.5)))
    vmax = max(percentiles)
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Shared ROVir display window is invalid.")

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "rovir_ncc24_vs_ncc48_fixed_window.png"
    indices = tuple(size // 2 for size in arrays[0].shape)
    orientations = ("sagittal", "coronal", "axial")
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, (name, values) in enumerate(
        (("ROVir-24 FISTA λ=0", arrays[0]), ("ROVir-48 FISTA λ=0", arrays[1]))
    ):
        planes = (
            np.rot90(values[indices[0], :, :]),
            np.rot90(values[:, indices[1], :]),
            np.rot90(values[:, :, indices[2]]),
        )
        for column, (orientation, plane) in enumerate(zip(orientations, planes, strict=True)):
            axes[row, column].imshow(plane, cmap="gray", vmin=0, vmax=vmax)
            axes[row, column].set_title(f"{name}\n{orientation} center")
            axes[row, column].axis("off")
    figure.suptitle("Matched ROVir coil-count comparison; shared absolute window")
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    manifest = {
        "format_version": 1,
        "status": "mprage_rovir_coil_count_qc_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rovir_ncc24": _file_record(paths[0]),
        "rovir_ncc48": _file_record(paths[1]),
        "magnitude_normalization_restoration": normalization,
        "shape": list(arrays[0].shape),
        "affine": np.asarray(images[0].affine).tolist(),
        "center_indices": list(indices),
        "display_window": {
            "vmin": 0.0,
            "vmax": vmax,
            "per_branch_positive_p99_5": percentiles,
            "anchor": "maximum of both positive-voxel p99.5 values",
            "shared_between_rows": True,
        },
        "figure": _file_record(figure_path),
        "automatic_winner_selected": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def write_mprage_rovir_mask_comparison_qc(
    ro000_030_nifti: str | Path,
    ro000_020_nifti: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Compare ROVir-24 reconstructions from the RO 0--30 and 0--20 masks.

    Args:
        ro000_030_nifti: Existing ROVir-24 magnitude NIfTI using RO 0--30 as
            the negative estimation region.
        ro000_020_nifti: New ROVir-24 magnitude NIfTI using RO 0--20.
        output_directory: User-approved QC output directory.

    Returns:
        Manifest describing scale restoration, shared window, and figure.

    Raises:
        ValueError: If geometry, values, or reversible display-normalization
            provenance is incompatible.

    Side Effects:
        Writes a two-row, three-orientation PNG and JSON manifest.
    """
    import matplotlib.pyplot as plt

    paths = [Path(ro000_030_nifti).resolve(), Path(ro000_020_nifti).resolve()]
    images, arrays, normalization = _load_restored_magnitudes(paths)
    if arrays[0].ndim != 3 or arrays[0].shape != arrays[1].shape or not np.allclose(
        images[0].affine, images[1].affine, atol=1e-5
    ):
        raise ValueError("ROVir mask-comparison NIfTIs have different geometry.")
    percentiles = [float(np.percentile(values[values > 0], 99.5)) for values in arrays]
    vmax = max(percentiles)
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Shared ROVir mask-comparison window is invalid.")

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "rovir24_ro000_030_vs_ro000_020_fixed_window.png"
    indices = tuple(size // 2 for size in arrays[0].shape)
    orientations = ("sagittal", "coronal", "axial")
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, (name, values) in enumerate(
        (("ROVir-24, negative RO 0--30", arrays[0]), ("ROVir-24, negative RO 0--20", arrays[1]))
    ):
        planes = (
            np.rot90(values[indices[0], :, :]),
            np.rot90(values[:, indices[1], :]),
            np.rot90(values[:, :, indices[2]]),
        )
        for column, (orientation, plane) in enumerate(zip(orientations, planes, strict=True)):
            axes[row, column].imshow(plane, cmap="gray", vmin=0, vmax=vmax)
            axes[row, column].set_title(f"{name}\n{orientation} center")
            axes[row, column].axis("off")
    figure.suptitle("ROVir negative-ROI comparison; restored magnitude and shared window")
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    manifest = {
        "format_version": 1,
        "status": "mprage_rovir_negative_roi_comparison_qc_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ro000_030": _file_record(paths[0]),
        "ro000_020": _file_record(paths[1]),
        "magnitude_normalization_restoration": normalization,
        "shape": list(arrays[0].shape),
        "affine": np.asarray(images[0].affine).tolist(),
        "center_indices": list(indices),
        "display_window": {
            "vmin": 0.0,
            "vmax": vmax,
            "per_branch_restored_positive_p99_5": percentiles,
            "anchor": "maximum of both restored positive-voxel p99.5 values",
            "shared_between_rows": True,
        },
        "figure": _file_record(figure_path),
        "automatic_winner_selected": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _load_restored_magnitudes(
    paths: Sequence[Path],
) -> tuple[list[Any], list[np.ndarray], list[dict[str, Any]]]:
    """Load display NIfTIs and reverse their recorded p99 normalization.

    Args:
        paths: Magnitude NIfTI paths with adjacent JSON sidecars.

    Returns:
        Loaded NIfTI images, restored float32 arrays, and normalization records.

    Raises:
        ValueError: If a sidecar lacks reversible normalization provenance or
            an image contains invalid magnitude values.
    """
    import nibabel as nib

    images: list[Any] = []
    arrays: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for path in paths:
        image = nib.load(str(path))
        displayed = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(displayed).all() or np.any(displayed < 0):
            raise ValueError("ROVir magnitude NIfTIs must be finite and nonnegative.")
        name = path.name
        if not name.endswith(".nii.gz"):
            raise ValueError(f"Expected a .nii.gz magnitude image: {path}")
        sidecar_path = path.with_name(name[:-7] + ".json")
        sidecar = _read_json(sidecar_path)
        normalization = sidecar.get("MagnitudeNormalization")
        if not isinstance(normalization, Mapping) or normalization.get("Clipped") is not False:
            raise ValueError("Magnitude normalization must be recorded and unclipped.")
        input_value = float(normalization.get("InputPercentileValue", np.nan))
        output_value = float(normalization.get("OutputPercentileValue", np.nan))
        if not np.isfinite(input_value) or input_value <= 0 or output_value != 1.0:
            raise ValueError("Magnitude normalization scale is not reversibly recorded.")
        restored = displayed * np.float32(input_value / output_value)
        images.append(image)
        arrays.append(restored)
        records.append(
            {
                "sidecar": _file_record(sidecar_path),
                "display_to_restored_scale": input_value / output_value,
                "method": normalization.get("Method"),
                "percentile": normalization.get("Percentile"),
                "clipped": False,
            }
        )
    return images, arrays, records


def _write_shared_manifest(
    root: Path,
    source: Mapping[str, Any],
    feasibility: Path,
    transform_record: Mapping[str, Any],
    transform_validation: Mapping[str, Any],
    branches: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Write the shared manifest after validating all requested branches.

    Args:
        root: Comparison output root.
        source: Validated source contract.
        feasibility: Reviewed ROVir feasibility root.
        transform_record: Exact full-transform CFL record.
        transform_validation: Orthonormality validation report.
        branches: Prepared branch manifests keyed by retained coil count.

    Returns:
        Shared JSON-compatible manifest.

    Side Effects:
        Atomically writes ``shared/manifest.json``.
    """
    records = {}
    for count in sorted(branches):
        path = root / f"rovir_ncc{count}" / "bart_inputs" / "manifest.json"
        records[f"rovir_ncc{count}"] = _file_record(path)
    manifest = {
        "format_version": 1,
        "status": "measured_wave_mprage_rovir_coil_count_comparison_ready",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "feasibility_root": str(feasibility),
        "transform": dict(transform_record),
        "transform_validation": dict(transform_validation),
        "channel_counts": sorted(branches),
        "branches": records,
        "same_transform_image_source_acs_psf_and_sampling": True,
        "automatic_winner_selected": False,
    }
    _write_json(root / "shared" / "manifest.json", manifest)
    return manifest


def _validate_rovir_branch_reuse(
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    feasibility: Path,
    destination: Path,
    transform_record: Mapping[str, Any],
    count: int,
) -> None:
    """Validate an existing prepared ROVir branch for exact reuse.

    Args:
        manifest: Existing branch manifest.
        source: Current accepted source contract.
        feasibility: Current feasibility root.
        destination: Existing BART-input directory.
        transform_record: Current ROVir transform record.
        count: Requested retained virtual-coil count.

    Raises:
        ValueError: If provenance or any prepared artifact differs.
    """
    rovir = manifest.get("rovir", {})
    if (
        manifest.get("status") != "measured_wave_mprage_rovir_control_ready"
        or manifest.get("source") != source
        or rovir.get("virtual_coils") != count
        or not _same_cfl_content(rovir.get("transform_source", {}), transform_record)
        or rovir.get("transform_qc_manifest", {}).get("sha256")
        != sha256_file(feasibility / ROVIR_QC_MANIFEST)
    ):
        raise ValueError(f"Existing ROVir-{count} inputs use a different contract.")
    for name in ("wave_kspace", "kspace_calib"):
        if not _same_cfl_content(
            manifest.get("artifacts", {}).get(name, {}), cfl_record(destination / name)
        ):
            raise ValueError(f"Existing ROVir-{count} {name} failed exact hash reuse.")
    if not _same_cfl_content(
        manifest.get("psf_calibration", {}).get("copied", {}),
        cfl_record(destination / "psf"),
    ):
        raise ValueError(f"Existing ROVir-{count} PSF failed exact hash reuse.")
    basis = destination / "rovir_projection_basis.npy"
    if sha256_file(basis) != rovir.get("projection_basis_sha256"):
        raise ValueError(f"Existing ROVir-{count} projection basis failed hash reuse.")
