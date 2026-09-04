#!/usr/bin/env python3
"""Prepare, reconstruct, and evaluate the two-echo synthetic-Wave GRE sweep."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
TOOL_ROOT = SCRIPT_ROOT.parent
REPOSITORY_ROOT = TOOL_ROOT.parents[1]
RETRO_TOOL_ROOT = REPOSITORY_ROOT / "tools" / "wave_retro_lr_recon"
GRE_ORIENTATION_POLICY_VERSION = 3
GRE_LOGICAL_AXIS_ROLES = ("readout", "phase", "slice")
GRE_AFFINE_AXIS_FLIPS = (False, True, False)
GRE_BRAIN_MASK_CANDIDATE_GRID_VERSION = 4
GRE_PRIOR_MASK_CANDIDATE_ID = "prior_approved_mprage_f0p59_d1"
GRE_PRIOR_MASK_ANTERIOR_D1_CANDIDATE_ID = "prior_approved_mprage_f0p59_d1_anterior-d1"
GRE_BART_OUTPUT_CONVENTION_VERSION = 2
GRE_NIFTI_EXPORT_VERSION = 3
if str(RETRO_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRO_TOOL_ROOT))

from gre_synthetic_wave import (  # noqa: E402
    CASE_IDS,
    COARSE_LAMBDAS,
    ECHO_IDS,
    ECHO_TIMES_S,
    EXTENDED_READOUT,
    LLR_BLOCK_SIZES,
    NATIVE_FOV_MM,
    NATIVE_MATRIX,
    SOURCE_MATRIX,
    VIRTUAL_COILS,
    apply_sampling_mask,
    bart_wave_restoration_factor,
    build_case_mask,
    build_wave_command,
    candidate_name,
    case_definitions,
    circular_phase_metrics,
    coarse_candidate_settings,
    completed_manifest_reusable,
    crop_native_for_case,
    inter_echo_metrics,
    json_sha256,
    restore_bart_normalization,
    theoretical_psf,
    validate_config_document,
    validate_echo_counters,
)
from wave_retro_lr.bart_io import (  # noqa: E402
    bart_base,
    cfl_record,
    create_cfl,
    open_cfl,
    read_shape,
    recombine_split_complex_cfl,
    sha256_file,
)
from wave_retro_lr.core import centered_fftn  # noqa: E402
from wave_retro_lr.retrospective import (  # noqa: E402
    resample_sensitivity_maps,
    synthesize_wave_from_no_wave_crop,
)
from wave_retro_lr.sampling import validate_pure_cartesian_image_lattice  # noqa: E402


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp.

    Returns:
        ISO-8601 UTC timestamp string.
    """

    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON document by atomic replacement.

    Args:
        path: Destination JSON path.
        payload: JSON-compatible mapping to serialize.

    Returns:
        None. The destination is replaced only after serialization completes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path, label: str) -> dict[str, Any]:
    """Load one required JSON object.

    Args:
        path: Existing JSON file.
        label: Human-readable input name for validation errors.

    Returns:
        Parsed JSON object.
    """

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return document


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate one ignored machine-local sweep configuration.

    Args:
        path: Path to the private local JSON configuration.

    Returns:
        Parsed configuration and resolved validation metadata.
    """

    resolved = path.expanduser().resolve()
    config = load_json(resolved, "GRE sweep configuration")
    validated = validate_config_document(config)
    for label, value in config.get("inputs", {}).items():
        if label == "dicom_globs":
            continue
        input_path = Path(str(value)).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Configured {label} file not found: {input_path}")
    validated["config_path"] = str(resolved)
    validated["config_sha256"] = sha256_file(resolved)
    return config, validated


def require_confirmed_root(validated: Mapping[str, Any], confirmation: Path | None) -> Path:
    """Require an exact user-approved run-root confirmation before writing.

    Args:
        validated: Resolved configuration metadata.
        confirmation: Explicit run-root value supplied on the command line.

    Returns:
        Confirmed run root, created if it does not yet exist.
    """

    configured = Path(str(validated["run_root"])).expanduser()
    if confirmation is None or confirmation.expanduser().resolve() != configured.resolve():
        raise ValueError(f"Writing requires --confirm-run-root {configured}")
    configured.mkdir(parents=True, exist_ok=True)
    return configured


def stage_manifest_path(root: Path, operation: str) -> Path:
    """Return the stable manifest path for one preparation operation.

    Args:
        root: Approved experiment run root.
        operation: Stable preparation operation name.

    Returns:
        Operation manifest path below the preparation tree.
    """

    return root / "preparation" / operation / "manifest.json"


def file_identity(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    """Record one input file's path, size, timestamp, and optional SHA-256.

    Args:
        path: Existing file to identify.
        include_hash: Compute and include a full SHA-256 digest when true.

    Returns:
        JSON-native file provenance record.
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


def _verify_embedded_hashes(value: Any) -> None:
    """Recursively verify manifest records that bind paths to hashes.

    Args:
        value: Arbitrarily nested manifest value.

    Returns:
        None. A mismatch raises instead of accepting stale artifacts.
    """

    if isinstance(value, Mapping):
        path_value = value.get("path")
        digest = value.get("sha256")
        if isinstance(path_value, str) and isinstance(digest, str):
            path = Path(path_value)
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"Manifest-bound file changed or disappeared: {path}")
        base_value = value.get("base")
        payload_digest = value.get("payload_sha256")
        if isinstance(base_value, str) and isinstance(payload_digest, str):
            payload = Path(base_value).with_suffix(".cfl")
            if not payload.is_file() or sha256_file(payload) != payload_digest:
                raise ValueError(f"Manifest-bound BART payload changed or disappeared: {payload}")
        for nested in value.values():
            _verify_embedded_hashes(nested)
    elif isinstance(value, list):
        for nested in value:
            _verify_embedded_hashes(nested)


def _reuse_completed_operation(
    manifest_path: Path,
    validated: Mapping[str, Any],
    *,
    allowed_statuses: tuple[str, ...] = ("complete",),
) -> dict[str, Any] | None:
    """Reuse one stage only after configuration and artifact hash validation.

    Args:
        manifest_path: Existing operation manifest candidate.
        validated: Current resolved configuration metadata.
        allowed_statuses: Status values that are safe to reuse.

    Returns:
        Validated manifest, or ``None`` when no reusable manifest exists.
    """

    if not manifest_path.is_file():
        return None
    manifest = load_json(manifest_path, "existing operation manifest")
    if manifest.get("config_sha256") != validated["config_sha256"]:
        raise ValueError(f"Existing operation uses a different configuration: {manifest_path}")
    if manifest.get("status") not in allowed_statuses:
        return None
    _verify_embedded_hashes(manifest)
    return manifest


def _select_twix_measurement(root: Any) -> tuple[int, Any]:
    """Select the TWIX measurement with the largest populated image stream.

    Args:
        root: mapVBVD single measurement or multi-measurement collection.

    Returns:
        Selected zero-based measurement index and measurement object.
    """

    measurements = list(root) if isinstance(root, (list, tuple)) else [root]
    candidates = []
    for index, measurement in enumerate(measurements):
        image = getattr(measurement, "image", None)
        if image is not None and int(getattr(image, "NAcq", 0)) > 0:
            candidates.append((int(image.NAcq), index, measurement))
    if not candidates:
        raise ValueError("No populated TWIX image stream was found.")
    _, index, measurement = max(candidates)
    return index, measurement


def _twix_value(mapping: Any, key: tuple[str, ...], default: Any = None) -> Any:
    """Read one mapVBVD MeasYaps value across mapping implementations.

    Args:
        mapping: MeasYaps-compatible mapping.
        key: Tuple key used by mapVBVD.
        default: Value returned when the key is unavailable.

    Returns:
        Stored value or ``default``.
    """

    try:
        value = mapping.get(key, default)
    except Exception:
        try:
            value = mapping[key]
        except Exception:
            value = default
    return default if value is None else value


def _open_source_twix(path: Path) -> tuple[int, Any, Any, dict[str, Any]]:
    """Open and validate the fully sampled two-echo R1 GRE source.

    Args:
        path: Fully sampled Siemens TWIX file.

    Returns:
        Measurement index, measurement, image stream, and echo metadata.
    """

    import mapvbvd

    root = mapvbvd.mapVBVD(str(path), quiet=True)
    index, measurement = _select_twix_measurement(root)
    image = measurement.image
    image.flagRemoveOS = True
    image.squeeze = True
    shape = tuple(int(value) for value in image.sqzSize)
    dims = list(image.sqzDims)
    expected_shape = (256, 44, 256, 72, 2)
    if shape != expected_shape or dims != ["Col", "Cha", "Lin", "Par", "Eco"]:
        raise ValueError(f"Unexpected source TWIX layout {dims} {shape}; expected {expected_shape}.")
    lines = np.asarray(image.Lin, dtype=np.int64)
    partitions = np.asarray(image.Par, dtype=np.int64)
    echoes = np.asarray(image.Eco, dtype=np.int64)
    yaps = measurement.hdr["MeasYaps"]
    te_s = tuple(float(_twix_value(yaps, ("alTE", str(i)), math.nan)) * 1e-6 for i in range(2))
    if not np.allclose(te_s, ECHO_TIMES_S, rtol=0.0, atol=1e-9):
        raise ValueError(f"Source TWIX echo times {te_s} do not match {ECHO_TIMES_S}.")
    echo_records = validate_echo_counters(
        lines,
        partitions,
        echoes,
        matrix_lin_par=(256, 72),
        echo_times_s=te_s,
    )
    metadata = {
        "measurement_index": index,
        "mapvbvd_squeezed_dimensions": dims,
        "mapvbvd_shape": list(shape),
        "echo_times_s": list(te_s),
        "echoes": echo_records,
        "source_matrix_ro_lin_par": list(SOURCE_MATRIX),
    }
    return index, measurement, image, metadata


def _iter_source_chunks(image: Any, *, echo_index: int, partition_chunk: int) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield centered 250x250 target chunks in coil-last layout.

    Args:
        image: Configured five-dimensional mapVBVD image stream.
        echo_index: Zero-based Eco dimension index.
        partition_chunk: Maximum PAR partitions per payload read.

    Yields:
        Start, stop, and complex64 RO/LIN/PAR/COIL chunk.
    """

    for start in range(0, 72, partition_chunk):
        stop = min(start + partition_chunk, 72)
        raw = np.asarray(image[:, :, :, start:stop, echo_index], dtype=np.complex64)
        expected = (256, 44, 256, stop - start)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError(f"Source chunk has shape {raw.shape}; expected {expected}.")
        cropped = raw[3:253, :, 3:253, :]
        yield start, stop, np.transpose(cropped, (0, 2, 3, 1))


def _coil_basis(image: Any, *, partition_chunk: int, readout_step: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate one shared 44-to-12 basis from balanced echo covariances.

    Args:
        image: Configured source mapVBVD image stream.
        partition_chunk: Maximum PAR partitions per payload read.
        readout_step: Readout subsampling step used only for covariance estimation.

    Returns:
        Orthonormal compression basis and covariance diagnostics.
    """

    echo_covariances = []
    echo_rows = []
    for echo in range(2):
        covariance = np.zeros((44, 44), dtype=np.complex128)
        rows = 0
        for _, _, chunk in _iter_source_chunks(image, echo_index=echo, partition_chunk=partition_chunk):
            design = np.ascontiguousarray(chunk[::readout_step]).reshape(-1, 44)
            usable = design[np.any(design != 0, axis=1)]
            covariance += usable.conj().T @ usable
            rows += int(usable.shape[0])
        covariance = 0.5 * (covariance + covariance.conj().T)
        trace = float(np.trace(covariance).real)
        if trace <= 0 or not np.isfinite(covariance).all():
            raise ValueError(f"Echo {echo + 1} coil covariance is invalid.")
        echo_covariances.append(covariance / trace)
        echo_rows.append(rows)
    combined = 0.5 * (echo_covariances[0] + echo_covariances[1])
    eigenvalues, eigenvectors = np.linalg.eigh(combined)
    order = np.argsort(eigenvalues)[::-1]
    basis = np.asarray(eigenvectors[:, order[:VIRTUAL_COILS]], dtype=np.complex64)
    residual = float(np.max(np.abs(basis.conj().T @ basis - np.eye(VIRTUAL_COILS))))
    if residual > 1e-5:
        raise ValueError(f"Coil basis orthonormality residual is {residual:g}.")
    retained = float(np.sum(eigenvalues[order[:VIRTUAL_COILS]]) / np.sum(eigenvalues))
    return basis, {
        "physical_coils": 44,
        "virtual_coils": VIRTUAL_COILS,
        "calibration_source": "both fully sampled echoes with trace-balanced covariance",
        "usable_rows_per_echo": echo_rows,
        "readout_step": readout_step,
        "retained_normalized_covariance_fraction": retained,
        "orthonormality_maximum_residual": residual,
    }


def prepare_source(config: Mapping[str, Any], validated: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Prepare shared coil compression and two native-grid no-Wave sources.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.

    Returns:
        Completed, hash-bound source preparation manifest.
    """

    metadata = load_json(root / "metadata" / "manifest.json", "metadata manifest")
    if metadata.get("status") != "complete":
        raise ValueError("Source preparation requires completed metadata validation.")
    output = root / "preparation" / "source"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    reusable = _reuse_completed_operation(manifest_path, validated)
    if reusable is not None:
        return reusable
    source_path = Path(str(config["inputs"]["source_twix"])).expanduser().resolve()
    source_record = file_identity(source_path)
    _, _, image, twix_metadata = _open_source_twix(source_path)
    settings = config["coil_compression"]
    basis, basis_metadata = _coil_basis(
        image,
        partition_chunk=int(settings["partition_chunk"]),
        readout_step=int(settings["readout_step"]),
    )
    basis_path = output / "coil_basis.npy"
    np.save(basis_path, basis, allow_pickle=False)
    echoes = []
    for echo_index, echo_id in enumerate(ECHO_IDS):
        path = output / echo_id / "no_wave_kspace.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        target = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.complex64, shape=(*NATIVE_MATRIX, VIRTUAL_COILS)
        )
        squared_norm = 0.0
        for start, stop, chunk in _iter_source_chunks(
            image,
            echo_index=echo_index,
            partition_chunk=int(settings["partition_chunk"]),
        ):
            compressed = np.asarray(chunk @ basis, dtype=np.complex64)
            if not np.isfinite(compressed).all():
                raise ValueError(f"Compressed {echo_id} chunk is non-finite.")
            target[:, :, start:stop, :] = compressed
            squared_norm += float(np.vdot(compressed, compressed).real)
        target.flush()
        del target
        echoes.append(
            {
                "echo": echo_index + 1,
                "echo_id": echo_id,
                "te_s": ECHO_TIMES_S[echo_index],
                "path": str(path),
                "shape": [*NATIVE_MATRIX, VIRTUAL_COILS],
                "sha256": sha256_file(path),
                "l2_norm": math.sqrt(squared_norm),
            }
        )
    manifest = {
        "format_version": 1,
        "status": "complete",
        "operation": "source",
        "created_utc": utc_now(),
        "config_sha256": validated["config_sha256"],
        "source_twix": source_record,
        "twix_metadata": twix_metadata,
        "source_to_native_operation": "centered logical k-space crop without image interpolation",
        "source_to_native_crop_bounds_ro_lin_par": [[3, 253], [3, 253], [0, 72]],
        "native_geometry": validated["geometry"],
        "coil_basis": {**basis_metadata, "path": str(basis_path), "sha256": sha256_file(basis_path)},
        "echoes": echoes,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _load_upstream_gre_module() -> Any:
    """Load the pinned upstream GRE module without invoking its CLI.

    Returns:
        Imported pinned Wave-GRE Python module.
    """

    path = REPOSITORY_ROOT / "external" / "wave-gre-flow-comp" / "recon" / "recon_wave_gre_from_twix_integrated_nifti.py"
    upstream_root = path.parent
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    spec = importlib.util.spec_from_file_location("pinned_wave_gre_focused", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned Wave-GRE module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_operator(config: Mapping[str, Any], validated: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Prepare echo-specific theoretical sequence-derived PSFs on both grids.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.

    Returns:
        Completed theoretical trajectory and PSF manifest.
    """

    metadata = load_json(root / "metadata" / "manifest.json", "metadata manifest")
    if metadata.get("status") != "complete":
        raise ValueError("Operator preparation requires completed metadata validation.")
    output = root / "preparation" / "theoretical_operator"
    output.mkdir(parents=True, exist_ok=True)
    reusable = _reuse_completed_operation(output / "manifest.json", validated)
    if reusable is not None:
        return reusable
    sequence_path = Path(str(config["inputs"]["sequence"])).expanduser().resolve()
    native = _load_upstream_gre_module()
    sequence = native._load_sequence(sequence_path)
    cfg = native._derive_gre_config(sequence, yflip_override=None, zflip_override=None)
    required = {
        "Nx": 250,
        "Ny": 250,
        "Nz": 72,
        "Nx_os": 1000,
        "Necho": 2,
        "orientation": "TRA",
    }
    for key, value in required.items():
        if cfg[key] != value:
            raise ValueError(f"Sequence {key}={cfg[key]!r}; expected {value!r}.")
    if not np.allclose(cfg["FOVxyz_m"], np.asarray(NATIVE_FOV_MM) / 1000.0, atol=1e-12, rtol=0):
        raise ValueError("Sequence FOV does not match 220x220x180 mm.")
    if not np.allclose(cfg["TE_s"], ECHO_TIMES_S, atol=1e-9, rtol=0):
        raise ValueError("Sequence echo times do not match 10/20 ms.")
    image_lines, _ = native._split_adc_trajectory(sequence, cfg)
    trajectories = native._echo_theoretical_wave_trajectories(image_lines, cfg)
    echoes = []
    for echo_index, (delta_ky, delta_kz) in enumerate(trajectories):
        echo_id = ECHO_IDS[echo_index]
        echo_dir = output / echo_id
        echo_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = echo_dir / "trajectory.npz"
        np.savez(trajectory_path, delta_ky_index=delta_ky, delta_kz_index=delta_kz)
        case_records = {}
        for case_id, case in case_definitions().items():
            psf = theoretical_psf(
                delta_ky,
                delta_kz,
                nlin=case.matrix_ro_lin_par[1],
                npar=case.matrix_ro_lin_par[2],
                yflip=int(cfg["yflip"]),
                zflip=int(cfg["zflip"]),
            )
            psf_base = echo_dir / case_id / "psf"
            target = create_cfl(psf_base, (*psf.shape, 1, 1))
            target[..., 0, 0] = psf
            target.flush()
            del target
            magnitude_error = float(np.max(np.abs(np.abs(psf) - 1.0)))
            if magnitude_error > 2e-5:
                raise ValueError(f"{echo_id}/{case_id} PSF is not unit magnitude.")
            case_records[case_id] = {
                **cfl_record(psf_base),
                "unit_magnitude_maximum_error": magnitude_error,
            }
        echoes.append(
            {
                "echo": echo_index + 1,
                "echo_id": echo_id,
                "te_s": ECHO_TIMES_S[echo_index],
                "trajectory": file_identity(trajectory_path),
                "cases": case_records,
            }
        )
    manifest = {
        "format_version": 1,
        "status": "complete",
        "operation": "theoretical_operator",
        "created_utc": utc_now(),
        "config_sha256": validated["config_sha256"],
        "sequence": file_identity(sequence_path),
        "operator_model": "theoretical sequence trajectory without measured PSF calibration",
        "measured_wave_samples_or_coefficients_used": False,
        "sequence_config": {
            key: (np.asarray(cfg[key]).tolist() if isinstance(cfg[key], np.ndarray) else cfg[key])
            for key in ("Nx", "Ny", "Nz", "Nx_os", "Necho", "TE_s", "FOVxyz_m", "orientation", "yflip", "zflip")
        },
        "echoes": echoes,
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _invert_full_wave_encoding(
    wave_kspace: np.ndarray,
    psf: np.ndarray,
    *,
    logical_readout: int,
    workers: int,
) -> np.ndarray:
    """Invert the full-sampling synthetic Wave operator for validation.

    Args:
        wave_kspace: Fully sampled Wave-encoded k-space.
        psf: Matching unit-magnitude theoretical PSF.
        logical_readout: Logical image readout matrix before Wave extension.
        workers: Maximum SciPy FFT worker count.

    Returns:
        Recovered logical no-Wave k-space.
    """

    encoded = np.asarray(wave_kspace, dtype=np.complex64)
    modulation = np.asarray(psf, dtype=np.complex64)
    if encoded.shape != modulation.shape or logical_readout > encoded.shape[0]:
        raise ValueError("Full Wave data, PSF, and logical readout are incompatible.")
    hybrid = centered_fftn(encoded, axes=(1, 2), inverse=True, workers=workers)
    hybrid *= np.conj(modulation)
    extended_image = centered_fftn(hybrid, axes=(0,), inverse=True, workers=workers)
    start = encoded.shape[0] // 2 - logical_readout // 2
    image = extended_image[start : start + logical_readout]
    return centered_fftn(image, axes=(0, 1, 2), workers=workers)


def validate_operator_roundtrip(
    config: Mapping[str, Any], validated: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Validate PSF-one and theoretical-PSF full-sampling round trips.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.

    Returns:
        Completed per-echo operator-validation manifest.
    """

    source = _require_complete(root, "source")
    operator = _require_complete(root, "theoretical_operator")
    output = root / "preparation" / "operator_validation"
    output.mkdir(parents=True, exist_ok=True)
    reusable = _reuse_completed_operation(output / "manifest.json", validated)
    if reusable is not None:
        return reusable
    workers = int(config["runtime"]["fft_workers"])
    records = []
    for echo_index, echo_id in enumerate(ECHO_IDS):
        no_wave_all = np.load(source["echoes"][echo_index]["path"], mmap_mode="r", allow_pickle=False)
        no_wave = np.asarray(no_wave_all[..., 0], dtype=np.complex64)
        psf_base = operator["echoes"][echo_index]["cases"]["native_r3x1"]["base"]
        psf = np.asarray(open_cfl(psf_base)).squeeze()
        tests = {}
        for label, modulation in (
            ("identity_psf", np.ones_like(psf)),
            ("theoretical_psf", psf),
        ):
            encoded = synthesize_wave_from_no_wave_crop(
                no_wave,
                modulation,
                readout_oversampled=EXTENDED_READOUT,
                target_mask=None,
                fft_workers=workers,
            )
            recovered = _invert_full_wave_encoding(
                encoded,
                modulation,
                logical_readout=NATIVE_MATRIX[0],
                workers=workers,
            )
            difference = recovered - no_wave
            relative = float(np.linalg.norm(difference) / np.linalg.norm(no_wave))
            maximum = float(np.max(np.abs(difference)))
            if not np.isfinite(recovered).all() or relative > 5e-5:
                raise ValueError(f"{echo_id} {label} operator round trip failed: {relative:g}.")
            tests[label] = {
                "relative_complex_l2_error": relative,
                "maximum_complex_error": maximum,
                "finite": True,
            }
        records.append({"echo_id": echo_id, "echo": echo_index + 1, "te_s": ECHO_TIMES_S[echo_index], "tests": tests})
    manifest = {
        "format_version": 1,
        "status": "complete",
        "operation": "operator_validation",
        "created_utc": utc_now(),
        "config_sha256": validated["config_sha256"],
        "validation_scope": "full-sampling forward/inverse round trip on one shared-basis coil per echo",
        "tolerance_relative_complex_l2": 5e-5,
        "echoes": records,
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _require_complete(root: Path, operation: str) -> dict[str, Any]:
    """Load one completed preparation manifest.

    Args:
        root: Confirmed experiment run root.
        operation: Required preparation operation name.

    Returns:
        Parsed completed operation manifest.
    """

    path = stage_manifest_path(root, operation)
    manifest = load_json(path, f"{operation} manifest")
    if manifest.get("status") != "complete":
        raise ValueError(f"Preparation operation is not complete: {operation}")
    return manifest


def _stream_npy_to_cfl(source_path: Path, output_base: Path) -> dict[str, Any]:
    """Export one RO/LIN/PAR/COIL complex64 NPY to a BART CFL pair.

    Args:
        source_path: Native no-Wave complex64 NPY file.
        output_base: Destination BART basename.

    Returns:
        Hash-bound BART CFL record.
    """

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.shape != (*NATIVE_MATRIX, VIRTUAL_COILS) or source.dtype != np.complex64:
        raise ValueError("Native no-Wave source has the wrong shape or dtype.")
    target = create_cfl(output_base, (*source.shape, 1))
    for start in range(0, source.shape[2], 4):
        stop = min(start + 4, source.shape[2])
        target[:, :, start:stop, :, 0] = source[:, :, start:stop, :]
    target.flush()
    del target
    return cfl_record(output_base)


def _run_logged(
    command: Sequence[str], log_path: Path, *, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Run one command while streaming and retaining combined output.

    Args:
        command: Exact external command argument vector.
        log_path: Combined stdout/stderr log path.
        environment: Optional complete child-process environment.

    Returns:
        Command, timing, return-code, and log provenance.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=None if environment is None else dict(environment),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    record = {
        "command": list(command),
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "return_code": returncode,
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }
    if returncode != 0:
        raise RuntimeError(f"Command failed with status {returncode}: {' '.join(command)}")
    return record


def _fsl_runtime_environment(bet: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve and freeze the FSL runtime required by a BET wrapper.

    Args:
        bet: Resolved BET executable or site wrapper path.

    Returns:
        Complete subprocess environment and FSL provenance record.

    Raises:
        FileNotFoundError: No complete FSL installation can be resolved from
            the current environment or configured BET wrapper.
    """

    bet_path = Path(bet)
    candidates: list[Path] = []
    configured_root = os.environ.get("FSLDIR")
    if configured_root:
        candidates.append(Path(configured_root))
    try:
        wrapper_text = bet_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        wrapper_text = ""
    for target in re.findall(r"(?m)^\s*(/[^\s\"']+/bin/bet)(?:\s|$)", wrapper_text):
        candidates.append(Path(target).parent.parent)
    candidates.append(bet_path.parent.parent)
    required_relative = (
        Path("bin/bet"),
        Path("bin/remove_ext"),
        Path("bin/bet2"),
        Path("bin/fslstats"),
        Path("bin/fast"),
        Path("bin/fslmaths"),
        Path("bin/standard_space_roi"),
        Path("etc/fslconf/fsl.sh"),
    )
    fsldir = next(
        (
            candidate
            for candidate in candidates
            if all((candidate / relative).is_file() for relative in required_relative)
            and (candidate / "data/standard").is_dir()
        ),
        None,
    )
    if fsldir is None:
        raise FileNotFoundError(f"Could not resolve a complete FSL installation from BET wrapper: {bet_path}")
    search_roots = [fsldir / "share/fsl/bin", fsldir / "bin"]
    search_path = os.pathsep.join(
        [str(path) for path in search_roots if path.is_dir()] + [os.environ.get("PATH", "")]
    )
    environment = {
        **os.environ,
        "FSLDIR": str(fsldir),
        "FSLOUTPUTTYPE": "NIFTI_GZ",
        "FSLMULTIFILEQUIT": "TRUE",
        "FSLTCLSH": str(fsldir / "bin/fsltclsh"),
        "FSLWISH": str(fsldir / "bin/fslwish"),
        "FSL_LOAD_NIFTI_EXTENSIONS": "0",
        "FSL_SKIP_GLOBAL": "0",
        "PATH": search_path,
    }
    version_path = fsldir / "etc/fslversion"
    provenance = {
        "configured_bet": file_identity(bet_path),
        "fsldir": str(fsldir),
        "fsl_version": version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else None,
        "required_tools": {
            relative.name: file_identity(fsldir / relative) for relative in required_relative
        },
        "environment": {
            key: environment[key]
            for key in (
                "FSLDIR",
                "FSLOUTPUTTYPE",
                "FSLMULTIFILEQUIT",
                "FSLTCLSH",
                "FSLWISH",
                "FSL_LOAD_NIFTI_EXTENSIONS",
                "FSL_SKIP_GLOBAL",
                "PATH",
            )
        },
    }
    return environment, provenance


def prepare_csm(config: Mapping[str, Any], validated: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Run the approved one-time echo-1 BART ecalib CSM generation.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.

    Returns:
        Completed shared native/LR CSM manifest.
    """

    source = _require_complete(root, "source")
    output = root / "preparation" / "csm"
    output.mkdir(parents=True, exist_ok=True)
    reusable = _reuse_completed_operation(output / "manifest.json", validated)
    if reusable is not None:
        return reusable
    echo1 = Path(source["echoes"][0]["path"])
    calibration_base = output / "native_echo-01_calibration_kspace"
    calibration_record = _stream_npy_to_cfl(echo1, calibration_base)
    bart = shutil.which(str(config["runtime"]["bart"]))
    if bart is None:
        raise FileNotFoundError(f"BART executable not found: {config['runtime']['bart']}")
    maps_base = output / "native" / "coil_sens"
    maps_base.parent.mkdir(parents=True, exist_ok=True)
    command = [
        bart,
        "ecalib",
        "-m",
        "1",
        "-c",
        f"{float(config['csm']['ecalib_crop']):.12g}",
        "-r",
        "250:32:32",
        str(calibration_base),
        str(maps_base),
    ]
    run = _run_logged(command, output / "ecalib.log")
    if read_shape(maps_base)[:5] != (*NATIVE_MATRIX, VIRTUAL_COILS, 1):
        raise ValueError(f"ecalib maps have unexpected shape {read_shape(maps_base)}.")
    maps = open_cfl(maps_base)
    if not np.isfinite(maps).all():
        raise ValueError("ecalib maps contain non-finite values.")
    rss_squared = np.zeros(NATIVE_MATRIX, dtype=np.float32)
    for coil in range(VIRTUAL_COILS):
        sensitivity = np.asarray(maps[:, :, :, coil, 0, ...]).squeeze()
        rss_squared += np.abs(sensitivity).astype(np.float32) ** 2
    rss = np.sqrt(rss_squared)
    support = rss > 1e-6
    if not np.any(support):
        raise ValueError("ecalib maps contain no supported voxels.")
    maximum_rss_error = float(np.max(np.abs(rss[support] - 1.0)))
    if maximum_rss_error > 5e-3:
        raise ValueError(f"ecalib CSM RSS normalization failed: {maximum_rss_error:g}.")
    low_base = output / "lin_low_resolution" / "coil_sens"
    low_base.parent.mkdir(parents=True, exist_ok=True)
    resample_sensitivity_maps(maps_base, low_base, target_lin_par=(148, 72))
    manifest = {
        "format_version": 1,
        "status": "complete",
        "operation": "csm",
        "created_utc": utc_now(),
        "config_sha256": validated["config_sha256"],
        "shared_across_echoes": True,
        "calibration_echo": 1,
        "calibration_te_s": ECHO_TIMES_S[0],
        "calibration_region_ro_lin_par": [250, 32, 32],
        "calibration_kspace": calibration_record,
        "ecalib": run,
        "native": cfl_record(maps_base),
        "lin_low_resolution": cfl_record(low_base),
        "native_rss_normalization": {
            "support_threshold": 1e-6,
            "support_voxel_count": int(np.count_nonzero(support)),
            "maximum_absolute_error_from_one": maximum_rss_error,
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _sense_direct_reference(no_wave: np.ndarray, maps_base: Path, *, workers: int) -> np.ndarray:
    """Build a complex SENSE-combined direct-FFT reference with shared CSMs.

    Args:
        no_wave: Case-matched no-Wave k-space in RO/LIN/PAR/COIL order.
        maps_base: Case-matched shared CSM BART basename.
        workers: Maximum SciPy FFT worker count.

    Returns:
        Finite complex64 direct-FFT reference image.
    """

    maps = open_cfl(maps_base)
    expected = (*no_wave.shape[:3], no_wave.shape[3], 1)
    if read_shape(maps_base)[:5] != expected:
        raise ValueError(f"CSM shape {read_shape(maps_base)} does not match {expected}.")
    numerator = np.zeros(no_wave.shape[:3], dtype=np.complex64)
    denominator = np.zeros(no_wave.shape[:3], dtype=np.float32)
    for coil in range(no_wave.shape[3]):
        image = centered_fftn(no_wave[..., coil], axes=(0, 1, 2), inverse=True, workers=workers)
        sensitivity = np.asarray(maps[:, :, :, coil, 0, ...]).squeeze()
        numerator += np.conj(sensitivity) * image
        denominator += np.abs(sensitivity).astype(np.float32) ** 2
    support = denominator > 1e-8
    reference = np.zeros_like(numerator)
    reference[support] = numerator[support] / denominator[support]
    if not np.isfinite(reference).all() or not np.any(reference):
        raise ValueError("Complex direct-FFT reference is invalid.")
    return reference


def _gre_nifti_geometry(
    config: Mapping[str, Any], shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build GRE logical and canonical-RAS geometry without resampling.

    Args:
        config: Validated private workflow configuration.
        shape: Target logical image shape.

    Returns:
        Logical affine, canonical-RAS affine, and orientation diagnostics.
    """

    import nibabel as nib

    upstream = _load_upstream_gre_module()
    from utils.nifti_export_twix import make_nifti_affine_from_twix

    logical_affine, voxel, info = make_nifti_affine_from_twix(
        twix_file=Path(str(config["inputs"]["measured_wave_twix"])).expanduser().resolve(),
        scan_index=-1,
        npy_shape=shape,
        twix_array_axis_roles=GRE_LOGICAL_AXIS_ROLES,
        twix_array_axis_flips=GRE_AFFINE_AXIS_FLIPS,
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=True,
        twix_fov_override={"readout": 220.0, "phase": 220.0, "slice": 180.0},
    )
    del upstream
    logical_orientation = nib.orientations.io_orientation(logical_affine)
    ras_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    logical_to_ras = nib.orientations.ornt_transform(logical_orientation, ras_orientation)
    ras_to_logical = nib.orientations.ornt_transform(ras_orientation, logical_orientation)
    canonical_affine = logical_affine @ nib.orientations.inv_ornt_aff(logical_to_ras, shape)
    canonical_shape = tuple(int(shape[index]) for index in np.argsort(logical_to_ras[:, 0]))
    if tuple(nib.aff2axcodes(canonical_affine)) != ("R", "A", "S"):
        raise ValueError("GRE NIfTI geometry could not be canonicalized to RAS.")
    geometry = {
        "orientation_policy_version": GRE_ORIENTATION_POLICY_VERSION,
        "sequence_family": "GRE",
        "compatibility_scope": "GRE-specific; MPRAGE orientation policy is unchanged",
        "logical_axis_roles": list(GRE_LOGICAL_AXIS_ROLES),
        "affine_axis_flips": list(GRE_AFFINE_AXIS_FLIPS),
        "logical_shape": list(shape),
        "canonical_ras_shape": list(canonical_shape),
        "logical_affine": logical_affine.tolist(),
        "canonical_ras_affine": canonical_affine.tolist(),
        "logical_orientation_codes": list(nib.aff2axcodes(logical_affine)),
        "canonical_orientation_codes": ["R", "A", "S"],
        "logical_to_canonical_ras_transform": logical_to_ras.tolist(),
        "canonical_ras_to_logical_transform": ras_to_logical.tolist(),
        "voxel_size_mm_logical_axes": list(voxel),
        "twix_geometry": info,
        "validation": {
            "basis": "GRE source DICOM comparison",
            "opposite_all_three_physical_directions_rejected": True,
            "interpolation_during_orientation_transform": False,
        },
    }
    return logical_affine, canonical_affine, geometry


def _apply_orientation(array: np.ndarray, transform: Sequence[Sequence[float]]) -> np.ndarray:
    """Apply one lossless permutation/reversal orientation transform.

    Args:
        array: Three-dimensional logical or canonical image array.
        transform: nibabel orientation transform rows.

    Returns:
        Contiguous reoriented array with unchanged voxel values.
    """

    import nibabel as nib

    values = np.asarray(array)
    if values.ndim != 3:
        raise ValueError(f"Orientation transform requires a 3-D array, got {values.shape}.")
    oriented = nib.orientations.apply_orientation(values, np.asarray(transform, dtype=float))
    return np.ascontiguousarray(oriented)


def prepare_references(config: Mapping[str, Any], validated: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Create complex native and LR direct-FFT references for both echoes.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.

    Returns:
        Completed per-case and per-echo reference manifest.
    """

    import nibabel as nib

    source = _require_complete(root, "source")
    csm = _require_complete(root, "csm")
    output = root / "preparation" / "references"
    reusable = _reuse_completed_operation(output / "manifest.json", validated)
    if reusable is not None and reusable.get("orientation_policy_version") == GRE_ORIENTATION_POLICY_VERSION:
        return reusable
    workers = int(config["runtime"]["fft_workers"])
    records = {}
    for case_id, case in case_definitions().items():
        maps_record = csm["lin_low_resolution"] if case_id == "lin_low_resolution_r3x2" else csm["native"]
        maps_base = Path(maps_record["base"])
        _, canonical_affine, geometry = _gre_nifti_geometry(config, case.matrix_ro_lin_par)
        case_echoes = {}
        for echo_index, echo_id in enumerate(ECHO_IDS):
            native = np.load(source["echoes"][echo_index]["path"], mmap_mode="r", allow_pickle=False)
            no_wave = crop_native_for_case(native, case)
            reference = _sense_direct_reference(no_wave, maps_base, workers=workers)
            echo_dir = output / case_id / echo_id
            echo_dir.mkdir(parents=True, exist_ok=True)
            complex_path = echo_dir / "direct_fft_complex.npy"
            np.save(complex_path, reference, allow_pickle=False)
            magnitude_path = echo_dir / "direct_fft_magnitude.nii.gz"
            phase_path = echo_dir / "direct_fft_phase.nii.gz"
            magnitude_ras = _apply_orientation(
                np.abs(reference).astype(np.float32),
                geometry["logical_to_canonical_ras_transform"],
            )
            phase_ras = _apply_orientation(
                np.angle(reference).astype(np.float32),
                geometry["logical_to_canonical_ras_transform"],
            )
            nib.save(nib.Nifti1Image(magnitude_ras, canonical_affine), magnitude_path)
            nib.save(nib.Nifti1Image(phase_ras, canonical_affine), phase_path)
            case_echoes[echo_id] = {
                "echo": echo_index + 1,
                "te_s": ECHO_TIMES_S[echo_index],
                "complex": file_identity(complex_path),
                "magnitude_nifti": file_identity(magnitude_path),
                "phase_nifti": file_identity(phase_path),
                "nifti_orientation": geometry,
            }
        records[case_id] = {"geometry": case.to_json(), "nifti_geometry": geometry, "echoes": case_echoes}
    manifest = {
        "format_version": 1,
        "status": "complete",
        "operation": "references",
        "created_utc": utc_now(),
        "config_sha256": validated["config_sha256"],
        "orientation_policy_version": GRE_ORIENTATION_POLICY_VERSION,
        "orientation_policy": {
            "sequence_family": "GRE",
            "logical_axis_roles": list(GRE_LOGICAL_AXIS_ROLES),
            "affine_axis_flips": list(GRE_AFFINE_AXIS_FLIPS),
            "stored_nifti_orientation": "RAS",
            "voxel_reordering": "lossless permutation and reversal without interpolation",
            "mprage_policy_modified": False,
        },
        "combination": "complex SENSE combination with one shared echo-1-derived CSM",
        "cases": records,
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _dilate_mask_anterior(mask: np.ndarray) -> np.ndarray:
    """Expand a canonical-RAS mask by exactly one voxel toward anterior.

    Args:
        mask: Three-dimensional binary mask whose axis 1 increases toward anterior.

    Returns:
        Boolean mask containing the original support and its one-voxel A shift.
    """

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("Anterior mask dilation requires a three-dimensional mask.")
    expanded = values.copy()
    expanded[:, 1:, :] |= values[:, :-1, :]
    return expanded


def prepare_brain_mask_candidate(
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    root: Path,
    *,
    source_manifest_path: Path | None,
) -> dict[str, Any]:
    """Map a previously approved same-subject mask to the GRE reference.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.
        source_manifest_path: Manifest for the previously reviewed MPRAGE mask.

    Returns:
        Candidate manifest requiring explicit visual approval on the GRE image.
    """

    import nibabel as nib
    from nibabel.processing import resample_from_to
    from prepare_reference_brain_mask import make_mask_qc

    references = _require_complete(root, "references")
    if references.get("orientation_policy_version") != GRE_ORIENTATION_POLICY_VERSION:
        raise ValueError("References use the legacy GRE orientation; rerun the references stage first.")
    orientation = references["cases"]["native_r3x1"]["nifti_geometry"]
    input_path = Path(references["cases"]["native_r3x1"]["echoes"]["echo-01"]["magnitude_nifti"]["path"])
    output = root / "preparation" / "brain_mask"
    if source_manifest_path is None:
        raise ValueError("prepare-brain-mask requires --brain-mask-source-manifest.")
    source_manifest_path = source_manifest_path.expanduser().resolve()
    reusable = _reuse_completed_operation(
        output / "manifest.json",
        validated,
        allowed_statuses=("candidate_awaiting_user_approval", "complete"),
    )
    if (
        reusable is not None
        and reusable.get("candidate_grid_version") == GRE_BRAIN_MASK_CANDIDATE_GRID_VERSION
        and reusable.get("source_approved_mask_manifest", {}).get("path")
        == str(source_manifest_path)
    ):
        return reusable
    source_manifest = load_json(source_manifest_path, "previously approved brain-mask manifest")
    approval = source_manifest.get("approval", {})
    if (
        source_manifest.get("status") != "approved_for_metrics"
        or approval.get("mask_boundary_visually_approved") is not True
        or approval.get("left_right_orientation_visually_approved") is not True
    ):
        raise ValueError("The source brain-mask manifest does not record boundary and LR approval.")
    source_record = source_manifest.get("brain_mask", {})
    recorded_source = Path(str(source_record.get("path", ""))).expanduser()
    relocated_source = source_manifest_path.parent / recorded_source.name
    source_candidates = [relocated_source, recorded_source]
    source_mask_path = next((path for path in source_candidates if path.is_file()), None)
    if source_mask_path is None:
        raise FileNotFoundError(
            f"Approved source mask was not found at either {recorded_source} or {relocated_source}."
        )
    recorded_hash = source_record.get("sha256")
    if not isinstance(recorded_hash, str) or sha256_file(source_mask_path) != recorded_hash:
        raise ValueError("The approved source mask does not match its manifest SHA-256.")

    source_image = nib.load(str(input_path))
    if tuple(nib.aff2axcodes(source_image.affine)) != ("R", "A", "S"):
        raise ValueError("GRE mask target must use the validated canonical RAS orientation.")
    prior_mask_image = nib.load(str(source_mask_path))
    if tuple(nib.aff2axcodes(prior_mask_image.affine)) != ("R", "A", "S"):
        raise ValueError("Previously approved source mask must use canonical RAS orientation.")
    prior_values = np.asarray(prior_mask_image.dataobj)
    if (
        prior_values.ndim != 3
        or not np.isfinite(prior_values).all()
        or not np.all((prior_values == 0) | (prior_values == 1))
        or not np.any(prior_values)
    ):
        raise ValueError("Previously approved source mask must be a nonempty finite binary 3D image.")

    magnitude = np.asarray(source_image.dataobj)
    voxel_volume_ml = float(np.prod(source_image.header.get_zooms()[:3]) / 1000.0)
    mapped_image = resample_from_to(
        prior_mask_image,
        (source_image.shape, source_image.affine),
        order=0,
        mode="constant",
        cval=0.0,
    )
    base_mask = np.asarray(mapped_image.dataobj) > 0.5
    base_mask_voxels = int(base_mask.sum())
    if (
        base_mask.shape != NATIVE_MATRIX
        or base_mask_voxels <= 0
        or base_mask_voxels >= base_mask.size
    ):
        raise ValueError("Resampled prior mask has invalid GRE geometry or support.")
    source_volume_ml = int(np.count_nonzero(prior_values)) * float(
        np.prod(prior_mask_image.header.get_zooms()[:3]) / 1000.0
    )
    base_volume_ml = base_mask_voxels * voxel_volume_ml
    relative_volume_change = base_volume_ml / source_volume_ml - 1.0
    if not math.isfinite(relative_volume_change) or abs(relative_volume_change) > 0.05:
        raise ValueError(
            "Resampling changed approved-mask physical volume by more than 5%; "
            "verify cross-sequence geometry before proceeding."
        )
    anterior_d1_mask = _dilate_mask_anterior(base_mask)
    candidate_specs = (
        (
            GRE_PRIOR_MASK_CANDIDATE_ID,
            base_mask,
            {"kind": "none"},
        ),
        (
            GRE_PRIOR_MASK_ANTERIOR_D1_CANDIDATE_ID,
            anterior_d1_mask,
            {
                "kind": "directional_dilation",
                "direction": "anterior in canonical RAS",
                "distance_voxels": 1,
                "distance_mm": float(source_image.header.get_zooms()[1]),
                "shift_axis_only": True,
            },
        ),
    )
    candidates = {}
    for candidate_id, mask, postprocessing in candidate_specs:
        candidate_dir = output / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        mask_path = candidate_dir / "brain_mask_resampled_to_gre_ras.nii.gz"
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), source_image.affine), mask_path)
        qc_path = output / "qc" / f"brain_mask_candidate_{candidate_id}_overlay.png"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc = make_mask_qc(magnitude, mask, qc_path)
        mask_voxels = int(mask.sum())
        mapped_volume_ml = mask_voxels * voxel_volume_ml
        boundary_counts = {
            "x_min_left": int(mask[0, :, :].sum()),
            "x_max_right": int(mask[-1, :, :].sum()),
            "y_min_posterior": int(mask[:, 0, :].sum()),
            "y_max_anterior": int(mask[:, -1, :].sum()),
            "z_min_inferior": int(mask[:, :, 0].sum()),
            "z_max_superior": int(mask[:, :, -1].sum()),
        }
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "method": "nearest-neighbor physical-space resampling of an approved same-subject MPRAGE mask",
            "postprocessing": postprocessing,
            "candidate_mask": {
                **file_identity(mask_path),
                "voxel_count": mask_voxels,
                "mask_fraction": float(mask.mean()),
                "volume_ml": mapped_volume_ml,
                "boundary_voxel_counts": boundary_counts,
            },
            "source_mask_volume_ml": source_volume_ml,
            "relative_volume_change": mapped_volume_ml / source_volume_ml - 1.0,
            "qc_figure": {**file_identity(qc_path), **qc},
        }
    manifest = {
        "format_version": 1,
        "status": "candidate_awaiting_user_approval",
        "operation": "brain_mask",
        "created_utc": utc_now(),
        "config_sha256": validated["config_sha256"],
        "candidate_grid_version": GRE_BRAIN_MASK_CANDIDATE_GRID_VERSION,
        "candidate_ids": list(candidates),
        "candidates": candidates,
        "candidate_method": {
            "name": "approved_same_subject_mprage_mask_resampled_to_gre",
            "interpolation": "nearest neighbor in scanner physical coordinates",
            "original_reference_modified": False,
            "source_mask_modified": False,
            "ordinary_bet_threshold_sweep_rejected": True,
            "fsl_bet_B_rejected": True,
            "fsl_bet_B_rejection_reason": (
                "BET -B crashed in its recursive robust-center pass and FAST returned an essentially unity bias field."
            ),
        },
        "input": file_identity(input_path),
        "source_approved_mask_manifest": file_identity(source_manifest_path),
        "source_approved_mask": file_identity(source_mask_path),
        "source_approval": approval,
        "orientation": orientation,
        "qc_implementation": file_identity(SCRIPT_ROOT / "prepare_reference_brain_mask.py"),
        "approved": False,
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def approve_brain_mask(
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    root: Path,
    *,
    reviewer: str,
    decision: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Record the user's visual decision and freeze an approved mask.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.
        reviewer: Nonempty reviewer identity recorded in provenance.
        decision: Explicit approval decision.
        candidate_id: Explicit reviewed brain-mask candidate identifier.

    Returns:
        Completed approved-mask manifest.
    """

    import nibabel as nib

    manifest_path = root / "preparation" / "brain_mask" / "manifest.json"
    manifest = load_json(manifest_path, "brain mask candidate manifest")
    if manifest.get("status") != "candidate_awaiting_user_approval":
        raise ValueError("Brain mask is not awaiting a user decision.")
    if decision != "approve" or not reviewer.strip():
        raise ValueError("Only an explicit approve decision with reviewer identity can freeze the mask.")
    if manifest.get("candidate_grid_version") != GRE_BRAIN_MASK_CANDIDATE_GRID_VERSION:
        raise ValueError("Brain-mask approval requires the current reviewed candidate grid.")
    if candidate_id not in manifest.get("candidates", {}):
        raise ValueError(f"Select one reviewed brain-mask candidate from {manifest.get('candidate_ids', [])}.")
    selected = manifest["candidates"][candidate_id]
    source = Path(selected["candidate_mask"]["path"])
    if sha256_file(source) != selected["candidate_mask"]["sha256"]:
        raise ValueError("Brain mask candidate changed after QC generation.")
    orientation = manifest["orientation"]
    source_image = nib.load(str(source))
    source_values = (np.asarray(source_image.dataobj) > 0.5).astype(np.uint8)
    if tuple(nib.aff2axcodes(source_image.affine)) != ("R", "A", "S"):
        raise ValueError("Approved brain-mask candidate must be stored in canonical RAS orientation.")
    if list(source_values.shape) != orientation["canonical_ras_shape"]:
        raise ValueError("Brain-mask candidate shape differs from the recorded canonical-RAS geometry.")
    logical_values = _apply_orientation(
        source_values,
        orientation["canonical_ras_to_logical_transform"],
    ).astype(np.uint8, copy=False)
    if logical_values.shape != NATIVE_MATRIX or int(logical_values.sum()) != int(source_values.sum()):
        raise ValueError("Lossless brain-mask conversion back to GRE logical space failed.")
    approved_dir = manifest_path.parent / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True)
    approved_ras = approved_dir / "brain_mask_native_ras.nii.gz"
    approved_logical = approved_dir / "brain_mask_native_logical.nii.gz"
    shutil.copy2(source, approved_ras)
    nib.save(
        nib.Nifti1Image(logical_values, np.asarray(orientation["logical_affine"], dtype=float)),
        approved_logical,
    )
    manifest.update(
        {
            "status": "complete",
            "approved": True,
            "approval": {
                "decision": decision,
                "reviewer": reviewer,
                "candidate_id": candidate_id,
                "candidate_method": selected["method"],
                "candidate_postprocessing": selected["postprocessing"],
                "reviewed_qc_figure": selected["qc_figure"],
                "recorded_utc": utc_now(),
            },
            "approved_mask": {
                **file_identity(approved_logical),
                "array_space": "GRE logical RO/LIN/PAR",
                "shape": list(logical_values.shape),
                "voxel_count": int(logical_values.sum()),
            },
            "approved_mask_canonical_ras": {
                **file_identity(approved_ras),
                "array_space": "canonical RAS",
                "shape": list(source_values.shape),
                "voxel_count": int(source_values.sum()),
            },
        }
    )
    write_json_atomic(manifest_path, manifest)
    return manifest


def _map_mask_to_lr(native_path: Path, output_path: Path) -> dict[str, Any]:
    """Map the frozen native mask to LR using affine-aware nearest neighbors.

    Args:
        native_path: Approved native-grid binary mask NIfTI.
        output_path: Destination LIN-low-resolution mask NIfTI.

    Returns:
        Hash, shape, and voxel-count record for the mapped mask.
    """

    import nibabel as nib
    from nibabel.processing import resample_from_to

    source = nib.load(str(native_path))
    target_shape = (250, 148, 72)
    target_affine = source.affine.copy()
    old_vector = source.affine[:3, 1].copy()
    old_center = source.affine @ np.array([(250 - 1) / 2, (250 - 1) / 2, (72 - 1) / 2, 1.0])
    target_affine[:3, 1] = old_vector * (250.0 / 148.0)
    target_center_index = np.array([(250 - 1) / 2, (148 - 1) / 2, (72 - 1) / 2])
    target_affine[:3, 3] = old_center[:3] - target_affine[:3, :3] @ target_center_index
    mapped = resample_from_to(source, (target_shape, target_affine), order=0)
    values = (np.asarray(mapped.dataobj) > 0.5).astype(np.uint8)
    nib.save(nib.Nifti1Image(values, target_affine), output_path)
    return {**file_identity(output_path), "shape": list(values.shape), "voxel_count": int(values.sum())}


def prepare_cases(config: Mapping[str, Any], validated: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Generate case-matched pure-mask synthetic-Wave data and bind inputs.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.

    Returns:
        Completed manifest indexing all six prepared case/echo inputs.
    """

    source = _require_complete(root, "source")
    operator = _require_complete(root, "theoretical_operator")
    operator_validation = _require_complete(root, "operator_validation")
    csm = _require_complete(root, "csm")
    references = _require_complete(root, "references")
    brain = _require_complete(root, "brain_mask")
    metadata_path = root / "metadata" / "manifest.json"
    metadata = load_json(metadata_path, "metadata manifest")
    if metadata.get("status") != "complete":
        raise ValueError("Case preparation requires completed metadata validation.")
    run_manifest_path = root / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    bart = shutil.which(str(config["runtime"]["bart"]))
    if bart is None:
        raise FileNotFoundError(f"BART executable not found: {config['runtime']['bart']}")
    provenance = {
        "run_manifest": file_identity(run_manifest_path),
        "metadata_manifest": file_identity(metadata_path),
        "source_manifest": file_identity(stage_manifest_path(root, "source")),
        "operator_manifest": file_identity(stage_manifest_path(root, "theoretical_operator")),
        "operator_validation_manifest": file_identity(stage_manifest_path(root, "operator_validation")),
        "csm_manifest": file_identity(stage_manifest_path(root, "csm")),
        "reference_manifest": file_identity(stage_manifest_path(root, "references")),
        "brain_mask_manifest": file_identity(stage_manifest_path(root, "brain_mask")),
        "bart": _bart_identity(bart),
        "implementation": {
            "scientific_contract": file_identity(SCRIPT_ROOT / "gre_synthetic_wave.py"),
            "workflow": file_identity(Path(__file__).resolve()),
        },
    }
    if brain.get("approved") is not True:
        raise ValueError("Case preparation requires an explicitly approved BET mask.")
    output = root / "preparation" / "cases"
    reusable = _reuse_completed_operation(output / "manifest.json", validated)
    if reusable is not None:
        return reusable
    masks_dir = root / "preparation" / "brain_mask" / "mapped_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    native_mask_path = Path(brain["approved_mask"]["path"])
    lr_mask_record = _map_mask_to_lr(native_mask_path, masks_dir / "brain_mask_lin_low_resolution.nii.gz")
    case_records = {}
    for case_id, case in case_definitions().items():
        pure_mask, mask_metadata = build_case_mask(case)
        validate_pure_cartesian_image_lattice(pure_mask, mask_metadata)
        case_echoes = {}
        for echo_index, echo_id in enumerate(ECHO_IDS):
            native = np.load(source["echoes"][echo_index]["path"], mmap_mode="r", allow_pickle=False)
            no_wave = crop_native_for_case(native, case)
            psf_record = operator["echoes"][echo_index]["cases"][case_id]
            psf_base = Path(psf_record["base"])
            psf = np.asarray(open_cfl(psf_base))[..., 0, 0]
            echo_dir = output / case_id / echo_id
            echo_dir.mkdir(parents=True, exist_ok=True)
            mask_path = echo_dir / "sampling_mask.npy"
            np.save(mask_path, pure_mask, allow_pickle=False)
            wave_base = echo_dir / "wave_kspace"
            target = create_cfl(
                wave_base,
                (EXTENDED_READOUT, case.matrix_ro_lin_par[1], case.matrix_ro_lin_par[2], VIRTUAL_COILS, 1),
            )
            norm_squared = 0.0
            acquired_mismatch = 0
            outside_nonzero = 0
            for coil in range(VIRTUAL_COILS):
                full = synthesize_wave_from_no_wave_crop(
                    no_wave[..., coil],
                    psf,
                    readout_oversampled=EXTENDED_READOUT,
                    target_mask=None,
                    fft_workers=int(config["runtime"]["fft_workers"]),
                )
                masked, checks = apply_sampling_mask(full, pure_mask)
                if not all(checks.values()):
                    raise ValueError(f"{case_id}/{echo_id} sampling checks failed.")
                target[..., coil, 0] = masked
                acquired_mismatch += int(np.count_nonzero(masked[:, pure_mask] != full[:, pure_mask]))
                outside_nonzero += int(np.count_nonzero(masked[:, ~pure_mask]))
                norm_squared += float(np.vdot(masked, masked).real)
            target.flush()
            del target
            wave_record = cfl_record(wave_base)
            wave_record.update(
                {
                    "l2_norm": math.sqrt(norm_squared),
                    "acquired_mismatch_count": acquired_mismatch,
                    "unacquired_nonzero_count": outside_nonzero,
                }
            )
            maps_record = csm["lin_low_resolution"] if case_id == "lin_low_resolution_r3x2" else csm["native"]
            manifest = {
                "format_version": 1,
                "status": "complete",
                "case_id": case_id,
                "echo_id": echo_id,
                "echo": echo_index + 1,
                "te_s": ECHO_TIMES_S[echo_index],
                "geometry": case.to_json(),
                "sampling_mask": {**mask_metadata, "path": str(mask_path), "file_sha256": sha256_file(mask_path)},
                "bart_inputs": {"maps": maps_record, "psf": psf_record, "wave_kspace": wave_record},
                "direct_fft_reference": references["cases"][case_id]["echoes"][echo_id],
                "brain_mask": brain["approved_mask"] if case_id != "lin_low_resolution_r3x2" else lr_mask_record,
                "provenance": provenance,
                "calibration_samples_merged_into_wave_kspace": False,
                "measured_wave_samples_used": False,
            }
            case_manifest = echo_dir / "manifest.json"
            write_json_atomic(case_manifest, manifest)
            case_echoes[echo_id] = {"path": str(case_manifest), "sha256": sha256_file(case_manifest)}
        case_records[case_id] = case_echoes
    manifest = {
        "format_version": 1,
        "status": "complete",
        "operation": "cases",
        "created_utc": utc_now(),
        "config_sha256": validated["config_sha256"],
        "provenance": provenance,
        "cases": case_records,
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _dicom_echo_time_ms(dataset: Any) -> float:
    """Find one effective echo time recursively in a DICOM dataset.

    Args:
        dataset: pydicom dataset or nested sequence item.

    Returns:
        Effective echo time in milliseconds.
    """

    direct = getattr(dataset, "EffectiveEchoTime", None)
    if direct is None:
        direct = getattr(dataset, "EchoTime", None)
    if direct is not None:
        return float(direct)
    for element in dataset:
        if element.VR == "SQ":
            for item in element.value:
                try:
                    return _dicom_echo_time_ms(item)
                except ValueError:
                    pass
    raise ValueError("DICOM object contains no effective echo time.")


def _dicom_keyword_values(dataset: Any, keyword: str) -> list[Any]:
    """Collect values for one DICOM keyword through all nested sequences.

    Args:
        dataset: pydicom dataset or nested sequence item.
        keyword: Standard DICOM keyword to collect.

    Returns:
        Values in traversal order.
    """

    values = []
    for element in dataset:
        if element.keyword == keyword:
            values.append(element.value)
        if element.VR == "SQ":
            for item in element.value:
                values.extend(_dicom_keyword_values(item, keyword))
    return values


def inspect_metadata(config: Mapping[str, Any], validated: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Validate all metadata without mixing measured-Wave image data.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.

    Returns:
        Completed source/measured-Wave/DICOM/sequence metadata manifest.
    """

    import glob
    import mapvbvd
    import pydicom

    output = root / "metadata"
    output.mkdir(parents=True, exist_ok=True)
    reusable = _reuse_completed_operation(output / "manifest.json", validated)
    if reusable is not None:
        return reusable
    source_path = Path(str(config["inputs"]["source_twix"])).expanduser().resolve()
    _, _, _, source_metadata = _open_source_twix(source_path)
    measured_path = Path(str(config["inputs"]["measured_wave_twix"])).expanduser().resolve()
    measured_root = mapvbvd.mapVBVD(str(measured_path), quiet=True)
    measurement_index, measured = _select_twix_measurement(measured_root)
    measured_image = measured.image
    measured_image.flagRemoveOS = True
    measured_image.squeeze = True
    measured_shape = tuple(int(value) for value in measured_image.sqzSize)
    if measured_shape != (500, 44, 249, 72, 2):
        raise ValueError(f"Measured-Wave image layout changed: {measured_shape}.")
    echo_values = sorted({int(value) for value in measured_image.Eco})
    if echo_values != [0, 1]:
        raise ValueError(f"Measured-Wave Eco counters changed: {echo_values}.")
    measured_metadata = {
        "measurement_index": measurement_index,
        "mapvbvd_shape_after_remove_os": list(measured_shape),
        "header_logical_matrix_ro_lin_par": [250, 250, 72],
        "observed_image_lin_indices": sorted({int(value) for value in measured_image.Lin}),
        "observed_image_par_indices": sorted({int(value) for value in measured_image.Par}),
        "observed_echo_counters": echo_values,
        "role": "geometry, metadata, orientation, and qualitative validation only",
        "image_samples_used": False,
        "refscan_or_acs_samples_used": False,
        "psf_coefficients_used": False,
    }
    measured_yaps = measured.hdr["MeasYaps"]
    measured_te_s = [
        float(_twix_value(measured_yaps, ("alTE", str(index)), math.nan)) * 1e-6
        for index in range(2)
    ]
    if not np.allclose(measured_te_s, ECHO_TIMES_S, rtol=0.0, atol=1e-9):
        raise ValueError(f"Measured-Wave echo times changed: {measured_te_s}.")
    measured_metadata.update(
        {
            "echo_times_s": measured_te_s,
            "twix_header_matrix_base_phase_partition": [
                int(_twix_value(measured_yaps, ("sKSpace", "lBaseResolution"), -1)),
                int(_twix_value(measured_yaps, ("sKSpace", "lPhaseEncodingLines"), -1)),
                int(_twix_value(measured_yaps, ("sKSpace", "lPartitions"), -1)),
            ],
            "nominal_fov_mm_ro_lin_par": [
                float(_twix_value(measured_yaps, ("sSliceArray", "asSlice", "0", "dReadoutFOV"), math.nan)),
                float(_twix_value(measured_yaps, ("sSliceArray", "asSlice", "0", "dPhaseFOV"), math.nan)),
                float(_twix_value(measured_yaps, ("sSliceArray", "asSlice", "0", "dThickness"), math.nan)),
            ],
        }
    )
    if not np.allclose(measured_metadata["nominal_fov_mm_ro_lin_par"], NATIVE_FOV_MM, rtol=0.0, atol=1e-6):
        raise ValueError("Measured-Wave nominal FOV does not match the approved geometry.")
    dicom_paths: list[Path] = []
    for pattern in config["inputs"]["dicom_globs"]:
        dicom_paths.extend(Path(value).resolve() for value in glob.glob(str(pattern)))
    dicom_records = []
    for path in sorted(set(dicom_paths)):
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        echo_time_ms = _dicom_echo_time_ms(dataset)
        if not any(math.isclose(echo_time_ms, expected, abs_tol=1e-6) for expected in (10.0, 20.0)):
            raise ValueError(f"Unexpected DICOM echo time {echo_time_ms} ms in {path}.")
        record = {
            "path": str(path),
            "sop_instance_uid": str(dataset.SOPInstanceUID),
            "series_instance_uid": str(dataset.SeriesInstanceUID),
            "series_number": int(dataset.SeriesNumber),
            "echo_time_ms": echo_time_ms,
            "echo": 1 if math.isclose(echo_time_ms, 10.0, abs_tol=1e-6) else 2,
            "rows": int(dataset.Rows),
            "columns": int(dataset.Columns),
            "number_of_frames": int(dataset.NumberOfFrames),
        }
        pixel_spacings = _dicom_keyword_values(dataset, "PixelSpacing")
        slice_thicknesses = _dicom_keyword_values(dataset, "SliceThickness")
        orientations = _dicom_keyword_values(dataset, "ImageOrientationPatient")
        positions = _dicom_keyword_values(dataset, "ImagePositionPatient")
        if not pixel_spacings or not slice_thicknesses or not orientations or not positions:
            raise ValueError(f"DICOM geometry metadata is incomplete: {path}")
        unique_pixel_spacing = sorted({tuple(float(value) for value in item) for item in pixel_spacings})
        unique_slice_thickness = sorted({float(value) for value in slice_thicknesses})
        unique_orientation = sorted({tuple(float(value) for value in item) for item in orientations})
        position_values = [tuple(float(value) for value in item) for item in positions]
        record.update(
            {
                "pixel_spacing_mm": [list(value) for value in unique_pixel_spacing],
                "slice_thickness_mm": unique_slice_thickness,
                "image_orientation_patient_lps": [list(value) for value in unique_orientation],
                "image_position_patient_lps_first": list(position_values[0]),
                "image_position_patient_lps_last": list(position_values[-1]),
                "image_position_count": len(position_values),
            }
        )
        if (record["rows"], record["columns"], record["number_of_frames"]) != (256, 256, 72):
            raise ValueError(f"Unexpected DICOM matrix in {path}: {record}.")
        for orientation in unique_orientation:
            row = np.asarray(orientation[:3], dtype=float)
            column = np.asarray(orientation[3:], dtype=float)
            normal = np.cross(row, column)
            if not np.allclose(np.abs(normal), [0.0, 0.0, 1.0], rtol=0.0, atol=1e-6):
                raise ValueError(f"DICOM reference is not transverse: {path}")
        if unique_pixel_spacing != [(0.859375, 0.859375)] or unique_slice_thickness != [2.5]:
            raise ValueError(f"Unexpected DICOM voxel spacing in {path}.")
        if len(position_values) != 72:
            raise ValueError(f"DICOM reference does not contain 72 frame positions: {path}")
        dicom_records.append(record)
    if not dicom_records or {record["echo"] for record in dicom_records} != {1, 2}:
        raise ValueError("DICOM metadata did not provide both 10 ms and 20 ms echo references.")
    sequence_path = Path(str(config["inputs"]["sequence"])).expanduser().resolve()
    manifest = {
        "format_version": 1,
        "status": "complete",
        "operation": "metadata",
        "created_utc": utc_now(),
        "config_sha256": validated["config_sha256"],
        "source_twix": {**file_identity(source_path), "metadata": source_metadata},
        "measured_wave_twix": {**file_identity(measured_path), "metadata": measured_metadata},
        "sequence": file_identity(sequence_path),
        "dicom_references": dicom_records,
        "echo_mapping": [
            {"echo": 1, "te_s": 0.010, "twix_eco_counter": 0},
            {"echo": 2, "te_s": 0.020, "twix_eco_counter": 1},
        ],
        "delta_te_s": 0.010,
        "geometry": validated["geometry"],
    }
    write_json_atomic(output / "manifest.json", manifest)
    repository = {
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "submodules": subprocess.run(
            ["git", "submodule", "status", "--recursive"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
    }
    run_manifest = {
        "format_version": 1,
        "status": "initialized",
        "workflow": config["workflow"],
        "run_name": validated["run_name"],
        "run_root": str(root),
        "created_utc": utc_now(),
        "config": {
            "path": validated["config_path"],
            "sha256": validated["config_sha256"],
            "snapshot": config,
        },
        "repository": repository,
        "metadata_manifest": {
            "path": str(output / "manifest.json"),
            "sha256": sha256_file(output / "manifest.json"),
        },
        "production_execution_by_agent": False,
    }
    write_json_atomic(root / "run_manifest.json", run_manifest)
    return manifest


def _bart_identity(bart: str) -> dict[str, Any]:
    """Record the exact BART executable, hash, and reported version.

    Args:
        bart: Resolved BART executable path.

    Returns:
        Executable path/hash and version command output.
    """

    executable = Path(bart).resolve()
    process = subprocess.run(
        [str(executable), "version"], capture_output=True, text=True, check=False
    )
    version_text = (process.stdout + process.stderr).strip()
    return {
        "path": str(executable),
        "sha256": sha256_file(executable),
        "version_output": version_text,
        "version_return_code": process.returncode,
    }


def _validated_case_manifests(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load all six hash-bound prepared case/echo manifests.

    Args:
        root: Confirmed experiment run root.

    Returns:
        Mapping from case/echo keys to paths and validated manifests.
    """

    preparation = _require_complete(root, "cases")
    result = {}
    for case_id in CASE_IDS:
        for echo_id in ECHO_IDS:
            record = preparation["cases"][case_id][echo_id]
            path = Path(record["path"])
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"Prepared case manifest changed: {path}")
            manifest = load_json(path, f"{case_id}/{echo_id} manifest")
            if manifest.get("status") != "complete":
                raise ValueError(f"Prepared case is not complete: {case_id}/{echo_id}")
            mask = np.load(manifest["sampling_mask"]["path"], allow_pickle=False)
            validate_pure_cartesian_image_lattice(mask, manifest["sampling_mask"])
            for input_record in manifest["bart_inputs"].values():
                base = Path(input_record["base"])
                if sha256_file(base.with_suffix(".cfl")) != input_record["payload_sha256"]:
                    raise ValueError(f"Prepared BART input changed: {base}")
            result[(case_id, echo_id)] = {"path": path, "manifest": manifest}
    return result


def _fine_settings(root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Load user-recorded fine settings bound to coarse results.

    Args:
        root: Confirmed experiment run root.

    Returns:
        Explicit fine settings for every case/echo group.
    """

    path = root / "selections" / "refinement_request.json"
    document = load_json(path, "refinement request")
    if document.get("status") != "approved_for_fine_sweep":
        raise ValueError("Fine sweep requires an approved refinement request.")
    result = {}
    for case_id in CASE_IDS:
        for echo_id in ECHO_IDS:
            settings = document["groups"][case_id][echo_id]
            result[(case_id, echo_id)] = [dict(value) for value in settings]
    return result


def _parse_bart_log(path: Path) -> dict[str, Any]:
    """Extract available optimizer diagnostics from one BART log.

    Args:
        path: Combined BART stdout/stderr log.

    Returns:
        Parsed eigenvalue, timing, iteration, and convergence fields.
    """

    text = path.read_text(encoding="utf-8")
    diagnostics: dict[str, Any] = {"convergence_reported": False}
    patterns = {
        "maximum_eigenvalue": r"Max eval:\s*([0-9.eE+-]+)",
        "internal_reconstruction_seconds": r"Reconstruction time:\s*([0-9.eE+-]+) seconds",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            diagnostics[key] = float(match.group(1))
    iteration_matches = re.findall(r"(?:iter(?:ation)?)[^0-9]*([0-9]+)", text, flags=re.IGNORECASE)
    diagnostics["last_reported_iteration"] = int(iteration_matches[-1]) if iteration_matches else None
    diagnostics["convergence_reported"] = bool(re.search(r"converged", text, flags=re.IGNORECASE))
    return diagnostics


def run_sweep(
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    root: Path,
    *,
    sweep: str,
    resume: bool,
    validate_only: bool,
) -> dict[str, Any]:
    """Validate or execute a coarse/fine GPU BART Wave sweep.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.
        sweep: Coarse or fine sweep identifier.
        resume: Reuse only exact hash- and command-matched completed jobs.
        validate_only: Validate job inputs/counts without launching BART.

    Returns:
        Validation summary or completed sweep manifest.
    """

    if sweep not in {"coarse", "fine"}:
        raise ValueError("Sweep must be coarse or fine.")
    cases = _validated_case_manifests(root)
    bart = shutil.which(str(config["runtime"]["bart"]))
    if bart is None:
        raise FileNotFoundError(f"BART executable not found: {config['runtime']['bart']}")
    identity = _bart_identity(bart)
    settings_by_group = (
        {key: coarse_candidate_settings() for key in cases}
        if sweep == "coarse"
        else _fine_settings(root)
    )
    expected_jobs = sum(len(values) for values in settings_by_group.values())
    if validate_only:
        return {"status": "validated_only", "sweep": sweep, "job_count": expected_jobs}
    output_root = root / "reconstructions" / sweep
    records = []
    for (case_id, echo_id), case_record in cases.items():
        case = case_record["manifest"]
        for setting in settings_by_group[(case_id, echo_id)]:
            name = candidate_name(setting)
            output = output_root / case_id / echo_id / name
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            final_base = output / "image_wave"
            partial_base = output / "image_wave_partial"
            command = build_wave_command(
                bart,
                setting,
                maps=case["bart_inputs"]["maps"]["base"],
                psf=case["bart_inputs"]["psf"]["base"],
                kspace=case["bart_inputs"]["wave_kspace"]["base"],
                output=partial_base,
            )
            signature_payload = {
                "case_manifest_sha256": sha256_file(case_record["path"]),
                "setting": setting,
                "command": command,
                "bart": identity,
                "backend": "gpu",
            }
            signature = json_sha256(signature_payload)
            if manifest_path.is_file():
                old = load_json(manifest_path, "candidate manifest")
                final_hash = (
                    sha256_file(final_base.with_suffix(".cfl"))
                    if final_base.with_suffix(".cfl").is_file()
                    else None
                )
                if resume and completed_manifest_reusable(old, signature, final_hash):
                    records.append({"path": str(manifest_path), "sha256": sha256_file(manifest_path)})
                    continue
                if final_base.with_suffix(".cfl").exists() or final_base.with_suffix(".hdr").exists():
                    raise ValueError(f"Refusing to overwrite stale completed-looking output: {final_base}")
            pending = {
                "format_version": 1,
                "status": "running",
                "sweep": sweep,
                "case_id": case_id,
                "echo_id": echo_id,
                "echo": case["echo"],
                "te_s": case["te_s"],
                "setting": setting,
                "signature_sha256": signature,
                "signature": signature_payload,
                "started_utc": utc_now(),
            }
            write_json_atomic(manifest_path, pending)
            run = _run_logged(command, output / "bart.log")
            if setting["method"] == "llr":
                recombination = recombine_split_complex_cfl(partial_base, final_base)
            else:
                partial_base.with_suffix(".hdr").replace(final_base.with_suffix(".hdr"))
                partial_base.with_suffix(".cfl").replace(final_base.with_suffix(".cfl"))
                recombination = None
            expected_shape = tuple(case["geometry"]["matrix_ro_lin_par"])
            observed = read_shape(final_base)
            if observed[:3] != expected_shape or any(value != 1 for value in observed[3:]):
                raise ValueError(f"BART output shape {observed} differs from {expected_shape}.")
            image = open_cfl(final_base)
            if not np.isfinite(image).all():
                raise ValueError(f"BART output is non-finite: {final_base}")
            pending.update(
                {
                    "status": "complete",
                    "ended_utc": utc_now(),
                    "run": run,
                    "bart_diagnostics": _parse_bart_log(Path(run["log"])),
                    "requested_maximum_iterations": 100,
                    "requested_tolerance": 1e-6,
                    "input_kspace_norm_for_restoration": case["bart_inputs"]["wave_kspace"]["l2_norm"],
                    "input_encoding_shape_for_restoration": case["bart_inputs"]["wave_kspace"]["shape"][:3],
                    "recombination": recombination,
                    "output": cfl_record(final_base),
                }
            )
            write_json_atomic(manifest_path, pending)
            records.append({"path": str(manifest_path), "sha256": sha256_file(manifest_path)})
    sweep_manifest = {
        "format_version": 1,
        "status": "complete",
        "sweep": sweep,
        "created_utc": utc_now(),
        "backend": "gpu",
        "bart": identity,
        "job_count": expected_jobs,
        "candidate_manifests": records,
    }
    write_json_atomic(output_root / "sweep_manifest.json", sweep_manifest)
    return sweep_manifest


def _load_sweep(root: Path, sweep: str) -> tuple[Path, dict[str, Any]]:
    """Load and hash-validate every candidate in a completed sweep.

    Args:
        root: Confirmed experiment run root.
        sweep: Coarse or fine sweep identifier.

    Returns:
        Sweep manifest path and parsed validated manifest.
    """

    path = root / "reconstructions" / sweep / "sweep_manifest.json"
    manifest = load_json(path, f"{sweep} sweep manifest")
    if manifest.get("status") != "complete" or manifest.get("sweep") != sweep:
        raise ValueError(f"{sweep} sweep is not complete.")
    for record in manifest["candidate_manifests"]:
        candidate_path = Path(record["path"])
        if sha256_file(candidate_path) != record["sha256"]:
            raise ValueError(f"Candidate manifest changed: {candidate_path}")
        candidate = load_json(candidate_path, "candidate manifest")
        output = Path(candidate["output"]["base"])
        if sha256_file(output.with_suffix(".cfl")) != candidate["output"]["payload_sha256"]:
            raise ValueError(f"Candidate payload changed: {output}")
    return path, manifest


def _candidate_documents(sweep: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index candidates without scanning reconstruction directories.

    Args:
        sweep: Completed sweep manifest.

    Returns:
        Mapping from case/echo/candidate keys to job manifests.
    """

    result = {}
    for record in sweep["candidate_manifests"]:
        document = load_json(Path(record["path"]), "candidate manifest")
        key = (document["case_id"], document["echo_id"], candidate_name(document["setting"]))
        if key in result:
            raise ValueError(f"Sweep repeats candidate {key}.")
        result[key] = document
    return result


def _restoration_encoding_shape(document: Mapping[str, Any]) -> tuple[int, int, int]:
    """Resolve the extended BART Wave FFT grid for one reconstruction.

    Args:
        document: Completed reconstruction job manifest.

    Returns:
        Extended-RO/LIN/PAR encoding shape used by BART ``wave``.
    """

    recorded = document.get("input_encoding_shape_for_restoration")
    if recorded is not None:
        shape = tuple(int(value) for value in recorded)
    else:
        output_shape = tuple(int(value) for value in document["output"]["shape"][:3])
        shape = (EXTENDED_READOUT, output_shape[1], output_shape[2])
    if len(shape) != 3 or shape[0] != EXTENDED_READOUT:
        raise ValueError(f"Unexpected BART Wave restoration grid: {shape}.")
    return shape


def _restoration_record(document: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the fixed scale and phase correction for one BART output.

    Args:
        document: Completed reconstruction job manifest.

    Returns:
        JSON-native restoration inputs and complex factor components.
    """

    image_shape = tuple(int(value) for value in document["output"]["shape"][:3])
    encoding_shape = _restoration_encoding_shape(document)
    kspace_norm = float(document["input_kspace_norm_for_restoration"])
    factor = bart_wave_restoration_factor(image_shape, kspace_norm, encoding_shape)
    return {
        "convention_version": GRE_BART_OUTPUT_CONVENTION_VERSION,
        "policy": (
            "restore BART input L2 normalization, unnormalized extended-grid FFT scale, "
            "and deterministic fftmod global phase"
        ),
        "kspace_l2_norm": kspace_norm,
        "image_shape_ro_lin_par": list(image_shape),
        "encoding_shape_extended_ro_lin_par": list(encoding_shape),
        "factor_real": float(factor.real),
        "factor_imaginary": float(factor.imag),
        "factor_magnitude": float(abs(factor)),
        "candidate_specific_fit": False,
    }


def _scaling_matches_current_bart_convention(
    recorded: Any, expected: Mapping[str, Any]
) -> tuple[bool, str]:
    """Compare an evaluation scaling record with the current fixed convention.

    Args:
        recorded: Scaling metadata stored for one evaluated candidate.
        expected: Scaling metadata recomputed from its reconstruction manifest.

    Returns:
        A match flag and the provenance basis. Records made immediately before
        the explicit convention marker was added are accepted only when every
        other field exactly matches the current convention.
    """

    if not isinstance(recorded, Mapping):
        return False, "invalid"
    recorded_dict = dict(recorded)
    expected_dict = dict(expected)
    if recorded_dict == expected_dict:
        return True, "explicit convention marker"
    legacy_expected = dict(expected_dict)
    legacy_expected.pop("convention_version", None)
    if "convention_version" not in recorded_dict and recorded_dict == legacy_expected:
        return True, "exact legacy scaling audit"
    return False, "mismatch"


def _audit_evaluation_bart_convention(
    evaluation: Mapping[str, Any], sweep: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove that a completed evaluation used the current BART output scaling.

    Args:
        evaluation: Completed coarse evaluation manifest.
        sweep: Hash-validated coarse sweep manifest.

    Returns:
        JSON-native audit record describing the convention and validation basis.

    Raises:
        ValueError: If metric provenance, candidate coverage, or any fixed
            restoration record does not match the current convention.
    """

    if evaluation.get("status") != "complete":
        raise ValueError("Coarse evaluation is not complete.")
    recorded_version = evaluation.get("bart_output_convention_version")
    if recorded_version not in (None, GRE_BART_OUTPUT_CONVENTION_VERSION):
        raise ValueError("Coarse evaluation records an incompatible BART output convention.")
    _verify_embedded_hashes(evaluation)
    metrics_record = evaluation.get("metrics")
    if not isinstance(metrics_record, Mapping) or not isinstance(metrics_record.get("path"), str):
        raise ValueError("Coarse evaluation does not bind its metrics file.")
    metrics = load_json(Path(metrics_record["path"]), "coarse evaluation metrics")
    records = metrics.get("per_echo")
    if not isinstance(records, list):
        raise ValueError("Coarse evaluation metrics do not contain per-echo records.")

    candidates = _candidate_documents(sweep)
    if len(records) != len(candidates):
        raise ValueError("Coarse evaluation candidate count does not match the coarse sweep.")
    seen: set[tuple[str, str, str]] = set()
    bases: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Coarse evaluation contains an invalid per-echo record.")
        key = (str(record.get("case_id")), str(record.get("echo_id")), str(record.get("candidate_name")))
        document = candidates.get(key)
        if document is None or key in seen:
            raise ValueError(f"Coarse evaluation has missing or repeated candidate provenance: {key}.")
        expected_manifest = Path(document["output"]["base"]).parent / "manifest.json"
        candidate_record = record.get("candidate_manifest")
        if (
            not isinstance(candidate_record, Mapping)
            or candidate_record.get("path") != str(expected_manifest)
        ):
            raise ValueError(f"Coarse evaluation candidate manifest does not match the sweep: {key}.")
        matches, basis = _scaling_matches_current_bart_convention(
            record.get("scaling"), _restoration_record(document)
        )
        if not matches:
            raise ValueError(f"Coarse evaluation uses stale BART output scaling for {key}.")
        bases.add(basis)
        seen.add(key)
    if seen != set(candidates):
        raise ValueError("Coarse evaluation does not cover every coarse candidate exactly once.")
    return {
        "convention_version": GRE_BART_OUTPUT_CONVENTION_VERSION,
        "validation": sorted(bases),
        "candidate_records_verified": len(seen),
        "metrics_sha256": str(metrics_record["sha256"]),
    }


def _complex_candidate(document: Mapping[str, Any]) -> np.ndarray:
    """Load one BART image and restore its internal normalization.

    Args:
        document: Completed reconstruction job manifest.

    Returns:
        Three-dimensional complex64 image on restored input scale.
    """

    values = np.asarray(open_cfl(document["output"]["base"])).squeeze()
    restored = restore_bart_normalization(
        values,
        float(document["input_kspace_norm_for_restoration"]),
        _restoration_encoding_shape(document),
    )
    if restored.ndim != 3:
        raise ValueError(f"Restored BART candidate must be 3-D, got {restored.shape}.")
    return restored


def _candidate_nifti_arrays(
    candidate: np.ndarray, orientation: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one logical complex reconstruction to viewable RAS arrays.

    Args:
        candidate: Three-dimensional complex reconstruction in GRE logical order.
        orientation: Validated logical-to-canonical-RAS geometry record.

    Returns:
        Float32 magnitude and wrapped-phase arrays in canonical RAS order.
    """

    values = np.asarray(candidate)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("NIfTI export requires one finite three-dimensional candidate.")
    canonical = _apply_orientation(
        values,
        orientation["logical_to_canonical_ras_transform"],
    )
    expected_shape = tuple(int(value) for value in orientation["canonical_ras_shape"])
    if canonical.shape != expected_shape:
        raise ValueError(
            f"Canonical candidate shape {canonical.shape} differs from {expected_shape}."
        )
    magnitude = np.abs(canonical).astype(np.float32)
    phase = np.angle(canonical).astype(np.float32)
    return magnitude, phase


def _normalize_magnitude_for_display(
    magnitude: np.ndarray, *, percentile: float = 99.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the documented MPRAGE-style positive-voxel normalization.

    Args:
        magnitude: Finite nonnegative magnitude image on restored scale.
        percentile: Positive-voxel percentile mapped to one.

    Returns:
        Float32 display magnitude and JSON-native normalization metadata.
    """

    values = np.asarray(magnitude, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Display normalization requires a finite nonnegative 3-D magnitude.")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("Display normalization percentile must lie in (0, 100].")
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("Magnitude image has no positive voxels to normalize.")
    scale = float(np.percentile(positive, percentile))
    if not math.isfinite(scale) or scale <= np.finfo(np.float32).tiny:
        raise ValueError("Magnitude display normalization produced an invalid scale.")
    normalized = np.ascontiguousarray((values / scale).astype(np.float32, copy=False))
    return normalized, {
        "Method": "positive-finite-percentile",
        "Percentile": percentile,
        "InputPercentileValue": scale,
        "OutputPercentileValue": 1.0,
        "Clipped": False,
    }


def _save_nifti_atomic(image: Any, path: Path) -> None:
    """Save one NIfTI through an adjacent temporary file and atomic rename.

    Args:
        image: Nibabel image object to serialize.
        path: Final ``.nii`` or ``.nii.gz`` output path.

    Returns:
        None. The destination is replaced only after serialization succeeds.
    """

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.nii.gz")
    try:
        import nibabel as nib

        nib.save(image, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_sweep_nifti(
    validated: Mapping[str, Any], root: Path, *, sweep_name: str
) -> dict[str, Any]:
    """Export every restored sweep candidate as magnitude and phase NIfTI.

    Args:
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.
        sweep_name: Coarse or fine sweep identifier.

    Returns:
        Completed export manifest with candidate-level image identities.
    """

    import nibabel as nib

    sweep_path, sweep = _load_sweep(root, sweep_name)
    sweep_identity = file_identity(sweep_path)
    cases = _validated_case_manifests(root)
    candidates = _candidate_documents(sweep)
    export_manifest_path = root / "reconstructions" / sweep_name / "nifti_export_manifest.json"
    if export_manifest_path.is_file():
        previous = load_json(export_manifest_path, f"{sweep_name} NIfTI export manifest")
        if (
            previous.get("status") == "complete"
            and previous.get("config_sha256") == validated["config_sha256"]
            and previous.get("export_version") == GRE_NIFTI_EXPORT_VERSION
            and previous.get("sweep_manifest", {}).get("sha256") == sweep_identity["sha256"]
        ):
            _verify_embedded_hashes(previous)
            return previous

    records = []
    for key in sorted(candidates):
        case_id, echo_id, name = key
        document = candidates[key]
        case = cases[(case_id, echo_id)]["manifest"]
        orientation = case["direct_fft_reference"]["nifti_orientation"]
        output_base = Path(document["output"]["base"])
        candidate_manifest_path = output_base.parent / "manifest.json"
        restoration = _restoration_record(document)
        signature_payload = {
            "export_version": GRE_NIFTI_EXPORT_VERSION,
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "candidate_payload_sha256": document["output"]["payload_sha256"],
            "restoration": restoration,
            "orientation": orientation,
            "components": ["magnitude", "wrapped_phase"],
            "magnitude_display_normalization": {
                "method": "positive-finite-percentile",
                "percentile": 99.0,
            },
        }
        signature = json_sha256(signature_payload)
        job_manifest_path = output_base.parent / "nifti_export_manifest.json"
        if job_manifest_path.is_file():
            previous_job = load_json(job_manifest_path, "candidate NIfTI export manifest")
            if previous_job.get("status") == "complete" and previous_job.get("signature_sha256") == signature:
                _verify_embedded_hashes(previous_job)
                records.append(file_identity(job_manifest_path))
                continue
            previous_version = previous_job.get("signature", {}).get("export_version")
            if not isinstance(previous_version, int) or previous_version >= GRE_NIFTI_EXPORT_VERSION:
                raise ValueError(f"Stale NIfTI export manifest requires review: {job_manifest_path}")

        candidate = _complex_candidate(document)
        restored_magnitude, phase = _candidate_nifti_arrays(candidate, orientation)
        magnitude, magnitude_normalization = _normalize_magnitude_for_display(
            restored_magnitude,
            percentile=99.0,
        )
        affine = np.asarray(orientation["canonical_ras_affine"], dtype=float)
        magnitude_path = output_base.parent / "image_wave_magnitude_ras.nii.gz"
        phase_path = output_base.parent / "image_wave_phase_ras.nii.gz"
        magnitude_image = nib.Nifti1Image(magnitude, affine)
        magnitude_image.header["cal_min"] = 0.0
        magnitude_image.header["cal_max"] = float(np.percentile(magnitude, 99.5))
        magnitude_image.set_qform(affine, code=1)
        magnitude_image.set_sform(affine, code=1)
        phase_image = nib.Nifti1Image(phase, affine)
        phase_image.header["cal_min"] = -math.pi
        phase_image.header["cal_max"] = math.pi
        phase_image.set_qform(affine, code=1)
        phase_image.set_sform(affine, code=1)
        _save_nifti_atomic(magnitude_image, magnitude_path)
        _save_nifti_atomic(phase_image, phase_path)
        job_manifest = {
            "format_version": 1,
            "status": "complete",
            "created_utc": utc_now(),
            "sweep": sweep_name,
            "case_id": case_id,
            "echo_id": echo_id,
            "candidate_name": name,
            "setting": document["setting"],
            "signature_sha256": signature,
            "signature": signature_payload,
            "bart_wave_restoration": restoration,
            "source_candidate_manifest": file_identity(candidate_manifest_path),
            "magnitude_nifti": {
                **file_identity(magnitude_path),
                "dtype": "float32",
                "shape": list(magnitude.shape),
                "orientation": "RAS",
                "units": "normalized arbitrary",
                "normalization": magnitude_normalization,
                "header_calibration_window": [
                    float(magnitude_image.header["cal_min"]),
                    float(magnitude_image.header["cal_max"]),
                ],
            },
            "phase_nifti": {
                **file_identity(phase_path),
                "dtype": "float32",
                "shape": list(phase.shape),
                "orientation": "RAS",
                "units": "radians",
                "range": [-math.pi, math.pi],
                "header_calibration_window": [-math.pi, math.pi],
            },
            "reference_magnitude_nifti": case["direct_fft_reference"]["magnitude_nifti"],
            "reference_phase_nifti": case["direct_fft_reference"]["phase_nifti"],
        }
        write_json_atomic(job_manifest_path, job_manifest)
        records.append(file_identity(job_manifest_path))

    manifest = {
        "format_version": 1,
        "status": "complete",
        "created_utc": utc_now(),
        "sweep": sweep_name,
        "config_sha256": validated["config_sha256"],
        "export_version": GRE_NIFTI_EXPORT_VERSION,
        "bart_output_convention_version": GRE_BART_OUTPUT_CONVENTION_VERSION,
        "sweep_manifest": sweep_identity,
        "candidate_count": len(records),
        "components": ["magnitude", "wrapped_phase"],
        "stored_orientation": "canonical RAS",
        "magnitude_display_normalization": "positive-voxel p99 mapped to 1, matching MPRAGE export convention",
        "quantitative_metrics_source": "restored complex BART CFL, not display-normalized NIfTI",
        "candidate_export_manifests": records,
    }
    write_json_atomic(export_manifest_path, manifest)
    return manifest


def _magnitude_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute magnitude fidelity and fixed-background QC metrics.

    Args:
        reference: Complex direct-FFT reference.
        candidate: Complex reconstruction on restored scale.
        mask: Approved fixed brain mask.

    Returns:
        Magnitude, gradient, background, finite, and dynamic-range metrics.
    """

    from skimage.metrics import structural_similarity

    ref = np.abs(np.asarray(reference, dtype=np.complex64)).astype(np.float64)
    cand = np.abs(np.asarray(candidate, dtype=np.complex64)).astype(np.float64)
    brain = np.asarray(mask, dtype=bool)
    if ref.shape != cand.shape or ref.shape != brain.shape or not np.any(brain):
        raise ValueError("Magnitude metric arrays and approved brain mask must match.")
    ref_values = ref[brain]
    cand_values = cand[brain]
    error = cand_values - ref_values
    reference_norm = float(np.linalg.norm(ref_values))
    data_range = float(np.percentile(ref_values, 99.5) - np.percentile(ref_values, 0.5))
    if reference_norm <= 0 or data_range <= 0:
        raise ValueError("Magnitude reference has invalid scale in the brain mask.")
    centered_ref = ref_values - np.mean(ref_values)
    centered_cand = cand_values - np.mean(cand_values)
    ncc_denominator = float(np.linalg.norm(centered_ref) * np.linalg.norm(centered_cand))
    gradient_error = 0.0
    gradient_reference = 0.0
    gradient_candidate = 0.0
    gradient_cross = 0.0
    for axis in range(3):
        ref_gradient = np.gradient(ref, axis=axis)[brain]
        cand_gradient = np.gradient(cand, axis=axis)[brain]
        difference = cand_gradient - ref_gradient
        gradient_error += float(np.vdot(difference, difference).real)
        gradient_reference += float(np.vdot(ref_gradient, ref_gradient).real)
        gradient_candidate += float(np.vdot(cand_gradient, cand_gradient).real)
        gradient_cross += float(np.vdot(ref_gradient, cand_gradient).real)
    background = ~brain
    low_reference = ref <= 0.01 * float(np.percentile(ref_values, 99))
    background &= low_reference
    if not np.any(background):
        background = ~brain
    absolute_error = np.abs(error)
    _, ssim_map = structural_similarity(ref, cand, data_range=data_range, full=True)
    return {
        "nrmse_brain": float(np.linalg.norm(error) / reference_norm),
        "ssim_brain": float(np.mean(ssim_map[brain])),
        "ncc_brain": float(np.vdot(centered_ref, centered_cand).real / ncc_denominator) if ncc_denominator > 0 else math.nan,
        "mean_bias_brain": float(np.mean(error)),
        "median_bias_brain": float(np.median(error)),
        "absolute_error_p50": float(np.percentile(absolute_error, 50)),
        "absolute_error_p90": float(np.percentile(absolute_error, 90)),
        "absolute_error_p95": float(np.percentile(absolute_error, 95)),
        "absolute_error_p99": float(np.percentile(absolute_error, 99)),
        "gradient_nrmse_brain": float(math.sqrt(gradient_error / gradient_reference)) if gradient_reference > 0 else math.nan,
        "gradient_ncc_brain": float(gradient_cross / math.sqrt(gradient_reference * gradient_candidate)) if gradient_reference > 0 and gradient_candidate > 0 else math.nan,
        "background_energy_fraction": float(np.vdot(cand[background], cand[background]).real / np.vdot(cand, cand).real),
        "finite_fraction": float(np.mean(np.isfinite(cand))),
        "candidate_maximum": float(np.max(cand)),
        "candidate_p999": float(np.percentile(cand, 99.9)),
    }


def _global_echo_metrics(
    references: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    phase_supports: Sequence[np.ndarray],
    per_echo_metrics: Sequence[Mapping[str, Any]],
    measured_kspace_norms: Sequence[float],
) -> dict[str, float]:
    """Compute energy-pooled fidelity metrics across multiple GRE echoes.

    Args:
        references: Complex direct-FFT reference volumes ordered by echo.
        candidates: Complex reconstructions matching ``references``.
        masks: Fixed brain masks for the corresponding echo grids.
        phase_supports: Reference-derived nonempty phase supports.
        per_echo_metrics: Existing convention-v2 metric records used for SSIM
            and acquired-data residual aggregation.
        measured_kspace_norms: Positive measured-data L2 norms by echo.

    Returns:
        Joint magnitude, gradient, phase, background, SSIM, and data-residual
        metrics. NRMSE denominators pool reference energy before division.

    Raises:
        ValueError: If echo counts, geometry, supports, or scales are invalid.
    """

    count = len(references)
    if count < 2 or not (
        len(candidates)
        == len(masks)
        == len(phase_supports)
        == len(per_echo_metrics)
        == len(measured_kspace_norms)
        == count
    ):
        raise ValueError("Global echo metrics require aligned inputs for at least two echoes.")

    reference_parts = []
    candidate_parts = []
    error_parts = []
    phase_error_parts = []
    gradient_error = 0.0
    gradient_reference = 0.0
    gradient_candidate = 0.0
    gradient_cross = 0.0
    background_energy = 0.0
    candidate_energy = 0.0
    residual_error_squared = 0.0
    measured_squared = 0.0
    for reference, candidate, mask, phase_support, metrics, measured_norm in zip(
        references,
        candidates,
        masks,
        phase_supports,
        per_echo_metrics,
        measured_kspace_norms,
        strict=True,
    ):
        ref_complex = np.asarray(reference, dtype=np.complex64)
        cand_complex = np.asarray(candidate, dtype=np.complex64)
        brain = np.asarray(mask, dtype=bool)
        support = np.asarray(phase_support, dtype=bool)
        if (
            ref_complex.shape != cand_complex.shape
            or ref_complex.shape != brain.shape
            or ref_complex.shape != support.shape
            or not np.any(brain)
            or not np.any(support)
            or not np.isfinite(ref_complex).all()
            or not np.isfinite(cand_complex).all()
        ):
            raise ValueError("Global echo metric arrays, masks, and supports must match and be finite.")
        if not math.isfinite(float(measured_norm)) or float(measured_norm) <= 0:
            raise ValueError("Global data-residual aggregation requires positive k-space norms.")

        ref = np.abs(ref_complex).astype(np.float64)
        cand = np.abs(cand_complex).astype(np.float64)
        ref_values = ref[brain]
        cand_values = cand[brain]
        reference_parts.append(ref_values)
        candidate_parts.append(cand_values)
        error_parts.append(cand_values - ref_values)
        phase_error_parts.append(
            np.angle(cand_complex[support] * np.conj(ref_complex[support])).astype(np.float64)
        )
        for axis in range(3):
            ref_gradient = np.gradient(ref, axis=axis)[brain]
            cand_gradient = np.gradient(cand, axis=axis)[brain]
            difference = cand_gradient - ref_gradient
            gradient_error += float(np.vdot(difference, difference).real)
            gradient_reference += float(np.vdot(ref_gradient, ref_gradient).real)
            gradient_candidate += float(np.vdot(cand_gradient, cand_gradient).real)
            gradient_cross += float(np.vdot(ref_gradient, cand_gradient).real)

        background = (~brain) & (ref <= 0.01 * float(np.percentile(ref_values, 99)))
        if not np.any(background):
            background = ~brain
        background_energy += float(np.vdot(cand[background], cand[background]).real)
        candidate_energy += float(np.vdot(cand, cand).real)
        residual = float(metrics["normalized_acquired_data_residual"])
        residual_error_squared += (residual * float(measured_norm)) ** 2
        measured_squared += float(measured_norm) ** 2

    reference_values = np.concatenate(reference_parts)
    candidate_values = np.concatenate(candidate_parts)
    errors = np.concatenate(error_parts)
    phase_errors = np.concatenate(phase_error_parts)
    reference_norm = float(np.linalg.norm(reference_values))
    centered_reference = reference_values - np.mean(reference_values)
    centered_candidate = candidate_values - np.mean(candidate_values)
    ncc_denominator = float(np.linalg.norm(centered_reference) * np.linalg.norm(centered_candidate))
    phase_vector = np.mean(np.exp(1j * phase_errors))
    absolute_errors = np.abs(errors)
    absolute_phase_errors = np.abs(phase_errors)
    if reference_norm <= 0 or ncc_denominator <= 0 or gradient_reference <= 0:
        raise ValueError("Global echo reference energy or variance is zero.")
    return {
        "global_magnitude_nrmse_brain": float(np.linalg.norm(errors) / reference_norm),
        "global_magnitude_ncc_brain": float(
            np.vdot(centered_reference, centered_candidate).real / ncc_denominator
        ),
        "mean_echo_ssim_brain": float(
            np.mean([float(metrics["ssim_brain"]) for metrics in per_echo_metrics])
        ),
        "global_magnitude_mean_bias_brain": float(np.mean(errors)),
        "global_magnitude_median_bias_brain": float(np.median(errors)),
        "global_magnitude_absolute_error_p95": float(np.percentile(absolute_errors, 95)),
        "global_gradient_nrmse_brain": float(math.sqrt(gradient_error / gradient_reference)),
        "global_gradient_ncc_brain": float(
            gradient_cross / math.sqrt(gradient_reference * gradient_candidate)
        ),
        "global_background_energy_fraction": float(background_energy / candidate_energy),
        "global_circular_mean_error_rad": float(np.angle(phase_vector)),
        "global_circular_dispersion": float(1.0 - abs(phase_vector)),
        "global_median_absolute_phase_error_rad": float(np.median(absolute_phase_errors)),
        "global_p95_absolute_phase_error_rad": float(np.percentile(absolute_phase_errors, 95)),
        "global_normalized_acquired_data_residual": float(
            math.sqrt(residual_error_squared / measured_squared)
        ),
    }


def _data_consistency(
    candidate: np.ndarray,
    case: Mapping[str, Any],
    *,
    workers: int,
) -> dict[str, float | int | bool]:
    """Evaluate the Wave forward residual on intended acquired samples.

    Args:
        candidate: Complex reconstructed image.
        case: Prepared case/echo manifest.
        workers: Maximum SciPy FFT worker count.

    Returns:
        Normalized acquired residual and outside-mask input checks.
    """

    mask = np.load(case["sampling_mask"]["path"], allow_pickle=False)
    validate_pure_cartesian_image_lattice(mask, case["sampling_mask"])
    maps = open_cfl(case["bart_inputs"]["maps"]["base"])
    psf = np.asarray(open_cfl(case["bart_inputs"]["psf"]["base"])).squeeze()
    measured = open_cfl(case["bart_inputs"]["wave_kspace"]["base"])
    error_squared = 0.0
    measured_squared = 0.0
    outside_nonzero = 0
    for coil in range(VIRTUAL_COILS):
        sensitivity = np.asarray(maps[:, :, :, coil, 0, ...]).squeeze()
        coil_image = np.asarray(candidate * sensitivity, dtype=np.complex64)
        no_wave = centered_fftn(coil_image, axes=(0, 1, 2), workers=workers)
        predicted = synthesize_wave_from_no_wave_crop(
            no_wave,
            psf,
            readout_oversampled=EXTENDED_READOUT,
            target_mask=None,
            fft_workers=workers,
        )
        observed = np.asarray(measured[:, :, :, coil, 0, ...]).squeeze()
        difference = predicted[:, mask] - observed[:, mask]
        error_squared += float(np.vdot(difference, difference).real)
        measured_squared += float(np.vdot(observed[:, mask], observed[:, mask]).real)
        outside_nonzero += int(np.count_nonzero(observed[:, ~mask]))
    if measured_squared <= 0:
        raise ValueError("Acquired Wave data have zero norm.")
    return {
        "normalized_acquired_data_residual": math.sqrt(error_squared / measured_squared),
        "input_unacquired_nonzero_count": outside_nonzero,
        "input_zero_outside_mask": outside_nonzero == 0,
    }


def evaluate_sweep(
    config: Mapping[str, Any], validated: Mapping[str, Any], root: Path, *, sweep_name: str
) -> dict[str, Any]:
    """Evaluate one manifest-backed sweep without candidate ranking.

    Args:
        config: Validated private workflow configuration.
        validated: Resolved configuration metadata.
        root: Confirmed experiment run root.
        sweep_name: Coarse or fine sweep identifier.

    Returns:
        Completed evaluation manifest with no automatic selection.
    """

    import csv
    import nibabel as nib

    sweep_path, sweep = _load_sweep(root, sweep_name)
    candidates = _candidate_documents(sweep)
    cases = _validated_case_manifests(root)
    output = root / "evaluation" / sweep_name
    output.mkdir(parents=True, exist_ok=True)
    metric_records = []
    complex_cache: dict[tuple[str, str, str], np.ndarray] = {}
    for key, document in candidates.items():
        case_id, echo_id, name = key
        case = cases[(case_id, echo_id)]["manifest"]
        reference = np.load(case["direct_fft_reference"]["complex"]["path"], allow_pickle=False)
        brain = np.asarray(nib.load(case["brain_mask"]["path"]).dataobj) > 0.5
        candidate = _complex_candidate(document)
        if candidate.shape != reference.shape or candidate.shape != brain.shape:
            raise ValueError(f"Evaluation geometry mismatch for {key}.")
        signal_threshold = 0.05 * float(np.percentile(np.abs(reference[brain]), 99))
        phase_support = brain & (np.abs(reference) >= signal_threshold)
        metrics = {
            **_magnitude_metrics(reference, candidate, brain),
            **circular_phase_metrics(reference, candidate, phase_support),
            **_data_consistency(candidate, case, workers=int(config["runtime"]["fft_workers"])),
        }
        record = {
            "case_id": case_id,
            "echo_id": echo_id,
            "echo": case["echo"],
            "te_s": case["te_s"],
            "candidate_name": name,
            "setting": document["setting"],
            "candidate_manifest": {
                "path": str(Path(document["output"]["base"]).parent / "manifest.json"),
            },
            "scaling": _restoration_record(document),
            "metrics": metrics,
        }
        metric_records.append(record)
        complex_cache[key] = candidate
    cross_echo = []
    for case_id in CASE_IDS:
        names_echo1 = {key[2] for key in candidates if key[:2] == (case_id, "echo-01")}
        names_echo2 = {key[2] for key in candidates if key[:2] == (case_id, "echo-02")}
        for name in sorted(names_echo1 & names_echo2):
            case1 = cases[(case_id, "echo-01")]["manifest"]
            case2 = cases[(case_id, "echo-02")]["manifest"]
            reference1 = np.load(case1["direct_fft_reference"]["complex"]["path"], allow_pickle=False)
            reference2 = np.load(case2["direct_fft_reference"]["complex"]["path"], allow_pickle=False)
            brain = np.asarray(nib.load(case1["brain_mask"]["path"]).dataobj) > 0.5
            threshold = 0.05 * float(np.percentile(np.abs(reference1[brain]), 99))
            support = brain & (np.abs(reference1) >= threshold) & (np.abs(reference2) >= threshold)
            cross_echo.append(
                {
                    "case_id": case_id,
                    "candidate_name": name,
                    "pairing": "same-method and same-parameter",
                    "delta_te_s": ECHO_TIMES_S[1] - ECHO_TIMES_S[0],
                    "support_voxel_count": int(np.count_nonzero(support)),
                    "excluded_low_signal_fraction_within_brain": float(
                        1.0 - np.count_nonzero(support) / np.count_nonzero(brain)
                    ),
                    "metrics": inter_echo_metrics(
                        reference1,
                        reference2,
                        complex_cache[(case_id, "echo-01", name)],
                        complex_cache[(case_id, "echo-02", name)],
                        support,
                        delta_te_s=ECHO_TIMES_S[1] - ECHO_TIMES_S[0],
                    ),
                }
            )
    metrics_path = output / "metrics.json"
    write_json_atomic(metrics_path, {"per_echo": metric_records, "same_setting_cross_echo": cross_echo})
    csv_path = output / "metrics.csv"
    metric_names = sorted({key for record in metric_records for key in record["metrics"]})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["case_id", "echo_id", "method", "lambda", "block_size", *metric_names],
        )
        writer.writeheader()
        for record in metric_records:
            writer.writerow(
                {
                    "case_id": record["case_id"],
                    "echo_id": record["echo_id"],
                    "method": record["setting"]["method"],
                    "lambda": record["setting"]["lambda"],
                    "block_size": record["setting"]["block_size"],
                    **record["metrics"],
                }
            )
    manifest = {
        "format_version": 1,
        "status": "complete",
        "sweep": sweep_name,
        "created_utc": utc_now(),
        "sweep_manifest": {"path": str(sweep_path), "sha256": sha256_file(sweep_path)},
        "metrics": file_identity(metrics_path),
        "metrics_csv": file_identity(csv_path),
        "candidate_count": len(metric_records),
        "bart_output_convention_version": GRE_BART_OUTPUT_CONVENTION_VERSION,
        "automatic_selection_performed": False,
        "composite_score_computed": False,
    }
    write_json_atomic(output / "evaluation_manifest.json", manifest)
    return manifest


def _curve_key(setting: Mapping[str, Any]) -> tuple[str, int | None]:
    """Return the regularizer-family key used for plots and review.

    Args:
        setting: Candidate method and optional LLR block configuration.

    Returns:
        Method and explicit optional block-size tuple.
    """

    method = str(setting["method"])
    block = None if setting.get("block_size") is None else int(setting["block_size"])
    return method, block


def plot_sweep(root: Path, *, sweep_name: str) -> dict[str, Any]:
    """Generate fixed-window images and curves without winner markers.

    Args:
        root: Confirmed experiment run root.
        sweep_name: Coarse or fine sweep identifier.

    Returns:
        Completed figure manifest bound to its evaluation.
    """

    import matplotlib.pyplot as plt

    evaluation_path = root / "evaluation" / sweep_name / "evaluation_manifest.json"
    evaluation = load_json(evaluation_path, f"{sweep_name} evaluation manifest")
    if evaluation.get("status") != "complete" or evaluation.get("automatic_selection_performed") is not False:
        raise ValueError("Plotting requires a completed non-ranking evaluation.")
    metrics_document = load_json(Path(evaluation["metrics"]["path"]), "evaluation metrics")
    _, sweep = _load_sweep(root, sweep_name)
    candidates = _candidate_documents(sweep)
    coarse_candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    if sweep_name == "fine":
        _, coarse = _load_sweep(root, "coarse")
        coarse_candidates = _candidate_documents(coarse)
    cases = _validated_case_manifests(root)
    output = root / "figures" / sweep_name
    curves_dir = output / "metric_curves"
    images_dir = output / "fixed_window"
    curves_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    curve_metrics = (
        "nrmse_brain",
        "ssim_brain",
        "normalized_acquired_data_residual",
        "median_absolute_error_rad",
        "background_energy_fraction",
    )
    for case_id in CASE_IDS:
        for echo_id in ECHO_IDS:
            records = [
                record
                for record in metrics_document["per_echo"]
                if record["case_id"] == case_id and record["echo_id"] == echo_id
            ]
            positive = [record for record in records if float(record["setting"]["lambda"]) > 0]
            curve_keys = sorted(
                {_curve_key(record["setting"]) for record in positive},
                key=lambda value: (value[0], -1 if value[1] is None else value[1]),
            )
            case_geometry = cases[(case_id, echo_id)]["manifest"]["geometry"]
            te_ms = 1000.0 * float(cases[(case_id, echo_id)]["manifest"]["te_s"])
            geometry_label = (
                f"TE {te_ms:g} ms, matrix {case_geometry['matrix_ro_lin_par']}, "
                f"voxel {case_geometry['voxel_mm_ro_lin_par']} mm"
            )
            figure, axes = plt.subplots(len(curve_metrics), 1, figsize=(7, 3 * len(curve_metrics)))
            for method, block in curve_keys:
                selected = sorted(
                    [record for record in positive if _curve_key(record["setting"]) == (method, block)],
                    key=lambda record: float(record["setting"]["lambda"]),
                )
                label = method if block is None else f"{method}, block {block}"
                lambdas = [float(record["setting"]["lambda"]) for record in selected]
                for axis, metric in zip(axes, curve_metrics, strict=True):
                    axis.plot(lambdas, [record["metrics"][metric] for record in selected], marker="o", label=label)
                    axis.set_xscale("log")
                    axis.set_xlabel("lambda")
                    axis.set_ylabel(metric)
                    axis.grid(True, alpha=0.25)
            axes[0].legend()
            figure.suptitle(
                f"{case_id}, {echo_id}, {geometry_label}: quantitative curves "
                "(no automatic selection)"
            )
            figure.tight_layout()
            curve_path = curves_dir / f"{case_id}_{echo_id}_curves.png"
            figure.savefig(curve_path, dpi=180)
            plt.close(figure)
            figures.append(file_identity(curve_path))
            case = cases[(case_id, echo_id)]["manifest"]
            reference = np.load(case["direct_fft_reference"]["complex"]["path"], allow_pickle=False)
            window = float(np.percentile(np.abs(reference), 99.5))
            control_key = (case_id, echo_id, "fista_lambda-0")
            control_document = candidates.get(control_key) or coarse_candidates.get(control_key)
            if control_document is None:
                raise ValueError(f"No FISTA control is available for {case_id}/{echo_id}.")
            for curve_key in curve_keys:
                selected_records = sorted(
                    [record for record in positive if _curve_key(record["setting"]) == curve_key],
                    key=lambda record: float(record["setting"]["lambda"]),
                )
                volumes = [("direct FFT", reference), ("FISTA lambda 0", _complex_candidate(control_document))]
                for record in selected_records:
                    document = candidates[(case_id, echo_id, record["candidate_name"])]
                    setting = record["setting"]
                    label = f"lambda {setting['lambda']:g}"
                    volumes.append((label, _complex_candidate(document)))
                figure, axes = plt.subplots(3, len(volumes), figsize=(3 * len(volumes), 9), squeeze=False)
                orientation = case["direct_fft_reference"]["nifti_orientation"]
                shape = tuple(orientation["canonical_ras_shape"])
                slices = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
                for column, (label, volume) in enumerate(volumes):
                    magnitude = np.abs(
                        _apply_orientation(
                            volume,
                            orientation["logical_to_canonical_ras_transform"],
                        )
                    )
                    planes = (
                        magnitude[slices[0], :, :].T,
                        magnitude[:, slices[1], :].T,
                        magnitude[:, :, slices[2]].T,
                    )
                    for row, plane in enumerate(planes):
                        axes[row, column].imshow(plane, cmap="gray", origin="lower", vmin=0, vmax=window)
                        axes[row, column].axis("off")
                    axes[0, column].set_title(label)
                axes[0, 0].set_ylabel("sagittal")
                axes[1, 0].set_ylabel("coronal")
                axes[2, 0].set_ylabel("axial")
                method, block = curve_key
                descriptor = method if block is None else f"{method}_block-{block}"
                figure.suptitle(
                    f"{case_id}, {echo_id}, {descriptor}, {geometry_label}; "
                    f"fixed window [0, {window:.5g}]"
                )
                figure.tight_layout()
                image_path = images_dir / f"{case_id}_{echo_id}_{descriptor}.png"
                figure.savefig(image_path, dpi=150)
                plt.close(figure)
                figures.append(file_identity(image_path))
    manifest = {
        "format_version": 1,
        "status": "complete",
        "sweep": sweep_name,
        "created_utc": utc_now(),
        "evaluation_manifest": {"path": str(evaluation_path), "sha256": sha256_file(evaluation_path)},
        "bart_output_convention_version": GRE_BART_OUTPUT_CONVENTION_VERSION,
        "figures": figures,
        "fixed_candidate_independent_windows": True,
        "automatic_winner_markers": False,
    }
    write_json_atomic(output / "figure_manifest.json", manifest)
    return manifest


def _validated_explicit_refinement_settings(
    entries: Any, *, case_id: str, echo_id: str
) -> list[dict[str, Any]]:
    """Validate explicit user-reviewed fine-sweep lambda lists.

    Args:
        entries: Per-group list of regularizer and lambda-list mappings.
        case_id: GRE geometry identifier used in validation errors.
        echo_id: Echo identifier used in validation errors.

    Returns:
        Ordered BART setting records for one case/echo group.
    """

    if not isinstance(entries, list):
        raise ValueError(f"Refinement entries must be a list for {case_id}/{echo_id}.")
    settings = []
    for entry in entries:
        if not isinstance(entry, Mapping) or "lambdas" not in entry:
            raise ValueError(f"Refinement entries require explicit lambdas for {case_id}/{echo_id}.")
        method = str(entry.get("method", ""))
        block = None if entry.get("block_size") is None else int(entry["block_size"])
        if method not in {"wavelet", "llr"}:
            raise ValueError("Refinement method must be wavelet or llr.")
        if (method == "wavelet" and block is not None) or (
            method == "llr" and block not in LLR_BLOCK_SIZES
        ):
            raise ValueError("Refinement block contract is invalid.")
        raw_lambdas = entry["lambdas"]
        if not isinstance(raw_lambdas, list) or not raw_lambdas or len(raw_lambdas) > 32:
            raise ValueError("Each explicit refinement lambda list must contain 1 to 32 values.")
        lambdas = [float(value) for value in raw_lambdas]
        if (
            any(not math.isfinite(value) or value <= 0 for value in lambdas)
            or lambdas != sorted(lambdas)
            or len(lambdas) != len(set(lambdas))
        ):
            raise ValueError("Explicit refinement lambdas must be positive, finite, unique, and increasing.")
        settings.extend(
            {"method": method, "lambda": value, "block_size": block}
            for value in lambdas
        )
    names = [candidate_name(setting) for setting in settings]
    if len(names) != len(set(names)):
        raise ValueError(f"Refinement review repeats a candidate for {case_id}/{echo_id}.")
    return settings


def record_refinement(root: Path, review_path: Path, *, reviewer: str) -> dict[str, Any]:
    """Validate and record explicit user-reviewed fine-sweep settings.

    Args:
        root: Confirmed experiment run root.
        review_path: User-edited local refinement review JSON.
        reviewer: Nonempty reviewer identity.

    Returns:
        Hash-bound explicit fine-sweep request.
    """

    if not reviewer.strip():
        raise ValueError("A reviewer identity is required.")
    coarse_path, coarse = _load_sweep(root, "coarse")
    evaluation_path = root / "evaluation" / "coarse" / "evaluation_manifest.json"
    figures_path = root / "figures" / "coarse" / "figure_manifest.json"
    evaluation = load_json(evaluation_path, evaluation_path.name)
    try:
        convention_audit = _audit_evaluation_bart_convention(evaluation, coarse)
    except ValueError as error:
        raise ValueError(
            "Rerun coarse-evaluate with the current BART output convention first. "
            f"Audit failed: {error}"
        ) from error
    figures = load_json(figures_path, figures_path.name)
    if (
        figures.get("status") != "complete"
        or figures.get("bart_output_convention_version")
        not in (None, GRE_BART_OUTPUT_CONVENTION_VERSION)
        or figures.get("evaluation_manifest", {}).get("sha256") != sha256_file(evaluation_path)
    ):
        raise ValueError("Rerun coarse-plot against the current coarse evaluation first.")
    _verify_embedded_hashes(figures)
    review = load_json(review_path.expanduser().resolve(), "refinement review")
    if review.get("format_version") != 2:
        raise ValueError("Explicit GRE refinement review requires format_version 2.")
    groups = review.get("groups")
    if not isinstance(groups, Mapping) or set(groups) != set(CASE_IDS):
        raise ValueError("Refinement review must contain all three cases.")
    settings_by_group = {}
    for case_id in CASE_IDS:
        if not isinstance(groups[case_id], Mapping) or set(groups[case_id]) != set(ECHO_IDS):
            raise ValueError(f"Refinement review must contain both echoes for {case_id}.")
        settings_by_group[case_id] = {}
        for echo_id in ECHO_IDS:
            settings_by_group[case_id][echo_id] = _validated_explicit_refinement_settings(
                groups[case_id][echo_id],
                case_id=case_id,
                echo_id=echo_id,
            )
    document = {
        "format_version": 1,
        "status": "approved_for_fine_sweep",
        "recorded_utc": utc_now(),
        "reviewer": reviewer,
        "source_review": file_identity(review_path.expanduser().resolve()),
        "coarse_sweep": {"path": str(coarse_path), "sha256": sha256_file(coarse_path), "job_count": coarse["job_count"]},
        "coarse_evaluation": file_identity(evaluation_path),
        "coarse_evaluation_bart_convention_audit": convention_audit,
        "coarse_figures": file_identity(figures_path),
        "refinement_rule": "explicit user-reviewed lambda lists; no automatic interpolation or ranking",
        "fine_job_count": sum(len(settings_by_group[case][echo]) for case in CASE_IDS for echo in ECHO_IDS),
        "groups": settings_by_group,
        "automatic_selection_performed": False,
    }
    output = root / "selections" / "refinement_request.json"
    write_json_atomic(output, document)
    return document


def record_final_selection(root: Path, review_path: Path, *, reviewer: str) -> dict[str, Any]:
    """Record explicit Wavelet and LLR selections for every group.

    Args:
        root: Confirmed experiment run root.
        review_path: User-edited local final-selection JSON.
        reviewer: Nonempty reviewer identity.

    Returns:
        Completed selection manifest with no automatic winner logic.
    """

    if not reviewer.strip():
        raise ValueError("A reviewer identity is required.")
    review = load_json(review_path.expanduser().resolve(), "final selection review")
    available: dict[tuple[str, str, str], dict[str, Any]] = {}
    sweep_bindings = []
    for sweep_name in ("coarse", "fine"):
        path = root / "reconstructions" / sweep_name / "sweep_manifest.json"
        if not path.is_file():
            continue
        resolved, sweep = _load_sweep(root, sweep_name)
        available.update(_candidate_documents(sweep))
        sweep_bindings.append({"path": str(resolved), "sha256": sha256_file(resolved)})
    groups = review.get("groups")
    if not isinstance(groups, Mapping) or set(groups) != set(CASE_IDS):
        raise ValueError("Final selection must contain all three cases.")
    selected = {}
    for case_id in CASE_IDS:
        if not isinstance(groups[case_id], Mapping) or set(groups[case_id]) != set(ECHO_IDS):
            raise ValueError(f"Final selection must contain both echoes for {case_id}.")
        selected[case_id] = {}
        for echo_id in ECHO_IDS:
            entries = groups[case_id][echo_id]
            if not isinstance(entries, Mapping) or set(entries) != {"wavelet", "llr"}:
                raise ValueError(f"Select exactly one Wavelet and one LLR candidate for {case_id}/{echo_id}.")
            selected[case_id][echo_id] = {}
            for family in ("wavelet", "llr"):
                setting = dict(entries[family])
                setting["method"] = family
                setting.setdefault("block_size", None)
                name = candidate_name(setting)
                key = (case_id, echo_id, name)
                if key not in available:
                    raise ValueError(f"Selected candidate is not in a completed sweep: {key}")
                candidate_manifest = Path(available[key]["output"]["base"]).parent / "manifest.json"
                selected[case_id][echo_id][family] = {
                    "setting": setting,
                    "candidate_name": name,
                    "candidate_manifest": {"path": str(candidate_manifest), "sha256": sha256_file(candidate_manifest)},
                }
    document = {
        "format_version": 1,
        "status": "complete",
        "recorded_utc": utc_now(),
        "reviewer": reviewer,
        "source_review": file_identity(review_path.expanduser().resolve()),
        "sweep_manifests": sweep_bindings,
        "groups": selected,
        "automatic_selection_performed": False,
    }
    output = root / "selections" / "final_selection.json"
    write_json_atomic(output, document)
    return document


def evaluate_final_selection(root: Path) -> dict[str, Any]:
    """Evaluate independently selected echo pairs for Wavelet and LLR.

    Args:
        root: Confirmed experiment run root.

    Returns:
        Final cross-echo evaluation manifest.
    """

    import nibabel as nib

    selection_path = root / "selections" / "final_selection.json"
    selection = load_json(selection_path, "final selection")
    if selection.get("status") != "complete" or selection.get("automatic_selection_performed") is not False:
        raise ValueError("A complete explicit final selection is required.")
    cases = _validated_case_manifests(root)
    records = []
    for case_id in CASE_IDS:
        case1 = cases[(case_id, "echo-01")]["manifest"]
        case2 = cases[(case_id, "echo-02")]["manifest"]
        reference1 = np.load(case1["direct_fft_reference"]["complex"]["path"], allow_pickle=False)
        reference2 = np.load(case2["direct_fft_reference"]["complex"]["path"], allow_pickle=False)
        brain = np.asarray(nib.load(case1["brain_mask"]["path"]).dataobj) > 0.5
        threshold = 0.05 * float(np.percentile(np.abs(reference1[brain]), 99))
        support = brain & (np.abs(reference1) >= threshold) & (np.abs(reference2) >= threshold)
        for family in ("wavelet", "llr"):
            echo_documents = []
            settings = []
            for echo_id in ECHO_IDS:
                selected = selection["groups"][case_id][echo_id][family]
                path = Path(selected["candidate_manifest"]["path"])
                if sha256_file(path) != selected["candidate_manifest"]["sha256"]:
                    raise ValueError(f"Selected candidate manifest changed: {path}")
                echo_documents.append(load_json(path, "selected candidate"))
                settings.append(selected["setting"])
            candidate1, candidate2 = (_complex_candidate(document) for document in echo_documents)
            records.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "pairing": "independently user-selected echo settings",
                    "echo_1_setting": settings[0],
                    "echo_2_setting": settings[1],
                    "delta_te_s": ECHO_TIMES_S[1] - ECHO_TIMES_S[0],
                    "support_voxel_count": int(np.count_nonzero(support)),
                    "excluded_low_signal_fraction_within_brain": float(
                        1.0 - np.count_nonzero(support) / np.count_nonzero(brain)
                    ),
                    "scaling_policy": (
                        "fixed BART L2, extended-grid FFT, and fftmod convention restoration; "
                        "no per-echo LSQ scaling"
                    ),
                    "metrics": inter_echo_metrics(
                        reference1,
                        reference2,
                        candidate1,
                        candidate2,
                        support,
                        delta_te_s=ECHO_TIMES_S[1] - ECHO_TIMES_S[0],
                    ),
                }
            )
    output = root / "evaluation" / "final_selection"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "inter_echo_metrics.json"
    write_json_atomic(metrics_path, {"records": records})
    manifest = {
        "format_version": 1,
        "status": "complete",
        "created_utc": utc_now(),
        "selection": {"path": str(selection_path), "sha256": sha256_file(selection_path)},
        "metrics": file_identity(metrics_path),
        "automatic_selection_performed": False,
    }
    write_json_atomic(output / "evaluation_manifest.json", manifest)
    return manifest


def evaluate_shared_lambda(root: Path) -> dict[str, Any]:
    """Evaluate each fine Wavelet lambda after pooling both GRE echoes.

    Args:
        root: Confirmed experiment run root.

    Returns:
        Completed non-ranking evaluation manifest for three shared-lambda curves.
    """

    import csv
    import nibabel as nib

    fine_evaluation_path = root / "evaluation" / "fine" / "evaluation_manifest.json"
    fine_evaluation = load_json(fine_evaluation_path, "fine evaluation manifest")
    fine_sweep_path, fine_sweep = _load_sweep(root, "fine")
    convention_audit = _audit_evaluation_bart_convention(fine_evaluation, fine_sweep)
    fine_metrics = load_json(Path(fine_evaluation["metrics"]["path"]), "fine metrics")
    candidates = _candidate_documents(fine_sweep)
    cases = _validated_case_manifests(root)
    per_echo_index = {
        (record["case_id"], record["echo_id"], record["candidate_name"]): record
        for record in fine_metrics["per_echo"]
    }
    cross_echo_index = {
        (record["case_id"], record["candidate_name"]): record
        for record in fine_metrics["same_setting_cross_echo"]
    }
    records = []
    for case_id in CASE_IDS:
        echo_names = [
            {
                key[2]
                for key, document in candidates.items()
                if key[:2] == (case_id, echo_id) and document["setting"]["method"] == "wavelet"
            }
            for echo_id in ECHO_IDS
        ]
        shared_names = set.intersection(*echo_names)
        if len(shared_names) != len(echo_names[0]) or any(names != shared_names for names in echo_names):
            raise ValueError(f"Fine Wavelet candidates are not echo-matched for {case_id}.")

        references = []
        masks = []
        phase_supports = []
        for echo_id in ECHO_IDS:
            case = cases[(case_id, echo_id)]["manifest"]
            reference = np.load(case["direct_fft_reference"]["complex"]["path"], allow_pickle=False)
            brain = np.asarray(nib.load(case["brain_mask"]["path"]).dataobj) > 0.5
            threshold = 0.05 * float(np.percentile(np.abs(reference[brain]), 99))
            references.append(reference)
            masks.append(brain)
            phase_supports.append(brain & (np.abs(reference) >= threshold))

        ordered_names = sorted(
            shared_names,
            key=lambda name: float(candidates[(case_id, ECHO_IDS[0], name)]["setting"]["lambda"]),
        )
        for name in ordered_names:
            documents = [candidates[(case_id, echo_id, name)] for echo_id in ECHO_IDS]
            settings = [document["setting"] for document in documents]
            lambdas = [float(setting["lambda"]) for setting in settings]
            if lambdas[0] != lambdas[1]:
                raise ValueError(f"Candidate {case_id}/{name} does not use one shared lambda.")
            echo_metric_records = [
                per_echo_index[(case_id, echo_id, name)] for echo_id in ECHO_IDS
            ]
            candidates_by_echo = [_complex_candidate(document) for document in documents]
            global_metrics = _global_echo_metrics(
                references,
                candidates_by_echo,
                masks,
                phase_supports,
                [record["metrics"] for record in echo_metric_records],
                [float(document["input_kspace_norm_for_restoration"]) for document in documents],
            )
            records.append(
                {
                    "case_id": case_id,
                    "candidate_name": name,
                    "shared_setting": settings[0],
                    "echoes": [
                        {
                            "echo_id": echo_id,
                            "te_s": ECHO_TIMES_S[index],
                            "candidate_manifest": file_identity(
                                Path(documents[index]["output"]["base"]).parent / "manifest.json"
                            ),
                        }
                        for index, echo_id in enumerate(ECHO_IDS)
                    ],
                    "global_metrics": global_metrics,
                    "cross_echo_metrics": cross_echo_index[(case_id, name)]["metrics"],
                }
            )
            del candidates_by_echo

    output = root / "evaluation" / "fine_shared_lambda"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    write_json_atomic(
        metrics_path,
        {
            "format_version": 1,
            "status": "complete",
            "pooling_policy": {
                "magnitude_nrmse": "pool squared error and reference energy over both echoes before division",
                "magnitude_ncc": "concatenate brain-mask voxels from both echoes before centering",
                "gradient_metrics": "pool gradient energies and cross-products over echoes and axes",
                "phase_metrics": "concatenate reference-supported circular phase errors from both echoes",
                "ssim": "equal arithmetic mean of the two echo-specific brain SSIM values",
                "data_residual": "pool squared residuals using measured k-space L2 norms",
            },
            "record_count": len(records),
            "records": records,
        },
    )
    csv_path = output / "metrics.csv"
    global_names = sorted(records[0]["global_metrics"])
    cross_names = sorted(records[0]["cross_echo_metrics"])
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case_id",
                "method",
                "lambda",
                *global_names,
                *(f"cross_echo_{name}" for name in cross_names),
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "case_id": record["case_id"],
                    "method": record["shared_setting"]["method"],
                    "lambda": record["shared_setting"]["lambda"],
                    **record["global_metrics"],
                    **{
                        f"cross_echo_{name}": value
                        for name, value in record["cross_echo_metrics"].items()
                    },
                }
            )
    manifest = {
        "format_version": 1,
        "status": "complete",
        "created_utc": utc_now(),
        "fine_sweep": file_identity(fine_sweep_path),
        "fine_evaluation": file_identity(fine_evaluation_path),
        "bart_convention_audit": convention_audit,
        "metrics": file_identity(metrics_path),
        "metrics_csv": file_identity(csv_path),
        "case_count": len(CASE_IDS),
        "lambda_count_per_case": len(records) // len(CASE_IDS),
        "automatic_selection_performed": False,
    }
    write_json_atomic(output / "evaluation_manifest.json", manifest)
    return manifest


def plot_shared_lambda(root: Path) -> dict[str, Any]:
    """Plot global two-echo metrics for each fine shared Wavelet lambda.

    Args:
        root: Confirmed experiment run root.

    Returns:
        Completed figure manifest with one curve panel set per geometry.
    """

    import matplotlib.pyplot as plt

    evaluation_path = root / "evaluation" / "fine_shared_lambda" / "evaluation_manifest.json"
    evaluation = load_json(evaluation_path, "shared-lambda evaluation manifest")
    if evaluation.get("status") != "complete" or evaluation.get("automatic_selection_performed") is not False:
        raise ValueError("Shared-lambda plotting requires a complete non-ranking evaluation.")
    _verify_embedded_hashes(evaluation)
    metrics = load_json(Path(evaluation["metrics"]["path"]), "shared-lambda metrics")
    output = root / "figures" / "fine_shared_lambda"
    output.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("global", "global_magnitude_nrmse_brain", "Global magnitude NRMSE ↓"),
        ("global", "global_magnitude_ncc_brain", "Global magnitude NCC ↑"),
        ("global", "mean_echo_ssim_brain", "Mean echo SSIM ↑"),
        ("global", "global_gradient_nrmse_brain", "Global gradient NRMSE ↓"),
        ("global", "global_gradient_ncc_brain", "Global gradient NCC ↑"),
        ("global", "global_p95_absolute_phase_error_rad", "Global phase |error| p95 ↓"),
        ("global", "global_circular_dispersion", "Global phase dispersion ↓"),
        ("global", "global_normalized_acquired_data_residual", "Global acquired residual ↓"),
        ("cross", "unwrapped_delta_b0_nrmse", "Cross-echo ΔB0 NRMSE ↓"),
        ("cross", "unwrapped_delta_b0_mae_hz", "Cross-echo ΔB0 MAE (Hz) ↓"),
        ("cross", "wrapped_delta_b0_mae_hz", "Wrapped ΔB0 MAE (Hz) ↓"),
        ("cross", "magnitude_ratio_mad", "Magnitude-ratio MAD ↓"),
    )
    figures = []
    for case_id in CASE_IDS:
        records = sorted(
            [record for record in metrics["records"] if record["case_id"] == case_id],
            key=lambda record: float(record["shared_setting"]["lambda"]),
        )
        lambdas = [float(record["shared_setting"]["lambda"]) for record in records]
        figure, axes = plt.subplots(4, 3, figsize=(15, 16), squeeze=False)
        for axis, (source, name, label) in zip(axes.flat, specifications, strict=True):
            key = "global_metrics" if source == "global" else "cross_echo_metrics"
            axis.plot(lambdas, [float(record[key][name]) for record in records], marker="o")
            axis.set_xscale("log")
            axis.set_xlabel("Shared Wavelet lambda")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
        figure.suptitle(
            f"{case_id}: fine shared-lambda two-echo metrics (no automatic selection)",
            fontsize=15,
        )
        figure.tight_layout()
        path = output / f"{case_id}_shared_lambda_curves.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figures.append(file_identity(path))
    manifest = {
        "format_version": 1,
        "status": "complete",
        "created_utc": utc_now(),
        "evaluation": file_identity(evaluation_path),
        "figures": figures,
        "case_count": len(figures),
        "automatic_selection_markers": False,
    }
    write_json_atomic(output / "figure_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the stage-oriented command-line interface.

    Returns:
        Argument parser for all validation, preparation, and review operations.
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "operation",
        choices=(
            "validate-config",
            "inspect-metadata",
            "prepare-source",
            "prepare-operator",
            "validate-operator",
            "prepare-csm",
            "prepare-references",
            "prepare-brain-mask",
            "approve-brain-mask",
            "prepare-cases",
            "reconstruct",
            "export-nifti",
            "evaluate",
            "plot",
            "record-refinement",
            "record-selection",
            "evaluate-selection",
            "evaluate-shared-lambda",
            "plot-shared-lambda",
        ),
    )
    parser.add_argument("--confirm-run-root", type=Path)
    parser.add_argument("--sweep", choices=("coarse", "fine"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--decision", choices=("approve", "reject"))
    parser.add_argument("--brain-mask-candidate")
    parser.add_argument("--brain-mask-source-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute exactly one user-selected workflow operation.

    Args:
        argv: Optional command-line arguments; ``None`` uses process arguments.

    Returns:
        Zero after successful validation or operation completion.
    """

    args = build_parser().parse_args(argv)
    config, validated = load_config(args.config)
    if args.operation == "validate-config":
        print(json.dumps(validated, indent=2, sort_keys=True))
        return 0
    root = require_confirmed_root(validated, args.confirm_run_root)
    operation_functions = {
        "inspect-metadata": inspect_metadata,
        "prepare-source": prepare_source,
        "prepare-operator": prepare_operator,
        "validate-operator": validate_operator_roundtrip,
        "prepare-csm": prepare_csm,
        "prepare-references": prepare_references,
        "prepare-cases": prepare_cases,
    }
    if args.operation == "prepare-brain-mask":
        result = prepare_brain_mask_candidate(
            config,
            validated,
            root,
            source_manifest_path=args.brain_mask_source_manifest,
        )
    elif args.operation in operation_functions:
        result = operation_functions[args.operation](config, validated, root)
    elif args.operation == "approve-brain-mask":
        result = approve_brain_mask(
            config,
            validated,
            root,
            reviewer=args.reviewer or "",
            decision=args.decision or "",
            candidate_id=args.brain_mask_candidate or "",
        )
    elif args.operation == "reconstruct":
        if args.sweep is None:
            raise ValueError("reconstruct requires --sweep coarse or fine.")
        result = run_sweep(
            config,
            validated,
            root,
            sweep=args.sweep,
            resume=args.resume,
            validate_only=args.validate_only,
        )
    elif args.operation == "export-nifti":
        if args.sweep is None:
            raise ValueError("export-nifti requires --sweep coarse or fine.")
        result = export_sweep_nifti(validated, root, sweep_name=args.sweep)
    elif args.operation == "evaluate":
        if args.sweep is None:
            raise ValueError("evaluate requires --sweep coarse or fine.")
        result = evaluate_sweep(config, validated, root, sweep_name=args.sweep)
    elif args.operation == "plot":
        if args.sweep is None:
            raise ValueError("plot requires --sweep coarse or fine.")
        result = plot_sweep(root, sweep_name=args.sweep)
    elif args.operation == "record-refinement":
        if args.review is None:
            raise ValueError("record-refinement requires --review.")
        result = record_refinement(root, args.review, reviewer=args.reviewer or "")
    elif args.operation == "record-selection":
        if args.review is None:
            raise ValueError("record-selection requires --review.")
        result = record_final_selection(root, args.review, reviewer=args.reviewer or "")
    elif args.operation == "evaluate-selection":
        result = evaluate_final_selection(root)
    elif args.operation == "evaluate-shared-lambda":
        result = evaluate_shared_lambda(root)
    elif args.operation == "plot-shared-lambda":
        result = plot_shared_lambda(root)
    else:
        raise AssertionError(f"Unhandled operation: {args.operation}")
    print(json.dumps({"operation": args.operation, "status": result.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
