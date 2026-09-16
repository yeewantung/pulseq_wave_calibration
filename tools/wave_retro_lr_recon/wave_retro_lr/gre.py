"""Prepare measured single- or multi-echo Wave-GRE inputs for BART."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bart_io import create_cfl, logical_array_sha256, open_cfl, read_shape, sha256_file
from .core import PE_MATRIX_MULTIPLE, center_crop_bounds
from .psf import (
    PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME,
    PSF_COEFFICIENT_PLOT_NAME,
    evaluate_calibrated_psf,
    write_psf_coefficient_plot,
)
from .retrospective import (
    link_bart_pair,
    resample_sensitivity_maps,
    validate_same_grid_masked_wave,
)
from .sampling import pure_cartesian_image_lattice_mask, validate_pure_cartesian_image_lattice

NATIVE_MATRIX_RO_LIN_PAR = (250, 250, 72)
NATIVE_FOV_MM_RO_LIN_PAR = (220.0, 220.0, 180.0)
EXTENDED_READOUT = 1000
VIRTUAL_COILS = 12
LOW_RESOLUTION_LIN_BOUNDS = (51, 199)
LOW_RESOLUTION_LIN_MM = 1.5
WAVELET_SELECTION_BASENAME = "wavelet_shared_echo_selection.json"
WAVELET_SELECTION_SHA256 = "0c43a9d31672e90ad851decfca66c253c362cbd67ca5ba97c4fd8ef1f5a61afd"
GRE_GEOMETRY_IDS = (
    "native_r3x1",
    "native_r3x2",
    "lin_low_resolution_r3x2",
    "native_r3x3",
)
GRE_SHARED_WAVELET_LAMBDA = 0.015
GRE_LOGICAL_AXIS_ROLES = ("readout", "phase", "slice")
GRE_BART_ARRAY_AXIS_FLIPS = (False, True, False)
GRE_BART_OUTPUT_CONVENTION_VERSION = 2

NORMAL_INPUT_RELATIVE = Path("normal") / "bart_inputs"
NORMAL_OUTPUT_RELATIVE = Path("normal") / "bart_output"
RETRO_RELATIVE = Path("retro")
R3X3_CASE_ID = "native_r3x3"
NORMAL_REUSE_ATTESTATION_NAME = "NORMAL_INPUT_REUSE_ATTESTATION.json"


@dataclass(frozen=True)
class GreCase:
    """Describe one measured GRE geometry and pure Cartesian image lattice."""

    case_id: str
    source_matrix_ro_lin_par: tuple[int, int, int]
    matrix_ro_lin_par: tuple[int, int, int]
    fov_mm_ro_lin_par: tuple[float, float, float]
    crop_bounds_from_native: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    acceleration_lin_par: tuple[int, int]
    residue_lin_par: tuple[int, int]
    shared_wavelet_lambda: float

    @property
    def voxel_mm_ro_lin_par(self) -> tuple[float, float, float]:
        """Return unchanged-FOV voxel sizes in logical axis order.

        Returns:
            Readout, phase, and slice voxel sizes in millimeters.
        """

        return tuple(
            fov / matrix
            for fov, matrix in zip(
                self.fov_mm_ro_lin_par, self.matrix_ro_lin_par, strict=True
            )
        )

    def to_json(self) -> dict[str, Any]:
        """Convert the case contract to JSON-native values.

        Returns:
            Geometry, crop, sampling, and shared-echo Wavelet lambda.
        """

        return {
            "case_id": self.case_id,
            "matrix_ro_lin_par": list(self.matrix_ro_lin_par),
            "fov_mm_ro_lin_par": list(self.fov_mm_ro_lin_par),
            "voxel_mm_ro_lin_par": list(self.voxel_mm_ro_lin_par),
            "crop_bounds_from_native": [list(value) for value in self.crop_bounds_from_native],
            "acceleration_lin_par": list(self.acceleration_lin_par),
            "residue_lin_par": list(self.residue_lin_par),
            "shared_wavelet_lambda": self.shared_wavelet_lambda,
        }


def gre_echo_ids(echo_count: int) -> tuple[str, ...]:
    """Return consecutive one-based echo labels for a positive echo count.

    Args:
        echo_count: Number of GRE echoes.

    Returns:
        Labels from ``echo-01`` through the requested count.

    Raises:
        ValueError: If ``echo_count`` is not positive.
    """

    count = int(echo_count)
    if count <= 0:
        raise ValueError("GRE echo count must be positive.")
    return tuple(f"echo-{index:02d}" for index in range(1, count + 1))


def resolve_gre_wavelet_lambda(
    geometry_id: str,
    *,
    method: str = "wavelet",
    echo_ids: Sequence[str] | None = None,
    shared_lambda: float | None = None,
    wavelet_lambda_by_echo: Mapping[str, float] | Sequence[float] | None = None,
    selection_manifest_basename: str = WAVELET_SELECTION_BASENAME,
    selection_manifest_sha256: str = WAVELET_SELECTION_SHA256,
) -> float:
    """Resolve and validate the authoritative shared-echo GRE Wavelet value.

    Args:
        geometry_id: One of the three reviewed GRE geometry identifiers.
        method: Selected regularization method, which must be ``wavelet``.
        echo_ids: Optional complete ordered echo identifiers.
        shared_lambda: Optional explicit shared value.
        wavelet_lambda_by_echo: Optional legacy per-echo representation.
        selection_manifest_basename: Authoritative selection-record basename.
        selection_manifest_sha256: Authoritative selection-record SHA-256.

    Returns:
        The validated shared Wavelet lambda.

    Raises:
        ValueError: If geometry, echoes, method, values, or provenance differ
            from the reviewed shared-echo selection.
    """

    if geometry_id not in GRE_GEOMETRY_IDS:
        raise ValueError(f"Unknown GRE geometry ID: {geometry_id!r}.")
    if str(method).strip().lower() != "wavelet":
        raise ValueError("The reviewed GRE selection method must be wavelet.")
    ordered_echoes: tuple[str, ...] | None = None
    if echo_ids is not None:
        ordered_echoes = tuple(str(value) for value in echo_ids)
        if not ordered_echoes or ordered_echoes != gre_echo_ids(len(ordered_echoes)):
            raise ValueError("GRE Wavelet selection requires consecutive ordered echo labels.")
    if selection_manifest_basename != WAVELET_SELECTION_BASENAME:
        raise ValueError("GRE Wavelet selection manifest basename does not match provenance.")
    if selection_manifest_sha256 != WAVELET_SELECTION_SHA256:
        raise ValueError("GRE Wavelet selection manifest SHA-256 does not match provenance.")

    legacy_values: tuple[float, ...] | None = None
    if wavelet_lambda_by_echo is not None:
        if ordered_echoes is None:
            raise ValueError("Legacy echo lambdas require explicit ordered echo labels.")
        if isinstance(wavelet_lambda_by_echo, Mapping):
            if tuple(wavelet_lambda_by_echo) != ordered_echoes:
                raise ValueError("GRE Wavelet lambdas must include every ordered echo.")
            legacy_values = tuple(float(wavelet_lambda_by_echo[key]) for key in ordered_echoes)
        else:
            legacy_values = tuple(float(value) for value in wavelet_lambda_by_echo)
            if len(legacy_values) != len(ordered_echoes):
                raise ValueError("GRE Wavelet selection requires one lambda per echo.")
        if not all(math.isfinite(value) for value in legacy_values):
            raise ValueError("GRE Wavelet lambdas must be finite.")
        if any(value != legacy_values[0] for value in legacy_values[1:]):
            raise ValueError("GRE Wavelet lambdas must be equal across all echoes.")

    resolved = GRE_SHARED_WAVELET_LAMBDA if shared_lambda is None else float(shared_lambda)
    if not math.isfinite(resolved) or resolved != GRE_SHARED_WAVELET_LAMBDA:
        raise ValueError(f"GRE shared Wavelet lambda must be {GRE_SHARED_WAVELET_LAMBDA}.")
    if legacy_values is not None and any(value != resolved for value in legacy_values):
        raise ValueError("Legacy echo lambdas do not match the shared GRE Wavelet lambda.")
    return resolved


def gre_wavelet_selection_provenance(
    geometry_id: str, echo_count: int
) -> dict[str, Any]:
    """Return hash-bound shared-echo selection provenance for one geometry.

    Args:
        geometry_id: Reviewed GRE geometry identifier.
        echo_count: Positive number of echoes sharing the value.

    Returns:
        JSON-native selection provenance and shared regularization policy.
    """

    echo_ids = gre_echo_ids(echo_count)
    shared_lambda = resolve_gre_wavelet_lambda(geometry_id, echo_ids=echo_ids)
    return {
        "selection_kind": "gre_wavelet_shared_echo_regularization",
        "selection_manifest_basename": WAVELET_SELECTION_BASENAME,
        "selection_manifest_sha256": WAVELET_SELECTION_SHA256,
        "geometry_id": geometry_id,
        "method": "wavelet",
        "shared_lambda": shared_lambda,
        "echo_ids": list(echo_ids),
        "echo_count": len(echo_ids),
        "lambda_constraint": "shared_within_geometry",
        "echo_application_policy": "apply the shared value to every validated echo",
        "reconstruction_coupling": "none; reconstruct each echo separately",
        "llr_selection_recorded": False,
    }


def gre_cases(
    native_matrix_ro_lin_par: Sequence[int] = NATIVE_MATRIX_RO_LIN_PAR,
    native_fov_mm_ro_lin_par: Sequence[float] = NATIVE_FOV_MM_RO_LIN_PAR,
) -> dict[str, GreCase]:
    """Resolve normal and retrospective GRE cases for one native geometry.

    Args:
        native_matrix_ro_lin_par: Sequence-derived native logical matrix.
        native_fov_mm_ro_lin_par: Sequence-derived native logical FOV in
            millimeters.

    Returns:
        Cases keyed by their stable output-directory identifiers.

    Raises:
        ValueError: If the fixed readout/slice contract or the requested
            LIN-low-resolution crop cannot be represented.
    """

    matrix = tuple(int(value) for value in native_matrix_ro_lin_par)
    fov = tuple(float(value) for value in native_fov_mm_ro_lin_par)
    if len(matrix) != 3 or matrix[0] != 250 or matrix[2] != 72 or matrix[1] <= 0:
        raise ValueError(
            "GRE native matrix must have readout 250, slice 72, and a positive LIN size."
        )
    if len(fov) != 3 or not np.isfinite(fov).all() or any(value <= 0 for value in fov):
        raise ValueError("GRE native FOV must contain three positive finite values.")
    ideal_low_lin = fov[1] / LOW_RESOLUTION_LIN_MM
    low_lin = PE_MATRIX_MULTIPLE * int(
        math.floor(ideal_low_lin / PE_MATRIX_MULTIPLE + 0.5)
    )
    if low_lin <= 0 or low_lin >= matrix[1]:
        raise ValueError(
            "GRE LIN-low-resolution request must resolve to a positive grid "
            "smaller than the native LIN matrix."
        )
    low_lin_bounds = center_crop_bounds(matrix[1], low_lin)
    low_lin_residue = (2 - low_lin_bounds[0]) % 3
    native_bounds = ((0, matrix[0]), (0, matrix[1]), (0, matrix[2]))
    return {
        "native_r3x1": GreCase(
            "native_r3x1",
            matrix,
            matrix,
            fov,
            native_bounds,
            (3, 1),
            (2, 0),
            resolve_gre_wavelet_lambda("native_r3x1"),
        ),
        "native_r3x2": GreCase(
            "native_r3x2",
            matrix,
            matrix,
            fov,
            native_bounds,
            (3, 2),
            (2, 0),
            resolve_gre_wavelet_lambda("native_r3x2"),
        ),
        "lin_low_resolution_r3x2": GreCase(
            "lin_low_resolution_r3x2",
            matrix,
            (matrix[0], low_lin, matrix[2]),
            fov,
            ((0, matrix[0]), low_lin_bounds, (0, matrix[2])),
            (3, 2),
            (low_lin_residue, 0),
            resolve_gre_wavelet_lambda("lin_low_resolution_r3x2"),
        ),
        "native_r3x3": GreCase(
            "native_r3x3",
            matrix,
            matrix,
            fov,
            native_bounds,
            (3, 3),
            (2, (matrix[2] // 2) % 3),
            resolve_gre_wavelet_lambda("native_r3x3"),
        ),
    }


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp.

    Returns:
        ISO-8601 timestamp string.
    """

    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one JSON manifest.

    Args:
        path: Destination JSON path.
        payload: JSON-compatible mapping.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object.

    Args:
        path: JSON file to read.

    Returns:
        Parsed top-level object.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _file_identity(path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    """Record an input file identity for safe reuse.

    Args:
        path: Existing file.
        include_hash: Include a complete SHA-256 digest when true.

    Returns:
        Resolved path, size, timestamp, and optional digest.
    """

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        result["sha256"] = sha256_file(resolved)
    return result


def _repository_root() -> Path:
    """Return the enclosing parent repository root.

    Returns:
        Absolute path inferred from this module location.
    """

    return Path(__file__).resolve().parents[3]


def load_wave_gre_helpers() -> Any:
    """Load the pinned Wave-GRE implementation without invoking its CLI.

    Returns:
        Imported upstream module containing focused sequence, TWIX, coil, and
        calibration helpers.
    """

    source = (
        _repository_root()
        / "external"
        / "wave-gre-flow-comp"
        / "recon"
        / "recon_wave_gre_from_twix_integrated_nifti.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Pinned Wave-GRE implementation not found: {source}")
    upstream_root = source.parent
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    spec = importlib.util.spec_from_file_location("pinned_wave_gre_adapter", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import pinned Wave-GRE implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_psf_settings(
    processing: str, fit_kx_min: int | None, fit_kx_max: int | None
) -> dict[str, Any]:
    """Validate shared GRE PSF coefficient-processing settings.

    Args:
        processing: ``smooth`` or ``sine-line``.
        fit_kx_min: Optional inclusive manual fit bound.
        fit_kx_max: Optional exclusive manual fit bound.

    Returns:
        JSON-compatible normalized request.
    """

    mode = str(processing).strip().lower()
    if mode not in {"smooth", "sine-line"}:
        raise ValueError("PSF coefficient processing must be 'smooth' or 'sine-line'.")
    if mode == "smooth" and (fit_kx_min is not None or fit_kx_max is not None):
        raise ValueError("PSF fit bounds are valid only with sine-line processing.")
    if mode == "sine-line" and (fit_kx_min is None) != (fit_kx_max is None):
        raise ValueError("Sine-line processing requires both manual bounds or neither.")
    if fit_kx_min is not None and not (0 <= int(fit_kx_min) < int(fit_kx_max) <= EXTENDED_READOUT):
        raise ValueError("PSF fit bounds must satisfy 0 <= min < max <= 1000.")
    return {
        "coefficient_processing": mode,
        "requested_fit_kx_range": (
            None if fit_kx_min is None else [int(fit_kx_min), int(fit_kx_max)]
        ),
        "fit_kx_range_convention": "half-open [min, max)",
    }


def validate_gre_echo_consistency(
    sequence_echo_times_s: Sequence[Any], twix_echo_times_s: Sequence[Any]
) -> tuple[float, ...]:
    """Require identical positive echo counts and TE values in sequence and TWIX.

    Args:
        sequence_echo_times_s: Ordered sequence TE values in seconds.
        twix_echo_times_s: Ordered TWIX TE values in seconds.

    Returns:
        Validated ordered echo times in seconds.

    Raises:
        ValueError: If either list is invalid or their counts/values disagree.
    """

    sequence_times = np.asarray(sequence_echo_times_s, dtype=np.float64)
    twix_times = np.asarray(twix_echo_times_s, dtype=np.float64)
    for label, values in (("sequence", sequence_times), ("TWIX", twix_times)):
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"GRE {label} must contain at least one echo time.")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"GRE {label} echo times must be positive and finite.")
    if sequence_times.size != twix_times.size:
        raise ValueError(
            "GRE sequence/TWIX echo counts disagree: "
            f"{sequence_times.size} versus {twix_times.size}."
        )
    if not np.allclose(sequence_times, twix_times, rtol=0.0, atol=1e-9):
        raise ValueError("GRE sequence/TWIX echo times disagree.")
    return tuple(float(value) for value in sequence_times)


def validate_gre_echo_sampling(
    lines: Sequence[Any],
    partitions: Sequence[Any],
    echoes: Sequence[Any],
    *,
    echo_times_s: Sequence[Any],
    matrix_lin_par: Sequence[int] = NATIVE_MATRIX_RO_LIN_PAR[1:],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate identical measured native R3x1 image lattices for all echoes.

    Args:
        lines: TWIX LIN counters.
        partitions: TWIX PAR counters aligned with ``lines``.
        echoes: Zero-based TWIX Eco counters aligned with ``lines``.
        echo_times_s: Ordered TWIX echo times in seconds.
        matrix_lin_par: Sequence-derived native LIN/PAR matrix.

    Returns:
        Native boolean mask and exact pure-lattice/echo metadata.
    """

    arrays = [np.asarray(value) for value in (lines, partitions, echoes)]
    if any(value.ndim != 1 for value in arrays) or len({value.size for value in arrays}) != 1:
        raise ValueError("TWIX LIN, PAR, and Eco counters must be aligned one-dimensional arrays.")
    if arrays[0].size == 0:
        raise ValueError("TWIX image counter arrays are empty.")
    converted = []
    for values, label in zip(arrays, ("LIN", "PAR", "Eco"), strict=True):
        numeric = values.astype(np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
            raise ValueError(f"TWIX {label} counters must be finite integers.")
        converted.append(numeric.astype(np.int64))
    lin, par, eco = converted
    unique_echoes = tuple(sorted(set(eco.tolist())))
    if not unique_echoes or unique_echoes != tuple(range(len(unique_echoes))):
        raise ValueError("Measured GRE Eco counters must be consecutive from zero.")
    times = np.asarray(echo_times_s, dtype=np.float64)
    if times.ndim != 1 or times.size != len(unique_echoes):
        raise ValueError("TWIX Eco counter count and echo-time count disagree.")
    if not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError("TWIX echo times must be positive and finite.")

    matrix = tuple(int(value) for value in matrix_lin_par)
    if len(matrix) != 2 or any(value <= 0 for value in matrix):
        raise ValueError("GRE LIN/PAR matrix must contain two positive dimensions.")
    expected_mask, sampling = pure_cartesian_image_lattice_mask(
        matrix, acceleration_lin_par=(3, 1), residue_lin_par=(2, 0)
    )
    expected_coordinates = set(zip(*np.nonzero(expected_mask), strict=True))
    echo_records = []
    for echo_index, te_s in enumerate(times):
        selected = eco == echo_index
        coordinates = list(zip(lin[selected].tolist(), par[selected].tolist(), strict=True))
        if len(coordinates) != len(set(coordinates)):
            raise ValueError(f"Echo {echo_index + 1} contains duplicate LIN/PAR coordinates.")
        if set(coordinates) != expected_coordinates:
            raise ValueError(f"Echo {echo_index + 1} is not the complete native residue-2 R3x1 lattice.")
        echo_records.append(
            {
                "echo": echo_index + 1,
                "eco_counter": echo_index,
                "te_s": float(te_s),
                "acquired_coordinate_count": len(coordinates),
                "sampling_logical_sha256": sampling["logical_sha256"],
            }
        )
    validate_pure_cartesian_image_lattice(expected_mask, sampling)
    sampling["echoes"] = echo_records
    sampling["shared_identical_across_echoes"] = True
    return expected_mask, sampling


def _twix_value(mapping: Any, key: tuple[str, ...], default: Any = None) -> Any:
    """Read one mapVBVD tuple-key header value.

    Args:
        mapping: MeasYaps-compatible mapping.
        key: Tuple key.
        default: Value returned when unavailable.

    Returns:
        Stored header value or ``default``.
    """

    try:
        value = mapping.get(key, default)
    except Exception:
        try:
            value = mapping[key]
        except Exception:
            value = default
    return default if value is None else value


def _resolve_gre_twix_logical_matrix(
    *,
    base_resolution: int,
    header_partition_count: int,
    mdh_partitions: Sequence[Any],
    expected_matrix_ro_lin_par: Sequence[int],
) -> tuple[tuple[int, int, int], dict[str, Any]]:
    """Resolve GRE geometry from sequence dimensions and measured MDH support.

    Siemens Pulseq TWIX headers can retain ``lPartitions=1`` even when a 3D
    acquisition contains a complete measured PAR range. Therefore the header
    value is recorded but never used as the logical partition dimension.

    Args:
        base_resolution: TWIX base readout resolution.
        header_partition_count: Raw ``sKSpace.lPartitions`` header value.
        mdh_partitions: Measured image-stream PAR counters.
        expected_matrix_ro_lin_par: Sequence-derived logical matrix.

    Returns:
        Validated logical matrix and header/MDH provenance.

    Raises:
        ValueError: If the base resolution or measured PAR support disagrees
            with the sequence geometry.
    """

    expected = tuple(int(value) for value in expected_matrix_ro_lin_par)
    if len(expected) != 3 or any(value <= 0 for value in expected):
        raise ValueError("GRE sequence matrix must contain three positive dimensions.")
    if int(base_resolution) != expected[0]:
        raise ValueError(
            f"TWIX base resolution {base_resolution} does not match sequence matrix {expected}."
        )
    values = np.asarray(mdh_partitions, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("TWIX MDH PAR counters must be a populated finite vector.")
    if not np.equal(values, np.rint(values)).all():
        raise ValueError("TWIX MDH PAR counters must be integers.")
    unique_partitions = tuple(sorted(set(np.rint(values).astype(np.int64).tolist())))
    expected_partitions = tuple(range(expected[2]))
    if unique_partitions != expected_partitions:
        raise ValueError(
            "TWIX measured MDH PAR support does not match the sequence partition grid."
        )
    return expected, {
        "raw_header_partition_count": int(header_partition_count),
        "raw_header_partition_count_used_as_geometry": False,
        "mdh_partition_min": unique_partitions[0],
        "mdh_partition_max": unique_partitions[-1],
        "mdh_partition_count": len(unique_partitions),
        "mdh_partition_support_matches_sequence": True,
    }


def inspect_gre_twix(
    twix_path: str | Path,
    *,
    expected_echo_times_s: Sequence[Any],
    expected_matrix_ro_lin_par: Sequence[int],
    expected_fov_mm_ro_lin_par: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate measured GRE TWIX geometry, echo order, and image sampling.

    Sparse Siemens phase-line extent and partition-count tags are recorded but
    are not used as logical dimensions. The matrix is bound by the sequence,
    TWIX base resolution, and complete measured MDH sampling support.

    Args:
        twix_path: Measured Wave-GRE TWIX file.
        expected_echo_times_s: Ordered TE values derived from the sequence.
        expected_matrix_ro_lin_par: Sequence-derived logical matrix.
        expected_fov_mm_ro_lin_par: Sequence-derived logical FOV in
            millimeters.

    Returns:
        Native mask and JSON validation metadata.
    """

    import mapvbvd

    root = mapvbvd.mapVBVD(str(Path(twix_path)), quiet=True)
    measurements = list(root) if isinstance(root, (list, tuple)) else [root]
    candidates = []
    for index, measurement in enumerate(measurements):
        image = getattr(measurement, "image", None)
        acquisitions = int(getattr(image, "NAcq", 0)) if image is not None else 0
        if acquisitions:
            candidates.append((acquisitions, index, measurement))
    if not candidates:
        raise ValueError("No populated TWIX image stream was found.")
    _, index, measurement = max(candidates, key=lambda item: (item[0], item[1]))
    upstream_index = 1 if len(measurements) > 1 else 0
    if index != upstream_index:
        raise ValueError("Selected TWIX measurement disagrees with the pinned GRE loader.")
    image = measurement.image
    yaps = measurement.hdr["MeasYaps"]
    base = int(float(_twix_value(yaps, ("sKSpace", "lBaseResolution"), -1)))
    header_partitions = int(float(_twix_value(yaps, ("sKSpace", "lPartitions"), -1)))
    sparse_phase_extent = int(float(_twix_value(yaps, ("sKSpace", "lPhaseEncodingLines"), -1)))
    logical_matrix, partition_evidence = _resolve_gre_twix_logical_matrix(
        base_resolution=base,
        header_partition_count=header_partitions,
        mdh_partitions=image.Par,
        expected_matrix_ro_lin_par=expected_matrix_ro_lin_par,
    )
    fov = tuple(
        float(_twix_value(yaps, key, math.nan))
        for key in (
            ("sSliceArray", "asSlice", "0", "dReadoutFOV"),
            ("sSliceArray", "asSlice", "0", "dPhaseFOV"),
            ("sSliceArray", "asSlice", "0", "dThickness"),
        )
    )
    expected_fov = tuple(float(value) for value in expected_fov_mm_ro_lin_par)
    if len(expected_fov) != 3 or not np.allclose(
        fov, expected_fov, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"TWIX nominal FOV {fov} does not match sequence FOV {expected_fov} mm."
        )
    raw_echoes = np.asarray(image.Eco, dtype=np.float64)
    if raw_echoes.ndim != 1 or raw_echoes.size == 0 or not np.isfinite(raw_echoes).all():
        raise ValueError("TWIX Eco counters must be a populated finite vector.")
    echo_count = len(set(np.rint(raw_echoes).astype(np.int64).tolist()))
    echo_times = tuple(
        float(_twix_value(yaps, ("alTE", str(index)), math.nan)) * 1e-6
        for index in range(echo_count)
    )
    validated_echo_times = validate_gre_echo_consistency(
        expected_echo_times_s, echo_times
    )
    mask, sampling = validate_gre_echo_sampling(
        image.Lin,
        image.Par,
        image.Eco,
        echo_times_s=validated_echo_times,
        matrix_lin_par=logical_matrix[1:],
    )
    return mask, {
        "measurement_index": index,
        "logical_matrix_ro_lin_par": list(logical_matrix),
        **partition_evidence,
        "raw_sparse_phase_extent_tag": sparse_phase_extent,
        "raw_sparse_phase_extent_used_as_geometry": False,
        "nominal_fov_mm_ro_lin_par": list(fov),
        "echo_count": len(validated_echo_times),
        "echo_times_s": list(validated_echo_times),
        "sequence_echo_count_match": True,
        "sequence_echo_times_match": True,
        "sampling": sampling,
        "historical_encoded_slab_thickness_and_target_fov_tags_used": False,
    }


def validate_gre_sequence(native: Any, sequence_path: str | Path) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray]:
    """Validate the matching Pulseq geometry and split its ADC trajectory.

    Args:
        native: Imported pinned GRE helper module.
        sequence_path: Matching Pulseq sequence.

    Returns:
        Sequence, validated config, image trajectories, and calibration trajectories.
    """

    sequence = native._load_sequence(Path(sequence_path))
    if "FOV" not in sequence.definitions:
        raise ValueError("GRE sequence must define authoritative FOV; TargetFOV is ignored.")
    cfg = native._derive_gre_config(sequence, yflip_override=None, zflip_override=None)
    required = {
        "Nx": 250,
        "Nz": 72,
        "Nx_os": 1000,
        "orientation": "TRA",
        "Ry": 3,
        "Rz": 1,
        "Nz_meas": 72,
    }
    for key, expected in required.items():
        if cfg[key] != expected:
            raise ValueError(f"Sequence {key}={cfg[key]!r}; expected {expected!r}.")
    ny = int(cfg["Ny"])
    if ny <= 0:
        raise ValueError("Sequence Ny must be positive.")
    expected_measured_lines = len(range(2, ny, 3))
    if int(cfg["Ny_meas"]) != expected_measured_lines:
        raise ValueError(
            f"Sequence Ny_meas={cfg['Ny_meas']!r}; expected {expected_measured_lines} "
            f"for the residue-2 R3 lattice on Ny={ny}."
        )
    fov = np.asarray(cfg["FOVxyz_m"], dtype=np.float64)
    if fov.shape != (3,) or not np.isfinite(fov).all() or np.any(fov <= 0):
        raise ValueError("Sequence FOV must contain three positive finite values.")
    fixed_fov_m = np.asarray(NATIVE_FOV_MM_RO_LIN_PAR, dtype=np.float64) / 1000.0
    if not np.allclose(fov[[0, 2]], fixed_fov_m[[0, 2]], rtol=0.0, atol=1e-12):
        raise ValueError(
            "Sequence readout/slice FOV must remain 220x180 mm; phase FOV may vary."
        )
    sequence_times = np.asarray(cfg["TE_s"], dtype=np.float64)
    if int(cfg["Necho"]) <= 0 or sequence_times.ndim != 1:
        raise ValueError("GRE sequence must define one or more ordered echoes.")
    if sequence_times.size != int(cfg["Necho"]):
        raise ValueError("GRE sequence echo count and TE list length disagree.")
    if not np.isfinite(sequence_times).all() or np.any(sequence_times <= 0):
        raise ValueError("GRE sequence echo times must be positive and finite.")
    image_lines, calibration_lines = native._split_adc_trajectory(sequence, cfg)
    if native._detect_image_wave_mode(image_lines, cfg) != "wave":
        raise ValueError("GRE sequence does not contain two-axis Wave image encoding.")
    return sequence, cfg, image_lines, calibration_lines


def _embed_measured_echoes(
    data: Any,
    mask: np.ndarray,
    *,
    echo_count: int,
    coil_count: int,
    matrix_ro_lin_par: Sequence[int] = NATIVE_MATRIX_RO_LIN_PAR,
) -> Any:
    """Embed mapVBVD multi-echo payloads on the exact native logical grid.

    Args:
        data: Upstream normalized tensor in RO/LIN/PAR/ECHO/COIL order.
        mask: Validated native image sampling mask.
        echo_count: Positive sequence/TWIX-matched echo count.
        coil_count: Expected coil count after any prior compression.
        matrix_ro_lin_par: Sequence-derived native logical matrix.

    Returns:
        Complex Torch tensor on the extended-readout native LIN/PAR grid.
    """

    import torch

    matrix = tuple(int(value) for value in matrix_ro_lin_par)
    if len(matrix) != 3 or matrix[0] != 250 or any(value <= 0 for value in matrix):
        raise ValueError("GRE native matrix must use the supported 250-point readout.")
    _, nlin, npar = matrix
    if np.asarray(mask).shape != (nlin, npar):
        raise ValueError("GRE sampling mask does not match the native LIN/PAR matrix.")
    values = data if torch.is_tensor(data) else torch.as_tensor(data)
    if (
        values.ndim != 5
        or values.shape[0] != EXTENDED_READOUT
        or values.shape[3] != echo_count
    ):
        raise ValueError("GRE image payload must have RO/LIN/PAR/ECHO/COIL dimensions.")
    if values.shape[-1] != coil_count or values.shape[1] > nlin or values.shape[2] > npar:
        raise ValueError("GRE image payload geometry or coil count is incompatible with preparation.")
    full = torch.zeros(
        (EXTENDED_READOUT, nlin, npar, echo_count, coil_count),
        dtype=torch.complex64,
    )
    full[:, : values.shape[1], : values.shape[2], :, :] = values.to(torch.complex64)
    logical_mask = torch.from_numpy(np.asarray(mask, dtype=bool)).view(
        1, nlin, npar, 1, 1
    )
    # Bound support validation because the physical-coil full grid is large.
    for start in range(0, EXTENDED_READOUT, 8):
        outside = full[start : start + 8].masked_select(~logical_mask)
        if outside.numel() and torch.count_nonzero(outside).item():
            raise ValueError("GRE payload contains nonzero data outside the validated R3x1 image lattice.")
    return full * logical_mask


def _shared_calibration_id(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> str:
    """Hash one processed a/b/c solution for cross-echo identity checks.

    Args:
        a: LIN phase-plane coefficient vector.
        b: PAR phase-plane coefficient vector.
        c: Constant phase coefficient vector.

    Returns:
        SHA-256 over a canonical stacked float64 array.
    """

    vectors = np.stack([np.asarray(value, dtype=np.float64) for value in (a, b, c)])
    return logical_array_sha256(vectors)


def _evaluate_echo_psfs(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: Mapping[str, Any],
    case: GreCase,
) -> list[np.ndarray]:
    """Evaluate one shared calibration with each echo-specific trajectory.

    Args:
        trajectories: One sequence-derived LIN/PAR pair per echo.
        coefficients: Shared processed a/b/c vectors.
        cfg: Validated GRE sequence configuration.
        case: Target geometry.

    Returns:
        One calibrated unit-magnitude target-grid PSF per echo.
    """

    if not trajectories or len(trajectories) != int(cfg["Necho"]):
        raise ValueError("GRE trajectory count must match the positive sequence echo count.")
    a_fit, b_fit, c_fit = coefficients
    nlin, npar = case.matrix_ro_lin_par[1:]
    return [
        evaluate_calibrated_psf(
            delta_lin,
            delta_par,
            a_fit,
            b_fit,
            c_fit,
            nlin=nlin,
            npar=npar,
            lin_sign=int(cfg["yflip"]),
            par_sign=int(cfg["zflip"]),
        )
        for delta_lin, delta_par in trajectories
    ]


def _case_mask(case: GreCase) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one exact pure Cartesian case mask.

    Args:
        case: GRE case definition.

    Returns:
        Boolean LIN/PAR mask and canonical metadata.
    """

    mask, metadata = pure_cartesian_image_lattice_mask(
        case.matrix_ro_lin_par[1:],
        acceleration_lin_par=case.acceleration_lin_par,
        residue_lin_par=case.residue_lin_par,
    )
    return mask, validate_pure_cartesian_image_lattice(mask, metadata)


def _crop_gre_wave_coil_in_pe(
    values: np.ndarray,
    *,
    lin_bounds: tuple[int, int],
    par_bounds: tuple[int, int],
    target_mask: np.ndarray,
) -> np.ndarray:
    """Crop one GRE Wave coil only in phase-encoding dimensions.

    Args:
        values: One complex coil in extended-RO/LIN/PAR order.
        lin_bounds: Half-open source LIN crop bounds.
        par_bounds: Half-open source PAR crop bounds.
        target_mask: Boolean target LIN/PAR sampling mask.

    Returns:
        A complex64 copy with the full extended readout retained and the target
        sampling mask applied.

    Raises:
        ValueError: If dimensionality, bounds, or mask shape are inconsistent.
    """

    source = np.asarray(values)
    mask = np.asarray(target_mask, dtype=bool)
    if source.ndim != 3:
        raise ValueError("One GRE Wave coil must have extended-RO/LIN/PAR dimensions.")
    lin_start, lin_stop = (int(value) for value in lin_bounds)
    par_start, par_stop = (int(value) for value in par_bounds)
    if not (0 <= lin_start < lin_stop <= source.shape[1]):
        raise ValueError("GRE LIN crop bounds lie outside the source grid.")
    if not (0 <= par_start < par_stop <= source.shape[2]):
        raise ValueError("GRE PAR crop bounds lie outside the source grid.")
    if mask.shape != (lin_stop - lin_start, par_stop - par_start):
        raise ValueError("GRE target mask does not match the PE crop dimensions.")
    cropped = np.array(
        source[:, lin_start:lin_stop, par_start:par_stop],
        dtype=np.complex64,
        copy=True,
    )
    cropped *= mask[None, :, :]
    return cropped


def _write_echo_wave(
    values: np.ndarray,
    output_base: Path,
    case: GreCase,
    target_mask: np.ndarray,
) -> float:
    """Directly crop and mask one measured-Wave echo for a GRE case.

    Args:
        values: Native measured-Wave data in extended-RO/LIN/PAR/COIL order.
        output_base: Destination BART CFL basename.
        case: Target case with exact half-open crop bounds.
        target_mask: Pure target image lattice.

    Returns:
        L2 norm of the written BART k-space.
    """

    source = np.asarray(values)
    source_matrix = case.source_matrix_ro_lin_par
    expected_source_shape = (
        EXTENDED_READOUT,
        source_matrix[1],
        source_matrix[2],
    )
    if source.ndim != 4 or source.shape[:3] != expected_source_shape:
        raise ValueError("Native measured-Wave echo has an unexpected shape.")
    ro_bounds, lin_bounds, par_bounds = case.crop_bounds_from_native
    if ro_bounds != (0, source_matrix[0]):
        raise ValueError("GRE retrospective cases must retain the full image readout FOV.")
    target_shape = (
        1000,
        case.matrix_ro_lin_par[1],
        case.matrix_ro_lin_par[2],
        source.shape[3],
    )
    output = create_cfl(output_base, (*target_shape, 1))
    squared_norm = 0.0
    # Process one coil at a time to avoid a second multi-gigabyte case array.
    for coil in range(source.shape[3]):
        cropped = _crop_gre_wave_coil_in_pe(
            source[:, :, :, coil],
            lin_bounds=lin_bounds,
            par_bounds=par_bounds,
            target_mask=target_mask,
        )
        if cropped.shape != target_shape[:3]:
            raise AssertionError("Measured-Wave crop produced the wrong GRE grid.")
        output[:, :, :, coil, 0] = cropped
        squared_norm += float(np.vdot(cropped, cropped).real)
    output.flush()
    del output
    return float(math.sqrt(squared_norm))


def _validate_recoverable_retro_directory(
    directory: Path, echo_count: int
) -> None:
    """Allow only known partial GRE preparation artifacts before resuming.

    Args:
        directory: Retrospective case BART-input directory without a manifest.
        echo_count: Positive number of expected echoes.

    Returns:
        None. Known partial files may be overwritten by deterministic
        preparation; unexpected content is never removed or modified.

    Raises:
        FileExistsError: If the incomplete directory contains unknown entries.
    """

    if not directory.exists():
        return
    allowed = {"sampling_mask.npy"}
    for echo_label in gre_echo_ids(echo_count):
        for stem in (f"wave_kspace_{echo_label}", f"psf_{echo_label}"):
            allowed.update({f"{stem}.hdr", f"{stem}.cfl"})
    unexpected = sorted(path.name for path in directory.iterdir() if path.name not in allowed)
    if unexpected:
        raise FileExistsError(
            f"Incomplete retrospective GRE directory contains unexpected entries: "
            f"{directory}: {unexpected}"
        )


def _source_matches(existing: Mapping[str, Any], twix: Path, sequence: Path, settings: Mapping[str, Any]) -> bool:
    """Check whether a native manifest can be safely reused.

    Args:
        existing: Existing native manifest.
        twix: Current measured TWIX path.
        sequence: Current sequence path.
        settings: Current PSF-processing request.

    Returns:
        True only when source identities and settings match.
    """

    source = existing.get("source", {})
    echoes = existing.get("echoes")
    if not isinstance(echoes, list) or not echoes:
        return False
    echo_count = len(echoes)
    if [item.get("echo") for item in echoes] != list(range(1, echo_count + 1)):
        return False
    return bool(
        source.get("twix") == _file_identity(twix)
        and source.get("sequence") == _file_identity(sequence, include_hash=True)
        and existing.get("psf_calibration", {}).get("request") == settings
        and existing.get("wavelet_selection")
        == gre_wavelet_selection_provenance("native_r3x1", echo_count)
    )


def _normal_manifest_matches_reusable_artifact(
    manifest: Mapping[str, Any],
    twix: Path,
    sequence: Path,
    settings: Mapping[str, Any],
) -> bool:
    """Check whether legacy GRE normal artifacts may be reused unchanged.

    This retrospective-only path binds the recorded sources and accepts the
    already materialized measured PSFs without claiming that current defaults
    or implementation code produced them. Manual fit overrides remain strict.

    Args:
        manifest: Existing normal GRE preparation manifest.
        twix: Requested measured GRE TWIX path.
        sequence: Requested matching Pulseq sequence path.
        settings: Normalized current PSF-processing request.

    Returns:
        ``True`` when source identity and the narrow compatibility policy match.
    """

    source = manifest.get("source", {})
    echoes = manifest.get("echoes")
    if (
        manifest.get("status") != "measured_multi_echo_wave_gre_bart_inputs_ready"
        or source.get("twix") != _file_identity(twix)
        or source.get("sequence") != _file_identity(sequence, include_hash=True)
        or not isinstance(manifest.get("geometry"), Mapping)
        or not isinstance(manifest.get("sampling"), Mapping)
        or not isinstance(echoes, list)
        or not echoes
        or [item.get("echo") for item in echoes]
        != list(range(1, len(echoes) + 1))
    ):
        return False
    if settings.get("requested_fit_kx_range") is not None:
        return False
    psf_calibration = manifest.get("psf_calibration", {})
    if not isinstance(psf_calibration, Mapping):
        return False
    recorded_request = psf_calibration.get("request")
    if isinstance(recorded_request, Mapping):
        recorded_mode = recorded_request.get("coefficient_processing")
        if recorded_mode not in {None, "smooth", "sine-line"}:
            return False
    return True


def _validated_normal_sampling(
    destination: Path, manifest: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate the measured normal GRE image-stream lattice.

    Args:
        destination: Existing ``normal/bart_inputs`` directory.
        manifest: Normal preparation manifest containing sampling metadata.

    Returns:
        Boolean source mask and recomputed canonical pure-lattice metadata.

    Raises:
        FileNotFoundError: If the local sampling mask is absent.
        ValueError: If the mask differs from its recorded pure-lattice contract.
    """

    mask_path = destination / "sampling_mask.npy"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing normal GRE sampling mask: {mask_path}")
    mask = np.load(mask_path, allow_pickle=False)
    metadata = dict(manifest.get("sampling", {}))
    canonical = validate_pure_cartesian_image_lattice(mask, metadata)
    return np.asarray(mask, dtype=bool), canonical


def _finite_cfl(base: Path, *, readout_chunk: int = 8) -> bool:
    """Check one BART CFL for finite values without a full-memory copy.

    Args:
        base: BART CFL basename.
        readout_chunk: Maximum first-axis planes checked together.

    Returns:
        ``True`` only when every complex sample is finite.
    """

    if readout_chunk < 1:
        raise ValueError("Readout finite-check chunk must be positive.")
    values = open_cfl(base)
    for start in range(0, values.shape[0], readout_chunk):
        if not np.isfinite(np.asarray(values[start : start + readout_chunk])).all():
            return False
    return True


def _normal_core_artifact_records(
    destination: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Validate reusable multi-echo GRE core artifacts and record identities.

    Args:
        destination: Existing ``normal/bart_inputs`` directory.
        manifest: Historical manifest supplying geometry and echo filenames.

    Returns:
        Core artifact records and missing nonessential legacy diagnostics.

    Raises:
        FileNotFoundError: If required measured k-space, calibration, mask, or
            per-echo PSF files are absent.
        ValueError: If geometry or finite-value checks fail.
    """

    geometry = manifest["geometry"]
    matrix = tuple(int(value) for value in geometry["matrix_ro_lin_par"])
    if len(matrix) != 3 or any(value <= 0 for value in matrix):
        raise ValueError("Legacy normal GRE manifest has invalid geometry.")
    source_mask, sampling = _validated_normal_sampling(destination, manifest)
    if source_mask.shape != matrix[1:]:
        raise ValueError("Legacy normal GRE mask and geometry disagree.")
    if tuple(sampling["acceleration_lin_par"]) != (3, 1):
        raise ValueError("Legacy normal GRE source is not a regular R3x1 lattice.")

    echoes = manifest["echoes"]
    calibration_shape = read_shape(destination / str(manifest.get("kspace_calib", "kspace_calib")))
    padded_calibration = calibration_shape + (1,) * max(0, 5 - len(calibration_shape))
    if (
        padded_calibration[:3] != matrix
        or padded_calibration[3] <= 0
        or any(value != 1 for value in padded_calibration[4:])
    ):
        raise ValueError("Legacy normal GRE calibration geometry is inconsistent.")
    records: dict[str, Any] = {
        "sampling_mask": {
            "shape_lin_par": list(source_mask.shape),
            "logical_sha256": sampling["logical_sha256"],
            "acquired_coordinate_count": sampling["acquired_coordinate_count"],
        },
        "kspace_calib": {
            "shape": list(calibration_shape),
            "header_sha256": sha256_file(
                (destination / str(manifest.get("kspace_calib", "kspace_calib"))).with_suffix(".hdr")
            ),
            "payload_size_bytes": (
                destination / str(manifest.get("kspace_calib", "kspace_calib"))
            ).with_suffix(".cfl").stat().st_size,
        },
        "echoes": [],
    }
    expected_coils = padded_calibration[3]
    for echo_index, echo in enumerate(echoes, start=1):
        wave_name = str(echo.get("wave_kspace", f"wave_kspace_echo-{echo_index:02d}"))
        psf_name = str(echo.get("psf", f"psf_echo-{echo_index:02d}"))
        wave_base = destination / wave_name
        psf_base = destination / psf_name
        wave_shape = read_shape(wave_base)
        psf_shape = read_shape(psf_base)
        padded_wave = wave_shape + (1,) * max(0, 5 - len(wave_shape))
        padded_psf = psf_shape + (1,) * max(0, 5 - len(psf_shape))
        if (
            padded_wave[:5] != (EXTENDED_READOUT, matrix[1], matrix[2], expected_coils, 1)
            or any(value != 1 for value in padded_wave[5:])
            or padded_psf[:3] != (EXTENDED_READOUT, matrix[1], matrix[2])
            or any(value != 1 for value in padded_psf[3:])
        ):
            raise ValueError(
                f"Legacy normal GRE echo {echo_index} has inconsistent core geometry."
            )
        if not _finite_cfl(psf_base):
            raise ValueError(f"Legacy normal GRE echo {echo_index} PSF is non-finite.")
        records["echoes"].append(
            {
                "echo": echo_index,
                "wave_kspace": wave_name,
                "wave_kspace_shape": list(wave_shape),
                "wave_kspace_header_sha256": sha256_file(wave_base.with_suffix(".hdr")),
                "wave_kspace_payload_size_bytes": wave_base.with_suffix(".cfl").stat().st_size,
                "psf": psf_name,
                "psf_shape": list(psf_shape),
                "psf_header_sha256": sha256_file(psf_base.with_suffix(".hdr")),
                "psf_payload_sha256": sha256_file(psf_base.with_suffix(".cfl")),
                "psf_finite": True,
            }
        )

    missing: list[str] = []
    coefficient_path = destination / "shared_psf_coefficients.npz"
    if coefficient_path.is_file():
        coefficients, _ = _read_shared_psf_coefficients(coefficient_path)
        if not all(np.isfinite(value).all() for value in coefficients):
            raise ValueError("Legacy normal GRE PSF coefficients are non-finite.")
        records["shared_psf_coefficients"] = {
            "sha256": sha256_file(coefficient_path),
            "processed_values_finite": True,
        }
    else:
        missing.append("shared_psf_coefficients.npz")
    for echo_index, echo in enumerate(echoes, start=1):
        trajectory_name = echo.get("sequence_trajectory")
        if trajectory_name is None or not (destination / str(trajectory_name)).is_file():
            missing.append(str(trajectory_name or f"sequence_trajectory_echo-{echo_index:02d}.npz"))
            continue
        trajectory_path = destination / str(trajectory_name)
        with np.load(trajectory_path) as values:
            delta_lin = np.asarray(values["delta_lin"])
            delta_par = np.asarray(values["delta_par"])
        if (
            delta_lin.size != EXTENDED_READOUT
            or delta_par.size != EXTENDED_READOUT
            or not np.isfinite(delta_lin).all()
            or not np.isfinite(delta_par).all()
        ):
            raise ValueError(f"Legacy normal GRE echo {echo_index} trajectory is invalid.")
    return records, missing


def _write_normal_reuse_attestation(
    destination: Path,
    manifest: Mapping[str, Any],
    settings: Mapping[str, Any],
    artifact_records: Mapping[str, Any],
    missing_artifacts: Sequence[str],
) -> Path:
    """Record non-destructive retrospective reuse of legacy GRE normal inputs.

    Args:
        destination: Existing ``normal/bart_inputs`` directory.
        manifest: Unmodified historical normal manifest.
        settings: Current default request that was not applied to the PSF.
        artifact_records: Validated core artifact identities.
        missing_artifacts: Nonessential legacy diagnostics not present.

    Returns:
        Atomically written attestation path beside ``bart_inputs``.
    """

    manifest_path = destination / "manifest.json"
    calibration = manifest.get("psf_calibration", {})
    recorded_request = (
        calibration.get("request", {}) if isinstance(calibration, Mapping) else {}
    )
    path = destination.parent / NORMAL_REUSE_ATTESTATION_NAME
    stable_payload = {
        "format_version": 1,
        "status": "legacy_normal_inputs_accepted_for_retrospective_reuse",
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "modified": False,
        },
        "reuse_scope": "retrospective preparation only; no PSF refit or recalibration",
        "recorded_psf_coefficient_processing": (
            recorded_request.get("coefficient_processing")
            if isinstance(recorded_request, Mapping)
            else None
        ),
        "current_default_request_not_applied_to_existing_psf": dict(settings),
        "core_artifacts": artifact_records,
        "missing_nonessential_artifacts": list(missing_artifacts),
        "scientific_artifacts_modified": False,
    }
    if path.is_file():
        existing = _load_json(path)
        existing_stable = {
            key: value for key, value in existing.items() if key != "created_at_utc"
        }
        if existing_stable == stable_payload:
            return path
    _write_json(path, {**stable_payload, "created_at_utc": _utc_now()})
    return path


def _native_r3x3_residue(
    sampling: Mapping[str, Any], npar: int
) -> tuple[int, int]:
    """Derive a source-subset native R3x3 residue from measured GRE sampling.

    Args:
        sampling: Validated normal pure-lattice sampling metadata.
        npar: Native logical slice/partition count.

    Returns:
        LIN residue inherited from measured R3x1 and center-aligned PAR residue.

    Raises:
        ValueError: If the source is not a regular single-residue R3x1 lattice.
    """

    acceleration = tuple(int(value) for value in sampling["acceleration_lin_par"])
    residue = tuple(int(value) for value in sampling["residue_lin_par"])
    if acceleration != (3, 1) or len(residue) != 2 or residue[1] != 0:
        raise ValueError(
            "Native GRE R3x3 retrospective undersampling requires regular R3x1 data."
        )
    if not 0 <= residue[0] < 3 or int(npar) <= 0:
        raise ValueError("Measured GRE R3x1 residue or partition count is invalid.")
    return residue[0], (int(npar) // 2) % 3


def _normal_shared_calibration_id(
    destination: Path, manifest: Mapping[str, Any]
) -> str:
    """Resolve one stable shared-calibration identity for legacy GRE echoes.

    Args:
        destination: Existing normal GRE BART-input directory.
        manifest: Normal preparation manifest.

    Returns:
        Recorded coefficient identity, recomputed coefficient identity, or a
        clearly labeled digest of the immutable per-echo PSF payload set.

    Raises:
        ValueError: If recorded echo calibration identities disagree.
    """

    echoes = manifest.get("echoes", [])
    recorded = {echo.get("shared_calibration_id") for echo in echoes}
    recorded.discard(None)
    if len(recorded) > 1:
        raise ValueError("Normal GRE echoes disagree on shared calibration identity.")
    if recorded:
        return str(next(iter(recorded)))
    calibration = manifest.get("psf_calibration", {})
    calibration_id = (
        calibration.get("shared_calibration_id")
        if isinstance(calibration, Mapping)
        else None
    )
    if calibration_id:
        return str(calibration_id)
    coefficient_path = destination / "shared_psf_coefficients.npz"
    if coefficient_path.is_file():
        coefficients, _ = _read_shared_psf_coefficients(coefficient_path)
        return _shared_calibration_id(*coefficients)
    digest = hashlib.sha256()
    for echo in echoes:
        psf = destination / str(echo["psf"])
        digest.update(sha256_file(psf.with_suffix(".hdr")).encode("ascii"))
        digest.update(sha256_file(psf.with_suffix(".cfl")).encode("ascii"))
    return f"legacy-psf-set-sha256:{digest.hexdigest()}"


def _single_map_wave_values(base: Path) -> np.ndarray:
    """Expose one normal GRE Wave CFL as extended-RO/LIN/PAR/COIL data.

    Args:
        base: Normal per-echo measured-Wave BART basename.

    Returns:
        Memory-mapped logical values with singleton map/trailing axes removed.

    Raises:
        ValueError: If the BART input contains more than one map or a
            non-singleton trailing dimension.
    """

    shape = read_shape(base)
    padded = shape + (1,) * max(0, 5 - len(shape))
    if padded[4] != 1 or any(value != 1 for value in padded[5:]):
        raise ValueError("Normal GRE measured-Wave input must contain one map.")
    return open_cfl(base).reshape((*padded[:4], -1), order="F")[..., 0]


def _coefficient_arrays(
    values: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert three tensor-like PSF coefficient vectors to NumPy arrays.

    Args:
        values: Processed or raw ``a``, ``b``, and ``c`` coefficient vectors.

    Returns:
        Three float64 one-dimensional NumPy arrays.

    Raises:
        ValueError: If the vectors do not share one positive length.
    """

    if len(values) != 3:
        raise ValueError("PSF coefficients must contain a, b, and c vectors.")
    arrays = tuple(
        np.asarray(
            value.detach().cpu() if hasattr(value, "detach") else value,
            dtype=np.float64,
        ).reshape(-1)
        for value in values
    )
    if not arrays[0].size or len({value.size for value in arrays}) != 1:
        raise ValueError("PSF coefficient vectors must share one positive length.")
    return arrays


def _write_shared_psf_coefficients(
    path: Path,
    processed: tuple[np.ndarray, np.ndarray, np.ndarray],
    raw: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Persist processed and raw shared GRE PSF coefficients in one archive.

    Args:
        path: Destination NumPy ``.npz`` path.
        processed: Reconstruction coefficients ``a``, ``b``, and ``c``.
        raw: Corresponding measured coefficient samples.

    Returns:
        None.
    """

    np.savez(
        path,
        a=processed[0],
        b=processed[1],
        c=processed[2],
        a_raw=raw[0],
        b_raw=raw[1],
        c_raw=raw[2],
    )


def _read_shared_psf_coefficients(
    path: Path,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray] | None,
]:
    """Load processed and optional raw GRE PSF coefficients from an archive.

    Args:
        path: Existing NumPy ``.npz`` path.

    Returns:
        Processed coefficients and raw samples, or ``None`` for a legacy
        processed-only archive.

    Raises:
        ValueError: If required or raw archive keys are incomplete.
    """

    with np.load(path) as values:
        if not all(key in values for key in ("a", "b", "c")):
            raise ValueError(f"Shared PSF archive is incomplete: {path}")
        processed = _coefficient_arrays(tuple(values[key] for key in ("a", "b", "c")))
        raw_presence = tuple(key in values for key in ("a_raw", "b_raw", "c_raw"))
        if any(raw_presence) and not all(raw_presence):
            raise ValueError(f"Raw shared PSF archive keys are incomplete: {path}")
        raw = (
            _coefficient_arrays(
                tuple(values[key] for key in ("a_raw", "b_raw", "c_raw"))
            )
            if all(raw_presence)
            else None
        )
    return processed, raw


def _recover_legacy_raw_psf_coefficients(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Recover raw coefficients written by the pinned GRE calibration helper.

    Args:
        directory: Existing native GRE BART-input directory.

    Returns:
        Raw ``a``, ``b``, and combined ``c`` vectors, or ``None`` when the
        legacy projection caches are unavailable.
    """

    paths = {
        "a": directory / "a_fit_all_projy_shared.npy",
        "b": directory / "b_fit_all_projz_shared.npy",
        "c_y": directory / "c_fit_all_projy_shared.npy",
        "c_z": directory / "c_fit_all_projz_shared.npy",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    return _coefficient_arrays(
        (
            np.load(paths["a"], allow_pickle=False),
            np.load(paths["b"], allow_pickle=False),
            np.load(paths["c_y"], allow_pickle=False)
            + np.load(paths["c_z"], allow_pickle=False),
        )
    )


def _write_gre_psf_coefficient_plots(
    normal_directory: Path,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray],
    raw_coefficients: tuple[np.ndarray, np.ndarray, np.ndarray],
    settings: Mapping[str, Any],
    processing_diagnostics: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write fixed-range and full-range GRE coefficient diagnostics.

    Args:
        normal_directory: Dataset ``normal`` output directory.
        coefficients: Processed shared ``a``, ``b``, and ``c`` vectors.
        raw_coefficients: Corresponding measured coefficient samples.
        settings: Normalized coefficient-processing request.
        processing_diagnostics: Accepted upstream fitting diagnostics.

    Returns:
        Fixed-range and full-range diagnostic PNG paths, in that order.
    """

    selected_range = processing_diagnostics.get("kx_range")
    fit_kx_range = (
        None
        if selected_range is None
        else (int(selected_range[0]), int(selected_range[1]))
    )
    common_arguments = {
        "processing": str(settings["coefficient_processing"]),
        "fit_kx_range": fit_kx_range,
        "fit_range_selection": processing_diagnostics.get("fit_range_selection"),
        "raw_coefficients": raw_coefficients,
    }
    fixed_range_path = write_psf_coefficient_plot(
        *coefficients,
        normal_directory / PSF_COEFFICIENT_PLOT_NAME,
        **common_arguments,
    )
    full_range_path = write_psf_coefficient_plot(
        *coefficients,
        normal_directory / PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME,
        **common_arguments,
        full_range=True,
    )
    print(f"PSF coefficient visual-assessment plot: {fixed_range_path}")
    print(f"PSF coefficient full-range plot: {full_range_path}")
    return fixed_range_path, full_range_path


def prepare_normal_gre(
    twix: str | Path,
    output_root: str | Path,
    sequence: str | Path,
    *,
    psf_coefficient_processing: str = "sine-line",
    psf_fit_kx_min: int | None = None,
    psf_fit_kx_max: int | None = None,
    reuse: bool = True,
    allow_legacy_artifact_reuse: bool = False,
) -> dict[str, Any]:
    """Prepare native measured R3x1 GRE BART inputs for one or more echoes.

    Args:
        twix: Measured Wave-GRE TWIX file.
        output_root: Exact user-selected output root.
        sequence: Matching integrated Wave-GRE Pulseq file.
        psf_coefficient_processing: Shared ``sine-line`` mode by default, or
            explicit ``smooth`` processing.
        psf_fit_kx_min: Optional inclusive manual sine-line bound.
        psf_fit_kx_max: Optional exclusive manual sine-line bound.
        reuse: Reuse an identity-matched completed input set.
        allow_legacy_artifact_reuse: Permit retrospective callers to reuse
            source-matched legacy measured k-space and PSFs without rewriting
            their historical manifest or recalibrating the operator.

    Returns:
        Native preparation manifest.
    """

    import torch

    twix_path = Path(twix).expanduser().resolve()
    sequence_path = Path(sequence).expanduser().resolve()
    settings = _normalize_psf_settings(psf_coefficient_processing, psf_fit_kx_min, psf_fit_kx_max)
    destination = Path(output_root).expanduser().resolve() / NORMAL_INPUT_RELATIVE
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file() and reuse:
        existing = _load_json(manifest_path)
        exact_match = _source_matches(existing, twix_path, sequence_path, settings)
        artifact_match = allow_legacy_artifact_reuse and (
            _normal_manifest_matches_reusable_artifact(
                existing, twix_path, sequence_path, settings
            )
        )
        if not exact_match and not artifact_match:
            raise ValueError("Existing normal GRE inputs use different sources or PSF settings.")
        if artifact_match and not exact_match:
            artifact_records, missing_artifacts = _normal_core_artifact_records(
                destination, existing
            )
            attestation = _write_normal_reuse_attestation(
                destination,
                existing,
                settings,
                artifact_records,
                missing_artifacts,
            )
            calibration = existing.get("psf_calibration", {})
            recorded_request = (
                calibration.get("request", {})
                if isinstance(calibration, Mapping)
                else {}
            )
            recorded_mode = (
                recorded_request.get("coefficient_processing")
                if isinstance(recorded_request, Mapping)
                else None
            )
            print(
                "Accepted existing normal GRE artifacts without recalibration; "
                f"recorded PSF mode: {recorded_mode or 'unknown'}; "
                f"current defaults were not applied; reuse attestation: {attestation}"
            )
            return existing
        for echo in existing["echoes"]:
            read_shape(destination / str(echo["wave_kspace"]))
            read_shape(destination / str(echo["psf"]))
        read_shape(destination / "kspace_calib")
        coefficient_path = destination / "shared_psf_coefficients.npz"
        if allow_legacy_artifact_reuse and not coefficient_path.is_file():
            artifact_records, missing_artifacts = _normal_core_artifact_records(
                destination, existing
            )
            attestation = _write_normal_reuse_attestation(
                destination,
                existing,
                settings,
                artifact_records,
                missing_artifacts,
            )
            print(
                "Accepted existing normal GRE artifacts without diagnostic upgrade; "
                f"reuse attestation: {attestation}"
            )
            return existing
        coefficients, raw_coefficients = _read_shared_psf_coefficients(coefficient_path)
        if raw_coefficients is None:
            raw_coefficients = _recover_legacy_raw_psf_coefficients(destination)
            if raw_coefficients is None:
                if allow_legacy_artifact_reuse:
                    artifact_records, missing_artifacts = _normal_core_artifact_records(
                        destination, existing
                    )
                    attestation = _write_normal_reuse_attestation(
                        destination,
                        existing,
                        settings,
                        artifact_records,
                        missing_artifacts,
                    )
                    print(
                        "Accepted existing normal GRE artifacts without raw-coefficient "
                        f"diagnostic upgrade; reuse attestation: {attestation}"
                    )
                    return existing
                raise ValueError(
                    "Existing GRE inputs predate raw PSF persistence and their "
                    "projection caches are unavailable; use a new output root to "
                    "regenerate the PSF scatter diagnostic."
                )
            _write_shared_psf_coefficients(
                coefficient_path, coefficients, raw_coefficients
            )
        processing = existing.get("psf_calibration", {}).get(
            "processing_diagnostics", {}
        )
        _write_gre_psf_coefficient_plots(
            destination.parent,
            coefficients,
            raw_coefficients,
            settings,
            processing,
        )
        calibration_record = existing.setdefault("psf_calibration", {})
        calibration_record.update(
            {
                "coefficient_file": "shared_psf_coefficients.npz",
                "processed_coefficient_keys": ["a", "b", "c"],
                "raw_coefficient_keys": ["a_raw", "b_raw", "c_raw"],
                "visual_assessment_plot_relative_to_output_root": (
                    f"normal/{PSF_COEFFICIENT_PLOT_NAME}"
                ),
                "full_range_plot_relative_to_output_root": (
                    f"normal/{PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME}"
                ),
            }
        )
        existing["output_orientation"] = {
            "logical_axis_roles": list(GRE_LOGICAL_AXIS_ROLES),
            "validated_array_axis_flips": list(GRE_BART_ARRAY_AXIS_FLIPS),
            "stored_coordinate_system": "canonical RAS",
            "interpolation": False,
        }
        existing["format_version"] = max(int(existing.get("format_version", 1)), 2)
        _write_json(manifest_path, existing)
        return existing
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Normal GRE BART input directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    native = load_wave_gre_helpers()
    _, cfg, image_lines, calibration_lines = validate_gre_sequence(native, sequence_path)
    native_matrix = (int(cfg["Nx"]), int(cfg["Ny"]), int(cfg["Nz"]))
    native_fov_mm = tuple(
        float(value) * 1000.0 for value in np.asarray(cfg["FOVxyz_m"])
    )
    cases = gre_cases(native_matrix, native_fov_mm)
    echo_count = int(cfg["Necho"])
    echo_times = tuple(float(value) for value in cfg["TE_s"])
    source_mask, twix_metadata = inspect_gre_twix(
        twix_path,
        expected_echo_times_s=echo_times,
        expected_matrix_ro_lin_par=native_matrix,
        expected_fov_mm_ro_lin_par=native_fov_mm,
    )
    reference = native._check_integrated_refscan_shape(
        native.load_ref(str(twix_path)),
        ncalib1=int(cfg["Ncalib1"]),
        nacs=int(cfg["Nacs"]),
        nsets=int(cfg["Nsets"]),
    )
    physical_coils = int(reference.shape[-1])
    if reference.shape[0] != EXTENDED_READOUT or physical_coils < VIRTUAL_COILS:
        raise ValueError("GRE refscan readout or physical coil count is incompatible.")
    # The set-4 integrated ACS determines one coil basis and one native map input.
    nacs = int(cfg["Nacs"])
    os_factor = int(cfg["os_factor"])
    if nacs <= 0 or nacs > min(native_matrix[1:]):
        raise ValueError("GRE ACS matrix does not fit within the native LIN/PAR grid.")
    integrated_acs = reference[:, :nacs, :nacs, int(cfg["ACSSetID"]), :]
    basis, singular_values, retained_energy = native.estimate_cc_matrix_coillast(
        integrated_acs, ncc=VIRTUAL_COILS, acs=nacs, x_step=os_factor
    )
    image = native._normalize_gre_image_data(native.load_img(str(twix_path)), cfg)
    if int(image.shape[-1]) != physical_coils:
        raise ValueError("GRE image and integrated refscan coil counts disagree.")
    compressed_compact = torch.empty(
        (*image.shape[:-1], VIRTUAL_COILS), dtype=torch.complex64
    )
    for echo_index in range(echo_count):
        compressed_compact[:, :, :, echo_index, :] = native.apply_cc_coillast_torch(
            image[:, :, :, echo_index, :], basis, x_chunk=8
        )
    del image
    compressed = _embed_measured_echoes(
        compressed_compact,
        source_mask,
        echo_count=echo_count,
        coil_count=VIRTUAL_COILS,
        matrix_ro_lin_par=native_matrix,
    )
    del compressed_compact
    compressed_acs = native.apply_cc_coillast_torch(integrated_acs, basis, x_chunk=8)[::os_factor]
    if tuple(compressed_acs.shape) != (
        native_matrix[0],
        nacs,
        nacs,
        VIRTUAL_COILS,
    ):
        raise ValueError("Compressed GRE ACS has an unexpected shape.")
    if not torch.isfinite(compressed).all() or not torch.isfinite(compressed_acs).all():
        raise ValueError("Compressed GRE image or ACS contains non-finite values.")
    calibration = torch.zeros((*native_matrix, VIRTUAL_COILS), dtype=torch.complex64)
    lin_start = native_matrix[1] // 2 - nacs // 2
    par_start = native_matrix[2] // 2 - nacs // 2
    calibration[:, lin_start : lin_start + nacs, par_start : par_start + nacs, :] = compressed_acs

    # Fit a/b/c once, then combine that identity with every echo trajectory.
    a_raw, b_raw, c_raw, evidence = native.fit_wave_psf_deviation_from_projection(
        twix_file=twix_path,
        calib_lines=calibration_lines,
        cfg=cfg,
        out_folder=destination,
        file_tag="shared",
        return_diagnostics=True,
    )
    requested_range = settings["requested_fit_kx_range"]
    a_fit, b_fit, c_fit, processing = native._process_psf_coefficients(
        a_raw,
        b_raw,
        c_raw,
        nx_os=EXTENDED_READOUT,
        processing=settings["coefficient_processing"],
        fit_kx_min=None if requested_range is None else requested_range[0],
        fit_kx_max=None if requested_range is None else requested_range[1],
        fit_quality=evidence["projection_quality"],
        out_folder=destination,
        file_tag="shared",
        return_diagnostics=True,
    )
    coefficients = _coefficient_arrays((a_fit, b_fit, c_fit))
    raw_coefficients = _coefficient_arrays((a_raw, b_raw, c_raw))
    calibration_id = _shared_calibration_id(*coefficients)
    trajectories = native._echo_theoretical_wave_trajectories(image_lines, cfg)
    psfs = _evaluate_echo_psfs(
        trajectories, coefficients, cfg, cases["native_r3x1"]
    )

    _write_shared_psf_coefficients(
        destination / "shared_psf_coefficients.npz",
        coefficients,
        raw_coefficients,
    )
    _write_gre_psf_coefficient_plots(
        destination.parent,
        coefficients,
        raw_coefficients,
        settings,
        processing,
    )
    mask_path = destination / "sampling_mask.npy"
    np.save(mask_path, source_mask, allow_pickle=False)
    calibration_output = create_cfl(destination / "kspace_calib", calibration.shape)
    calibration_output[:] = calibration.cpu().numpy()
    calibration_output.flush()
    del calibration_output

    echo_records = []
    for echo_index, (trajectory, psf) in enumerate(zip(trajectories, psfs, strict=True)):
        echo_number = echo_index + 1
        echo_label = f"echo-{echo_number:02d}"
        wave_name = f"wave_kspace_{echo_label}"
        psf_name = f"psf_{echo_label}"
        trajectory_name = f"wave_trajectory_{echo_label}.npz"
        wave_values = compressed[:, :, :, echo_index, :].cpu().numpy()
        wave_output = create_cfl(destination / wave_name, (*wave_values.shape, 1))
        wave_output[..., 0] = wave_values
        wave_output.flush()
        del wave_output
        psf_output = create_cfl(destination / psf_name, (*psf.shape, 1, 1))
        psf_output[..., 0, 0] = psf
        psf_output.flush()
        del psf_output
        np.savez(destination / trajectory_name, delta_lin=np.asarray(trajectory[0]), delta_par=np.asarray(trajectory[1]))
        echo_records.append(
            {
                "echo": echo_number,
                "eco_counter": echo_index,
                "te_s": echo_times[echo_index],
                "wave_kspace": wave_name,
                "wave_kspace_shape": list(read_shape(destination / wave_name)),
                "wave_kspace_norm": float(torch.linalg.vector_norm(compressed[:, :, :, echo_index, :]).item()),
                "psf": psf_name,
                "psf_shape": list(read_shape(destination / psf_name)),
                "sequence_trajectory": trajectory_name,
                "shared_calibration_id": calibration_id,
                "selected_wavelet_lambda": cases["native_r3x1"].shared_wavelet_lambda,
            }
        )

    manifest = {
        "format_version": 2,
        "status": "measured_multi_echo_wave_gre_bart_inputs_ready",
        "echo_count": echo_count,
        "source": {
            "twix": _file_identity(twix_path),
            "sequence": _file_identity(sequence_path, include_hash=True),
            "pinned_wave_gre_helper": "external/wave-gre-flow-comp@d3772bda7077da9af16e776fce148ba2cec8fdcf",
        },
        "geometry": cases["native_r3x1"].to_json(),
        "twix_validation": twix_metadata,
        "sampling": {**twix_metadata["sampling"], "path": str(mask_path)},
        "coil_compression": {
            "physical_coils": physical_coils,
            "virtual_coils": VIRTUAL_COILS,
            "basis_source": "one integrated set-4 ACS shared across all echoes",
            "retained_energy": float(retained_energy[VIRTUAL_COILS - 1]),
            "leading_singular_values": [float(value) for value in singular_values[:VIRTUAL_COILS]],
        },
        "native_csm_policy": "estimate once from kspace_calib and share unchanged across echoes",
        "kspace_calib": "kspace_calib",
        "kspace_calib_shape": list(read_shape(destination / "kspace_calib")),
        "psf_calibration": {
            "request": settings,
            "processing_diagnostics": processing,
            "coefficient_fit_count": 1,
            "shared_across_echoes": True,
            "shared_calibration_id": calibration_id,
            "coefficient_file": "shared_psf_coefficients.npz",
            "processed_coefficient_keys": ["a", "b", "c"],
            "raw_coefficient_keys": ["a_raw", "b_raw", "c_raw"],
            "visual_assessment_plot_relative_to_output_root": (
                f"normal/{PSF_COEFFICIENT_PLOT_NAME}"
            ),
            "full_range_plot_relative_to_output_root": (
                f"normal/{PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME}"
            ),
            "echo_specific_component": "Pulseq sequence trajectory",
        },
        "wavelet_selection": gre_wavelet_selection_provenance(
            "native_r3x1", echo_count
        ),
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "output_orientation": {
            "logical_axis_roles": list(GRE_LOGICAL_AXIS_ROLES),
            "validated_array_axis_flips": list(GRE_BART_ARRAY_AXIS_FLIPS),
            "stored_coordinate_system": "canonical RAS",
            "interpolation": False,
        },
        "echoes": echo_records,
        "prepared_at_utc": _utc_now(),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _load_shared_operator(
    inputs: Path, echoes: Sequence[Mapping[str, Any]]
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Load shared coefficients and ordered per-echo trajectories.

    Args:
        inputs: Native GRE BART-input directory.
        echoes: Ordered native echo records.

    Returns:
        Shared a/b/c tuple and one trajectory pair per echo.
    """

    with np.load(inputs / "shared_psf_coefficients.npz") as values:
        coefficients = tuple(np.asarray(values[key]) for key in ("a", "b", "c"))
    trajectories = []
    if not echoes or [item.get("echo") for item in echoes] != list(
        range(1, len(echoes) + 1)
    ):
        raise ValueError("Normal GRE manifest echoes are incomplete or out of order.")
    for echo in echoes:
        with np.load(inputs / str(echo["sequence_trajectory"])) as values:
            trajectories.append((np.asarray(values["delta_lin"]), np.asarray(values["delta_par"])))
    return coefficients, trajectories


def prepare_retro_gre(
    twix: str | Path,
    output_root: str | Path,
    sequence: str | Path,
    *,
    psf_coefficient_processing: str = "sine-line",
    psf_fit_kx_min: int | None = None,
    psf_fit_kx_max: int | None = None,
) -> list[dict[str, Any]]:
    """Prepare measured native and LIN-cropped retrospective R3x2 GRE inputs.

    Args:
        twix: Measured native R3x1 Wave-GRE TWIX file.
        output_root: Exact user-selected output root.
        sequence: Matching integrated Pulseq file.
        psf_coefficient_processing: Shared ``sine-line`` mode by default, or
            explicit ``smooth`` processing.
        psf_fit_kx_min: Optional inclusive manual sine-line bound.
        psf_fit_kx_max: Optional exclusive manual sine-line bound.

    Returns:
        Native-R3x2 and LIN-low-resolution-R3x2 manifests.
    """

    root = Path(output_root).expanduser().resolve()
    normal = prepare_normal_gre(
        twix,
        root,
        sequence,
        psf_coefficient_processing=psf_coefficient_processing,
        psf_fit_kx_min=psf_fit_kx_min,
        psf_fit_kx_max=psf_fit_kx_max,
        reuse=True,
        allow_legacy_artifact_reuse=True,
    )
    normal_inputs = root / NORMAL_INPUT_RELATIVE
    normal_echoes = normal.get("echoes")
    if not isinstance(normal_echoes, list) or not normal_echoes:
        raise ValueError("Normal GRE manifest has no ordered echoes.")
    echo_count = len(normal_echoes)
    coefficients, trajectories = _load_shared_operator(normal_inputs, normal_echoes)
    native = load_wave_gre_helpers()
    _, cfg, _, _ = validate_gre_sequence(native, Path(sequence).expanduser().resolve())
    if int(cfg["Necho"]) != echo_count:
        raise ValueError("Normal manifest and sequence echo counts disagree.")
    geometry = normal.get("geometry", {})
    native_matrix = tuple(int(value) for value in geometry["matrix_ro_lin_par"])
    native_fov_mm = tuple(float(value) for value in geometry["fov_mm_ro_lin_par"])
    sequence_matrix = (int(cfg["Nx"]), int(cfg["Ny"]), int(cfg["Nz"]))
    sequence_fov_mm = tuple(
        float(value) * 1000.0 for value in np.asarray(cfg["FOVxyz_m"])
    )
    if sequence_matrix != native_matrix or not np.allclose(
        sequence_fov_mm, native_fov_mm, rtol=0.0, atol=1e-9
    ):
        raise ValueError("Normal manifest and sequence GRE geometry disagree.")
    cases = gre_cases(native_matrix, native_fov_mm)
    echo_times = validate_gre_echo_consistency(
        cfg["TE_s"], [echo["te_s"] for echo in normal_echoes]
    )
    calibration_id = _shared_calibration_id(*coefficients)
    if calibration_id != normal["psf_calibration"]["shared_calibration_id"]:
        raise ValueError("Stored GRE shared calibration identity changed.")

    results = []
    for case_id in ("native_r3x2", "lin_low_resolution_r3x2"):
        case = cases[case_id]
        inputs = root / RETRO_RELATIVE / case_id / "bart_inputs"
        manifest_path = inputs / "manifest.json"
        if manifest_path.is_file():
            existing = _load_json(manifest_path)
            if existing.get("source_normal_manifest") != str(normal_inputs / "manifest.json"):
                raise ValueError(f"Existing retrospective GRE inputs use another source: {inputs}")
            if existing.get("case") != case.to_json():
                raise ValueError(f"Existing retrospective GRE geometry differs: {inputs}")
            if existing.get("wavelet_selection") != gre_wavelet_selection_provenance(
                case_id, echo_count
            ):
                raise ValueError(f"Existing retrospective GRE selection provenance differs: {inputs}")
            existing["output_orientation"] = {
                "logical_axis_roles": list(GRE_LOGICAL_AXIS_ROLES),
                "validated_array_axis_flips": list(GRE_BART_ARRAY_AXIS_FLIPS),
                "stored_coordinate_system": "canonical RAS",
                "interpolation": False,
            }
            _write_json(manifest_path, existing)
            results.append(existing)
            continue
        _validate_recoverable_retro_directory(inputs, echo_count)
        inputs.mkdir(parents=True, exist_ok=True)
        target_mask, sampling = _case_mask(case)
        mask_path = inputs / "sampling_mask.npy"
        np.save(mask_path, target_mask, allow_pickle=False)
        psfs = _evaluate_echo_psfs(trajectories, coefficients, cfg, case)
        echo_records = []
        for echo_index, psf in enumerate(psfs):
            echo_number = echo_index + 1
            echo_label = f"echo-{echo_number:02d}"
            source_name = str(normal["echoes"][echo_index]["wave_kspace"])
            source = np.asarray(open_cfl(normal_inputs / source_name))[..., 0]
            wave_name = f"wave_kspace_{echo_label}"
            norm = _write_echo_wave(source, inputs / wave_name, case, target_mask)
            psf_name = f"psf_{echo_label}"
            psf_output = create_cfl(inputs / psf_name, (*psf.shape, 1, 1))
            psf_output[..., 0, 0] = psf
            psf_output.flush()
            del psf_output
            echo_records.append(
                {
                    "echo": echo_number,
                    "eco_counter": echo_index,
                    "te_s": echo_times[echo_index],
                    "wave_kspace": wave_name,
                    "wave_kspace_shape": list(read_shape(inputs / wave_name)),
                    "wave_kspace_norm": norm,
                    "psf": psf_name,
                    "psf_shape": list(read_shape(inputs / psf_name)),
                    "shared_calibration_id": calibration_id,
                    "selected_wavelet_lambda": case.shared_wavelet_lambda,
                }
            )
        manifest = {
            "format_version": 1,
            "status": "direct_measured_wave_gre_crop_bart_inputs_ready",
            "source_normal_manifest": str(normal_inputs / "manifest.json"),
            "case": case.to_json(),
            "operator": (
                "direct half-open LIN/PAR crop of measured Wave k-space with "
                "the full extended readout retained, followed by pure "
                "Cartesian masking"
            ),
            "interpolation": False,
            "forward_simulation_from_no_wave_data": False,
            "sampling": {**sampling, "path": str(mask_path)},
            "psf_calibration": {
                "coefficient_fit_count": 1,
                "shared_across_echoes": True,
                "shared_calibration_id": calibration_id,
                "per_echo_sequence_trajectories_reused": True,
            },
            "csm_policy": (
                "reuse native maps unchanged"
                if case_id == "native_r3x2"
                else "centered Fourier PE resampling at unchanged FOV followed by coil-RSS normalization"
            ),
            "echo_count": echo_count,
            "wavelet_selection": gre_wavelet_selection_provenance(
                case_id, echo_count
            ),
            "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
            "output_orientation": {
                "logical_axis_roles": list(GRE_LOGICAL_AXIS_ROLES),
                "validated_array_axis_flips": list(GRE_BART_ARRAY_AXIS_FLIPS),
                "stored_coordinate_system": "canonical RAS",
                "interpolation": False,
            },
            "echoes": echo_records,
            "prepared_at_utc": _utc_now(),
        }
        _write_json(manifest_path, manifest)
        results.append(manifest)
    _write_json(
        root / RETRO_RELATIVE / "manifest.json",
        {
            "format_version": 1,
            "status": "measured_multi_echo_gre_retro_cases_ready",
            "echo_count": echo_count,
            "source_normal_manifest": str(normal_inputs / "manifest.json"),
            "cases": ["native_r3x2", "lin_low_resolution_r3x2"],
            "wavelet_selection_by_case": {
                case_id: gre_wavelet_selection_provenance(case_id, echo_count)
                for case_id in ("native_r3x2", "lin_low_resolution_r3x2")
            },
            "prepared_at_utc": _utc_now(),
        },
    )
    return results


def _validate_gre_r3x3_case_inputs(
    normal_inputs: Path,
    inputs: Path,
    normal_echoes: Sequence[Mapping[str, Any]],
    target_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Validate every reused native-R3x3 GRE echo and measured PSF.

    Args:
        normal_inputs: Existing native measured GRE BART-input directory.
        inputs: Native-R3x3 case-local BART-input directory.
        normal_echoes: Ordered normal-manifest echo records.
        target_mask: Exact pure Cartesian native-R3x3 mask.

    Returns:
        One acquired-equality, zero-exterior, and finite-value record per echo.

    Raises:
        FileNotFoundError: If a required case or source CFL pair is absent.
        ValueError: If any echo, PSF, or same-grid data contract fails.
    """

    validation: list[dict[str, Any]] = []
    for echo_index, echo in enumerate(normal_echoes, start=1):
        echo_label = f"echo-{echo_index:02d}"
        source_wave = normal_inputs / str(echo["wave_kspace"])
        target_wave = inputs / f"wave_kspace_{echo_label}"
        source_psf = normal_inputs / str(echo["psf"])
        target_psf = inputs / f"psf_{echo_label}"
        metrics = validate_same_grid_masked_wave(
            source_wave, target_wave, target_mask
        )
        source_psf_shape = read_shape(source_psf)
        target_psf_shape = read_shape(target_psf)
        if source_psf_shape != target_psf_shape:
            raise ValueError(f"Native GRE R3x3 {echo_label} PSF geometry changed.")
        if not _finite_cfl(target_psf):
            raise ValueError(f"Native GRE R3x3 {echo_label} PSF is non-finite.")
        validation.append(
            {
                "echo": echo_index,
                **metrics,
                "psf_geometry_equal_source": True,
                "psf_finite": True,
            }
        )
    return validation


def prepare_retro_gre_r3x3(
    twix: str | Path,
    output_root: str | Path,
    sequence: str | Path,
    *,
    psf_coefficient_processing: str = "sine-line",
    psf_fit_kx_min: int | None = None,
    psf_fit_kx_max: int | None = None,
) -> dict[str, Any]:
    """Prepare only native-grid R3x3 retrospective multi-echo GRE inputs.

    The LIN residue is inherited from the measured regular R3x1 image stream;
    the PAR residue is center aligned. Each echo is masked independently on
    the same native grid, while its measured-data PSF and the native CSM remain
    separate reusable artifacts.

    Args:
        twix: Measured regular-R3x1 Wave-GRE TWIX file.
        output_root: Reconstruction root containing or receiving normal inputs.
        sequence: Matching integrated Wave-GRE Pulseq sequence.
        psf_coefficient_processing: Existing normal PSF processing contract.
        psf_fit_kx_min: Optional inclusive manual sine-line bound.
        psf_fit_kx_max: Optional exclusive manual sine-line bound.

    Returns:
        Complete native-R3x3 case-local preparation manifest.

    Raises:
        ValueError: If source, echo, geometry, mask, PSF, or exact data checks fail.
        FileExistsError: If incompatible existing or partial case inputs are found.
    """

    root = Path(output_root).expanduser().resolve()
    normal = prepare_normal_gre(
        twix,
        root,
        sequence,
        psf_coefficient_processing=psf_coefficient_processing,
        psf_fit_kx_min=psf_fit_kx_min,
        psf_fit_kx_max=psf_fit_kx_max,
        reuse=True,
        allow_legacy_artifact_reuse=True,
    )
    normal_inputs = root / NORMAL_INPUT_RELATIVE
    normal_echoes = normal.get("echoes")
    if not isinstance(normal_echoes, list) or not normal_echoes:
        raise ValueError("Normal GRE manifest has no ordered echoes.")
    echo_count = len(normal_echoes)
    if [item.get("echo") for item in normal_echoes] != list(
        range(1, echo_count + 1)
    ):
        raise ValueError("Normal GRE manifest echoes are incomplete or out of order.")

    geometry = normal.get("geometry", {})
    native_matrix = tuple(int(value) for value in geometry["matrix_ro_lin_par"])
    native_fov_mm = tuple(float(value) for value in geometry["fov_mm_ro_lin_par"])
    source_mask, source_sampling = _validated_normal_sampling(normal_inputs, normal)
    residue = _native_r3x3_residue(source_sampling, native_matrix[2])
    case = replace(
        gre_cases(native_matrix, native_fov_mm)[R3X3_CASE_ID],
        residue_lin_par=residue,
    )
    target_mask, sampling = _case_mask(case)
    if np.any(target_mask & ~source_mask):
        raise ValueError("Native GRE R3x3 mask is not a subset of measured R3x1 samples.")
    selection = gre_wavelet_selection_provenance(R3X3_CASE_ID, echo_count)
    calibration_id = _normal_shared_calibration_id(normal_inputs, normal)
    reuse_attestation_path = normal_inputs.parent / NORMAL_REUSE_ATTESTATION_NAME
    reuse_attestation = None
    if reuse_attestation_path.is_file():
        attestation_payload = _load_json(reuse_attestation_path)
        normal_manifest_hash = sha256_file(normal_inputs / "manifest.json")
        if (
            attestation_payload.get("status")
            == "legacy_normal_inputs_accepted_for_retrospective_reuse"
            and attestation_payload.get("source_manifest", {}).get("sha256")
            == normal_manifest_hash
        ):
            reuse_attestation = {
                "path": str(reuse_attestation_path),
                "sha256": sha256_file(reuse_attestation_path),
            }

    inputs = root / RETRO_RELATIVE / R3X3_CASE_ID / "bart_inputs"
    manifest_path = inputs / "manifest.json"
    if manifest_path.is_file():
        existing = _load_json(manifest_path)
        if (
            existing.get("source") != normal.get("source")
            or existing.get("case") != case.to_json()
            or existing.get("sampling", {}).get("logical_sha256")
            != sampling["logical_sha256"]
            or existing.get("wavelet_selection") != selection
            or int(existing.get("echo_count", 0)) != echo_count
        ):
            raise ValueError(f"Existing native GRE R3x3 inputs are incompatible: {inputs}")
        recorded_attestation = existing.get("normal_reuse_attestation")
        if recorded_attestation is not None and (
            not isinstance(recorded_attestation, Mapping)
            or not reuse_attestation_path.is_file()
            or recorded_attestation.get("path") != str(reuse_attestation_path)
            or recorded_attestation.get("sha256")
            != sha256_file(reuse_attestation_path)
        ):
            raise ValueError(
                "Existing native GRE R3x3 inputs reference a changed reuse attestation."
            )
        mask = np.load(inputs / "sampling_mask.npy", allow_pickle=False)
        validate_pure_cartesian_image_lattice(mask, existing["sampling"])
        _validate_gre_r3x3_case_inputs(
            normal_inputs, inputs, normal_echoes, np.asarray(mask, dtype=bool)
        )
        print(f"Reusing compatible native GRE R3x3 BART inputs: {inputs}")
        return existing

    _validate_recoverable_retro_directory(inputs, echo_count)
    inputs.mkdir(parents=True, exist_ok=True)
    mask_path = inputs / "sampling_mask.npy"
    np.save(mask_path, target_mask, allow_pickle=False)
    echo_records: list[dict[str, Any]] = []
    sampling_validation: list[dict[str, Any]] = []
    psf_records: list[dict[str, Any]] = []
    for echo_index, echo in enumerate(normal_echoes, start=1):
        echo_label = f"echo-{echo_index:02d}"
        source_wave = normal_inputs / str(echo["wave_kspace"])
        wave_name = f"wave_kspace_{echo_label}"
        norm = _write_echo_wave(
            _single_map_wave_values(source_wave),
            inputs / wave_name,
            case,
            target_mask,
        )
        validation = validate_same_grid_masked_wave(
            source_wave, inputs / wave_name, target_mask
        )
        sampling_validation.append({"echo": echo_index, **validation})

        source_psf = normal_inputs / str(echo["psf"])
        psf_name = f"psf_{echo_label}"
        target_psf = inputs / psf_name
        for suffix in (".hdr", ".cfl"):
            partial = target_psf.with_suffix(suffix)
            if partial.exists() or partial.is_symlink():
                partial.unlink()
        link_bart_pair(source_psf, target_psf)
        psf_shape = read_shape(target_psf)
        if read_shape(source_psf) != psf_shape or not _finite_cfl(target_psf):
            raise ValueError(f"Native GRE R3x3 {echo_label} PSF reuse failed validation.")
        psf_records.append(
            {
                "echo": echo_index,
                "source": str(source_psf),
                "case_local": psf_name,
                "header_sha256": sha256_file(source_psf.with_suffix(".hdr")),
                "payload_sha256": sha256_file(source_psf.with_suffix(".cfl")),
                "reused_without_recalibration": True,
                "finite": True,
            }
        )
        echo_records.append(
            {
                "echo": echo_index,
                "eco_counter": int(echo.get("eco_counter", echo_index - 1)),
                "te_s": float(echo["te_s"]),
                "wave_kspace": wave_name,
                "wave_kspace_shape": list(read_shape(inputs / wave_name)),
                "wave_kspace_norm": norm,
                "psf": psf_name,
                "psf_shape": list(psf_shape),
                "shared_calibration_id": calibration_id,
                "selected_wavelet_lambda": case.shared_wavelet_lambda,
            }
        )

    manifest = {
        "format_version": 1,
        "status": "direct_measured_multi_echo_wave_gre_r3x3_bart_inputs_ready",
        "source": normal["source"],
        "source_normal_manifest": {
            "path": str(normal_inputs / "manifest.json"),
            "sha256": sha256_file(normal_inputs / "manifest.json"),
        },
        "normal_reuse_attestation": reuse_attestation,
        "case": case.to_json(),
        "operator": "same-grid retrospective Cartesian undersampling of measured multi-echo Wave k-space",
        "interpolation": False,
        "forward_simulation_from_no_wave_data": False,
        "sampling": {**sampling, "path": str(mask_path)},
        "source_subset_validation": {
            "source_sampling_logical_sha256": source_sampling["logical_sha256"],
            "target_sampling_logical_sha256": sampling["logical_sha256"],
            "target_is_exact_source_subset": True,
            "source_acquired_coordinate_count": source_sampling[
                "acquired_coordinate_count"
            ],
            "target_acquired_coordinate_count": sampling[
                "acquired_coordinate_count"
            ],
        },
        "sampling_validation_by_echo": sampling_validation,
        "echo_count": echo_count,
        "echoes": echo_records,
        "psf_by_echo": psf_records,
        "psf_calibration": {
            "reused_without_recalibration": True,
            "shared_coefficient_fit_across_echoes": True,
            "echo_specific_measured_psfs_preserved": True,
        },
        "calibration_kspace_included": False,
        "csm_policy": "reuse native measured GRE maps unchanged across echoes",
        "wavelet_selection": selection,
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "output_orientation": {
            "logical_axis_roles": list(GRE_LOGICAL_AXIS_ROLES),
            "validated_array_axis_flips": list(GRE_BART_ARRAY_AXIS_FLIPS),
            "stored_coordinate_system": "canonical RAS",
            "interpolation": False,
        },
        "prepared_at_utc": _utc_now(),
    }
    _write_json(manifest_path, manifest)
    print(
        f"Prepared {R3X3_CASE_ID}: logical={case.matrix_ro_lin_par}, "
        f"mask coordinates={sampling['acquired_coordinate_count']}, echoes={echo_count}"
    )
    return manifest


def prepare_retro_gre_sensitivity_maps(output_root: str | Path) -> None:
    """Create only the LIN-low-resolution maps from one native ecalib result.

    Args:
        output_root: User-selected GRE output root containing prepared cases.

    Returns:
        None. Native maps are not duplicated; LR maps are Fourier-resampled and
        coil-RSS normalized.
    """

    root = Path(output_root).expanduser().resolve()
    source = root / NORMAL_OUTPUT_RELATIVE / "coil_sens"
    read_shape(source)
    inputs = root / RETRO_RELATIVE / "lin_low_resolution_r3x2" / "bart_inputs"
    manifest = _load_json(inputs / "manifest.json")
    case = manifest.get("case", {})
    target_matrix = tuple(int(value) for value in case["matrix_ro_lin_par"])
    if len(target_matrix) != 3:
        raise ValueError("LIN-low-resolution GRE manifest has an invalid matrix.")
    target = inputs / "coil_sens"
    if target.with_suffix(".hdr").is_file() and target.with_suffix(".cfl").is_file():
        if read_shape(target)[:3] != target_matrix:
            raise ValueError("Existing low-resolution GRE sensitivity maps have the wrong shape.")
        return
    resample_sensitivity_maps(
        source, target, target_lin_par=target_matrix[1:]
    )
    manifest["coil_sens"] = "coil_sens"
    manifest["coil_sens_shape"] = list(read_shape(target))
    manifest["coil_sens_source"] = str(source)
    manifest["coil_sens_shared_across_echoes"] = True
    _write_json(inputs / "manifest.json", manifest)


def build_gre_wave_command(
    *,
    maps: str | Path,
    psf: str | Path,
    kspace: str | Path,
    output: str | Path,
    regularization: float,
    gpu: bool = False,
) -> list[str]:
    """Construct one explicit FISTA-r0 or Wavelet BART Wave command.

    Args:
        maps: Native or case-resampled sensitivity-map basename.
        psf: Echo-specific calibrated PSF basename.
        kspace: Echo-specific measured-Wave k-space basename.
        output: Complex reconstruction basename.
        regularization: Nonnegative Wavelet lambda; zero is the FISTA control.
        gpu: Add BART's GPU option when true.

    Returns:
        Exact command argument vector.
    """

    value = float(regularization)
    if not math.isfinite(value) or value < 0:
        raise ValueError("GRE Wavelet regularization must be finite and nonnegative.")
    options = ["-g"] if gpu else []
    return [
        "bart",
        "wave",
        *options,
        "-w",
        "-f",
        "-r",
        f"{value:.12g}",
        "-i",
        "100",
        "-t",
        "1e-6",
        str(maps),
        str(psf),
        str(kspace),
        str(output),
    ]


def bart_wave_restoration_factor(
    image_shape: Sequence[int], kspace_norm: float, encoding_shape: Sequence[int]
) -> complex:
    """Return the validated GRE BART Wave amplitude and phase correction.

    Args:
        image_shape: Logical reconstructed RO/LIN/PAR matrix.
        kspace_norm: L2 norm removed internally by BART Wave.
        encoding_shape: Extended-RO/LIN/PAR Wave grid.

    Returns:
        Complex correction ``norm*sqrt(extended_RO*LIN*PAR)*1j*(-1)**(LIN//2)``.
    """

    logical = tuple(int(value) for value in image_shape)
    encoded = tuple(int(value) for value in encoding_shape)
    if len(logical) != 3 or len(encoded) != 3 or any(value <= 0 for value in logical + encoded):
        raise ValueError("GRE BART restoration shapes must contain three positive dimensions.")
    if encoded[0] < logical[0] or encoded[1:] != logical[1:]:
        raise ValueError("GRE BART image and Wave encoding grids are incompatible.")
    if any(value % 2 for value in logical + encoded):
        raise ValueError("The validated GRE BART restoration requires even dimensions.")
    if not math.isfinite(kspace_norm) or kspace_norm <= 0:
        raise ValueError("GRE BART k-space norm must be positive and finite.")
    amplitude = kspace_norm * math.sqrt(math.prod(encoded))
    phase = 1j * ((-1) ** (logical[1] // 2))
    return complex(amplitude * phase)


def restore_bart_wave_image(
    image: np.ndarray, *, kspace_norm: float, encoding_shape: Sequence[int]
) -> np.ndarray:
    """Restore one complex BART Wave GRE image to quantitative input scale.

    Args:
        image: Three-dimensional BART complex reconstruction.
        kspace_norm: Recorded measured-Wave input norm.
        encoding_shape: Extended-RO/LIN/PAR Wave grid.

    Returns:
        Finite complex64 restored image.
    """

    values = np.asarray(image, dtype=np.complex64)
    if values.ndim != 3:
        raise ValueError("GRE BART restoration requires a three-dimensional image.")
    result = values * np.complex64(bart_wave_restoration_factor(values.shape, kspace_norm, encoding_shape))
    if not np.isfinite(result).all():
        raise ValueError("Restored GRE BART image contains non-finite values.")
    return result


def prepare_gre(
    twix: str | Path,
    output_root: str | Path,
    sequence: str | Path,
    *,
    retrospective: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Compatibility entry point for normal or retrospective measured GRE preparation.

    Args:
        twix: Measured single- or multi-echo Wave-GRE TWIX path.
        output_root: Exact user-selected output root.
        sequence: Matching integrated Pulseq path.
        retrospective: Prepare R3x2 cases in addition to reusable normal inputs.
        **kwargs: PSF coefficient-processing options forwarded to the selected preparation.

    Returns:
        Normal manifest or the two retrospective case manifests.
    """

    if retrospective:
        return prepare_retro_gre(twix, output_root, sequence, **kwargs)
    return prepare_normal_gre(twix, output_root, sequence, **kwargs)
