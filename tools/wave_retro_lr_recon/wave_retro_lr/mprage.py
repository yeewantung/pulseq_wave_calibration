"""Prepare measured Wave-MPRAGE data for explicit BART reconstruction."""

from __future__ import annotations

import copy
import importlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .bart_io import create_cfl, open_cfl, read_shape, sha256_file
from .core import CaseSpec, Geometry, resolve_case
from .psf import (
    PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME,
    PSF_COEFFICIENT_PLOT_NAME,
    PSF_COEFFICIENT_REJECTED_DIAGNOSTICS_NAME,
    PSF_COEFFICIENT_REJECTED_FULL_RANGE_PLOT_NAME,
    PSF_COEFFICIENT_REJECTED_PLOT_NAME,
    PSF_PLANE_COMPARISON_PLOT_NAME,
    PSF_PLANE_REJECTED_DIAGNOSTICS_NAME,
    PSF_PLANE_REJECTED_PLOT_NAME,
    evaluate_calibrated_psf,
    write_psf_coefficient_plot,
    write_psf_plane_comparison_plot,
)
from .projection_psf import (
    SPATIAL_SELECTION_VERSION,
    AutomaticSpatialRegionRejected,
    align_constant_phase_branch,
    centered_spatial_core_bounds,
    select_spatial_projection_region,
)
from .retrospective import resample_sensitivity_maps, write_measured_wave_crop
from .sampling import SamplingPattern, inspect_twix_sampling

NORMAL_INPUT_RELATIVE = Path("normal") / "bart_inputs"
NORMAL_OUTPUT_RELATIVE = Path("normal") / "bart_output"
RETRO_RELATIVE = Path("retro")
RETRO_CASES = (
    ("native_r3x2", None),
    ("lr_x_1p5mm_r3x2", (1.5, 1.0)),
    ("lr_y_1p5mm_r3x2", (1.0, 1.5)),
    ("lr_xy_1p25mm_r3x2", (1.25, 1.25)),
)
AUTOMATIC_C_RECOVERY_VERSION = 1
AUTOMATIC_Y_CORE_RETRY_VERSION = 1
SUSTAINED_COEFFICIENT_CORRUPTION_ERROR = (
    "Automatic PSF range selection detected sustained coefficient corruption"
)
AB_MAXIMUM_RELATIVE_FREQUENCY_DIFFERENCE = 0.02
SEQUENCE_MAXIMUM_RELATIVE_FREQUENCY_DIFFERENCE = 0.03
C_FIXED_FREQUENCY_MAXIMUM_CONDITION_NUMBER = 1.0e8
C_FIXED_FREQUENCY_MAXIMUM_RAW_RELATIVE_RMSE = 0.5
C_FIXED_FREQUENCY_MAXIMUM_TRIMMED_RELATIVE_L2 = 2.0


class AutomaticPsfFitRejected(ValueError):
    """Carry a rejected automatic fit and its measured coefficient samples."""

    def __init__(
        self,
        message: str,
        *,
        raw_coefficients: tuple[np.ndarray, np.ndarray, np.ndarray],
        candidate_coefficients: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        diagnostics: Mapping[str, Any],
        plane_diagnostics: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        """Initialize an automatic-fit rejection with diagnostic payloads.

        Args:
            message: Upstream validation error.
            raw_coefficients: Measured per-readout ``a``, ``b``, and ``c``.
            candidate_coefficients: Reconstructed rejected fit curves when
                upstream parameter diagnostics are available.
            diagnostics: Upstream automatic range and validation details.
            plane_diagnostics: Optional theoretical/measured/fitted projection
                planes retained when later coefficient processing is rejected.

        Returns:
            None.
        """
        super().__init__(message)
        self.raw_coefficients = raw_coefficients
        self.candidate_coefficients = candidate_coefficients
        self.diagnostics = dict(diagnostics)
        self.plane_diagnostics = {
            str(name): dict(record)
            for name, record in (plane_diagnostics or {}).items()
        }


def _normalize_psf_coefficient_settings(
    processing: str,
    fit_kx_min: int | None,
    fit_kx_max: int | None,
) -> dict[str, Any]:
    """Validate and normalize PSF coefficient-processing settings.

    Args:
        processing: Upstream processing mode, either ``smooth`` or
            ``sine-line``.
        fit_kx_min: Inclusive first readout index used by sine-line fitting.
        fit_kx_max: Exclusive final readout index used by sine-line fitting.

    Returns:
        A JSON-compatible request mapping that distinguishes automatic and
        manual half-open fit-range selection.

    Raises:
        ValueError: If the mode is unsupported or its kx bounds are invalid.
    """
    mode = str(processing).strip().lower()
    if mode not in {"smooth", "sine-line"}:
        raise ValueError("PSF coefficient processing must be 'smooth' or 'sine-line'.")
    if mode == "smooth":
        if fit_kx_min is not None or fit_kx_max is not None:
            raise ValueError("PSF fit kx bounds are valid only with sine-line processing.")
        return {
            "coefficient_processing": "smooth",
            "fit_range_selection": None,
            "requested_fit_kx_range": None,
            "fit_kx_range_convention": "half-open",
        }
    if (fit_kx_min is None) != (fit_kx_max is None):
        raise ValueError(
            "Sine-line PSF processing requires both manual fit kx bounds or neither."
        )
    if fit_kx_min is None:
        return {
            "coefficient_processing": "sine-line",
            "fit_range_selection": "automatic",
            "requested_fit_kx_range": None,
            "fit_kx_range_convention": "half-open",
        }
    lower = int(fit_kx_min)
    upper = int(fit_kx_max)
    if lower < 0 or upper <= lower:
        raise ValueError("PSF fit kx bounds must satisfy 0 <= min < max.")
    return {
        "coefficient_processing": "sine-line",
        "fit_range_selection": "manual",
        "requested_fit_kx_range": [lower, upper],
        "fit_kx_range_convention": "half-open",
    }


def _normalize_psf_spatial_settings(
    fit_y_min: int | None,
    fit_y_max: int | None,
    fit_z_min: int | None,
    fit_z_max: int | None,
) -> dict[str, Any]:
    """Normalize global projection-space fit-region settings.

    Args:
        fit_y_min: Optional inclusive manual sin-projection index.
        fit_y_max: Optional exclusive manual sin-projection index.
        fit_z_min: Optional inclusive manual cos-projection index.
        fit_z_max: Optional exclusive manual cos-projection index.

    Returns:
        JSON-compatible automatic/manual region-selection request.

    Raises:
        ValueError: If either manual interval is incomplete or empty.
    """
    bounds = {}
    for axis, lower, upper in (
        ("y", fit_y_min, fit_y_max),
        ("z", fit_z_min, fit_z_max),
    ):
        if (lower is None) != (upper is None):
            raise ValueError(
                f"Manual PSF {axis} spatial fitting requires both min and max bounds."
            )
        if lower is not None:
            normalized = (int(lower), int(upper))
            if normalized[0] < 0 or normalized[1] <= normalized[0]:
                raise ValueError(
                    f"PSF {axis} spatial bounds must satisfy 0 <= min < max."
                )
            bounds[axis] = list(normalized)
        else:
            bounds[axis] = None
    return {
        "spatial_region_selection": "global-automatic-with-manual-override",
        "spatial_region_selection_version": SPATIAL_SELECTION_VERSION,
        "requested_fit_y_range": bounds["y"],
        "requested_fit_z_range": bounds["z"],
        "fit_spatial_range_convention": "half-open calibration-image indices",
    }


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        A timezone-aware ISO-8601 timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a mapping as formatted JSON.

    Args:
        path: Destination JSON file.
        payload: Mapping to serialize.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path: JSON file to read.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: If the top-level JSON value is not an object.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _file_identity(path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    """Describe a source file for safe input reuse checks.

    Args:
        path: Source file to identify.
        include_hash: Whether to include the SHA-256 digest.

    Returns:
        A mapping containing the resolved path, size, modification time, and
        optional digest.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    identity: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        identity["sha256"] = sha256_file(resolved)
    return identity


def _repo_root() -> Path:
    """Return the root of the enclosing source repository.

    Returns:
        The absolute repository path inferred from this module location.
    """
    return Path(__file__).resolve().parents[3]


def _psf_processing_implementation_identity() -> dict[str, Any]:
    """Identify pinned upstream files that determine MPRAGE PSF calibration.

    Returns:
        Relative upstream paths and SHA-256 digests used to reject prepared
        inputs created by a different calibration implementation.

    Raises:
        FileNotFoundError: If an expected pinned upstream source file is absent.
    """

    repository = _repo_root()
    relative_paths = (
        Path("external/wave-mprage/recon/recon_wave_mprage_from_twix_integrated_nifti.py"),
        Path("external/wave-mprage/recon/utils/psf_coefficient_processing.py"),
        Path("external/wave-mprage/recon/utils/psf_wrapped_phase_fit.py"),
        Path("tools/wave_retro_lr_recon/wave_retro_lr/mprage.py"),
        Path("tools/wave_retro_lr_recon/wave_retro_lr/projection_psf.py"),
        Path("tools/wave_retro_lr_recon/wave_retro_lr/psf.py"),
    )
    files = {}
    for relative_path in relative_paths:
        source = repository / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Pinned PSF calibration source not found: {source}")
        files[str(relative_path)] = sha256_file(source)
    return {"files_sha256": files}


def load_wave_mprage_helpers() -> Any:
    """Import focused helpers from the pinned Wave-MPRAGE implementation.

    TWIX I/O and coil compression come directly from upstream ``utils``.
    Calibration, sagittal geometry, and the accepted NIfTI wrapper currently
    live only in the upstream reconstruction module, which is imported as a
    library; its ``main()`` and Python CG-SENSE reconstruction are never called.

    Returns:
        A namespace containing the focused upstream helper callables.

    Raises:
        FileNotFoundError: If the pinned Wave-MPRAGE implementation is absent.
    """

    recon_root = _repo_root() / "external" / "wave-mprage" / "recon"
    script = recon_root / "recon_wave_mprage_from_twix_integrated_nifti.py"
    if not script.is_file():
        raise FileNotFoundError(f"Pinned Wave-MPRAGE implementation not found: {script}")
    if str(recon_root) not in sys.path:
        sys.path.insert(0, str(recon_root))
    twix_import = importlib.import_module("utils.twix_import")
    coil_compression = importlib.import_module("utils.coil_compression_kspace")
    native = importlib.import_module("recon_wave_mprage_from_twix_integrated_nifti")
    return SimpleNamespace(
        supports_spatial_projection_selection=True,
        load_img=twix_import.load_img,
        load_ref=twix_import.load_ref,
        estimate_cc_matrix_coillast=coil_compression.estimate_cc_matrix_coillast,
        apply_cc_coillast_torch=coil_compression.apply_cc_coillast_torch,
        fit_wave_psf_deviation_from_projection=(
            native.fit_wave_psf_deviation_from_projection
        ),
        fit_wrapped_phase_planes=native.fit_wrapped_phase_planes,
        _projection_fit_quality_summary=native._projection_fit_quality_summary,
        _get_fov_yz=native._get_fov_yz,
        ifft3call=native.ifft3call,
        fftc_dim=native.fftc_dim,
        generate_theoretical_wave_trajectory=(
            native.generate_theoretical_wave_trajectory
        ),
        smooth_1d_nan=native.smooth_1d_nan,
        AUTO_FIT_PREFILTER_WINDOW=native.AUTO_FIT_PREFILTER_WINDOW,
        _process_psf_coefficients=native._process_psf_coefficients,
        _resolve_mprage_wave_mode=native._resolve_mprage_wave_mode,
        _check_integrated_refscan_shape=native._check_integrated_refscan_shape,
        _derive_hardcoded_sag_logical_geometry=(
            native._derive_hardcoded_sag_logical_geometry
        ),
        _assert_sag_geometry=native._assert_sag_geometry,
        save_mprage_output_to_nifti=native.save_mprage_output_to_nifti,
        _sanitize_filename_component=native._sanitize_filename_component,
    )


def _read_sequence(sequence_path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Read a sagittal MPRAGE sequence and derive its logical geometry.

    Args:
        sequence_path: Pulseq sequence file to read.

    Returns:
        A pair containing sequence definitions and upstream geometry metadata.
    """
    import pypulseq as pp

    native = load_wave_mprage_helpers()
    sequence = pp.Sequence()
    sequence.read(str(sequence_path), remove_duplicates=False)
    definitions = sequence.definitions
    native._assert_sag_geometry(definitions)
    geometry = native._derive_hardcoded_sag_logical_geometry(definitions)
    return definitions, geometry


def _embed_image_stream(
    loaded: Any,
    sampling: SamplingPattern,
    *,
    readout_oversampled: int,
    physical_coils: int,
) -> Any:
    """Place mapVBVD's compact PE payload on the declared logical grid.

    Args:
        loaded: Compact or full-grid TWIX image payload.
        sampling: Validated measured sampling pattern.
        readout_oversampled: Expected oversampled readout length.
        physical_coils: Expected receive-coil count.

    Returns:
        A complex Torch tensor on the full logical PE grid. When ``loaded``
        already has the full logical shape and complex64 dtype, its storage is
        reused and samples outside the measured lattice are zeroed in place.

    Raises:
        ValueError: If payload dimensions, coils, support, or samples disagree
            with the sequence and MDH sampling information.
    """

    import torch

    image = loaded if torch.is_tensor(loaded) else torch.as_tensor(loaded)
    if image.ndim != 4 or image.shape[0] != readout_oversampled:
        raise ValueError(
            "TWIX image payload must have shape [RO_os, LIN, PAR, coil] with "
            f"RO_os={readout_oversampled}; got {tuple(image.shape)}."
        )
    if image.shape[-1] != physical_coils:
        raise ValueError("TWIX image/refscan physical coil counts disagree.")
    nlin, npar = sampling.matrix_lin_par
    if tuple(image.shape[1:3]) == (nlin, npar):
        full = image.to(torch.complex64)
    else:
        skip_lin, skip_par = sampling.skip_lin_par
        stop_lin = skip_lin + int(image.shape[1])
        stop_par = skip_par + int(image.shape[2])
        if not (0 <= skip_lin < stop_lin <= nlin and 0 <= skip_par < stop_par <= npar):
            raise ValueError(
                "mapVBVD compact image support cannot be embedded in the sequence grid: "
                f"skip={sampling.skip_lin_par}, payload={tuple(image.shape[1:3])}, "
                f"grid={(nlin, npar)}."
            )
        full = torch.zeros(
            (readout_oversampled, nlin, npar, physical_coils), dtype=torch.complex64
        )
        full[:, skip_lin:stop_lin, skip_par:stop_par, :] = image.to(torch.complex64)
    mask = torch.from_numpy(sampling.mask()).view(1, nlin, npar, 1)

    # Check the unmeasured lattice in bounded readout blocks. Materializing the
    # complete boolean selection can otherwise consume many GiB for 52 coils.
    for start in range(0, readout_oversampled, 8):
        outside = full[start : start + 8].masked_select(~mask)
        if outside.numel() and torch.count_nonzero(outside).item():
            raise ValueError(
                "TWIX payload contains nonzero samples outside its MDH sampling mask."
            )

    # The full-grid tensor is owned by this preparation stage and is no longer
    # needed unmasked, so avoid allocating a second physical-coil volume.
    full.mul_(mask)
    return full


def _curve_from_fit_parameters(
    parameters: Mapping[str, Any], readout_oversampled: int
) -> np.ndarray | None:
    """Reconstruct one sine-line curve from recorded fit parameters.

    Args:
        parameters: Mapping containing ``A``, ``w``, ``phi``, ``C1``, and
            ``C2`` sine-line parameters.
        readout_oversampled: Number of samples on the full readout grid.

    Returns:
        A finite full-readout curve, or ``None`` for incomplete parameters.
    """
    kx = np.arange(readout_oversampled, dtype=np.float64)
    try:
        amplitude = float(parameters["A"])
        angular_frequency = float(parameters["w"])
        phase = float(parameters["phi"])
        slope = float(parameters["C1"])
        intercept = float(parameters["C2"])
    except (KeyError, TypeError, ValueError):
        return None
    curve = (
        amplitude * np.sin(angular_frequency * kx + phase)
        + slope * kx
        + intercept
    )
    return curve if np.isfinite(curve).all() else None


def _candidate_curves_from_fit_diagnostics(
    diagnostics: Mapping[str, Any], readout_oversampled: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Reconstruct rejected sine-line curves from upstream fit parameters.

    Args:
        diagnostics: Upstream sine-line diagnostics containing one parameter
            mapping for each of ``a``, ``b``, and ``c``.
        readout_oversampled: Number of samples on the full readout grid.

    Returns:
        Three finite rejected candidate curves, or ``None`` when complete
        parameter diagnostics were not produced before failure.
    """
    coefficient_diagnostics = diagnostics.get("coefficients")
    if not isinstance(coefficient_diagnostics, Mapping):
        return None
    curves: list[np.ndarray] = []
    for name in ("a", "b", "c"):
        parameters = coefficient_diagnostics.get(name)
        if not isinstance(parameters, Mapping):
            return None
        curve = _curve_from_fit_parameters(parameters, readout_oversampled)
        if curve is None:
            return None
        curves.append(curve)
    return tuple(curves)  # type: ignore[return-value]


def _sequence_wave_angular_frequency(
    delta_lin: np.ndarray, delta_par: np.ndarray
) -> float:
    """Estimate the common Wave angular frequency from sequence trajectories.

    Args:
        delta_lin: Sequence-derived LIN displacement over readout.
        delta_par: Sequence-derived PAR displacement over readout.

    Returns:
        Dominant nonzero angular frequency in radians per readout sample.

    Raises:
        ValueError: If no finite oscillatory sequence component is present.
    """
    vectors = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (delta_lin, delta_par)
    )
    if not vectors[0].size or vectors[0].size != vectors[1].size:
        raise ValueError("Sequence Wave trajectories must have one common length.")
    sample_index = np.arange(vectors[0].size, dtype=np.float64)
    combined_power = np.zeros(vectors[0].size // 2 + 1, dtype=np.float64)
    usable_components = 0
    for vector in vectors:
        if not np.isfinite(vector).all():
            raise ValueError("Sequence Wave trajectories must be finite.")
        slope, intercept = np.polyfit(sample_index, vector, 1)
        detrended = vector - (slope * sample_index + intercept)
        power = np.abs(np.fft.rfft(detrended)) ** 2
        power[0] = 0.0
        maximum = float(np.max(power))
        if maximum > np.finfo(np.float64).eps:
            combined_power += power / maximum
            usable_components += 1
    if not usable_components or not np.any(combined_power[1:] > 0.0):
        raise ValueError("Sequence Wave trajectories contain no oscillatory component.")
    frequency_bin = int(np.argmax(combined_power[1:]) + 1)
    return float(2.0 * np.pi * frequency_bin / vectors[0].size)


def _fit_fixed_frequency_sine_line(
    values: np.ndarray,
    fit_indices: np.ndarray,
    *,
    angular_frequency: float,
    readout_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit sine/cosine amplitudes plus a line at one fixed frequency.

    Args:
        values: One-dimensional coefficient samples used by the fit.
        fit_indices: Readout indices selected as reliable observations.
        angular_frequency: Fixed angular frequency in radians per sample.
        readout_size: Full oversampled readout length.

    Returns:
        The full-readout fitted curve and JSON-compatible fit diagnostics.

    Raises:
        ValueError: If inputs are insufficient, non-finite, or rank deficient.
    """
    observations = np.asarray(values, dtype=np.float64).reshape(-1)
    indices = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
    if observations.size != readout_size:
        raise ValueError("Fixed-frequency c samples must match the readout size.")
    indices = indices[(indices >= 0) & (indices < observations.size)]
    indices = indices[np.isfinite(observations[indices])]
    if indices.size < 6:
        raise ValueError(
            "Fixed-frequency c fitting requires at least 6 finite samples."
        )
    t = indices.astype(np.float64)
    center = float(np.mean(t))
    scale = float(np.ptp(t) / 2.0)
    if (
        not np.isfinite(angular_frequency)
        or angular_frequency <= 0.0
        or scale <= 0.0
    ):
        raise ValueError(
            "Fixed-frequency c fitting requires a positive frequency and span."
        )
    centered = (t - center) / scale
    design = np.column_stack(
        (
            np.sin(angular_frequency * t),
            np.cos(angular_frequency * t),
            centered,
            np.ones_like(t),
        )
    )
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, observations[indices], rcond=None
    )
    if rank != 4 or not np.isfinite(coefficients).all():
        raise ValueError("Fixed-frequency c fit is rank deficient or non-finite.")
    condition_number = float(singular_values[0] / singular_values[-1])
    sine_weight, cosine_weight, scaled_slope, centered_intercept = coefficients
    slope = float(scaled_slope / scale)
    intercept = float(centered_intercept - slope * center)
    amplitude = float(np.hypot(sine_weight, cosine_weight))
    phase = float(np.arctan2(cosine_weight, sine_weight))
    full_index = np.arange(readout_size, dtype=np.float64)
    curve = (
        amplitude * np.sin(angular_frequency * full_index + phase)
        + slope * full_index
        + intercept
    )
    prediction = curve[indices]
    residual_rmse = float(
        np.sqrt(np.mean((prediction - observations[indices]) ** 2))
    )
    return curve, {
        "model": "A*sin(w*kx+phi)+C1*kx+C2",
        "frequency_constraint": "fixed",
        "A": amplitude,
        "w": float(angular_frequency),
        "phi": phase,
        "C1": slope,
        "C2": intercept,
        "fit_sample_count": int(indices.size),
        "design_rank": int(rank),
        "design_condition_number": condition_number,
        "fit_input_residual_rmse": residual_rmse,
    }


def _recover_c_only_automatic_rejection(
    native: Any,
    *,
    raw_coefficients: tuple[np.ndarray, np.ndarray, np.ndarray],
    rejected_diagnostics: Mapping[str, Any],
    delta_lin: np.ndarray,
    delta_par: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    dict[str, Any],
] | None:
    """Recover a c-only automatic rejection with constrained fit or smoothing.

    Args:
        native: Namespace exposing the pinned nine-point smoothing helper.
        raw_coefficients: Measured per-readout ``a``, ``b``, and ``c`` samples.
        rejected_diagnostics: Upstream independent sine-line fit diagnostics.
        delta_lin: Sequence-derived LIN Wave trajectory.
        delta_par: Sequence-derived PAR Wave trajectory.

    Returns:
        Accepted coefficient vectors and explicit hybrid diagnostics, or
        ``None`` when strict ``a/b`` and sequence eligibility checks fail.
    """
    coefficient_diagnostics = rejected_diagnostics.get("coefficients")
    if not isinstance(coefficient_diagnostics, Mapping):
        return None
    failed_names = [
        name
        for name in ("a", "b", "c")
        if not bool(
            isinstance(coefficient_diagnostics.get(name), Mapping)
            and coefficient_diagnostics[name].get("validation_passed")
        )
    ]
    if failed_names != ["c"]:
        return None
    original_curves = _candidate_curves_from_fit_diagnostics(
        rejected_diagnostics, raw_coefficients[0].size
    )
    if original_curves is None:
        return None
    try:
        frequency_a = float(coefficient_diagnostics["a"]["w"])
        frequency_b = float(coefficient_diagnostics["b"]["w"])
    except (KeyError, TypeError, ValueError):
        return None
    common_frequency = 0.5 * (frequency_a + frequency_b)
    if frequency_a <= 0.0 or frequency_b <= 0.0 or common_frequency <= 0.0:
        return None
    ab_relative_difference = abs(frequency_a - frequency_b) / common_frequency
    sequence_frequency = _sequence_wave_angular_frequency(delta_lin, delta_par)
    sequence_relative_difference = (
        abs(common_frequency - sequence_frequency) / sequence_frequency
    )
    if (
        ab_relative_difference > AB_MAXIMUM_RELATIVE_FREQUENCY_DIFFERENCE
        or sequence_relative_difference
        > SEQUENCE_MAXIMUM_RELATIVE_FREQUENCY_DIFFERENCE
    ):
        return None

    fit_range = rejected_diagnostics.get("kx_range")
    range_diagnostics = rejected_diagnostics.get("range_selection_diagnostics")
    if (
        not isinstance(fit_range, (list, tuple))
        or len(fit_range) != 2
        or not isinstance(range_diagnostics, Mapping)
    ):
        return None
    lower, upper = (int(value) for value in fit_range)
    readout_size = raw_coefficients[2].size
    if not 0 <= lower < upper <= readout_size:
        return None
    fit_mask = np.zeros(readout_size, dtype=bool)
    fit_mask[lower:upper] = True
    excluded = np.asarray(
        range_diagnostics.get("excluded_sample_indices_within_interval", []),
        dtype=np.int64,
    )
    if excluded.size and (np.any(excluded < lower) or np.any(excluded >= upper)):
        return None
    fit_mask[excluded] = False
    fit_indices = np.flatnonzero(fit_mask)

    import torch

    raw_c_tensor = torch.as_tensor(raw_coefficients[2], dtype=torch.float64)
    masked_c = raw_c_tensor.clone()
    masked_c[~torch.from_numpy(fit_mask)] = torch.nan
    fit_input = (
        native.smooth_1d_nan(
            masked_c,
            window=int(native.AUTO_FIT_PREFILTER_WINDOW),
        )
        .cpu()
        .numpy()
    )
    try:
        constrained_curve, constrained = _fit_fixed_frequency_sine_line(
            fit_input,
            fit_indices,
            angular_frequency=common_frequency,
            readout_size=readout_size,
        )
        raw_fit_values = raw_coefficients[2][fit_indices]
        raw_residual_rmse = float(
            np.sqrt(
                np.mean((constrained_curve[fit_indices] - raw_fit_values) ** 2)
            )
        )
        raw_range = max(float(np.ptp(raw_fit_values)), np.finfo(np.float64).eps)
        raw_relative_rmse = raw_residual_rmse / raw_range

        trim = max(2, int(np.ceil(0.05 * (upper - lower))))
        trimmed_indices = fit_indices[
            (fit_indices >= lower + trim) & (fit_indices < upper - trim)
        ]
        trimmed_success = False
        trimmed_relative_l2 = None
        try:
            trimmed_curve, _ = _fit_fixed_frequency_sine_line(
                fit_input,
                trimmed_indices,
                angular_frequency=common_frequency,
                readout_size=readout_size,
            )
            trimmed_relative_l2 = float(
                np.linalg.norm(constrained_curve - trimmed_curve)
                / max(np.linalg.norm(constrained_curve), np.finfo(np.float64).eps)
            )
            trimmed_success = True
        except ValueError:
            pass
        gates = {
            "design_condition_number_at_most_1e8": bool(
                constrained["design_condition_number"]
                <= C_FIXED_FREQUENCY_MAXIMUM_CONDITION_NUMBER
            ),
            "raw_residual_rmse_relative_to_range_at_most_0p5": bool(
                raw_relative_rmse <= C_FIXED_FREQUENCY_MAXIMUM_RAW_RELATIVE_RMSE
            ),
            "endpoint_trim_refit_succeeded": trimmed_success,
            "endpoint_trim_full_readout_relative_l2_at_most_2": bool(
                trimmed_relative_l2 is not None
                and trimmed_relative_l2
                <= C_FIXED_FREQUENCY_MAXIMUM_TRIMMED_RELATIVE_L2
            ),
        }
        constrained.update(
            {
                "raw_observation_residual_rmse": raw_residual_rmse,
                "raw_observation_residual_rmse_relative_to_range": raw_relative_rmse,
                "endpoint_trim_samples_per_side": trim,
                "endpoint_trim_full_readout_relative_l2_difference": (
                    trimmed_relative_l2
                ),
                "validation_gates": gates,
                "validation_passed": all(gates.values()),
            }
        )
    except ValueError as exc:
        constrained_curve = None
        constrained = {
            "model": "A*sin(w*kx+phi)+C1*kx+C2",
            "frequency_constraint": "fixed",
            "w": common_frequency,
            "validation_passed": False,
            "error": str(exc),
        }

    accepted_c = constrained_curve
    outcome = "constrained_common_frequency_c"
    effective_processing = "sine_line_ab_constrained_common_frequency_c"
    if constrained_curve is None or not constrained["validation_passed"]:
        accepted_c = (
            native.smooth_1d_nan(raw_c_tensor, window=9).cpu().numpy()
        )
        if not np.isfinite(accepted_c).all():
            return None
        outcome = "smooth_c_fallback"
        effective_processing = "sine_line_ab_smooth_c"

    accepted_diagnostics = copy.deepcopy(dict(rejected_diagnostics))
    original_c_diagnostics = copy.deepcopy(coefficient_diagnostics["c"])
    accepted_diagnostics["original_independent_fit_validation_passed"] = False
    accepted_diagnostics["validation_passed"] = True
    accepted_diagnostics["validation_policy"] = "local-c-only-recovery-v1"
    accepted_diagnostics["effective_coefficient_processing"] = effective_processing
    accepted_diagnostics["automatic_c_recovery"] = {
        "version": AUTOMATIC_C_RECOVERY_VERSION,
        "trigger": "only independently fitted c failed upstream validation",
        "a_b_strict_validation_passed": True,
        "a_b_angular_frequencies_rad_per_sample": [frequency_a, frequency_b],
        "a_b_relative_frequency_difference": ab_relative_difference,
        "sequence_angular_frequency_rad_per_sample": sequence_frequency,
        "sequence_relative_frequency_difference": sequence_relative_difference,
        "common_angular_frequency_rad_per_sample": common_frequency,
        "policy_thresholds": {
            "maximum_a_b_relative_frequency_difference": (
                AB_MAXIMUM_RELATIVE_FREQUENCY_DIFFERENCE
            ),
            "maximum_sequence_relative_frequency_difference": (
                SEQUENCE_MAXIMUM_RELATIVE_FREQUENCY_DIFFERENCE
            ),
            "maximum_c_design_condition_number": (
                C_FIXED_FREQUENCY_MAXIMUM_CONDITION_NUMBER
            ),
            "maximum_c_raw_relative_rmse": (
                C_FIXED_FREQUENCY_MAXIMUM_RAW_RELATIVE_RMSE
            ),
            "maximum_c_trimmed_relative_l2": (
                C_FIXED_FREQUENCY_MAXIMUM_TRIMMED_RELATIVE_L2
            ),
        },
        "original_rejected_c_fit": original_c_diagnostics,
        "constrained_c_fit": constrained,
        "outcome": outcome,
        "smooth_c_window_samples": 9 if outcome == "smooth_c_fallback" else None,
        "fallback_was_silent": False,
    }
    accepted_coefficients = accepted_diagnostics["coefficients"]
    accepted_coefficients["c"] = (
        constrained
        if outcome == "constrained_common_frequency_c"
        else {
            "coefficient_processing": "smooth",
            "window_samples": 9,
            "validation_passed": True,
            "validation_gates": {"finite_after_nine_point_smoothing": True},
            "fallback_reason": "relaxed constrained-frequency c gates failed",
        }
    )
    return (
        (original_curves[0], original_curves[1], accepted_c),
        accepted_diagnostics,
    )


def _fit_projection_coefficients_with_spatial_selection(
    native: Any,
    *,
    twix_path: Path,
    sequence_path: Path,
    ncalib: int,
    nacs: int,
    fit_y_bounds: tuple[int, int] | None,
    fit_z_bounds: tuple[int, int] | None,
) -> tuple[
    Any,
    Any,
    Any,
    int,
    dict[str, Any],
    dict[str, dict[str, object]],
    dict[str, Any] | None,
]:
    """Fit integrated projection planes with global spatial-region selection.

    The parent adapter retains the pinned upstream TWIX, trajectory, Fourier,
    and wrapped-plane implementations. It adds one fixed spatial interval per
    projection because the upstream high-level helper does not expose that
    selection boundary.

    Args:
        native: Focused pinned Wave-MPRAGE helper namespace.
        twix_path: Integrated measured TWIX file.
        sequence_path: Matching Pulseq sequence.
        ncalib: Projection calibration matrix width.
        nacs: Integrated ACS matrix width.
        fit_y_bounds: Optional manual sin-projection interval.
        fit_z_bounds: Optional manual cos-projection interval.

    Returns:
        Raw ``a``, ``b``, and total ``c`` coefficients, calibration ADC count,
        structured quality evidence, plot-ready sin/cos PSF planes, and an
        optional cached central-y retry payload for automatic kx rejection.

    Raises:
        AutomaticSpatialRegionRejected: If no reliable automatic spatial
            interval exists for either projection.
        ValueError: If refscan or trajectory dimensions are incompatible.
    """
    import pypulseq as pp
    import torch

    reference = native.load_ref(str(twix_path))
    native._check_integrated_refscan_shape(reference, Nacs=nacs, Ncalib=ncalib)
    readout_oversampled = int(reference.shape[0])
    sequence = pp.Sequence()
    sequence.read(str(sequence_path), remove_duplicates=False)
    definitions = sequence.definitions
    trajectory_adc, _, _, _, _ = sequence.calculate_kspace()
    calibration_samples = int(readout_oversampled * (ncalib * 4 + nacs * nacs))
    if trajectory_adc.shape[1] < calibration_samples:
        raise ValueError("Sequence trajectory is shorter than its integrated calibration tail.")
    trajectory = np.asarray(
        trajectory_adc[:, -calibration_samples:], dtype=np.float64
    ).reshape(3, -1, readout_oversampled)
    fov_y, fov_z = native._get_fov_yz(definitions, slice_orientation="SAG")

    coefficient_results = []
    projection_quality = {}
    selection_records = {}
    plane_records: dict[str, dict[str, object]] = {}
    y_core_retry_component: dict[str, Any] | None = None
    for wave_mode in ("sin", "cos"):
        if wave_mode == "sin":
            nowave_kspace = reference[:, :ncalib, :1, 0, :]
            wave_kspace = reference[:, :ncalib, :1, 1, :]
            delta = trajectory[1, ncalib : 2 * ncalib][ncalib // 2] * fov_y
            y_coordinates = (np.arange(ncalib) - ncalib / 2.0) / ncalib
            z_coordinates = np.array([0.0])
            theoretical = np.exp(
                1j * 2.0 * np.pi * delta[:, None] * y_coordinates[None, :]
            ).astype(np.complex64)[..., None]
            axis_name = "y"
            manual_bounds = fit_y_bounds
        else:
            nowave_kspace = reference[:, :1, :ncalib, 2, :]
            wave_kspace = reference[:, :1, :ncalib, 3, :]
            delta = trajectory[0, 3 * ncalib : 4 * ncalib][ncalib // 2] * fov_z
            y_coordinates = np.array([0.0])
            z_coordinates = (np.arange(ncalib) - ncalib / 2.0) / ncalib
            theoretical = np.exp(
                1j * 2.0 * np.pi * delta[:, None] * z_coordinates[None, :]
            ).astype(np.complex64)[:, None, :]
            axis_name = "z"
            manual_bounds = fit_z_bounds

        image_nowave = native.ifft3call(nowave_kspace)
        image_wave = native.ifft3call(wave_kspace)
        hybrid_nowave = native.fftc_dim(image_nowave, dim=0)
        hybrid_wave = native.fftc_dim(image_wave, dim=0)
        cross = hybrid_wave * torch.conj(hybrid_nowave) / (
            1e-8 + hybrid_nowave * torch.conj(hybrid_nowave)
        )
        measured = torch.exp(1j * torch.angle(cross.mean(dim=-1)))
        theoretical_tensor = torch.as_tensor(theoretical, device=measured.device)
        psf_difference = torch.angle(torch.conj(theoretical_tensor) * measured)
        fit_cache: dict[tuple[int, int], Mapping[str, Any]] = {}

        def fit_region(bounds: tuple[int, int]) -> Mapping[str, Any]:
            """Run the pinned fitter on one original-coordinate interval.

            Args:
                bounds: Half-open spatial calibration interval.

            Returns:
                Upstream wrapped-plane fit result.
            """
            if bounds in fit_cache:
                return fit_cache[bounds]
            lower, upper = bounds
            if axis_name == "y":
                difference_subset = psf_difference[:, lower:upper, :]
                hybrid_subset = hybrid_nowave[:, lower:upper, :, :]
                y_subset = y_coordinates[lower:upper]
                z_subset = z_coordinates
            else:
                difference_subset = psf_difference[:, :, lower:upper]
                hybrid_subset = hybrid_nowave[:, :, lower:upper, :]
                y_subset = y_coordinates
                z_subset = z_coordinates[lower:upper]
            fitted_region = native.fit_wrapped_phase_planes(
                psf_diff=difference_subset,
                hyb_nowave=hybrid_subset,
                y_norm=y_subset,
                z_norm=z_subset,
                mask_mode="combined",
                mag_abs_floor=0.0,
                local_window_size=5,
                coherence_threshold=0.75,
                use_phase_coherence_weight=True,
                phase_weight_power=2.0,
                use_residual_coherence_refinement=True,
                residual_window_size=5,
                residual_coherence_threshold=0.75,
                use_residual_coherence_weight=True,
                residual_weight_power=2.0,
                n_irls=10,
                huber_delta=0.7,
                return_quality_maps=True,
                verbose=False,
            )
            fit_cache[bounds] = fitted_region
            return fitted_region

        spatial_coordinates = y_coordinates if axis_name == "y" else z_coordinates
        try:
            result, selection = select_spatial_projection_region(
                psf_difference,
                hybrid_nowave,
                spatial_coordinates,
                axis_name=axis_name,
                fit_region=fit_region,
                manual_bounds=manual_bounds,
            )
        except AutomaticSpatialRegionRejected as exc:
            full_result = exc.plane.get("full_result")
            if isinstance(full_result, Mapping):
                slope_key = "a_fit_all" if axis_name == "y" else "b_fit_all"
                slope = np.asarray(full_result[slope_key], dtype=np.float64)
                constant = np.asarray(full_result["c_fit_all"], dtype=np.float64)
                prediction = slope[:, None] * spatial_coordinates[None, :] + constant[:, None]
                fitted = np.squeeze(theoretical) * np.exp(1j * prediction)
            else:
                fitted = np.squeeze(theoretical)
            exc.plane.update(
                {
                    "theoretical": np.squeeze(theoretical),
                    "measured": np.squeeze(measured.detach().cpu().numpy()),
                    "fitted": fitted,
                    "residual": np.conj(fitted)
                    * np.squeeze(measured.detach().cpu().numpy()),
                }
            )
            raise

        coefficient_results.append(result)
        projection_quality[wave_mode] = native._projection_fit_quality_summary(result)
        selection_records[wave_mode] = selection
        slope_key = "a_fit_all" if axis_name == "y" else "b_fit_all"
        slope = np.asarray(result[slope_key].detach().cpu(), dtype=np.float64)
        constant = np.asarray(result["c_fit_all"].detach().cpu(), dtype=np.float64)
        prediction = slope[:, None] * spatial_coordinates[None, :] + constant[:, None]
        theoretical_2d = np.squeeze(theoretical)
        measured_2d = np.squeeze(measured.detach().cpu().numpy())
        fitted_2d = theoretical_2d * np.exp(1j * prediction)
        plane_records[wave_mode] = {
            "theoretical": theoretical_2d,
            "measured": measured_2d,
            "fitted": fitted_2d,
            "residual": np.conj(fitted_2d) * measured_2d,
            "selected_bounds": tuple(selection["selected_bounds"]),
        }

        # Cache the already evaluated central-y model for one downstream
        # retry without reloading the large integrated reference scan.
        if (
            wave_mode == "sin"
            and manual_bounds is None
            and selection.get("outcome") == "full_region_preserved"
        ):
            core_bounds = centered_spatial_core_bounds(spatial_coordinates.size)
            core_result = fit_cache.get(core_bounds)
            core_quality = selection.get("central_core_quality")
            if isinstance(core_result, Mapping) and isinstance(core_quality, Mapping):
                core_is_reliable = (
                    float(core_quality.get("valid_readout_fraction", 0.0))
                    >= float(selection["minimum_valid_readout_fraction"])
                    and float(core_quality.get("median_wrapped_rms_rad", float("inf")))
                    <= float(selection["maximum_median_wrapped_rms_rad"])
                )
                if core_is_reliable:
                    core_slope = np.asarray(
                        core_result["a_fit_all"].detach().cpu(), dtype=np.float64
                    )
                    core_constant = np.asarray(
                        core_result["c_fit_all"].detach().cpu(), dtype=np.float64
                    )
                    core_prediction = (
                        core_slope[:, None] * spatial_coordinates[None, :]
                        + core_constant[:, None]
                    )
                    core_fitted = theoretical_2d * np.exp(1j * core_prediction)
                    y_core_retry_component = {
                        "bounds": core_bounds,
                        "result": core_result,
                        "projection_quality": native._projection_fit_quality_summary(
                            core_result
                        ),
                        "selection": {
                            **selection,
                            "selection": "automatic-fallback",
                            "outcome": "centered_y_core_retry",
                            "selected_bounds": list(core_bounds),
                            "selected_quality": dict(core_quality),
                            "clean_case_no_op": False,
                        },
                        "plane": {
                            "theoretical": theoretical_2d,
                            "measured": measured_2d,
                            "fitted": core_fitted,
                            "residual": np.conj(core_fitted) * measured_2d,
                            "selected_bounds": core_bounds,
                        },
                    }

    sin_result, cos_result = coefficient_results
    a_raw = sin_result["a_fit_all"]
    b_raw = cos_result["b_fit_all"]
    c_raw = sin_result["c_fit_all"] + cos_result["c_fit_all"]
    evidence = {
        "readout_index": np.arange(readout_oversampled, dtype=np.int64),
        "projection_quality": projection_quality,
        "spatial_region_selection": selection_records,
        "component_projection_sources": {
            "a": ["sin"],
            "b": ["cos"],
            "c": ["sin", "cos"],
        },
    }
    y_core_retry = None
    if y_core_retry_component is not None:
        core_sin_result = y_core_retry_component["result"]
        y_core_retry = {
            "bounds": y_core_retry_component["bounds"],
            "raw_coefficients": (
                core_sin_result["a_fit_all"],
                cos_result["b_fit_all"],
                core_sin_result["c_fit_all"] + cos_result["c_fit_all"],
            ),
            "calibration_evidence": {
                **evidence,
                "projection_quality": {
                    **projection_quality,
                    "sin": y_core_retry_component["projection_quality"],
                },
                "spatial_region_selection": {
                    **selection_records,
                    "sin": y_core_retry_component["selection"],
                },
            },
            "plane_diagnostics": {
                **plane_records,
                "sin": y_core_retry_component["plane"],
            },
        }
    return (
        a_raw,
        b_raw,
        c_raw,
        calibration_samples,
        evidence,
        plane_records,
        y_core_retry,
    )


def _coefficient_processing_inputs(
    a_raw: Any,
    b_raw: Any,
    c_raw: Any,
    *,
    readout_oversampled: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[Any, Any, Any],
    np.ndarray,
    dict[str, Any],
]:
    """Prepare raw coefficient vectors for smoothing or sine-line fitting.

    Args:
        a_raw: Raw LIN slope coefficients on the oversampled readout grid.
        b_raw: Raw PAR slope coefficients on the oversampled readout grid.
        c_raw: Raw constant phase coefficients on the same grid.
        readout_oversampled: Required coefficient-vector length.

    Returns:
        Original NumPy vectors, branch-aligned processing vectors, native
        processing inputs, integer c branch turns, and alignment provenance.

    Raises:
        ValueError: If any raw coefficient has an unexpected length.
    """
    raw_vectors = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (a_raw, b_raw, c_raw)
    )
    if any(value.size != readout_oversampled for value in raw_vectors):
        raise ValueError(
            "Raw calibrated PSF vectors do not match the oversampled readout."
        )
    c_aligned, c_branch_turns = align_constant_phase_branch(raw_vectors[2])
    processing_vectors = (raw_vectors[0], raw_vectors[1], c_aligned)
    if hasattr(c_raw, "detach"):
        import torch

        c_processing_input = torch.as_tensor(
            c_aligned, dtype=c_raw.dtype, device=c_raw.device
        )
    else:
        c_processing_input = np.asarray(c_aligned, dtype=np.asarray(c_raw).dtype)
    native_processing_vectors = (a_raw, b_raw, c_processing_input)
    branch_alignment_record = {
        "method": "center-anchored-continuous-modulo-2pi",
        "complex_phase_invariant": True,
        "nonzero_shift_count": int(np.count_nonzero(c_branch_turns)),
        "minimum_turns": int(np.min(c_branch_turns)),
        "maximum_turns": int(np.max(c_branch_turns)),
    }
    return (
        raw_vectors,  # type: ignore[return-value]
        processing_vectors,
        native_processing_vectors,
        c_branch_turns,
        branch_alignment_record,
    )


def _calibrated_psf_inputs(
    native: Any,
    *,
    twix_path: Path,
    sequence_path: Path,
    readout_oversampled: int,
    ncalib: int,
    nacs: int,
    coefficient_processing: str,
    fit_kx_min: int | None,
    fit_kx_max: int | None,
    fit_y_bounds: tuple[int, int] | None = None,
    fit_z_bounds: tuple[int, int] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    dict[str, Any],
    dict[str, dict[str, object]],
]:
    """Fit ``a,b,c`` and return them with the sequence Wave trajectories.

    Args:
        native: Namespace of focused upstream Wave-MPRAGE helpers.
        twix_path: Measured TWIX file containing the calibration projection.
        sequence_path: Pulseq sequence used for the acquisition.
        readout_oversampled: Oversampled readout length.
        ncalib: Number of projection-calibration readouts.
        nacs: Integrated reference-scan ACS width.
        coefficient_processing: Upstream ``smooth`` or ``sine-line`` mode.
        fit_kx_min: Inclusive first sine-line fitting index, if selected.
        fit_kx_max: Exclusive final sine-line fitting index, if selected.
        fit_y_bounds: Optional manual sin-projection spatial interval.
        fit_z_bounds: Optional manual cos-projection spatial interval.

    Returns:
        ``delta_lin``, ``delta_par``, and fitted ``a``, ``b``, ``c`` vectors,
        followed by original coefficients, branch-aligned processing inputs,
        integer ``c`` branch turns, processing diagnostics, and plot-ready
        projection planes. All vectors use the oversampled readout grid.

    Raises:
        AutomaticPsfFitRejected: If an automatic sine-line candidate fails
            upstream numerical or extrapolation-stability validation.
        ValueError: If any returned vector has an unexpected length.
    """

    plane_diagnostics: dict[str, dict[str, object]] = {}
    y_core_retry: dict[str, Any] | None = None
    if getattr(native, "supports_spatial_projection_selection", False) is True:
        (
            a_raw,
            b_raw,
            c_raw,
            calibration_samples,
            calibration_evidence,
            plane_diagnostics,
            y_core_retry,
        ) = _fit_projection_coefficients_with_spatial_selection(
            native,
            twix_path=twix_path,
            sequence_path=sequence_path,
            ncalib=ncalib,
            nacs=nacs,
            fit_y_bounds=fit_y_bounds,
            fit_z_bounds=fit_z_bounds,
        )
    else:
        # Test doubles and legacy focused namespaces retain the pinned
        # high-level behavior; production helpers always use spatial selection.
        with tempfile.TemporaryDirectory(prefix="wave_mprage_psf_") as temporary:
            output_prefix = str(Path(temporary)) + "/"
            (
                a_raw,
                b_raw,
                c_raw,
                calibration_samples,
                calibration_evidence,
            ) = native.fit_wave_psf_deviation_from_projection(
                mprage_data_file=str(twix_path),
                mprage_seq_file=str(sequence_path),
                out_folder=output_prefix,
                file_tag="temporary",
                yflip=-1,
                zflip=-1,
                Ncalib=ncalib,
                Nacs=nacs,
                slice_orientation="SAG",
                return_diagnostics=True,
            )
    delta_lin, delta_par = native.generate_theoretical_wave_trajectory(
        fn_seq=str(sequence_path),
        Nx_os=readout_oversampled,
        Nacs_total=calibration_samples,
        slice_orientation="SAG",
    )
    delta_vectors = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (delta_lin, delta_par)
    )
    automatic_spatial_fallback: dict[str, Any] | None = None
    while True:
        (
            raw_vectors,
            processing_vectors,
            native_processing_vectors,
            c_branch_turns,
            branch_alignment_record,
        ) = _coefficient_processing_inputs(
            a_raw,
            b_raw,
            c_raw,
            readout_oversampled=readout_oversampled,
        )
        with tempfile.TemporaryDirectory(
            prefix="wave_mprage_psf_processing_"
        ) as temporary:
            diagnostic_path = (
                Path(temporary) / "psf_sine_line_fit_automatic_candidate.json"
            )
            try:
                a_fit, b_fit, c_fit, processing_diagnostics = (
                    native._process_psf_coefficients(
                        *native_processing_vectors,
                        Nx_os=readout_oversampled,
                        coefficient_processing=coefficient_processing,
                        fit_kx_min=fit_kx_min,
                        fit_kx_max=fit_kx_max,
                        fit_quality=calibration_evidence["projection_quality"],
                        out_folder=temporary,
                        file_tag="automatic_candidate",
                        return_diagnostics=True,
                    )
                )
            except ValueError as exc:
                is_automatic_sine_line = (
                    str(coefficient_processing).strip().lower() == "sine-line"
                    and fit_kx_min is None
                    and fit_kx_max is None
                )
                if not is_automatic_sine_line:
                    raise
                try:
                    rejected_diagnostics = (
                        _load_json(diagnostic_path)
                        if diagnostic_path.is_file()
                        else {}
                    )
                except (OSError, ValueError):
                    rejected_diagnostics = {}
                candidate_coefficients = _candidate_curves_from_fit_diagnostics(
                    rejected_diagnostics, readout_oversampled
                )
                try:
                    recovery = _recover_c_only_automatic_rejection(
                        native,
                        raw_coefficients=processing_vectors,
                        rejected_diagnostics=rejected_diagnostics,
                        delta_lin=delta_vectors[0],
                        delta_par=delta_vectors[1],
                    )
                except ValueError:
                    recovery = None
                if recovery is not None:
                    (a_fit, b_fit, c_fit), processing_diagnostics = recovery
                    break

                should_retry_y_core = (
                    automatic_spatial_fallback is None
                    and fit_y_bounds is None
                    and y_core_retry is not None
                    and SUSTAINED_COEFFICIENT_CORRUPTION_ERROR in str(exc)
                )
                if should_retry_y_core:
                    initial_spatial_selection = calibration_evidence.get(
                        "spatial_region_selection"
                    )
                    automatic_spatial_fallback = {
                        "version": AUTOMATIC_Y_CORE_RETRY_VERSION,
                        "trigger": "automatic-kx-sustained-coefficient-corruption",
                        "axis": "y",
                        "initial_error": str(exc),
                        "initial_spatial_region_selection": copy.deepcopy(
                            initial_spatial_selection
                        ),
                        "fallback_bounds": list(y_core_retry["bounds"]),
                        "outcome": "retry_in_progress",
                        "fallback_was_silent": False,
                    }
                    a_raw, b_raw, c_raw = y_core_retry["raw_coefficients"]
                    calibration_evidence = y_core_retry["calibration_evidence"]
                    plane_diagnostics = y_core_retry["plane_diagnostics"]
                    continue

                rejected_diagnostics["c_phase_branch_alignment"] = (
                    branch_alignment_record
                )
                spatial_selection = calibration_evidence.get(
                    "spatial_region_selection"
                )
                if isinstance(spatial_selection, Mapping):
                    rejected_diagnostics["spatial_region_selection"] = dict(
                        spatial_selection
                    )
                if automatic_spatial_fallback is not None:
                    automatic_spatial_fallback["outcome"] = "retry_rejected"
                    rejected_diagnostics["automatic_spatial_fallback"] = (
                        automatic_spatial_fallback
                    )
                raise AutomaticPsfFitRejected(
                    str(exc),
                    raw_coefficients=raw_vectors,
                    candidate_coefficients=candidate_coefficients,
                    diagnostics=rejected_diagnostics,
                    plane_diagnostics=plane_diagnostics,
                ) from exc
            else:
                break
    processing_diagnostics = dict(processing_diagnostics)
    if automatic_spatial_fallback is not None:
        automatic_spatial_fallback["outcome"] = "centered_y_core_retry_accepted"
        automatic_spatial_fallback["validation_passed"] = True
        processing_diagnostics["automatic_spatial_fallback"] = (
            automatic_spatial_fallback
        )
    processing_diagnostics["c_phase_branch_alignment"] = branch_alignment_record
    spatial_selection = calibration_evidence.get("spatial_region_selection")
    if isinstance(spatial_selection, Mapping):
        processing_diagnostics["spatial_region_selection"] = dict(spatial_selection)
    vectors = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (*delta_vectors, a_fit, b_fit, c_fit)
    )
    if any(
        value.size != readout_oversampled for value in (*vectors, *raw_vectors)
    ):
        raise ValueError("Calibrated PSF vectors do not match the oversampled readout.")
    return (
        *vectors,
        raw_vectors,
        processing_vectors,
        c_branch_turns,
        processing_diagnostics,
        plane_diagnostics,
    )  # type: ignore[return-value]


def _write_real_vectors(base: Path, vectors: tuple[np.ndarray, ...]) -> None:
    """Write equally sized real vectors as a two-dimensional BART array.

    Args:
        base: BART CFL/HDR base path.
        vectors: Real vectors stored as columns.

    Returns:
        None.

    Raises:
        ValueError: If the vectors are empty or have inconsistent lengths.
    """
    if not vectors or any(vector.size != vectors[0].size for vector in vectors):
        raise ValueError("Real vector records must be non-empty and equally sized.")
    output = create_cfl(base, (vectors[0].size, len(vectors)))
    for index, vector in enumerate(vectors):
        output[:, index] = np.asarray(vector, dtype=np.float32)
    output.flush()
    del output


def _read_real_vectors(base: Path, expected_count: int) -> tuple[np.ndarray, ...]:
    """Read real vector columns from a BART array.

    Args:
        base: BART CFL/HDR base path.
        expected_count: Required number of vector columns.

    Returns:
        A tuple of one-dimensional float64 vectors.

    Raises:
        ValueError: If the array shape or imaginary content is invalid.
    """
    data = np.asarray(open_cfl(base))
    if data.ndim != 2 or data.shape[1] != expected_count:
        raise ValueError(f"Unexpected vector record shape for {base}: {data.shape}.")
    if np.max(np.abs(data.imag)) > 1e-7:
        raise ValueError(f"Real vector record contains imaginary values: {base}.")
    return tuple(
        np.asarray(data[:, index].real, dtype=np.float64)
        for index in range(expected_count)
    )


def _ensure_r3x1_psf_coefficient_plot(
    normal_directory: Path,
    sampling_name: str,
    coefficient_vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    psf_settings: Mapping[str, Any],
    *,
    raw_coefficient_vectors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    processing_diagnostics: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Create and announce the native R3x1 PSF coefficient diagnostic.

    Args:
        normal_directory: Dataset ``normal`` output directory.
        sampling_name: Validated measured sampling class.
        coefficient_vectors: Processed ``a``, ``b``, and ``c`` vectors used
            to evaluate the reconstruction PSF.
        psf_settings: Normalized coefficient-processing settings.
        raw_coefficient_vectors: Optional original ``a``, ``b``, and ``c``
            samples to overlay as scatter points.
        processing_diagnostics: Optional accepted automatic-recovery details
            used to label hybrid curves and show a rejected constrained c.
        overwrite: Replace an existing diagnostic with the supplied vectors.

    Returns:
        The diagnostic PNG path for R3x1, otherwise ``None``.
    """
    if sampling_name != "R3x1":
        return None
    destination = normal_directory / PSF_COEFFICIENT_PLOT_NAME
    full_range_destination = normal_directory / PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME
    fit_range = psf_settings.get("fit_kx_range")
    normalized_range = (
        None if fit_range is None else (int(fit_range[0]), int(fit_range[1]))
    )
    plot_processing = str(psf_settings["coefficient_processing"])
    curve_labels = None
    comparison_coefficients = None
    comparison_labels = None
    if processing_diagnostics is not None:
        effective = processing_diagnostics.get("effective_coefficient_processing")
        recovery = processing_diagnostics.get("automatic_c_recovery")
        if isinstance(effective, str) and isinstance(recovery, Mapping):
            plot_processing = effective
            outcome = recovery.get("outcome")
            if outcome == "constrained_common_frequency_c":
                curve_labels = (
                    "strict sine-line fit",
                    "strict sine-line fit",
                    "accepted common-frequency sine-line fit",
                )
            elif outcome == "smooth_c_fallback":
                curve_labels = (
                    "strict sine-line fit",
                    "strict sine-line fit",
                    "accepted 9-point smooth fallback",
                )
                constrained = recovery.get("constrained_c_fit")
                comparison_c = (
                    _curve_from_fit_parameters(constrained, coefficient_vectors[0].size)
                    if isinstance(constrained, Mapping)
                    else None
                )
                if comparison_c is not None:
                    comparison_coefficients = (None, None, comparison_c)
                    comparison_labels = ("", "", "rejected constrained c fit")
    if overwrite or not destination.is_file():
        write_psf_coefficient_plot(
            *coefficient_vectors,
            destination,
            processing=plot_processing,
            fit_kx_range=normalized_range,
            fit_range_selection=psf_settings.get("fit_range_selection"),
            raw_coefficients=raw_coefficient_vectors,
            curve_labels=curve_labels,
            comparison_coefficients=comparison_coefficients,
            comparison_labels=comparison_labels,
        )
    if overwrite or not full_range_destination.is_file():
        write_psf_coefficient_plot(
            *coefficient_vectors,
            full_range_destination,
            processing=plot_processing,
            fit_kx_range=normalized_range,
            fit_range_selection=psf_settings.get("fit_range_selection"),
            raw_coefficients=raw_coefficient_vectors,
            curve_labels=curve_labels,
            comparison_coefficients=comparison_coefficients,
            comparison_labels=comparison_labels,
            full_range=True,
        )
    print(f"PSF coefficient visual-assessment plot: {destination}")
    print(f"PSF coefficient full-range plot: {full_range_destination}")
    print(
        "If reconstruction has unexpected artifacts, inspect this plot and "
        "tools/wave_retro_lr_recon/TROUBLESHOOTING.md."
    )
    return destination


def _write_automatic_psf_rejection_diagnostics(
    normal_directory: Path,
    rejection: AutomaticPsfFitRejected,
    *,
    twix_path: Path,
    sequence_path: Path,
) -> tuple[Path, Path]:
    """Persist a rejected automatic fit without creating reusable BART inputs.

    Args:
        normal_directory: Dataset ``normal`` output directory.
        rejection: Rejected fit with raw samples and optional candidate curves.
        twix_path: Measured TWIX source identity to record.
        sequence_path: Pulseq sequence source identity to record.

    Returns:
        Paths to the rejected-fit PNG and JSON diagnostics.
    """
    fit_range = rejection.diagnostics.get("kx_range")
    normalized_range = None
    if isinstance(fit_range, (list, tuple)) and len(fit_range) == 2:
        normalized_range = (int(fit_range[0]), int(fit_range[1]))
    candidate = rejection.candidate_coefficients
    plot_path = normal_directory / PSF_COEFFICIENT_REJECTED_PLOT_NAME
    write_psf_coefficient_plot(
        *(candidate if candidate is not None else (None, None, None)),
        plot_path,
        processing="sine-line",
        fit_kx_range=normalized_range,
        fit_range_selection="automatic",
        raw_coefficients=rejection.raw_coefficients,
        accepted_for_reconstruction=False,
    )
    full_range_plot_path = (
        normal_directory / PSF_COEFFICIENT_REJECTED_FULL_RANGE_PLOT_NAME
    )
    write_psf_coefficient_plot(
        *(candidate if candidate is not None else (None, None, None)),
        full_range_plot_path,
        processing="sine-line",
        fit_kx_range=normalized_range,
        fit_range_selection="automatic",
        raw_coefficients=rejection.raw_coefficients,
        accepted_for_reconstruction=False,
        full_range=True,
    )
    plane_plot_path = None
    if rejection.plane_diagnostics:
        plane_plot_path = write_psf_plane_comparison_plot(
            rejection.plane_diagnostics,
            normal_directory / PSF_PLANE_COMPARISON_PLOT_NAME,
        )
    json_path = normal_directory / PSF_COEFFICIENT_REJECTED_DIAGNOSTICS_NAME
    _write_json(
        json_path,
        {
            "format_version": 1,
            "status": "automatic_sine_line_psf_fit_rejected",
            "created_utc": _utc_now(),
            "error": str(rejection),
            "source": {
                "twix": _file_identity(twix_path),
                "sequence": _file_identity(sequence_path, include_hash=True),
            },
            "plot_relative_to_output_root": (
                f"normal/{PSF_COEFFICIENT_REJECTED_PLOT_NAME}"
            ),
            "full_range_plot_relative_to_output_root": (
                f"normal/{PSF_COEFFICIENT_REJECTED_FULL_RANGE_PLOT_NAME}"
            ),
            "accepted_for_reconstruction": False,
            **(
                {
                    "plane_comparison_plot_relative_to_output_root": (
                        f"normal/{PSF_PLANE_COMPARISON_PLOT_NAME}"
                    )
                }
                if plane_plot_path is not None
                else {}
            ),
            "manual_override": {
                "fit_kx_range_convention": "half-open [min, max)",
                "required_arguments": ["--psf-fit-kx-min", "--psf-fit-kx-max"],
            },
            "upstream_fit_diagnostics": rejection.diagnostics,
        },
    )
    return plot_path, json_path


def _write_spatial_region_rejection_diagnostics(
    normal_directory: Path,
    rejection: AutomaticSpatialRegionRejected,
    *,
    twix_path: Path,
    sequence_path: Path,
) -> tuple[Path, Path]:
    """Persist a rejected spatial-region selection before BART input export.

    Args:
        normal_directory: Dataset ``normal`` output directory.
        rejection: Automatic spatial selector rejection and plane evidence.
        twix_path: Measured TWIX source.
        sequence_path: Matching Pulseq source.

    Returns:
        Rejected plane-comparison PNG and JSON paths.
    """
    plane_name = str(rejection.diagnostics.get("axis", "y"))
    mode = "sin" if plane_name == "y" else "cos"
    plot_path = normal_directory / PSF_PLANE_REJECTED_PLOT_NAME
    write_psf_plane_comparison_plot({mode: rejection.plane}, plot_path)
    json_path = normal_directory / PSF_PLANE_REJECTED_DIAGNOSTICS_NAME
    _write_json(
        json_path,
        {
            "format_version": 1,
            "status": "automatic_projection_spatial_region_rejected",
            "created_utc": _utc_now(),
            "error": str(rejection),
            "source": {
                "twix": _file_identity(twix_path),
                "sequence": _file_identity(sequence_path, include_hash=True),
            },
            "plot_relative_to_output_root": f"normal/{PSF_PLANE_REJECTED_PLOT_NAME}",
            "selection_diagnostics": rejection.diagnostics,
            "manual_override": {
                "fit_spatial_range_convention": "half-open calibration-image indices",
                "y_arguments": ["--psf-fit-y-min", "--psf-fit-y-max"],
                "z_arguments": ["--psf-fit-z-min", "--psf-fit-z-max"],
            },
        },
    )
    return plot_path, json_path


def _native_manifest_matches(
    manifest: Mapping[str, Any],
    twix_path: Path,
    sequence_path: Path,
    psf_settings: Mapping[str, Any],
) -> bool:
    """Check whether a native manifest matches sources and PSF settings.

    Args:
        manifest: Existing preparation manifest.
        twix_path: Requested measured TWIX file.
        sequence_path: Requested Pulseq sequence file.
        psf_settings: Normalized coefficient-processing settings.

    Returns:
        ``True`` when the status, source identities, and PSF settings match.
    """
    recorded_psf = manifest.get("psf_calibration", {})
    recorded_processing = recorded_psf.get("processing_diagnostics", {})
    if isinstance(recorded_processing, Mapping):
        recovery = recorded_processing.get("automatic_c_recovery")
        if isinstance(recovery, Mapping) and recovery.get("version") != (
            AUTOMATIC_C_RECOVERY_VERSION
        ):
            return False
    recorded_mode = recorded_psf.get("coefficient_processing", "smooth")
    recorded_selection = recorded_psf.get("fit_range_selection")
    if recorded_selection is None and recorded_mode == "sine-line":
        recorded_selection = (
            "manual" if recorded_psf.get("fit_kx_range") is not None else "automatic"
        )
    recorded_requested_range = recorded_psf.get("requested_fit_kx_range")
    if recorded_requested_range is None and recorded_selection == "manual":
        recorded_requested_range = recorded_psf.get("fit_kx_range")
    recorded_settings = {
        "coefficient_processing": recorded_mode,
        "fit_range_selection": recorded_selection,
        "requested_fit_kx_range": recorded_requested_range,
        "fit_kx_range_convention": recorded_psf.get(
            "fit_kx_range_convention", "half-open"
        ),
        "spatial_region_selection": recorded_psf.get("spatial_region_selection"),
        "spatial_region_selection_version": recorded_psf.get(
            "spatial_region_selection_version"
        ),
        "requested_fit_y_range": recorded_psf.get("requested_fit_y_range"),
        "requested_fit_z_range": recorded_psf.get("requested_fit_z_range"),
        "fit_spatial_range_convention": recorded_psf.get(
            "fit_spatial_range_convention"
        ),
        "processing_implementation": recorded_psf.get("processing_implementation"),
    }
    return (
        manifest.get("status") == "measured_wave_mprage_bart_inputs_ready"
        and manifest.get("source", {}).get("twix") == _file_identity(twix_path)
        and manifest.get("source", {}).get("sequence")
        == _file_identity(sequence_path, include_hash=True)
        and recorded_settings == dict(psf_settings)
    )


def prepare_normal_mprage(
    twix: str | Path,
    output_root: str | Path,
    sequence: str | Path,
    *,
    psf_coefficient_processing: str = "smooth",
    psf_fit_kx_min: int | None = None,
    psf_fit_kx_max: int | None = None,
    psf_fit_y_min: int | None = None,
    psf_fit_y_max: int | None = None,
    psf_fit_z_min: int | None = None,
    psf_fit_z_max: int | None = None,
    reuse: bool = True,
) -> dict[str, Any]:
    """Prepare native measured-Wave k-space, calibration k-space, and PSF.

    Args:
        twix: Measured Wave-MPRAGE TWIX file.
        output_root: Dataset-specific output root.
        sequence: Pulseq sequence file used for the acquisition.
        psf_coefficient_processing: Upstream ``smooth`` or ``sine-line`` mode.
        psf_fit_kx_min: Inclusive first sine-line fitting index, if selected.
        psf_fit_kx_max: Exclusive final sine-line fitting index, if selected.
        psf_fit_y_min: Optional inclusive manual sin-projection spatial index.
        psf_fit_y_max: Optional exclusive manual sin-projection spatial index.
        psf_fit_z_min: Optional inclusive manual cos-projection spatial index.
        psf_fit_z_max: Optional exclusive manual cos-projection spatial index.
        reuse: Reuse compatible inputs already present under ``output_root``.

    Returns:
        The native BART-input manifest.

    Raises:
        ValueError: If acquisition metadata, data dimensions, or existing
            inputs are incompatible.
        FileExistsError: If a non-reusable input directory is non-empty.
    """

    import torch

    twix_path = Path(twix).expanduser().resolve()
    sequence_path = Path(sequence).expanduser().resolve()
    psf_settings = _normalize_psf_coefficient_settings(
        psf_coefficient_processing, psf_fit_kx_min, psf_fit_kx_max
    )
    psf_settings.update(
        _normalize_psf_spatial_settings(
            psf_fit_y_min,
            psf_fit_y_max,
            psf_fit_z_min,
            psf_fit_z_max,
        )
    )
    psf_settings["processing_implementation"] = (
        _psf_processing_implementation_identity()
    )
    destination = Path(output_root).expanduser().resolve() / NORMAL_INPUT_RELATIVE
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file() and reuse:
        existing = _load_json(manifest_path)
        if not _native_manifest_matches(existing, twix_path, sequence_path, psf_settings):
            raise ValueError(
                "Existing normal BART inputs use different sources or PSF "
                "coefficient-processing settings."
            )
        for name in (
            "wave_kspace",
            "kspace_calib",
            "psf",
            "wave_trajectory",
            "psf_coefficients",
            "psf_coefficients_processing_input",
            "psf_coefficient_c_branch_turns",
        ):
            read_shape(destination / name)
        a_fit, b_fit, c_fit = _read_real_vectors(destination / "psf_coefficients", 3)
        raw_name = existing.get("psf_calibration", {}).get("raw_psf_coefficients")
        raw_coefficients = (
            None
            if raw_name is None
            else _read_real_vectors(destination / str(raw_name), 3)
        )
        effective_psf_settings = {
            **psf_settings,
            "fit_kx_range": existing.get("psf_calibration", {}).get("fit_kx_range"),
        }
        diagnostic = _ensure_r3x1_psf_coefficient_plot(
            destination.parent,
            str(existing["sampling"]["name"]),
            (a_fit, b_fit, c_fit),
            effective_psf_settings,
            raw_coefficient_vectors=raw_coefficients,
            processing_diagnostics=existing.get("psf_calibration", {}).get(
                "processing_diagnostics"
            ),
        )
        if diagnostic is not None:
            expected_relative = f"normal/{PSF_COEFFICIENT_PLOT_NAME}"
            expected_full_relative = (
                f"normal/{PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME}"
            )
            calibration = existing.setdefault("psf_calibration", {})
            recorded_relative = calibration.get(
                "visual_assessment_plot_relative_to_output_root"
            )
            if (
                recorded_relative != expected_relative
                or calibration.get("full_range_plot_relative_to_output_root")
                != expected_full_relative
            ):
                calibration[
                    "visual_assessment_plot_relative_to_output_root"
                ] = expected_relative
                calibration["full_range_plot_relative_to_output_root"] = (
                    expected_full_relative
                )
                _write_json(manifest_path, existing)
        print(f"Reusing compatible normal BART inputs: {destination}")
        return existing
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Normal BART input directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    native = load_wave_mprage_helpers()
    definitions, upstream_geometry = _read_sequence(sequence_path)
    nro = int(upstream_geometry["Nro"])
    nlin = int(upstream_geometry["Nlin"])
    npar = int(upstream_geometry["Npar"])
    os_factor = int(definitions.get("ReadoutOversamplingFactor", 4))
    ro_os = nro * os_factor
    ncalib = int(definitions.get("Calibration_Ncalib1", 72))
    nacs = int(definitions.get("Calibration_Nacs", 32))
    requested_fit_kx_range = psf_settings["requested_fit_kx_range"]
    if requested_fit_kx_range is not None and int(requested_fit_kx_range[1]) > ro_os:
        raise ValueError(
            f"PSF fit kx max {requested_fit_kx_range[1]} exceeds oversampled readout {ro_os}."
        )
    for axis in ("y", "z"):
        requested = psf_settings[f"requested_fit_{axis}_range"]
        if requested is not None and int(requested[1]) > ncalib:
            raise ValueError(
                f"PSF {axis} spatial fit max {requested[1]} exceeds calibration "
                f"matrix {ncalib}."
            )
    native._resolve_mprage_wave_mode(
        "wave", str(sequence_path), ro_os, ncalib, nacs, slice_orientation="SAG"
    )

    # Validate the measured Wave image stream independently of the set-4
    # no-Wave reference scan used for sensitivity-map calibration.
    sampling, _ = inspect_twix_sampling(twix_path, matrix_lin_par=(nlin, npar))
    # Load and consume the large dense reference before the image stream so
    # the two physical-coil payloads are never resident together unnecessarily.
    reference = native.load_ref(str(twix_path))
    native._check_integrated_refscan_shape(reference, Nacs=nacs, Ncalib=ncalib)
    physical_coils = int(reference.shape[-1])
    if reference.shape[0] != ro_os:
        raise ValueError(
            f"Integrated refscan readout {reference.shape[0]} does not match expected {ro_os}."
        )
    if physical_coils < 12:
        raise ValueError("MPRAGE BART preparation requires at least 12 physical receive coils.")
    # Compute one shared coil-compression basis from the integrated reference
    # scan, then apply it consistently to image and calibration data.
    integrated_acs = reference[:, :nacs, :nacs, -1, :]
    if nacs > nlin or nacs > npar:
        raise ValueError(f"Integrated ACS size {nacs} does not fit the PE grid {(nlin, npar)}.")
    basis, singular_values, retained_energy = native.estimate_cc_matrix_coillast(
        integrated_acs,
        ncc=12,
        acs=nacs,
        x_step=os_factor,
    )
    compressed_acs = native.apply_cc_coillast_torch(integrated_acs, basis, x_chunk=8)[
        ::os_factor
    ].contiguous()
    del integrated_acs, reference

    image = native.load_img(str(twix_path))
    full_image = _embed_image_stream(
        image,
        sampling,
        readout_oversampled=ro_os,
        physical_coils=physical_coils,
    )
    compressed = native.apply_cc_coillast_torch(full_image, basis, x_chunk=8)
    del full_image, image
    if tuple(compressed_acs.shape) != (nro, nacs, nacs, 12):
        raise ValueError(f"Unexpected compressed ACS shape: {tuple(compressed_acs.shape)}.")
    if not torch.isfinite(compressed).all() or not torch.isfinite(compressed_acs).all():
        raise ValueError("Compressed image or calibration k-space contains non-finite values.")
    calibration = torch.zeros((nro, nlin, npar, 12), dtype=torch.complex64)
    lin_start = nlin // 2 - nacs // 2
    par_start = npar // 2 - nacs // 2
    calibration[
        :, lin_start : lin_start + nacs, par_start : par_start + nacs, :
    ] = compressed_acs

    # Evaluate the calibrated Wave model on the native acquisition grid.
    try:
        (
            delta_lin,
            delta_par,
            a_fit,
            b_fit,
            c_fit,
            raw_coefficients,
            processing_input_coefficients,
            c_branch_turns,
            psf_processing_diagnostics,
            projection_planes,
        ) = _calibrated_psf_inputs(
            native,
            twix_path=twix_path,
            sequence_path=sequence_path,
            readout_oversampled=ro_os,
            ncalib=ncalib,
            nacs=nacs,
            coefficient_processing=str(psf_settings["coefficient_processing"]),
            fit_kx_min=(
                None if requested_fit_kx_range is None else requested_fit_kx_range[0]
            ),
            fit_kx_max=(
                None if requested_fit_kx_range is None else requested_fit_kx_range[1]
            ),
            fit_y_bounds=(
                None
                if psf_settings["requested_fit_y_range"] is None
                else tuple(psf_settings["requested_fit_y_range"])
            ),
            fit_z_bounds=(
                None
                if psf_settings["requested_fit_z_range"] is None
                else tuple(psf_settings["requested_fit_z_range"])
            ),
        )
    except AutomaticSpatialRegionRejected as exc:
        plot_path, json_path = _write_spatial_region_rejection_diagnostics(
            destination.parent,
            exc,
            twix_path=twix_path,
            sequence_path=sequence_path,
        )
        raise ValueError(
            f"{exc} Rejected plane PNG: {plot_path}. Selection diagnostics: "
            f"{json_path}. Review the plane comparison, then provide the matching "
            "--psf-fit-y-min/--psf-fit-y-max or "
            "--psf-fit-z-min/--psf-fit-z-max manual interval."
        ) from exc
    except AutomaticPsfFitRejected as exc:
        plot_path, json_path = _write_automatic_psf_rejection_diagnostics(
            destination.parent,
            exc,
            twix_path=twix_path,
            sequence_path=sequence_path,
        )
        raise ValueError(
            f"{exc} Rejected-fit PNG: {plot_path}. Fit diagnostics: {json_path}. "
            "Review the raw samples and shaded interval, then rerun with both "
            "--psf-fit-kx-min and --psf-fit-kx-max for a manual half-open range."
        ) from exc
    automatic_spatial_fallback = psf_processing_diagnostics.get(
        "automatic_spatial_fallback"
    )
    if isinstance(automatic_spatial_fallback, Mapping):
        bounds = automatic_spatial_fallback["fallback_bounds"]
        print(
            "Accepted automatic centered-y PSF retry after full-region "
            f"coefficient corruption: [{bounds[0]}, {bounds[1]})"
        )
    selected_fit_kx_range = psf_processing_diagnostics.get("kx_range")
    effective_psf_settings = {
        **psf_settings,
        "fit_kx_range": selected_fit_kx_range,
    }
    calibrated_psf = evaluate_calibrated_psf(
        delta_lin,
        delta_par,
        a_fit,
        b_fit,
        c_fit,
        nlin=nlin,
        npar=npar,
    )

    # Persist BART arrays and the reusable physical calibration vectors.
    wave_output = create_cfl(destination / "wave_kspace", (ro_os, nlin, npar, 12, 1))
    wave_output[:, :, :, :, 0] = compressed.cpu().numpy()
    wave_output.flush()
    del wave_output
    calibration_output = create_cfl(destination / "kspace_calib", (nro, nlin, npar, 12))
    calibration_output[:] = calibration.cpu().numpy()
    calibration_output.flush()
    del calibration_output
    psf_output = create_cfl(destination / "psf", (ro_os, nlin, npar, 1, 1))
    psf_output[:, :, :, 0, 0] = calibrated_psf
    psf_output.flush()
    del psf_output
    _write_real_vectors(destination / "wave_trajectory", (delta_lin, delta_par))
    _write_real_vectors(destination / "psf_coefficients", (a_fit, b_fit, c_fit))
    _write_real_vectors(destination / "psf_coefficients_raw", raw_coefficients)
    _write_real_vectors(
        destination / "psf_coefficients_processing_input",
        processing_input_coefficients,
    )
    _write_real_vectors(
        destination / "psf_coefficient_c_branch_turns",
        (c_branch_turns.astype(np.float64),),
    )
    plane_diagnostic = None
    if projection_planes:
        plane_diagnostic = write_psf_plane_comparison_plot(
            projection_planes,
            destination.parent / PSF_PLANE_COMPARISON_PLOT_NAME,
        )
    diagnostic = _ensure_r3x1_psf_coefficient_plot(
        destination.parent,
        sampling.name,
        (a_fit, b_fit, c_fit),
        effective_psf_settings,
        raw_coefficient_vectors=raw_coefficients,
        processing_diagnostics=psf_processing_diagnostics,
        overwrite=True,
    )

    physical_fov_mm_xyz = tuple(
        float(value) * 1000.0 for value in upstream_geometry["FOVxyz"]
    )
    manifest: dict[str, Any] = {
        "format_version": 2,
        "status": "measured_wave_mprage_bart_inputs_ready",
        "source": {
            "twix": _file_identity(twix_path),
            "sequence": _file_identity(sequence_path, include_hash=True),
            "wave_mprage_helpers": (
                "external/wave-mprage/recon/utils direct TWIX/coil helpers plus focused "
                "calibration/orientation callables imported from the reconstruction module; "
                "Python reconstruction main() not called"
            ),
        },
        "geometry": {
            "physical_fov_mm_xyz": list(physical_fov_mm_xyz),
            "logical_matrix_ro_lin_par": [nro, nlin, npar],
            "readout_oversampling_factor": os_factor,
        },
        "sampling": sampling.to_json(),
        "coil_compression": {
            "physical_coils": physical_coils,
            "virtual_coils": 12,
            "method": "integrated ACS covariance eigendecomposition",
            "retained_energy": float(retained_energy[11]),
            "leading_singular_values": [float(value) for value in singular_values[:12]],
        },
        "psf_calibration": {
            "method": "sequence trajectory plus processed integrated projection a,b,c",
            **psf_settings,
            "effective_coefficient_processing": psf_processing_diagnostics.get(
                "effective_coefficient_processing",
                psf_settings["coefficient_processing"],
            ),
            "fit_kx_range": selected_fit_kx_range,
            "processing_diagnostics": psf_processing_diagnostics,
            "trajectory_sign_lin_par": [-1, -1],
            "ncalib": ncalib,
            "nacs": nacs,
            "wave_trajectory": "wave_trajectory",
            "psf_coefficients": "psf_coefficients",
            "raw_psf_coefficients": "psf_coefficients_raw",
            "processing_input_psf_coefficients": (
                "psf_coefficients_processing_input"
            ),
            "c_phase_branch_turns": "psf_coefficient_c_branch_turns",
            "c_phase_branch_turn_units": "integer multiples of 2*pi radians",
            **(
                {
                    "visual_assessment_plot_relative_to_output_root": (
                        f"normal/{PSF_COEFFICIENT_PLOT_NAME}"
                    )
                }
                if diagnostic is not None
                else {}
            ),
            **(
                {
                    "full_range_plot_relative_to_output_root": (
                        f"normal/{PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME}"
                    ),
                    "plane_comparison_plot_relative_to_output_root": (
                        f"normal/{PSF_PLANE_COMPARISON_PLOT_NAME}"
                    ),
                }
                if plane_diagnostic is not None
                else {
                    "full_range_plot_relative_to_output_root": (
                        f"normal/{PSF_COEFFICIENT_FULL_RANGE_PLOT_NAME}"
                    )
                }
            ),
        },
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "kspace_calib": "kspace_calib",
        "kspace_calib_shape": list(read_shape(destination / "kspace_calib")),
        "echoes": [
            {
                "echo": 1,
                "wave_kspace": "wave_kspace",
                "wave_kspace_shape": list(read_shape(destination / "wave_kspace")),
                "wave_kspace_norm": float(torch.linalg.vector_norm(compressed).item()),
                "psf": "psf",
                "psf_shape": list(read_shape(destination / "psf")),
            }
        ],
        "prepared_at_utc": _utc_now(),
    }
    _write_json(manifest_path, manifest)
    (destination / "sampling_class.txt").write_text(sampling.name + "\n", encoding="utf-8")
    print(f"Prepared normal measured-Wave BART inputs: {destination}")
    return manifest


def _sampling_from_manifest(payload: Mapping[str, Any]) -> SamplingPattern:
    """Reconstruct a sampling pattern from a normal-input manifest.

    Args:
        payload: Normal-input manifest containing a ``sampling`` object.

    Returns:
        The reconstructed sampling-pattern record.
    """
    sampling = payload["sampling"]
    return SamplingPattern(
        name=str(sampling["name"]),
        acceleration_lin_par=tuple(int(value) for value in sampling["acceleration_lin_par"]),
        lin_residue=None if sampling["lin_residue"] is None else int(sampling["lin_residue"]),
        matrix_lin_par=tuple(int(value) for value in sampling["matrix_lin_par"]),
        acquired_lin=tuple(int(value) for value in sampling["acquired_lin"]),
        acquired_par=tuple(int(value) for value in sampling["acquired_par"]),
        measurement_index=sampling.get("measurement_index"),
        skip_lin_par=tuple(int(value) for value in sampling["skip_lin_par"]),
    )


def prepare_retro_mprage(
    twix: str | Path,
    output_root: str | Path,
    sequence: str | Path,
    *,
    psf_coefficient_processing: str = "smooth",
    psf_fit_kx_min: int | None = None,
    psf_fit_kx_max: int | None = None,
    psf_fit_y_min: int | None = None,
    psf_fit_y_max: int | None = None,
    psf_fit_z_min: int | None = None,
    psf_fit_z_max: int | None = None,
) -> list[dict[str, Any]]:
    """Prepare native R3x2 and three direct-crop LR R3x2 BART input sets.

    Args:
        twix: Measured Wave-MPRAGE TWIX file.
        output_root: Dataset-specific output root shared with normal preparation.
        sequence: Pulseq sequence file used for the acquisition.
        psf_coefficient_processing: Upstream ``smooth`` or ``sine-line`` mode.
        psf_fit_kx_min: Inclusive first sine-line fitting index, if selected.
        psf_fit_kx_max: Exclusive final sine-line fitting index, if selected.
        psf_fit_y_min: Optional inclusive manual sin-projection spatial index.
        psf_fit_y_max: Optional exclusive manual sin-projection spatial index.
        psf_fit_z_min: Optional inclusive manual cos-projection spatial index.
        psf_fit_z_max: Optional exclusive manual cos-projection spatial index.

    Returns:
        One manifest per resolved native or low-resolution case.

    Raises:
        ValueError: If requested resolutions collapse to duplicate grids or
            existing cases were prepared from different sources.
        FileExistsError: If an incompatible case input directory is non-empty.
    """

    output_path = Path(output_root).expanduser().resolve()
    normal = prepare_normal_mprage(
        twix,
        output_path,
        sequence,
        psf_coefficient_processing=psf_coefficient_processing,
        psf_fit_kx_min=psf_fit_kx_min,
        psf_fit_kx_max=psf_fit_kx_max,
        psf_fit_y_min=psf_fit_y_min,
        psf_fit_y_max=psf_fit_y_max,
        psf_fit_z_min=psf_fit_z_min,
        psf_fit_z_max=psf_fit_z_max,
        reuse=True,
    )
    normal_inputs = output_path / NORMAL_INPUT_RELATIVE
    retro_root = output_path / RETRO_RELATIVE
    retro_root.mkdir(parents=True, exist_ok=True)

    geometry_payload = normal["geometry"]
    geometry = Geometry(
        physical_fov_mm_xyz=tuple(
            float(value) for value in geometry_payload["physical_fov_mm_xyz"]
        ),
        logical_matrix_ro_lin_par=tuple(
            int(value) for value in geometry_payload["logical_matrix_ro_lin_par"]
        ),
    )
    source_sampling = _sampling_from_manifest(normal)
    source_mask = source_sampling.mask()
    delta_lin, delta_par = _read_real_vectors(normal_inputs / "wave_trajectory", 2)
    a_fit, b_fit, c_fit = _read_real_vectors(normal_inputs / "psf_coefficients", 3)
    native_resolution = geometry.physical_resolution_mm_xyz
    source_identity = normal["source"]
    results: list[dict[str, Any]] = []
    # Resolve requested physical resolutions to the nearest centered PE grids
    # whose dimensions remain divisible by four.
    resolved_cases = []
    for directory_name, requested_xy in RETRO_CASES:
        requested = (
            native_resolution
            if requested_xy is None
            else (requested_xy[0], requested_xy[1], native_resolution[2])
        )
        case = resolve_case(CaseSpec(requested, (3, 2), directory_name), geometry)
        resolved_cases.append((directory_name, case))
    matrices = [case.target_logical_matrix_ro_lin_par for _, case in resolved_cases]
    if len(set(matrices)) != len(matrices):
        raise ValueError(
            "Two requested retrospective resolutions resolve to the same PE matrix; "
            "this source geometry cannot represent all four distinct cases."
        )

    # Directly center-crop measured Wave samples and re-evaluate the calibrated
    # PSF on each case grid; no interpolation or forward simulation is used.
    for directory_name, case in resolved_cases:
        case_dir = retro_root / directory_name
        inputs = case_dir / "bart_inputs"
        manifest_path = inputs / "manifest.json"
        if manifest_path.is_file():
            existing = _load_json(manifest_path)
            if existing.get("source") != source_identity or existing.get("case") != case.to_json():
                raise ValueError(f"Existing retrospective case has different inputs: {case_dir}")
            for name in ("wave_kspace", "psf"):
                read_shape(inputs / name)
            print(f"Reusing compatible retrospective BART inputs: {inputs}")
            results.append(existing)
            continue
        if inputs.exists() and any(inputs.iterdir()):
            raise FileExistsError(f"Retrospective BART input directory is not empty: {inputs}")
        inputs.mkdir(parents=True, exist_ok=True)
        _, target_lin, target_par = case.target_logical_matrix_ro_lin_par
        psf = evaluate_calibrated_psf(
            delta_lin,
            delta_par,
            a_fit,
            b_fit,
            c_fit,
            nlin=target_lin,
            npar=target_par,
        )
        psf_output = create_cfl(inputs / "psf", (psf.shape[0], target_lin, target_par, 1, 1))
        psf_output[:, :, :, 0, 0] = psf
        psf_output.flush()
        del psf_output
        crop_metrics = write_measured_wave_crop(
            normal_inputs / "wave_kspace",
            inputs / "wave_kspace",
            case,
            source_mask,
            source_sampling.acceleration_lin_par,
        )
        manifest = {
            "format_version": 1,
            "status": "direct_measured_wave_crop_bart_inputs_ready",
            "source": source_identity,
            "source_normal_manifest": str(normal_inputs / "manifest.json"),
            "case_directory": directory_name,
            "case": case.to_json(),
            "operator": "direct center crop of measured Wave k-space in LIN/PAR",
            "interpolation": False,
            "forward_simulation": False,
            "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
            "echoes": [
                {
                    "echo": 1,
                    "wave_kspace": "wave_kspace",
                    "wave_kspace_shape": list(read_shape(inputs / "wave_kspace")),
                    "wave_kspace_norm": crop_metrics["wave_kspace_norm"],
                    "psf": "psf",
                    "psf_shape": list(read_shape(inputs / "psf")),
                }
            ],
            "sampling": crop_metrics,
            "prepared_at_utc": _utc_now(),
        }
        _write_json(manifest_path, manifest)
        print(
            f"Prepared {directory_name}: logical={case.target_logical_matrix_ro_lin_par}, "
            f"achieved={case.achieved_resolution_mm_xyz} mm"
        )
        results.append(manifest)
    _write_json(
        retro_root / "manifest.json",
        {
            "format_version": 1,
            "status": "mprage_retro_cases_ready",
            "source": source_identity,
            "normal_inputs": str(normal_inputs),
            "cases": [payload["case_directory"] for payload in results],
            "prepared_at_utc": _utc_now(),
        },
    )
    return results


def prepare_retro_sensitivity_maps(output_root: str | Path) -> None:
    """Derive all LR CSM grids from the one native BART ecalib result.

    Args:
        output_root: Dataset-specific output root containing normal and retro
            BART inputs and the native ``coil_sens`` result.

    Returns:
        None.

    Raises:
        ValueError: If an existing low-resolution map has the wrong grid.
    """

    root = Path(output_root).expanduser().resolve()
    source_maps = root / NORMAL_OUTPUT_RELATIVE / "coil_sens"
    read_shape(source_maps)
    retro_root = root / RETRO_RELATIVE
    for directory_name, requested_xy in RETRO_CASES:
        if requested_xy is None:
            continue
        inputs = retro_root / directory_name / "bart_inputs"
        manifest = _load_json(inputs / "manifest.json")
        target = tuple(int(value) for value in manifest["case"]["target_logical_matrix_ro_lin_par"])
        output_maps = inputs / "coil_sens"
        if output_maps.with_suffix(".hdr").is_file() and output_maps.with_suffix(".cfl").is_file():
            if read_shape(output_maps)[:3] != target:
                raise ValueError(
                    f"Existing LR sensitivity maps have the wrong shape: {output_maps}"
                )
            print(f"Reusing compatible LR sensitivity maps: {output_maps}")
            continue
        resample_sensitivity_maps(
            source_maps, output_maps, target_lin_par=(target[1], target[2])
        )
        manifest["coil_sens"] = "coil_sens"
        manifest["coil_sens_shape"] = list(read_shape(output_maps))
        manifest["coil_sens_source"] = str(source_maps)
        manifest["coil_sens_resampling"] = (
            "same-FOV centered Fourier PE resampling plus RSS normalization"
        )
        _write_json(inputs / "manifest.json", manifest)
        print(f"Prepared LR sensitivity maps: {output_maps}")
