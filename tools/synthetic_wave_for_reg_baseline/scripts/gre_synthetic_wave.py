"""Scientific contracts for the two-echo synthetic-Wave GRE sweep."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TOOL_ROOT.parents[1]
RETRO_TOOL_ROOT = REPOSITORY_ROOT / "tools" / "wave_retro_lr_recon"

import sys

if str(RETRO_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(RETRO_TOOL_ROOT))

from wave_retro_lr.bart_io import bart_base  # noqa: E402
from wave_retro_lr.core import center_crop_bounds  # noqa: E402
from wave_retro_lr.sampling import (  # noqa: E402
    PURE_CARTESIAN_IMAGE_LATTICE,
    pure_cartesian_image_lattice_mask,
    validate_pure_cartesian_image_lattice,
)

WORKFLOW_NAME = "synthetic_wave_gre_regularization_sweep"
CASE_IDS = ("native_r3x1", "native_r3x2", "lin_low_resolution_r3x2")
ECHO_IDS = ("echo-01", "echo-02")
NATIVE_MATRIX = (250, 250, 72)
NATIVE_FOV_MM = (220.0, 220.0, 180.0)
NATIVE_VOXEL_MM = (0.88, 0.88, 2.5)
SOURCE_MATRIX = (256, 256, 72)
SOURCE_TO_NATIVE_BOUNDS = ((3, 253), (3, 253), (0, 72))
LOW_RESOLUTION_MATRIX = (250, 148, 72)
LOW_RESOLUTION_LIN_BOUNDS = (51, 199)
EXTENDED_READOUT = 1000
VIRTUAL_COILS = 12
ECHO_TIMES_S = (0.010, 0.020)
COARSE_LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
LLR_BLOCK_SIZES = (4, 8, 16)


@dataclass(frozen=True)
class GreCase:
    """Describe one target geometry and pure sampling lattice."""

    case_id: str
    matrix_ro_lin_par: tuple[int, int, int]
    fov_mm_ro_lin_par: tuple[float, float, float]
    voxel_mm_ro_lin_par: tuple[float, float, float]
    crop_bounds_from_native: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    acceleration_lin_par: tuple[int, int]
    residue_lin_par: tuple[int, int]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-native representation of the case.

        Returns:
            Geometry, crop, acceleration, and residue values using JSON lists.
        """

        return {
            "case_id": self.case_id,
            "matrix_ro_lin_par": list(self.matrix_ro_lin_par),
            "fov_mm_ro_lin_par": list(self.fov_mm_ro_lin_par),
            "voxel_mm_ro_lin_par": list(self.voxel_mm_ro_lin_par),
            "crop_bounds_from_native": [list(bounds) for bounds in self.crop_bounds_from_native],
            "acceleration_lin_par": list(self.acceleration_lin_par),
            "residue_lin_par": list(self.residue_lin_par),
        }


def case_definitions() -> dict[str, GreCase]:
    """Return the three immutable GRE geometry and sampling definitions.

    Returns:
        Mapping from stable case identifiers to validated case definitions.
    """

    native_crop = ((0, 250), (0, 250), (0, 72))
    low_crop = ((0, 250), LOW_RESOLUTION_LIN_BOUNDS, (0, 72))
    return {
        "native_r3x1": GreCase(
            "native_r3x1",
            NATIVE_MATRIX,
            NATIVE_FOV_MM,
            NATIVE_VOXEL_MM,
            native_crop,
            (3, 1),
            (2, 0),
        ),
        "native_r3x2": GreCase(
            "native_r3x2",
            NATIVE_MATRIX,
            NATIVE_FOV_MM,
            NATIVE_VOXEL_MM,
            native_crop,
            (3, 2),
            (2, 0),
        ),
        "lin_low_resolution_r3x2": GreCase(
            "lin_low_resolution_r3x2",
            LOW_RESOLUTION_MATRIX,
            NATIVE_FOV_MM,
            (
                NATIVE_FOV_MM[0] / LOW_RESOLUTION_MATRIX[0],
                NATIVE_FOV_MM[1] / LOW_RESOLUTION_MATRIX[1],
                NATIVE_FOV_MM[2] / LOW_RESOLUTION_MATRIX[2],
            ),
            low_crop,
            (3, 2),
            (2, 0),
        ),
    }


def validate_geometry_contract() -> dict[str, Any]:
    """Validate exact source, native, and low-resolution crop geometry.

    Returns:
        JSON-native source, native, and per-case geometry metadata.

    Raises:
        AssertionError: If a centered crop or voxel-size invariant changes.
    """

    if center_crop_bounds(256, 250) != (3, 253):
        raise AssertionError("The 256-to-250 centered crop contract changed.")
    if center_crop_bounds(250, 148) != LOW_RESOLUTION_LIN_BOUNDS:
        raise AssertionError("The 250-to-148 LIN crop contract changed.")
    cases = case_definitions()
    if tuple(NATIVE_FOV_MM[i] / NATIVE_MATRIX[i] for i in range(3)) != NATIVE_VOXEL_MM:
        raise AssertionError("Native GRE voxel dimensions are inconsistent.")
    return {
        "source_matrix_ro_lin_par": list(SOURCE_MATRIX),
        "source_to_native_crop_bounds": [list(bounds) for bounds in SOURCE_TO_NATIVE_BOUNDS],
        "native_matrix_ro_lin_par": list(NATIVE_MATRIX),
        "native_fov_mm_ro_lin_par": list(NATIVE_FOV_MM),
        "native_voxel_mm_ro_lin_par": list(NATIVE_VOXEL_MM),
        "cases": {key: value.to_json() for key, value in cases.items()},
    }


def build_case_mask(case: GreCase) -> tuple[np.ndarray, dict[str, Any]]:
    """Build and validate a case's pure Cartesian image-lattice mask.

    Args:
        case: Target geometry, acceleration, and lattice residue.

    Returns:
        Boolean LIN/PAR mask and canonical validation metadata.
    """

    mask, metadata = pure_cartesian_image_lattice_mask(
        case.matrix_ro_lin_par[1:],
        acceleration_lin_par=case.acceleration_lin_par,
        residue_lin_par=case.residue_lin_par,
    )
    validate_pure_cartesian_image_lattice(mask, metadata)
    return mask, metadata


def expected_mask_records() -> dict[str, dict[str, Any]]:
    """Return exact mask metadata for every GRE case.

    Returns:
        Mapping from case identifier to count, coordinates, and logical hash.
    """

    return {case_id: build_case_mask(case)[1] for case_id, case in case_definitions().items()}


def validate_echo_counters(
    lines: Sequence[int],
    partitions: Sequence[int],
    echoes: Sequence[int],
    *,
    matrix_lin_par: tuple[int, int],
    echo_times_s: Sequence[float],
) -> list[dict[str, Any]]:
    """Validate duplicate-free fully sampled counters and bind echo times.

    Args:
        lines: TWIX LIN counters for all image acquisitions.
        partitions: TWIX PAR counters aligned with ``lines``.
        echoes: TWIX Eco counters aligned with ``lines``.
        matrix_lin_par: Required fully sampled logical PE matrix.
        echo_times_s: Positive echo times ordered by Eco counter.

    Returns:
        One acquisition-count and TE record per validated echo.

    Raises:
        ValueError: If counters, coverage, duplicates, or echo times are invalid.
    """

    lin = np.asarray(lines, dtype=np.int64)
    par = np.asarray(partitions, dtype=np.int64)
    eco = np.asarray(echoes, dtype=np.int64)
    times = tuple(float(value) for value in echo_times_s)
    if lin.shape != par.shape or lin.shape != eco.shape or len(times) < 1:
        raise ValueError("LIN, PAR, Eco counters and echo-time metadata are incompatible.")
    records = []
    expected_coordinates = int(matrix_lin_par[0] * matrix_lin_par[1])
    for echo_index, te_s in enumerate(times):
        selected = eco == echo_index
        coordinates = np.stack((lin[selected], par[selected]), axis=1)
        unique = np.unique(coordinates, axis=0)
        if selected.sum() != expected_coordinates or unique.shape[0] != expected_coordinates:
            raise ValueError(f"Echo {echo_index + 1} is not a duplicate-free fully sampled grid.")
        if set(unique[:, 0]) != set(range(matrix_lin_par[0])) or set(unique[:, 1]) != set(
            range(matrix_lin_par[1])
        ):
            raise ValueError(f"Echo {echo_index + 1} does not span the full LIN/PAR grid.")
        if not math.isfinite(te_s) or te_s <= 0:
            raise ValueError(f"Echo {echo_index + 1} has invalid TE {te_s}.")
        records.append(
            {
                "echo": echo_index + 1,
                "eco_counter": echo_index,
                "te_s": te_s,
                "acquisitions": int(selected.sum()),
            }
        )
    if set(np.unique(eco)) != set(range(len(times))):
        raise ValueError("TWIX contains undeclared or missing Eco counters.")
    return records


def crop_source_to_native(values: np.ndarray) -> np.ndarray:
    """Center-crop source k-space from 256x256x72 to 250x250x72.

    Args:
        values: Array beginning with source RO/LIN/PAR dimensions.

    Returns:
        Logical k-space view cropped by ``[3:253, 3:253, 0:72]``.
    """

    array = np.asarray(values)
    if array.shape[:3] != SOURCE_MATRIX:
        raise ValueError(f"Expected source RO/LIN/PAR {SOURCE_MATRIX}, received {array.shape}.")
    result = array[3:253, 3:253, :]
    if result.shape[:3] != NATIVE_MATRIX:
        raise AssertionError("Source-to-native crop produced the wrong matrix.")
    return result


def crop_native_for_case(values: np.ndarray, case: GreCase) -> np.ndarray:
    """Apply the exact native-to-case logical k-space crop.

    Args:
        values: Array beginning with native RO/LIN/PAR dimensions.
        case: Target case containing half-open native crop bounds.

    Returns:
        Logical k-space view on the case-specific grid.
    """

    array = np.asarray(values)
    if array.shape[:3] != NATIVE_MATRIX:
        raise ValueError(f"Expected native RO/LIN/PAR {NATIVE_MATRIX}, received {array.shape}.")
    slices = tuple(slice(start, stop) for start, stop in case.crop_bounds_from_native)
    result = array[slices]
    if result.shape[:3] != case.matrix_ro_lin_par:
        raise AssertionError("Native-to-case crop produced the wrong matrix.")
    return result


def theoretical_psf(
    delta_ky_index: np.ndarray,
    delta_kz_index: np.ndarray,
    *,
    nlin: int,
    npar: int,
    yflip: int,
    zflip: int,
) -> np.ndarray:
    """Evaluate one sequence-derived GRE Wave PSF on a target PE grid.

    Args:
        delta_ky_index: Sequence-derived normalized ky offset at each readout sample.
        delta_kz_index: Sequence-derived normalized kz offset at each readout sample.
        nlin: Target logical LIN dimension.
        npar: Target logical PAR dimension.
        yflip: Validated ky trajectory sign.
        zflip: Validated kz trajectory sign.

    Returns:
        Unit-magnitude complex64 PSF in ``(RO_os, LIN, PAR)`` order.
    """

    delta_ky = np.asarray(delta_ky_index, dtype=np.float64).reshape(-1)
    delta_kz = np.asarray(delta_kz_index, dtype=np.float64).reshape(-1)
    if delta_ky.shape != delta_kz.shape or delta_ky.size != EXTENDED_READOUT:
        raise ValueError("GRE trajectory vectors must both contain 1000 samples.")
    if yflip not in {-1, 1} or zflip not in {-1, 1}:
        raise ValueError("Wave trajectory flips must be -1 or +1.")
    y_norm = (np.arange(nlin, dtype=np.float64) - nlin / 2.0) / nlin
    z_norm = (np.arange(npar, dtype=np.float64) - npar / 2.0) / npar
    phase = (
        -2.0 * np.pi * yflip * delta_ky[:, None, None] * y_norm[None, :, None]
        -2.0 * np.pi * zflip * delta_kz[:, None, None] * z_norm[None, None, :]
    )
    psf = np.exp(1j * phase).astype(np.complex64)
    if psf.shape != (EXTENDED_READOUT, nlin, npar) or not np.isfinite(psf).all():
        raise ValueError("The theoretical GRE PSF is invalid.")
    return psf


def coarse_candidate_settings() -> list[dict[str, Any]]:
    """Return the fixed 21-job coarse setting list for one case and echo.

    Returns:
        One FISTA control, five Wavelet, and fifteen explicit LLR settings.
    """

    settings: list[dict[str, Any]] = [
        {"method": "fista_lambda0", "lambda": 0.0, "block_size": None}
    ]
    settings.extend(
        {"method": "wavelet", "lambda": value, "block_size": None}
        for value in COARSE_LAMBDAS
    )
    settings.extend(
        {"method": "llr", "lambda": value, "block_size": block}
        for block in LLR_BLOCK_SIZES
        for value in COARSE_LAMBDAS
    )
    return settings


def lambda_label(value: float) -> str:
    """Format a finite nonnegative lambda for deterministic path names.

    Args:
        value: Candidate regularization strength.

    Returns:
        Stable compact scientific-notation label.
    """

    if not math.isfinite(value) or value < 0:
        raise ValueError("Lambda must be finite and nonnegative.")
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.8e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def candidate_name(setting: Mapping[str, Any]) -> str:
    """Return a stable candidate directory name from an explicit setting.

    Args:
        setting: Method, lambda, and explicit optional LLR block size.

    Returns:
        Filesystem-safe deterministic candidate name.
    """

    method = str(setting["method"])
    label = lambda_label(float(setting["lambda"]))
    block = setting.get("block_size")
    if method == "fista_lambda0" and float(setting["lambda"]) == 0 and block is None:
        return "fista_lambda-0"
    if method == "wavelet" and float(setting["lambda"]) > 0 and block is None:
        return f"wavelet_lambda-{label}"
    if method == "llr" and float(setting["lambda"]) > 0 and int(block) in LLR_BLOCK_SIZES:
        return f"llr_block-{int(block)}_lambda-{label}"
    raise ValueError(f"Invalid GRE sweep setting: {dict(setting)}")


def build_wave_command(
    bart: str | Path,
    setting: Mapping[str, Any],
    *,
    maps: str | Path,
    psf: str | Path,
    kspace: str | Path,
    output: str | Path,
) -> list[str]:
    """Build an exact GPU BART Wave command for one GRE candidate.

    Args:
        bart: BART executable path or command.
        setting: Explicit FISTA, Wavelet, or LLR configuration.
        maps: Case-matched CSM basename.
        psf: Echo- and grid-matched theoretical PSF basename.
        kspace: Pure-mask synthetic-Wave k-space basename.
        output: Destination image basename.

    Returns:
        Exact argument vector using GPU FISTA, 100 iterations, and 1e-6 tolerance.
    """

    method = str(setting["method"])
    value = float(setting["lambda"])
    block = setting.get("block_size")
    candidate_name(setting)
    if method == "fista_lambda0":
        options = ["-g", "-w", "-f", "-r", "0"]
    elif method == "wavelet":
        options = ["-g", "-w", "-f", "-r", f"{value:.12g}"]
    else:
        options = [
            "-g",
            "-l",
            "-v",
            "-b",
            str(int(block)),
            "-f",
            "-r",
            f"{value:.12g}",
        ]
    return [
        str(bart),
        "wave",
        *options,
        "-i",
        "100",
        "-t",
        "1e-6",
        str(bart_base(maps)),
        str(bart_base(psf)),
        str(bart_base(kspace)),
        str(bart_base(output)),
    ]


def refinement_points(lower: float, upper: float) -> tuple[float, float]:
    """Return two logarithmic trisection points inside a reviewed bracket.

    Args:
        lower: Positive lower lambda endpoint.
        upper: Positive upper lambda endpoint greater than ``lower``.

    Returns:
        Two ordered interior lambda values in logarithmic space.
    """

    if not 0 < lower < upper or not all(math.isfinite(v) for v in (lower, upper)):
        raise ValueError("A refinement bracket must contain two positive increasing values.")
    ratio = upper / lower
    return lower * ratio ** (1.0 / 3.0), lower * ratio ** (2.0 / 3.0)


def apply_sampling_mask(
    full_wave: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a pure PE mask and prove acquired equality and outside zeros.

    Args:
        full_wave: Full synthetic-Wave data in RO_os/LIN/PAR[/COIL] order.
        mask: Boolean pure image-lattice mask in LIN/PAR order.

    Returns:
        Masked complex64 data and exact equality, zero, and finite checks.
    """

    source = np.asarray(full_wave, dtype=np.complex64)
    logical_mask = np.asarray(mask)
    if source.ndim not in {3, 4} or logical_mask.dtype != np.bool_:
        raise ValueError("Wave data must be 3-D/4-D and the mask must be boolean.")
    if source.shape[1:3] != logical_mask.shape or not np.isfinite(source).all():
        raise ValueError("Wave data and pure sampling mask are incompatible.")
    output = np.zeros_like(source)
    output[:, logical_mask, ...] = source[:, logical_mask, ...]
    acquired_equal = np.array_equal(output[:, logical_mask, ...], source[:, logical_mask, ...])
    outside_zero = int(np.count_nonzero(output[:, ~logical_mask, ...])) == 0
    if not acquired_equal or not outside_zero:
        raise AssertionError("Pure-mask application failed its exact equality contract.")
    return output, {
        "acquired_samples_equal_full_wave_bitwise": acquired_equal,
        "unacquired_samples_are_exact_zero": outside_zero,
        "finite": bool(np.isfinite(output).all()),
    }


def bart_wave_restoration_factor(
    image_shape: Sequence[int],
    kspace_norm: float,
    encoding_shape: Sequence[int],
) -> complex:
    """Calculate the fixed BART-Wave scale and phase convention correction.

    Args:
        image_shape: Even GRE logical image shape in RO/LIN/PAR order.
        kspace_norm: Positive L2 norm removed internally by BART ``wave``.
        encoding_shape: BART Wave FFT grid in extended-RO/LIN/PAR order.

    Returns:
        Complex factor restoring unitary centered-FFT scale and phase.
    """

    logical = tuple(int(value) for value in image_shape)
    encoded = tuple(int(value) for value in encoding_shape)
    if len(logical) != 3 or len(encoded) != 3 or any(value <= 0 for value in logical + encoded):
        raise ValueError("BART Wave restoration shapes must contain three positive dimensions.")
    if any(value % 2 for value in logical + encoded):
        raise ValueError("The validated GRE BART Wave restoration requires even dimensions.")
    if encoded[0] < logical[0] or encoded[1:] != logical[1:]:
        raise ValueError("BART Wave encoding and image shapes have incompatible LIN/PAR geometry.")
    if not math.isfinite(kspace_norm) or kspace_norm <= 0:
        raise ValueError("BART input k-space norm must be positive and finite.")

    # BART wave divides its input by this norm and uses an unnormalized FFT on
    # the extended encoding grid. Its fftmod convention contributes a fixed
    # +/-i phase for the supported even GRE LIN sizes.
    amplitude = kspace_norm * math.sqrt(math.prod(encoded))
    phase = 1j * ((-1) ** (logical[1] // 2))
    return complex(amplitude * phase)


def restore_bart_normalization(
    image: np.ndarray,
    kspace_norm: float,
    encoding_shape: Sequence[int],
) -> np.ndarray:
    """Restore BART ``wave`` output to the synthetic-input convention.

    Args:
        image: Complex BART reconstruction on normalized input scale.
        kspace_norm: Recorded positive L2 norm of the BART k-space input.
        encoding_shape: BART Wave FFT grid in extended-RO/LIN/PAR order.

    Returns:
        Finite complex64 reconstruction on restored scale and global phase.
    """

    values = np.asarray(image, dtype=np.complex64)
    if values.ndim != 3:
        raise ValueError("BART Wave restoration requires a three-dimensional image.")
    factor = bart_wave_restoration_factor(values.shape, kspace_norm, encoding_shape)
    restored = values * np.complex64(factor)
    if not np.isfinite(restored).all():
        raise ValueError("Restored BART image contains non-finite values.")
    return restored


def fit_shared_echo1_scale(reference_echo1: np.ndarray, candidate_echo1: np.ndarray) -> float:
    """Fit one positive echo-1 scale for optional shared convention correction.

    Args:
        reference_echo1: Complex echo-1 direct-FFT reference.
        candidate_echo1: Complex echo-1 control reconstruction.

    Returns:
        Nonnegative least-squares scalar to apply unchanged to both echoes.
    """

    reference = np.asarray(reference_echo1, dtype=np.complex64)
    candidate = np.asarray(candidate_echo1, dtype=np.complex64)
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Shared scaling inputs must be matching finite complex arrays.")
    denominator = float(np.vdot(candidate, candidate).real)
    if denominator <= 0:
        raise ValueError("Shared scaling candidate has zero energy.")
    return max(0.0, float(np.vdot(candidate, reference).real / denominator))


def circular_phase_metrics(
    reference: np.ndarray, candidate: np.ndarray, support: np.ndarray
) -> dict[str, float]:
    """Compute raw circular phase-error summaries in a fixed support.

    Args:
        reference: Complex reference image.
        candidate: Complex candidate image on the same grid and scale.
        support: Fixed reference-derived brain-and-signal support.

    Returns:
        Circular mean, dispersion, and robust absolute-error summaries.
    """

    ref = np.asarray(reference, dtype=np.complex64)
    cand = np.asarray(candidate, dtype=np.complex64)
    selected = np.asarray(support, dtype=bool)
    if ref.shape != cand.shape or ref.shape != selected.shape or not np.any(selected):
        raise ValueError("Phase metric arrays and nonempty support must match.")
    error = np.angle(cand[selected] * np.conj(ref[selected])).astype(np.float64)
    mean_vector = np.mean(np.exp(1j * error))
    absolute = np.abs(error)
    return {
        "circular_mean_error_rad": float(np.angle(mean_vector)),
        "circular_dispersion": float(1.0 - abs(mean_vector)),
        "median_absolute_error_rad": float(np.median(absolute)),
        "p95_absolute_error_rad": float(np.percentile(absolute, 95)),
    }


def inter_echo_metrics(
    reference_echo1: np.ndarray,
    reference_echo2: np.ndarray,
    candidate_echo1: np.ndarray,
    candidate_echo2: np.ndarray,
    support: np.ndarray,
    *,
    delta_te_s: float,
) -> dict[str, float]:
    """Measure magnitude-ratio and wrapped/unwrapped delta-B0 preservation.

    Args:
        reference_echo1: Complex direct-FFT reference for echo 1.
        reference_echo2: Complex direct-FFT reference for echo 2.
        candidate_echo1: Complex reconstruction for echo 1.
        candidate_echo2: Complex reconstruction for echo 2.
        support: Fixed reference-derived brain-and-signal support.
        delta_te_s: Validated positive echo-time separation in seconds.

    Returns:
        Ratio bias/spread and wrapped/unwrapped field-consistency metrics.
    """

    from skimage.restoration import unwrap_phase

    arrays = [
        np.asarray(value, dtype=np.complex64)
        for value in (reference_echo1, reference_echo2, candidate_echo1, candidate_echo2)
    ]
    selected = np.asarray(support, dtype=bool)
    if any(value.shape != selected.shape for value in arrays) or not np.any(selected):
        raise ValueError("Inter-echo arrays and nonempty support must match.")
    if not math.isfinite(delta_te_s) or delta_te_s <= 0:
        raise ValueError("Delta TE must be positive and finite.")
    reference1, reference2, candidate1, candidate2 = arrays
    r1, r2, c1, c2 = (value[selected] for value in arrays)
    floor = np.finfo(np.float32).eps
    ref_ratio = np.abs(r2) / np.maximum(np.abs(r1), floor)
    cand_ratio = np.abs(c2) / np.maximum(np.abs(c1), floor)
    ratio_error = cand_ratio - ref_ratio
    ref_phase_volume = np.angle(reference2 * np.conj(reference1))
    cand_phase_volume = np.angle(candidate2 * np.conj(candidate1))
    ref_phase = ref_phase_volume[selected]
    cand_phase = cand_phase_volume[selected]
    phase_error = np.angle(np.exp(1j * (cand_phase - ref_phase)))
    b0_error = phase_error / (2.0 * np.pi * delta_te_s)
    ref_std = float(np.std(ref_phase))
    cand_std = float(np.std(cand_phase))
    correlation = (
        float(np.corrcoef(ref_phase, cand_phase)[0, 1])
        if ref_phase.size > 1 and ref_std > 0 and cand_std > 0
        else math.nan
    )
    ref_unwrapped = unwrap_phase(np.ma.array(ref_phase_volume, mask=~selected))
    cand_unwrapped = unwrap_phase(np.ma.array(cand_phase_volume, mask=~selected))
    unwrapped_ref_hz = np.asarray(ref_unwrapped[selected], dtype=np.float64) / (
        2.0 * np.pi * delta_te_s
    )
    unwrapped_cand_hz = np.asarray(cand_unwrapped[selected], dtype=np.float64) / (
        2.0 * np.pi * delta_te_s
    )
    unwrapped_error = unwrapped_cand_hz - unwrapped_ref_hz
    unwrapped_reference_norm = float(np.linalg.norm(unwrapped_ref_hz))
    unwrapped_correlation = (
        float(np.corrcoef(unwrapped_ref_hz, unwrapped_cand_hz)[0, 1])
        if unwrapped_ref_hz.size > 1
        and float(np.std(unwrapped_ref_hz)) > 0
        and float(np.std(unwrapped_cand_hz)) > 0
        else math.nan
    )
    return {
        "magnitude_ratio_bias": float(np.median(ratio_error)),
        "magnitude_ratio_mad": float(np.median(np.abs(ratio_error - np.median(ratio_error)))),
        "wrapped_delta_b0_bias_hz": float(np.median(b0_error)),
        "wrapped_delta_b0_mae_hz": float(np.mean(np.abs(b0_error))),
        "wrapped_phase_difference_correlation": correlation,
        "unwrapped_delta_b0_bias_hz": float(np.median(unwrapped_error)),
        "unwrapped_delta_b0_mae_hz": float(np.mean(np.abs(unwrapped_error))),
        "unwrapped_delta_b0_nrmse": (
            float(np.linalg.norm(unwrapped_error) / unwrapped_reference_norm)
            if unwrapped_reference_norm > 0
            else math.nan
        ),
        "unwrapped_delta_b0_correlation": unwrapped_correlation,
        "unwrap_nonfinite_fraction": float(np.mean(~np.isfinite(unwrapped_error))),
    }


def json_sha256(payload: Any) -> str:
    """Hash one JSON-compatible value using canonical serialization.

    Args:
        payload: JSON-compatible value whose logical identity is required.

    Returns:
        Lowercase canonical JSON SHA-256 digest.
    """

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def completed_manifest_reusable(
    manifest: Mapping[str, Any], signature_sha256: str, output_sha256: str | None
) -> bool:
    """Return whether a completed job exactly matches its signature and output hash.

    Args:
        manifest: Existing reconstruction job manifest.
        signature_sha256: Expected complete command/input signature.
        output_sha256: Current output payload hash, or ``None`` when missing.

    Returns:
        True only when status, signature, and output hash all match.
    """

    if manifest.get("status") != "complete" or manifest.get("signature_sha256") != signature_sha256:
        return False
    recorded = manifest.get("output", {}).get("payload_sha256")
    return isinstance(recorded, str) and output_sha256 == recorded


def validate_config_document(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable local GRE sweep configuration fields.

    Args:
        config: Parsed ignored local configuration document.

    Returns:
        Resolved output identity, geometry, masks, and exact coarse job count.
    """

    if config.get("format_version") != 1 or config.get("workflow") != WORKFLOW_NAME:
        raise ValueError("Unsupported GRE sweep configuration schema.")
    geometry = config.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("geometry must be an object.")
    expected = {
        "source_matrix_ro_lin_par": list(SOURCE_MATRIX),
        "native_matrix_ro_lin_par": list(NATIVE_MATRIX),
        "fov_mm_ro_lin_par": list(NATIVE_FOV_MM),
        "extended_wave_readout": EXTENDED_READOUT,
        "low_resolution_matrix_ro_lin_par": list(LOW_RESOLUTION_MATRIX),
    }
    for key, value in expected.items():
        if geometry.get(key) != value:
            raise ValueError(f"geometry.{key} must equal {value!r}.")
    if config.get("runtime", {}).get("backend") != "gpu":
        raise ValueError("The approved GRE sweep backend is fixed to GPU.")
    sweep = config.get("sweep", {})
    if tuple(float(v) for v in sweep.get("wavelet_lambdas", ())) != COARSE_LAMBDAS:
        raise ValueError("The Wavelet lambda grid differs from the approved contract.")
    if tuple(float(v) for v in sweep.get("llr_lambdas", ())) != COARSE_LAMBDAS:
        raise ValueError("The LLR lambda grid differs from the approved contract.")
    if tuple(int(v) for v in sweep.get("llr_blocks", ())) != LLR_BLOCK_SIZES:
        raise ValueError("The LLR block list differs from the approved contract.")
    if config.get("sampling", {}).get("mask_kind") != PURE_CARTESIAN_IMAGE_LATTICE:
        raise ValueError("Historical ACS-union masks are forbidden.")
    if config.get("sampling", {}).get("residue_lin_par") != [2, 0]:
        raise ValueError("The approved GRE LIN/PAR sampling residue is [2, 0].")
    compression = config.get("coil_compression", {})
    if (
        compression.get("physical_coils") != 44
        or compression.get("virtual_coils") != VIRTUAL_COILS
        or compression.get("covariance") != "trace_balanced_across_two_echoes"
        or int(compression.get("partition_chunk", 0)) < 1
        or int(compression.get("readout_step", 0)) < 1
    ):
        raise ValueError("The shared 44-to-12 two-echo coil-compression contract changed.")
    csm = config.get("csm", {})
    if (
        csm.get("calibration_echo") != 1
        or csm.get("calibration_size_ro_lin_par") != [250, 32, 32]
        or csm.get("ecalib_maps") != 1
        or not math.isclose(float(csm.get("ecalib_crop", math.nan)), 0.6)
        or csm.get("shared_across_echoes") is not True
    ):
        raise ValueError("The approved shared echo-1 CSM calibration contract changed.")
    brain = config.get("brain_mask", {})
    if (
        brain.get("source_case") != "native_r3x1"
        or brain.get("source_echo") != 1
        or not math.isclose(float(brain.get("fractional_intensity_threshold", math.nan)), 0.3)
        or not math.isclose(float(brain.get("vertical_gradient", math.nan)), 0.0)
        or brain.get("robust_center") is not True
        or brain.get("dilation_voxels") != 0
    ):
        raise ValueError("The approved fixed native echo-1 BET candidate contract changed.")
    if sweep.get("iterations") != 100 or not math.isclose(
        float(sweep.get("tolerance", math.nan)), 1e-6, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("The approved BART iteration/tolerance contract changed.")
    if int(config.get("runtime", {}).get("fft_workers", 0)) < 1:
        raise ValueError("runtime.fft_workers must be positive.")
    output_parent = Path(str(config.get("output_parent", ""))).expanduser()
    run_name = str(config.get("run_name", ""))
    if (
        not output_parent.is_absolute()
        or not output_parent.is_dir()
        or not run_name
        or Path(run_name).name != run_name
    ):
        raise ValueError("The approved output parent and simple run name are required.")
    return {
        "output_parent": str(output_parent),
        "run_name": run_name,
        "run_root": str(output_parent / run_name),
        "geometry": validate_geometry_contract(),
        "masks": expected_mask_records(),
        "coarse_jobs_per_group": len(coarse_candidate_settings()),
        "coarse_job_count": len(CASE_IDS) * len(ECHO_IDS) * len(coarse_candidate_settings()),
    }
