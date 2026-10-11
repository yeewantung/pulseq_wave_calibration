"""Estimate, apply, and record receive-noise prewhitening for TWIX data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bart_io import sha256_file

PREWHITENING_VERSION = 1
WHITENING_MATRIX_NAME = "noise_prewhitening_matrix.npy"
NOISE_COVARIANCE_NAME = "noise_covariance.npy"
EMPIRICAL_WHITE_MAX_CORRELATION = 0.02
EMPIRICAL_WHITE_DIAGONAL_CV = 0.02


@dataclass(frozen=True)
class NoisePrewhiteningPlan:
    """Hold one validated prewhitening decision and its numerical arrays.

    Attributes:
        record: JSON-compatible decision and acquisition provenance.
        matrix: Coil-last right-multiplication matrix, or ``None`` when no
            transform is required.
        covariance: Mean-diagonal-normalized noise covariance, or ``None``
            when no usable noise scan exists or whitening was disabled.
    """

    record: dict[str, Any]
    matrix: np.ndarray | None
    covariance: np.ndarray | None


def prepare_noise_prewhitening(
    twix_path: str | Path,
    physical_coils: int,
    *,
    enabled: bool = True,
    measurements: Sequence[Any] | None = None,
) -> NoisePrewhiteningPlan:
    """Inspect a TWIX file and prepare one physical-coil whitening transform.

    Args:
        twix_path: Siemens TWIX file containing the image measurement and,
            when available, a noise-adjust acquisition.
        physical_coils: Validated receive-channel count of image/refscan data.
        enabled: Whether automatic noise prewhitening was requested.
        measurements: Optional measurements from an earlier mapVBVD geometry
            inspection. Supplying them avoids a duplicate TWIX metadata pass.

    Returns:
        A plan whose matrix can be applied before PCA or ROVir projection.

    Raises:
        FileNotFoundError: If ``twix_path`` is absent.
        ValueError: If populated noise data are ambiguous, non-finite,
            undersampled, singular, or use a different coil count.

    Notes:
        Absence of ``MDH_NOISEADJSCAN`` is an explicit safe fallback. A
        malformed populated noise scan is not treated as absence.
    """

    path = Path(twix_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    coils = int(physical_coils)
    if coils < 1:
        raise ValueError("Physical coil count must be positive.")
    request = {
        "enabled": bool(enabled),
        "default": True,
        "disable_flags": ["-no-noise-prewhitening", "--no-noise-prewhitening"],
    }
    if not enabled:
        return NoisePrewhiteningPlan(
            record={
                "version": PREWHITENING_VERSION,
                "request": request,
                "status": "disabled_by_user",
                "applied": False,
                "physical_coils": coils,
                "fallback": False,
            },
            matrix=None,
            covariance=None,
        )

    if measurements is None:
        import mapvbvd

        root = mapvbvd.mapVBVD(str(path), quiet=True)
        parsed_measurements = list(root) if isinstance(root, (list, tuple)) else [root]
    else:
        parsed_measurements = list(measurements)
    if not parsed_measurements:
        raise ValueError("TWIX inspection returned no measurements.")
    image_candidates: list[tuple[int, int, Any]] = []
    noise_candidates: list[tuple[int, int, Any]] = []
    for index, measurement in enumerate(parsed_measurements):
        image = _stream(measurement, "image")
        noise = _stream(measurement, "noise")
        image_acquisitions = int(getattr(image, "NAcq", 0)) if image is not None else 0
        noise_acquisitions = int(getattr(noise, "NAcq", 0)) if noise is not None else 0
        if image_acquisitions:
            image_candidates.append((image_acquisitions, index, image))
        if noise_acquisitions:
            noise_candidates.append((noise_acquisitions, index, noise))
    if not image_candidates:
        raise ValueError("No TWIX measurement contains a populated image stream.")
    _, image_index, image_stream = max(
        image_candidates, key=lambda item: (item[0], item[1])
    )
    correction = _raw_data_correction_record(parsed_measurements, image_index)
    base = {
        "version": PREWHITENING_VERSION,
        "request": request,
        "physical_coils": coils,
        "image_measurement_index": image_index,
        "noise_measurement_candidates": [
            {"measurement_index": index, "acquisitions": acquisitions}
            for acquisitions, index, _ in noise_candidates
        ],
        "scanner_side_whitening": {
            "explicitly_confirmed": False,
            "assessment": "no_reliable_explicit_twix_marker",
            "mdh_raw_data_correction": correction,
            "mdh_flag_interpreted_as_whitening": False,
        },
    }
    image_channels = _constant_channel_count(image_stream, "image")
    if image_channels != coils:
        raise ValueError(
            f"TWIX image stream has {image_channels} coils but {coils} were expected."
        )
    compatible = [
        candidate
        for candidate in noise_candidates
        if _constant_channel_count(candidate[2], "noise") == coils
    ]
    if not compatible:
        if noise_candidates:
            found = sorted(
                {_constant_channel_count(candidate[2], "noise") for candidate in noise_candidates}
            )
            raise ValueError(
                f"Noise-adjust scans use coil counts {found}, not the physical count {coils}."
            )
        return NoisePrewhiteningPlan(
            record={
                **base,
                "status": "not_applied_no_noise_scan",
                "applied": False,
                "fallback": True,
                "fallback_reason": "no MDH_NOISEADJSCAN acquisition",
            },
            matrix=None,
            covariance=None,
        )
    if len(compatible) != 1:
        indices = [candidate[1] for candidate in compatible]
        raise ValueError(
            f"Multiple compatible noise-adjust measurements are ambiguous: {indices}."
        )

    acquisitions, noise_index, noise_stream = compatible[0]
    values = np.asarray(noise_stream.unsorted())
    if values.ndim != 3 or values.shape[1] != coils:
        raise ValueError(
            "Noise-adjust payload must have unsorted [sample, coil, acquisition] "
            f"shape; found {values.shape}."
        )
    samples = np.transpose(values, (0, 2, 1)).reshape(-1, coils)
    if samples.shape[0] < 4 * coils:
        raise ValueError(
            f"Noise-adjust scan has only {samples.shape[0]} samples for {coils} coils."
        )
    samples = np.asarray(samples, dtype=np.complex128)
    if not np.isfinite(samples).all():
        raise ValueError("Noise-adjust scan contains non-finite samples.")
    samples -= samples.mean(axis=0, keepdims=True)
    covariance = samples.conj().T @ samples / float(samples.shape[0] - 1)
    covariance = 0.5 * (covariance + covariance.conj().T)
    diagonal = np.asarray(np.real(np.diag(covariance)), dtype=np.float64)
    mean_variance = float(diagonal.mean())
    if not np.isfinite(mean_variance) or mean_variance <= 0 or np.any(diagonal <= 0):
        raise ValueError("Noise covariance has invalid channel variances.")
    covariance /= mean_variance
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    maximum = float(eigenvalues[-1])
    minimum = float(eigenvalues[0])
    if minimum <= max(np.finfo(np.float64).eps * maximum, maximum * 1.0e-8):
        raise ValueError(
            "Noise covariance is singular or ill-conditioned; refusing regularized whitening."
        )
    std = np.sqrt(np.real(np.diag(covariance)))
    correlation = covariance / (std[:, None] * std[None, :])
    off_diagonal = correlation - np.diag(np.diag(correlation))
    maximum_correlation = float(np.max(np.abs(off_diagonal)))
    diagonal_cv = float(np.std(diagonal) / mean_variance)
    already_white = (
        maximum_correlation <= EMPIRICAL_WHITE_MAX_CORRELATION
        and diagonal_cv <= EMPIRICAL_WHITE_DIAGONAL_CV
    )
    covariance64 = np.asarray(covariance, dtype=np.complex64)
    noise_record = {
        "measurement_index": noise_index,
        "acquisitions": acquisitions,
        "samples_per_acquisition": int(values.shape[0]),
        "total_samples": int(samples.shape[0]),
        "coils": coils,
        "coil_order_validation": (
            "noise and image streams originate from the same TWIX file and "
            "have identical channel counts; pyMapVBVD does not expose a "
            "per-channel identity list, so independent order verification is unavailable"
        ),
        "sample_mean_removed_per_coil": True,
        "covariance_normalization": "divide by mean physical-coil variance",
        "mean_variance_before_normalization": mean_variance,
        "minimum_normalized_eigenvalue": minimum,
        "maximum_normalized_eigenvalue": maximum,
        "condition_number": maximum / minimum,
        "maximum_absolute_off_diagonal_correlation": maximum_correlation,
        "channel_variance_coefficient_of_variation": diagonal_cv,
    }
    empirical = {
        "assessment": "already_white" if already_white else "requires_prewhitening",
        "maximum_absolute_off_diagonal_correlation_threshold": (
            EMPIRICAL_WHITE_MAX_CORRELATION
        ),
        "channel_variance_coefficient_of_variation_threshold": (
            EMPIRICAL_WHITE_DIAGONAL_CV
        ),
    }
    if already_white:
        return NoisePrewhiteningPlan(
            record={
                **base,
                "status": "not_applied_empirically_already_white",
                "applied": False,
                "fallback": False,
                "noise_scan": noise_record,
                "empirical_whitening_assessment": empirical,
            },
            matrix=None,
            covariance=covariance64,
        )

    inverse_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    matrix = eigenvectors @ inverse_sqrt @ eigenvectors.conj().T
    matrix = np.asarray(matrix, dtype=np.complex64)
    whitened = matrix.conj().T @ covariance64 @ matrix
    residual = float(np.max(np.abs(whitened - np.eye(coils))))
    if not np.isfinite(matrix).all() or residual > 5.0e-5:
        raise ValueError("Computed noise prewhitening matrix failed identity validation.")
    return NoisePrewhiteningPlan(
        record={
            **base,
            "status": "applied",
            "applied": True,
            "fallback": False,
            "noise_scan": noise_record,
            "empirical_whitening_assessment": empirical,
            "application": {
                "convention": "coil-last right multiplication: data @ W",
                "covariance_validation": "W^H C W = I",
                "maximum_absolute_identity_residual": residual,
                "dwell_time_scaling": (
                    "not applied; global scale preserved by mean-variance normalization"
                ),
            },
        },
        matrix=matrix,
        covariance=covariance64,
    )


def apply_noise_prewhitening_coillast(data: Any, matrix: np.ndarray | None) -> Any:
    """Apply a whitening matrix to NumPy or PyTorch coil-last data.

    Args:
        data: Complex array or tensor with physical coils on the final axis.
        matrix: Square coil-last right-multiplication matrix, or ``None`` for
            an explicitly validated identity/no-op decision.

    Returns:
        Whitened data of the same array type and complex precision.

    Raises:
        TypeError: If ``data`` is neither NumPy nor PyTorch-compatible.
        ValueError: If dimensions, coil counts, or finite values are invalid.
    """

    if matrix is None:
        return data
    weights = np.asarray(matrix)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("Noise prewhitening matrix must be square.")
    if int(data.shape[-1]) != weights.shape[0]:
        raise ValueError("Data coil count does not match the prewhitening matrix.")
    try:
        import torch

        if isinstance(data, torch.Tensor):
            matrix_tensor = torch.as_tensor(
                weights, dtype=data.dtype, device=data.device
            )
            result = data @ matrix_tensor
            if not torch.isfinite(result).all():
                raise ValueError("Prewhitened data contain non-finite values.")
            return result
    except ImportError:
        pass
    values = np.asarray(data)
    if not np.iscomplexobj(values):
        raise TypeError("Noise prewhitening requires complex NumPy or PyTorch data.")
    result = values @ weights.astype(values.dtype, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Prewhitened data contain non-finite values.")
    return result


def combined_prewhitening_and_compression_basis(
    whitening_matrix: np.ndarray | None, compression_basis: np.ndarray
) -> np.ndarray:
    """Combine physical-coil whitening and PCA into one image-data transform.

    Args:
        whitening_matrix: Square physical-coil whitening matrix or ``None``.
        compression_basis: PCA basis estimated from already whitened ACS data.

    Returns:
        Complex64 physical-to-virtual coil transform.
    """

    basis = np.asarray(compression_basis)
    if basis.ndim != 2:
        raise ValueError("Coil-compression basis must be two-dimensional.")
    if whitening_matrix is None:
        return np.asarray(basis, dtype=np.complex64)
    whitening = np.asarray(whitening_matrix)
    if whitening.shape != (basis.shape[0], basis.shape[0]):
        raise ValueError("Whitening and coil-compression basis dimensions disagree.")
    return np.asarray(whitening @ basis, dtype=np.complex64)


def persist_noise_prewhitening(
    directory: str | Path, plan: NoisePrewhiteningPlan
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Persist numerical prewhitening evidence and return its manifest record.

    Args:
        directory: Existing BART-input directory.
        plan: Validated prewhitening plan.

    Returns:
        Updated JSON record and filenames to include in artifact hashing.

    Side Effects:
        Writes covariance and, when applied, transform arrays below
        ``directory``.
    """

    root = Path(directory)
    record = dict(plan.record)
    files: list[str] = []
    if plan.covariance is not None:
        path = root / NOISE_COVARIANCE_NAME
        np.save(path, np.asarray(plan.covariance), allow_pickle=False)
        record["noise_covariance"] = _array_file_record(path, plan.covariance)
        files.append(path.name)
    if plan.matrix is not None:
        path = root / WHITENING_MATRIX_NAME
        np.save(path, np.asarray(plan.matrix), allow_pickle=False)
        record["whitening_matrix"] = _array_file_record(path, plan.matrix)
        files.append(path.name)
    return record, tuple(files)


def prewhitening_request_matches(
    manifest: Mapping[str, Any], *, enabled: bool
) -> bool:
    """Check an existing manifest against one requested whitening mode.

    Args:
        manifest: Existing prepared-input manifest.
        enabled: Current automatic-prewhitening request.

    Returns:
        ``True`` for an exact recorded request or a historical manifest that
        predates the prewhitening field. Legacy artifacts remain unwhitened;
        this compatibility does not relabel them as a new default run.
    """

    if "noise_prewhitening" not in manifest:
        return True
    record = manifest.get("noise_prewhitening")
    if not isinstance(record, Mapping):
        return False
    request = record.get("request")
    return isinstance(request, Mapping) and request.get("enabled") is bool(enabled)


def load_recorded_whitening_matrix(
    input_directory: str | Path, manifest: Mapping[str, Any]
) -> np.ndarray | None:
    """Load and hash-validate the whitening matrix recorded by a manifest.

    Args:
        input_directory: Directory containing the manifest's numerical files.
        manifest: Prepared-input manifest with ``noise_prewhitening``.

    Returns:
        Recorded complex64 matrix, or ``None`` for a validated no-op status.

    Raises:
        ValueError: If whitening provenance or hashes are incomplete.
    """

    record = manifest.get("noise_prewhitening")
    if not isinstance(record, Mapping):
        raise ValueError("Source manifest predates noise-prewhitening provenance.")
    applied = record.get("applied")
    if applied is False:
        return None
    if applied is not True:
        raise ValueError("Noise-prewhitening manifest has an invalid applied state.")
    identity = record.get("whitening_matrix")
    if not isinstance(identity, Mapping) or not isinstance(identity.get("file"), str):
        raise ValueError("Applied prewhitening lacks a matrix identity.")
    path = Path(input_directory) / str(identity["file"])
    if not path.is_file() or sha256_file(path) != identity.get("sha256"):
        raise ValueError("Recorded noise-prewhitening matrix changed or is absent.")
    matrix = np.load(path, allow_pickle=False)
    if list(matrix.shape) != identity.get("shape"):
        raise ValueError("Recorded noise-prewhitening matrix shape changed.")
    return np.asarray(matrix, dtype=np.complex64)


def _stream(measurement: Any, name: str) -> Any | None:
    """Return one named mapVBVD stream.

    Args:
        measurement: Mapping- or attribute-style mapVBVD measurement.
        name: Stream key such as ``image``, ``refscan``, or ``noise``.

    Returns:
        The selected stream, or ``None`` when it is absent.
    """

    if hasattr(measurement, "get"):
        return measurement.get(name)
    return getattr(measurement, name, None)


def _constant_channel_count(stream: Any, label: str) -> int:
    """Validate one constant mapVBVD stream channel count.

    Args:
        stream: Populated mapVBVD stream with per-acquisition ``NCha`` values.
        label: Human-readable stream label for validation errors.

    Returns:
        Positive constant physical-channel count.

    Raises:
        ValueError: If channel metadata are absent, non-finite, non-integral,
            non-positive, or inconsistent across acquisitions.
    """

    raw = np.asarray(getattr(stream, "NCha", []), dtype=np.float64).reshape(-1)
    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError(f"{label} stream lacks a finite channel count.")
    rounded = np.rint(raw).astype(np.int64)
    if not np.allclose(raw, rounded) or len(set(rounded.tolist())) != 1:
        raise ValueError(f"{label} stream channel count changes across acquisitions.")
    count = int(rounded[0])
    if count < 1:
        raise ValueError(f"{label} stream channel count must be positive.")
    return count


def _raw_data_correction_record(
    measurements: list[Any], image_index: int
) -> dict[str, Any]:
    """Record Siemens raw-data-correction bits without calling them whitening.

    Args:
        measurements: Ordered mapVBVD measurements.
        image_index: Selected image-measurement index.

    Returns:
        JSON-compatible image/refscan flag counts and a semantic warning.
    """

    streams: dict[str, Any] = {}
    for label in ("image", "refscan"):
        stream = _stream(measurements[image_index], label)
        raw_flags = getattr(stream, "IsRawDataCorrect", None)
        flags = (
            np.asarray([], dtype=bool)
            if raw_flags is None
            else np.asarray(raw_flags, dtype=bool).reshape(-1)
        )
        streams[label] = {
            "acquisitions": int(flags.size),
            "flagged": int(np.count_nonzero(flags)),
            "all_flagged": bool(flags.size and flags.all()),
            "mixed": bool(flags.size and flags.any() and not flags.all()),
        }
    return {
        "measurement_index": image_index,
        "streams": streams,
        "semantic_note": (
            "MDH_RAWDATACORRECTION is recorded for audit only; pyMapVBVD does "
            "not implement that correction and the bit is not treated as proof "
            "of scanner-side noise whitening."
        ),
    }


def _array_file_record(path: Path, values: np.ndarray) -> dict[str, Any]:
    """Describe one persisted numerical array with logical and file hashes.

    Args:
        path: Existing NumPy file.
        values: Logical array written to ``path``.

    Returns:
        Filename, shape, dtype, byte count, logical hash, and file hash.
    """

    array = np.ascontiguousarray(values)
    logical = hashlib.sha256(array.view(np.uint8)).hexdigest()
    return {
        "file": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "logical_sha256": logical,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
