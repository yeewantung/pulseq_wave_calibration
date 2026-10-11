"""Apply optional, provenance-bound ANTs N4 correction to MPRAGE magnitude."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from .bart_io import sha256_file
from .nifti_collection import create_whole_head_mask

N4_MANIFEST_NAME = "n4_manifest.json"
N4_MANIFEST_STATUS = "mprage_n4_bias_correction_complete"


@dataclass(frozen=True)
class N4Parameters:
    """Define the fixed, reviewable ANTs N4 fitting contract.

    Attributes:
        shrink_factor: Integer spatial downsampling used during estimation.
        bspline_spacing_mm: Initial isotropic B-spline spacing in millimetres.
        convergence_iterations: Ordered multi-resolution iteration counts.
        convergence_threshold: Early-stopping coefficient-of-variation limit.
    """

    shrink_factor: int = 4
    bspline_spacing_mm: float = 180.0
    convergence_iterations: tuple[int, ...] = (50, 50, 50, 50)
    convergence_threshold: float = 0.0


def apply_mprage_n4_bias_correction(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    executable: str | Path = "N4BiasFieldCorrection",
    parameters: N4Parameters | None = None,
) -> dict[str, Any]:
    """Create or strictly reuse an N4-corrected MPRAGE magnitude derivative.

    Args:
        source_directory: Canonical branch-local magnitude/phase NIfTI folder.
        output_directory: Dedicated derivative folder outside the canonical
            reconstruction NIfTI tree.
        executable: ANTs ``N4BiasFieldCorrection`` executable name or path.
        parameters: Optional explicit N4 fitting parameters.

    Returns:
        Installed, fully validated N4 derivative manifest.

    Raises:
        FileNotFoundError: If sources, sidecars, or the ANTs executable are absent.
        FileExistsError: If an unowned or incomplete derivative directory exists.
        RuntimeError: If ANTs returns a failure status.
        ValueError: If geometry, values, parameters, hashes, or N4 outputs are invalid.

    Side Effects:
        Runs ANTs only when no complete matching derivative exists. Writes one
        corrected magnitude, one bias field, JSON sidecars, logs, and a manifest
        below ``output_directory``. The temporary estimation mask is deleted.
    """

    source = Path(source_directory).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    config = parameters or N4Parameters()
    _validate_parameters(config)
    magnitude, magnitude_sidecar, phase_records = _discover_mprage_sources(source)
    source_record = {
        "directory": str(source),
        "magnitude": _file_record(magnitude),
        "magnitude_sidecar": _file_record(magnitude_sidecar),
        "phase_files_unchanged": phase_records,
    }
    request = {
        "source": source_record,
        "parameters": _parameter_record(config),
        "policy": {
            "default_enabled": False,
            "magnitude_only": True,
            "original_niftis_modified": False,
            "phase_modified": False,
            "temporary_estimation_mask_retained": False,
        },
    }
    manifest_path = destination / "metadata" / N4_MANIFEST_NAME
    if manifest_path.is_file():
        return _validate_existing_derivative(destination, request)
    if destination.exists():
        raise FileExistsError(
            f"N4 derivative directory exists without a complete manifest: {destination}"
        )

    image, source_data = _load_magnitude(magnitude)
    mask_image, mask_details = create_whole_head_mask(image)
    mask_data = np.asarray(mask_image.dataobj, dtype=bool)
    tool = _tool_identity(executable)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        images = staging / "images"
        metadata = staging / "metadata"
        images.mkdir()
        metadata.mkdir()
        source_stem = magnitude.name.removesuffix(".nii.gz")
        corrected = images / f"{source_stem}_desc-N4BiasCorrected.nii.gz"
        bias = images / f"{source_stem}_desc-N4BiasField.nii.gz"
        with tempfile.TemporaryDirectory(prefix="n4-work-", dir=staging) as work:
            mask_path = Path(work) / "estimation_mask.nii.gz"
            nib.save(mask_image, str(mask_path))
            command = _n4_command(
                tool["path"], magnitude, corrected, bias, mask_path, config
            )
            completed = _execute_n4(command)
        (metadata / "n4_stdout.log").write_text(
            completed.stdout or "", encoding="utf-8"
        )
        (metadata / "n4_stderr.log").write_text(
            completed.stderr or "", encoding="utf-8"
        )
        validation = _validate_n4_outputs(
            image, source_data, mask_data, corrected, bias
        )
        source_metadata = _load_json(magnitude_sidecar)
        shared_metadata = {
            "N4BiasCorrectionApplied": True,
            "N4BiasCorrectionDefaultEnabled": False,
            "N4SourceMagnitude": str(magnitude),
            "N4SourceMagnitudeSHA256": source_record["magnitude"]["sha256"],
            "N4OriginalNIfTIsModified": False,
            "N4PhaseModified": False,
            "N4Parameters": _parameter_record(config),
            "N4Tool": tool,
            "N4TemporaryEstimationMaskRetained": False,
        }
        _write_json(
            corrected.with_name(corrected.name.removesuffix(".nii.gz") + ".json"),
            {
                **source_metadata,
                **shared_metadata,
                "N4DerivativeType": "bias-corrected magnitude",
            },
        )
        _write_json(
            bias.with_name(bias.name.removesuffix(".nii.gz") + ".json"),
            {
                **shared_metadata,
                "N4DerivativeType": "multiplicative bias field",
                "N4BiasFieldApplication": (
                    "corrected_magnitude = source_magnitude / bias_field "
                    "inside the estimation mask"
                ),
            },
        )
        output_paths = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != N4_MANIFEST_NAME
        )
        manifest = {
            "format_version": 1,
            "status": N4_MANIFEST_STATUS,
            "request": request,
            "tool": tool,
            "command": {
                "executable": tool["path"],
                "arguments_with_path_placeholders": _command_template(config),
                "shell_used": False,
            },
            "estimation_mask": {
                **mask_details,
                "logical_sha256": _logical_array_hash(mask_data.astype(np.uint8)),
                "retained": False,
                "purpose": "temporary N4 histogram-estimation mask only",
            },
            "validation": validation,
            "outputs": [
                _relative_file_record(path, staging) for path in output_paths
            ],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(metadata / N4_MANIFEST_NAME, manifest)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _validate_existing_derivative(destination, request)


def _validate_parameters(parameters: N4Parameters) -> None:
    """Validate one N4 parameter set before any external execution.

    Args:
        parameters: Proposed fitting parameters.

    Returns:
        None.

    Raises:
        ValueError: If a parameter is non-finite, non-positive, or malformed.
    """

    if parameters.shrink_factor < 1:
        raise ValueError("N4 shrink factor must be a positive integer.")
    if not np.isfinite(parameters.bspline_spacing_mm) or parameters.bspline_spacing_mm <= 0:
        raise ValueError("N4 B-spline spacing must be positive and finite.")
    if not parameters.convergence_iterations or any(
        int(value) != value or value < 1 for value in parameters.convergence_iterations
    ):
        raise ValueError("N4 convergence iterations must be positive integers.")
    if (
        not np.isfinite(parameters.convergence_threshold)
        or parameters.convergence_threshold < 0
    ):
        raise ValueError("N4 convergence threshold must be finite and nonnegative.")


def _discover_mprage_sources(
    directory: Path,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Resolve one canonical MPRAGE magnitude and its untouched phase pair.

    Args:
        directory: Branch-local canonical NIfTI directory.

    Returns:
        Magnitude path, magnitude sidecar, and phase file identity records.

    Raises:
        FileNotFoundError: If the directory or required sidecars are absent.
        ValueError: If magnitude or phase discovery is ambiguous.
    """

    if not directory.is_dir():
        raise FileNotFoundError(directory)
    magnitudes = sorted(directory.rglob("*_part-mag_*.nii.gz"))
    phases = sorted(directory.rglob("*_part-phase_*.nii.gz"))
    if len(magnitudes) != 1 or len(phases) != 1:
        raise ValueError(
            "N4 requires exactly one canonical MPRAGE magnitude and one phase NIfTI; "
            f"found {len(magnitudes)} magnitude and {len(phases)} phase files."
        )
    magnitude_sidecar = _sidecar_path(magnitudes[0])
    phase_sidecar = _sidecar_path(phases[0])
    for path in (magnitude_sidecar, phase_sidecar):
        if not path.is_file():
            raise FileNotFoundError(path)
    return magnitudes[0], magnitude_sidecar, [
        _file_record(phases[0]),
        _file_record(phase_sidecar),
    ]


def _load_magnitude(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load and validate one canonical finite nonnegative magnitude NIfTI.

    Args:
        path: Magnitude NIfTI path.

    Returns:
        Loaded image and float32 voxel array.

    Raises:
        ValueError: If dimensionality, orientation, geometry, or values fail.
    """

    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    if data.ndim != 3 or np.iscomplexobj(data) or not np.isfinite(data).all():
        raise ValueError("N4 source magnitude must be one finite real 3D image.")
    if np.any(data < 0) or not np.any(data > 0):
        raise ValueError("N4 source magnitude must be nonnegative with positive signal.")
    if tuple(nib.aff2axcodes(image.affine)) != ("R", "A", "S"):
        raise ValueError("N4 source magnitude must use canonical RAS orientation.")
    return image, np.asarray(data, dtype=np.float32)


def _tool_identity(executable: str | Path) -> dict[str, Any]:
    """Resolve and hash the requested ANTs executable and capture its version.

    Args:
        executable: Executable name or explicit filesystem path.

    Returns:
        JSON-compatible executable identity.

    Raises:
        FileNotFoundError: If the executable cannot be resolved.
        RuntimeError: If its version query fails.
    """

    token = os.fspath(executable)
    found = shutil.which(token) if Path(token).name == token else token
    if found is None:
        raise FileNotFoundError(
            "N4BiasFieldCorrection is not on PATH; install ANTs or activate its environment."
        )
    path = Path(found).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not os.access(path, os.X_OK):
        raise ValueError(f"N4BiasFieldCorrection is not executable: {path}")
    version = subprocess.run(
        [str(path), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if version.returncode != 0:
        raise RuntimeError("N4BiasFieldCorrection --version failed.")
    version_text = (version.stdout or version.stderr).strip()
    if not version_text:
        raise RuntimeError("N4BiasFieldCorrection returned an empty version string.")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "version": version_text,
    }


def _n4_command(
    executable: str,
    source: Path,
    corrected: Path,
    bias: Path,
    mask: Path,
    parameters: N4Parameters,
) -> list[str]:
    """Build one shell-free ANTs N4 command.

    Args:
        executable: Resolved ANTs executable path.
        source: Input magnitude NIfTI.
        corrected: Corrected-magnitude output path.
        bias: Multiplicative bias-field output path.
        mask: Temporary binary estimation-mask path.
        parameters: Validated fitting parameters.

    Returns:
        Argument vector suitable for ``subprocess.run``.
    """

    iterations = "x".join(str(value) for value in parameters.convergence_iterations)
    return [
        executable,
        "-d",
        "3",
        "-i",
        str(source),
        "-x",
        str(mask),
        "-s",
        str(parameters.shrink_factor),
        "-b",
        f"[{parameters.bspline_spacing_mm:g}]",
        "-c",
        f"[{iterations},{parameters.convergence_threshold:g}]",
        "-o",
        f"[{corrected},{bias}]",
        "-v",
        "1",
    ]


def _execute_n4(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one shell-free N4 command and capture its diagnostic streams.

    Args:
        command: Validated ANTs argument vector.

    Returns:
        Successful completed process.

    Raises:
        RuntimeError: If ANTs exits unsuccessfully.
    """

    completed = subprocess.run(
        list(command), check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"N4BiasFieldCorrection failed: {message}")
    return completed


def _validate_n4_outputs(
    source_image: nib.Nifti1Image,
    source_data: np.ndarray,
    mask: np.ndarray,
    corrected_path: Path,
    bias_path: Path,
) -> dict[str, Any]:
    """Validate N4 geometry, finite values, and multiplicative identity.

    Args:
        source_image: Loaded source magnitude image.
        source_data: Source magnitude voxel array.
        mask: Boolean N4 estimation mask.
        corrected_path: ANTs corrected-magnitude output.
        bias_path: ANTs multiplicative bias-field output.

    Returns:
        JSON-compatible numerical validation diagnostics.

    Raises:
        FileNotFoundError: If ANTs omitted an output.
        ValueError: If output geometry, values, or identity are invalid.
    """

    for path in (corrected_path, bias_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    corrected_image = nib.load(str(corrected_path))
    bias_image = nib.load(str(bias_path))
    corrected = np.asarray(corrected_image.dataobj, dtype=np.float64)
    bias = np.asarray(bias_image.dataobj, dtype=np.float64)
    for label, image, data in (
        ("corrected magnitude", corrected_image, corrected),
        ("bias field", bias_image, bias),
    ):
        if data.shape != source_data.shape or not np.isfinite(data).all():
            raise ValueError(f"N4 {label} shape or finite-value validation failed.")
        if not np.allclose(image.affine, source_image.affine, rtol=0.0, atol=1e-6):
            raise ValueError(f"N4 {label} affine differs from the source.")
    if np.any(corrected < 0) or np.any(bias[mask] <= 0):
        raise ValueError("N4 outputs contain negative magnitude or nonpositive bias.")
    expected = np.asarray(source_data[mask], dtype=np.float64) / bias[mask]
    scale = max(float(np.max(np.abs(expected))), np.finfo(np.float64).tiny)
    maximum_residual = float(np.max(np.abs(corrected[mask] - expected)) / scale)
    if maximum_residual > 5.0e-4:
        raise ValueError("N4 corrected magnitude does not equal input divided by bias field.")
    return {
        "shape": list(source_data.shape),
        "affine_matches_source": True,
        "finite": True,
        "corrected_magnitude_nonnegative": True,
        "bias_field_positive_inside_mask": True,
        "maximum_relative_full_scale_identity_residual_inside_mask": maximum_residual,
    }


def _validate_existing_derivative(
    destination: Path, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exact source binding and every output hash for N4 reuse.

    Args:
        destination: Installed derivative directory.
        request: Current source and parameter contract.

    Returns:
        Validated existing manifest.

    Raises:
        FileNotFoundError: If the manifest or an output is absent.
        ValueError: If the request, status, confinement, size, or hash differs.
    """

    manifest_path = destination / "metadata" / N4_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("status") != N4_MANIFEST_STATUS:
        raise ValueError("N4 derivative manifest has an unexpected status.")
    if manifest.get("request") != dict(request):
        raise ValueError("N4 derivative source or parameter contract changed.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("N4 derivative manifest has no output records.")
    for record in outputs:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValueError("N4 derivative contains a malformed output record.")
        path = (destination / str(record["path"])).resolve()
        if destination not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"N4 derivative output size changed: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"N4 derivative output hash changed: {path}")
    return manifest


def _parameter_record(parameters: N4Parameters) -> dict[str, Any]:
    """Convert N4 parameters to a JSON-native record.

    Args:
        parameters: Validated parameter dataclass.

    Returns:
        JSON-compatible parameter mapping.
    """

    record = asdict(parameters)
    record["convergence_iterations"] = list(parameters.convergence_iterations)
    record["source"] = "ANTs human T1w quick-start contract"
    return record


def _command_template(parameters: N4Parameters) -> list[str]:
    """Return a path-independent command template for durable provenance.

    Args:
        parameters: Validated N4 parameters.

    Returns:
        ANTs argument vector with semantic path placeholders.
    """

    return _n4_command(
        "<N4BiasFieldCorrection>",
        Path("<source_magnitude>"),
        Path("<corrected_magnitude>"),
        Path("<bias_field>"),
        Path("<temporary_estimation_mask>"),
        parameters,
    )


def _logical_array_hash(values: np.ndarray) -> str:
    """Hash array dtype, shape, and C-order payload.

    Args:
        values: Numerical array.

    Returns:
        SHA-256 hexadecimal digest.
    """

    import hashlib

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sidecar_path(nifti: Path) -> Path:
    """Resolve the JSON sidecar for a compressed NIfTI path.

    Args:
        nifti: ``.nii.gz`` file.

    Returns:
        Adjacent ``.json`` path.
    """

    return nifti.with_name(nifti.name.removesuffix(".nii.gz") + ".json")


def _file_record(path: Path) -> dict[str, Any]:
    """Create one absolute path, size, and SHA-256 identity.

    Args:
        path: Existing ordinary file.

    Returns:
        JSON-compatible file identity.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _relative_file_record(path: Path, root: Path) -> dict[str, Any]:
    """Create one root-relative output identity.

    Args:
        path: Existing output file.
        root: Staging root containing the file.

    Returns:
        JSON-compatible relative identity.
    """

    record = _file_record(path)
    record["path"] = str(path.relative_to(root))
    return record


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object.

    Args:
        path: JSON file.

    Returns:
        Parsed mapping.

    Raises:
        ValueError: If the top-level value is not an object.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic JSON object.

    Args:
        path: Destination file.
        payload: JSON-compatible mapping.

    Returns:
        None.

    Side Effects:
        Writes ``path`` with stable indentation and key ordering.
    """

    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
