"""Shared contracts for standard-PCA reconstruction variants."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bart_io import cfl_record, sha256_file


DEFAULT_VIRTUAL_COILS = 24
RECONSTRUCTION_PROFILES = ("reg-full", "wavelet-only", "fista-only")


def validate_virtual_coils(
    virtual_coils: int, physical_coils: int | None = None
) -> int:
    """Validate a requested standard-PCA channel count.

    Args:
        virtual_coils: Requested positive compressed-coil count.
        physical_coils: Optional measured physical receive-coil count.

    Returns:
        The validated integer count.

    Raises:
        ValueError: If the count is non-positive or exceeds the physical count.
    """

    count = int(virtual_coils)
    if count <= 0:
        raise ValueError("Standard-PCA virtual-coil count must be positive.")
    if physical_coils is not None and count > int(physical_coils):
        raise ValueError(
            f"Requested {count} standard virtual coils, but the acquisition has "
            f"only {int(physical_coils)} physical receive coils."
        )
    return count


def standard_variant_name(virtual_coils: int) -> str:
    """Return the immutable directory name for a standard-PCA count.

    Args:
        virtual_coils: Positive standard-PCA channel count.

    Returns:
        A directory name such as ``vcc24``.
    """

    return f"vcc{validate_virtual_coils(virtual_coils)}"


def standard_variant_root(output_root: str | Path, virtual_coils: int) -> Path:
    """Resolve a standard-PCA variant below a user-selected reconstruction root.

    Args:
        output_root: User-selected top-level reconstruction root.
        virtual_coils: Positive standard-PCA channel count.

    Returns:
        Absolute ``OUTPUT_ROOT/vccN`` path. The directory is not created.
    """

    return Path(output_root).expanduser().resolve() / standard_variant_name(
        virtual_coils
    )


def normalize_reconstruction_profile(profile: str) -> str:
    """Validate a standard reconstruction-profile identifier.

    Args:
        profile: One of ``reg-full``, ``wavelet-only``, or ``fista-only``.

    Returns:
        The normalized profile string.

    Raises:
        ValueError: If the profile is not supported.
    """

    value = str(profile).strip().lower()
    if value not in RECONSTRUCTION_PROFILES:
        raise ValueError(
            "Reconstruction profile must be one of: "
            + ", ".join(RECONSTRUCTION_PROFILES)
            + "."
        )
    return value


def profile_branches(profile: str) -> tuple[str, ...]:
    """Return the reconstruction branches selected by a profile.

    Args:
        profile: Valid standard reconstruction profile.

    Returns:
        Stable branch identifiers in execution order.
    """

    value = normalize_reconstruction_profile(profile)
    if value == "reg-full":
        return ("fista_r0", "wavelet")
    if value == "wavelet-only":
        return ("wavelet",)
    return ("fista_r0",)


def array_sha256(values: Any) -> str:
    """Hash an array's dtype, shape, and logical C-order values.

    Args:
        values: NumPy-compatible array or CPU-convertible tensor.

    Returns:
        Lowercase SHA-256 identity.
    """

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def write_pca_basis(path: str | Path, basis: Any) -> dict[str, Any]:
    """Persist and identify one standard-PCA compression basis.

    Args:
        path: Destination ``.npy`` path.
        basis: NumPy-compatible basis matrix or CPU-convertible tensor.

    Returns:
        JSON-native basis identity including file and logical-array hashes.

    Side Effects:
        Writes one NumPy array file at ``path``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(basis, "detach"):
        basis = basis.detach().cpu().numpy()
    values = np.asarray(basis)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Standard-PCA basis must be a finite two-dimensional array.")
    np.save(destination, values, allow_pickle=False)
    return {
        "path": destination.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "logical_sha256": array_sha256(values),
        "file_sha256": sha256_file(destination),
    }


def retained_energy_diagnostics(
    retained_energy: Sequence[float], virtual_coils: int
) -> dict[str, Any]:
    """Build fixed-count and selected-count PCA energy diagnostics.

    Args:
        retained_energy: Cumulative retained-energy vector by component count.
        virtual_coils: Selected positive compressed-coil count.

    Returns:
        Selected energy plus available diagnostics at counts 12, 20, and 24.
    """

    values = np.asarray(retained_energy, dtype=np.float64).reshape(-1)
    count = validate_virtual_coils(virtual_coils, values.size)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("PCA retained-energy diagnostics must be finite and nonnegative.")
    fixed = {
        str(candidate): float(values[candidate - 1])
        for candidate in (12, 20, 24)
        if candidate <= values.size
    }
    return {
        "selected": float(values[count - 1]),
        "cumulative_at_fixed_counts": fixed,
    }


def prepared_artifact_records(
    directory: str | Path,
    cfl_names: Sequence[str],
    file_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Hash prepared scientific outputs for strict resumable reuse.

    Args:
        directory: Directory containing the artifacts.
        cfl_names: BART CFL basenames relative to ``directory``.
        file_names: Ordinary files relative to ``directory``.

    Returns:
        JSON-native records keyed by relative artifact name.
    """

    root = Path(directory)
    records: dict[str, Any] = {}
    for name in cfl_names:
        records[str(name)] = cfl_record(root / name, include_hash=True)
    for name in file_names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Prepared artifact does not exist: {path}")
        records[str(name)] = {
            "path": str(name),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def validate_prepared_artifact_records(
    directory: str | Path, records: Mapping[str, Any]
) -> None:
    """Validate prepared outputs against their recorded hashes.

    Args:
        directory: Directory containing the recorded artifacts.
        records: Mapping returned by :func:`prepared_artifact_records`.

    Returns:
        None.

    Raises:
        ValueError: If a path, shape, size, or hash differs.
    """

    root = Path(directory)
    if not isinstance(records, Mapping) or not records:
        raise ValueError("Prepared manifest has no output artifact hashes.")
    for name, expected in records.items():
        if not isinstance(expected, Mapping):
            raise ValueError(f"Invalid prepared artifact record: {name}")
        if "payload_sha256" in expected:
            actual = cfl_record(root / name, include_hash=True)
            for key in ("shape", "dtype", "payload_bytes", "header_sha256", "payload_sha256"):
                if actual.get(key) != expected.get(key):
                    raise ValueError(f"Prepared CFL artifact changed: {root / name} ({key}).")
        else:
            path = root / name
            if (
                not path.is_file()
                or path.stat().st_size != int(expected.get("size_bytes", -1))
                or sha256_file(path) != expected.get("sha256")
            ):
                raise ValueError(f"Prepared file artifact changed: {path}.")


def validate_standard_pca_manifest(
    manifest: Mapping[str, Any], virtual_coils: int, *, variant_name: str | None = None
) -> Mapping[str, Any]:
    """Validate the count and basis identity in a standard preparation manifest.

    Args:
        manifest: Parsed MPRAGE or GRE normal preparation manifest.
        virtual_coils: Expected standard-PCA channel count.
        variant_name: Optional expected ``vccN`` directory label.

    Returns:
        The validated coil-compression mapping.

    Raises:
        ValueError: If count, variant, energy, or basis identity is missing or
            inconsistent.
    """

    count = validate_virtual_coils(virtual_coils)
    compression = manifest.get("coil_compression")
    if not isinstance(compression, Mapping):
        raise ValueError("Prepared manifest has no coil-compression contract.")
    if int(compression.get("virtual_coils", 0)) != count:
        raise ValueError(
            f"Prepared manifest virtual-coil count does not match vcc{count}."
        )
    physical = int(compression.get("physical_coils", 0))
    validate_virtual_coils(count, physical)
    energy = float(compression.get("retained_energy", math.nan))
    if not math.isfinite(energy) or not 0.0 <= energy <= 1.0 + 1e-6:
        raise ValueError("Prepared manifest has invalid retained PCA energy.")
    basis = compression.get("basis")
    if not isinstance(basis, Mapping) or not all(
        isinstance(basis.get(key), str) and basis.get(key)
        for key in ("path", "logical_sha256", "file_sha256")
    ):
        raise ValueError("Prepared manifest has no immutable PCA basis identity.")
    expected_variant = variant_name or standard_variant_name(count)
    if manifest.get("standard_pca_variant") != expected_variant:
        raise ValueError(
            f"Prepared manifest variant does not match {expected_variant}."
        )
    return compression
