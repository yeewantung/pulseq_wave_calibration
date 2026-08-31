"""Memory-bounded BART CFL I/O used by retrospective reconstruction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def bart_base(path: str | Path) -> Path:
    """Remove a BART payload or header suffix from a path.

    Args:
        path: BART basename, ``.cfl`` path, or ``.hdr`` path.

    Returns:
        Normalized BART basename as a ``Path``.
    """
    path = Path(path)
    return path.with_suffix("") if path.suffix in {".cfl", ".hdr"} else path


def read_shape(path: str | Path) -> tuple[int, ...]:
    """Read a CFL header and validate its complex64 payload size.

    Args:
        path: BART basename or either member of its CFL/HDR pair.

    Returns:
        Positive BART dimensions in stored order.
    """
    base = bart_base(path)
    header = base.with_suffix(".hdr")
    payload = base.with_suffix(".cfl")
    if not header.is_file() or not payload.is_file():
        raise FileNotFoundError(f"Missing BART CFL pair: {base}.{{hdr,cfl}}")
    dimension_line = next(
        (
            line.strip()
            for line in header.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        None,
    )
    if dimension_line is None:
        raise ValueError(f"BART header contains no dimensions: {header}")
    try:
        shape = tuple(int(value) for value in dimension_line.split())
    except ValueError as exc:
        raise ValueError(f"Invalid BART dimensions in {header}: {dimension_line!r}") from exc
    if not shape or any(size < 1 for size in shape):
        raise ValueError(f"BART dimensions must be positive: {shape}")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.complex64).itemsize
    if payload.stat().st_size != expected_bytes:
        raise ValueError(
            f"BART payload size mismatch for {payload}: expected {expected_bytes}, "
            f"found {payload.stat().st_size}."
        )
    return shape


def open_cfl(path: str | Path, mode: str = "r") -> np.memmap:
    """Open a BART payload in its logical Fortran-ordered shape.

    Args:
        path: BART basename or CFL/HDR path.
        mode: NumPy memory-map mode, normally ``r`` or ``r+``.

    Returns:
        Memory-mapped complex64 array with validated logical dimensions.
    """
    base = bart_base(path)
    return np.memmap(
        base.with_suffix(".cfl"),
        mode=mode,
        dtype=np.complex64,
        shape=read_shape(base),
        order="F",
    )


def create_cfl(path: str | Path, shape: Sequence[int]) -> np.memmap:
    """Create a non-empty BART CFL pair.

    Args:
        path: Destination basename or CFL/HDR path.
        shape: Positive output dimensions in BART order.

    Returns:
        Writable Fortran-ordered complex64 memory map.
    """
    base = bart_base(path)
    shape = tuple(int(value) for value in shape)
    if not shape or any(size < 1 for size in shape):
        raise ValueError(f"BART dimensions must be positive: {shape}")
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".hdr").write_text(
        "# Dimensions\n" + " ".join(str(size) for size in shape) + "\n",
        encoding="utf-8",
    )
    return np.memmap(
        base.with_suffix(".cfl"),
        mode="w+",
        dtype=np.complex64,
        shape=shape,
        order="F",
    )


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading a full payload into memory.

    Args:
        path: File to read without modification.
        chunk_bytes: Maximum bytes read per hashing iteration.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def logical_array_sha256(array: np.ndarray, axis_zero_chunk: int = 16) -> str:
    """Hash logical C-order array values without one full contiguous copy.

    Args:
        array: Array whose logical values and dtype are part of the identity.
        axis_zero_chunk: Maximum leading-axis planes copied per hash update.

    Returns:
        Lowercase hexadecimal SHA-256 digest of the C-order logical bytes.
    """
    values = np.asarray(array)
    if values.ndim < 1 or axis_zero_chunk < 1:
        raise ValueError("Logical array hashing requires an array and a positive chunk size.")
    digest = hashlib.sha256()
    for start in range(0, values.shape[0], axis_zero_chunk):
        block = np.ascontiguousarray(values[start : start + axis_zero_chunk])
        digest.update(block.view(np.uint8))
    return digest.hexdigest()


def logical_cfl_sha256(path: str | Path, axis_zero_chunk: int = 16) -> str:
    """Hash one BART CFL as logical C-order complex64 values.

    Args:
        path: BART basename or either member of its CFL/HDR pair.
        axis_zero_chunk: Maximum leading-axis planes copied per hash update.

    Returns:
        Lowercase hexadecimal SHA-256 digest independent of CFL storage order.
    """
    return logical_array_sha256(open_cfl(path), axis_zero_chunk=axis_zero_chunk)


def cfl_record(path: str | Path, *, include_hash: bool = True) -> dict[str, object]:
    """Build provenance for one validated CFL pair.

    Args:
        path: BART basename or CFL/HDR path.
        include_hash: Whether to hash both stored files.

    Returns:
        JSON-native dimensions, type, size, paths, and optional hashes.
    """
    base = bart_base(path)
    record: dict[str, object] = {
        "base": str(base),
        "shape": list(read_shape(base)),
        "dtype": "complex64",
        "payload_bytes": base.with_suffix(".cfl").stat().st_size,
    }
    if include_hash:
        record["header_sha256"] = sha256_file(base.with_suffix(".hdr"))
        record["payload_sha256"] = sha256_file(base.with_suffix(".cfl"))
    return record


def recombine_split_complex_cfl(
    split_path: str | Path,
    output_path: str | Path,
    *,
    partition_chunk: int = 8,
    residual_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Recombine BART's size-two ITER real/imaginary representation.

    Args:
        split_path: Split-complex BART image basename.
        output_path: Destination native-complex BART basename.
        partition_chunk: Maximum logical PAR planes processed per block.
        residual_tolerance: Maximum allowed off-axis complex residual.

    Returns:
        Shapes, hashes, recombination rule, and measured residual.
    """
    split = open_cfl(split_path)
    if split.ndim < 9 or split.shape[8] != 2 or partition_chunk < 1:
        raise ValueError(
            f"Expected a size-two BART ITER dimension and positive chunk; got {split.shape}."
        )
    if any(value != 1 for value in split.shape[9:]):
        raise ValueError(f"Unexpected dimensions after split-complex ITER: {split.shape}.")
    output_shape = list(split.shape)
    output_shape[8] = 1
    output = create_cfl(output_path, output_shape)
    maximum_residual = 0.0
    for start in range(0, split.shape[2], partition_chunk):
        stop = min(start + partition_chunk, split.shape[2])
        block = np.asarray(split[:, :, start:stop, ...])
        real_component = np.take(block, 0, axis=8)
        imaginary_component = np.take(block, 1, axis=8)
        maximum_residual = max(
            maximum_residual,
            float(np.max(np.abs(real_component.imag))),
            float(np.max(np.abs(imaginary_component.real))),
        )
        combined = real_component.real + 1j * imaginary_component.imag
        output[:, :, start:stop, ...] = np.expand_dims(combined, axis=8)
    output.flush()
    del output
    if maximum_residual > residual_tolerance:
        raise ValueError(
            "Split-complex components contain an off-axis residual of "
            f"{maximum_residual:g}."
        )
    return {
        "split_base": str(bart_base(split_path)),
        "split_shape": list(split.shape),
        "split_payload_sha256": sha256_file(bart_base(split_path).with_suffix(".cfl")),
        "recombined_base": str(bart_base(output_path)),
        "recombined_shape": output_shape,
        "recombined_payload_sha256": sha256_file(
            bart_base(output_path).with_suffix(".cfl")
        ),
        "rule": "complex = real(ITER[0]) + 1j * imag(ITER[1])",
        "maximum_off_axis_component_residual": maximum_residual,
        "residual_tolerance": residual_tolerance,
    }
