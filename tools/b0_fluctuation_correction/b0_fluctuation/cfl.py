"""Minimal BART CFL input/output used by the B0-fluctuation tools."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def _base(path: str | Path) -> Path:
    """Strip a ``.cfl``/``.hdr`` suffix from a BART path.

    Args:
        path: BART basename or one member of its file pair.

    Returns:
        BART basename.
    """
    path = Path(path)
    return path.with_suffix("") if path.suffix in {".cfl", ".hdr"} else path


def read_shape(path: str | Path) -> tuple[int, ...]:
    """Read the dimensions of a BART CFL pair and check the payload size.

    Args:
        path: BART basename.

    Returns:
        Stored dimensions, including trailing singleton dimensions.

    Raises:
        FileNotFoundError: If the header or payload is missing.
        ValueError: If the payload size disagrees with the header.
    """
    base = _base(path)
    header = base.with_suffix(".hdr")
    payload = base.with_suffix(".cfl")
    if not header.is_file() or not payload.is_file():
        raise FileNotFoundError(f"Missing BART CFL pair: {base}")
    lines = [line.strip() for line in header.read_text().splitlines()]
    dims_line = next(line for line in lines if line and not line.startswith("#"))
    shape = tuple(int(value) for value in dims_line.split())
    expected = int(np.prod(shape, dtype=np.int64)) * 8
    if payload.stat().st_size != expected:
        raise ValueError(f"{payload} holds {payload.stat().st_size} bytes; header implies {expected}.")
    return shape


def read_cfl(path: str | Path, ndim: int | None = None) -> np.memmap:
    """Memory-map a BART CFL in Fortran order.

    Args:
        path: BART basename.
        ndim: Optional number of leading dimensions to keep; dropped trailing
            dimensions must be singleton.

    Returns:
        Read-only complex64 memory map.

    Raises:
        ValueError: If a dropped dimension is not singleton.
    """
    base = _base(path)
    shape = read_shape(base)
    values = np.memmap(base.with_suffix(".cfl"), dtype=np.complex64, mode="r", shape=shape, order="F")
    if ndim is None:
        return values
    padded = shape + (1,) * max(0, ndim - len(shape))
    if any(size != 1 for size in padded[ndim:]):
        raise ValueError(f"{base} has nonsingleton dimensions beyond {ndim}: {shape}.")
    return values.reshape(padded[:ndim], order="F")


def write_cfl(path: str | Path, values: np.ndarray) -> None:
    """Write a complex array as a BART CFL pair (Fortran order).

    Args:
        path: Destination basename; parent directories are created.
        values: Array written with its own shape.
    """
    base = _base(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.complex64)
    base.with_suffix(".hdr").write_text("# Dimensions\n" + " ".join(str(n) for n in array.shape) + "\n")
    output = np.memmap(base.with_suffix(".cfl"), dtype=np.complex64, mode="w+", shape=array.shape, order="F")
    output[...] = array
    output.flush()
    del output


def create_cfl(path: str | Path, shape: Sequence[int]) -> np.memmap:
    """Create a writable BART CFL memory map.

    Args:
        path: Destination basename; parent directories are created.
        shape: Stored dimensions.

    Returns:
        Writable complex64 Fortran-ordered memory map.
    """
    base = _base(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".hdr").write_text("# Dimensions\n" + " ".join(str(int(n)) for n in shape) + "\n")
    return np.memmap(base.with_suffix(".cfl"), dtype=np.complex64, mode="w+", shape=tuple(shape), order="F")


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    """Hash one file in bounded memory.

    Args:
        path: File to hash.
        chunk: Bytes per read.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()
