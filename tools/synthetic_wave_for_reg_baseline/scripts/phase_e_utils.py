"""Small BART CFL, validation, and diagnostic helpers for the baseline sweep."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def bart_base(path: str | Path) -> Path:
    """Return a CFL basename with a possible .hdr/.cfl suffix removed."""
    path = Path(path)
    return path.with_suffix("") if path.suffix in {".hdr", ".cfl"} else path


def read_bart_shape(path: str | Path) -> tuple[int, ...]:
    """Parse and validate dimensions from a BART header."""
    header = bart_base(path).with_suffix(".hdr")
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
    shape = tuple(int(value) for value in dimension_line.split())
    if not shape or any(value < 1 for value in shape):
        raise ValueError(f"Invalid BART dimensions: {shape}")
    expected_bytes = int(np.prod(shape, dtype=np.int64) * np.dtype(np.complex64).itemsize)
    actual_bytes = bart_base(path).with_suffix(".cfl").stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"BART payload has {actual_bytes} bytes; dimensions require {expected_bytes}."
        )
    return shape


def open_bart_memmap(path: str | Path, mode: str = "r") -> np.memmap:
    """Open a BART complex64 payload in its column-major logical shape."""
    base = bart_base(path)
    shape = read_bart_shape(base)
    return np.memmap(base.with_suffix(".cfl"), mode=mode, dtype=np.complex64, shape=shape, order="F")


def write_bart_header(path: str | Path, shape: Sequence[int]) -> Path:
    """Write the header for a nonempty complex64 BART payload."""
    base = bart_base(path)
    shape = tuple(int(value) for value in shape)
    if not shape or any(value < 1 for value in shape):
        raise ValueError(f"Invalid BART shape: {shape}")
    base.with_suffix(".hdr").write_text(
        "# Dimensions\n" + " ".join(str(value) for value in shape) + "\n",
        encoding="utf-8",
    )
    return base


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Compute a streaming file digest without loading a CFL into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_finite_bart(path: str | Path, first_five: Sequence[int]) -> dict[str, object]:
    """Validate expected leading dimensions and finiteness in coil/map chunks."""
    array = open_bart_memmap(path)
    shape = tuple(int(value) for value in array.shape)
    expected = tuple(int(value) for value in first_five)
    padded = shape + (1,) * max(0, 5 - len(shape))
    if padded[:5] != expected or any(value != 1 for value in padded[5:]):
        raise ValueError(f"BART shape {shape} does not match expected leading dimensions {expected}.")

    finite = True
    squared_norm = 0.0
    # Chunk on PE2 to keep validation memory bounded for full 3D coil maps.
    for start in range(0, expected[2], 8):
        stop = min(start + 8, expected[2])
        block = np.asarray(array[:, :, start:stop, ...])
        finite &= bool(np.isfinite(block).all())
        squared_norm += float(np.vdot(block, block).real)
    if not finite:
        raise ValueError(f"BART payload contains non-finite values: {bart_base(path)}")
    return {
        "shape": list(expected),
        "dtype": "complex64",
        "all_samples_finite": finite,
        "norm": float(np.sqrt(squared_norm)),
        "size_bytes": bart_base(path).with_suffix(".cfl").stat().st_size,
    }
