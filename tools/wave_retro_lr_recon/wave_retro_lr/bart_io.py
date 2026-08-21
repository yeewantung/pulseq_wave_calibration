"""Memory-bounded BART CFL I/O used by retrospective reconstruction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def bart_base(path: str | Path) -> Path:
    """Return a BART basename with a possible payload/header suffix removed."""
    path = Path(path)
    return path.with_suffix("") if path.suffix in {".cfl", ".hdr"} else path


def read_shape(path: str | Path) -> tuple[int, ...]:
    """Read a CFL header and validate its complex64 payload size."""
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
    """Open a BART payload in its logical Fortran-ordered shape."""
    base = bart_base(path)
    return np.memmap(
        base.with_suffix(".cfl"),
        mode=mode,
        dtype=np.complex64,
        shape=read_shape(base),
        order="F",
    )


def create_cfl(path: str | Path, shape: Sequence[int]) -> np.memmap:
    """Create a non-empty BART CFL pair and return its writable payload."""
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
    """Hash a file without loading a full reconstruction payload into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def cfl_record(path: str | Path, *, include_hash: bool = True) -> dict[str, object]:
    """Return portable provenance for one validated CFL pair."""
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
