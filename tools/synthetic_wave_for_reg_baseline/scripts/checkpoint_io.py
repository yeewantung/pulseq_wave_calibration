"""Small shared helpers for safe, resumable NumPy reconstruction outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a small JSON document only after its complete content is written."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_coil_basis(path: Path, ncc: int) -> np.ndarray:
    """Load the requested leading columns of a nested virtual-coil basis."""
    with np.load(path) as archive:
        basis = np.asarray(archive["basis"], dtype=np.complex64)
    if basis.ndim != 2 or not 1 <= ncc <= basis.shape[1]:
        raise ValueError(f"Invalid basis {basis.shape} or requested Ncc={ncc}.")
    return basis[:, :ncc]


def open_or_create_complex64_npy(
    path: Path,
    shape: tuple[int, ...],
    *,
    resume: bool,
) -> np.memmap:
    """Open a compatible checkpoint or safely create a new complex64 NPY file."""
    if resume and path.is_file():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.complex64:
            raise ValueError(
                f"Checkpoint {path} has {array.shape} {array.dtype}, expected {shape}."
            )
        return array
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {path}")
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.complex64, shape=shape
    )


def validate_resume_pair(data_path: Path, progress_path: Path, *, resume: bool) -> None:
    """Reject a partial checkpoint pair that cannot establish a safe boundary."""
    if resume and data_path.is_file() != progress_path.is_file():
        raise ValueError(
            f"Resume requires both checkpoint files or neither: {data_path}, {progress_path}."
        )
