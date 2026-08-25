"""Safely install a phase NIfTI generated from an accepted complex image."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from bart_cfl import sha256_file


PHASE_KEYS = (
    "phase_nifti",
    "phase_nifti_sha256",
    "phase_sidecar",
    "phase_sidecar_sha256",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def validate_phase_record(
    nifti_record: dict[str, Any],
    *,
    expected_shape: tuple[int, int, int],
) -> bool:
    """Return whether a manifest's phase files are complete and hash-valid."""
    if any(not nifti_record.get(key) for key in PHASE_KEYS):
        return False
    phase_path = Path(nifti_record["phase_nifti"])
    sidecar_path = Path(nifti_record["phase_sidecar"])
    if not phase_path.is_file() or not sidecar_path.is_file():
        return False
    if sha256_file(phase_path) != nifti_record["phase_nifti_sha256"]:
        return False
    if sha256_file(sidecar_path) != nifti_record["phase_sidecar_sha256"]:
        return False

    image = nib.load(str(phase_path))
    if image.shape != expected_shape or nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        return False
    magnitude_path = Path(nifti_record.get("magnitude_nifti", ""))
    if not magnitude_path.is_file():
        return False
    magnitude = nib.load(str(magnitude_path))
    if magnitude.shape != image.shape or not np.array_equal(
        magnitude.affine, image.affine
    ):
        return False
    phase = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(phase).all():
        return False
    tolerance = 8.0 * np.finfo(np.float32).eps
    if float(np.min(phase)) < -math.pi - tolerance:
        return False
    if float(np.max(phase)) > math.pi + tolerance:
        return False
    sidecar = _load_json(sidecar_path)
    return sidecar.get("Part") == "phase" and sidecar.get("Units") == "rad"


def _assert_same_magnitude(existing_path: Path, generated_path: Path) -> None:
    existing = nib.load(str(existing_path))
    generated = nib.load(str(generated_path))
    if existing.shape != generated.shape or not np.array_equal(
        existing.affine, generated.affine
    ):
        raise ValueError("Regenerated magnitude geometry differs during phase backfill")
    existing_data = np.asarray(existing.dataobj, dtype=np.float32)
    generated_data = np.asarray(generated.dataobj, dtype=np.float32)
    if not np.array_equal(existing_data, generated_data):
        raise ValueError("Regenerated magnitude samples differ during phase backfill")


def _install_if_absent(source: Path, destination: Path) -> None:
    source_hash = sha256_file(source)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != source_hash:
            raise FileExistsError(
                f"Refusing to replace an existing phase artifact: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    shutil.copyfile(source, partial)
    if sha256_file(partial) != source_hash:
        raise RuntimeError(f"Phase artifact copy verification failed: {destination}")
    os.replace(partial, destination)


def install_phase_from_temporary_export(
    *,
    existing_record: dict[str, Any],
    generated_record: dict[str, Any],
    existing_nifti_root: Path,
    generated_nifti_root: Path,
    expected_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Install only phase files after proving the temporary magnitude is identical."""
    existing_nifti_root = existing_nifti_root.resolve()
    generated_nifti_root = generated_nifti_root.resolve()
    existing_magnitude = Path(existing_record["magnitude_nifti"]).resolve()
    generated_magnitude = Path(generated_record["magnitude_nifti"]).resolve()
    existing_relative = existing_magnitude.relative_to(existing_nifti_root)
    generated_relative = generated_magnitude.relative_to(generated_nifti_root)
    if existing_relative != generated_relative:
        raise ValueError("Temporary export does not target the accepted magnitude filename")
    if sha256_file(existing_magnitude) != existing_record["magnitude_nifti_sha256"]:
        raise ValueError("Accepted magnitude hash changed before phase backfill")
    _assert_same_magnitude(existing_magnitude, generated_magnitude)

    generated_phase = Path(generated_record["phase_nifti"]).resolve()
    generated_sidecar = Path(generated_record["phase_sidecar"]).resolve()
    phase_relative = generated_phase.relative_to(generated_nifti_root)
    sidecar_relative = generated_sidecar.relative_to(generated_nifti_root)
    destination_phase = existing_nifti_root / phase_relative
    destination_sidecar = existing_nifti_root / sidecar_relative

    # Install the sidecar first so the NIfTI acts as the final completion marker.
    _install_if_absent(generated_sidecar, destination_sidecar)
    _install_if_absent(generated_phase, destination_phase)
    updated = dict(existing_record)
    updated.update(
        {
            "phase_nifti": str(destination_phase),
            "phase_nifti_sha256": sha256_file(destination_phase),
            "phase_sidecar": str(destination_sidecar),
            "phase_sidecar_sha256": sha256_file(destination_sidecar),
        }
    )
    if not validate_phase_record(updated, expected_shape=expected_shape):
        raise RuntimeError("Installed phase NIfTI failed validation")
    if sha256_file(existing_magnitude) != existing_record["magnitude_nifti_sha256"]:
        raise RuntimeError("Accepted magnitude changed during phase backfill")
    return updated
