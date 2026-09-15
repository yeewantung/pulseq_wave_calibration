"""Read approved brain masks stored beside the reconstructions.

Masks are produced by the ``niftis_to_brain_masks_batch`` tool, which writes
them next to each subject's shipped head mask::

    <root>/FMP_199/MPRAGE_preGad/masks/
        brain_mask_hdbet.nii.gz
        brain_mask_hdbet.json

This module only consumes them. It never creates a mask: if one is absent, the
answer is to run the masking tool, not to improvise a mask mid-analysis.

A mask is usable only when its sidecar records visual approval and still
describes the mask on disk, so an unreviewed or edited mask cannot be scored
against by accident.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MASKS_DIRNAME = "masks"
MASK_NAME = "brain_mask_hdbet.nii.gz"
SIDECAR_NAME = "brain_mask_hdbet.json"

STATUS_APPROVED = "approved"
FORMAT_VERSION = 1


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest.

    Args:
        path: File to digest.
        chunk_bytes: Read size.

    Returns:
        The hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def mask_candidate(masks_dir: Path) -> tuple[Path, Path]:
    """Mask and sidecar paths inside one ``masks`` folder.

    Args:
        masks_dir: The subject/contrast ``masks`` folder.

    Returns:
        The mask path and its sidecar path, neither of which need exist.
    """
    return masks_dir / MASK_NAME, masks_dir / SIDECAR_NAME


def find_mask(contrast_dir: Path) -> Path | None:
    """Locate an approved brain mask for one subject/contrast.

    Args:
        contrast_dir: ``<root>/<subject>/<contrast>`` folder, or a mirror of it.

    Returns:
        The mask path when an approved, intact mask is present; ``None`` when no
        mask has been generated for this contrast yet.

    Raises:
        PermissionError: If a mask exists but has not been approved.
        ValueError: If the sidecar is unreadable, or no longer describes the
            mask beside it.
    """
    mask_path, sidecar_path = mask_candidate(contrast_dir / MASKS_DIRNAME)
    if not mask_path.is_file():
        return None

    if not sidecar_path.is_file():
        raise ValueError(
            f"Brain mask {mask_path} has no sidecar; regenerate it with the "
            "niftis_to_brain_masks_batch tool."
        )

    payload = _read_sidecar(sidecar_path)
    if not payload.get("approved", False):
        raise PermissionError(
            f"Brain mask {mask_path} is not approved (status "
            f"{payload.get('status')!r}). Review "
            f"{contrast_dir / MASKS_DIRNAME / payload.get('qc_image', '')} and run "
            "approve_brain_masks.py before scoring against it."
        )

    recorded = str(payload.get("mask_sha256", ""))
    if recorded and sha256_file(mask_path) != recorded:
        raise ValueError(
            f"Brain mask changed since approval: {mask_path}. Regenerate and "
            "re-approve it."
        )
    return mask_path


def describe(contrast_dir: Path) -> dict[str, Any] | None:
    """Read the sidecar of one subject/contrast, when present.

    Args:
        contrast_dir: ``<root>/<subject>/<contrast>`` folder.

    Returns:
        The sidecar payload, or ``None`` when no mask has been generated.
    """
    _, sidecar_path = mask_candidate(contrast_dir / MASKS_DIRNAME)
    if not sidecar_path.is_file():
        return None
    return _read_sidecar(sidecar_path)


def _read_sidecar(path: Path) -> dict[str, Any]:
    """Read and version-check one sidecar.

    Args:
        path: Sidecar JSON.

    Returns:
        The sidecar payload.

    Raises:
        ValueError: If the sidecar is malformed or of an unsupported version.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed brain-mask sidecar: {path}") from error
    if int(payload.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"Unsupported brain-mask sidecar version: {path}")
    return payload
