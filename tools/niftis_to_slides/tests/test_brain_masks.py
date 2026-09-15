"""Reading brain masks stored beside the reconstructions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from nifti_slides.brain_masks import (  # noqa: E402
    MASK_NAME,
    SIDECAR_NAME,
    describe,
    find_mask,
    sha256_file,
)


def _install(
    contrast_dir: Path, *, approved: bool, content: bytes = b"mask", **overrides
) -> Path:
    """Write a mask and sidecar the way the masking tool would."""
    masks = contrast_dir / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    mask_path = masks / MASK_NAME
    mask_path.write_bytes(content)

    payload = {
        "format_version": 1,
        "status": "approved" if approved else "visual_review_required",
        "approved": approved,
        "mask": MASK_NAME,
        "qc_image": "brain_mask_hdbet_qc.png",
        "mask_sha256": sha256_file(mask_path),
        "voxel_count": 1_234_567,
        "volume_ml": 1234.567,
    }
    payload.update(overrides)
    (masks / SIDECAR_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return mask_path


def test_absent_mask_reads_as_none(tmp_path: Path) -> None:
    assert find_mask(tmp_path) is None
    assert describe(tmp_path) is None


def test_approved_mask_is_returned(tmp_path: Path) -> None:
    expected = _install(tmp_path, approved=True)
    assert find_mask(tmp_path) == expected
    assert describe(tmp_path)["voxel_count"] == 1_234_567


def test_unapproved_mask_is_refused(tmp_path: Path) -> None:
    _install(tmp_path, approved=False)
    with pytest.raises(PermissionError, match="not approved"):
        find_mask(tmp_path)


def test_mask_changed_since_approval_is_refused(tmp_path: Path) -> None:
    mask_path = _install(tmp_path, approved=True)
    mask_path.write_bytes(b"edited after approval")
    with pytest.raises(ValueError, match="changed since approval"):
        find_mask(tmp_path)


def test_mask_without_a_sidecar_is_refused(tmp_path: Path) -> None:
    _install(tmp_path, approved=True)
    (tmp_path / "masks" / SIDECAR_NAME).unlink()
    with pytest.raises(ValueError, match="no sidecar"):
        find_mask(tmp_path)


def test_unsupported_sidecar_version_is_refused(tmp_path: Path) -> None:
    _install(tmp_path, approved=True, format_version=99)
    with pytest.raises(ValueError, match="Unsupported"):
        find_mask(tmp_path)


def test_malformed_sidecar_is_refused(tmp_path: Path) -> None:
    _install(tmp_path, approved=True)
    (tmp_path / "masks" / SIDECAR_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        find_mask(tmp_path)
