"""Mask discovery, reuse detection and the approval gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from brain_mask_batch.layout import (  # noqa: E402
    MASK_NAME,
    QC_NAME,
    SIDECAR_NAME,
    STATUS_APPROVED,
    MaskTarget,
    approve,
    baseline_matches,
    discover_targets,
    find_baseline,
    find_contrasts,
    find_subjects,
    is_complete,
    read_sidecar,
    sha256_file,
    write_sidecar,
)


def _touch(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture()
def share(tmp_path: Path) -> Path:
    """A miniature share with MPRAGE and a GRE-style branch name."""
    root = tmp_path / "scans"
    for subject in ("FMP_199", "FMP_9"):
        for contrast in ("MPRAGE_preGad", "MPRAGE_postGad"):
            base = root / subject / contrast / "original_nifti" / "optimal_wavelet"
            _touch(base / "normal" / "sub-normal_part-mag_Recon.nii.gz")
            _touch(base / "normal" / "sub-normal_part-phase_Recon.nii.gz")
    # SWI-style tree: a different branch name and a different baseline stem.
    swi = root / "FMP_199" / "SWI_preGad" / "original_nifti" / "selected_wavelet"
    _touch(swi / "normal" / "sub-native_r3x1_part-mag_Recon.nii.gz")
    return root


def test_find_subjects_orders_numerically(share: Path) -> None:
    assert find_subjects(share) == ["FMP_9", "FMP_199"]


def test_find_subjects_rejects_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_subjects(tmp_path / "absent")


def test_find_contrasts_lists_only_reconstruction_folders(share: Path) -> None:
    (share / "FMP_199" / "notes").mkdir()
    assert find_contrasts(share, "FMP_199") == [
        "MPRAGE_postGad",
        "MPRAGE_preGad",
        "SWI_preGad",
    ]


def test_find_baseline_handles_both_branch_names(share: Path) -> None:
    mprage = find_baseline(share / "FMP_199" / "MPRAGE_preGad")
    assert mprage is not None and "part-mag" in mprage.name

    swi = find_baseline(share / "FMP_199" / "SWI_preGad")
    assert swi is not None and swi.name.startswith("sub-native_r3x1")


def test_find_baseline_ignores_phase_only_folders(tmp_path: Path) -> None:
    contrast = tmp_path / "MPRAGE_preGad"
    _touch(
        contrast
        / "original_nifti"
        / "optimal_wavelet"
        / "normal"
        / "sub-normal_part-phase_Recon.nii.gz"
    )
    assert find_baseline(contrast) is None


def test_discover_targets_writes_beside_the_data(share: Path) -> None:
    targets = discover_targets(share, subjects=["FMP_199"])
    assert [target.contrast for target in targets] == [
        "MPRAGE_postGad",
        "MPRAGE_preGad",
        "SWI_preGad",
    ]
    target = targets[0]
    assert target.masks_dir == share / "FMP_199" / "MPRAGE_postGad" / "masks"
    assert target.mask_path.name == MASK_NAME
    assert target.qc_path.name == QC_NAME


def test_discover_targets_can_mirror_elsewhere(share: Path, tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    target = discover_targets(
        share, subjects=["FMP_199"], contrasts=["MPRAGE_preGad"], output_root=mirror
    )[0]
    assert target.masks_dir == mirror / "FMP_199" / "MPRAGE_preGad" / "masks"
    # The baseline still comes from the share.
    assert share in target.baseline.parents


def _write_mask(target: MaskTarget, content: bytes = b"mask") -> None:
    _touch(target.mask_path, content)
    _touch(target.qc_path, b"png")
    write_sidecar(
        target,
        generator={"tool": "HD-BET", "device": "cpu"},
        baseline_sha256=sha256_file(target.baseline),
        baseline_shape=(2, 2, 2),
        baseline_voxel_mm=(1.0, 1.0, 1.0),
        voxel_count=1234,
        volume_ml=1.234,
    )


def test_is_complete_requires_mask_sidecar_and_figure(share: Path) -> None:
    target = discover_targets(share, subjects=["FMP_199"], contrasts=["MPRAGE_preGad"])[0]
    assert not is_complete(target)

    _write_mask(target)
    assert is_complete(target)

    target.qc_path.unlink()
    assert not is_complete(target)


def test_is_complete_detects_an_edited_mask(share: Path) -> None:
    target = discover_targets(share, subjects=["FMP_199"], contrasts=["MPRAGE_preGad"])[0]
    _write_mask(target)
    target.mask_path.write_bytes(b"edited")
    assert not is_complete(target)


def test_baseline_matches_detects_a_new_reconstruction(share: Path) -> None:
    target = discover_targets(share, subjects=["FMP_199"], contrasts=["MPRAGE_preGad"])[0]
    _write_mask(target)
    payload = read_sidecar(target.sidecar_path)
    assert baseline_matches(target, payload)

    target.baseline.write_bytes(b"a different reconstruction")
    assert not baseline_matches(target, payload)


def test_new_masks_are_pending_review(share: Path) -> None:
    target = discover_targets(share, subjects=["FMP_199"], contrasts=["MPRAGE_preGad"])[0]
    _write_mask(target)
    payload = read_sidecar(target.sidecar_path)
    assert payload["approved"] is False
    assert payload["status"] == "visual_review_required"
    assert payload["baseline"]["sha256"] == sha256_file(target.baseline)


def test_approve_records_the_reviewer(share: Path) -> None:
    target = discover_targets(share, subjects=["FMP_199"], contrasts=["MPRAGE_preGad"])[0]
    _write_mask(target)

    payload = approve(target.sidecar_path, approved_by="Reviewer", note="looks right")
    assert payload["approved"] is True
    assert payload["status"] == STATUS_APPROVED
    assert payload["approval"]["approved_by"] == "Reviewer"

    on_disk = json.loads(target.sidecar_path.read_text(encoding="utf-8"))
    assert on_disk["approved"] is True


def test_approve_refuses_a_changed_mask(share: Path) -> None:
    target = discover_targets(share, subjects=["FMP_199"], contrasts=["MPRAGE_preGad"])[0]
    _write_mask(target)
    target.mask_path.write_bytes(b"edited after generation")

    with pytest.raises(ValueError, match="changed since it was written"):
        approve(target.sidecar_path, approved_by="Reviewer")


def test_read_sidecar_rejects_an_unknown_version(share: Path) -> None:
    target = discover_targets(share, subjects=["FMP_199"], contrasts=["MPRAGE_preGad"])[0]
    _write_mask(target)
    target.sidecar_path.write_text(json.dumps({"format_version": 99}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        read_sidecar(target.sidecar_path)
