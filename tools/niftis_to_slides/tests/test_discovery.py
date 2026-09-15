"""Discovery indexes both share layouts into ordered comparison cases."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from nifti_slides.discovery import (  # noqa: E402
    BRANCH_ORDER,
    SR_MODEL_ORDER,
    VARIANT_ORDER,
    Case,
    find_magnitude,
    find_subjects,
    group_by_section,
    index_cases,
    subject_contrast_dir,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _reconstruction_stem(case: str, model: str = "") -> str:
    suffix = f"_{model}" if model else ""
    return f"sub-{case}_part-mag_BARTWave{case}{suffix}.nii.gz"


@pytest.fixture()
def shares(tmp_path: Path) -> tuple[Path, Path]:
    """Build a miniature pair of shares covering both nesting spellings."""
    reference = tmp_path / "scans"
    sr_share = tmp_path / "scans_subtle"

    for subject, nested in (("FMP_199", True), ("FMP_213", False)):
        for contrast in ("MPRAGE_preGad", "MPRAGE_postGad"):
            base = reference / subject / contrast / "original_nifti"
            for branch in BRANCH_ORDER:
                _touch(base / branch / "normal" / _reconstruction_stem("normal"))
                # A phase file sits beside every magnitude file.
                _touch(
                    base
                    / branch
                    / "normal"
                    / "sub-normal_part-phase_BARTWavenormal.nii.gz"
                )
                for variant in VARIANT_ORDER:
                    _touch(
                        base / branch / "retro" / variant / _reconstruction_stem(variant)
                    )

            sr_base = (
                sr_share / subject / subject / contrast / "original_nifti"
                if nested
                else sr_share / subject / contrast / "original_nifti"
            )
            for branch in BRANCH_ORDER:
                for variant in VARIANT_ORDER:
                    for model in SR_MODEL_ORDER:
                        _touch(
                            sr_base
                            / branch
                            / "retro"
                            / variant
                            / _reconstruction_stem(variant, model)
                        )

    return sr_share, reference


def test_find_subjects_orders_numerically(tmp_path: Path) -> None:
    for name in ("FMP_213", "FMP_9", "FMP_199", "notes"):
        (tmp_path / name).mkdir()
    assert find_subjects(tmp_path) == ["FMP_9", "FMP_199", "FMP_213"]


def test_subject_contrast_dir_accepts_both_nestings(shares: tuple[Path, Path]) -> None:
    sr_share, _ = shares
    assert subject_contrast_dir(sr_share, "FMP_199", "MPRAGE_preGad") is not None
    assert subject_contrast_dir(sr_share, "FMP_213", "MPRAGE_preGad") is not None
    assert subject_contrast_dir(sr_share, "FMP_999", "MPRAGE_preGad") is None


def test_find_magnitude_ignores_phase_and_sr_outputs(tmp_path: Path) -> None:
    folder = tmp_path / "retro" / "native_r3x2"
    _touch(folder / "sub-native_r3x2_part-mag_Recon.nii.gz")
    _touch(folder / "sub-native_r3x2_part-phase_Recon.nii.gz")
    _touch(folder / "sub-native_r3x2_part-mag_Recon_shd.nii.gz")

    plain = find_magnitude(folder, SR_MODEL_ORDER)
    assert plain is not None and plain.name.endswith("Recon.nii.gz")

    sr_output = find_magnitude(folder, SR_MODEL_ORDER, sr_model="shd")
    assert sr_output is not None and sr_output.name.endswith("_shd.nii.gz")

    assert find_magnitude(folder, SR_MODEL_ORDER, sr_model="cond_unet_retrain") is None


def test_find_magnitude_rejects_ambiguity(tmp_path: Path) -> None:
    folder = tmp_path / "normal"
    _touch(folder / "sub-a_part-mag_One.nii.gz")
    _touch(folder / "sub-b_part-mag_Two.nii.gz")
    with pytest.raises(RuntimeError, match="Ambiguous"):
        find_magnitude(folder, SR_MODEL_ORDER)


def test_index_cases_orders_sections_and_stages(shares: tuple[Path, Path]) -> None:
    sr_share, reference = shares
    cases = index_cases(sr_share, reference)

    sections = group_by_section(cases)
    assert list(sections) == [
        ("FMP_199", "MPRAGE_preGad"),
        ("FMP_199", "MPRAGE_postGad"),
        ("FMP_213", "MPRAGE_preGad"),
        ("FMP_213", "MPRAGE_postGad"),
    ]

    section = sections[("FMP_199", "MPRAGE_preGad")]
    # One baseline, then every variant x branch x (Pre-SR + both SR models).
    assert len(section) == 1 + len(VARIANT_ORDER) * len(BRANCH_ORDER) * 3
    assert section[0].is_baseline
    assert section[0].variant == "normal"

    first_group = section[1:4]
    assert [case.stage for case in first_group] == ["Pre-SR", "Post-SR", "Post-SR"]
    assert [case.sr_model for case in first_group] == ["", *SR_MODEL_ORDER]
    assert {case.variant for case in first_group} == {VARIANT_ORDER[0]}


def test_index_cases_skips_contrast_without_baseline(shares: tuple[Path, Path]) -> None:
    sr_share, reference = shares
    baseline = (
        reference
        / "FMP_213"
        / "MPRAGE_postGad"
        / "original_nifti"
        / "optimal_wavelet"
        / "normal"
    )
    for path in baseline.glob("*part-mag*"):
        path.unlink()

    sections = group_by_section(index_cases(sr_share, reference))
    assert ("FMP_213", "MPRAGE_postGad") not in sections
    assert ("FMP_213", "MPRAGE_preGad") in sections


def test_image_stem_carries_full_provenance() -> None:
    case = Case(
        subject="FMP_199",
        contrast="MPRAGE_preGad",
        branch="fista_r0",
        variant="lr_x_1p5mm_r3x2",
        stage="Post-SR",
        sr_model="shd",
        path=Path("unused.nii.gz"),
    )
    assert case.image_stem == (
        "FMP199_preGad_fista_r0_lr_x_1p5mm_r3x2_Post-SR_shd"
    )
    assert case.subject_number == 199
