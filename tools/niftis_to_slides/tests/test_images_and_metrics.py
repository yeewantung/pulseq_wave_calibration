"""Strip rendering and masked metrics behave as the deck pattern requires."""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from nifti_slides.images import (  # noqa: E402
    ORIENTATION_ORDER,
    build_baseline_grid,
    center_slices,
    orientation_strip,
    resample_to_baseline,
    write_strip_tiff,
)
from nifti_slides.metrics import (  # noqa: E402
    build_foreground,
    compute_metrics,
    scale_to_baseline,
)


def _volume(shape=(12, 16, 16), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.random(shape, dtype=np.float32)
    data[2:-2, 2:-2, 2:-2] += 4.0  # a bright interior standing in for anatomy
    return data.astype(np.float32)


def _nifti(data: np.ndarray, voxel_mm: float = 1.0) -> nib.Nifti1Image:
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    return nib.Nifti1Image(data, affine)


def test_center_slices_use_the_middle_of_each_axis() -> None:
    data = _volume((9, 11, 13))
    indices, panels = center_slices(data)
    assert indices == {"sagittal": 4, "coronal": 5, "axial": 6}
    assert panels["sagittal"].shape == (13, 11)
    assert panels["coronal"].shape == (13, 9)
    assert panels["axial"].shape == (11, 9)


def test_orientation_strip_concatenates_without_a_gap() -> None:
    data = _volume((12, 16, 16))
    _, panels = center_slices(data)
    _, strip = orientation_strip(data)

    assert strip.shape[1] == sum(panels[name].shape[1] for name in ORIENTATION_ORDER)
    assert strip.shape[0] == max(panels[name].shape[0] for name in ORIENTATION_ORDER)

    # Panels land side by side in order, so the first columns are the sagittal panel.
    width = panels["sagittal"].shape[1]
    np.testing.assert_array_equal(strip[:, :width], panels["sagittal"])


def test_orientation_strip_pads_a_non_cubic_field_of_view() -> None:
    # Fewer superior rows than anterior rows makes the axial panel the tall one.
    data = _volume((12, 16, 10))
    _, strip = orientation_strip(data)
    assert strip.shape[0] == 16
    assert strip.shape[1] == 16 + 12 + 12


def test_write_strip_tiff_is_sixteen_bit_and_windowed(tmp_path: Path) -> None:
    strip = np.linspace(0.0, 2.0, 64, dtype=np.float32).reshape(8, 8)
    path = tmp_path / "strip.tiff"
    width, height = write_strip_tiff(strip, display_max=1.0, path=path)

    assert (width, height) == (8, 8)
    with Image.open(path) as image:
        assert image.mode == "I;16"
        pixels = np.array(image)

    assert pixels.dtype == np.uint16
    assert pixels.max() == 65535  # everything at or above the window clips
    assert pixels[0, 0] == 0


def test_write_strip_tiff_rejects_a_flat_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        write_strip_tiff(np.zeros((8, 8), np.float32), 0.0, tmp_path / "x.tiff")


def test_resample_to_baseline_reaches_the_baseline_grid() -> None:
    baseline = _nifti(_volume((12, 16, 16)))
    coarse = _nifti(_volume((6, 16, 16), seed=1), voxel_mm=2.0)

    same, resampled = resample_to_baseline(baseline, baseline)
    assert not resampled and same.shape == baseline.shape

    placed, resampled = resample_to_baseline(coarse, baseline)
    assert resampled and placed.shape == baseline.shape
    assert np.isfinite(placed).all() and (placed >= 0).all()


def test_build_baseline_grid_windows_on_the_mask(tmp_path: Path) -> None:
    data = _volume()
    path = tmp_path / "baseline.nii.gz"
    nib.save(_nifti(data), path)

    mask = data > 1.0
    grid = build_baseline_grid(path, mask, display_percentile=99.5)
    assert grid.display_max == pytest.approx(np.percentile(data[mask], 99.5), rel=1e-5)
    assert grid.data.shape == data.shape


def test_build_foreground_falls_back_when_no_head_mask() -> None:
    data = _volume()
    foreground = build_foreground(data, _nifti(data), mode="head", head_mask_path=None)
    assert foreground.source == "reference_intensity"
    assert 0 < foreground.voxel_count < data.size


def test_build_foreground_uses_a_shipped_head_mask(tmp_path: Path) -> None:
    data = _volume()
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[3:-3, 3:-3, 3:-3] = 1
    mask_path = tmp_path / "head_mask.nii.gz"
    nib.save(_nifti(mask.astype(np.uint8)), mask_path)

    foreground = build_foreground(
        data, _nifti(data), mode="head", head_mask_path=mask_path
    )
    assert foreground.source == "head_mask_from_normal"
    assert foreground.voxel_count == int(mask.sum())


def test_brain_mask_mode_requires_an_approved_mask() -> None:
    data = _volume()
    with pytest.raises(FileNotFoundError, match="approved mask"):
        build_foreground(data, _nifti(data), mode="brain")


def test_brain_mask_mode_uses_the_supplied_mask(tmp_path: Path) -> None:
    data = _volume()
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[4:-4, 4:-4, 4:-4] = 1
    path = tmp_path / "brain_mask.nii.gz"
    nib.save(_nifti(mask.astype(np.uint8)), path)

    foreground = build_foreground(
        data, _nifti(data), mode="brain", brain_mask_path=path
    )
    assert foreground.source == "brain_mask:brain_mask.nii.gz"
    assert foreground.voxel_count == int(mask.sum())


def test_scale_to_baseline_recovers_a_known_gain() -> None:
    baseline = _volume()
    mask = baseline > 1.0
    scaled, gain = scale_to_baseline(baseline, baseline * 0.4, mask, "lsq")
    assert gain == pytest.approx(2.5, rel=1e-4)
    np.testing.assert_allclose(scaled[mask], baseline[mask], rtol=1e-4)


def test_metrics_are_perfect_for_an_identical_volume() -> None:
    baseline = _volume()
    mask = baseline > 1.0
    scored = compute_metrics(baseline, baseline.copy(), mask)

    assert scored.nrmse == pytest.approx(0.0, abs=1e-6)
    assert scored.ssim_mask == pytest.approx(1.0, abs=1e-6)
    assert np.isinf(scored.psnr_db)
    assert scored.intensity_scale == pytest.approx(1.0, rel=1e-5)


def test_metrics_degrade_with_noise() -> None:
    baseline = _volume()
    mask = baseline > 1.0
    rng = np.random.default_rng(7)
    noisy = (baseline + rng.normal(0, 0.5, baseline.shape)).astype(np.float32)

    scored = compute_metrics(baseline, noisy, mask)
    assert scored.nrmse > 0.0
    assert 0.0 < scored.ssim_mask < 1.0
    assert np.isfinite(scored.psnr_db)


def test_masked_metrics_ignore_background_noise() -> None:
    """Background noise must not reach the reported metrics.

    A whole-volume SSIM would be dominated by the air around the head, which is
    the reason every metric here is restricted to a foreground mask.
    """
    from skimage.metrics import structural_similarity

    rng = np.random.default_rng(3)
    shape = (40, 40, 40)
    baseline = rng.random(shape, dtype=np.float32) * 0.05
    baseline[14:26, 14:26, 14:26] += 5.0

    mask = baseline > 1.0
    assert mask.sum() < baseline.size / 8  # anatomy is a small part of the volume

    # Anatomy identical, background redrawn from the same distribution.
    corrupted = baseline.copy()
    corrupted[~mask] = (rng.random(int((~mask).sum())) * 0.05).astype(np.float32)

    scored = compute_metrics(baseline, corrupted, mask)
    assert scored.nrmse == pytest.approx(0.0, abs=1e-6)
    assert scored.ssim_mask > 0.999

    whole_volume = structural_similarity(
        baseline,
        corrupted,
        data_range=float(baseline.max() - baseline.min()),
        channel_axis=None,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert whole_volume < scored.ssim_mask
