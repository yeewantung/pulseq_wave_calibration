"""Full-volume NRMSE, SSIM and PSNR against the fully sampled baseline.

Metrics are never taken over the whole field of view: air background outnumbers
anatomy and would inflate PSNR and SSIM. Each metric is restricted to a
foreground mask, and 3D SSIM is averaged over mask voxels rather than over the
bounding box, so background contributes nothing at all.

Mask modes
----------
``head``       the whole-head mask shipped beside the reconstructions, falling
               back to ``intensity`` when the share carries none
``intensity``  reference samples above a fraction of their own 99th percentile
``brain``      brain-only mask written beside the reconstruction by the
               niftis_to_brain_masks_batch tool, usable once visually approved

Intensity scaling is applied inside the mask before every metric, so a global
gain difference between reconstructions is not charged as error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from skimage.metrics import structural_similarity


MASK_MODES: tuple[str, ...] = ("head", "intensity", "brain")

#: Whole-head mask shipped beside the reconstructions of the reference share.
HEAD_MASK_RELATIVE = Path("masks") / "head_mask_from_normal.nii.gz"

#: Manifest the brain-mask stage writes, and the metrics stage checks.
BRAIN_MASK_MANIFEST = "brain_mask_manifest.json"

SCALE_MODES: tuple[str, ...] = ("none", "lsq", "median", "percentile")


@dataclass(frozen=True)
class Foreground:
    """The voxels one section is scored over.

    Attributes:
        mask: Boolean mask on the baseline grid.
        source: How the mask was derived, recorded alongside every metric row.
    """

    mask: np.ndarray
    source: str

    @property
    def voxel_count(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True)
class VolumeMetrics:
    """Metrics of one volume against its baseline.

    Attributes:
        nrmse: Root-mean-square error normalized by the baseline RMS.
        ssim_mask: 3D SSIM averaged over mask voxels.
        ssim_bbox: 3D SSIM averaged over the mask bounding box.
        psnr_db: Peak signal-to-noise ratio using the baseline 99th percentile.
        intensity_scale: Gain applied to the volume before scoring.
    """

    nrmse: float
    ssim_mask: float
    ssim_bbox: float
    psnr_db: float
    intensity_scale: float


def build_foreground(
    baseline: np.ndarray,
    baseline_image: nib.Nifti1Image,
    *,
    mode: str = "head",
    head_mask_path: Path | None = None,
    brain_mask_path: Path | None = None,
    intensity_fraction: float = 0.05,
) -> Foreground:
    """Derive one foreground mask a section is scored over.

    Args:
        baseline: Baseline magnitude samples.
        baseline_image: Baseline NIfTI, defining the grid a mask is mapped to.
        mode: One of :data:`MASK_MODES`.
        head_mask_path: Whole-head mask shipped with the reference share.
        brain_mask_path: Approved brain mask for this subject and contrast.
        intensity_fraction: Threshold as a fraction of the baseline 99th
            percentile, used by the ``intensity`` mode and by the fallback.

    Returns:
        The mask and a label describing how it was derived.

    Raises:
        FileNotFoundError: If ``mode`` is ``brain`` and no mask was supplied.
        ValueError: If ``mode`` is unknown or the mask ends up empty.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown mask mode: {mode}")

    if mode == "brain":
        if brain_mask_path is None or not brain_mask_path.is_file():
            raise FileNotFoundError(
                "Brain-mask metrics need an approved mask beside the "
                "reconstruction. Generate one with the "
                "niftis_to_brain_masks_batch tool and approve it, or drop "
                "'brain' from --metric-masks."
            )
        mask = _load_binary_mask(brain_mask_path, baseline_image)
        source = f"brain_mask:{brain_mask_path.name}"
    elif mode == "head" and head_mask_path is not None and head_mask_path.is_file():
        mask = _load_binary_mask(head_mask_path, baseline_image)
        source = "head_mask_from_normal"
    else:
        mask = _intensity_mask(baseline, intensity_fraction)
        source = "reference_intensity"

    mask = mask & np.isfinite(baseline)
    if int(mask.sum()) == 0:
        raise ValueError("Foreground mask is empty; lower the intensity fraction.")
    return Foreground(mask=mask, source=source)


def _load_binary_mask(path: Path, baseline_image: nib.Nifti1Image) -> np.ndarray:
    """Load a mask and place it on the baseline grid.

    Args:
        path: Mask NIfTI.
        baseline_image: Baseline defining the target grid.

    Returns:
        The boolean mask on the baseline grid.
    """
    image = nib.load(str(path))
    if image.shape != baseline_image.shape or not np.allclose(
        image.affine, baseline_image.affine, atol=1e-5
    ):
        image = resample_from_to(image, baseline_image, order=0)
    return np.asarray(image.dataobj) > 0.5


def _intensity_mask(baseline: np.ndarray, fraction: float) -> np.ndarray:
    """Threshold the baseline at a fraction of its positive 99th percentile."""
    positive = baseline[baseline > 0]
    if positive.size == 0:
        raise ValueError("Baseline has no positive samples.")
    return baseline > fraction * np.percentile(positive, 99.0)


def bounding_box(mask: np.ndarray, pad: int = 2) -> tuple[slice, ...]:
    """Padded bounding box of a 3D mask, clipped to the array.

    Args:
        mask: Boolean mask.
        pad: Voxels of padding added on every side.

    Returns:
        One slice per axis.
    """
    coordinates = np.argwhere(mask)
    low = np.maximum(coordinates.min(axis=0) - pad, 0)
    high = np.minimum(coordinates.max(axis=0) + 1 + pad, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def scale_to_baseline(
    baseline: np.ndarray, volume: np.ndarray, mask: np.ndarray, mode: str = "lsq"
) -> tuple[np.ndarray, float]:
    """Match a volume to the baseline intensity level inside the mask.

    Args:
        baseline: Baseline samples.
        volume: Volume to scale, on the baseline grid.
        mask: Foreground mask.
        mode: One of :data:`SCALE_MODES`.

    Returns:
        The scaled volume and the gain applied.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode == "none":
        return volume, 1.0

    reference = baseline[mask].astype(np.float64)
    test = volume[mask].astype(np.float64)
    epsilon = 1e-12

    if mode == "lsq":
        scale = float(np.sum(reference * test) / max(np.sum(test * test), epsilon))
    elif mode == "median":
        scale = float(np.median(reference) / max(np.median(test), epsilon))
    elif mode == "percentile":
        scale = float(
            np.percentile(reference, 99) / max(np.percentile(test, 99), epsilon)
        )
    else:
        raise ValueError(f"Unknown scale mode: {mode}")

    return (volume * scale).astype(np.float32), scale


def compute_metrics(
    baseline: np.ndarray,
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    scale_mode: str = "lsq",
) -> VolumeMetrics:
    """Score one volume against the baseline inside the foreground mask.

    Args:
        baseline: Baseline samples.
        volume: Volume on the baseline grid.
        mask: Foreground mask.
        scale_mode: Intensity match applied before scoring.

    Returns:
        The volume's metrics.
    """
    scaled, scale = scale_to_baseline(baseline, volume, mask, scale_mode)
    ssim_mask, ssim_bbox = _ssim_3d(baseline, scaled, mask)
    return VolumeMetrics(
        nrmse=_nrmse(baseline, scaled, mask),
        ssim_mask=ssim_mask,
        ssim_bbox=ssim_bbox,
        psnr_db=_psnr(baseline, scaled, mask),
        intensity_scale=scale,
    )


def _nrmse(baseline: np.ndarray, volume: np.ndarray, mask: np.ndarray) -> float:
    """RMS-normalized root-mean-square error inside the mask."""
    reference = baseline[mask].astype(np.float64)
    test = volume[mask].astype(np.float64)
    error = np.sqrt(np.mean((test - reference) ** 2))
    return float(error / max(np.sqrt(np.mean(reference**2)), 1e-12))


def _psnr(baseline: np.ndarray, volume: np.ndarray, mask: np.ndarray) -> float:
    """Peak signal-to-noise ratio using the baseline 99th percentile as peak."""
    reference = baseline[mask].astype(np.float64)
    test = volume[mask].astype(np.float64)
    error = np.sqrt(np.mean((test - reference) ** 2))
    if error <= 0:
        return float("inf")
    peak = float(np.percentile(reference, 99))
    if peak <= 0:
        return float("nan")
    return float(20.0 * np.log10(peak / error))


def _ssim_3d(
    baseline: np.ndarray, volume: np.ndarray, mask: np.ndarray, pad: int = 2
) -> tuple[float, float]:
    """Full-volume 3D SSIM, averaged over the mask and over the bounding box.

    The SSIM map is computed on the mask bounding box so the Gaussian windows
    still see real anatomical neighbourhoods, then averaged over mask voxels so
    air background is excluded from the reported value.

    Args:
        baseline: Baseline samples.
        volume: Volume on the baseline grid.
        mask: Foreground mask.
        pad: Bounding-box padding.

    Returns:
        The mask-averaged SSIM and the bounding-box mean SSIM.
    """
    box = bounding_box(mask, pad=pad)
    reference = baseline[box].astype(np.float32)
    test = volume[box].astype(np.float32)
    inside = mask[box]

    if min(reference.shape) < 7:  # skimage's default window
        return float("nan"), float("nan")

    positive = reference[reference > 0]
    if positive.size > 0:
        low, high = np.percentile(positive, [1, 99])
        data_range = float(high - low)
    else:
        data_range = float(reference.max() - reference.min())
    if data_range <= 0:
        return float("nan"), float("nan")

    bbox_mean, ssim_map = structural_similarity(
        reference,
        test,
        data_range=data_range,
        channel_axis=None,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        full=True,
    )
    return float(ssim_map[inside].mean()), float(bbox_mean)
