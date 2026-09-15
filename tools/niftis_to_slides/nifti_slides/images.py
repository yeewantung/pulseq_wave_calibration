"""Render three-orientation center-slice strips onto the baseline grid.

Every volume of one subject/contrast is resampled onto the fully sampled R3x1
baseline grid before slicing, so all strips of a section share their pixel
dimensions and anatomy location, and the retrospective low-resolution cases are
shown at the resolution they are compared at.

Strips are written as 16-bit LZW TIFF with no annotation of any kind. A single
display window, taken from the baseline, is shared by every volume of the
section so Pre-SR and Post-SR strips are directly comparable by eye.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from nibabel.processing import resample_from_to


#: Physical orientation of each display panel, left to right.
ORIENTATION_ORDER: tuple[str, ...] = ("sagittal", "coronal", "axial")

#: Array axis sliced for each orientation on a canonical RAS grid.
ORIENTATION_AXES = {"sagittal": 0, "coronal": 1, "axial": 2}

UINT16_MAX = 65535


@dataclass(frozen=True)
class BaselineGrid:
    """Reference geometry and display window shared by one section.

    Attributes:
        image: Baseline NIfTI, defining the grid every volume is resampled onto.
        data: Baseline magnitude samples.
        display_max: Upper end of the shared grayscale window.
    """

    image: nib.Nifti1Image
    data: np.ndarray
    display_max: float


def load_magnitude(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a magnitude NIfTI as non-negative, finite float32 samples.

    Args:
        path: Magnitude NIfTI.

    Returns:
        The loaded image and its sample array.
    """
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    data = np.abs(data)
    data[~np.isfinite(data)] = 0.0
    return image, data


def resample_to_baseline(
    image: nib.Nifti1Image, baseline: nib.Nifti1Image, *, interpolation_order: int = 3
) -> tuple[np.ndarray, bool]:
    """Resample a volume onto the baseline grid when its geometry differs.

    Args:
        image: Volume to place on the baseline grid.
        baseline: Baseline NIfTI defining the target grid.
        interpolation_order: Spline order; 3 is cubic, appropriate for
            upsampling a retrospective low-resolution reconstruction.

    Returns:
        The samples on the baseline grid, and whether resampling was needed.

    Raises:
        RuntimeError: If resampling did not produce the baseline shape.
    """
    same_shape = image.shape == baseline.shape
    same_affine = np.allclose(image.affine, baseline.affine, atol=1e-5)

    if same_shape and same_affine:
        placed = image
        resampled = False
    else:
        placed = resample_from_to(image, baseline, order=interpolation_order)
        resampled = True

    data = np.asarray(placed.get_fdata(dtype=np.float32), dtype=np.float32)
    data = np.abs(data)
    data[~np.isfinite(data)] = 0.0

    if data.shape != baseline.shape:
        raise RuntimeError(
            f"Resampling produced {data.shape}, expected the baseline {baseline.shape}"
        )
    return data, resampled


def build_baseline_grid(
    path: Path, mask: np.ndarray | None, *, display_percentile: float = 99.5
) -> BaselineGrid:
    """Load the baseline and derive the display window shared by its section.

    Args:
        path: Fully sampled R3x1 baseline magnitude NIfTI.
        mask: Foreground mask on the baseline grid, or ``None`` to take the
            window over every positive sample.
        display_percentile: Percentile of the baseline used as the window top.

    Returns:
        The baseline grid and its display window.

    Raises:
        ValueError: If the baseline carries no positive samples.
    """
    image, data = load_magnitude(path)
    samples = data[mask] if mask is not None else data[data > 0]
    if samples.size == 0:
        raise ValueError(f"Baseline has no positive samples: {path}")

    return BaselineGrid(
        image=image,
        data=data,
        display_max=float(np.percentile(samples, display_percentile)),
    )


def center_slices(volume: np.ndarray) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Take the center slice of each orientation in RAS display layout.

    Superior is up for the sagittal and coronal panels, anterior is up for the
    axial panel, matching the presentation TIFF convention used elsewhere in
    this repository.

    Args:
        volume: Magnitude samples on a canonical RAS grid.

    Returns:
        The slice index per orientation and the corresponding display panels.
    """
    indices = {name: volume.shape[axis] // 2 for name, axis in ORIENTATION_AXES.items()}
    panels = {
        "sagittal": np.flip(volume[indices["sagittal"], :, :].T, axis=0),
        "coronal": np.flip(volume[:, indices["coronal"], :].T, axis=0),
        "axial": np.flip(volume[:, :, indices["axial"]].T, axis=0),
    }
    return indices, panels


def orientation_strip(volume: np.ndarray) -> tuple[dict[str, int], np.ndarray]:
    """Butt the three center slices together with no separator.

    On a cubic-field-of-view baseline all three panels share their row count, so
    the strip is a plain horizontal concatenation at native voxel resolution.
    The centered zero padding guards the non-cubic case, where the axial panel
    counts anterior rows while the other two count superior rows.

    Args:
        volume: Magnitude samples on the baseline grid.

    Returns:
        The slice index per orientation and the assembled strip.
    """
    indices, panels = center_slices(volume)
    ordered = [panels[name] for name in ORIENTATION_ORDER]
    height = max(panel.shape[0] for panel in ordered)

    padded = []
    for panel in ordered:
        missing = height - panel.shape[0]
        if missing:
            top = missing // 2
            panel = np.pad(panel, ((top, missing - top), (0, 0)))
        padded.append(panel)

    return indices, np.concatenate(padded, axis=1)


def write_strip_tiff(strip: np.ndarray, display_max: float, path: Path) -> tuple[int, int]:
    """Write one strip as a 16-bit LZW TIFF, windowed and unannotated.

    Args:
        strip: Assembled display strip.
        display_max: Upper end of the shared grayscale window.
        path: Destination TIFF.

    Returns:
        The written ``(width, height)`` in pixels.

    Raises:
        ValueError: If the display window is not positive.
    """
    if display_max <= 0:
        raise ValueError("Display maximum must be positive.")

    scaled = np.clip(strip, 0.0, display_max) / display_max
    pixels = np.rint(scaled * UINT16_MAX).astype(np.uint16)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format="TIFF", compression="tiff_lzw")
    return pixels.shape[1], pixels.shape[0]
