"""Cut one HD-BET brain mask per baseline, with a figure for visual review.

HD-BET writes its mask on the grid of the input it was given, and the input here
is the fully sampled R3x1 baseline every metric is measured against, so the mask
lands on exactly the grid the metrics use. That is asserted rather than assumed:
a mask on a different shape or affine is rejected outright.

HD-BET's output is streamed to the caller's terminal rather than captured. The
first run downloads roughly 110 MB of weights before predicting anything, and on
a slow link that download dominates the wall time; without streaming it looks
indistinguishable from a hang.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from .layout import MaskTarget, sha256_file, write_sidecar


#: Devices selectable on the command line; ``auto`` picks the best available.
DEVICES: tuple[str, ...] = ("auto", "mps", "cpu", "cuda")

#: Preference order when resolving ``auto``, and when falling back.
DEVICE_PREFERENCE: tuple[str, ...] = ("cuda", "mps", "cpu")

#: Array axis sliced for each orientation on a canonical RAS grid.
ORIENTATION_AXES = {"sagittal": 0, "coronal": 1, "axial": 2}

#: Offsets from the mask centroid shown in each QC row.
QC_OFFSETS: tuple[int, ...] = (-32, 0, 32)

#: A plausible adult brain mask, used to catch a collapsed or runaway result.
MIN_MASK_VOXELS = 200_000
MAX_MASK_FRACTION = 0.8


def available_devices() -> tuple[str, ...]:
    """Torch devices this machine can actually predict on.

    CPU is always present. Probing is defensive because an environment without
    torch, or with a torch build lacking a backend, must degrade rather than
    raise: the caller only wants to know what it may ask for.

    Returns:
        The usable devices in :data:`DEVICE_PREFERENCE` order.
    """
    available = {"cpu"}
    try:
        import torch
    except Exception:  # pragma: no cover - torch is an HD-BET dependency
        return ("cpu",)

    try:
        if torch.cuda.is_available():
            available.add("cuda")
    except Exception:  # pragma: no cover - backend probing is best effort
        pass
    try:
        if torch.backends.mps.is_available():
            available.add("mps")
    except Exception:  # pragma: no cover - backend probing is best effort
        pass

    return tuple(name for name in DEVICE_PREFERENCE if name in available)


def resolve_device(requested: str, *, log: Callable[[str], None] = print) -> str:
    """Pick the device to predict on, falling back when one is unavailable.

    ``auto`` takes the best device this machine has. An explicit request that
    the machine cannot honour falls back to the best available rather than
    failing, so the same command works on an Apple silicon laptop, a CUDA box
    and a plain CPU server. The fallback is always announced.

    Args:
        requested: One of :data:`DEVICES`.
        log: Progress sink.

    Returns:
        The device to hand to HD-BET.

    Raises:
        ValueError: If ``requested`` is not a known device.
    """
    if requested not in DEVICES:
        raise ValueError(f"Unknown device: {requested!r}")

    available = available_devices()
    if requested == "auto":
        chosen = available[0]
        log(f"Device: {chosen} (auto; available: {', '.join(available)})")
        return chosen

    if requested in available:
        return requested

    fallback = available[0]
    log(
        f"[warn] device {requested!r} is not available on this machine "
        f"(available: {', '.join(available)}); falling back to {fallback!r}"
    )
    return fallback


@dataclass(frozen=True)
class MaskResult:
    """What one generated mask came out as.

    Attributes:
        target: The subject/contrast the mask belongs to.
        voxel_count: Mask voxels.
        volume_ml: Mask volume in millilitres.
    """

    target: MaskTarget
    voxel_count: int
    volume_ml: float


def hd_bet_version() -> str:
    """Installed HD-BET version, or ``unknown`` when it cannot be read."""
    try:
        import importlib.metadata as metadata

        return metadata.version("HD-BET")
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def generate_mask(
    target: MaskTarget,
    *,
    executable: str,
    device: str = "auto",
    disable_tta: bool = True,
    verbose: bool = False,
    log: Callable[[str], None] = print,
) -> MaskResult:
    """Run HD-BET on one baseline and install its mask, sidecar and QC figure.

    Args:
        target: The subject/contrast to mask.
        executable: Resolved ``hd-bet`` path.
        device: Torch device HD-BET predicts on; already resolved by
            :func:`resolve_device`.
        disable_tta: Skip test-time augmentation; roughly eight times faster.
        verbose: Ask HD-BET for verbose progress.
        log: Progress sink.

    Returns:
        What the generated mask came out as.

    Raises:
        RuntimeError: If HD-BET fails, or writes a mask this tool cannot accept.
    """
    target.masks_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        # HD-BET names its output after the input stem, so give it a fixed one
        # and keep the shared reconstruction tree read-only.
        staged = scratch_dir / "baseline.nii.gz"
        shutil.copyfile(target.baseline, staged)

        command = [
            executable,
            "-i",
            str(staged),
            "-o",
            str(scratch_dir / "stripped.nii.gz"),
            "-device",
            device,
            "--save_bet_mask",
            "--no_bet_image",
        ]
        if disable_tta:
            command.append("--disable_tta")
        if verbose:
            command.append("--verbose")

        log(f"      $ {' '.join(command)}")
        # Streamed, not captured: the first run's weight download and the
        # inference progress both need to be visible while they happen.
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"HD-BET failed for {target.subject} {target.contrast} with status "
                f"{completed.returncode}"
            )

        shutil.copyfile(_produced_mask(scratch_dir), target.mask_path)

    baseline_image = nib.load(str(target.baseline))
    mask = _validated_mask(target, baseline_image)

    baseline_data = np.asarray(baseline_image.get_fdata(dtype=np.float32))
    baseline_data = np.abs(baseline_data)
    baseline_data[~np.isfinite(baseline_data)] = 0.0
    write_qc_figure(baseline_data, mask, target)

    voxel_volume = float(np.prod(baseline_image.header.get_zooms()[:3]))
    voxel_count = int(mask.sum())
    volume_ml = voxel_count * voxel_volume / 1000.0

    write_sidecar(
        target,
        generator={
            "tool": "HD-BET",
            "version": hd_bet_version(),
            "executable": executable,
            "device": device,
            "test_time_augmentation": not disable_tta,
        },
        baseline_sha256=sha256_file(target.baseline),
        baseline_shape=tuple(baseline_image.shape),
        baseline_voxel_mm=tuple(
            float(value) for value in baseline_image.header.get_zooms()[:3]
        ),
        voxel_count=voxel_count,
        volume_ml=volume_ml,
    )
    return MaskResult(target=target, voxel_count=voxel_count, volume_ml=volume_ml)


def _produced_mask(scratch: Path) -> Path:
    """Locate the mask HD-BET wrote in its scratch folder.

    Args:
        scratch: Temporary folder HD-BET wrote into.

    Returns:
        The mask NIfTI.

    Raises:
        RuntimeError: If no mask, or more than one, was produced.
    """
    candidates = sorted(scratch.glob("*bet*.nii.gz")) or sorted(
        path for path in scratch.glob("*.nii.gz") if path.name != "baseline.nii.gz"
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one HD-BET mask in {scratch}, found: "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def _validated_mask(target: MaskTarget, baseline_image: nib.Nifti1Image) -> np.ndarray:
    """Check a mask shares the baseline grid and is anatomically plausible.

    Args:
        target: The subject/contrast the mask belongs to.
        baseline_image: Baseline defining the required grid.

    Returns:
        The boolean mask.

    Raises:
        RuntimeError: If the grid differs or the mask size is implausible.
    """
    image = nib.load(str(target.mask_path))
    if image.shape != baseline_image.shape or not np.allclose(
        image.affine, baseline_image.affine, atol=1e-5
    ):
        raise RuntimeError(
            f"HD-BET mask for {target.subject} {target.contrast} is not on the "
            f"baseline grid: {image.shape} vs {baseline_image.shape}"
        )

    mask = np.asarray(image.dataobj) > 0.5
    voxels = int(mask.sum())
    if not MIN_MASK_VOXELS < voxels < int(MAX_MASK_FRACTION * mask.size):
        raise RuntimeError(
            f"HD-BET mask for {target.subject} {target.contrast} is implausible: "
            f"{voxels} voxels of {mask.size}"
        )
    return mask


def write_qc_figure(
    baseline: np.ndarray, mask: np.ndarray, target: MaskTarget
) -> Path:
    """Render a three-by-three mask-boundary figure for visual review.

    Rows are the three orientations, columns step away from the mask centroid,
    and every panel is labelled with anatomical directions so a left/right flip
    cannot slip through unnoticed.

    Args:
        baseline: Baseline magnitude samples.
        mask: Brain mask on the same grid.
        target: The subject/contrast being reviewed.

    Returns:
        The QC figure path.
    """
    centroid = np.rint(np.argwhere(mask).mean(axis=0)).astype(int)
    display_max = float(np.percentile(baseline[mask], 99.5))

    figure, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
    for row, orientation in enumerate(("sagittal", "coronal", "axial")):
        axis_index = ORIENTATION_AXES[orientation]
        for column, offset in enumerate(QC_OFFSETS):
            index = int(
                np.clip(centroid[axis_index] + offset, 0, mask.shape[axis_index] - 1)
            )
            cell = axes[row, column]
            cell.imshow(
                _plane(baseline, orientation, index),
                cmap="gray",
                origin="lower",
                vmin=0.0,
                vmax=display_max,
            )
            outline = _plane(mask, orientation, index)
            if outline.any() and not outline.all():
                cell.contour(outline, levels=[0.5], colors="#00ff66", linewidths=0.8)
            cell.set_title(f"{orientation}, index {index}", fontsize=9)
            cell.set_axis_off()
            _label_directions(cell, orientation)

    figure.suptitle(
        f"{target.subject} {target.contrast} — HD-BET boundary (green) "
        f"on the R3x1 baseline",
        fontsize=14,
    )
    figure.savefig(target.qc_path, dpi=150)
    plt.close(figure)
    return target.qc_path


def _plane(volume: np.ndarray, orientation: str, index: int) -> np.ndarray:
    """Slice one orientation out of a canonical RAS volume."""
    if orientation == "sagittal":
        return volume[index, :, :].T
    if orientation == "coronal":
        return volume[:, index, :].T
    return volume[:, :, index].T


def _label_directions(axis, orientation: str) -> None:
    """Annotate anatomical directions so left and right cannot be confused."""
    left, right = ("P", "A") if orientation == "sagittal" else ("L", "R")
    bottom, top = ("P", "A") if orientation == "axial" else ("I", "S")
    style = {
        "color": "white",
        "fontsize": 8,
        "weight": "bold",
        "bbox": {"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1},
    }
    axis.text(0.02, 0.5, left, transform=axis.transAxes, va="center", **style)
    axis.text(
        0.98, 0.5, right, transform=axis.transAxes, ha="right", va="center", **style
    )
    axis.text(0.5, 0.02, bottom, transform=axis.transAxes, ha="center", **style)
    axis.text(0.5, 0.98, top, transform=axis.transAxes, ha="center", va="top", **style)


def resolve_executable(name: str) -> str:
    """Resolve the HD-BET executable on PATH.

    Args:
        name: Executable name or path.

    Returns:
        The resolved path.

    Raises:
        FileNotFoundError: If it is not installed.
    """
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(
            f"{name!r} is not on PATH. Install HD-BET into this environment."
        )
    return resolved


def summarize(results: Sequence[MaskResult]) -> str:
    """One line per generated mask, for the run summary.

    Args:
        results: Generated masks.

    Returns:
        The formatted summary.
    """
    return "\n".join(
        f"  {result.target.subject:8s} {result.target.contrast:16s} "
        f"{result.voxel_count:>9d} voxels  {result.volume_ml:7.1f} mL"
        for result in results
    )
