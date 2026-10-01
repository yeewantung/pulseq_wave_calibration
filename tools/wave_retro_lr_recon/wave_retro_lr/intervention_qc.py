"""Fixed-normalization review figures for Wave-MPRAGE intervention arms.

Every arm is compared with the accepted baseline under one rule set:

* Magnitudes are restored to the shared BART scale by undoing each NIfTI's
  export normalization, recorded as ``MagnitudeNormalization`` in its sidecar.
* One display constant C, the 99.5th percentile of the restored baseline
  inside the central observation box, sets the image window [0, C], the
  magnitude-difference window [-0.2 C, 0.2 C], and the air window [0, 0.1 C]
  for every arm. No per-arm gain is fitted.
* Observation boxes are inclusive voxel ranges of the RAS-stored NIfTI: axis 0
  is R-L, axis 1 is A-P, and axis 2 is S-I, with indices increasing toward R,
  A, and S.
* Montage slices are chosen mechanically: a fixed count at an exactly uniform
  integer step, centred in each range.

No reconstruction runs here, and nothing is written outside the QC directory.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import rovir_feasibility

FORMAT_VERSION = 1
STATUS = "mprage_intervention_review_written"
TOOL_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = ("wave_retro_lr/intervention_qc.py", "scripts/mprage_intervention_qc.py")
AXIS_CODES = ("R", "A", "S")
NORMALIZATION_PERCENTILE = 99.5
DIFFERENCE_FRACTION = 0.2
AIR_FRACTION = 0.1
ORIENTATIONS = ("axial", "sagittal", "coronal")
# The slice axis of each orientation, and the in-plane axes shown as (x, y).
_SLICE_AXIS = {"axial": 2, "sagittal": 0, "coronal": 1}
_PLANE_AXES = {"axial": (0, 1), "sagittal": (1, 2), "coronal": (0, 2)}
_AXIS_NAMES = ("R-L index (toward R)", "A-P index (toward A)", "S-I index (toward S)")

Box = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def parse_box(text: str, shape: Sequence[int] | None = None) -> Box:
    """Parse an inclusive ``r0:r1,a0:a1,s0:s1`` voxel box.

    Args:
        text: Box text in RAS-stored index space.
        shape: Optional volume shape the box must fit.

    Returns:
        Three inclusive ``(low, high)`` ranges.

    Raises:
        ValueError: If the text is malformed, a range is reversed, or the box
            leaves the volume.
    """
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(f"A box needs three ranges r0:r1,a0:a1,s0:s1; found {text!r}.")
    ranges = []
    for axis, part in enumerate(parts):
        try:
            low, high = (int(value) for value in part.split(":"))
        except ValueError as exc:
            raise ValueError(f"Range {part!r} is not an integer low:high pair.") from exc
        if low > high or low < 0 or (shape is not None and high >= shape[axis]):
            raise ValueError(f"Range {part!r} is reversed or outside axis {axis} of shape {shape}.")
        ranges.append((low, high))
    return tuple(ranges)  # type: ignore[return-value]


def uniform_indices(low: int, high: int, count: int = 5) -> list[int]:
    """Choose slice indices at an exactly uniform integer step.

    The step is ``(high - low) // (count - 1)``, and the pattern is centred
    in the inclusive range, so both ends are included when the span divides
    evenly.

    Args:
        low: Lowest allowed index.
        high: Highest allowed index.
        count: Number of slices.

    Returns:
        ``count`` increasing indices.

    Raises:
        ValueError: If the range cannot hold ``count`` distinct slices.
    """
    if count < 2 or high - low < count - 1:
        raise ValueError(f"Range {low}-{high} cannot hold {count} uniformly spaced slices.")
    step = (high - low) // (count - 1)
    start = low + ((high - low) - step * (count - 1)) // 2
    return [start + step * index for index in range(count)]


def find_magnitude(path: str | Path) -> Path:
    """Return one magnitude NIfTI, given its file or a directory holding it.

    Args:
        path: ``*_part-mag_*.nii.gz`` file or a directory searched recursively.

    Returns:
        Resolved NIfTI path.

    Raises:
        FileNotFoundError: If no magnitude NIfTI is found.
        ValueError: If a directory holds more than one.
    """
    source = Path(path).expanduser().resolve()
    if source.is_file():
        return source
    matches = sorted(source.rglob("*_part-mag_*.nii.gz")) if source.is_dir() else []
    if not matches:
        raise FileNotFoundError(f"No magnitude NIfTI found at {source}")
    if len(matches) > 1:
        raise ValueError(f"Expected one magnitude NIfTI under {source}; found {len(matches)}.")
    return matches[0]


def load_restored_magnitude(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a magnitude NIfTI and undo its export normalization.

    Args:
        path: Magnitude NIfTI file or a directory holding exactly one.

    Returns:
        ``(magnitude, affine, record)``: float32 magnitude on the shared BART
        scale, the affine, and file and normalization provenance.

    Raises:
        FileNotFoundError: If the NIfTI or its sidecar is missing.
        ValueError: If the sidecar normalization is absent, clipped, or not the
            positive-percentile method, or the image is not RAS-stored.
    """
    import nibabel as nib

    nifti = find_magnitude(path)
    sidecar = nifti.with_name(nifti.name.removesuffix(".nii.gz") + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"The magnitude sidecar is missing: {sidecar}")
    normalization = json.loads(sidecar.read_text(encoding="utf-8")).get("MagnitudeNormalization")
    if (
        not isinstance(normalization, dict)
        or normalization.get("Method") != "positive-finite-percentile"
        or normalization.get("Clipped") is not False
        or not float(normalization.get("InputPercentileValue", 0)) > 0
        or not float(normalization.get("OutputPercentileValue", 0)) > 0
    ):
        raise ValueError(f"The sidecar does not record an invertible magnitude normalization: {sidecar}")
    image = nib.load(str(nifti))
    if tuple(nib.aff2axcodes(image.affine)) != AXIS_CODES:
        raise ValueError(f"The magnitude NIfTI is not RAS-stored: {nifti}")
    scale = float(normalization["InputPercentileValue"]) / float(normalization["OutputPercentileValue"])
    magnitude = np.asarray(image.dataobj, dtype=np.float32) * np.float32(scale)
    if magnitude.ndim != 3 or not np.isfinite(magnitude).all():
        raise ValueError(f"The magnitude NIfTI must be a finite 3-D volume: {nifti}")
    record = {
        "file": rovir_feasibility._file_record(nifti),
        "sidecar": rovir_feasibility._file_record(sidecar),
        "magnitude_normalization": normalization,
        "restore_scale": scale,
    }
    return magnitude, np.asarray(image.affine, dtype=np.float64), record


def require_same_geometry(
    baseline_shape: Sequence[int],
    baseline_affine: np.ndarray,
    candidate_shape: Sequence[int],
    candidate_affine: np.ndarray,
) -> None:
    """Require the candidate to share the baseline voxel grid exactly.

    Args:
        baseline_shape: Baseline shape.
        baseline_affine: Baseline affine.
        candidate_shape: Candidate shape.
        candidate_affine: Candidate affine.

    Raises:
        ValueError: If shapes differ or affines differ by more than 1e-4 mm.
    """
    if tuple(baseline_shape) != tuple(candidate_shape):
        raise ValueError(f"Candidate shape {tuple(candidate_shape)} differs from baseline {tuple(baseline_shape)}.")
    if not np.allclose(baseline_affine, candidate_affine, rtol=0.0, atol=1e-4):
        raise ValueError("The candidate affine differs from the baseline affine.")


def _box_view(volume: np.ndarray, box: Box) -> np.ndarray:
    """Return the voxels of an inclusive box.

    Args:
        volume: RAS-stored volume.
        box: Inclusive ranges.

    Returns:
        View of the box.
    """
    return volume[tuple(slice(low, high + 1) for low, high in box)]


def display_constants(baseline: np.ndarray, central: Box) -> dict[str, Any]:
    """Derive the shared display windows from the restored baseline.

    Args:
        baseline: Restored baseline magnitude.
        central: Central observation box.

    Returns:
        The constant C and the image, difference, and air windows.
    """
    constant = float(np.percentile(_box_view(baseline, central), NORMALIZATION_PERCENTILE))
    return {
        "definition": (
            f"C = {NORMALIZATION_PERCENTILE}th percentile of the restored baseline magnitude in the "
            "central box; identical for every arm"
        ),
        "constant": constant,
        "image_window": [0.0, constant],
        "difference_window": [-DIFFERENCE_FRACTION * constant, DIFFERENCE_FRACTION * constant],
        "air_window": [0.0, AIR_FRACTION * constant],
    }


def box_statistics(baseline: np.ndarray, candidate: np.ndarray, box: Box) -> dict[str, float]:
    """Describe baseline and candidate magnitudes inside one box.

    Args:
        baseline: Restored baseline magnitude.
        candidate: Restored candidate magnitude.
        box: Inclusive ranges.

    Returns:
        Medians, 99.5th percentiles, their ratio, and the relative RMS
        difference. These numbers set no pass or fail criterion.
    """
    base = _box_view(baseline, box).astype(np.float64)
    cand = _box_view(candidate, box).astype(np.float64)
    base_median = float(np.median(base))
    return {
        "voxels": int(base.size),
        "baseline_median": base_median,
        "candidate_median": float(np.median(cand)),
        "median_ratio": float(np.median(cand) / base_median) if base_median > 0 else float("nan"),
        "baseline_p99_5": float(np.percentile(base, 99.5)),
        "candidate_p99_5": float(np.percentile(cand, 99.5)),
        "relative_rms_difference": float(np.linalg.norm(cand - base) / np.linalg.norm(base)),
    }


def air_statistics(volume: np.ndarray, si_min: int) -> dict[str, float]:
    """Describe the noise floor in the air band above the vertex.

    Args:
        volume: Restored magnitude.
        si_min: Lowest S-I index of the air band.

    Returns:
        Mean, standard deviation, and 99th percentile over the band.
    """
    band = volume[:, :, si_min:].astype(np.float64)
    return {
        "voxels": int(band.size),
        "mean": float(band.mean()),
        "std": float(band.std()),
        "p99": float(np.percentile(band, 99)),
    }


def _plane(volume: np.ndarray, orientation: str, index: int) -> np.ndarray:
    """Extract one display plane with its in-plane axes as (y, x).

    Args:
        volume: RAS-stored volume.
        orientation: ``axial``, ``sagittal``, or ``coronal``.
        index: Slice index along the orientation's slice axis.

    Returns:
        2-D array indexed ``[y, x]``.
    """
    plane = np.take(volume, index, axis=_SLICE_AXIS[orientation])
    return plane.T


def _render(
    rows: Sequence[tuple[str, np.ndarray, tuple[float, float], str]],
    orientation: str,
    indices: Sequence[int],
    crop: tuple[tuple[int, int], tuple[int, int]] | None,
    title: str,
    path: Path,
) -> dict[str, Any]:
    """Render one montage with fixed windows and write it as PNG.

    Args:
        rows: ``(label, volume, window, colormap)`` per montage row.
        orientation: Plane orientation.
        indices: Slice indices, one montage column each.
        crop: In-plane ``((x0, x1), (y0, y1))`` inclusive ranges, or ``None``
            for the full field of view.
        title: Figure title.
        path: PNG destination.

    Returns:
        File record of the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_axis, y_axis = _PLANE_AXES[orientation]
    figure, axes = plt.subplots(len(rows), len(indices), figsize=(2.6 * len(indices), 2.7 * len(rows)), squeeze=False)
    for row, (label, volume, window, colormap) in enumerate(rows):
        for column, index in enumerate(indices):
            plane = _plane(volume, orientation, index)
            if crop is None:
                (x0, x1), (y0, y1) = (0, plane.shape[1] - 1), (0, plane.shape[0] - 1)
            else:
                (x0, x1), (y0, y1) = crop
            axis = axes[row][column]
            axis.imshow(
                plane[y0 : y1 + 1, x0 : x1 + 1],
                cmap=colormap,
                vmin=window[0],
                vmax=window[1],
                origin="lower",
                extent=(x0 - 0.5, x1 + 0.5, y0 - 0.5, y1 + 0.5),
                interpolation="nearest",
            )
            axis.set_title(f"{label} · {orientation} {index}", fontsize=7)
            axis.tick_params(labelsize=6)
            if row == len(rows) - 1:
                axis.set_xlabel(_AXIS_NAMES[x_axis], fontsize=6)
            if column == 0:
                axis.set_ylabel(_AXIS_NAMES[y_axis], fontsize=6)
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=90)
    plt.close(figure)
    return rovir_feasibility._file_record(path)


def _figure_plan(
    shape: Sequence[int], central: Box, metal: Box, air_si_min: int, count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fix every montage's slices and crop before anything is drawn.

    Residual Wave spreading appears on the middle sagittal slices, at the
    anterior face edge and the posterior occiput/head edge. The edge strips are
    the A-P ranges outside the central box (anterior face above it, posterior
    occiput/head below it) over the full S-I extent. They are shown on the five
    sagittal views that span the central R-L range.

    Args:
        shape: Volume shape.
        central: Central observation box.
        metal: Metal-context box.
        air_si_min: Lowest S-I index of the air band.
        count: Slices per montage.

    Returns:
        ``(slices, figures)``: recorded slice indices, and per figure its
        orientation, indices, crop, window name, and title.
    """
    slices: dict[str, Any] = {}
    figures: dict[str, Any] = {}
    for name, box in (("central", central), ("metal", metal)):
        slices[name] = {}
        for orientation in ORIENTATIONS:
            axis = _SLICE_AXIS[orientation]
            indices = uniform_indices(*box[axis], count)
            slices[name][orientation] = indices
            x_axis, y_axis = _PLANE_AXES[orientation]
            for view, crop in (("full", None), ("crop", (box[x_axis], box[y_axis]))):
                figures[f"{name}_{orientation}_{view}"] = {
                    "orientation": orientation,
                    "indices": indices,
                    "crop": crop,
                    "window": "image",
                    "title": f"{name} box, {orientation}, {'full field of view' if crop is None else 'cropped to the box'}",
                }
    air_indices = uniform_indices(air_si_min, shape[2] - 1, count)
    slices["air"] = {"axial": air_indices}
    figures["air_axial_full"] = {
        "orientation": "axial",
        "indices": air_indices,
        "crop": None,
        "window": "air",
        "title": f"air reference, S-I >= {air_si_min}, air window",
    }
    (a_low, a_high) = central[1]
    edges = {
        "anterior": (a_high + 1, shape[1] - 1),
        "posterior": (0, a_low - 1),
    }
    edge_names = {"anterior": "anterior face edge", "posterior": "posterior occiput/head edge"}
    slices["edges"] = {"sagittal": slices["central"]["sagittal"], "a_p_strips": edges}
    for side, strip in edges.items():
        if strip[0] > strip[1]:
            raise ValueError(f"The {side} edge strip is empty for central A-P range {central[1]}.")
        figures[f"edges_sagittal_{side}"] = {
            "orientation": "sagittal",
            "indices": slices["central"]["sagittal"],
            "crop": (strip, (0, shape[2] - 1)),
            "window": "image",
            "title": (
                f"{edge_names[side]} (A-P {strip[0]}-{strip[1]}), sagittal views across the central "
                "R-L range, full S-I"
            ),
        }
    return slices, figures


def implementation_identity() -> dict[str, str]:
    """Hash the files that define these review figures.

    Returns:
        Tool-relative path mapped to SHA-256.
    """
    return {
        relative: hashlib.sha256((TOOL_ROOT / relative).read_bytes()).hexdigest()
        for relative in IMPLEMENTATION_FILES
    }


def write_review(
    baseline_path: str | Path,
    candidate_path: str | Path,
    candidate_label: str,
    output_dir: str | Path,
    *,
    central: Box,
    metal: Box,
    air_si_min: int,
    count: int = 5,
) -> dict[str, Any]:
    """Write the fixed-normalization review figures of one arm.

    Args:
        baseline_path: Accepted baseline magnitude NIfTI or its directory.
        candidate_path: Arm magnitude NIfTI or its directory.
        candidate_label: Short arm label for figure rows.
        output_dir: QC directory of the arm.
        central: Central observation box.
        metal: Metal-context box.
        air_si_min: Lowest S-I index of the air reference band.
        count: Slices per montage.

    Returns:
        QC manifest with inputs, geometry, boxes, slices, windows, statistics,
        and figure records.

    Raises:
        FileExistsError: If the QC manifest or a figure it would write exists.
        ValueError: If inputs are invalid, geometries differ, or a box or
            strip does not fit the volume.

    Side Effects:
        Writes PNG figures under ``output_dir/figures`` and
        ``output_dir/qc_manifest.json``.
    """
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "qc_manifest.json"
    baseline, baseline_affine, baseline_record = load_restored_magnitude(baseline_path)
    candidate, candidate_affine, candidate_record = load_restored_magnitude(candidate_path)
    require_same_geometry(baseline.shape, baseline_affine, candidate.shape, candidate_affine)
    for box in (central, metal):
        parse_box(",".join(f"{low}:{high}" for low, high in box), baseline.shape)
    if not 0 <= air_si_min < baseline.shape[2]:
        raise ValueError(f"The air band start {air_si_min} lies outside S-I 0-{baseline.shape[2] - 1}.")
    slices, plan = _figure_plan(baseline.shape, central, metal, air_si_min, count)
    targets = {name: output / "figures" / f"{name}.png" for name in plan}
    existing = [str(path) for path in (manifest_path, *targets.values()) if path.exists()]
    if existing:
        raise FileExistsError("Review outputs already exist; move them aside before rerunning: " + ", ".join(existing))

    constants = display_constants(baseline, central)
    difference = candidate - baseline
    windows = {
        "image": tuple(constants["image_window"]),
        "air": tuple(constants["air_window"]),
    }
    figures = {}
    for name, spec in plan.items():
        window = windows[spec["window"]]
        rows = (
            ("baseline", baseline, window, "gray"),
            (candidate_label, candidate, window, "gray"),
            ("candidate - baseline", difference, tuple(constants["difference_window"]), "RdBu_r"),
        )
        figures[name] = _render(rows, spec["orientation"], spec["indices"], spec["crop"], spec["title"], targets[name])
    payload = {
        "format_version": FORMAT_VERSION,
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": implementation_identity(),
        "coordinates": (
            "inclusive voxel indices of the RAS-stored NIfTI: axis 0 R-L, axis 1 A-P, axis 2 S-I, "
            "increasing toward R, A, and S"
        ),
        "baseline": {"label": "accepted baseline", **baseline_record},
        "candidate": {"label": candidate_label, **candidate_record},
        "geometry": {"shape": list(baseline.shape), "affine": baseline_affine.tolist(), "axis_codes": list(AXIS_CODES)},
        "boxes": {"central": [list(r) for r in central], "metal": [list(r) for r in metal], "air_si_min": air_si_min},
        "slice_rule": f"{count} slices at an exactly uniform integer step, centred in each inclusive range",
        "slices": slices,
        "normalization": constants,
        "statistics": {
            "central": box_statistics(baseline, candidate, central),
            "metal": box_statistics(baseline, candidate, metal),
            "air": {"baseline": air_statistics(baseline, air_si_min), "candidate": air_statistics(candidate, air_si_min)},
        },
        "figures": figures,
    }
    rovir_feasibility._write_json(manifest_path, payload)
    return payload
