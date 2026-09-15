"""Cross-dataset metric charts, one grouped bar panel per metric and mask.

Each chart averages one metric over every subject and contrast that was
scored, grouped by retrospective case along the x axis and by
regularization branch and SR stage within each group, so the effect of
super-resolution reads directly against the Pre-SR bar beside it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .deck import BRANCH_LABELS, STAGE_LABELS, case_label
from .discovery import BRANCH_ORDER, SR_MODEL_ORDER, VARIANT_ORDER


#: Metric column, axis label and whether lower is better.
METRIC_PANELS: tuple[tuple[str, str, bool], ...] = (
    ("nrmse", "NRMSE", True),
    ("ssim_3d_mask", "3D SSIM", False),
    ("psnr_p99_db", "PSNR p99 (dB)", False),
)


def condition_order() -> list[tuple[str, str, str]]:
    """Bar order within each retrospective case.

    Returns:
        ``(branch, stage, sr_model)`` triples, Pre-SR first in each branch.
    """
    conditions: list[tuple[str, str, str]] = []
    for branch in BRANCH_ORDER:
        conditions.append((branch, "Pre-SR", ""))
        for model in SR_MODEL_ORDER:
            conditions.append((branch, "Post-SR", model))
    return conditions


def condition_label(branch: str, stage: str, sr_model: str) -> str:
    """Legend wording for one bar.

    Args:
        branch: Regularization branch.
        stage: ``Pre-SR`` or ``Post-SR``.
        sr_model: SR model for ``Post-SR``, otherwise empty.

    Returns:
        The legend entry, using the deck's own wording.
    """
    stage_key = sr_model if stage == "Post-SR" else stage
    return f"{BRANCH_LABELS.get(branch, branch)}, {STAGE_LABELS.get(stage_key, stage_key)}"


def write_summary_charts(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    mask_mode: str,
    dpi: int = 150,
) -> list[tuple[str, Path]]:
    """Write one grouped bar chart per metric for a single mask mode.

    Args:
        rows: Metric rows, as written to the CSV.
        output_dir: Folder receiving the charts.
        mask_mode: Mask whose rows are charted.
        dpi: Raster resolution.

    Returns:
        ``(slide title, chart path)`` pairs in metric order, empty when the
        mask has no rows.
    """
    selected = [row for row in rows if row["mask_mode"] == mask_mode]
    if not selected:
        return []

    subjects = sorted({str(row["subject"]) for row in selected})
    contrasts = sorted({str(row["contrast"]) for row in selected})
    variants = [
        variant
        for variant in VARIANT_ORDER
        if any(row["variant"] == variant for row in selected)
    ]
    conditions = [
        condition
        for condition in condition_order()
        if any(_matches(row, condition) for row in selected)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, Path]] = []

    for column, axis_label, lower_is_better in METRIC_PANELS:
        figure, axis = plt.subplots(figsize=(11.0, 4.6))
        width = 0.8 / max(len(conditions), 1)

        for index, condition in enumerate(conditions):
            means = [
                _mean(selected, variant, condition, column) for variant in variants
            ]
            offsets = (
                np.arange(len(variants))
                + (index - (len(conditions) - 1) / 2) * width
            )
            axis.bar(offsets, means, width=width, label=condition_label(*condition))

        axis.set_xticks(np.arange(len(variants)))
        axis.set_xticklabels(
            [case_label(variant, (1.0, 1.0, 1.0)) for variant in variants], fontsize=9
        )
        direction = " (lower is better)" if lower_is_better else ""
        axis.set_ylabel(axis_label + direction)
        axis.set_title(
            f"{axis_label}{direction} in the {mask_mode} mask — mean over "
            f"{len(subjects)} subjects, {len(contrasts)} contrasts"
        )
        axis.legend(fontsize=8, ncol=2)
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()

        path = output_dir / f"summary_{mask_mode}_{column}.png"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)

        written.append((f"{axis_label} — all datasets, {mask_mode} mask", path))

    return written


def _matches(row: Mapping[str, object], condition: tuple[str, str, str]) -> bool:
    """Whether a metric row belongs to one legend condition."""
    branch, stage, sr_model = condition
    return (
        row["branch"] == branch
        and row["stage"] == stage
        and (row["sr_model"] or "") == sr_model
    )


def _mean(
    rows: Sequence[Mapping[str, object]],
    variant: str,
    condition: tuple[str, str, str],
    column: str,
) -> float:
    """Mean of one metric over every subject and contrast of a bar."""
    values = [
        float(row[column])
        for row in rows
        if row["variant"] == variant and _matches(row, condition)
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")
