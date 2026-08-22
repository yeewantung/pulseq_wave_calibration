#!/usr/bin/env python3
"""Create a reference-neutral visual and numerical regularization review."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bart_cfl import open_bart_memmap, sha256_file
from checkpoint_io import write_json_atomic


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-zero-manifest", required=True, type=Path)
    parser.add_argument("--sweep-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--regularizer", required=True, choices=("wavelet", "llr"))
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--lambda-labels", required=True, nargs="+")
    parser.add_argument(
        "--qualitative-transfer-only",
        action="store_true",
        help="Label the output as a transfer assessment that cannot select or retune lambda.",
    )
    return parser


def _run_name(regularizer: str, label: str, block_size: int) -> str:
    if regularizer == "wavelet":
        return f"wavelet_lambda-{label}"
    return f"llr_block-{block_size}_lambda-{label}"


def _relative_l2(candidate_base: Path, reference_base: Path) -> float:
    candidate = open_bart_memmap(candidate_base)
    reference = open_bart_memmap(reference_base)
    if candidate.shape != reference.shape:
        raise ValueError(f"BART shapes differ: {candidate.shape} versus {reference.shape}")
    error_squared = 0.0
    reference_squared = 0.0
    for start in range(0, candidate.shape[2], 8):
        stop = min(start + 8, candidate.shape[2])
        current = np.asarray(candidate[:, :, start:stop, ...])
        baseline = np.asarray(reference[:, :, start:stop, ...])
        error_squared += float(np.vdot(current - baseline, current - baseline).real)
        reference_squared += float(np.vdot(baseline, baseline).real)
    if reference_squared <= 0:
        raise ValueError("FISTA lambda-zero reference has zero energy.")
    return float(np.sqrt(error_squared / reference_squared))


def _load_restored_magnitude(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import nibabel as nib

    niftis = sorted(run_dir.glob("nifti/**/*part-mag*.nii.gz"))
    if len(niftis) != 1:
        raise ValueError(f"Expected one magnitude NIfTI under {run_dir}, found {niftis}")
    nifti_path = niftis[0]
    sidecar_path = Path(str(nifti_path).removesuffix(".nii.gz") + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    scale = float(sidecar["MagnitudeNormalization"]["InputPercentileValue"])
    image = nib.load(str(nifti_path))
    data = np.asarray(image.dataobj, dtype=np.float32) * np.float32(scale)
    if image.shape != (256, 256, 256) or not np.isfinite(data).all():
        raise ValueError(f"Invalid magnitude NIfTI: {nifti_path}")
    if tuple(__import__("nibabel").aff2axcodes(image.affine)) != ("R", "A", "S"):
        raise ValueError(f"Magnitude NIfTI is not canonical RAS: {nifti_path}")
    return data, np.asarray(image.affine), {
        "nifti": str(nifti_path),
        "nifti_sha256": sha256_file(nifti_path),
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "normalization_undone_with_input_percentile": scale,
    }


def _add_orientation_labels(axis: Any, *, view: str) -> None:
    if view == "coronal":
        labels = (
            (0.02, 0.5, "L"),
            (0.98, 0.5, "R"),
            (0.5, 0.98, "S"),
            (0.5, 0.02, "I"),
        )
    else:
        labels = (
            (0.02, 0.5, "L"),
            (0.98, 0.5, "R"),
            (0.5, 0.98, "A"),
            (0.5, 0.02, "P"),
        )
    for x, y, label in labels:
        axis.text(
            x,
            y,
            label,
            transform=axis.transAxes,
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    lambda_zero_manifest = args.lambda_zero_manifest.expanduser().resolve()
    sweep_root = args.sweep_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    labels = tuple(args.lambda_labels)
    if not labels or labels[0] != "0":
        raise ValueError("The first lambda label must be 0 for the matched reference.")
    if args.regularizer == "llr" and args.block_size < 1:
        raise ValueError("LLR block size must be positive.")
    regularizer_title = (
        "Wavelet" if args.regularizer == "wavelet" else f"LLR block {args.block_size}"
    )
    output_prefix = (
        "wavelet" if args.regularizer == "wavelet" else f"llr_block-{args.block_size}"
    )
    accepted = json.loads(lambda_zero_manifest.read_text(encoding="utf-8"))
    if float(accepted.get("config", {}).get("ecalib_crop", -1)) != 0.6:
        raise ValueError("Review requires the approved crop-0.6 lambda zero.")
    source_hash = sha256_file(lambda_zero_manifest)

    records = []
    volumes = []
    common_affine: np.ndarray | None = None
    fista_zero_base = (
        sweep_root / _run_name(args.regularizer, "0", args.block_size) / "bart" / "image_wave"
    )
    for label in labels:
        run_dir = sweep_root / _run_name(args.regularizer, label, args.block_size)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete regularization case: {manifest_path}")
        if manifest.get("config", {}).get("regularizer") != args.regularizer:
            raise ValueError(f"Regularizer/config mismatch: {manifest_path}")
        if manifest.get("config", {}).get("lambda_label") != label:
            raise ValueError(f"Lambda label/config mismatch: {manifest_path}")
        if args.regularizer == "llr" and manifest.get("config", {}).get(
            "block_size"
        ) != args.block_size:
            raise ValueError(f"LLR block-size/config mismatch: {manifest_path}")
        if manifest.get("source_provenance", {}).get("lambda_zero_manifest", {}).get(
            "sha256"
        ) != source_hash:
            raise ValueError(
                f"Case does not use the approved lambda zero: {manifest_path}"
            )
        volume, affine, nifti_record = _load_restored_magnitude(run_dir)
        if common_affine is None:
            common_affine = affine
        elif not np.allclose(affine, common_affine, atol=1e-5):
            raise ValueError(f"Regularization NIfTI affine mismatch: {manifest_path}")
        relative_l2 = _relative_l2(
            run_dir / "bart" / "image_wave", fista_zero_base
        )
        records.append(
            {
                "lambda": float(manifest["config"]["lambda"]),
                "lambda_label": label,
                "run_manifest": str(manifest_path),
                "run_manifest_sha256": sha256_file(manifest_path),
                "relative_l2_from_fista_lambda_zero": relative_l2,
                "nifti": nifti_record,
            }
        )
        volumes.append(volume)

    positive = volumes[0][volumes[0] > 0]
    display_window = [0.0, float(np.percentile(positive, 99.5))]
    figure, axes = plt.subplots(2, len(volumes), figsize=(18, 8.0), squeeze=False)
    for column, (label, volume) in enumerate(zip(labels, volumes)):
        planes = (
            volume[:, volume.shape[1] // 2, :],
            volume[:, :, volume.shape[2] // 2],
        )
        for row, (view, plane) in enumerate(zip(("coronal", "axial"), planes)):
            axis = axes[row, column]
            axis.imshow(
                plane.T,
                cmap="gray",
                origin="lower",
                vmin=display_window[0],
                vmax=display_window[1],
            )
            axis.set_title(
                ("FISTA λ=0" if label == "0" else f"{regularizer_title} λ={label}")
                + f"\n{view}"
            )
            _add_orientation_labels(axis, view=view)
            axis.axis("off")
    review_title = (
        f"R3 transfer of frozen {regularizer_title} — qualitative common window"
        if args.qualitative_transfer_only
        else f"R1 synthetic R3×2 {regularizer_title} sweep — common FISTA-λ=0 intensity window"
    )
    figure.suptitle(review_title)
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = output_dir / f"{output_prefix}_sweep_common_window.png"
    figure.savefig(comparison, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    positive_records = records[1:]
    axis.semilogx(
        [record["lambda"] for record in positive_records],
        [record["relative_l2_from_fista_lambda_zero"] for record in positive_records],
        marker="o",
    )
    axis.set_xlabel(f"{regularizer_title} lambda")
    axis.set_ylabel("Relative L2 from FISTA lambda zero")
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    difference_plot = output_dir / f"{output_prefix}_sweep_relative_l2.png"
    figure.savefig(difference_plot, dpi=150)
    plt.close(figure)

    report = {
        "format_version": 1,
        "status": (
            "awaiting_qualitative_transfer_assessment"
            if args.qualitative_transfer_only
            else "awaiting_visual_selection"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            f"reference-neutral {regularizer_title} "
            + (
                "cross-dataset transfer assessment; lambda is frozen and cannot be retuned"
                if args.qualitative_transfer_only
                else "review; no DICOM, BET, or candidate ranking"
            )
        ),
        "qualitative_transfer_only": args.qualitative_transfer_only,
        "regularizer": args.regularizer,
        "block_size": args.block_size if args.regularizer == "llr" else None,
        "lambda_zero_manifest": {
            "path": str(lambda_zero_manifest),
            "sha256": source_hash,
        },
        "display_window_restored_magnitude": display_window,
        "cases": records,
        "outputs": {
            "comparison": str(comparison),
            "comparison_sha256": sha256_file(comparison),
            "relative_l2_plot": str(difference_plot),
            "relative_l2_plot_sha256": sha256_file(difference_plot),
        },
    }
    write_json_atomic(output_dir / "review_manifest.json", report)
    print(f"Regularization review manifest: {output_dir / 'review_manifest.json'}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
