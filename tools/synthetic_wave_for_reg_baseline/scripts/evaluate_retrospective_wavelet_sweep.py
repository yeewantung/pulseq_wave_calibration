#!/usr/bin/env python3
"""Evaluate retro-LR Wavelet lambdas on exact native target grids.

Native references use ``wave_retro_lr.core.centered_fftn`` on center-cropped,
fully sampled no-Wave multi-coil k-space, followed by RSS. NIfTI geometry and
metrics reuse the established Wave exporter and retrospective metric functions.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy
from nibabel.processing import resample_from_to

SCRIPT_ROOT = Path(__file__).resolve().parent
TOOL_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = TOOL_ROOT.parents[1]
RETRO_TOOL_ROOT = REPO_ROOT / "tools" / "wave_retro_lr_recon"
sys.path.insert(0, str(RETRO_TOOL_ROOT))

from wave_retro_lr.bart_io import sha256_file  # noqa: E402
from wave_retro_lr.core import centered_fftn  # noqa: E402
from wave_retro_lr.pipeline import (  # noqa: E402
    _load_json,
    _load_upstream_exporter,
    _resolve_path,
    _write_json,
)

from analyze_retrospective_low_resolution import (  # noqa: E402
    _canonical_image,
    _reference_edge_components,
    _resample_binary_mask,
    matched_fidelity_metrics,
)
from presentation_metrics import (  # noqa: E402
    nifti_sidecar_path,
    validate_metrics_reference_manifest,
)
from run_retrospective_wavelet_sweep import _settings  # noqa: E402


PLOT_METRICS = (
    ("nrmse_brain", "Native-grid brain NRMSE ↓", "min"),
    ("ssim_axial_brain_bbox_mean", "Native-grid axial SSIM ↑", "max"),
    ("gradient_ncc_fixed_edge", "Native-grid edge-gradient NCC ↑", "max"),
    ("edge_gradient_preservation_ratio", "Native-grid edge ratio → 1", "one"),
)


def direct_fft_rss(
    source: np.ndarray,
    case: Mapping[str, Any],
    *,
    fft_workers: int,
) -> np.ndarray:
    """Fourier-crop fully sampled k-space and return target-grid coil RSS."""
    target_shape = tuple(int(value) for value in case["target_logical_matrix_ro_lin_par"])
    lin = slice(*[int(value) for value in case["crop_bounds_lin"]])
    par = slice(*[int(value) for value in case["crop_bounds_par"]])
    if source.ndim != 4 or (source.shape[0], lin.stop - lin.start, par.stop - par.start) != target_shape:
        raise ValueError(f"Source/crop does not produce target shape {target_shape}.")
    rss_squared = np.zeros(target_shape, dtype=np.float32)
    for coil in range(source.shape[3]):
        coil_image = centered_fftn(
            np.asarray(source[:, lin, par, coil], dtype=np.complex64),
            axes=(0, 1, 2),
            inverse=True,
            workers=fft_workers,
        )
        magnitude = np.abs(coil_image).astype(np.float32)
        rss_squared += magnitude * magnitude
    return np.sqrt(rss_squared, out=rss_squared)


def _candidate_by_case(sweep: Mapping[str, Any]) -> dict[tuple[float, str], Path]:
    candidates: dict[tuple[float, str], Path] = {}
    for run in sweep["lambda_runs"]:
        value = float(run["lambda"])
        for case in run["cases"]:
            path = Path(case["magnitude_nifti"]).expanduser().resolve()
            phase = Path(case["phase_nifti"]).expanduser().resolve()
            if not path.is_file() or not phase.is_file():
                raise FileNotFoundError(path if not path.is_file() else phase)
            candidates[(value, str(case["case_name"]))] = path
    expected = len(sweep["wavelet_lambdas"]) * len(sweep["cases"])
    if len(candidates) != expected:
        raise ValueError(f"Sweep has {len(candidates)} unique candidates; expected {expected}.")
    return candidates


def _nifti_geometry(image: nib.spatialimages.SpatialImage) -> dict[str, list[int] | list[float]]:
    """Return JSON-native canonical geometry values."""
    return {
        "shape_ras_xyz": [int(value) for value in image.shape],
        "voxel_size_mm_ras_xyz": [
            float(value) for value in image.header.get_zooms()[:3]
        ],
    }


def _finalize_reference_directory(
    *,
    temporary_dir: Path,
    case_dir: Path,
    case: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    expected_candidate: Path,
) -> dict[str, Any]:
    """Validate and atomically publish one exported native reference."""
    outputs = sorted((temporary_dir / "nifti").rglob("*part-mag*.nii.gz"))
    if len(outputs) != 1:
        raise ValueError(f"Expected one native direct-FFT magnitude NIfTI: {outputs}")
    sidecar = _load_json(nifti_sidecar_path(outputs[0]))
    if sidecar.get("RetrospectiveCase") != dict(case):
        raise ValueError(f"Temporary native reference has different case metadata: {temporary_dir}")
    reference_image = nib.load(str(outputs[0]))
    candidate_image = nib.load(str(expected_candidate))
    if (
        reference_image.shape != candidate_image.shape
        or not np.allclose(reference_image.affine, candidate_image.affine, atol=1e-5)
        or nib.aff2axcodes(reference_image.affine) != ("R", "A", "S")
    ):
        raise ValueError("Native direct-FFT reference and reconstruction grids differ.")
    final_nifti = case_dir / outputs[0].relative_to(temporary_dir)
    record = {
        "format_version": 1,
        "status": "complete",
        "case": dict(case),
        "source_no_wave_kspace": str(source_path),
        "source_no_wave_kspace_sha256": source_sha256,
        "construction": "center PE crop before centered orthonormal IFFT3; RSS across virtual coils",
        "candidate_interpolation_performed": False,
        "magnitude_nifti": str(final_nifti),
        "magnitude_nifti_sha256": sha256_file(outputs[0]),
        **_nifti_geometry(reference_image),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(temporary_dir / "reference_manifest.json", record)
    temporary_dir.replace(case_dir)
    return record


def _reference_record(
    *,
    reference_root: Path,
    source: np.ndarray,
    source_path: Path,
    source_sha256: str,
    twix: Path,
    subject: str,
    case: Mapping[str, Any],
    expected_candidate: Path,
    fft_workers: int,
    resume: bool,
) -> dict[str, Any]:
    case_dir = reference_root / str(case["case_name"])
    manifest_path = case_dir / "reference_manifest.json"
    if manifest_path.is_file() and resume:
        record = _load_json(manifest_path)
        nifti_path = Path(record.get("magnitude_nifti", ""))
        if (
            record.get("status") == "complete"
            and record.get("case") == case
            and record.get("source_no_wave_kspace_sha256") == source_sha256
            and nifti_path.is_file()
            and sha256_file(nifti_path) == record.get("magnitude_nifti_sha256")
        ):
            return record
    if case_dir.exists():
        raise FileExistsError(f"Native reference is not safely reusable: {case_dir}")
    reference_root.mkdir(parents=True, exist_ok=True)
    partials = sorted(
        path
        for path in reference_root.glob(f".{case['case_name']}-*")
        if path.is_dir()
    )
    if partials:
        if not resume or len(partials) != 1:
            raise FileExistsError(f"Ambiguous temporary native references: {partials}")
        return _finalize_reference_directory(
            temporary_dir=partials[0],
            case_dir=case_dir,
            case=case,
            source_path=source_path,
            source_sha256=source_sha256,
            expected_candidate=expected_candidate,
        )
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{case['case_name']}-", dir=reference_root)
    )

    rss = direct_fft_rss(source, case, fft_workers=fft_workers)
    native = _load_upstream_exporter(REPO_ROOT)
    achieved_x, achieved_y, achieved_z = [
        float(value) for value in case["achieved_resolution_mm_xyz"]
    ]
    native.save_mprage_output_to_nifti(
        image=rss,
        twix_file=str(twix),
        out_folder=str(temporary_dir / "nifti"),
        nifti_sub=f"{subject}-native-direct-fft-{case['case_name']}",
        suffix="DirectFFTRSSNativeGrid",
        tag_wave="nowave",
        voxel_size_mm=(achieved_z, achieved_y, achieved_x),
        crop_readout_os=1,
        save_phase=False,
        twix_array_axis_roles=("phase", "readout", "slice"),
        twix_array_axis_flips=(True, False, False),
        metadata={
            "Description": "Native target-grid direct-FFT RSS reference",
            "ReferenceConstruction": "center-crop full no-Wave PE k-space, IFFT3 per virtual coil, RSS",
            "CandidateInterpolationUsed": False,
            "RetrospectiveCase": case,
        },
    )
    return _finalize_reference_directory(
        temporary_dir=temporary_dir,
        case_dir=case_dir,
        case=case,
        source_path=source_path,
        source_sha256=source_sha256,
        expected_candidate=expected_candidate,
    )


def _metric_row(
    *,
    case: Mapping[str, Any],
    lambda_value: float,
    reference_key: str,
    candidate_path: Path,
    metrics: Mapping[str, float],
    interpolation: bool,
) -> dict[str, Any]:
    return {
        "case_name": case["case_name"],
        "case_label": case.get("label"),
        "achieved_resolution_x_mm": case["achieved_resolution_mm_xyz"][0],
        "achieved_resolution_y_mm": case["achieved_resolution_mm_xyz"][1],
        "achieved_resolution_z_mm": case["achieved_resolution_mm_xyz"][2],
        "lambda": lambda_value,
        "reference": reference_key,
        "candidate_nifti": str(candidate_path),
        "candidate_nifti_sha256": sha256_file(candidate_path),
        "candidate_interpolation_performed": interpolation,
        **metrics,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty metric table.")
    fields = list(rows[0].keys())
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def metric_leaders(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report per-case, per-metric leaders without making one selection."""
    leaders: dict[str, Any] = {}
    for case_name in sorted({str(row["case_name"]) for row in rows}):
        case_rows = [row for row in rows if row["case_name"] == case_name]
        leaders[case_name] = {}
        for field, _label, objective in PLOT_METRICS:
            if objective == "min":
                leader = min(case_rows, key=lambda row: float(row[field]))
            elif objective == "max":
                leader = max(case_rows, key=lambda row: float(row[field]))
            else:
                leader = min(case_rows, key=lambda row: abs(float(row[field]) - 1.0))
            leaders[case_name][field] = {
                "lambda": float(leader["lambda"]),
                "value": float(leader[field]),
                "objective": objective,
            }
    return leaders


def _plot(path_png: Path, path_pdf: Path, rows: Sequence[Mapping[str, Any]], leaders: Mapping[str, Any]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = ("#006ba4", "#ff800e", "#3e9651")
    case_names = sorted({str(row["case_name"]) for row in rows})
    for axis, (field, label, _objective) in zip(axes.ravel(), PLOT_METRICS):
        for index, case_name in enumerate(case_names):
            color = colors[index % len(colors)]
            case_rows = sorted(
                (row for row in rows if row["case_name"] == case_name),
                key=lambda row: float(row["lambda"]),
            )
            title = str(case_rows[0].get("case_label") or case_name)
            axis.plot(
                [float(row["lambda"]) for row in case_rows],
                [float(row[field]) for row in case_rows],
                color=color,
                marker="o",
                linewidth=1.6,
                label=title,
            )
            leader = leaders[case_name][field]
            axis.scatter(
                [leader["lambda"]], [leader["value"]], color=color, marker="*", s=110, zorder=5
            )
        axis.set_xscale("symlog", linthresh=1e-5, linscale=0.8)
        axis.set_xlabel("Wavelet λ (0 is FISTA control)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    figure.suptitle(
        "Retrospective low-resolution Wavelet sweep vs native direct-FFT RSS\n"
        "Stars mark single-metric leaders; no automatic lambda selection"
    )
    figure.savefig(path_png, dpi=200)
    figure.savefig(path_pdf)
    plt.close(figure)


def run(config_path: Path, *, resume: bool) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    settings = _settings(config_path)
    sweep_path = settings["output_root"] / "sweep_manifest.json"
    sweep = _load_json(sweep_path)
    if sweep.get("status") != "complete":
        raise ValueError("Retrospective Wavelet sweep is not complete.")
    if sweep["config"]["sha256"] != sha256_file(config_path):
        raise ValueError("Sweep config changed after reconstruction.")

    output_dir = settings["output_root"] / "evaluation"
    manifest_path = output_dir / "evaluation_manifest.json"
    sweep_sha256 = sha256_file(sweep_path)
    if manifest_path.is_file() and resume:
        prior = _load_json(manifest_path)
        if prior.get("sweep_manifest", {}).get("sha256") != sweep_sha256:
            raise ValueError("Incomplete evaluation belongs to a different sweep manifest.")
        if prior.get("status") == "complete" and prior.get("sweep_manifest", {}).get("sha256") == sweep_sha256:
            print(f"Reusing retrospective sweep evaluation: {manifest_path}")
            return prior
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Evaluation output is not empty; use --resume: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Nonempty output is not an owned evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    running = {
        "format_version": 1,
        "status": "running",
        "purpose": "native-grid and matched-grid retrospective Wavelet sweep evaluation",
        "sweep_manifest": {"path": str(sweep_path), "sha256": sweep_sha256},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, running)

    base = _load_json(settings["base_config"])
    source_path = _resolve_path(
        base["source"]["no_wave_kspace"],
        settings["base_config"].parent,
        "source.no_wave_kspace",
    )
    twix = _resolve_path(
        base["source"]["twix"], settings["base_config"].parent, "source.twix"
    )
    if not source_path.is_file() or not twix.is_file():
        raise FileNotFoundError(source_path if not source_path.is_file() else twix)
    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 4 or source.dtype != np.complex64:
        raise ValueError(f"Expected complex64 [RO,LIN,PAR,coil] source: {source.shape} {source.dtype}")
    source_hash = sha256_file(source_path)
    candidates = _candidate_by_case(sweep)
    fft_workers = int(settings["config"].get("fft_workers", 4))
    if fft_workers < 1:
        raise ValueError("fft_workers must be positive.")

    metrics_context = validate_metrics_reference_manifest(settings["metrics_reference"])
    full_image, full_reference = _canonical_image(metrics_context["reference_path"])
    mask_image = nib.as_closest_canonical(nib.load(str(metrics_context["mask_path"])))
    full_brain = np.asarray(mask_image.dataobj) > 0
    if full_image.shape != mask_image.shape or not np.allclose(full_image.affine, mask_image.affine, atol=1e-5):
        raise ValueError("Approved full-resolution reference and BET mask grids differ.")
    full_zooms = full_image.header.get_zooms()[:3]
    _normalized, _gradient, full_edge, _threshold = _reference_edge_components(
        full_reference, full_brain, full_zooms
    )

    references = []
    native_rows = []
    matched_rows = []
    reference_root = output_dir / "native_direct_fft_references"
    first_lambda = float(sweep["wavelet_lambdas"][0])
    for case in sweep["cases"]:
        case_name = str(case["case_name"])
        reference_record = _reference_record(
            reference_root=reference_root,
            source=source,
            source_path=source_path,
            source_sha256=source_hash,
            twix=twix,
            subject=settings["subject"],
            case=case,
            expected_candidate=candidates[(first_lambda, case_name)],
            fft_workers=fft_workers,
            resume=resume,
        )
        references.append(reference_record)
        native_image, native_reference = _canonical_image(Path(reference_record["magnitude_nifti"]))
        native_brain = _resample_binary_mask(full_brain, full_image, native_image)
        native_zooms = native_image.header.get_zooms()[:3]
        _normalized, _gradient, native_edge, _threshold = _reference_edge_components(
            native_reference, native_brain, native_zooms
        )
        for value in [float(item) for item in sweep["wavelet_lambdas"]]:
            candidate_path = candidates[(value, case_name)]
            candidate_image, candidate = _canonical_image(candidate_path)
            if candidate_image.shape != native_image.shape or not np.allclose(
                candidate_image.affine, native_image.affine, atol=1e-5
            ):
                raise ValueError(f"Candidate is not on its exact native reference grid: {candidate_path}")
            native_metrics = matched_fidelity_metrics(
                native_reference, candidate, native_brain, native_edge, native_zooms
            )
            native_rows.append(
                _metric_row(
                    case=case,
                    lambda_value=value,
                    reference_key="native_target_grid_direct_fft_rss",
                    candidate_path=candidate_path,
                    metrics=native_metrics,
                    interpolation=False,
                )
            )
            matched = np.abs(
                np.asarray(resample_from_to(candidate_image, full_image, order=1).dataobj)
            ).astype(np.float32)
            matched_metrics = matched_fidelity_metrics(
                full_reference, matched, full_brain, full_edge, full_zooms
            )
            matched_rows.append(
                _metric_row(
                    case=case,
                    lambda_value=value,
                    reference_key="full_resolution_direct_fft_rss",
                    candidate_path=candidate_path,
                    metrics=matched_metrics,
                    interpolation=True,
                )
            )

    native_rows.sort(key=lambda row: (row["case_name"], float(row["lambda"])))
    matched_rows.sort(key=lambda row: (row["case_name"], float(row["lambda"])))
    leaders = metric_leaders(native_rows)
    native_csv = output_dir / "native_grid_metrics.csv"
    matched_csv = output_dir / "matched_1mm_metrics.csv"
    png_path = output_dir / "native_grid_metrics_vs_lambda.png"
    pdf_path = output_dir / "native_grid_metrics_vs_lambda.pdf"
    _write_csv(native_csv, native_rows)
    _write_csv(matched_csv, matched_rows)
    _plot(png_path, pdf_path, native_rows, leaders)

    manifest = {
        **running,
        "status": "complete",
        "scientific_scope": {
            "primary_reference": "case-specific direct-FFT RSS after exact Fourier crop to target PE grid",
            "primary_candidate_interpolation_performed": False,
            "approved_bet_mask_transfer_to_native_grid": "nearest-neighbor only",
            "secondary_reference": "full-resolution 1 mm direct-FFT RSS",
            "secondary_candidate_interpolation": "linear to the full-resolution RAS grid",
            "intensity_scaling": "one least-squares scalar inside the approved BET brain mask",
            "automatic_lambda_selection_performed": False,
        },
        "source_no_wave_kspace": {
            "path": str(source_path),
            "sha256": source_hash,
        },
        "metrics_reference_manifest": {
            "path": str(settings["metrics_reference"]),
            "sha256": metrics_context["manifest_sha256"],
        },
        "native_references": references,
        "native_metric_leaders": leaders,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (native_csv, matched_csv, png_path, pdf_path)
        ],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    print(f"Retrospective Wavelet sweep evaluation: {manifest_path}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run(args.config, resume=args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
