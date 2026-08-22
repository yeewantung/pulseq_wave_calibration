#!/usr/bin/env python3
"""Audit, convert, and compare matched R1 DICOMs with direct no-Wave FFT RSS."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bart_cfl import sha256_file
from checkpoint_io import write_json_atomic
from export_grappa_rss import run as export_multicoil_rss


PRIVATE_FRAME_RECONSTRUCTION = (0x0021, 0x1176)
PRIVATE_FRAME_CONTAINER = (0x0021, 0x11FE)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalize-off-dicom", required=True, type=Path)
    parser.add_argument("--normalize-on-dicom", required=True, type=Path)
    parser.add_argument(
        "--acc-normalize-on-dicom",
        type=Path,
        help="Optional unfiltered Adaptive Combine, Prescan-Normalize-on series.",
    )
    parser.add_argument(
        "--additional-unfiltered-dicom",
        action="append",
        default=[],
        type=Path,
        help="Also audit and convert this unfiltered DICOM without adding it to the figure.",
    )
    parser.add_argument("--kspace", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--measurement-index", type=int, default=1)
    parser.add_argument("--dcm2niix", type=Path, default=Path("dcm2niix"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a complete output only after validating its recorded file hashes.",
    )
    return parser


def classify_reconstruction(
    series_description: str,
    reconstruction_tokens: Sequence[str],
    *,
    protocol_coil_combine_mode: int,
    contains_dis2d: bool,
    contains_dis3d: bool,
) -> dict[str, Any]:
    """Classify the exact stored output using Siemens per-frame provenance."""
    tokens = tuple(str(token) for token in reconstruction_tokens)
    coil_combination = {
        1: "Sum of Squares",
        2: "Adaptive Combine",
    }.get(protocol_coil_combine_mode, "unknown")
    return {
        "coil_combination": coil_combination,
        "per_frame_coil_combination_token": (
            "CC:SoS" if "CC:SoS" in tokens else "not_reported"
        ),
        "prescan_normalize": "NormalizeAlgo:PreScan" in tokens,
        "distortion_correction": (
            "filtered"
            if contains_dis2d or contains_dis3d or not series_description.endswith("_ND")
            else "unfiltered_ND"
        ),
        "contains_dis2d_marker": contains_dis2d,
        "contains_dis3d_marker": contains_dis3d,
    }


def _frame_reconstruction_tokens(frame: Any) -> tuple[str, ...]:
    container = frame.get(PRIVATE_FRAME_CONTAINER)
    if container is None or len(container.value) != 1:
        raise ValueError("Enhanced DICOM frame lacks the Siemens reconstruction container.")
    element = container.value[0].get(PRIVATE_FRAME_RECONSTRUCTION)
    if element is None:
        raise ValueError("Enhanced DICOM frame lacks reconstruction provenance tokens.")
    value = element.value
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def audit_dicom(path: Path) -> dict[str, Any]:
    """Require one internally consistent 256-frame enhanced magnitude series."""
    import pydicom

    source = path.expanduser().resolve()
    dataset = pydicom.dcmread(source, stop_before_pixels=True)
    frames = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
    if frames is None or len(frames) != int(getattr(dataset, "NumberOfFrames", 0)):
        raise ValueError(f"Invalid enhanced-DICOM frame structure: {source}")
    if len(frames) != 256 or int(dataset.Rows) != 256 or int(dataset.Columns) != 256:
        raise ValueError(f"Expected one 256x256x256 enhanced DICOM: {source}")
    frame_tokens = {_frame_reconstruction_tokens(frame) for frame in frames}
    if len(frame_tokens) != 1:
        raise ValueError(f"DICOM reconstruction tokens differ between frames: {source}")
    tokens = next(iter(frame_tokens))
    raw = source.read_bytes()
    protocol_coil_modes = {
        int(value)
        for value in re.findall(rb"ucCoilCombineMode\s*=\s*(\d+)", raw)
    }
    if len(protocol_coil_modes) != 1:
        raise ValueError(f"Could not resolve one protocol coil-combine mode: {source}")
    series_description = str(dataset.SeriesDescription)
    classification = classify_reconstruction(
        series_description,
        tokens,
        protocol_coil_combine_mode=next(iter(protocol_coil_modes)),
        contains_dis2d=b"DIS2D" in raw,
        contains_dis3d=b"DIS3D" in raw,
    )
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "series_number": int(dataset.SeriesNumber),
        "series_instance_uid": str(dataset.SeriesInstanceUID),
        "study_instance_uid": str(dataset.StudyInstanceUID),
        "frame_of_reference_uid": str(dataset.FrameOfReferenceUID),
        "series_description": series_description,
        "protocol_name": str(dataset.ProtocolName),
        "acquisition_datetime": str(dataset.AcquisitionDateTime),
        "number_of_frames": len(frames),
        "matrix": [int(dataset.Rows), int(dataset.Columns), len(frames)],
        "image_type": [str(value) for value in dataset.ImageType],
        "reconstruction_tokens": list(tokens),
        "protocol_coil_combine_mode": next(iter(protocol_coil_modes)),
        **classification,
    }


def validate_matched_pair(off: dict[str, Any], on: dict[str, Any]) -> None:
    """Ensure normalization is the only decisive reconstruction difference."""
    for field in (
        "study_instance_uid",
        "frame_of_reference_uid",
        "protocol_name",
        "acquisition_datetime",
        "matrix",
        "coil_combination",
        "protocol_coil_combine_mode",
        "distortion_correction",
    ):
        if off[field] != on[field]:
            raise ValueError(f"DICOM pair differs in {field}: {off[field]} vs {on[field]}")
    if off["coil_combination"] != "Sum of Squares":
        raise ValueError("Both comparison DICOMs must use Sum of Squares.")
    if off["distortion_correction"] != "unfiltered_ND":
        raise ValueError("Both comparison DICOMs must be unfiltered ND outputs.")
    if off["prescan_normalize"] is not False or on["prescan_normalize"] is not True:
        raise ValueError("DICOM normalization states do not match their requested roles.")
    normalize_token = "NormalizeAlgo:PreScan"
    if set(off["reconstruction_tokens"]) != set(on["reconstruction_tokens"]) - {
        normalize_token
    }:
        raise ValueError("DICOM per-frame reconstruction tokens differ beyond NormalizeAlgo.")


def validate_acc_normalize_on(reference: dict[str, Any], acc: dict[str, Any]) -> None:
    """Require a geometry-matched unfiltered ACC, Normalize-on comparison case."""
    for field in (
        "study_instance_uid",
        "frame_of_reference_uid",
        "protocol_name",
        "acquisition_datetime",
        "matrix",
        "distortion_correction",
    ):
        if reference[field] != acc[field]:
            raise ValueError(
                f"ACC and SOS DICOMs differ in {field}: "
                f"{reference[field]} vs {acc[field]}"
            )
    if acc["coil_combination"] != "Adaptive Combine":
        raise ValueError("ACC comparison DICOM does not use Adaptive Combine.")
    if acc["prescan_normalize"] is not True:
        raise ValueError("ACC comparison DICOM is not Prescan-Normalize-on.")
    if acc["distortion_correction"] != "unfiltered_ND":
        raise ValueError("ACC comparison DICOM is not an unfiltered ND output.")


def _comparison_cases(
    off: dict[str, Any],
    on: dict[str, Any],
    no_wave_nifti: str,
    acc: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the stable left-to-right presentation order."""
    cases = [
        {
            "key": "dicom_sos_normalize_off",
            "title": (
                "DICOM: SOS + Normalize off\n"
                f"Unfiltered ND (series {off['series_number']})"
            ),
            "nifti": off["conversion"]["nifti"],
        },
        {
            "key": "dicom_sos_normalize_on",
            "title": (
                "DICOM: SOS + Normalize on\n"
                f"Unfiltered ND (series {on['series_number']})"
            ),
            "nifti": on["conversion"]["nifti"],
        },
    ]
    if acc is not None:
        cases.append(
            {
                "key": "dicom_acc_normalize_on",
                "title": (
                    "DICOM: ACC + Normalize on\n"
                    f"Unfiltered ND (series {acc['series_number']})"
                ),
                "nifti": acc["conversion"]["nifti"],
            }
        )
    cases.append(
        {
            "key": "no_wave_fft_rss",
            "title": "Direct FFT RSS\nFully sampled no-Wave k-space, NCC=12",
            "nifti": no_wave_nifti,
        }
    )
    return cases


def _convert_dicom(
    source: Path, label: str, output_dir: Path, dcm2niix: Path
) -> dict[str, Any]:
    """Convert one exact enhanced series without dcm2niix crop or rotation."""
    import nibabel as nib

    case_dir = output_dir / "dicom" / label
    staging_dir = case_dir / "source"
    converted_dir = case_dir / "converted"
    staging_dir.mkdir(parents=True, exist_ok=False)
    converted_dir.mkdir(parents=True, exist_ok=False)
    link = staging_dir / "enhanced.dcm"
    link.symlink_to(source.resolve())
    command = [
        str(dcm2niix),
        "-b", "y",
        "-ba", "y",
        "-f", label,
        "-l", "o",
        "-m", "n",
        "-q", "n",
        "-w", "0",
        "-x", "i",
        "-z", "i",
        "-o", str(converted_dir),
        str(staging_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log = completed.stdout + completed.stderr
    (case_dir / "dcm2niix.log").write_text(log, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"dcm2niix failed for {source}:\n{log}")
    candidates = sorted(converted_dir.glob("*.nii.gz"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one converted NIfTI for {source}, found {candidates}")
    converted = nib.load(str(candidates[0]))
    canonical = nib.as_closest_canonical(converted)
    destination = output_dir / "dicom" / f"{label}_ras.nii.gz"
    nib.save(canonical, str(destination))
    if canonical.shape != (256, 256, 256):
        raise ValueError(f"Unexpected converted DICOM shape: {canonical.shape}")
    if tuple(nib.aff2axcodes(canonical.affine)) != ("R", "A", "S"):
        raise ValueError(f"Converted DICOM is not canonical RAS: {destination}")
    data = np.asarray(canonical.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f"Converted DICOM contains non-finite values: {destination}")
    sidecars = sorted(converted_dir.glob("*.json"))
    return {
        "nifti": str(destination),
        "nifti_sha256": sha256_file(destination),
        "shape": list(canonical.shape),
        "orientation": list(nib.aff2axcodes(canonical.affine)),
        "voxel_size_mm": [float(value) for value in canonical.header.get_zooms()[:3]],
        "dcm2niix_command": command,
        "dcm2niix_log": str(case_dir / "dcm2niix.log"),
        "dcm2niix_sidecars": [str(path) for path in sidecars],
    }


def _add_orientation_labels(axis: Any, view: str) -> None:
    labels = (
        ((0.02, 0.5, "L"), (0.98, 0.5, "R"), (0.5, 0.98, "S"), (0.5, 0.02, "I"))
        if view == "coronal"
        else ((0.02, 0.5, "L"), (0.98, 0.5, "R"), (0.5, 0.98, "A"), (0.5, 0.02, "P"))
    )
    for x, y, text in labels:
        axis.text(
            x,
            y,
            text,
            transform=axis.transAxes,
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
        )


def _make_review(
    cases: list[dict[str, Any]], output_dir: Path, *, replace: bool = False
) -> dict[str, Any]:
    """Show relative contrast after independent robust display normalization."""
    import matplotlib.pyplot as plt
    import nibabel as nib

    volumes = []
    case_records = []
    for case in cases:
        image = nib.load(case["nifti"])
        if tuple(nib.aff2axcodes(image.affine)) != ("R", "A", "S"):
            raise ValueError(f"Review input is not canonical RAS: {case['nifti']}")
        data = np.asarray(image.dataobj, dtype=np.float32)
        positive = data[data > 0]
        if positive.size == 0 or not np.isfinite(data).all():
            raise ValueError(f"Invalid review volume: {case['nifti']}")
        scale = float(np.percentile(positive, 99.0))
        volumes.append(data / np.float32(scale))
        case_records.append({**case, "display_positive_voxel_p99": scale})

    figure, axes = plt.subplots(
        2, len(cases), figsize=(4.5 * len(cases), 8.5), squeeze=False
    )
    for column, (case, volume) in enumerate(zip(case_records, volumes)):
        planes = (
            volume[:, volume.shape[1] // 2, :],
            volume[:, :, volume.shape[2] // 2],
        )
        for row, (view, plane) in enumerate(zip(("coronal", "axial"), planes)):
            axis = axes[row, column]
            axis.imshow(plane.T, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
            axis.set_title(f"{case['title']}\n{view}")
            _add_orientation_labels(axis, view)
            axis.axis("off")
    figure.suptitle(
        "R1 receive-profile comparison — each volume divided by its own positive-voxel p99"
    )
    figure.text(
        0.5,
        0.01,
        "Unregistered qualitative display; no BET mask, intensity matching, or ranking",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96), h_pad=2.2)
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=replace)
    figure_path = review_dir / "dicom_coil_combination_vs_no_wave_fft.png"
    if figure_path.exists() and not replace:
        raise FileExistsError(figure_path)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return {
        "figure": str(figure_path),
        "figure_sha256": sha256_file(figure_path),
        "display_scaling": "each volume independently divided by positive-voxel p99",
        "cases": case_records,
    }


def _add_unfiltered_conversions(
    output_dir: Path,
    paths: Sequence[Path],
    dcm2niix: Path,
    manifest: dict[str, Any],
) -> bool:
    """Add non-comparison unfiltered DICOM exports without changing the figure."""
    records = manifest["dicom"].setdefault("additional_unfiltered", [])
    by_path = {Path(record["path"]).resolve(): record for record in records}
    changed = False
    for requested in paths:
        path = requested.expanduser().resolve()
        if path in by_path:
            conversion = by_path[path]["conversion"]
            nifti = Path(conversion["nifti"])
            if not nifti.is_file() or sha256_file(nifti) != conversion["nifti_sha256"]:
                raise ValueError(f"Additional DICOM conversion failed hash validation: {nifti}")
            continue
        audit = audit_dicom(path)
        if audit["distortion_correction"] != "unfiltered_ND":
            raise ValueError(f"Additional DICOM is distortion-filtered: {path}")
        if audit["coil_combination"] == "unknown":
            raise ValueError(f"Additional DICOM coil combination is unknown: {path}")
        normalize = "on" if audit["prescan_normalize"] else "off"
        coil_label = (
            "sos"
            if audit["coil_combination"] == "Sum of Squares"
            else "adaptive-combine"
        )
        label = (
            f"{coil_label}_normalize-{normalize}_unfiltered_"
            f"series-{audit['series_number']}"
        )
        conversion = _convert_dicom(path, label, output_dir, dcm2niix)
        records.append({**audit, "conversion": conversion})
        changed = True
    return changed


def _reuse_complete(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_qualitative_comparison_only":
        raise ValueError(f"Existing comparison is not complete: {manifest_path}")
    expected_sources = {
        "normalize-off DICOM": (
            Path(manifest["dicom"]["normalize_off"]["path"]),
            args.normalize_off_dicom,
        ),
        "normalize-on DICOM": (
            Path(manifest["dicom"]["normalize_on"]["path"]),
            args.normalize_on_dicom,
        ),
        "no-Wave k-space": (Path(manifest["no_wave_fft"]["kspace"]), args.kspace),
        "source report": (
            Path(manifest["no_wave_fft"]["source_report"]),
            args.source_report,
        ),
    }
    for label, (recorded, requested) in expected_sources.items():
        if recorded.resolve() != requested.expanduser().resolve():
            raise ValueError(f"Existing comparison uses a different {label}.")
    refreshed_off = audit_dicom(args.normalize_off_dicom)
    refreshed_on = audit_dicom(args.normalize_on_dicom)
    validate_matched_pair(refreshed_off, refreshed_on)
    refreshed_off["conversion"] = manifest["dicom"]["normalize_off"]["conversion"]
    refreshed_on["conversion"] = manifest["dicom"]["normalize_on"]["conversion"]
    manifest["dicom"]["normalize_off"] = refreshed_off
    manifest["dicom"]["normalize_on"] = refreshed_on
    records = [
        manifest["dicom"]["normalize_off"]["conversion"],
        manifest["dicom"]["normalize_on"]["conversion"],
    ]
    for record in records:
        path = Path(record["nifti"])
        if not path.is_file() or sha256_file(path) != record["nifti_sha256"]:
            raise ValueError(f"Existing DICOM NIfTI failed hash validation: {path}")
    for section, path_key, hash_key in (
        (manifest["no_wave_fft"], "nifti", "nifti_sha256"),
        (manifest["review"], "figure", "figure_sha256"),
    ):
        path = Path(section[path_key])
        if not path.is_file() or sha256_file(path) != section[hash_key]:
            raise ValueError(f"Existing comparison output failed hash validation: {path}")
    changed = False
    if args.acc_normalize_on_dicom is not None:
        acc_path = args.acc_normalize_on_dicom.expanduser().resolve()
        acc = audit_dicom(acc_path)
        validate_acc_normalize_on(refreshed_off, acc)
        existing_acc = manifest["dicom"].get("acc_normalize_on")
        if existing_acc is None:
            conversion = _convert_dicom(
                acc_path,
                "adaptive-combine_normalize-on_unfiltered",
                output_dir,
                args.dcm2niix,
            )
            manifest["dicom"]["acc_normalize_on"] = {**acc, "conversion": conversion}
            manifest["dicom"]["additional_unfiltered"] = [
                record
                for record in manifest["dicom"].get("additional_unfiltered", [])
                if Path(record["path"]).resolve() != acc_path
            ]
            changed = True
        elif Path(existing_acc["path"]).resolve() != acc_path:
            raise ValueError("Existing comparison uses a different ACC Normalize-on DICOM.")
        else:
            conversion = existing_acc["conversion"]
            nifti = Path(conversion["nifti"])
            if not nifti.is_file() or sha256_file(nifti) != conversion["nifti_sha256"]:
                raise ValueError(f"ACC DICOM NIfTI failed hash validation: {nifti}")
        acc_record = manifest["dicom"]["acc_normalize_on"]
        manifest["review"] = _make_review(
            _comparison_cases(
                refreshed_off,
                refreshed_on,
                manifest["no_wave_fft"]["nifti"],
                acc_record,
            ),
            output_dir,
            replace=True,
        )
        changed = True
    if _add_unfiltered_conversions(
        output_dir, args.additional_unfiltered_dicom, args.dcm2niix, manifest
    ):
        changed = True
    if changed:
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(manifest_path, manifest)
    print(f"Reusing validated comparison: {output_dir}")
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if args.resume:
            return _reuse_complete(output_dir, args)
        raise FileExistsError(f"Comparison output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    off_path = args.normalize_off_dicom.expanduser().resolve()
    on_path = args.normalize_on_dicom.expanduser().resolve()
    off = audit_dicom(off_path)
    on = audit_dicom(on_path)
    validate_matched_pair(off, on)
    acc: dict[str, Any] | None = None
    acc_path: Path | None = None
    if args.acc_normalize_on_dicom is not None:
        acc_path = args.acc_normalize_on_dicom.expanduser().resolve()
        acc = audit_dicom(acc_path)
        validate_acc_normalize_on(off, acc)

    try:
        off_conversion = _convert_dicom(
            off_path, "sos_normalize-off_unfiltered", output_dir, args.dcm2niix
        )
        on_conversion = _convert_dicom(
            on_path, "sos_normalize-on_unfiltered", output_dir, args.dcm2niix
        )
        if acc is not None and acc_path is not None:
            acc_conversion = _convert_dicom(
                acc_path,
                "adaptive-combine_normalize-on_unfiltered",
                output_dir,
                args.dcm2niix,
            )
            acc["conversion"] = acc_conversion
        off["conversion"] = off_conversion
        on["conversion"] = on_conversion
        source_report_path = args.source_report.expanduser().resolve()
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        kspace = args.kspace.expanduser().resolve()
        assembly = source_report.get("assembly", {})
        if Path(assembly.get("output", "")).resolve() != kspace:
            raise ValueError("Source report does not identify the requested no-Wave k-space.")
        if assembly.get("shape") != [256, 256, 256, 12] or not assembly.get("finite"):
            raise ValueError("No-Wave source report does not certify finite 256^3x12 k-space.")
        no_wave_dir = output_dir / "no_wave"
        no_wave_output = no_wave_dir / "fft_rss_ncc12_ras.nii.gz"
        fft_metadata = export_multicoil_rss(
            Namespace(
                kspace=kspace,
                twix=args.twix,
                output=no_wave_output,
                measurement_index=args.measurement_index,
                canonical_ras=True,
                reference_recon=(
                    Path(__file__).resolve().parents[3]
                    / "external"
                    / "wave-mprage"
                    / "recon"
                ),
            )
        )
        review = _make_review(
            _comparison_cases(off, on, str(no_wave_output), acc), output_dir
        )
        manifest = {
            "format_version": 1,
            "status": "complete_qualitative_comparison_only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "compare Prescan Normalize receive-profile effects with direct "
                "no-Wave FFT RSS"
            ),
            "selection_policy": (
                "matched unfiltered ND SOS pair plus optional ACC; no intensity ranking"
            ),
            "dicom": {
                "normalize_off": off,
                "normalize_on": on,
                **({"acc_normalize_on": acc} if acc is not None else {}),
            },
            "no_wave_fft": {
                "kspace": str(kspace),
                "kspace_sha256": sha256_file(kspace),
                "source_report": str(source_report_path),
                "source_report_sha256": sha256_file(source_report_path),
                "nifti": str(no_wave_output),
                "nifti_sha256": sha256_file(no_wave_output),
                "metadata": fft_metadata,
            },
            "review": review,
            "scientific_limitations": [
                "DICOMs are qualitative comparison inputs, not ranking baselines.",
                "Each display column has independent positive-voxel p99 scaling.",
                "The direct FFT uses RSS after 64-to-12 virtual-coil compression.",
                "No registration, BET mask, or candidate-specific intensity matching was applied.",
            ],
        }
        _add_unfiltered_conversions(
            output_dir, args.additional_unfiltered_dicom, args.dcm2niix, manifest
        )
        write_json_atomic(output_dir / "manifest.json", manifest)
    except Exception as exc:
        write_json_atomic(
            output_dir / "manifest.json",
            {
                "format_version": 1,
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "normalize_off_dicom": str(off_path),
                "normalize_on_dicom": str(on_path),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    print(f"Comparison manifest: {output_dir / 'manifest.json'}")
    print(f"Comparison figure: {review['figure']}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
