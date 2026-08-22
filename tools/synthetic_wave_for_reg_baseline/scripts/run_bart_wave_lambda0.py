#!/usr/bin/env python3
"""Run timed BART ecalib and unregularized Wave CG, then export NIfTIs."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bart_cfl import (
    bart_base,
    open_bart_memmap,
    sha256_file,
    validate_finite_bart,
)
from checkpoint_io import write_json_atomic
from dataset_manifest import (
    DatasetManifestError,
    load_dataset_manifest,
    load_passed_inspection,
)


AXIS_ROLES = ("phase", "readout", "slice")
AXIS_FLIPS = (True, False, False)
AFFINE_AXIS_FLIPS = (True, False, True)


def _build_parser() -> argparse.ArgumentParser:
    """Build the unregularized BART acceptance-run command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Use the validated dataset contract for all inputs, outputs, and settings.",
    )
    parser.add_argument("--bart", type=Path)
    parser.add_argument("--bart-input-dir", type=Path)
    parser.add_argument(
        "--calibration-base",
        type=Path,
        help=(
            "BART ACS basename for ecalib. Defaults to "
            "<bart-input-dir>/kspace_calib."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--twix", type=Path)
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--measurement-index", type=int)
    parser.add_argument("--subject")
    parser.add_argument("--ecalib-crop", type=float)
    parser.add_argument(
        "--ecalib-intensity-correction",
        action="store_true",
        default=None,
        help="Pass -I to BART ecalib for adaptive-combine-like map normalization.",
    )
    parser.add_argument("--cg-iterations", type=int)
    parser.add_argument("--cg-tolerance", type=float)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only a complete reconstruction with matching provenance and hashes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _run_logged(command: list[str], log_path: Path) -> dict[str, Any]:
    """Stream a subprocess to the terminal while retaining its complete timed log."""
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    lines: list[str] = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    returncode = process.wait()
    wall_seconds = time.perf_counter() - started
    log_path.write_text("".join(lines), encoding="utf-8")
    if returncode:
        raise RuntimeError(f"Command failed with status {returncode}: {' '.join(command)}")
    return {"command": command, "wall_seconds": wall_seconds, "log": str(log_path)}


def build_ecalib_command(
    bart: Path,
    calibration_base: Path,
    maps_base: Path,
    eigenvalues_base: Path,
    *,
    crop: float,
    intensity_correction: bool = False,
) -> list[str]:
    """Build hard-crop one-map ESPIRiT calibration with eigenvalue output."""
    if not 0.0 <= crop <= 1.0:
        raise ValueError("ESPIRiT crop must be between 0 and 1.")
    command = [
        str(bart),
        "ecalib",
        "-m",
        "1",
        "-c",
        str(crop),
    ]
    if intensity_correction:
        command.append("-I")
    return command + [str(calibration_base), str(maps_base), str(eigenvalues_base)]


def build_wave_command(
    bart: Path,
    maps_base: Path,
    psf_base: Path,
    kspace_base: Path,
    image_base: Path,
    *,
    iterations: int,
    tolerance: float,
) -> list[str]:
    """Build the GPU-only unregularized Wave reconstruction command."""
    if iterations < 1 or tolerance <= 0:
        raise ValueError("Wave iterations and tolerance must be positive.")
    return [
        str(bart),
        "wave",
        "-g",
        "-i",
        str(iterations),
        "-t",
        str(tolerance),
        str(maps_base),
        str(psf_base),
        str(kspace_base),
        str(image_base),
    ]


def _save_map_montages(maps_base: Path, output_dir: Path) -> list[str]:
    """Save central-slice magnitude and phase montages for all active maps/coils."""
    import matplotlib.pyplot as plt

    maps = open_bart_memmap(maps_base)
    logical = maps.reshape((*maps.shape[:5], -1), order="F")
    if logical.shape[4] != 1 or logical.shape[5] != 1:
        raise ValueError("Map diagnostics require one ESPIRiT map set and trailing singleton axes.")
    coil_count = int(logical.shape[3])
    columns = min(4, coil_count)
    rows = math.ceil(coil_count / columns)
    outputs = []
    for part in ("mag", "phase"):
        figure, axes = plt.subplots(
            rows, columns, figsize=(3.25 * columns, 3.1 * rows), squeeze=False
        )
        for coil, axis in enumerate(axes.ravel()):
            if coil >= coil_count:
                axis.axis("off")
                continue
            plane = np.asarray(logical[:, :, logical.shape[2] // 2, coil, 0, 0])
            data = np.abs(plane) if part == "mag" else np.angle(plane)
            kwargs = {"cmap": "gray"} if part == "mag" else {
                "cmap": "twilight",
                "vmin": -np.pi,
                "vmax": np.pi,
            }
            axis.imshow(data.T, origin="lower", **kwargs)
            axis.set_title(f"Virtual coil {coil + 1:02d}")
            axis.axis("off")
        figure.suptitle(f"BART ESPIRiT maps — {part}")
        figure.tight_layout()
        path = output_dir / f"espirit_maps_{part}_central_slice.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        outputs.append(str(path))
    return outputs


def _convert_nifti(
    image_base: Path,
    *,
    bart_input_dir: Path,
    output_dir: Path,
    twix: Path,
    sequence: Path,
    measurement_index: int,
    subject: str,
    matrix_rolinpar: tuple[int, int, int],
    fov_mm_rolinpar: tuple[float, float, float],
) -> dict[str, Any]:
    """Convert logical BART output using the upstream TWIX orientation convention."""
    import nibabel as nib
    import pypulseq as pp

    reference_recon = Path(__file__).resolve().parents[3] / "external" / "wave-mprage" / "recon"
    sys.path.insert(0, str(reference_recon))
    from utils.nifti_export_twix import (
        apply_array_axis_flips,
        canonicalize_arrays_to_ras,
        make_nifti_affine_from_twix,
        normalize_magnitude,
        save_nifti_with_json,
    )

    image = open_bart_memmap(image_base)
    logical = np.asarray(image).reshape(image.shape[:3], order="F")
    if logical.shape != matrix_rolinpar:
        raise ValueError(
            f"Expected logical BART image {matrix_rolinpar}, got {logical.shape}."
        )

    input_manifest = json.loads((bart_input_dir / "manifest.json").read_text(encoding="utf-8"))
    kspace_norm = float(input_manifest["masked_wave_kspace"]["norm"])
    restored = logical.astype(np.complex64, copy=False) * np.float32(kspace_norm)
    magnitude, magnitude_normalization = normalize_magnitude(
        np.abs(restored).astype(np.float32), percentile=99.0
    )
    phase = np.angle(restored).astype(np.float32)
    magnitude, phase = apply_array_axis_flips((magnitude, phase), AXIS_FLIPS)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seq = pp.Sequence()
        seq.read(str(sequence), remove_duplicates=False)
    definitions = seq.definitions
    fov_mm = np.asarray(definitions["FOV"], dtype=float) * 1000.0
    if not np.allclose(fov_mm, fov_mm_rolinpar, atol=1e-3):
        raise ValueError(
            "Sequence and dataset-contract FOV disagree: "
            f"{fov_mm.tolist()} versus {list(fov_mm_rolinpar)} mm."
        )
    voxel_size_mm = tuple(float(value) for value in fov_mm / np.asarray(logical.shape))
    affine, _, twix_info = make_nifti_affine_from_twix(
        twix_file=twix,
        scan_index=measurement_index,
        npy_shape=logical.shape,
        twix_array_axis_roles=AXIS_ROLES,
        twix_array_axis_flips=AFFINE_AXIS_FLIPS,
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=voxel_size_mm,
    )
    (magnitude, phase), affine, orientation_transform = canonicalize_arrays_to_ras(
        (magnitude, phase), affine
    )
    stored_voxel_size_mm = np.linalg.norm(affine[:3, :3], axis=0)

    nifti_dir = output_dir / "nifti"
    nifti_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    common = {
        "Description": "BART unregularized Wave CG-SENSE reconstruction",
        "ReconstructionSoftware": "BART wave",
        "Regularization": "none",
        "Lambda": 0.0,
        "BARTImageInput": str(image_base),
        "BARTInternalNormalizationRestored": True,
        "BARTWaveKspaceNormRestored": kspace_norm,
        "BARTOutputLogicalShape": list(logical.shape),
        "BARTOutputAlreadyReadoutDeoversampled": True,
        "SequenceReadoutOversamplingFactor": int(definitions["ReadoutOversamplingFactor"]),
        "AdditionalReadoutCropApplied": False,
        "ReadoutCropReason": (
            "bart wave output is already on the logical sensitivity-map grid"
        ),
        "SourceTwix": str(twix),
        "SourceSequence": str(sequence),
        "NIfTITwixArrayAxisRoles": list(AXIS_ROLES),
        "NIfTIPhysicalArrayFlipsApplied": list(AXIS_FLIPS),
        "NIfTIAffineAxisFlips": list(AFFINE_AXIS_FLIPS),
        "NIfTICanonicalRAS": True,
        "NIfTIOrientationTransform": orientation_transform,
        "NIfTIStoredImageShape": [int(value) for value in magnitude.shape],
        "NIfTIVoxelSizeMm": [float(value) for value in stored_voxel_size_mm],
        "TwixOrientation": twix_info,
        "PyPulseqWarnings": [str(item.message) for item in caught],
    }
    for part, data in (("mag", magnitude), ("phase", phase)):
        stem = f"sub-{subject}_part-{part}_BARTCGSENSELambda0"
        nii_path = nifti_dir / f"{stem}.nii.gz"
        json_path = nifti_dir / f"{stem}.json"
        metadata = {
            **common,
            "Part": part,
            "Units": "normalized arbitrary" if part == "mag" else "rad",
            "MagnitudeNormalization": magnitude_normalization if part == "mag" else None,
        }
        save_nifti_with_json(data, affine, nii_path, json_path, metadata)
        saved = nib.load(str(nii_path))
        if saved.shape != logical.shape or not np.isfinite(np.asanyarray(saved.dataobj)).all():
            raise ValueError(f"Saved NIfTI validation failed: {nii_path}")
        outputs.append(
            {
                "part": part,
                "nifti": str(nii_path),
                "nifti_sha256": sha256_file(nii_path),
                "json": str(json_path),
                "json_sha256": sha256_file(json_path),
            }
        )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    planes = (
        magnitude[magnitude.shape[0] // 2, :, :],
        magnitude[:, magnitude.shape[1] // 2, :],
        magnitude[:, :, magnitude.shape[2] // 2],
    )
    vmax = float(np.percentile(magnitude[magnitude > 0], 99.5))
    for axis, plane, title in zip(axes, planes, ("axis 0 center", "axis 1 center", "axis 2 center")):
        axis.imshow(plane.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("BART Wave CG-SENSE lambda=0")
    figure.tight_layout()
    quicklook = output_dir / "lambda0_nifti_central_slices.png"
    figure.savefig(quicklook, dpi=140)
    plt.close(figure)
    return {
        "outputs": outputs,
        "voxel_size_mm": [float(value) for value in stored_voxel_size_mm],
        "central_slice_quicklook": str(quicklook),
    }


def _extract_bart_times(log_path: Path) -> dict[str, float]:
    """Extract BART's internal timing fields from a reconstruction log."""
    text = log_path.read_text(encoding="utf-8")
    result = {}
    for key, pattern in (
        ("maximum_eigenvalue", r"Max eval:\s*([0-9.eE+-]+)"),
        ("internal_reconstruction_seconds", r"Reconstruction time:\s*([0-9.eE+-]+) seconds"),
        ("bart_total_seconds", r"Total time:\s*([0-9.eE+-]+) seconds"),
    ):
        match = re.search(pattern, text)
        if match:
            result[key] = float(match.group(1))
    return result


def _completed_reconstruction_reusable(
    manifest_path: Path,
    *,
    dataset_sha256: str | None,
    bart_input_manifest_sha256: str,
    expected_config: dict[str, Any],
) -> bool:
    """Accept reuse only when upstream provenance and persisted outputs still match."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "lambda0_complete_awaiting_visual_review":
            return False
        dataset_record = manifest.get("dataset_manifest")
        if dataset_sha256 is None:
            if dataset_record is not None:
                return False
        elif not isinstance(dataset_record, dict) or dataset_record.get(
            "sha256"
        ) != dataset_sha256:
            return False
        if manifest.get("bart_input_manifest", {}).get(
            "sha256"
        ) != bart_input_manifest_sha256:
            return False
        if manifest.get("config") != expected_config:
            return False
        for record in manifest["nifti"]["outputs"]:
            if sha256_file(Path(record["nifti"])) != record["nifti_sha256"]:
                return False
            if sha256_file(Path(record["json"])) != record["json_sha256"]:
                return False
        for section in ("ecalib", "wave_lambda0"):
            base = Path(manifest[section]["output_base"])
            if sha256_file(base.with_suffix(".cfl")) != manifest[section][
                "output_cfl_sha256"
            ]:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _resolve_run(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve either a manifest-backed run or the compatible explicit interface."""
    manifest_incompatible_options = (
        args.bart,
        args.bart_input_dir,
        args.calibration_base,
        args.twix,
        args.sequence,
        args.measurement_index,
        args.subject,
        args.ecalib_intensity_correction,
        args.cg_iterations,
        args.cg_tolerance,
    )
    if args.dataset_manifest is not None:
        if any(value is not None for value in manifest_incompatible_options) or args.overwrite:
            raise ValueError(
                "--dataset-manifest cannot be combined with explicit reconstruction "
                "options other than --ecalib-crop/--output-dir, or --overwrite"
            )
        dataset = load_dataset_manifest(args.dataset_manifest)
        inspection = load_passed_inspection(dataset)
        bart_command = shutil.which("bart")
        if bart_command is None:
            raise FileNotFoundError(
                "bart is not on PATH; source /path/to/user_workspace/bart/bart_startup.sh first"
            )
        bart_settings = dataset.payload["reconstruction"]["bart"]
        if "lambda0_reconstruction_dir" not in dataset.payload["outputs"]:
            raise ValueError(
                "Manifest-backed lambda zero requires "
                "outputs.lambda0_reconstruction_dir"
            )
        output_dir = (
            dataset.output_path("lambda0_reconstruction_dir")
            if args.output_dir is None
            else args.output_dir.expanduser().resolve()
        )
        if not output_dir.is_relative_to(dataset.output_root):
            raise ValueError("Manifest output override must remain below outputs.root")
        manifest_crop = float(bart_settings.get("ecalib_crop", 0.8))
        selected_crop = manifest_crop if args.ecalib_crop is None else args.ecalib_crop
        return {
            "dataset": dataset,
            "bart": Path(bart_command).resolve(),
            "bart_input": dataset.output_path("bart_export_dir") / "bart_inputs",
            "calibration_base": None,
            "output_dir": output_dir,
            "twix": dataset.input_path("twix"),
            "sequence": dataset.input_path("wave_sequence"),
            "measurement_index": int(inspection["twix"]["selected_measurement_index"]),
            "subject": dataset.subject,
            "crop": selected_crop,
            "intensity_correction": bool(
                bart_settings.get("ecalib_intensity_correction", False)
            ),
            "iterations": int(bart_settings.get("lambda0_iterations", 300)),
            "tolerance": float(bart_settings.get("lambda0_tolerance", 1e-3)),
            "matrix": tuple(int(value) for value in dataset.payload["geometry"]["matrix"]),
            "fov_mm": tuple(float(value) for value in dataset.payload["geometry"]["fov_mm"]),
            "virtual_coils": int(dataset.payload["reconstruction"]["virtual_coils"]),
            "manifest_overrides": {
                "ecalib_crop": selected_crop if args.ecalib_crop is not None else None,
                "output_dir": str(output_dir) if args.output_dir is not None else None,
            },
        }

    required = {
        "--bart": args.bart,
        "--bart-input-dir": args.bart_input_dir,
        "--output-dir": args.output_dir,
        "--twix": args.twix,
        "--sequence": args.sequence,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Use --dataset-manifest, or provide " + ", ".join(missing)
        )
    return {
        "dataset": None,
        "bart": args.bart.expanduser().resolve(),
        "bart_input": args.bart_input_dir.expanduser().resolve(),
        "calibration_base": args.calibration_base,
        "output_dir": args.output_dir.expanduser().resolve(),
        "twix": args.twix.expanduser().resolve(),
        "sequence": args.sequence.expanduser().resolve(),
        "measurement_index": 1 if args.measurement_index is None else args.measurement_index,
        "subject": "20260817product" if args.subject is None else args.subject,
        "crop": 0.8 if args.ecalib_crop is None else args.ecalib_crop,
        "intensity_correction": bool(args.ecalib_intensity_correction),
        "iterations": 300 if args.cg_iterations is None else args.cg_iterations,
        "tolerance": 1e-3 if args.cg_tolerance is None else args.cg_tolerance,
        "matrix": (256, 256, 256),
        "fov_mm": (256.0, 256.0, 256.0),
        "virtual_coils": 12,
        "manifest_overrides": None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Calibrate ESPIRiT maps, run lambda-zero Wave CG, and export review files."""
    resolved = _resolve_run(args)
    dataset = resolved["dataset"]
    bart = resolved["bart"]
    bart_input = resolved["bart_input"]
    calibration_base = bart_base(
        resolved["calibration_base"].expanduser().resolve()
        if resolved["calibration_base"] is not None
        else bart_input / "kspace_calib"
    )
    output_dir = resolved["output_dir"]
    twix = resolved["twix"]
    sequence = resolved["sequence"]
    matrix = resolved["matrix"]
    ncc = resolved["virtual_coils"]
    for path in (bart, twix, sequence):
        if not path.is_file():
            raise FileNotFoundError(path)
    for base in (calibration_base, bart_input / "psf", bart_input / "wave_kspace"):
        for suffix in (".hdr", ".cfl"):
            path = base.with_suffix(suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
    bart_input_manifest_path = bart_input / "manifest.json"
    if not bart_input_manifest_path.is_file():
        raise FileNotFoundError(bart_input_manifest_path)
    bart_input_manifest = json.loads(
        bart_input_manifest_path.read_text(encoding="utf-8")
    )
    bart_input_status = bart_input_manifest.get("status")
    if dataset is not None:
        if bart_input_status != "calibration_kspace_ready_for_ecalib":
            raise ValueError("Measured ACS is not ready for BART ecalib.")
    elif bart_input_status not in {
        "calibration_kspace_ready_for_ecalib",
        "masked_wave_inputs_ready_for_reconstruction_with_existing_maps",
    }:
        raise ValueError(
            "Explicit BART inputs are not in a recognized reconstruction-ready state."
        )
    elif (
        bart_input_status
        == "masked_wave_inputs_ready_for_reconstruction_with_existing_maps"
        and resolved["calibration_base"] is None
    ):
        raise ValueError(
            "Legacy masked Wave inputs require an explicit --calibration-base."
        )
    if dataset is not None and bart_input_manifest.get("dataset_manifest", {}).get(
        "sha256"
    ) != dataset.sha256:
        raise ValueError("BART inputs use a stale dataset manifest.")
    bart_input_manifest_sha256 = sha256_file(bart_input_manifest_path)
    run_config = {
        "matrix_rolinpar": list(matrix),
        "fov_mm_rolinpar": list(resolved["fov_mm"]),
        "virtual_coils": ncc,
        "ecalib_crop": resolved["crop"],
        "ecalib_intensity_correction": resolved["intensity_correction"],
        "lambda0_iterations": resolved["iterations"],
        "lambda0_tolerance": resolved["tolerance"],
        "gpu_wave_reconstruction": True,
        "dataset_manifest_overrides": resolved["manifest_overrides"],
    }
    manifest_path = output_dir / "manifest.json"
    if (
        args.resume
        and _completed_reconstruction_reusable(
            manifest_path,
            dataset_sha256=dataset.sha256 if dataset is not None else None,
            bart_input_manifest_sha256=bart_input_manifest_sha256,
            expected_config=run_config,
        )
    ):
        print(f"Reusing validated lambda-zero reconstruction: {output_dir}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    version = subprocess.run(
        [str(bart), "version"], check=True, capture_output=True, text=True
    )
    bart_version_output = (version.stdout + version.stderr).strip()
    maps_base = output_dir / "coil_sens_bart"
    eigenvalues_base = output_dir / "eigenvalue_maps_bart"
    ecalib = _run_logged(
        build_ecalib_command(
            bart,
            calibration_base,
            maps_base,
            eigenvalues_base,
            crop=resolved["crop"],
            intensity_correction=resolved["intensity_correction"],
        ),
        output_dir / "ecalib.log",
    )
    ecalib["output"] = validate_finite_bart(maps_base, (*matrix, ncc, 1))
    ecalib["output_base"] = str(maps_base)
    ecalib["output_cfl_sha256"] = sha256_file(maps_base.with_suffix(".cfl"))
    ecalib["eigenvalue_output"] = validate_finite_bart(
        eigenvalues_base, (*matrix, 1, 1)
    )
    ecalib["eigenvalue_cfl_sha256"] = sha256_file(
        eigenvalues_base.with_suffix(".cfl")
    )
    ecalib["intensity_correction"] = resolved["intensity_correction"]
    ecalib["input_base"] = str(calibration_base)
    ecalib["diagnostic_montages"] = _save_map_montages(maps_base, output_dir)

    image_base = output_dir / "image_wave"
    wave = _run_logged(
        build_wave_command(
            bart,
            maps_base,
            bart_input / "psf",
            bart_input / "wave_kspace",
            image_base,
            iterations=resolved["iterations"],
            tolerance=resolved["tolerance"],
        ),
        output_dir / "wave_lambda0.log",
    )
    wave.update(_extract_bart_times(Path(wave["log"])))
    wave["algorithm"] = "conjugate gradient selected by bart wave without -w or -l"
    wave["backend"] = "gpu"
    wave["lambda"] = 0.0
    wave["output"] = validate_finite_bart(image_base, (*matrix, 1, 1))
    wave["output_base"] = str(image_base)
    wave["output_cfl_sha256"] = sha256_file(image_base.with_suffix(".cfl"))

    nifti_started = time.perf_counter()
    nifti = _convert_nifti(
        image_base,
        bart_input_dir=bart_input,
        output_dir=output_dir,
        twix=twix,
        sequence=sequence,
        measurement_index=resolved["measurement_index"],
        subject=resolved["subject"],
        matrix_rolinpar=matrix,
        fov_mm_rolinpar=resolved["fov_mm"],
    )
    nifti["conversion_wall_seconds"] = time.perf_counter() - nifti_started

    manifest = {
        "format_version": 1,
        "status": "lambda0_complete_awaiting_visual_review",
        "bart_executable": str(bart),
        "bart_version_output": bart_version_output,
        "bart_input_dir": str(bart_input),
        "bart_input_manifest": {
            "path": str(bart_input_manifest_path),
            "sha256": bart_input_manifest_sha256,
        },
        "dataset_manifest": dataset.provenance() if dataset is not None else None,
        "config": run_config,
        "ecalib": ecalib,
        "wave_lambda0": wave,
        "nifti": nifti,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if sha256_file(bart_input_manifest_path) != bart_input_manifest_sha256:
        raise ValueError("BART input manifest changed during reconstruction.")
    write_json_atomic(manifest_path, manifest)
    print(f"Run manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the lambda-zero acceptance reconstruction from CLI arguments."""
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
