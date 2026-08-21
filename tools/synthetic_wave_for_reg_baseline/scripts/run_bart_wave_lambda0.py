#!/usr/bin/env python3
"""Run timed BART ecalib and unregularized Wave CG, then export NIfTIs."""

from __future__ import annotations

import argparse
import json
import re
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


AXIS_ROLES = ("phase", "readout", "slice")
AXIS_FLIPS = (True, False, False)
AFFINE_AXIS_FLIPS = (True, False, True)


def _build_parser() -> argparse.ArgumentParser:
    """Build the unregularized BART acceptance-run command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bart", required=True, type=Path)
    parser.add_argument("--bart-input-dir", required=True, type=Path)
    parser.add_argument(
        "--calibration-base",
        type=Path,
        help=(
            "BART ACS basename for ecalib. Defaults to "
            "<bart-input-dir>/kspace_calib."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--measurement-index", type=int, default=1)
    parser.add_argument("--subject", default="20260817product")
    parser.add_argument("--ecalib-crop", type=float, default=0.8)
    parser.add_argument(
        "--ecalib-intensity-correction",
        action="store_true",
        help="Pass -I to BART ecalib for adaptive-combine-like map normalization.",
    )
    parser.add_argument("--cg-iterations", type=int, default=300)
    parser.add_argument("--cg-tolerance", type=float, default=1e-3)
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
    outputs = []
    for part in ("mag", "phase"):
        figure, axes = plt.subplots(3, 4, figsize=(13, 10), squeeze=False)
        for coil, axis in enumerate(axes.ravel()):
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
    if logical.shape != (256, 256, 256):
        raise ValueError(f"Expected logical BART image [256,256,256], got {logical.shape}.")

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
    if not np.allclose(fov_mm, [256.0, 256.0, 256.0], atol=1e-3):
        raise ValueError(f"Unexpected sequence FOV for logical output: {fov_mm.tolist()} mm.")
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
        "ReadoutCropReason": "bart wave output is already on the 256-sample sensitivity-map grid",
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
        outputs.append({"part": part, "nifti": str(nii_path), "json": str(json_path)})

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
        "voxel_size_mm": list(affine_voxel_size),
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Calibrate ESPIRiT maps, run lambda-zero Wave CG, and export review files."""
    bart = args.bart.expanduser().resolve()
    bart_input = args.bart_input_dir.expanduser().resolve()
    calibration_base = bart_base(
        args.calibration_base.expanduser().resolve()
        if args.calibration_base is not None
        else bart_input / "kspace_calib"
    )
    output_dir = args.output_dir.expanduser().resolve()
    twix = args.twix.expanduser().resolve()
    sequence = args.sequence.expanduser().resolve()
    for path in (bart, twix, sequence):
        if not path.is_file():
            raise FileNotFoundError(path)
    for base in (calibration_base, bart_input / "psf", bart_input / "wave_kspace"):
        for suffix in (".hdr", ".cfl"):
            path = base.with_suffix(suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
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
            crop=args.ecalib_crop,
            intensity_correction=args.ecalib_intensity_correction,
        ),
        output_dir / "ecalib.log",
    )
    ecalib["output"] = validate_finite_bart(maps_base, (256, 256, 256, 12, 1))
    ecalib["output_cfl_sha256"] = sha256_file(maps_base.with_suffix(".cfl"))
    ecalib["eigenvalue_output"] = validate_finite_bart(
        eigenvalues_base, (256, 256, 256, 1, 1)
    )
    ecalib["eigenvalue_cfl_sha256"] = sha256_file(
        eigenvalues_base.with_suffix(".cfl")
    )
    ecalib["intensity_correction"] = args.ecalib_intensity_correction
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
            iterations=args.cg_iterations,
            tolerance=args.cg_tolerance,
        ),
        output_dir / "wave_lambda0.log",
    )
    wave.update(_extract_bart_times(Path(wave["log"])))
    wave["algorithm"] = "conjugate gradient selected by bart wave without -w or -l"
    wave["backend"] = "gpu"
    wave["lambda"] = 0.0
    wave["output"] = validate_finite_bart(image_base, (256, 256, 256, 1, 1))
    wave["output_cfl_sha256"] = sha256_file(image_base.with_suffix(".cfl"))

    nifti_started = time.perf_counter()
    nifti = _convert_nifti(
        image_base,
        bart_input_dir=bart_input,
        output_dir=output_dir,
        twix=twix,
        sequence=sequence,
        measurement_index=args.measurement_index,
        subject=args.subject,
    )
    nifti["conversion_wall_seconds"] = time.perf_counter() - nifti_started

    manifest = {
        "format_version": 1,
        "status": "lambda0_complete_awaiting_visual_review",
        "bart_executable": str(bart),
        "bart_version_output": bart_version_output,
        "bart_input_dir": str(bart_input),
        "ecalib": ecalib,
        "wave_lambda0": wave,
        "nifti": nifti,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
