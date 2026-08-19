#!/usr/bin/env python3
"""Calibrate BART ESPIRiT maps and run unregularized SigPy no-wave SENSE.

Inputs are the measured, coil-compressed no-wave product k-space and the
separately exported measured ACS. BART performs one-map ESPIRiT calibration;
SigPy then solves the exact masked Cartesian SENSE problem with lambda zero.
The reconstructed object and model-consistent 12-coil k-space are retained for
later Wave synthesis, but no Wave encoding is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from phase_e_utils import bart_base, open_bart_memmap, sha256_file, validate_finite_bart


GRID_SHAPE = (256, 256, 256)
NCC = 12
PHYSICAL_ARRAY_FLIPS = (True, False, False)
AFFINE_AXIS_FLIPS = (True, False, True)
AXIS_ROLES = ("phase", "readout", "slice")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for BART-map SigPy SENSE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bart", required=True, type=Path)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--calib-base", required=True, type=Path)
    parser.add_argument("--calib-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--measurement-index", type=int, default=1)
    parser.add_argument("--subject", default="20260817product")
    parser.add_argument("--ecalib-crop", type=float, default=0.8)
    parser.add_argument(
        "--ecalib-soft",
        action="store_true",
        help="Pass -S to BART ecalib so the crop boundary has a smooth transition.",
    )
    parser.add_argument(
        "--save-ecalib-eigenvalues",
        action="store_true",
        help="Save BART eigenvalue maps for sensitivity-support diagnostics.",
    )
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--coil-batch-size", type=int, default=1)
    parser.add_argument("--skip-modeled-kspace", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object top-level values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_provenance(
    prepare_manifest: dict[str, Any], calibration_manifest: dict[str, Any]
) -> None:
    """Require measured ACS and imaging data to share the exact 12-coil basis."""
    expected_columns = [0, NCC]
    for label, manifest in (
        ("prepared imaging", prepare_manifest),
        ("calibration", calibration_manifest),
    ):
        if manifest.get("basis_columns_half_open") != expected_columns:
            raise ValueError(f"{label} does not use basis columns {expected_columns}.")
    keys = ("coil_basis", "coil_basis_file_sha256")
    for key in keys:
        if prepare_manifest.get(key) != calibration_manifest.get(key):
            raise ValueError(f"Prepared imaging and calibration differ in {key}.")
    if prepare_manifest.get("contains_grappa_samples") is not False:
        raise ValueError("Prepared no-wave SENSE input has ambiguous GRAPPA provenance.")
    if calibration_manifest.get("source") != "measured no-wave product TWIX refscan ACS":
        raise ValueError("ESPIRiT calibration input is not the measured no-wave ACS.")


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """Create a dedicated output directory without touching unrelated files."""
    owned_names = {
        "coil_sens_bart.cfl",
        "coil_sens_bart.hdr",
        "eigenvalue_maps_bart.cfl",
        "eigenvalue_maps_bart.hdr",
        "ecalib.log",
        "image_sense_lambda0.npy",
        "model_no_wave_kspace_ncc12.npy",
        "manifest.json",
        "sense_quicklook.png",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in owned_names if (output_dir / name).exists()]
    nifti_dir = output_dir / "nifti"
    if nifti_dir.exists():
        existing.append(nifti_dir)
    if existing and not overwrite:
        raise FileExistsError(f"SENSE outputs already exist: {existing}")
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _run_logged(command: list[str], log_path: Path) -> dict[str, Any]:
    """Run one external command while streaming and preserving its output."""
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    returncode = process.wait()
    log_path.write_text("".join(lines), encoding="utf-8")
    if returncode:
        raise RuntimeError(f"Command failed with status {returncode}: {' '.join(command)}")
    return {
        "command": command,
        "wall_seconds": time.perf_counter() - started,
        "log": str(log_path),
    }


def build_ecalib_command(
    bart: Path,
    calib_base: Path,
    maps_base: Path,
    crop: float,
    *,
    soft: bool,
    eigenvalue_base: Path | None,
) -> list[str]:
    """Build one-map BART ESPIRiT calibration with explicit support controls."""
    command = [str(bart), "ecalib", "-m", "1", "-c", str(crop)]
    if soft:
        command.append("-S")
    command.extend([str(bart_base(calib_base)), str(maps_base)])
    if eigenvalue_base is not None:
        command.append(str(eigenvalue_base))
    return command


def _calibrate_maps(
    bart: Path,
    calib_base: Path,
    maps_base: Path,
    crop: float,
    output_dir: Path,
    *,
    soft: bool,
    save_eigenvalues: bool,
) -> dict[str, Any]:
    """Run BART one-map ESPIRiT calibration on the compressed measured ACS."""
    eigenvalue_base = output_dir / "eigenvalue_maps_bart" if save_eigenvalues else None
    result = _run_logged(
        build_ecalib_command(
            bart,
            calib_base,
            maps_base,
            crop,
            soft=soft,
            eigenvalue_base=eigenvalue_base,
        ),
        output_dir / "ecalib.log",
    )
    result["crop"] = crop
    result["soft_sense"] = soft
    result["map_sets"] = 1
    result["output"] = validate_finite_bart(maps_base, (*GRID_SHAPE, NCC, 1))
    result["output_cfl_sha256"] = sha256_file(maps_base.with_suffix(".cfl"))
    if eigenvalue_base is not None:
        result["eigenvalue_output"] = validate_finite_bart(
            eigenvalue_base, (*GRID_SHAPE, 1, 1)
        )
        result["eigenvalue_output"]["base"] = str(eigenvalue_base)
        result["eigenvalue_output"]["cfl_sha256"] = sha256_file(
            eigenvalue_base.with_suffix(".cfl")
        )
    return result


def bart_maps_to_coilfirst(maps: np.ndarray) -> np.ndarray:
    """Expose one BART map set as a coil-first view without copying 1.6 GB."""
    padded_shape = maps.shape + (1,) * max(0, 6 - maps.ndim)
    logical = maps.reshape(padded_shape[:5] + (-1,), order="F")
    if logical.shape[:5] != (*GRID_SHAPE, NCC, 1) or logical.shape[5] != 1:
        raise ValueError(f"Expected one [256,256,256,12] map set, got {maps.shape}.")
    return np.moveaxis(logical[:, :, :, :, 0, 0], -1, 0)


def bart_kspace_to_coilfirst(kspace: np.ndarray) -> np.ndarray:
    """Expose BART Cartesian k-space as a coil-first view for SigPy."""
    if kspace.shape != (*GRID_SHAPE, NCC):
        raise ValueError(f"Expected measured k-space {(*GRID_SHAPE, NCC)}, got {kspace.shape}.")
    return np.moveaxis(kspace, -1, 0)


def run_sigpy_sense(
    kspace_cf: np.ndarray,
    maps_cf: np.ndarray,
    pe1_mask: np.ndarray,
    *,
    max_iterations: int,
    absolute_tolerance: float,
    coil_batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve masked Cartesian SENSE with SigPy's conjugate-gradient application."""
    import sigpy as sp
    import sigpy.mri as mr

    if kspace_cf.shape != (NCC, *GRID_SHAPE) or maps_cf.shape != kspace_cf.shape:
        raise ValueError("SigPy k-space and map arrays must both be [12,256,256,256].")
    if pe1_mask.shape != (GRID_SHAPE[1],) or pe1_mask.dtype != np.bool_:
        raise ValueError("The explicit PE1 sampling mask must be bool[256].")
    if max_iterations < 1 or coil_batch_size < 1:
        raise ValueError("CG iteration and coil-batch counts must be positive.")

    # A singleton-broadcast mask avoids allocating a redundant 256^3 weight array.
    weights = pe1_mask.astype(np.float32, copy=False)[None, :, None]
    started = time.perf_counter()
    application = mr.app.SenseRecon(
        kspace_cf,
        maps_cf,
        lamda=0.0,
        weights=weights,
        device=sp.cpu_device,
        coil_batch_size=min(coil_batch_size, NCC),
        max_iter=max_iterations,
        tol=absolute_tolerance,
        show_pbar=True,
        leave_pbar=True,
    )
    initial_residual = float(application.alg.resid)
    image = np.asarray(application.run(), dtype=np.complex64)
    final_residual = float(application.alg.resid)
    if image.shape != GRID_SHAPE or not np.isfinite(image).all():
        raise ValueError("SigPy SENSE returned an invalid image.")
    return image, {
        "implementation": "sigpy.mri.app.SenseRecon",
        "sigpy_version": sp.__version__,
        "device": "CPU",
        "lambda": 0.0,
        "explicit_sampling_mask": True,
        "max_iterations": max_iterations,
        "completed_iterations": int(application.alg.iter),
        "absolute_tolerance": absolute_tolerance,
        "coil_batch_size": min(coil_batch_size, NCC),
        "initial_normal_equation_residual": initial_residual,
        "final_normal_equation_residual": final_residual,
        "relative_normal_equation_residual": final_residual / initial_residual,
        "not_positive_definite": bool(application.alg.not_positive_definite),
        "wall_seconds": time.perf_counter() - started,
    }


def synthesize_model_kspace(
    image: np.ndarray,
    maps_cf: np.ndarray,
    measured_cf: np.ndarray,
    pe1_mask: np.ndarray,
    output_path: Path | None,
) -> dict[str, Any]:
    """Apply ``F(Sx)`` coil-wise, save it, and measure acquired-data consistency."""
    import sigpy as sp

    output = None
    if output_path is not None:
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.complex64, shape=(*GRID_SHAPE, NCC)
        )
    residual_squared = 0.0
    measured_squared = 0.0
    started = time.perf_counter()
    for coil in range(NCC):
        coil_image = np.asarray(maps_cf[coil]) * image
        predicted = np.asarray(
            sp.fft(coil_image, axes=(0, 1, 2), center=True), dtype=np.complex64
        )
        if output is not None:
            output[:, :, :, coil] = predicted
        acquired_prediction = predicted[:, pe1_mask, :]
        acquired_measured = np.asarray(measured_cf[coil])[:, pe1_mask, :]
        difference = acquired_prediction - acquired_measured
        residual_squared += float(np.vdot(difference, difference).real)
        measured_squared += float(np.vdot(acquired_measured, acquired_measured).real)
    if output is not None:
        output.flush()
        del output
    residual = float(np.sqrt(residual_squared))
    measured_norm = float(np.sqrt(measured_squared))
    return {
        "model": "coil_kspace[c] = centered_orthonormal_fft3(map[c] * image)",
        "output": None if output_path is None else str(output_path),
        "output_shape": [*GRID_SHAPE, NCC],
        "output_size_bytes": None if output_path is None else output_path.stat().st_size,
        "acquired_residual_norm": residual,
        "acquired_measured_norm": measured_norm,
        "acquired_relative_residual": residual / measured_norm,
        "wall_seconds": time.perf_counter() - started,
    }


def canonicalize_to_ras(
    data: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Reorient a lossless 3D array/affine pair to canonical RAS storage."""
    import nibabel as nib

    source = nib.orientations.io_orientation(affine)
    target = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(source, target)
    canonical = nib.orientations.apply_orientation(data, transform)
    canonical_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
    return np.ascontiguousarray(canonical), canonical_affine, transform.tolist()


def export_niftis(
    image: np.ndarray,
    *,
    twix: Path,
    measurement_index: int,
    output_dir: Path,
    subject: str,
    reconstruction: dict[str, Any],
) -> dict[str, Any]:
    """Save RAS-canonical magnitude/phase NIfTIs with corrected product geometry."""
    import nibabel as nib

    reference_recon = Path(__file__).resolve().parents[3] / "external" / "wave-mprage" / "recon"
    sys.path.insert(0, str(reference_recon))
    from utils.nifti_export_twix import (
        apply_array_axis_flips,
        make_nifti_affine_from_twix,
        normalize_magnitude,
        save_nifti_with_json,
    )

    magnitude, magnitude_info = normalize_magnitude(np.abs(image).astype(np.float32))
    phase = np.angle(image).astype(np.float32)
    magnitude, phase = apply_array_axis_flips(
        (magnitude, phase), PHYSICAL_ARRAY_FLIPS
    )
    native_affine, voxel_size, twix_info = make_nifti_affine_from_twix(
        twix_file=twix,
        scan_index=measurement_index,
        npy_shape=GRID_SHAPE,
        twix_array_axis_roles=AXIS_ROLES,
        twix_array_axis_flips=AFFINE_AXIS_FLIPS,
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=(1.0, 1.0, 1.0),
    )
    native_codes = list(nib.aff2axcodes(native_affine))
    magnitude, canonical_affine, transform = canonicalize_to_ras(magnitude, native_affine)
    phase, phase_affine, phase_transform = canonicalize_to_ras(phase, native_affine)
    if not np.allclose(canonical_affine, phase_affine) or transform != phase_transform:
        raise ValueError("Magnitude and phase orientation transforms differ.")
    canonical_codes = list(nib.aff2axcodes(canonical_affine))
    if canonical_codes != ["R", "A", "S"]:
        raise ValueError(f"Corrected NIfTI orientation is not RAS: {canonical_codes}.")

    nifti_dir = output_dir / "nifti"
    outputs = []
    common = {
        "Description": "Emergency Phase G unregularized no-wave SENSE reconstruction",
        "SourceTwix": str(twix),
        "CoilCount": NCC,
        "SensitivityCalibration": "BART ecalib from measured compressed no-wave ACS",
        "SenseImplementation": reconstruction["implementation"],
        "Lambda": 0.0,
        "NativeArrayAxisRoles": list(AXIS_ROLES),
        "PhysicalArrayFlipsApplied": list(PHYSICAL_ARRAY_FLIPS),
        "AffineAxisFlips": list(AFFINE_AXIS_FLIPS),
        "NativeAffineAxisCodes": native_codes,
        "CanonicalOrientationTransform": transform,
        "StoredAxisCodes": canonical_codes,
        "VoxelSizeMm": list(voxel_size),
        "TwixOrientation": twix_info,
        "OrientationCorrectionReason": (
            "Product DICOM audit showed stored data maps to RAS by transpose(2,1,0) "
            "without flips; the previous IAL affine had reversed axis signs."
        ),
    }
    for part, data in (("mag", magnitude), ("phase", phase)):
        stem = f"sub-{subject}_part-{part}_NoWaveSENSELambda0"
        nii_path = nifti_dir / f"{stem}.nii.gz"
        json_path = nifti_dir / f"{stem}.json"
        metadata = {
            **common,
            "Part": part,
            "Units": "normalized arbitrary" if part == "mag" else "rad",
            "MagnitudeNormalization": magnitude_info if part == "mag" else None,
        }
        save_nifti_with_json(data, canonical_affine, nii_path, json_path, metadata)
        saved = nib.load(str(nii_path))
        if saved.shape != GRID_SHAPE or nib.aff2axcodes(saved.affine) != ("R", "A", "S"):
            raise ValueError(f"Saved NIfTI geometry validation failed: {nii_path}")
        outputs.append({"part": part, "nifti": str(nii_path), "json": str(json_path)})

    quicklook = _save_quicklook(magnitude, output_dir / "sense_quicklook.png")
    return {
        "outputs": outputs,
        "stored_shape": list(GRID_SHAPE),
        "stored_axis_codes": canonical_codes,
        "canonical_affine": canonical_affine.tolist(),
        "quicklook": str(quicklook),
    }


def _save_quicklook(magnitude_ras: np.ndarray, path: Path) -> Path:
    """Save central planes and both interpretations of the reported axial slice 75."""
    import matplotlib.pyplot as plt

    axial_indices = (75, magnitude_ras.shape[2] - 1 - 75)
    panels = (
        (magnitude_ras[magnitude_ras.shape[0] // 2, :, :], "mid sagittal"),
        (magnitude_ras[:, magnitude_ras.shape[1] // 2, :], "mid coronal"),
        (magnitude_ras[:, :, axial_indices[0]], f"axial {axial_indices[0]}"),
        (magnitude_ras[:, :, axial_indices[1]], f"axial {axial_indices[1]} (reverse count)"),
    )
    positive = magnitude_ras[magnitude_ras > 0]
    vmax = float(np.percentile(positive, 99.5))
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, (plane, title) in zip(axes, panels):
        axis.imshow(plane.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("No-wave SENSE lambda=0 — canonical RAS")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate provenance, calibrate maps, reconstruct, and export diagnostics."""
    started = time.perf_counter()
    bart = args.bart.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    calib_base = bart_base(args.calib_base.expanduser().resolve())
    calib_manifest_path = args.calib_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    twix = args.twix.expanduser().resolve()
    prepare_manifest_path = prepared_dir / "prepare_manifest.json"
    kspace_base = prepared_dir / "no_wave_kspace_measured"
    mask_path = prepared_dir / "pe1_sampling_mask.npy"
    required = (
        bart,
        twix,
        prepare_manifest_path,
        kspace_base.with_suffix(".cfl"),
        mask_path,
        calib_base.with_suffix(".cfl"),
        calib_manifest_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0 < args.ecalib_crop <= 1 or args.absolute_tolerance < 0:
        raise ValueError("Invalid ecalib crop or CG tolerance.")

    prepare_manifest = _load_json(prepare_manifest_path)
    calibration_manifest = _load_json(calib_manifest_path)
    _validate_provenance(prepare_manifest, calibration_manifest)
    if Path(prepare_manifest["source_twix"]).resolve() != twix:
        raise ValueError("Prepared k-space and requested TWIX paths differ.")
    _prepare_output_directory(output_dir, args.overwrite)

    version = subprocess.run(
        [str(bart), "version"], check=True, capture_output=True, text=True
    )
    maps_base = output_dir / "coil_sens_bart"
    ecalib = _calibrate_maps(
        bart,
        calib_base,
        maps_base,
        args.ecalib_crop,
        output_dir,
        soft=args.ecalib_soft,
        save_eigenvalues=args.save_ecalib_eigenvalues,
    )

    kspace = open_bart_memmap(kspace_base)
    maps = open_bart_memmap(maps_base)
    kspace_cf = bart_kspace_to_coilfirst(kspace)
    maps_cf = bart_maps_to_coilfirst(maps)
    pe1_mask = np.asarray(np.load(mask_path), dtype=bool)
    image, reconstruction = run_sigpy_sense(
        kspace_cf,
        maps_cf,
        pe1_mask,
        max_iterations=args.max_iterations,
        absolute_tolerance=args.absolute_tolerance,
        coil_batch_size=args.coil_batch_size,
    )
    image_path = output_dir / "image_sense_lambda0.npy"
    np.save(image_path, image)
    reconstruction["output"] = str(image_path)
    reconstruction["output_size_bytes"] = image_path.stat().st_size

    modeled_path = None
    if not args.skip_modeled_kspace:
        modeled_path = output_dir / "model_no_wave_kspace_ncc12.npy"
    model_consistency = synthesize_model_kspace(
        image, maps_cf, kspace_cf, pe1_mask, modeled_path
    )
    nifti = export_niftis(
        image,
        twix=twix,
        measurement_index=args.measurement_index,
        output_dir=output_dir,
        subject=args.subject,
        reconstruction=reconstruction,
    )

    manifest = {
        "format_version": 1,
        "status": "no_wave_sense_lambda0_complete_awaiting_visual_review",
        "pipeline_label": "Emergency Phase G",
        "prepared_input_manifest": str(prepare_manifest_path),
        "calibration_input_manifest": str(calib_manifest_path),
        "coil_compression": {
            "physical_coils": 64,
            "virtual_coils": NCC,
            "basis": prepare_manifest["coil_basis"],
            "basis_file_sha256": prepare_manifest["coil_basis_file_sha256"],
            "basis_columns_half_open": [0, NCC],
        },
        "bart_executable": str(bart),
        "bart_version_output": (version.stdout + version.stderr).strip(),
        "ecalib": ecalib,
        "sense": reconstruction,
        "model_consistency": model_consistency,
        "nifti": nifti,
        "total_wall_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "manifest.json"
    temporary = Path(str(manifest_path) + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"SENSE image: {image_path}")
    print(f"Relative acquired residual: {model_consistency['acquired_relative_residual']:.6g}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and map expected failures to status 2."""
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
