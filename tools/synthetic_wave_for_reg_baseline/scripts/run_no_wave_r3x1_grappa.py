#!/usr/bin/env python3
"""Run accepted 5x5x5 GRAPPA on retrospective no-Wave R3x1 data.

The input is fully sampled NCC=12 no-Wave k-space in ``[RO, PE1, PE2,
coil]`` order. The measured product R3x1 lattice and its 24-line ACS band
are applied before the existing joint-coil GRAPPA implementation is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from bart_cfl import bart_base, open_bart_memmap, sha256_file
from checkpoint_io import (
    open_or_create_complex64_npy,
    validate_resume_pair,
    write_json_atomic,
)
from export_multicoil_nifti import centered_ifft3
from grappa_3d_r3 import apply_grappa_3d_block, pe2_offsets
from nifti_phase_resume import validate_phase_record
from presentation_metrics import (
    evaluate_against_direct_fft,
    nifti_sidecar_path,
    validate_metrics_reference_manifest,
)
from reconstruct_no_wave_grappa_3d import calibrate_weights
from run_no_wave_r3x1_pics_sweep import (
    FULLY_SAMPLED_PE1_LINES,
    GRID_SHAPE,
    NCC,
    PE1_ACCELERATION,
    PE1_RESIDUE,
    build_r3x1_pe1_mask,
)


ACCEPTED_PE2_KERNEL_SIZE = 5
ACCEPTED_REGULARIZATION = 0.01


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _progress_start(path: Path, *, resume: bool) -> int:
    if resume and path.is_file():
        payload = _load_json(path)
        return int(payload["next_partition"])
    return 0


def reconstruct_retrospective_grappa(
    source: np.ndarray,
    acquired_mask: np.ndarray,
    weights: dict[int, np.ndarray],
    output_path: Path,
    progress_path: Path,
    *,
    chunk_size: int,
    pe2_kernel_size: int,
    resume: bool,
) -> dict[str, Any]:
    """Complete missing PE1 samples in haloed PE2 chunks."""
    if source.ndim != 4 or source.dtype != np.complex64:
        raise ValueError(f"Expected complex64 [RO, PE1, PE2, coil], got {source.shape}.")
    shape = tuple(int(value) for value in source.shape)
    mask = np.asarray(acquired_mask, dtype=bool)
    if mask.shape != (shape[1],) or chunk_size < 1:
        raise ValueError("Invalid acquired mask or reconstruction chunk size.")
    validate_resume_pair(output_path, progress_path, resume=resume)
    start_partition = _progress_start(progress_path, resume=resume)
    if not 0 <= start_partition <= shape[2]:
        raise ValueError("Resume partition is outside the reconstruction volume.")

    output = open_or_create_complex64_npy(output_path, shape, resume=resume)
    acquired_lines = np.flatnonzero(mask)
    halo = pe2_kernel_size // 2
    started = time.perf_counter()
    for start in range(start_partition, shape[2], chunk_size):
        stop = min(start + chunk_size, shape[2])
        halo_start = max(0, start - halo)
        halo_stop = min(shape[2], stop + halo)
        block = np.zeros(
            (shape[0], shape[1], halo_stop - halo_start, shape[3]),
            dtype=np.complex64,
        )
        block[:, acquired_lines, :, :] = np.asarray(
            source[:, acquired_lines, halo_start:halo_stop, :], dtype=np.complex64
        )
        core = np.arange(start - halo_start, stop - halo_start)
        reconstructed = apply_grappa_3d_block(
            block,
            core,
            mask,
            weights,
            acquired_residue=PE1_RESIDUE,
            pe2_kernel_size=pe2_kernel_size,
        )
        expected = np.asarray(source[:, acquired_lines, start:stop, :])
        if not np.array_equal(reconstructed[:, acquired_lines, :, :], expected):
            raise RuntimeError("GRAPPA changed acquired source samples.")
        if not np.isfinite(reconstructed).all():
            raise RuntimeError("GRAPPA produced non-finite samples.")
        output[:, :, start:stop, :] = reconstructed
        output.flush()
        write_json_atomic(
            progress_path,
            {"next_partition": stop, "complete": stop == shape[2]},
        )
        print(f"GRAPPA reconstruction checkpoint: PE2 {stop}/{shape[2]}", flush=True)

    acquired_error_count = 0
    nonfinite_count = 0
    predicted_nonzero_count = 0
    for start in range(0, shape[2], max(1, chunk_size)):
        stop = min(start + max(1, chunk_size), shape[2])
        block = np.asarray(output[:, :, start:stop, :])
        nonfinite_count += int(np.count_nonzero(~np.isfinite(block)))
        acquired_error_count += int(
            np.count_nonzero(
                block[:, acquired_lines, :, :]
                != np.asarray(source[:, acquired_lines, start:stop, :])
            )
        )
        predicted_nonzero_count += int(np.count_nonzero(block[:, ~mask, :, :]))
    if nonfinite_count or acquired_error_count or predicted_nonzero_count == 0:
        raise RuntimeError("Completed GRAPPA checkpoint failed sample validation.")
    return {
        "path": str(output_path),
        "shape": list(shape),
        "dtype": "complex64",
        "all_samples_finite": True,
        "acquired_samples_equal_source_bitwise": True,
        "predicted_nonzero_count": predicted_nonzero_count,
        "runtime_seconds_this_invocation": time.perf_counter() - started,
    }


def rss_magnitude_with_sensitivity_phase(
    kspace: np.ndarray,
    maps: np.ndarray,
    *,
    map_power_threshold_fraction: float,
) -> np.ndarray:
    """Return standard RSS magnitude with a sensitivity-aligned phase.

    Sensitivity maps affect only the phase reference. The magnitude remains
    the conventional GRAPPA root-sum-of-squares coil combination.
    """
    if kspace.ndim != 4 or not np.iscomplexobj(kspace):
        raise ValueError("GRAPPA k-space must be a complex four-dimensional array.")
    map_array = np.asarray(maps)
    while map_array.ndim > 4 and map_array.shape[-1] == 1:
        map_array = map_array[..., 0]
    if map_array.shape != kspace.shape or not np.iscomplexobj(map_array):
        raise ValueError(f"Map shape {map_array.shape} does not match {kspace.shape}.")
    if (
        not math.isfinite(map_power_threshold_fraction)
        or not 0 <= map_power_threshold_fraction < 1
    ):
        raise ValueError("Map-power threshold fraction must be finite in [0, 1).")

    rss_squared = np.zeros(kspace.shape[:3], dtype=np.float32)
    phase_reference = np.zeros(kspace.shape[:3], dtype=np.complex64)
    map_power = np.zeros(kspace.shape[:3], dtype=np.float32)
    for coil in range(kspace.shape[-1]):
        coil_image = centered_ifft3(np.asarray(kspace[..., coil]))
        sensitivity = np.asarray(map_array[..., coil], dtype=np.complex64)
        if not np.isfinite(coil_image).all() or not np.isfinite(sensitivity).all():
            raise ValueError("Coil image or sensitivity map contains non-finite samples.")
        rss_squared += np.square(np.abs(coil_image), dtype=np.float32)
        phase_reference += np.conj(sensitivity) * coil_image
        map_power += np.square(np.abs(sensitivity), dtype=np.float32)
        print(f"Coil combination: {coil + 1}/{kspace.shape[-1]}", flush=True)
    rss = np.sqrt(rss_squared, out=rss_squared)
    threshold = float(map_power_threshold_fraction * np.max(map_power))
    supported = map_power > threshold
    combined = rss.astype(np.complex64)
    combined[supported] *= np.exp(
        1j * np.angle(phase_reference[supported])
    ).astype(np.complex64)
    if not np.isfinite(combined).all() or not np.any(np.abs(combined) > 0):
        raise RuntimeError("Combined GRAPPA image is empty or non-finite.")
    return combined


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    os.replace(temporary, path)


def _load_or_combine_image(
    kspace: np.ndarray,
    maps: np.ndarray,
    output_path: Path,
    *,
    map_power_threshold_fraction: float,
    resume: bool,
) -> np.ndarray:
    if resume and output_path.is_file():
        image = np.load(output_path, mmap_mode="r")
        if image.shape != kspace.shape[:3] or image.dtype != np.complex64:
            raise ValueError("Saved combined image is incompatible with this run.")
        if not np.isfinite(image).all() or not np.any(np.abs(image) > 0):
            raise ValueError("Saved combined image is invalid.")
        print(f"Reusing combined GRAPPA image: {output_path}")
        return image
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite combined image: {output_path}")
    image = rss_magnitude_with_sensitivity_phase(
        kspace,
        maps,
        map_power_threshold_fraction=map_power_threshold_fraction,
    )
    _save_npy_atomic(output_path, image)
    return np.load(output_path, mmap_mode="r")


def _export_nifti(
    image: np.ndarray,
    *,
    output_dir: Path,
    twix: Path,
    sequence: Path,
    subject: str,
    map_power_threshold_fraction: float,
    complex_image_sha256: str,
) -> dict[str, Any]:
    recon_root = Path(__file__).resolve().parents[3] / "external" / "wave-mprage" / "recon"
    if str(recon_root) not in sys.path:
        sys.path.insert(0, str(recon_root))
    import recon_wave_mprage_from_twix_integrated_nifti as native

    seq = native.pp.Sequence()
    seq.read(str(sequence), remove_duplicates=False)
    definitions = seq.definitions
    geometry = native._derive_hardcoded_sag_logical_geometry(definitions)
    native._assert_sag_geometry(definitions)
    voxel_size_mm = native._derive_nifti_voxel_size_mm(definitions, geometry)
    metadata = {
        "Description": "Retrospectively sampled no-Wave R3x1 GRAPPA reconstruction",
        "ReconstructionSoftware": "Local accepted joint-coil GRAPPA implementation",
        "ReconstructionModel": "Cartesian 5x5x5 GRAPPA without Wave PSF",
        "Acceleration": {"PE1": PE1_ACCELERATION, "PE2": 1},
        "PE1Residue": PE1_RESIDUE,
        "FullySampledPE1Lines": list(FULLY_SAMPLED_PE1_LINES),
        "VirtualCoils": NCC,
        "GRAPPAKernelROPE1PE2": [5, 5, ACCEPTED_PE2_KERNEL_SIZE],
        "GRAPPARegularization": ACCEPTED_REGULARIZATION,
        "MagnitudeCombination": "root-sum-of-squares across virtual coils",
        "PhaseCombination": "phase of sum(conj(sensitivity_map) * coil_image)",
        "PhaseMapPowerThresholdFraction": map_power_threshold_fraction,
        "SensitivityMapsUsedForReconstruction": False,
        "ComplexCombinedImageSHA256": complex_image_sha256,
    }
    nifti_dir = output_dir / "nifti"
    native.save_mprage_output_to_nifti(
        image=np.asarray(image),
        twix_file=str(twix),
        out_folder=nifti_dir,
        nifti_sub=f"{subject}-no-wave-r3x1-grappa",
        suffix="GRAPPANoWaveR3x1",
        tag_wave="nowave",
        file_tag="grappa-5x5x5",
        voxel_size_mm=voxel_size_mm,
        crop_readout_os=1,
        save_phase=True,
        metadata=metadata,
    )
    magnitude_outputs = sorted(nifti_dir.rglob("*part-mag*.nii.gz"))
    phase_outputs = sorted(nifti_dir.rglob("*part-phase*.nii.gz"))
    if len(magnitude_outputs) != 1 or len(phase_outputs) != 1:
        raise ValueError("Expected exactly one magnitude and one phase NIfTI.")
    magnitude = nib.load(str(magnitude_outputs[0]))
    if magnitude.shape != GRID_SHAPE or nib.aff2axcodes(magnitude.affine) != ("R", "A", "S"):
        raise ValueError("GRAPPA magnitude NIfTI geometry validation failed.")
    magnitude_sidecar = nifti_sidecar_path(magnitude_outputs[0])
    phase_sidecar = nifti_sidecar_path(phase_outputs[0])
    record = {
        "magnitude_nifti": str(magnitude_outputs[0]),
        "magnitude_nifti_sha256": sha256_file(magnitude_outputs[0]),
        "magnitude_sidecar": str(magnitude_sidecar),
        "magnitude_sidecar_sha256": sha256_file(magnitude_sidecar),
        "shape": list(magnitude.shape),
        "axis_codes": list(nib.aff2axcodes(magnitude.affine)),
        "phase_nifti": str(phase_outputs[0]),
        "phase_nifti_sha256": sha256_file(phase_outputs[0]),
        "phase_sidecar": str(phase_sidecar),
        "phase_sidecar_sha256": sha256_file(phase_sidecar),
    }
    if not validate_phase_record(record, expected_shape=GRID_SHAPE):
        raise ValueError("GRAPPA phase NIfTI validation failed.")
    return record


def _reuse_existing_nifti(
    output_dir: Path,
    *,
    complex_image_sha256: str,
) -> dict[str, Any] | None:
    """Reuse only a complete export bound to the unchanged combined image."""
    nifti_dir = output_dir / "nifti"
    if not nifti_dir.exists():
        return None
    magnitude_outputs = sorted(nifti_dir.rglob("*part-mag*.nii.gz"))
    phase_outputs = sorted(nifti_dir.rglob("*part-phase*.nii.gz"))
    if not magnitude_outputs and not phase_outputs:
        return None
    if len(magnitude_outputs) != 1 or len(phase_outputs) != 1:
        raise FileExistsError("Existing GRAPPA NIfTI export is incomplete or ambiguous.")
    magnitude_sidecar = nifti_sidecar_path(magnitude_outputs[0])
    phase_sidecar = nifti_sidecar_path(phase_outputs[0])
    if not magnitude_sidecar.is_file() or not phase_sidecar.is_file():
        raise FileExistsError("Existing GRAPPA NIfTI export has a missing sidecar.")
    for sidecar in (magnitude_sidecar, phase_sidecar):
        if _load_json(sidecar).get("ComplexCombinedImageSHA256") != complex_image_sha256:
            raise FileExistsError("Existing GRAPPA NIfTI is not bound to this complex image.")
    magnitude = nib.load(str(magnitude_outputs[0]))
    record = {
        "magnitude_nifti": str(magnitude_outputs[0]),
        "magnitude_nifti_sha256": sha256_file(magnitude_outputs[0]),
        "magnitude_sidecar": str(magnitude_sidecar),
        "magnitude_sidecar_sha256": sha256_file(magnitude_sidecar),
        "shape": list(magnitude.shape),
        "axis_codes": list(nib.aff2axcodes(magnitude.affine)),
        "phase_nifti": str(phase_outputs[0]),
        "phase_nifti_sha256": sha256_file(phase_outputs[0]),
        "phase_sidecar": str(phase_sidecar),
        "phase_sidecar_sha256": sha256_file(phase_sidecar),
    }
    if not validate_phase_record(record, expected_shape=GRID_SHAPE):
        raise FileExistsError("Existing GRAPPA NIfTI export failed validation.")
    print(f"Reusing complete GRAPPA magnitude/phase export: {nifti_dir}")
    return record


def _complete_manifest_reusable(
    manifest_path: Path,
    *,
    metrics_reference_manifest: Path,
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    nifti = manifest.get("nifti", {})
    magnitude = Path(nifti.get("magnitude_nifti", ""))
    metrics = manifest.get("direct_fft_metrics", {})
    if (
        manifest.get("status") == "complete"
        and magnitude.is_file()
        and sha256_file(magnitude) == nifti.get("magnitude_nifti_sha256")
        and validate_phase_record(nifti, expected_shape=GRID_SHAPE)
        and metrics.get("status") == "complete"
        and metrics.get("metrics_reference_manifest", {}).get("sha256")
        == sha256_file(metrics_reference_manifest)
    ):
        return manifest
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_no_wave_kspace.expanduser().resolve()
    maps_base = bart_base(args.maps.expanduser().resolve())
    twix = args.twix.expanduser().resolve()
    sequence = args.sequence.expanduser().resolve()
    metrics_reference_manifest = args.metrics_reference_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    required = (
        source_path,
        maps_base.with_suffix(".hdr"),
        maps_base.with_suffix(".cfl"),
        twix,
        sequence,
        metrics_reference_manifest,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.pe2_kernel_size != ACCEPTED_PE2_KERNEL_SIZE:
        raise ValueError("The presentation GRAPPA contract requires a 5x5x5 kernel.")
    if args.regularization != ACCEPTED_REGULARIZATION:
        raise ValueError("The presentation GRAPPA contract requires regularization 0.01.")
    if args.calibration_chunk < 1 or args.reconstruction_chunk < 1:
        raise ValueError("Chunk sizes must be positive.")
    pe2_offsets(args.pe2_kernel_size)
    metrics_context = validate_metrics_reference_manifest(metrics_reference_manifest)

    source = np.load(source_path, mmap_mode="r")
    if source.shape != (*GRID_SHAPE, NCC) or source.dtype != np.complex64:
        raise ValueError(f"Unexpected no-Wave source: {source.shape}, {source.dtype}")
    maps = open_bart_memmap(maps_base)
    padded_maps_shape = maps.shape + (1,) * max(0, 5 - maps.ndim)
    if padded_maps_shape[:5] != (*GRID_SHAPE, NCC, 1) or any(
        value != 1 for value in padded_maps_shape[5:]
    ):
        raise ValueError(f"Unexpected one-set sensitivity maps: {maps.shape}")
    mask = build_r3x1_pe1_mask()
    if args.validate_only:
        print("No-Wave R3x1 retrospective GRAPPA structural validation: PASSED")
        print("Source shape:", source.shape)
        print("Acquired PE1 lines:", int(np.count_nonzero(mask)))
        print("GRAPPA kernel: 5x5x5; Ncc: 12; regularization: 0.01")
        print("Direct-FFT metrics reference:", metrics_context["reference_path"])
        print("Output directory:", output_dir)
        return {"status": "validated"}

    manifest_path = output_dir / "manifest.json"
    if args.resume:
        reusable = _complete_manifest_reusable(
            manifest_path,
            metrics_reference_manifest=metrics_reference_manifest,
        )
        if reusable is not None:
            print(f"Reusing complete no-Wave R3x1 GRAPPA run: {manifest_path}")
            return reusable
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"GRAPPA output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / "pe1_sampling_mask.npy"
    if mask_path.is_file():
        if not args.resume or not np.array_equal(np.load(mask_path), mask):
            raise ValueError("Existing GRAPPA sampling mask is not safely reusable.")
    else:
        np.save(mask_path, mask)

    equations_path = output_dir / "normal_equations.npz"
    weights_path = output_dir / "weights.npz"
    acs_start = FULLY_SAMPLED_PE1_LINES[0]
    acs_stop = FULLY_SAMPLED_PE1_LINES[-1] + 1
    acs = source[:, acs_start:acs_stop, :, :]
    weights = calibrate_weights(
        acs,
        equations_path,
        weights_path,
        chunk_size=args.calibration_chunk,
        regularization=args.regularization,
        pe2_kernel_size=args.pe2_kernel_size,
        resume=args.resume,
    )
    kspace_path = output_dir / "grappa_completed_kspace.npy"
    progress_path = output_dir / "grappa_reconstruction_progress.json"
    reconstruction = reconstruct_retrospective_grappa(
        source,
        mask,
        weights,
        kspace_path,
        progress_path,
        chunk_size=args.reconstruction_chunk,
        pe2_kernel_size=args.pe2_kernel_size,
        resume=args.resume,
    )
    kspace = np.load(kspace_path, mmap_mode="r")
    complex_image_path = output_dir / "image_grappa_complex_combined.npy"
    combined = _load_or_combine_image(
        kspace,
        maps,
        complex_image_path,
        map_power_threshold_fraction=args.map_power_threshold_fraction,
        resume=args.resume,
    )
    complex_image_sha256 = sha256_file(complex_image_path)
    nifti = (
        _reuse_existing_nifti(
            output_dir,
            complex_image_sha256=complex_image_sha256,
        )
        if args.resume
        else None
    )
    if nifti is None:
        nifti = _export_nifti(
            combined,
            output_dir=output_dir,
            twix=twix,
            sequence=sequence,
            subject=args.subject,
            map_power_threshold_fraction=args.map_power_threshold_fraction,
            complex_image_sha256=complex_image_sha256,
        )
    direct_fft_metrics = evaluate_against_direct_fft(
        Path(nifti["magnitude_nifti"]), metrics_reference_manifest
    )
    implementation = Path(__file__).with_name("grappa_3d_r3.py")
    payload = {
        "format_version": 1,
        "status": "complete",
        "purpose": "no-Wave R3x1 presentation GRAPPA reconstruction",
        "config": {
            "algorithm": "existing joint-coil local GRAPPA",
            "kernel_ro_pe1_pe2": [5, 5, args.pe2_kernel_size],
            "virtual_coils": NCC,
            "regularization": args.regularization,
            "sampling": {
                "pe1_acceleration": PE1_ACCELERATION,
                "pe1_residue": PE1_RESIDUE,
                "pe2_acceleration": 1,
                "fully_sampled_pe1_lines": list(FULLY_SAMPLED_PE1_LINES),
                "acquired_pe1_line_count": int(np.count_nonzero(mask)),
            },
        },
        "implementation": {
            "path": str(implementation),
            "sha256": sha256_file(implementation),
            "reused_functions": [
                "calibrate_weights",
                "apply_grappa_3d_block",
                "solve_weights_3d",
            ],
        },
        "inputs": {
            "source_no_wave_kspace": str(source_path),
            "source_sha256": sha256_file(source_path),
            "maps_base": str(maps_base),
            "maps_cfl_sha256": sha256_file(maps_base.with_suffix(".cfl")),
            "twix": str(twix),
            "sequence": str(sequence),
            "sequence_sha256": sha256_file(sequence),
        },
        "sampling_mask": {
            "path": str(mask_path),
            "sha256": sha256_file(mask_path),
        },
        "calibration": {
            "normal_equations": str(equations_path),
            "normal_equations_sha256": sha256_file(equations_path),
            "weights": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
        },
        "reconstruction": {
            **reconstruction,
            "sha256": sha256_file(kspace_path),
        },
        "coil_combination": {
            "complex_image": str(complex_image_path),
            "complex_image_sha256": complex_image_sha256,
            "magnitude": "root-sum-of-squares across virtual coils",
            "phase": "sensitivity-aligned phase; maps do not affect magnitude or GRAPPA",
            "map_power_threshold_fraction": args.map_power_threshold_fraction,
        },
        "nifti": nifti,
        "direct_fft_metrics": direct_fft_metrics,
        "metrics_reference_manifest": {
            "path": str(metrics_reference_manifest),
            "sha256": metrics_context["manifest_sha256"],
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(manifest_path, payload)
    print(f"No-Wave R3x1 GRAPPA manifest: {manifest_path}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-no-wave-kspace", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--metrics-reference-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--pe2-kernel-size", type=int, default=ACCEPTED_PE2_KERNEL_SIZE)
    parser.add_argument("--regularization", type=float, default=ACCEPTED_REGULARIZATION)
    parser.add_argument("--calibration-chunk", type=int, default=4)
    parser.add_argument("--reconstruction-chunk", type=int, default=2)
    parser.add_argument("--map-power-threshold-fraction", type=float, default=1e-8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume calibration/reconstruction checkpoints and reuse complete outputs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and the frozen scientific contract without writing outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
