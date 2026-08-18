#!/usr/bin/env python3
"""Synthesize full Wave k-space with a sequence-derived theoretical PSF."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from wave_synthesis import (
    SPATIAL_AXES,
    apply_wave_forward,
    build_theoretical_psf,
    center_embed_readout,
    centered_fftn,
    generate_theoretical_wave_trajectory,
    logical_bart_cfl_sha256,
    logical_array_sha256,
    sha256_file,
)


AXIS_ROLES = ("phase", "readout", "slice")
AXIS_FLIPS = (True, False, False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kspace", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--twix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", default="20260817product")
    parser.add_argument("--measurement-index", type=int, default=1)
    parser.add_argument("--nx-extended", type=int, default=1024)
    parser.add_argument("--ncalib", type=int, default=72)
    parser.add_argument("--nacs", type=int, default=32)
    parser.add_argument("--orientation", default="SAG", choices=("SAG", "TRA"))
    parser.add_argument("--yflip", type=int, default=-1, choices=(-1, 1))
    parser.add_argument("--zflip", type=int, default=-1, choices=(-1, 1))
    parser.add_argument("--diagnostic-coils", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument("--fft-workers", type=int, default=4)
    parser.add_argument(
        "--reference-recon",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "wave-mprage" / "recon",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _save_diagnostic_nifti(
    image: np.ndarray,
    *,
    coil_number: int,
    args: argparse.Namespace,
    affine: np.ndarray,
    voxel_size_mm: Sequence[float],
    twix_info: dict[str, Any],
    output_dir: Path,
    save_nifti_with_json: Any,
    apply_array_axis_flips: Any,
) -> list[dict[str, Any]]:
    """Save unnormalized magnitude/phase for one direct-IFFT Wave coil."""
    magnitude = np.abs(image).astype(np.float32, copy=False)
    phase = np.angle(image).astype(np.float32, copy=False)
    magnitude, phase = apply_array_axis_flips((magnitude, phase), AXIS_FLIPS)
    outputs = []
    for part, data in (("mag", magnitude), ("phase", phase)):
        stem = (
            f"sub-{args.subject}_echo-01_coil-{coil_number:02d}_"
            f"part-{part}_fullwave-directifft"
        )
        nii_path = output_dir / f"{stem}.nii.gz"
        json_path = output_dir / f"{stem}.json"
        sidecar = {
            "Description": "Direct IFFT of full synthetic Wave k-space; not de-Waved",
            "SourceKSpace": str(args.kspace.expanduser().resolve()),
            "SourceSequence": str(args.sequence.expanduser().resolve()),
            "EchoNumber": 1,
            "VirtualCoilNumber": coil_number,
            "Part": part,
            "Units": "arbitrary" if part == "mag" else "rad",
            "MagnitudeNormalization": "none",
            "ImageShape": list(data.shape),
            "VoxelSizeMm": [float(value) for value in voxel_size_mm],
            "ExtendedReadoutFovMm": float(data.shape[0]),
            "NIfTITwixArrayAxisRoles": list(AXIS_ROLES),
            "NIfTIPhysicalArrayFlipsApplied": list(AXIS_FLIPS),
            "TwixOrientation": twix_info,
        }
        save_nifti_with_json(data, affine, nii_path, json_path, sidecar)
        outputs.append({"part": part, "nifti": str(nii_path), "json": str(json_path)})
    return outputs


def _save_montage(diagnostic_dir: Path, coils: Sequence[int], part: str) -> Path:
    """Save a compact full-extended-FOV central-slice montage."""
    import matplotlib.pyplot as plt
    import nibabel as nib

    paths = [
        next(diagnostic_dir.glob(f"*_coil-{coil:02d}_part-{part}_*.nii.gz"))
        for coil in coils
    ]
    figure, axes = plt.subplots(len(paths), 1, figsize=(12, 2.8 * len(paths)), squeeze=False)
    for axis, path, coil in zip(axes.ravel(), paths, coils):
        image = nib.load(str(path))
        plane = np.asanyarray(image.dataobj[:, :, image.shape[2] // 2])
        if part == "mag":
            positive = plane[plane > 0]
            vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
            axis.imshow(plane.T, cmap="gray", origin="lower", vmin=0, vmax=vmax, aspect="auto")
        else:
            axis.imshow(
                plane.T, cmap="twilight", origin="lower", vmin=-np.pi, vmax=np.pi, aspect="auto"
            )
        axis.set_title(f"Virtual coil {coil:02d} — {part}")
        axis.axis("off")
    figure.tight_layout()
    output = diagnostic_dir / f"fullwave_directifft_{part}_central_slice_montage.png"
    figure.savefig(output, dpi=120)
    plt.close(figure)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    kspace_path = args.kspace.expanduser().resolve()
    sequence_path = args.sequence.expanduser().resolve()
    twix_path = args.twix.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    reference_recon = args.reference_recon.expanduser().resolve()
    for path in (kspace_path, sequence_path, twix_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(reference_recon))
    from bart.bart_utils.bart_io import write_cfl
    from utils.nifti_export_twix import (
        apply_array_axis_flips,
        make_nifti_affine_from_twix,
        save_nifti_with_json,
    )

    no_wave = np.load(kspace_path, mmap_mode="r")
    expected_input = (256, 256, 256, 12)
    if no_wave.shape != expected_input or no_wave.dtype != np.complex64:
        raise ValueError(
            f"Expected active [RO,PE1,PE2,Ncc] k-space {expected_input} complex64, "
            f"got {no_wave.shape} {no_wave.dtype}."
        )
    diagnostic_coils = sorted(set(int(value) for value in args.diagnostic_coils))
    if any(value < 1 or value > no_wave.shape[3] for value in diagnostic_coils):
        raise ValueError("Diagnostic coil numbers are one-based and must be within Ncc.")

    started = time.perf_counter()
    delta_ky, delta_kz, trajectory_info = generate_theoretical_wave_trajectory(
        sequence_path,
        nx_os=args.nx_extended,
        ncalib=args.ncalib,
        nacs=args.nacs,
        orientation=args.orientation,
    )
    psf = build_theoretical_psf(
        delta_ky,
        delta_kz,
        ny=no_wave.shape[1],
        nz=no_wave.shape[2],
        yflip=args.yflip,
        zflip=args.zflip,
    )
    expected_psf = (args.nx_extended, no_wave.shape[1], no_wave.shape[2])
    if psf.shape != expected_psf or not np.allclose(np.abs(psf), 1.0, atol=2e-7):
        raise ValueError("Theoretical PSF shape or unit-magnitude invariant failed.")
    psf_hash = logical_array_sha256(psf)
    psf_path = output_dir / "theoretical_psf.npy"
    np.save(psf_path, psf)
    bart_dir = output_dir / "bart_inputs"
    bart_psf_base = write_cfl(bart_dir / "psf", psf[..., None, None])
    bart_psf_shape = (*expected_psf, 1, 1)
    bart_psf_hash = logical_bart_cfl_sha256(bart_psf_base, bart_psf_shape)
    if bart_psf_hash != psf_hash:
        raise ValueError("Canonical and BART theoretical PSF logical hashes differ.")

    output_path = output_dir / "full_wave_kspace.npy"
    full_wave = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(args.nx_extended, no_wave.shape[1], no_wave.shape[2], no_wave.shape[3]),
        fortran_order=True,
    )
    affine, voxel_size_mm, twix_info = make_nifti_affine_from_twix(
        twix_file=twix_path,
        scan_index=args.measurement_index,
        npy_shape=expected_psf,
        twix_array_axis_roles=AXIS_ROLES,
        twix_array_axis_flips=(False, False, False),
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=(1.0, 1.0, 1.0),
    )

    diagnostics: list[dict[str, Any]] = []
    squared_norm = 0.0
    finite = True
    support = None
    for coil_index in range(no_wave.shape[3]):
        coil_number = coil_index + 1
        coil_kspace = np.asarray(no_wave[..., coil_index], dtype=np.complex64)
        coil_image = centered_fftn(
            coil_kspace, axes=SPATIAL_AXES, inverse=True, workers=args.fft_workers
        )
        extended, current_support = center_embed_readout(coil_image, args.nx_extended)
        support = current_support if support is None else support
        if current_support != support:
            raise RuntimeError("Readout embedding support changed between coils.")
        wave_kspace = apply_wave_forward(extended, psf, workers=args.fft_workers)
        finite &= bool(np.isfinite(wave_kspace).all())
        squared_norm += float(np.vdot(wave_kspace, wave_kspace).real)
        full_wave[..., coil_index] = wave_kspace

        if coil_number in diagnostic_coils:
            direct_image = centered_fftn(
                wave_kspace, axes=SPATIAL_AXES, inverse=True, workers=args.fft_workers
            )
            diagnostics.append(
                {
                    "coil": coil_number,
                    "finite": bool(np.isfinite(direct_image).all()),
                    "outputs": _save_diagnostic_nifti(
                        direct_image,
                        coil_number=coil_number,
                        args=args,
                        affine=affine,
                        voxel_size_mm=voxel_size_mm,
                        twix_info=twix_info,
                        output_dir=diagnostic_dir,
                        save_nifti_with_json=save_nifti_with_json,
                        apply_array_axis_flips=apply_array_axis_flips,
                    ),
                }
            )
        print(f"Completed Wave synthesis coil {coil_number:02d}/{no_wave.shape[3]:02d}", flush=True)

    full_wave.flush()
    del full_wave
    montages = [
        str(_save_montage(diagnostic_dir, diagnostic_coils, part))
        for part in ("mag", "phase")
    ]
    bart_psf_bytes = bart_psf_base.with_suffix(".cfl").stat().st_size
    expected_bart_bytes = int(np.prod(bart_psf_shape, dtype=np.int64) * 8)
    if bart_psf_bytes != expected_bart_bytes:
        raise ValueError("BART PSF byte count does not match its expected complex64 shape.")

    manifest = {
        "format_version": 1,
        "status": "awaiting_visual_review_before_mask_and_bart",
        "source_no_wave_kspace": str(kspace_path),
        "source_sequence": str(sequence_path),
        "source_sequence_sha256": sha256_file(sequence_path),
        "theoretical_trajectory_reference": str(
            reference_recon / "recon_wave_mprage_from_twix_integrated_nifti.py"
        ),
        "theoretical_trajectory_reference_sha256": sha256_file(
            reference_recon / "recon_wave_mprage_from_twix_integrated_nifti.py"
        ),
        "source_twix": str(twix_path),
        "input_layout": ["RO", "PE1", "PE2", "Ncc"],
        "input_shape": list(expected_input),
        "readout_axis": 0,
        "extended_image_shape": [args.nx_extended, 256, 256, 12],
        "readout_embedding_half_open": [int(support.start), int(support.stop)],
        "fft_convention": "fftshift(fftn/ifftn(ifftshift(x)), norm='ortho')",
        "forward_operator": "F_RO -> theoretical_PSF -> F_PE1_PE2",
        "trajectory": trajectory_info,
        "psf": {
            "kind": "theoretical_sequence_trajectory_without_calibrated_correction",
            "shape": list(psf.shape),
            "dtype": str(psf.dtype),
            "yflip": int(args.yflip),
            "zflip": int(args.zflip),
            "logical_sha256": psf_hash,
            "npy": str(psf_path),
            "bart_base": str(bart_psf_base),
            "bart_shape": list(bart_psf_shape),
            "bart_logical_sha256": bart_psf_hash,
            "bart_cfl_bytes": bart_psf_bytes,
        },
        "full_wave_kspace": {
            "path": str(output_path),
            "shape": [args.nx_extended, 256, 256, 12],
            "dtype": "complex64",
            "fortran_order": True,
            "size_bytes": output_path.stat().st_size,
            "all_samples_finite": finite,
            "norm": float(np.sqrt(squared_norm)),
            "sampling_mask_applied": False,
        },
        "diagnostic_coils": diagnostic_coils,
        "diagnostics": diagnostics,
        "montages": montages,
        "runtime_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
