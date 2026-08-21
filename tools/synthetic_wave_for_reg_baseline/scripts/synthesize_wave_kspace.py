#!/usr/bin/env python3
"""Synthesize full Wave k-space with a sequence-derived theoretical PSF."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from checkpoint_io import write_json_atomic
from dataset_manifest import (
    DatasetManifest,
    DatasetManifestError,
    load_dataset_manifest,
    load_passed_inspection,
)
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


@dataclass(frozen=True)
class WaveSynthesisInputs:
    """Resolved dataset and sequence contract for full Wave encoding."""

    dataset_manifest: DatasetManifest | None
    inspection_report: Path | None
    source_report: Path | None
    kspace: Path
    sequence: Path
    twix: Path
    output_dir: Path
    subject: str
    measurement_index: int
    matrix_rolinpar: tuple[int, int, int]
    fov_mm_rolinpar: tuple[float, float, float]
    virtual_coils: int
    nx_extended: int
    ncalib: int
    nacs: int
    orientation: str
    yflip: int
    zflip: int
    diagnostic_coils: tuple[int, ...]


def _build_parser() -> argparse.ArgumentParser:
    """Build the theoretical Wave synthesis command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--kspace", type=Path)
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--twix", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subject")
    parser.add_argument("--measurement-index", type=int)
    parser.add_argument("--nx-extended", type=int)
    parser.add_argument("--ncalib", type=int)
    parser.add_argument("--nacs", type=int)
    parser.add_argument("--orientation", choices=("SAG", "TRA"))
    parser.add_argument("--yflip", type=int, choices=(-1, 1))
    parser.add_argument("--zflip", type=int, choices=(-1, 1))
    parser.add_argument("--diagnostic-coils", nargs="+", type=int)
    parser.add_argument("--fft-workers", type=int, default=4)
    parser.add_argument(
        "--reference-recon",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "external" / "wave-mprage" / "recon",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only a complete manifest-backed synthesis with intact provenance.",
    )
    return parser


def _source_record(report: Mapping[str, Any]) -> Mapping[str, Any]:
    record = report.get("assembly", report.get("reconstruction"))
    if not isinstance(record, Mapping):
        raise ValueError("Source report has neither direct assembly nor GRAPPA reconstruction.")
    return record


def resolve_wave_synthesis_inputs(args: argparse.Namespace) -> WaveSynthesisInputs:
    """Resolve one manifest-backed dataset or the compatible explicit interface."""
    explicit = (
        args.kspace,
        args.sequence,
        args.twix,
        args.output_dir,
        args.subject,
        args.measurement_index,
        args.nx_extended,
        args.ncalib,
        args.nacs,
        args.orientation,
        args.yflip,
        args.zflip,
        args.diagnostic_coils,
    )
    if args.dataset_manifest is not None:
        if any(value is not None for value in explicit) or args.overwrite:
            raise ValueError(
                "--dataset-manifest cannot be combined with explicit dataset/Wave options "
                "or --overwrite"
            )
        manifest = load_dataset_manifest(args.dataset_manifest)
        inspection = load_passed_inspection(manifest)
        contract = manifest.payload
        reconstruction = contract["reconstruction"]
        wave = contract["wave_synthesis"]
        prefix = manifest.output_path("source_reconstruction_prefix")
        kspace = prefix.with_name(
            prefix.name + f"_full_ncc{int(reconstruction['virtual_coils'])}.npy"
        )
        source_report = prefix.with_name(prefix.name + "_report.json")
        if not source_report.is_file():
            raise FileNotFoundError(f"Source reconstruction report not found: {source_report}")
        report = json.loads(source_report.read_text(encoding="utf-8"))
        if report.get("dataset_manifest", {}).get("sha256") != manifest.sha256:
            raise ValueError("Source reconstruction report uses a stale dataset manifest.")
        record = _source_record(report)
        expected_shape = [
            *[int(value) for value in contract["geometry"]["matrix"]],
            int(reconstruction["virtual_coils"]),
        ]
        if (
            Path(record.get("output", "")).resolve() != kspace
            or record.get("shape") != expected_shape
            or record.get("finite") is not True
        ):
            raise ValueError("Source reconstruction report does not validate expected k-space.")
        return WaveSynthesisInputs(
            dataset_manifest=manifest,
            inspection_report=manifest.inspection_report,
            source_report=source_report,
            kspace=kspace,
            sequence=manifest.input_path("wave_sequence"),
            twix=manifest.input_path("twix"),
            output_dir=manifest.output_path("wave_synthesis_dir"),
            subject=manifest.subject,
            measurement_index=int(inspection["twix"]["selected_measurement_index"]),
            matrix_rolinpar=tuple(int(value) for value in contract["geometry"]["matrix"]),
            fov_mm_rolinpar=tuple(float(value) for value in contract["geometry"]["fov_mm"]),
            virtual_coils=int(reconstruction["virtual_coils"]),
            nx_extended=int(wave["extended_readout_samples"]),
            ncalib=int(wave["calibration_ncalib1"]),
            nacs=int(wave["calibration_nacs"]),
            orientation=str(wave["orientation"]),
            yflip=int(wave["pe1_phase_sign"]),
            zflip=int(wave["pe2_phase_sign"]),
            diagnostic_coils=tuple(int(value) for value in wave["diagnostic_coils"]),
        )

    required = (args.kspace, args.sequence, args.twix, args.output_dir)
    if any(value is None for value in required):
        raise ValueError(
            "Use --dataset-manifest, or provide --kspace, --sequence, --twix, and --output-dir"
        )
    if args.resume:
        raise ValueError("--resume requires --dataset-manifest")
    matrix = (256, 256, 256)
    return WaveSynthesisInputs(
        dataset_manifest=None,
        inspection_report=None,
        source_report=None,
        kspace=args.kspace.expanduser().resolve(),
        sequence=args.sequence.expanduser().resolve(),
        twix=args.twix.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        subject="20260817product" if args.subject is None else args.subject,
        measurement_index=1 if args.measurement_index is None else args.measurement_index,
        matrix_rolinpar=matrix,
        fov_mm_rolinpar=(256.0, 256.0, 256.0),
        virtual_coils=12,
        nx_extended=1024 if args.nx_extended is None else args.nx_extended,
        ncalib=72 if args.ncalib is None else args.ncalib,
        nacs=32 if args.nacs is None else args.nacs,
        orientation="SAG" if args.orientation is None else args.orientation,
        yflip=-1 if args.yflip is None else args.yflip,
        zflip=-1 if args.zflip is None else args.zflip,
        diagnostic_coils=tuple(
            (1, 2, 3, 4) if args.diagnostic_coils is None else args.diagnostic_coils
        ),
    )


def _save_diagnostic_nifti(
    image: np.ndarray,
    *,
    coil_number: int,
    inputs: WaveSynthesisInputs,
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
            f"sub-{inputs.subject}_echo-01_coil-{coil_number:02d}_"
            f"part-{part}_fullwave-directifft"
        )
        nii_path = output_dir / f"{stem}.nii.gz"
        json_path = output_dir / f"{stem}.json"
        sidecar = {
            "Description": "Direct IFFT of full synthetic Wave k-space; not de-Waved",
            "SourceKSpace": str(inputs.kspace),
            "SourceSequence": str(inputs.sequence),
            "EchoNumber": 1,
            "VirtualCoilNumber": coil_number,
            "Part": part,
            "Units": "arbitrary" if part == "mag" else "rad",
            "MagnitudeNormalization": "none",
            "ImageShape": list(data.shape),
            "VoxelSizeMm": [float(value) for value in voxel_size_mm],
            "ExtendedReadoutFovMm": float(data.shape[0] * voxel_size_mm[0]),
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


def completed_synthesis_reusable(
    manifest_path: Path,
    inputs: WaveSynthesisInputs,
) -> bool:
    """Accept complete reuse only when dataset/source provenance and payloads match."""
    if inputs.dataset_manifest is None or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "awaiting_visual_review_before_mask_and_bart":
            return False
        if manifest.get("dataset_manifest", {}).get("sha256") != inputs.dataset_manifest.sha256:
            return False
        if manifest.get("source_reconstruction_report", {}).get("sha256") != sha256_file(
            inputs.source_report
        ):
            return False
        full_wave_info = manifest["full_wave_kspace"]
        psf_info = manifest["psf"]
        output = Path(full_wave_info["path"])
        psf = Path(psf_info["npy"])
        bart_psf_base = Path(psf_info["bart_base"])
        if not output.is_file() or not psf.is_file():
            return False
        expected_output = (
            inputs.nx_extended,
            inputs.matrix_rolinpar[1],
            inputs.matrix_rolinpar[2],
            inputs.virtual_coils,
        )
        output_array = np.load(output, mmap_mode="r")
        psf_array = np.load(psf, mmap_mode="r")
        expected_bart_psf = (*expected_output[:3], 1, 1)
        return (
            output.resolve() == inputs.output_dir / "full_wave_kspace.npy"
            and psf.resolve() == inputs.output_dir / "theoretical_psf.npy"
            and output_array.shape == expected_output
            and output_array.dtype == np.complex64
            and full_wave_info.get("all_samples_finite") is True
            and float(full_wave_info.get("norm", 0.0)) > 0
            and psf_array.shape == expected_output[:3]
            and psf_array.dtype == np.complex64
            and logical_array_sha256(psf_array) == psf_info["logical_sha256"]
            and tuple(psf_info["bart_shape"]) == expected_bart_psf
            and logical_bart_cfl_sha256(bart_psf_base, expected_bart_psf)
            == psf_info["logical_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Apply the sequence-derived Wave forward model coil by coil."""
    inputs = resolve_wave_synthesis_inputs(args)
    kspace_path = inputs.kspace
    sequence_path = inputs.sequence
    twix_path = inputs.twix
    output_dir = inputs.output_dir
    reference_recon = args.reference_recon.expanduser().resolve()
    for path in (kspace_path, sequence_path, twix_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.resume and completed_synthesis_reusable(manifest_path, inputs):
            print(f"Reusing validated full-Wave synthesis: {output_dir}")
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not safely reusable: {output_dir}")
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
    expected_input = (*inputs.matrix_rolinpar, inputs.virtual_coils)
    if no_wave.shape != expected_input or no_wave.dtype != np.complex64:
        raise ValueError(
            f"Expected active [RO,PE1,PE2,Ncc] k-space {expected_input} complex64, "
            f"got {no_wave.shape} {no_wave.dtype}."
        )
    diagnostic_coils = sorted(set(inputs.diagnostic_coils))
    if any(value < 1 or value > no_wave.shape[3] for value in diagnostic_coils):
        raise ValueError("Diagnostic coil numbers are one-based and must be within Ncc.")

    started = time.perf_counter()
    delta_ky, delta_kz, trajectory_info = generate_theoretical_wave_trajectory(
        sequence_path,
        nx_os=inputs.nx_extended,
        ncalib=inputs.ncalib,
        nacs=inputs.nacs,
        orientation=inputs.orientation,
    )
    psf = build_theoretical_psf(
        delta_ky,
        delta_kz,
        ny=no_wave.shape[1],
        nz=no_wave.shape[2],
        yflip=inputs.yflip,
        zflip=inputs.zflip,
    )
    expected_psf = (inputs.nx_extended, no_wave.shape[1], no_wave.shape[2])
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
        shape=(inputs.nx_extended, no_wave.shape[1], no_wave.shape[2], no_wave.shape[3]),
        fortran_order=True,
    )
    affine, voxel_size_mm, twix_info = make_nifti_affine_from_twix(
        twix_file=twix_path,
        scan_index=inputs.measurement_index,
        npy_shape=expected_psf,
        twix_array_axis_roles=AXIS_ROLES,
        twix_array_axis_flips=(False, False, False),
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=tuple(
            fov / matrix
            for fov, matrix in zip(inputs.fov_mm_rolinpar, inputs.matrix_rolinpar)
        ),
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
        extended, current_support = center_embed_readout(coil_image, inputs.nx_extended)
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
                        inputs=inputs,
                        affine=affine,
                        voxel_size_mm=voxel_size_mm,
                        twix_info=twix_info,
                        output_dir=diagnostic_dir,
                        save_nifti_with_json=save_nifti_with_json,
                        apply_array_axis_flips=apply_array_axis_flips,
                    ),
                }
            )
        print(
            f"Completed Wave synthesis coil {coil_number:02d}/{no_wave.shape[3]:02d}",
            flush=True,
        )

    full_wave.flush()
    del full_wave
    if not finite or squared_norm <= 0:
        raise ValueError("Full synthetic Wave k-space is non-finite or has no signal energy.")
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
        "extended_image_shape": [
            inputs.nx_extended,
            inputs.matrix_rolinpar[1],
            inputs.matrix_rolinpar[2],
            inputs.virtual_coils,
        ],
        "readout_embedding_half_open": [int(support.start), int(support.stop)],
        "fft_convention": "fftshift(fftn/ifftn(ifftshift(x)), norm='ortho')",
        "forward_operator": "F_RO -> theoretical_PSF -> F_PE1_PE2",
        "trajectory": trajectory_info,
        "psf": {
            "kind": "theoretical_sequence_trajectory_without_calibrated_correction",
            "shape": list(psf.shape),
            "dtype": str(psf.dtype),
            "yflip": inputs.yflip,
            "zflip": inputs.zflip,
            "logical_sha256": psf_hash,
            "npy": str(psf_path),
            "bart_base": str(bart_psf_base),
            "bart_shape": list(bart_psf_shape),
            "bart_logical_sha256": bart_psf_hash,
            "bart_cfl_bytes": bart_psf_bytes,
        },
        "full_wave_kspace": {
            "path": str(output_path),
            "shape": [
                inputs.nx_extended,
                inputs.matrix_rolinpar[1],
                inputs.matrix_rolinpar[2],
                inputs.virtual_coils,
            ],
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
    if inputs.dataset_manifest is not None:
        manifest["dataset_manifest"] = inputs.dataset_manifest.provenance()
        manifest["dataset_inspection"] = {
            "path": str(inputs.inspection_report),
            "sha256": sha256_file(inputs.inspection_report),
        }
        manifest["source_reconstruction_report"] = {
            "path": str(inputs.source_report),
            "sha256": sha256_file(inputs.source_report),
        }
    write_json_atomic(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run synthetic Wave generation from command-line arguments."""
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        DatasetManifestError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
