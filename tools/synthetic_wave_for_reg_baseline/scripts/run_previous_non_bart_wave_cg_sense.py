#!/usr/bin/env python3
"""Run the previous Torch Wave PCG-SENSE implementation on accepted inputs.

This adapter does not define a new reconstruction algorithm. It loads the
accepted BART-formatted synthetic-Wave k-space, theoretical PSF, and ESPIRiT
maps, embeds the native-readout maps on the oversampled grid exactly as the
historical integrated reconstruction did, and calls the existing
``cg_sense_wave`` implementation from ``external/wave-mprage/recon``. It then
saves exact-grid metrics against the approved direct-FFT R1 reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch

from bart_cfl import bart_base, open_bart_memmap, sha256_file
from presentation_metrics import (
    evaluate_against_direct_fft,
    magnitude_sidecar_path,
    validate_metrics_reference_manifest,
)
from wave_synthesis import logical_array_sha256


NATIVE_SHAPE = (256, 256, 256)
EXTENDED_READOUT = 1024
NCC = 12


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _coil_first_wave_view(array: np.ndarray) -> np.ndarray:
    shape = array.shape + (1,) * max(0, 5 - array.ndim)
    logical = array.reshape(shape[:5] + (-1,), order="F")
    if logical.shape[:5] != (EXTENDED_READOUT, 256, 256, NCC, 1):
        raise ValueError(f"Unexpected Wave k-space shape: {array.shape}")
    if logical.shape[5] != 1:
        raise ValueError(f"Wave k-space has unexpected trailing dimensions: {array.shape}")
    return np.moveaxis(logical[:, :, :, :, 0, 0], -1, 0)


def _psf_view(array: np.ndarray) -> np.ndarray:
    shape = array.shape + (1,) * max(0, 5 - array.ndim)
    logical = array.reshape(shape[:5] + (-1,), order="F")
    if logical.shape[:5] != (EXTENDED_READOUT, 256, 256, 1, 1):
        raise ValueError(f"Unexpected PSF shape: {array.shape}")
    if logical.shape[5] != 1:
        raise ValueError(f"PSF has unexpected trailing dimensions: {array.shape}")
    return logical[:, :, :, 0, 0, 0]


def _coil_first_map_view(array: np.ndarray) -> np.ndarray:
    shape = array.shape + (1,) * max(0, 5 - array.ndim)
    logical = array.reshape(shape[:5] + (-1,), order="F")
    if logical.shape[:5] != (*NATIVE_SHAPE, NCC, 1):
        raise ValueError(f"Unexpected sensitivity-map shape: {array.shape}")
    if logical.shape[5] != 1:
        raise ValueError(f"Sensitivity maps have multiple map sets: {array.shape}")
    return np.moveaxis(logical[:, :, :, :, 0, 0], -1, 0)


def _validate_input_manifest(
    manifest_path: Path, wave_base: Path, psf_base: Path
) -> tuple[dict[str, Any], np.ndarray]:
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "calibration_kspace_ready_for_ecalib":
        raise ValueError("BART input manifest has an unexpected status.")
    echoes = manifest.get("echoes")
    if not isinstance(echoes, list) or len(echoes) != 1:
        raise ValueError("Expected exactly one Wave-MPRAGE echo.")
    if Path(manifest.get("sampling_mask", {}).get("path", "")).is_file() is False:
        raise FileNotFoundError(manifest.get("sampling_mask", {}).get("path", ""))
    expected_wave = (manifest_path.parent / str(echoes[0].get("wave_kspace", ""))).resolve()
    expected_psf = (manifest_path.parent / str(echoes[0].get("psf", ""))).resolve()
    if expected_wave != wave_base:
        raise ValueError("Requested Wave k-space differs from the input manifest.")
    if expected_psf != psf_base:
        raise ValueError("Requested PSF differs from the input manifest.")
    mask = np.asarray(np.load(manifest["sampling_mask"]["path"]), dtype=bool)
    if mask.shape != (256, 256):
        raise ValueError(f"Unexpected sampling mask shape: {mask.shape}")
    if logical_array_sha256(mask) != manifest["sampling_mask"]["logical_sha256"]:
        raise ValueError("Sampling-mask logical hash changed.")
    return manifest, mask


def _export_nifti(
    image: np.ndarray,
    *,
    output_dir: Path,
    twix: Path,
    sequence: Path,
    subject: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    recon_root = Path(__file__).resolve().parents[3] / "external" / "wave-mprage" / "recon"
    if str(recon_root) not in sys.path:
        sys.path.insert(0, str(recon_root))
    import recon_wave_mprage_from_twix_integrated_nifti as native

    seq = native.pp.Sequence()
    seq.read(str(sequence), remove_duplicates=False)
    defs = seq.definitions
    geom = native._derive_hardcoded_sag_logical_geometry(defs)
    native._assert_sag_geometry(defs)
    voxel_size_mm = native._derive_nifti_voxel_size_mm(defs, geom)
    metadata = {
        "Description": "Previous non-BART synthetic-Wave R3x2 CG-SENSE reconstruction",
        "ReconstructionSoftware": "Torch wave-mprage legacy PCG-SENSE",
        "ReconstructionSource": (
            "external/wave-mprage/recon/utils/wave_cg_sense_precondition.py"
        ),
        "ReconstructionModel": "M * F_PE * PSF * F_RO * S",
        "Acceleration": {"PE1": 3, "PE2": 2},
        "Iterations": config["iterations"],
        "RelativeResidualTolerance": config["tolerance"],
        "Initialization": config["initialization"],
        "DiagonalCoilPowerPreconditioner": config["preconditioner"],
        "BARTReconstructionUsed": False,
    }
    nifti_dir = output_dir / "nifti"
    native.save_mprage_output_to_nifti(
        image=image,
        twix_file=str(twix),
        out_folder=nifti_dir,
        nifti_sub=subject,
        suffix="PreviousNonBARTWaveCGSENSE",
        tag_wave="wave",
        file_tag="previous-non-bart-r3x2-cg-sense",
        voxel_size_mm=voxel_size_mm,
        crop_readout_os=4,
        save_phase=False,
        metadata=metadata,
    )
    outputs = sorted(nifti_dir.rglob("*part-mag*.nii.gz"))
    if len(outputs) != 1:
        raise ValueError(f"Expected one previous-CG magnitude NIfTI, found {outputs}")
    saved = nib.load(str(outputs[0]))
    if saved.shape != NATIVE_SHAPE or nib.aff2axcodes(saved.affine) != ("R", "A", "S"):
        raise ValueError("Previous non-BART NIfTI geometry validation failed.")
    if not np.isfinite(np.asanyarray(saved.dataobj)).all():
        raise ValueError("Previous non-BART NIfTI contains non-finite samples.")
    sidecar = magnitude_sidecar_path(outputs[0])
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    return {
        "magnitude_nifti": str(outputs[0]),
        "magnitude_nifti_sha256": sha256_file(outputs[0]),
        "magnitude_sidecar": str(sidecar),
        "magnitude_sidecar_sha256": sha256_file(sidecar),
        "shape": list(saved.shape),
        "axis_codes": list(nib.aff2axcodes(saved.affine)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    wave_base = bart_base(args.wave_kspace.expanduser().resolve())
    psf_base = bart_base(args.psf.expanduser().resolve())
    maps_base = bart_base(args.maps.expanduser().resolve())
    input_manifest_path = args.bart_input_manifest.expanduser().resolve()
    twix = args.twix.expanduser().resolve()
    sequence = args.sequence.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metrics_reference_manifest = (
        args.metrics_reference_manifest.expanduser().resolve()
    )
    manifest_path = output_dir / "manifest.json"
    required = (
        wave_base.with_suffix(".hdr"),
        wave_base.with_suffix(".cfl"),
        psf_base.with_suffix(".hdr"),
        psf_base.with_suffix(".cfl"),
        maps_base.with_suffix(".hdr"),
        maps_base.with_suffix(".cfl"),
        input_manifest_path,
        twix,
        sequence,
        metrics_reference_manifest,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.iterations < 1 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        raise ValueError("Iterations and tolerance must be positive.")
    metrics_context = validate_metrics_reference_manifest(metrics_reference_manifest)
    input_manifest, mask = _validate_input_manifest(
        input_manifest_path, wave_base, psf_base
    )
    wave_memmap = open_bart_memmap(wave_base)
    psf_memmap = open_bart_memmap(psf_base)
    maps_memmap = open_bart_memmap(maps_base)
    wave_view = _coil_first_wave_view(wave_memmap)
    psf_view = _psf_view(psf_memmap)
    maps_view = _coil_first_map_view(maps_memmap)
    if not np.isfinite(psf_view).all():
        raise ValueError("PSF contains non-finite samples.")
    if args.validate_only:
        print("Previous non-BART Wave CG-SENSE structural validation: PASSED")
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(args.device))
            print(
                f"CUDA device: {torch.cuda.get_device_name(torch.device(args.device))}; "
                f"free/total GiB: {free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f}"
            )
        else:
            print(
                "CUDA visibility: unavailable in this validation session; the real run "
                "will stop unless CUDA is visible."
            )
        print(f"Wave k-space shape: {wave_view.shape}")
        print(f"PSF shape: {psf_view.shape}")
        print(f"Native maps shape: {maps_view.shape}")
        print(f"Acquired PE coordinates: {int(mask.sum())}")
        print("Direct-FFT metrics reference:", metrics_context["reference_path"])
        return {"status": "validated"}
    if manifest_path.is_file() and args.resume:
        manifest = _load_json(manifest_path)
        nifti_path = Path(manifest.get("nifti", {}).get("magnitude_nifti", ""))
        metrics_record = manifest.get("direct_fft_metrics", {})
        if (
            manifest.get("status") == "complete"
            and nifti_path.is_file()
            and sha256_file(nifti_path)
            == manifest.get("nifti", {}).get("magnitude_nifti_sha256")
            and metrics_record.get("status") == "complete"
            and metrics_record.get("metrics_reference_manifest", {}).get("sha256")
            == metrics_context["manifest_sha256"]
        ):
            print(f"Reusing previous non-BART reconstruction: {manifest_path}")
            return manifest
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not safely reusable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The tmux production launcher requires a visible CUDA device.")
    torch.cuda.set_device(device)
    print(f"Torch device: {torch.cuda.get_device_name(device)}", flush=True)

    recon_root = Path(__file__).resolve().parents[3] / "external" / "wave-mprage" / "recon"
    if str(recon_root) not in sys.path:
        sys.path.insert(0, str(recon_root))
    from utils.wave_cg_sense_precondition import cg_sense_wave

    print("Transferring Wave k-space to CUDA...", flush=True)
    y = torch.from_numpy(wave_view).to(device=device, dtype=torch.complex64)
    print("Transferring PSF to CUDA...", flush=True)
    psf = torch.from_numpy(psf_view).to(device=device, dtype=torch.complex64)
    print("Embedding accepted maps on the oversampled readout grid...", flush=True)
    native_maps = torch.from_numpy(maps_view).to(device=device, dtype=torch.complex64)
    sens = torch.zeros(
        (NCC, EXTENDED_READOUT, 256, 256), dtype=torch.complex64, device=device
    )
    start = EXTENDED_READOUT // 2 - NATIVE_SHAPE[0] // 2
    stop = start + NATIVE_SHAPE[0]
    sens[:, start:stop, :, :] = native_maps
    del native_maps
    mask_t = torch.from_numpy(mask).to(device=device).view(1, 1, 256, 256)

    config = {
        "iterations": int(args.iterations),
        "tolerance": float(args.tolerance),
        "initialization": "zero",
        "preconditioner": True,
        "combined_pe_fft": True,
        "direct_full_sampling_shortcut": False,
        "device": str(device),
        "readout_embedding_half_open": [start, stop],
    }
    reconstructed = cg_sense_wave(
        y=y,
        sens=sens,
        psf_to_use=psf,
        mask_t=mask_t,
        n_iter=args.iterations,
        tol=args.tolerance,
        init="zero",
        use_preconditioner=True,
        use_direct_if_full=False,
        combine_yz_fft=True,
    )
    if reconstructed.shape != (EXTENDED_READOUT, 256, 256):
        raise ValueError(f"Unexpected legacy reconstruction shape: {reconstructed.shape}")
    image = reconstructed.detach().cpu().numpy().astype(np.complex64, copy=False)
    if not np.isfinite(image).all():
        raise ValueError("Previous non-BART reconstruction contains non-finite samples.")
    image_path = output_dir / "image_previous_non_bart_wave_cg_sense.npy"
    np.save(image_path, image)
    nifti = _export_nifti(
        image,
        output_dir=output_dir,
        twix=twix,
        sequence=sequence,
        subject=args.subject,
        config=config,
    )
    complex_image = {
        "path": str(image_path),
        "sha256": sha256_file(image_path),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "all_samples_finite": True,
    }
    del reconstructed, image, y, sens, psf, mask_t
    torch.cuda.empty_cache()
    direct_fft_metrics = evaluate_against_direct_fft(
        Path(nifti["magnitude_nifti"]), metrics_reference_manifest
    )
    source_code = recon_root / "utils" / "wave_cg_sense_precondition.py"
    payload = {
        "format_version": 1,
        "status": "complete",
        "purpose": "previous non-BART synthetic-Wave R3x2 CG-SENSE presentation result",
        "config": config,
        "scientific_scope": {
            "new_reconstruction_algorithm_introduced": False,
            "bart_reconstruction_used": False,
            "existing_sensitivity_maps_reused": True,
            "presentation_processing_during_reconstruction": False,
        },
        "implementation": {
            "path": str(source_code),
            "sha256": sha256_file(source_code),
            "callable": "cg_sense_wave",
        },
        "inputs": {
            "bart_input_manifest": {
                "path": str(input_manifest_path),
                "sha256": sha256_file(input_manifest_path),
            },
            "wave_kspace": {
                "base": str(wave_base),
                "cfl_sha256": sha256_file(wave_base.with_suffix(".cfl")),
            },
            "psf": {
                "base": str(psf_base),
                "cfl_sha256": sha256_file(psf_base.with_suffix(".cfl")),
                "logical_sha256": input_manifest["psf"]["logical_sha256"],
            },
            "maps": {
                "base": str(maps_base),
                "cfl_sha256": sha256_file(maps_base.with_suffix(".cfl")),
            },
            "sampling_mask_logical_sha256": input_manifest["sampling_mask"][
                "logical_sha256"
            ],
        },
        "complex_image": complex_image,
        "nifti": nifti,
        "direct_fft_metrics": direct_fft_metrics,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_path, payload)
    print(f"Previous non-BART reconstruction manifest: {manifest_path}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bart-input-manifest", required=True, type=Path)
    parser.add_argument("--wave-kspace", required=True, type=Path)
    parser.add_argument("--psf", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metrics-reference-manifest", required=True, type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs, shapes, mask, and CUDA without reconstructing.",
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
