#!/usr/bin/env python3
"""Export Phase C multicoil k-space as per-coil magnitude/phase NIfTI files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_AXIS_ROLES = ("phase", "readout", "slice")
DEFAULT_AXIS_FLIPS = (True, False, False)


def centered_ifft3(kspace: np.ndarray) -> np.ndarray:
    """Apply the reference Wave reconstruction's centered orthonormal 3D IFFT."""
    kspace = np.asarray(kspace)
    if kspace.ndim != 3:
        raise ValueError(f"Expected 3D single-coil k-space, got {kspace.shape}.")
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace), norm="ortho")
    ).astype(np.complex64, copy=False)


def output_basename(subject: str, echo: int, coil: int, part: str) -> str:
    """Return a BIDS-like filename stem with explicit echo, coil, and component."""
    if part not in {"mag", "phase"}:
        raise ValueError("part must be 'mag' or 'phase'.")
    return f"sub-{subject}_echo-{echo:02d}_coil-{coil:02d}_part-{part}_GRAPPA"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kspace", type=Path, required=True)
    parser.add_argument("--twix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", default="20260817product")
    parser.add_argument("--measurement-index", type=int, default=1)
    parser.add_argument("--echo-index", type=int, default=1)
    parser.add_argument(
        "--reference-recon",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "wave-mprage" / "recon",
        help="Wave-MPRAGE recon folder containing utils/nifti_export_twix.py.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_existing(nii_path: Path, json_path: Path, expected_shape: tuple[int, ...]) -> bool:
    """Return true when an existing output pair is complete and has the right shape."""
    if not nii_path.is_file() or not json_path.is_file():
        return False
    import nibabel as nib

    return tuple(int(v) for v in nib.load(str(nii_path)).shape) == expected_shape


def run(args: argparse.Namespace) -> dict[str, Any]:
    kspace_path = args.kspace.expanduser().resolve()
    twix_path = args.twix.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    reference_recon = args.reference_recon.expanduser().resolve()
    if not kspace_path.is_file() or not twix_path.is_file():
        raise FileNotFoundError("The reconstructed k-space or matching TWIX file is missing.")
    if not (reference_recon / "utils" / "nifti_export_twix.py").is_file():
        raise FileNotFoundError(f"Reference NIfTI utility not found below {reference_recon}.")

    sys.path.insert(0, str(reference_recon))
    from utils.nifti_export_twix import (  # pylint: disable=import-error
        apply_array_axis_flips,
        make_nifti_affine_from_twix,
        save_nifti_with_json,
    )

    kspace = np.load(kspace_path, mmap_mode="r")
    if kspace.ndim != 4 or not np.iscomplexobj(kspace):
        raise ValueError(
            f"Expected complex [RO, PE1, PE2, coil] k-space, got {kspace.shape} {kspace.dtype}."
        )
    spatial_shape = tuple(int(v) for v in kspace.shape[:3])
    ncoil = int(kspace.shape[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    # The reference MPRAGE pipeline physically flips axis 0 before constructing
    # its affine, so the affine itself is requested without a second flip.
    affine, voxel_size_mm, twix_info = make_nifti_affine_from_twix(
        twix_file=twix_path,
        scan_index=args.measurement_index,
        npy_shape=spatial_shape,
        twix_array_axis_roles=DEFAULT_AXIS_ROLES,
        twix_array_axis_flips=(False, False, False),
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=(1.0, 1.0, 1.0),
    )

    started = time.perf_counter()
    outputs: list[dict[str, Any]] = []
    for coil_index in range(ncoil):
        coil_number = coil_index + 1
        image = centered_ifft3(kspace[..., coil_index])
        magnitude = np.abs(image).astype(np.float32, copy=False)
        phase = np.angle(image).astype(np.float32, copy=False)
        magnitude, phase = apply_array_axis_flips(
            (magnitude, phase), DEFAULT_AXIS_FLIPS
        )

        for part, data in (("mag", magnitude), ("phase", phase)):
            basename = output_basename(
                args.subject, args.echo_index, coil_number, part
            )
            nii_path = output_dir / f"{basename}.nii.gz"
            json_path = output_dir / f"{basename}.json"
            if not args.overwrite and _validate_existing(
                nii_path, json_path, spatial_shape
            ):
                status = "reused"
            else:
                sidecar = {
                    "SourceKSpace": str(kspace_path),
                    "SourceTwix": str(twix_path),
                    "MeasurementIndex": int(args.measurement_index),
                    "EchoNumber": int(args.echo_index),
                    "VirtualCoilNumber": coil_number,
                    "Part": part,
                    "Units": "arbitrary" if part == "mag" else "rad",
                    "MagnitudeNormalization": "none",
                    "FFTConvention": "fftshift(ifftn(ifftshift(kspace), norm='ortho'))",
                    "InputArrayLayout": ["readout", "LIN", "PAR", "virtual_coil"],
                    "NIfTITwixArrayAxisRoles": list(DEFAULT_AXIS_ROLES),
                    "NIfTIPhysicalArrayFlipsApplied": list(DEFAULT_AXIS_FLIPS),
                    "NIfTIVoxelSizeMm": list(voxel_size_mm),
                    "OrientationSource": "TwixMeasYaps via wave-mprage reference utility",
                    "TwixOrientation": twix_info,
                }
                save_nifti_with_json(data, affine, nii_path, json_path, sidecar)
                status = "written"
            outputs.append(
                {
                    "coil": coil_number,
                    "echo": int(args.echo_index),
                    "part": part,
                    "nifti": str(nii_path),
                    "json": str(json_path),
                    "status": status,
                }
            )
        print(f"Completed virtual coil {coil_number:02d}/{ncoil:02d}", flush=True)

    manifest = {
        "format_version": 1,
        "source_kspace": str(kspace_path),
        "source_twix": str(twix_path),
        "input_shape": list(kspace.shape),
        "input_dtype": str(kspace.dtype),
        "echo_count": 1,
        "echo_index": int(args.echo_index),
        "virtual_coil_count": ncoil,
        "parts": ["mag", "phase"],
        "output_shape": list(spatial_shape),
        "voxel_size_mm": list(voxel_size_mm),
        "axis_roles": list(DEFAULT_AXIS_ROLES),
        "axis_flips": list(DEFAULT_AXIS_FLIPS),
        "magnitude_normalization": "none",
        "runtime_seconds": time.perf_counter() - started,
        "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
