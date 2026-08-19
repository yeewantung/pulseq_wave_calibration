#!/usr/bin/env python3
"""Export a compact RSS NIfTI from GRAPPA-completed multicoil k-space."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from phase_c_export_coil_nifti import (
    DEFAULT_AXIS_FLIPS,
    DEFAULT_AXIS_ROLES,
    centered_ifft3,
)
from run_no_wave_sense import AFFINE_AXIS_FLIPS, canonicalize_to_ras


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse paths and TWIX orientation settings for the RSS export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kspace", type=Path, required=True)
    parser.add_argument("--twix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-index", type=int, default=1)
    parser.add_argument(
        "--canonical-ras",
        action="store_true",
        help="Apply the product-DICOM-validated affine correction and store RAS data.",
    )
    parser.add_argument(
        "--reference-recon",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "external"
        / "wave-mprage"
        / "recon",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Compute coil RSS one channel at a time and save it with TWIX geometry."""
    kspace_path = args.kspace.expanduser().resolve()
    twix_path = args.twix.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reference_recon = args.reference_recon.expanduser().resolve()
    if not kspace_path.is_file() or not twix_path.is_file():
        raise FileNotFoundError("The k-space or matching TWIX file does not exist.")

    sys.path.insert(0, str(reference_recon))
    from utils.nifti_export_twix import (  # pylint: disable=import-error
        apply_array_axis_flips,
        make_nifti_affine_from_twix,
        save_nifti_with_json,
    )

    kspace = np.load(kspace_path, mmap_mode="r")
    if kspace.ndim != 4 or not np.iscomplexobj(kspace):
        raise ValueError(f"Expected complex [RO, PE1, PE2, coil], got {kspace.shape}.")
    spatial_shape = tuple(int(value) for value in kspace.shape[:3])
    rss_squared = np.zeros(spatial_shape, dtype=np.float32)
    started = time.perf_counter()
    for coil in range(kspace.shape[-1]):
        coil_image = centered_ifft3(kspace[..., coil])
        rss_squared += np.abs(coil_image).astype(np.float32) ** 2
    rss = np.sqrt(rss_squared, out=rss_squared)
    (rss,) = apply_array_axis_flips((rss,), DEFAULT_AXIS_FLIPS)

    affine_flips = AFFINE_AXIS_FLIPS if args.canonical_ras else (False, False, False)
    affine, voxel_size_mm, twix_info = make_nifti_affine_from_twix(
        twix_file=twix_path,
        scan_index=args.measurement_index,
        npy_shape=spatial_shape,
        twix_array_axis_roles=DEFAULT_AXIS_ROLES,
        twix_array_axis_flips=affine_flips,
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=(1.0, 1.0, 1.0),
    )
    orientation_transform = None
    if args.canonical_ras:
        rss, affine, orientation_transform = canonicalize_to_ras(rss, affine)
    sidecar_path = output_path.with_suffix("").with_suffix(".json")
    metadata: dict[str, object] = {
        "SourceKSpace": str(kspace_path),
        "SourceTwix": str(twix_path),
        "Combination": "root-sum-of-squares across virtual coils",
        "FFTConvention": "fftshift(ifftn(ifftshift(kspace), norm='ortho'))",
        "InputArrayLayout": ["readout", "LIN", "PAR", "virtual_coil"],
        "NIfTITwixArrayAxisRoles": list(DEFAULT_AXIS_ROLES),
        "NIfTIPhysicalArrayFlipsApplied": list(DEFAULT_AXIS_FLIPS),
        "NIfTIAffineAxisFlips": list(affine_flips),
        "NIfTICanonicalRAS": bool(args.canonical_ras),
        "NIfTIOrientationTransform": orientation_transform,
        "NIfTIVoxelSizeMm": list(voxel_size_mm),
        "TwixOrientation": twix_info,
        "RuntimeSeconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti_with_json(rss, affine, output_path, sidecar_path, metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line RSS export and report its output path."""
    args = _parse_args(argv)
    metadata = run(args)
    print(f"RSS NIfTI: {args.output.expanduser().resolve()}")
    print(f"Runtime: {metadata['RuntimeSeconds']:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
