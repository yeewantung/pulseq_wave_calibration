#!/usr/bin/env python3
"""Convert one BART Wave image to MPRAGE magnitude and phase NIfTI files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import open_cfl  # noqa: E402
from wave_retro_lr.mprage import load_wave_mprage_helpers  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and export one BART image as two NIfTI parts.

    Args:
        argv: Optional argument vector; ``None`` reads the process arguments.

    Returns:
        Zero after magnitude and phase files are written successfully.
    """
    args = _parser().parse_args(argv)
    convert(args.bart_inputs, args.image, args.twix, args.seq, args.output, args.suffix)
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the BART-image-to-NIfTI command-line interface.

    Returns:
        Parser describing the prepared inputs, image, raw data, and output.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bart-inputs", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path, help="BART image CFL basename.")
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--seq", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="NIfTI destination.")
    parser.add_argument("--suffix", default="BARTWaveMPRAGE")
    return parser


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load one prepared-input JSON manifest.

    Args:
        path: Manifest path expected to contain a JSON object.

    Returns:
        Parsed manifest mapping.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _case_geometry(
    manifest: dict[str, Any],
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Resolve logical shape and physical XYZ resolution from a manifest.

    Args:
        manifest: Native or retrospective BART-input manifest.

    Returns:
        Logical ``(RO, LIN, PAR)`` shape and physical ``(X, Y, Z)`` spacing.
    """
    if "case" in manifest:
        case = manifest["case"]
        logical = tuple(int(value) for value in case["target_logical_matrix_ro_lin_par"])
        physical = tuple(float(value) for value in case["achieved_resolution_mm_xyz"])
        return logical, physical
    geometry = manifest["geometry"]
    logical = tuple(int(value) for value in geometry["logical_matrix_ro_lin_par"])
    fov = tuple(float(value) for value in geometry["physical_fov_mm_xyz"])
    physical_matrix = (logical[2], logical[1], logical[0])
    physical = tuple(value / size for value, size in zip(fov, physical_matrix, strict=True))
    return logical, physical


def _recorded_commands(inputs: Path, image: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Read the exact ecalib and Wave commands associated with an image.

    Args:
        inputs: Prepared BART-input directory for the current case.
        image: BART reconstructed-image basename.
        manifest: Native or retrospective prepared-input manifest.

    Returns:
        NIfTI metadata fields containing both shell-escaped commands.
    """
    wave_record = image.parent / "wave_command.txt"
    dataset_root = inputs.parents[2] if "case" in manifest else inputs.parents[1]
    ecalib_record = dataset_root / "normal" / "bart_output" / "ecalib_command.txt"
    for label, path in (("wave", wave_record), ("ecalib", ecalib_record)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing recorded {label} command: {path}")
    return {
        "BARTWaveCommand": wave_record.read_text(encoding="utf-8").strip(),
        "BARTEcalibCommand": ecalib_record.read_text(encoding="utf-8").strip(),
    }


def convert(
    bart_inputs: Path,
    image_base: Path,
    twix: Path,
    sequence: Path,
    output: Path,
    suffix: str,
) -> None:
    """Restore BART normalization and write magnitude plus phase NIfTI files.

    Args:
        bart_inputs: Directory containing the prepared case manifest.
        image_base: BART complex image CFL basename.
        twix: Source TWIX path used to derive the physical affine.
        sequence: Matching sequence path retained for provenance validation.
        output: Destination directory for NIfTI files and JSON sidecars.
        suffix: Dataset-independent suffix used in output filenames.

    Returns:
        None. The function writes magnitude and phase NIfTI/JSON pairs.
    """
    inputs = bart_inputs.expanduser().resolve()
    image_path = image_base.expanduser().resolve()
    twix_path = twix.expanduser().resolve()
    sequence_path = sequence.expanduser().resolve()
    manifest = _load_manifest(inputs / "manifest.json")
    logical_shape, physical_resolution_xyz = _case_geometry(manifest)
    image = np.asarray(open_cfl(image_path)).squeeze()
    if image.shape != logical_shape or not np.isfinite(image).all():
        raise ValueError(
            f"BART image shape/finite check failed: {image.shape}, "
            f"expected {logical_shape}."
        )
    kspace_norm = float(manifest["echoes"][0]["wave_kspace_norm"])
    if not np.isfinite(kspace_norm) or kspace_norm <= 0:
        raise ValueError("BART Wave k-space norm must be positive and finite.")
    restored = image.astype(np.complex64, copy=False) * kspace_norm

    native = load_wave_mprage_helpers()
    metadata = {
        "Reconstruction": "BART Wave MPRAGE",
        "ReconstructionSoftware": "BART wave",
        "BARTWaveKspaceNormRestored": kspace_norm,
        "BARTInternalNormalizationRestored": True,
        "BARTOutputAlreadyReadoutDeoversampled": True,
        "PreparedInputManifest": str(inputs / "manifest.json"),
        **_recorded_commands(inputs, image_path, manifest),
    }
    # The native exporter expects logical RO/LIN/PAR voxel sizes, whereas the
    # case manifest records physical X/Y/Z.
    voxel_size_logical = (
        physical_resolution_xyz[2],
        physical_resolution_xyz[1],
        physical_resolution_xyz[0],
    )
    native.save_mprage_output_to_nifti(
        image=restored,
        twix_file=str(twix_path),
        out_folder=str(output.expanduser().resolve()),
        nifti_sub=inputs.parent.name,
        suffix=native._sanitize_filename_component(suffix),
        tag_wave="wave",
        file_tag="",
        voxel_size_mm=voxel_size_logical,
        crop_readout_os=1,
        save_phase=True,
        twix_array_axis_roles=("phase", "readout", "slice"),
        twix_array_axis_flips=(True, False, False),
        metadata=metadata,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
