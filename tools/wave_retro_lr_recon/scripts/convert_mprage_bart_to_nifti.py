#!/usr/bin/env python3
"""Convert BART Wave MPRAGE map images to magnitude and phase NIfTI files."""

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

# BART images use logical (RO, LIN, PAR) axes with physical roles
# (SI, AP, LR) for sagittal MPRAGE. Relative to the prior (True, False, False)
# convention, DICOM comparison retained AP and reversed only SI and LR.
MPRAGE_BART_ARRAY_AXIS_FLIPS = (False, False, True)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and export one or two BART map images.

    Args:
        argv: Optional argument vector; ``None`` reads the process arguments.

    Returns:
        Zero after magnitude and phase files are written successfully.
    """
    args = _parser().parse_args(argv)
    convert(
        args.bart_inputs,
        args.image,
        args.twix,
        args.seq,
        args.output,
        args.suffix,
        map_count=args.map_count,
        ecalib_record=args.ecalib_record,
    )
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
    parser.add_argument(
        "--map-count",
        type=int,
        choices=(1, 2),
        default=1,
        help="Expected ESPIRiT map count in BART dimension 4.",
    )
    parser.add_argument(
        "--ecalib-record",
        type=Path,
        help="Optional exact ecalib-command record for an isolated experiment.",
    )
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


def _recorded_commands(
    inputs: Path,
    image: Path,
    manifest: dict[str, Any],
    ecalib_record: Path | None = None,
) -> dict[str, str]:
    """Read the exact ecalib and Wave commands associated with an image.

    Args:
        inputs: Prepared BART-input directory for the current case.
        image: BART reconstructed-image basename.
        manifest: Native or retrospective prepared-input manifest.
        ecalib_record: Optional experiment-specific ecalib command record.

    Returns:
        NIfTI metadata fields containing both shell-escaped commands.
    """
    wave_record = image.parent / "wave_command.txt"
    if ecalib_record is None:
        dataset_root = inputs.parents[2] if "case" in manifest else inputs.parents[1]
        resolved_ecalib_record = (
            dataset_root / "normal" / "bart_output" / "ecalib_command.txt"
        )
    else:
        resolved_ecalib_record = ecalib_record.expanduser().resolve()
    for label, path in (("wave", wave_record), ("ecalib", resolved_ecalib_record)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing recorded {label} command: {path}")
    return {
        "BARTWaveCommand": wave_record.read_text(encoding="utf-8").strip(),
        "BARTEcalibCommand": resolved_ecalib_record.read_text(encoding="utf-8").strip(),
    }


def _load_bart_map_images(
    image_path: Path,
    logical_shape: tuple[int, int, int],
    expected_map_count: int,
) -> np.ndarray:
    """Load and validate one BART Wave image with an explicit map dimension.

    Args:
        image_path: BART CFL basename to load.
        logical_shape: Expected logical ``(RO, LIN, PAR)`` image dimensions.
        expected_map_count: Required ESPIRiT map count in BART dimension 4.

    Returns:
        Finite complex array shaped ``(RO, LIN, PAR, map)``.

    Raises:
        ValueError: If image dimensions, map count, or values are invalid.
    """
    if expected_map_count not in (1, 2):
        raise ValueError("Expected ESPIRiT map count must be 1 or 2.")
    image = np.asarray(open_cfl(image_path))
    padded_shape = image.shape + (1,) * max(0, 5 - image.ndim)
    if (
        padded_shape[:3] != logical_shape
        or padded_shape[3] != 1
        or padded_shape[4] != expected_map_count
        or any(size != 1 for size in padded_shape[5:])
        or not np.isfinite(image).all()
    ):
        raise ValueError(
            "BART image shape/finite check failed: "
            f"{image.shape}, expected {logical_shape + (1, expected_map_count)} "
            "with only trailing singleton dimensions."
        )
    normalized = image.reshape(
        logical_shape + (1, expected_map_count), order="F"
    )
    return normalized[:, :, :, 0, :]


def _export_map_image(
    *,
    native: Any,
    image: np.ndarray,
    twix_path: Path,
    output: Path,
    nifti_sub: str,
    suffix: str,
    voxel_size_logical: tuple[float, float, float],
    save_phase: bool,
    metadata: dict[str, Any],
) -> None:
    """Export one logical MPRAGE image through the validated native writer.

    Args:
        native: Loaded upstream MPRAGE helper module.
        image: Logical ``(RO, LIN, PAR)`` complex or real image.
        twix_path: Source TWIX path used to derive the physical affine.
        output: Destination directory for NIfTI files and JSON sidecars.
        nifti_sub: Dataset-independent output name component.
        suffix: Output filename suffix.
        voxel_size_logical: Logical ``(RO, LIN, PAR)`` voxel sizes in mm.
        save_phase: Whether to export a wrapped-phase NIfTI.
        metadata: JSON-compatible reconstruction provenance fields.

    Returns:
        None. The native writer creates NIfTI files and JSON sidecars.
    """
    native.save_mprage_output_to_nifti(
        image=image,
        twix_file=str(twix_path),
        out_folder=str(output),
        nifti_sub=nifti_sub,
        suffix=native._sanitize_filename_component(suffix),
        tag_wave="wave",
        file_tag="",
        voxel_size_mm=voxel_size_logical,
        crop_readout_os=1,
        save_phase=save_phase,
        twix_array_axis_roles=("phase", "readout", "slice"),
        twix_array_axis_flips=MPRAGE_BART_ARRAY_AXIS_FLIPS,
        metadata=metadata,
    )


def convert(
    bart_inputs: Path,
    image_base: Path,
    twix: Path,
    sequence: Path,
    output: Path,
    suffix: str,
    *,
    map_count: int = 1,
    ecalib_record: Path | None = None,
) -> None:
    """Restore BART normalization and write magnitude plus phase NIfTI files.

    Args:
        bart_inputs: Directory containing the prepared case manifest.
        image_base: BART complex image CFL basename.
        twix: Source TWIX path used to derive the physical affine.
        sequence: Matching sequence path retained for provenance validation.
        output: Destination directory for NIfTI files and JSON sidecars.
        suffix: Dataset-independent suffix used in output filenames.
        map_count: Expected ESPIRiT map count in the BART image.
        ecalib_record: Optional experiment-specific ecalib command record.

    Returns:
        None. The function writes per-map magnitude/phase pairs and, for the
        two-map experiment, one display-only map-RSS magnitude.
    """
    inputs = bart_inputs.expanduser().resolve()
    image_path = image_base.expanduser().resolve()
    twix_path = twix.expanduser().resolve()
    sequence_path = sequence.expanduser().resolve()
    manifest = _load_manifest(inputs / "manifest.json")
    logical_shape, physical_resolution_xyz = _case_geometry(manifest)
    map_images = _load_bart_map_images(image_path, logical_shape, map_count)
    kspace_norm = float(manifest["echoes"][0]["wave_kspace_norm"])
    if not np.isfinite(kspace_norm) or kspace_norm <= 0:
        raise ValueError("BART Wave k-space norm must be positive and finite.")
    restored = map_images.astype(np.complex64, copy=False) * kspace_norm

    native = load_wave_mprage_helpers()
    metadata = {
        "Reconstruction": "BART Wave MPRAGE",
        "ReconstructionSoftware": "BART wave",
        "BARTWaveKspaceNormRestored": kspace_norm,
        "BARTInternalNormalizationRestored": True,
        "BARTOutputAlreadyReadoutDeoversampled": True,
        "MPRAGEBARTOrientationCorrection": (
            "original AP direction retained; physical SI and LR directions "
            "reversed after real-data DICOM comparison"
        ),
        "PreparedInputManifest": str(inputs / "manifest.json"),
        **_recorded_commands(inputs, image_path, manifest, ecalib_record),
    }
    # The native exporter expects logical RO/LIN/PAR voxel sizes, whereas the
    # case manifest records physical X/Y/Z.
    voxel_size_logical = (
        physical_resolution_xyz[2],
        physical_resolution_xyz[1],
        physical_resolution_xyz[0],
    )
    resolved_output = output.expanduser().resolve()
    if map_count == 1:
        _export_map_image(
            native=native,
            image=restored[:, :, :, 0],
            twix_path=twix_path,
            output=resolved_output,
            nifti_sub=inputs.parent.name,
            suffix=suffix,
            voxel_size_logical=voxel_size_logical,
            save_phase=True,
            metadata=metadata,
        )
        return

    # Retain both quantitative complex components as separate magnitude/phase
    # exports. Their common RSS is diagnostic display data and has no phase.
    for map_index in range(map_count):
        component_metadata = {
            **metadata,
            "Experimental": True,
            "BARTESPIRiTMapCount": map_count,
            "BARTESPIRiTMapComponent": map_index + 1,
        }
        _export_map_image(
            native=native,
            image=restored[:, :, :, map_index],
            twix_path=twix_path,
            output=resolved_output,
            nifti_sub=inputs.parent.name,
            suffix=f"{suffix}Map{map_index + 1:02d}",
            voxel_size_logical=voxel_size_logical,
            save_phase=True,
            metadata=component_metadata,
        )
    rss_display = np.sqrt(np.sum(np.abs(restored) ** 2, axis=3)).astype(
        np.float32, copy=False
    )
    _export_map_image(
        native=native,
        image=rss_display,
        twix_path=twix_path,
        output=resolved_output,
        nifti_sub=inputs.parent.name,
        suffix=f"{suffix}MapsRSSDisplay",
        voxel_size_logical=voxel_size_logical,
        save_phase=False,
        metadata={
            **metadata,
            "Experimental": True,
            "DisplayOnly": True,
            "BARTESPIRiTMapCount": map_count,
            "BARTESPIRiTMapCombination": "sqrt(sum(abs(map_image)^2))",
            "CombinedPhaseAvailable": False,
        },
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
