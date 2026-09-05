#!/usr/bin/env python3
"""Restore and export one multi-echo BART Wave-GRE reconstruction branch."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import open_cfl  # noqa: E402
from wave_retro_lr.gre import (  # noqa: E402
    EXTENDED_READOUT,
    GRE_BART_ARRAY_AXIS_FLIPS,
    GRE_BART_OUTPUT_CONVENTION_VERSION,
    GRE_LOGICAL_AXIS_ROLES,
    bart_wave_restoration_factor,
    load_wave_gre_helpers,
    resolve_gre_wavelet_lambda,
    restore_bart_wave_image,
    validate_gre_echo_consistency,
    validate_gre_sequence,
)


def _parser() -> argparse.ArgumentParser:
    """Build the multi-echo GRE conversion CLI.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bart-inputs", required=True, type=Path)
    parser.add_argument(
        "--image",
        required=True,
        action="append",
        type=Path,
        help="Ordered BART image basename; repeat once per echo.",
    )
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--seq", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suffix", default="BARTWaveGRE")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object.

    Args:
        path: JSON file to read.

    Returns:
        Parsed mapping.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON conversion manifest.

    Args:
        path: Destination JSON path.
        payload: JSON-compatible mapping.

    Returns:
        None.
    """

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _case_geometry(manifest: Mapping[str, Any]) -> tuple[str, tuple[int, int, int], tuple[float, float, float]]:
    """Read a normal or retrospective GRE geometry from its manifest.

    Args:
        manifest: Prepared BART-input manifest.

    Returns:
        Case identifier, logical matrix, and logical voxel sizes in millimeters.
    """

    case = manifest.get("case", manifest.get("geometry"))
    if not isinstance(case, Mapping):
        raise ValueError("Prepared GRE manifest has no case geometry.")
    return (
        str(case["case_id"]),
        tuple(int(value) for value in case["matrix_ro_lin_par"]),
        tuple(float(value) for value in case["voxel_mm_ro_lin_par"]),
    )


def _command_records(inputs: Path, images: Sequence[Path], manifest: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Read the exact ecalib and echo-specific Wave command records.

    Args:
        inputs: Prepared case input directory.
        images: Ordered reconstructed image basenames.
        manifest: Prepared case manifest.

    Returns:
        One ecalib command and one ordered Wave command per echo.
    """

    root = inputs.parents[2] if "case" in manifest else inputs.parents[1]
    ecalib_path = root / "normal" / "bart_output" / "ecalib_command.txt"
    wave_paths = [image.parent / "wave_command.txt" for image in images]
    for path in [ecalib_path, *wave_paths]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing BART command record: {path}")
    return (
        ecalib_path.read_text(encoding="utf-8").strip(),
        [path.read_text(encoding="utf-8").strip() for path in wave_paths],
    )


def _canonicalize_saved_nifti(nifti_path: Path, json_path: Path) -> dict[str, Any]:
    """Reorient a saved GRE NIfTI to canonical RAS without interpolation.

    Args:
        nifti_path: Magnitude or wrapped-phase NIfTI produced with the TWIX affine.
        json_path: Matching JSON sidecar to update with the stored orientation.

    Returns:
        Original/final axis codes and the no-interpolation reorientation method.
    """

    import nibabel as nib

    image = nib.load(str(nifti_path))
    original = tuple(str(value) for value in nib.aff2axcodes(image.affine))
    canonical = nib.as_closest_canonical(image, enforce_diag=False)
    final = tuple(str(value) for value in nib.aff2axcodes(canonical.affine))
    if final != ("R", "A", "S"):
        raise ValueError(f"GRE NIfTI did not resolve to canonical RAS: {final}.")
    nib.save(canonical, str(nifti_path))
    sidecar = _load_json(json_path)
    sidecar["CanonicalRASReorientation"] = {
        "OriginalAxisCodes": list(original),
        "StoredAxisCodes": list(final),
        "Method": "nibabel orientation permutation and axis flips",
        "Interpolation": False,
    }
    _write_json(json_path, sidecar)
    return {
        "original_axis_codes": list(original),
        "stored_axis_codes": list(final),
        "method": "axis permutation and flips",
        "interpolation": False,
    }


def convert(
    bart_inputs: Path,
    images: Sequence[Path],
    twix: Path,
    sequence: Path,
    output: Path,
    suffix: str,
) -> dict[str, Any]:
    """Restore complex scale and export display magnitude and wrapped phase.

    Args:
        bart_inputs: Prepared normal or retrospective BART-input directory.
        images: Ordered BART complex image basenames, one per echo.
        twix: Source measured GRE TWIX path used for the affine.
        sequence: Matching Pulseq sequence path.
        output: Branch-specific output directory.
        suffix: Dataset-independent NIfTI suffix.

    Returns:
        Conversion manifest. Quantitative complex NumPy data remain separate
        from display-normalized magnitude NIfTI files.
    """

    inputs = bart_inputs.expanduser().resolve()
    image_paths = [image.expanduser().resolve() for image in images]
    twix_path = twix.expanduser().resolve()
    sequence_path = sequence.expanduser().resolve()
    destination = output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(inputs / "manifest.json")
    case_id, matrix, voxel_mm = _case_geometry(manifest)
    echoes = manifest.get("echoes")
    if not isinstance(echoes, list) or not echoes:
        raise ValueError("Prepared GRE manifest must contain one or more echoes.")
    echo_count = len(echoes)
    if [item.get("echo") for item in echoes] != list(range(1, echo_count + 1)):
        raise ValueError("Prepared GRE echoes are not consecutive and ordered.")
    if len(image_paths) != echo_count:
        raise ValueError(
            f"Prepared GRE has {echo_count} echoes but {len(image_paths)} images were supplied."
        )
    manifest_echo_times = [float(item.get("te_s", math.nan)) for item in echoes]
    native = load_wave_gre_helpers()
    _, cfg, _, _ = validate_gre_sequence(native, sequence_path)
    echo_times = validate_gre_echo_consistency(cfg["TE_s"], manifest_echo_times)
    calibration_ids = {item.get("shared_calibration_id") for item in echoes}
    if len(calibration_ids) != 1 or None in calibration_ids:
        raise ValueError("Prepared GRE echoes do not share one calibration identity.")
    selection = manifest.get("wavelet_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Prepared GRE manifest has no shared-echo Wavelet provenance.")
    selected_lambda = resolve_gre_wavelet_lambda(
        case_id,
        method=str(selection.get("method", "")),
        echo_ids=selection.get("echo_ids", ()),
        shared_lambda=selection.get("shared_lambda"),
        wavelet_lambda_by_echo={
            f"echo-{int(item['echo']):02d}": float(item["selected_wavelet_lambda"])
            for item in echoes
        },
        selection_manifest_basename=str(selection.get("selection_manifest_basename", "")),
        selection_manifest_sha256=str(selection.get("selection_manifest_sha256", "")),
    )

    restored = []
    restoration_records = []
    encoding_shape = (EXTENDED_READOUT, matrix[1], matrix[2])
    quantitative = destination / "quantitative_complex"
    quantitative.mkdir(parents=True, exist_ok=True)
    for echo_index, (image_base, echo) in enumerate(
        zip(image_paths, echoes, strict=True)
    ):
        image = np.asarray(open_cfl(image_base)).squeeze()
        if image.shape != matrix or not np.isfinite(image).all():
            raise ValueError(f"Echo {echo_index + 1} BART image shape/finite check failed: {image.shape}.")
        norm = float(echo["wave_kspace_norm"])
        corrected = restore_bart_wave_image(image, kspace_norm=norm, encoding_shape=encoding_shape)
        complex_path = quantitative / f"echo-{echo_index + 1:02d}_complex.npy"
        np.save(complex_path, corrected, allow_pickle=False)
        restored.append(corrected)
        factor = bart_wave_restoration_factor(matrix, norm, encoding_shape)
        restoration_records.append(
            {
                "echo": echo_index + 1,
                "te_s": float(echo["te_s"]),
                "kspace_norm": norm,
                "encoding_shape_extended_ro_lin_par": list(encoding_shape),
                "amplitude": float(abs(factor)),
                "phase_factor_real": float(np.real(factor / abs(factor))),
                "phase_factor_imag": float(np.imag(factor / abs(factor))),
                "formula": "amplitude=kspace_norm*sqrt(extended_RO*LIN*PAR); phase=1j*(-1)**(LIN//2)",
                "quantitative_complex_npy": str(complex_path),
            }
        )

    positive = np.abs(restored[0])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        raise ValueError("Restored echo-1 magnitude has no positive finite voxels.")
    shared_display_scale = float(np.percentile(positive, 99.0))
    ecalib_command, wave_commands = _command_records(inputs, image_paths, manifest)

    cfg = dict(cfg)
    cfg.update(
        {
            "Nx": matrix[0],
            "Ny": matrix[1],
            "Nz": matrix[2],
            "res_xyz_m": tuple(value / 1000.0 for value in voxel_mm),
        }
    )
    nifti_records = []
    for echo_index, corrected in enumerate(restored):
        metadata = {
            "Reconstruction": "BART Wave measured multi-echo GRE",
            "EchoNumber": echo_index + 1,
            "EchoTime": float(echo_times[echo_index]),
            "EchoTimeUnits": "s",
            "PreparedInputManifest": str(inputs / "manifest.json"),
            "CaseID": case_id,
            "BARTEcalibCommand": ecalib_command,
            "BARTWaveCommand": wave_commands[echo_index],
            "BARTOutputConventionVersion": GRE_BART_OUTPUT_CONVENTION_VERSION,
            "BARTOutputNormalization": restoration_records[echo_index],
            "GRESharedWaveletSelection": dict(selection),
            "GRESelectedWaveletLambda": selected_lambda,
            "QuantitativeComplexData": restoration_records[echo_index]["quantitative_complex_npy"],
            "MagnitudeNIfTIIsDisplayNormalized": True,
            "WrappedPhaseUnits": "radian",
            "OrientationPolicy": {
                "logical_axis_roles": list(GRE_LOGICAL_AXIS_ROLES),
                "array_axis_flips": list(GRE_BART_ARRAY_AXIS_FLIPS),
                "canonical_coordinate_system": "RAS",
                "interpolation": False,
            },
            "PresentationMaskApplied": False,
        }
        saved = native.save_gre_echo_to_nifti(
            image=corrected,
            twix_file=twix_path,
            out_folder=destination,
            nifti_sub=case_id,
            suffix=suffix,
            mode="wave",
            echo_idx=echo_index,
            cfg=cfg,
            save_phase=True,
            twix_array_axis_roles=GRE_LOGICAL_AXIS_ROLES,
            twix_array_axis_flips=GRE_BART_ARRAY_AXIS_FLIPS,
            twix_coord_system="LPS",
            twix_inplane_rot_sign=-1.0,
            twix_use_fov_for_voxel_size=False,
            metadata=metadata,
            voxel_size_mm=voxel_mm,
            magnitude_normalization_scale=shared_display_scale,
            crop_readout_os=1,
        )
        canonical_outputs = []
        for nifti_path, json_path in saved:
            canonical_outputs.append(
                {
                    "nifti": str(nifti_path),
                    "json": str(json_path),
                    "canonicalization": _canonicalize_saved_nifti(nifti_path, json_path),
                }
            )
        nifti_records.append(
            {
                "echo": echo_index + 1,
                "outputs": canonical_outputs,
            }
        )

    result = {
        "format_version": 1,
        "status": "multi_echo_gre_nifti_export_complete",
        "echo_count": echo_count,
        "echo_times_s": list(echo_times),
        "case_id": case_id,
        "prepared_input_manifest": str(inputs / "manifest.json"),
        "bart_commands": {"ecalib": ecalib_command, "wave_by_echo": wave_commands},
        "bart_output_restoration": restoration_records,
        "wavelet_selection": dict(selection),
        "display_magnitude": {
            "normalization": "echo-1 positive-finite 99th percentile shared across echoes",
            "scale": shared_display_scale,
            "clipped": False,
        },
        "orientation": {
            "logical_axis_roles": list(GRE_LOGICAL_AXIS_ROLES),
            "array_axis_flips": list(GRE_BART_ARRAY_AXIS_FLIPS),
            "canonical_coordinate_system": "RAS",
            "interpolation": False,
        },
        "quantitative_complex_separate_from_display_nifti": True,
        "presentation_mask_applied": False,
        "nifti": nifti_records,
    }
    _write_json(destination / "conversion_manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Convert the parsed multi-echo BART output branch.

    Args:
        argv: Optional argument vector.

    Returns:
        Zero after successful conversion.
    """

    args = _parser().parse_args(argv)
    convert(
        args.bart_inputs,
        args.image,
        args.twix,
        args.seq,
        args.output,
        args.suffix,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
