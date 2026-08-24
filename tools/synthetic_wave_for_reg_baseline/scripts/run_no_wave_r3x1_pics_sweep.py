#!/usr/bin/env python3
"""Run a GPU BART PICS sweep on retrospectively sampled no-Wave R3x1 data.

The source is fully sampled, compressed no-Wave k-space. This script applies
the measured product R3x1 PE1 lattice plus its 24-line ACS support, runs one
unregularized CG-SENSE control and a compact Wavelet/FISTA sweep, and exports
canonical-RAS magnitude NIfTIs. Completed cases are safely reusable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from bart_cfl import (
    bart_base,
    open_bart_memmap,
    sha256_file,
    validate_finite_bart,
    write_bart_header,
)


GRID_SHAPE = (256, 256, 256)
NCC = 12
PE1_ACCELERATION = 3
PE1_RESIDUE = 1
FULLY_SAMPLED_PE1_LINES = tuple(range(115, 139))


def build_r3x1_pe1_mask(
    size: int = GRID_SHAPE[1],
    acceleration: int = PE1_ACCELERATION,
    residue: int = PE1_RESIDUE,
    fully_sampled_lines: Sequence[int] = FULLY_SAMPLED_PE1_LINES,
) -> np.ndarray:
    """Return the no-Wave R3x1 image-plus-ACS PE1 mask."""
    if size < 1 or acceleration < 1 or not 0 <= residue < acceleration:
        raise ValueError("Invalid R3x1 mask dimensions or residue.")
    mask = np.zeros(size, dtype=bool)
    mask[residue::acceleration] = True
    for line in fully_sampled_lines:
        if not 0 <= int(line) < size:
            raise ValueError(f"Fully sampled PE1 line is outside the grid: {line}")
        mask[int(line)] = True
    return mask


def lambda_label(value: float) -> str:
    """Return a stable scientific-notation label for one positive lambda."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Wavelet lambda must be positive and finite.")
    return f"{value:.8g}".replace("+", "")


def build_pics_command(
    bart: Path,
    kspace: Path,
    maps: Path,
    output: Path,
    *,
    iterations: int,
    regularizer: str,
    lambda_value: float | None = None,
) -> list[str]:
    """Build one mandatory-GPU BART PICS command."""
    if iterations < 1:
        raise ValueError("PICS iteration count must be positive.")
    command = [
        str(bart),
        "pics",
        "-g",
        "-S",
        "-e",
        "-i",
        str(iterations),
    ]
    if regularizer == "cg_sense":
        if lambda_value is not None:
            raise ValueError("CG-SENSE does not accept a Wavelet lambda.")
    elif regularizer == "wavelet":
        if lambda_value is None:
            raise ValueError("Wavelet reconstruction requires lambda.")
        command.extend(["-l1", "-n", "--fista", "-r", f"{lambda_value:.12g}"])
    else:
        raise ValueError(f"Unsupported PICS regularizer: {regularizer}")
    command.extend([str(bart_base(kspace)), str(bart_base(maps)), str(bart_base(output))])
    return command


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_logged(command: list[str], log_path: Path) -> dict[str, Any]:
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
        "log": str(log_path),
        "wall_seconds": time.perf_counter() - started,
    }


def _prepare_input(
    source_path: Path,
    output_dir: Path,
    *,
    pe2_chunk: int,
    resume: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / "input_manifest.json"
    output_base = output_dir / "no_wave_kspace_r3x1"
    if manifest_path.is_file() and resume:
        manifest = _load_json(manifest_path)
        if (
            manifest.get("status") == "complete"
            and Path(manifest.get("source_no_wave_kspace", "")).resolve() == source_path
            and output_base.with_suffix(".cfl").is_file()
            and sha256_file(output_base.with_suffix(".cfl"))
            == manifest.get("output_cfl_sha256")
        ):
            print(f"Reusing prepared no-Wave R3x1 input: {manifest_path}")
            return manifest
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"No-Wave input directory is not safely reusable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = np.load(source_path, mmap_mode="r")
    expected_shape = (*GRID_SHAPE, NCC)
    if source.shape != expected_shape or source.dtype != np.complex64:
        raise ValueError(f"Unexpected no-Wave source: {source.shape}, {source.dtype}")
    mask = build_r3x1_pe1_mask()
    lines = np.flatnonzero(mask)
    partial = Path(str(output_base.with_suffix(".cfl")) + ".partial")
    target = np.memmap(
        partial, mode="w+", dtype=np.complex64, shape=expected_shape, order="F"
    )
    target.fill(np.complex64(0.0))
    acquired_error_count = 0
    missing_nonzero_count = 0
    nonfinite_count = 0
    for start in range(0, GRID_SHAPE[2], pe2_chunk):
        stop = min(start + pe2_chunk, GRID_SHAPE[2])
        block = np.asarray(source[:, :, start:stop, :], dtype=np.complex64)
        target[:, lines, start:stop, :] = block[:, lines, :, :]
        written = np.asarray(target[:, :, start:stop, :])
        nonfinite_count += int(np.count_nonzero(~np.isfinite(written)))
        acquired_error_count += int(
            np.count_nonzero(written[:, lines, :, :] != block[:, lines, :, :])
        )
        missing_nonzero_count += int(np.count_nonzero(written[:, ~mask, :, :]))
        target.flush()
        print(f"Prepared no-Wave R3x1 PE2 {stop}/{GRID_SHAPE[2]}", flush=True)
    del target
    if nonfinite_count or acquired_error_count or missing_nonzero_count:
        raise RuntimeError("Prepared no-Wave R3x1 input failed sample validation.")
    os.replace(partial, output_base.with_suffix(".cfl"))
    write_bart_header(output_base, expected_shape)
    np.save(output_dir / "pe1_sampling_mask.npy", mask)
    payload = {
        "format_version": 1,
        "status": "complete",
        "purpose": "retrospective no-Wave R3x1 BART PICS input",
        "source_no_wave_kspace": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output_base": str(output_base),
        "output_cfl_sha256": sha256_file(output_base.with_suffix(".cfl")),
        "shape": list(expected_shape),
        "sampling": {
            "pe1_acceleration": PE1_ACCELERATION,
            "pe1_residue": PE1_RESIDUE,
            "pe2_acceleration": 1,
            "image_pe1_lines": list(range(PE1_RESIDUE, GRID_SHAPE[1], PE1_ACCELERATION)),
            "fully_sampled_pe1_lines": list(FULLY_SAMPLED_PE1_LINES),
            "acquired_pe1_lines": lines.tolist(),
            "sampling_mask": str(output_dir / "pe1_sampling_mask.npy"),
        },
        "validation": {
            "all_samples_finite": True,
            "acquired_samples_equal_source_bitwise": True,
            "missing_samples_are_exact_zero": True,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_path, payload)
    return payload


def _export_nifti(
    image_base: Path,
    *,
    output_dir: Path,
    twix: Path,
    sequence: Path,
    subject: str,
    regularizer: str,
    lambda_value: float | None,
    command: list[str],
) -> dict[str, Any]:
    recon_root = Path(__file__).resolve().parents[3] / "external" / "wave-mprage" / "recon"
    if str(recon_root) not in sys.path:
        sys.path.insert(0, str(recon_root))
    import recon_wave_mprage_from_twix_integrated_nifti as native
    from bart.bart_utils.bart_io import read_cfl

    image = read_cfl(image_base)
    if image.shape != GRID_SHAPE or not np.isfinite(image).all():
        raise ValueError(f"Invalid PICS image for NIfTI export: {image.shape}")
    seq = native.pp.Sequence()
    seq.read(str(sequence), remove_duplicates=False)
    defs = seq.definitions
    geom = native._derive_hardcoded_sag_logical_geometry(defs)
    native._assert_sag_geometry(defs)
    voxel_size_mm = native._derive_nifti_voxel_size_mm(defs, geom)
    label = "cg-sense" if regularizer == "cg_sense" else f"wavelet_lambda-{lambda_label(lambda_value)}"
    suffix = "BARTNoWaveCGSENSE" if regularizer == "cg_sense" else "BARTNoWaveWavelet"
    nifti_subject = f"{subject}-no-wave-r3x1-{label}"
    metadata = {
        "Description": "Retrospectively sampled no-Wave R3x1 reconstruction",
        "ReconstructionSoftware": "BART pics",
        "ReconstructionModel": "Cartesian SENSE without Wave PSF",
        "Regularizer": regularizer,
        "Lambda": lambda_value,
        "Optimizer": "CG" if regularizer == "cg_sense" else "FISTA",
        "Acceleration": {"PE1": 3, "PE2": 1},
        "FullySampledPE1Lines": list(FULLY_SAMPLED_PE1_LINES),
        "BARTCommand": command,
        "BARTDataScaleRestoredWithS": True,
    }
    nifti_dir = output_dir / "nifti"
    native.save_mprage_output_to_nifti(
        image=image,
        twix_file=str(twix),
        out_folder=nifti_dir,
        nifti_sub=nifti_subject,
        suffix=suffix,
        tag_wave="nowave",
        file_tag=label,
        voxel_size_mm=voxel_size_mm,
        crop_readout_os=1,
        save_phase=False,
        metadata=metadata,
    )
    outputs = sorted(nifti_dir.rglob("*part-mag*.nii.gz"))
    if len(outputs) != 1:
        raise ValueError(f"Expected one magnitude NIfTI, found {outputs}")
    saved = nib.load(str(outputs[0]))
    if saved.shape != GRID_SHAPE or nib.aff2axcodes(saved.affine) != ("R", "A", "S"):
        raise ValueError("No-Wave PICS NIfTI geometry validation failed.")
    return {
        "magnitude_nifti": str(outputs[0]),
        "magnitude_nifti_sha256": sha256_file(outputs[0]),
        "shape": list(saved.shape),
        "axis_codes": list(nib.aff2axcodes(saved.affine)),
    }


def _run_case(
    *,
    bart: Path,
    kspace_base: Path,
    input_manifest: Path,
    maps_base: Path,
    output_dir: Path,
    twix: Path,
    sequence: Path,
    subject: str,
    iterations: int,
    regularizer: str,
    lambda_value: float | None,
    resume: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file() and resume:
        manifest = _load_json(manifest_path)
        nifti_path = Path(manifest.get("nifti", {}).get("magnitude_nifti", ""))
        if (
            manifest.get("status") == "complete"
            and nifti_path.is_file()
            and sha256_file(nifti_path)
            == manifest.get("nifti", {}).get("magnitude_nifti_sha256")
        ):
            print(f"Reusing complete no-Wave case: {manifest_path}")
            return manifest
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"No-Wave case is not safely reusable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_base = output_dir / "image"
    command = build_pics_command(
        bart,
        kspace_base,
        maps_base,
        image_base,
        iterations=iterations,
        regularizer=regularizer,
        lambda_value=lambda_value,
    )
    execution = _run_logged(command, output_dir / "bart_pics.log")
    output_validation = validate_finite_bart(image_base, (*GRID_SHAPE, 1, 1))
    nifti = _export_nifti(
        image_base,
        output_dir=output_dir,
        twix=twix,
        sequence=sequence,
        subject=subject,
        regularizer=regularizer,
        lambda_value=lambda_value,
        command=command,
    )
    payload = {
        "format_version": 1,
        "status": "complete",
        "purpose": "no-Wave R3x1 presentation reconstruction",
        "config": {
            "regularizer": regularizer,
            "lambda": lambda_value,
            "optimizer": "CG" if regularizer == "cg_sense" else "FISTA",
            "iterations": iterations,
            "gpu_required": True,
            "bart_data_scale_restored": True,
        },
        "input_manifest": {
            "path": str(input_manifest),
            "sha256": sha256_file(input_manifest),
        },
        "maps": {
            "base": str(maps_base),
            "cfl_sha256": sha256_file(maps_base.with_suffix(".cfl")),
        },
        "execution": execution,
        "bart_output": {
            **output_validation,
            "base": str(image_base),
            "cfl_sha256": sha256_file(image_base.with_suffix(".cfl")),
        },
        "nifti": nifti,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_path, payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    bart = args.bart.expanduser().resolve()
    source = args.source_no_wave_kspace.expanduser().resolve()
    maps = bart_base(args.maps.expanduser().resolve())
    twix = args.twix.expanduser().resolve()
    sequence = args.sequence.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for path in (
        bart,
        source,
        maps.with_suffix(".hdr"),
        maps.with_suffix(".cfl"),
        twix,
        sequence,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.pe2_chunk < 1 or args.iterations < 1:
        raise ValueError("Chunk and iteration counts must be positive.")
    lambdas = [float(value) for value in args.wavelet_lambdas]
    if len(set(lambdas)) != len(lambdas):
        raise ValueError("Wavelet lambda list contains duplicates.")
    for value in lambdas:
        lambda_label(value)

    source_array = np.load(source, mmap_mode="r")
    if source_array.shape != (*GRID_SHAPE, NCC) or source_array.dtype != np.complex64:
        raise ValueError(
            f"Unexpected no-Wave source: {source_array.shape}, {source_array.dtype}"
        )
    maps_array = open_bart_memmap(maps)
    padded_maps_shape = maps_array.shape + (1,) * max(0, 5 - maps_array.ndim)
    if padded_maps_shape[:5] != (*GRID_SHAPE, NCC, 1) or any(
        value != 1 for value in padded_maps_shape[5:]
    ):
        raise ValueError(f"Unexpected one-set sensitivity maps: {maps_array.shape}")
    if args.validate_only:
        cg_command = build_pics_command(
            bart,
            output_root / "bart_inputs" / "no_wave_kspace_r3x1",
            maps,
            output_root / "cg_sense_lambda-0" / "image",
            iterations=args.iterations,
            regularizer="cg_sense",
        )
        wavelet_command = build_pics_command(
            bart,
            output_root / "bart_inputs" / "no_wave_kspace_r3x1",
            maps,
            output_root / f"wavelet_lambda-{lambda_label(lambdas[0])}" / "image",
            iterations=args.iterations,
            regularizer="wavelet",
            lambda_value=lambdas[0],
        )
        print("No-Wave R3x1 sweep structural validation: PASSED")
        print("CG command:", " ".join(cg_command))
        print("Wavelet command example:", " ".join(wavelet_command))
        return {"status": "validated", "case_count": 1 + len(lambdas)}

    version = subprocess.run(
        [str(bart), "version"], check=True, capture_output=True, text=True
    )
    input_dir = output_root / "bart_inputs"
    input_payload = _prepare_input(
        source, input_dir, pe2_chunk=args.pe2_chunk, resume=args.resume
    )
    input_manifest = input_dir / "input_manifest.json"
    kspace_base = Path(input_payload["output_base"])
    records = [
        _run_case(
            bart=bart,
            kspace_base=kspace_base,
            input_manifest=input_manifest,
            maps_base=maps,
            output_dir=output_root / "cg_sense_lambda-0",
            twix=twix,
            sequence=sequence,
            subject=args.subject,
            iterations=args.iterations,
            regularizer="cg_sense",
            lambda_value=None,
            resume=args.resume,
        )
    ]
    for value in lambdas:
        records.append(
            _run_case(
                bart=bart,
                kspace_base=kspace_base,
                input_manifest=input_manifest,
                maps_base=maps,
                output_dir=output_root / f"wavelet_lambda-{lambda_label(value)}",
                twix=twix,
                sequence=sequence,
                subject=args.subject,
                iterations=args.iterations,
                regularizer="wavelet",
                lambda_value=value,
                resume=args.resume,
            )
        )
    payload = {
        "format_version": 1,
        "status": "complete",
        "purpose": "no-Wave R3x1 CG-SENSE control and Wavelet sweep",
        "bart_executable": str(bart),
        "bart_version_output": (version.stdout + version.stderr).strip(),
        "input_manifest": str(input_manifest),
        "maps_base": str(maps),
        "wavelet_lambdas": lambdas,
        "cases": [str(Path(record["nifti"]["magnitude_nifti"])) for record in records],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(output_root / "sweep_manifest.json", payload)
    print(f"No-Wave R3x1 sweep manifest: {output_root / 'sweep_manifest.json'}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bart", required=True, type=Path)
    parser.add_argument("--source-no-wave-kspace", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--pe2-chunk", type=int, default=4)
    parser.add_argument(
        "--wavelet-lambdas",
        nargs="+",
        type=float,
        default=(1e-4, 1e-3, 1e-2, 1.5e-2, 2e-2, 5e-2),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and commands without creating outputs or running BART.",
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
