#!/usr/bin/env python3
"""Run one resumable BART Wave regularization case via the pinned wrapper."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bart_cfl import bart_base, open_bart_memmap, sha256_file, validate_finite_bart


def _build_parser() -> argparse.ArgumentParser:
    """Build the dataset-independent regularized reconstruction interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", required=True, type=Path)
    parser.add_argument("--bart", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--bart-input-dir", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--expected-maps-sha256", required=True)
    parser.add_argument("--lambda-zero-base", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--twix", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--regularizer", required=True, choices=("wavelet", "llr"))
    parser.add_argument("--lambda-value", required=True, type=float)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-eigenvalue", type=float, default=6.70e7)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--subject", default="20260817product")
    parser.add_argument("--resume", action="store_true")
    return parser


def canonical_lambda(value: float) -> str:
    """Return a stable compact scientific label for paths and manifests."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Regularization lambda must be positive and finite.")
    mantissa, exponent = f"{value:.8e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def validate_settings(
    regularizer: str,
    lambda_value: float,
    block_size: int,
    iterations: int,
    tolerance: float,
    max_eigenvalue: float,
) -> None:
    """Reject ambiguous or unsupported optimizer/regularizer settings."""
    canonical_lambda(lambda_value)
    if regularizer not in {"wavelet", "llr"}:
        raise ValueError(f"Unsupported regularizer: {regularizer}.")
    if regularizer == "llr" and block_size < 1:
        raise ValueError("LLR block size must be positive.")
    if iterations < 1:
        raise ValueError("Iteration count must be positive.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Tolerance must be positive and finite.")
    if not np.isfinite(max_eigenvalue) or max_eigenvalue <= 0.0:
        raise ValueError("Maximum eigenvalue must be positive and finite.")


def run_name(regularizer: str, lambda_value: float, block_size: int = 8) -> str:
    """Build the deterministic output-directory name for one parameter case."""
    label = canonical_lambda(lambda_value)
    if regularizer == "wavelet":
        return f"wavelet_lambda-{label}"
    if regularizer == "llr":
        return f"llr_block-{block_size}_lambda-{label}"
    raise ValueError(f"Unsupported regularizer: {regularizer}.")


def build_wave_options(
    regularizer: str,
    lambda_value: float,
    *,
    block_size: int,
    iterations: int,
    tolerance: float,
    max_eigenvalue: float,
    backend: str,
) -> list[str]:
    """Build the option section passed unchanged to upstream ``bart wave``."""
    validate_settings(
        regularizer, lambda_value, block_size, iterations, tolerance, max_eigenvalue
    )
    options = ["-w"] if regularizer == "wavelet" else ["-l", "-b", str(block_size)]
    options.extend(
        [
            "-f",
            "-i",
            str(iterations),
            "-t",
            f"{tolerance:.12g}",
            "-e",
            f"{max_eigenvalue:.12g}",
            "-r",
            f"{lambda_value:.12g}",
        ]
    )
    if backend == "gpu":
        options.append("-g")
    elif backend != "cpu":
        raise ValueError(f"Unsupported backend: {backend}.")
    return options


def build_wrapper_command(
    wrapper: Path,
    *,
    bart_input_dir: Path,
    bart_output_dir: Path,
    maps: Path,
    twix: Path,
    sequence: Path,
    nifti_output_dir: Path,
    nifti_subject: str,
    nifti_suffix: str,
    wave_options: Sequence[str],
) -> list[str]:
    """Build one direct call to the pinned Wave-MPRAGE reconstruction wrapper."""
    return [
        "bash",
        str(wrapper),
        "--bart-input",
        str(bart_input_dir),
        "--bart-output",
        str(bart_output_dir),
        "--maps-source",
        "existing",
        "--existing-maps",
        str(maps),
        "--twix",
        str(twix),
        "--seq",
        str(sequence),
        "--nifti-output",
        str(nifti_output_dir),
        "--save-phase",
        "--wave-options",
        *wave_options,
        "--end-wave-options",
        "--nifti-options",
        "--nifti-sub",
        nifti_subject,
        "--nifti-suffix",
        nifti_suffix,
        "--end-nifti-options",
    ]


def build_conversion_command(
    wrapper: Path,
    *,
    python: Path,
    bart_input_dir: Path,
    bart_output_dir: Path,
    twix: Path,
    sequence: Path,
    nifti_output_dir: Path,
    nifti_subject: str,
    nifti_suffix: str,
) -> list[str]:
    """Build the upstream conversion command used to recover a finished BART run."""
    return [
        str(python),
        str(wrapper.parent / "wave_to_nifti.py"),
        "--bart-input-dir",
        str(bart_input_dir),
        "--bart-output-dir",
        str(bart_output_dir),
        "--twix",
        str(twix),
        "--seq",
        str(sequence),
        "--out",
        str(nifti_output_dir),
        "--save-phase",
        "--nifti-sub",
        nifti_subject,
        "--nifti-suffix",
        nifti_suffix,
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a run manifest to avoid valid-looking partial JSON."""
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_streamed(command: Sequence[str], log_path: Path, env: dict[str, str]) -> float:
    """Stream the upstream wrapper while retaining its complete combined log."""
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    lines: list[str] = []
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    returncode = process.wait()
    log_path.write_text("".join(lines), encoding="utf-8")
    elapsed = time.perf_counter() - started
    if returncode:
        raise RuntimeError(f"Wave-MPRAGE wrapper failed with status {returncode}.")
    return elapsed


def _parse_bart_log(path: Path) -> dict[str, float]:
    """Extract BART eigenvalue and reconstruction timings from the wrapper log."""
    text = path.read_text(encoding="utf-8")
    fields = {}
    for name, pattern in (
        ("maximum_eigenvalue", r"Max eval:\s*([0-9.eE+-]+)"),
        ("internal_reconstruction_seconds", r"Reconstruction time:\s*([0-9.eE+-]+) seconds"),
        ("bart_total_seconds", r"Total time:\s*([0-9.eE+-]+) seconds"),
    ):
        match = re.search(pattern, text)
        if match:
            fields[name] = float(match.group(1))
    return fields


def _relative_bart_difference(candidate_base: Path, reference_base: Path) -> float:
    """Measure chunked relative L2 difference from the accepted lambda-zero image."""
    candidate = open_bart_memmap(candidate_base)
    reference = open_bart_memmap(reference_base)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Candidate/reference BART shapes differ: {candidate.shape} vs {reference.shape}."
        )
    error_squared = 0.0
    reference_squared = 0.0
    for start in range(0, candidate.shape[2], 8):
        stop = min(start + 8, candidate.shape[2])
        current = np.asarray(candidate[:, :, start:stop, ...])
        baseline = np.asarray(reference[:, :, start:stop, ...])
        difference = current - baseline
        error_squared += float(np.vdot(difference, difference).real)
        reference_squared += float(np.vdot(baseline, baseline).real)
    if reference_squared <= 0.0:
        raise ValueError("Lambda-zero reference has zero norm.")
    return float(np.sqrt(error_squared / reference_squared))


def _validate_niftis(nifti_dir: Path) -> list[dict[str, Any]]:
    """Validate the wrapper's matched magnitude/phase NIfTIs and sidecars."""
    import nibabel as nib

    paths = sorted(nifti_dir.rglob("*.nii.gz"))
    if len(paths) != 2 or not any("part-mag" in path.name for path in paths) or not any(
        "part-phase" in path.name for path in paths
    ):
        raise ValueError(f"Expected one magnitude and one phase NIfTI, found {paths}.")
    outputs = []
    for path in paths:
        image = nib.load(str(path))
        if image.shape != (256, 256, 256):
            raise ValueError(f"Unexpected NIfTI shape {image.shape}: {path}")
        data = np.asanyarray(image.dataobj)
        if not np.isfinite(data).all():
            raise ValueError(f"NIfTI contains non-finite values: {path}")
        sidecar = Path(str(path).removesuffix(".nii.gz") + ".json")
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        outputs.append(
            {
                "part": "phase" if "part-phase" in path.name else "mag",
                "nifti": str(path),
                "nifti_sha256": sha256_file(path),
                "json": str(sidecar),
                "json_sha256": sha256_file(sidecar),
                "shape": list(image.shape),
                "orientation": list(__import__("nibabel").aff2axcodes(image.affine)),
                "all_samples_finite": True,
            }
        )
    return outputs


def completed_manifest_reusable(
    manifest_path: Path, expected_config: dict[str, Any], expected_maps_hash: str
) -> bool:
    """Return true only for a complete, matching run with intact hashed outputs."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            return False
        if manifest.get("config") != expected_config:
            return False
        if manifest.get("maps", {}).get("cfl_sha256") != expected_maps_hash:
            return False
        records = [manifest["bart_output"], *manifest["nifti_outputs"]]
        for record in records:
            path = Path(record.get("path", record.get("nifti", "")))
            expected_hash = record.get("sha256", record.get("nifti_sha256"))
            if not path.is_file() or sha256_file(path) != expected_hash:
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def failed_run_recoverable(
    manifest_path: Path, expected_config: dict[str, Any], expected_maps_hash: str
) -> bool:
    """Recognize a finished BART solve whose wrapper failed only during conversion."""
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"failed", "recovering_conversion"}:
            return False
        if manifest.get("config") != expected_config:
            return False
        if manifest.get("maps", {}).get("cfl_sha256") != expected_maps_hash:
            return False
        run_dir = manifest_path.parent
        log_path = run_dir / "wrapper.log"
        image_base = run_dir / "bart" / "image_wave"
        fields = _parse_bart_log(log_path)
        if "bart_total_seconds" not in fields:
            return False
        validation = validate_finite_bart(image_base, (256, 256, 256, 1, 1))
        return float(validation["norm"]) > 0.0
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _resolved(path: Path) -> Path:
    """Expand and resolve a user-provided path without requiring existence."""
    return path.expanduser().resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate provenance, invoke the wrapper once, and persist a complete manifest."""
    wrapper = _resolved(args.wrapper)
    bart = _resolved(args.bart)
    python = _resolved(args.python)
    bart_input = _resolved(args.bart_input_dir)
    maps = bart_base(_resolved(args.maps))
    lambda_zero = bart_base(_resolved(args.lambda_zero_base))
    output_root = _resolved(args.output_root)
    twix = _resolved(args.twix)
    sequence = _resolved(args.sequence)
    validate_settings(
        args.regularizer,
        args.lambda_value,
        args.block_size,
        args.iterations,
        args.tolerance,
        args.max_eigenvalue,
    )
    for path in (wrapper, bart, python, twix, sequence, bart_input / "manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for base in (maps, lambda_zero, bart_input / "psf", bart_input / "wave_kspace"):
        for suffix in (".hdr", ".cfl"):
            if not base.with_suffix(suffix).is_file():
                raise FileNotFoundError(base.with_suffix(suffix))
    expected_maps_hash = args.expected_maps_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_maps_hash):
        raise ValueError("--expected-maps-sha256 must contain 64 lowercase hex characters.")
    actual_maps_hash = sha256_file(maps.with_suffix(".cfl"))
    if actual_maps_hash != expected_maps_hash:
        raise ValueError(
            f"ESPIRiT map hash mismatch: expected {expected_maps_hash}, got {actual_maps_hash}."
        )

    wave_options = build_wave_options(
        args.regularizer,
        args.lambda_value,
        block_size=args.block_size,
        iterations=args.iterations,
        tolerance=args.tolerance,
        max_eigenvalue=args.max_eigenvalue,
        backend=args.backend,
    )
    name = run_name(args.regularizer, args.lambda_value, args.block_size)
    run_dir = output_root / name
    bart_output = run_dir / "bart"
    nifti_output = run_dir / "nifti"
    manifest_path = run_dir / "manifest.json"
    config = {
        "regularizer": args.regularizer,
        "lambda": args.lambda_value,
        "lambda_label": canonical_lambda(args.lambda_value),
        "block_size": args.block_size if args.regularizer == "llr" else None,
        "optimizer": "FISTA",
        "iterations": args.iterations,
        "tolerance": args.tolerance,
        "maximum_eigenvalue": args.max_eigenvalue,
        "backend": args.backend,
    }
    if args.resume and completed_manifest_reusable(manifest_path, config, expected_maps_hash):
        print(f"Reusing validated completed run: {run_dir}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    recover_conversion = args.resume and failed_run_recoverable(
        manifest_path, config, expected_maps_hash
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not recover_conversion:
        raise FileExistsError(
            f"Output exists but is not a reusable completed run: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    nifti_subject = f"{args.subject}-{name}"
    nifti_suffix = "BARTWaveRegularized"
    wrapper_command = build_wrapper_command(
        wrapper,
        bart_input_dir=bart_input,
        bart_output_dir=bart_output,
        maps=maps,
        twix=twix,
        sequence=sequence,
        nifti_output_dir=nifti_output,
        nifti_subject=nifti_subject,
        nifti_suffix=nifti_suffix,
        wave_options=wave_options,
    )
    effective_bart_command = [
        str(bart),
        "wave",
        *wave_options,
        str(maps),
        str(bart_input / "psf"),
        str(bart_input / "wave_kspace"),
        str(bart_output / "image_wave"),
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any]
    recovered_failure: dict[str, Any] | None = None
    if recover_conversion:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "error" in manifest:
            recovered_failure = {
                "error": manifest.pop("error"),
                "failed_at_utc": manifest.pop("failed_at_utc", None),
            }
        manifest.update(
            {
                "status": "recovering_conversion",
                "conversion_recovery_started_at_utc": started_at,
                "python_executable": str(python),
            }
        )
    else:
        manifest = {
            "format_version": 1,
            "status": "running",
            "config": config,
            "started_at_utc": started_at,
            "wrapper": str(wrapper),
            "wrapper_command": wrapper_command,
            "effective_bart_command": effective_bart_command,
            "python_executable": str(python),
            "maps": {"base": str(maps), "cfl_sha256": actual_maps_hash},
            "lambda_zero_reference": str(lambda_zero),
            "bart_input_dir": str(bart_input),
        }
    _write_json(manifest_path, manifest)
    log_path = run_dir / "wrapper.log"
    environment = {
        **os.environ,
        "BART_BIN": str(bart),
        "PYTHON_BIN": str(python),
    }
    try:
        if recover_conversion:
            conversion_command = build_conversion_command(
                wrapper,
                python=python,
                bart_input_dir=bart_input,
                bart_output_dir=bart_output,
                twix=twix,
                sequence=sequence,
                nifti_output_dir=nifti_output,
                nifti_subject=nifti_subject,
                nifti_suffix=nifti_suffix,
            )
            conversion_log = run_dir / "conversion_recovery.log"
            conversion_wall_seconds = _run_streamed(
                conversion_command, conversion_log, environment
            )
            manifest["conversion_recovery"] = {
                "command": conversion_command,
                "wall_seconds": conversion_wall_seconds,
                "log": str(conversion_log),
                "reason": "upstream wrapper BART solve completed before conversion failure",
            }
            wrapper_wall_seconds = None
        else:
            wrapper_wall_seconds = _run_streamed(wrapper_command, log_path, environment)
        image_base = bart_output / "image_wave"
        validation = validate_finite_bart(image_base, (256, 256, 256, 1, 1))
        if float(validation["norm"]) <= 0.0:
            raise ValueError("Regularized BART output has zero norm.")
        difference = _relative_bart_difference(image_base, lambda_zero)
        if difference <= 1e-7:
            raise ValueError("Regularized output is indistinguishable from lambda-zero.")
        nifti_outputs = _validate_niftis(nifti_output)
        version = subprocess.run(
            [str(bart), "version"], check=True, capture_output=True, text=True
        )
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wrapper_wall_seconds": wrapper_wall_seconds,
                "wrapper_log": str(log_path),
                "bart_version_output": (version.stdout + version.stderr).strip(),
                "bart_log_fields": _parse_bart_log(log_path),
                "bart_output": {
                    **validation,
                    "base": str(image_base),
                    "path": str(image_base.with_suffix(".cfl")),
                    "sha256": sha256_file(image_base.with_suffix(".cfl")),
                    "relative_l2_difference_from_lambda_zero": difference,
                },
                "nifti_outputs": nifti_outputs,
            }
        )
        if recovered_failure is not None:
            manifest["recovered_failure"] = recovered_failure
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(manifest_path, manifest)
        raise
    _write_json(manifest_path, manifest)
    print(f"Completed run manifest: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run one regularization case and map expected failures to status 2."""
    run(_build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
