#!/usr/bin/env python3
"""Validate BART ``wave -v`` by recombining and comparing with lambda zero."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from bart_cfl import bart_base, sha256_file, validate_finite_bart
from run_bart_regularization import (
    _relative_bart_difference,
    _run_streamed,
    recombine_split_complex_bart,
)


def build_command(
    bart: Path,
    maps: Path,
    psf: Path,
    kspace: Path,
    output: Path,
    *,
    split_complex: bool,
    block_size: int,
    iterations: int,
    tolerance: float,
    max_eigenvalue: float,
    backend: str,
) -> list[str]:
    options = ["-l"]
    if split_complex:
        options.append("-v")
    options.extend([
        "-b",
        str(block_size),
        "-f",
        "-r",
        "0",
        "-i",
        str(iterations),
        "-t",
        f"{tolerance:.12g}",
        "-e",
        f"{max_eigenvalue:.12g}",
    ])
    if backend == "gpu":
        options.append("-g")
    elif backend != "cpu":
        raise ValueError(f"Unsupported backend: {backend}")
    return [
        str(bart),
        "wave",
        *options,
        str(maps),
        str(psf),
        str(kspace),
        str(output),
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bart", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--psf", required=True, type=Path)
    parser.add_argument("--kspace", required=True, type=Path)
    parser.add_argument("--lambda-zero", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-eigenvalue", type=float, default=6.70e7)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--maximum-relative-difference", type=float, default=1e-5)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    bart = args.bart.expanduser().resolve()
    maps = bart_base(args.maps.expanduser().resolve())
    psf = bart_base(args.psf.expanduser().resolve())
    kspace = bart_base(args.kspace.expanduser().resolve())
    lambda_zero = bart_base(args.lambda_zero.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Validation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (bart,):
        if not path.is_file():
            raise FileNotFoundError(path)
    for base in (maps, psf, kspace, lambda_zero):
        for suffix in (".hdr", ".cfl"):
            if not base.with_suffix(suffix).is_file():
                raise FileNotFoundError(base.with_suffix(suffix))

    native_base = output_dir / "lambda0_native_complex"
    split_base = output_dir / "lambda0_split_real_imag"
    combined_base = output_dir / "lambda0_recombined"
    native_command = build_command(
        bart,
        maps,
        psf,
        kspace,
        native_base,
        split_complex=False,
        block_size=args.block_size,
        iterations=args.iterations,
        tolerance=args.tolerance,
        max_eigenvalue=args.max_eigenvalue,
        backend=args.backend,
    )
    split_command = build_command(
        bart,
        maps,
        psf,
        kspace,
        split_base,
        split_complex=True,
        block_size=args.block_size,
        iterations=args.iterations,
        tolerance=args.tolerance,
        max_eigenvalue=args.max_eigenvalue,
        backend=args.backend,
    )
    native_log = output_dir / "bart_wave_native_lambda0.log"
    split_log = output_dir / "bart_wave_split_lambda0.log"
    native_wall_seconds = _run_streamed(native_command, native_log, dict(os.environ))
    split_wall_seconds = _run_streamed(split_command, split_log, dict(os.environ))
    recombination = recombine_split_complex_bart(split_base, combined_base)
    native_validation = validate_finite_bart(native_base, (256, 256, 256, 1, 1))
    validation = validate_finite_bart(combined_base, (256, 256, 256, 1, 1))
    relative_difference = _relative_bart_difference(combined_base, native_base)
    if relative_difference > args.maximum_relative_difference:
        raise ValueError(
            "Recombined split lambda zero does not match native-complex FISTA lambda zero: "
            f"{relative_difference} > {args.maximum_relative_difference}"
        )
    version = subprocess.run(
        [str(bart), "version"], check=True, capture_output=True, text=True
    )
    manifest = {
        "format_version": 1,
        "status": "accepted",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "prove LLR/FISTA lambda-zero equivalence with and without wave -v before corrected LLR",
        "commands": {"native_complex": native_command, "split_complex": split_command},
        "wall_seconds": {
            "native_complex": native_wall_seconds,
            "split_complex": split_wall_seconds,
        },
        "logs": {
            "native_complex": {"path": str(native_log), "sha256": sha256_file(native_log)},
            "split_complex": {"path": str(split_log), "sha256": sha256_file(split_log)},
        },
        "bart_version_output": (version.stdout + version.stderr).strip(),
        "inputs": {
            name: {"base": str(base), "cfl_sha256": sha256_file(base.with_suffix(".cfl"))}
            for name, base in {
                "maps": maps,
                "psf": psf,
                "kspace": kspace,
                "accepted_lambda_zero": lambda_zero,
            }.items()
        },
        "recombination": recombination,
        "native_complex_validation": native_validation,
        "recombined_validation": validation,
        "relative_l2_difference_recombined_split_vs_native_complex": relative_difference,
        "relative_l2_difference_native_fista_vs_accepted_cg": _relative_bart_difference(
            native_base, lambda_zero
        ),
        "relative_l2_difference_recombined_split_vs_accepted_cg": _relative_bart_difference(
            combined_base, lambda_zero
        ),
        "maximum_accepted_relative_difference": args.maximum_relative_difference,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Accepted split-complex validation: {manifest_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
