#!/usr/bin/env python3
"""Run a resumable BART-Wave FISTA lambda sweep on prepared retro-LR cases.

This file does not implement reconstruction. Each new lambda is one call to
``wave_retro_lr.pipeline.run_config``, whose backend is BART ``wave`` with
``-w -f -r <lambda> -g``. Completed lambda runs may be declared as controls.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
TOOL_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = TOOL_ROOT.parents[1]
RETRO_TOOL_ROOT = REPO_ROOT / "tools" / "wave_retro_lr_recon"
sys.path.insert(0, str(RETRO_TOOL_ROOT))

from wave_retro_lr.bart_io import sha256_file  # noqa: E402
from wave_retro_lr.pipeline import (  # noqa: E402
    OUTPUT_FOLDER_NAME,
    _load_json,
    _resolve_path,
    _write_json,
    run_config,
)

from presentation_metrics import validate_metrics_reference_manifest  # noqa: E402


def lambda_label(value: float) -> str:
    """Return a stable path label for a finite nonnegative lambda."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("Wavelet lambdas must be finite and nonnegative.")
    return f"{value:.8g}".replace("+", "").replace(".", "p")


def _settings(config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if config.get("format_version") != 1:
        raise ValueError("Sweep config format_version must be 1.")
    lambdas = [float(value) for value in config.get("wavelet_lambdas", [])]
    if not lambdas or len(set(lambdas)) != len(lambdas):
        raise ValueError("wavelet_lambdas must be a nonempty list without duplicates.")
    if lambdas != sorted(lambdas):
        raise ValueError("wavelet_lambdas must be in increasing order.")
    for value in lambdas:
        lambda_label(value)

    raw_existing = config.get("existing_runs", {})
    if not isinstance(raw_existing, Mapping):
        raise ValueError("existing_runs must map lambda strings to workflow roots.")
    existing: dict[float, Path] = {}
    for raw_lambda, path_value in raw_existing.items():
        value = float(raw_lambda)
        if value not in lambdas or value in existing:
            raise ValueError(f"Invalid or duplicate existing-run lambda: {raw_lambda}")
        existing[value] = _resolve_path(
            path_value, config_path.parent, f"existing_runs.{raw_lambda}"
        )

    base_config = _resolve_path(
        config.get("base_reconstruction_config"),
        config_path.parent,
        "base_reconstruction_config",
    )
    prepared_root = _resolve_path(
        config.get("prepared_cases_root"),
        config_path.parent,
        "prepared_cases_root",
    )
    output_root = _resolve_path(
        config.get("output_root"), config_path.parent, "output_root"
    )
    metrics_reference = _resolve_path(
        config.get("metrics_reference_manifest"),
        config_path.parent,
        "metrics_reference_manifest",
    )
    for path in (base_config, prepared_root / "batch_manifest.json", metrics_reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    subject = str(config.get("subject", "")).strip()
    if not subject:
        raise ValueError("subject must be a nonempty NIfTI subject label.")
    return {
        "config": config,
        "base_config": base_config,
        "prepared_root": prepared_root,
        "output_root": output_root,
        "metrics_reference": metrics_reference,
        "lambdas": lambdas,
        "existing": existing,
        "subject": subject,
    }


def _lambda_from_options(options: Sequence[str]) -> float:
    if not {"-w", "-f", "-g"}.issubset(options) or "-r" not in options:
        raise ValueError("Expected BART wave -w -f -r <lambda> -g options.")
    return float(options[options.index("-r") + 1])


def validate_completed_run(
    workflow_root: Path,
    lambda_value: float,
    expected_cases: Sequence[Mapping[str, Any]],
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one complete batch and its magnitude/phase outputs."""
    batch_path = workflow_root / "batch_manifest.json"
    batch = _load_json(batch_path)
    if batch.get("status") != "complete":
        raise ValueError(f"Retrospective batch is not complete: {batch_path}")
    if batch.get("source") != expected_source or batch.get("cases") != list(expected_cases):
        raise ValueError(f"Retrospective batch source or cases differ: {batch_path}")
    reconstruction = batch.get("reconstruction")
    options = (
        [str(value) for value in reconstruction.get("wave_options", [])]
        if isinstance(reconstruction, Mapping)
        else []
    )
    if options and not math.isclose(
        _lambda_from_options(options), lambda_value, rel_tol=0, abs_tol=1e-15
    ):
        raise ValueError(f"Retrospective batch has the wrong lambda: {batch_path}")

    case_records = []
    for case in expected_cases:
        case_path = workflow_root / str(case["case_name"]) / "case_manifest.json"
        payload = _load_json(case_path)
        if payload.get("status") != "complete" or payload.get("case") != case:
            raise ValueError(f"Retrospective case is incomplete or differs: {case_path}")
        command = [str(value) for value in payload.get("reconstruction", {}).get("command", [])]
        if "wave" not in command or "-g" not in command or "-w" not in command:
            raise ValueError(f"Case does not record BART Wavelet GPU reconstruction: {case_path}")
        if not math.isclose(
            _lambda_from_options(command), lambda_value, rel_tol=0, abs_tol=1e-15
        ):
            raise ValueError(f"Case records the wrong Wavelet lambda: {case_path}")
        if not options:
            options = command[command.index("wave") + 1 :]
        outputs = [Path(value) for value in payload["reconstruction"]["nifti_outputs"]]
        magnitude = [path for path in outputs if "_part-mag_" in path.name]
        phase = [path for path in outputs if "_part-phase_" in path.name]
        if len(magnitude) != 1 or len(phase) != 1 or not all(path.is_file() for path in outputs):
            raise ValueError(f"Case magnitude/phase outputs are incomplete: {case_path}")
        case_records.append(
            {
                "case_name": case["case_name"],
                "case_manifest": str(case_path),
                "case_manifest_sha256": sha256_file(case_path),
                "magnitude_nifti": str(magnitude[0]),
                "phase_nifti": str(phase[0]),
            }
        )
    return {
        "lambda": lambda_value,
        "workflow_root": str(workflow_root),
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": sha256_file(batch_path),
        "wave_options": options,
        "cases": case_records,
    }


def build_lambda_config(
    base: Mapping[str, Any],
    *,
    lambda_value: float,
    output_root: Path,
    prepared_root: Path,
    subject: str,
) -> dict[str, Any]:
    """Specialize the existing retrospective pipeline config for one lambda."""
    config = copy.deepcopy(dict(base))
    config["output_root"] = str(output_root)
    config["prepared_cases_root"] = str(prepared_root)
    config["source"]["subject"] = subject
    config["reconstruction"]["regularizer"] = "wavelet"
    config["reconstruction"]["lambda"] = lambda_value
    return config


def _write_or_validate_config(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if _load_json(path) != payload:
            raise ValueError(f"Generated lambda config changed: {path}")
        return
    _write_json(path, payload)


def run(config_path: Path, *, resume: bool, validate_only: bool) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    settings = _settings(config_path)
    reference = validate_metrics_reference_manifest(settings["metrics_reference"])
    base_summary = run_config(
        settings["base_config"], repo_root=REPO_ROOT, validate_only=True
    )
    expected_cases = base_summary["cases"]
    expected_source = base_summary["source"]
    completed = {
        value: validate_completed_run(root, value, expected_cases, expected_source)
        for value, root in settings["existing"].items()
    }
    if validate_only:
        print(f"Validated {len(settings['lambdas'])} lambdas; {len(completed)} reuse completed runs.")
        print("Backend for new cases: BART wave -w -f -r <lambda> -g")
        return {"status": "validated", "lambda_count": len(settings["lambdas"])}

    output_root: Path = settings["output_root"]
    manifest_path = output_root / "sweep_manifest.json"
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"Sweep output is not empty; use --resume: {output_root}")
    if output_root.exists() and any(output_root.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Nonempty output is not an owned sweep: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "status": "running",
        "purpose": "retrospective low-resolution BART-Wave Wavelet lambda sweep",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "snapshot": settings["config"],
        },
        "base_reconstruction_config": {
            "path": str(settings["base_config"]),
            "sha256": sha256_file(settings["base_config"]),
        },
        "prepared_cases_root": str(settings["prepared_root"]),
        "metrics_reference_manifest": {
            "path": str(settings["metrics_reference"]),
            "sha256": reference["manifest_sha256"],
        },
        "wavelet_lambdas": settings["lambdas"],
        "cases": expected_cases,
        "source": expected_source,
        "lambda_runs": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)

    base = _load_json(settings["base_config"])
    for value in settings["lambdas"]:
        if value in completed:
            record = {**completed[value], "reused_existing_run": True}
        else:
            lambda_root = output_root / "reconstructions" / f"wavelet_lambda-{lambda_label(value)}"
            generated_config = build_lambda_config(
                base,
                lambda_value=value,
                output_root=lambda_root,
                prepared_root=settings["prepared_root"],
                subject=settings["subject"],
            )
            generated_path = lambda_root / "run_config.json"
            _write_or_validate_config(generated_path, generated_config)
            run_config(generated_path, repo_root=REPO_ROOT, resume=resume)
            workflow_root = lambda_root / OUTPUT_FOLDER_NAME
            record = {
                **validate_completed_run(
                    workflow_root, value, expected_cases, expected_source
                ),
                "reused_existing_run": False,
                "generated_config": str(generated_path),
                "generated_config_sha256": sha256_file(generated_path),
            }
        manifest["lambda_runs"] = [
            prior for prior in manifest["lambda_runs"] if prior["lambda"] != value
        ] + [record]
        manifest["lambda_runs"].sort(key=lambda item: item["lambda"])
        _write_json(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    print(f"Retrospective Wavelet sweep manifest: {manifest_path}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate sources, prepared cases, controls, and backend without output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run(args.config, resume=args.resume, validate_only=args.validate_only)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
