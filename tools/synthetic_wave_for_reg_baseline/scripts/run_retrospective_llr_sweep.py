#!/usr/bin/env python3
"""Run a resumable corrected-LLR sweep on prepared retro-LR cases.

This file does not implement reconstruction. Every setting is one call to
``wave_retro_lr.pipeline.run_config``; its backend is BART ``wave`` with
split-complex corrected LLR options ``-l -v -b <block> -f -r <lambda> -g``.
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
sys.path.insert(0, str(REPO_ROOT / "tools" / "wave_retro_lr_recon"))

from wave_retro_lr.bart_io import sha256_file  # noqa: E402
from wave_retro_lr.core import build_wave_options  # noqa: E402
from wave_retro_lr.pipeline import (  # noqa: E402
    OUTPUT_FOLDER_NAME,
    _load_json,
    _resolve_path,
    _write_json,
    run_config,
)

from run_retrospective_wavelet_sweep import (  # noqa: E402
    lambda_label,
    validate_completed_run,
)


def _settings(config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if config.get("format_version") != 1:
        raise ValueError("LLR sweep config format_version must be 1.")
    raw_settings = config.get("llr_settings")
    if not isinstance(raw_settings, list) or not raw_settings:
        raise ValueError("llr_settings must be a nonempty list.")
    settings = []
    keys = set()
    for raw in raw_settings:
        if not isinstance(raw, Mapping):
            raise ValueError("Each LLR setting must be an object.")
        block_size = int(raw["block_size"])
        lambdas = [float(value) for value in raw["lambdas"]]
        if block_size < 1 or not lambdas or lambdas != sorted(lambdas):
            raise ValueError("LLR blocks must be positive and lambdas increasing.")
        for value in lambdas:
            if not math.isfinite(value) or value <= 0:
                raise ValueError("LLR sweep lambdas must be positive and finite.")
            key = (block_size, value)
            if key in keys:
                raise ValueError(f"Duplicate LLR setting: {key}")
            keys.add(key)
            settings.append({"block_size": block_size, "lambda": value})
    subject = str(config.get("subject", "")).strip()
    if not subject:
        raise ValueError("subject must be a nonempty NIfTI subject label.")
    resolved = {
        "config": config,
        "base_config": _resolve_path(
            config.get("base_reconstruction_config"),
            config_path.parent,
            "base_reconstruction_config",
        ),
        "prepared_root": _resolve_path(
            config.get("prepared_cases_root"),
            config_path.parent,
            "prepared_cases_root",
        ),
        "control_root": _resolve_path(
            config.get("fista_lambda0_control_root"),
            config_path.parent,
            "fista_lambda0_control_root",
        ),
        "output_root": _resolve_path(
            config.get("output_root"), config_path.parent, "output_root"
        ),
        "settings": settings,
        "subject": subject,
    }
    for path in (
        resolved["base_config"],
        resolved["prepared_root"] / "batch_manifest.json",
        resolved["control_root"] / "batch_manifest.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    return resolved


def _command_setting(command: Sequence[str]) -> tuple[int, float]:
    required = {"wave", "-l", "-v", "-b", "-f", "-r", "-g"}
    if not required.issubset(command):
        raise ValueError("Expected corrected GPU BART Wave LLR options.")
    return int(command[command.index("-b") + 1]), float(command[command.index("-r") + 1])


def validate_completed_llr_run(
    workflow_root: Path,
    *,
    block_size: int,
    lambda_value: float,
    expected_cases: Sequence[Mapping[str, Any]],
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one completed corrected-LLR batch and both NIfTI parts."""
    batch_path = workflow_root / "batch_manifest.json"
    batch = _load_json(batch_path)
    if (
        batch.get("status") != "complete"
        or batch.get("source") != expected_source
        or batch.get("cases") != list(expected_cases)
    ):
        raise ValueError(f"LLR batch is incomplete or differs: {batch_path}")
    records = []
    for case in expected_cases:
        case_path = workflow_root / str(case["case_name"]) / "case_manifest.json"
        payload = _load_json(case_path)
        if payload.get("status") != "complete" or payload.get("case") != case:
            raise ValueError(f"LLR case is incomplete or differs: {case_path}")
        command = [str(value) for value in payload["reconstruction"]["command"]]
        found_block, found_lambda = _command_setting(command)
        if found_block != block_size or not math.isclose(
            found_lambda, lambda_value, rel_tol=0, abs_tol=1e-15
        ):
            raise ValueError(f"LLR case records the wrong block/lambda: {case_path}")
        outputs = [Path(value) for value in payload["reconstruction"]["nifti_outputs"]]
        magnitude = [path for path in outputs if "_part-mag_" in path.name]
        phase = [path for path in outputs if "_part-phase_" in path.name]
        if len(magnitude) != 1 or len(phase) != 1 or not all(path.is_file() for path in outputs):
            raise ValueError(f"LLR magnitude/phase outputs are incomplete: {case_path}")
        records.append(
            {
                "case_name": case["case_name"],
                "case_manifest": str(case_path),
                "case_manifest_sha256": sha256_file(case_path),
                "magnitude_nifti": str(magnitude[0]),
                "phase_nifti": str(phase[0]),
            }
        )
    return {
        "block_size": block_size,
        "lambda": lambda_value,
        "workflow_root": str(workflow_root),
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": sha256_file(batch_path),
        "cases": records,
    }


def build_llr_config(
    base: Mapping[str, Any],
    *,
    block_size: int,
    lambda_value: float,
    output_root: Path,
    prepared_root: Path,
    subject: str,
) -> dict[str, Any]:
    """Specialize the existing retrospective pipeline config for corrected LLR."""
    config = copy.deepcopy(dict(base))
    config["output_root"] = str(output_root)
    config["prepared_cases_root"] = str(prepared_root)
    config["source"]["subject"] = subject
    config["reconstruction"].update(
        {"regularizer": "llr", "lambda": lambda_value, "block_size": block_size}
    )
    return config


def _write_or_validate(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if _load_json(path) != payload:
            raise ValueError(f"Generated LLR config changed: {path}")
        return
    _write_json(path, payload)


def run(config_path: Path, *, validate_only: bool, resume: bool) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    settings = _settings(config_path)
    base_summary = run_config(
        settings["base_config"], repo_root=REPO_ROOT, validate_only=True
    )
    expected_cases = base_summary["cases"]
    expected_source = base_summary["source"]
    control = validate_completed_run(
        settings["control_root"], 0.0, expected_cases, expected_source
    )
    for item in settings["settings"]:
        build_wave_options(
            "llr",
            item["lambda"],
            block_size=item["block_size"],
            iterations=100,
            tolerance=1e-6,
            maximum_eigenvalue=None,
        )
    if validate_only:
        print(
            f"Validated {len(settings['settings'])} corrected-LLR settings across "
            f"{len(expected_cases)} retrospective geometries."
        )
        print("Backend: BART wave -l -v -b <block> -f -r <lambda> -g")
        print("FISTA lambda-zero control: reused for evaluation only")
        return {"status": "validated", "reconstruction_count": len(settings["settings"]) * len(expected_cases)}

    output_root: Path = settings["output_root"]
    manifest_path = output_root / "sweep_manifest.json"
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"LLR sweep output is not empty; use --resume: {output_root}")
    if output_root.exists() and any(output_root.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Nonempty output is not an owned LLR sweep: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "status": "running",
        "purpose": "retrospective low-resolution corrected-LLR block/lambda sweep",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "snapshot": settings["config"],
        },
        "prepared_cases_root": str(settings["prepared_root"]),
        "cases": expected_cases,
        "source": expected_source,
        "llr_settings": settings["settings"],
        "fista_lambda0_control": control,
        "llr_runs": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    base = _load_json(settings["base_config"])
    for item in settings["settings"]:
        block_size = item["block_size"]
        lambda_value = item["lambda"]
        setting_root = (
            output_root
            / "reconstructions"
            / f"llr_block-{block_size}_lambda-{lambda_label(lambda_value)}"
        )
        generated = build_llr_config(
            base,
            block_size=block_size,
            lambda_value=lambda_value,
            output_root=setting_root,
            prepared_root=settings["prepared_root"],
            subject=settings["subject"],
        )
        generated_path = setting_root / "run_config.json"
        _write_or_validate(generated_path, generated)
        run_config(generated_path, repo_root=REPO_ROOT, resume=resume)
        record = {
            **validate_completed_llr_run(
                setting_root / OUTPUT_FOLDER_NAME,
                block_size=block_size,
                lambda_value=lambda_value,
                expected_cases=expected_cases,
                expected_source=expected_source,
            ),
            "generated_config": str(generated_path),
            "generated_config_sha256": sha256_file(generated_path),
        }
        manifest["llr_runs"] = [
            prior
            for prior in manifest["llr_runs"]
            if (prior["block_size"], prior["lambda"]) != (block_size, lambda_value)
        ] + [record]
        manifest["llr_runs"].sort(key=lambda run: (run["block_size"], run["lambda"]))
        _write_json(manifest_path, manifest)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    print(f"Retrospective LLR sweep manifest: {manifest_path}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run(args.config, validate_only=args.validate_only, resume=args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
