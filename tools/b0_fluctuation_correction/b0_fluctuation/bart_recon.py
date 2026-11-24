"""BART Wave reconstruction, NIfTI export, and QC for a corrected tree.

The corrected tree uses the standard GRE layout, so this module reuses the
Wave command builder and NIfTI converter of ``tools/wave_retro_lr_recon``
(imported from the sibling directory of this repository).
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import plots
from .standard_tree import write_json

TOOLS_ROOT = Path(__file__).resolve().parents[2]
WAVE_RETRO_ROOT = TOOLS_ROOT / "wave_retro_lr_recon"
GRE_WAVELET_LAMBDA = 0.015
BRANCH_LAMBDAS = {"fista_r0": 0.0, "wavelet": GRE_WAVELET_LAMBDA}


def _wave_retro_modules() -> tuple[Any, Any]:
    """Import the GRE command builder and the NIfTI converter script.

    Returns:
        ``(wave_retro_lr.gre, convert_gre_bart_to_nifti)`` modules.

    Raises:
        FileNotFoundError: If the sibling tool is missing.
    """
    if not WAVE_RETRO_ROOT.is_dir():
        raise FileNotFoundError(f"Sibling tool not found: {WAVE_RETRO_ROOT}")
    if str(WAVE_RETRO_ROOT) not in sys.path:
        sys.path.insert(0, str(WAVE_RETRO_ROOT))
    from wave_retro_lr import gre  # noqa: PLC0415

    script = WAVE_RETRO_ROOT / "scripts" / "convert_gre_bart_to_nifti.py"
    spec = importlib.util.spec_from_file_location("convert_gre_bart_to_nifti", script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    return gre, converter


def _cases(output_root: Path) -> list[tuple[str, Path]]:
    """List the prepared cases of a corrected tree.

    Args:
        output_root: Corrected tree root.

    Returns:
        ``(case_id, case_root)`` pairs, normal first.
    """
    cases = [("normal", output_root / "normal")]
    retro = output_root / "retro"
    if retro.is_dir():
        cases += [(p.name, p) for p in sorted(retro.iterdir()) if (p / "bart_inputs" / "manifest.json").is_file()]
    return cases


def _standard_branch(standard_case: Path, branch: str) -> Path | None:
    """Find the standard reconstruction matching a branch, if any.

    Args:
        standard_case: Standard ``normal`` or ``retro/<case>`` directory.
        branch: ``fista_r0`` or ``wavelet``.

    Returns:
        Path of the standard quantitative complex echo-1 array, or ``None``.
    """
    exact = standard_case / "nifti" / branch / "quantitative_complex" / "echo-01_complex.npy"
    if exact.is_file():
        return exact
    if branch == "wavelet":
        matches = sorted((standard_case / "nifti").glob("wavelet*/quantitative_complex/echo-01_complex.npy"))
        if matches:
            return matches[0]
    return None


def reconstruct_tree(
    output_root: str | Path,
    twix: str | Path,
    sequence: str | Path,
    *,
    branches: Sequence[str] = ("fista_r0", "wavelet"),
    gpu: bool = False,
    bart: str = "bart",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run BART Wave on every case and branch, export NIfTI, and write QC.

    Args:
        output_root: Corrected tree root written by ``estimate_and_correct``.
        twix: Source TWIX (NIfTI affine).
        sequence: Matching Pulseq sequence.
        branches: Subset of ``fista_r0`` and ``wavelet``.
        gpu: Use BART's GPU path.
        bart: BART executable.
        log: Progress callback.

    Returns:
        QC manifest.

    Raises:
        ValueError: If a branch name is unknown.

    Side Effects:
        Launches ``bart wave``; writes images, command records, NIfTIs, and
        QC figures below ``output_root``.
    """
    unknown = sorted(set(branches) - set(BRANCH_LAMBDAS))
    if unknown:
        raise ValueError(f"Unknown branches: {unknown}.")
    gre, converter = _wave_retro_modules()
    root = Path(output_root).expanduser().resolve()
    estimation = json.loads((root / "estimation" / "manifest.json").read_text())
    standard_root = Path(estimation["standard_root"])
    maps = root / "normal" / "bart_output" / "coil_sens"
    log_path = root / "logs" / "bart.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    qc_records = []
    for case_id, case_root in _cases(root):
        inputs = case_root / "bart_inputs"
        manifest = json.loads((inputs / "manifest.json").read_text())
        for branch in branches:
            regularization = BRANCH_LAMBDAS[branch]
            images = []
            for echo in manifest["echoes"]:
                label = f"echo-{int(echo['echo']):02d}"
                image = case_root / "bart_output" / branch / label / "image_wave"
                record = image.parent / "wave_command.txt"
                command = gre.build_gre_wave_command(
                    maps=maps, psf=inputs / echo["psf"], kspace=inputs / echo["wave_kspace"],
                    output=image, regularization=regularization, gpu=gpu,
                )
                command[0] = bart
                expected = shlex.join(command)
                done = image.with_suffix(".cfl").is_file() and record.is_file()
                if done and record.read_text().strip() == expected:
                    log(f"reusing {case_id}/{branch}/{label}")
                else:
                    image.parent.mkdir(parents=True, exist_ok=True)
                    record.unlink(missing_ok=True)
                    with log_path.open("a") as stream:
                        stream.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {expected}\n")
                        stream.flush()
                        log(f"$ {expected}")
                        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                    record.write_text(expected + "\n")
                images.append(image)
            nifti = case_root / "nifti" / branch
            if not (nifti / "conversion_manifest.json").is_file():
                case_text = "Normal" if case_id == "normal" else "".join(p.capitalize() for p in case_id.split("_"))
                suffix = f"BARTWaveGREB0Corrected{case_text}{'FISTAR0' if branch == 'fista_r0' else 'Wavelet'}"
                converter.convert(inputs, images, Path(twix), Path(sequence), nifti, suffix)
            corrected = nifti / "quantitative_complex" / "echo-01_complex.npy"
            standard_case = standard_root / "normal" if case_id == "normal" else standard_root / "retro" / case_id
            standard = _standard_branch(standard_case, branch)
            if standard is None:
                continue
            voxel = manifest.get("case", manifest.get("geometry"))["voxel_mm_ro_lin_par"]
            figure = plots.plot_image_comparison(
                {"standard": np.load(standard), "B0-corrected": np.load(corrected)},
                root / "qc" / f"compare_{case_id}_{branch}.png",
                title=f"{case_id} {branch}: standard vs B0-fluctuation corrected ({estimation['method']})",
                voxel_mm=voxel,
            )
            qc_records.append({"case_id": case_id, "branch": branch, "standard": str(standard),
                               "corrected": str(corrected), "figure": str(figure)})
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "comparisons": qc_records}
    write_json(root / "qc" / "manifest.json", result)
    return result
