"""Read a standard GRE ``vccN`` reconstruction tree and write a corrected copy.

The standard tree is produced by ``tools/wave_retro_lr_recon``::

    <vccN>/normal/bart_inputs/{manifest.json, wave_kspace_echo-NN, psf_echo-NN, ...}
    <vccN>/normal/bart_output/{coil_sens, ecalib_command.txt, <branch>/...}
    <vccN>/retro/<case>/bart_inputs/{manifest.json, wave_kspace_echo-NN, psf_echo-NN, sampling_mask.npy}

The corrected tree written here keeps that layout (corrected k-space, copied
PSFs, copied coil maps and ecalib record, updated manifests), so the existing
BART ``wave`` commands and ``convert_gre_bart_to_nifti.py`` work unchanged.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .cfl import create_cfl, read_cfl, read_shape


@dataclass(frozen=True)
class CaseInputs:
    """Paths and metadata of one prepared GRE case.

    Attributes:
        case_id: ``normal`` or a retrospective case identifier.
        inputs: The case's ``bart_inputs`` directory.
        manifest: Parsed ``bart_inputs/manifest.json``.
    """

    case_id: str
    inputs: Path
    manifest: dict[str, Any]

    @property
    def echoes(self) -> list[dict[str, Any]]:
        """Ordered echo records.

        Returns:
            Echo dictionaries from the manifest.
        """
        return list(self.manifest["echoes"])

    @property
    def geometry(self) -> dict[str, Any]:
        """Case geometry (``case`` for retrospective, ``geometry`` for normal).

        Returns:
            Geometry dictionary.
        """
        return dict(self.manifest.get("case", self.manifest.get("geometry")))

    @property
    def voxel_mm(self) -> tuple[float, float, float]:
        """Voxel size in millimetres.

        Returns:
            ``(RO, LIN, PAR)`` voxel size.
        """
        return tuple(float(v) for v in self.geometry["voxel_mm_ro_lin_par"])  # type: ignore[return-value]

    def echo_index(self, echo: int) -> int:
        """Position of a one-based echo number in :attr:`echoes`.

        Args:
            echo: One-based echo number.

        Returns:
            Zero-based list index.

        Raises:
            ValueError: If the echo is absent.
        """
        for index, record in enumerate(self.echoes):
            if int(record["echo"]) == int(echo):
                return index
        raise ValueError(f"Echo {echo} is not in {self.inputs}.")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object.

    Args:
        path: JSON file.

    Returns:
        Parsed mapping.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write sorted, indented JSON.

    Args:
        path: Destination.
        payload: JSON-compatible mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars and arrays for JSON.

    Args:
        value: Object that ``json`` cannot encode.

    Returns:
        JSON-compatible value.

    Raises:
        TypeError: For unsupported objects.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def load_case(standard_root: str | Path, case_id: str = "normal") -> CaseInputs:
    """Locate one prepared case of a standard tree.

    Args:
        standard_root: Standard ``vccN`` root.
        case_id: ``normal`` or a retrospective case directory name.

    Returns:
        Case paths and manifest.

    Raises:
        FileNotFoundError: If the case manifest is absent.
    """
    root = Path(standard_root).expanduser().resolve()
    inputs = root / "normal" / "bart_inputs" if case_id == "normal" else root / "retro" / case_id / "bart_inputs"
    return CaseInputs(case_id, inputs, _read_json(inputs / "manifest.json"))


def maps_base(standard_root: str | Path) -> Path:
    """BART basename of the standard coil maps.

    Args:
        standard_root: Standard ``vccN`` root.

    Returns:
        ``normal/bart_output/coil_sens``.
    """
    return Path(standard_root).expanduser().resolve() / "normal" / "bart_output" / "coil_sens"


def load_kspace(base: Path, device: torch.device) -> torch.Tensor:
    """Load Wave k-space ``(Xe, Y, Z, C[, 1])`` as ``(C, Xe, Y, Z)`` on a device.

    Args:
        base: BART basename.
        device: Target device.

    Returns:
        Complex64 tensor.
    """
    values = np.asarray(read_cfl(base, 4))
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(values, 3, 0))).to(device)


def load_maps(base: Path, device: torch.device) -> torch.Tensor:
    """Load single-set ESPIRiT maps ``(X, Y, Z, C)`` as ``(C, X, Y, Z)``.

    Args:
        base: BART basename.
        device: Target device.

    Returns:
        Complex64 tensor.
    """
    values = np.asarray(read_cfl(base, 4))
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(values, 3, 0))).to(device)


def load_psf(base: Path, device: torch.device) -> torch.Tensor:
    """Load a Wave PSF ``(Xe, Y, Z)``.

    Args:
        base: BART basename.
        device: Target device.

    Returns:
        Complex64 tensor.
    """
    return torch.from_numpy(np.ascontiguousarray(np.asarray(read_cfl(base, 3)))).to(device)


def write_kspace(base: Path, kspace: torch.Tensor) -> float:
    """Write ``(C, Xe, Y, Z)`` k-space as BART ``(Xe, Y, Z, C, 1)``.

    Args:
        base: Destination basename (written via a temporary name).
        kspace: Device tensor.

    Returns:
        L2 norm of the written data (BART's internal normalization).
    """
    coils, readout, lin, par = kspace.shape
    temporary = base.with_name(f".{base.name}-partial")
    output = create_cfl(temporary, (readout, lin, par, coils, 1))
    for coil in range(coils):
        output[:, :, :, coil, 0] = kspace[coil].cpu().numpy()
    output.flush()
    del output
    for suffix in (".hdr", ".cfl"):
        temporary.with_suffix(suffix).replace(base.with_suffix(suffix))
    return float(torch.linalg.vector_norm(kspace))


def copy_cfl(source: Path, destination: Path) -> None:
    """Copy one CFL pair.

    Args:
        source: Existing basename.
        destination: New basename.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".hdr", ".cfl"):
        shutil.copyfile(source.with_suffix(suffix), destination.with_suffix(suffix))


def install_shared_outputs(standard_root: str | Path, output_root: Path) -> None:
    """Copy the standard coil maps and their ecalib record into the new tree.

    Args:
        standard_root: Standard ``vccN`` root.
        output_root: Corrected tree root.
    """
    source = Path(standard_root).expanduser().resolve() / "normal" / "bart_output"
    destination = output_root / "normal" / "bart_output"
    if not (destination / "coil_sens.cfl").is_file():
        copy_cfl(source / "coil_sens", destination / "coil_sens")
    shutil.copyfile(source / "ecalib_command.txt", destination / "ecalib_command.txt")


def write_case_inputs(
    case: CaseInputs,
    output_root: Path,
    kspace_by_echo: Mapping[int, torch.Tensor],
    correction_record: Mapping[str, Any],
    *,
    sampling_mask: np.ndarray | None = None,
) -> Path:
    """Write corrected k-space, copied PSFs, and an updated manifest for one case.

    Args:
        case: Standard case being corrected.
        output_root: Corrected tree root.
        kspace_by_echo: Corrected ``(C, Xe, Y, Z)`` k-space per one-based echo.
        correction_record: Provenance stored under ``b0_fluctuation_correction``.
        sampling_mask: Optional ``(LIN, PAR)`` mask saved as ``sampling_mask.npy``.

    Returns:
        The written ``bart_inputs`` directory.

    Raises:
        ValueError: If an echo of the case has no corrected k-space.
    """
    inputs = (
        output_root / "normal" / "bart_inputs"
        if case.case_id == "normal"
        else output_root / "retro" / case.case_id / "bart_inputs"
    )
    inputs.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(json.dumps(case.manifest))
    # Artifact hashes of the standard tree do not describe the corrected files.
    manifest.pop("output_artifacts", None)
    echoes = []
    for record in manifest["echoes"]:
        echo = int(record["echo"])
        if echo not in kspace_by_echo:
            raise ValueError(f"No corrected k-space for echo {echo}.")
        wave = inputs / str(record["wave_kspace"])
        record["wave_kspace_norm"] = write_kspace(wave, kspace_by_echo[echo])
        record["wave_kspace_shape"] = list(read_shape(wave))
        copy_cfl(case.inputs / str(record["psf"]), inputs / str(record["psf"]))
        record["standard_wave_kspace"] = str(case.inputs / str(record["wave_kspace"]))
        echoes.append(record)
    manifest["echoes"] = echoes
    if sampling_mask is not None:
        np.save(inputs / "sampling_mask.npy", np.asarray(sampling_mask, dtype=bool), allow_pickle=False)
        if isinstance(manifest.get("sampling"), dict):
            manifest["sampling"]["path"] = str(inputs / "sampling_mask.npy")
    manifest["standard_input_manifest"] = str(case.inputs / "manifest.json")
    manifest["b0_fluctuation_correction"] = dict(correction_record)
    manifest["prepared_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(inputs / "manifest.json", manifest)
    return inputs
