#!/usr/bin/env python3
"""Batch retrospective low-resolution and undersampling Wave-MPRAGE recon.

This program consumes reusable files produced by the standard wave-mprage
reconstruction. It never modifies or relocates those source files. New results
are written beneath <wave-mprage-out-folder>/retro-LR-us/.

The implementation reuses the verified Wave-MPRAGE PSF, CG-SENSE, and NIfTI
helpers from a pinned external wave-mprage checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import zoom


RETRO_FOLDER_NAME = "retro-LR-us"
MANIFEST_CANDIDATES = (
    "wave_mprage_manifest.json",
    "reconstruction_manifest.json",
    "recon_manifest.json",
    "manifest.json",
)


@dataclass(frozen=True)
class SourceGeometry:
    physical_matrix_xyz: tuple[int, int, int]
    physical_fov_mm_xyz: tuple[float, float, float]
    logical_matrix_ro_lin_par: tuple[int, int, int]
    logical_fov_mm_ro_lin_par: tuple[float, float, float]
    readout_oversampling_factor: int
    ncalib: int
    nacs: int

    @property
    def physical_resolution_mm_xyz(self) -> tuple[float, float, float]:
        return tuple(
            fov / matrix
            for fov, matrix in zip(
                self.physical_fov_mm_xyz,
                self.physical_matrix_xyz,
                strict=True,
            )
        )


@dataclass(frozen=True)
class RequestedCase:
    resolution_mm_xyz: tuple[float, float, float]
    acceleration_ry_rz: tuple[int, int]
    label: str | None = None


@dataclass(frozen=True)
class ResolvedCase:
    requested_resolution_mm_xyz: tuple[float, float, float]
    achieved_resolution_mm_xyz: tuple[float, float, float]
    target_physical_matrix_xyz: tuple[int, int, int]
    target_logical_matrix_ro_lin_par: tuple[int, int, int]
    source_acceleration_ry_rz: tuple[int, int]
    target_acceleration_ry_rz: tuple[int, int]
    crop_bounds_lin: tuple[int, int]
    crop_bounds_par: tuple[int, int]
    case_name: str
    label: str | None = None


@dataclass(frozen=True)
class SourceFiles:
    out_folder: Path
    manifest: Path | None
    seq: Path
    twix: Path
    kspace_cc: Path
    csm_acs: Path
    processed_psf_coefficients: Path | None
    a_projy: Path | None
    b_projz: Path | None
    c_projy: Path | None
    c_projz: Path | None
    file_tag: str


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _path_from_value(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _get_nested(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_manifest_value(mapping: Mapping[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        value = _get_nested(mapping, path)
        if value not in (None, ""):
            return value
    return None


def _load_manifest(out_folder: Path, explicit: Path | None) -> tuple[Path | None, dict[str, Any]]:
    if explicit is not None:
        manifest_path = explicit.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    else:
        manifest_path = next(
            (out_folder / name for name in MANIFEST_CANDIDATES if (out_folder / name).is_file()),
            None,
        )
    if manifest_path is None:
        return None, {}
    with manifest_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must contain a JSON object: {manifest_path}")
    return manifest_path, data


def _candidate_matches(
    folder: Path,
    patterns: Sequence[str],
    file_tag: str,
) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in folder.glob(pattern) if path.is_file())
    unique = sorted(set(path.resolve() for path in matches))
    if file_tag:
        tagged = [path for path in unique if file_tag in path.name]
        if tagged:
            unique = tagged
    return unique


def _select_unique_file(
    folder: Path,
    label: str,
    patterns: Sequence[str],
    file_tag: str,
    required: bool = True,
) -> Path | None:
    matches = _candidate_matches(folder, patterns, file_tag)
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    if not matches:
        raise FileNotFoundError(
            f"Could not find {label} in {folder}. Tried: {', '.join(patterns)}"
        )
    formatted = "\n".join(f"  - {path.name}" for path in matches)
    raise RuntimeError(
        f"Multiple candidates were found for {label}:\n{formatted}\n"
        "Use --file-tag or record the exact path in the source manifest."
    )


def _manifest_file(
    manifest: Mapping[str, Any],
    base: Path,
    aliases: Sequence[str],
) -> Path | None:
    return _path_from_value(_first_manifest_value(manifest, aliases), base)


def discover_source_files(args: argparse.Namespace) -> tuple[SourceFiles, dict[str, Any]]:
    out_folder = Path(args.wave_mprage_out_folder).expanduser().resolve()
    if not out_folder.is_dir():
        raise NotADirectoryError(f"Wave-MPRAGE output folder not found: {out_folder}")

    explicit_manifest = Path(args.manifest).expanduser() if args.manifest else None
    manifest_path, manifest = _load_manifest(out_folder, explicit_manifest)
    manifest_base = manifest_path.parent if manifest_path else out_folder

    manifest_tag = _first_manifest_value(
        manifest,
        ("file_tag", "FileTag", "reconstruction.file_tag", "config.file_tag"),
    )
    file_tag = str(args.file_tag if args.file_tag is not None else (manifest_tag or ""))

    seq = _path_from_value(args.seq, Path.cwd()) if args.seq else _manifest_file(
        manifest,
        manifest_base,
        ("seq", "seq_file", "sequence_file", "inputs.seq", "inputs.sequence", "source.seq"),
    )
    twix = _path_from_value(args.twix, Path.cwd()) if args.twix else _manifest_file(
        manifest,
        manifest_base,
        ("twix", "twix_file", "data_file", "inputs.twix", "inputs.data", "source.twix"),
    )
    if seq is None or not seq.is_file():
        raise FileNotFoundError(
            "The matching Pulseq .seq file was not resolved. Record it in the source "
            "manifest or provide --seq PATH."
        )
    if twix is None or not twix.is_file():
        raise FileNotFoundError(
            "The source TWIX .dat file was not resolved. Record it in the source "
            "manifest or provide --twix PATH. It is required for NIfTI orientation."
        )

    kspace_cc = _manifest_file(
        manifest,
        manifest_base,
        ("files.kspace_cc", "outputs.kspace_cc", "artifacts.kspace_cc", "kspace_cc"),
    )
    if kspace_cc is None:
        kspace_cc = _select_unique_file(
            out_folder,
            "coil-compressed k-space",
            ("kspace_cc.npy", "kspace_*_cc_*.npy", "kspace*_cc*.npy"),
            file_tag,
        )

    csm_acs = _manifest_file(
        manifest,
        manifest_base,
        ("files.csm_acs", "outputs.csm_acs", "artifacts.csm_acs", "csm_acs"),
    )
    if csm_acs is None:
        csm_acs = _select_unique_file(
            out_folder,
            "low-resolution ESPIRiT CSM",
            ("csm_acs.npy", "csm_acs_*.npy", "csm_acs*.npy"),
            file_tag,
        )

    processed = _manifest_file(
        manifest,
        manifest_base,
        (
            "files.psf_coefficients",
            "outputs.psf_coefficients",
            "artifacts.psf_coefficients",
            "psf_coefficients",
        ),
    )
    if processed is None:
        processed = _select_unique_file(
            out_folder,
            "processed PSF coefficients",
            (
                "psf_coefficients_processed*.npz",
                "psf_coefficients_processed*.npy",
                "psf_integrated_calib_fit*.npy",
            ),
            file_tag,
            required=False,
        )

    a_projy = b_projz = c_projy = c_projz = None
    if processed is None:
        a_projy = _select_unique_file(
            out_folder,
            "sine/LIN a(kx) fit",
            ("a_fit_all_projy_*.npy", "a_fit_all_projy*.npy"),
            file_tag,
        )
        b_projz = _select_unique_file(
            out_folder,
            "cosine/PAR b(kx) fit",
            ("b_fit_all_projz_*.npy", "b_fit_all_projz*.npy"),
            file_tag,
        )
        c_projy = _select_unique_file(
            out_folder,
            "sine/LIN c(kx) fit",
            ("c_fit_all_projy_*.npy", "c_fit_all_projy*.npy"),
            file_tag,
        )
        c_projz = _select_unique_file(
            out_folder,
            "cosine/PAR c(kx) fit",
            ("c_fit_all_projz_*.npy", "c_fit_all_projz*.npy"),
            file_tag,
        )

    for label, path in (("k-space", kspace_cc), ("CSM", csm_acs)):
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Resolved {label} file does not exist: {path}")

    return (
        SourceFiles(
            out_folder=out_folder,
            manifest=manifest_path,
            seq=seq,
            twix=twix,
            kspace_cc=kspace_cc,
            csm_acs=csm_acs,
            processed_psf_coefficients=processed,
            a_projy=a_projy,
            b_projz=b_projz,
            c_projy=c_projy,
            c_projz=c_projz,
            file_tag=file_tag,
        ),
        manifest,
    )


def load_upstream_module(repo_path: Path):
    repo_path = repo_path.expanduser().resolve()
    recon_dir = repo_path / "recon"
    script = recon_dir / "recon_wave_mprage_from_twix_integrated_nifti.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"wave-mprage reconstruction script not found at {script}. "
            "Initialize the external/wave-mprage submodule or provide --wave-mprage-repo."
        )
    sys.path.insert(0, str(recon_dir))
    spec = importlib.util.spec_from_file_location("wave_mprage_upstream_recon", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import upstream reconstruction module: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_source_geometry(upstream: Any, seq_path: Path) -> tuple[SourceGeometry, Mapping[str, Any]]:
    pp = upstream.pp
    seq = pp.Sequence()
    seq.read(str(seq_path), remove_duplicates=False)
    defs = seq.definitions
    upstream._assert_sag_geometry(defs)
    geom = upstream._derive_hardcoded_sag_logical_geometry(defs)
    os_factor = int(defs.get("ReadoutOversamplingFactor", 4))
    geometry = SourceGeometry(
        physical_matrix_xyz=tuple(int(v) for v in geom["Nxyz"]),
        physical_fov_mm_xyz=tuple(float(v) * 1e3 for v in geom["FOVxyz"]),
        logical_matrix_ro_lin_par=(int(geom["Nro"]), int(geom["Nlin"]), int(geom["Npar"])),
        logical_fov_mm_ro_lin_par=(
            float(geom["FOVro"]) * 1e3,
            float(geom["FOVlin"]) * 1e3,
            float(geom["FOVpar"]) * 1e3,
        ),
        readout_oversampling_factor=os_factor,
        ncalib=int(defs.get("Calibration_Ncalib1", 72)),
        nacs=int(defs.get("Calibration_Nacs", 32)),
    )
    return geometry, defs


def infer_sampling_mask(kspace_cc: np.ndarray) -> np.ndarray:
    """Infer acquired LIN/PAR lines once from zero-filled coil-compressed k-space."""
    if kspace_cc.ndim != 4:
        raise ValueError(
            "Expected kspace_cc with shape (Nro_os, Nlin, Npar, Ncoil); "
            f"got {kspace_cc.shape}."
        )
    line_energy = np.sum(np.abs(kspace_cc) ** 2, axis=(0, 3))
    return line_energy > 0


def _infer_axis_acceleration(sampled: np.ndarray, axis_name: str) -> int:
    sampled = np.asarray(sampled, dtype=bool).reshape(-1)
    n = sampled.size
    center = n // 2
    if not sampled[center]:
        raise ValueError(f"The source k-space center line is missing on {axis_name}.")
    if np.all(sampled):
        return 1

    indices = np.flatnonzero(sampled)
    offsets = np.abs(indices - center)
    nonzero_offsets = offsets[offsets > 0]
    if nonzero_offsets.size == 0:
        raise ValueError(f"Cannot infer {axis_name} acceleration from only one sampled line.")

    candidate = int(np.gcd.reduce(nonzero_offsets.astype(np.int64)))
    if candidate < 2:
        differences = np.diff(indices)
        candidate = int(np.gcd.reduce(differences.astype(np.int64))) if differences.size else 1
    if candidate < 2:
        raise ValueError(
            f"The {axis_name} sampling pattern is not a regular center-referenced lattice."
        )

    expected = ((np.arange(n) - center) % candidate) == 0
    outside = sampled & ~expected
    coverage = float(np.count_nonzero(sampled & expected)) / max(1, np.count_nonzero(expected))
    if np.any(outside) or coverage < 0.80:
        raise ValueError(
            f"The {axis_name} sampling mask is not compatible with a regular R={candidate} "
            f"centered lattice (coverage={coverage:.1%})."
        )
    return candidate


def infer_source_acceleration(mask_2d: np.ndarray) -> tuple[int, int]:
    if mask_2d.ndim != 2:
        raise ValueError(f"Expected a 2D sampling mask; got {mask_2d.shape}.")
    sampled_lin = np.any(mask_2d, axis=1)
    sampled_par = np.any(mask_2d, axis=0)
    return (
        _infer_axis_acceleration(sampled_lin, "LIN/Ry"),
        _infer_axis_acceleration(sampled_par, "PAR/Rz"),
    )


def center_crop_bounds(source_size: int, target_size: int) -> tuple[int, int]:
    if not 1 <= target_size <= source_size:
        raise ValueError(
            f"Target size must satisfy 1 <= target <= source; got {target_size} and {source_size}."
        )
    left = source_size // 2 - target_size // 2
    right = left + target_size
    if left + target_size // 2 != source_size // 2:
        raise AssertionError("Centered crop failed to preserve the Python center index.")
    return left, right


def center_crop_lin_par(array: np.ndarray, target_lin: int, target_par: int) -> np.ndarray:
    if array.ndim == 4:
        source_lin, source_par = array.shape[1], array.shape[2]
        lin = center_crop_bounds(source_lin, target_lin)
        par = center_crop_bounds(source_par, target_par)
        return array[:, lin[0] : lin[1], par[0] : par[1], ...]
    if array.ndim == 2:
        source_lin, source_par = array.shape
        lin = center_crop_bounds(source_lin, target_lin)
        par = center_crop_bounds(source_par, target_par)
        return array[lin[0] : lin[1], par[0] : par[1]]
    raise ValueError(f"Only 2D masks and 4D coil-last k-space are supported; got {array.shape}.")


def make_retrospective_mask(
    nlin: int,
    npar: int,
    source_acceleration_ry_rz: tuple[int, int],
    target_acceleration_ry_rz: tuple[int, int],
) -> np.ndarray:
    source_ry, source_rz = source_acceleration_ry_rz
    target_ry, target_rz = target_acceleration_ry_rz
    for source, target, name in (
        (source_ry, target_ry, "Ry"),
        (source_rz, target_rz, "Rz"),
    ):
        if target < 1:
            raise ValueError(f"{name} must be a positive integer; got {target}.")
        if source > 1 and target != source:
            raise ValueError(
                f"Source {name}={source} is already accelerated. The retrospective patch "
                f"must keep it unchanged, but target {name}={target} was requested."
            )

    lin_keep = np.ones(nlin, dtype=bool)
    par_keep = np.ones(npar, dtype=bool)
    if source_ry == 1:
        lin_keep = ((np.arange(nlin) - nlin // 2) % target_ry) == 0
    if source_rz == 1:
        par_keep = ((np.arange(npar) - npar // 2) % target_rz) == 0
    return lin_keep[:, None] & par_keep[None, :]


def _round_matrix_size(fov_mm: float, requested_resolution_mm: float) -> int:
    if not np.isfinite(requested_resolution_mm) or requested_resolution_mm <= 0:
        raise ValueError(f"Resolution must be finite and positive; got {requested_resolution_mm}.")
    return int(math.floor(fov_mm / requested_resolution_mm + 0.5))


def _format_resolution_component(value: float) -> str:
    rounded = round(float(value) + 0.0, 2)
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def format_case_name(
    achieved_resolution_mm_xyz: Sequence[float],
    target_acceleration_ry_rz: tuple[int, int],
) -> str:
    resolution = "x".join(_format_resolution_component(v) for v in achieved_resolution_mm_xyz)
    ry, rz = target_acceleration_ry_rz
    return f"res{resolution}mm_R{ry}x{rz}"


def resolve_case(
    requested: RequestedCase,
    geometry: SourceGeometry,
    source_acceleration_ry_rz: tuple[int, int],
) -> ResolvedCase:
    source_nx, source_ny, source_nz = geometry.physical_matrix_xyz
    fov_x, fov_y, fov_z = geometry.physical_fov_mm_xyz
    source_res_x, source_res_y, source_res_z = geometry.physical_resolution_mm_xyz
    req_x, req_y, req_z = requested.resolution_mm_xyz

    readout_tolerance = max(1e-4, source_res_z * 1e-3)
    if not math.isclose(req_z, source_res_z, rel_tol=0.0, abs_tol=readout_tolerance):
        raise ValueError(
            "This patch crops only the two phase-encoding dimensions. Requested physical-z "
            f"readout resolution is {req_z:g} mm, but the source readout resolution is "
            f"{source_res_z:g} mm."
        )

    target_nx = _round_matrix_size(fov_x, req_x)
    target_ny = _round_matrix_size(fov_y, req_y)
    target_nz = source_nz
    if target_nx > source_nx or target_ny > source_ny:
        raise ValueError(
            "Requested resolution is finer than the acquired phase-encoding resolution: "
            f"source={geometry.physical_resolution_mm_xyz}, requested={requested.resolution_mm_xyz}."
        )
    if target_nx < 1 or target_ny < 1:
        raise ValueError("Requested resolution produced an invalid target matrix.")

    achieved_xyz = (fov_x / target_nx, fov_y / target_ny, fov_z / target_nz)
    # SAG mapping: logical (RO, LIN, PAR) = physical (z, y, x).
    target_logical = (target_nz, target_ny, target_nx)
    source_ro, source_lin, source_par = geometry.logical_matrix_ro_lin_par
    if target_logical[0] != source_ro:
        raise AssertionError("The readout matrix must remain unchanged.")
    lin_bounds = center_crop_bounds(source_lin, target_logical[1])
    par_bounds = center_crop_bounds(source_par, target_logical[2])

    source_ry, source_rz = source_acceleration_ry_rz
    target_ry, target_rz = requested.acceleration_ry_rz
    # Validate the axis restriction before naming the folder.
    make_retrospective_mask(3, 3, (source_ry, source_rz), (target_ry, target_rz))

    return ResolvedCase(
        requested_resolution_mm_xyz=tuple(float(v) for v in requested.resolution_mm_xyz),
        achieved_resolution_mm_xyz=tuple(float(v) for v in achieved_xyz),
        target_physical_matrix_xyz=(target_nx, target_ny, target_nz),
        target_logical_matrix_ro_lin_par=target_logical,
        source_acceleration_ry_rz=source_acceleration_ry_rz,
        target_acceleration_ry_rz=requested.acceleration_ry_rz,
        crop_bounds_lin=lin_bounds,
        crop_bounds_par=par_bounds,
        case_name=format_case_name(achieved_xyz, requested.acceleration_ry_rz),
        label=requested.label,
    )


def load_cases(path: Path) -> list[RequestedCase]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Cases JSON must contain a non-empty 'cases' list.")

    cases: list[RequestedCase] = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"Case {index} must be a JSON object.")
        resolution = raw.get("resolution_mm", raw.get("resolution_mm_xyz"))
        acceleration = raw.get("acceleration", raw.get("acceleration_ry_rz"))
        if not isinstance(resolution, list) or len(resolution) != 3:
            raise ValueError(f"Case {index} resolution_mm must contain physical [x, y, z].")
        if not isinstance(acceleration, list) or len(acceleration) != 2:
            raise ValueError(f"Case {index} acceleration must contain [Ry, Rz].")
        cases.append(
            RequestedCase(
                resolution_mm_xyz=tuple(float(v) for v in resolution),
                acceleration_ry_rz=tuple(int(v) for v in acceleration),
                label=str(raw["label"]) if raw.get("label") not in (None, "") else None,
            )
        )
    return cases


def _ensure_unique_case_names(cases: list[ResolvedCase]) -> list[ResolvedCase]:
    seen_keys: set[tuple[tuple[int, int, int], tuple[int, int]]] = set()
    used_names: dict[str, tuple[int, int, int]] = {}
    resolved: list[ResolvedCase] = []
    for case in cases:
        key = (case.target_physical_matrix_xyz, case.target_acceleration_ry_rz)
        if key in seen_keys:
            print(f"Skipping duplicate case resolving to {key}: {case.case_name}")
            continue
        seen_keys.add(key)
        name = case.case_name
        previous_matrix = used_names.get(name)
        if previous_matrix is not None and previous_matrix != case.target_physical_matrix_xyz:
            nx, ny, nz = case.target_physical_matrix_xyz
            name = f"{name}_N{nx}x{ny}x{nz}"
        used_names[name] = case.target_physical_matrix_xyz
        if name != case.case_name:
            case = ResolvedCase(**{**asdict(case), "case_name": name})
        resolved.append(case)
    return resolved


def _load_numeric_vector(path: Path, label: str) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False)).squeeze()
    if values.ndim != 1:
        raise ValueError(f"{label} must reduce to one dimension; got {values.shape} from {path}.")
    return values


def _load_processed_coefficients(path: Path, nx_os: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if not all(key in data for key in ("a", "b", "c")):
                raise ValueError(f"Processed coefficient NPZ must contain a, b, and c: {path}")
            a, b, c = (np.asarray(data[key]).squeeze() for key in ("a", "b", "c"))
    else:
        array = np.asarray(np.load(path, allow_pickle=False)).squeeze()
        if array.shape == (3, nx_os):
            a, b, c = array[0], array[1], array[2]
        elif array.shape == (nx_os, 3):
            a, b, c = array[:, 0], array[:, 1], array[:, 2]
        else:
            raise ValueError(
                f"Processed coefficient NPY must have shape (3, Nx_os) or (Nx_os, 3); "
                f"got {array.shape} from {path}."
            )
    vectors = tuple(np.asarray(v, dtype=np.float32).reshape(-1) for v in (a, b, c))
    if any(v.size != nx_os for v in vectors):
        raise ValueError(f"Processed coefficient length must equal Nx_os={nx_os}: {path}")
    return vectors  # type: ignore[return-value]


def load_psf_coefficients(
    source: SourceFiles,
    upstream: Any,
    nx_os: int,
    processing: str,
    fit_kx_min: int | None,
    fit_kx_max: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if source.processed_psf_coefficients is not None:
        try:
            a, b, c = _load_processed_coefficients(source.processed_psf_coefficients, nx_os)
        except ValueError as exc:
            print(
                "WARNING: Processed PSF coefficient candidate could not be used; "
                f"falling back to raw projection fits. Reason: {exc}"
            )
        else:
            return a, b, c, f"processed:{source.processed_psf_coefficients.name}"

    a_path = source.a_projy or _select_unique_file(
        source.out_folder,
        "sine/LIN a(kx) fit",
        ("a_fit_all_projy_*.npy", "a_fit_all_projy*.npy"),
        source.file_tag,
    )
    b_path = source.b_projz or _select_unique_file(
        source.out_folder,
        "cosine/PAR b(kx) fit",
        ("b_fit_all_projz_*.npy", "b_fit_all_projz*.npy"),
        source.file_tag,
    )
    cy_path = source.c_projy or _select_unique_file(
        source.out_folder,
        "sine/LIN c(kx) fit",
        ("c_fit_all_projy_*.npy", "c_fit_all_projy*.npy"),
        source.file_tag,
    )
    cz_path = source.c_projz or _select_unique_file(
        source.out_folder,
        "cosine/PAR c(kx) fit",
        ("c_fit_all_projz_*.npy", "c_fit_all_projz*.npy"),
        source.file_tag,
    )
    assert a_path and b_path and cy_path and cz_path
    a_raw = _load_numeric_vector(a_path, "a_projy")
    b_raw = _load_numeric_vector(b_path, "b_projz")
    c_raw = _load_numeric_vector(cy_path, "c_projy") + _load_numeric_vector(
        cz_path, "c_projz"
    )
    a_processed, b_processed, c_processed = upstream._process_psf_coefficients(
        a_raw,
        b_raw,
        c_raw,
        Nx_os=nx_os,
        coefficient_processing=processing,
        fit_kx_min=fit_kx_min,
        fit_kx_max=fit_kx_max,
        out_folder=None,
        file_tag=source.file_tag,
    )
    vectors = []
    for name, value in zip(("a", "b", "c"), (a_processed, b_processed, c_processed), strict=True):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        vector = np.asarray(value, dtype=np.float32).squeeze()
        if vector.ndim != 1 or vector.size != nx_os:
            raise ValueError(f"Processed {name}(kx) has shape {vector.shape}; expected ({nx_os},).")
        vectors.append(vector)
    return vectors[0], vectors[1], vectors[2], f"raw:{processing}"


def build_target_psf(
    delta_ky_idx: np.ndarray,
    delta_kz_idx: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    nlin: int,
    npar: int,
    yflip: int,
    zflip: int,
) -> np.ndarray:
    y_norm = (np.arange(nlin, dtype=np.float32) - nlin / 2.0) / nlin
    z_norm = (np.arange(npar, dtype=np.float32) - npar / 2.0) / npar
    theory_phase = (
        -float(yflip) * 2.0 * np.pi * delta_ky_idx[:, None, None] * y_norm[None, :, None]
        -float(zflip) * 2.0 * np.pi * delta_kz_idx[:, None, None] * z_norm[None, None, :]
    )
    correction_phase = (
        a[:, None, None] * y_norm[None, :, None]
        + b[:, None, None] * z_norm[None, None, :]
        + c[:, None, None]
    )
    correction_phase = np.nan_to_num(correction_phase, nan=0.0, posinf=0.0, neginf=0.0)
    phase = theory_phase + correction_phase
    return np.exp(1j * phase).astype(np.complex64, copy=False)


def _center_crop_or_pad(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    output = np.zeros(target_shape, dtype=array.dtype)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(array.shape, target_shape, strict=True):
        copy_size = min(source_size, target_size)
        source_start = source_size // 2 - copy_size // 2
        target_start = target_size // 2 - copy_size // 2
        source_slices.append(slice(source_start, source_start + copy_size))
        target_slices.append(slice(target_start, target_start + copy_size))
    output[tuple(target_slices)] = array[tuple(source_slices)]
    return output


def interpolate_target_csm(
    csm_acs: np.ndarray,
    target_unoversampled_shape_ro_lin_par: tuple[int, int, int],
) -> np.ndarray:
    if csm_acs.ndim != 4:
        raise ValueError(f"Expected csm_acs shape (Ncoil, Nro, NlinACS, NparACS); got {csm_acs.shape}.")
    target_shape = (csm_acs.shape[0],) + target_unoversampled_shape_ro_lin_par
    factors = tuple(target / source for target, source in zip(target_shape, csm_acs.shape, strict=True))
    csm = (
        zoom(csm_acs.real, factors, order=1)
        + 1j * zoom(csm_acs.imag, factors, order=1)
    ).astype(np.complex64, copy=False)
    if csm.shape != target_shape:
        csm = _center_crop_or_pad(csm, target_shape)
    rss = np.sqrt(np.sum(np.abs(csm) ** 2, axis=0, keepdims=True))
    csm /= np.maximum(rss, 1e-8)
    return csm


def embed_readout_oversampling(csm: np.ndarray, nx_os: int) -> np.ndarray:
    ncoil, nro, nlin, npar = csm.shape
    if nro > nx_os:
        raise ValueError(f"Unoversampled CSM readout {nro} exceeds Nx_os={nx_os}.")
    sens = np.zeros((ncoil, nx_os, nlin, npar), dtype=np.complex64)
    start = nx_os // 2 - nro // 2
    sens[:, start : start + nro] = csm
    return sens


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid Boolean value: {value!r}")


def _manifest_setting(manifest: Mapping[str, Any], aliases: Sequence[str], default: Any) -> Any:
    value = _first_manifest_value(manifest, aliases)
    return default if value in (None, "") else value


def resolve_runtime_settings(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    def choose(cli_value: Any, aliases: Sequence[str], default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        return _manifest_setting(manifest, aliases, default)

    settings = {
        "yflip": int(choose(args.yflip, ("yflip", "PSFSignConvention.yflip", "config.yflip"), -1)),
        "zflip": int(choose(args.zflip, ("zflip", "PSFSignConvention.zflip", "config.zflip"), -1)),
        "psf_processing": str(
            choose(
                args.psf_coefficient_processing,
                ("psf_coefficient_processing", "config.psf_coefficient_processing"),
                "smooth",
            )
        ),
        "fit_kx_min": choose(
            args.psf_fit_kx_min,
            ("psf_fit_kx_min", "config.psf_fit_kx_min"),
            None,
        ),
        "fit_kx_max": choose(
            args.psf_fit_kx_max,
            ("psf_fit_kx_max", "config.psf_fit_kx_max"),
            None,
        ),
        "nifti_sub": str(
            choose(args.nifti_sub, ("nifti_sub", "nifti.subject", "config.nifti_sub"), "retro")
        ),
        "nifti_suffix": str(
            choose(args.nifti_suffix, ("nifti_suffix", "nifti.suffix", "config.nifti_suffix"), "MPRAGE")
        ),
        "nifti_axis_roles": tuple(
            choose(
                args.nifti_axis_roles,
                ("nifti_axis_roles", "nifti.axis_roles", "config.nifti_axis_roles"),
                ("phase", "readout", "slice"),
            )
        ),
        "nifti_axis_flips": tuple(
            _parse_bool(v)
            for v in choose(
                args.nifti_axis_flips,
                ("nifti_axis_flips", "nifti.axis_flips", "config.nifti_axis_flips"),
                (True, False, False),
            )
        ),
        "twix_coord_system": str(
            choose(
                args.twix_coord_system,
                ("twix_coord_system", "nifti.twix_coord_system"),
                "LPS",
            )
        ),
        "twix_inplane_rot_sign": float(
            choose(
                args.twix_inplane_rot_sign,
                ("twix_inplane_rot_sign", "nifti.twix_inplane_rot_sign"),
                -1.0,
            )
        ),
    }
    if settings["yflip"] not in (-1, 1) or settings["zflip"] not in (-1, 1):
        raise ValueError("yflip and zflip must each be -1 or +1.")
    if settings["psf_processing"] not in ("smooth", "sine-line"):
        raise ValueError("PSF coefficient processing must be 'smooth' or 'sine-line'.")
    if len(settings["nifti_axis_roles"]) != 3 or len(settings["nifti_axis_flips"]) != 3:
        raise ValueError("NIfTI axis roles and flips must each contain exactly three entries.")
    return settings


def _sanitize_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return value.strip("-_") or "retro"


def save_case_nifti(
    upstream: Any,
    image: np.ndarray,
    twix_file: Path,
    case_dir: Path,
    case: ResolvedCase,
    geometry: SourceGeometry,
    settings: Mapping[str, Any],
    save_phase: bool,
    metadata: Mapping[str, Any],
) -> list[str]:
    from utils.nifti_export_twix import (  # type: ignore[import-not-found]
        apply_array_axis_flips,
        crop_readout_oversampling,
        make_nifti_affine_from_twix,
        prepare_image_array,
        save_nifti_with_json,
    )

    case_dir.mkdir(parents=True, exist_ok=True)
    cropped = crop_readout_oversampling(
        np.asarray(image),
        crop_readout_os=geometry.readout_oversampling_factor,
    )
    outputs = [("mag", prepare_image_array(cropped, part="mag"))]
    if save_phase:
        outputs.append(("phase", prepare_image_array(cropped, part="phase")))
    arrays = apply_array_axis_flips(
        [array for _, array in outputs],
        settings["nifti_axis_flips"],
    )
    outputs = [(part, array) for (part, _), array in zip(outputs, arrays, strict=True)]

    # Logical RO/LIN/PAR spacing = physical z/y/x spacing.
    achieved_x, achieved_y, achieved_z = case.achieved_resolution_mm_xyz
    logical_voxel_size_mm = (achieved_z, achieved_y, achieved_x)
    affine, voxel_size_from_affine, twix_info = make_nifti_affine_from_twix(
        twix_file=str(twix_file),
        npy_shape=outputs[0][1].shape,
        twix_array_axis_roles=settings["nifti_axis_roles"],
        twix_array_axis_flips=(False, False, False),
        twix_coord_system=settings["twix_coord_system"],
        twix_inplane_rot_sign=settings["twix_inplane_rot_sign"],
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=logical_voxel_size_mm,
    )

    subject = _sanitize_component(settings["nifti_sub"])
    if not subject.startswith("sub-"):
        subject = "sub-" + subject
    suffix = _sanitize_component(settings["nifti_suffix"])
    written: list[str] = []
    for part, array in outputs:
        base = f"{subject}_part-{part}_{suffix}"
        nii_path = case_dir / f"{base}.nii.gz"
        json_path = case_dir / f"{base}.json"
        sidecar = dict(metadata)
        sidecar.update(
            {
                "Part": part,
                "Units": "rad" if part == "phase" else "arbitrary",
                "NIfTIVoxelSizeMm": [float(v) for v in voxel_size_from_affine],
                "NIfTILogicalVoxelSizeMmRO_LIN_PAR": [float(v) for v in logical_voxel_size_mm],
                "NIfTIPhysicalArrayFlipsApplied": [bool(v) for v in settings["nifti_axis_flips"]],
                "NIfTITwixArrayAxisRoles": list(settings["nifti_axis_roles"]),
                "NIfTIOrientation": {
                    "OrientationSource": "TwixMeasYaps",
                    "TwixOrientation": twix_info,
                },
            }
        )
        save_nifti_with_json(array, affine, nii_path, json_path, metadata=sidecar)
        written.extend([str(nii_path), str(json_path)])
    return written


def _case_payload(
    case: ResolvedCase,
    geometry: SourceGeometry,
    source: SourceFiles,
    save_intermediate: str,
    psf_source: str,
) -> dict[str, Any]:
    return {
        "case_name": case.case_name,
        "label": case.label,
        "requested_resolution_mm_xyz": list(case.requested_resolution_mm_xyz),
        "achieved_resolution_mm_xyz": list(case.achieved_resolution_mm_xyz),
        "folder_resolution_rounding_decimals": 2,
        "source_physical_matrix_xyz": list(geometry.physical_matrix_xyz),
        "target_physical_matrix_xyz": list(case.target_physical_matrix_xyz),
        "source_logical_matrix_ro_lin_par": list(geometry.logical_matrix_ro_lin_par),
        "target_logical_matrix_ro_lin_par": list(case.target_logical_matrix_ro_lin_par),
        "source_physical_fov_mm_xyz": list(geometry.physical_fov_mm_xyz),
        "source_acceleration_ry_rz": list(case.source_acceleration_ry_rz),
        "target_acceleration_ry_rz": list(case.target_acceleration_ry_rz),
        "crop_bounds_half_open": {
            "LIN": list(case.crop_bounds_lin),
            "PAR": list(case.crop_bounds_par),
        },
        "sampling_mask_saved": False,
        "save_intermediate": save_intermediate,
        "source_kspace_cc": str(source.kspace_cc),
        "source_csm_acs": str(source.csm_acs),
        "source_seq": str(source.seq),
        "source_twix": str(source.twix),
        "psf_coefficient_source": psf_source,
    }


def _prepare_case_directories(
    retro_root: Path,
    case_name: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    case_dir = retro_root / case_name
    nifti_dir = retro_root / "nifti" / case_name
    existing = [path for path in (case_dir, nifti_dir) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Case output already exists for {case_name}: {existing}. Use --overwrite to replace it."
        )
    if overwrite:
        for path in existing:
            shutil.rmtree(path)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir, nifti_dir


def run_batch(args: argparse.Namespace) -> int:
    source, manifest = discover_source_files(args)
    repo_path = Path(args.wave_mprage_repo).expanduser()
    upstream = load_upstream_module(repo_path)
    import torch

    geometry, defs = read_source_geometry(upstream, source.seq)
    kspace_cc = np.load(source.kspace_cc, mmap_mode=None, allow_pickle=False)
    if not np.iscomplexobj(kspace_cc):
        raise ValueError(f"kspace_cc must be complex-valued: {source.kspace_cc}")
    source_ro, source_lin, source_par = geometry.logical_matrix_ro_lin_par
    expected_shape = (
        source_ro * geometry.readout_oversampling_factor,
        source_lin,
        source_par,
    )
    if tuple(kspace_cc.shape[:3]) != expected_shape:
        raise ValueError(
            f"kspace_cc spatial shape {kspace_cc.shape[:3]} does not match sequence-derived "
            f"logical shape {expected_shape}."
        )
    nx_os = kspace_cc.shape[0]

    source_mask = infer_sampling_mask(kspace_cc)
    source_acceleration = infer_source_acceleration(source_mask)
    seq_acceleration = (
        int(defs.get("MPRAGE_PE2_R", source_acceleration[0])),
        int(defs.get("MPRAGE_PE1_R", source_acceleration[1])),
    )
    if source_acceleration != seq_acceleration:
        raise ValueError(
            "Acceleration inferred from kspace_cc disagrees with the .seq definitions: "
            f"mask={source_acceleration}, seq={seq_acceleration}."
        )

    settings = resolve_runtime_settings(args, manifest)
    requested_cases = load_cases(Path(args.cases).expanduser().resolve())
    resolved_cases = _ensure_unique_case_names(
        [resolve_case(case, geometry, source_acceleration) for case in requested_cases]
    )
    if not resolved_cases:
        raise ValueError("No unique cases remain after resolution/matrix de-duplication.")

    mode = upstream._resolve_mprage_wave_mode(
        requested_mode="auto",
        mprage_seq_file=str(source.seq),
        Nx_os=nx_os,
        Ncalib=geometry.ncalib,
        Nacs=geometry.nacs,
        slice_orientation="SAG",
    )
    if mode != "wave":
        raise ValueError("This retrospective PSF-calibrated patch currently supports Wave-MPRAGE only.")

    a, b, c, psf_source = load_psf_coefficients(
        source=source,
        upstream=upstream,
        nx_os=nx_os,
        processing=settings["psf_processing"],
        fit_kx_min=None if settings["fit_kx_min"] is None else int(settings["fit_kx_min"]),
        fit_kx_max=None if settings["fit_kx_max"] is None else int(settings["fit_kx_max"]),
    )
    nacs_total = int(nx_os * (4 * geometry.ncalib + geometry.nacs * geometry.nacs))
    delta_ky_idx, delta_kz_idx = upstream.generate_theoretical_wave_trajectory(
        fn_seq=str(source.seq),
        Nx_os=nx_os,
        Nacs_total=nacs_total,
        slice_orientation="SAG",
    )
    delta_ky_idx = np.asarray(delta_ky_idx, dtype=np.float32).reshape(-1)
    delta_kz_idx = np.asarray(delta_kz_idx, dtype=np.float32).reshape(-1)
    if delta_ky_idx.size != nx_os or delta_kz_idx.size != nx_os:
        raise ValueError("Theoretical trajectory length does not match kspace_cc readout length.")

    csm_acs = np.load(source.csm_acs, allow_pickle=False)
    if csm_acs.shape[0] != kspace_cc.shape[-1]:
        raise ValueError(
            f"CSM virtual-coil count {csm_acs.shape[0]} does not match k-space {kspace_cc.shape[-1]}."
        )

    retro_root = source.out_folder / RETRO_FOLDER_NAME
    retro_root.mkdir(parents=True, exist_ok=True)
    (retro_root / "nifti").mkdir(parents=True, exist_ok=True)

    batch_payload: dict[str, Any] = {
        "source_out_folder": str(source.out_folder),
        "source_manifest": str(source.manifest) if source.manifest else None,
        "source_files": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(source).items()},
        "source_geometry": asdict(geometry),
        "source_acceleration_ry_rz": list(source_acceleration),
        "save_intermediate": args.save_intermediate,
        "save_nifti": bool(args.save_nifti),
        "save_nifti_phase": bool(args.save_nifti_phase),
        "cg": {"iterations": int(args.cg_iters), "tolerance": float(args.cg_tol)},
        "settings": settings,
        "cases": [],
    }

    print("Retrospective Wave-MPRAGE batch")
    print(f"  Source output: {source.out_folder}")
    print(f"  Source k-space: {source.kspace_cc.name} {kspace_cc.shape}")
    print(f"  Source acceleration inferred from k-space: R{source_acceleration[0]}x{source_acceleration[1]}")
    print(f"  Cases: {len(resolved_cases)}")
    for case in resolved_cases:
        print(
            f"  - {case.case_name}: requested={case.requested_resolution_mm_xyz} mm, "
            f"achieved={tuple(round(v, 6) for v in case.achieved_resolution_mm_xyz)} mm, "
            f"matrix={case.target_physical_matrix_xyz}, R={case.target_acceleration_ry_rz}"
        )

    if args.validate_only:
        batch_payload["cases"] = [asdict(case) for case in resolved_cases]
        _json_dump(retro_root / "batch_info.json", batch_payload)
        print("Validation complete; reconstruction was not run.")
        return 0

    for case_index, case in enumerate(resolved_cases, start=1):
        print(f"\n[{case_index}/{len(resolved_cases)}] {case.case_name}")
        case_dir, nifti_dir = _prepare_case_directories(
            retro_root, case.case_name, overwrite=args.overwrite
        )
        _, target_lin, target_par = case.target_logical_matrix_ro_lin_par
        kspace_lr = center_crop_lin_par(kspace_cc, target_lin, target_par)
        source_mask_lr = center_crop_lin_par(source_mask, target_lin, target_par)
        extra_mask = make_retrospective_mask(
            target_lin,
            target_par,
            source_acceleration,
            case.target_acceleration_ry_rz,
        )
        case_mask = source_mask_lr & extra_mask
        kspace_case = (
            kspace_lr * case_mask[None, :, :, None]
        ).astype(np.complex64, copy=False)

        csm_target = interpolate_target_csm(
            csm_acs,
            case.target_logical_matrix_ro_lin_par,
        )
        sens_np = embed_readout_oversampling(csm_target, nx_os)
        psf_target = build_target_psf(
            delta_ky_idx,
            delta_kz_idx,
            a,
            b,
            c,
            target_lin,
            target_par,
            yflip=settings["yflip"],
            zflip=settings["zflip"],
        )

        y_meas = torch.from_numpy(kspace_case).permute(3, 0, 1, 2).contiguous()
        sens = torch.from_numpy(sens_np).contiguous()
        psf_t = torch.from_numpy(psf_target).contiguous()
        mask_t = torch.from_numpy(case_mask.astype(np.float32)).view(1, 1, target_lin, target_par)
        image = upstream.cg_sense_wave(
            y=y_meas,
            sens=sens,
            psf_to_use=psf_t,
            mask_t=mask_t,
            n_iter=int(args.cg_iters),
            tol=float(args.cg_tol),
            init="zero",
            use_preconditioner=True,
            use_direct_if_full=True,
        )
        image_np = image.detach().cpu().numpy().astype(np.complex64, copy=False)

        case_payload = _case_payload(
            case,
            geometry,
            source,
            args.save_intermediate,
            psf_source,
        )
        case_payload["sampled_line_count"] = int(np.count_nonzero(case_mask))
        case_payload["effective_acceleration"] = float(case_mask.size / np.count_nonzero(case_mask))
        case_payload["outputs"] = {}

        if args.save_intermediate in ("standard", "all"):
            image_path = case_dir / "image.npy"
            np.save(image_path, image_np)
            case_payload["outputs"]["image_npy"] = str(image_path)
        if args.save_intermediate == "all":
            kspace_path = case_dir / "kspace_lr_undersampled.npy"
            csm_path = case_dir / "csm_target.npy"
            psf_path = case_dir / "psf_target.npy"
            np.save(kspace_path, kspace_case)
            # Save the unoversampled normalized CSM; the oversampled zero-padding is reproducible.
            np.save(csm_path, csm_target)
            np.save(psf_path, psf_target)
            case_payload["outputs"].update(
                {
                    "kspace_lr_undersampled_npy": str(kspace_path),
                    "csm_target_npy": str(csm_path),
                    "psf_target_npy": str(psf_path),
                }
            )

        if args.save_nifti:
            nifti_outputs = save_case_nifti(
                upstream=upstream,
                image=image_np,
                twix_file=source.twix,
                case_dir=nifti_dir,
                case=case,
                geometry=geometry,
                settings=settings,
                save_phase=args.save_nifti_phase,
                metadata={
                    **case_payload,
                    "Reconstruction": "Retrospective LR + undersampling + calibrated Wave CG-SENSE",
                    "RetrospectiveLowResolution": True,
                    "RetrospectiveUndersampling": True,
                },
            )
            case_payload["outputs"]["nifti"] = nifti_outputs

        _json_dump(case_dir / "case_info.json", case_payload)
        batch_payload["cases"].append(case_payload)

        del y_meas, sens, psf_t, mask_t, image

    _json_dump(retro_root / "batch_info.json", batch_payload)
    print(f"\nCompleted {len(resolved_cases)} case(s). Results: {retro_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run batch retrospective low-resolution + undersampling + calibrated "
            "Wave-MPRAGE reconstruction from an existing wave-mprage output folder."
        )
    )
    parser.add_argument(
        "--wave-mprage-out-folder",
        required=True,
        help="Existing output folder produced by the standard wave-mprage reconstruction.",
    )
    parser.add_argument("--cases", required=True, help="JSON file containing desired resolution/acceleration cases.")
    parser.add_argument(
        "--wave-mprage-repo",
        default=str(Path(__file__).resolve().parents[1] / "external" / "wave-mprage"),
        help="Pinned upstream wave-mprage checkout. Default: external/wave-mprage.",
    )
    parser.add_argument("--manifest", default=None, help="Optional explicit source manifest path.")
    parser.add_argument("--seq", default=None, help="Override/fallback matching integrated .seq path.")
    parser.add_argument("--twix", default=None, help="Override/fallback source TWIX .dat path for NIfTI geometry.")
    parser.add_argument("--file-tag", default=None, help="Resolve tagged source artifacts when filenames are ambiguous.")
    parser.add_argument(
        "--save-intermediate",
        choices=("none", "standard", "all"),
        default="standard",
        help=(
            "none: metadata/NIfTI only; standard: also image.npy; all: also undersampled "
            "LR k-space, target CSM, and target PSF. Default: standard."
        ),
    )
    parser.add_argument(
        "--save-nifti",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save magnitude NIfTI output for every case. Default: enabled.",
    )
    parser.add_argument(
        "--save-nifti-phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save phase NIfTI output. Default: disabled.",
    )
    parser.add_argument("--nifti-sub", default=None, help="NIfTI subject stem; default from manifest or 'retro'.")
    parser.add_argument("--nifti-suffix", default=None, help="NIfTI suffix; default MPRAGE.")
    parser.add_argument("--nifti-axis-roles", nargs=3, default=None, metavar=("AXIS0", "AXIS1", "AXIS2"))
    parser.add_argument("--nifti-axis-flips", nargs=3, default=None, metavar=("AXIS0", "AXIS1", "AXIS2"))
    parser.add_argument("--twix-coord-system", choices=("LPS", "RAS"), default=None)
    parser.add_argument("--twix-inplane-rot-sign", type=float, default=None)
    parser.add_argument("--yflip", type=int, choices=(-1, 1), default=None)
    parser.add_argument("--zflip", type=int, choices=(-1, 1), default=None)
    parser.add_argument(
        "--psf-coefficient-processing",
        choices=("smooth", "sine-line"),
        default=None,
        help="Used only when processed a/b/c coefficients are not already saved.",
    )
    parser.add_argument("--psf-fit-kx-min", type=int, default=None)
    parser.add_argument("--psf-fit-kx-max", type=int, default=None)
    parser.add_argument("--cg-iters", type=int, default=50)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs for matching cases.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs/cases and write batch_info.json only.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cg_iters < 1:
        parser.error("--cg-iters must be positive.")
    if not np.isfinite(args.cg_tol) or args.cg_tol <= 0:
        parser.error("--cg-tol must be finite and positive.")
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
