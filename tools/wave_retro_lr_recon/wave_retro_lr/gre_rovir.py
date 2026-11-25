"""Experimental ROVir workflow for measured Wave-GRE with box/ellipsoid ROIs.

ROVir (region-optimized virtual coils) replaces standard PCA coil compression
by a physical-to-virtual transform that keeps signal from a desired region and
suppresses signal from a nuisance region. The GRE use case is a face region
that aliases into the brain under acceleration. A face box spanning all
partitions also covers the frontal lobe, so this workflow accepts spheres and
ellipsoids (see :mod:`wave_retro_lr.rovir_regions`) in addition to boxes.

The workflow reuses an existing standard-PCA GRE reconstruction tree
(``vccN/normal`` and optional ``vccN/retro/<case>``) for geometry, the
calibrated PSF, the CSM crop, and the regularization choice. It reads physical
coils from the original TWIX and never consumes the standard PCA k-space.

Output layout below one user-approved root::

    physical_inputs/{manifest.json, bart/, figures/}
    roi/<region_id>/{manifest.json, masks/, inputs/, transform/, figures/}
    rovir_<region_id>_vcc<N>/
        manifest.json
        normal/{bart_inputs, bart_output, nifti}
        retro/<case>/{bart_inputs, bart_output, nifti}
        qc/
    logs/

Every stage is resumable: completed stages are validated and reused, and
BART outputs are written under temporary names and renamed after success.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .bart_io import cfl_record, create_cfl, open_cfl, read_shape, sha256_file
from .gre import (
    EXTENDED_READOUT,
    GRE_SHARED_WAVELET_LAMBDA,
    _case_mask,
    build_gre_wave_command,
    gre_cases,
    gre_wavelet_selection_provenance,
    inspect_gre_twix,
    load_wave_gre_helpers,
    validate_gre_sequence,
)
from .noise_prewhitening import (
    apply_noise_prewhitening_coillast,
    load_recorded_whitening_matrix,
)
from .refscan import read_refscan_set, refscan_shape
from .rovir import (
    region_correlation_diagnostics,
    region_energy_curve,
    rovir_matrix_view,
    validate_rovir_transform,
)
from .rovir_regions import null_region_union
from .twix_streaming import load_compressed_image

FORMAT_VERSION = 1
PHYSICAL_DIRECTORY = "physical_inputs"
ROI_DIRECTORY = "roi"
LOG_DIRECTORY = "logs"
DEFAULT_ROVIR_VIRTUAL_COILS = 24
DEFAULT_RETRO_CASES = ("native_r3x3",)
SUPPORTED_RETRO_CASES = ("native_r3x2", "native_r3x3")
BRANCH_LAMBDAS = {"fista_r0": 0.0, "wavelet": GRE_SHARED_WAVELET_LAMBDA}
DEFAULT_BRANCHES = ("fista_r0", "wavelet")
ENERGY_CURVE_COUNTS = (4, 8, 12, 16, 20, 24, 28, 32, 40, 48)
STANDARD_COMPARISON_BRANCHES = {
    # (ROVir case, ROVir branch) -> standard branch directory name.
    ("normal", "wavelet"): "wavelet_selected_vcc24",
    ("normal", "fista_r0"): "fista_r0",
    ("native_r3x3", "fista_r0"): "fista_r0",
    ("native_r3x3", "wavelet"): "wavelet",
    ("native_r3x2", "fista_r0"): "fista_r0",
}
SOURCE_CONSISTENCY_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return the current UTC time in ISO-8601 form.

    Returns:
        Timestamp text.
    """
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON object.

    Args:
        path: JSON file.

    Returns:
        Parsed mapping.

    Raises:
        ValueError: If the file does not contain a JSON object.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one sorted, indented JSON object.

    Args:
        path: Destination file.
        payload: JSON-compatible mapping.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _file_identity(path: Path, *, include_hash: bool) -> dict[str, Any]:
    """Record path, size, mtime, and optionally SHA-256 of one source file.

    Args:
        path: Existing file.
        include_hash: Whether to hash the file contents.

    Returns:
        JSON-compatible identity record.
    """
    resolved = Path(path).expanduser().resolve()
    status = resolved.stat()
    record: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(status.st_size),
        "mtime_ns": int(status.st_mtime_ns),
    }
    if include_hash:
        record["sha256"] = sha256_file(resolved)
    return record


def _cfl_array(path: str | Path, ndim: int) -> np.ndarray:
    """Open a CFL and drop trailing singleton dimensions beyond ``ndim``.

    Args:
        path: BART basename.
        ndim: Number of leading dimensions to keep.

    Returns:
        Read-only memory-mapped view with exactly ``ndim`` dimensions.

    Raises:
        ValueError: If a dropped trailing dimension is not singleton.
    """
    values = open_cfl(path)
    shape = tuple(values.shape)
    if len(shape) < ndim:
        shape = shape + (1,) * (ndim - len(shape))
    if any(size != 1 for size in shape[ndim:]):
        raise ValueError(f"{path} has nonsingleton dimensions beyond {ndim}: {shape}.")
    return values.reshape(shape[:ndim], order="F")


def _remove_cfl(base: Path) -> None:
    """Delete one CFL pair if present.

    Args:
        base: BART basename.
    """
    for suffix in (".hdr", ".cfl"):
        candidate = base.with_suffix(suffix)
        if candidate.exists() or candidate.is_symlink():
            candidate.unlink()


def _rename_cfl(source: Path, destination: Path) -> None:
    """Rename one CFL pair, replacing nothing.

    Args:
        source: Existing BART basename.
        destination: New BART basename.

    Raises:
        FileExistsError: If the destination pair already exists.
    """
    for suffix in (".hdr", ".cfl"):
        if destination.with_suffix(suffix).exists():
            raise FileExistsError(destination.with_suffix(suffix))
    for suffix in (".hdr", ".cfl"):
        source.with_suffix(suffix).replace(destination.with_suffix(suffix))


def _cfl_exists(base: Path) -> bool:
    """Report whether both files of one CFL pair exist.

    Args:
        base: BART basename.

    Returns:
        True when header and payload are present.
    """
    return base.with_suffix(".hdr").is_file() and base.with_suffix(".cfl").is_file()


def run_logged(command: Sequence[str], log_path: Path) -> None:
    """Run one external command and append its output to a log.

    Args:
        command: Argument vector.
        log_path: Log file receiving the command line, stdout, and stderr.

    Raises:
        subprocess.CalledProcessError: If the command fails.

    Side Effects:
        Launches the external program and appends to ``log_path``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] $ {shlex.join(command)}\n")
        log.flush()
        print(f"$ {shlex.join(command)}", flush=True)
        subprocess.run(list(command), check=True, stdout=log, stderr=subprocess.STDOUT)


def bart_to(
    destination: Path,
    build: Callable[[Path], Sequence[str]],
    log_path: Path,
) -> str:
    """Run one BART command into a temporary basename, then rename it.

    Args:
        destination: Final BART output basename.
        build: Function mapping an output basename to the argument vector.
        log_path: Command log.

    Returns:
        Command text with the final output path (for provenance records).

    Side Effects:
        Launches BART and installs the output pair at ``destination``.
    """
    temporary = destination.with_name(f".{destination.name}-partial")
    _remove_cfl(temporary)
    run_logged(build(temporary), log_path)
    _rename_cfl(temporary, destination)
    return shlex.join(build(destination))


# ---------------------------------------------------------------------------
# Standard-tree binding
# ---------------------------------------------------------------------------


def load_standard_source(
    standard_root: str | Path,
    twix: str | Path,
    sequence: str | Path,
) -> dict[str, Any]:
    """Bind the GRE ROVir run to one completed standard-PCA ``vccN`` tree.

    Args:
        standard_root: Standard variant root containing ``normal/bart_inputs``.
        twix: Measured Wave-GRE TWIX file.
        sequence: Matching Pulseq sequence file.

    Returns:
        JSON-compatible binding with geometry, echoes, PSF paths, sampling
        hash, coil-compression basis path, whitening policy, and CSM crop.

    Raises:
        FileNotFoundError: If required standard artifacts are absent.
        ValueError: If the TWIX or sequence differs from the recorded source.
    """
    root = Path(standard_root).expanduser().resolve()
    inputs = root / "normal" / "bart_inputs"
    manifest_path = inputs / "manifest.json"
    manifest = _read_json(manifest_path)
    twix_identity = _file_identity(Path(twix), include_hash=False)
    sequence_identity = _file_identity(Path(sequence), include_hash=True)
    recorded = manifest.get("source", {})
    recorded_twix = recorded.get("twix", {})
    recorded_sequence = recorded.get("sequence", {})
    if int(recorded_twix.get("size_bytes", -1)) != twix_identity["size_bytes"]:
        raise ValueError("TWIX size differs from the standard GRE source record.")
    if (
        recorded_twix.get("path") != twix_identity["path"]
        and int(recorded_twix.get("mtime_ns", -1)) != twix_identity["mtime_ns"]
    ):
        raise ValueError("TWIX path and modification time both differ from the standard record.")
    if recorded_sequence.get("sha256") != sequence_identity["sha256"]:
        raise ValueError("Sequence SHA-256 differs from the standard GRE source record.")
    geometry = manifest.get("geometry", {})
    if geometry.get("case_id") != "native_r3x1":
        raise ValueError("The standard normal GRE manifest must describe native_r3x1.")
    echoes = manifest.get("echoes")
    if not isinstance(echoes, list) or not echoes:
        raise ValueError("The standard normal GRE manifest lists no echoes.")
    for echo in echoes:
        if not _cfl_exists(inputs / str(echo["psf"])):
            raise FileNotFoundError(inputs / str(echo["psf"]))

    # Follow the standard tree's whitening policy so the comparison is like-for-like.
    whitening_record = manifest.get("noise_prewhitening")
    if whitening_record is None:
        whitening_policy = "legacy source without a whitening record; no whitening applied"
        whitening_applied = False
    else:
        whitening_applied = bool(whitening_record.get("applied"))
        whitening_policy = (
            "recorded standard whitening matrix applied to physical coils"
            if whitening_applied
            else "standard source recorded a whitening no-op"
        )

    ecalib_record = root / "normal" / "bart_output" / "ecalib_command.txt"
    crop = 0.6
    crop_source = "normal launcher default"
    if ecalib_record.is_file():
        match = re.search(r"\s-c\s+([0-9.eE+-]+)\s", " " + ecalib_record.read_text() + " ")
        if match:
            crop = float(match.group(1))
            crop_source = str(ecalib_record)
    return {
        "standard_root": str(root),
        "normal_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "twix": twix_identity,
        "sequence": sequence_identity,
        "twix_recorded_path": recorded_twix.get("path"),
        "geometry": dict(geometry),
        "echoes": [dict(echo) for echo in echoes],
        "sampling_logical_sha256": manifest.get("sampling", {}).get("logical_sha256"),
        "physical_coils": int(manifest["coil_compression"]["physical_coils"]),
        "standard_virtual_coils": int(manifest["coil_compression"]["virtual_coils"]),
        "coil_compression_basis": str(inputs / "coil_compression_basis.npy"),
        "noise_prewhitening": {"applied": whitening_applied, "policy": whitening_policy},
        "ecalib_crop": crop,
        "ecalib_crop_source": crop_source,
        "wavelet_selection": manifest.get("wavelet_selection"),
        "output_orientation": manifest.get("output_orientation"),
        "dimension_order": manifest.get("dimension_order"),
    }


def _standard_whitening_matrix(binding: Mapping[str, Any]) -> np.ndarray | None:
    """Load the standard tree's whitening matrix when one was applied.

    Args:
        binding: Result of :func:`load_standard_source`.

    Returns:
        Coil-last whitening matrix or ``None``.
    """
    if not binding["noise_prewhitening"]["applied"]:
        return None
    manifest_path = Path(binding["normal_manifest"]["path"])
    return load_recorded_whitening_matrix(manifest_path.parent, _read_json(manifest_path))


# ---------------------------------------------------------------------------
# Stage 1: physical-coil export and calibration views
# ---------------------------------------------------------------------------


def export_physical_inputs(
    twix: str | Path,
    sequence: str | Path,
    standard_root: str | Path,
    output_root: str | Path,
    *,
    bart: str = "bart",
) -> dict[str, Any]:
    """Export physical-coil ACS and Wave k-space and the ACS RSS views.

    Args:
        twix: Measured Wave-GRE TWIX file.
        sequence: Matching Pulseq sequence file.
        standard_root: Completed standard ``vccN`` tree for the same scan.
        output_root: User-approved GRE ROVir output root.
        bart: BART executable.

    Returns:
        Physical-input manifest.

    Raises:
        ValueError: If the TWIX disagrees with the standard tree, or the
            exported physical data do not reproduce the standard PCA data.

    Side Effects:
        Reads the TWIX, writes ``physical_inputs/`` (about 7.5 GB per echo for
        52 coils plus 4 GB of ACS arrays), and launches ``bart fft``/``rss``.
    """
    import torch

    root = Path(output_root).expanduser().resolve()
    destination = root / PHYSICAL_DIRECTORY
    manifest_path = destination / "manifest.json"
    binding = load_standard_source(standard_root, twix, sequence)
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        same = (
            existing.get("source", {}).get("normal_manifest") == binding["normal_manifest"]
            and existing.get("source", {}).get("sequence", {}).get("sha256")
            == binding["sequence"]["sha256"]
            and existing.get("source", {}).get("twix", {}).get("size_bytes")
            == binding["twix"]["size_bytes"]
        )
        if not same:
            raise ValueError("Existing physical inputs were exported from another source.")
        for name in existing["cfl"]:
            read_shape(destination / "bart" / name)
        print(f"Reusing physical-coil inputs: {destination}", flush=True)
        return existing
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Incomplete physical-input directory exists: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{PHYSICAL_DIRECTORY}-", dir=root))
    log_path = root / LOG_DIRECTORY / "physical_inputs.log"
    try:
        bart_dir = staging / "bart"
        bart_dir.mkdir()
        native = load_wave_gre_helpers()
        _, cfg, _, _ = validate_gre_sequence(native, Path(sequence).expanduser().resolve())
        matrix = (int(cfg["Nx"]), int(cfg["Ny"]), int(cfg["Nz"]))
        if list(matrix) != list(binding["geometry"]["matrix_ro_lin_par"]):
            raise ValueError("Sequence matrix differs from the standard GRE geometry.")
        fov_mm = tuple(float(value) * 1000.0 for value in np.asarray(cfg["FOVxyz_m"]))
        echo_count = int(cfg["Necho"])
        if echo_count != len(binding["echoes"]):
            raise ValueError("Sequence echo count differs from the standard GRE tree.")
        measurements: list[Any] = []
        source_mask, twix_metadata = inspect_gre_twix(
            Path(twix).expanduser().resolve(),
            expected_echo_times_s=tuple(float(value) for value in cfg["TE_s"]),
            expected_matrix_ro_lin_par=matrix,
            expected_fov_mm_ro_lin_par=fov_mm,
            measurement_sink=measurements,
        )
        if twix_metadata["sampling"]["logical_sha256"] != binding["sampling_logical_sha256"]:
            raise ValueError("TWIX sampling differs from the standard GRE sampling record.")
        measurement = measurements[int(twix_metadata["measurement_index"])]
        nacs = int(cfg["Nacs"])
        reference_shape = refscan_shape(
            measurement.refscan, ncalib=int(cfg["Ncalib1"]), nacs=nacs, nsets=int(cfg["Nsets"])
        )
        physical_coils = int(reference_shape[-1])
        if physical_coils != binding["physical_coils"]:
            raise ValueError("Physical-coil count differs from the standard GRE tree.")
        whitening = _standard_whitening_matrix(binding)

        # Set-4 integrated ACS, readout oversampling removed, embedded at the grid centre.
        packed = native.remove_readout_oversampling_kspace(
            read_refscan_set(measurement.refscan, int(cfg["ACSSetID"]), lin_count=nacs, par_count=nacs),
            int(cfg["os_factor"]),
            axis=0,
        )
        packed = apply_noise_prewhitening_coillast(packed, whitening)
        packed = np.asarray(packed.cpu().numpy() if hasattr(packed, "cpu") else packed, dtype=np.complex64)
        if packed.shape != (matrix[0], nacs, nacs, physical_coils):
            raise ValueError(f"Unexpected physical ACS shape {packed.shape}.")
        lin_start = matrix[1] // 2 - nacs // 2
        par_start = matrix[2] // 2 - nacs // 2
        acs = create_cfl(bart_dir / "acs_kspace", (*matrix, physical_coils))
        acs[...] = 0
        acs[:, lin_start : lin_start + nacs, par_start : par_start + nacs, :] = packed
        acs.flush()
        del acs

        # Consistency: physical ACS projected by the standard PCA basis must
        # reproduce the standard kspace_calib.
        basis = np.load(binding["coil_compression_basis"])
        standard_calib = _cfl_array(
            Path(binding["standard_root"]) / "normal" / "bart_inputs" / "kspace_calib", 4
        )[:, lin_start : lin_start + nacs, par_start : par_start + nacs, :]
        reproduced = packed @ basis
        acs_error = float(
            np.linalg.norm(reproduced - standard_calib) / np.linalg.norm(standard_calib)
        )
        if acs_error > SOURCE_CONSISTENCY_TOLERANCE:
            raise ValueError(f"Physical ACS does not reproduce the standard ACS (rel. error {acs_error:g}).")
        del packed, reproduced, standard_calib

        # Physical-coil Wave k-space: identity (or whitening) instead of PCA.
        image_matrix = (
            np.eye(physical_coils, dtype=np.complex64) if whitening is None else np.asarray(whitening, dtype=np.complex64)
        )
        image = load_compressed_image(
            measurement.image,
            image_matrix,
            output_shape=(EXTENDED_READOUT, matrix[1], matrix[2], echo_count),
            sampling_mask=source_mask,
        )
        if not torch.isfinite(image).all():
            raise ValueError("Physical-coil Wave k-space contains nonfinite values.")
        echo_records = []
        for echo_index in range(echo_count):
            label = f"echo-{echo_index + 1:02d}"
            values = image[:, :, :, echo_index, :].numpy()
            output = create_cfl(bart_dir / f"wave_kspace_{label}", (*values.shape, 1))
            output[..., 0] = values
            output.flush()
            del output
            # Spot-check three partitions against the standard PCA k-space.
            standard_wave = _cfl_array(
                Path(binding["standard_root"]) / "normal" / "bart_inputs" / str(binding["echoes"][echo_index]["wave_kspace"]),
                4,
            )
            errors = []
            for par in (matrix[2] // 4, matrix[2] // 2, 3 * matrix[2] // 4):
                reference = np.asarray(standard_wave[:, :, par, :])
                errors.append(
                    float(np.linalg.norm(values[:, :, par, :] @ basis - reference) / np.linalg.norm(reference))
                )
            if max(errors) > SOURCE_CONSISTENCY_TOLERANCE:
                raise ValueError(f"Physical {label} k-space does not reproduce the standard data: {errors}.")
            echo_records.append(
                {
                    "echo": echo_index + 1,
                    "wave_kspace": f"wave_kspace_{label}",
                    "shape": [*values.shape, 1],
                    "standard_pca_reproduction_relative_error_by_par": errors,
                }
            )
        del image

        # Low-resolution physical-coil calibration images and their RSS.
        run_logged([bart, "fft", "-iu", "7", str(bart_dir / "acs_kspace"), str(bart_dir / "acs_coil_images")], log_path)
        run_logged([bart, "rss", "8", str(bart_dir / "acs_coil_images"), str(bart_dir / "acs_rss")], log_path)
        rss = np.abs(np.asarray(_cfl_array(bart_dir / "acs_rss", 3)))
        figures = write_calibration_views(
            rss, staging / "figures", voxel_mm=binding["geometry"]["voxel_mm_ro_lin_par"]
        )

        manifest = {
            "format_version": FORMAT_VERSION,
            "status": "gre_rovir_physical_inputs_ready",
            "created_at_utc": _utc_now(),
            "source": {
                **{key: binding[key] for key in ("normal_manifest", "twix", "sequence", "standard_root")},
                "twix_recorded_path_in_standard_tree": binding["twix_recorded_path"],
            },
            "matrix_ro_lin_par": list(matrix),
            "voxel_mm_ro_lin_par": list(binding["geometry"]["voxel_mm_ro_lin_par"]),
            "physical_coils": physical_coils,
            "echo_count": echo_count,
            "noise_prewhitening": binding["noise_prewhitening"],
            "acs": {
                "set_index": int(cfg["ACSSetID"]),
                "width_lin_par": nacs,
                "embedding_start_lin_par": [lin_start, par_start],
                "standard_pca_reproduction_relative_error": acs_error,
            },
            "echoes": echo_records,
            "cfl": ["acs_kspace", "acs_coil_images", "acs_rss", *[e["wave_kspace"] for e in echo_records]],
            "commands": {
                "ifft": f"{bart} fft -iu 7 acs_kspace acs_coil_images",
                "rss": f"{bart} rss 8 acs_coil_images acs_rss",
            },
            "figures": [str(Path("figures") / path.name) for path in figures],
            "axis_convention": "native BART array: RO, LIN (0 = anterior), PAR (0 = inferior)",
        }
        _write_json(staging / "manifest.json", manifest)
        if destination.exists():
            destination.rmdir()
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _index_axes(ax: Any, width: int, height: int, step_x: int, step_y: int) -> None:
    """Draw native-index ticks and a faint grid on one image axis.

    Args:
        ax: Matplotlib axis.
        width: Horizontal array size.
        height: Vertical array size.
        step_x: Horizontal tick spacing.
        step_y: Vertical tick spacing.
    """
    ax.set_xticks(range(0, width, step_x))
    ax.set_yticks(range(0, height, step_y))
    ax.grid(color="yellow", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)


def _show_plane(
    ax: Any,
    volume: np.ndarray,
    plane: str,
    index: int,
    voxel_mm: Sequence[float],
    vmax: float,
    outline: np.ndarray | None = None,
    title: str | None = None,
    cmap: str = "gray",
    vmin: float = 0.0,
) -> None:
    """Show one native-orientation plane with index ticks and an optional outline.

    Args:
        ax: Matplotlib axis.
        volume: Native ``(RO, LIN, PAR)`` real volume.
        plane: ``sagittal`` (fixed RO), ``axial`` (fixed PAR), or ``coronal``
            (fixed LIN).
        index: Fixed-axis index.
        voxel_mm: Native voxel size used for the display aspect ratio.
        vmax: Display maximum.
        outline: Optional boolean null-region mask drawn as a red contour.
        title: Optional panel title.
        cmap: Matplotlib colormap.
        vmin: Display minimum.
    """
    # Sagittal: LIN horizontal (anterior left), PAR vertical (inferior bottom).
    # Axial: RO horizontal, LIN vertical (anterior top). Coronal: RO vs PAR.
    if plane == "sagittal":
        image, mask = volume[index].T, None if outline is None else outline[index].T
        aspect, origin, labels = voxel_mm[2] / voxel_mm[1], "lower", ("LIN", "PAR")
    elif plane == "axial":
        image, mask = volume[:, :, index].T, None if outline is None else outline[:, :, index].T
        aspect, origin, labels = voxel_mm[1] / voxel_mm[0], "upper", ("RO", "LIN")
    elif plane == "coronal":
        image, mask = volume[:, index, :].T, None if outline is None else outline[:, index, :].T
        aspect, origin, labels = voxel_mm[2] / voxel_mm[0], "lower", ("RO", "PAR")
    else:
        raise ValueError(f"Unknown plane {plane!r}.")
    ax.imshow(image, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax, aspect=aspect, interpolation="nearest")
    if mask is not None and mask.any() and not mask.all():
        # No contour origin: Z[0, 0] then sits at data (0, 0), the same pixel
        # centre imshow uses for either origin, so the outline never flips.
        ax.contour(mask.astype(float), levels=[0.5], colors="red", linewidths=1.0)
    _index_axes(ax, image.shape[1], image.shape[0], 25 if image.shape[1] > 100 else 6, 25 if image.shape[0] > 100 else 6)
    ax.set_xlabel(labels[0], fontsize=8)
    ax.set_ylabel(labels[1], fontsize=8)
    ax.set_title(title or f"{plane} {index}", fontsize=9)


def _plane_indices(shape: Sequence[int]) -> dict[str, list[int]]:
    """Choose review planes spread across the native grid.

    Args:
        shape: Native ``(RO, LIN, PAR)`` grid.

    Returns:
        Indices for sagittal, axial, and coronal review planes.
    """
    ro, lin, par = (int(value) for value in shape)
    return {
        "sagittal": [int(round(f * (ro - 1))) for f in (0.3, 0.4, 0.5, 0.6, 0.7)],
        "axial": [int(round(f * (par - 1))) for f in (0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.8)],
        "coronal": [int(round(f * (lin - 1))) for f in (0.08, 0.16, 0.28, 0.4)],
    }


def write_calibration_views(
    rss: np.ndarray,
    output_directory: Path,
    *,
    voxel_mm: Sequence[float] = (0.88, 0.88, 2.5),
    outline: np.ndarray | None = None,
    stem: str = "acs_rss",
) -> list[Path]:
    """Write native-index RSS views used to place ROIs.

    Args:
        rss: Native ``(RO, LIN, PAR)`` RSS volume.
        output_directory: Figure directory.
        voxel_mm: Native voxel size for aspect ratios.
        outline: Optional null-region mask drawn in red.
        stem: Output filename stem.

    Returns:
        Written PNG paths (one per plane orientation).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    vmax = float(np.percentile(rss, 99.5)) * 0.4
    paths = []
    for plane, indices in _plane_indices(rss.shape).items():
        fig, axes = plt.subplots(1, len(indices), figsize=(4.2 * len(indices), 4.2))
        for ax, index in zip(np.atleast_1d(axes), indices, strict=True):
            _show_plane(ax, rss, plane, index, voxel_mm, vmax, outline)
        fig.suptitle(
            f"{stem}: native BART indices (LIN 0 = anterior, PAR 0 = inferior); gain x2.5"
            + ("; red = null region" if outline is not None else ""),
            fontsize=10,
        )
        fig.tight_layout()
        path = output_directory / f"{stem}_{plane}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Stage 2: ROI masks, BART ROVir transform, and energy curves
# ---------------------------------------------------------------------------


def _standard_reference_image(standard_root: Path) -> tuple[np.ndarray | None, str | None]:
    """Load one standard reconstruction magnitude in native order for overlays.

    Args:
        standard_root: Standard ``vccN`` tree.

    Returns:
        Magnitude volume and its path, or ``(None, None)`` when absent.
    """
    candidates = sorted((standard_root / "normal" / "nifti").glob("*/quantitative_complex/echo-01_complex.npy"))
    if not candidates:
        return None, None
    return np.abs(np.load(candidates[0])), str(candidates[0])


def prepare_roi(
    output_root: str | Path,
    *,
    boxes: Sequence[str | Mapping[str, Any]] = (),
    ellipsoids: Sequence[str | Mapping[str, Any]] = (),
    bart: str = "bart",
    voxel_chunk: int = 65536,
) -> dict[str, Any]:
    """Create masks, run ``bart rovir``, and write ROI review diagnostics.

    Args:
        output_root: GRE ROVir output root with completed physical inputs.
        boxes: Null boxes (``ro=a:b,lin=c:d,par=e:f`` or JSON mappings).
        ellipsoids: Null spheres/ellipsoids (see ``rovir_regions``).
        bart: BART executable.
        voxel_chunk: Voxels per correlation/energy accumulation block.

    Returns:
        ROI manifest including the region identifier and energy curve.

    Raises:
        ValueError: If the union is invalid, the interference correlation is
            singular, or the BART transform is not orthonormal.

    Side Effects:
        Writes ``roi/<region_id>/`` (masks, two masked coil-image arrays of
        about 1.9 GB each for 52 coils, the transform, and figures) and
        launches ``bart rovir``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_root).expanduser().resolve()
    physical_dir = root / PHYSICAL_DIRECTORY
    physical = _read_json(physical_dir / "manifest.json")
    shape = tuple(physical["matrix_ro_lin_par"])
    voxel_mm = tuple(physical["voxel_mm_ro_lin_par"])
    null_mask, region = null_region_union(shape, voxel_mm, boxes=boxes, ellipsoids=ellipsoids)
    region_id = region["region_id"]
    destination = root / ROI_DIRECTORY / region_id
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("region", {}).get("negative_union_sha256") != region["negative_union_sha256"]:
            raise ValueError(f"Existing ROI {region_id} has a different union.")
        print(f"Reusing ROI {region_id}: {destination}", flush=True)
        return existing
    if destination.exists():
        raise FileExistsError(f"ROI directory without a manifest exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{region_id}-", dir=destination.parent))
    log_path = root / LOG_DIRECTORY / f"roi_{region_id}.log"
    try:
        positive = (~null_mask).astype(np.float32)
        negative = null_mask.astype(np.float32)
        for name, mask in (("positive_mask", positive), ("negative_mask", negative)):
            output = create_cfl(staging / "masks" / name, shape)
            output[...] = mask
            output.flush()
            del output

        # Review overlays on the ACS RSS and, if available, the standard image.
        rss = np.abs(np.asarray(_cfl_array(physical_dir / "bart" / "acs_rss", 3)))
        figures = write_calibration_views(rss, staging / "figures", voxel_mm=voxel_mm, outline=null_mask, stem="acs_rss_with_null_region")
        standard_image, standard_path = _standard_reference_image(Path(physical["source"]["standard_root"]))
        if standard_image is not None and standard_image.shape == shape:
            figures += write_calibration_views(
                standard_image / np.percentile(standard_image, 99.5) * np.percentile(rss, 99.5),
                staging / "figures",
                voxel_mm=voxel_mm,
                outline=null_mask,
                stem="standard_image_with_null_region",
            )

        # Region-masked physical-coil calibration images for the BART eigensolver.
        images = _cfl_array(physical_dir / "bart" / "acs_coil_images", 4)
        correlation = region_correlation_diagnostics(images, positive, negative, coil_axis=3, voxel_chunk=voxel_chunk)
        positive_images = create_cfl(staging / "inputs" / "positive_signal_images", images.shape)
        negative_images = create_cfl(staging / "inputs" / "negative_interference_images", images.shape)
        for start in range(0, shape[0], 8):
            stop = min(start + 8, shape[0])
            block = np.asarray(images[start:stop])
            positive_images[start:stop] = block * positive[start:stop, ..., np.newaxis]
            negative_images[start:stop] = block * negative[start:stop, ..., np.newaxis]
        positive_images.flush()
        negative_images.flush()
        del positive_images, negative_images
        (staging / "transform").mkdir()
        run_logged(
            [
                bart,
                "rovir",
                str(staging / "inputs" / "positive_signal_images"),
                str(staging / "inputs" / "negative_interference_images"),
                str(staging / "transform" / "rovir_transform"),
            ],
            log_path,
        )
        transform = np.asarray(open_cfl(staging / "transform" / "rovir_transform"))
        transform_validation = validate_rovir_transform(transform)
        counts = [count for count in ENERGY_CURVE_COUNTS if count <= physical["physical_coils"]]
        if physical["physical_coils"] not in counts:
            counts.append(int(physical["physical_coils"]))
        curve = region_energy_curve(images, transform, positive, negative, counts, coil_axis=3, voxel_chunk=voxel_chunk)

        # Energy-curve table and plot for choosing the retained ROVir coil count.
        with (staging / "figures" / "rovir_energy_curve.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["virtual_coils", "signal_retention_fraction", "null_region_remaining_fraction"])
            for entry in curve["channel_counts"]:
                writer.writerow([entry["virtual_coils"], entry["signal_retention_fraction"], entry["interference_remaining_fraction"]])
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ncc = [entry["virtual_coils"] for entry in curve["channel_counts"]]
        ax.plot(ncc, [entry["signal_retention_fraction"] for entry in curve["channel_counts"]], "o-", label="kept: outside null region")
        ax.plot(ncc, [entry["interference_remaining_fraction"] for entry in curve["channel_counts"]], "s-", label="remaining: null region")
        ax.set_xlabel("retained ROVir virtual coils")
        ax.set_ylabel("fraction of physical-coil ACS energy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f"{region_id}: ROVir energy (ACS images)", fontsize=10)
        fig.tight_layout()
        fig.savefig(staging / "figures" / "rovir_energy_curve.png", dpi=100)
        plt.close(fig)
        figures.append(staging / "figures" / "rovir_energy_curve.png")

        manifest = {
            "format_version": FORMAT_VERSION,
            "status": "gre_rovir_roi_transform_ready",
            "created_at_utc": _utc_now(),
            "physical_inputs_manifest_sha256": sha256_file(physical_dir / "manifest.json"),
            "region": region,
            "masks": {
                "positive": "masks/positive_mask (exact complement of the null region)",
                "negative": "masks/negative_mask (null-region union)",
            },
            "correlation_diagnostics": correlation,
            "transform": {
                "path": "transform/rovir_transform",
                "command": f"{bart} rovir inputs/positive_signal_images inputs/negative_interference_images transform/rovir_transform",
                "validation": transform_validation,
                "record": cfl_record(staging / "transform" / "rovir_transform"),
            },
            "energy_curve": curve,
            "standard_overlay_source": standard_path,
            "figures": [str(Path("figures") / path.name) for path in figures],
        }
        _write_json(staging / "manifest.json", manifest)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _read_json(manifest_path)


# ---------------------------------------------------------------------------
# Stage 3: ROVir-projected BART inputs and CSM
# ---------------------------------------------------------------------------


def variant_root(output_root: str | Path, region_id: str, virtual_coils: int) -> Path:
    """Return the reconstruction tree for one ROI and coil count.

    Args:
        output_root: GRE ROVir output root.
        region_id: ROI identifier from :func:`prepare_roi`.
        virtual_coils: Retained ROVir virtual coils.

    Returns:
        ``<output_root>/rovir_<region_id>_vcc<N>``.
    """
    return Path(output_root).expanduser().resolve() / f"rovir_{region_id}_vcc{int(virtual_coils)}"


def _copy_cfl(source: Path, destination: Path) -> None:
    """Copy one CFL pair through a temporary name.

    Args:
        source: Existing BART basename.
        destination: Destination basename.
    """
    temporary = destination.with_name(f".{destination.name}-partial")
    _remove_cfl(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".hdr", ".cfl"):
        shutil.copyfile(source.with_suffix(suffix), temporary.with_suffix(suffix))
    _rename_cfl(temporary, destination)


def _cfl_norm(base: Path, *, chunk: int = 50) -> float:
    """Compute the L2 norm of one CFL in bounded memory.

    Args:
        base: BART basename.
        chunk: Leading-axis planes per block.

    Returns:
        Euclidean norm of all samples.
    """
    values = open_cfl(base)
    total = 0.0
    for start in range(0, values.shape[0], chunk):
        block = np.asarray(values[start : start + chunk])
        total += float(np.vdot(block, block).real)
    return math.sqrt(total)


def _check_projection(
    physical_acs: Path, projected_acs: Path, transform: np.ndarray, virtual_coils: int, readout: int
) -> float:
    """Verify one ``ccapply`` output plane against ``X @ conj(T[:, :N])``.

    Args:
        physical_acs: Physical-coil ACS basename.
        projected_acs: ROVir-projected ACS basename.
        transform: Full square ROVir transform (physical rows, virtual columns).
        virtual_coils: Retained virtual coils.
        readout: Readout plane to compare.

    Returns:
        Relative error of the BART projection.

    Raises:
        ValueError: If the projection differs from the expected convention.
    """
    source = np.asarray(_cfl_array(physical_acs, 4)[readout])
    target = np.asarray(_cfl_array(projected_acs, 4)[readout])
    expected = source @ transform[:, :virtual_coils].conj()
    error = float(np.linalg.norm(target - expected) / max(np.linalg.norm(expected), 1e-30))
    if error > 1e-4:
        raise ValueError(f"BART ccapply output differs from X @ conj(T) (rel. error {error:g}).")
    return error


def prepare_variant(
    output_root: str | Path,
    region_id: str,
    *,
    virtual_coils: int = DEFAULT_ROVIR_VIRTUAL_COILS,
    retro_cases: Sequence[str] = DEFAULT_RETRO_CASES,
    ecalib_crop: float | None = None,
    bart: str = "bart",
) -> dict[str, Any]:
    """Project physical data with ROVir, estimate the CSM, and prepare cases.

    Args:
        output_root: GRE ROVir output root.
        region_id: Completed ROI identifier.
        virtual_coils: Leading ROVir virtual coils retained.
        retro_cases: Retrospective same-grid cases to prepare.
        ecalib_crop: ESPIRiT crop; ``None`` inherits the standard tree's crop.
        bart: BART executable.

    Returns:
        Variant manifest.

    Raises:
        ValueError: If the coil count, cases, or projections are invalid.

    Side Effects:
        Writes the variant's ``bart_inputs`` and CSM and launches ``bart
        ccapply`` and ``bart ecalib``.
    """
    root = Path(output_root).expanduser().resolve()
    physical_dir = root / PHYSICAL_DIRECTORY
    physical = _read_json(physical_dir / "manifest.json")
    roi_dir = root / ROI_DIRECTORY / region_id
    roi = _read_json(roi_dir / "manifest.json")
    if not 1 <= int(virtual_coils) <= int(physical["physical_coils"]):
        raise ValueError(f"Virtual coils must lie in [1, {physical['physical_coils']}].")
    unknown = sorted(set(retro_cases) - set(SUPPORTED_RETRO_CASES))
    if unknown:
        raise ValueError(f"Unsupported retrospective GRE ROVir cases: {unknown}.")
    standard_root = Path(physical["source"]["standard_root"])
    standard_normal = _read_json(standard_root / "normal" / "bart_inputs" / "manifest.json")
    binding_crop = ecalib_crop
    if binding_crop is None:
        record = standard_root / "normal" / "bart_output" / "ecalib_command.txt"
        match = re.search(r"\s-c\s+([0-9.eE+-]+)\s", " " + record.read_text() + " ") if record.is_file() else None
        binding_crop = float(match.group(1)) if match else 0.6
    variant = variant_root(root, region_id, virtual_coils)
    log_path = root / LOG_DIRECTORY / f"{variant.name}.log"
    transform_base = roi_dir / "transform" / "rovir_transform"
    transform = rovir_matrix_view(np.asarray(open_cfl(transform_base)))
    rovir_record = {
        "region_id": region_id,
        "roi_manifest_sha256": sha256_file(roi_dir / "manifest.json"),
        "transform": str(transform_base),
        "virtual_coils": int(virtual_coils),
        "physical_coils": int(physical["physical_coils"]),
        "energy_at_selected_count": next(
            (entry for entry in roi["energy_curve"]["channel_counts"] if entry["virtual_coils"] == int(virtual_coils)),
            None,
        ),
        "projection": "bart ccapply -p N (SVD type: no readout FFT; output = X @ conj(T[:, :N]))",
    }

    # Normal (prospective R3x1) inputs: projected k-space, projected ACS, copied PSFs.
    normal_inputs = variant / "normal" / "bart_inputs"
    normal_inputs.mkdir(parents=True, exist_ok=True)
    echo_records = []
    for echo, standard_echo in zip(physical["echoes"], standard_normal["echoes"], strict=True):
        label = f"echo-{int(echo['echo']):02d}"
        wave = normal_inputs / f"wave_kspace_{label}"
        if not _cfl_exists(wave):
            bart_to(
                wave,
                lambda out, e=echo: [bart, "ccapply", "-p", str(int(virtual_coils)), str(physical_dir / "bart" / e["wave_kspace"]), str(transform_base), str(out)],
                log_path,
            )
        psf = normal_inputs / f"psf_{label}"
        if not _cfl_exists(psf):
            _copy_cfl(standard_root / "normal" / "bart_inputs" / str(standard_echo["psf"]), psf)
        echo_records.append(
            {
                **{key: standard_echo[key] for key in ("echo", "eco_counter", "te_s", "shared_calibration_id", "selected_wavelet_lambda")},
                "wave_kspace": wave.name,
                "wave_kspace_shape": list(read_shape(wave)),
                "wave_kspace_norm": _cfl_norm(wave),
                "psf": psf.name,
                "psf_shape": list(read_shape(psf)),
                "psf_source": str(standard_root / "normal" / "bart_inputs" / str(standard_echo["psf"])),
                "psf_sha256": sha256_file(psf.with_suffix(".cfl")),
            }
        )
    calib = normal_inputs / "kspace_calib"
    if not _cfl_exists(calib):
        bart_to(
            calib,
            lambda out: [bart, "ccapply", "-p", str(int(virtual_coils)), str(physical_dir / "bart" / "acs_kspace"), str(transform_base), str(out)],
            log_path,
        )
    projection_error = _check_projection(
        physical_dir / "bart" / "acs_kspace", calib, transform, int(virtual_coils), readout=physical["matrix_ro_lin_par"][0] // 2
    )
    normal_manifest = {
        "format_version": FORMAT_VERSION,
        "status": "gre_rovir_normal_bart_inputs_ready",
        "prepared_at_utc": _utc_now(),
        "geometry": standard_normal["geometry"],
        "echo_count": len(echo_records),
        "echoes": echo_records,
        "kspace_calib": "kspace_calib",
        "kspace_calib_shape": list(read_shape(calib)),
        "wavelet_selection": standard_normal["wavelet_selection"],
        "dimension_order": standard_normal.get("dimension_order"),
        "output_orientation": standard_normal.get("output_orientation"),
        "sampling": standard_normal.get("sampling"),
        "rovir": {**rovir_record, "ccapply_projection_check_relative_error": projection_error},
        "source": physical["source"],
    }
    _write_json(normal_inputs / "manifest.json", normal_manifest)

    # One CSM in the ROVir basis, shared by the normal and retrospective cases.
    normal_output = variant / "normal" / "bart_output"
    normal_output.mkdir(parents=True, exist_ok=True)
    maps = normal_output / "coil_sens"
    ecalib_record = normal_output / "ecalib_command.txt"
    expected_ecalib = shlex.join([bart, "ecalib", "-m", "1", "-c", f"{binding_crop:g}", str(calib), str(maps)])
    if _cfl_exists(maps) and ecalib_record.is_file():
        if ecalib_record.read_text(encoding="utf-8").strip() != expected_ecalib:
            raise ValueError("Existing ROVir CSM was estimated with a different command.")
    else:
        _remove_cfl(maps)
        bart_to(maps, lambda out: [bart, "ecalib", "-m", "1", "-c", f"{binding_crop:g}", str(calib), str(out)], log_path)
        ecalib_record.write_text(expected_ecalib + "\n", encoding="utf-8")

    # Retrospective same-grid cases: mask the projected normal k-space.
    cases = gre_cases(tuple(standard_normal["geometry"]["matrix_ro_lin_par"]), tuple(standard_normal["geometry"]["fov_mm_ro_lin_par"]))
    case_records = []
    for case_id in retro_cases:
        case = cases[case_id]
        mask, sampling = _case_mask(case)
        standard_case_manifest = standard_root / "retro" / case_id / "bart_inputs" / "manifest.json"
        if standard_case_manifest.is_file():
            recorded = _read_json(standard_case_manifest).get("sampling", {}).get("logical_sha256")
            if recorded != sampling["logical_sha256"]:
                raise ValueError(f"{case_id} sampling differs from the standard retrospective case.")
        inputs = variant / "retro" / case_id / "bart_inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        np.save(inputs / "sampling_mask.npy", mask, allow_pickle=False)
        case_echoes = []
        for echo in echo_records:
            label = f"echo-{int(echo['echo']):02d}"
            target = inputs / f"wave_kspace_{label}"
            if not _cfl_exists(target):
                temporary = target.with_name(f".{target.name}-partial")
                _remove_cfl(temporary)
                source = open_cfl(normal_inputs / echo["wave_kspace"])
                output = create_cfl(temporary, source.shape)
                for start in range(0, source.shape[0], 50):
                    stop = min(start + 50, source.shape[0])
                    output[start:stop] = np.asarray(source[start:stop]) * mask.reshape(1, *mask.shape, *([1] * (source.ndim - 3)))
                output.flush()
                del output, source
                _rename_cfl(temporary, target)
            psf = inputs / f"psf_{label}"
            if not _cfl_exists(psf):
                _copy_cfl(normal_inputs / echo["psf"], psf)
            case_echoes.append(
                {
                    **echo,
                    "wave_kspace": target.name,
                    "wave_kspace_shape": list(read_shape(target)),
                    "wave_kspace_norm": _cfl_norm(target),
                    "psf": psf.name,
                }
            )
        case_manifest = {
            "format_version": FORMAT_VERSION,
            "status": "gre_rovir_retro_case_ready",
            "prepared_at_utc": _utc_now(),
            "case": case.to_json(),
            "echo_count": len(case_echoes),
            "echoes": case_echoes,
            "sampling": {**sampling, "path": str(inputs / "sampling_mask.npy")},
            "csm_policy": "reuse the ROVir normal CSM unchanged (same native grid)",
            "wavelet_selection": gre_wavelet_selection_provenance(case_id, len(case_echoes)),
            "dimension_order": standard_normal.get("dimension_order"),
            "output_orientation": standard_normal.get("output_orientation"),
            "rovir": rovir_record,
            "source": physical["source"],
        }
        _write_json(inputs / "manifest.json", case_manifest)
        case_records.append({"case_id": case_id, "manifest": str(inputs / "manifest.json")})

    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "gre_rovir_variant_inputs_ready",
        "updated_at_utc": _utc_now(),
        "variant_root": str(variant),
        "rovir": rovir_record,
        "ecalib_command": expected_ecalib,
        "ecalib_crop": binding_crop,
        "normal_manifest": str(normal_inputs / "manifest.json"),
        "retro_cases": case_records,
    }
    _write_json(variant / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Stage 4: BART Wave reconstruction and NIfTI export
# ---------------------------------------------------------------------------


def _load_converter() -> Any:
    """Import the existing GRE BART-to-NIfTI converter script as a module.

    Returns:
        Loaded ``convert_gre_bart_to_nifti`` module.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "convert_gre_bart_to_nifti.py"
    spec = importlib.util.spec_from_file_location("convert_gre_bart_to_nifti", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nifti_suffix(case_id: str, branch: str, region_id: str, virtual_coils: int) -> str:
    """Build a dataset-independent NIfTI suffix for one ROVir branch.

    Args:
        case_id: ``normal`` or a retrospective case identifier.
        branch: Reconstruction branch name.
        region_id: ROI identifier.
        virtual_coils: Retained ROVir virtual coils.

    Returns:
        Alphanumeric suffix.
    """
    case_text = "Normal" if case_id == "normal" else "".join(part.capitalize() for part in case_id.split("_"))
    branch_text = "FISTAR0" if branch == "fista_r0" else "Wavelet"
    return f"BARTWaveGRERovir{case_text}{branch_text}VCC{int(virtual_coils)}{region_id.replace('_', '')}"


def reconstruct_variant(
    output_root: str | Path,
    region_id: str,
    twix: str | Path,
    sequence: str | Path,
    *,
    virtual_coils: int = DEFAULT_ROVIR_VIRTUAL_COILS,
    branches: Sequence[str] = DEFAULT_BRANCHES,
    gpu: bool = False,
    bart: str = "bart",
) -> list[dict[str, Any]]:
    """Run BART Wave for every prepared case and branch, then export NIfTI.

    Args:
        output_root: GRE ROVir output root.
        region_id: ROI identifier.
        twix: Measured TWIX (used for the NIfTI affine).
        sequence: Matching Pulseq sequence.
        virtual_coils: Retained ROVir virtual coils.
        branches: Subset of ``fista_r0`` and ``wavelet``.
        gpu: Use BART's GPU path.
        bart: BART executable (the Wave command records ``bart`` as in the
            standard launchers).

    Returns:
        One record per reconstructed case and branch.

    Side Effects:
        Launches ``bart wave`` and writes images, command records, and NIfTIs.
    """
    unknown = sorted(set(branches) - set(BRANCH_LAMBDAS))
    if unknown:
        raise ValueError(f"Unknown GRE ROVir branches: {unknown}.")
    variant = variant_root(output_root, region_id, virtual_coils)
    manifest = _read_json(variant / "manifest.json")
    maps = variant / "normal" / "bart_output" / "coil_sens"
    log_path = Path(output_root).expanduser().resolve() / LOG_DIRECTORY / f"{variant.name}.log"
    case_roots = [("normal", variant / "normal")] + [
        (record["case_id"], variant / "retro" / record["case_id"]) for record in manifest["retro_cases"]
    ]
    converter = _load_converter()
    results = []
    for case_id, case_root in case_roots:
        inputs = case_root / "bart_inputs"
        case_manifest = _read_json(inputs / "manifest.json")
        for branch in branches:
            regularization = BRANCH_LAMBDAS[branch]
            images = []
            for echo in case_manifest["echoes"]:
                label = f"echo-{int(echo['echo']):02d}"
                image = case_root / "bart_output" / branch / label / "image_wave"
                record = image.parent / "wave_command.txt"
                command = build_gre_wave_command(
                    maps=maps, psf=inputs / echo["psf"], kspace=inputs / echo["wave_kspace"],
                    output=image, regularization=regularization, gpu=gpu,
                )
                expected = shlex.join(command)
                if _cfl_exists(image) and record.is_file() and record.read_text(encoding="utf-8").strip() == expected:
                    print(f"Reusing {case_id}/{branch}/{label}", flush=True)
                else:
                    image.parent.mkdir(parents=True, exist_ok=True)
                    record.unlink(missing_ok=True)
                    _remove_cfl(image)
                    bart_to(
                        image,
                        lambda out, e=echo: [bart, *build_gre_wave_command(
                            maps=maps, psf=inputs / e["psf"], kspace=inputs / e["wave_kspace"],
                            output=out, regularization=regularization, gpu=gpu,
                        )[1:]],
                        log_path,
                    )
                    record.write_text(expected + "\n", encoding="utf-8")
                images.append(image)
            nifti = case_root / "nifti" / branch
            conversion = nifti / "conversion_manifest.json"
            if not conversion.is_file():
                converter.convert(
                    inputs, images, Path(twix), Path(sequence), nifti,
                    _nifti_suffix(case_id, branch, region_id, virtual_coils),
                )
            results.append({"case_id": case_id, "branch": branch, "regularization": regularization, "nifti": str(nifti)})
    return results


# ---------------------------------------------------------------------------
# Stage 5: standard-versus-ROVir QC
# ---------------------------------------------------------------------------


def comparison_metrics(
    standard: np.ndarray,
    rovir: np.ndarray,
    null_mask: np.ndarray,
    head_mask: np.ndarray,
) -> dict[str, float]:
    """Summarize null-region suppression and inside-head agreement.

    Args:
        standard: Standard-PCA magnitude in native order.
        rovir: ROVir magnitude in native order.
        null_mask: Boolean null (face) region.
        head_mask: Boolean inside-head reference region outside the null region.

    Returns:
        Means, the null-region ROVir/standard ratio, the head median ratio,
        and the head relative difference after median matching.
    """
    head_ratio = float(np.median(rovir[head_mask]) / np.median(standard[head_mask]))
    matched = rovir / head_ratio
    null_standard = float(np.mean(standard[null_mask]))
    null_rovir = float(np.mean(matched[null_mask]))
    return {
        "head_median_ratio_rovir_over_standard": head_ratio,
        "null_region_mean_standard": null_standard,
        "null_region_mean_rovir_head_matched": null_rovir,
        "null_region_rovir_over_standard": null_rovir / null_standard if null_standard > 0 else float("nan"),
        "head_relative_rms_difference": float(
            np.sqrt(np.mean((matched[head_mask] - standard[head_mask]) ** 2)) / np.sqrt(np.mean(standard[head_mask] ** 2))
        ),
    }


def local_intensity_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    support: np.ndarray,
    voxel_mm: Sequence[float],
    *,
    sigma_mm: float = 12.0,
) -> np.ndarray:
    """Estimate a smooth ``numerator / denominator`` shading inside a support.

    ROVir keeps a voxel-dependent fraction of each voxel's coil-sensitivity
    energy, and the ESPIRiT-combined image scales with the square root of that
    fraction. This smooth ratio separates that shading (the SNR-loss map) from
    local artifact and noise differences.

    Args:
        numerator: Magnitude image whose shading is estimated (ROVir).
        denominator: Reference magnitude image (standard PCA).
        support: Boolean voxels used for the normalized convolution.
        voxel_mm: Native voxel size in millimetres.
        sigma_mm: Isotropic Gaussian width in millimetres.

    Returns:
        Smooth positive ratio volume (1 outside the dilated support).
    """
    from scipy import ndimage

    sigma = [sigma_mm / float(step) for step in voxel_mm]
    weight = support.astype(np.float64)
    smooth_weight = ndimage.gaussian_filter(weight, sigma)
    top = ndimage.gaussian_filter(numerator * weight, sigma)
    bottom = ndimage.gaussian_filter(denominator * weight, sigma)
    valid = (smooth_weight > 0.05) & (bottom > 0)
    ratio = np.ones_like(numerator, dtype=np.float64)
    ratio[valid] = top[valid] / bottom[valid]
    return np.maximum(ratio, 1e-6)


def write_comparison_qc(
    output_root: str | Path,
    region_id: str,
    *,
    virtual_coils: int = DEFAULT_ROVIR_VIRTUAL_COILS,
) -> dict[str, Any]:
    """Write standard-versus-ROVir figures and metrics for every matching branch.

    Args:
        output_root: GRE ROVir output root.
        region_id: ROI identifier.
        virtual_coils: Retained ROVir virtual coils.

    Returns:
        QC manifest with per-pair metrics and figure paths.

    Side Effects:
        Writes PNG figures and ``qc/comparison.json`` below the variant root.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_root).expanduser().resolve()
    physical = _read_json(root / PHYSICAL_DIRECTORY / "manifest.json")
    voxel_mm = tuple(physical["voxel_mm_ro_lin_par"])
    standard_root = Path(physical["source"]["standard_root"])
    variant = variant_root(root, region_id, virtual_coils)
    null_mask = np.abs(np.asarray(_cfl_array(root / ROI_DIRECTORY / region_id / "masks" / "negative_mask", 3))) > 0.5
    rss = np.abs(np.asarray(_cfl_array(root / PHYSICAL_DIRECTORY / "bart" / "acs_rss", 3)))
    # Inside-head reference: bright ACS RSS voxels outside the null region.
    head_mask = (rss > 0.25 * np.percentile(rss, 99)) & ~null_mask
    qc_dir = variant / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for (case_id, branch), standard_branch in STANDARD_COMPARISON_BRANCHES.items():
        case_dir = variant / "normal" if case_id == "normal" else variant / "retro" / case_id
        standard_case = standard_root / "normal" if case_id == "normal" else standard_root / "retro" / case_id
        rovir_path = case_dir / "nifti" / branch / "quantitative_complex" / "echo-01_complex.npy"
        standard_path = standard_case / "nifti" / standard_branch / "quantitative_complex" / "echo-01_complex.npy"
        if not rovir_path.is_file() or not standard_path.is_file():
            continue
        standard = np.abs(np.load(standard_path))
        rovir = np.abs(np.load(rovir_path))
        metrics = comparison_metrics(standard, rovir, null_mask, head_mask)
        matched = rovir / metrics["head_median_ratio_rovir_over_standard"]
        # Smooth ROVir/standard shading; its square approximates retained SNR^2.
        shading = local_intensity_ratio(rovir, standard, head_mask, voxel_mm)
        locally_matched = rovir / shading
        relative_shading = shading / np.median(shading[head_mask])
        metrics.update(
            {
                "shading_head_percentiles_5_50_95_relative_to_median": [
                    float(value) for value in np.percentile(relative_shading[head_mask], (5, 50, 95))
                ],
                "shading_head_percentiles_5_50_95_absolute": [
                    float(value) for value in np.percentile(shading[head_mask], (5, 50, 95))
                ],
            }
        )
        vmax = float(np.percentile(standard[head_mask], 99.5))
        planes = [("sagittal", index) for index in _plane_indices(standard.shape)["sagittal"][1:4]] + [
            ("axial", index) for index in _plane_indices(standard.shape)["axial"][1:6]
        ]
        fig, axes = plt.subplots(5, len(planes), figsize=(3.6 * len(planes), 17.5))
        for column, (plane, index) in enumerate(planes):
            _show_plane(axes[0, column], standard, plane, index, voxel_mm, vmax, null_mask, f"standard vcc24 {plane} {index}")
            _show_plane(axes[1, column], matched, plane, index, voxel_mm, vmax, null_mask, f"ROVir vcc{virtual_coils} {plane} {index}")
            _show_plane(
                axes[2, column], shading, plane, index, voxel_mm, 1.2, null_mask,
                f"ROVir/standard shading (0-1.2) {plane} {index}", cmap="viridis",
            )
            _show_plane(
                axes[3, column], locally_matched, plane, index, voxel_mm, vmax, null_mask,
                f"ROVir / shading {plane} {index}",
            )
            _show_plane(
                axes[4, column], locally_matched - standard, plane, index, voxel_mm, 0.25 * vmax, None,
                f"(ROVir/shading) - standard (±25%) {plane} {index}", cmap="RdBu_r", vmin=-0.25 * vmax,
            )
        fig.suptitle(
            f"{case_id} {branch} vs standard {standard_branch}; red = null region {region_id}; "
            f"null-region mean ROVir/standard = {metrics['null_region_rovir_over_standard']:.3f}; "
            "rows: standard | ROVir (head-median matched) | smooth shading | ROVir/shading | difference",
            fontsize=11,
        )
        fig.tight_layout()
        figure = qc_dir / f"compare_{case_id}_{branch}.png"
        fig.savefig(figure, dpi=80)
        plt.close(fig)
        pairs.append(
            {
                "case_id": case_id,
                "rovir_branch": branch,
                "standard_branch": standard_branch,
                "rovir": str(rovir_path),
                "standard": str(standard_path),
                "metrics": metrics,
                "figure": str(figure),
            }
        )
    result = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": _utc_now(),
        "region_id": region_id,
        "virtual_coils": int(virtual_coils),
        "head_mask": "ACS RSS > 25% of its 99th percentile, outside the null region",
        "display_scaling": "ROVir magnitude divided by its head-median ratio to the standard image",
        "pairs": pairs,
    }
    _write_json(qc_dir / "comparison.json", result)
    return result
