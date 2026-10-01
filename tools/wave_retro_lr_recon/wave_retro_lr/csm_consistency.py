"""Orchestrate calibration-only set-4 coil-sensitivity consistency diagnostics.

The workflow is driven by ``scripts/sample_mprage_csm_consistency.sh`` and has
four explicit stages:

* ``prepare`` binds a measured TWIX file and its Pulseq sequence to an accepted
  normal reconstruction root, exports the physical-coil set-4 ACS through the
  existing ROVir-named exporter, and records noise, channel identity, and the
  accepted PCA-12 arrays by hash;
* ``record-calibration`` validates the outputs of the shell command
  ``bart ecalib -m 2 -c 0``, which is a diagnostic only and never feeds Wave
  reconstruction;
* ``roi-template`` exports a canonical-RAS reference image and an empty
  five-label template for manual review; and
* ``diagnose`` computes phase-free projection residuals, conditional
  residual-to-noise ratios, eigenvalue, local-rank, and coherence summaries,
  figures, and a report.

Scientific framing: FLASH set 4 and MPRAGE share the readout gradient,
polarity, sample count, and dwell, so their off-resonance readout
displacement per unit frequency is identical. For an individual isochromat
under a constant readout gradient, off-resonance phase is equivalent to a
readout-direction shift, and Wave does not add EPI-like B0 distortion. Near
metal, however, non-invertible pile-up, intravoxel dephasing, excitation
differences, signal voids, and displaced or mixed coil sensitivities can
still violate the single-map SENSE model. These diagnostics describe
calibration evidence only; they do not select a winner or prove a mechanism.

This module never launches BART or any other process.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shlex
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import csm_consistency_metrics as metrics
from . import csm_consistency_roi as roi
from . import mprage as mprage_module
from . import rovir_feasibility
from . import twix_noise
from .bart_io import bart_base, cfl_record, open_cfl, read_shape, sha256_file

FORMAT_VERSION = 1
ACS_SET_INDEX = 4
EXPECTED_REFSCAN_SETS = 5
TWO_MAP_ECALIB_ARGUMENTS = ("ecalib", "-m", "2", "-c", "0")
ACCEPTED_BASELINE_ECALIB_ARGUMENTS = ("ecalib", "-m", "1", "-c", "0")
# Flags written by scripts/sample_mprage_normal_recon.sh for its FISTA-r0
# branch, after the optional leading -g; other branches differ in -r or output.
ACCEPTED_FISTA_R0_WAVE_FLAGS = ("-w", "-f", "-r", "0", "-i", "100", "-t", "1e-6")
ROI_REFERENCE_NAME = "reference_fista_r0_magnitude_ras.nii.gz"
ROI_TEMPLATE_NAME = "csm_consistency_roi_labels_template.nii.gz"
ROI_INSTRUCTIONS_NAME = "README.txt"
DEFAULT_SNR_KAPPA = 25.0
ROBUSTNESS_SNR_KAPPAS = (10.0, 100.0)
RANK_TOLERANCE = 1e-3
EIGENVALUE_THRESHOLDS = (0.5, 0.8, 0.9, 0.95)
LAMBDA1_LOW_THRESHOLD = 0.8
MAP_REPRODUCTION_LAMBDA1_MIN = 0.9
COHERENCE_LAMBDA1_MIN = 0.5
SUPPORT_MISMATCH_LAMBDA1 = 0.5
LOCAL_RANK_NEIGHBORHOODS = {"ro5": (5, 1, 1), "ro5_lin3_par3": (5, 3, 3)}
LOCAL_RANK_MIN_MEMBERS = 4
PAR_CHUNK = 8
PCA_BASIS_RELATIVE_RESIDUAL_LIMIT = 1e-3
EDGE_CONTROL_SNR_BINS = 10
EDGE_CONTROL_SEED = 0
SAMPLING_NULL_REPEATS = 16
SAMPLING_NULL_SEED = 0
INTERPRETATION_POLICY = (
    "Thresholds are pre-registered descriptive defaults recorded in this "
    "manifest. Metrics do not select a winner, prove a mechanism, or authorize "
    "a reconstruction; interpretation requires visual and scientific review."
)
RNR_POLICY = (
    "RNR uses the empirical background-air coil covariance and is interpretable "
    "only when the matrix-level comparison with the independent noise scan is "
    "compatible; otherwise it is reported as uncalibrated and not used to "
    "support or reject a hypothesis."
)

LAYOUT = {
    "physical_calibration": Path("inputs/physical_calibration/physical_set4_kspace"),
    "noise_covariance": Path("inputs/noise/noise_covariance.npy"),
    "map1_reference": Path("csm/map1/accepted_map1_reference.json"),
    "two_map_maps": Path("csm/map2_uncropped/coil_sens"),
    "two_map_command": Path("csm/map2_uncropped/ecalib_command.txt"),
    "two_map_eigenvalues": Path("csm/eigenvalues/ev_m2_c0"),
    "two_map_input": Path("csm/map2_uncropped/ecalib_input.sha256"),
    "prepare_manifest": Path("manifests/csm_consistency_prepare.json"),
    "calibration_manifest": Path("manifests/two_map_calibration.json"),
    "roi_template_manifest": Path("manifests/roi_template.json"),
    "diagnostics_manifest": Path("manifests/csm_consistency_diagnostics.json"),
    "roi_template_directory": Path("rois/template"),
    "roi_reviewed_labels": Path(
        "rois/reviewed/csm_consistency_roi_labels_reviewed.nii.gz"
    ),
    "calibration_views": Path("diagnostics/calibration_views"),
    "projection_residuals": Path("diagnostics/coil_projection_residuals"),
    "local_rank": Path("diagnostics/local_rank"),
    "csm_coherence": Path("diagnostics/csm_coherence"),
    "roi_overlays": Path("diagnostics/roi_overlays"),
    "report": Path("reports/csm_consistency_report.md"),
    "roi_summary_csv": Path("reports/roi_summary.csv"),
    "bart_version": Path("logs/bart_version.txt"),
    "bart_binary_hash": Path("logs/bart_binary.sha256"),
    "ecalib_log": Path("logs/ecalib_m2_c0.log"),
}
# Files written only by the diagnose stage; any of them without a manifest
# marks an interrupted run.
_DIAGNOSTICS_OWNED_PATHS = (
    *(
        LAYOUT[key]
        for key in (
            "calibration_views",
            "projection_residuals",
            "local_rank",
            "csm_coherence",
            "roi_overlays",
        )
    ),
    LAYOUT["report"],
    LAYOUT["roi_summary_csv"],
)
_ENTRY_POINTS = ("scripts/mprage_csm_consistency.py", "scripts/sample_mprage_csm_consistency.sh")
# Pinned upstream sources whose TWIX-import, coil-compression, NIfTI-geometry,
# and sequence-geometry helpers the workflow calls.
_UPSTREAM_IMPLEMENTATION_FILES = (
    "external/wave-mprage/recon/recon_wave_mprage_from_twix_integrated_nifti.py",
    "external/wave-mprage/recon/utils/coil_compression_kspace.py",
    "external/wave-mprage/recon/utils/nifti_export_twix.py",
    "external/wave-mprage/recon/utils/twix_import.py",
)
# Metrics exported as NIfTI views and the figures written by diagnose; the
# local-rank figure exists only when a native basis has local-rank results.
_NIFTI_METRICS = (
    "lambda1",
    "lambda2",
    "rho1_hann",
    "rho2_hann",
    "map1_reproduction_alpha",
    "coherence_c1",
    "coherence_c2",
)
_FIGURES = {
    "eigenvalues_and_residuals": LAYOUT["projection_residuals"] / "eigenvalues_and_residuals.png",
    "map_components": LAYOUT["projection_residuals"] / "map_components.png",
    "coherence_and_support": LAYOUT["csm_coherence"] / "coherence_and_support.png",
    "local_rank_native": LAYOUT["local_rank"] / "local_rank_native.png",
    "roi_overlay": LAYOUT["roi_overlays"] / "roi_overlay.png",
}
# Fixed output contract of diagnose, independent of any manifest: the
# accepted-grid metrics it always computes, both native coil bases, and the
# ROI and control masks every summary covers.
_ACCEPTED_METRICS = (
    "coherence_c1",
    "coherence_c2",
    "eigenvalue_gap",
    "lambda1",
    "lambda2",
    "log10_snr_hann",
    "map1_reproduction_alpha",
    "map_orthonormality_error",
    "map_switching",
    "rho1_hann",
    "rho1_unapodized",
    "rho2_hann",
    "rho2_unapodized",
    "rnr1_hann",
    "rnr2_hann",
)
_NATIVE_BASES = ("physical", "pca")
_SUMMARY_MASKS = (
    "alias_partner",
    "background_air",
    "edge_control",
    "edge_control_snr_matched",
    "fringe",
    "metal",
    "metal_pileup",
    "metal_void",
    "preserved_anatomy",
)
# BART opens names with these endings as one file instead of a .hdr/.cfl pair.
_BART_SINGLE_FILE_SUFFIXES = (".ra", ".coo", ".shm", ".mem", ".fifo")
_FALSE_FLAGS = {
    "bart_launched_by_python": False,
    "wave_reconstruction_launched": False,
    "soft_sense_used": False,
    "psf_recalibrated": False,
    "refscan_sets_0_to_3_used": False,
    "normal_defaults_changed": False,
    "two_map_csm_used_for_reconstruction": False,
}


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        Timezone-aware ISO-8601 timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def output_path(output_root: str | Path, key: str) -> Path:
    """Resolve one named artifact below a diagnostic output root.

    Args:
        output_root: User-approved diagnostic output root.
        key: Key of :data:`LAYOUT`.

    Returns:
        Absolute artifact path.

    Raises:
        KeyError: If ``key`` is not part of the layout.
    """
    return Path(output_root).expanduser().resolve() / LAYOUT[key]


def accepted_artifact_paths(accepted_root: str | Path) -> dict[str, Path]:
    """List the accepted normal-root artifacts consumed by the diagnostics.

    Args:
        accepted_root: Accepted normal reconstruction root with ``normal/``.

    Returns:
        Absolute paths of the input manifest, calibration k-space, PSF, wave
        k-space, one-map CSM, its ecalib record, and the FISTA-r0 image and
        command record.
    """
    root = Path(accepted_root).expanduser().resolve()
    return {
        "root": root,
        "manifest": root / "normal" / "bart_inputs" / "manifest.json",
        "kspace_calib": root / "normal" / "bart_inputs" / "kspace_calib",
        "psf": root / "normal" / "bart_inputs" / "psf",
        "wave_kspace": root / "normal" / "bart_inputs" / "wave_kspace",
        "coil_sens": root / "normal" / "bart_output" / "coil_sens",
        "ecalib_command": root / "normal" / "bart_output" / "ecalib_command.txt",
        "fista_r0_image": root / "normal" / "bart_output" / "fista_r0" / "image_wave",
        "fista_r0_command": root
        / "normal"
        / "bart_output"
        / "fista_r0"
        / "wave_command.txt",
        "fista_r0_nifti_directory": root / "normal" / "nifti" / "fista_r0",
    }


def two_map_ecalib_argv(
    accepted_root: str | Path, output_root: str | Path
) -> list[str]:
    """Build the reviewed diagnostic two-map ESPIRiT command.

    The shell entry point constructs the same command independently and runs
    it; Python only validates the recorded command against this argument list.

    Args:
        accepted_root: Accepted normal reconstruction root.
        output_root: Diagnostic output root.

    Returns:
        Argument list ``bart ecalib -m 2 -c 0 KSPACE MAPS EIGENVALUES`` with
        resolved absolute paths.
    """
    accepted = accepted_artifact_paths(accepted_root)
    return [
        "bart",
        *TWO_MAP_ECALIB_ARGUMENTS,
        str(accepted["kspace_calib"]),
        str(output_path(output_root, "two_map_maps")),
        str(output_path(output_root, "two_map_eigenvalues")),
    ]


def format_command(argv: Sequence[str]) -> str:
    """Quote an argument list as one POSIX shell command line.

    Args:
        argv: Command arguments.

    Returns:
        Shell-quoted command text.
    """
    return shlex.join([str(value) for value in argv])


def cfl_pair_state(path: str | Path) -> str:
    """Classify a BART CFL pair as absent, partial, or complete.

    Args:
        path: BART basename or either member of the pair.

    Returns:
        ``"absent"``, ``"partial"``, or ``"complete"``.
    """
    base = bart_base(path)
    present = (base.with_suffix(".hdr").is_file(), base.with_suffix(".cfl").is_file())
    if all(present):
        return "complete"
    return "partial" if any(present) else "absent"


def require_complete_cfl_pair(path: str | Path, label: str) -> tuple[int, ...]:
    """Require one complete, size-consistent CFL pair.

    Args:
        path: BART basename or either member of the pair.
        label: Human-readable artifact name for error messages.

    Returns:
        Stored BART dimensions.

    Raises:
        FileNotFoundError: If both members are absent.
        ValueError: If only one member exists or the payload size is wrong.
    """
    state = cfl_pair_state(path)
    if state == "absent":
        raise FileNotFoundError(f"{label} is absent: {bart_base(path)}")
    if state == "partial":
        raise ValueError(
            f"{label} is a partial CFL pair; inspect and move it aside before "
            f"rerunning: {bart_base(path)}"
        )
    return read_shape(path)


def _active_shape(shape: Sequence[int], rank: int, label: str) -> tuple[int, ...]:
    """Return leading dimensions after requiring singleton trailing dimensions.

    Args:
        shape: Stored BART dimensions.
        rank: Number of leading dimensions to keep.
        label: Artifact name for error messages.

    Returns:
        The leading ``rank`` dimensions, padded with ones when shorter.

    Raises:
        ValueError: If a trailing dimension is not singleton.
    """
    padded = tuple(int(value) for value in shape) + (1,) * max(0, rank - len(shape))
    if any(value != 1 for value in padded[rank:]):
        raise ValueError(f"{label} has unexpected trailing dimensions: {tuple(shape)}.")
    return padded[:rank]


def _cfl_view(path: str | Path, rank: int, label: str) -> np.ndarray:
    """Open a CFL pair as a read-only view with ``rank`` active dimensions.

    Args:
        path: BART basename.
        rank: Number of leading dimensions to keep.
        label: Artifact name for error messages.

    Returns:
        Memory-mapped Fortran-ordered view without trailing singleton axes.
    """
    shape = _active_shape(require_complete_cfl_pair(path, label), rank, label)
    return np.reshape(open_cfl(path), shape, order="F")


def implementation_identity() -> dict[str, str]:
    """Hash every source file that can change the diagnostics.

    The diagnostics import most of the ``wave_retro_lr`` package
    transitively, so every module of the package is included, together with
    both entry points and the pinned upstream Wave-MPRAGE helper sources.

    Returns:
        Tool-relative paths of local files and repository-relative paths of
        upstream files, each mapped to its SHA-256 digest.

    Raises:
        FileNotFoundError: If an implementation file is missing.
    """
    tool_root = Path(__file__).resolve().parents[1]
    repository = tool_root.parents[1]
    modules = sorted(
        path.relative_to(tool_root).as_posix() for path in (tool_root / "wave_retro_lr").glob("*.py")
    )
    identity = {name: sha256_file(tool_root / name) for name in (*modules, *_ENTRY_POINTS)}
    for name in _UPSTREAM_IMPLEMENTATION_FILES:
        identity[name] = sha256_file(repository / name)
    return identity


def environment_log_path(output_root: str | Path, environment_log: str | Path | None) -> Path | None:
    """Validate the shell environment log of one stage invocation.

    The shell writes a new, uniquely named log under ``logs/environment`` for
    every invocation, so a log recorded by a manifest is never rewritten.

    Args:
        output_root: Diagnostic output root.
        environment_log: Log written by the shell for this invocation, or
            ``None`` for direct Python calls.

    Returns:
        Resolved log path, or ``None``.

    Raises:
        FileNotFoundError: If the log does not exist.
        ValueError: If the log is not a file directly inside
            ``logs/environment`` of this output root, or its path depends on
            the process.
    """
    if environment_log is None:
        return None
    output = Path(output_root).expanduser().resolve()
    if _process_dependent(str(environment_log)):
        raise ValueError(f"The environment log path depends on the process: {environment_log}")
    path = Path(environment_log).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"The environment log is absent: {path}")
    try:
        relative = path.relative_to(output)
    except ValueError as exc:
        raise ValueError(f"The environment log must lie in {output / 'logs' / 'environment'}: {path}") from exc
    if relative.parts[:2] != ("logs", "environment") or len(relative.parts) != 3:
        raise ValueError(f"The environment log must lie in {output / 'logs' / 'environment'}: {path}")
    return path


def environment_record(output_root: str | Path, environment_log: str | Path | None = None) -> dict[str, Any]:
    """Record the Python environment and the shell log of this invocation.

    Args:
        output_root: Diagnostic output root.
        environment_log: Log validated by :func:`environment_log_path`, or
            ``None`` for direct Python calls.

    Returns:
        JSON-native interpreter, package, host, and log provenance. The log
        record holds the file record, the output-root-relative path, and the
        exact text.

    Raises:
        FileNotFoundError: If the log does not exist.
        ValueError: If the log lies outside ``logs/environment``.
    """
    import scipy

    try:
        import nibabel

        nibabel_version = nibabel.__version__
    except ImportError:  # pragma: no cover - nibabel is a declared dependency
        nibabel_version = None
    output = Path(output_root).expanduser().resolve()
    log_path = environment_log_path(output, environment_log)
    log_record = None
    if log_path is not None:
        log_record = {
            **rovir_feasibility._file_record(log_path),
            "relative_path": log_path.relative_to(output).as_posix(),
            "text": log_path.read_text(encoding="utf-8"),
        }
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "nibabel": nibabel_version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "shell_environment_log": log_record,
    }


def _verify_environment(manifest: Mapping[str, Any], output_root: Path, label: str) -> None:
    """Verify the shell environment log recorded by a manifest, if any.

    Args:
        manifest: Stored stage manifest.
        output_root: Diagnostic output root.
        label: Manifest name for error messages.

    Raises:
        FileNotFoundError: If the recorded log is missing.
        ValueError: If the environment record is malformed, or the recorded
            log lies outside ``logs/environment`` or changed after it was
            recorded.
    """
    log = _mapping(manifest, "environment", label).get("shell_environment_log")
    if log is None:
        return
    relative = Path(str(log.get("relative_path", ""))) if isinstance(log, Mapping) else None
    if (
        relative is None
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("logs", "environment")
        or len(relative.parts) != 3
    ):
        raise ValueError(f"{label} has no valid environment log record.")
    expected = output_root / relative
    _verify_file(log, expected, f"{label} environment log")
    if expected.read_text(encoding="utf-8") != log.get("text"):
        raise ValueError(f"{label} environment log changed after it was recorded: {expected}")


def _definition_number(definitions: Mapping[str, Any], key: str) -> float:
    """Read one scalar numeric Pulseq definition.

    Args:
        definitions: Pulseq ``[DEFINITIONS]`` mapping.
        key: Definition name.

    Returns:
        The definition value as ``float``.

    Raises:
        ValueError: If the definition is absent or not a single number.
    """
    if key not in definitions:
        raise ValueError(f"Sequence definition {key} is absent.")
    value = definitions[key]
    values = np.atleast_1d(np.asarray(value, dtype=float))
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f"Sequence definition {key} must be one finite number: {value!r}")
    return float(values[0])


def sequence_contract(
    definitions: Mapping[str, Any], accepted_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the integrated-refscan set-4 contract of a Pulseq sequence.

    Args:
        definitions: Pulseq ``[DEFINITIONS]`` mapping of the matching sequence.
        accepted_manifest: Accepted normal-input manifest.

    Returns:
        JSON-native record of the verified set-4, readout, and TE definitions.

    Raises:
        ValueError: If set identity, set count, ACS size or counters, or the
            readout sampling disagree with the accepted geometry.
    """
    geometry = accepted_manifest["geometry"]
    nro = int(geometry["logical_matrix_ro_lin_par"][0])
    oversampling = int(geometry["readout_oversampling_factor"])
    nacs = int(accepted_manifest["psf_calibration"]["nacs"])
    expected = {
        "Calibration_ACSSetID": ACS_SET_INDEX,
        "Calibration_RefscanNSets": EXPECTED_REFSCAN_SETS,
        "Calibration_Nacs": nacs,
        "Calibration_ACSLocalStart0": 0,
        "Calibration_ACSLocalStop0": nacs - 1,
        "Calibration_ReadoutSamples": nro * oversampling,
        "ReadoutOversamplingFactor": oversampling,
    }
    observed = {key: _definition_number(definitions, key) for key in expected}
    failures = [
        f"{key}={observed[key]:g} (expected {value})"
        for key, value in expected.items()
        if observed[key] != value
    ]
    if failures:
        raise ValueError("Sequence set-4 contract failed: " + "; ".join(failures))
    duration = _definition_number(definitions, "Calibration_ReadoutDuration")
    samples = observed["Calibration_ReadoutSamples"]
    return {
        "verified": {key: int(value) for key, value in observed.items()},
        "calibration_te_s": _definition_number(definitions, "Calibration_TE"),
        "mprage_te_s": _definition_number(definitions, "TE"),
        "calibration_readout_duration_s": duration,
        "calibration_dwell_s": duration / samples,
        "readout_axis": str(definitions.get("ReadoutAxis", "")),
        "calibration_readout_axis": str(definitions.get("Calibration_ReadoutAxis", "")),
    }


def _same_file(left: str | Path, right: str | Path) -> bool:
    """Return whether two paths name the same existing file.

    Device and inode are compared rather than the spelling of the path, so
    symlink and automount aliases of one file compare equal.

    Args:
        left: One path.
        right: The other path.

    Returns:
        True when both paths exist and name the same file.
    """
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _process_dependent(path: str) -> bool:
    """Return whether an absolute path resolves through /proc or /dev.

    Paths such as ``/proc/self/cwd/...`` or ``/dev/fd/N/...`` name different
    files in different processes. The path is resolved component by component
    like the kernel, following every symbolic link, and is process dependent
    when any step lies on the file system mounted at ``/proc`` or ``/dev``.
    Every spelling is therefore caught, including ``//proc``, ``/./proc``,
    ``/../proc``, and links that point into ``/proc``.

    Args:
        path: Absolute path as recorded.

    Returns:
        True when the path names or passes through ``/proc`` or ``/dev``.
    """
    lexical = os.path.normpath("/" + path.lstrip("/"))
    if lexical.split("/")[1] in ("proc", "dev"):
        return True
    root_device = os.stat("/").st_dev
    special = set()
    for mount in ("/proc", "/dev"):
        try:
            device = os.stat(mount).st_dev
        except OSError:
            continue
        if device != root_device:
            special.add(device)
    current, pending, links = "/", path.split("/"), 0
    while pending:
        part = pending.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            current = os.path.dirname(current)
            continue
        candidate = os.path.join(current, part)
        try:
            info = os.lstat(candidate)
        except OSError:
            # A missing component cannot be bound to any existing file.
            return False
        if info.st_dev in special:
            return True
        if stat.S_ISLNK(info.st_mode):
            links += 1
            if links > 40:
                return True
            target = os.readlink(candidate)
            if target.startswith("/"):
                current = "/"
            pending = target.split("/") + pending
            continue
        current = candidate
    return False


def _stable_path(path: object) -> bool:
    """Return whether a recorded path names the same file in every process.

    A stable path is an absolute string that does not resolve through ``/proc``
    or ``/dev``. A relative path depends on the working directory and a
    ``/proc`` or ``/dev`` path on the calling process, so ``samefile()`` alone
    could bind either one to the expected file in one process only.

    Args:
        path: Recorded path.

    Returns:
        True for a stable absolute path.
    """
    return isinstance(path, str) and os.path.isabs(path) and not _process_dependent(path)


def _split_command(text: str, label: str) -> list[str]:
    """Split one recorded shell command into its argument list.

    Args:
        text: Recorded command text.
        label: Command name for error messages.

    Returns:
        Argument list.

    Raises:
        ValueError: If the text is not valid POSIX shell quoting.
    """
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"The recorded {label} cannot be parsed: {text!r}") from exc


def _bound_cfl_token(token: str, expected: Path, label: str, command: str) -> dict[str, str]:
    """Require a recorded BART path to name the current accepted CFL pair.

    Args:
        token: BART basename exactly as recorded in the command.
        expected: Basename of the current accepted artifact.
        label: Role of the path, for error messages.
        command: Name of the recorded command, for error messages.

    Returns:
        Recorded, resolved, and accepted basenames.

    Raises:
        ValueError: If the token is relative or process-dependent, has another
            basename (which includes BART's single-file suffixes), or if its
            ``.hdr`` or ``.cfl`` member is not the same file as the accepted
            artifact.
    """
    recorded = Path(token)
    if not recorded.is_absolute() or _process_dependent(token):
        raise ValueError(
            f"The accepted {command} records a relative or process-dependent {label} path "
            f"{token!r}, which cannot be bound to {expected}."
        )
    # BART opens names such as x.ra as one file rather than x.ra.hdr/.cfl, so
    # the basename must match before the pair is compared.
    if recorded.name != expected.name or recorded.name.endswith(_BART_SINGLE_FILE_SUFFIXES):
        raise ValueError(
            f"The accepted {command} names {label} {token!r}; the basename must be "
            f"{expected.name!r}."
        )
    if not all(
        _same_file(f"{token}{suffix}", f"{expected}{suffix}") for suffix in (".hdr", ".cfl")
    ):
        raise ValueError(
            f"The accepted {command} names {label} {token!r}, which is not the current "
            f"accepted artifact {expected}."
        )
    return {"recorded": token, "resolved": str(Path(token).resolve()), "accepted": str(expected)}


def _accepted_ecalib_command(text: str, paths: Mapping[str, Path]) -> dict[str, Any]:
    """Parse and bind the recorded one-map ``ecalib -m 1 -c 0`` command.

    Args:
        text: Recorded command text.
        paths: Current accepted artifact paths.

    Returns:
        Argument list and the bound calibration k-space and CSM paths.

    Raises:
        ValueError: If the command is not exactly ``bart ecalib -m 1 -c 0
            KSPACE_CALIB COIL_SENS`` naming this root's artifacts.
    """
    tokens = _split_command(text, "ecalib command")
    if len(tokens) != 8 or tuple(tokens[:6]) != ("bart", *ACCEPTED_BASELINE_ECALIB_ARGUMENTS):
        raise ValueError(
            "The accepted baseline CSM must be recorded as 'bart ecalib -m 1 -c 0 "
            f"KSPACE_CALIB COIL_SENS'; found {text!r}."
        )
    command = "ecalib command"
    return {
        "argv": tokens,
        "bound_artifacts": {
            "kspace_calib": _bound_cfl_token(
                tokens[6], paths["kspace_calib"], "calibration k-space", command
            ),
            "coil_sens": _bound_cfl_token(tokens[7], paths["coil_sens"], "CSM output", command),
        },
    }


def _accepted_fista_r0_command(text: str, paths: Mapping[str, Path]) -> dict[str, Any]:
    """Parse and bind the recorded unregularized FISTA-r0 Wave command.

    Args:
        text: Recorded command text.
        paths: Current accepted artifact paths.

    Returns:
        Argument list, the GPU flag, and the bound CSM, PSF, wave k-space,
        and output image paths.

    Raises:
        ValueError: If the command is not exactly ``bart wave [-g] -w -f -r 0
            -i 100 -t 1e-6 COIL_SENS PSF WAVE_KSPACE IMAGE`` naming this
            root's artifacts; regularized and other-branch commands fail.
    """
    tokens = _split_command(text, "FISTA-r0 command")
    gpu = tokens[2:3] == ["-g"]
    expected_length = 2 + int(gpu) + len(ACCEPTED_FISTA_R0_WAVE_FLAGS) + 4
    if (
        len(tokens) != expected_length
        or tokens[:2] != ["bart", "wave"]
        or tuple(tokens[2 + int(gpu) : -4]) != ACCEPTED_FISTA_R0_WAVE_FLAGS
    ):
        raise ValueError(
            "The accepted FISTA-r0 image must be recorded as 'bart wave [-g] "
            f"{' '.join(ACCEPTED_FISTA_R0_WAVE_FLAGS)} COIL_SENS PSF WAVE_KSPACE IMAGE'; "
            f"found {text!r}."
        )
    command = "FISTA-r0 command"
    csm, psf, kspace, image = tokens[-4:]
    return {
        "argv": tokens,
        "gpu": gpu,
        "bound_artifacts": {
            "coil_sens": _bound_cfl_token(csm, paths["coil_sens"], "CSM", command),
            "psf": _bound_cfl_token(psf, paths["psf"], "PSF", command),
            "wave_kspace": _bound_cfl_token(kspace, paths["wave_kspace"], "wave k-space", command),
            "fista_r0_image": _bound_cfl_token(
                image, paths["fista_r0_image"], "output image", command
            ),
        },
    }


def _identity_cfl_record(path: Path) -> dict[str, Any]:
    """Describe a large accepted CFL pair without hashing its payload.

    The FISTA-r0 PSF and wave k-space are bound to the recorded command by
    file identity; the diagnostics never read them, so only their header is
    hashed.

    Args:
        path: BART basename of a complete pair.

    Returns:
        Base, shape, dtype, payload size, header SHA-256, and hash policy.
    """
    base = bart_base(path)
    return {
        **cfl_record(base, include_hash=False),
        "header_sha256": sha256_file(base.with_suffix(".hdr")),
        "payload_hash": "not computed; bound by file identity and never read by the diagnostics",
    }


def _require_same_cfl_content(
    recorded: Mapping[str, Any], current: Mapping[str, Any], label: str
) -> None:
    """Compare the shape, size, and hashes of two CFL records.

    Args:
        recorded: Record stored in a manifest.
        current: Freshly computed record of the same artifact.
        label: Artifact name for error messages.

    Raises:
        ValueError: If any field present in either record differs.
    """
    for key in ("shape", "payload_bytes", "header_sha256", "payload_sha256"):
        if (key in recorded or key in current) and recorded.get(key) != current.get(key):
            raise ValueError(f"{label} changed after it was recorded ({key} differs).")


def _verify_file(record: object, expected: Path, label: str) -> None:
    """Verify one recorded file at the location the current stage reads.

    Args:
        record: Manifest file record with ``path``, ``size_bytes``, and
            ``sha256``.
        expected: Location of the file in the current output root.
        label: Artifact name for error messages.

    Raises:
        FileNotFoundError: If the expected file is absent.
        ValueError: If the record is incomplete, its path is not a stable
            absolute path (see :func:`_stable_path`), it names another file,
            or the size or SHA-256 changed.
    """
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or not isinstance(record.get("sha256"), str)
    ):
        raise ValueError(f"{label} has no complete file record.")
    if not expected.is_file():
        raise FileNotFoundError(f"{label} is missing: {expected}")
    if not _stable_path(record["path"]) or not _same_file(record["path"], expected):
        raise ValueError(
            f"{label} record names {record['path']!r}, not {expected} by a stable absolute path."
        )
    actual = rovir_feasibility._file_record(expected)
    if actual["size_bytes"] != int(record["size_bytes"]) or actual["sha256"] != str(record["sha256"]):
        raise ValueError(f"{label} changed after it was recorded: {expected}")


def _verify_cfl(record: object, expected: Path, label: str) -> None:
    """Verify one recorded CFL pair at the location the current stage reads.

    Args:
        record: Manifest CFL record from :func:`bart_io.cfl_record`.
        expected: BART basename in the current output root.
        label: Artifact name for error messages.

    Raises:
        FileNotFoundError: If the pair is absent.
        ValueError: If the pair is partial, the recorded base is not a stable
            absolute path (see :func:`_stable_path`), the record names another
            pair, or the shape, size, or hashes changed.
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("base"), str):
        raise ValueError(f"{label} has no complete CFL record.")
    require_complete_cfl_pair(expected, label)
    base, recorded = bart_base(expected), bart_base(str(record["base"]))
    if not _stable_path(record["base"]) or not all(
        _same_file(recorded.with_suffix(suffix), base.with_suffix(suffix))
        for suffix in (".hdr", ".cfl")
    ):
        raise ValueError(
            f"{label} record names {record['base']!r}, not {base} by a stable absolute path."
        )
    _require_same_cfl_content(record, cfl_record(base), label)


def validate_accepted_baseline(
    accepted_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and hash the accepted one-map ``c=0`` baseline artifacts.

    Args:
        accepted_root: Accepted normal reconstruction root.

    Returns:
        ``(record, manifest)``: JSON-native identities of the accepted manifest,
        calibration k-space, PSF, wave k-space, one-map CSM, FISTA-r0 image,
        and both parsed command records with their bound artifact paths, plus
        the decoded accepted input manifest.

    Raises:
        FileNotFoundError: If a required artifact is absent.
        ValueError: If the readout crop is not the alias-free contract, array
            geometry disagrees with the manifest, the recorded CSM command is
            not ``bart ecalib -m 1 -c 0`` on this root's calibration k-space
            and CSM, or the recorded FISTA-r0 command is not the unregularized
            ``bart wave [-g] -w -f -r 0 -i 100 -t 1e-6`` on this root's CSM,
            PSF, wave k-space, and FISTA-r0 image. Recorded paths are bound by
            file identity, so symlink aliases pass and other roots fail.
    """
    paths = accepted_artifact_paths(accepted_root)
    manifest = rovir_feasibility._read_json(paths["manifest"])
    if not mprage_module._uses_alias_free_coil_calibration(manifest):
        raise ValueError(
            "Accepted normal inputs must record the versioned alias-free "
            "centered-image-domain readout crop; stride-derived inputs are not reusable."
        )
    nro, nlin, npar = (int(value) for value in manifest["geometry"]["logical_matrix_ro_lin_par"])
    virtual = int(manifest["coil_compression"]["virtual_coils"])
    grid_sizes = {
        "RO": nro,
        "LIN": nlin,
        "PAR": npar,
        "ACS": int(manifest["psf_calibration"]["nacs"]),
        "calibration grid": int(manifest["psf_calibration"]["ncalib"]),
    }
    odd = [name for name, size in grid_sizes.items() if size % 2]
    if odd:
        # The fftshift convention used for coil images equals BART's centered
        # FFT grid only for even sizes, which image/map alignment relies on.
        raise ValueError(f"Odd grid sizes are not supported for map alignment: {odd}.")
    kspace_shape = _active_shape(
        require_complete_cfl_pair(paths["kspace_calib"], "accepted kspace_calib"),
        4,
        "accepted kspace_calib",
    )
    csm_shape = _active_shape(
        require_complete_cfl_pair(paths["coil_sens"], "accepted one-map CSM"),
        5,
        "accepted one-map CSM",
    )
    image_shape = _active_shape(
        require_complete_cfl_pair(paths["fista_r0_image"], "accepted FISTA-r0 image"),
        3,
        "accepted FISTA-r0 image",
    )
    if kspace_shape != (nro, nlin, npar, virtual):
        raise ValueError(f"Accepted kspace_calib shape {kspace_shape} disagrees with the manifest.")
    if csm_shape != (nro, nlin, npar, virtual, 1):
        raise ValueError(f"Accepted CSM must hold one map on {(nro, nlin, npar, virtual)}; found {csm_shape}.")
    if image_shape != (nro, nlin, npar):
        raise ValueError(f"Accepted FISTA-r0 image shape {image_shape} disagrees with the manifest.")
    require_complete_cfl_pair(paths["psf"], "accepted PSF")
    require_complete_cfl_pair(paths["wave_kspace"], "accepted wave k-space")
    # Both records must name exactly this root's artifacts; matching path
    # suffixes alone would accept commands recorded for another root.
    command_text = paths["ecalib_command"].read_text(encoding="utf-8").strip()
    ecalib = _accepted_ecalib_command(command_text, paths)
    fista_text = paths["fista_r0_command"].read_text(encoding="utf-8").strip()
    fista = _accepted_fista_r0_command(fista_text, paths)
    record = {
        "root": str(paths["root"]),
        "manifest": rovir_feasibility._file_record(paths["manifest"]),
        "geometry": dict(manifest["geometry"]),
        "virtual_coils": virtual,
        "physical_coils": int(manifest["coil_compression"]["physical_coils"]),
        "retained_energy": manifest["coil_compression"].get("retained_energy"),
        "nacs": int(manifest["psf_calibration"]["nacs"]),
        "ncalib": int(manifest["psf_calibration"]["ncalib"]),
        "readout_oversampling_removal": dict(
            manifest["coil_compression"]["readout_oversampling_removal"]
        ),
        "kspace_calib": cfl_record(paths["kspace_calib"]),
        "coil_sens": cfl_record(paths["coil_sens"]),
        "fista_r0_image": cfl_record(paths["fista_r0_image"]),
        "psf": _identity_cfl_record(paths["psf"]),
        "wave_kspace": _identity_cfl_record(paths["wave_kspace"]),
        "ecalib_command": {
            **rovir_feasibility._file_record(paths["ecalib_command"]),
            "text": command_text,
            **ecalib,
        },
        "fista_r0_command": {
            **rovir_feasibility._file_record(paths["fista_r0_command"]),
            "text": fista_text,
            **fista,
        },
    }
    return record, manifest


def noise_and_channel_record(
    twix: str | Path,
    output_root: str | Path,
    *,
    physical_coils: int,
    nacs: int,
    acs_dwell_ns: float,
) -> dict[str, Any]:
    """Record measurement-0 noise, channel identity, and header metadata.

    Args:
        twix: Measured TWIX file with an adjustment and an acquisition raid entry.
        output_root: Diagnostic output root receiving the covariance array.
        physical_coils: Physical coil count recorded by the accepted manifest.
        nacs: Set-4 ACS width along LIN and PAR.
        acs_dwell_ns: Set-4 ACS dwell from the sequence contract.

    Returns:
        JSON-native noise, channel-identity, coil-select, metadata, and
        held-out covariance records.

    Raises:
        ValueError: If the raid layout, channel identity, set-4 lattice, or
            block-0 coil selection is inconsistent.

    Side Effects:
        Writes ``inputs/noise/noise_covariance.npy``.
    """
    twix_path = Path(twix).expanduser().resolve()
    table = twix_noise.read_multiraid_table_from_file(twix_path)
    if len(table) != 2:
        raise ValueError(
            "Expected exactly two raid measurements (adjustment scan with noise, "
            f"then the acquisition); found {len(table)}."
        )
    noise_walk = twix_noise.walk_measurement(twix_path, table[0])
    acquisition_walk = twix_noise.walk_measurement(twix_path, table[1])
    for label, walk in (("noise", noise_walk), ("acquisition", acquisition_walk)):
        if not walk.get("acqend_found"):
            raise ValueError(
                f"The {label} MDH walk ended without ACQEND "
                f"({walk.get('termination')}); channel identity would be incomplete."
            )
    identity = twix_noise.channel_identity_report(
        *_identity_walks(noise_walk, acquisition_walk),
        required_roles=("noise", "image", f"refscan_set{ACS_SET_INDEX}"),
    )
    if not identity["identical"] or not identity["required_roles_present"]:
        raise ValueError(
            "Noise, image, and refscan MDH channel IDs are not identical: "
            + "; ".join(str(item) for item in identity.get("differences", []))
        )
    if int(identity["channel_count"]) != int(physical_coils):
        raise ValueError(
            f"MDH channel count {identity['channel_count']} disagrees with "
            f"{physical_coils} physical coils in the accepted manifest."
        )
    refscan_roles = sorted(
        role for role in acquisition_walk["roles"] if role.startswith("refscan_set")
    )
    expected_roles = [f"refscan_set{index}" for index in range(EXPECTED_REFSCAN_SETS)]
    if refscan_roles != expected_roles:
        raise ValueError(f"Refscan sets {refscan_roles} differ from {expected_roles}.")
    set4 = acquisition_walk["roles"][f"refscan_set{ACS_SET_INDEX}"]
    lattice_ok = (
        int(set4["lines"]) == nacs * nacs
        and list(set4["lin_range"]) == [0, nacs - 1]
        and list(set4["par_range"]) == [0, nacs - 1]
        and int(set4["unique_lin_par_pairs"]) == nacs * nacs
    )
    if not lattice_ok:
        raise ValueError(f"Set-4 MDH lattice is not a complete {nacs}x{nacs} ACS: {set4}.")

    headers = twix_noise.load_measurement_headers(twix_path)
    if len(headers) != 2:
        raise ValueError("Expected headers for exactly two raid measurements.")
    coil_select = twix_noise.compare_coil_select(
        headers[0]["MeasYaps"], headers[1]["MeasYaps"], block=0
    )
    if not coil_select["identical"]:
        raise ValueError(
            "Block-0 coil-element to ADC-channel maps differ between the noise "
            f"and acquisition measurements: {coil_select['differences']}"
        )
    noise_metadata = twix_noise.measurement_metadata(
        headers[0]["MeasYaps"], headers[0].get("Meas")
    )
    acquisition_metadata = twix_noise.measurement_metadata(
        headers[1]["MeasYaps"], headers[1].get("Meas")
    )

    noise = twix_noise.load_measurement0_noise(twix_path)
    lines = np.asarray(noise["lines"])
    if lines.ndim != 3 or lines.shape[-1] != int(physical_coils):
        raise ValueError(f"Noise lines must be (lines, samples, {physical_coils}); found {lines.shape}.")
    covariance = twix_noise.noise_covariance(lines.reshape(-1, lines.shape[-1]))
    estimation, validation, split = twix_noise.split_noise_lines(lines, rule="alternate")
    held_out = twix_noise.compare_covariances(
        twix_noise.noise_covariance(estimation.reshape(-1, lines.shape[-1])),
        twix_noise.noise_covariance(validation.reshape(-1, lines.shape[-1])),
    )
    covariance_path = output_path(output_root, "noise_covariance")
    covariance_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(covariance_path, covariance.astype(np.complex128), allow_pickle=False)
    noise_dwell = noise_metadata.get("dwell_ns")
    return {
        "raid_measurements": [entry.to_json() for entry in table],
        "noise_measurement_index": 0,
        "acquisition_measurement_index": 1,
        "noise_walk": noise_walk,
        "acquisition_walk": acquisition_walk,
        "channel_identity": identity,
        "set4_mdh_lattice": {**set4, "complete": True},
        "coil_select_block0": coil_select,
        "noise_metadata": noise_metadata,
        "acquisition_metadata": acquisition_metadata,
        "noise_lines_shape": list(lines.shape),
        "noise_covariance": rovir_feasibility._file_record(covariance_path),
        "noise_covariance_statistics": twix_noise.correlation_statistics(covariance),
        "held_out_split": split,
        "held_out_comparison": held_out,
        "held_out_compatibility": twix_noise.covariance_compatibility(held_out),
        "acs_dwell_ns": float(acs_dwell_ns),
        "expected_white_noise_variance_ratio_acs_over_noise": (
            None
            if noise_dwell is None
            else twix_noise.expected_white_noise_variance_ratio(
                float(noise_dwell), float(acs_dwell_ns)
            )
        ),
        "absolute_acs_noise_calibration_claimed": False,
        "fft_scale_applied": False,
        "raw_data_correction_applied": False,
    }


def _identity_walks(
    noise_walk: Mapping[str, Any], acquisition_walk: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restrict MDH walks to the roles whose channels must match.

    The adjustment measurement also stores its own coil-sensitivity reference
    lines, including two-channel body-coil lines, which never enter the
    diagnostics. Only its noise lines are compared with the acquisition image
    and refscan lines.

    Args:
        noise_walk: Walk of the adjustment measurement.
        acquisition_walk: Walk of the acquisition measurement.

    Returns:
        Copies of both walks restricted to the compared roles.
    """
    noise_roles = {
        role: value for role, value in noise_walk["roles"].items() if role == "noise"
    }
    acquisition_roles = {
        role: value
        for role, value in acquisition_walk["roles"].items()
        if role == "image" or role.startswith("refscan_set")
    }
    return (
        {**noise_walk, "roles": noise_roles},
        {**acquisition_walk, "roles": acquisition_roles},
    )


def _source_identity(
    twix: str | Path, sequence: str | Path, accepted_root: str | Path
) -> dict[str, Any]:
    """Bind TWIX and sequence to the accepted manifest without hashing TWIX.

    Args:
        twix: Measured TWIX file.
        sequence: Matching Pulseq sequence.
        accepted_root: Accepted normal reconstruction root.

    Returns:
        Source contract from the existing ROVir-named validator.

    Raises:
        ValueError: If either source differs from the accepted manifest.
    """
    return rovir_feasibility._validated_source_contract(
        twix, sequence, accepted_root, include_twix_hash=False
    )


def _mapping(container: object, key: str, label: str) -> Mapping[str, Any]:
    """Return one nested manifest record, requiring a mapping.

    Args:
        container: Parent record.
        key: Child key.
        label: Name of the parent record for error messages.

    Returns:
        The child mapping.

    Raises:
        ValueError: If the parent or the child is not a mapping.
    """
    value = container.get(key) if isinstance(container, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} has no valid {key!r} record.")
    return value


def _validate_prepare_reuse(
    existing: Mapping[str, Any],
    source: Mapping[str, Any],
    baseline: Mapping[str, Any],
    output_root: Path,
) -> None:
    """Reject reuse of a prepare manifest whose inputs changed.

    Args:
        existing: Stored prepare manifest.
        source: Freshly validated source contract.
        baseline: Freshly hashed accepted-baseline record.
        output_root: Diagnostic output root.

    Raises:
        FileNotFoundError: If a recorded source, accepted file, or output is
            missing.
        ValueError: If the manifest is malformed; a recorded path or base is
            not a stable absolute path; a source, accepted array, accepted
            manifest, or command record changed or names another file; any
            other accepted field differs from the current accepted root; or a
            recorded prepare output changed.
    """
    label = "The prepare manifest"
    sources = _mapping(existing, "sources", label)
    recorded_twix = _mapping(sources, "twix", f"{label} sources")
    if not _stable_path(recorded_twix.get("path")):
        raise ValueError(
            f"{label} records the TWIX as {recorded_twix.get('path')!r}, which is not a stable "
            "absolute path."
        )
    current_twix = source["twix"]
    for key in ("path", "size_bytes", "mtime_ns"):
        if recorded_twix.get(key) != current_twix.get(key):
            raise ValueError(f"TWIX {key} differs from the prepared diagnostic inputs.")
    # Recorded source and accepted files are verified as complete file records
    # (stable path, same file, size, and SHA-256), never by their hash alone.
    _verify_file(sources.get("sequence"), Path(source["sequence"]["path"]), "Recorded sequence")
    _verify_file(
        sources.get("accepted_normal_manifest"),
        Path(source["normal_manifest"]["path"]),
        "Recorded accepted normal manifest",
    )
    accepted = _mapping(existing, "accepted", label)
    _verify_file(accepted.get("manifest"), Path(baseline["manifest"]["path"]), "Recorded accepted manifest")
    for key in ("ecalib_command", "fista_r0_command"):
        _verify_file(accepted.get(key), Path(baseline[key]["path"]), f"Recorded accepted {key}")
    for key in ("kspace_calib", "coil_sens", "fista_r0_image", "psf", "wave_kspace"):
        recorded = accepted.get(key)
        if not isinstance(recorded, Mapping) or not isinstance(recorded.get("base"), str):
            raise ValueError(
                f"The prepare manifest has no accepted {key} record; prepare again in a new "
                "output root."
            )
        if not _stable_path(recorded["base"]):
            raise ValueError(
                f"The recorded accepted {key} base {recorded['base']!r} is not a stable absolute path."
            )
        recorded_base, current_base = bart_base(recorded["base"]), bart_base(baseline[key]["base"])
        if not all(
            _same_file(recorded_base.with_suffix(suffix), current_base.with_suffix(suffix))
            for suffix in (".hdr", ".cfl")
        ):
            raise ValueError(f"Accepted {key} is not the file recorded at preparation.")
        _require_same_cfl_content(recorded, baseline[key], f"Accepted {key}")
    # Every other accepted field, including command texts and bound paths, must
    # equal the record computed now from the accepted root.
    current = json.loads(json.dumps(_json_ready(baseline)))
    for key in sorted(set(accepted) | set(current)):
        if accepted.get(key) != current.get(key):
            raise ValueError(f"The recorded accepted {key} differs from the current accepted root.")
    _verify_prepare_outputs(existing, output_root)
    _verify_environment(existing, output_root, label)


def _verify_prepare_outputs(existing: Mapping[str, Any], output_root: Path) -> None:
    """Verify every output recorded by the prepare stage.

    Args:
        existing: Stored prepare manifest.
        output_root: Diagnostic output root.

    Raises:
        FileNotFoundError: If a recorded output is missing.
        ValueError: If a record is malformed, an output changed, or its
            record names another file.
    """
    label = "The prepare manifest"
    _verify_file(
        _mapping(existing, "noise", label).get("noise_covariance"),
        output_root / LAYOUT["noise_covariance"],
        "Noise covariance",
    )
    physical = _mapping(existing, "physical_calibration", label)
    _verify_file(
        physical.get("manifest"),
        output_root / rovir_feasibility.PHYSICAL_CALIBRATION_MANIFEST,
        "Physical set-4 export manifest",
    )
    _verify_cfl(
        physical.get("kspace"),
        output_root / LAYOUT["physical_calibration"],
        "Physical set-4 ACS export",
    )
    _verify_file(
        existing.get("map1_reference"), output_root / LAYOUT["map1_reference"], "Map-1 reference record"
    )


def _existing_outputs(root: Path, relative_paths: Sequence[Path]) -> list[Path]:
    """List stage-owned files that already exist below an output root.

    Args:
        root: Diagnostic output root.
        relative_paths: Stage-owned files or directories; directories count
            only when they contain a file.

    Returns:
        Existing stage-owned files or nonempty directories.
    """
    found = []
    for relative in relative_paths:
        path = root / relative
        if path.is_file() or (path.is_dir() and any(item.is_file() for item in path.rglob("*"))):
            found.append(path)
    return found


def prepare_csm_consistency(
    twix: str | Path,
    sequence: str | Path,
    accepted_root: str | Path,
    output_root: str | Path,
    *,
    environment_log: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare hash-bound inputs for the set-4 CSM consistency diagnostics.

    Args:
        twix: Measured Wave-MPRAGE TWIX file.
        sequence: Matching Pulseq sequence.
        accepted_root: Accepted normal root with the one-map ``c=0`` CSM.
        output_root: User-approved diagnostic output root.
        environment_log: Shell log of this invocation under ``logs/environment``;
            recorded only when a new manifest is written.

    Returns:
        Prepare manifest; an existing manifest is returned unchanged after
        every recorded input and output has been revalidated.

    Raises:
        FileExistsError: If prepare outputs exist without a prepare manifest.
        ValueError: If sources, sequence contract, channel identity, set-4
            lattice, coil selection, accepted artifacts, or recorded outputs
            are inconsistent.

    Side Effects:
        Writes the physical set-4 export (existing exporter), the noise
        covariance, the map-1 reference record, and the prepare manifest.
        Reads the TWIX file; launches no process.
    """
    twix_path = Path(twix).expanduser().resolve()
    sequence_path = Path(sequence).expanduser().resolve()
    accepted = Path(accepted_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    log_path = environment_log_path(output, environment_log)
    manifest_path = output / LAYOUT["prepare_manifest"]
    source = _source_identity(twix_path, sequence_path, accepted)
    baseline, accepted_manifest = validate_accepted_baseline(accepted)
    if manifest_path.is_file():
        existing = rovir_feasibility._read_json(manifest_path)
        _validate_prepare_reuse(existing, source, baseline, output)
        return existing
    # Outputs without a manifest come from an interrupted run; they are never
    # overwritten. The reused exporter validates its own outputs.
    leftovers = _existing_outputs(output, (LAYOUT["noise_covariance"], LAYOUT["map1_reference"]))
    if leftovers:
        raise FileExistsError(
            "Prepare outputs exist without a prepare manifest; inspect and move them aside "
            "before rerunning: " + ", ".join(str(path) for path in leftovers)
        )

    definitions, _ = mprage_module._read_sequence(sequence_path)
    contract = sequence_contract(definitions, accepted_manifest)
    # Reuse the existing set-4 exporter unchanged; its manifest keeps the
    # inherited ROVir-named status string documented in the diagnostics guide.
    physical_manifest = rovir_feasibility.export_mprage_physical_calibration(
        twix_path, sequence_path, accepted, output, acs_set_index=ACS_SET_INDEX
    )
    physical_manifest_path = output / rovir_feasibility.PHYSICAL_CALIBRATION_MANIFEST
    noise = noise_and_channel_record(
        twix_path,
        output,
        physical_coils=baseline["physical_coils"],
        nacs=baseline["nacs"],
        acs_dwell_ns=contract["calibration_dwell_s"] * 1e9,
    )
    map1_path = output / LAYOUT["map1_reference"]
    rovir_feasibility._write_json(
        map1_path,
        {
            "role": "accepted one-map c=0 reconstruction baseline, referenced without copying",
            "coil_sens": baseline["coil_sens"],
            "ecalib_command": baseline["ecalib_command"],
        },
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "status": "mprage_csm_consistency_inputs_ready",
        "created_at_utc": _utc_now(),
        "implementation": implementation_identity(),
        "environment": environment_record(output, log_path),
        "sources": {
            "twix": dict(physical_manifest["source"]["twix"]),
            "sequence": dict(source["sequence"]),
            "accepted_normal_manifest": dict(source["normal_manifest"]),
        },
        "sequence_contract": contract,
        "accepted": baseline,
        "physical_calibration": {
            "manifest": rovir_feasibility._file_record(physical_manifest_path),
            "inherited_status": physical_manifest.get("status"),
            "kspace": cfl_record(output / LAYOUT["physical_calibration"]),
            "set_index_zero_based": ACS_SET_INDEX,
        },
        "noise": noise,
        "map1_reference": rovir_feasibility._file_record(map1_path),
        "flags": dict(_FALSE_FLAGS),
    }
    rovir_feasibility._write_json(manifest_path, payload)
    return payload


def load_prepared_inputs(
    twix: str | Path,
    sequence: str | Path,
    accepted_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Load a prepare manifest after revalidating every recorded input.

    Args:
        twix: Measured TWIX file.
        sequence: Matching sequence.
        accepted_root: Accepted normal root.
        output_root: Diagnostic output root.

    Returns:
        The validated prepare manifest.

    Raises:
        FileNotFoundError: If the prepare stage has not run or a recorded
            output is missing.
        ValueError: If any recorded input or output changed.
    """
    output = Path(output_root).expanduser().resolve()
    manifest_path = output / LAYOUT["prepare_manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run the prepare stage before this stage: {manifest_path}")
    existing = rovir_feasibility._read_json(manifest_path)
    source = _source_identity(twix, sequence, accepted_root)
    baseline, _ = validate_accepted_baseline(accepted_root)
    _validate_prepare_reuse(existing, source, baseline, output)
    return existing


def _finite_by_partition(view: np.ndarray, label: str) -> None:
    """Require finite values while reading bounded PAR slabs.

    Args:
        view: Memory-mapped array with PAR on axis 2.
        label: Artifact name for error messages.

    Raises:
        ValueError: If any value is not finite.
    """
    for start in range(0, view.shape[2], PAR_CHUNK):
        if not np.isfinite(np.asarray(view[:, :, start : start + PAR_CHUNK])).all():
            raise ValueError(f"{label} contains non-finite values.")


def two_map_input_record(output_root: str | Path, prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the two-map outputs to the calibration k-space that ecalib read.

    The shell writes ``sha256sum`` lines for ``kspace_calib.hdr`` and
    ``kspace_calib.cfl`` immediately before it runs ``bart ecalib``.

    Args:
        output_root: Diagnostic output root.
        prepared: Validated prepare manifest.

    Returns:
        File record of the hash record plus the header and payload digests.

    Raises:
        FileNotFoundError: If the hash record is absent.
        ValueError: If the record is malformed, names a path that is not a
            stable absolute path, or does not name exactly the prepared
            accepted ``kspace_calib`` pair with its recorded hashes.
    """
    path = Path(output_root).expanduser().resolve() / LAYOUT["two_map_input"]
    if not path.is_file():
        raise FileNotFoundError(f"Two-map input hash record is absent: {path}")
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64 or line.startswith("\\"):
            raise ValueError(f"Two-map input hash record is malformed: {path}")
        entries.append((parts[0].lower(), parts[1].lstrip("*")))
    unstable = [name for _, name in entries if not _stable_path(name)]
    if unstable:
        raise ValueError(
            f"Two-map input hash record names paths that are not stable absolute paths: {unstable}"
        )
    expected = _mapping(prepared, "accepted", "The prepare manifest")["kspace_calib"]
    base = bart_base(expected["base"])
    digests = {}
    for suffix, key in ((".hdr", "header_sha256"), (".cfl", "payload_sha256")):
        matches = [digest for digest, name in entries if _same_file(name, base.with_suffix(suffix))]
        if len(entries) != 2 or len(matches) != 1 or matches[0] != expected[key]:
            raise ValueError(
                "The two-map outputs were not computed from the prepared accepted kspace_calib; "
                f"move the two-map outputs aside and rerun calibrate: {path}"
            )
        digests[key] = matches[0]
    return {**rovir_feasibility._file_record(path), **digests}


def verify_two_map_calibration(
    calibration: Mapping[str, Any],
    prepared: Mapping[str, Any],
    accepted_root: str | Path,
    output_root: str | Path,
) -> None:
    """Verify a recorded two-map calibration against its current artifacts.

    Args:
        calibration: Stored calibration manifest.
        prepared: Validated prepare manifest.
        accepted_root: Accepted normal root.
        output_root: Diagnostic output root.

    Raises:
        FileNotFoundError: If a recorded artifact is missing.
        ValueError: If the manifest is malformed, the calibration was recorded
            against another prepare manifest or accepted calibration k-space,
            the maps were computed from another k-space, the command differs
            from the reviewed command, or the command record, input hash
            record, maps, eigenvalues, or BART logs changed.
    """
    output = Path(output_root).expanduser().resolve()
    _verify_file(
        calibration.get("prepare_manifest"),
        output / LAYOUT["prepare_manifest"],
        "Prepare manifest of the two-map calibration",
    )
    if calibration.get("accepted_kspace_calib") != prepared["accepted"]["kspace_calib"]:
        raise ValueError("The two-map calibration used a different accepted kspace_calib.")
    recorded_input = calibration.get("ecalib_input")
    _verify_file(recorded_input, output / LAYOUT["two_map_input"], "Two-map input hash record")
    current_input = two_map_input_record(output, prepared)
    if any(recorded_input.get(key) != current_input[key] for key in ("header_sha256", "payload_sha256")):
        raise ValueError("The two-map input hash record no longer matches the calibration manifest.")
    command = _mapping(calibration, "command", "The two-map calibration manifest")
    _verify_file(command.get("record"), output / LAYOUT["two_map_command"], "Two-map command record")
    expected = two_map_ecalib_argv(accepted_root, output)
    if command.get("argv") != expected or _split_command(
        str(command.get("text", "")), "two-map ecalib command"
    ) != expected:
        raise ValueError(
            f"The recorded two-map command is not the reviewed command {format_command(expected)!r}."
        )
    _verify_environment(calibration, output, "The two-map calibration manifest")
    _verify_cfl(calibration.get("maps"), output / LAYOUT["two_map_maps"], "Two-map CSM")
    _verify_cfl(
        calibration.get("eigenvalues"), output / LAYOUT["two_map_eigenvalues"], "Eigenvalue maps"
    )
    for key, label in (
        ("bart_version", "BART version record"),
        ("bart_binary", "BART binary hash record"),
        ("ecalib_log", "Two-map ecalib log"),
    ):
        layout_key = "bart_binary_hash" if key == "bart_binary" else key
        _verify_file(calibration.get(key), output / LAYOUT[layout_key], label)


def record_two_map_calibration(
    twix: str | Path,
    sequence: str | Path,
    accepted_root: str | Path,
    output_root: str | Path,
    *,
    environment_log: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and record the explicit diagnostic ``ecalib -m 2 -c 0`` run.

    Args:
        twix: Measured TWIX file.
        sequence: Matching sequence.
        accepted_root: Accepted normal root whose ``kspace_calib`` was used.
        output_root: Diagnostic output root.
        environment_log: Shell log of this invocation under ``logs/environment``;
            recorded only when a new manifest is written.

    Returns:
        Calibration manifest; an existing manifest is returned only after
        :func:`verify_two_map_calibration` has verified every recorded
        artifact.

    Raises:
        FileNotFoundError: If an output, command record, input hash record,
            or log is absent.
        ValueError: If a CFL pair is partial, geometry or map count is wrong,
            values are non-finite, the recorded command differs from the
            reviewed command, the maps were not computed from the prepared
            accepted calibration k-space, or a recorded artifact changed.

    Side Effects:
        Writes ``manifests/two_map_calibration.json``.
    """
    output = Path(output_root).expanduser().resolve()
    log_path = environment_log_path(output, environment_log)
    prepared = load_prepared_inputs(twix, sequence, accepted_root, output)
    manifest_path = output / LAYOUT["calibration_manifest"]
    if manifest_path.is_file():
        existing = rovir_feasibility._read_json(manifest_path)
        verify_two_map_calibration(existing, prepared, accepted_root, output)
        return existing
    geometry = prepared["accepted"]["geometry"]
    spatial = tuple(int(value) for value in geometry["logical_matrix_ro_lin_par"])
    coils = int(prepared["accepted"]["virtual_coils"])
    maps_base = output / LAYOUT["two_map_maps"]
    eigen_base = output / LAYOUT["two_map_eigenvalues"]
    maps_shape = require_complete_cfl_pair(maps_base, "two-map CSM")
    eigen_shape = require_complete_cfl_pair(eigen_base, "two-map eigenvalue maps")
    metrics.validate_two_map_outputs(maps_shape, eigen_shape, spatial, coils)
    command_path = output / LAYOUT["two_map_command"]
    if not command_path.is_file():
        raise FileNotFoundError(f"Two-map ecalib command record is absent: {command_path}")
    recorded = command_path.read_text(encoding="utf-8").strip()
    expected = two_map_ecalib_argv(accepted_root, output)
    if _split_command(recorded, "two-map ecalib command") != expected:
        raise ValueError(
            "Recorded ecalib command differs from the reviewed diagnostic command "
            f"{format_command(expected)!r}: {recorded!r}"
        )
    # The maps must come from the accepted kspace_calib recorded at prepare;
    # a matching command text alone does not identify the input content.
    input_record = two_map_input_record(output, prepared)
    _finite_by_partition(_cfl_view(maps_base, 5, "two-map CSM"), "Two-map CSM")
    _finite_by_partition(_cfl_view(eigen_base, 5, "two-map eigenvalue maps"), "Eigenvalue maps")
    maps_record = cfl_record(maps_base)
    eigen_record = cfl_record(eigen_base)
    for key in ("bart_version", "bart_binary_hash", "ecalib_log"):
        if not (output / LAYOUT[key]).is_file():
            raise FileNotFoundError(f"Calibration log {LAYOUT[key]} is absent.")
    payload = {
        "format_version": FORMAT_VERSION,
        "status": "mprage_csm_two_map_calibration_recorded",
        "created_at_utc": _utc_now(),
        "implementation": implementation_identity(),
        "environment": environment_record(output, log_path),
        "prepare_manifest": rovir_feasibility._file_record(output / LAYOUT["prepare_manifest"]),
        "command": {
            "argv": expected,
            "text": recorded,
            "record": rovir_feasibility._file_record(command_path),
            "backend": "BART ecalib run by scripts/sample_mprage_csm_consistency.sh",
        },
        "map_count": 2,
        "crop_value": 0.0,
        "soft_sense": False,
        "diagnostic_only": True,
        "used_for_wave_reconstruction": False,
        "maps": maps_record,
        "eigenvalues": eigen_record,
        "accepted_kspace_calib": prepared["accepted"]["kspace_calib"],
        "ecalib_input": input_record,
        "bart_version": rovir_feasibility._version_record(output / LAYOUT["bart_version"]),
        "bart_binary": {
            **rovir_feasibility._file_record(output / LAYOUT["bart_binary_hash"]),
            "text": (output / LAYOUT["bart_binary_hash"]).read_text(encoding="utf-8").strip(),
        },
        "ecalib_log": rovir_feasibility._file_record(output / LAYOUT["ecalib_log"]),
        "flags": dict(_FALSE_FLAGS),
    }
    rovir_feasibility._write_json(manifest_path, payload)
    return payload


def _write_nifti(
    path: Path,
    values: np.ndarray,
    affine: np.ndarray,
    *,
    label_image: bool,
    intent_name: str = "CSMConsistencyROI",
) -> None:
    """Write one 3D NIfTI with matching qform and sform.

    This mirrors ``rovir_feasibility._write_geometry_bound_nifti`` but uses a
    workflow-specific label intent name and permits NaN in metric images.

    Args:
        path: Destination ``.nii.gz`` path.
        values: Three-dimensional array; label images must be finite.
        affine: Finite nonsingular voxel-to-RAS affine.
        label_image: Whether to store uint8 labels.
        intent_name: NIfTI intent name for label images.

    Raises:
        ValueError: If shape, values, or affine are invalid.

    Side Effects:
        Writes one NIfTI file.
    """
    import nibabel as nib

    array = np.asarray(values)
    matrix = np.asarray(affine, dtype=np.float64)
    if array.ndim != 3 or (label_image and not np.isfinite(array).all()):
        raise ValueError("NIfTI data must be three-dimensional; labels must be finite.")
    if matrix.shape != (4, 4) or abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-8:
        raise ValueError("NIfTI affine must be a finite nonsingular 4x4 matrix.")
    image = nib.Nifti1Image(array.astype(np.uint8 if label_image else np.float32), matrix)
    image.set_qform(matrix, code=1)
    image.set_sform(matrix, code=1)
    image.header.set_xyzt_units("mm")
    if label_image:
        image.header.set_intent("label", name=intent_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _accepted_image_magnitude(prepared: Mapping[str, Any]) -> np.ndarray:
    """Load the accepted FISTA-r0 magnitude on the BART logical grid.

    Args:
        prepared: Validated prepare manifest.

    Returns:
        Float32 magnitude array ``(RO, LIN, PAR)``.
    """
    base = Path(prepared["accepted"]["fista_r0_image"]["base"])
    return np.abs(np.asarray(_cfl_view(base, 3, "accepted FISTA-r0 image"))).astype(np.float32)


def _logical_voxel_size(prepared: Mapping[str, Any], shape: Sequence[int]) -> tuple[float, float, float]:
    """Return logical RO/LIN/PAR voxel sizes for a grid with the accepted FOV.

    Args:
        prepared: Validated prepare manifest.
        shape: Grid shape ``(RO, LIN, PAR)``.

    Returns:
        Voxel sizes in millimetres.
    """
    return rovir_feasibility._calibration_voxel_size_logical_mm(
        prepared["accepted"]["geometry"], shape
    )


def verify_roi_template(
    template: Mapping[str, Any], output_root: str | Path, accepted_root: str | Path
) -> None:
    """Verify a recorded ROI template against its files, inputs, and records.

    Args:
        template: Stored ROI-template manifest.
        output_root: Diagnostic output root.
        accepted_root: Accepted normal root holding the FISTA-r0 NIfTI.

    Raises:
        FileNotFoundError: If a template file is missing.
        ValueError: If the manifest is malformed, a template file changed,
            a recorded path is not the stable absolute path of its expected
            file or destination, the template was exported from another
            prepare manifest or reference image, or the accepted FISTA-r0
            NIfTI comparison no longer matches its record.
    """
    output = Path(output_root).expanduser().resolve()
    directory = output / LAYOUT["roi_template_directory"]
    for key, name, label in (
        ("reference_nifti", ROI_REFERENCE_NAME, "ROI reference NIfTI"),
        ("label_template_nifti", ROI_TEMPLATE_NAME, "ROI label template"),
        ("instructions", ROI_INSTRUCTIONS_NAME, "ROI instructions"),
    ):
        _verify_file(template.get(key), directory / name, label)
    # The destination is printed on reuse and may not exist yet, so it is
    # bound by its exact stable spelling instead of by file identity.
    destination = template.get("reviewed_label_destination")
    if destination != str(output / LAYOUT["roi_reviewed_labels"]) or not _stable_path(destination):
        raise ValueError(
            f"The ROI-template manifest names reviewed-label destination {destination!r}, "
            f"not {output / LAYOUT['roi_reviewed_labels']}."
        )
    _verify_environment(template, output, "The ROI-template manifest")
    geometry = _mapping(template, "geometry", "The ROI-template manifest")
    if geometry.get("source_manifest_sha256") != sha256_file(output / LAYOUT["prepare_manifest"]):
        raise ValueError("The ROI template was exported from a different prepare manifest.")
    if geometry.get("reference_sha256") != template["reference_nifti"]["sha256"]:
        raise ValueError("The ROI template geometry does not describe its reference image.")
    try:
        current = _accepted_nifti_geometry(
            accepted_artifact_paths(accepted_root)["fista_r0_nifti_directory"],
            [int(value) for value in geometry["stored_shape"]],
            np.asarray(geometry["stored_affine"], dtype=np.float64),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The ROI-template geometry record is malformed.") from exc
    if _json_ready(current) != template.get("accepted_fista_r0_nifti_geometry"):
        raise ValueError(
            "The accepted FISTA-r0 NIfTI changed after the ROI template was exported; move "
            "the ROI template and its manifest aside and export it again."
        )


def write_roi_template(
    twix: str | Path,
    sequence: str | Path,
    accepted_root: str | Path,
    output_root: str | Path,
    *,
    environment_log: str | Path | None = None,
) -> dict[str, Any]:
    """Export a geometry-bound reference image and an empty five-label template.

    Args:
        twix: Measured TWIX file used only for its geometry header.
        sequence: Matching sequence.
        accepted_root: Accepted normal root providing the FISTA-r0 reference.
        output_root: Diagnostic output root.
        environment_log: Shell log of this invocation under ``logs/environment``;
            recorded only when a new manifest is written.

    Returns:
        ROI-template manifest; an existing manifest is returned only after
        :func:`verify_roi_template` has verified it.

    Raises:
        FileExistsError: If an unrecognized template directory exists.
        ValueError: If the orientation round trip is not the identity or an
            existing template changed.

    Side Effects:
        Atomically writes ``rois/template`` and ``manifests/roi_template.json``.
    """
    output = Path(output_root).expanduser().resolve()
    log_path = environment_log_path(output, environment_log)
    prepared = load_prepared_inputs(twix, sequence, accepted_root, output)
    manifest_path = output / LAYOUT["roi_template_manifest"]
    destination = output / LAYOUT["roi_template_directory"]
    if manifest_path.is_file():
        existing = rovir_feasibility._read_json(manifest_path)
        verify_roi_template(existing, output, accepted_root)
        return existing
    if destination.exists():
        raise FileExistsError(f"Unrecognized ROI template directory exists: {destination}")

    magnitude = _accepted_image_magnitude(prepared)
    voxel_size = _logical_voxel_size(prepared, magnitude.shape)
    helpers = mprage_module.load_wave_mprage_helpers()
    source_affine, _, twix_info = helpers.make_nifti_affine_from_twix(
        twix_file=str(Path(twix).expanduser().resolve()),
        npy_shape=magnitude.shape,
        twix_array_axis_roles=rovir_feasibility.MPRAGE_NIFTI_AXIS_ROLES,
        twix_array_axis_flips=rovir_feasibility.MPRAGE_NIFTI_AFFINE_AXIS_FLIPS,
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=voxel_size,
    )
    flips = rovir_feasibility.MPRAGE_NIFTI_ARRAY_AXIS_FLIPS
    round_trip = roi.orientation_round_trip(magnitude.shape, flips, source_affine, helpers)
    if not round_trip["identity"]:
        raise ValueError("ROI orientation round trip is not the identity; refusing to export.")
    scale = roi.display_scale(magnitude)
    stored, canonical_affine, transform = roi.to_stored_orientation(
        magnitude / np.float32(scale["scale"]), flips, source_affine, helpers
    )
    accepted_nifti = _accepted_nifti_geometry(
        accepted_artifact_paths(accepted_root)["fista_r0_nifti_directory"],
        stored.shape,
        canonical_affine,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".roi-template-", dir=destination.parent))
    try:
        reference_path = staging / ROI_REFERENCE_NAME
        template_path = staging / ROI_TEMPLATE_NAME
        _write_nifti(reference_path, stored, canonical_affine, label_image=False)
        _write_nifti(template_path, np.zeros(stored.shape, np.uint8), canonical_affine, label_image=True)
        instructions = staging / ROI_INSTRUCTIONS_NAME
        instructions.write_text(_roi_instructions(), encoding="utf-8")
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    reference_record = rovir_feasibility._file_record(destination / reference_path.name)
    geometry = roi.roi_geometry_record(
        bart_shape=magnitude.shape,
        stored_shape=stored.shape,
        stored_affine=canonical_affine,
        source_affine=source_affine,
        array_flips=flips,
        orientation_transform=transform,
        reference_sha256=reference_record["sha256"],
        source_manifest_sha256=sha256_file(output / LAYOUT["prepare_manifest"]),
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "status": "mprage_csm_consistency_roi_template_ready",
        "created_at_utc": _utc_now(),
        "implementation": implementation_identity(),
        "environment": environment_record(output, log_path),
        "geometry": geometry,
        "orientation_round_trip": round_trip,
        "display_scale": scale,
        "accepted_fista_r0_nifti_geometry": accepted_nifti,
        "twix_orientation": twix_info,
        "label_definitions": {str(key): value for key, value in roi.ROI_LABELS.items()},
        "label_descriptions": {str(key): value for key, value in roi.ROI_LABEL_DESCRIPTIONS.items()},
        "reference_nifti": reference_record,
        "label_template_nifti": rovir_feasibility._file_record(destination / template_path.name),
        "instructions": rovir_feasibility._file_record(destination / ROI_INSTRUCTIONS_NAME),
        "reviewed_label_destination": str(output / LAYOUT["roi_reviewed_labels"]),
        "automatic_roi_detection": False,
        "flags": dict(_FALSE_FLAGS),
    }
    rovir_feasibility._write_json(manifest_path, payload)
    return payload


def _accepted_nifti_geometry(
    directory: Path, stored_shape: Sequence[int], canonical_affine: np.ndarray
) -> dict[str, Any]:
    """Compare the template geometry with the accepted FISTA-r0 magnitude NIfTI.

    Args:
        directory: Accepted ``normal/nifti/fista_r0`` directory.
        stored_shape: Template stored shape.
        canonical_affine: Template RAS affine.

    Returns:
        Comparison record; ``available`` is false when no unique NIfTI exists.
    """
    import nibabel as nib

    matches = sorted(directory.rglob("*_part-mag_*.nii.gz")) if directory.is_dir() else []
    if len(matches) != 1:
        return {"available": False, "candidates": len(matches)}
    image = nib.load(str(matches[0]))
    difference = float(np.max(np.abs(np.asarray(image.affine) - np.asarray(canonical_affine))))
    return {
        "available": True,
        "file": rovir_feasibility._file_record(matches[0]),
        "shape_matches": tuple(image.shape) == tuple(stored_shape),
        "max_abs_affine_difference_mm": difference,
        "affine_matches": difference <= 1e-3,
    }


def _roi_instructions() -> str:
    """Return the five-label annotation instructions.

    Returns:
        Plain-text instructions written beside the template.
    """
    lines = [
        "Set-4 CSM consistency ROI annotation",
        "",
        f"Open {ROI_REFERENCE_NAME} and",
        f"{ROI_TEMPLATE_NAME} together.",
        "",
    ]
    for value, name in roi.ROI_LABELS.items():
        lines.append(f"Label {value} ({name}): {roi.ROI_LABEL_DESCRIPTIONS.get(value, name)}")
    lines += [
        "",
        "Draw background air (label 5) superior to the scalp so that no head",
        "signal shares its readout rows. Do not resample, crop, reorient, or",
        "change the affine. Save the completed map as",
        "rois/reviewed/csm_consistency_roi_labels_reviewed.nii.gz, or pass",
        "equivalent inclusive BART-index boxes with --roi-box.",
    ]
    return "\n".join(lines) + "\n"


def load_reviewed_rois(
    output_root: str | Path,
    bart_shape: Sequence[int],
    *,
    labels_path: str | Path | None = None,
    boxes: Sequence[str] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load reviewed five-label ROIs from a NIfTI label map or explicit boxes.

    Args:
        output_root: Diagnostic output root with an ROI-template manifest.
        bart_shape: Accepted BART logical grid ``(RO, LIN, PAR)``.
        labels_path: Reviewed label NIfTI drawn on the exported template.
        boxes: Inclusive ``LABEL=ro=a:b,lin=c:d,par=e:f`` specifications.

    Returns:
        ``(labels, record)`` with uint8 labels on the BART grid and provenance.

    Raises:
        ValueError: If neither or both ROI sources are given, the stored label
            values are not finite integers in 0-5, or validation fails.
    """
    import nibabel as nib

    if (labels_path is None) == (not boxes):
        raise ValueError("Provide exactly one of a reviewed label NIfTI or ROI boxes.")
    output = Path(output_root).expanduser().resolve()
    if boxes:
        labels, record = roi.labels_from_boxes(boxes, bart_shape)
        source: dict[str, Any] = {"source": "boxes", "boxes": record}
    else:
        template = rovir_feasibility._read_json(output / LAYOUT["roi_template_manifest"])
        geometry = template["geometry"]
        try:
            image = nib.load(str(Path(labels_path).expanduser().resolve()))
            values = np.asanyarray(image.dataobj)
        except (nib.filebasedimages.ImageFileError, nib.spatialimages.HeaderDataError) as exc:
            raise ValueError(f"The reviewed label NIfTI cannot be read: {labels_path}: {exc}") from exc
        roi.validate_roi_geometry(values.shape, image.affine, geometry)
        # Validate the stored values exactly as read, before any narrowing:
        # casting before this check would wrap values such as 257 into labels.
        roi.validate_label_volume(values, values.shape)
        labels = roi.to_bart_orientation(
            np.asarray(values).astype(np.uint8),
            geometry["array_flips"],
            np.asarray(geometry["source_affine"], dtype=np.float64),
        )
        source = {
            "source": "label_nifti",
            "file": rovir_feasibility._file_record(labels_path),
            "roi_template_manifest_sha256": sha256_file(output / LAYOUT["roi_template_manifest"]),
        }
    labels = np.asarray(labels).astype(np.uint8)
    validation = roi.validate_label_volume(labels, bart_shape)
    return labels, {**source, "validation": validation}


def recover_accepted_pca_basis(
    physical_acs: np.ndarray,
    accepted_acs: np.ndarray,
    virtual_coils: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover the accepted PCA basis, including its per-channel phases.

    The accepted normal preparation compresses ``X @ W`` with ``W`` from
    the upstream ``estimate_cc_matrix_coillast``; only singular values are
    stored, so the basis is recomputed from the physical set-4 ACS and each
    virtual channel's arbitrary unit phase is fitted against the accepted
    compressed ACS.

    Args:
        physical_acs: Physical-coil set-4 ACS block ``(RO, n, n, Nc)``.
        accepted_acs: Accepted compressed ACS block ``(RO, n, n, Ncc)``.
        virtual_coils: Accepted virtual-coil count.

    Returns:
        ``(W, record)`` where compressed data equal ``X @ W`` and the record
        holds the relative residual and phases.

    Raises:
        ValueError: If shapes disagree or the relative residual exceeds
            :data:`PCA_BASIS_RELATIVE_RESIDUAL_LIMIT`.
    """
    helpers = mprage_module.load_wave_mprage_helpers()
    physical = np.asarray(physical_acs, dtype=np.complex64)
    accepted = np.asarray(accepted_acs, dtype=np.complex64)
    if physical.shape[:3] != accepted.shape[:3] or accepted.shape[3] != virtual_coils:
        raise ValueError("Physical and accepted ACS blocks have incompatible shapes.")
    basis, singular_values, retained = helpers.estimate_cc_matrix_coillast(
        physical, ncc=int(virtual_coils), acs=physical.shape[1], x_step=1
    )
    basis = np.asarray(basis, dtype=np.complex128)
    samples = physical.reshape(-1, physical.shape[-1]).astype(np.complex128)
    target = accepted.reshape(-1, virtual_coils).astype(np.complex128)
    candidate = samples @ basis
    phases = np.ones(virtual_coils, dtype=np.complex128)
    for index in range(virtual_coils):
        inner = np.vdot(candidate[:, index], target[:, index])
        if abs(inner) > 0:
            phases[index] = inner / abs(inner)
    fitted = basis * phases[None, :]
    residual = float(
        np.linalg.norm(samples @ fitted - target) / max(np.linalg.norm(target), 1e-30)
    )
    if residual > PCA_BASIS_RELATIVE_RESIDUAL_LIMIT:
        raise ValueError(
            f"Recomputed PCA basis does not reproduce the accepted compressed ACS "
            f"(relative residual {residual:.3e})."
        )
    return fitted, {
        "method": "upstream estimate_cc_matrix_coillast recomputed on the physical set-4 ACS",
        "relative_residual": residual,
        "residual_limit": PCA_BASIS_RELATIVE_RESIDUAL_LIMIT,
        "retained_energy": float(np.asarray(retained)[int(virtual_coils) - 1]),
        "leading_singular_values": [float(value) for value in np.asarray(singular_values)[:virtual_coils]],
        "phase_angles_rad": [float(np.angle(value)) for value in phases],
    }


def _embedded_window_power(
    grid_shape: Sequence[int], acs_window: np.ndarray
) -> tuple[np.ndarray, slice, slice]:
    """Embed the squared ACS k-space window on the image LIN/PAR grid.

    Args:
        grid_shape: Image grid ``(RO, LIN, PAR)``.
        acs_window: Real nonnegative ``(nacs, nacs)`` weights of the measured
            ACS block.

    Returns:
        ``(power, lin_block, par_block)`` where ``power`` is the float64
        ``(LIN, PAR)`` array of squared weights, zero outside the block.

    Raises:
        ValueError: If the window is not a finite nonnegative square array
            that fits the grid.
    """
    weights = np.asarray(acs_window, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"The ACS window must be square; got shape {weights.shape}.")
    if not np.isfinite(weights).all() or (weights < 0).any() or not weights.any():
        raise ValueError("The ACS window must be finite, nonnegative, and nonzero.")
    _, nlin, npar = (int(value) for value in grid_shape)
    lin_block, par_block = metrics.acs_block_slices(nlin, npar, int(weights.shape[0]))
    power = np.zeros((nlin, npar), dtype=np.float64)
    power[lin_block, par_block] = weights**2
    return power, lin_block, par_block


def effective_air_samples(air_mask: np.ndarray, acs_window: np.ndarray) -> float:
    """Return the effective number of independent noise vectors in an air covariance.

    The air covariance is the mean of ``d d^H`` over the mask. For white
    k-space noise weighted by ``acs_window`` inside the measured ACS block
    and zero-filled onto the image LIN/PAR grid, image noise is independent
    between RO rows (the readout is fully sampled, unweighted, and
    transformed orthonormally) and has the circular LIN/PAR correlation
    kernel ``K = ifft2(|w|^2)``. The estimate is then a weighted sum of
    independent rank-one Wishart terms with weights ``g``, and the returned
    value is ``(sum g)^2 / sum g^2 = M^2 K(0)^2 / sum_rows sum_D |K(D)|^2
    A_row(D)``, where ``A_row`` is the circular autocorrelation of the mask
    in one RO row and ``M`` the number of masked voxels. It equals ``M`` for
    independent voxels and ``(sum w^2)^2 / sum w^4`` per fully masked row.

    Args:
        air_mask: Boolean mask on the ``(RO, LIN, PAR)`` image grid.
        acs_window: Real nonnegative ``(nacs, nacs)`` weights of the measured
            ACS block, embedded with :func:`metrics.acs_block_slices`.

    Returns:
        Effective sample count; 0.0 for an empty mask.

    Raises:
        ValueError: If the mask is not three-dimensional or the window is
            invalid.
    """
    mask = np.asarray(air_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"The air mask must be (RO, LIN, PAR); got shape {mask.shape}.")
    voxels = int(np.count_nonzero(mask))
    if voxels == 0:
        return 0.0
    power, _, _ = _embedded_window_power(mask.shape, acs_window)
    kernel_power = np.abs(np.fft.ifft2(power)) ** 2
    kernel_zero = float(power.sum()) / power.size
    denominator = 0.0
    for row in np.flatnonzero(mask.any(axis=(1, 2))):
        plane = mask[row]
        if plane.all():
            # A fully masked row sums |K|^2 over every lag with weight LIN * PAR.
            denominator += float(kernel_power.sum()) * plane.size
            continue
        spectrum = np.fft.fft2(plane.astype(np.float64))
        autocorrelation = np.fft.ifft2(np.abs(spectrum) ** 2).real
        denominator += float(np.sum(kernel_power * autocorrelation))
    return float((voxels * kernel_zero) ** 2 / denominator)


def sampling_null_reference(
    reference: np.ndarray,
    air_mask: np.ndarray,
    acs_window: np.ndarray,
    *,
    repeats: int = SAMPLING_NULL_REPEATS,
    seed: int = SAMPLING_NULL_SEED,
) -> dict[str, Any] | None:
    """Describe compatibility-criterion values expected from sampling alone.

    The simulation reproduces the estimator behind the candidate covariance.
    Complex Gaussian noise with the reference channel covariance is drawn
    independently for every measured sample of the ACS block, weighted by
    ``acs_window``, zero-filled onto the image LIN/PAR grid, transformed with
    the centered orthonormal inverse FFT, and averaged as ``d d^H`` over
    ``air_mask``. Because the readout is fully sampled, unweighted, and
    transformed orthonormally, RO rows are independent and are drawn
    directly in the image domain; a fully masked row uses the equivalent
    Parseval sum over k-space samples. Each simulated estimate is compared
    with the reference using the same criteria as the gate. The gate itself
    is unchanged; the reference only shows whether a failed criterion could
    plausibly come from the finite, window-correlated sample under the
    white-noise approximation.

    Args:
        reference: Hermitian positive-definite reference covariance.
        air_mask: Boolean air mask on the ``(RO, LIN, PAR)`` image grid.
        acs_window: Real nonnegative ``(nacs, nacs)`` ACS k-space weights;
            all ones for unapodized data.
        repeats: Number of simulated draws.
        seed: Random seed.

    Returns:
        Median and maximum of each criterion and of the trace ratio over the
        draws that gave a positive-definite estimate, with the masked voxel
        count, the effective sample count, the window power gain ``K(0)``
        (the expected trace ratio for identical per-sample noise), and the
        number of singular draws; ``None`` when the mask holds no more voxels
        than channels.
    """
    matrix = np.asarray(reference, dtype=np.complex128)
    matrix = 0.5 * (matrix + matrix.conj().T)
    channels = int(matrix.shape[0])
    mask = np.asarray(air_mask, dtype=bool)
    voxels = int(np.count_nonzero(mask))
    if voxels <= channels:
        return None
    power, lin_block, par_block = _embedded_window_power(mask.shape, acs_window)
    weights = np.sqrt(power[lin_block, par_block])
    factor = np.linalg.cholesky(matrix)
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(mask.any(axis=(1, 2)))
    values: dict[str, list[float]] = {
        name: [] for name in twix_noise.DEFAULT_COVARIANCE_COMPATIBILITY_LIMITS
    }
    trace_ratios: list[float] = []
    singular = 0
    grid = np.zeros(power.shape + (channels,), dtype=np.complex128)
    for _ in range(repeats):
        accumulator = np.zeros((channels, channels), dtype=np.complex128)
        for row in rows:
            white = rng.standard_normal(weights.shape + (channels,)) + 1j * rng.standard_normal(
                weights.shape + (channels,)
            )
            samples = ((white / np.sqrt(2.0)) @ factor.T) * weights[..., None]
            plane = mask[row]
            if plane.all():
                flat = samples.reshape(-1, channels)
                accumulator += flat.T @ flat.conj()
                continue
            grid[...] = 0.0
            grid[lin_block, par_block] = samples
            image = metrics.centered_ifft(grid, axes=(0, 1))[plane]
            accumulator += image.T @ image.conj()
        try:
            comparison = twix_noise.compare_covariances(matrix, accumulator / voxels)
        except ValueError:
            singular += 1
            continue
        trace_ratios.append(float(comparison["trace_ratio"]))
        for name, record in twix_noise.covariance_compatibility(comparison)["criteria"].items():
            values[name].append(float(record["value"]))

    def summary(items: Sequence[float]) -> dict[str, float | None]:
        """Summarize simulated criterion values.

        Args:
            items: Values from the positive-definite draws.

        Returns:
            ``{"median", "max"}``, both ``None`` when no draw was usable.
        """
        if not items:
            return {"median": None, "max": None}
        return {"median": float(np.median(items)), "max": float(np.max(items))}

    return {
        "method": (
            "Monte Carlo of the air-covariance estimator: white complex Gaussian ACS noise "
            "with the reference covariance, the same k-space window, zero fill, orthonormal "
            "inverse FFT, and air mask"
        ),
        "voxels": voxels,
        "effective_samples": effective_air_samples(mask, weights),
        "window_power_gain": float(power.sum()) / power.size,
        "repeats": int(repeats),
        "seed": int(seed),
        "singular_draws": singular,
        "criteria": {name: summary(items) for name, items in values.items()},
        "trace_ratio": summary(trace_ratios),
        "note": (
            "Descriptive only; the pre-registered limits are unchanged, and the white-noise "
            "model ignores readout-filter and inter-sample noise correlations."
        ),
    }


def covariance_check(
    reference: np.ndarray,
    candidate: np.ndarray | None,
    air_mask: np.ndarray,
    acs_window: np.ndarray,
    *,
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    """Compare a noise-scan covariance with an empirical air covariance.

    Args:
        reference: Noise-scan covariance in the basis of the candidate.
        candidate: Empirical air covariance, or ``None`` when it could not be
            formed.
        air_mask: Air mask used for the candidate on its image grid.
        acs_window: ``(nacs, nacs)`` ACS k-space weights used for the
            candidate image.
        reference_label: Description of the reference.
        candidate_label: Description of the candidate.

    Returns:
        JSON-ready record with ``comparison``, ``compatibility``, and
        ``sampling_null_reference``. A comparison error, such as a singular
        air covariance, is recorded and makes the check incompatible instead
        of stopping the diagnostics.
    """
    record: dict[str, Any] = {"reference": reference_label, "candidate": candidate_label}
    try:
        if candidate is None:
            raise ValueError("No empirical air covariance could be formed.")
        comparison = twix_noise.compare_covariances(reference, candidate)
    except ValueError as exc:
        record.update(
            {
                "comparison": {"error": str(exc)},
                "compatibility": {
                    "compatible": False,
                    "criteria": {},
                    "failed_criteria": ["comparison_error"],
                },
                "sampling_null_reference": None,
            }
        )
        return record
    record.update(
        {
            "comparison": comparison,
            "compatibility": twix_noise.covariance_compatibility(comparison),
            "sampling_null_reference": sampling_null_reference(reference, air_mask, acs_window),
        }
    )
    return record


def rnr_status(
    noise_model: Mapping[str, Any] | None, rnr_basis_check: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Decide whether the conditional RNR may be read as calibrated.

    RNR is calibrated only when the physical-coil noise scan matches the
    native background-air covariance and the noise scan transformed into the
    accepted PCA basis matches the Hann air covariance that RNR divides by.

    Args:
        noise_model: Physical-coil check from :func:`covariance_check`, or
            ``None`` when no air covariance was available.
        rnr_basis_check: Accepted-basis check, or ``None``.

    Returns:
        ``{"status": "calibrated" | "uncalibrated", "failed_checks": [...]}``.
    """
    failed = [
        name
        for name, check in (
            ("physical_noise_model", noise_model),
            ("accepted_pca_basis", rnr_basis_check),
        )
        if not (check and check.get("compatibility", {}).get("compatible") is True)
    ]
    return {"status": "uncalibrated" if failed else "calibrated", "failed_checks": failed}


def transform_covariance(covariance: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Express a physical-coil covariance in a virtual-coil basis.

    Compressed samples are ``X @ W`` for row-vector samples, so each virtual
    vector is ``y = W^T x`` and its covariance is ``W^T Psi conj(W)``.

    Args:
        covariance: Physical-coil covariance ``Psi[i, j] = E[x_i conj(x_j)]``.
        basis: Compression matrix ``W`` shaped ``(physical, virtual)``.

    Returns:
        Hermitian virtual-coil covariance.
    """
    matrix = np.asarray(basis, dtype=np.complex128)
    transformed = matrix.T @ np.asarray(covariance, dtype=np.complex128) @ matrix.conj()
    return 0.5 * (transformed + transformed.conj().T)


def _nan_array(shape: Sequence[int]) -> np.ndarray:
    """Allocate a float32 metric array filled with NaN.

    Args:
        shape: Array shape.

    Returns:
        NaN-filled float32 array.
    """
    return np.full(tuple(shape), np.nan, dtype=np.float32)


def _accepted_grid_metrics(
    prepared: Mapping[str, Any],
    calibration: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute PCA-12 projection, eigenvalue, reproduction, and coherence maps.

    Args:
        prepared: Validated prepare manifest.
        calibration: Validated two-map calibration manifest.
        masks: Label masks on the accepted grid.

    Returns:
        Metric arrays, support masks, noise covariances, and QC records.
    """
    geometry = prepared["accepted"]["geometry"]
    nro, nlin, npar = (int(value) for value in geometry["logical_matrix_ro_lin_par"])
    coils = int(prepared["accepted"]["virtual_coils"])
    nacs = int(prepared["accepted"]["nacs"])
    kspace = np.asarray(
        _cfl_view(Path(prepared["accepted"]["kspace_calib"]["base"]), 4, "accepted kspace_calib")
    )
    window = metrics.acs_apodization_window((nro, nlin, npar), nacs)
    images = {
        "hann": metrics.coil_images(kspace, window),
        "unapodized": metrics.coil_images(kspace, None),
    }
    air = masks["background_air"]
    covariance: dict[str, np.ndarray] = {}
    snr: dict[str, np.ndarray] = {}
    for variant, values in images.items():
        covariance[variant], air_count = metrics.empirical_coil_covariance(values, air)
        snr[variant] = metrics.signal_to_noise_energy(
            values, float(np.real(np.trace(covariance[variant])))
        )
    support = {
        str(kappa): metrics.support_mask(snr["hann"], kappa)
        for kappa in (DEFAULT_SNR_KAPPA, *ROBUSTNESS_SNR_KAPPAS)
    }
    primary = support[str(DEFAULT_SNR_KAPPA)]
    maps = _cfl_view(Path(calibration["maps"]["base"]), 5, "two-map CSM")
    eigenvalues = metrics.eigenvalue_arrays(
        np.asarray(_cfl_view(Path(calibration["eigenvalues"]["base"]), 5, "eigenvalue maps"))
    )
    accepted_map = _cfl_view(Path(prepared["accepted"]["coil_sens"]["base"]), 4, "accepted CSM")
    shape = (nro, nlin, npar)
    arrays = {
        name: _nan_array(shape)
        for name in (
            "rho1_hann",
            "rho2_hann",
            "rho1_unapodized",
            "rho2_unapodized",
            "rnr1_hann",
            "rnr2_hann",
            "map1_reproduction_alpha",
            "map_orthonormality_error",
        )
    }
    rank2 = np.zeros(shape, dtype=np.int8)
    for start in range(0, npar, PAR_CHUNK):
        part = slice(start, min(npar, start + PAR_CHUNK))
        block = np.asarray(maps[:, :, part])
        basis1, _ = metrics.orthonormal_map_basis(block[..., :1], RANK_TOLERANCE)
        basis2, rank = metrics.orthonormal_map_basis(block, RANK_TOLERANCE)
        rank2[:, :, part] = rank
        mask = primary[:, :, part]
        for variant in ("hann", "unapodized"):
            values = images[variant][:, :, part]
            arrays[f"rho1_{variant}"][:, :, part] = metrics.projection_residual(values, basis1, mask)
            arrays[f"rho2_{variant}"][:, :, part] = metrics.projection_residual(values, basis2, mask)
        arrays["rnr1_hann"][:, :, part] = metrics.residual_to_noise_ratio(
            images["hann"][:, :, part], basis1, covariance["hann"], mask
        )
        arrays["rnr2_hann"][:, :, part] = metrics.residual_to_noise_ratio(
            images["hann"][:, :, part], basis2, covariance["hann"], mask
        )
        reproduction_mask = mask & (eigenvalues[:, :, part, 0] >= MAP_REPRODUCTION_LAMBDA1_MIN)
        arrays["map1_reproduction_alpha"][:, :, part] = metrics.map_reproduction(
            block[..., 0], np.asarray(accepted_map[:, :, part]), reproduction_mask
        )
        arrays["map_orthonormality_error"][:, :, part] = metrics.map_orthonormality_error(block)

    coherence_mask = primary & (eigenvalues[..., 0] >= COHERENCE_LAMBDA1_MIN)
    arrays["coherence_c1"] = _nan_array(shape)
    arrays["coherence_c2"] = _nan_array(shape)
    # Coherence needs face neighbours, so each PAR slab is read with a one-plane halo.
    for start in range(0, npar, PAR_CHUNK):
        stop = min(npar, start + PAR_CHUNK)
        low, high = max(0, start - 1), min(npar, stop + 1)
        block = np.asarray(maps[:, :, low:high])
        basis2, rank = metrics.orthonormal_map_basis(block, RANK_TOLERANCE)
        c1, c2 = metrics.csm_coherence(block[..., 0], basis2, rank, coherence_mask[:, :, low:high])
        keep = slice(start - low, start - low + (stop - start))
        arrays["coherence_c1"][:, :, start:stop] = c1[:, :, keep]
        arrays["coherence_c2"][:, :, start:stop] = c2[:, :, keep]
    arrays["lambda1"] = eigenvalues[..., 0].astype(np.float32)
    arrays["lambda2"] = eigenvalues[..., 1].astype(np.float32)
    arrays["eigenvalue_gap"] = metrics.eigenvalue_gap(eigenvalues).astype(np.float32)
    arrays["map_switching"] = metrics.map_switching_mask(
        arrays["coherence_c1"], arrays["coherence_c2"], arrays["lambda1"], arrays["lambda2"]
    ).astype(np.float32)
    arrays["log10_snr_hann"] = np.log10(np.maximum(snr["hann"], 1e-12)).astype(np.float32)
    rss = np.sqrt(np.sum(np.abs(images["hann"]) ** 2, axis=-1)).astype(np.float32)
    alpha = arrays["map1_reproduction_alpha"]
    return {
        "arrays": arrays,
        "support": support,
        "primary_support": primary,
        "snr_hann": snr["hann"],
        "rss_hann": rss,
        "two_map_rank": rank2,
        "air_covariance": covariance,
        "air_voxels": int(air_count),
        "eigenvalue_qc_all": metrics.eigenvalue_qc(eigenvalues),
        "eigenvalue_qc_support": metrics.eigenvalue_qc(eigenvalues, primary),
        "rank_deficient_two_map_voxels": int(np.count_nonzero((rank2 < 2) & primary)),
        "map1_reproduction": {
            "voxels": int(np.count_nonzero(np.isfinite(alpha))),
            "median": _nanquantile(alpha, 0.5),
            "q01": _nanquantile(alpha, 0.01),
            "lambda1_min": MAP_REPRODUCTION_LAMBDA1_MIN,
            "expected_median_at_least": 0.999,
        },
    }


def _nanquantile(values: np.ndarray, quantile: float) -> float | None:
    """Return a NaN-aware quantile or ``None`` for an empty selection.

    Args:
        values: Array possibly containing NaN.
        quantile: Quantile in ``[0, 1]``.

    Returns:
        Quantile as ``float`` or ``None``.
    """
    finite = np.asarray(values)[np.isfinite(values)]
    return None if finite.size == 0 else float(np.quantile(finite, quantile))


def _native_grid_metrics(
    prepared: Mapping[str, Any],
    output: Path,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute physical-coil and PCA-12 local rank on the native ACS grid.

    Args:
        prepared: Validated prepare manifest.
        output: Diagnostic output root.
        masks: Label and derived masks on the accepted grid.

    Returns:
        Native-grid masks, local-rank arrays, PCA-basis reproduction, and the
        conditional noise-model comparison.
    """
    geometry = prepared["accepted"]["geometry"]
    nro, nlin, npar = (int(value) for value in geometry["logical_matrix_ro_lin_par"])
    nacs = int(prepared["accepted"]["nacs"])
    ncalib = int(prepared["accepted"]["ncalib"])
    coils = int(prepared["accepted"]["virtual_coils"])
    physical = _cfl_view(output / LAYOUT["physical_calibration"], 4, "physical set-4 ACS")
    lin_block, par_block = metrics.acs_block_slices(ncalib, ncalib, nacs)
    native_kspace = np.asarray(physical[:, lin_block, par_block, :])
    native_images = metrics.coil_images(native_kspace, None)
    native_shape = (nro, nacs, nacs)
    native_masks = {
        name: roi.map_mask_to_grid(mask, native_shape)[0] for name, mask in masks.items()
    }
    accepted_kspace = _cfl_view(
        Path(prepared["accepted"]["kspace_calib"]["base"]), 4, "accepted kspace_calib"
    )
    full_lin, full_par = metrics.acs_block_slices(nlin, npar, nacs)
    basis, basis_record = recover_accepted_pca_basis(
        native_kspace, np.asarray(accepted_kspace[:, full_lin, full_par, :]), coils
    )
    bases = {
        "physical": native_images,
        "pca": np.einsum("...c,cv->...v", native_images, basis.astype(np.complex64)),
    }
    scan_covariance = np.load(output / LAYOUT["noise_covariance"], allow_pickle=False)
    results: dict[str, Any] = {
        "masks": native_masks,
        "pca_basis": basis_record,
        "pca_basis_matrix": basis,
        "scan_covariance": scan_covariance,
        "bases": {},
    }
    for name, values in bases.items():
        record: dict[str, Any] = {}
        try:
            air_covariance, air_count = metrics.empirical_coil_covariance(
                values, native_masks["background_air"]
            )
        except ValueError as exc:
            # Every basis must yield SNR and local rank, so the output set stays
            # the fixed contract; a too-small air ROI stops the stage.
            raise ValueError(
                f"The background-air ROI is too small on the native set-4 grid for the {name} "
                f"coil basis ({exc}); draw a larger background-air region."
            ) from exc
        record["air_voxels"] = int(air_count)
        if name == "physical":
            # Native voxels come from the unweighted ACS block without zero fill,
            # so the sampling null uses an all-ones window on the native grid.
            record["noise_model"] = covariance_check(
                scan_covariance,
                air_covariance,
                native_masks["background_air"],
                np.ones((nacs, nacs)),
                reference_label="measurement-0 noise scan covariance (physical coils)",
                candidate_label="empirical background-air covariance of the native set-4 image",
            )
        snr = metrics.signal_to_noise_energy(values, float(np.real(np.trace(air_covariance))))
        support = metrics.support_mask(snr, DEFAULT_SNR_KAPPA)
        record["support_voxels"] = int(np.count_nonzero(support))
        record["log10_snr"] = np.log10(np.maximum(snr, 1e-12)).astype(np.float32)
        record["local_rank"] = {
            label: metrics.local_coil_rank(
                values, support, neighborhood=size, min_members=LOCAL_RANK_MIN_MEMBERS
            )
            for label, size in LOCAL_RANK_NEIGHBORHOODS.items()
        }
        record["rss"] = np.sqrt(np.sum(np.abs(values) ** 2, axis=-1)).astype(np.float32)
        results["bases"][name] = record
    return results


def _rnr_basis_check(
    native: Mapping[str, Any],
    accepted: Mapping[str, Any],
    air_mask: np.ndarray,
    nacs: int,
) -> dict[str, Any]:
    """Compare the covariance used by RNR with the noise scan in the same basis.

    RNR uses the Hann-apodized background-air covariance of the accepted PCA
    image. Its shape is compared with the noise-scan covariance transformed
    into the recovered accepted basis; the scalar window and grid factors
    cancel under trace normalization. The Hann window and zero fill correlate
    neighbouring air voxels, so the sampling null simulates that estimator
    and reports its effective sample count instead of the voxel count.

    Args:
        native: Native-grid bundle with the noise-scan covariance and basis.
        accepted: Accepted-grid bundle with the air covariances.
        air_mask: Background-air mask on the accepted grid.
        nacs: ACS edge length.

    Returns:
        Record from :func:`covariance_check`.
    """
    reference = transform_covariance(native["scan_covariance"], native["pca_basis_matrix"])
    profile = metrics.dc_centered_hann(nacs)
    return covariance_check(
        reference,
        accepted["air_covariance"]["hann"],
        air_mask,
        np.outer(profile, profile),
        reference_label="noise-scan covariance transformed into the recovered accepted PCA basis",
        candidate_label="Hann-apodized background-air covariance of the accepted PCA image (used by RNR)",
    )


def _derived_masks(
    prepared: Mapping[str, Any],
    output: Path,
    masks: Mapping[str, np.ndarray],
    snr: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the pre-registered alias-partner and edge-matched control masks.

    Args:
        prepared: Validated prepare manifest.
        output: Diagnostic output root with the ROI template.
        masks: Label masks on the accepted grid.
        snr: Primary coil-image SNR energy on the accepted grid.

    Returns:
        ``(derived masks, records)`` for the partner band, head mask, and the
        edge-matched control with its SNR-matched subsample.
    """
    import nibabel as nib

    from .nifti_collection import create_whole_head_mask

    template = rovir_feasibility._read_json(output / LAYOUT["roi_template_manifest"])
    geometry = template["geometry"]
    reference = nib.load(template["reference_nifti"]["path"])
    head_image, head_record = create_whole_head_mask(reference)
    head = roi.to_bart_orientation(
        np.asarray(head_image.dataobj).astype(np.uint8),
        geometry["array_flips"],
        np.asarray(geometry["source_affine"], dtype=np.float64),
    ).astype(bool)
    metal = masks["metal"]
    voxel_size = _logical_voxel_size(prepared, metal.shape)
    edge = roi.edge_matched_control(head, metal, voxel_size)
    matched, matched_record = roi.snr_matched_mask(
        edge, metal, snr, bins=EDGE_CONTROL_SNR_BINS, seed=EDGE_CONTROL_SEED
    )
    derived = {
        "alias_partner": roi.alias_partner_mask(metal),
        "edge_control": edge,
        "edge_control_snr_matched": matched,
    }
    return derived, {
        "head_mask": {**head_record, "voxels": int(np.count_nonzero(head))},
        "edge_control_voxels": int(np.count_nonzero(edge)),
        "edge_control_snr_matched": matched_record,
        "voxel_size_mm_ro_lin_par": list(voxel_size),
    }


def _support_mismatch(
    magnitude: np.ndarray, lambda1: np.ndarray, masks: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Share of accepted MPRAGE image energy where the FLASH map is unsupported.

    Args:
        magnitude: Accepted FISTA-r0 magnitude.
        lambda1: Map-1 ESPIRiT eigenvalue.
        masks: Masks to summarize.

    Returns:
        Per-mask descriptive energy fractions; circular by construction.
    """
    energy = np.asarray(magnitude, dtype=np.float64) ** 2
    unsupported = lambda1 < SUPPORT_MISMATCH_LAMBDA1
    result = {}
    for name in sorted(masks):
        total = float(np.sum(energy[masks[name]]))
        result[name] = None if total <= 0 else float(np.sum(energy[masks[name] & unsupported]) / total)
    return {
        "lambda1_threshold": SUPPORT_MISMATCH_LAMBDA1,
        "fractions": result,
        "note": "Descriptive only: the accepted image was reconstructed with the one-map CSM.",
    }


def _save_metric(path: Path, values: np.ndarray, records: dict[str, Any], root: Path) -> None:
    """Save one float32 metric array and record its hash.

    Args:
        path: Destination ``.npy`` path.
        values: Metric array.
        records: Output record mapping updated in place.
        root: Output root used for relative keys.

    Side Effects:
        Writes one NumPy file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float32), allow_pickle=False)
    records[str(path.relative_to(root))] = rovir_feasibility._file_record(path)


def _diagnostics_move_aside(output: Path) -> str:
    """Name every diagnose-owned path that must be moved aside to recompute.

    Args:
        output: Diagnostic output root.

    Returns:
        Comma-separated paths.
    """
    paths = [output / LAYOUT["diagnostics_manifest"], output / "diagnostics", output / "reports"]
    return ", ".join(str(path) for path in paths)


def _native_summary_names() -> set[str]:
    """List the native-grid summaries that diagnose always writes.

    Returns:
        ``log10_snr_<basis>`` and ``<e1|kappa2>_<basis>_<neighbourhood>`` names.
    """
    names = {f"log10_snr_{basis}" for basis in _NATIVE_BASES}
    names.update(
        f"{key}_{basis}_{label}"
        for basis in _NATIVE_BASES
        for label in LOCAL_RANK_NEIGHBORHOODS
        for key in ("e1", "kappa2")
    )
    return names


def _diagnostics_output_contract() -> set[str]:
    """List every output that diagnose always writes, from constants only.

    The contract does not depend on any manifest, so removing a summary, its
    output record, and its file together cannot shrink it.

    Returns:
        Output-root-relative paths of every metric array, local-rank array,
        NIfTI view, figure, and the CSV.
    """
    expected = {(_metric_directory(name) / f"{name}.npy").as_posix() for name in _ACCEPTED_METRICS}
    expected.update(
        (LAYOUT["local_rank"] / f"{name}.npy").as_posix()
        for name in _native_summary_names()
        if not name.startswith("log10_snr_")
    )
    expected.update(
        (_metric_directory(name) / f"{name}_ras.nii.gz").as_posix() for name in _NIFTI_METRICS
    )
    expected.update(relative.as_posix() for relative in _FIGURES.values())
    expected.add(LAYOUT["roi_summary_csv"].as_posix())
    return expected


def _require_summary_contract(manifest: Mapping[str, Any]) -> None:
    """Require every fixed metric summary over every ROI and control mask.

    Args:
        manifest: Diagnostics manifest or payload.

    Raises:
        ValueError: If a summary is missing or extra, or a summary does not
            cover exactly the fixed mask set.
    """
    for key, names in (
        ("roi_summaries", set(_ACCEPTED_METRICS)),
        ("native_grid_roi_summaries", _native_summary_names()),
    ):
        summaries = manifest.get(key)
        if not isinstance(summaries, Mapping) or set(summaries) != names:
            present = set(summaries) if isinstance(summaries, Mapping) else set()
            raise ValueError(
                f"The diagnostics {key} differ from the fixed metric set: missing "
                f"{sorted(names - present)}, unexpected {sorted(present - names)}."
            )
        for name, summary in summaries.items():
            if not isinstance(summary, Mapping) or set(summary) != set(_SUMMARY_MASKS):
                raise ValueError(
                    f"The diagnostics summary {key}[{name!r}] does not cover exactly the "
                    "fixed ROI and control masks."
                )


def verify_diagnostics(
    existing: Mapping[str, Any], output_root: str | Path, roi_record: Mapping[str, Any]
) -> None:
    """Verify an existing diagnostics manifest before it is reused.

    Args:
        existing: Stored diagnostics manifest.
        output_root: Diagnostic output root.
        roi_record: ROI provenance of the current invocation.

    Raises:
        FileExistsError: If the diagnostics were computed by a different
            implementation or recorded another ROI source, and must be moved
            aside to be recomputed.
        FileNotFoundError: If a recorded input or output is missing.
        ValueError: If the manifest is malformed, an input manifest, output,
            figure, CSV, or the report changed, an implied output is not
            recorded, or a diagnose-owned file is not recorded.
    """
    output = Path(output_root).expanduser().resolve()
    if existing.get("implementation") != implementation_identity():
        raise FileExistsError(
            "Existing diagnostics were computed by a different implementation; move "
            f"these aside before recomputing: {_diagnostics_move_aside(output)}"
        )
    _verify_environment(existing, output, "The diagnostics manifest")
    inputs = _mapping(existing, "inputs", "The diagnostics manifest")
    if inputs.get("rois") != json.loads(json.dumps(_json_ready(roi_record))):
        raise FileExistsError(
            "Existing diagnostics record another ROI source or label file; move these aside "
            f"before recomputing: {_diagnostics_move_aside(output)}"
        )
    for key, label in (
        ("prepare_manifest", "Prepare manifest"),
        ("calibration_manifest", "Two-map calibration manifest"),
        ("roi_template_manifest", "ROI-template manifest"),
    ):
        _verify_file(inputs.get(key), output / LAYOUT[key], f"{label} of the diagnostics")
    outputs = existing.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError("The diagnostics manifest records no outputs.")
    for relative, record in outputs.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Diagnostics output key {relative!r} is not a relative path.")
        _verify_file(record, output / relative_path, f"Diagnostics output {relative}")
    contract = _diagnostics_output_contract()
    if set(outputs) != contract:
        raise ValueError(
            "The diagnostics manifest does not record the fixed output contract: missing "
            f"{sorted(contract - set(outputs))}, unexpected {sorted(set(outputs) - contract)}."
        )
    _require_summary_contract(existing)
    # Every file the diagnose stage owns must be recorded, so a trimmed
    # manifest cannot hide deleted or extra outputs.
    present = {
        path.relative_to(output).as_posix()
        for owned in _DIAGNOSTICS_OWNED_PATHS
        for path in ([output / owned] if (output / owned).is_file() else sorted((output / owned).rglob("*")))
        if path.is_file()
    }
    unrecorded = sorted(present - set(outputs) - {LAYOUT["report"].as_posix()})
    if unrecorded:
        raise ValueError(f"Diagnose-owned files are not recorded in the manifest: {unrecorded}")
    figures = existing.get("figures")
    if not isinstance(figures, list):
        raise ValueError("The diagnostics manifest has no valid figure list.")
    recorded_figures = [figure.get("relative_path") for figure in figures if isinstance(figure, Mapping)]
    if sorted(recorded_figures) != sorted(relative.as_posix() for relative in _FIGURES.values()):
        raise ValueError("The diagnostics manifest does not list exactly the fixed figures.")
    for figure in figures:
        entry = outputs.get(figure.get("relative_path")) if isinstance(figure, Mapping) else None
        if not isinstance(entry, Mapping) or any(
            figure.get(key) != entry.get(key) for key in ("path", "size_bytes", "sha256")
        ):
            raise ValueError("A diagnostics figure record differs from its verified output record.")
    _verify_file(existing.get("report"), output / LAYOUT["report"], "Diagnostics report")


def diagnose_csm_consistency(
    twix: str | Path,
    sequence: str | Path,
    accepted_root: str | Path,
    output_root: str | Path,
    *,
    labels_path: str | Path | None = None,
    boxes: Sequence[str] = (),
    environment_log: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the calibration-only diagnostics, figures, and report.

    Args:
        twix: Measured TWIX file.
        sequence: Matching sequence.
        accepted_root: Accepted normal root.
        output_root: Diagnostic output root after prepare, calibrate, and
            roi-template stages.
        labels_path: Reviewed five-label NIfTI drawn on the exported template.
        boxes: Alternative inclusive BART-index ROI boxes.
        environment_log: Shell log of this invocation under ``logs/environment``;
            recorded only when a new manifest is written.

    Returns:
        Diagnostics manifest. An existing manifest is returned only when its
        ROI identity and implementation match and
        :func:`verify_diagnostics` has verified every recorded output.

    Raises:
        FileExistsError: If diagnostics exist for a different ROI identity or
            implementation, or diagnostics outputs exist without a manifest.
        FileNotFoundError: If a recorded input or output is missing.
        ValueError: If inputs, ROIs, or recorded artifacts are inconsistent.

    Side Effects:
        Writes metric arrays, NIfTI views, figures, CSV, the report, and the
        diagnostics manifest. Launches no process.
    """
    import nibabel as nib

    output = Path(output_root).expanduser().resolve()
    log_path = environment_log_path(output, environment_log)
    prepared = load_prepared_inputs(twix, sequence, accepted_root, output)
    calibration = rovir_feasibility._read_json(output / LAYOUT["calibration_manifest"])
    verify_two_map_calibration(calibration, prepared, accepted_root, output)
    template = rovir_feasibility._read_json(output / LAYOUT["roi_template_manifest"])
    verify_roi_template(template, output, accepted_root)
    geometry = prepared["accepted"]["geometry"]
    shape = tuple(int(value) for value in geometry["logical_matrix_ro_lin_par"])
    labels, roi_record = load_reviewed_rois(output, shape, labels_path=labels_path, boxes=boxes)
    roi_identity = sha256_bytes(np.ascontiguousarray(labels).tobytes())
    manifest_path = output / LAYOUT["diagnostics_manifest"]
    if manifest_path.is_file():
        existing = rovir_feasibility._read_json(manifest_path)
        if existing.get("roi_identity_sha256") != roi_identity:
            raise FileExistsError(
                "Diagnostics already exist for different ROIs; move these aside before "
                f"recomputing: {_diagnostics_move_aside(output)}"
            )
        verify_diagnostics(existing, output, roi_record)
        return existing
    leftovers = _existing_outputs(output, _DIAGNOSTICS_OWNED_PATHS)
    if leftovers:
        raise FileExistsError(
            "Diagnostics outputs exist without a diagnostics manifest; inspect and move "
            "them aside before rerunning: " + ", ".join(str(path) for path in leftovers)
        )

    label_masks = roi.label_masks(labels)
    accepted = _accepted_grid_metrics(prepared, calibration, label_masks)
    derived, derived_record = _derived_masks(prepared, output, label_masks, accepted["snr_hann"])
    masks = {**label_masks, **derived}
    native = _native_grid_metrics(prepared, output, masks)
    noise_model = native["bases"]["physical"].get("noise_model")
    rnr_basis_check = _rnr_basis_check(
        native, accepted, label_masks["background_air"], int(prepared["accepted"]["nacs"])
    )
    rnr_decision = rnr_status(noise_model, rnr_basis_check)

    arrays = accepted["arrays"]
    if set(arrays) != set(_ACCEPTED_METRICS):
        raise RuntimeError("Internal error: accepted-grid metrics differ from the output contract.")
    summaries: dict[str, Any] = {}
    threshold_by_metric = {"lambda1": (LAMBDA1_LOW_THRESHOLD,), "lambda2": EIGENVALUE_THRESHOLDS, "map_switching": (0.5,)}
    for name, values in sorted(arrays.items()):
        summaries[name] = roi.summarize_metric(values, masks, thresholds=threshold_by_metric.get(name, ()))
    native_summaries: dict[str, Any] = {}
    for basis_name, record in native["bases"].items():
        if "log10_snr" in record:
            # e1 has a noise floor, so it is read against the per-ROI SNR.
            native_summaries[f"log10_snr_{basis_name}"] = roi.summarize_metric(
                record["log10_snr"], native["masks"]
            )
        for label, rank in record.get("local_rank", {}).items():
            for key in ("e1", "kappa2"):
                native_summaries[f"{key}_{basis_name}_{label}"] = roi.summarize_metric(
                    rank[key], native["masks"]
                )
    magnitude = _accepted_image_magnitude(prepared)
    mismatch = _support_mismatch(magnitude, arrays["lambda1"], masks)
    partner = roi.partner_test(label_masks["fringe"], label_masks["metal"])

    outputs: dict[str, Any] = {}
    for name, values in sorted(arrays.items()):
        directory = _metric_directory(name)
        _save_metric(output / directory / f"{name}.npy", values, outputs, output)
    for basis_name, record in native["bases"].items():
        for label, rank in record.get("local_rank", {}).items():
            for key in ("e1", "kappa2"):
                _save_metric(
                    output / LAYOUT["local_rank"] / f"{key}_{basis_name}_{label}.npy",
                    rank[key],
                    outputs,
                    output,
                )
    _write_metric_niftis(output, template, arrays, outputs)
    figures = _write_figures(output, magnitude, template, accepted, native, masks, label_masks)
    for figure in figures:
        # Figures are outputs too, so reuse verifies them by relative path.
        figure["relative_path"] = str(Path(figure["path"]).relative_to(output))
        outputs[figure["relative_path"]] = {key: figure[key] for key in ("path", "size_bytes", "sha256")}
    rows = []
    for name, summary in sorted(summaries.items()):
        rows.extend(roi.summary_rows(name, summary))
    for name, summary in sorted(native_summaries.items()):
        rows.extend(roi.summary_rows(f"native_grid_{name}", summary))
    csv_path = output / LAYOUT["roi_summary_csv"]
    _write_rows(csv_path, rows)
    outputs[str(LAYOUT["roi_summary_csv"])] = rovir_feasibility._file_record(csv_path)
    if set(outputs) != _diagnostics_output_contract():
        raise RuntimeError("Internal error: diagnose outputs differ from the fixed output contract.")

    payload = {
        "format_version": FORMAT_VERSION,
        "status": "mprage_csm_consistency_diagnostics_ready",
        "created_at_utc": _utc_now(),
        "implementation": implementation_identity(),
        "environment": environment_record(output, log_path),
        "inputs": {
            "prepare_manifest": rovir_feasibility._file_record(output / LAYOUT["prepare_manifest"]),
            "calibration_manifest": rovir_feasibility._file_record(output / LAYOUT["calibration_manifest"]),
            "roi_template_manifest": rovir_feasibility._file_record(output / LAYOUT["roi_template_manifest"]),
            "rois": roi_record,
        },
        "roi_identity_sha256": roi_identity,
        "parameters": _parameter_record(),
        "noise_model": {
            **_json_ready(noise_model or {"available": False}),
            "rnr_basis_check": _json_ready(rnr_basis_check),
            "rnr_status": rnr_decision["status"],
            "rnr_failed_checks": rnr_decision["failed_checks"],
            "policy": RNR_POLICY,
            "held_out_noise_compatibility": prepared["noise"]["held_out_compatibility"],
            "expected_white_noise_variance_ratio_acs_over_noise": prepared["noise"][
                "expected_white_noise_variance_ratio_acs_over_noise"
            ],
            "air_voxels_accepted_grid": accepted["air_voxels"],
            "air_voxels_native_grid": native["bases"]["physical"]["air_voxels"],
        },
        "pca_basis_reproduction": native["pca_basis"],
        "map1_reproduction": accepted["map1_reproduction"],
        "eigenvalue_qc": {
            "all_voxels": accepted["eigenvalue_qc_all"],
            "primary_support": accepted["eigenvalue_qc_support"],
            "rank_deficient_two_map_support_voxels": accepted["rank_deficient_two_map_voxels"],
        },
        "support_voxels": {kappa: int(np.count_nonzero(mask)) for kappa, mask in accepted["support"].items()},
        "native_grid_support_voxels": {
            name: record.get("support_voxels") for name, record in native["bases"].items()
        },
        "derived_masks": derived_record,
        "roi_summaries": summaries,
        "native_grid_roi_summaries": native_summaries,
        "alias_partner_test": partner,
        "support_mismatch": mismatch,
        "figures": figures,
        "outputs": outputs,
        "interpretation_policy": INTERPRETATION_POLICY,
        "flags": dict(_FALSE_FLAGS),
    }
    try:
        _require_summary_contract(payload)
    except ValueError as exc:
        raise RuntimeError(f"Internal error: {exc}") from exc
    report_path = output / LAYOUT["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_markdown(payload, prepared, calibration), encoding="utf-8")
    payload["report"] = rovir_feasibility._file_record(report_path)
    rovir_feasibility._write_json(manifest_path, _json_ready(payload))
    return rovir_feasibility._read_json(manifest_path)


def sha256_bytes(payload: bytes) -> str:
    """Hash an in-memory byte string.

    Args:
        payload: Bytes to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _metric_directory(name: str) -> Path:
    """Choose the diagnostics subdirectory for one accepted-grid metric.

    Args:
        name: Metric name.

    Returns:
        Output-root-relative directory.
    """
    if name.startswith("coherence") or name == "map_switching":
        return LAYOUT["csm_coherence"]
    if name.startswith("log10_snr"):
        return LAYOUT["calibration_views"]
    return LAYOUT["projection_residuals"]


def _write_metric_niftis(
    output: Path,
    template: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    outputs: dict[str, Any],
) -> None:
    """Export key accepted-grid metrics as NIfTI aligned with the ROI template.

    Args:
        output: Diagnostic output root.
        template: ROI-template manifest with the forward geometry.
        arrays: Accepted-grid metric arrays.
        outputs: Output record mapping updated in place.

    Side Effects:
        Writes NIfTI files beside the NumPy metrics.
    """
    geometry = template["geometry"]
    helpers = mprage_module.load_wave_mprage_helpers()
    source_affine = np.asarray(geometry["source_affine"], dtype=np.float64)
    for name in _NIFTI_METRICS:
        stored, affine, _ = roi.to_stored_orientation(
            arrays[name], geometry["array_flips"], source_affine, helpers
        )
        path = output / _metric_directory(name) / f"{name}_ras.nii.gz"
        _write_nifti(path, stored, affine, label_image=False)
        outputs[str(path.relative_to(output))] = rovir_feasibility._file_record(path)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic CSV rows.

    Args:
        path: Destination CSV path.
        rows: Flat mappings with identical keys.

    Side Effects:
        Writes one CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parameter_record() -> dict[str, Any]:
    """Return every pre-registered descriptive default used by ``diagnose``.

    Returns:
        JSON-native parameter record.
    """
    return {
        "pre_registered_descriptive_defaults": True,
        "snr_kappa_primary": DEFAULT_SNR_KAPPA,
        "snr_kappa_robustness": list(ROBUSTNESS_SNR_KAPPAS),
        "snr_noise_source": "empirical background-air coil covariance trace",
        "rank_tolerance": RANK_TOLERANCE,
        "eigenvalue_thresholds": list(EIGENVALUE_THRESHOLDS),
        "lambda1_low_threshold": LAMBDA1_LOW_THRESHOLD,
        "map_reproduction_lambda1_min": MAP_REPRODUCTION_LAMBDA1_MIN,
        "coherence_lambda1_min": COHERENCE_LAMBDA1_MIN,
        "support_mismatch_lambda1": SUPPORT_MISMATCH_LAMBDA1,
        "apodization": {"primary": "dc-centered Hann over the ACS block", "secondary": "none"},
        "local_rank": {
            "grid": "native set-4 ACS grid, no zero filling",
            "neighborhoods": {key: list(value) for key, value in LOCAL_RANK_NEIGHBORHOODS.items()},
            "min_members": LOCAL_RANK_MIN_MEMBERS,
            "noise_debiasing": False,
        },
        "alias_partner": {
            "acceleration": roi.DEFAULT_ALIAS_ACCELERATION,
            "tolerance": roi.DEFAULT_ALIAS_TOLERANCE,
            "ro_dilations": [value for value in roi.DEFAULT_RO_DILATIONS],
            "status": "pre-registered testable hypothesis",
        },
        "edge_control": {
            "band_mm": roi.DEFAULT_EDGE_BAND_MM,
            "exclusion_mm": roi.DEFAULT_EDGE_EXCLUSION_MM,
            "snr_bins": EDGE_CONTROL_SNR_BINS,
            "seed": EDGE_CONTROL_SEED,
        },
        "display": {
            "percentile": roi.DEFAULT_DISPLAY_PERCENTILE,
            "windows": {key: list(value) for key, value in roi.FIXED_DISPLAY_WINDOWS.items()},
        },
        "noise_compatibility_limits": dict(twix_noise.DEFAULT_COVARIANCE_COMPATIBILITY_LIMITS),
        "par_chunk": PAR_CHUNK,
    }


def _json_ready(value: Any) -> Any:
    """Convert NumPy containers and scalars into JSON-native values.

    Array values larger than a scalar are omitted from manifests; they are
    stored as files instead.

    Args:
        value: Nested structure.

    Returns:
        JSON-compatible structure.
    """
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
            if not (isinstance(item, np.ndarray) and item.size > 1)
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_figures(
    output: Path,
    magnitude: np.ndarray,
    template: Mapping[str, Any],
    accepted: Mapping[str, Any],
    native: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    label_masks: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Write fixed-window review figures; maps are never combined into RSS only.

    Args:
        output: Diagnostic output root.
        magnitude: Accepted FISTA-r0 magnitude.
        template: ROI-template manifest (display scale).
        accepted: Accepted-grid metric bundle.
        native: Native-grid metric bundle.
        masks: All masks on the accepted grid.
        label_masks: Reviewed label masks on the accepted grid.

    Returns:
        File records of the written figures.

    Side Effects:
        Writes PNG files below ``diagnostics/``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = accepted["arrays"]
    focus = label_masks["metal"] | label_masks["fringe"]
    slices = roi.representative_slices(focus, axis=2, count=3)
    scale = roi.display_scale(magnitude, label_masks["preserved_anatomy"])["scale"]
    outline_styles = (("metal", "solid"), ("fringe", "dashed"), ("alias_partner", "dotted"), ("preserved_anatomy", "dashdot"))
    records = []

    def outlines(axis: Any, index: int) -> None:
        """Draw ROI outlines with distinct line styles on one sagittal panel.

        Args:
            axis: Matplotlib axis.
            index: PAR slice index.
        """
        for name, style in outline_styles:
            plane = masks[name][:, :, index]
            if plane.any():
                axis.contour(plane.astype(float), levels=[0.5], colors="white", linewidths=0.8, linestyles=style)

    panels = [
        ("reference magnitude", magnitude / scale, (0.0, 1.0), "gray"),
        ("lambda 1", arrays["lambda1"], roi.window_for("lambda"), "viridis"),
        ("lambda 2", arrays["lambda2"], roi.window_for("lambda"), "viridis"),
        ("rho 1 (Hann)", arrays["rho1_hann"], roi.window_for("rho"), "cividis"),
        ("rho 2 (Hann)", arrays["rho2_hann"], roi.window_for("rho"), "cividis"),
    ]
    figure, axes = plt.subplots(len(slices), len(panels), figsize=(3.2 * len(panels), 3.0 * len(slices)), squeeze=False)
    for row, index in enumerate(slices):
        for column, (title, values, window, cmap) in enumerate(panels):
            axis = axes[row][column]
            image = axis.imshow(roi.apply_display_window(values[:, :, index], window), vmin=window[0], vmax=window[1], cmap=cmap, origin="lower", aspect="auto")
            outlines(axis, index)
            axis.set_title(f"{title}, PAR {index}", fontsize=8)
            axis.set_xlabel("LIN index", fontsize=7)
            axis.set_ylabel("RO index", fontsize=7)
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("Fixed windows; outlines: metal solid, fringe dashed, partner dotted, anatomy dash-dot", fontsize=9)
    records.append(_save_figure(figure, output / _FIGURES["eigenvalues_and_residuals"]))

    maps = _cfl_view(Path(rovir_feasibility._read_json(output / LAYOUT["calibration_manifest"])["maps"]["base"]), 5, "two-map CSM")
    index = slices[len(slices) // 2]
    plane = np.asarray(maps[:, :, index])
    coils = min(4, plane.shape[2])
    figure, axes = plt.subplots(2, coils, figsize=(3.0 * coils, 6.0), squeeze=False)
    for map_index in range(2):
        for coil in range(coils):
            axis = axes[map_index][coil]
            image = axis.imshow(np.abs(plane[:, :, coil, map_index]), vmin=0.0, vmax=1.0, cmap="magma", origin="lower", aspect="auto")
            axis.set_title(f"|map {map_index + 1}|, virtual coil {coil}, PAR {index}", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("Two-map components shown separately (no RSS); window [0, 1]", fontsize=9)
    records.append(_save_figure(figure, output / _FIGURES["map_components"]))

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), squeeze=False)
    for column, (title, values, window) in enumerate(
        (
            ("coherence C1", arrays["coherence_c1"], roi.window_for("coherence")),
            ("coherence C2", arrays["coherence_c2"], roi.window_for("coherence")),
            ("log10 SNR energy (Hann)", arrays["log10_snr_hann"], (0.0, 4.0)),
        )
    ):
        axis = axes[0][column]
        image = axis.imshow(roi.apply_display_window(values[:, :, index], window), vmin=window[0], vmax=window[1], cmap="viridis", origin="lower", aspect="auto")
        outlines(axis, index)
        axis.set_title(f"{title}, PAR {index}", fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046)
    records.append(_save_figure(figure, output / _FIGURES["coherence_and_support"]))

    native_index = int(round((index - magnitude.shape[2] // 2) * native["masks"]["metal"].shape[2] / magnitude.shape[2] + native["masks"]["metal"].shape[2] // 2))
    native_index = min(max(native_index, 0), native["masks"]["metal"].shape[2] - 1)
    columns = [(name, record) for name, record in native["bases"].items() if "local_rank" in record]
    if columns:
        figure, axes = plt.subplots(1, 2 * len(columns), figsize=(3.2 * 2 * len(columns), 3.2), squeeze=False)
        for position, (name, record) in enumerate(columns):
            rss = record["rss"][:, :, native_index]
            axis = axes[0][2 * position]
            image = axis.imshow(rss / max(float(np.max(rss)), 1e-30), vmin=0.0, vmax=1.0, cmap="gray", origin="lower", aspect="auto")
            axis.set_title(f"{name} set-4 RSS (native), PAR {native_index}", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.046)
            axis = axes[0][2 * position + 1]
            window = roi.window_for("e1")
            image = axis.imshow(roi.apply_display_window(record["local_rank"]["ro5"]["e1"][:, :, native_index], window), vmin=window[0], vmax=window[1], cmap="viridis", origin="lower", aspect="auto")
            axis.set_title(f"{name} local e1 (RO 5), PAR {native_index}", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.046)
        records.append(_save_figure(figure, output / _FIGURES["local_rank_native"]))

    figure, axes = plt.subplots(1, len(slices), figsize=(3.6 * len(slices), 3.4), squeeze=False)
    for column, index in enumerate(slices):
        axis = axes[0][column]
        axis.imshow(roi.apply_display_window(magnitude[:, :, index] / scale, (0.0, 1.0)), vmin=0.0, vmax=1.0, cmap="gray", origin="lower", aspect="auto")
        outlines(axis, index)
        for name, style in (("edge_control_snr_matched", "solid"), ("background_air", "dashed")):
            plane = masks[name][:, :, index]
            if plane.any():
                axis.contour(plane.astype(float), levels=[0.5], colors="yellow", linewidths=0.8, linestyles=style)
        axis.set_title(f"ROIs on reference, PAR {index}", fontsize=8)
    records.append(_save_figure(figure, output / _FIGURES["roi_overlay"]))
    return records


def _save_figure(figure: Any, path: Path) -> dict[str, Any]:
    """Save and close one Matplotlib figure.

    Args:
        figure: Matplotlib figure.
        path: Destination PNG.

    Returns:
        File record of the written PNG.

    Side Effects:
        Writes one PNG file.
    """
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return rovir_feasibility._file_record(path)


def _format_number(value: Any) -> str:
    """Format a summary value for Markdown tables.

    Args:
        value: Number or ``None``.

    Returns:
        Compact text.
    """
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.3g}"


def _control_lines(derived: Mapping[str, Any]) -> list[str]:
    """Render the edge-matched control section of the report.

    Args:
        derived: ``derived_masks`` record of the diagnostics manifest.

    Returns:
        Markdown lines, starting with a blank line.
    """
    matched = derived.get("edge_control_snr_matched") or {}
    selected = int(matched.get("selected_voxels") or 0)
    empty_bins = sum(
        1
        for wanted, available in zip(matched.get("target_counts", []), matched.get("available_counts", []))
        if wanted > 0 and available == 0
    )
    lines = [
        "",
        "## Edge-matched control (descriptive)",
        "",
        f"- Edge control: {_format_number(derived.get('edge_control_voxels'))} head voxels within "
        f"{_format_number(roi.DEFAULT_EDGE_BAND_MM)} mm of the head boundary and more than "
        f"{_format_number(roi.DEFAULT_EDGE_EXCLUSION_MM)} mm from metal.",
        f"- SNR-matched subsample (matched to the metal SNR histogram in "
        f"{_format_number(matched.get('bins_effective'))} quantile bins, seed {_format_number(matched.get('seed'))}): "
        f"{selected} voxels; complete match {matched.get('complete_match')}; common scale factor "
        f"{_format_number(matched.get('scale_factor'))}; {empty_bins} bins without an eligible control voxel.",
    ]
    if selected == 0:
        lines.append(
            "- The SNR-matched subsample is empty: at least one metal SNR bin has no eligible "
            "control voxel, and the pre-registered common scaling keeps the histogram shape. "
            "Read the unmatched edge control together with the per-ROI log10 SNR instead."
        )
    return lines


def _report_markdown(
    payload: Mapping[str, Any],
    prepared: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> str:
    """Render the reviewed diagnostic report.

    Args:
        payload: Diagnostics manifest payload (without the report record).
        prepared: Prepare manifest.
        calibration: Two-map calibration manifest.

    Returns:
        Markdown text.
    """
    summaries = payload["roi_summaries"]
    nacs = int(prepared["accepted"]["nacs"])
    roi_names = ("metal_void", "metal_pileup", "metal", "fringe", "alias_partner", "preserved_anatomy", "edge_control", "edge_control_snr_matched", "background_air")

    def median(metric: str, name: str) -> str:
        """Return one per-ROI median as text.

        Args:
            metric: Metric key.
            name: Mask name.
        """
        record = summaries.get(metric, {}).get(name, {})
        return _format_number(record.get("quantiles", {}).get("0.5"))

    def fraction(metric: str, name: str, threshold: float) -> str:
        """Return one per-ROI threshold fraction as text.

        Args:
            metric: Metric key.
            name: Mask name.
            threshold: Threshold value.
        """
        record = summaries.get(metric, {}).get(name, {})
        return _format_number(record.get("fraction_ge", {}).get(str(threshold)))

    status = payload["noise_model"]["rnr_status"]
    failed_checks = payload["noise_model"].get("rnr_failed_checks") or []
    lines = [
        "# Set-4 coil-sensitivity consistency diagnostics",
        "",
        "Calibration-only diagnostics of the integrated FLASH set-4 ACS. No Wave",
        "reconstruction, Soft-SENSE, PSF recalibration, or refscan sets 0-3 were used.",
        "",
        "## Provenance",
        "",
        f"- Diagnostic command: `{calibration['command']['text']}` (diagnostic only; never used for reconstruction).",
        f"- BART: `{calibration['bart_version']['output']}`.",
        f"- Accepted baseline CSM: `{prepared['accepted']['ecalib_command']['text']}`.",
        f"- Sequence: FLASH TE {prepared['sequence_contract']['calibration_te_s'] * 1e3:.4f} ms, MPRAGE TE {prepared['sequence_contract']['mprage_te_s'] * 1e3:.4f} ms, ACS dwell {prepared['sequence_contract']['calibration_dwell_s'] * 1e6:.3f} us.",
        f"- Map-1 reproduction median {_format_number(payload['map1_reproduction']['median'])} (q01 {_format_number(payload['map1_reproduction']['q01'])}).",
        f"- PCA basis reproduction relative residual {_format_number(payload['pca_basis_reproduction']['relative_residual'])}.",
        "",
        "## Noise model and RNR status",
        "",
        f"RNR status: **{status}**"
        + (f" (failed checks: {', '.join(failed_checks)})" if failed_checks else "")
        + f". {RNR_POLICY}",
        "",
        "Sampling-null values come from a Monte Carlo of each air-covariance estimator "
        "(white noise with the reference covariance, the same k-space window, zero fill, "
        "and air mask). They show what finite, window-correlated samples alone would give; "
        "the pre-registered limits are unchanged.",
        "",
    ]
    checks = (
        ("Physical coils, native grid: noise scan versus empirical air", payload["noise_model"]),
        (
            "Accepted PCA basis used by RNR: transformed noise scan versus Hann air",
            payload["noise_model"].get("rnr_basis_check") or {},
        ),
    )
    for title, check in checks:
        lines += [f"{title}:", ""]
        comparison = check.get("comparison") or {}
        if "error" in comparison:
            lines += [f"- Comparison not available: {comparison['error']}", ""]
            continue
        sampling = check.get("sampling_null_reference") or {}
        null = sampling.get("criteria", {})
        for name, criterion in sorted(check.get("compatibility", {}).get("criteria", {}).items()):
            reference = null.get(name, {})
            lines.append(
                f"- {name}: {_format_number(criterion.get('value'))} (limit "
                f"{_format_number(criterion.get('limit'))}, passed {criterion.get('passed')}; "
                f"sampling-null median {_format_number(reference.get('median'))}, "
                f"max {_format_number(reference.get('max'))})"
            )
        gain = sampling.get("window_power_gain")
        trace_ratio = comparison.get("trace_ratio")
        relative = trace_ratio / gain if trace_ratio is not None and gain else None
        lines += [
            f"- Air voxels {_format_number(sampling.get('voxels'))}; effective independent "
            f"samples {_format_number(sampling.get('effective_samples'))}.",
            f"- Trace ratio (air / noise scan): {_format_number(trace_ratio)}; window power "
            f"gain {_format_number(gain)}; ratio / gain {_format_number(relative)} (reported "
            "only; also includes export scaling).",
            "",
        ]
    lines += [
        f"White-noise dwell expectation for the ACS/noise variance ratio: {_format_number(payload['noise_model']['expected_white_noise_variance_ratio_acs_over_noise'])} (approximation only; not an absolute calibration).",
        "",
        "## Eigenvalues and projection residuals (accepted PCA grid)",
        "",
        "| ROI | voxels | median lambda1 | median lambda2 | frac lambda2 >= 0.8 | median rho1 Hann | median rho2 Hann | median rho1 unapodized | median rho2 unapodized | median C1 | median C2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in roi_names:
        voxels = summaries.get("lambda1", {}).get(name, {}).get("voxels")
        lines.append(
            f"| {name} | {_format_number(voxels)} | {median('lambda1', name)} | {median('lambda2', name)} | "
            f"{fraction('lambda2', name, 0.8)} | {median('rho1_hann', name)} | {median('rho2_hann', name)} | "
            f"{median('rho1_unapodized', name)} | {median('rho2_unapodized', name)} | "
            f"{median('coherence_c1', name)} | {median('coherence_c2', name)} |"
        )
    lines += [
        "",
        f"Conditional RNR medians ({status}; do not interpret when uncalibrated):",
        "",
        "| ROI | median RNR1 | median RNR2 |",
        "| --- | ---: | ---: |",
    ]
    for name in roi_names:
        lines.append(f"| {name} | {median('rnr1_hann', name)} | {median('rnr2_hann', name)} |")
    lines += [
        "",
        "## Model-free local coil-vector rank (native set-4 grid)",
        "",
        "e1 has a noise floor without debiasing; read it against the per-ROI log10 SNR rows and the edge-matched controls.",
        "",
        "| metric | ROI | voxels | finite | median |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for metric, summary in sorted(payload["native_grid_roi_summaries"].items()):
        if not metric.startswith(("e1_", "log10_snr_")):
            continue
        for name in roi_names:
            record = summary.get(name)
            if record:
                lines.append(f"| {metric} | {name} | {_format_number(record['voxels'])} | {_format_number(record['finite'])} | {_format_number(record['quantiles'].get('0.5'))} |")
    lines += [
        "",
        "## Pre-registered alias-partner test (R = 3 on LIN)",
        "",
        "Testable hypothesis, not a conclusion. Exceedance fraction = share of non-partner LIN shifts whose overlap is at least the observed overlap; it is descriptive and not a p-value.",
        "",
        "| RO dilation | partner shift | observed overlap | exceedance fraction | null shifts |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in payload["alias_partner_test"]:
        lines.append(
            f"| {record.get('ro_half_width', 'full') if record.get('ro_half_width') is not None else 'full'} | {_format_number(record.get('partner_shift'))} | "
            f"{_format_number(record.get('observed_overlap'))} | {_format_number(record.get('exceedance_fraction'))} | {len(record.get('null_shifts', []))} |"
        )
    lines += _control_lines(payload["derived_masks"])
    lines += [
        "",
        "## FLASH-support mismatch (descriptive)",
        "",
        f"Share of accepted-image energy where lambda1 < {_format_number(payload['support_mismatch']['lambda1_threshold'])}; "
        "n/a means the mask holds no accepted-image energy.",
        "",
    ]
    for name, value in sorted(payload["support_mismatch"]["fractions"].items()):
        lines.append(f"- {name}: {_format_number(value)}")
    lines += [
        "",
        "## Figures",
        "",
    ]
    for record in payload["figures"]:
        lines.append(f"- `{Path(record['path']).name}`")
    lines += [
        "",
        "## Limitations",
        "",
        "- Calibration self-fit is optimistic: a residual failure indicates model insufficiency; a pass is not proof.",
        f"- Set-4 ACS resolution is {nacs} x {nacs} in PE; low-resolution blurring and Gibbs ringing can mix sensitivities near bright edges, which is why the Hann variant is primary and an edge-matched control is reported.",
        "- FLASH-to-MPRAGE differences in TE, contrast, preparation, and excitation cannot be tested from FLASH data alone.",
        "- For an individual isochromat under a constant readout gradient, off-resonance phase is equivalent to a readout-direction shift, and Wave does not add EPI-like B0 distortion. Near metal, however, non-invertible pile-up, intravoxel dephasing, excitation differences, signal voids, and displaced or mixed coil sensitivities can still violate the single-map SENSE model.",
        "- Unacquired or dephased signal cannot be recovered by any calibration change.",
        "",
        "## Interpretation policy",
        "",
        INTERPRETATION_POLICY,
        "",
    ]
    return "\n".join(lines)
