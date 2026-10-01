"""Integration tests for the set-4 coil-sensitivity consistency diagnostics."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr import csm_consistency, twix_noise  # noqa: E402
from wave_retro_lr.bart_io import create_cfl, sha256_file  # noqa: E402
from wave_retro_lr.mprage import (  # noqa: E402
    COIL_CALIBRATION_READOUT_OVERSAMPLING_REMOVAL,
    load_wave_mprage_helpers,
)

SHELL_SCRIPT = TOOL_ROOT / "scripts" / "sample_mprage_csm_consistency.sh"
PYTHON_CLI = TOOL_ROOT / "scripts" / "mprage_csm_consistency.py"
SYNTHETIC_BOXES = (
    "metal_void=ro=6:9,lin=24:27,par=14:17",
    "metal_pileup=ro=10:13,lin=24:27,par=14:17",
    "fringe=ro=6:13,lin=12:15,par=14:17",
    "preserved_anatomy=ro=20:25,lin=16:21,par=6:10",
    "background_air=ro=35:39,lin=all,par=all",
)
_REAL_HELPERS: Any = None
# Stand-in for BART in shell-level tests: it answers "version" and copies
# fixture maps for exactly "ecalib -m 2 -c 0 KSPACE MAPS EIGENVALUES".
STUB_BART = """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
    version) echo "v1.0.00-stub" ;;
    ecalib)
        [[ "$2 $3 $4 $5" == "-m 2 -c 0" ]] || { echo "unexpected ecalib arguments: $*" >&2; exit 1; }
        for suffix in hdr cfl; do
            cp "$STUB_MAPS.$suffix" "$7.$suffix"
            cp "$STUB_EIGEN.$suffix" "$8.$suffix"
        done ;;
    *) echo "unexpected stub call: $*" >&2; exit 1 ;;
esac
"""


def _real_helpers() -> Any:
    """Load the pinned upstream helpers once for fixtures.

    Returns:
        Upstream helper namespace.
    """
    global _REAL_HELPERS
    if _REAL_HELPERS is None:
        _REAL_HELPERS = load_wave_mprage_helpers()
    return _REAL_HELPERS


def _centered_fft(values: np.ndarray) -> np.ndarray:
    """Apply the centered orthonormal 3D FFT over the spatial axes.

    Args:
        values: Array with spatial axes 0, 1, and 2.

    Returns:
        Centered k-space.
    """
    axes = (0, 1, 2)
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(values, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def _write_cfl(path: Path, values: np.ndarray) -> None:
    """Write one synthetic complex64 CFL pair.

    Args:
        path: Destination basename.
        values: Array to store.
    """
    array = np.asarray(values, dtype=np.complex64)
    output = create_cfl(path, array.shape)
    output[...] = array
    output.flush()
    del output


def _unit(values: np.ndarray) -> np.ndarray:
    """Normalize coil vectors along the last axis.

    Args:
        values: Complex vectors.

    Returns:
        Unit-norm vectors.
    """
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def fista_r0_command(
    root: Path,
    *,
    gpu: bool = False,
    flags: Sequence[str] = csm_consistency.ACCEPTED_FISTA_R0_WAVE_FLAGS,
    csm: Path | None = None,
    psf: Path | None = None,
    kspace: Path | None = None,
    image: Path | None = None,
) -> str:
    """Format a FISTA-r0 Wave command record as the normal launcher writes it.

    Args:
        root: Accepted root whose artifacts are named by default.
        gpu: Whether to include the leading ``-g``.
        flags: Wave flags after ``-g``.
        csm: Optional CSM path override.
        psf: Optional PSF path override.
        kspace: Optional wave k-space path override.
        image: Optional output image path override.

    Returns:
        Shell-quoted command text without a trailing newline.
    """
    inputs, outputs = root / "normal" / "bart_inputs", root / "normal" / "bart_output"
    return shlex.join(
        [
            "bart",
            "wave",
            *(["-g"] if gpu else []),
            *flags,
            str(csm or outputs / "coil_sens"),
            str(psf or inputs / "psf"),
            str(kspace or inputs / "wave_kspace"),
            str(image or outputs / "fista_r0" / "image_wave"),
        ]
    )


def build_synthetic_example(root: Path, seed: int = 2026) -> SimpleNamespace:
    """Create an accepted root with rank-1 anatomy and a rank-2 pile-up region.

    Args:
        root: Temporary directory.
        seed: Random seed for noise.

    Returns:
        Namespace with paths, arrays, and patch payloads for the fixture.
    """
    rng = np.random.default_rng(seed)
    nro, nlin, npar = 40, 36, 32
    nacs, ncalib, physical, virtual = 8, 16, 6, 4
    ro, lin, par = np.meshgrid(np.arange(nro), np.arange(nlin), np.arange(npar), indexing="ij")
    centers = ((-6, 18, 16), (46, 18, 16), (18, -6, 16), (18, 42, 16), (18, 18, -6), (18, 18, 38))
    sensitivities = np.stack(
        [
            np.exp(-((ro - a) ** 2 + (lin - b) ** 2 + (par - c) ** 2) / (2 * 18.0**2))
            * np.exp(1j * (0.04 * (k + 1) * ro + 0.03 * (k - 2) * lin + 0.02 * k * par))
            for k, (a, b, c) in enumerate(centers)
        ],
        axis=-1,
    )
    head = ((ro - 18) / 14.0) ** 2 + ((lin - 18) / 14.0) ** 2 + ((par - 16) / 12.0) ** 2 <= 1.0
    tissue = (head * (1.0 + 0.15 * np.cos(ro / 5.0) * np.cos(lin / 6.0))).astype(np.complex128)
    void = (slice(6, 10), slice(24, 28), slice(14, 18))
    pileup = (slice(10, 14), slice(24, 28), slice(14, 18))
    tissue[void] = 0.0
    image = tissue[..., None] * sensitivities
    # Pile-up adds signal whose coil vector comes from tissue 12 RO voxels away.
    displaced = np.roll(sensitivities, 12, axis=0)
    weight = np.zeros(tissue.shape)
    weight[pileup] = (1.5 + 0.5 * np.sin(np.arange(4)))[:, None, None]
    image = image + weight[..., None] * displaced
    kspace = _centered_fft(image)
    lin0, par0 = nlin // 2 - nacs // 2, npar // 2 - nacs // 2
    acs = kspace[:, lin0 : lin0 + nacs, par0 : par0 + nacs, :]
    mixing = rng.normal(size=(physical, physical)) + 1j * rng.normal(size=(physical, physical))
    covariance = mixing @ mixing.conj().T / physical + 0.5 * np.eye(physical)
    covariance *= 1e-4 * float(np.mean(np.abs(acs) ** 2)) / float(np.real(np.trace(covariance)) / physical)
    factor = np.linalg.cholesky(covariance)
    white = (rng.normal(size=acs.shape) + 1j * rng.normal(size=acs.shape)) / np.sqrt(2.0)
    acs = (acs + white @ factor.T).astype(np.complex64)
    export = np.zeros((nro, ncalib, ncalib, physical), dtype=np.complex64)
    start = ncalib // 2 - nacs // 2
    export[:, start : start + nacs, start : start + nacs, :] = acs
    basis, _, energy = _real_helpers().estimate_cc_matrix_coillast(acs, ncc=virtual, acs=nacs, x_step=1)
    compressed = (acs.reshape(-1, physical) @ basis).reshape(nro, nacs, nacs, virtual)
    kspace_calib = np.zeros((nro, nlin, npar, virtual), dtype=np.complex64)
    kspace_calib[:, lin0 : lin0 + nacs, par0 : par0 + nacs, :] = compressed

    map_one = _unit(sensitivities @ basis)
    reference = np.ones(virtual) / np.sqrt(virtual)
    map_two = _unit(reference - map_one * np.sum(map_one.conj() * reference, axis=-1, keepdims=True))
    mixed = displaced @ basis
    mixed = _unit(mixed - map_one * np.sum(map_one.conj() * mixed, axis=-1, keepdims=True))
    pileup_mask = np.zeros(tissue.shape, dtype=bool)
    pileup_mask[pileup] = True
    map_two[pileup_mask] = mixed[pileup_mask]
    maps = np.stack([map_one, map_two], axis=-1)
    eigenvalues = np.stack(
        [np.where(head, 1.0, 0.2), np.where(pileup_mask, 0.95, 0.05)], axis=-1
    )
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=tissue.shape))
    accepted_map = map_one * phases[..., None]

    accepted = root / "accepted"
    inputs = accepted / "normal" / "bart_inputs"
    outputs = accepted / "normal" / "bart_output"
    _write_cfl(inputs / "kspace_calib", kspace_calib)
    _write_cfl(outputs / "coil_sens", accepted_map[..., None])
    _write_cfl(outputs / "fista_r0" / "image_wave", np.abs(tissue + weight)[..., None, None])
    # FISTA-r0 inputs are never read by the diagnostics; they only have to
    # exist so that the recorded Wave command can be bound to them.
    _write_cfl(inputs / "psf", np.ones((2 * nro, nlin, npar), dtype=np.complex64))
    _write_cfl(inputs / "wave_kspace", np.zeros((2 * nro, nlin, npar, virtual), dtype=np.complex64))
    (outputs / "ecalib_command.txt").write_text(
        f"bart ecalib -m 1 -c 0 {inputs / 'kspace_calib'} {outputs / 'coil_sens'}\n",
        encoding="utf-8",
    )
    (outputs / "fista_r0" / "wave_command.txt").write_text(
        fista_r0_command(accepted) + "\n",
        encoding="utf-8",
    )
    twix = root / "synthetic.dat"
    sequence = root / "synthetic.seq"
    twix.write_bytes(b"synthetic twix identity")
    sequence.write_text("synthetic sequence identity\n", encoding="utf-8")
    manifest = {
        "source": {
            "twix": {"path": str(twix), "size_bytes": twix.stat().st_size, "mtime_ns": twix.stat().st_mtime_ns},
            "sequence": {
                "path": str(sequence),
                "size_bytes": sequence.stat().st_size,
                "mtime_ns": sequence.stat().st_mtime_ns,
                "sha256": sha256_file(sequence),
            },
        },
        "geometry": {
            "logical_matrix_ro_lin_par": [nro, nlin, npar],
            "readout_oversampling_factor": 2,
            "physical_fov_mm_xyz": [2.0 * npar, 2.0 * nlin, 2.0 * nro],
        },
        "coil_compression": {
            "physical_coils": physical,
            "virtual_coils": virtual,
            "retained_energy": float(energy[virtual - 1]),
            "method": "fixture",
            "readout_oversampling_removal": {
                **COIL_CALIBRATION_READOUT_OVERSAMPLING_REMOVAL,
                "oversampling_factor": 2,
                "input_readout": 2 * nro,
                "output_readout": nro,
            },
        },
        "psf_calibration": {"ncalib": ncalib, "nacs": nacs},
    }
    (inputs / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    noise_factor = np.linalg.cholesky(covariance / 0.8)
    noise_white = (rng.normal(size=(64, 128, physical)) + 1j * rng.normal(size=(64, 128, physical))) / np.sqrt(2.0)
    return SimpleNamespace(
        root=root,
        accepted=accepted,
        output=root / "diagnostics",
        twix=twix,
        sequence=sequence,
        export=export,
        maps=maps,
        eigenvalues=eigenvalues,
        noise_lines=(noise_white @ noise_factor.T).astype(np.complex64),
        shape=(nro, nlin, npar),
        physical=physical,
        nacs=nacs,
    )


def _definitions(example: SimpleNamespace) -> dict[str, Any]:
    """Return Pulseq definitions matching the synthetic fixture.

    Args:
        example: Synthetic fixture.

    Returns:
        Definitions mapping.
    """
    samples = 2 * example.shape[0]
    return {
        "Calibration_ACSSetID": 4.0,
        "Calibration_RefscanNSets": 5.0,
        "Calibration_Nacs": float(example.nacs),
        "Calibration_ACSLocalStart0": 0.0,
        "Calibration_ACSLocalStop0": float(example.nacs - 1),
        "Calibration_ReadoutSamples": float(samples),
        "ReadoutOversamplingFactor": 2.0,
        "Calibration_ReadoutDuration": samples * 5e-6,
        "Calibration_TE": 0.0043625,
        "TE": 0.0036325,
        "ReadoutAxis": "z",
        "Calibration_ReadoutAxis": "z",
    }


def _walk(roles: dict[str, tuple[int, tuple[int, int], tuple[int, int], int]], channels: list[int]) -> dict[str, Any]:
    """Build a synthetic MDH walk record in the twix_noise schema.

    Args:
        roles: Role name mapped to (lines, LIN range, PAR range, unique pairs).
        channels: Channel-ID sequence used by every line.

    Returns:
        Walk record.
    """
    return {
        "scans": sum(value[0] for value in roles.values()),
        "acqend_found": True,
        "flag_counts": {},
        "dma_length_mismatches": 0,
        "roles": {
            name: {
                "lines": lines,
                "channel_id_sequences": [{"samples": 16, "channel_ids": list(channels), "lines": lines}],
                "raw_data_correction_lines": 0,
                "lin_range": list(lin_range),
                "par_range": list(par_range),
                "unique_lin_par_pairs": pairs,
            }
            for name, (lines, lin_range, par_range, pairs) in roles.items()
        },
    }


def _coil_yaps(dwell_ns: float, elements: list[str]) -> dict[tuple[str, ...], Any]:
    """Build a synthetic MeasYaps mapping with one block-0 coil selection.

    Args:
        dwell_ns: Dwell time in nanoseconds.
        elements: Element names ordered by ADC channel.

    Returns:
        Tuple-keyed header mapping.
    """
    yaps: dict[tuple[str, ...], Any] = {("sRXSPEC", "alDwellTime", "0"): float(dwell_ns)}
    for index, element in enumerate(elements):
        prefix = ("sCoilSelectMeas", "aRxCoilSelectData", "0", "asList", str(index))
        yaps[prefix + ("lADCChannelConnected",)] = float(index + 1)
        yaps[prefix + ("sCoilElementID", "tElement")] = f'"{element}"'
    return yaps


@contextlib.contextmanager
def synthetic_patches(
    example: SimpleNamespace,
    *,
    acquisition_channels: list[int] | None = None,
    acquisition_elements: list[str] | None = None,
    noise_channels: list[int] | None = None,
) -> Iterator[None]:
    """Patch TWIX, sequence, exporter, and upstream helpers for the fixture.

    Args:
        example: Synthetic fixture.
        acquisition_channels: Optional acquisition channel-ID override.
        acquisition_elements: Optional acquisition element-name override.
        noise_channels: Optional noise channel-ID override.

    Yields:
        None while the patches are active.
    """
    channels = list(range(example.physical))
    elements = [f"E{index}" for index in range(example.physical)]
    table = [
        twix_noise.RaidMeasurement(0, 11, 21, 10240, 1000, "synthetic_adjustment"),
        twix_noise.RaidMeasurement(1, 12, 22, 20000, 5000, "synthetic_acquisition"),
    ]
    nacs = example.nacs
    noise_walk = _walk({"noise": (64, (0, 63), (0, 0), 64)}, noise_channels or channels)
    # Real adjustment scans also hold their own reference lines, including
    # two-channel body-coil lines, which the identity check must ignore.
    noise_walk["roles"]["image"] = {
        "lines": 20,
        "channel_id_sequences": [
            {"samples": 128, "channel_ids": [0, 1], "lines": 10},
            {"samples": 128, "channel_ids": list(channels), "lines": 10},
        ],
        "raw_data_correction_lines": 20,
        "lin_range": [0, 9],
        "par_range": [0, 0],
        "unique_lin_par_pairs": 10,
    }
    acquisition_roles = {"image": (100, (2, 35), (0, 31), 100)}
    for index in range(4):
        acquisition_roles[f"refscan_set{index}"] = (16, (0, 15), (0, 0), 16)
    acquisition_roles["refscan_set4"] = (nacs * nacs, (0, nacs - 1), (0, nacs - 1), nacs * nacs)
    acquisition_walk = _walk(acquisition_roles, acquisition_channels or channels)
    headers = [
        {"MeasYaps": _coil_yaps(4000.0, elements), "Meas": {"flReadoutOSFactor": 2.0}},
        {"MeasYaps": _coil_yaps(5000.0, acquisition_elements or elements), "Meas": {"flReadoutOSFactor": 2.0}},
    ]

    def export(twix: Any, sequence: Any, normal: Any, output: Any, *, acs_set_index: int = 4) -> dict[str, Any]:
        """Write the synthetic physical set-4 export like the real exporter.

        Args:
            twix: TWIX path.
            sequence: Sequence path.
            normal: Accepted root.
            output: Diagnostic root.
            acs_set_index: Requested set index.

        Returns:
            Synthetic exporter manifest.
        """
        if acs_set_index != 4:
            raise ValueError("fixture accepts only set 4")
        root = Path(output)
        _write_cfl(root / "inputs" / "physical_calibration" / "physical_set4_kspace", example.export)
        payload = {
            "format_version": 2,
            "status": "mprage_rovir_physical_calibration_ready",
            "source": {"twix": {"path": str(twix), "size_bytes": Path(twix).stat().st_size, "mtime_ns": Path(twix).stat().st_mtime_ns, "sha256": sha256_file(twix)}},
        }
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        (root / "manifests" / "physical_calibration.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    real = _real_helpers()
    affine = np.asarray([[0.0, 0.0, -2.0, 30.0], [0.0, 2.0, 0.0, -36.0], [2.0, 0.0, 0.0, -40.0], [0.0, 0.0, 0.0, 1.0]])
    helper = SimpleNamespace(
        make_nifti_affine_from_twix=lambda **_: (affine, (2.0, 2.0, 2.0), {"fixture": True}),
        apply_array_axis_flips=real.apply_array_axis_flips,
        canonicalize_arrays_to_ras=real.canonicalize_arrays_to_ras,
        estimate_cc_matrix_coillast=real.estimate_cc_matrix_coillast,
    )
    noise = {
        "lines": example.noise_lines,
        "measurement_index": 0,
        "data_type": "noise",
        "remove_oversampling": False,
        "metadata": {"dwell_ns": 4000.0},
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("wave_retro_lr.mprage._read_sequence", return_value=(_definitions(example), {})))
        stack.enter_context(patch("wave_retro_lr.mprage.load_wave_mprage_helpers", return_value=helper))
        stack.enter_context(patch("wave_retro_lr.rovir_feasibility.export_mprage_physical_calibration", side_effect=export))
        stack.enter_context(patch("wave_retro_lr.twix_noise.read_multiraid_table_from_file", return_value=table))
        stack.enter_context(
            patch(
                "wave_retro_lr.twix_noise.walk_measurement",
                side_effect=lambda path, entry: noise_walk if entry.index == 0 else acquisition_walk,
            )
        )
        stack.enter_context(patch("wave_retro_lr.twix_noise.load_measurement_headers", return_value=headers))
        stack.enter_context(patch("wave_retro_lr.twix_noise.load_measurement0_noise", return_value=noise))
        yield


def write_synthetic_calibration(example: SimpleNamespace, *, command: str | None = None) -> None:
    """Emulate the shell calibrate stage with exact synthetic two-map outputs.

    Args:
        example: Synthetic fixture.
        command: Optional command text overriding the reviewed command.
    """
    output = example.output
    argv = csm_consistency.two_map_ecalib_argv(example.accepted, output)
    _write_cfl(output / "csm" / "map2_uncropped" / "coil_sens", example.maps)
    kspace = Path(argv[6])
    (output / csm_consistency.LAYOUT["two_map_input"]).write_text(
        "".join(
            f"{sha256_file(kspace.with_suffix(suffix))}  {kspace.with_suffix(suffix)}\n"
            for suffix in (".hdr", ".cfl")
        ),
        encoding="utf-8",
    )
    _write_cfl(output / "csm" / "eigenvalues" / "ev_m2_c0", example.eigenvalues[:, :, :, None, :])
    (output / "csm" / "map2_uncropped" / "ecalib_command.txt").write_text(
        (command or csm_consistency.format_command(argv)) + "\n", encoding="utf-8"
    )
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "bart_version.txt").write_text("v1.0.00-synthetic\n", encoding="utf-8")
    (logs / "bart_binary.sha256").write_text("0" * 64 + "  bart\n", encoding="utf-8")
    (logs / "ecalib_m2_c0.log").write_text("synthetic ecalib log\n", encoding="utf-8")


def run_synthetic_pipeline(example: SimpleNamespace) -> dict[str, Any]:
    """Run prepare, record, template, and diagnose on the synthetic fixture.

    Args:
        example: Synthetic fixture.

    Returns:
        Diagnostics manifest.
    """
    arguments = (example.twix, example.sequence, example.accepted, example.output)
    with synthetic_patches(example):
        csm_consistency.prepare_csm_consistency(*arguments)
        write_synthetic_calibration(example)
        csm_consistency.record_two_map_calibration(*arguments)
        csm_consistency.write_roi_template(*arguments)
        return csm_consistency.diagnose_csm_consistency(*arguments, boxes=SYNTHETIC_BOXES)


class CommandContractTests(unittest.TestCase):
    """Keep BART explicit in the shell and absent from Python."""

    def test_shell_entry_point_has_only_the_diagnostic_ecalib_command(self) -> None:
        """Allow exactly one BART computation: two-map ecalib without cropping.

        Returns:
            None.
        """
        subprocess.run(["bash", "-n", str(SHELL_SCRIPT)], check=True)
        executable = [
            line.strip()
            for line in SHELL_SCRIPT.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        run_lines = [line for line in executable if line.startswith("BART_DEBUG_LEVEL=3")]
        self.assertEqual(len(run_lines), 1)
        self.assertIn("bart ecalib -m 2 -c 0 ", run_lines[0])
        bart_commands = [
            line for line in executable if line.startswith("bart ") or " bart " in f" {line} "
        ]
        allowed = ("bart version", "BART_DEBUG_LEVEL=3 /usr/bin/time -v bart ecalib -m 2 -c 0 ", "printf -v ECALIB_COMMAND")
        for line in bart_commands:
            if "command -v bart" in line or line.startswith(("echo", "calibrate ")):
                continue
            self.assertTrue(any(token in line for token in allowed), line)
        text = "\n".join(executable)
        for forbidden in ("bart wave", "bart fft", "bart rss", "bart pics", " -S ", "-m 1 "):
            self.assertNotIn(forbidden, text)

    def test_help_and_print_command_create_nothing_and_match_python(self) -> None:
        """Print the identical command from shell and Python without side effects.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "not_created"
            help_run = subprocess.run(["bash", str(SHELL_SCRIPT), "--help"], check=True, capture_output=True, text=True)
            self.assertIn("print-calibration-command", help_run.stdout)
            arguments = [str(root / "a.dat"), str(root / "a.seq"), str(root / "accepted"), str(output)]
            shell = subprocess.run(
                ["bash", str(SHELL_SCRIPT), "print-calibration-command", *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
            python = subprocess.run(
                [sys.executable, str(PYTHON_CLI), "print-calibration-command", *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(shlex.split(shell.stdout), shlex.split(python.stdout))
            self.assertEqual(shlex.split(shell.stdout)[:6], ["bart", "ecalib", "-m", "2", "-c", "0"])
            bad = subprocess.run(["bash", str(SHELL_SCRIPT), "unknown", *arguments], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertFalse(output.exists())
            cli_help = subprocess.run([sys.executable, str(PYTHON_CLI), "--help"], check=True, capture_output=True, text=True)
            self.assertIn("diagnose", cli_help.stdout)

    def test_python_modules_never_launch_processes(self) -> None:
        """Keep library modules and the CLI free of process launching.

        Returns:
            None.
        """
        for relative in (
            "wave_retro_lr/csm_consistency.py",
            "wave_retro_lr/csm_consistency_metrics.py",
            "wave_retro_lr/csm_consistency_roi.py",
            "wave_retro_lr/twix_noise.py",
            "scripts/mprage_csm_consistency.py",
        ):
            source = (TOOL_ROOT / relative).read_text(encoding="utf-8")
            for forbidden in ("subprocess", "os.system", "Popen", "execv"):
                self.assertNotIn(forbidden, source, relative)

    def test_implementation_identity_covers_every_dependency(self) -> None:
        """Hash every package module, both entry points, and upstream helpers.

        Returns:
            None.
        """
        identity = csm_consistency.implementation_identity()
        modules = {f"wave_retro_lr/{path.name}" for path in (TOOL_ROOT / "wave_retro_lr").glob("*.py")}
        self.assertLessEqual(modules, set(identity))
        for name in (
            "wave_retro_lr/nifti_collection.py",
            "wave_retro_lr/mprage.py",
            "wave_retro_lr/bart_io.py",
            "wave_retro_lr/rovir.py",
            *csm_consistency._ENTRY_POINTS,
            *csm_consistency._UPSTREAM_IMPLEMENTATION_FILES,
        ):
            self.assertIn(name, identity)
        self.assertTrue(all(len(value) == 64 for value in identity.values()))

    def test_shell_writes_one_new_environment_log_per_invocation(self) -> None:
        """Never rewrite a recorded environment log.

        Returns:
            None.
        """
        script = SHELL_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("environment_${stage}.txt", script)
        self.assertIn('mktemp --suffix=.txt "$directory/${stage}_', script)
        self.assertEqual(script.count('--environment-log "$ENVIRONMENT_LOG"'), 4)

    def test_normal_reconstruction_defaults_are_unchanged(self) -> None:
        """Keep the one-map normal ecalib default and PCA-12 preparation.

        Returns:
            None.
        """
        normal = (TOOL_ROOT / "scripts" / "sample_mprage_normal_recon.sh").read_text(encoding="utf-8")
        self.assertIn('ECALIB_CROP="0.6"', normal)
        self.assertIn('bart ecalib -m 1 -c "$ECALIB_CROP"', normal)
        mprage = (TOOL_ROOT / "wave_retro_lr" / "mprage.py").read_text(encoding="utf-8")
        self.assertIn("ncc=12", mprage)


class ContractValidationTests(unittest.TestCase):
    """Reject incompatible inputs before any metric is computed."""

    def test_cfl_pair_states_reject_partial_outputs(self) -> None:
        """Distinguish absent, partial, and complete CFL pairs.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "maps"
            self.assertEqual(csm_consistency.cfl_pair_state(base), "absent")
            with self.assertRaises(FileNotFoundError):
                csm_consistency.require_complete_cfl_pair(base, "maps")
            base.with_suffix(".hdr").write_text("# Dimensions\n2 2\n", encoding="utf-8")
            self.assertEqual(csm_consistency.cfl_pair_state(base), "partial")
            with self.assertRaisesRegex(ValueError, "partial"):
                csm_consistency.require_complete_cfl_pair(base, "maps")
            _write_cfl(base, np.ones((2, 2)))
            self.assertEqual(csm_consistency.require_complete_cfl_pair(base, "maps"), (2, 2))

    def test_accepted_nifti_geometry_is_recorded_without_raising(self) -> None:
        """Record agreement with the unique accepted magnitude NIfTI.

        Returns:
            None.
        """
        import nibabel as nib

        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = (-10.0, 4.0, 7.5)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fista_r0"
            self.assertEqual(
                csm_consistency._accepted_nifti_geometry(directory, (4, 5, 6), affine),
                {"available": False, "candidates": 0},
            )
            directory.mkdir()
            path = directory / "sub-x_part-mag_fista.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), np.float32), affine), str(path))
            record = csm_consistency._accepted_nifti_geometry(directory, (4, 5, 6), affine)
            self.assertTrue(record["available"])
            self.assertTrue(record["shape_matches"])
            self.assertTrue(record["affine_matches"])
            shifted = affine.copy()
            shifted[0, 3] += 0.01
            record = csm_consistency._accepted_nifti_geometry(directory, (4, 5, 7), shifted)
            self.assertFalse(record["shape_matches"])
            self.assertFalse(record["affine_matches"])
            self.assertAlmostEqual(record["max_abs_affine_difference_mm"], 0.01, places=5)
            nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), np.float32), affine), str(directory / "b_part-mag_x.nii.gz"))
            self.assertEqual(
                csm_consistency._accepted_nifti_geometry(directory, (4, 5, 6), affine),
                {"available": False, "candidates": 2},
            )

    def test_sequence_contract_requires_set_four_and_complete_counters(self) -> None:
        """Reject another ACS set, set count, or local counter range.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            manifest = json.loads((example.accepted / "normal" / "bart_inputs" / "manifest.json").read_text())
            definitions = _definitions(example)
            record = csm_consistency.sequence_contract(definitions, manifest)
            self.assertAlmostEqual(record["calibration_dwell_s"], 5e-6)
            for key, value in (
                ("Calibration_ACSSetID", 3.0),
                ("Calibration_RefscanNSets", 6.0),
                ("Calibration_ACSLocalStop0", float(example.nacs)),
            ):
                with self.subTest(key=key):
                    with self.assertRaisesRegex(ValueError, key):
                        csm_consistency.sequence_contract({**definitions, key: value}, manifest)

    def test_accepted_baseline_requires_alias_free_one_map_c0(self) -> None:
        """Reject striding provenance, other crops, and two-map accepted CSMs.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            record, _ = csm_consistency.validate_accepted_baseline(example.accepted)
            self.assertEqual(record["virtual_coils"], 4)
            command = example.accepted / "normal" / "bart_output" / "ecalib_command.txt"
            original = command.read_text(encoding="utf-8")
            command.write_text(original.replace("-c 0", "-c 0.6"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ecalib -m 1 -c 0"):
                csm_consistency.validate_accepted_baseline(example.accepted)
            command.write_text(original, encoding="utf-8")
            manifest_path = example.accepted / "normal" / "bart_inputs" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["coil_compression"].pop("readout_oversampling_removal")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alias-free"):
                csm_consistency.validate_accepted_baseline(example.accepted)

    def test_prepare_rejects_channel_or_coil_selection_mismatch(self) -> None:
        """Fail closed when channel IDs or block-0 element maps disagree.

        Returns:
            None.
        """
        for override in (
            {"acquisition_channels": [1, 0, 2, 3, 4, 5]},
            {"noise_channels": [0, 1, 2, 3, 5, 4]},
            {"acquisition_elements": ["E1", "E0", "E2", "E3", "E4", "E5"]},
        ):
            with self.subTest(override=list(override)):
                with tempfile.TemporaryDirectory() as temporary:
                    example = build_synthetic_example(Path(temporary))
                    with synthetic_patches(example, **override):
                        with self.assertRaises(ValueError):
                            csm_consistency.prepare_csm_consistency(
                                example.twix, example.sequence, example.accepted, example.output
                            )

    def test_record_calibration_rejects_wrong_command_partial_and_changed_input(self) -> None:
        """Accept only the reviewed command, complete pairs, and unchanged inputs.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            arguments = (example.twix, example.sequence, example.accepted, example.output)
            with synthetic_patches(example):
                prepared = csm_consistency.prepare_csm_consistency(*arguments)
                self.assertEqual(prepared["status"], "mprage_csm_consistency_inputs_ready")
                self.assertFalse(prepared["noise"]["absolute_acs_noise_calibration_claimed"])
                again = csm_consistency.prepare_csm_consistency(*arguments)
                self.assertEqual(again["created_at_utc"], prepared["created_at_utc"])
                write_synthetic_calibration(example, command="bart ecalib -m 2 -S -c 0.8 a b c")
                with self.assertRaisesRegex(ValueError, "differs from the reviewed"):
                    csm_consistency.record_two_map_calibration(*arguments)
                write_synthetic_calibration(example)
                (example.output / "csm" / "eigenvalues" / "ev_m2_c0.cfl").unlink()
                with self.assertRaisesRegex(ValueError, "partial"):
                    csm_consistency.record_two_map_calibration(*arguments)
                write_synthetic_calibration(example)
                record = csm_consistency.record_two_map_calibration(*arguments)
                self.assertTrue(record["diagnostic_only"])
                self.assertFalse(record["used_for_wave_reconstruction"])
                self.assertEqual(record["map_count"], 2)
                kspace = example.accepted / "normal" / "bart_inputs" / "kspace_calib.cfl"
                payload = bytearray(kspace.read_bytes())
                payload[0] ^= 1
                kspace.write_bytes(bytes(payload))
                with self.assertRaisesRegex(ValueError, "kspace_calib changed"):
                    csm_consistency.record_two_map_calibration(*arguments)


def _random_covariance(channels: int, seed: int) -> np.ndarray:
    """Draw a well-conditioned complex Hermitian positive-definite covariance.

    Args:
        channels: Matrix size.
        seed: Random seed.

    Returns:
        complex128 covariance.
    """
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((channels, channels)) + 1j * rng.standard_normal((channels, channels))
    return factor @ factor.conj().T / channels + np.eye(channels)


def _hann_block(nacs: int) -> np.ndarray:
    """Return the separable DC-centered Hann weights of one ACS block.

    Args:
        nacs: ACS edge length.

    Returns:
        ``(nacs, nacs)`` float64 weights.
    """
    profile = csm_consistency.metrics.dc_centered_hann(nacs)
    return np.outer(profile, profile)


def _brute_force_effective_samples(mask: np.ndarray, window: np.ndarray) -> float:
    """Compute the effective sample count from the explicit Gram eigenvalues.

    Every masked voxel is written as a linear combination of the measured
    k-space samples by transforming unit impulses with the production
    centered inverse FFT, and the estimator weights are the eigenvalues of
    the resulting Gram matrix.

    Args:
        mask: Boolean ``(RO, LIN, PAR)`` air mask.
        window: ``(nacs, nacs)`` k-space weights.

    Returns:
        ``(sum g)^2 / sum g^2`` of the Gram eigenvalues ``g``.
    """
    nro, nlin, npar = mask.shape
    nacs = window.shape[0]
    lin_block, par_block = csm_consistency.metrics.acs_block_slices(nlin, npar, nacs)
    responses = []
    for kl in range(nacs):
        for kp in range(nacs):
            impulse = np.zeros((nlin, npar), dtype=np.complex128)
            impulse[lin_block.start + kl, par_block.start + kp] = window[kl, kp]
            responses.append(csm_consistency.metrics.centered_ifft(impulse, axes=(0, 1)))
    responses = np.stack(responses, axis=-1)
    rows = []
    for ro, lin, par in zip(*np.nonzero(mask)):
        row = np.zeros((nro, nacs * nacs), dtype=np.complex128)
        row[ro] = responses[lin, par]
        rows.append(row.ravel())
    design = np.asarray(rows)
    gram = design.T @ design.conj() / design.shape[0]
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.conj().T))
    return float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))


class NoiseModelNullTests(unittest.TestCase):
    """Validate the effective sample count, the estimator null, and the RNR decision."""

    def test_effective_samples_match_brute_force_gram_eigenvalues(self) -> None:
        """Match the FFT formula to explicit eigenvalues and closed forms.

        Returns:
            None.
        """
        rng = np.random.default_rng(11)
        mask = rng.random((3, 8, 6)) < 0.4
        mask[1] = True
        for window in (_hann_block(4), np.ones((4, 4)), rng.random((4, 4)) + 0.1):
            self.assertAlmostEqual(
                csm_consistency.effective_air_samples(mask, window),
                _brute_force_effective_samples(mask, window),
                delta=1e-9 * mask.sum(),
            )
        native = rng.random((5, 4, 4)) < 0.5
        self.assertAlmostEqual(
            csm_consistency.effective_air_samples(native, np.ones((4, 4))), float(native.sum()), places=9
        )
        hann = _hann_block(8)
        full = np.ones((2, 16, 16), dtype=bool)
        expected = 2 * np.sum(hann**2) ** 2 / np.sum(hann**4)
        self.assertAlmostEqual(csm_consistency.effective_air_samples(full, hann), expected, places=9)
        self.assertAlmostEqual(csm_consistency.effective_air_samples(full, np.ones((8, 8))), 128.0, places=9)
        self.assertEqual(csm_consistency.effective_air_samples(np.zeros((2, 4, 4), dtype=bool), hann[:4, :4]), 0.0)
        with self.assertRaises(ValueError):
            csm_consistency.effective_air_samples(full, np.ones((8, 7)))
        with self.assertRaises(ValueError):
            csm_consistency.effective_air_samples(full, -np.ones((8, 8)))

    def test_sampling_null_simulates_the_windowed_estimator(self) -> None:
        """Reproduce the zero-filled windowed air-covariance estimator.

        Returns:
            None.
        """
        reference = _random_covariance(4, seed=5)
        full = np.ones((6, 16, 16), dtype=bool)
        hann = _hann_block(8)
        hann_record = csm_consistency.sampling_null_reference(reference, full, hann, repeats=48, seed=3)
        self.assertEqual(hann_record, csm_consistency.sampling_null_reference(reference, full, hann, repeats=48, seed=3))
        unapodized = csm_consistency.sampling_null_reference(reference, full, np.ones((8, 8)), repeats=48, seed=3)
        self.assertAlmostEqual(hann_record["effective_samples"], 6 * np.sum(hann**2) ** 2 / np.sum(hann**4), places=9)
        self.assertAlmostEqual(unapodized["effective_samples"], 384.0, places=9)
        self.assertAlmostEqual(hann_record["window_power_gain"], np.sum(hann**2) / 256.0)
        self.assertAlmostEqual(unapodized["window_power_gain"], 0.25)
        for record in (hann_record, unapodized):
            self.assertEqual(record["voxels"], full.sum())
            self.assertEqual(record["singular_draws"], 0)
            self.assertLess(abs(record["trace_ratio"]["median"] / record["window_power_gain"] - 1.0), 0.1)
        spread = "generalized_eigenvalue_spread"
        self.assertGreater(hann_record["criteria"][spread]["median"], unapodized["criteria"][spread]["median"])
        too_small = np.zeros_like(full)
        too_small[0, 0, :4] = True
        self.assertIsNone(csm_consistency.sampling_null_reference(reference, too_small, hann))

        # The row-wise hybrid-space draw must match the production path that
        # windows full 3D k-space noise, applies the 3D inverse FFT, and
        # averages over a partial air mask.
        channels, shape, nacs = 3, (8, 8, 8), 4
        reference = _random_covariance(channels, seed=9)
        rng = np.random.default_rng(21)
        mask = rng.random(shape) < 0.35
        window3d = csm_consistency.metrics.acs_apodization_window(shape, nacs)
        lin_block, par_block = csm_consistency.metrics.acs_block_slices(shape[1], shape[2], nacs)
        factor = np.linalg.cholesky(reference)
        direct: dict[str, list[float]] = {name: [] for name in twix_noise.DEFAULT_COVARIANCE_COMPATIBILITY_LIMITS}
        for _ in range(160):
            kspace = np.zeros(shape + (channels,), dtype=np.complex128)
            white = rng.standard_normal((shape[0], nacs, nacs, channels)) + 1j * rng.standard_normal(
                (shape[0], nacs, nacs, channels)
            )
            kspace[:, lin_block, par_block] = (white / np.sqrt(2.0)) @ factor.T
            images = csm_consistency.metrics.coil_images(kspace, window3d, dtype=np.complex128)
            candidate, _ = csm_consistency.metrics.empirical_coil_covariance(images, mask)
            comparison = twix_noise.compare_covariances(reference, candidate)
            for name, record in twix_noise.covariance_compatibility(comparison)["criteria"].items():
                direct[name].append(record["value"])
        simulated = csm_consistency.sampling_null_reference(reference, mask, _hann_block(nacs), repeats=160, seed=4)
        for name, values in direct.items():
            with self.subTest(criterion=name):
                self.assertLess(abs(simulated["criteria"][name]["median"] / np.median(values) - 1.0), 0.12)

    def test_covariance_check_records_errors_and_rnr_needs_both_checks(self) -> None:
        """Record comparison failures and require both compatible checks.

        Returns:
            None.
        """
        reference = _random_covariance(3, seed=2)
        mask = np.ones((4, 4, 4), dtype=bool)
        window = np.ones((4, 4))
        labels = {"reference_label": "reference", "candidate_label": "candidate"}
        for candidate in (None, np.diag([1.0, 1.0, 0.0]).astype(np.complex128)):
            record = csm_consistency.covariance_check(reference, candidate, mask, window, **labels)
            self.assertFalse(record["compatibility"]["compatible"])
            self.assertEqual(record["compatibility"]["failed_criteria"], ["comparison_error"])
            self.assertIn("error", record["comparison"])
            self.assertIsNone(record["sampling_null_reference"])
            json.dumps(csm_consistency._json_ready(record), allow_nan=False)
        record = csm_consistency.covariance_check(reference, reference.copy(), mask, window, **labels)
        self.assertTrue(record["compatibility"]["compatible"])
        self.assertAlmostEqual(record["sampling_null_reference"]["effective_samples"], 64.0, places=9)

        passed = {"compatibility": {"compatible": True}}
        failed = {"compatibility": {"compatible": False}}
        cases = {
            (None, None): ["physical_noise_model", "accepted_pca_basis"],
            ("passed", "passed"): [],
            ("passed", "failed"): ["accepted_pca_basis"],
            ("failed", "passed"): ["physical_noise_model"],
            ("passed", None): ["accepted_pca_basis"],
        }
        lookup = {None: None, "passed": passed, "failed": failed}
        for (physical, accepted), expected in cases.items():
            with self.subTest(physical=physical, accepted=accepted):
                decision = csm_consistency.rnr_status(lookup[physical], lookup[accepted])
                self.assertEqual(decision["failed_checks"], expected)
                self.assertEqual(decision["status"], "uncalibrated" if expected else "calibrated")


def _copy_cfl(source: Path, destination: Path) -> None:
    """Copy one CFL pair to another basename.

    Args:
        source: Source BART basename.
        destination: Destination BART basename.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".hdr", ".cfl"):
        shutil.copyfile(source.with_suffix(suffix), destination.with_suffix(suffix))


@contextlib.contextmanager
def _inside(directory: Path) -> Iterator[int]:
    """Enter a directory and hold an open descriptor on it.

    While active, ``/proc/self/cwd`` and ``/dev/fd/<descriptor>`` name the
    directory, so process-dependent spellings would otherwise bind to it.

    Args:
        directory: Directory to enter.

    Yields:
        File descriptor of the directory.
    """
    previous = os.getcwd()
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    os.chdir(directory)
    try:
        yield descriptor
    finally:
        os.chdir(previous)
        os.close(descriptor)


def _process_dependent_spellings(root: Path, relative: str, descriptor: int, link: Path) -> dict[str, str]:
    """Spell one file below ``root`` in process-dependent ways.

    Args:
        root: Directory that is the current directory and held by ``descriptor``.
        relative: File path relative to ``root``.
        descriptor: Open descriptor on ``root``.
        link: Symbolic link to ``/proc/self/cwd``.

    Returns:
        Case name mapped to an absolute path that names the file only in this
        process.
    """
    return {
        "//proc": f"//proc/self/cwd/{relative}",
        "//dev": f"//dev/fd/{descriptor}/{relative}",
        "///proc": f"///proc/self/cwd/{relative}",
        "/./proc": f"/./proc/self/cwd/{relative}",
        "/../proc": f"/../proc/self/cwd/{relative}",
        "/proc": f"/proc/self/cwd/{relative}",
        "/dev": f"/dev/fd/{descriptor}/{relative}",
        "link into /proc": f"{link}/{relative}",
    }


def _foreign_copy(example: SimpleNamespace) -> Path:
    """Copy the accepted artifacts to an unrelated root with the same layout.

    Args:
        example: Synthetic fixture.

    Returns:
        Unrelated root whose files match the accepted files byte for byte.
    """
    foreign = example.root / "unrelated"
    for relative in (
        "normal/bart_inputs/kspace_calib",
        "normal/bart_inputs/psf",
        "normal/bart_inputs/wave_kspace",
        "normal/bart_output/coil_sens",
        "normal/bart_output/fista_r0/image_wave",
    ):
        _copy_cfl(example.accepted / relative, foreign / relative)
    return foreign


class AcceptedCommandBindingTests(unittest.TestCase):
    """Bind both accepted command records to the current accepted artifacts."""

    def test_ecalib_record_is_bound_to_the_current_accepted_root(self) -> None:
        """Reject foreign or relative paths and accept symlink aliases.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            foreign = _foreign_copy(example)
            alias = example.root / "alias"
            alias.symlink_to(example.accepted, target_is_directory=True)
            command = example.accepted / "normal" / "bart_output" / "ecalib_command.txt"
            original = command.read_text(encoding="utf-8")

            def ecalib(kspace: object, csm: object) -> str:
                """Format an ecalib record naming the given paths.

                Args:
                    kspace: Calibration k-space path.
                    csm: CSM output path.

                Returns:
                    Command text.
                """
                return shlex.join(["bart", "ecalib", "-m", "1", "-c", "0", str(kspace), str(csm)]) + "\n"

            accepted_kspace = example.accepted / "normal" / "bart_inputs" / "kspace_calib"
            accepted_csm = example.accepted / "normal" / "bart_output" / "coil_sens"
            # BART opens x.ra itself, so x.ra.hdr/.cfl aliases must not bind it.
            single = example.root / "single"
            single.mkdir()
            for suffix in (".hdr", ".cfl"):
                (single / f"kspace_calib.ra{suffix}").symlink_to(accepted_kspace.with_suffix(suffix))
            (single / "kspace_calib.ra").write_bytes(b"unrelated single-file image")
            rejected = {
                "BART single-file suffix": ecalib(single / "kspace_calib.ra", accepted_csm),
                "process-dependent path": ecalib(
                    "/proc/self/cwd/normal/bart_inputs/kspace_calib", accepted_csm
                ),
                "reviewer reproduction": ecalib(
                    "/unrelated/normal/bart_inputs/kspace_calib", "/unrelated/normal/bart_output/coil_sens"
                ),
                "foreign root with identical files": ecalib(
                    foreign / "normal" / "bart_inputs" / "kspace_calib",
                    foreign / "normal" / "bart_output" / "coil_sens",
                ),
                "foreign CSM only": ecalib(accepted_kspace, foreign / "normal" / "bart_output" / "coil_sens"),
                "relative paths": ecalib("normal/bart_inputs/kspace_calib", "normal/bart_output/coil_sens"),
                "swapped roles": ecalib(accepted_csm, accepted_kspace),
                "crop 0.0": original.replace("-c 0 ", "-c 0.0 "),
                "extra token": original.strip() + " extra\n",
                "unbalanced quote": "bart ecalib -m 1 -c 0 'unterminated\n",
            }
            for name, text in rejected.items():
                with self.subTest(case=name):
                    command.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        csm_consistency.validate_accepted_baseline(example.accepted)
            # Process-dependent spellings name the accepted files in this
            # process, so only the explicit rule can reject them.
            link = example.root / "cwd_link"
            link.symlink_to("/proc/self/cwd")
            with _inside(example.accepted) as descriptor:
                spellings = _process_dependent_spellings(
                    example.accepted, "normal/bart_inputs/kspace_calib", descriptor, link
                )
                for name, token in spellings.items():
                    with self.subTest(process_dependent=name):
                        self.assertTrue(os.path.samefile(token + ".hdr", accepted_kspace.with_suffix(".hdr")))
                        command.write_text(ecalib(token, accepted_csm), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, "process-dependent"):
                            csm_consistency.validate_accepted_baseline(example.accepted)
            command.write_text(
                ecalib(
                    alias / "normal" / "bart_inputs" / "kspace_calib",
                    alias / "normal" / "bart_output" / "coil_sens",
                ),
                encoding="utf-8",
            )
            record, _ = csm_consistency.validate_accepted_baseline(example.accepted)
            bound = record["ecalib_command"]["bound_artifacts"]
            self.assertEqual(bound["coil_sens"]["resolved"], str(accepted_csm.resolve()))
            self.assertTrue(bound["kspace_calib"]["recorded"].startswith(str(alias)))
            # Nothing is written when the accepted baseline is rejected.
            command.write_text(rejected["foreign root with identical files"], encoding="utf-8")
            with synthetic_patches(example):
                with self.assertRaises(ValueError):
                    csm_consistency.prepare_csm_consistency(
                        example.twix, example.sequence, example.accepted, example.output
                    )
            self.assertFalse(example.output.exists())

    def test_fista_r0_record_is_the_bound_unregularized_branch(self) -> None:
        """Reject regularized, other-branch, and foreign-path Wave records.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            foreign = _foreign_copy(example)
            _copy_cfl(
                example.accepted / "normal" / "bart_output" / "fista_r0" / "image_wave",
                example.accepted / "normal" / "bart_output" / "optimal_wavelet" / "image_wave",
            )
            alias = example.root / "alias"
            alias.symlink_to(example.accepted, target_is_directory=True)
            command = example.accepted / "normal" / "bart_output" / "fista_r0" / "wave_command.txt"
            regularized = list(csm_consistency.ACCEPTED_FISTA_R0_WAVE_FLAGS)
            regularized[regularized.index("-r") + 1] = "0.2"
            outputs = foreign / "normal" / "bart_output"
            single = example.root / "single"
            single.mkdir()
            image_base = example.accepted / "normal" / "bart_output" / "fista_r0" / "image_wave"
            for suffix in (".hdr", ".cfl"):
                (single / f"image_wave.ra{suffix}").symlink_to(image_base.with_suffix(suffix))
            (single / "image_wave.ra").write_bytes(b"unrelated single-file image")
            rejected = {
                "reviewer reproduction": "bart wave -w -f -r 0.2 wrong_sens wrong_psf wrong_kspace wrong_output",
                "BART single-file suffix output": fista_r0_command(example.accepted, image=single / "image_wave.ra"),
                "regularized": fista_r0_command(example.accepted, flags=regularized),
                "foreign CSM": fista_r0_command(example.accepted, csm=outputs / "coil_sens"),
                "foreign PSF": fista_r0_command(example.accepted, psf=foreign / "normal" / "bart_inputs" / "psf"),
                "foreign wave k-space": fista_r0_command(
                    example.accepted, kspace=foreign / "normal" / "bart_inputs" / "wave_kspace"
                ),
                "foreign output": fista_r0_command(example.accepted, image=outputs / "fista_r0" / "image_wave"),
                "other branch output": fista_r0_command(
                    example.accepted,
                    image=example.accepted / "normal" / "bart_output" / "optimal_wavelet" / "image_wave",
                ),
                "PSF in the k-space slot": fista_r0_command(
                    example.accepted,
                    psf=example.accepted / "normal" / "bart_inputs" / "wave_kspace",
                    kspace=example.accepted / "normal" / "bart_inputs" / "psf",
                ),
                "missing iteration flags": fista_r0_command(example.accepted, flags=("-w", "-f", "-r", "0")),
                "reordered flags": fista_r0_command(
                    example.accepted, flags=("-f", "-w", "-r", "0", "-i", "100", "-t", "1e-6")
                ),
                "gpu flag after the others": fista_r0_command(
                    example.accepted, flags=(*csm_consistency.ACCEPTED_FISTA_R0_WAVE_FLAGS, "-g")
                ),
                "relative paths": "bart wave -w -f -r 0 -i 100 -t 1e-6 coil_sens psf wave_kspace image_wave",
                "not wave": fista_r0_command(example.accepted).replace("bart wave", "bart pics", 1),
            }
            for name, text in rejected.items():
                with self.subTest(case=name):
                    command.write_text(text + "\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        csm_consistency.validate_accepted_baseline(example.accepted)
            link = example.root / "cwd_link"
            link.symlink_to("/proc/self/cwd")
            with _inside(example.accepted) as descriptor:
                for role, relative in (
                    ("csm", "normal/bart_output/coil_sens"),
                    ("image", "normal/bart_output/fista_r0/image_wave"),
                ):
                    spellings = _process_dependent_spellings(example.accepted, relative, descriptor, link)
                    for name, token in spellings.items():
                        with self.subTest(process_dependent=name, role=role):
                            command.write_text(
                                fista_r0_command(example.accepted, **{role: token}) + "\n",
                                encoding="utf-8",
                            )
                            with self.assertRaisesRegex(ValueError, "process-dependent"):
                                csm_consistency.validate_accepted_baseline(example.accepted)
            for gpu, root in ((False, example.accepted), (True, example.accepted), (True, alias)):
                with self.subTest(gpu=gpu, root=root.name):
                    command.write_text(fista_r0_command(root, gpu=gpu) + "\n", encoding="utf-8")
                    record, _ = csm_consistency.validate_accepted_baseline(example.accepted)
                    fista = record["fista_r0_command"]
                    self.assertEqual(fista["gpu"], gpu)
                    self.assertEqual(
                        set(fista["bound_artifacts"]), {"coil_sens", "psf", "wave_kspace", "fista_r0_image"}
                    )
                    self.assertEqual(record["psf"]["payload_hash"].split(";")[0], "not computed")
                    self.assertIn("header_sha256", record["wave_kspace"])


class ReviewedLabelValueTests(unittest.TestCase):
    """Validate reviewed label values before any integer narrowing."""

    def test_out_of_range_values_never_wrap_into_labels(self) -> None:
        """Reject 257 and other invalid stored values; accept exact integers.

        Returns:
            None.
        """
        import nibabel as nib

        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            arguments = (example.twix, example.sequence, example.accepted, example.output)
            with synthetic_patches(example):
                csm_consistency.prepare_csm_consistency(*arguments)
                template = csm_consistency.write_roi_template(*arguments)
                geometry = template["geometry"]
                labels, _ = csm_consistency.roi.labels_from_boxes(SYNTHETIC_BOXES, example.shape)
                stored, _, _ = csm_consistency.roi.to_stored_orientation(
                    labels, tuple(geometry["array_flips"]), np.asarray(geometry["source_affine"])
                )
                affine = nib.load(template["reference_nifti"]["path"]).affine
                unassigned = tuple(int(value) for value in np.argwhere(stored == 0)[0])
                reviewed = example.root / "reviewed.nii.gz"

                def load(values: np.ndarray) -> np.ndarray:
                    """Save stored-orientation values and load them as reviewed ROIs.

                    Args:
                        values: Label values in stored orientation.

                    Returns:
                        Labels on the BART grid.
                    """
                    nib.save(nib.Nifti1Image(values, affine), str(reviewed))
                    return csm_consistency.load_reviewed_rois(
                        example.output, example.shape, labels_path=reviewed
                    )[0]

                for dtype in (np.uint8, np.int16, np.float32):
                    with self.subTest(dtype=np.dtype(dtype).name):
                        np.testing.assert_array_equal(load(stored.astype(dtype)), labels)
                invalid = {257: np.int16, 258: np.int16, 259: np.int16, 6: np.int16, -1: np.int16, 2.5: np.float32, np.nan: np.float32}
                for value, dtype in invalid.items():
                    with self.subTest(value=value):
                        values = stored.astype(dtype)
                        values[unassigned] = value
                        with self.assertRaises(ValueError):
                            load(values)
                # An unreadable scaling header is a ValueError, not a nibabel error.
                uncompressed = example.root / "reviewed_scaling.nii"
                nib.save(nib.Nifti1Image(stored.astype(np.int16), affine), str(uncompressed))
                header = bytearray(uncompressed.read_bytes())
                struct.pack_into("<ff", header, 112, 1.0, float("nan"))
                uncompressed.write_bytes(bytes(header))
                with self.assertRaisesRegex(ValueError, "cannot be read"):
                    csm_consistency.load_reviewed_rois(example.output, example.shape, labels_path=uncompressed)


class TwoMapInputBindingTests(unittest.TestCase):
    """Bind two-map outputs to the calibration k-space that ecalib read."""

    def test_record_requires_the_prepared_calibration_kspace(self) -> None:
        """Reject missing, stale, foreign, or malformed input hash records.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            arguments = (example.twix, example.sequence, example.accepted, example.output)
            with synthetic_patches(example):
                csm_consistency.prepare_csm_consistency(*arguments)
                write_synthetic_calibration(example)
                record_path = example.output / csm_consistency.LAYOUT["two_map_input"]
                original = record_path.read_text(encoding="utf-8")
                kspace = example.accepted / "normal" / "bart_inputs" / "kspace_calib"
                foreign = example.root / "foreign_kspace" / "kspace_calib"
                _copy_cfl(kspace, foreign)
                spelled = original.replace(str(example.accepted), "//proc/self/cwd")
                cases = {
                    "missing": None,
                    "//proc names while the cwd is the accepted root": spelled,
                    "relative names while the cwd is the accepted root": original.replace(
                        str(example.accepted) + "/", ""
                    ),
                    "./ names while the cwd is the accepted root": original.replace(
                        str(example.accepted) + "/", "./"
                    ),
                    "../accepted names while the cwd is the accepted root": original.replace(
                        str(example.accepted) + "/", f"../{example.accepted.name}/"
                    ),
                    "stale payload hash": original.replace(sha256_file(kspace.with_suffix(".cfl")), "0" * 64),
                    "foreign copy with identical bytes": original.replace(str(kspace), str(foreign)),
                    "extra line": original + original.splitlines()[0] + "\n",
                    "escaped name": "\\" + original,
                }
                for name, content in cases.items():
                    with self.subTest(case=name):
                        if content is None:
                            record_path.unlink()
                        else:
                            record_path.write_text(content, encoding="utf-8")
                        try:
                            with _inside(example.accepted):
                                if "while the cwd is the accepted root" in name:
                                    # Each spelled name resolves to the prepared k-space here.
                                    for line in content.splitlines():
                                        spelled_name = line.split("  ", 1)[1]
                                        self.assertTrue(
                                            os.path.samefile(spelled_name, kspace.with_suffix(Path(spelled_name).suffix))
                                        )
                                with self.assertRaises((ValueError, FileNotFoundError)):
                                    csm_consistency.record_two_map_calibration(*arguments)
                        finally:
                            record_path.write_text(original, encoding="utf-8")
                self.assertFalse((example.output / csm_consistency.LAYOUT["calibration_manifest"]).exists())
                record = csm_consistency.record_two_map_calibration(*arguments)
                self.assertEqual(
                    record["ecalib_input"]["payload_sha256"], sha256_file(kspace.with_suffix(".cfl"))
                )

    def test_shell_calibrate_reuses_outputs_only_through_their_manifest(self) -> None:
        """Run the real calibrate stage with a stub BART and refuse leftovers.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            with synthetic_patches(example):
                csm_consistency.prepare_csm_consistency(
                    example.twix, example.sequence, example.accepted, example.output
                )
            stub = example.root / "stub"
            _write_cfl(stub / "maps", example.maps)
            _write_cfl(stub / "eigen", example.eigenvalues[:, :, :, None, :])
            bart = stub / "bin" / "bart"
            bart.parent.mkdir(parents=True)
            bart.write_text(STUB_BART, encoding="utf-8")
            bart.chmod(bart.stat().st_mode | stat.S_IXUSR)
            environment = {
                **os.environ,
                "PATH": f"{bart.parent}{os.pathsep}{os.environ['PATH']}",
                "STUB_MAPS": str(stub / "maps"),
                "STUB_EIGEN": str(stub / "eigen"),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            command = [
                "bash",
                str(SHELL_SCRIPT),
                "calibrate",
                str(example.twix),
                str(example.sequence),
                str(example.accepted),
                str(example.output),
            ]
            manifest_path = example.output / csm_consistency.LAYOUT["calibration_manifest"]

            def run() -> subprocess.CompletedProcess[str]:
                """Run the calibrate stage once.

                Returns:
                    Completed process with captured output.
                """
                return subprocess.run(command, env=environment, capture_output=True, text=True, timeout=600)

            initial = run()
            self.assertEqual(initial.returncode, 0, initial.stderr)
            logs = example.output / "logs" / "environment"
            recorded_log = json.loads(manifest_path.read_text(encoding="utf-8"))["environment"]["shell_environment_log"]
            self.assertEqual(Path(recorded_log["relative_path"]).parent, Path("logs/environment"))
            self.assertIn("stage: calibrate", recorded_log["text"])
            recorded_path = example.output / recorded_log["relative_path"]
            recorded_bytes = recorded_path.read_bytes()
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            prepared = json.loads(
                (example.output / csm_consistency.LAYOUT["prepare_manifest"]).read_text(encoding="utf-8")
            )
            for key in ("header_sha256", "payload_sha256"):
                self.assertEqual(recorded["ecalib_input"][key], prepared["accepted"]["kspace_calib"][key])
            self.assertEqual(recorded["bart_version"]["output"], "v1.0.00-stub")
            repeated = run()
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("Verifying the recorded diagnostic two-map calibration", repeated.stdout)
            # Reuse writes its own new log and leaves the recorded one unchanged.
            self.assertEqual(recorded_path.read_bytes(), recorded_bytes)
            self.assertEqual(len(list(logs.glob("calibrate_*.txt"))), 2)
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["environment"]["shell_environment_log"],
                recorded_log,
            )
            recorded_path.write_bytes(recorded_bytes + b"edited after recording\n")
            try:
                edited = run()
                self.assertEqual(edited.returncode, 2)
                self.assertIn("environment log", edited.stderr)
            finally:
                recorded_path.write_bytes(recorded_bytes)
            aside = example.root / "aside_calibration.json"
            manifest_path.rename(aside)
            try:
                interrupted = run()
                self.assertEqual(interrupted.returncode, 2)
                self.assertIn("without a calibration manifest", interrupted.stderr)
            finally:
                aside.rename(manifest_path)
            self.assertEqual(run().returncode, 0)


class EnvironmentLogTests(unittest.TestCase):
    """Record one environment log per new manifest and verify it on reuse."""

    def test_recorded_logs_are_verified_and_never_rewritten(self) -> None:
        """Reuse keeps every recorded log and refuses a changed or missing one.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            output = example.output
            arguments = (example.twix, example.sequence, example.accepted, output)
            directory = output / "logs" / "environment"
            directory.mkdir(parents=True)

            def log(stage: str, invocation: int) -> Path:
                """Write one invocation log the way the shell does.

                Args:
                    stage: Stage name.
                    invocation: Invocation number.

                Returns:
                    Path of the new log.
                """
                path = directory / f"{stage}_{invocation}.txt"
                path.write_text(f"stage: {stage}\ninvocation: {invocation}\n", encoding="utf-8")
                return path

            stages = {
                "prepare": functools.partial(csm_consistency.prepare_csm_consistency, *arguments),
                "calibrate": functools.partial(csm_consistency.record_two_map_calibration, *arguments),
                "roi-template": functools.partial(csm_consistency.write_roi_template, *arguments),
                "diagnose": functools.partial(
                    csm_consistency.diagnose_csm_consistency, *arguments, boxes=SYNTHETIC_BOXES
                ),
            }
            with synthetic_patches(example):
                recorded = {}
                for stage, run in stages.items():
                    if stage == "calibrate":
                        write_synthetic_calibration(example)
                    manifest = run(environment_log=log(stage, 1))
                    recorded[stage] = manifest["environment"]["shell_environment_log"]
                    self.assertEqual(recorded[stage]["relative_path"], f"logs/environment/{stage}_1.txt")
                before = {stage: (output / entry["relative_path"]).read_bytes() for stage, entry in recorded.items()}
                for stage, run in stages.items():
                    with self.subTest(reuse=stage):
                        again = run(environment_log=log(stage, 2))
                        self.assertEqual(again["environment"]["shell_environment_log"], recorded[stage])
                for stage, entry in recorded.items():
                    self.assertEqual((output / entry["relative_path"]).read_bytes(), before[stage])

                dependents = {
                    "prepare": tuple(stages),
                    "calibrate": ("calibrate", "diagnose"),
                    "roi-template": ("roi-template", "diagnose"),
                    "diagnose": ("diagnose",),
                }
                for stage, users in dependents.items():
                    path = output / recorded[stage]["relative_path"]
                    for mode in ("modified", "deleted"):
                        for user in users:
                            with self.subTest(log=stage, user=user, mode=mode):
                                if mode == "modified":
                                    path.write_bytes(before[stage] + b"edited after recording\n")
                                else:
                                    path.unlink()
                                try:
                                    with self.assertRaisesRegex(
                                        (ValueError, FileNotFoundError), "environment log"
                                    ):
                                        stages[user](environment_log=log(user, 3))
                                finally:
                                    path.write_bytes(before[stage])

                elsewhere = output / "logs" / "prepare_elsewhere.txt"
                elsewhere.write_text("stage: prepare\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "logs/environment"):
                    stages["prepare"](environment_log=elsewhere)
                with _inside(output):
                    with self.assertRaisesRegex(ValueError, "depends on the process"):
                        stages["prepare"](environment_log="//proc/self/cwd/logs/environment/prepare_1.txt")

    def test_too_small_air_roi_fails_instead_of_dropping_outputs(self) -> None:
        """Require native local rank for both coil bases.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            arguments = (example.twix, example.sequence, example.accepted, example.output)
            boxes = SYNTHETIC_BOXES[:-1] + ("background_air=ro=39:39,lin=0:4,par=0:0",)
            with synthetic_patches(example):
                csm_consistency.prepare_csm_consistency(*arguments)
                write_synthetic_calibration(example)
                csm_consistency.record_two_map_calibration(*arguments)
                csm_consistency.write_roi_template(*arguments)
                with self.assertRaisesRegex(ValueError, "background-air ROI is too small"):
                    csm_consistency.diagnose_csm_consistency(*arguments, boxes=boxes)
            self.assertFalse((example.output / csm_consistency.LAYOUT["diagnostics_manifest"]).exists())


# A relative or process-dependent spelling of a recorded path is refused by the
# stable-path rule itself or by exact equality with a record computed now from
# stable absolute paths.
_PATH_REJECTION = (
    "stable absolute path|differs from the current accepted root|differs from its verified "
    "output record|another ROI source or label file|reviewed-label destination|not the reviewed "
    "command|used a different accepted kspace_calib"
)


def _string_leaves(data: Any, keys: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], str]]:
    """Yield every string stored in a decoded manifest.

    Args:
        data: Decoded JSON value.
        keys: Keys and list indices leading to ``data``.

    Yields:
        ``(keys, value)`` for each string leaf.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            yield from _string_leaves(value, (*keys, key))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from _string_leaves(value, (*keys, index))
    elif isinstance(data, str):
        yield keys, data


def _replace_recorded(data: Any, keys: Sequence[Any], recorded: str, value: str) -> None:
    """Replace one recorded manifest string after checking its current value.

    Args:
        data: Decoded manifest, edited in place.
        keys: Keys and list indices leading to the field.
        recorded: Value the field must hold before the edit.
        value: Replacement value.

    Raises:
        AssertionError: If the field does not hold ``recorded``.
    """
    for key in keys[:-1]:
        data = data[key]
    if data[keys[-1]] != recorded:
        raise AssertionError(f"{keys} holds {data[keys[-1]]!r}, not {recorded!r}")
    data[keys[-1]] = value


def _cwd_spellings(cwd: Path, recorded: str, descriptor: int) -> dict[str, str]:
    """Spell a recorded path relative to, or through, the working directory.

    Args:
        cwd: Current working directory, held open by ``descriptor``.
        recorded: Recorded absolute path below ``cwd``.
        descriptor: Open descriptor on ``cwd``.

    Returns:
        Case name mapped to a spelling that names ``recorded`` only from this
        working directory or process.
    """
    plain = os.path.relpath(recorded, cwd)
    return {
        "relative": plain,
        "./relative": f"./{plain}",
        "../<cwd>/relative": os.path.join("..", cwd.name, plain),
        "//proc/self/cwd": f"//proc/self/cwd/{plain}",
        "/dev/fd": f"/dev/fd/{descriptor}/{plain}",
    }


def _reviewed_label_file(example: SimpleNamespace) -> Path:
    """Draw the synthetic ROI boxes on the exported template as reviewed labels.

    Args:
        example: Synthetic fixture after the ROI-template stage.

    Returns:
        Reviewed label NIfTI written at the recorded destination.
    """
    import nibabel as nib

    template = json.loads(
        (example.output / csm_consistency.LAYOUT["roi_template_manifest"]).read_text(encoding="utf-8")
    )
    geometry = template["geometry"]
    labels, _ = csm_consistency.roi.labels_from_boxes(SYNTHETIC_BOXES, example.shape)
    stored, _, _ = csm_consistency.roi.to_stored_orientation(
        labels, tuple(geometry["array_flips"]), np.asarray(geometry["source_affine"])
    )
    path = Path(template["reviewed_label_destination"])
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = nib.load(template["reference_nifti"]["path"]).affine
    nib.save(nib.Nifti1Image(stored.astype(np.uint8), affine), str(path))
    return path


class StableRecordedPathTests(unittest.TestCase):
    """Bind every recorded path only through stable absolute paths."""

    def test_every_recorded_path_refuses_cwd_dependent_spellings(self) -> None:
        """Refuse relative and process-dependent spellings of every recorded path.

        Every recorded absolute path in the four manifests is replaced by
        relative, ``/proc``, and ``/dev`` spellings. Each spelling is checked
        from a working directory where it names the recorded file, so only the
        stable-path rule or an exact comparison with a stable record can
        reject it. Both ROI sources and all four environment logs are covered.

        Returns:
            None.
        """
        for source in ("boxes", "label NIfTI"):
            with self.subTest(roi_source=source), tempfile.TemporaryDirectory() as temporary:
                example = build_synthetic_example(Path(temporary))
                output = example.output
                arguments = (example.twix, example.sequence, example.accepted, output)
                logs = output / "logs" / "environment"
                logs.mkdir(parents=True)

                def invocation_log(stage: str) -> Path:
                    """Write one environment log the way the shell does.

                    Args:
                        stage: Manifest key of the stage.

                    Returns:
                        Path of the new log.
                    """
                    path = logs / f"{stage}_recorded.txt"
                    path.write_text(f"stage: {stage}\n", encoding="utf-8")
                    return path

                stages = {
                    "prepare_manifest": functools.partial(csm_consistency.prepare_csm_consistency, *arguments),
                    "calibration_manifest": functools.partial(csm_consistency.record_two_map_calibration, *arguments),
                    "roi_template_manifest": functools.partial(csm_consistency.write_roi_template, *arguments),
                }
                with synthetic_patches(example):
                    for key, stage in stages.items():
                        if key == "calibration_manifest":
                            write_synthetic_calibration(example)
                        stage(environment_log=invocation_log(key))
                    roi_source: dict[str, Any] = (
                        {"boxes": SYNTHETIC_BOXES}
                        if source == "boxes"
                        else {"labels_path": _reviewed_label_file(example)}
                    )
                    stages["diagnostics_manifest"] = functools.partial(
                        csm_consistency.diagnose_csm_consistency, *arguments, **roi_source
                    )
                    stages["diagnostics_manifest"](environment_log=invocation_log("diagnostics_manifest"))

                    fields = set()
                    for key, stage in stages.items():
                        manifest_path = output / csm_consistency.LAYOUT[key]
                        original = manifest_path.read_bytes()
                        for field, recorded in _string_leaves(json.loads(original)):
                            if not recorded.startswith(f"{example.root}/"):
                                continue
                            fields.add((key, ".".join(map(str, field))))
                            cwd = next(
                                directory
                                for directory in (output, example.accepted, example.root)
                                if recorded.startswith(f"{directory}/")
                            )
                            with _inside(cwd) as descriptor:
                                for spelling, spelled in _cwd_spellings(cwd, recorded, descriptor).items():
                                    with self.subTest(
                                        roi_source=source, manifest=key, field=".".join(map(str, field)), spelling=spelling
                                    ):
                                        # The spelling names the recorded file from this directory.
                                        self.assertEqual(os.path.realpath(spelled), os.path.realpath(recorded))
                                        for suffix in ("", ".hdr"):
                                            if os.path.exists(recorded + suffix):
                                                self.assertTrue(os.path.samefile(spelled + suffix, recorded + suffix))
                                                break
                                        edited = json.loads(original)
                                        _replace_recorded(edited, field, recorded, spelled)
                                        manifest_path.write_text(json.dumps(edited), encoding="utf-8")
                                        try:
                                            with self.assertRaisesRegex((ValueError, FileExistsError), _PATH_REJECTION):
                                                stage()
                                        finally:
                                            manifest_path.write_bytes(original)

                    # The walk must reach every kind of recorded path.
                    required = {
                        ("prepare_manifest", "sources.twix.path"),
                        ("prepare_manifest", "sources.sequence.path"),
                        ("prepare_manifest", "sources.accepted_normal_manifest.path"),
                        ("prepare_manifest", "accepted.root"),
                        ("prepare_manifest", "accepted.manifest.path"),
                        ("prepare_manifest", "accepted.ecalib_command.path"),
                        ("prepare_manifest", "accepted.fista_r0_command.path"),
                        ("prepare_manifest", "accepted.kspace_calib.base"),
                        ("prepare_manifest", "accepted.wave_kspace.base"),
                        ("prepare_manifest", "noise.noise_covariance.path"),
                        ("prepare_manifest", "physical_calibration.manifest.path"),
                        ("prepare_manifest", "physical_calibration.kspace.base"),
                        ("prepare_manifest", "map1_reference.path"),
                        ("calibration_manifest", "prepare_manifest.path"),
                        ("calibration_manifest", "command.record.path"),
                        ("calibration_manifest", "ecalib_input.path"),
                        ("calibration_manifest", "maps.base"),
                        ("calibration_manifest", "eigenvalues.base"),
                        ("calibration_manifest", "ecalib_log.path"),
                        ("roi_template_manifest", "reference_nifti.path"),
                        ("roi_template_manifest", "label_template_nifti.path"),
                        ("roi_template_manifest", "instructions.path"),
                        ("roi_template_manifest", "reviewed_label_destination"),
                        ("diagnostics_manifest", "inputs.prepare_manifest.path"),
                        ("diagnostics_manifest", "outputs.reports/roi_summary.csv.path"),
                        ("diagnostics_manifest", "report.path"),
                        ("diagnostics_manifest", "figures.0.path"),
                        *((key, "environment.shell_environment_log.path") for key in stages),
                    }
                    if source == "label NIfTI":
                        required.add(("diagnostics_manifest", "inputs.rois.file.path"))
                    self.assertLessEqual(required, fields)
                    self.assertGreater(len(fields), 100)
                    for key, stage in stages.items():
                        with self.subTest(roi_source=source, unchanged=key):
                            recorded = json.loads((output / csm_consistency.LAYOUT[key]).read_text(encoding="utf-8"))
                            self.assertEqual(stage(), recorded)

    def test_prepare_file_records_name_their_files(self) -> None:
        """Refuse source, accepted-manifest, and command records naming another file.

        These records used to be checked by their hash alone. A nonexistent
        path or an identical copy elsewhere now fails, although the size and
        SHA-256 in the record still match.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            run_synthetic_pipeline(example)
            arguments = (example.twix, example.sequence, example.accepted, example.output)
            inputs = example.accepted / "normal" / "bart_inputs"
            outputs = example.accepted / "normal" / "bart_output"
            prepare_path = example.output / csm_consistency.LAYOUT["prepare_manifest"]
            original = prepare_path.read_bytes()
            copies = example.root / "identical_copies"
            copies.mkdir()
            with synthetic_patches(example):
                for field, target in (
                    (("sources", "twix", "path"), example.twix),
                    (("sources", "sequence", "path"), example.sequence),
                    (("sources", "accepted_normal_manifest", "path"), inputs / "manifest.json"),
                    (("accepted", "manifest", "path"), inputs / "manifest.json"),
                    (("accepted", "ecalib_command", "path"), outputs / "ecalib_command.txt"),
                    (("accepted", "fista_r0_command", "path"), outputs / "fista_r0" / "wave_command.txt"),
                ):
                    copy = copies / "_".join(field)
                    shutil.copy2(target, copy)
                    for case, value in (("nonexistent", f"/nonexistent/{target.name}"), ("identical copy", str(copy))):
                        with self.subTest(field=".".join(field), case=case):
                            edited = json.loads(original)
                            _replace_recorded(edited, field, str(target), value)
                            prepare_path.write_text(json.dumps(edited), encoding="utf-8")
                            try:
                                with self.assertRaisesRegex(ValueError, "record names|TWIX path differs"):
                                    csm_consistency.prepare_csm_consistency(*arguments)
                            finally:
                                prepare_path.write_bytes(original)
                self.assertEqual(csm_consistency.prepare_csm_consistency(*arguments), json.loads(original))


class ReuseIntegrityTests(unittest.TestCase):
    """Reuse a stage only when every recorded input and output is unchanged."""

    @staticmethod
    def _tamper(path: Path) -> None:
        """Change one file while keeping JSON files parseable.

        Args:
            path: File to modify in place.
        """
        if path.suffix == ".json":
            path.write_text(path.read_text(encoding="utf-8").rstrip("\n") + " \n", encoding="utf-8")
            return
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 0xFF
        path.write_bytes(bytes(payload))

    def test_every_recorded_artifact_is_verified_before_reuse(self) -> None:
        """Detect deleted or modified artifacts, relocation, and leftovers.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            manifest = run_synthetic_pipeline(example)
            output = example.output
            arguments = (example.twix, example.sequence, example.accepted, output)
            stages = {
                "prepare": functools.partial(csm_consistency.prepare_csm_consistency, *arguments),
                "record-calibration": functools.partial(
                    csm_consistency.record_two_map_calibration, *arguments
                ),
                "roi-template": functools.partial(csm_consistency.write_roi_template, *arguments),
                "diagnose": functools.partial(
                    csm_consistency.diagnose_csm_consistency, *arguments, boxes=SYNTHETIC_BOXES
                ),
            }
            every_stage = tuple(stages)
            consumers = {
                **dict.fromkeys(
                    (
                        "inputs/noise/noise_covariance.npy",
                        "inputs/physical_calibration/physical_set4_kspace.cfl",
                        "csm/map1/accepted_map1_reference.json",
                        "manifests/physical_calibration.json",
                    ),
                    every_stage,
                ),
                "manifests/csm_consistency_prepare.json": ("record-calibration", "roi-template", "diagnose"),
                **dict.fromkeys(
                    (
                        "csm/map2_uncropped/ecalib_command.txt",
                        "csm/map2_uncropped/ecalib_input.sha256",
                        "csm/map2_uncropped/coil_sens.cfl",
                        "csm/eigenvalues/ev_m2_c0.cfl",
                        "logs/bart_version.txt",
                        "logs/bart_binary.sha256",
                        "logs/ecalib_m2_c0.log",
                    ),
                    ("record-calibration", "diagnose"),
                ),
                **dict.fromkeys(
                    (
                        "rois/template/reference_fista_r0_magnitude_ras.nii.gz",
                        "rois/template/csm_consistency_roi_labels_template.nii.gz",
                        "rois/template/README.txt",
                    ),
                    ("roi-template", "diagnose"),
                ),
                "manifests/two_map_calibration.json": ("diagnose",),
                "manifests/roi_template.json": ("diagnose",),
                "reports/csm_consistency_report.md": ("diagnose",),
            }
            # verify_diagnostics checks every recorded output with one loop, so
            # the lowest and highest sorted arrays plus one output of every kind suffice.
            recorded = sorted(manifest["outputs"])
            arrays = [key for key in recorded if key.endswith(".npy")]
            sample = {
                arrays[0],
                arrays[-1],
                next(key for key in recorded if key.startswith("diagnostics/local_rank/")),
                next(key for key in recorded if key.endswith(".nii.gz")),
                "reports/roi_summary.csv",
                *(key for key in recorded if key.endswith(".png")),
            }
            self.assertEqual(sum(key.endswith(".png") for key in sample), len(manifest["figures"]))
            consumers.update(dict.fromkeys(sorted(sample), ("diagnose",)))
            with synthetic_patches(example):
                for relative, dependents in consumers.items():
                    path = output / relative
                    original = path.read_bytes()
                    for stage in dependents:
                        for mode in ("modified", "deleted"):
                            with self.subTest(artifact=relative, stage=stage, mode=mode):
                                if mode == "modified":
                                    self._tamper(path)
                                else:
                                    path.unlink()
                                try:
                                    with self.assertRaises((ValueError, FileNotFoundError)):
                                        stages[stage]()
                                finally:
                                    path.write_bytes(original)

                changed = {**manifest["implementation"], "wave_retro_lr/csm_consistency.py": "0" * 64}
                with patch.object(csm_consistency, "implementation_identity", return_value=changed):
                    with self.assertRaisesRegex(FileExistsError, "different implementation"):
                        stages["diagnose"]()

                relocated = example.root / "relocated"
                shutil.copytree(output, relocated, symlinks=True)
                with self.assertRaisesRegex(ValueError, "record names"):
                    csm_consistency.diagnose_csm_consistency(
                        example.twix, example.sequence, example.accepted, relocated, boxes=SYNTHETIC_BOXES
                    )

                import nibabel as nib

                # F3: a new accepted FISTA-r0 NIfTI invalidates the recorded comparison.
                nifti_directory = example.accepted / "normal" / "nifti" / "fista_r0"
                nifti_directory.mkdir(parents=True, exist_ok=True)
                accepted_nifti = nifti_directory / "sub-x_part-mag_fista.nii.gz"
                nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.float32), np.eye(4)), str(accepted_nifti))
                try:
                    for stage in ("roi-template", "diagnose"):
                        with self.subTest(accepted_nifti=stage):
                            with self.assertRaisesRegex(ValueError, "accepted FISTA-r0 NIfTI changed"):
                                stages[stage]()
                finally:
                    accepted_nifti.unlink()

                # F3: identical labels from another source are not silently reused.
                template = json.loads((output / csm_consistency.LAYOUT["roi_template_manifest"]).read_text())
                geometry = template["geometry"]
                box_labels, _ = csm_consistency.roi.labels_from_boxes(SYNTHETIC_BOXES, example.shape)
                stored, _, _ = csm_consistency.roi.to_stored_orientation(
                    box_labels, tuple(geometry["array_flips"]), np.asarray(geometry["source_affine"])
                )
                reviewed = example.root / "reviewed_equivalent.nii.gz"
                reference_affine = nib.load(template["reference_nifti"]["path"]).affine
                nib.save(nib.Nifti1Image(stored.astype(np.uint8), reference_affine), str(reviewed))
                with self.assertRaisesRegex(FileExistsError, "another ROI source"):
                    csm_consistency.diagnose_csm_consistency(*arguments, labels_path=reviewed)

                # F4: a trimmed output list cannot hide deleted outputs, and every
                # diagnose-owned file must be recorded.
                manifest_path = output / csm_consistency.LAYOUT["diagnostics_manifest"]
                original_manifest = manifest_path.read_bytes()
                kept = "reports/roi_summary.csv"
                trimmed = json.loads(original_manifest)
                trimmed["outputs"] = {kept: trimmed["outputs"][kept]}
                trimmed["figures"] = []
                moved = example.root / "trimmed_outputs"
                others = [relative for relative in manifest["outputs"] if relative != kept]
                manifest_path.write_text(json.dumps(trimmed), encoding="utf-8")
                for relative in others:
                    (moved / relative).parent.mkdir(parents=True, exist_ok=True)
                    (output / relative).rename(moved / relative)
                try:
                    with self.assertRaisesRegex(ValueError, "fixed output contract"):
                        stages["diagnose"]()
                finally:
                    for relative in others:
                        (moved / relative).rename(output / relative)
                    manifest_path.write_bytes(original_manifest)
                extra = output / csm_consistency.LAYOUT["local_rank"] / "unrecorded.npy"
                extra.write_bytes(b"not recorded")
                try:
                    with self.assertRaisesRegex(ValueError, "not recorded"):
                        stages["diagnose"]()
                finally:
                    extra.unlink()

                # Coordinated removal: a summary, its output record, and its file
                # removed together must not shrink the fixed output contract.
                removals = (
                    ("roi_summaries", "rnr1_hann", "diagnostics/coil_projection_residuals/rnr1_hann.npy"),
                    ("native_grid_roi_summaries", "e1_pca_ro5", "diagnostics/local_rank/e1_pca_ro5.npy"),
                    (None, None, "diagnostics/coil_projection_residuals/lambda1_ras.nii.gz"),
                    ("figures", None, "diagnostics/roi_overlays/roi_overlay.png"),
                )
                for summary_key, metric, relative in removals:
                    with self.subTest(coordinated_removal=relative):
                        edited = json.loads(original_manifest)
                        if summary_key == "figures":
                            edited["figures"] = [
                                figure for figure in edited["figures"] if figure["relative_path"] != relative
                            ]
                        elif summary_key is not None:
                            del edited[summary_key][metric]
                        del edited["outputs"][relative]
                        manifest_path.write_text(json.dumps(edited), encoding="utf-8")
                        stash = example.root / "coordinated" / relative
                        stash.parent.mkdir(parents=True, exist_ok=True)
                        (output / relative).rename(stash)
                        try:
                            with self.assertRaisesRegex(ValueError, "fixed output contract"):
                                stages["diagnose"]()
                        finally:
                            stash.rename(output / relative)
                            manifest_path.write_bytes(original_manifest)
                for description, mutate_key, message in (
                    ("summary removed alone", ("roi_summaries", "rnr1_hann", None), "fixed metric set"),
                    ("one ROI row removed", ("roi_summaries", "lambda1", "fringe"), "fixed ROI and control masks"),
                ):
                    with self.subTest(summary_edit=description):
                        edited = json.loads(original_manifest)
                        key, metric, mask = mutate_key
                        if mask is None:
                            del edited[key][metric]
                        else:
                            del edited[key][metric][mask]
                        manifest_path.write_text(json.dumps(edited), encoding="utf-8")
                        try:
                            with self.assertRaisesRegex(ValueError, message):
                                stages["diagnose"]()
                        finally:
                            manifest_path.write_bytes(original_manifest)

                # Process-dependent spellings of a recorded output path.
                prepare_path = output / csm_consistency.LAYOUT["prepare_manifest"]
                original_prepare = prepare_path.read_bytes()
                link = example.root / "output_cwd_link"
                link.symlink_to("/proc/self/cwd")
                with _inside(output) as descriptor:
                    spellings = _process_dependent_spellings(
                        output, "inputs/noise/noise_covariance.npy", descriptor, link
                    )
                    for name, spelled in spellings.items():
                        with self.subTest(recorded_path=name):
                            edited = json.loads(original_prepare)
                            edited["noise"]["noise_covariance"]["path"] = spelled
                            prepare_path.write_text(json.dumps(edited), encoding="utf-8")
                            try:
                                with self.assertRaisesRegex(ValueError, "record names"):
                                    stages["prepare"]()
                            finally:
                                prepare_path.write_bytes(original_prepare)

                # Wrongly typed records fail as ValueError, not as a traceback.
                def twix_as_list(data: dict[str, Any]) -> None:
                    """Replace the recorded TWIX identity with a list.

                    Args:
                        data: Prepare manifest.
                    """
                    data["sources"]["twix"] = [1]

                def base_as_number(data: dict[str, Any]) -> None:
                    """Replace the recorded two-map CSM base with a number.

                    Args:
                        data: Calibration manifest.
                    """
                    data["maps"]["base"] = 7

                def figures_as_null(data: dict[str, Any]) -> None:
                    """Replace the recorded figure list with null.

                    Args:
                        data: Diagnostics manifest.
                    """
                    data["figures"] = None

                for key, mutate, stage in (
                    ("prepare_manifest", twix_as_list, "prepare"),
                    ("calibration_manifest", base_as_number, "record-calibration"),
                    ("diagnostics_manifest", figures_as_null, "diagnose"),
                ):
                    with self.subTest(malformed=key):
                        target = output / csm_consistency.LAYOUT[key]
                        original = target.read_bytes()
                        data = json.loads(original)
                        mutate(data)
                        target.write_text(json.dumps(data), encoding="utf-8")
                        try:
                            with self.assertRaises(ValueError):
                                stages[stage]()
                        finally:
                            target.write_bytes(original)

                aside = example.root / "aside"
                aside.mkdir()
                for key, stage, message in (
                    ("diagnostics_manifest", "diagnose", "without a diagnostics manifest"),
                    ("prepare_manifest", "prepare", "without a prepare manifest"),
                ):
                    with self.subTest(interrupted=stage):
                        moved = aside / csm_consistency.LAYOUT[key].name
                        (output / csm_consistency.LAYOUT[key]).rename(moved)
                        try:
                            with self.assertRaisesRegex(FileExistsError, message):
                                stages[stage]()
                        finally:
                            moved.rename(output / csm_consistency.LAYOUT[key])

                manifest_keys = {
                    "prepare": "prepare_manifest",
                    "record-calibration": "calibration_manifest",
                    "roi-template": "roi_template_manifest",
                    "diagnose": "diagnostics_manifest",
                }
                for name, stage in stages.items():
                    with self.subTest(unchanged=name):
                        stored = json.loads((output / csm_consistency.LAYOUT[manifest_keys[name]]).read_text())
                        self.assertEqual(stage()["created_at_utc"], stored["created_at_utc"])
                self.assertEqual(stages["diagnose"]()["created_at_utc"], manifest["created_at_utc"])


class SyntheticDiagnosticsTests(unittest.TestCase):
    """Exercise the complete Python pipeline on a synthetic rank-1/rank-2 phantom."""

    def test_rank_two_pileup_is_localized_and_reported(self) -> None:
        """Localize one-map insufficiency to the rank-2 pile-up and keep provenance.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            example = build_synthetic_example(Path(temporary))
            manifest = run_synthetic_pipeline(example)
            self.assertEqual(manifest["status"], "mprage_csm_consistency_diagnostics_ready")
            self.assertTrue(all(value is False for value in manifest["flags"].values()))
            summaries = manifest["roi_summaries"]

            def median(metric: str, name: str) -> float:
                """Return one per-ROI median from the manifest summaries.

                Args:
                    metric: Metric key.
                    name: ROI name.

                Returns:
                    Median value.
                """
                return summaries[metric][name]["quantiles"]["0.5"]

            self.assertGreater(median("rho1_hann", "metal_pileup"), 2.0 * median("rho1_hann", "preserved_anatomy"))
            self.assertLess(median("rho2_hann", "metal_pileup"), 0.5 * median("rho1_hann", "metal_pileup"))
            self.assertEqual(summaries["lambda2"]["metal_pileup"]["fraction_ge"]["0.8"], 1.0)
            self.assertEqual(summaries["lambda2"]["preserved_anatomy"]["fraction_ge"]["0.8"], 0.0)
            self.assertGreaterEqual(manifest["map1_reproduction"]["median"], 0.999)
            self.assertLess(manifest["pca_basis_reproduction"]["relative_residual"], 1e-3)
            partner = manifest["alias_partner_test"][0]
            self.assertEqual(partner["partner_shift"], 12)
            self.assertAlmostEqual(partner["observed_overlap"], 1.0)
            noise_model = manifest["noise_model"]
            status = noise_model["rnr_status"]
            self.assertIn(status, ("calibrated", "uncalibrated"))
            # RNR is calibrated only when both covariance checks pass.
            both_compatible = bool(
                noise_model["compatibility"]["compatible"]
                and noise_model["rnr_basis_check"]["compatibility"]["compatible"]
            )
            self.assertEqual(status == "calibrated", both_compatible)
            self.assertEqual(bool(noise_model["rnr_failed_checks"]), not both_compatible)
            for check in (noise_model, noise_model["rnr_basis_check"]):
                self.assertIn("sampling_null_reference", check)
            # The Hann window and zero fill leave fewer independent air vectors than
            # accepted-grid voxels, and even fewer than native unapodized voxels.
            hann_null = noise_model["rnr_basis_check"]["sampling_null_reference"]
            native_null = noise_model["sampling_null_reference"]
            self.assertEqual(hann_null["voxels"], noise_model["air_voxels_accepted_grid"])
            self.assertLess(hann_null["effective_samples"], native_null["effective_samples"])
            self.assertAlmostEqual(native_null["effective_samples"], native_null["voxels"], places=6)
            report = Path(manifest["report"]["path"]).read_text(encoding="utf-8")
            for heading in ("Noise model and RNR status", "alias-partner test", "Limitations", "Interpretation policy"):
                self.assertIn(heading, report)
            manifest_text = json.dumps(manifest).lower()
            for term in ("first", "paired", "psf anomaly"):
                self.assertNotIn(term, report.lower())
                self.assertNotIn(term, manifest_text)
            for record in manifest["figures"]:
                self.assertTrue(Path(record["path"]).is_file())
            for relative, record in manifest["outputs"].items():
                self.assertEqual(sha256_file(example.output / relative), record["sha256"])
            arguments = (example.twix, example.sequence, example.accepted, example.output)
            with synthetic_patches(example):
                again = csm_consistency.diagnose_csm_consistency(*arguments, boxes=SYNTHETIC_BOXES)
                self.assertEqual(again["created_at_utc"], manifest["created_at_utc"])
                with self.assertRaises(FileExistsError):
                    csm_consistency.diagnose_csm_consistency(
                        *arguments, boxes=SYNTHETIC_BOXES[:-1] + ("background_air=ro=36:39,lin=all,par=all",)
                    )


if __name__ == "__main__":
    unittest.main()
