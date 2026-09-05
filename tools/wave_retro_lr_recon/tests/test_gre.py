"""Focused interface and scientific-contract tests for measured multi-echo GRE."""

from __future__ import annotations

import inspect
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TOOL_ROOT / "scripts"
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import create_cfl, open_cfl, read_shape  # noqa: E402
from wave_retro_lr.gre import (  # noqa: E402
    GRE_BART_ARRAY_AXIS_FLIPS,
    GRE_GEOMETRY_IDS,
    GRE_LOGICAL_AXIS_ROLES,
    GRE_SHARED_WAVELET_LAMBDA,
    LOW_RESOLUTION_LIN_BOUNDS,
    WAVELET_SELECTION_BASENAME,
    WAVELET_SELECTION_SHA256,
    _crop_gre_wave_coil_in_pe,
    _evaluate_echo_psfs,
    _normalize_psf_settings,
    _read_shared_psf_coefficients,
    _resolve_gre_twix_logical_matrix,
    _shared_calibration_id,
    _validate_recoverable_retro_directory,
    _write_shared_psf_coefficients,
    bart_wave_restoration_factor,
    build_gre_wave_command,
    gre_echo_ids,
    gre_cases,
    gre_wavelet_selection_provenance,
    prepare_normal_gre,
    prepare_retro_gre,
    resolve_gre_wavelet_lambda,
    restore_bart_wave_image,
    validate_gre_echo_consistency,
    validate_gre_echo_sampling,
    validate_gre_sequence,
)
from wave_retro_lr.retrospective import resample_sensitivity_maps  # noqa: E402
from scripts.convert_gre_bart_to_nifti import _canonicalize_saved_nifti  # noqa: E402
from scripts.prepare_gre_normal import _parser as normal_parser  # noqa: E402
from scripts.prepare_gre_retro import _parser as retro_parser  # noqa: E402


class GreGeometryAndEchoTests(unittest.TestCase):
    """Verify geometry, echo binding, calibration sharing, and PSFs."""

    def test_exact_geometry_crop_and_shared_lambda_defaults(self) -> None:
        """Require one shared Wavelet value per approved geometry."""

        cases = gre_cases()
        self.assertEqual(cases["native_r3x1"].matrix_ro_lin_par, (250, 250, 72))
        self.assertEqual(cases["native_r3x2"].matrix_ro_lin_par, (250, 250, 72))
        low = cases["lin_low_resolution_r3x2"]
        self.assertEqual(low.matrix_ro_lin_par, (250, 148, 72))
        self.assertEqual(low.crop_bounds_from_native[1], LOW_RESOLUTION_LIN_BOUNDS)
        self.assertEqual(tuple(cases), GRE_GEOMETRY_IDS)
        for case in cases.values():
            self.assertEqual(case.shared_wavelet_lambda, GRE_SHARED_WAVELET_LAMBDA)
            self.assertEqual(case.to_json()["shared_wavelet_lambda"], 0.015)
            self.assertNotIn("wavelet_lambda_by_echo", case.to_json())
        self.assertEqual(WAVELET_SELECTION_BASENAME, "wavelet_shared_echo_selection.json")
        self.assertEqual(
            WAVELET_SELECTION_SHA256,
            "0c43a9d31672e90ad851decfca66c253c362cbd67ca5ba97c4fd8ef1f5a61afd",
        )

    def test_shared_selection_rejects_invalid_echoes_values_method_and_provenance(self) -> None:
        """Hard-fail every superseded or incomplete selection representation."""

        self.assertEqual(
            resolve_gre_wavelet_lambda(
                "native_r3x1",
                echo_ids=("echo-01", "echo-02", "echo-03"),
                wavelet_lambda_by_echo={
                    "echo-01": 0.015,
                    "echo-02": 0.015,
                    "echo-03": 0.015,
                },
            ),
            0.015,
        )
        invalid_calls = (
            {"geometry_id": "unknown"},
            {"geometry_id": "native_r3x1", "echo_ids": ("echo-02",)},
            {
                "geometry_id": "native_r3x1",
                "echo_ids": ("echo-01", "echo-02"),
                "wavelet_lambda_by_echo": {"echo-01": 0.015},
            },
            {
                "geometry_id": "native_r3x1",
                "echo_ids": ("echo-01", "echo-02"),
                "wavelet_lambda_by_echo": (0.015, 0.020),
            },
            {"geometry_id": "native_r3x1", "method": "llr"},
            {
                "geometry_id": "native_r3x1",
                "selection_manifest_basename": "wavelet_selection.json",
            },
            {"geometry_id": "native_r3x1", "selection_manifest_sha256": "0" * 64},
        )
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                resolve_gre_wavelet_lambda(**arguments)

        provenance = gre_wavelet_selection_provenance("native_r3x1", 3)
        self.assertEqual(provenance["echo_ids"], ["echo-01", "echo-02", "echo-03"])
        self.assertEqual(provenance["echo_count"], 3)
        self.assertEqual(provenance["lambda_constraint"], "shared_within_geometry")
        self.assertEqual(provenance["reconstruction_coupling"], "none; reconstruct each echo separately")
        self.assertFalse(provenance["llr_selection_recorded"])

    def test_arbitrary_echo_counts_use_identical_ordered_r3x1_lattices(self) -> None:
        """Accept single/multi-echo counters and reject a missing coordinate."""

        coordinates = [(line, par) for line in range(2, 250, 3) for par in range(72)]
        for echo_count in (1, 3):
            times = tuple(0.005 * index for index in range(1, echo_count + 1))
            lines = [value[0] for value in coordinates] * echo_count
            pars = [value[1] for value in coordinates] * echo_count
            echoes = [index for index in range(echo_count) for _ in coordinates]
            mask, metadata = validate_gre_echo_sampling(
                lines, pars, echoes, echo_times_s=times
            )
            self.assertEqual(mask.shape, (250, 72))
            self.assertEqual(int(mask.sum()), 83 * 72)
            self.assertEqual(
                [item["echo"] for item in metadata["echoes"]],
                list(range(1, echo_count + 1)),
            )
            self.assertEqual(
                [item["te_s"] for item in metadata["echoes"]], list(times)
            )
            with self.assertRaisesRegex(ValueError, f"Echo {echo_count}"):
                validate_gre_echo_sampling(
                    lines[:-1], pars[:-1], echoes[:-1], echo_times_s=times
                )

    def test_sequence_and_twix_echo_count_and_te_must_match(self) -> None:
        """Reject count or TE disagreement before reconstruction preparation."""

        self.assertEqual(
            validate_gre_echo_consistency((0.01,), (0.01,)), (0.01,)
        )
        self.assertEqual(
            validate_gre_echo_consistency((0.01, 0.02, 0.03), (0.01, 0.02, 0.03)),
            (0.01, 0.02, 0.03),
        )
        with self.assertRaisesRegex(ValueError, "echo counts disagree"):
            validate_gre_echo_consistency((0.01,), (0.01, 0.02))
        with self.assertRaisesRegex(ValueError, "echo times disagree"):
            validate_gre_echo_consistency((0.01, 0.02), (0.01, 0.021))

    def test_stale_header_partition_count_is_not_used_as_geometry(self) -> None:
        """Accept lPartitions=1 when measured PAR exactly covers sequence Nz."""

        matrix, evidence = _resolve_gre_twix_logical_matrix(
            base_resolution=250,
            header_partition_count=1,
            mdh_partitions=np.repeat(np.arange(72), 83),
            expected_matrix_ro_lin_par=(250, 250, 72),
        )
        self.assertEqual(matrix, (250, 250, 72))
        self.assertEqual(evidence["raw_header_partition_count"], 1)
        self.assertFalse(evidence["raw_header_partition_count_used_as_geometry"])
        self.assertEqual(evidence["mdh_partition_count"], 72)
        self.assertTrue(evidence["mdh_partition_support_matches_sequence"])

        with self.assertRaisesRegex(ValueError, "MDH PAR support"):
            _resolve_gre_twix_logical_matrix(
                base_resolution=250,
                header_partition_count=72,
                mdh_partitions=np.arange(71),
                expected_matrix_ro_lin_par=(250, 250, 72),
            )

    def test_sequence_validation_requires_authoritative_exact_geometry(self) -> None:
        """Accept arbitrary positive echo counts while enforcing exact geometry."""

        sequence = SimpleNamespace(definitions={"FOV": [0.22, 0.22, 0.18], "TargetFOV": [9, 9, 9]})
        cfg = {
            "Nx": 250,
            "Ny": 250,
            "Nz": 72,
            "Nx_os": 1000,
            "Necho": 3,
            "orientation": "TRA",
            "Ry": 3,
            "Rz": 1,
            "Ny_meas": 83,
            "Nz_meas": 72,
            "FOVxyz_m": (0.22, 0.22, 0.18),
            "TE_s": np.asarray((0.005, 0.010, 0.015)),
        }
        native = Mock()
        native._load_sequence.return_value = sequence
        native._derive_gre_config.return_value = cfg
        native._split_adc_trajectory.return_value = (np.zeros((3, 2, 1000)), np.zeros((3, 3, 1000)))
        native._detect_image_wave_mode.return_value = "wave"
        _, received, image_lines, calibration_lines = validate_gre_sequence(native, Path("gre.seq"))
        self.assertIs(received, cfg)
        self.assertEqual(image_lines.shape[-1], 1000)
        self.assertEqual(calibration_lines.shape[-1], 1000)
        sequence.definitions = {"TargetFOV": [0.22, 0.22, 0.18]}
        with self.assertRaisesRegex(ValueError, "authoritative FOV"):
            validate_gre_sequence(native, Path("gre.seq"))

    def test_shared_calibration_identity_and_echo_specific_psfs(self) -> None:
        """Reuse byte-identical a/b/c while retaining distinct echo trajectories."""

        length = 12
        coefficients = tuple(np.linspace(0, 0.1, length) for _ in range(3))
        identity = _shared_calibration_id(*coefficients)
        self.assertEqual(identity, _shared_calibration_id(*coefficients))
        trajectories = [
            (np.zeros(length), np.zeros(length)),
            (np.linspace(-1, 1, length), np.linspace(1, -1, length)),
            (np.linspace(-0.5, 0.5, length), np.linspace(0.25, -0.25, length)),
        ]
        psfs = _evaluate_echo_psfs(
            trajectories,
            coefficients,
            {"yflip": 1, "zflip": 1, "Necho": 3},
            gre_cases()["lin_low_resolution_r3x2"],
        )
        self.assertEqual(
            [value.shape for value in psfs], [(12, 148, 72)] * 3
        )
        self.assertFalse(np.array_equal(psfs[0], psfs[1]))
        self.assertFalse(np.array_equal(psfs[1], psfs[2]))
        self.assertLess(max(float(np.max(np.abs(np.abs(value) - 1))) for value in psfs), 2e-6)

        source = (TOOL_ROOT / "wave_retro_lr" / "gre.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("native.fit_wave_psf_deviation_from_projection("), 1)
        self.assertIn('"coefficient_fit_count": 1', source)

    def test_shared_psf_archive_retains_raw_samples_for_scatter_plot(self) -> None:
        """Persist raw samples separately from reconstruction coefficients."""

        processed = tuple(np.linspace(index, index + 1, 8) for index in range(3))
        raw = tuple(value + 0.125 for value in processed)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shared_psf_coefficients.npz"
            _write_shared_psf_coefficients(path, processed, raw)
            loaded_processed, loaded_raw = _read_shared_psf_coefficients(path)
        self.assertIsNotNone(loaded_raw)
        for expected, observed in zip(processed, loaded_processed, strict=True):
            np.testing.assert_array_equal(observed, expected)
        for expected, observed in zip(raw, loaded_raw, strict=True):
            np.testing.assert_array_equal(observed, expected)

        source = (TOOL_ROOT / "wave_retro_lr" / "gre.py").read_text(encoding="utf-8")
        self.assertIn("raw_coefficients=raw_coefficients", source)
        self.assertIn('"raw_coefficient_keys": ["a_raw", "b_raw", "c_raw"]', source)


class GreCsmCommandAndOutputTests(unittest.TestCase):
    """Verify map resampling, BART commands, normalization, and orientation."""

    def test_lr_csm_fourier_resampling_is_rss_normalized(self) -> None:
        """Resample PE only at unchanged readout and normalize coil RSS."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = create_cfl(root / "native", (4, 8, 6, 2, 1))
            source[:, :, :, 0, 0] = 1
            source[:, :, :, 1, 0] = 1j
            source.flush()
            del source
            resample_sensitivity_maps(root / "native", root / "low", target_lin_par=(4, 6))
            self.assertEqual(read_shape(root / "low")[:5], (4, 4, 6, 2, 1))
            maps = np.asarray(open_cfl(root / "low"))
            rss = np.sqrt(np.sum(np.abs(maps[..., 0]) ** 2, axis=3))
            self.assertTrue(np.allclose(rss[rss > 0], 1.0, atol=1e-6))

    def test_retro_wave_crop_preserves_readout_and_slices_only_pe(self) -> None:
        """Match MPRAGE's direct PE crop without interpolation or RO loss."""

        source = np.arange(5 * 6 * 4, dtype=np.float32).reshape(5, 6, 4).astype(
            np.complex64
        )
        mask = np.asarray(
            [[True, False], [False, True], [True, True], [False, False]],
            dtype=bool,
        )
        cropped = _crop_gre_wave_coil_in_pe(
            source,
            lin_bounds=(1, 5),
            par_bounds=(1, 3),
            target_mask=mask,
        )
        expected = np.array(source[:, 1:5, 1:3], copy=True)
        expected *= mask[None, :, :]
        self.assertEqual(cropped.shape, (5, 4, 2))
        np.testing.assert_array_equal(cropped, expected)
        np.testing.assert_array_equal(cropped[:, 0, 0], source[:, 1, 1])

    def test_partial_retro_preparation_resumes_only_known_artifacts(self) -> None:
        """Permit deterministic overwrite of the failed crop's owned files."""

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "sampling_mask.npy").touch()
            (directory / "wave_kspace_echo-01.hdr").touch()
            (directory / "wave_kspace_echo-01.cfl").touch()
            _validate_recoverable_retro_directory(directory, 1)
            (directory / "unowned.txt").touch()
            with self.assertRaisesRegex(FileExistsError, "unexpected entries"):
                _validate_recoverable_retro_directory(directory, 1)

    def test_fista_and_wavelet_command_construction(self) -> None:
        """Build explicit 100-iteration FISTA-r0 and selected Wavelet commands."""

        control = build_gre_wave_command(
            maps="maps", psf="psf1", kspace="data1", output="out1", regularization=0
        )
        selected = build_gre_wave_command(
            maps="maps", psf="psf2", kspace="data2", output="out2", regularization=0.02, gpu=True
        )
        self.assertEqual(control[:7], ["bart", "wave", "-w", "-f", "-r", "0", "-i"])
        self.assertEqual(selected[:8], ["bart", "wave", "-g", "-w", "-f", "-r", "0.02", "-i"])
        self.assertIn("1e-6", control)

    def test_all_echo_counts_use_shared_lambda_and_echo_specific_records(self) -> None:
        """Use r=0.015 for every echo while retaining distinct records."""

        records = []
        echo_ids = gre_echo_ids(3)
        echo_times = (0.010, 0.020, 0.030)
        for geometry_id, case in gre_cases().items():
            for echo_id, te_s in zip(echo_ids, echo_times, strict=True):
                record = {
                    "geometry_id": geometry_id,
                    "echo_id": echo_id,
                    "te_s": te_s,
                    "psf": f"{geometry_id}/psf_{echo_id}",
                    "kspace": f"{geometry_id}/wave_kspace_{echo_id}",
                    "output": f"{geometry_id}/selected_wavelet/{echo_id}/image_wave",
                    "nifti": f"{geometry_id}/selected_wavelet/{echo_id}_magnitude.nii.gz",
                }
                record["command"] = build_gre_wave_command(
                    maps=f"{geometry_id}/coil_sens",
                    psf=record["psf"],
                    kspace=record["kspace"],
                    output=record["output"],
                    regularization=case.shared_wavelet_lambda,
                )
                records.append(record)

        self.assertEqual(len(records), 9)
        self.assertTrue(all(record["command"][5] == "0.015" for record in records))
        for geometry_id in GRE_GEOMETRY_IDS:
            echoes = [record for record in records if record["geometry_id"] == geometry_id]
            for key in ("echo_id", "te_s", "psf", "kspace", "output", "nifti"):
                self.assertEqual(len({record[key] for record in echoes}), 3, key)

    def test_validated_bart_restoration_amplitude_and_phase(self) -> None:
        """Restore norm, extended-grid FFT amplitude, and LIN-dependent phase."""

        factor = bart_wave_restoration_factor((250, 250, 72), 2.5, (1000, 250, 72))
        self.assertAlmostEqual(abs(factor), 2.5 * math.sqrt(1000 * 250 * 72))
        self.assertEqual(factor / abs(factor), 1j * ((-1) ** 125))
        image = np.ones((2, 4, 2), dtype=np.complex64)
        restored = restore_bart_wave_image(image, kspace_norm=2.0, encoding_shape=(8, 4, 2))
        self.assertTrue(np.allclose(restored, 16j))

    def test_orientation_is_gre_specific_and_mprage_remains_unchanged(self) -> None:
        """Require GRE RAS flips without changing sagittal MPRAGE policy."""

        self.assertEqual(GRE_LOGICAL_AXIS_ROLES, ("readout", "phase", "slice"))
        self.assertEqual(GRE_BART_ARRAY_AXIS_FLIPS, (False, True, False))
        mprage = (SCRIPTS / "convert_mprage_bart_to_nifti.py").read_text(encoding="utf-8")
        self.assertIn("MPRAGE_BART_ARRAY_AXIS_FLIPS = (False, False, True)", mprage)

    def test_nifti_canonicalization_uses_only_axis_reorientation(self) -> None:
        """Resolve a noncanonical affine to RAS without changing voxel values."""

        import json
        import nibabel as nib

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nifti = root / "image.nii.gz"
            sidecar = root / "image.json"
            values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            affine = np.asarray([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 3], [0, 0, 0, 1]], dtype=float)
            nib.save(nib.Nifti1Image(values, affine), nifti)
            sidecar.write_text(json.dumps({"EchoNumber": 1}), encoding="utf-8")
            record = _canonicalize_saved_nifti(nifti, sidecar)
            saved = nib.load(str(nifti))
            self.assertEqual(nib.aff2axcodes(saved.affine), ("R", "A", "S"))
            self.assertEqual(record["interpolation"], False)
            self.assertEqual(np.asarray(saved.dataobj).size, values.size)

    def test_psf_settings_accept_explicit_smooth_and_sine_line_bounds(self) -> None:
        """Validate the retained smooth mode and manual sine-line bounds."""

        self.assertEqual(_normalize_psf_settings("smooth", None, None)["coefficient_processing"], "smooth")
        self.assertEqual(_normalize_psf_settings("sine-line", 100, 900)["requested_fit_kx_range"], [100, 900])
        with self.assertRaisesRegex(ValueError, "both"):
            _normalize_psf_settings("sine-line", 100, None)


class GreSampleInterfaceTests(unittest.TestCase):
    """Verify readable samples and the no-BART Python boundary."""

    def test_samples_parse_and_publish_shared_selected_lambda(self) -> None:
        """Keep one shared GRE default and visible echo-specific BART commands."""

        normal = SCRIPTS / "sample_gre_normal_recon.sh"
        retro = SCRIPTS / "sample_gre_retro_lr_recon.sh"
        for script in (normal, retro):
            subprocess.run(["bash", "-n", str(script)], check=True)
            help_result = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True)
            self.assertEqual(help_result.returncode, 0)
            self.assertIn("TWIX.dat OUTPUT_ROOT SEQUENCE.seq", help_result.stdout)
            source = script.read_text(encoding="utf-8")
            self.assertIn("bart ecalib -m 1", source)
            self.assertIn("bart wave -g -w -f -r", source)
            self.assertIn("bart wave -w -f -r", source)
            self.assertIn("wave_command.txt", source)
            self.assertIn('PSF_COEFFICIENT_PROCESSING="sine-line"', source)
            ecalib_commands = [
                line.strip() for line in source.splitlines() if line.strip().startswith("bart ecalib ")
            ]
            self.assertEqual(len(ecalib_commands), 1)
        normal_source = normal.read_text(encoding="utf-8")
        retro_source = retro.read_text(encoding="utf-8")
        for source in (normal_source, retro_source):
            self.assertIn('GRE_SHARED_WAVELET_LAMBDA="0.015"', source)
            self.assertNotIn("ECHO1_LAMBDA", source)
            self.assertNotIn("ECHO2_LAMBDA", source)
            self.assertIn("ECHO_COUNT", source)
            self.assertIn("echo_number <= ECHO_COUNT", source)
            self.assertIn("psf_$echo_label", source)
            self.assertIn("wave_kspace_$echo_label", source)
            self.assertIn("conversion_args+=(--image", source)
        self.assertNotIn("-l -v", normal_source + retro_source)

    def test_gre_preparation_defaults_to_automatic_sine_line(self) -> None:
        """Keep the sample, preparation CLI, and Python API defaults aligned."""

        arguments = ["input.dat", "output", "input.seq"]
        for parser in (normal_parser, retro_parser):
            parsed = parser().parse_args(arguments)
            self.assertEqual(parsed.psf_coefficient_processing, "sine-line")
            self.assertIsNone(parsed.psf_fit_kx_min)
            self.assertIsNone(parsed.psf_fit_kx_max)
        for function in (prepare_normal_gre, prepare_retro_gre):
            parameter = inspect.signature(function).parameters[
                "psf_coefficient_processing"
            ]
            self.assertEqual(parameter.default, "sine-line")

    def test_python_preparation_and_conversion_never_launch_bart(self) -> None:
        """Keep BART execution exclusively in the user-facing Bash samples."""

        paths = [
            TOOL_ROOT / "wave_retro_lr" / "gre.py",
            SCRIPTS / "prepare_gre_normal.py",
            SCRIPTS / "prepare_gre_retro.py",
            SCRIPTS / "prepare_gre_retro_maps.py",
            SCRIPTS / "convert_gre_bart_to_nifti.py",
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("subprocess", source, path.name)
            self.assertNotIn("Popen", source, path.name)

    def test_converter_records_quantitative_complex_and_no_masking(self) -> None:
        """Keep display NIfTI normalization separate from complex source data."""

        source = (SCRIPTS / "convert_gre_bart_to_nifti.py").read_text(encoding="utf-8")
        self.assertIn('action="append"', source)
        self.assertNotIn("--image-echo1", source)
        self.assertNotIn("--image-echo2", source)
        self.assertIn('quantitative / f"echo-{echo_index + 1:02d}_complex.npy"', source)
        self.assertIn('"MagnitudeNIfTIIsDisplayNormalized": True', source)
        self.assertIn('"PresentationMaskApplied": False', source)
        self.assertIn('"GRESharedWaveletSelection": dict(selection)', source)
        self.assertIn('"GRESelectedWaveletLambda": selected_lambda', source)
        self.assertNotIn("brain_mask", source.lower())


if __name__ == "__main__":
    unittest.main()
