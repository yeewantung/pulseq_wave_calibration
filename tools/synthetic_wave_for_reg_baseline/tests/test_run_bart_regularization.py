"""Tests for the publishable BART Wave regularization driver."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from bart_cfl import open_bart_memmap, sha256_file, write_bart_header  # noqa: E402
from run_bart_regularization import (  # noqa: E402
    _build_parser,
    build_conversion_command,
    build_wave_options,
    build_wrapper_command,
    canonical_lambda,
    completed_manifest_reusable,
    failed_run_recoverable,
    recombine_split_complex_bart,
    resolve_regularization_inputs,
    run,
    run_name,
)


class NamingAndCommandTests(unittest.TestCase):
    """Verify stable naming and exact wrapper/BART option forwarding."""

    def test_names_requested_smoke_cases(self) -> None:
        self.assertEqual(canonical_lambda(1e-3), "1e-3")
        self.assertEqual(canonical_lambda(0), "0")
        self.assertEqual(canonical_lambda(1.4e-2), "1.4e-2")
        self.assertEqual(run_name("wavelet", 1e-3), "wavelet_lambda-1e-3")
        self.assertEqual(run_name("llr", 2e-3, 8), "llr_block-8_lambda-2e-3")
        self.assertEqual(run_name("llr", 1.4e-2, 16), "llr_block-16_lambda-1.4e-2")

    def test_wavelet_options_freeze_fista_settings(self) -> None:
        self.assertEqual(
            build_wave_options(
                "wavelet",
                1e-3,
                block_size=8,
                iterations=100,
                tolerance=1e-6,
                max_eigenvalue=6.70e7,
                backend="cpu",
            ),
            ["-w", "-f", "-i", "100", "-t", "1e-06", "-e", "67000000", "-r", "0.001"],
        )

    def test_llr_wrapper_uses_existing_maps_and_direct_option_section(self) -> None:
        options = build_wave_options(
            "llr",
            2e-3,
            block_size=8,
            iterations=100,
            tolerance=1e-6,
            max_eigenvalue=6.70e7,
            backend="cpu",
        )
        command = build_wrapper_command(
            Path("wrapper.sh"),
            bart_input_dir=Path("inputs"),
            bart_output_dir=Path("output"),
            maps=Path("maps"),
            twix=Path("meas.dat"),
            sequence=Path("sequence.seq"),
            nifti_output_dir=Path("nifti"),
            nifti_subject="subject-llr",
            nifti_suffix="BARTWaveRegularized",
            wave_options=options,
        )
        self.assertEqual(command[:2], ["bash", "wrapper.sh"])
        self.assertEqual(command[command.index("--maps-source") + 1], "existing")
        start = command.index("--wave-options") + 1
        stop = command.index("--end-wave-options")
        self.assertEqual(command[start:stop], options)

    def test_gpu_backend_adds_bart_gpu_flag(self) -> None:
        options = build_wave_options(
            "llr",
            2e-3,
            block_size=8,
            iterations=100,
            tolerance=1e-6,
            max_eigenvalue=6.70e7,
            backend="gpu",
        )
        self.assertIn("-g", options)
        self.assertEqual(options.count("-g"), 1)
        self.assertEqual(options[:5], ["-l", "-v", "-b", "8", "-f"])

    def test_conversion_recovery_calls_upstream_converter(self) -> None:
        command = build_conversion_command(
            Path("recon/bart/run_wave_recon.sh"),
            python=Path("env/bin/python"),
            bart_input_dir=Path("inputs"),
            bart_output_dir=Path("output"),
            twix=Path("meas.dat"),
            sequence=Path("sequence.seq"),
            nifti_output_dir=Path("nifti"),
            nifti_subject="subject-wavelet",
            nifti_suffix="BARTWaveRegularized",
        )
        self.assertEqual(command[:2], ["env/bin/python", "recon/bart/wave_to_nifti.py"])
        self.assertIn("--save-phase", command)

    def test_rejects_negative_lambda(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative and finite"):
            canonical_lambda(-1e-6)


class ManifestInputTests(unittest.TestCase):
    def test_freezes_approved_crop_maps_and_measured_solver_scale(self) -> None:
        from dataset_manifest import load_dataset_manifest

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            example = (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "incoming_r1_dataset.example.json"
            )
            payload = json.loads(example.read_text(encoding="utf-8"))
            payload["outputs"]["root"] = "outputs"
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = load_dataset_manifest(dataset_path)
            dataset.inspection_report.parent.mkdir(parents=True)
            dataset.inspection_report.write_text(
                json.dumps(
                    {
                        "dataset_manifest": dataset.provenance(),
                        "contract_checks": {"all_passed": True},
                    }
                ),
                encoding="utf-8",
            )

            bart_input = root / "bart_inputs"
            bart_input.mkdir()
            bart_manifest = bart_input / "manifest.json"
            bart_manifest.write_text('{"status":"ready"}', encoding="utf-8")
            maps = root / "accepted" / "coil_sens_bart"
            lambda_zero = root / "accepted" / "image_wave"
            maps.parent.mkdir()
            for base, content in ((maps, b"maps"), (lambda_zero, b"lambda-zero")):
                base.with_suffix(".hdr").write_text("# Dimensions\n1\n", encoding="utf-8")
                base.with_suffix(".cfl").write_bytes(content)
            lambda_manifest = root / "accepted" / "manifest.json"
            lambda_manifest.write_text(
                json.dumps(
                    {
                        "status": "lambda0_complete_awaiting_visual_review",
                        "config": {
                            "ecalib_crop": 0.6,
                            "gpu_wave_reconstruction": True,
                        },
                        "dataset_manifest": dataset.provenance(),
                        "bart_input_manifest": {
                            "path": str(bart_manifest),
                            "sha256": sha256_file(bart_manifest),
                        },
                        "ecalib": {
                            "output_base": str(maps),
                            "output_cfl_sha256": sha256_file(maps.with_suffix(".cfl")),
                        },
                        "wave_lambda0": {
                            "output_base": str(lambda_zero),
                            "output_cfl_sha256": sha256_file(
                                lambda_zero.with_suffix(".cfl")
                            ),
                            "maximum_eigenvalue": 12345.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            fake_bart = root / "bart"
            fake_bart.touch()
            args = _build_parser().parse_args(
                [
                    "--lambda-zero-manifest",
                    str(lambda_manifest),
                    "--output-root",
                    str(dataset.output_root / "regularization"),
                    "--regularizer",
                    "wavelet",
                    "--lambda-value",
                    "1e-4",
                ]
            )

            with mock.patch(
                "run_bart_regularization.shutil.which", return_value=str(fake_bart)
            ):
                resolved, provenance, matrix = resolve_regularization_inputs(args)

            self.assertEqual(resolved.backend, "gpu")
            self.assertEqual(resolved.max_eigenvalue, 12345.0)
            self.assertEqual(resolved.maps, maps)
            self.assertEqual(matrix, (256, 256, 256))
            self.assertEqual(
                provenance["dataset_manifest"]["sha256"], dataset.sha256
            )

    def test_explicit_inputs_bind_to_lambda_zero_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            maps = root / "lambda0" / "coil_sens_bart"
            lambda_zero = root / "lambda0" / "image_wave"
            maps.parent.mkdir()
            for base, content in ((maps, b"maps"), (lambda_zero, b"lambda-zero")):
                base.with_suffix(".hdr").write_text("# Dimensions\n1\n", encoding="utf-8")
                base.with_suffix(".cfl").write_bytes(content)
            lambda_manifest = root / "lambda0" / "manifest.json"
            lambda_manifest.write_text(
                json.dumps(
                    {
                        "status": "lambda0_complete_awaiting_visual_review",
                        "config": {
                            "ecalib_crop": 0.6,
                            "gpu_wave_reconstruction": True,
                        },
                        "bart_input_manifest": None,
                        "ecalib": {
                            "output_base": str(maps),
                            "output_cfl_sha256": sha256_file(maps.with_suffix(".cfl")),
                        },
                        "wave_lambda0": {
                            "output_base": str(lambda_zero),
                            "output_cfl_sha256": sha256_file(
                                lambda_zero.with_suffix(".cfl")
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            required_files = {
                "--wrapper": root / "run_wave_recon.sh",
                "--bart": root / "bart",
                "--python": root / "python",
                "--bart-input-dir": root / "inputs",
                "--twix": root / "meas.dat",
                "--sequence": root / "sequence.seq",
            }
            required_files["--bart-input-dir"].mkdir()
            for option, path in required_files.items():
                if option != "--bart-input-dir":
                    path.touch()
            args_list = [
                "--source-lambda-zero-manifest",
                str(lambda_manifest),
                "--maps",
                str(maps),
                "--expected-maps-sha256",
                sha256_file(maps.with_suffix(".cfl")),
                "--lambda-zero-base",
                str(lambda_zero),
                "--output-root",
                str(root / "output"),
                "--regularizer",
                "wavelet",
                "--lambda-value",
                "1.5e-2",
            ]
            for option, path in required_files.items():
                args_list.extend((option, str(path)))

            resolved, provenance, matrix = resolve_regularization_inputs(
                _build_parser().parse_args(args_list)
            )

            self.assertEqual(matrix, (256, 256, 256))
            self.assertEqual(resolved.backend, "gpu")
            self.assertEqual(
                provenance["lambda_zero_manifest"]["sha256"],
                sha256_file(lambda_manifest),
            )


class ResumeTests(unittest.TestCase):
    """Require matching configuration and intact hashes before skipping work."""

    def test_reuses_only_intact_completed_manifest(self) -> None:
        config = {"regularizer": "wavelet", "lambda": 0.001}
        maps_hash = "a" * 64
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bart_output = root / "image_wave.cfl"
            nifti = root / "image.nii.gz"
            bart_output.write_bytes(b"bart")
            nifti.write_bytes(b"nifti")
            manifest = {
                "status": "complete",
                "config": config,
                "maps": {"cfl_sha256": maps_hash},
                "bart_output": {
                    "path": str(bart_output),
                    "sha256": sha256_file(bart_output),
                },
                "nifti_outputs": [
                    {"nifti": str(nifti), "nifti_sha256": sha256_file(nifti)}
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(completed_manifest_reusable(path, config, maps_hash))
            nifti.write_bytes(b"changed")
            self.assertFalse(completed_manifest_reusable(path, config, maps_hash))

    def test_recovers_finished_bart_solve_after_conversion_failure(self) -> None:
        config = {"regularizer": "wavelet", "lambda": 0.001}
        maps_hash = "b" * 64
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "wrapper.log").write_text(
                "Reconstruction... Done.\nTotal time: 12.5 seconds.\n", encoding="utf-8"
            )
            manifest = {
                "status": "failed",
                "config": config,
                "maps": {"cfl_sha256": maps_hash},
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "run_bart_regularization.validate_finite_bart",
                return_value={"norm": 1.0},
            ):
                self.assertTrue(failed_run_recoverable(path, config, maps_hash))

    def test_clean_first_pass_success_finalizes_manifest(self) -> None:
        """A fresh successful wrapper run must not enter recovery-only state."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wrapper = root / "run_wave_recon.sh"
            bart = root / "bart"
            python = root / "python"
            twix = root / "meas.dat"
            sequence = root / "sequence.seq"
            bart_input = root / "inputs"
            output_root = root / "output"
            maps = root / "maps"
            lambda_zero = root / "lambda_zero"
            bart_input.mkdir()
            for path in (wrapper, bart, python, twix, sequence, bart_input / "manifest.json"):
                path.write_text("test", encoding="utf-8")
            for base in (maps, lambda_zero, bart_input / "psf", bart_input / "wave_kspace"):
                base.with_suffix(".hdr").write_text("test", encoding="utf-8")
                base.with_suffix(".cfl").write_bytes(b"test")
            args = argparse.Namespace(
                lambda_zero_manifest=None,
                wrapper=wrapper,
                bart=bart,
                python=python,
                bart_input_dir=bart_input,
                maps=maps,
                expected_maps_sha256="a" * 64,
                lambda_zero_base=lambda_zero,
                output_root=output_root,
                twix=twix,
                sequence=sequence,
                regularizer="llr",
                lambda_value=2e-3,
                block_size=8,
                iterations=100,
                tolerance=1e-6,
                max_eigenvalue=6.70e7,
                backend="gpu",
                subject="test",
                resume=True,
            )
            version = mock.Mock(stdout="v1.0\n", stderr="")
            with (
                mock.patch("run_bart_regularization.sha256_file", return_value="a" * 64),
                mock.patch("run_bart_regularization._run_streamed", return_value=1.0),
                mock.patch(
                    "run_bart_regularization.recombine_split_complex_bart",
                    return_value={"rule": "tested"},
                ),
                mock.patch(
                    "run_bart_regularization.validate_finite_bart",
                    return_value={"norm": 1.0, "all_samples_finite": True},
                ),
                mock.patch("run_bart_regularization._relative_bart_difference", return_value=0.1),
                mock.patch("run_bart_regularization._validate_niftis", return_value=[]),
                mock.patch("run_bart_regularization._parse_bart_log", return_value={}),
                mock.patch("run_bart_regularization.subprocess.run", return_value=version),
            ):
                manifest = run(args)
            self.assertEqual(manifest["status"], "complete")
            self.assertNotIn("recovered_failure", manifest)


class SplitComplexTests(unittest.TestCase):
    def test_recombines_iter_dimension_real_and_imaginary_components(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split_base = root / "split"
            output_base = root / "combined"
            shape = (3, 2, 2, 1, 1, 1, 1, 1, 2)
            write_bart_header(split_base, shape)
            split = np.memmap(
                split_base.with_suffix(".cfl"),
                mode="w+",
                dtype=np.complex64,
                shape=shape,
                order="F",
            )
            expected = np.arange(12, dtype=np.float32).reshape((3, 2, 2), order="F")
            split[:, :, :, 0, 0, 0, 0, 0, 0] = expected
            split[:, :, :, 0, 0, 0, 0, 0, 1] = 1j * (expected + 10)
            split.flush()

            record = recombine_split_complex_bart(split_base, output_base)
            combined = np.squeeze(np.asarray(open_bart_memmap(output_base)))
            np.testing.assert_allclose(combined, expected + 1j * (expected + 10))
            self.assertEqual(record["split_shape"][8], 2)
            self.assertEqual(record["recombined_shape"][8], 1)


if __name__ == "__main__":
    unittest.main()
