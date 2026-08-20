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
    build_conversion_command,
    build_wave_options,
    build_wrapper_command,
    canonical_lambda,
    completed_manifest_reusable,
    failed_run_recoverable,
    recombine_split_complex_bart,
    run,
    run_name,
)


class NamingAndCommandTests(unittest.TestCase):
    """Verify stable naming and exact wrapper/BART option forwarding."""

    def test_names_requested_smoke_cases(self) -> None:
        self.assertEqual(canonical_lambda(1e-3), "1e-3")
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

    def test_rejects_nonpositive_lambda(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            canonical_lambda(0.0)


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
                backend="cpu",
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
