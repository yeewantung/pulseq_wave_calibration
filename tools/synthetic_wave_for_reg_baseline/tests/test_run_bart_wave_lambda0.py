"""Focused tests for the unregularized BART Wave acceptance runner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_manifest import load_dataset_manifest  # noqa: E402
from run_bart_wave_lambda0 import (  # noqa: E402
    _build_parser,
    _completed_reconstruction_reusable,
    _resolve_run,
    build_ecalib_command,
    build_wave_command,
)


class EcalibCommandTests(unittest.TestCase):
    """Verify hard-crop calibration records both required BART outputs."""

    def test_builds_one_map_hard_crop_with_eigenvalues(self) -> None:
        command = build_ecalib_command(
            Path("/opt/bart"),
            Path("inputs/kspace_calib"),
            Path("output/coil_sens"),
            Path("output/eigenvalues"),
            crop=0.5,
        )
        self.assertEqual(
            command,
            [
                "/opt/bart",
                "ecalib",
                "-m",
                "1",
                "-c",
                "0.5",
                "inputs/kspace_calib",
                "output/coil_sens",
                "output/eigenvalues",
            ],
        )
        self.assertNotIn("-S", command)

    def test_requests_intensity_corrected_maps(self) -> None:
        command = build_ecalib_command(
            Path("bart"),
            Path("inputs/kspace_calib"),
            Path("output/coil_sens"),
            Path("output/eigenvalues"),
            crop=0.5,
            intensity_correction=True,
        )
        self.assertEqual(
            command[2:8],
            ["-m", "1", "-c", "0.5", "-I", "inputs/kspace_calib"],
        )

    def test_rejects_out_of_range_crop(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            build_ecalib_command(
                Path("bart"), Path("calib"), Path("maps"), Path("eigen"), crop=1.1
            )


class WaveCommandTests(unittest.TestCase):
    def test_gpu_is_mandatory(self) -> None:
        command = build_wave_command(
            Path("bart"),
            Path("maps"),
            Path("psf"),
            Path("kspace"),
            Path("image"),
            iterations=300,
            tolerance=1e-3,
        )
        self.assertEqual(command[:5], ["bart", "wave", "-g", "-i", "300"])


class ManifestResolutionTests(unittest.TestCase):
    def test_resolves_geometry_and_lambda_zero_settings_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            example_path = (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "incoming_r1_dataset.example.json"
            )
            payload = json.loads(example_path.read_text(encoding="utf-8"))
            payload["outputs"]["root"] = "outputs"
            payload["geometry"]["matrix"] = [256, 240, 192]
            payload["geometry"]["fov_mm"] = [256.0, 240.0, 192.0]
            manifest_path = root / "dataset.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = load_dataset_manifest(manifest_path)
            dataset.inspection_report.parent.mkdir(parents=True)
            dataset.inspection_report.write_text(
                json.dumps(
                    {
                        "dataset_manifest": dataset.provenance(),
                        "contract_checks": {"all_passed": True},
                        "twix": {"selected_measurement_index": 2},
                    }
                ),
                encoding="utf-8",
            )
            fake_bart = root / "bart"
            fake_bart.touch()
            args = _build_parser().parse_args(
                ["--dataset-manifest", str(manifest_path), "--resume"]
            )

            with patch("run_bart_wave_lambda0.shutil.which", return_value=str(fake_bart)):
                resolved = _resolve_run(args)

            self.assertEqual(resolved["matrix"], (256, 240, 192))
            self.assertEqual(resolved["virtual_coils"], 12)
            self.assertEqual(resolved["measurement_index"], 2)
            self.assertEqual(resolved["iterations"], 300)
            self.assertFalse(resolved["intensity_correction"])
            self.assertEqual(
                resolved["output_dir"],
                root / "outputs" / "reconstructions" / "synthetic_wave" / "lambda0",
            )

            pilot_dir = root / "outputs" / "reconstructions" / "crop-pilot"
            pilot_args = _build_parser().parse_args(
                [
                    "--dataset-manifest",
                    str(manifest_path),
                    "--ecalib-crop",
                    "0.6",
                    "--output-dir",
                    str(pilot_dir),
                    "--resume",
                ]
            )
            with patch("run_bart_wave_lambda0.shutil.which", return_value=str(fake_bart)):
                pilot = _resolve_run(pilot_args)
            self.assertEqual(pilot["crop"], 0.6)
            self.assertEqual(pilot["output_dir"], pilot_dir)
            self.assertEqual(pilot["manifest_overrides"]["ecalib_crop"], 0.6)

            outside_args = _build_parser().parse_args(
                [
                    "--dataset-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "outside"),
                ]
            )
            with (
                patch("run_bart_wave_lambda0.shutil.which", return_value=str(fake_bart)),
                self.assertRaisesRegex(ValueError, "below outputs.root"),
            ):
                _resolve_run(outside_args)

    def test_explicit_interface_allows_hash_validated_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bart = root / "bart"
            fake_bart.touch()
            args = _build_parser().parse_args(
                [
                    "--bart",
                    str(fake_bart),
                    "--bart-input-dir",
                    str(root / "inputs"),
                    "--calibration-base",
                    str(root / "calibration"),
                    "--output-dir",
                    str(root / "output"),
                    "--twix",
                    str(root / "measurement.dat"),
                    "--sequence",
                    str(root / "sequence.seq"),
                    "--resume",
                ]
            )

            resolved = _resolve_run(args)

            self.assertIsNone(resolved["dataset"])
            self.assertEqual(resolved["calibration_base"], root / "calibration")


class ResumeValidationTests(unittest.TestCase):
    def test_explicit_resume_rejects_manifest_bound_to_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "lambda0_complete_awaiting_visual_review",
                        "dataset_manifest": {"sha256": "dataset-hash"},
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(
                _completed_reconstruction_reusable(
                    manifest_path,
                    dataset_sha256=None,
                    bart_input_manifest_sha256="input-hash",
                    expected_config={},
                )
            )


if __name__ == "__main__":
    unittest.main()
