"""Focused tests for the unregularized BART Wave acceptance runner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from run_bart_wave_lambda0 import build_ecalib_command, build_wave_command  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
