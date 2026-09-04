"""Static interface tests for the readable MPRAGE sample workflows."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.nifti_collection import HeadMaskParameters  # noqa: E402

SCRIPTS = TOOL_ROOT / "scripts"


class SampleCommandTests(unittest.TestCase):
    def test_normal_script_keeps_bart_commands_explicit(self) -> None:
        """Verify normal ecalib and explicit dual-branch Wave commands.

        Returns:
            None.
        """
        source = (SCRIPTS / "sample_mprage_normal_recon.sh").read_text(encoding="utf-8")
        commands = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("bart ")
        ]
        self.assertEqual(sum(line.startswith("bart ecalib -m 1 ") for line in commands), 1)
        self.assertEqual(sum(line.startswith("bart wave -g ") for line in commands), 3)
        self.assertEqual(sum(line.startswith("bart wave -w ") for line in commands), 3)
        self.assertIn('ECALIB_CROP="0.6"', source)
        self.assertIn('R3_LAMBDA="3.5e-2"', source)
        self.assertIn("USE_GPU=false", source)
        self.assertIn("-g) USE_GPU=true; shift ;;", source)
        self.assertIn('PSF_COEFFICIENT_PROCESSING="smooth"', source)
        self.assertIn("--psf-coefficient-processing sine-line", source)
        self.assertIn('--psf-fit-kx-min "$PSF_FIT_KX_MIN"', source)
        self.assertIn('--psf-fit-kx-max "$PSF_FIT_KX_MAX"', source)
        self.assertEqual(sum(line.startswith("bart wave -g -w -f -r 0 ") for line in commands), 2)
        self.assertEqual(sum(line.startswith("bart wave -w -f -r 0 ") for line in commands), 2)
        self.assertIn('bart wave -g -w -f -r "$R3_LAMBDA" ', source)
        self.assertIn('bart wave -w -f -r "$R3_LAMBDA" ', source)
        self.assertIn('ECALIB_RECORD="$BART_OUTPUT_ROOT/ecalib_command.txt"', source)
        self.assertIn('$BART_OUTPUT_ROOT/fista_r0/wave_command.txt"', source)
        self.assertIn('$BART_OUTPUT_ROOT/optimal_wavelet/wave_command.txt"', source)
        self.assertNotIn("build_mprage_nifti_collection.py", source)
        self.assertIn("PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png", source)
        self.assertIn("TROUBLESHOOTING.md", source)

    def test_retro_script_has_one_ecalib_and_eight_wave_commands(self) -> None:
        """Verify one ecalib and explicit CPU/GPU Wave branches per case.

        Returns:
            None.
        """
        source = (SCRIPTS / "sample_mprage_retro_lr_recon.sh").read_text(encoding="utf-8")
        commands = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("bart ")
        ]
        self.assertEqual(sum(line.startswith("bart ecalib -m 1 ") for line in commands), 1)
        self.assertEqual(sum(line.startswith("bart wave -g ") for line in commands), 8)
        self.assertEqual(sum(line.startswith("bart wave -w ") for line in commands), 8)
        self.assertIn("USE_GPU=false", source)
        self.assertIn("-g) USE_GPU=true; shift ;;", source)
        self.assertIn('PSF_COEFFICIENT_PROCESSING="smooth"', source)
        self.assertIn("--psf-coefficient-processing sine-line", source)
        self.assertIn('--psf-fit-kx-min "$PSF_FIT_KX_MIN"', source)
        self.assertIn('--psf-fit-kx-max "$PSF_FIT_KX_MAX"', source)
        self.assertEqual(sum(line.startswith("bart wave -g -w -f -r 0 ") for line in commands), 4)
        self.assertEqual(sum(line.startswith("bart wave -w -f -r 0 ") for line in commands), 4)
        self.assertEqual(sum("bart wave -g -w -f -r 3.5e-2 " in line for line in commands), 1)
        self.assertEqual(sum("bart wave -w -f -r 3.5e-2 " in line for line in commands), 1)
        self.assertEqual(sum("bart wave -g -w -f -r 2.5e-2 " in line for line in commands), 2)
        self.assertEqual(sum("bart wave -w -f -r 2.5e-2 " in line for line in commands), 2)
        self.assertEqual(sum("bart wave -g -w -f -r 2.2e-2 " in line for line in commands), 1)
        self.assertEqual(sum("bart wave -w -f -r 2.2e-2 " in line for line in commands), 1)
        self.assertNotIn("bart ecalib -g", source)
        self.assertEqual(source.count('wave_command.txt"'), 8)
        self.assertEqual(source.count("bart_output/fista_r0/image_wave"), 20)
        self.assertEqual(source.count("bart_output/optimal_wavelet/image_wave"), 20)
        self.assertNotIn("sample_mprage_normal_recon.sh", source)
        self.assertNotIn("build_mprage_nifti_collection.py", source)
        self.assertIn("PSF_COEFFICIENTS_VISUAL_ASSESSMENT.png", source)
        self.assertIn("TROUBLESHOOTING.md", source)

    def test_samples_parse_and_offer_dataset_independent_help(self) -> None:
        """Verify all Bash samples parse and expose path-agnostic help.

        Returns:
            None.
        """
        samples = {
            "sample_mprage_normal_recon.sh": "TWIX.dat OUTPUT_ROOT SEQUENCE.seq",
            "sample_mprage_retro_lr_recon.sh": "TWIX.dat OUTPUT_ROOT SEQUENCE.seq",
            "sample_mprage_nifti_collection.sh": "OUTPUT_ROOT",
        }
        for name, expected_help in samples.items():
            path = SCRIPTS / name
            subprocess.run(["bash", "-n", str(path)], check=True)
            completed = subprocess.run(
                ["bash", str(path), "--help"], check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(expected_help, completed.stdout)
            if name != "sample_mprage_nifti_collection.sh":
                self.assertIn("[-g]", completed.stdout)
                self.assertIn("--psf-coefficient-processing", completed.stdout)
                self.assertIn("--psf-fit-kx-min", completed.stdout)

        defaults = HeadMaskParameters()
        self.assertEqual(defaults.relative_threshold, 0.02)
        self.assertEqual(defaults.core_relative_threshold, 0.05)
        self.assertEqual(defaults.maximum_growth_distance_mm, 12.0)
        self.assertEqual(defaults.smoothing_mm, 1.0)
        self.assertEqual(defaults.opening_radius_mm, 0.0)
        self.assertEqual(defaults.closing_radius_mm, 1.5)
        self.assertEqual(defaults.dilation_radius_mm, 0.0)

    def test_new_python_workflow_does_not_launch_bart(self) -> None:
        """Verify measured-data Python modules never launch BART processes.

        Returns:
            None.
        """
        measured_sources = [
            TOOL_ROOT / "wave_retro_lr" / "mprage.py",
            SCRIPTS / "prepare_mprage_normal.py",
            SCRIPTS / "prepare_mprage_retro.py",
            SCRIPTS / "prepare_mprage_retro_maps.py",
            SCRIPTS / "convert_mprage_bart_to_nifti.py",
            SCRIPTS / "build_mprage_nifti_collection.py",
            TOOL_ROOT / "wave_retro_lr" / "nifti_collection.py",
        ]
        for path in measured_sources:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("subprocess", source, path.name)
            self.assertNotIn("Popen", source, path.name)

    def test_mprage_converter_keeps_dicom_validated_orientation(self) -> None:
        """Verify the exporter retains the DICOM-validated SI/LR correction.

        Returns:
            None.
        """
        source = (SCRIPTS / "convert_mprage_bart_to_nifti.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MPRAGE_BART_ARRAY_AXIS_FLIPS = (False, False, True)", source)
        self.assertIn(
            "twix_array_axis_flips=MPRAGE_BART_ARRAY_AXIS_FLIPS", source
        )

    def test_mprage_orientation_policy_is_not_replaced_by_gre(self) -> None:
        """Verify GRE integration leaves the validated MPRAGE flips unchanged.

        Returns:
            None.
        """

        source = (SCRIPTS / "convert_mprage_bart_to_nifti.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MPRAGE_BART_ARRAY_AXIS_FLIPS = (False, False, True)", source)


if __name__ == "__main__":
    unittest.main()
