"""Static interface tests for the readable MPRAGE sample workflows."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.gre import prepare_gre  # noqa: E402

SCRIPTS = TOOL_ROOT / "scripts"


class SampleCommandTests(unittest.TestCase):
    def test_normal_script_keeps_bart_commands_explicit(self) -> None:
        """Verify normal ecalib and both sampling-dependent Wave commands.

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
        self.assertEqual(sum(line.startswith("bart wave -g ") for line in commands), 2)
        self.assertIn('ECALIB_CROP="0.6"', source)
        self.assertIn('R3_LAMBDA="2.2e-2"', source)
        self.assertIn("bart wave -g -w -f -r 0 ", source)
        self.assertIn('ECALIB_RECORD="$BART_OUTPUT/ecalib_command.txt"', source)
        self.assertIn('> "$BART_OUTPUT/wave_command.txt"', source)

    def test_retro_script_has_one_ecalib_and_four_wave_commands(self) -> None:
        """Verify one ecalib and four explicit GPU Wave commands.

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
        self.assertEqual(sum(line.startswith("bart wave -g ") for line in commands), 4)
        self.assertEqual(sum(line.startswith("bart wave -g -w -f -r 0 ") for line in commands), 3)
        self.assertIn("bart wave -g -w -f -r 1.5e-2 ", source)
        self.assertNotIn("bart ecalib -g", source)
        self.assertEqual(source.count('bart_output/wave_command.txt"'), 4)

    def test_samples_parse_and_offer_dataset_independent_help(self) -> None:
        """Verify both Bash samples parse and expose path-agnostic help.

        Returns:
            None.
        """
        for name in ("sample_mprage_normal_recon.sh", "sample_mprage_retro_lr_recon.sh"):
            path = SCRIPTS / name
            subprocess.run(["bash", "-n", str(path)], check=True)
            completed = subprocess.run(
                ["bash", str(path), "--help"], check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("TWIX.dat OUTPUT_ROOT SEQUENCE.seq", completed.stdout)

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

    def test_gre_placeholder_cannot_launch_reconstruction(self) -> None:
        """Verify the deferred GRE adapter is explicitly non-runnable.

        Returns:
            None.
        """
        with self.assertRaisesRegex(NotImplementedError, "deferred"):
            prepare_gre()


if __name__ == "__main__":
    unittest.main()
