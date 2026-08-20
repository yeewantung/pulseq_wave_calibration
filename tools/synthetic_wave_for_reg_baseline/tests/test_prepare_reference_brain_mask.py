"""Focused tests for the fixed FSL BET mask wrapper."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_reference_brain_mask import build_bet_command  # noqa: E402


class BetCommandTests(unittest.TestCase):
    def test_command_freezes_mask_and_vertical_gradient_options(self) -> None:
        self.assertEqual(
            build_bet_command(Path("bet"), Path("reference.nii.gz"), Path("brain"), 0.25),
            ["bet", "reference.nii.gz", "brain", "-m", "-f", "0.25", "-g", "0"],
        )

    def test_rejects_invalid_fractional_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly between"):
            build_bet_command(Path("bet"), Path("reference.nii.gz"), Path("brain"), 1.0)

    def test_robust_center_is_explicit(self) -> None:
        self.assertEqual(
            build_bet_command(
                Path("bet"), Path("reference.nii.gz"), Path("brain"), 0.55, True
            ),
            ["bet", "reference.nii.gz", "brain", "-R", "-m", "-f", "0.55", "-g", "0"],
        )


if __name__ == "__main__":
    unittest.main()
