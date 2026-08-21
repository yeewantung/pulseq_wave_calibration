"""Focused tests for the fixed FSL BET mask wrapper."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_reference_brain_mask import build_bet_command, expand_mask  # noqa: E402


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


class MaskExpansionTests(unittest.TestCase):
    def test_one_voxel_dilation_is_face_connected(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True

        expanded = expand_mask(mask, 1)

        self.assertEqual(int(expanded.sum()), 7)
        self.assertTrue(expanded[1, 2, 2])
        self.assertFalse(expanded[1, 1, 2])

    def test_zero_dilation_returns_independent_boolean_mask(self) -> None:
        mask = np.ones((2, 2, 2), dtype=np.uint8)
        expanded = expand_mask(mask, 0)
        self.assertEqual(expanded.dtype, np.bool_)
        self.assertFalse(np.shares_memory(mask, expanded))

    def test_rejects_negative_dilation(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            expand_mask(np.ones((2, 2, 2), dtype=bool), -1)


if __name__ == "__main__":
    unittest.main()
