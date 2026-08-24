"""Tests for no-Wave R3x1 PICS sweep configuration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_no_wave_r3x1_pics_sweep import (  # noqa: E402
    FULLY_SAMPLED_PE1_LINES,
    build_pics_command,
    build_r3x1_pe1_mask,
)


class NoWaveR3x1SweepTests(unittest.TestCase):
    def test_mask_is_r3x1_union_full_acs(self) -> None:
        mask = build_r3x1_pe1_mask()
        expected = np.zeros(256, dtype=bool)
        expected[1::3] = True
        expected[list(FULLY_SAMPLED_PE1_LINES)] = True
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(int(mask.sum()), 101)

    def test_commands_are_gpu_and_scaling_explicit(self) -> None:
        common = {
            "bart": Path("/bart"),
            "kspace": Path("/kspace"),
            "maps": Path("/maps"),
            "output": Path("/output"),
            "iterations": 100,
        }
        cg = build_pics_command(**common, regularizer="cg_sense")
        wavelet = build_pics_command(
            **common, regularizer="wavelet", lambda_value=1.5e-2
        )
        for command in (cg, wavelet):
            self.assertIn("-g", command)
            self.assertIn("-S", command)
            self.assertIn("-e", command)
        self.assertNotIn("--fista", cg)
        self.assertIn("--fista", wavelet)
        self.assertIn("-l1", wavelet)
        self.assertIn("-n", wavelet)


if __name__ == "__main__":
    unittest.main()
