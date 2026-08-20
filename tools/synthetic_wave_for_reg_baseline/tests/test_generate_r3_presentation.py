"""Tests for deterministic R3 presentation-case selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from generate_r3_presentation import select_representatives  # noqa: E402


class SelectionTests(unittest.TestCase):
    def test_selects_lowest_composite_rank_within_each_regularizer(self) -> None:
        records = [
            {"case": "lambda0", "kind": "lambda0"},
            {"case": "worse-wavelet", "kind": "wavelet", "composite_mean_rank": 2.0},
            {"case": "best-wavelet", "kind": "wavelet", "composite_mean_rank": 1.0},
            {"case": "best-llr", "kind": "llr", "composite_mean_rank": 1.5},
        ]
        selected = select_representatives(records)
        self.assertEqual(selected["wavelet"]["case"], "best-wavelet")
        self.assertEqual(selected["llr"]["case"], "best-llr")


if __name__ == "__main__":
    unittest.main()
