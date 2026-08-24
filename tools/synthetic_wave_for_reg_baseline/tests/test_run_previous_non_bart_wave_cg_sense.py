"""Shape-view tests for the previous non-BART Wave adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_previous_non_bart_wave_cg_sense import (  # noqa: E402
    _coil_first_map_view,
    _coil_first_wave_view,
    _psf_view,
)


class PreviousNonBartAdapterTests(unittest.TestCase):
    def test_bart_views_reject_incorrect_shapes(self) -> None:
        wrong = np.zeros((2, 3, 4), dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, "Wave k-space"):
            _coil_first_wave_view(wrong)
        with self.assertRaisesRegex(ValueError, "PSF"):
            _psf_view(wrong)
        with self.assertRaisesRegex(ValueError, "sensitivity-map"):
            _coil_first_map_view(wrong)


if __name__ == "__main__":
    unittest.main()
