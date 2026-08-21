"""Focused tests for DICOM reconstruction-state classification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_r1_reference_comparison import classify_reconstruction  # noqa: E402


class DicomClassificationTests(unittest.TestCase):
    def test_unfiltered_sos_normalize_off(self) -> None:
        result = classify_reconstruction(
            "t1_mprage_sag_p2_RR_ND",
            ["ChannelMixing:ND=true_CMM=1_CDM=1", "CC:SoS"],
            contains_dis2d=False,
            contains_dis3d=False,
        )
        self.assertEqual(result["coil_combination"], "SoS")
        self.assertFalse(result["prescan_normalize"])
        self.assertEqual(result["distortion_correction"], "unfiltered_ND")

    def test_distortion_marker_overrides_nd_suffix(self) -> None:
        result = classify_reconstruction(
            "t1_mprage_sag_p2_RR_ND",
            ["CC:SoS", "NormalizeAlgo:PreScan"],
            contains_dis2d=True,
            contains_dis3d=True,
        )
        self.assertTrue(result["prescan_normalize"])
        self.assertEqual(result["distortion_correction"], "filtered")


if __name__ == "__main__":
    unittest.main()
