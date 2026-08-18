"""Unit tests for metadata-only Phase A inspection helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from phase_a_inspect import (  # noqa: E402
    _stream_summary,
    parse_dcmdump_records,
    summarize_sampling,
)


class SamplingSummaryTests(unittest.TestCase):
    def test_regular_r3_image_and_full_pe2_reference_strip(self) -> None:
        image_lines = [1, 4, 7, 1, 4, 7]
        image_partitions = [0, 0, 0, 1, 1, 1]
        ref_lines = [3, 4, 3, 4]
        ref_partitions = [0, 0, 1, 1]

        result = summarize_sampling(
            image_lines,
            image_partitions,
            ref_lines,
            ref_partitions,
            matrix_pe1=8,
            matrix_pe2=2,
        )

        self.assertEqual(result["image_inferred_pe1_stride"], 3)
        self.assertEqual(result["image_pe1_residues_for_inferred_stride"], [1])
        self.assertEqual(result["image_unique_coordinate_count"], 6)
        self.assertEqual(result["refscan_unique_coordinate_count"], 4)
        self.assertEqual(result["union_unique_coordinate_count"], 8)
        self.assertTrue(result["refscan_is_cartesian_rectangle"])
        self.assertTrue(result["refscan_covers_full_pe2"])
        self.assertEqual(result["out_of_range_coordinates"], [])
        self.assertEqual(result["image_patterns_by_pe2"][0]["partitions"], [0, 1])

    def test_duplicate_coordinates_are_reported(self) -> None:
        result = summarize_sampling(
            [1, 1],
            [0, 0],
            [],
            [],
            matrix_pe1=4,
            matrix_pe2=1,
        )
        self.assertEqual(result["image_duplicate_coordinate_count"], 1)


class DicomDumpParserTests(unittest.TestCase):
    def test_parses_bracketed_and_unbracketed_values(self) -> None:
        text = """# dcmdump (1/1): /data/Dicom_ima_00001.dcm
(0008,103e) LO [t1_mprage_sag_p2_ND] # 20, 1 SeriesDescription
(0020,000e) UI [1.2.3] # 6, 1 SeriesInstanceUID
(0020,0013) IS [1] # 2, 1 InstanceNumber
(0028,0010) US 256 # 2, 1 Rows
(0028,0101) US 12 # 2, 1 BitsStored
"""
        records = parse_dcmdump_records(text)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["filename"], "Dicom_ima_00001.dcm")
        self.assertEqual(records[0]["series_description"], "t1_mprage_sag_p2_ND")
        self.assertEqual(records[0]["instance_number"], 1)
        self.assertEqual(records[0]["rows"], 256)
        self.assertEqual(records[0]["bits_stored"], 12)


class StreamSummaryTests(unittest.TestCase):
    def test_reports_layout_without_reading_sample_payloads(self) -> None:
        class FakeStream:
            dataType = "image"
            NAcq = 2
            NCol = 512
            NCha = 64
            dataDims = ["Col", "Cha", "Lin", "Par"]
            dataSize = np.array([256, 64, 2, 1])
            sqzDims = ["Col", "Cha", "Lin"]
            sqzSize = np.array([256, 64, 2])
            Lin = np.array([1.0, 4.0])
            Par = np.array([0.0, 0.0])
            Sli = Ave = Phs = Eco = Rep = Set = Seg = Ida = Idb = Idc = Idd = Ide = np.zeros(2)
            centerLin = np.array([127.0, 127.0])
            centerPar = np.array([128.0, 128.0])
            IsReflected = np.array([False, True])

        result = _stream_summary(FakeStream())
        self.assertEqual(result["acquisition_count"], 2)
        self.assertEqual(result["readout_oversampling_factor"], 2.0)
        self.assertEqual(result["counters"]["Lin"]["unique_values"], [1, 4])
        self.assertEqual(result["reflected_acquisition_count"], 1)


if __name__ == "__main__":
    unittest.main()
