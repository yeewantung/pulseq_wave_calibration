"""Tests for exact product-mask parsing and streamed BART export."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from sampling_mask import (  # noqa: E402
    product_mask_from_report,
    retrospective_cartesian_mask,
    write_masked_bart_kspace,
)


def _report() -> dict:
    return {
        "twix": {
            "selected_measurement_sampling": {
                "matrix_pe1": 6,
                "matrix_pe2": 4,
                "merged_patterns_by_pe2": [
                    {"partitions": [0, 2], "pe1_lines": [1, 4]},
                    {"partitions": [1, 3], "pe1_lines": [0, 1, 4]},
                ],
                "union_unique_coordinate_count": 10,
                "image_inferred_pe1_stride": 3,
                "image_pe1_residues_for_inferred_stride": [1],
                "refscan_unique_pe1_lines": [0, 1],
                "refscan_covers_full_pe2": True,
            }
        }
    }


class ProductMaskTests(unittest.TestCase):
    def test_mask_uses_reported_patterns(self) -> None:
        mask, info = product_mask_from_report(_report())
        expected = np.zeros((6, 4), dtype=bool)
        expected[np.ix_([1, 4], [0, 2])] = True
        expected[np.ix_([0, 1, 4], [1, 3])] = True
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(info["acquired_coordinate_count"], 10)

    def test_duplicate_partition_is_rejected(self) -> None:
        report = _report()
        report["twix"]["selected_measurement_sampling"]["merged_patterns_by_pe2"][1][
            "partitions"
        ] = [0, 1, 3]
        with self.assertRaisesRegex(ValueError, "more than one"):
            product_mask_from_report(report)

    def test_streamed_bart_export_preserves_and_zeros_samples(self) -> None:
        rng = np.random.default_rng(73)
        source = (
            rng.standard_normal((8, 6, 4, 3)) + 1j * rng.standard_normal((8, 6, 4, 3))
        ).astype(np.complex64)
        mask, _ = product_mask_from_report(_report())
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source_path = folder / "full.npy"
            np.save(source_path, np.asfortranarray(source))
            info = write_masked_bart_kspace(source_path, mask, folder / "wave_kspace")
            payload = np.memmap(
                folder / "wave_kspace.cfl",
                mode="r",
                dtype=np.complex64,
                shape=(8, 6, 4, 3, 1),
                order="F",
            )[..., 0]
            np.testing.assert_array_equal(payload[:, mask, :], source[:, mask, :])
            self.assertFalse(np.any(payload[:, ~mask, :]))
            self.assertEqual(info["shape"], [8, 6, 4, 3, 1])
            self.assertTrue(info["unacquired_samples_are_exact_zero"])
            self.assertEqual(
                info["full_chunked_readback_validation"]["acquired_mismatch_count"], 0
            )


class RetrospectiveMaskTests(unittest.TestCase):
    def test_r3x2_lattice_preserves_full_pe2_acs(self) -> None:
        mask, info = retrospective_cartesian_mask(
            (256, 256),
            accelerations=(3, 2),
            residues=(1, 0),
            fully_sampled_pe1_lines=np.arange(115, 139),
        )
        self.assertEqual(int(mask.sum()), 16000)
        self.assertEqual(info["nominal_acceleration"], 6)
        self.assertAlmostEqual(info["effective_acceleration_including_acs"], 4.096)
        self.assertTrue(np.all(mask[115:139, :]))
        outside_acs = np.r_[0:115, 139:256]
        expected = (
            outside_acs[:, None] % 3 == 1
        ) & (np.arange(256)[None, :] % 2 == 0)
        np.testing.assert_array_equal(mask[outside_acs, :], expected)
        self.assertEqual(info["image_coordinate_count"], 10880)
        self.assertEqual(info["acs_coordinate_count"], 6144)
        self.assertEqual(info["image_acs_overlap_coordinate_count"], 1024)

    def test_invalid_retrospective_residue_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "PE2 residue"):
            retrospective_cartesian_mask(
                (8, 8),
                accelerations=(3, 2),
                residues=(1, 2),
                fully_sampled_pe1_lines=[3, 4],
            )


if __name__ == "__main__":
    unittest.main()
