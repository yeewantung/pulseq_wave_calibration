"""Tests for exact product-mask parsing and streamed BART export."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from sampling_mask import product_mask_from_report, write_masked_bart_kspace  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
