"""Focused tests for no-wave SENSE mask, provenance, and orientation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_no_wave_sense import (  # noqa: E402
    EXPECTED_ACS_LINES,
    EXPECTED_IMAGE_LINES,
    build_pe1_mask,
)
from run_no_wave_sense import (  # noqa: E402
    _validate_provenance,
    build_ecalib_command,
    canonicalize_to_ras,
)


class SamplingMaskTests(unittest.TestCase):
    """Check the exact product image/refscan PE1 union."""

    def test_validated_union_has_101_lines(self) -> None:
        """The 85 R3 image lines and 24 ACS lines overlap at eight coordinates."""
        mask = build_pe1_mask(EXPECTED_IMAGE_LINES, EXPECTED_ACS_LINES)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(int(mask.sum()), 101)
        np.testing.assert_array_equal(
            np.flatnonzero(mask),
            sorted(set(EXPECTED_IMAGE_LINES) | set(EXPECTED_ACS_LINES)),
        )


class ProvenanceTests(unittest.TestCase):
    """Prevent map calibration and imaging data from using different bases."""

    def test_matching_measured_provenance(self) -> None:
        """Identical basis identity and columns are accepted."""
        common = {
            "coil_basis": "/tmp/basis.npz",
            "coil_basis_file_sha256": "abc",
            "basis_columns_half_open": [0, 12],
        }
        prepared = {**common, "contains_grappa_samples": False}
        calibration = {
            **common,
            "source": "measured no-wave product TWIX refscan ACS",
        }
        _validate_provenance(prepared, calibration)

    def test_mismatched_basis_is_rejected(self) -> None:
        """A separately estimated calibration basis must not silently pass."""
        prepared = {
            "coil_basis": "/tmp/a.npz",
            "coil_basis_file_sha256": "abc",
            "basis_columns_half_open": [0, 12],
            "contains_grappa_samples": False,
        }
        calibration = {
            "coil_basis": "/tmp/b.npz",
            "coil_basis_file_sha256": "def",
            "basis_columns_half_open": [0, 12],
            "source": "measured no-wave product TWIX refscan ACS",
        }
        with self.assertRaisesRegex(ValueError, "differ"):
            _validate_provenance(prepared, calibration)


class EcalibCommandTests(unittest.TestCase):
    """Keep crop, smooth-support, and diagnostic outputs explicit."""

    def test_soft_maps_and_eigenvalues_are_requested(self) -> None:
        """The review run must use one threshold and retain its eigenvalue map."""
        command = build_ecalib_command(
            Path("/opt/bart"),
            Path("/data/acs.cfl"),
            Path("/data/maps"),
            0.7,
            soft=True,
            eigenvalue_base=Path("/data/eigenvalues"),
        )
        self.assertEqual(
            command,
            [
                "/opt/bart",
                "ecalib",
                "-m",
                "1",
                "-c",
                "0.7",
                "-S",
                "/data/acs",
                "/data/maps",
                "/data/eigenvalues",
            ],
        )


class OrientationTests(unittest.TestCase):
    """Lock the content-derived SAR-to-RAS permutation used for this product scan."""

    def test_sar_native_data_becomes_ras_by_transpose_only(self) -> None:
        """Canonicalization must transpose axes 2,1,0 without array flips."""
        data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        affine = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        canonical, canonical_affine, transform = canonicalize_to_ras(data, affine)
        np.testing.assert_array_equal(canonical, np.transpose(data, (2, 1, 0)))
        self.assertEqual(nib.aff2axcodes(canonical_affine), ("R", "A", "S"))
        self.assertEqual(transform, [[2.0, 1.0], [1.0, 1.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
