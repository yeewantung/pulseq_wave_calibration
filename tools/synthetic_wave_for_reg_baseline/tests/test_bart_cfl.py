"""Tests for product ACS geometry and BART CFL helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from export_bart_calibration_acs import validate_refscan_rectangle  # noqa: E402
from bart_cfl import open_bart_memmap, read_bart_shape, write_bart_header  # noqa: E402


class RefscanGeometryTests(unittest.TestCase):
    def test_complete_rectangle(self) -> None:
        lines = [line for partition in range(4) for line in (2, 3, 4)]
        partitions = [partition for partition in range(4) for _ in (2, 3, 4)]
        self.assertEqual(
            validate_refscan_rectangle(lines, partitions, npe1=8, npe2=4),
            ([2, 3, 4], [0, 1, 2, 3]),
        )

    def test_missing_coordinate_is_rejected(self) -> None:
        lines = [2, 3, 2]
        partitions = [0, 0, 1]
        with self.assertRaisesRegex(ValueError, "complete rectangle"):
            validate_refscan_rectangle(lines, partitions, npe1=8, npe2=2)


class BartCflTests(unittest.TestCase):
    def test_header_and_fortran_payload(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "array"
            shape = (4, 3, 2, 2)
            expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape, order="F")
            np.ravel(expected.astype(np.complex64), order="F").tofile(base.with_suffix(".cfl"))
            write_bart_header(base, shape)
            self.assertEqual(read_bart_shape(base), shape)
            np.testing.assert_array_equal(open_bart_memmap(base), expected)


if __name__ == "__main__":
    unittest.main()
