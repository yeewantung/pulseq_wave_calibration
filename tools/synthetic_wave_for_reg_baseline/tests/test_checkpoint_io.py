"""Tests for shared safe NumPy checkpoint helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_io import (  # noqa: E402
    load_coil_basis,
    open_or_create_complex64_npy,
    validate_resume_pair,
    write_json_atomic,
)


class CheckpointIoTests(unittest.TestCase):
    def test_json_write_and_nested_basis_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = root / "progress.json"
            write_json_atomic(document, {"next_partition": 4})
            self.assertEqual(json.loads(document.read_text())["next_partition"], 4)

            basis_path = root / "basis.npz"
            basis = np.eye(4, dtype=np.complex64)
            np.savez(basis_path, basis=basis)
            np.testing.assert_array_equal(load_coil_basis(basis_path, 2), basis[:, :2])

    def test_resume_requires_compatible_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.npy"
            created = open_or_create_complex64_npy(path, (2, 3), resume=False)
            created[:] = 1
            created.flush()
            resumed = open_or_create_complex64_npy(path, (2, 3), resume=True)
            self.assertEqual(resumed.shape, (2, 3))
            with self.assertRaisesRegex(ValueError, "expected"):
                open_or_create_complex64_npy(path, (3, 2), resume=True)

    def test_resume_pair_rejects_one_missing_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.npy"
            progress = root / "progress.json"
            np.save(data, np.zeros(1, dtype=np.complex64))
            with self.assertRaisesRegex(ValueError, "both checkpoint files"):
                validate_resume_pair(data, progress, resume=True)


if __name__ == "__main__":
    unittest.main()
