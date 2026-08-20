"""Tests for the BART split-complex lambda-zero validation command."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_bart_split_complex import build_command  # noqa: E402


class SplitValidationCommandTests(unittest.TestCase):
    def test_gpu_command_uses_split_without_a_regularizer(self) -> None:
        command = build_command(
            Path("bart"),
            Path("maps"),
            Path("psf"),
            Path("kspace"),
            Path("output"),
            split_complex=True,
            block_size=8,
            iterations=300,
            tolerance=1e-3,
            max_eigenvalue=6.7e7,
            backend="gpu",
        )
        self.assertEqual(command[:4], ["bart", "wave", "-l", "-v"])
        self.assertIn("-g", command)
        self.assertIn("-l", command)
        self.assertNotIn("-w", command)


if __name__ == "__main__":
    unittest.main()
