"""Tests for the retrospective-acceleration BART input exporter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from export_bart_wave_inputs_retrospective import _completed_reusable  # noqa: E402
from wave_synthesis import logical_array_sha256, sha256_file  # noqa: E402


class RetrospectiveExportResumeTests(unittest.TestCase):
    def test_reuses_only_matching_intact_export(self) -> None:
        config = {
            "tag": "r3x2",
            "pe1_acceleration": 3,
            "pe2_acceleration": 2,
            "pe1_residue": 1,
            "pe2_residue": 0,
            "acs_pe1_start": 115,
            "acs_pe1_stop_exclusive": 139,
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            mask_path = root / "mask.npy"
            kspace_path = root / "wave_kspace.cfl"
            psf_path = root / "psf.cfl"
            mask = np.eye(4, dtype=bool)
            np.save(mask_path, mask)
            kspace_path.write_bytes(b"kspace")
            psf_path.write_bytes(b"psf")
            manifest = {
                "status": "retrospective_bart_inputs_ready",
                "config": config,
                "source_synthesis_dir": str(source),
                "sampling_mask": {
                    "path": str(mask_path),
                    "logical_sha256": logical_array_sha256(mask),
                },
                "masked_wave_kspace": {
                    "cfl": str(kspace_path),
                    "cfl_sha256": sha256_file(kspace_path),
                },
                "psf": {
                    "cfl": str(psf_path),
                    "cfl_sha256": sha256_file(psf_path),
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(_completed_reusable(manifest_path, config, source.resolve()))
            kspace_path.write_bytes(b"changed")
            self.assertFalse(_completed_reusable(manifest_path, config, source.resolve()))


if __name__ == "__main__":
    unittest.main()
