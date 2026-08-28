"""Tests for branching a target mask from accepted full-Wave encoding."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from bart_cfl import sha256_file, write_bart_header  # noqa: E402
from export_bart_wave_target_branch import run  # noqa: E402
from wave_synthesis import logical_bart_cfl_sha256  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bart(base: Path, array: np.ndarray) -> None:
    write_bart_header(base, array.shape)
    output = np.memmap(
        base.with_suffix(".cfl"),
        mode="w+",
        dtype=np.complex64,
        shape=array.shape,
        order="F",
    )
    output[...] = array
    output.flush()
    del output


class TargetBranchTests(unittest.TestCase):
    def test_exports_new_mask_and_links_validated_psf_and_acs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            _write_json(dataset_path, {"dataset_id": "test"})
            dataset = {
                "path": str(dataset_path),
                "sha256": sha256_file(dataset_path),
                "dataset_id": "test",
                "subject": "test",
            }

            full_wave = (
                np.arange(8 * 4 * 3 * 2, dtype=np.float32).reshape(8, 4, 3, 2)
                + 1j
            ).astype(np.complex64)
            full_wave_path = root / "full_wave.npy"
            np.save(full_wave_path, full_wave)
            psf_base = root / "psf"
            psf = np.ones((8, 4, 3, 1, 1), dtype=np.complex64)
            _write_bart(psf_base, psf)
            calibration_base = root / "calibration"
            calibration = np.zeros((4, 4, 3, 2), dtype=np.complex64)
            calibration[:, 1:3, :, :] = 1
            _write_bart(calibration_base, calibration)

            synthesis_path = root / "synthesis.json"
            synthesis = {
                "status": "awaiting_visual_review_before_mask_and_bart",
                "dataset_manifest": dataset,
                "full_wave_kspace": {
                    "path": str(full_wave_path),
                    "shape": list(full_wave.shape),
                    "dtype": "complex64",
                    "sampling_mask_applied": False,
                },
                "psf": {
                    "bart_base": str(psf_base),
                    "bart_shape": list(psf.shape),
                    "logical_sha256": logical_bart_cfl_sha256(psf_base, psf.shape),
                },
            }
            _write_json(synthesis_path, synthesis)
            synthesis_record = {
                "path": str(synthesis_path),
                "sha256": sha256_file(synthesis_path),
            }
            source_bart_path = root / "source_bart.json"
            _write_json(
                source_bart_path,
                {
                    "status": "calibration_kspace_ready_for_ecalib",
                    "dataset_manifest": dataset,
                    "source_synthesis_manifest": synthesis_record,
                    "kspace_calib": {
                        "bart_base": str(calibration_base),
                        "bart_shape": list(calibration.shape),
                        "bart_cfl_sha256": sha256_file(
                            calibration_base.with_suffix(".cfl")
                        ),
                        "pe1_lines_half_open": [1, 3],
                    },
                },
            )
            operator_path = root / "operator.json"
            _write_json(
                operator_path,
                {
                    "status": "passed",
                    "dataset_manifest": dataset,
                    "synthesis_manifest": synthesis_record,
                },
            )
            output_dir = root / "target"
            args = argparse.Namespace(
                source_synthesis_manifest=synthesis_path,
                source_bart_input_manifest=source_bart_path,
                operator_validation_manifest=operator_path,
                output_dir=output_dir,
                target_id="synthetic-wave-r2x1",
                pe1_acceleration=2,
                pe2_acceleration=1,
                pe1_residue=1,
                pe2_residue=0,
                acs_pe1_start=1,
                acs_pe1_stop_exclusive=3,
                pe2_chunk=2,
                confirm_full_wave_reviewed=True,
                check_only=False,
                resume=False,
            )

            manifest = run(args)

            self.assertEqual(manifest["status"], "calibration_kspace_ready_for_ecalib")
            self.assertEqual(manifest["sampling_mask"]["nominal_acceleration"], 2)
            self.assertTrue(Path(manifest["psf"]["cfl"]).is_symlink())
            self.assertTrue(Path(manifest["kspace_calib"]["cfl"]).is_symlink())
            self.assertTrue(
                manifest["masked_wave_kspace"][
                    "acquired_samples_equal_full_wave_bitwise"
                ]
            )
            args.resume = True
            self.assertEqual(run(args)["target_config"], manifest["target_config"])


if __name__ == "__main__":
    unittest.main()
