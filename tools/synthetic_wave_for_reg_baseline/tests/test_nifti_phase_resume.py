"""Tests for non-destructive phase-NIfTI resume installation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bart_cfl import sha256_file  # noqa: E402
from nifti_phase_resume import (  # noqa: E402
    install_phase_from_temporary_export,
    validate_phase_record,
)


class NiftiPhaseResumeTests(unittest.TestCase):
    def _save_part(self, root: Path, part: str, data: np.ndarray) -> tuple[Path, Path]:
        folder = root / "sub-example"
        folder.mkdir(parents=True, exist_ok=True)
        nifti = folder / f"sub-example_part-{part}_Example.nii.gz"
        sidecar = folder / f"sub-example_part-{part}_Example.json"
        nib.save(nib.Nifti1Image(data, np.eye(4)), nifti)
        sidecar.write_text(
            json.dumps(
                {
                    "Part": part,
                    "Units": "rad" if part == "phase" else "relative",
                }
            ),
            encoding="utf-8",
        )
        return nifti, sidecar

    def test_installs_phase_without_replacing_magnitude(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing_root = root / "accepted"
            generated_root = root / "temporary"
            magnitude = np.arange(8**3, dtype=np.float32).reshape((8, 8, 8))
            phase = np.linspace(-np.pi, np.pi, 8**3, dtype=np.float32).reshape(
                (8, 8, 8)
            )
            existing_magnitude, existing_sidecar = self._save_part(
                existing_root, "mag", magnitude
            )
            generated_magnitude, generated_sidecar = self._save_part(
                generated_root, "mag", magnitude
            )
            generated_phase, generated_phase_sidecar = self._save_part(
                generated_root, "phase", phase
            )
            magnitude_hash = sha256_file(existing_magnitude)
            existing_record = {
                "magnitude_nifti": str(existing_magnitude),
                "magnitude_nifti_sha256": magnitude_hash,
                "magnitude_sidecar": str(existing_sidecar),
                "magnitude_sidecar_sha256": sha256_file(existing_sidecar),
            }
            generated_record = {
                "magnitude_nifti": str(generated_magnitude),
                "magnitude_nifti_sha256": sha256_file(generated_magnitude),
                "magnitude_sidecar": str(generated_sidecar),
                "magnitude_sidecar_sha256": sha256_file(generated_sidecar),
                "phase_nifti": str(generated_phase),
                "phase_nifti_sha256": sha256_file(generated_phase),
                "phase_sidecar": str(generated_phase_sidecar),
                "phase_sidecar_sha256": sha256_file(generated_phase_sidecar),
            }
            updated = install_phase_from_temporary_export(
                existing_record=existing_record,
                generated_record=generated_record,
                existing_nifti_root=existing_root,
                generated_nifti_root=generated_root,
                expected_shape=(8, 8, 8),
            )
            self.assertTrue(validate_phase_record(updated, expected_shape=(8, 8, 8)))
            self.assertEqual(sha256_file(existing_magnitude), magnitude_hash)

    def test_missing_phase_is_not_reusable(self) -> None:
        self.assertFalse(validate_phase_record({}, expected_shape=(8, 8, 8)))

    def test_magnitude_mismatch_blocks_phase_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing_root = root / "accepted"
            generated_root = root / "temporary"
            existing_magnitude, existing_sidecar = self._save_part(
                existing_root, "mag", np.zeros((8, 8, 8), dtype=np.float32)
            )
            generated_magnitude, generated_sidecar = self._save_part(
                generated_root, "mag", np.ones((8, 8, 8), dtype=np.float32)
            )
            generated_phase, generated_phase_sidecar = self._save_part(
                generated_root, "phase", np.zeros((8, 8, 8), dtype=np.float32)
            )
            existing_record = {
                "magnitude_nifti": str(existing_magnitude),
                "magnitude_nifti_sha256": sha256_file(existing_magnitude),
                "magnitude_sidecar": str(existing_sidecar),
                "magnitude_sidecar_sha256": sha256_file(existing_sidecar),
            }
            generated_record = {
                "magnitude_nifti": str(generated_magnitude),
                "magnitude_nifti_sha256": sha256_file(generated_magnitude),
                "magnitude_sidecar": str(generated_sidecar),
                "magnitude_sidecar_sha256": sha256_file(generated_sidecar),
                "phase_nifti": str(generated_phase),
                "phase_nifti_sha256": sha256_file(generated_phase),
                "phase_sidecar": str(generated_phase_sidecar),
                "phase_sidecar_sha256": sha256_file(generated_phase_sidecar),
            }
            with self.assertRaisesRegex(ValueError, "magnitude samples differ"):
                install_phase_from_temporary_export(
                    existing_record=existing_record,
                    generated_record=generated_record,
                    existing_nifti_root=existing_root,
                    generated_nifti_root=generated_root,
                    expected_shape=(8, 8, 8),
                )
            self.assertFalse(
                (
                    existing_root
                    / "sub-example/sub-example_part-phase_Example.nii.gz"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
