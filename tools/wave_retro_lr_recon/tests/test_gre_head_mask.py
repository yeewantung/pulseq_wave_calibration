"""Tests for review-driven GRE whole-head-mask parameter derivation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TOOL_ROOT / "scripts"
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.gre_head_mask import (  # noqa: E402
    derive_gre_head_mask_candidates,
)


class GreHeadMaskDerivationTests(unittest.TestCase):
    """Verify strict source validation and non-ranking candidate artifacts."""

    def test_sweep_writes_review_artifacts_without_selecting_default(self) -> None:
        """Create one valid candidate while leaving selection to visual review."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self._write_source(root)
            output = root / "mask_parameter_sweep"
            manifest = derive_gre_head_mask_candidates(
                source,
                output,
                relative_thresholds=(0.2,),
                core_relative_thresholds=(0.5,),
                maximum_growth_distances_mm=(0.0,),
                smoothing_mm=0.0,
                boundary_width_mm=1.0,
                closing_radius_mm=0.0,
            )
            self.assertEqual(manifest["candidate_count"], 1)
            self.assertEqual(manifest["ready_for_visual_review_count"], 1)
            self.assertEqual(manifest["selection"]["status"], "not_selected")
            self.assertFalse(manifest["selection"]["automatic_ranking"])
            self.assertFalse(manifest["selection"]["automatic_selection"])
            record = manifest["candidates"][0]
            self.assertTrue((output / record["mask"]["path"]).is_file())
            self.assertTrue((output / record["overlay"]["path"]).is_file())
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "REVIEW_INSTRUCTIONS.txt").is_file())

    def test_source_requires_corrected_gre_orientation(self) -> None:
        """Reject a GRE magnitude carrying the superseded flip provenance."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self._write_source(root, flips=(True, False, True))
            with self.assertRaisesRegex(ValueError, "orientation correction"):
                derive_gre_head_mask_candidates(
                    source,
                    root / "output",
                    relative_thresholds=(0.2,),
                    core_relative_thresholds=(0.5,),
                    maximum_growth_distances_mm=(0.0,),
                )

    def test_local_and_python_launchers_parse(self) -> None:
        """Keep both derivation launchers executable and self-documenting."""

        local = SCRIPTS / "derive_gre_head_mask.local.sh"
        python = SCRIPTS / "derive_gre_head_mask_parameters.py"
        subprocess.run(["bash", "-n", str(local)], check=True)
        result = subprocess.run(
            [sys.executable, str(python), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--relative-thresholds", result.stdout)
        self.assertIn("--maximum-growth-distances-mm", result.stdout)

    @staticmethod
    def _write_source(
        root: Path, *, flips: tuple[bool, bool, bool] = (False, True, False)
    ) -> Path:
        """Write a canonical synthetic GRE magnitude and matching sidecar.

        Args:
            root: Temporary test root.
            flips: Orientation provenance stored in the sidecar.

        Returns:
            Synthetic magnitude NIfTI path.
        """

        directory = root / "normal" / "nifti" / "selected_wavelet"
        directory.mkdir(parents=True)
        path = directory / "sub-test_echo-01_part-mag_BARTWaveGRE.nii.gz"
        grid = np.indices((20, 22, 18), dtype=np.float32)
        radius = (
            ((grid[0] - 9.5) / 7.0) ** 2
            + ((grid[1] - 10.5) / 8.0) ** 2
            + ((grid[2] - 8.5) / 6.0) ** 2
        )
        magnitude = (radius <= 1.0).astype(np.float32)
        nib.save(nib.Nifti1Image(magnitude, np.eye(4)), str(path))
        sidecar = path.with_name(path.name.removesuffix(".nii.gz") + ".json")
        sidecar.write_text(
            json.dumps(
                {
                    "ImagePart": "mag",
                    "EchoNumber": 1,
                    "CaseID": "native_r3x1",
                    "GRESelectedWaveletLambda": 0.015,
                    "OrientationPolicy": {"array_axis_flips": list(flips)},
                }
            ),
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
