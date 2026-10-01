"""Tests for the fixed-normalization intervention review figures."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr import intervention_qc  # noqa: E402


def write_magnitude(path: Path, restored: np.ndarray, scale: float, *, affine: np.ndarray | None = None, **normalization: object) -> Path:
    """Write a magnitude NIfTI and sidecar the way the converter exports them.

    Args:
        path: ``*_part-mag_*.nii.gz`` destination.
        restored: Magnitude on the shared BART scale.
        scale: Input percentile value; the stored image is ``restored / scale``.
        affine: Optional affine; the default is RAS-stored.
        normalization: Overrides of sidecar ``MagnitudeNormalization`` fields.

    Returns:
        Written NIfTI path.
    """
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image((restored / scale).astype(np.float32), np.eye(4) if affine is None else affine), str(path))
    record = {
        "Method": "positive-finite-percentile",
        "Percentile": 99.0,
        "InputPercentileValue": scale,
        "OutputPercentileValue": 1.0,
        "Clipped": False,
        **normalization,
    }
    path.with_name(path.name.removesuffix(".nii.gz") + ".json").write_text(
        json.dumps({"MagnitudeNormalization": record}), encoding="utf-8"
    )
    return path


def phantom(shape: tuple[int, int, int]) -> np.ndarray:
    """Return a smooth positive magnitude volume with an air band on top.

    Args:
        shape: Volume shape in RAS-stored order.

    Returns:
        Float32 magnitude.
    """
    r, a, s = np.meshgrid(*(np.linspace(-1.0, 1.0, size) for size in shape), indexing="ij")
    head = (r**2 + a**2 + s**2) < 0.8
    return (1e-8 * (0.02 + head * (1.0 + 0.3 * np.cos(4 * r) * np.cos(3 * a)))).astype(np.float32)


class SliceAndBoxTests(unittest.TestCase):
    """Pin the mechanical slice rule and box parsing."""

    def test_uniform_indices_for_the_reviewed_ranges(self) -> None:
        """Reproduce the reviewed slice table exactly.

        Returns:
            None.
        """
        expected = {
            (50, 170): [50, 80, 110, 140, 170],
            (64, 151): [65, 86, 107, 128, 149],
            (100, 192): [100, 123, 146, 169, 192],
            (64, 158): [65, 88, 111, 134, 157],
            (137, 221): [137, 158, 179, 200, 221],
            (0, 70): [1, 18, 35, 52, 69],
            (233, 255): [234, 239, 244, 249, 254],
        }
        for (low, high), indices in expected.items():
            with self.subTest(range=(low, high)):
                found = intervention_qc.uniform_indices(low, high, 5)
                self.assertEqual(found, indices)
                self.assertEqual(len(set(np.diff(found))), 1)
                self.assertTrue(low <= found[0] and found[-1] <= high)
        with self.assertRaises(ValueError):
            intervention_qc.uniform_indices(10, 12, 5)

    def test_boxes_parse_and_reject(self) -> None:
        """Parse inclusive RAS boxes and refuse malformed or outside ones.

        Returns:
            None.
        """
        self.assertEqual(intervention_qc.parse_box("50:170,64:151,100:192"), ((50, 170), (64, 151), (100, 192)))
        for text in ("50:170,64:151", "170:50,64:151,100:192", "a:b,64:151,100:192", "50:170,64:151,100:256"):
            with self.subTest(box=text), self.assertRaises(ValueError):
                intervention_qc.parse_box(text, (220, 256, 256))


class RestoredMagnitudeTests(unittest.TestCase):
    """Undo the export normalization and refuse anything ambiguous."""

    def test_restoration_and_refusals(self) -> None:
        """Restore the shared scale; refuse clipped, unknown, or non-RAS exports.

        Returns:
            None.
        """
        restored = phantom((12, 14, 16))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = write_magnitude(root / "a" / "sub_part-mag_A.nii.gz", restored, 5e-8)
            loaded, affine, record = intervention_qc.load_restored_magnitude(root / "a")
            np.testing.assert_allclose(loaded, restored, rtol=1e-6)
            self.assertEqual(record["restore_scale"], 5e-8)
            self.assertEqual(intervention_qc.find_magnitude(path), path.resolve())
            refused = {
                "clipped": {"Clipped": True},
                "other method": {"Method": "max"},
                "zero scale": {"InputPercentileValue": 0.0},
            }
            for name, override in refused.items():
                with self.subTest(case=name), self.assertRaises(ValueError):
                    intervention_qc.load_restored_magnitude(
                        write_magnitude(root / name / "x_part-mag_B.nii.gz", restored, 5e-8, **override)
                    )
            flipped = np.diag([-1.0, 1.0, 1.0, 1.0])
            with self.assertRaisesRegex(ValueError, "RAS"):
                intervention_qc.load_restored_magnitude(
                    write_magnitude(root / "las" / "x_part-mag_C.nii.gz", restored, 5e-8, affine=flipped)
                )
            path.with_name("sub_part-mag_A.json").unlink()
            with self.assertRaises(FileNotFoundError):
                intervention_qc.load_restored_magnitude(path)


class ReviewTests(unittest.TestCase):
    """Render one arm against the baseline with shared fixed windows."""

    def test_review_uses_one_baseline_constant_and_never_overwrites(self) -> None:
        """Derive C from the baseline only, record everything, refuse rewrites.

        Returns:
            None.
        """
        shape = (40, 48, 48)
        baseline = phantom(shape)
        central = ((8, 31), (10, 29), (12, 35))
        metal = ((10, 29), (25, 40), (0, 10))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = write_magnitude(root / "base" / "s_part-mag_Base.nii.gz", baseline, 4e-8)
            outputs = {}
            for name, candidate in (("scaled", 1.1 * baseline), ("noisy", baseline + 2e-10)):
                cand_path = write_magnitude(root / name / "s_part-mag_Cand.nii.gz", candidate, 7e-8)
                outputs[name] = intervention_qc.write_review(
                    base_path, cand_path, name, root / name / "qc", central=central, metal=metal, air_si_min=42, count=3
                )
            scaled = outputs["scaled"]
            box = baseline[8:32, 10:30, 12:36]
            # The stored float32 image is restored by one scale, so agreement is to float32 precision.
            np.testing.assert_allclose(scaled["normalization"]["constant"], np.percentile(box, 99.5), rtol=1e-6)
            self.assertEqual(scaled["normalization"], outputs["noisy"]["normalization"])
            self.assertAlmostEqual(scaled["statistics"]["central"]["median_ratio"], 1.1, places=5)
            self.assertEqual(scaled["slices"]["edges"]["a_p_strips"], {"anterior": (30, 47), "posterior": (0, 9)})
            self.assertEqual(scaled["slices"]["central"]["axial"], [12, 23, 34])
            self.assertEqual(len(scaled["figures"]), 15)
            for record in scaled["figures"].values():
                self.assertTrue(Path(record["path"]).is_file())
            self.assertTrue((root / "scaled" / "qc" / "qc_manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                intervention_qc.write_review(
                    base_path, root / "scaled", "scaled", root / "scaled" / "qc", central=central, metal=metal, air_si_min=42, count=3
                )
            shifted = np.eye(4)
            shifted[0, 3] = 1.0
            moved = write_magnitude(root / "moved" / "s_part-mag_M.nii.gz", baseline, 4e-8, affine=shifted)
            with self.assertRaisesRegex(ValueError, "affine"):
                intervention_qc.write_review(
                    base_path, moved, "moved", root / "moved" / "qc", central=central, metal=metal, air_si_min=42, count=3
                )
            self.assertFalse((root / "moved" / "qc").exists())

    def test_module_launches_no_process(self) -> None:
        """Keep the review code free of process launching.

        Returns:
            None.
        """
        for relative in intervention_qc.IMPLEMENTATION_FILES:
            source = (TOOL_ROOT / relative).read_text(encoding="utf-8")
            for forbidden in ("subprocess", "os.system", "Popen", "execv"):
                self.assertNotIn(forbidden, source, relative)


if __name__ == "__main__":
    unittest.main()
