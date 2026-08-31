"""Tests for canonical and whole-head-masked MPRAGE NIfTI collections."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import sha256_file  # noqa: E402
from wave_retro_lr.nifti_collection import (  # noqa: E402
    HeadMaskParameters,
    RETRO_CASES,
    build_mprage_nifti_collection,
)


class NiftiCollectionTests(unittest.TestCase):
    def test_complete_collection_preserves_originals_and_masks_every_grid(self) -> None:
        """Verify byte copies, mask provenance, and physical LR mask mapping.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reconstruction"
            source_paths = self._write_complete_source_tree(root)
            source_hashes = {path: sha256_file(path) for path in source_paths}

            mask_parameters = HeadMaskParameters(
                relative_threshold=0.02,
                core_relative_threshold=0.05,
                maximum_growth_distance_mm=4.0,
                opening_radius_mm=0.0,
            )
            manifest = build_mprage_nifti_collection(
                root, require_retro=True, parameters=mask_parameters
            )

            collection = root / "nifti_collection"
            self.assertEqual(manifest["builder"], "wave_retro_lr.nifti_collection")
            self.assertEqual(len(manifest["cases"]), 5)
            self.assertFalse(manifest["head_mask"]["bet_used"])
            self.assertTrue(
                manifest["scientific_scope"]["original_niftis_copied_byte_for_byte"]
            )
            for source, digest in source_hashes.items():
                self.assertEqual(sha256_file(source), digest)

            normal_magnitude = next(
                path for path in source_paths if "normal/nifti" in str(path) and "part-mag" in path.name
            )
            copied_normal = collection / "original_nifti" / "normal" / normal_magnitude.name
            self.assertEqual(sha256_file(copied_normal), sha256_file(normal_magnitude))
            masked_normal = collection / "head_masked_nifti" / "normal" / normal_magnitude.name
            masked_data = np.asanyarray(nib.load(str(masked_normal)).dataobj)
            self.assertGreater(float(masked_data[16, 16, 16]), 0)
            self.assertEqual(float(masked_data[2, 2, 2]), 0)
            self.assertEqual(float(masked_data[31, 16, 16]), 0)

            for case in RETRO_CASES:
                masked_files = sorted(
                    (collection / "head_masked_nifti" / "retro" / case).glob("*.nii.gz")
                )
                self.assertEqual(len(masked_files), 2)
                for path in masked_files:
                    data = np.asanyarray(nib.load(str(path)).dataobj)
                    self.assertGreater(np.count_nonzero(data), 0)
                    sidecar = path.with_name(path.name[: -len(".nii.gz")] + ".json")
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    self.assertTrue(metadata["WholeHeadMaskApplied"])
                    self.assertFalse(metadata["WholeHeadMaskBETUsed"])

            # A second build safely replaces only the utility-owned collection.
            refreshed = build_mprage_nifti_collection(
                root, require_retro=True, parameters=mask_parameters
            )
            self.assertEqual(len(refreshed["cases"]), 5)

            protected = collection / "original_nifti" / "normal" / normal_magnitude.name
            protected.write_bytes(b"user-modified")
            with self.assertRaisesRegex(FileExistsError, "changed since its manifest"):
                build_mprage_nifti_collection(
                    root, require_retro=True, parameters=mask_parameters
                )
            self.assertEqual(protected.read_bytes(), b"user-modified")

    def test_normal_only_collection_and_unowned_output_protection(self) -> None:
        """Verify partial normal use and refusal to replace an unowned directory.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reconstruction"
            self._write_case(root / "normal" / "nifti", (32, 32, 32), (1.0, 1.0, 1.0))
            manifest = build_mprage_nifti_collection(root)
            self.assertEqual([record["case"] for record in manifest["cases"]], ["normal"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reconstruction"
            self._write_case(root / "normal" / "nifti", (32, 32, 32), (1.0, 1.0, 1.0))
            collection = root / "nifti_collection"
            collection.mkdir()
            (collection / "user_file.txt").write_text("keep\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not tool-owned"):
                build_mprage_nifti_collection(root)
            self.assertEqual((collection / "user_file.txt").read_text(encoding="utf-8"), "keep\n")

    def _write_complete_source_tree(self, root: Path) -> list[Path]:
        """Create representative normal and four-grid canonical source pairs.

        Args:
            root: Temporary reconstruction output root.

        Returns:
            All source NIfTI and JSON paths created for mutation checks.
        """
        paths = self._write_case(
            root / "normal" / "nifti", (32, 32, 32), (1.0, 1.0, 1.0)
        )
        geometries = {
            "native_r3x2": ((32, 32, 32), (1.0, 1.0, 1.0)),
            "lr_x_1p5mm_r3x2": ((22, 32, 32), (1.5, 1.0, 1.0)),
            "lr_y_1p5mm_r3x2": ((32, 22, 32), (1.0, 1.5, 1.0)),
            "lr_xy_1p25mm_r3x2": ((26, 26, 32), (1.25, 1.25, 1.0)),
        }
        for case, (shape, spacing) in geometries.items():
            paths.extend(self._write_case(root / "retro" / case / "nifti", shape, spacing))
        return paths

    def _write_case(
        self,
        directory: Path,
        shape: tuple[int, int, int],
        spacing: tuple[float, float, float],
    ) -> list[Path]:
        """Write one canonical magnitude/phase pair with a distant noise island.

        Args:
            directory: Canonical source NIfTI directory to create.
            shape: Target XYZ matrix.
            spacing: Positive XYZ voxel sizes in millimetres.

        Returns:
            Magnitude, magnitude JSON, phase, and phase JSON paths.
        """
        directory.mkdir(parents=True, exist_ok=True)
        affine = np.eye(4, dtype=np.float64)
        affine[0, 0], affine[1, 1], affine[2, 2] = spacing
        affine[:3, 3] = [-(size - 1) * step / 2 for size, step in zip(shape, spacing)]
        coordinates = np.meshgrid(
            *[
                (np.arange(size) - (size - 1) / 2) * step
                for size, step in zip(shape, spacing)
            ],
            indexing="ij",
        )
        head = (
            (coordinates[0] / 9.0) ** 2
            + (coordinates[1] / 10.0) ** 2
            + (coordinates[2] / 11.0) ** 2
        ) <= 1.0
        magnitude = np.full(shape, 0.001, dtype=np.float32)
        magnitude[head] = 1.0
        magnitude[(1, 1, 1)] = 0.2
        # This dim, thick appendage is connected to the head and therefore
        # survives largest-component filtering and modest binary opening.
        if shape == (32, 32, 32):
            magnitude[23:32, 14:19, 14:19] = 0.03
        phase = np.zeros(shape, dtype=np.float32)
        phase[head] = 0.4
        phase[(1, 1, 1)] = 1.2

        created: list[Path] = []
        for part, data in (("mag", magnitude), ("phase", phase)):
            nifti = directory / "sub-test" / f"sub-test_part-{part}_BARTWaveMPRAGE.nii.gz"
            nifti.parent.mkdir(exist_ok=True)
            nib.save(nib.Nifti1Image(data, affine), str(nifti))
            sidecar = nifti.with_name(nifti.name[: -len(".nii.gz")] + ".json")
            sidecar.write_text(json.dumps({"Part": part}) + "\n", encoding="utf-8")
            created.extend((nifti, sidecar))
        return created


if __name__ == "__main__":
    unittest.main()
