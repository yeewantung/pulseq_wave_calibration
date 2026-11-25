"""Tests for box/sphere/ellipsoid ROVir null-region geometry."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.rovir_regions import (  # noqa: E402
    ellipsoid_mask,
    null_region_union,
    parse_null_box,
    parse_null_ellipsoid,
    union_sha256,
)

SHAPE = (40, 30, 12)
VOXEL = (1.0, 1.0, 2.5)


class EllipsoidParsingTests(unittest.TestCase):
    """Validate the text specification of spheres and ellipsoids."""

    def test_sphere_shorthand_and_per_axis_radii(self) -> None:
        """Accept a sphere radius and three explicit per-axis radii.

        Returns:
            None.
        """
        sphere = parse_null_ellipsoid("ro=20,lin=-3.5,par=0,radius_mm=8", SHAPE, VOXEL)
        self.assertEqual(sphere, {"center_index": [20.0, -3.5, 0.0], "radii_mm": [8.0, 8.0, 8.0]})
        ellipsoid = parse_null_ellipsoid(
            " RO=20 , lin=5, par=6, ro_radius_mm=3, lin_radius_mm=4, par_radius_mm=5 ", SHAPE, VOXEL
        )
        self.assertEqual(ellipsoid["radii_mm"], [3.0, 4.0, 5.0])

    def test_invalid_specifications_are_rejected(self) -> None:
        """Reject missing, duplicated, mixed, unknown, or empty geometry.

        Returns:
            None.
        """
        bad = [
            "ro=20,lin=5,radius_mm=3",  # missing centre axis
            "ro=20,lin=5,par=6",  # missing radius
            "ro=20,lin=5,par=6,radius_mm=3,ro_radius_mm=3,lin_radius_mm=3,par_radius_mm=3",
            "ro=20,lin=5,par=6,radius_mm=-1",
            "ro=20,lin=5,par=6,radius_mm=nan",
            "ro=20,lin=5,par=6,radius_mm=3,tilt=4",
            "ro=20,ro=21,lin=5,par=6,radius_mm=3",
            "ro=20,lin=5,par=6,radius_mm=3,",
            "ro=500,lin=500,par=500,radius_mm=3",  # does not touch the grid
        ]
        for specification in bad:
            with self.subTest(specification=specification):
                with self.assertRaises(ValueError):
                    parse_null_ellipsoid(specification, SHAPE, VOXEL)


class EllipsoidGeometryTests(unittest.TestCase):
    """Check rasterization in physical millimetres on anisotropic grids."""

    def test_sphere_is_round_in_millimetres(self) -> None:
        """Extend 2.5x fewer voxels along the 2.5 mm axis than in-plane.

        Returns:
            None.
        """
        mask = ellipsoid_mask(SHAPE, VOXEL, (20, 15, 6), (5, 5, 5))
        self.assertTrue(mask[20, 15, 6])
        self.assertTrue(mask[25, 15, 6] and not mask[26, 15, 6])  # 5 mm in-plane
        self.assertTrue(mask[20, 10, 6] and not mask[20, 9, 6])
        self.assertTrue(mask[20, 15, 8] and not mask[20, 15, 9])  # 5 mm = 2 slices
        self.assertTrue(mask[20, 15, 4] and not mask[20, 15, 3])

    def test_outside_centre_clips_to_a_cap(self) -> None:
        """Keep only the in-grid cap of a sphere centred below the grid.

        Returns:
            None.
        """
        mask = ellipsoid_mask(SHAPE, VOXEL, (20, 15, -2), (10, 10, 10))
        # Slice z lies (z + 2) * 2.5 mm from the centre; z = 2 is 10 mm away.
        self.assertTrue(mask[:, :, 0].any() and mask[20, 15, 2])
        self.assertFalse(mask[:, :, 3].any())
        expected_radius_at_z0 = np.sqrt(10**2 - 5**2)
        row = np.flatnonzero(mask[:, 15, 0])
        self.assertAlmostEqual((row.max() - row.min()) / 2, np.floor(expected_radius_at_z0), delta=0.5)


class UnionTests(unittest.TestCase):
    """Validate unions, identity hashing, and box compatibility."""

    def test_union_is_order_independent_and_deduplicated(self) -> None:
        """Produce one identity regardless of argument order or duplicates.

        Returns:
            None.
        """
        box = "ro=0:5,lin=0:3,par=all"
        sphere = "ro=30,lin=20,par=5,radius_mm=4"
        first, record_a = null_region_union(SHAPE, VOXEL, boxes=[box, box], ellipsoids=[sphere])
        second, record_b = null_region_union(
            SHAPE, VOXEL, boxes=[box], ellipsoids=[{"center_index": [30, 20, 5], "radii_mm": [4, 4, 4]}, sphere]
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(record_a["region_id"], record_b["region_id"])
        self.assertEqual(record_a["canonical_boxes"], [parse_null_box(box, SHAPE)])
        self.assertEqual(len(record_b["canonical_ellipsoids"]), 1)
        self.assertEqual(
            record_a["union_voxel_count"],
            int(np.count_nonzero(first)),
        )

    def test_hash_matches_mprage_candidate_convention(self) -> None:
        """Hash shape (little-endian int64) then uint8 voxels, as for MPRAGE.

        Returns:
            None.
        """
        mask, record = null_region_union(SHAPE, VOXEL, boxes=["ro=1:2,lin=3:4,par=5:6"])
        hasher = hashlib.sha256()
        hasher.update(np.asarray(SHAPE, dtype="<i8").tobytes())
        hasher.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
        self.assertEqual(record["negative_union_sha256"], hasher.hexdigest())
        self.assertEqual(union_sha256(mask), hasher.hexdigest())
        self.assertEqual(record["mprage_equivalent_candidate_id"], f"negative_union_{hasher.hexdigest()[:16]}")

    def test_empty_or_full_union_is_rejected(self) -> None:
        """Require a nonempty null region and a nonempty complement.

        Returns:
            None.
        """
        with self.assertRaises(ValueError):
            null_region_union(SHAPE, VOXEL)
        with self.assertRaises(ValueError):
            null_region_union(SHAPE, VOXEL, boxes=["ro=all,lin=all,par=all"])


if __name__ == "__main__":
    unittest.main()
