"""Tests for canonical ROVir consumption by MPRAGE retro cases."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr.bart_io import cfl_record, create_cfl, sha256_file  # noqa: E402
from wave_retro_lr.core import ResolvedCase  # noqa: E402
from wave_retro_lr.rovir_retro import (  # noqa: E402
    ROVIR_RETRO_CASES,
    prepare_mprage_rovir_retro,
)
from wave_retro_lr.sampling import SamplingPattern  # noqa: E402


class RovirRetroTests(unittest.TestCase):
    """Exercise five-case preparation and immutable canonical reuse."""

    def test_prepares_all_cases_and_rejects_changed_transform(self) -> None:
        """Bind every sibling branch to one normal contract and transform.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reconstruction"
            transform = self._write_fixture(root)
            manifests = prepare_mprage_rovir_retro(root)
            self.assertEqual([entry["case_directory"] for entry in manifests], list(ROVIR_RETRO_CASES))
            for entry in manifests:
                case = entry["case_directory"]
                self.assertEqual(entry["coil_processing"]["candidate_id"], "reviewed_union")
                self.assertTrue(entry["fista_lambda_zero_required"])
                self.assertFalse(entry["selected_regularization"]["optimized_for_rovir"])
                self.assertTrue(
                    (root / "retro" / case / "rovir" / "bart_inputs" / "wave_kspace.cfl").is_file()
                )
            reused = prepare_mprage_rovir_retro(root)
            self.assertEqual(len(reused), 5)

            changed = create_cfl(transform, (1, 1, 1, 2, 2))
            changed[...] = 0
            changed.flush()
            del changed
            with self.assertRaisesRegex(ValueError, "transform differs"):
                prepare_mprage_rovir_retro(root)

    def _write_fixture(self, root: Path) -> Path:
        """Write a compact normal ROVir contract and five standard cases.

        Args:
            root: Temporary reconstruction root.

        Returns:
            Full ROVir transform basename.
        """
        twix = root / "source.dat"
        sequence = root / "source.seq"
        root.mkdir(parents=True)
        twix.write_bytes(b"twix-fixture")
        sequence.write_bytes(b"sequence-fixture")
        source = {
            "twix": self._source_record(twix),
            "sequence": self._source_record(sequence),
        }
        sampling = SamplingPattern(
            name="R1",
            acceleration_lin_par=(1, 1),
            lin_residue=None,
            matrix_lin_par=(8, 8),
            acquired_lin=tuple(range(8)),
            acquired_par=tuple(range(8)),
            measurement_index=0,
            skip_lin_par=(0, 0),
        )
        normal_manifest = root / "normal" / "bart_inputs" / "manifest.json"
        self._write_json(normal_manifest, {"source": source, "sampling": sampling.to_json()})

        rovir_inputs = root / "normal" / "rovir" / "bart_inputs"
        wave = self._write_cfl(rovir_inputs / "wave_kspace", (8, 8, 8, 2, 1), seed=1)
        psf = self._write_cfl(rovir_inputs / "psf", (8, 8, 8, 1, 1), seed=2)
        calibration = self._write_cfl(
            rovir_inputs / "kspace_calib", (4, 8, 8, 2), seed=5
        )
        basis = rovir_inputs / "rovir_projection_basis.npy"
        np.save(basis, np.eye(2, dtype=np.complex64), allow_pickle=False)
        physical_base = self._write_cfl(
            root
            / "normal"
            / "rovir"
            / "feasibility"
            / "inputs"
            / "physical_calibration"
            / "physical_set4_kspace",
            (4, 8, 8, 2),
            seed=6,
        )
        physical_manifest = (
            root / "normal" / "rovir" / "feasibility" / "manifests" / "physical_calibration.json"
        )
        self._write_json(physical_manifest, {"status": "fixture"})
        rovir_inputs_manifest = rovir_inputs / "manifest.json"
        self._write_json(
            rovir_inputs_manifest,
            {
                "source": source,
                "artifacts": {
                    "wave_kspace": cfl_record(wave),
                    "kspace_calib": cfl_record(calibration),
                },
                "psf_calibration": {"copied": cfl_record(psf)},
                "physical_calibration": {
                    "manifest_path": str(physical_manifest),
                    "manifest_sha256": sha256_file(physical_manifest),
                    "cfl": cfl_record(physical_base),
                },
                "rovir": {
                    "projection_basis_file": basis.name,
                    "projection_basis_sha256": sha256_file(basis),
                    "physical_coils": 2,
                    "virtual_coils": 2,
                },
            },
        )
        rovir_output = root / "normal" / "rovir" / "bart_output"
        csm = self._write_cfl(rovir_output / "coil_sens", (4, 8, 8, 2, 1), seed=3)
        ecalib = rovir_output / "ecalib_command.txt"
        ecalib.write_text("bart ecalib -m 1 -c 0.1 input output\n", encoding="utf-8")
        feasibility = root / "normal" / "rovir" / "feasibility"
        transform = self._write_cfl(
            feasibility / "transforms" / "rovir_full" / "transform",
            (1, 1, 1, 2, 2),
            seed=4,
        )
        approved = feasibility / "masks" / "approved" / "manifest.json"
        self._write_json(approved, {"candidate_id": "reviewed_union"})

        case_shapes = {
            "native_r3x2": (8, 8),
            "lr_x_1p5mm_r3x2": (8, 4),
            "lr_y_1p5mm_r3x2": (4, 8),
            "lr_xy_1p25mm_r3x2": (4, 4),
            "native_r3x3": (8, 8),
        }
        for index, (case_name, (lin, par)) in enumerate(case_shapes.items(), start=10):
            standard = root / "retro" / case_name / "bart_inputs"
            self._write_cfl(standard / "psf", (8, lin, par, 1, 1), seed=index)
            case = ResolvedCase(
                requested_resolution_mm_xyz=(8 / par, 8 / lin, 1.0),
                achieved_resolution_mm_xyz=(8 / par, 8 / lin, 1.0),
                source_logical_matrix_ro_lin_par=(4, 8, 8),
                target_logical_matrix_ro_lin_par=(4, lin, par),
                target_physical_matrix_xyz=(par, lin, 4),
                crop_bounds_lin=((8 - lin) // 2, (8 - lin) // 2 + lin),
                crop_bounds_par=((8 - par) // 2, (8 - par) // 2 + par),
                acceleration_ry_rz=((3, 3) if case_name == "native_r3x3" else (3, 2)),
                case_name=case_name,
                label=case_name,
            )
            manifest = {"case": case.to_json()}
            if case_name == "native_r3x3":
                mask = np.zeros((8, 8), dtype=bool)
                mask[1::3, 1::3] = True
                np.save(standard / "sampling_mask.npy", mask)
                manifest["selected_regularization"] = {"method": "wavelet", "lambda": 0.045}
            self._write_json(standard / "manifest.json", manifest)

        contract = {
            "status": "mprage_normal_rovir_complete",
            "retro_consumable": True,
            "source": source,
            "normal_source_manifest": {
                "path": str(normal_manifest),
                "sha256": sha256_file(normal_manifest),
            },
            "coil_processing": {
                "candidate_id": "reviewed_union",
                "virtual_coils": 2,
                "transform": cfl_record(transform),
                "approved_mask_manifest": self._file_record(approved),
            },
            "ecalib": {
                "coil_sens": cfl_record(csm),
                "command_record": self._file_record(ecalib),
            },
            "artifacts": {
                "prepared_inputs_manifest": self._file_record(rovir_inputs_manifest)
            },
        }
        self._write_json(root / "normal" / "rovir" / "manifest.json", contract)
        return transform

    @staticmethod
    def _write_cfl(base: Path, shape: tuple[int, ...], *, seed: int) -> Path:
        """Write one finite nonzero complex BART fixture.

        Args:
            base: Destination basename.
            shape: BART array shape.
            seed: Deterministic random seed.

        Returns:
            Input basename.
        """
        base.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        output = create_cfl(base, shape)
        output[...] = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        output.flush()
        del output
        return base

    @staticmethod
    def _source_record(path: Path) -> dict[str, object]:
        """Return the source identity format used by the canonical contract.

        Args:
            path: Existing source file.

        Returns:
            Path, stat, and SHA-256 record.
        """
        stat = path.stat()
        return {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }

    @staticmethod
    def _file_record(path: Path) -> dict[str, object]:
        """Return an immutable ordinary-file record.

        Args:
            path: Existing file.

        Returns:
            Path, size, and SHA-256 record.
        """
        return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        """Write one JSON fixture after creating parents.

        Args:
            path: Destination path.
            payload: JSON-native object.

        Side Effects:
            Writes the fixture file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
