"""Focused tests for optional, immutable MPRAGE N4 derivatives."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from scripts.apply_mprage_n4_bias_correction import _parser  # noqa: E402
from wave_retro_lr.n4_bias_correction import (  # noqa: E402
    N4Parameters,
    apply_mprage_n4_bias_correction,
    apply_mprage_shared_n4_bias_field,
)


class N4BiasCorrectionTests(unittest.TestCase):
    """Validate opt-in defaults, numerical binding, privacy, and reuse."""

    def test_cli_uses_reviewed_human_t1w_defaults(self) -> None:
        """Expose fixed official quick-start defaults without enabling a pipeline."""

        args = _parser().parse_args(["source", "output"])
        self.assertEqual(args.shrink_factor, 4)
        self.assertEqual(args.bspline_spacing_mm, 180.0)
        self.assertEqual(args.convergence_iterations, [50, 50, 50, 50])
        self.assertEqual(args.convergence_threshold, 0.0)

    def test_derivative_is_magnitude_only_hash_bound_and_resumable(self) -> None:
        """Keep originals unchanged and strictly reuse a complete N4 derivative."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self._write_source_pair(root / "nifti")
            destination = root / "n4_bias_corrected_nifti"
            before = self._tree_hashes(source)
            with (
                patch(
                    "wave_retro_lr.n4_bias_correction._tool_identity",
                    return_value={
                        "path": "N4BiasFieldCorrection-fixture",
                        "size_bytes": 7,
                        "sha256": "fixture-tool-sha256",
                        "version": "ANTs fixture",
                    },
                ),
                patch(
                    "wave_retro_lr.n4_bias_correction._execute_n4",
                    side_effect=self._fake_n4,
                ) as execute,
            ):
                manifest = apply_mprage_n4_bias_correction(source, destination)
                reused = apply_mprage_n4_bias_correction(source, destination)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(manifest, reused)
            self.assertEqual(before, self._tree_hashes(source))
            self.assertFalse(any(destination.rglob("*mask*")))
            self.assertTrue(manifest["validation"]["affine_matches_source"])
            self.assertFalse(manifest["request"]["policy"]["default_enabled"])
            self.assertFalse(manifest["request"]["policy"]["phase_modified"])
            self.assertFalse(
                manifest["request"]["parameters"]["rescale_intensities"]
            )
            mask_request = manifest["request"]["estimation_mask"]
            self.assertEqual(mask_request["contract_version"], 1)
            self.assertEqual(
                mask_request["parameters"],
                manifest["estimation_mask"]["parameters"],
            )
            command = manifest["command"]["arguments_with_path_placeholders"]
            self.assertEqual(command[command.index("-r") + 1], "0")
            output_names = {
                Path(record["path"]).name for record in manifest["outputs"]
            }
            self.assertTrue(any("N4BiasCorrected" in name for name in output_names))
            self.assertTrue(any("N4BiasField" in name for name in output_names))

            phase = next(source.rglob("*_part-phase_*.nii.gz"))
            recorded_phase = manifest["request"]["source"]["phase_files_unchanged"][0]
            self.assertEqual(recorded_phase["sha256"], self._sha256(phase))
            (destination / "metadata" / "n4_stdout.log").write_text(
                "changed\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, r"output (size|hash) changed"):
                apply_mprage_n4_bias_correction(source, destination)

    def test_source_change_rejects_existing_derivative(self) -> None:
        """Reject reuse after any canonical magnitude or phase source change."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self._write_source_pair(root / "nifti")
            destination = root / "n4"
            with (
                patch(
                    "wave_retro_lr.n4_bias_correction._tool_identity",
                    return_value={
                        "path": "N4BiasFieldCorrection-fixture",
                        "size_bytes": 7,
                        "sha256": "fixture-tool-sha256",
                        "version": "ANTs fixture",
                    },
                ),
                patch(
                    "wave_retro_lr.n4_bias_correction._execute_n4",
                    side_effect=self._fake_n4,
                ),
            ):
                apply_mprage_n4_bias_correction(source, destination)
            phase_sidecar = next(source.rglob("*_part-phase_*.json"))
            phase_sidecar.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source or parameter"):
                apply_mprage_n4_bias_correction(source, destination)

    def test_shared_normal_field_is_resampled_and_strictly_reused(self) -> None:
        """Apply one fitted normal field to retro geometry without another fit."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            normal = self._write_source_pair(root / "normal" / "nifti")
            reference = root / "normal" / "n4"
            retro = self._write_source_pair(root / "retro" / "nifti")
            retro_shape = (16, 32, 32)
            retro_affine = np.diag([2.0, 1.0, 1.0, 1.0])
            coordinates = np.indices(retro_shape, dtype=np.float32)
            retro_magnitude = 80.0 + coordinates[0]
            for nifti in retro.glob("*.nii.gz"):
                values = (
                    retro_magnitude
                    if "_part-mag_" in nifti.name
                    else np.zeros(retro_shape, dtype=np.float32)
                )
                nib.save(nib.Nifti1Image(values, retro_affine), str(nifti))
            with (
                patch(
                    "wave_retro_lr.n4_bias_correction._tool_identity",
                    return_value={
                        "path": "N4BiasFieldCorrection-fixture",
                        "size_bytes": 7,
                        "sha256": "fixture-tool-sha256",
                        "version": "ANTs fixture",
                    },
                ),
                patch(
                    "wave_retro_lr.n4_bias_correction._execute_n4",
                    side_effect=self._fake_n4,
                ) as execute,
            ):
                apply_mprage_n4_bias_correction(normal, reference)
                destination = root / "retro" / "n4"
                manifest = apply_mprage_shared_n4_bias_field(
                    retro, destination, reference
                )
                reused = apply_mprage_shared_n4_bias_field(
                    retro, destination, reference
                )
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(manifest, reused)
            self.assertEqual(manifest["mode"], "shared_field_application")
            self.assertTrue(manifest["validation"]["physical_space_resampling"])
            self.assertFalse(manifest["request"]["policy"]["ants_refit_per_target"])
            self.assertEqual(
                manifest["request"]["shared_bias_field"]["reference"]["manifest"][
                    "sha256"
                ],
                self._sha256(reference / "metadata" / "n4_manifest.json"),
            )
            corrected = next(
                destination.rglob("*_desc-N4BiasCorrected.nii.gz")
            )
            corrected_data = np.asarray(nib.load(str(corrected)).dataobj)
            self.assertEqual(corrected_data.shape, retro_shape)
            self.assertTrue(np.isfinite(corrected_data).all())

    def test_mprage_launchers_are_opt_in_and_gre_is_untouched(self) -> None:
        """Expose N4 only through MPRAGE launchers and keep GRE unchanged."""

        mprage = (
            "sample_mprage_normal_recon.sh",
            "sample_mprage_retro_lr_recon.sh",
            "sample_mprage_retro_r3x3_recon.sh",
            "sample_mprage_rovir_recon.sh",
            "sample_mprage_rovir_retro_recon.sh",
        )
        for name in mprage:
            source = (TOOL_ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("--n4-bias-correction", source, name)
            self.assertIn("apply_mprage_n4_bias_correction.py", source, name)
        for name in (
            "sample_gre_normal_recon.sh",
            "sample_gre_retro_lr_recon.sh",
            "sample_gre_retro_r3x3_recon.sh",
        ):
            source = (TOOL_ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertNotIn("--n4-bias-correction", source, name)

        rovir = (
            TOOL_ROOT / "scripts" / "sample_mprage_rovir_recon.sh"
        ).read_text(encoding="utf-8")
        self.assertLess(
            rovir.index('python "$WORKFLOW" finalize "$RECONSTRUCTION_ROOT"'),
            rovir.index('run_optional_n4 "$NIFTI_OUTPUT"'),
        )
        self.assertLess(
            rovir.index('python "$WORKFLOW" finalize-wavelet'),
            rovir.index('"$ROVIR_ROOT/n4_bias_corrected_nifti/optimal_wavelet"'),
        )
        rovir_retro = (
            TOOL_ROOT / "scripts" / "sample_mprage_rovir_retro_recon.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--reference-derivative", rovir_retro)
        self.assertIn(
            "normal/rovir/n4_bias_corrected_nifti/optimal_wavelet", rovir_retro
        )

    @staticmethod
    def _write_source_pair(directory: Path) -> Path:
        """Write one synthetic canonical-RAS magnitude/phase NIfTI pair."""

        directory.mkdir(parents=True)
        coordinates = np.indices((32, 32, 32), dtype=np.float32)
        radius = np.sqrt(
            sum((coordinates[axis] - 15.5) ** 2 for axis in range(3))
        )
        magnitude = np.zeros((32, 32, 32), dtype=np.float32)
        magnitude[radius < 11] = 100.0 + coordinates[0][radius < 11]
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        for part, data in (
            ("mag", magnitude),
            ("phase", np.zeros_like(magnitude)),
        ):
            nifti = directory / f"sub-fixture_part-{part}_wave.nii.gz"
            nib.save(nib.Nifti1Image(data, affine), str(nifti))
            nifti.with_name(nifti.name.removesuffix(".nii.gz") + ".json").write_text(
                json.dumps({"ImagePart": part}) + "\n", encoding="utf-8"
            )
        return directory

    @staticmethod
    def _fake_n4(command: list[str]) -> subprocess.CompletedProcess[str]:
        """Emulate ANTs output while preserving its multiplicative contract."""

        source = Path(command[command.index("-i") + 1])
        mask = Path(command[command.index("-x") + 1])
        output = command[command.index("-o") + 1]
        corrected_text, bias_text = output[1:-1].split(",", maxsplit=1)
        source_image = nib.load(str(source))
        values = np.asarray(source_image.dataobj, dtype=np.float32)
        selected = np.asarray(nib.load(str(mask)).dataobj, dtype=bool)
        bias = np.ones(values.shape, dtype=np.float32)
        bias[selected] = 2.0
        corrected = values.copy()
        corrected[selected] /= bias[selected]
        nib.save(
            nib.Nifti1Image(corrected, source_image.affine, source_image.header),
            corrected_text,
        )
        nib.save(
            nib.Nifti1Image(bias, source_image.affine, source_image.header),
            bias_text,
        )
        return subprocess.CompletedProcess(command, 0, stdout="fixture N4\n", stderr="")

    @classmethod
    def _tree_hashes(cls, directory: Path) -> dict[str, str]:
        """Hash every file below one test source tree."""

        return {
            str(path.relative_to(directory)): cls._sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        """Hash one test file."""

        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
