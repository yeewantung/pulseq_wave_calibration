"""Focused contracts for TWIX noise-scan prewhitening."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from scripts.prepare_gre_normal import _parser as gre_parser  # noqa: E402
from scripts.prepare_mprage_normal import _parser as mprage_parser  # noqa: E402
from wave_retro_lr.noise_prewhitening import (  # noqa: E402
    apply_noise_prewhitening_coillast,
    combined_prewhitening_and_compression_basis,
    load_recorded_whitening_matrix,
    persist_noise_prewhitening,
    prepare_noise_prewhitening,
    prewhitening_request_matches,
)


class _Stream:
    """Provide the small mapVBVD stream interface used by the tests."""

    def __init__(self, channels: int, acquisitions: int, values: np.ndarray | None = None):
        """Create a stream with constant channel metadata and optional data."""

        self.NAcq = acquisitions
        self.NCha = np.full(acquisitions, channels, dtype=np.float64)
        self.IsRawDataCorrect = np.zeros(acquisitions, dtype=bool)
        self._values = values

    def unsorted(self) -> np.ndarray:
        """Return synthetic [sample, coil, acquisition] noise data."""

        if self._values is None:
            raise RuntimeError("Synthetic stream has no payload.")
        return self._values


class NoisePrewhiteningTests(unittest.TestCase):
    """Validate default, fallback, numerical, and provenance behavior."""

    def test_cli_defaults_and_both_disable_spellings(self) -> None:
        """Enable whitening by default and accept both requested flag forms."""

        for parser in (mprage_parser(), gre_parser()):
            default = parser.parse_args(["input.dat", "output", "input.seq"])
            self.assertTrue(default.noise_prewhitening)
            for flag in ("-no-noise-prewhitening", "--no-noise-prewhitening"):
                parsed = parser.parse_args(
                    ["input.dat", "output", "input.seq", flag]
                )
                self.assertFalse(parsed.noise_prewhitening)

        standard_launchers = (
            "sample_mprage_normal_recon.sh",
            "sample_mprage_retro_lr_recon.sh",
            "sample_mprage_retro_r3x3_recon.sh",
            "sample_gre_normal_recon.sh",
            "sample_gre_retro_lr_recon.sh",
            "sample_gre_retro_r3x3_recon.sh",
        )
        for name in (
            *standard_launchers,
            "sample_mprage_rovir_recon.sh",
        ):
            source = (TOOL_ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("-no-noise-prewhitening", source, name)
            self.assertIn("--no-noise-prewhitening", source, name)
        for name in standard_launchers:
            source = (TOOL_ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("wavelet_prewhitening_unvalidated", source, name)
            self.assertIn(
                '.get("noise_prewhitening", {}).get("applied", False)',
                source,
                name,
            )

    def test_disabled_mode_does_not_open_twix(self) -> None:
        """Honor explicit opt-out without importing or reading mapVBVD."""

        with tempfile.TemporaryDirectory() as folder:
            twix = Path(folder) / "input.dat"
            twix.write_bytes(b"placeholder")
            with patch.dict(sys.modules, {"mapvbvd": None}):
                plan = prepare_noise_prewhitening(twix, 4, enabled=False)
        self.assertEqual(plan.record["status"], "disabled_by_user")
        self.assertIsNone(plan.matrix)

    def test_legacy_manifest_is_compatible_without_relabeling(self) -> None:
        """Reuse a pre-feature manifest in either automatic or forced-off mode."""

        legacy = {"format_version": 4}
        self.assertTrue(prewhitening_request_matches(legacy, enabled=True))
        self.assertTrue(prewhitening_request_matches(legacy, enabled=False))
        self.assertFalse(
            prewhitening_request_matches(
                {"noise_prewhitening": None}, enabled=True
            )
        )
        explicitly_disabled = {
            "noise_prewhitening": {"request": {"enabled": False}}
        }
        self.assertFalse(
            prewhitening_request_matches(explicitly_disabled, enabled=True)
        )
        self.assertTrue(
            prewhitening_request_matches(explicitly_disabled, enabled=False)
        )

    def test_absent_noise_scan_is_recorded_fallback(self) -> None:
        """Treat only an absent MDH noise stream as a safe fallback."""

        measurement = {"image": _Stream(4, 10), "refscan": _Stream(4, 3)}
        plan = self._prepare_with_measurements([{}, measurement], coils=4)
        self.assertEqual(plan.record["status"], "not_applied_no_noise_scan")
        self.assertTrue(plan.record["fallback"])
        self.assertIsNone(plan.matrix)

        streams = plan.record["scanner_side_whitening"][
            "mdh_raw_data_correction"
        ]["streams"]
        self.assertEqual(streams["image"]["acquisitions"], 10)
        self.assertEqual(streams["refscan"]["acquisitions"], 3)

    def test_supplied_measurements_avoid_duplicate_twix_parse(self) -> None:
        """Reuse geometry-inspection metadata without importing mapVBVD again."""

        measurement = {"image": _Stream(4, 10)}
        with tempfile.TemporaryDirectory() as folder:
            twix = Path(folder) / "input.dat"
            twix.write_bytes(b"placeholder")
            with patch.dict(sys.modules, {"mapvbvd": None}):
                plan = prepare_noise_prewhitening(
                    twix, 4, measurements=[{}, measurement]
                )
        self.assertEqual(plan.record["status"], "not_applied_no_noise_scan")
        self.assertEqual(
            plan.record["scanner_side_whitening"]["mdh_raw_data_correction"][
                "streams"
            ]["refscan"]["acquisitions"],
            0,
        )

    def test_correlated_noise_is_whitened_and_hash_bound(self) -> None:
        """Whiten correlated complex noise and persist exact matrix hashes."""

        rng = np.random.default_rng(12)
        independent = (
            rng.standard_normal((1600, 3))
            + 1j * rng.standard_normal((1600, 3))
        ).astype(np.complex64)
        mixing = np.asarray(
            [[1.0, 0.0, 0.0], [0.7, 0.8, 0.0], [0.2, 0.4, 0.7]],
            dtype=np.complex64,
        )
        rows = independent @ mixing.T
        values = np.transpose(rows.reshape(40, 40, 3), (0, 2, 1))
        measurement = {
            "image": _Stream(3, 12),
            "refscan": _Stream(3, 5),
            "noise": _Stream(3, 40, values),
        }
        plan = self._prepare_with_measurements([{}, measurement], coils=3)
        self.assertEqual(plan.record["status"], "applied")
        whitened = apply_noise_prewhitening_coillast(rows, plan.matrix)
        whitened -= whitened.mean(axis=0, keepdims=True)
        covariance = whitened.conj().T @ whitened / (whitened.shape[0] - 1)
        covariance /= np.real(np.diag(covariance)).mean()
        self.assertLess(float(np.max(np.abs(covariance - np.eye(3)))), 5.0e-5)

        with tempfile.TemporaryDirectory() as folder:
            record, files = persist_noise_prewhitening(folder, plan)
            manifest = {"noise_prewhitening": record}
            self.assertEqual(
                set(files),
                {"noise_covariance.npy", "noise_prewhitening_matrix.npy"},
            )
            loaded = load_recorded_whitening_matrix(folder, manifest)
            np.testing.assert_array_equal(loaded, plan.matrix)
            self.assertTrue(prewhitening_request_matches(manifest, enabled=True))
            self.assertFalse(prewhitening_request_matches(manifest, enabled=False))

    def test_noise_coil_mismatch_is_not_a_fallback(self) -> None:
        """Reject a populated noise scan whose physical channel count differs."""

        values = np.ones((16, 3, 4), dtype=np.complex64)
        measurement = {
            "image": _Stream(4, 12),
            "refscan": _Stream(4, 5),
            "noise": _Stream(3, 4, values),
        }
        with self.assertRaisesRegex(ValueError, "coil counts"):
            self._prepare_with_measurements([{}, measurement], coils=4)

    def test_combined_basis_matches_two_stage_application(self) -> None:
        """Apply whitening plus compression exactly via one combined basis."""

        rng = np.random.default_rng(4)
        data = (
            rng.standard_normal((5, 4)) + 1j * rng.standard_normal((5, 4))
        ).astype(np.complex64)
        whitening = np.diag(np.asarray([1.0, 0.8, 1.2, 0.7], dtype=np.complex64))
        basis = (
            rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
        ).astype(np.complex64)
        combined = combined_prewhitening_and_compression_basis(whitening, basis)
        np.testing.assert_allclose(
            data @ combined, (data @ whitening) @ basis, rtol=1.0e-6
        )

    def _prepare_with_measurements(
        self, measurements: list[object], *, coils: int
    ):
        """Run preparation against one synthetic mapVBVD measurement list."""

        with tempfile.TemporaryDirectory() as folder:
            twix = Path(folder) / "input.dat"
            twix.write_bytes(b"placeholder")
            module = SimpleNamespace(mapVBVD=lambda *_args, **_kwargs: measurements)
            with patch.dict(sys.modules, {"mapvbvd": module}):
                return prepare_noise_prewhitening(twix, coils)


if __name__ == "__main__":
    unittest.main()
