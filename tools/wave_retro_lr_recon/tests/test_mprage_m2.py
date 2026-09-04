"""Interface and conversion tests for the isolated MPRAGE m2 experiment."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from wave_retro_lr.bart_io import create_cfl

TOOL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TOOL_ROOT / "scripts"


def _load_converter() -> ModuleType:
    """Load the conversion entry point as an importable test module.

    Returns:
        Loaded ``convert_mprage_bart_to_nifti`` module.
    """
    path = SCRIPTS / "convert_mprage_bart_to_nifti.py"
    spec = importlib.util.spec_from_file_location("mprage_m2_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load converter module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERTER = _load_converter()


def _write_manifest(inputs: Path, norm: float = 2.0) -> None:
    """Write one minimal normal-input manifest used by converter tests.

    Args:
        inputs: Prepared normal BART-input directory.
        norm: Positive Wave k-space normalization to record.

    Returns:
        None. A JSON manifest is written below ``inputs``.
    """
    inputs.mkdir(parents=True, exist_ok=True)
    payload = {
        "geometry": {
            "logical_matrix_ro_lin_par": [4, 3, 2],
            "physical_fov_mm_xyz": [8.0, 6.0, 4.0],
        },
        "echoes": [{"wave_kspace_norm": norm}],
    }
    (inputs / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_cfl(path: Path, values: np.ndarray) -> None:
    """Create one BART CFL pair and fill it with test values.

    Args:
        path: Destination BART basename.
        values: Complex values and dimensions to store.

    Returns:
        None. The CFL memory map is flushed and closed.
    """
    target = create_cfl(path, values.shape)
    target[...] = values
    target.flush()
    del target


def _fake_native() -> tuple[SimpleNamespace, mock.Mock]:
    """Create a recording stand-in for the native MPRAGE NIfTI writer.

    Returns:
        Namespace exposing the expected helper API and its writer mock.
    """
    writer = mock.Mock()
    native = SimpleNamespace(
        _sanitize_filename_component=lambda value: value,
        save_mprage_output_to_nifti=writer,
    )
    return native, writer


class MprageM2InterfaceTests(unittest.TestCase):
    """Verify the normal-only m2 shell interface remains isolated and explicit."""

    def test_shell_option_and_commands_are_explicit(self) -> None:
        """Verify opt-in parsing, output isolation, and visible BART commands.

        Returns:
            None.
        """
        path = SCRIPTS / "sample_mprage_normal_recon.sh"
        source = path.read_text(encoding="utf-8")
        commands = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("bart ")
        ]
        self.assertIn('ECALIB_MAPS="1"', source)
        self.assertIn("--ecalib-maps) ECALIB_MAPS=", source)
        self.assertIn('normal/experimental_m2"', source)
        self.assertEqual(
            sum(line.startswith("bart ecalib -m 1 ") for line in commands), 1
        )
        self.assertEqual(
            sum(line.startswith("bart ecalib -m 2 ") for line in commands), 1
        )
        m2_command = next(
            line for line in commands if line.startswith("bart ecalib -m 2 ")
        )
        self.assertIn('"$BART_OUTPUT_ROOT/eigenvalue_maps"', m2_command)
        self.assertIn("--map-count 2", source)
        self.assertIn("--ecalib-record", source)

        subprocess.run(["bash", "-n", str(path)], check=True)
        completed = subprocess.run(
            ["bash", str(path), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--ecalib-maps 1|2", completed.stdout)

    def test_python_converter_does_not_launch_bart(self) -> None:
        """Verify the extended converter remains a conversion-only module.

        Returns:
            None.
        """
        source = (SCRIPTS / "convert_mprage_bart_to_nifti.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("subprocess", source)
        self.assertNotIn("Popen", source)


class MprageM2ConversionTests(unittest.TestCase):
    """Verify map dimensions, normalization, provenance, and orientation."""

    def test_map_loader_accepts_one_and_two_maps(self) -> None:
        """Verify BART map dimension 4 is retained instead of squeezed away.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = np.ones((4, 3, 2, 1, 1), dtype=np.complex64)
            two = np.empty((4, 3, 2, 1, 2), dtype=np.complex64)
            two[:, :, :, 0, 0] = 1.0 + 2.0j
            two[:, :, :, 0, 1] = 3.0 + 4.0j
            _write_cfl(root / "one", one)
            _write_cfl(root / "two", two)

            loaded_one = CONVERTER._load_bart_map_images(
                root / "one", (4, 3, 2), 1
            )
            loaded_two = CONVERTER._load_bart_map_images(
                root / "two", (4, 3, 2), 2
            )

            self.assertEqual(loaded_one.shape, (4, 3, 2, 1))
            self.assertEqual(loaded_two.shape, (4, 3, 2, 2))
            np.testing.assert_array_equal(loaded_two[:, :, :, 0], 1.0 + 2.0j)
            np.testing.assert_array_equal(loaded_two[:, :, :, 1], 3.0 + 4.0j)

    def test_map_loader_rejects_wrong_map_and_trailing_dimensions(self) -> None:
        """Verify accidental map-count and non-singleton trailing axes fail.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_cfl(
                root / "two", np.ones((4, 3, 2, 1, 2), dtype=np.complex64)
            )
            _write_cfl(
                root / "trailing",
                np.ones((4, 3, 2, 1, 2, 2), dtype=np.complex64),
            )

            with self.assertRaisesRegex(ValueError, "shape/finite check"):
                CONVERTER._load_bart_map_images(root / "two", (4, 3, 2), 1)
            with self.assertRaisesRegex(ValueError, "trailing singleton"):
                CONVERTER._load_bart_map_images(
                    root / "trailing", (4, 3, 2), 2
                )
            with self.assertRaisesRegex(ValueError, "must be 1 or 2"):
                CONVERTER._load_bart_map_images(root / "two", (4, 3, 2), 3)

    def test_default_m1_conversion_remains_one_complex_export(self) -> None:
        """Verify the default path keeps its suffix, phase, and orientation.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "normal" / "bart_inputs"
            image = root / "normal" / "bart_output" / "fista_r0" / "image_wave"
            _write_manifest(inputs)
            _write_cfl(image, np.ones((4, 3, 2, 1, 1), dtype=np.complex64))
            (image.parent / "wave_command.txt").write_text(
                "bart wave m1\n", encoding="utf-8"
            )
            (root / "normal" / "bart_output" / "ecalib_command.txt").write_text(
                "bart ecalib -m 1\n", encoding="utf-8"
            )
            native, writer = _fake_native()

            with mock.patch.object(
                CONVERTER, "load_wave_mprage_helpers", return_value=native
            ):
                CONVERTER.convert(
                    inputs,
                    image,
                    root / "source.dat",
                    root / "source.seq",
                    root / "normal" / "nifti" / "fista_r0",
                    "OriginalSuffix",
                )

            writer.assert_called_once()
            call = writer.call_args.kwargs
            np.testing.assert_array_equal(call["image"], 2.0 + 0.0j)
            self.assertEqual(call["suffix"], "OriginalSuffix")
            self.assertTrue(call["save_phase"])
            self.assertEqual(
                call["twix_array_axis_flips"],
                CONVERTER.MPRAGE_BART_ARRAY_AXIS_FLIPS,
            )
            self.assertNotIn("Experimental", call["metadata"])
            self.assertNotIn("BARTESPIRiTMapCount", call["metadata"])

    def test_m2_conversion_exports_components_and_display_rss(self) -> None:
        """Verify m2 normalization, per-map phase, RSS, and exact commands.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "normal" / "bart_inputs"
            experiment = root / "normal" / "experimental_m2"
            image = experiment / "fista_r0" / "image_wave"
            ecalib_record = experiment / "ecalib_command.txt"
            _write_manifest(inputs)
            values = np.empty((4, 3, 2, 1, 2), dtype=np.complex64)
            values[:, :, :, 0, 0] = 1.0 + 2.0j
            values[:, :, :, 0, 1] = 3.0 + 4.0j
            _write_cfl(image, values)
            (image.parent / "wave_command.txt").write_text(
                "bart wave m2\n", encoding="utf-8"
            )
            ecalib_record.parent.mkdir(parents=True, exist_ok=True)
            ecalib_record.write_text("bart ecalib -m 2\n", encoding="utf-8")
            native, writer = _fake_native()

            with mock.patch.object(
                CONVERTER, "load_wave_mprage_helpers", return_value=native
            ):
                CONVERTER.convert(
                    inputs,
                    image,
                    root / "source.dat",
                    root / "source.seq",
                    experiment / "nifti" / "fista_r0",
                    "ExperimentalM2",
                    map_count=2,
                    ecalib_record=ecalib_record,
                )

            self.assertEqual(writer.call_count, 3)
            first, second, display = [call.kwargs for call in writer.call_args_list]
            np.testing.assert_array_equal(first["image"], 2.0 + 4.0j)
            np.testing.assert_array_equal(second["image"], 6.0 + 8.0j)
            np.testing.assert_allclose(
                display["image"], np.sqrt(120.0), rtol=1e-6
            )
            self.assertEqual(first["suffix"], "ExperimentalM2Map01")
            self.assertEqual(second["suffix"], "ExperimentalM2Map02")
            self.assertEqual(display["suffix"], "ExperimentalM2MapsRSSDisplay")
            self.assertTrue(first["save_phase"])
            self.assertTrue(second["save_phase"])
            self.assertFalse(display["save_phase"])
            self.assertEqual(first["metadata"]["BARTESPIRiTMapCount"], 2)
            self.assertEqual(first["metadata"]["BARTESPIRiTMapComponent"], 1)
            self.assertEqual(second["metadata"]["BARTESPIRiTMapComponent"], 2)
            self.assertTrue(display["metadata"]["DisplayOnly"])
            self.assertFalse(display["metadata"]["CombinedPhaseAvailable"])
            self.assertEqual(
                first["metadata"]["BARTEcalibCommand"], "bart ecalib -m 2"
            )
            self.assertEqual(first["metadata"]["BARTWaveCommand"], "bart wave m2")
            for call in (first, second, display):
                self.assertEqual(
                    call["twix_array_axis_flips"],
                    CONVERTER.MPRAGE_BART_ARRAY_AXIS_FLIPS,
                )


if __name__ == "__main__":
    unittest.main()
