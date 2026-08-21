"""Tests for manifest-backed and matrix-derived 3D GRAPPA orchestration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_manifest import load_dataset_manifest  # noqa: E402
from reconstruct_no_wave_grappa_3d import (  # noqa: E402
    _open_or_create_memmap,
    reconstruct,
    resolve_grappa_run_inputs,
)


def _manifest_with_inspection(root: Path, acceleration: list[int]) -> Path:
    example = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "incoming_r1_dataset.example.json"
    )
    payload = json.loads(example.read_text(encoding="utf-8"))
    payload["inputs"]["twix"] = "inputs/scan.dat"
    payload["outputs"]["root"] = "outputs"
    payload["sampling"]["source_acceleration_pe1_pe2"] = acceleration
    payload["sampling"]["require_complete_source_grid"] = acceleration == [1, 1]
    manifest_path = root / "dataset.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_dataset_manifest(manifest_path)
    manifest.inspection_report.parent.mkdir(parents=True)
    manifest.inspection_report.write_text(
        json.dumps(
            {
                "dataset_manifest": {"sha256": manifest.sha256},
                "contract_checks": {"all_passed": True},
                "twix": {
                    "selected_measurement_sampling": {
                        "image_inferred_pe1_stride": acceleration[0],
                        "image_pe1_residues_for_inferred_stride": [1]
                        if acceleration[0] == 3
                        else [],
                        "refscan_covers_full_pe2": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _manifest_namespace(path: Path) -> Namespace:
    return Namespace(
        dataset_manifest=path,
        twix=None,
        coil_basis=None,
        output_prefix=None,
        ncc=None,
        regularization=None,
        pe2_kernel_size=None,
        matrix_rolinpar=None,
    )


class ManifestGrappaTests(unittest.TestCase):
    def test_r3_manifest_resolves_geometry_paths_and_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _manifest_with_inspection(root, [3, 1])

            inputs = resolve_grappa_run_inputs(_manifest_namespace(path))

            self.assertEqual(inputs.twix, root / "inputs" / "scan.dat")
            self.assertEqual(
                inputs.coil_basis,
                root / "outputs" / "calibration" / "coil_compression.npz",
            )
            self.assertEqual(
                inputs.output_prefix,
                root / "outputs" / "reconstructions" / "no_wave" / "source",
            )
            self.assertEqual(inputs.matrix_rolinpar, (256, 256, 256))
            self.assertEqual(inputs.pe2_kernel_size, 5)

    def test_r1_manifest_is_not_silently_sent_to_r3_grappa(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _manifest_with_inspection(Path(temporary), [1, 1])

            with self.assertRaisesRegex(ValueError, "separate direct source-reconstruction"):
                resolve_grappa_run_inputs(_manifest_namespace(path))


class MatrixDerivedReconstructionTests(unittest.TestCase):
    class FakeStream:
        def __init__(self, *, npe1: int, lines: list[int], skip_line: int = 0):
            self.sqzSize = (4, 2, npe1, 5)
            self.Lin = np.asarray(lines)
            self.skipLin = skip_line
            self.skipPar = 0

        def __getitem__(self, key):
            partition_slice = key[3]
            partitions = partition_slice.stop - partition_slice.start
            return np.zeros((4, 2, self.sqzSize[2], partitions), dtype=np.complex64)

    def test_reconstruction_allocation_and_loops_use_declared_matrix(self) -> None:
        image = self.FakeStream(npe1=3, lines=[0, 2])
        refscan = self.FakeStream(npe1=1, lines=[1], skip_line=1)
        basis = np.eye(2, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "reconstruction.npy"
            progress = root / "progress.json"

            with patch(
                "reconstruct_no_wave_grappa_3d.apply_grappa_3d_block",
                side_effect=lambda block, core, *_args, **_kwargs: block[:, :, core, :],
            ):
                report = reconstruct(
                    image,
                    refscan,
                    basis,
                    {},
                    output,
                    progress,
                    chunk_size=2,
                    pe2_kernel_size=3,
                    matrix_rolinpar=(4, 3, 5),
                    resume=False,
                )

            self.assertEqual(report["shape"], [4, 3, 5, 2])
            self.assertEqual(np.load(output, mmap_mode="r").shape, (4, 3, 5, 2))
            self.assertEqual(json.loads(progress.read_text())["next_partition"], 5)

    def test_nonresume_checkpoint_creation_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.npy"
            np.save(path, np.zeros((2, 2), dtype=np.complex64))

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                _open_or_create_memmap(path, (2, 2), resume=False)


if __name__ == "__main__":
    unittest.main()
