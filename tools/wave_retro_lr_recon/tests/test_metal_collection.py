"""Metal preference, atomic synchronization, reference survival and provenance tests."""

from __future__ import annotations

from contextlib import contextmanager
import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from test_metal_pipeline import arguments, pipeline_fixture, fake_cpu, fake_convert
import test_standard_redesign as standard_tests
from wave_retro_lr.bart_io import sha256_file
from wave_retro_lr.metal_pipeline import run_pipeline
from wave_retro_lr.metal_workflow import read_json, write_json, file_record
from wave_retro_lr.nifti_collection import build_mprage_nifti_collection
from wave_retro_lr.reconstruction_state import (
    _request_contract, _cfl_records, _nifti_records, record_completed_reconstruction,
)


def collection_fixture(root: Path) -> tuple[Path, Path]:
    """Create fully manifested ordinary and metal-compatible baseline inputs in a temporary tree."""
    config = pipeline_fixture(root)
    source = read_json(config)
    manifest_path = Path(source["source_manifest"])
    old = read_json(manifest_path)
    request = old["request"]
    reference = Path(source["reference_nifti"])
    phase = reference.with_name(reference.name.replace("part-mag", "part-phase"))
    image = nib.load(reference)
    nib.save(nib.Nifti1Image(np.zeros(image.shape, np.float32), image.affine), phase)
    write_json(phase.with_name(phase.name.replace(".nii.gz", ".json")), {"Part": "phase"})
    inputs = root/"vcc2/normal/bart_inputs"
    (inputs/"sampling_class.txt").write_text("R3x1\n")
    contract = _request_contract(
        prepared_manifest=inputs/"manifest.json", normal_manifest=inputs/"manifest.json",
        profile="wavelet-only", case="normal", branch=manifest_path.parent.name,
        method="wavelet", regularization=.03, maps=Path(request["inputs"]["maps"]["base"]),
        psfs=[inputs/"psf"], kspaces=[inputs/"wave_kspace"],
        images=[Path(old["outputs"][0]["base"])],
        command_records=[Path(request["command_records"][0])],
        expected_commands=request["expected_commands"], nifti_directory=reference.parent)
    payload = {"format_version": 1, "status": old["status"], "request": contract,
               "outputs": _cfl_records([Path(old["outputs"][0]["base"])], require_finite=True),
               "nifti_outputs": _nifti_records(reference.parent)}
    manifest_path.write_text(json.dumps(payload))
    return config, reference


@contextmanager
def mocked_science():
    """Keep all end-to-end orchestration tests synthetic and prevent external scientific jobs."""
    with patch("wave_retro_lr.metal_runner.memory_guard"), \
         patch("wave_retro_lr.metal_pipeline.memory_guard"), \
         patch("wave_retro_lr.metal_pipeline.validate_invocation", return_value={}), \
         patch("wave_retro_lr.mprage.prepare_retro_mprage"), \
         patch("wave_retro_lr.mprage.prepare_retro_mprage_r3x3"), \
         patch("wave_retro_lr.metal_pipeline.run_cpu_reconstruction", side_effect=fake_cpu), \
         patch("scripts.convert_mprage_bart_to_nifti.convert", side_effect=fake_convert):
        yield


class MetalCollectionTests(unittest.TestCase):
    """Exercise actual collection replacement with tiny hash-bound metal fixtures."""

    def test_same_names_paired_override_new_retro_and_other_vcc_retained(self):
        """Prefer matched metal magnitude/phase, add metal-only retro and leave other VCCs intact."""
        with tempfile.TemporaryDirectory() as temporary, mocked_science():
            root = Path(temporary)
            _, reference = collection_fixture(root)
            helper = standard_tests.StandardPcaStateAndCollectionTests()
            other = helper._write_variant_branch(root, 16, case="normal", branch="fista_r0")
            record_completed_reconstruction(other.pop("run_manifest"), **other)
            before = build_mprage_nifti_collection(root)
            original = next(c for c in before["cases"] if c["collection_id"] == "vcc2")
            standard_names = {r["original_nifti"] for r in original["files"]}
            source_hash = sha256_file(reference)
            destination = run_pipeline(arguments(root))
            run_pipeline(arguments(root, "retro"))
            result = build_mprage_nifti_collection(root, require_retro=True)
            normal = next(c for c in result["cases"] if c["collection_id"] == "vcc2" and c["case"] == "normal")
            self.assertEqual({r["original_nifti"] for r in normal["files"]}, standard_names)
            for record in normal["files"]:
                self.assertEqual(record["collection_source_selection"], "metal_csm")
                self.assertEqual(sha256_file(root/"nifti_collection/vcc2"/record["original_nifti"]), record["collected_source_nifti_sha256"])
                self.assertIn("vcc2_metal_csm/", record["collected_source_nifti"])
                self.assertIsNotNone(record["metal_csm_provenance"]["replaced_standard_branch"])
            self.assertEqual(result["storage_policy"]["metal_csm_selection"]["selected_image_count"], 8)
            self.assertEqual(len([c for c in result["cases"] if c["collection_id"] == "vcc2"]), 4)
            self.assertTrue(all(f["collection_source_selection"] == "canonical" for c in result["cases"]
                                if c["collection_id"] == "vcc16" for f in c["files"]))
            self.assertFalse((root/"nifti_collection/vcc2_metal_csm").exists())
            self.assertEqual(sha256_file(reference), source_hash)
            build_mprage_nifti_collection(root, require_retro=True)

    def test_incomplete_fallback_corruption_atomicity_and_no_downgrade(self):
        """Skip unfinished derivatives, reject corrupt completed exports, and preserve previous collections."""
        with tempfile.TemporaryDirectory() as temporary, mocked_science():
            root = Path(temporary)
            collection_fixture(root)
            destination = run_pipeline(arguments(root, "normal", "--metal-prepare-only"))
            ordinary = build_mprage_nifti_collection(root)
            self.assertEqual(ordinary["storage_policy"]["metal_csm_selection"]["selected_image_count"], 0)
            run_pipeline(arguments(root))
            build_mprage_nifti_collection(root)
            previous = sha256_file(root/"nifti_collection/manifest.json")
            export = destination/"normal/bart_output/wavelet_metal_unvalidated/metadata/export.json"
            payload = export.read_text()
            export.write_text('{"status":"complete"}')
            with self.assertRaises((ValueError, KeyError)):
                build_mprage_nifti_collection(root)
            self.assertEqual(sha256_file(root/"nifti_collection/manifest.json"), previous)
            export.write_text(payload)
            export.rename(export.with_suffix(".incomplete"))
            with self.assertRaisesRegex(FileNotFoundError, "Previously collected metal"):
                build_mprage_nifti_collection(root)
            self.assertEqual(sha256_file(root/"nifti_collection/manifest.json"), previous)

    def test_reference_survives_collection_replacement_and_old_n4_is_not_used(self):
        """Cache a collection-derived baseline before replacement and keep old N4 out of metal selection."""
        with tempfile.TemporaryDirectory() as temporary, mocked_science():
            root = Path(temporary)
            config, reference = collection_fixture(root)
            first = build_mprage_nifti_collection(root)
            record = next(f for f in first["cases"][0]["files"] if "part-mag" in f["original_nifti"])
            collected = root/"nifti_collection/vcc2"/record["original_nifti"]
            original_hash = sha256_file(collected)
            payload = read_json(config)
            payload["reference_nifti"] = str(collected)
            config.write_text(json.dumps(payload))
            destination = run_pipeline(arguments(root))
            # An incomplete ordinary N4 tree must not be used for a metal image.
            (root/"vcc2/normal/n4_bias_corrected_nifti/wavelet_selected").mkdir(parents=True)
            result = build_mprage_nifti_collection(root)
            self.assertNotEqual(sha256_file(collected), original_hash)
            self.assertEqual(sha256_file(destination/"source_reference"/reference.name), original_hash)
            self.assertEqual(result["storage_policy"]["magnitude_selection"]["n4_bias_corrected_magnitude_count"], 0)
            run_pipeline(arguments(root))
            build_mprage_nifti_collection(root)

    def test_wrong_count_rejected(self):
        """Reject mismatched VCC directory identity without replacing the previous collection."""
        with tempfile.TemporaryDirectory() as temporary, mocked_science():
            root = Path(temporary)
            collection_fixture(root)
            destination = run_pipeline(arguments(root))
            build_mprage_nifti_collection(root)
            previous = sha256_file(root/"nifti_collection/manifest.json")
            destination.rename(root/"vcc3_metal_csm")
            with self.assertRaisesRegex(ValueError, "count or reference"):
                build_mprage_nifti_collection(root)
            self.assertEqual(sha256_file(root/"nifti_collection/manifest.json"), previous)

    def test_hash_and_geometry_rejection_preserve_previous_collection(self):
        """Reject modified completed payloads even if an export record is updated to a wrong grid."""
        with tempfile.TemporaryDirectory() as temporary, mocked_science():
            root = Path(temporary)
            collection_fixture(root)
            destination = run_pipeline(arguments(root))
            build_mprage_nifti_collection(root)
            previous = sha256_file(root/"nifti_collection/manifest.json")
            export = destination/"normal/bart_output/wavelet_metal_unvalidated/metadata/export.json"
            payload = read_json(export)
            image = next(Path(r["path"]) for r in payload["outputs"] if "part-mag" in r["path"] and r["path"].endswith(".nii.gz"))
            old = nib.load(image)
            nib.save(nib.Nifti1Image(np.ones((3, 4, 5), np.float32), old.affine), image)
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                build_mprage_nifti_collection(root)
            payload["outputs"] = [file_record(image) if r["path"] == str(image) else r for r in payload["outputs"]]
            export.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "geometry/finite-value"):
                build_mprage_nifti_collection(root)
            self.assertEqual(sha256_file(root/"nifti_collection/manifest.json"), previous)

    def test_rovir_isolation_and_ambiguous_wavelet_targets(self):
        """Never overlay ROVir; refuse an ambiguous choice among multiple same-lambda standard branches."""
        from wave_retro_lr.metal_collection import merge_metal
        from wave_retro_lr.nifti_collection import _discover_collection_variants

        with tempfile.TemporaryDirectory() as temporary, mocked_science():
            root = Path(temporary)
            collection_fixture(root)
            ordinary = _discover_collection_variants(root)
            rovir = {"collection_id": "rovir", "kind": "canonical_rovir", "sources": {"independent": {}},
                     "source_contract": {"must_remain": True}}
            variants = copy.deepcopy(ordinary)+[rovir]
            expected_rovir = copy.deepcopy(rovir)
            run_pipeline(arguments(root))
            merge_metal(root, variants)
            self.assertEqual(rovir, expected_rovir)
            ambiguous = copy.deepcopy(ordinary)
            variant = ambiguous[0]
            branch = next(iter(variant["sources"]))
            variant["sources"]["wavelet_candidate"] = copy.deepcopy(variant["sources"][branch])
            record = variant["run_manifests"][f"{branch}:normal"]
            variant["run_manifests"]["wavelet_candidate:normal"] = copy.deepcopy(record)
            # Both standard candidates match the selected metal lambda.
            record["lambda"] = .035
            variant["run_manifests"]["wavelet_candidate:normal"]["lambda"] = .035
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                merge_metal(root, ambiguous)

    def test_pruned_normal_niftis_keep_previous_collection_names(self):
        """Keep historical names and grouping when only complete metal exports remain available."""
        with tempfile.TemporaryDirectory() as temporary, mocked_science():
            root = Path(temporary)
            collection_fixture(root)
            first = build_mprage_nifti_collection(root)
            run_pipeline(arguments(root))
            source = root/"vcc2/normal/nifti"
            source.rename(root/"retained_original_niftis")
            source.mkdir()
            result = build_mprage_nifti_collection(root)
            self.assertEqual(result["cases"][0]["branch"], first["cases"][0]["branch"])
            self.assertEqual({f["original_nifti"] for f in result["cases"][0]["files"]},
                             {f["original_nifti"] for f in first["cases"][0]["files"]})
            run_pipeline(arguments(root))
            build_mprage_nifti_collection(root)

    def test_head_mask_uses_selected_metal_reference(self):
        """Derive optional masking from the preferred metal normal magnitude, not the replaced original."""
        with tempfile.TemporaryDirectory() as temporary, mocked_science():
            root = Path(temporary)
            collection_fixture(root)
            run_pipeline(arguments(root))
            result = build_mprage_nifti_collection(root, include_head_mask=True)
            self.assertIn("vcc2_metal_csm", result["variants"][0]["head_mask"]["source_nifti"])
            self.assertEqual(result["storage_policy"]["metal_csm_selection"]["selected_image_count"], 2)


if __name__ == "__main__":
    unittest.main()
