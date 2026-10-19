"""Non-destructive path migration, raw attestation and strict provenance checks."""

import json
import os
from pathlib import Path
import tempfile
import unittest
import shutil
from unittest.mock import patch

from wave_retro_lr.relocation import RECEIPT, checked_hash, read_json, read_command, relocation_record


def fixture(dataset: Path) -> tuple[Path, Path, Path]:
    """Create a tiny moved dataset and an explicit receipt; return root, manifest and raw."""
    root = dataset/"reconstruction"
    root.mkdir()
    raw = dataset/"raw.dat"
    raw.write_bytes(b"raw sample")
    old = "/historical/disk/dataset"
    manifest = root/"manifest.json"
    manifest.write_text(json.dumps({"source": {"path": old+"/raw.dat"},
        "base": old+"/reconstruction/maps", "sha256": "unchanged",
        "relative": "maps", "unrelated": old+"-other/maps",
        "expected_commands": [f"bart wave {old}/reconstruction/maps"]}))
    command = root/"ecalib_command.txt"
    command.write_text(f"bart ecalib '{old}/reconstruction/maps'\n")
    stat = raw.stat()
    receipt = {"schema": "mprage-path-relocation-v1", "root": str(root),
        "target_dataset": str(dataset), "source_prefixes": [old],
        "documents": {"manifest.json": checked_hash(manifest), "ecalib_command.txt": checked_hash(command)},
        "raw_sources": [{"path": str(raw), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": checked_hash(raw), "historical_sha256_available": False,
            "verification": "user-confirmed-relocation-plus-size-mtime; current-sha256"}]}
    (root/RECEIPT).parent.mkdir()
    (root/RECEIPT).write_text(json.dumps(receipt))
    return root, manifest, raw


class RelocationTests(unittest.TestCase):
    """Exercise receipts with tiny temporary data and no production writes."""

    def test_view_preserves_original_bytes_and_hashes(self):
        """Rebase exact prefixes/command tokens only; leave unlisted new documents unchanged."""
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, raw = fixture(Path(temporary))
            original = manifest.read_bytes()
            value = read_json(manifest)
            self.assertEqual(value["source"]["path"], str(raw))
            self.assertEqual(value["sha256"], "unchanged")
            self.assertEqual(value["relative"], "maps")
            self.assertEqual(value["unrelated"], "/historical/disk/dataset-other/maps")
            self.assertEqual(value["expected_commands"], [f"bart wave {root}/maps"])
            self.assertEqual(read_command(root/"ecalib_command.txt"), f"bart ecalib {root}/maps")
            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(relocation_record(manifest)["sha256"], checked_hash(root/RECEIPT))
            new = root/"new.json"
            new.write_bytes(original)
            self.assertEqual(read_json(new), json.loads(original))

    def test_modified_original_and_raw_same_size_mtime_rejected(self):
        """Reject source edits, including raw changes that preserve legacy size/mtime."""
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, raw = fixture(Path(temporary))
            read_json(manifest)
            original = manifest.read_bytes()
            manifest.write_bytes(original+b" ")
            with self.assertRaisesRegex(ValueError, "document changed"):
                read_json(manifest)
            manifest.write_bytes(original)
            stat = raw.stat()
            raw.write_bytes(b"bad sample")
            os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "raw identity changed"):
                read_json(manifest)

    def test_receipt_root_scope_and_escape_rejected(self):
        """Reject copied receipts, broad roots, and path traversal outside the dataset."""
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest, raw = fixture(Path(temporary))
            receipt_path = root/RECEIPT
            receipt = json.loads(receipt_path.read_text())
            for prefix in ("/", "/historical/../dataset"):
                receipt_path.write_text(json.dumps({**receipt, "source_prefixes": [prefix]}))
                with self.assertRaisesRegex(ValueError, "source prefix"):
                    read_json(manifest)
            receipt_path.write_text(json.dumps(receipt))
            manifest.write_text(json.dumps({"path": "/historical/disk/dataset/../../escape"}))
            receipt["documents"]["manifest.json"] = checked_hash(manifest)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "escapes"):
                read_json(manifest)

    def test_without_receipt_is_unchanged(self):
        """Keep ordinary source behavior unchanged without explicit relocation registration."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"manifest.json"
            path.write_text('{"path": "/historical/disk/dataset/raw.dat"}')
            self.assertEqual(read_json(path), json.loads(path.read_text()))
            self.assertIsNone(relocation_record(path))

    def test_registered_move_normal_retro_collection_and_resume(self):
        """Audit a real fixture move and exercise automatic source/collection without changing originals."""
        from test_metal_auto import automatic_fixture, mocked_auto
        from test_metal_pipeline import arguments
        from test_retro_legacy_reuse import upgrade_retro_fixture
        from wave_retro_lr.mprage import prepare_retro_mprage, prepare_retro_mprage_r3x3
        from scripts.register_mprage_relocation import build_receipt
        from wave_retro_lr.mprage import _file_identity, _write_normal_reuse_attestation
        from wave_retro_lr.metal_pipeline import run_pipeline
        from wave_retro_lr.nifti_collection import build_mprage_nifti_collection

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            old = parent/"old"/"dataset"
            old.mkdir(parents=True)
            root = old/"reconstruction"
            automatic_fixture(root, keep_original_maps=True)
            upgrade_retro_fixture(root)
            (old/"raw.dat").write_bytes(b"raw")
            (old/"scan.seq").write_bytes(b"sequence")
            source = {"twix": _file_identity(old/"raw.dat"),
                      "sequence": _file_identity(old/"scan.seq", include_hash=True)}
            normal_path = root/"vcc2/normal/bart_inputs/manifest.json"
            normal = json.loads(normal_path.read_text())
            normal["source"] = source
            normal_path.write_text(json.dumps(normal))
            for path in (root/"vcc2/retro").glob("*/bart_inputs/manifest.json"):
                payload = json.loads(path.read_text())
                payload["source"] = source
                payload["source_normal_manifest_identity"]["sha256"] = checked_hash(normal_path)
                path.write_text(json.dumps(payload))
            branch_path = root/"vcc2/normal/bart_output/wavelet_selected/reconstruction_manifest.json"
            branch = json.loads(branch_path.read_text())
            branch["request"]["source"] = source
            for key in ("normal_manifest", "prepared_manifest"):
                branch["request"][key]["sha256"] = checked_hash(normal_path)
            branch_path.write_text(json.dumps(branch))
            historical_attestation = root/"vcc2/normal/NORMAL_INPUT_REUSE_ATTESTATION.json"
            historical_attestation.write_text('{"historical": true}')
            new = parent/"new_dataset"
            shutil.move(old, new)
            root = new/"reconstruction"
            original = {str(p.relative_to(root)): p.read_bytes() for p in (root/"vcc2").rglob("*") if p.is_file()}
            receipt = build_receipt(root, [str(old)], 2)
            (root/RECEIPT).parent.mkdir()
            (root/RECEIPT).write_text(json.dumps(receipt))
            # Do not mock the preparation boundary that failed on real legacy
            # data: run all crop/mask/reuse checks on the moved tiny fixtures.
            with mocked_auto(root), \
                 patch("wave_retro_lr.mprage.prepare_retro_mprage", wraps=prepare_retro_mprage), \
                 patch("wave_retro_lr.mprage.prepare_retro_mprage_r3x3", wraps=prepare_retro_mprage_r3x3):
                for mode in ("normal", "retro", "normal", "retro", "r3x3"):
                    args = arguments(root, mode)
                    args.twix, args.seq = new/"raw.dat", new/"scan.seq"
                    destination = run_pipeline(args)
                result = build_mprage_nifti_collection(root, require_retro=True)
                self.assertEqual(len(result["cases"]), 4)
                self.assertEqual(read_json(destination/"metadata/source.json")["path_relocation"],
                                 relocation_record(root/"vcc2/normal/bart_inputs/manifest.json"))
            self.assertEqual(original, {str(p.relative_to(root)): p.read_bytes()
                                       for p in (root/"vcc2").rglob("*") if p.is_file()})
            inputs = root/"vcc2/normal/bart_inputs"
            new_attestation = _write_normal_reuse_attestation(inputs, read_json(inputs/"manifest.json"), {}, {}, [])
            self.assertEqual(new_attestation.name, "NORMAL_INPUT_REUSE_ATTESTATION_RELOCATED.json")
            self.assertEqual((root/"vcc2/normal/NORMAL_INPUT_REUSE_ATTESTATION.json").read_text(),
                             '{"historical": true}')
