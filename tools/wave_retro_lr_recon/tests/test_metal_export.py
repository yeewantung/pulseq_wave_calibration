"""Real nested export layout, exact legacy recovery and immutable reconstruction resume."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_metal_auto import automatic_fixture, mocked_auto, raw_files
from test_metal_pipeline import arguments, fake_convert
from wave_retro_lr.bart_io import sha256_file
from wave_retro_lr.metal_export import (LEGACY_PIPELINE_SHA256, LEGACY_PIPELINE_BYTES,
    LAYOUT_FIX_PIPELINE_SHA256, export_files, legacy_export_request)
from wave_retro_lr.metal_pipeline import run_pipeline
from wave_retro_lr.metal_workflow import read_json
from wave_retro_lr.nifti_collection import build_mprage_nifti_collection


def legacy_failure(root: Path) -> Path:
    """Create the exact old nested-export failure with a completed, hash-bound mock BART execution."""
    raw_files(root)
    automatic_fixture(root, keep_original_maps=True)

    def broken_export(*args, **kwargs):
        """Produce the converter's four nested files, then emulate the old directory-as-file error."""
        fake_convert(*args, **kwargs)
        raise IsADirectoryError('nifti/sub-fixture')

    with mocked_auto(root), patch('scripts.convert_mprage_bart_to_nifti.convert', side_effect=broken_export):
        try:
            run_pipeline(arguments(root, 'normal', '-g'))
        except IsADirectoryError:
            pass
        else:
            raise AssertionError('Expected legacy export failure')
    variant = root/'vcc2_metal_csm'
    root_path = variant/'metadata/request.json'
    request = read_json(root_path)
    request['implementation']['metal_pipeline.py'].update(
        sha256=LEGACY_PIPELINE_SHA256, size_bytes=LEGACY_PIPELINE_BYTES)
    root_path.write_text(json.dumps(request))
    branch = variant/'normal/bart_output/wavelet_metal_unvalidated'
    request_path = branch/'metadata/request.json'
    request = read_json(request_path)
    request['software_sha256']['metal_pipeline.py'] = LEGACY_PIPELINE_SHA256
    del request['software_sha256']['metal_export.py']
    request_path.write_text(json.dumps(request))
    preparation_path = branch/'metadata/preparation.json'
    preparation = read_json(preparation_path)
    preparation['request']['pipeline'] = request
    preparation_path.write_text(json.dumps(preparation))
    execution_path = branch/'metadata/reconstruction.json'
    execution = read_json(execution_path)
    execution['identity']['preparation_sha256'] = sha256_file(preparation_path)
    execution_path.write_text(json.dumps(execution))
    return branch


class MetalExportTests(unittest.TestCase):
    """Test the production layout rather than assuming mocked exports are flat."""

    def test_exact_upgrade_only(self):
        """Reject unknown old/new code, solver changes and converter changes during compatibility reuse."""
        saved = {'software_sha256': {'metal_pipeline.py': LEGACY_PIPELINE_SHA256, 'metal_runner.py': 'same'},
                 'converter_sha256': 'same', 'lambda': .03}
        current = deepcopy(saved)
        current['software_sha256'].update({'metal_pipeline.py': LAYOUT_FIX_PIPELINE_SHA256, 'metal_export.py': 'new'})
        self.assertTrue(legacy_export_request(saved, current))
        for key, value in [('lambda', .04), ('converter_sha256', 'changed')]:
            modified = dict(current, **{key: value})
            self.assertFalse(legacy_export_request(saved, modified))
        current['software_sha256']['metal_pipeline.py'] = 'unreviewed-new-code'
        self.assertFalse(legacy_export_request(saved, current))

    def test_failed_nested_export_recovery_then_retro_collection(self):
        """Recover only export registration, preserving original BART/provenance bytes and image hashes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            branch = legacy_failure(root)
            variant = root/'vcc2_metal_csm'
            frozen = [variant/'metadata/request.json', *(branch/'metadata').glob('*.json'),
                      *(branch/'reconstruction').iterdir(), *(branch/'nifti').rglob('*.nii.gz')]
            before = {p: sha256_file(p) for p in frozen}
            with mocked_auto(root):
                run_pipeline(arguments(root, 'normal', '-g'))
                registration = read_json(branch/'metadata/export_registration.json')
                self.assertTrue(registration['legacy_layout_revalidated'])
                self.assertEqual(before, {p: sha256_file(p) for p in frozen})
                for sidecar in (branch/'nifti').rglob('*.json'):
                    self.assertTrue(read_json(sidecar)['MetalCSM']['GPU'])
                with patch('scripts.convert_mprage_bart_to_nifti.convert', side_effect=AssertionError('must not reconvert')):
                    run_pipeline(arguments(root, 'normal', '-g'))
                run_pipeline(arguments(root, 'retro', '-g'))
                collection = build_mprage_nifti_collection(root, require_retro=True)
                self.assertEqual(len(collection['cases']), 4)

    def test_unknown_or_tampered_partial_export_rejected(self):
        """Never adopt edited NIfTIs/sidecars merely because a matching completed BART image exists."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            branch = legacy_failure(root)
            sidecar = next((branch/'nifti').rglob('*.json'))
            sidecar.write_text('{"tampered":true}')
            before = {p: sha256_file(p) for p in branch.rglob('*') if p.is_file()}
            with mocked_auto(root), self.assertRaisesRegex(ValueError, 'differs from verified conversion'):
                run_pipeline(arguments(root, 'normal', '-g'))
            self.assertEqual(before, {p: sha256_file(p) for p in branch.rglob('*') if p.is_file()})
            self.assertFalse((branch/'metadata/export.json').exists())

    def test_inventory_rejects_extras_and_symlinks(self):
        """Do not hide extra files/empty directories behind recursive file-only enumeration."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            branch = legacy_failure(root)
            nifti = branch/'nifti'
            self.assertEqual(len(export_files(nifti)), 4)
            (nifti/'extra').mkdir()
            with self.assertRaisesRegex(ValueError, 'layout'):
                export_files(nifti)
            (nifti/'extra').rmdir()
            (nifti/'alias').symlink_to(next(nifti.rglob('*.nii.gz')))
            with self.assertRaisesRegex(ValueError, 'Symlinks'):
                export_files(nifti)

    def test_registration_resume_after_completion_write_interruption(self):
        """Resume a hash-bound registration journal without reconversion after sidecars are augmented."""
        from wave_retro_lr.metal_workflow import write_json

        def interrupt_completion(path, value):
            """Emulate interruption at the final completion write, leaving prior journal/images intact."""
            if path.name == 'export.json':
                raise InterruptedError('completion interrupted')
            write_json(path, value)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            branch = legacy_failure(root)
            with mocked_auto(root), patch('wave_retro_lr.metal_export.write_json', side_effect=interrupt_completion):
                with self.assertRaises(InterruptedError):
                    run_pipeline(arguments(root, 'normal', '-g'))
            before = {p: sha256_file(p) for p in branch.rglob('*') if p.is_file()}
            with mocked_auto(root), patch('scripts.convert_mprage_bart_to_nifti.convert',
                                         side_effect=AssertionError('journal must resume without conversion')):
                run_pipeline(arguments(root, 'normal', '-g'))
            self.assertEqual(before, {p: sha256_file(p) for p in before})
            self.assertEqual(read_json(branch/'metadata/export.json')['status'], 'complete')
