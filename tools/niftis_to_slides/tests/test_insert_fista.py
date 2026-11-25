"""Raw FISTA comparisons follow each uncond pair without changing reviewed pages."""

import csv
import sys
from pathlib import Path

import nibabel as nib
import pytest
from pptx import Presentation

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT / "scripts"))

from build_retrospective_slides import build
from insert_baseline_slides import insert as insert_baselines
from insert_fista_slides import find_fista, insert
from test_insert_baselines import pictures, write_baseline
from test_retrospective import write_volume


def write_fista(references, root, source):
    path = references / source.relative_to(root)
    path = path.with_name(path.name.replace("_uncond", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.load(source)
    nib.save(nib.Nifti1Image(image.get_fdata() * 4, image.affine), path)
    return path


def test_fista_pair_order_echo_matching_and_reviewed_slide_preservation(tmp_path):
    root, references = tmp_path / "input", tmp_path / "references"
    for echo in ("01", "02"):
        for variant in ("native_r3x2", "native_r3x3"):
            source = write_volume(root, variant, echo=echo)
            raw = write_fista(references, root, source)
            # An SR output in the same directory must not count as raw FISTA.
            raw.with_name(raw.name.replace(".nii.gz", "_uncond.nii.gz")).write_bytes(raw.read_bytes())
        write_baseline(references, source, echo)
    template = TOOL_ROOT / "assets/slide_template.pptx"
    initial = tmp_path / "comparison.pptx"
    build(root, initial, template=template, reference_variant="native_r3x2")
    manifest = tmp_path / "comparison_assets/manifest.csv"
    with_baseline = tmp_path / "baseline/comparison.pptx"
    insert_baselines(root, references, initial, with_baseline, template=template)
    before = Presentation(with_baseline)
    output = tmp_path / "final/comparison.pptx"
    assert insert(root, references, with_baseline, output,
                  template=template, manifest=manifest) == 4
    after = Presentation(output)
    assert len(after.slides) == 11
    for start, echo in ((1, 1), (6, 2)):
        titles = [after.slides[index].shapes.title.text for index in range(start, start + 5)]
        assert titles[0] == f"Wave R3x1 – wavelet, echo {echo}"
        assert titles[1].startswith("Wave R3x2") and titles[1].endswith(", uncond")
        assert titles[2].startswith("Wave R3x3") and titles[2].endswith(", uncond")
        assert titles[3].startswith("Wave R3x2") and titles[3].endswith(", FISTA")
        assert titles[4].startswith("Wave R3x3") and titles[4].endswith(", FISTA")
        for position in (1, 2):
            uncond = pictures(after.slides[start + position])[0]
            fista = pictures(after.slides[start + position + 2])[0]
            assert (uncond.left, uncond.top, uncond.width, uncond.height) == (
                fista.left, fista.top, fista.width, fista.height)
            assert uncond.image.blob == fista.image.blob
    for old, index in zip(before.slides, (0, 1, 2, 3, 6, 7, 8)):
        new = after.slides[index]
        assert old._element.xml == new._element.xml
        assert [p.image.blob for p in pictures(old)] == [p.image.blob for p in pictures(new)]
    with (output.parent / "comparison_assets/fista_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(float(row["intensity_scale"]) == .25 for row in rows)
    with pytest.raises(ValueError, match="picture count"):
        insert(root, references, output, tmp_path / "duplicate.pptx",
               template=template, manifest=manifest)


def test_fista_matching_rejects_missing_and_ambiguous_originals(tmp_path):
    root, references = tmp_path / "input", tmp_path / "references"
    source = write_volume(root, "native_r3x2")
    row = dict(subject="FMP_1", contrast="SWI_preGad", source=str(source.relative_to(root)))
    raw = write_fista(references, root, source)
    wrong_echo = raw.with_name(raw.name.replace("echo-01", "echo-02"))
    raw.rename(wrong_echo)
    with pytest.raises(ValueError, match="found 0"):
        find_fista(references, row)
    wrong_echo.rename(raw)
    assert find_fista(references, row) == raw
    raw.with_name("duplicate_" + raw.name).write_bytes(raw.read_bytes())
    with pytest.raises(ValueError, match="found 2"):
        find_fista(references, row)
