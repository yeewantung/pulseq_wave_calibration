"""Validate nested MPRAGE exports and register completed images without rerunning BART."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from .metal_workflow import file_record, read_json, write_json


LEGACY_PIPELINE_SHA256 = "8c158f559308c1bef8a2337d1978730ae7e7bbb27ba6dbff20c092c71e6566c1"
LEGACY_PIPELINE_BYTES = 23326
LAYOUT_FIX_PIPELINE_SHA256 = "2b37387e44c477b4aeba64b371e797f20c73b9d161a42ccbccae157782d99dd9"


def legacy_export_request(saved: dict, current: dict) -> bool:
    """Allow only the reviewed export-layout-only upgrade; preserve all scientific request fields.

    Match the exact historical pipeline implementation, not arbitrary old code.
    Original request/preparation files are retained so completed BART execution
    remains bound to its original preparation hash.
    """
    candidate = deepcopy(saved)
    changed = False
    if "software_sha256" in candidate:
        old = candidate["software_sha256"]
        new = current.get("software_sha256", {})
        if (old.get("metal_pipeline.py") == LEGACY_PIPELINE_SHA256 and "metal_export.py" not in old
                and new.get("metal_pipeline.py") == LAYOUT_FIX_PIPELINE_SHA256):
            old["metal_pipeline.py"] = new.get("metal_pipeline.py")
            old["metal_export.py"] = new.get("metal_export.py")
            changed = True
    old_impl, new_impl = candidate.get("implementation", {}), current.get("implementation", {})
    if isinstance(old_impl, dict) and isinstance(new_impl, dict):
        old_record = old_impl.get("metal_pipeline.py", old_impl)
        new_record = new_impl.get("metal_pipeline.py", new_impl)
        expected = {"path": new_record.get("path"), "sha256": LEGACY_PIPELINE_SHA256,
                    "size_bytes": LEGACY_PIPELINE_BYTES}
        if old_record == expected and new_record.get("sha256") == LAYOUT_FIX_PIPELINE_SHA256:
            if "metal_pipeline.py" in old_impl:
                candidate["implementation"]["metal_pipeline.py"] = new_record
            else:
                candidate["implementation"] = new_record
            changed = True
    return changed and candidate == current


def export_files(root: Path) -> list[Path]:
    """Return exactly two paired NIfTI/JSON exports, flat or in one subject directory.

    Reject symlinks, extra files, mixed layouts, unexpected nesting and empty
    directories rather than silently excluding unregistered artifacts.
    """
    entries = list(root.rglob("*"))
    if root.is_symlink() or any(p.is_symlink() for p in entries):
        raise ValueError("Symlinks in metal NIfTI exports are not supported")
    files = sorted(p for p in entries if p.is_file())
    parents = {p.parent for p in files}
    if len(files) != 4 or len(parents) != 1:
        raise ValueError("Expected magnitude/phase NIfTI plus two JSON sidecars")
    parent = next(iter(parents))
    expected_dirs = set() if parent == root else {parent}
    if (parent != root and (parent.parent != root or not parent.name.startswith("sub-"))) or {
            p for p in entries if p.is_dir()} != expected_dirs:
        raise ValueError("Unexpected metal NIfTI directory layout")
    paired = set()
    for part in ("mag", "phase"):
        images = [p for p in files if f"_part-{part}_" in p.name and p.name.endswith(".nii.gz")]
        if len(images) != 1:
            raise ValueError(f"Expected exactly one {part} NIfTI")
        sidecar = images[0].with_name(images[0].name.removesuffix(".nii.gz")+".json")
        paired.update((images[0], sidecar))
    if paired != set(files):
        raise ValueError("Metal NIfTI/JSON pairing mismatch")
    return files


def validate_images(files: list[Path], target: tuple) -> None:
    """Check output shape, affine agreement and finite values for both image parts."""
    affine = None
    for path in files:
        if not path.name.endswith(".nii.gz"):
            continue
        image = nib.load(path)
        if (image.shape != target[::-1] or not np.isfinite(image.affine).all()
                or not np.isfinite(np.asarray(image.dataobj)).all()
                or (affine is not None and not np.allclose(affine, image.affine, rtol=0, atol=1e-6))):
            raise ValueError("Metal NIfTI geometry/finite-value validation failed")
        affine = image.affine


def sidecar_bytes(payload: dict, metal: dict) -> bytes:
    """Return deterministic JSON bytes with exact metal preparation/execution provenance."""
    return (json.dumps(dict(payload, MetalCSM=metal), indent=2, allow_nan=False)+"\n").encode()


def finish_export(directory: Path, target: tuple, metal: dict, convert, *, legacy: bool = False) -> None:
    """Convert/register or safely finish the known legacy layout failure without BART.

    A journal hash-binds raw exports before sidecar augmentation. Unjournaled
    legacy files are adopted only after independently reconverting the already
    validated complex reconstruction to temporary files and comparing exact
    NIfTI hashes and sidecar content. Mismatches preserve all existing files.
    """
    root = directory/"nifti"
    completion = directory/"metadata/export.json"
    journal_path = directory/"metadata/export_registration.json"
    execution = file_record(directory/"metadata/reconstruction.json")
    if completion.exists():
        saved = read_json(completion)
        files = export_files(root)
        if (saved.get("status") != "complete" or saved["execution"] != execution
                or saved["outputs"] != [file_record(p) for p in files]):
            raise ValueError("Metal NIfTI output changed or execution identity changed")
        if "export_registration" in saved and file_record(journal_path) != saved["export_registration"]:
            raise ValueError("Metal export registration changed")
        validate_images(files, target)
        for path in files:
            if path.suffix == ".json" and read_json(path).get("MetalCSM") != metal:
                raise ValueError("Metal NIfTI sidecar provenance changed")
        return
    if journal_path.exists():
        journal = read_json(journal_path)
        if journal["execution"] != execution or journal["metal"] != metal:
            raise ValueError("Metal export registration source mismatch")
        files = export_files(root)
    else:
        recovery = False
        if any(root.iterdir()):
            if not legacy:
                raise ValueError("Partial metal NIfTI export; refusing overwrite")
            files = export_files(root)
            if files[0].parent == root:
                raise ValueError("Legacy layout recovery requires the nested converter layout")
            # This repeats conversion only, never calibration or reconstruction.
            print("Revalidating nested NIfTI exports from the completed complex image; no BART reconstruction.", flush=True)
            with tempfile.TemporaryDirectory(prefix=".export-layout-check-", dir=directory) as temporary:
                regenerated = Path(temporary)
                convert(regenerated)
                fresh = export_files(regenerated)
                if [p.relative_to(root) for p in files] != [p.relative_to(regenerated) for p in fresh]:
                    raise ValueError("Legacy export filenames differ from verified conversion")
                validate_images(fresh, target)
                for old, new in zip(files, fresh, strict=True):
                    equal = (read_json(old) == read_json(new) if old.suffix == ".json"
                             else file_record(old)["sha256"] == file_record(new)["sha256"])
                    if not equal:
                        raise ValueError("Unregistered legacy export differs from verified conversion; files preserved")
            recovery = True
        else:
            convert(root)
            files = export_files(root)
        validate_images(files, target)
        journal = {"execution": execution, "metal": metal,
                   "original_outputs": [file_record(p) for p in files],
                   "original_sidecars": {str(p): read_json(p) for p in files if p.suffix == ".json"},
                   "legacy_layout_revalidated": recovery, "export_implementation": file_record(Path(__file__))}
        write_json(journal_path, journal)
    if {str(p) for p in files} != {r["path"] for r in journal["original_outputs"]}:
        raise ValueError("Metal export file inventory changed")
    updates = {}
    for record in journal["original_outputs"]:
        path = Path(record["path"])
        actual = file_record(path)
        if path.suffix == ".json":
            content = sidecar_bytes(journal["original_sidecars"][str(path)], metal)
            final = dict(record, size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            if actual not in (record, final):
                raise ValueError("Metal sidecar differs from registered original and final metadata")
            if actual != final:
                updates[path] = content
        elif actual != record:
            raise ValueError("Metal NIfTI image differs from registered export")
    validate_images(files, target)
    for path, content in updates.items():
        path.write_bytes(content)
    write_json(completion, {"status": "complete", "outputs": [file_record(p) for p in files],
                           "execution": execution, "export_registration": file_record(journal_path)})
