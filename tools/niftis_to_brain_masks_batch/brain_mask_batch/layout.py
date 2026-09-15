"""Where brain masks live beside the reconstructions they describe.

A brain mask is a durable derivative of one fully sampled R3x1 baseline, so it
is stored next to that baseline's own head mask rather than in a per-run output
folder::

    <root>/FMP_199/MPRAGE_preGad/masks/
        head_mask_from_normal.nii.gz     shipped with the share
        brain_mask_hdbet.nii.gz          this tool
        brain_mask_hdbet.json            provenance and approval
        brain_mask_hdbet_qc.png          boundary figure for visual review

The sidecar records the baseline's digest, so a mask left behind by an earlier
reconstruction can be told apart from one that still matches its baseline.
Consumers refuse to score against a mask whose sidecar is not approved.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SUBJECT_PATTERN = re.compile(r"^FMP_(\d+)$")

MASKS_DIRNAME = "masks"
MASK_NAME = "brain_mask_hdbet.nii.gz"
SIDECAR_NAME = "brain_mask_hdbet.json"
QC_NAME = "brain_mask_hdbet_qc.png"

STATUS_PENDING = "visual_review_required"
STATUS_APPROVED = "approved"

FORMAT_VERSION = 1

#: Branches that may hold the fully sampled baseline, in preference order.
#: MPRAGE uses ``optimal_wavelet``; the GRE/SWI trees use ``selected_wavelet``.
BASELINE_BRANCHES: tuple[str, ...] = ("optimal_wavelet", "selected_wavelet", "fista_r0")

MAGNITUDE_TOKEN = "part-mag"


@dataclass(frozen=True)
class MaskTarget:
    """One subject/contrast a mask can be produced for.

    Attributes:
        subject: Subject folder name, for example ``FMP_199``.
        contrast: Contrast folder name, for example ``MPRAGE_preGad``.
        baseline: Fully sampled magnitude NIfTI the mask is cut from.
        masks_dir: Folder the mask, sidecar and QC figure are written to.
    """

    subject: str
    contrast: str
    baseline: Path
    masks_dir: Path

    @property
    def mask_path(self) -> Path:
        return self.masks_dir / MASK_NAME

    @property
    def sidecar_path(self) -> Path:
        return self.masks_dir / SIDECAR_NAME

    @property
    def qc_path(self) -> Path:
        return self.masks_dir / QC_NAME


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest.

    Args:
        path: File to digest.
        chunk_bytes: Read size.

    Returns:
        The hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def find_subjects(root: Path) -> list[str]:
    """List ``FMP_<n>`` subject folders under a share, in numeric order.

    Args:
        root: Reconstruction share root.

    Returns:
        Subject folder names sorted by their numeric id.

    Raises:
        FileNotFoundError: If the share root does not exist.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Share root does not exist: {root}")
    subjects = [
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and SUBJECT_PATTERN.match(entry.name)
    ]
    return sorted(subjects, key=lambda name: int(SUBJECT_PATTERN.match(name).group(1)))


def find_contrasts(root: Path, subject: str) -> list[str]:
    """List contrast folders of one subject that hold reconstructions.

    Args:
        root: Share root.
        subject: Subject folder name.

    Returns:
        Contrast folder names, sorted.
    """
    directory = root / subject
    if not directory.is_dir():
        return []
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.is_dir() and (entry / "original_nifti").is_dir()
    )


def find_baseline(contrast_dir: Path) -> Path | None:
    """Locate the fully sampled magnitude reconstruction of one contrast.

    Branches are tried in :data:`BASELINE_BRANCHES` order so MPRAGE and the
    GRE/SWI trees both resolve without naming the branch explicitly. Phase files
    share the folder, so the match is anchored on ``part-mag``.

    Args:
        contrast_dir: ``<root>/<subject>/<contrast>`` folder.

    Returns:
        The baseline magnitude NIfTI, or ``None`` when the contrast has none.

    Raises:
        RuntimeError: If a branch's ``normal`` folder is ambiguous.
    """
    for branch in BASELINE_BRANCHES:
        normal = contrast_dir / "original_nifti" / branch / "normal"
        if not normal.is_dir():
            continue
        matches = sorted(normal.glob(f"*{MAGNITUDE_TOKEN}_*.nii.gz"))
        if not matches:
            continue
        if len(matches) > 1:
            raise RuntimeError(f"Ambiguous baseline magnitude NIfTI in {normal}")
        return matches[0]
    return None


def discover_targets(
    root: Path,
    *,
    subjects: list[str] | None = None,
    contrasts: list[str] | None = None,
    output_root: Path | None = None,
) -> list[MaskTarget]:
    """Find every subject/contrast a brain mask can be produced for.

    Args:
        root: Share holding the reconstructions.
        subjects: Subjects to include, or ``None`` for every subject found.
        contrasts: Contrasts to include, or ``None`` for every contrast found.
        output_root: Mirror masks under this root instead of beside the data,
            keeping the same ``<subject>/<contrast>/masks`` layout.

    Returns:
        Targets in subject then contrast order.
    """
    wanted_subjects = subjects if subjects is not None else find_subjects(root)
    targets: list[MaskTarget] = []

    for subject in wanted_subjects:
        available = find_contrasts(root, subject)
        for contrast in contrasts if contrasts is not None else available:
            contrast_dir = root / subject / contrast
            if not contrast_dir.is_dir():
                continue
            baseline = find_baseline(contrast_dir)
            if baseline is None:
                continue
            base = output_root / subject / contrast if output_root else contrast_dir
            targets.append(
                MaskTarget(
                    subject=subject,
                    contrast=contrast,
                    baseline=baseline,
                    masks_dir=base / MASKS_DIRNAME,
                )
            )
    return targets


def write_sidecar(
    target: MaskTarget,
    *,
    generator: Mapping[str, Any],
    baseline_sha256: str,
    baseline_shape: tuple[int, ...],
    baseline_voxel_mm: tuple[float, ...],
    voxel_count: int,
    volume_ml: float,
) -> Path:
    """Record a freshly generated mask as pending visual review.

    Args:
        target: The subject/contrast the mask belongs to.
        generator: How the mask was produced.
        baseline_sha256: Digest of the baseline it was cut from.
        baseline_shape: Baseline array shape.
        baseline_voxel_mm: Baseline voxel size.
        voxel_count: Mask voxels.
        volume_ml: Mask volume in millilitres.

    Returns:
        The sidecar path.
    """
    payload = {
        "format_version": FORMAT_VERSION,
        "status": STATUS_PENDING,
        "approved": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": target.subject,
        "contrast": target.contrast,
        "mask": MASK_NAME,
        "qc_image": QC_NAME,
        "mask_sha256": sha256_file(target.mask_path),
        "voxel_count": voxel_count,
        "volume_ml": volume_ml,
        "generator": dict(generator),
        "baseline": {
            "path": str(target.baseline),
            "sha256": baseline_sha256,
            "shape": [int(value) for value in baseline_shape],
            "voxel_size_mm": [float(value) for value in baseline_voxel_mm],
        },
    }
    return _write_json(target.sidecar_path, payload)


def read_sidecar(path: Path) -> dict[str, Any]:
    """Read a mask sidecar.

    Args:
        path: Sidecar JSON.

    Returns:
        The sidecar payload.

    Raises:
        FileNotFoundError: If the sidecar is absent.
        ValueError: If the sidecar format is unsupported.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No brain-mask sidecar at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"Unsupported brain-mask sidecar version: {path}")
    return payload


def is_complete(target: MaskTarget) -> bool:
    """Whether a usable mask already exists for this target.

    The mask, its sidecar and its QC figure must all be present and the sidecar
    must still describe the mask on disk, so a half-finished or hand-edited mask
    is regenerated rather than silently reused.

    Args:
        target: The subject/contrast to check.

    Returns:
        True when the mask can be reused as is.
    """
    if not (
        target.mask_path.is_file()
        and target.sidecar_path.is_file()
        and target.qc_path.is_file()
    ):
        return False
    try:
        payload = read_sidecar(target.sidecar_path)
    except (ValueError, json.JSONDecodeError):
        return False
    return payload.get("mask_sha256") == sha256_file(target.mask_path)


def baseline_matches(target: MaskTarget, payload: Mapping[str, Any]) -> bool:
    """Whether a sidecar still describes the baseline now on disk.

    Args:
        target: The subject/contrast to check.
        payload: Its sidecar payload.

    Returns:
        True when the baseline digest is unchanged. Reading the baseline forces
        a download of an online-only file, so callers opt into this check.
    """
    recorded = str(payload.get("baseline", {}).get("sha256", ""))
    return bool(recorded) and recorded == sha256_file(target.baseline)


def approve(path: Path, *, approved_by: str, note: str = "") -> dict[str, Any]:
    """Record visual approval in one mask sidecar.

    Args:
        path: Sidecar JSON.
        approved_by: Who reviewed the QC figure.
        note: Optional reviewer comment.

    Returns:
        The updated sidecar payload.

    Raises:
        FileNotFoundError: If the mask the sidecar describes is missing.
        ValueError: If the mask changed since it was written.
    """
    payload = read_sidecar(path)
    mask_path = path.parent / str(payload["mask"])
    if not mask_path.is_file():
        raise FileNotFoundError(f"Sidecar describes a missing mask: {mask_path}")
    if sha256_file(mask_path) != payload.get("mask_sha256"):
        raise ValueError(
            f"Brain mask changed since it was written: {mask_path}. Regenerate it."
        )

    payload["status"] = STATUS_APPROVED
    payload["approved"] = True
    payload["approval"] = {
        "approved_by": approved_by,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    _write_json(path, payload)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON atomically so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
