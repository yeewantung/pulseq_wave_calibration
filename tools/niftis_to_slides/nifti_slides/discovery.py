"""Index measured Wave-MPRAGE reconstruction NIfTIs into comparison cases.

The reconstruction shares are laid out as::

    <root>/FMP_<n>/<contrast>/original_nifti/<branch>/normal/<mag>.nii.gz
    <root>/FMP_<n>/<contrast>/original_nifti/<branch>/retro/<variant>/<mag>.nii.gz

Super-resolution shares repeat that tree without ``normal``, carrying one file
per SR model with the model appended to the reconstruction stem. Some subjects
in the SR share nest the subject folder twice (``FMP_199/FMP_199/...``); both
spellings are accepted.

Every folder holds a ``part-phase`` file beside the ``part-mag`` file, so
magnitude selection is anchored on ``part-mag`` rather than on file count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


SUBJECT_PATTERN = re.compile(r"^FMP_(\d+)$")

#: Regularization branches, in the order sections present them.
BRANCH_ORDER: tuple[str, ...] = ("fista_r0", "optimal_wavelet")

#: The fully sampled branch that supplies the baseline grid and display window.
BASELINE_BRANCH = "optimal_wavelet"

#: Retrospective cases, in the order sections present them.
VARIANT_ORDER: tuple[str, ...] = (
    "native_r3x2",
    "lr_x_1p5mm_r3x2",
    "lr_y_1p5mm_r3x2",
    "lr_xy_1p25mm_r3x2",
)

#: Super-resolution models, in the order sections present them.
SR_MODEL_ORDER: tuple[str, ...] = ("shd", "cond_unet_retrain")

#: Contrasts handled by this tool. SWI uses different branch and variant names
#: and is deliberately out of scope.
CONTRAST_ORDER: tuple[str, ...] = ("MPRAGE_preGad", "MPRAGE_postGad")

MAGNITUDE_TOKEN = "part-mag"


@dataclass(frozen=True)
class Case:
    """One volume to render and score.

    Attributes:
        subject: Subject folder name, for example ``FMP_199``.
        contrast: Contrast folder name, for example ``MPRAGE_preGad``.
        branch: Regularization branch, for example ``fista_r0``.
        variant: Retrospective case, or ``normal`` for the baseline.
        stage: ``Reference``, ``Pre-SR`` or ``Post-SR``. The baseline keeps
            the ``Reference`` token so image filenames match the established
            naming convention.
        sr_model: Super-resolution model for ``Post-SR``, otherwise empty.
        path: Magnitude NIfTI on disk.
    """

    subject: str
    contrast: str
    branch: str
    variant: str
    stage: str
    sr_model: str
    path: Path

    @property
    def subject_number(self) -> int:
        """Numeric subject id used for ordering."""
        match = SUBJECT_PATTERN.match(self.subject)
        if match is None:
            raise ValueError(f"Subject folder is not FMP_<n>: {self.subject}")
        return int(match.group(1))

    @property
    def is_baseline(self) -> bool:
        return self.stage == "Reference"

    @property
    def image_stem(self) -> str:
        """Filename stem carrying the full provenance of the volume.

        The stem names the subject, contrast, regularization branch, the
        retrospective case (which itself encodes native-versus-LR and the
        acceleration), the SR stage and the SR model.
        """
        parts = [
            self.subject.replace("_", ""),
            self.contrast.replace("MPRAGE_", ""),
            self.branch,
            self.variant,
            self.stage,
        ]
        if self.sr_model:
            parts.append(self.sr_model)
        return "_".join(parts)


def find_subjects(root: Path) -> list[str]:
    """List ``FMP_<n>`` subject folders under a share, in numeric order.

    Args:
        root: Reconstruction share root.

    Returns:
        Subject folder names sorted by their numeric id.
    """
    subjects = [
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and SUBJECT_PATTERN.match(entry.name)
    ]
    return sorted(subjects, key=lambda name: int(SUBJECT_PATTERN.match(name).group(1)))


def subject_contrast_dir(root: Path, subject: str, contrast: str) -> Path | None:
    """Resolve a subject/contrast folder across both nesting spellings.

    Args:
        root: Share root.
        subject: Subject folder name.
        contrast: Contrast folder name.

    Returns:
        The existing directory, or ``None`` when the scan is absent.
    """
    for candidate in (root / subject / subject / contrast, root / subject / contrast):
        if candidate.is_dir():
            return candidate
    return None


def find_magnitude(directory: Path, sr_models: Sequence[str], sr_model: str = "") -> Path | None:
    """Resolve the single magnitude NIfTI in a reconstruction folder.

    Args:
        directory: Folder holding one reconstruction's NIfTI outputs.
        sr_models: Known SR model suffixes, used to keep SR outputs out of the
            plain reconstruction match.
        sr_model: SR model whose output is wanted, or empty for the plain
            reconstruction.

    Returns:
        The magnitude NIfTI, or ``None`` when the folder has none.

    Raises:
        RuntimeError: If the folder holds more than one matching magnitude file.
    """
    if not directory.is_dir():
        return None

    suffix = f"_{sr_model}" if sr_model else ""
    matches = sorted(directory.glob(f"*{MAGNITUDE_TOKEN}_*{suffix}.nii.gz"))

    if not sr_model:
        model_tags = tuple(f"_{model}.nii.gz" for model in sr_models)
        matches = [path for path in matches if not path.name.endswith(model_tags)]

    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous magnitude NIfTI in {directory}: {matches}")
    return matches[0]


def baseline_path(reference_dir: Path, branch: str, sr_models: Sequence[str]) -> Path | None:
    """Locate the fully sampled R3x1 reconstruction of one branch.

    Args:
        reference_dir: Subject/contrast folder inside the reference share.
        branch: Regularization branch.
        sr_models: Known SR model suffixes.

    Returns:
        The baseline magnitude NIfTI, or ``None`` when the branch has none.
    """
    return find_magnitude(
        reference_dir / "original_nifti" / branch / "normal", sr_models
    )


def index_cases(
    input_root: Path,
    reference_root: Path,
    *,
    subjects: Sequence[str] | None = None,
    contrasts: Sequence[str] = CONTRAST_ORDER,
    branches: Sequence[str] = BRANCH_ORDER,
    variants: Sequence[str] = VARIANT_ORDER,
    sr_models: Sequence[str] = SR_MODEL_ORDER,
) -> list[Case]:
    """Index every renderable volume across both shares, in presentation order.

    The input share decides which subjects and contrasts appear. Pre-SR volumes
    and the fully sampled baseline are read from the reference share, which is
    the only place they exist when the input share holds SR outputs only.

    Args:
        input_root: Share whose subjects drive the run.
        reference_root: Share holding the fully sampled baselines and the
            unmodified retrospective reconstructions.
        subjects: Subject folders to include; ``None`` takes every subject found
            in the input share.
        contrasts: Contrast folders to include.
        branches: Regularization branches to include.
        variants: Retrospective cases to include.
        sr_models: Super-resolution models to include.

    Returns:
        Cases ordered subject, contrast, baseline, then retro case, branch and
        SR stage, which is the order the slides follow.
    """
    wanted = list(subjects) if subjects is not None else find_subjects(input_root)
    cases: list[Case] = []

    for subject in wanted:
        for contrast in contrasts:
            reference_dir = subject_contrast_dir(reference_root, subject, contrast)
            if reference_dir is None:
                continue
            cases.extend(
                _index_subject_contrast(
                    input_root=input_root,
                    reference_dir=reference_dir,
                    subject=subject,
                    contrast=contrast,
                    branches=branches,
                    variants=variants,
                    sr_models=sr_models,
                )
            )

    return cases


def _index_subject_contrast(
    *,
    input_root: Path,
    reference_dir: Path,
    subject: str,
    contrast: str,
    branches: Sequence[str],
    variants: Sequence[str],
    sr_models: Sequence[str],
) -> Iterator[Case]:
    """Yield the cases of one subject/contrast in presentation order."""
    baseline = baseline_path(reference_dir, BASELINE_BRANCH, sr_models)
    if baseline is None:
        return

    yield Case(
        subject=subject,
        contrast=contrast,
        branch=BASELINE_BRANCH,
        variant="normal",
        stage="Reference",
        sr_model="",
        path=baseline,
    )

    input_dir = subject_contrast_dir(input_root, subject, contrast)

    for variant in variants:
        for branch in branches:
            relative = Path("original_nifti") / branch / "retro" / variant

            pre_sr = find_magnitude(reference_dir / relative, sr_models)
            if pre_sr is not None:
                yield Case(
                    subject=subject,
                    contrast=contrast,
                    branch=branch,
                    variant=variant,
                    stage="Pre-SR",
                    sr_model="",
                    path=pre_sr,
                )

            if input_dir is None:
                continue

            for model in sr_models:
                post_sr = find_magnitude(input_dir / relative, sr_models, sr_model=model)
                if post_sr is not None:
                    yield Case(
                        subject=subject,
                        contrast=contrast,
                        branch=branch,
                        variant=variant,
                        stage="Post-SR",
                        sr_model=model,
                        path=post_sr,
                    )


def group_by_section(cases: Sequence[Case]) -> dict[tuple[str, str], list[Case]]:
    """Group cases by the (subject, contrast) pair that becomes one section.

    Args:
        cases: Indexed cases.

    Returns:
        Mapping from (subject, contrast) to its cases, preserving input order.
    """
    sections: dict[tuple[str, str], list[Case]] = {}
    for case in cases:
        sections.setdefault((case.subject, case.contrast), []).append(case)
    return sections
