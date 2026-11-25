#!/usr/bin/env python3
"""Build a visual comparison of retrospective magnitude NIfTIs without R3x1 data.

Supports <subject>/<contrast>/<container>/<branch>/retro/<variant>/*.nii[.gz].
Contrasts, branches, acceleration factors and echo numbers are discovered rather
than enumerated. Uses the existing TIFF renderer and PowerPoint template.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from nifti_slides import deck, images, metrics  # noqa: E402


def natural_key(value: str) -> tuple:
    return tuple(int(item) if item.isdigit() else item.lower()
                 for item in re.split(r"(\d+)", value))


@dataclass(frozen=True)
class Volume:
    path: Path
    subject: str
    contrast: str
    context: tuple[str, ...]
    variant: str
    signature: str

    @property
    def group(self) -> tuple:
        # A separate window/grid for each echo, branch and reconstruction type.
        return self.subject, self.contrast, self.context, self.signature

    @property
    def acceleration(self) -> str:
        match = re.search(r"r\d+x\d+", self.variant, re.IGNORECASE)
        return match[0][0].upper() + match[0][1:].lower() if match else self.variant


def discover(root: Path) -> list[Volume]:
    """Index magnitude files, retaining echo/model distinctions when pairing."""
    volumes = []
    for path in root.rglob("*.nii*"):
        if not path.name.endswith((".nii", ".nii.gz")) or "part-mag" not in path.name:
            continue
        parts = path.relative_to(root).parts
        if "retro" not in parts or len(parts) < 6:
            continue
        index = parts.index("retro")
        if index < 3 or len(parts) != index + 3:
            continue
        # Reconstruction names may spell NativeR3x3 versus native_r3x2.
        signature = re.sub(r"r\d+x\d+", "rx", path.name.lower())
        signature = re.sub(r"[^a-z0-9]", "", signature)
        volumes.append(Volume(path, parts[0], parts[1], parts[2:index],
                              parts[index + 1], signature))
    volumes.sort(key=lambda item: natural_key(str(item.path.relative_to(root))))
    if not volumes:
        raise ValueError(f"No retrospective part-mag NIfTIs found under {root}")
    return volumes


def canonical_magnitude(path: Path) -> nib.Nifti1Image:
    image, data = images.load_magnitude(path)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {data.shape}: {path}")
    return nib.as_closest_canonical(nib.Nifti1Image(data, image.affine))


def physical_strip(data: np.ndarray, voxel_sizes: tuple) -> np.ndarray:
    """Keep physical aspect ratios for anisotropic acquisitions such as SWI."""
    _, panels = images.center_slices(data)
    sx, sy, sz = voxel_sizes
    pixel_size = min(voxel_sizes)
    spacings = ((sz, sy), (sz, sx), (sy, sx))
    rendered = [zoom(panels[name], tuple(float(v / pixel_size) for v in spacing),
                     order=1, prefilter=False)
                for name, spacing in zip(images.ORIENTATION_ORDER, spacings)]
    height = max(panel.shape[0] for panel in rendered)
    padded = []
    for panel in rendered:
        missing = height - panel.shape[0]
        padded.append(np.pad(panel, ((missing // 2, missing - missing // 2), (0, 0))))
    return np.concatenate(padded, axis=1)


def slide_title(volume: Volume, voxel_sizes: tuple) -> str:
    branch = volume.context[-1]
    details = [deck.case_label(volume.variant, voxel_sizes),
               deck.BRANCH_LABELS.get(branch, branch)]
    echo = re.search(r"echo-(\d+)", volume.path.name)
    if echo:
        details.append(f"echo {int(echo[1])}")
    stem = volume.path.name.removesuffix(".gz").removesuffix(".nii")
    # Preserve reconstruction suffixes (e.g. uncond or SR model) in the title.
    if "BART" in stem:
        suffix = re.split(re.escape(branch.replace("_", "")),
                          stem.split("BART", 1)[1], maxsplit=1, flags=re.IGNORECASE)
        if len(suffix) == 2:
            if suffix[1].strip("_"):
                details.append(suffix[1].strip("_"))
    return f"Wave {volume.acceleration} – " + ", ".join(details)


def build(root: Path, output: Path, *, template: Path, reference_variant: str,
          percentile: float = 99.5, scale_mode: str = "lsq") -> tuple[int, int]:
    """Render every input once; the first acceleration supplies the display grid.

    This is a display reference only. No accuracy metrics are computed.
    """
    if not 0 < percentile <= 100:
        raise ValueError("Display percentile must be in (0, 100].")
    volumes = discover(root)
    groups = defaultdict(list)
    for volume in volumes:
        groups[volume.group].append(volume)
    # Catch ambiguous pairing before writing anything.
    for group in groups.values():
        variants = [volume.variant for volume in group]
        if len(set(variants)) != len(variants):
            raise ValueError(f"Duplicate variants in comparison group: {group}")

    assets = output.parent / f"{output.stem}_assets"
    builder = deck.DeckBuilder(template)
    rows = []
    current_section = None
    sections = 0
    for group in groups.values():
        group.sort(key=lambda volume: (volume.variant != reference_variant,
                                       natural_key(volume.variant)))
        reference = group[0]
        section = reference.subject, reference.contrast
        if section != current_section:
            builder.add_section(" ".join(section).replace("_", " "))
            current_section = section
            sections += 1
        print(f"{reference.subject}/{reference.contrast}: "
              f"display reference {reference.variant}", flush=True)
        reference_image = canonical_magnitude(reference.path)
        baseline = reference_image.get_fdata(dtype=np.float32)
        foreground = metrics.build_foreground(baseline, reference_image,
                                               mode="intensity").mask
        display_max = float(np.percentile(baseline[foreground], percentile))
        voxel_sizes = nib.affines.voxel_sizes(reference_image.affine)
        for volume in group:
            source_image = canonical_magnitude(volume.path)
            data, resampled = images.resample_to_baseline(source_image, reference_image)
            shown, gain = metrics.scale_to_baseline(baseline, data, foreground,
                                                    scale_mode)
            strip = physical_strip(shown, voxel_sizes)
            relative = volume.path.relative_to(root)
            tiff = assets / "pptx_img" / relative.parent / f"{relative.name}.tiff"
            images.write_strip_tiff(strip, display_max, tiff)
            raster, width, height = deck.brightened_png_bytes(tiff)
            builder.add_picture_slide(deck.SlideSpec(
                slide_title(volume, tuple(nib.affines.voxel_sizes(source_image.affine))),
                raster, width, height, with_axes=False))
            rows.append(dict(subject=volume.subject, contrast=volume.contrast,
                             variant=volume.variant, source=str(relative),
                             display_reference=str(reference.path.relative_to(root)),
                             display_max=display_max, intensity_scale=gain,
                             scale_mode=scale_mode, resampled=resampled,
                             orientation="RAS", image=str(tiff.relative_to(output.parent))))
            print(f"  {volume.acceleration}: {volume.path.name}", flush=True)
    builder.save(output)
    with (assets / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}: {sections} sections, {len(rows)} picture slides.", flush=True)
    return sections, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output", type=Path,
                        help="PPTX path; defaults to INPUT_ROOT/nifti_comparison.pptx")
    parser.add_argument("--template", type=Path,
                        default=TOOL_ROOT / "assets" / "slide_template.pptx")
    parser.add_argument("--reference-variant", default="native_r3x2",
                        help="Preferred display reference per group; otherwise first variant")
    parser.add_argument("--display-percentile", type=float, default=99.5)
    parser.add_argument("--scale", choices=metrics.SCALE_MODES, default="lsq")
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    output = (args.output or root / "nifti_comparison.pptx").expanduser().resolve()
    build(root, output, template=args.template, reference_variant=args.reference_variant,
          percentile=args.display_percentile, scale_mode=args.scale)


if __name__ == "__main__":
    main()
