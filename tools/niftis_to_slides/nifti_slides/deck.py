"""Assemble the comparison deck from rendered center-slice strips.

Slide pattern, matched to the hand-built reference deck::

    section : one "Section Header" slide per subject/contrast
    first   : the fully sampled baseline, titled "Wave R3x1 - wavelet"
    then    : "Wave Fake R3x2 - <LR case>, <regularization>, <SR>"
    picture : width 13 in, exactly centered, sent to back, +20% brightness
    caption : optional NRMSE / SSIM / PSNR line below the picture
    order   : retro LR case  >  regularization  >  SR stage

Brightness is handled the way PowerPoint stores it. PowerPoint renders the
pixels it holds and treats ``brightnessContrast`` as pane state, so a picture
whose raster is uncorrected renders uncorrected however the pane reads. The
correction is therefore baked into the embedded PNG, and the XML effect is
written so the Format Picture pane still reports +20%.

The theme, layouts and the x/y axis indicator group all come from the template
deck in ``assets/``; nothing about the styling is synthesised here.
"""

from __future__ import annotations

import copy
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from lxml import etree
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt


PICTURE_WIDTH = Inches(13.0)
PICTURE_LAYOUT = "Title and Content"
SECTION_LAYOUT = "Section Header"

BASELINE_TITLE = "Wave R3x1 – wavelet"
SECTION_TITLE = "FMP {number} MPRAGE {contrast}"
COMPARISON_TITLE = "Wave Fake {acceleration} – {case}, {branch}, {stage}"

# Charts occupy the content area below the title, never the picture band.
FIGURE_TOP = Inches(1.15)
FIGURE_MAX_WIDTH = Inches(12.4)
FIGURE_MAX_HEIGHT = Inches(5.9)

AXIS_GROUP_NAME = "Group 8"
TEMPLATE_ASSET_TITLE = "TEMPLATE ASSETS"

METRICS_SHAPE_NAME = "MetricsCaption"
# The tallest strip ends at 6.35 in, leaving the band below it for captions.
METRICS_TOP = Inches(6.40)
METRICS_HEIGHT = Inches(1.00)
METRICS_FONT = Pt(14)

BRIGHTNESS = 20000  # 1/1000 percent, i.e. +20%
A14_NS = "http://schemas.microsoft.com/office/drawing/2010/main"
IMG_PROPS_EXT_URI = "{BEBA8EAE-BF5A-486C-A8C5-ECC9F3942E4B}"

#: PowerPoint's own +20% brightness curve, measured by pairing the rasters of a
#: hand-corrected deck against the strips they were made from. It is close to a
#: 1.25x gain but rolls off in the shadows and highlights; reproducing it keeps
#: generated slides pixel-matched to hand-made ones, within 2/255 on held-out
#: slides with 97% of pixels within 1/255. Calibrated for +20% only.
POWERPOINT_BRIGHT20_LUT: tuple[int, ...] = (
      0,   1,   3,   5,   6,   9,  10,  11,  13,  14,  16,  17,  18,  19,  20,  21,
     23,  24,  25,  26,  27,  28,  31,  32,  33,  34,  35,  36,  37,  39,  40,  41,
     42,  44,  45,  46,  48,  49,  50,  51,  52,  53,  55,  56,  57,  58,  60,  61,
     62,  63,  65,  66,  67,  68,  69,  70,  71,  74,  75,  76,  77,  78,  79,  80,
     82,  83,  84,  85,  86,  87,  90,  91,  92,  93,  94,  95,  97,  98,  99, 100,
    101, 102, 104, 105, 106, 107, 108, 110, 111, 113, 114, 115, 116, 118, 119, 120,
    121, 122, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 136, 137, 138, 139,
    140, 142, 143, 144, 145, 146, 148, 149, 150, 151, 152, 154, 155, 156, 157, 159,
    160, 161, 162, 163, 165, 166, 167, 168, 170, 171, 172, 173, 174, 176, 177, 178,
    179, 181, 182, 183, 184, 186, 187, 188, 189, 190, 192, 193, 194, 195, 197, 198,
    199, 200, 202, 203, 204, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 218,
    219, 220, 222, 223, 224, 225, 227, 228, 229, 230, 232, 233, 234, 236, 237, 238,
    239, 240, 241, 242, 243, 245, 246, 247, 248, 250, 251, 252, 253, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
)

#: Retrospective case folder names, parsed into their title wording.
NATIVE_PATTERN = re.compile(r"^native_(r\d+x\d+)$")
LOW_RESOLUTION_PATTERN = re.compile(r"^lr_([a-z]+)_(\d+p\d+)mm_(r\d+x\d+)$")

BRANCH_LABELS = {"fista_r0": "no reg", "optimal_wavelet": "wavelet"}
STAGE_LABELS = {
    "Pre-SR": "no SR",
    "cond_unet_retrain": "retrained U-Net",
    "shd": "SHD",
}


@dataclass(frozen=True)
class SlideSpec:
    """One slide to build.

    Attributes:
        title: Slide title.
        image: Brightened 8-bit PNG bytes to embed, held in memory.
        width_px: Strip width in pixels.
        height_px: Strip height in pixels.
        with_axes: Whether the x/y axis indicator belongs on this slide.
        captions: One metric line per scored mask; empty for none.
    """

    title: str
    image: bytes
    width_px: int
    height_px: int
    with_axes: bool
    captions: Sequence[str] = field(default=())


def section_title(subject: str, contrast: str) -> str:
    """Title of the section header for one subject/contrast.

    Args:
        subject: Subject folder name, for example ``FMP_199``.
        contrast: Contrast folder name, for example ``MPRAGE_preGad``.

    Returns:
        The section header title.
    """
    number = "".join(character for character in subject if character.isdigit())
    return SECTION_TITLE.format(number=number, contrast=contrast.replace("MPRAGE_", ""))


def case_label(variant: str, voxel_size_mm: tuple[float, float, float]) -> str:
    """Title wording for one retrospective case.

    The native case is named after its measured voxel size rather than assuming
    1 mm, and low-resolution cases are named after the axes and size in their
    folder name.

    Args:
        variant: Retrospective case folder name.
        voxel_size_mm: Voxel size of the reconstruction, used for the native case.

    Returns:
        The wording used inside the slide title.
    """
    if NATIVE_PATTERN.match(variant):
        sizes = {round(float(value), 2) for value in voxel_size_mm}
        if len(sizes) == 1:
            return f"{_trim(sizes.pop())}mm iso"
        return " x ".join(_trim(round(float(value), 2)) for value in voxel_size_mm) + "mm"

    low_resolution = LOW_RESOLUTION_PATTERN.match(variant)
    if low_resolution is not None:
        axes, size, _ = low_resolution.groups()
        return f"LR {axes} {size.replace('p', '.')}mm"

    return variant


def acceleration_label(variant: str) -> str:
    """Acceleration factor carried by a retrospective case folder name.

    Args:
        variant: Retrospective case folder name.

    Returns:
        The acceleration, for example ``R3x2``; falls back to ``R3x2`` when the
        folder name carries none. Only the leading letter is capitalized, so the
        separator stays lowercase as the deck writes it.
    """
    for pattern in (NATIVE_PATTERN, LOW_RESOLUTION_PATTERN):
        match = pattern.match(variant)
        if match is not None:
            factor = match.groups()[-1]
            return factor[0].upper() + factor[1:]
    return "R3x2"


def comparison_title(
    variant: str, branch: str, stage: str, sr_model: str, voxel_size_mm
) -> str:
    """Build the title of one comparison slide.

    Args:
        variant: Retrospective case folder name.
        branch: Regularization branch.
        stage: ``Pre-SR`` or ``Post-SR``.
        sr_model: SR model for ``Post-SR``, otherwise empty.
        voxel_size_mm: Voxel size of the reconstruction.

    Returns:
        The slide title.
    """
    stage_key = sr_model if stage == "Post-SR" else stage
    return COMPARISON_TITLE.format(
        acceleration=acceleration_label(variant),
        case=case_label(variant, voxel_size_mm),
        branch=BRANCH_LABELS.get(branch, branch),
        stage=STAGE_LABELS.get(stage_key, stage_key),
    )


def format_caption(mask_label: str, nrmse: float, ssim: float, psnr_db: float) -> str:
    """Render one metric line, labelled by the mask it was scored over.

    Args:
        mask_label: Foreground the metrics were taken over, e.g. ``head``.
        nrmse: RMS-normalized error.
        ssim: 3D SSIM averaged over the mask.
        psnr_db: Peak signal-to-noise ratio in decibels.

    Returns:
        The caption line shown under the picture.
    """
    return (
        f"{mask_label:>5s} mask     NRMSE {nrmse:.4f}     "
        f"SSIM {ssim:.4f}     PSNR {psnr_db:.2f} dB"
    )


def brightened_png_bytes(tiff_path: Path) -> tuple[bytes, int, int]:
    """Turn a 16-bit strip into the brightened 8-bit PNG PowerPoint embeds.

    The PNG never reaches the filesystem: python-pptx accepts a file-like
    object, so the only copy is the one inside the deck.

    Args:
        tiff_path: 16-bit center-slice strip.

    Returns:
        The PNG bytes and the ``(width, height)`` in pixels.

    Raises:
        ValueError: If the strip is not 16-bit.
    """
    with Image.open(tiff_path) as image:
        data = np.asarray(image)
    if data.dtype != np.uint16:
        raise ValueError(f"Expected a 16-bit TIFF: {tiff_path} ({data.dtype})")

    eight_bit = np.rint(data.astype(np.float64) / 65535.0 * 255.0).astype(np.uint8)
    brightened = np.asarray(POWERPOINT_BRIGHT20_LUT, dtype=np.uint8)[eight_bit]

    buffer = io.BytesIO()
    Image.fromarray(brightened, mode="L").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), brightened.shape[1], brightened.shape[0]


class DeckBuilder:
    """Build a comparison deck on top of the styling template.

    Attributes:
        presentation: The deck under construction.
    """

    def __init__(self, template_path: Path) -> None:
        """Open the template and take its styling assets.

        Args:
            template_path: Template deck carrying the theme, the layouts and the
                x/y axis indicator group.

        Raises:
            ValueError: If the template lacks the axis indicator or a layout.
        """
        self.presentation = Presentation(str(template_path))
        self._axis_group = self._take_axis_group()
        layouts = self.presentation.slide_masters[0].slide_layouts
        self._picture_layout = layouts.get_by_name(PICTURE_LAYOUT)
        self._section_layout = layouts.get_by_name(SECTION_LAYOUT)
        if self._picture_layout is None or self._section_layout is None:
            raise ValueError(
                f"Template needs {PICTURE_LAYOUT!r} and {SECTION_LAYOUT!r} layouts."
            )
        self._drop_template_slides()

    def add_section(self, title: str) -> None:
        """Append a section header slide.

        Args:
            title: Section header title.
        """
        slide = self.presentation.slides.add_slide(self._section_layout)
        slide.shapes.title.text_frame.text = title

    def add_picture_slide(self, spec: SlideSpec) -> None:
        """Append one picture slide built to the deck pattern.

        Args:
            spec: What to put on the slide.
        """
        slide = self.presentation.slides.add_slide(self._picture_layout)
        self._drop_body_placeholders(slide)
        slide.shapes.title.text_frame.text = spec.title

        height = Emu(int(round(int(PICTURE_WIDTH) * spec.height_px / spec.width_px)))
        left = Emu(int((self.presentation.slide_width - int(PICTURE_WIDTH)) // 2))
        top = Emu(int((self.presentation.slide_height - int(height)) // 2))

        picture = slide.shapes.add_picture(
            io.BytesIO(spec.image), left, top, width=PICTURE_WIDTH, height=height
        )
        _record_brightness(picture, BRIGHTNESS)
        _send_to_back(slide, picture)

        if spec.with_axes and self._axis_group is not None:
            self._add_axis_group(slide)
        if spec.captions:
            _add_caption(slide, spec.captions, left)

    def add_figure_slide(self, title: str, figure_path: Path) -> None:
        """Append a slide holding one chart, fitted to the content area.

        Charts are ordinary figures: no brightness correction, no axis
        indicator and no caption, scaled to fit rather than fixed at the
        picture width used for anatomy strips.

        Args:
            title: Slide title.
            figure_path: Chart image on disk.
        """
        slide = self.presentation.slides.add_slide(self._picture_layout)
        self._drop_body_placeholders(slide)
        slide.shapes.title.text_frame.text = title

        with Image.open(figure_path) as image:
            width_px, height_px = image.size

        scale = min(
            int(FIGURE_MAX_WIDTH) / width_px, int(FIGURE_MAX_HEIGHT) / height_px
        )
        width = Emu(int(round(width_px * scale)))
        height = Emu(int(round(height_px * scale)))
        left = Emu(int((self.presentation.slide_width - int(width)) // 2))
        top = Emu(
            int(FIGURE_TOP)
            + max(0, (int(FIGURE_MAX_HEIGHT) - int(height)) // 2)
        )
        slide.shapes.add_picture(str(figure_path), left, top, width=width, height=height)

    def move_last_slides_to_front(self, count: int) -> None:
        """Move the most recently added slides to the start of the deck.

        Sections are built subject by subject, but the evaluation charts that
        summarise them can only be drawn once every section is scored. Building
        them last and moving them here keeps the deck opening on the summary.

        Args:
            count: How many trailing slides to move, order preserved.

        Raises:
            ValueError: If the deck holds fewer slides than requested.
        """
        id_list = self.presentation.slides._sldIdLst
        entries = list(id_list)
        if not 0 <= count <= len(entries):
            raise ValueError(f"Cannot move {count} of {len(entries)} slides.")
        for offset, entry in enumerate(entries[len(entries) - count :]):
            id_list.remove(entry)
            id_list.insert(offset, entry)

    def save(self, path: Path) -> None:
        """Write the deck.

        Args:
            path: Destination ``.pptx``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.presentation.save(str(path))

    def _take_axis_group(self):
        """Copy the x/y axis indicator group out of the template."""
        for slide in self.presentation.slides:
            for shape in slide.shapes:
                if (
                    shape.shape_type == MSO_SHAPE_TYPE.GROUP
                    and shape.name == AXIS_GROUP_NAME
                ):
                    return copy.deepcopy(shape._element)
        raise ValueError(
            f"Template has no {AXIS_GROUP_NAME!r} shape to copy onto LR slides."
        )

    def _drop_template_slides(self) -> None:
        """Remove the template's asset slides, keeping its styling behind."""
        id_list = self.presentation.slides._sldIdLst
        for entry in list(id_list):
            self.presentation.part.drop_rel(entry.get(qn("r:id")))
            id_list.remove(entry)

    def _add_axis_group(self, slide) -> None:
        """Place the axis indicator at the front, keeping its template offsets."""
        element = copy.deepcopy(self._axis_group)
        used = {
            int(node.get("id"))
            for node in slide.shapes._spTree.iter(qn("p:cNvPr"))
            if node.get("id")
        }
        next_id = max(used, default=1) + 1
        for node in element.iter(qn("p:cNvPr")):
            node.set("id", str(next_id))
            next_id += 1
        slide.shapes._spTree.append(element)

    @staticmethod
    def _drop_body_placeholders(slide) -> None:
        """Remove the layout's content placeholder, keeping only the title."""
        for shape in list(slide.placeholders):
            if shape.placeholder_format.idx == 0:
                continue
            shape._element.getparent().remove(shape._element)


def _record_brightness(picture, bright: int) -> None:
    """Write the brightness the way PowerPoint writes it.

    This drives the Format Picture pane only; the pixels themselves are
    corrected in :func:`tiff_to_embed_png`. It is written as an ``a:ext``
    extension, so a consumer that does not understand the 2010 image properties
    ignores it rather than failing.

    Args:
        picture: Picture shape to annotate.
        bright: Brightness in 1/1000 percent.
    """
    blip = picture._element.blipFill.find(qn("a:blip"))
    ext_list = blip.find(qn("a:extLst"))
    if ext_list is None:
        ext_list = etree.SubElement(blip, qn("a:extLst"))

    ext = etree.SubElement(ext_list, qn("a:ext"))
    ext.set("uri", IMG_PROPS_EXT_URI)
    properties = etree.SubElement(ext, f"{{{A14_NS}}}imgProps", nsmap={"a14": A14_NS})
    layer = etree.SubElement(properties, f"{{{A14_NS}}}imgLayer")
    effect = etree.SubElement(layer, f"{{{A14_NS}}}imgEffect")
    etree.SubElement(
        effect, f"{{{A14_NS}}}brightnessContrast", attrib={"bright": str(bright)}
    )


def _send_to_back(slide, shape) -> None:
    """Move a shape to the front of the shape tree, the bottom of the z-order."""
    sp_tree = slide.shapes._spTree
    element = shape._element
    sp_tree.remove(element)
    # Children 0 and 1 are nvGrpSpPr and grpSpPr; drawables start at index 2.
    sp_tree.insert(2, element)


def _add_caption(slide, lines: Sequence[str], left) -> None:
    """Add the centered metric lines in the band below the picture.

    Args:
        slide: Slide to caption.
        lines: One line per scored mask, in the order they were scored.
        left: Left offset shared with the picture.
    """
    box = slide.shapes.add_textbox(left, METRICS_TOP, PICTURE_WIDTH, METRICS_HEIGHT)
    box.name = METRICS_SHAPE_NAME

    frame = box.text_frame
    frame.word_wrap = False
    for index, text in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = text
        run.font.size = METRICS_FONT


def _trim(value: float) -> str:
    """Render a voxel size without a trailing ``.0``."""
    return f"{value:g}"
