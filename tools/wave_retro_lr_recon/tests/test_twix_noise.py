"""Synthetic tests for TWIX noise, channel-identity, and covariance checks."""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from wave_retro_lr import twix_noise  # noqa: E402
from wave_retro_lr.twix_noise import (  # noqa: E402
    DEFAULT_COVARIANCE_COMPATIBILITY_LIMITS,
    EVAL_INFO_BITS,
    LOOP_COUNTER_NAMES,
    RaidMeasurement,
    channel_identity_report,
    coil_select_tables,
    compare_coil_select,
    compare_covariances,
    correlation_statistics,
    covariance_compatibility,
    expected_white_noise_variance_ratio,
    fft_scale_factors,
    fft_scale_factors_by_block,
    load_measurement0_noise,
    load_measurement_headers,
    measurement_metadata,
    noise_covariance,
    noise_lines_from_mapvbvd,
    read_multiraid_table,
    read_multiraid_table_from_file,
    split_noise_lines,
    walk_measurement,
    walk_scan_headers,
)

MAPVBVD_AVAILABLE = importlib.util.find_spec("mapvbvd") is not None
BITS = {name: 1 << position for name, position in EVAL_INFO_BITS.items()}
CHANNEL_IDS = (0, 1, 2, 3)
NOISE_LINE_COUNT = 12
NOISE_SAMPLES = 16
ACQUISITION_SAMPLES = 24
RAW_CORRECTION_NOISE_LINES = (2, 5, 8)
SYNC_AFTER_NOISE_LINE = 4
ARRAY_COIL = "SyntheticArray"
BODY_COIL = "SyntheticBody"
ARRAY_ELEMENTS = ("A1", "A2", "A3", "A4")
BODY_ELEMENTS = ("B1", "B2")
NOISE_FFT_SCALE = (1.25, 1.5, 1.75, 2.0)
ACQUISITION_FFT_SCALE = (3.0, 3.25, 3.5, 3.75)
BODY_FFT_SCALE = (180.75, 172.5)
MEASUREMENT_TABLE = (
    (11111, 31, "synthetic_adjust"),
    (22222, 32, "synthetic_imaging"),
)
SYNTHETIC_PATIENT = b"SYNTHETIC^NOBODY"


def scan_bytes(
    mask: int,
    samples: int,
    channel_ids: Sequence[int],
    *,
    counters: Mapping[str, int] | None = None,
    data: np.ndarray | None = None,
    dma_adjust: int = 0,
) -> bytes:
    """Build one regular VD/VE scan with channel headers and samples.

    Args:
        mask: Eval-info mask word 1.
        samples: Complex samples per channel.
        channel_ids: MDH channel IDs in stored order.
        counters: Loop-counter values by name; absent counters are zero.
        data: Optional complex samples shaped ``(samples, channels)``.
        dma_adjust: Deliberate DMA-length error in bytes.

    Returns:
        192-byte scan header followed by one block per channel.
    """
    channel_count = len(channel_ids)
    size = 192 + channel_count * (32 + 8 * samples)
    header = bytearray(192)
    struct.pack_into("<I", header, 0, size + dma_adjust)
    struct.pack_into("<II", header, 40, mask, 0)
    struct.pack_into("<HH", header, 48, samples, channel_count)
    values = dict(counters or {})
    struct.pack_into(
        "<14H", header, 52, *(int(values.get(name, 0)) for name in LOOP_COUNTER_NAMES)
    )
    if data is None:
        data = np.zeros((samples, channel_count), dtype=np.complex64)
    payload = bytearray()
    for column, channel_id in enumerate(channel_ids):
        channel_header = bytearray(32)
        struct.pack_into("<I", channel_header, 0, 32 + 8 * samples)
        struct.pack_into("<H", channel_header, 24, channel_id)
        payload += channel_header
        payload += np.ascontiguousarray(data[:, column], dtype="<c8").tobytes()
    return bytes(header) + bytes(payload)


def sync_bytes(payload_bytes: int = 64, *, dma_length: int | None = None) -> bytes:
    """Build one SYNCDATA scan that must be skipped by its DMA length.

    Args:
        payload_bytes: Opaque synchronization payload size.
        dma_length: Optional explicit DMA length; defaults to header plus
            payload.

    Returns:
        SYNCDATA header and payload bytes.
    """
    header = bytearray(192)
    length = 192 + payload_bytes if dma_length is None else dma_length
    # Control bits above the 25-bit DMA length must be masked by readers.
    struct.pack_into("<I", header, 0, length | (1 << 26))
    struct.pack_into("<II", header, 40, BITS["SYNCDATA"], 0)
    # Misleading sample/channel fields prove that the DMA length is used.
    struct.pack_into("<HH", header, 48, 7, 3)
    return bytes(header) + b"\xab" * payload_bytes


def acqend_bytes() -> bytes:
    """Build one ACQEND scan header.

    Returns:
        192-byte ACQEND header.
    """
    header = bytearray(192)
    struct.pack_into("<I", header, 0, 192)
    struct.pack_into("<II", header, 40, BITS["ACQEND"], 0)
    struct.pack_into("<HH", header, 48, 16, 1)
    return bytes(header)


def measurement_header(buffers: Sequence[tuple[str, str]]) -> bytes:
    """Build a measurement header in the layout read by mapVBVD.

    Args:
        buffers: ``(name, text)`` header buffers.

    Returns:
        Header bytes whose leading uint32 is the total header length.
    """
    body = bytearray(struct.pack("<I", len(buffers)))
    for name, text in buffers:
        payload = text.encode("latin-1")
        body += name.encode("ascii") + b"\0" + struct.pack("<I", len(payload)) + payload
    unpadded = 4 + len(body)
    total = unpadded + (-unpadded) % 32
    return struct.pack("<I", total) + bytes(body) + bytes(total - unpadded)


def ascconv(lines: Sequence[str]) -> str:
    """Wrap assignment lines in an ASCCONV block.

    Args:
        lines: ``name = value`` assignments.

    Returns:
        ASCCONV text.
    """
    return "### ASCCONV BEGIN ###\n" + "\n".join(lines) + "\n### ASCCONV END ###\n"


def coil_lines(
    block: int, coil: str, elements: Sequence[str], *, rx_offset: int = 0
) -> list[str]:
    """Build ASCCONV coil-select assignments for one block.

    Args:
        block: ``aRxCoilSelectData`` block index.
        coil: Synthetic coil identifier.
        elements: Element names; list index ``i`` uses ADC channel ``i + 1``.
        rx_offset: Offset added to the receiver-channel numbers.

    Returns:
        ASCCONV assignment lines including the container attribute line.
    """
    prefix = f"sCoilSelectMeas.aRxCoilSelectData[{block}]"
    lines: list[str] = []
    for index, element in enumerate(elements):
        entry = f"{prefix}.asList[{index}]"
        lines += [
            f"{entry}.lADCChannelConnected = {index + 1}",
            f"{entry}.lRxChannelConnected = {index + 1 + rx_offset}",
            f'{entry}.sCoilElementID.tCoilID = "{coil}"',
            f'{entry}.sCoilElementID.tElement = "{element}"',
        ]
    lines.append(f"{prefix}.asList.__attribute__.size = {len(elements)}")
    return lines


def fft_scale_lines(block: int, factors: Sequence[float]) -> list[str]:
    """Build the FFT-scale assignments of one coil-select block.

    VD/VE headers store FFT-scale factors per ``aRxCoilSelectData`` block,
    with hexadecimal ``bValid`` flags, as in the measured pilot header.

    Args:
        block: ``aRxCoilSelectData`` block index.
        factors: FFT-scale factors in index order.

    Returns:
        ASCCONV assignment lines including the container attribute line.
    """
    prefix = f"sCoilSelectMeas.aRxCoilSelectData[{block}].aFFT_SCALE"
    lines: list[str] = []
    for index, factor in enumerate(factors):
        lines += [f"{prefix}[{index}].flFactor = {factor}", f"{prefix}[{index}].bValid = 0x1"]
    lines.append(f"{prefix}.__attribute__.size = {len(factors)}")
    return lines


def header_buffers(
    protocol: str, dwell_ns: int, fft_scale: Sequence[float], *, body_block: bool
) -> list[tuple[str, str]]:
    """Build synthetic ``Meas`` and ``MeasYaps`` header buffers.

    Args:
        protocol: Synthetic protocol name.
        dwell_ns: Dwell time in nanoseconds.
        fft_scale: FFT-scale factors to record.
        body_block: Whether to add a block-1 body-coil selection that reuses
            list indices 0/1 and ADC channels 1/2 and has its own FFT-scale
            factors, as in a measurement-0 adjustment scan.

    Returns:
        Header buffers for :func:`measurement_header`.
    """
    lines = [f'tProtocolName = "{protocol}"', f"sRXSPEC.alDwellTime[0] = {dwell_ns}"]
    lines += coil_lines(0, ARRAY_COIL, ARRAY_ELEMENTS)
    lines.append("sCoilSelectMeas.aRxCoilSelectData[0].ucNoiseDecorrMode = 0x2")
    lines += fft_scale_lines(0, fft_scale)
    if body_block:
        lines += coil_lines(1, BODY_COIL, BODY_ELEMENTS, rx_offset=len(ARRAY_ELEMENTS))
        lines += fft_scale_lines(1, BODY_FFT_SCALE)
    xprotocol = (
        "<XProtocol>\n{\n"
        '<ParamDouble."flReadoutOSFactor">  { <Precision> 6  2.000000  }\n'
        '<ParamDouble."dRawDataCorrectionFactorRe">  { }\n'
        '<ParamDouble."dRawDataCorrectionFactorIm">  { }\n'
        "}\n"
    )
    return [("Meas", xprotocol), ("MeasYaps", ascconv(lines))]


def measurement_bytes(
    buffers: Sequence[tuple[str, str]], scans: Sequence[bytes]
) -> tuple[bytes, int]:
    """Assemble one padded measurement.

    Args:
        buffers: Header buffers.
        scans: Scan byte blocks in acquisition order.

    Returns:
        Measurement bytes padded to 512 bytes and the header length.
    """
    header = measurement_header(buffers)
    body = header + b"".join(scans)
    return body + bytes((-len(body)) % 512), len(header)


def multiraid_bytes(
    measurements: Sequence[tuple[tuple[int, int, str], bytes]],
) -> tuple[bytes, list[tuple[int, int]]]:
    """Assemble a multi-raid file with a 10240-byte measurement table.

    Args:
        measurements: ``((measurement_id, file_id, protocol), payload)``.

    Returns:
        File bytes and ``(offset, length)`` per measurement.
    """
    header = bytearray(10240)
    struct.pack_into("<II", header, 0, 0, len(measurements))
    ranges: list[tuple[int, int]] = []
    offset = 10240
    for index, ((measurement_id, file_id, protocol), payload) in enumerate(measurements):
        start = 8 + 152 * index
        struct.pack_into(
            "<IIQQ", header, start, measurement_id, file_id, offset, len(payload)
        )
        header[start + 24 : start + 24 + len(SYNTHETIC_PATIENT)] = SYNTHETIC_PATIENT
        name = protocol.encode("latin-1")
        header[start + 88 : start + 88 + len(name)] = name
        ranges.append((offset, len(payload)))
        offset += len(payload)
    return bytes(header) + b"".join(payload for _, payload in measurements), ranges


def build_synthetic_twix(
    directory: Path,
    *,
    include_noise_measurement: bool = True,
    refscan_channel_ids: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Write a small synthetic multi-raid TWIX file.

    Measurement 0 holds 12 noise-adjustment lines (three flagged for raw-data
    correction), one SYNCDATA block after noise line 4, and ACQEND.
    Measurement 1 holds a 4 x 3 image grid (one scan with a wrong DMA length),
    two refscan lines for each of sets 0..3, the full 4 x 4 set-4 LIN/PAR
    grid once each, and ACQEND.

    Args:
        directory: Temporary directory receiving the file.
        include_noise_measurement: Whether to write measurement 0.
        refscan_channel_ids: Optional channel-ID override per refscan set.

    Returns:
        File path, written noise samples, measurement byte ranges, header
        lengths, and ACQEND offsets.
    """
    rng = np.random.default_rng(20260929)
    shape = (NOISE_LINE_COUNT, NOISE_SAMPLES, len(CHANNEL_IDS))
    noise = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )
    noise_scans: list[bytes] = []
    for line in range(NOISE_LINE_COUNT):
        mask = BITS["NOISEADJSCAN"]
        if line in RAW_CORRECTION_NOISE_LINES:
            mask |= BITS["RAWDATACORRECTION"]
        noise_scans.append(
            scan_bytes(mask, NOISE_SAMPLES, CHANNEL_IDS, counters={"Lin": line}, data=noise[line])
        )
        if line == SYNC_AFTER_NOISE_LINE:
            noise_scans.append(sync_bytes())
    noise_scans.append(acqend_bytes())

    def acquisition_data() -> np.ndarray:
        """Draw deterministic complex acquisition samples.

        Returns:
            Complex64 samples shaped ``(samples, channels)``.
        """
        values = rng.standard_normal((ACQUISITION_SAMPLES, len(CHANNEL_IDS), 2))
        return (values[..., 0] + 1j * values[..., 1]).astype(np.complex64)

    overrides = dict(refscan_channel_ids or {})
    acquisition_scans: list[bytes] = []
    for par in range(3):
        for lin in range(4):
            acquisition_scans.append(
                scan_bytes(
                    BITS["ONLINE"],
                    ACQUISITION_SAMPLES,
                    CHANNEL_IDS,
                    counters={"Lin": lin, "Par": par},
                    data=acquisition_data(),
                    dma_adjust=8 if (lin, par) == (1, 1) else 0,
                )
            )
    for set_index in range(4):
        for lin in range(2):
            acquisition_scans.append(
                scan_bytes(
                    BITS["PATREFSCAN"],
                    ACQUISITION_SAMPLES,
                    overrides.get(set_index, CHANNEL_IDS),
                    counters={"Lin": 10 + lin, "Par": 5, "Set": set_index},
                    data=acquisition_data(),
                )
            )
    for lin in range(4):
        for par in range(4):
            acquisition_scans.append(
                scan_bytes(
                    BITS["PATREFSCAN"],
                    ACQUISITION_SAMPLES,
                    overrides.get(4, CHANNEL_IDS),
                    counters={"Lin": lin, "Par": par, "Set": 4},
                    data=acquisition_data(),
                )
            )
    acquisition_scans.append(acqend_bytes())

    noise_payload, noise_header = measurement_bytes(
        header_buffers("synthetic_adjust", 4000, NOISE_FFT_SCALE, body_block=True),
        noise_scans,
    )
    acquisition_payload, acquisition_header = measurement_bytes(
        header_buffers("synthetic_imaging", 5000, ACQUISITION_FFT_SCALE, body_block=False),
        acquisition_scans,
    )
    entries = [(MEASUREMENT_TABLE[1], acquisition_payload)]
    header_lengths = [acquisition_header]
    scan_lengths = [sum(len(scan) for scan in acquisition_scans[:-1])]
    if include_noise_measurement:
        entries.insert(0, (MEASUREMENT_TABLE[0], noise_payload))
        header_lengths.insert(0, noise_header)
        scan_lengths.insert(0, sum(len(scan) for scan in noise_scans[:-1]))
    content, ranges = multiraid_bytes(entries)
    path = directory / "synthetic_multiraid.dat"
    path.write_bytes(content)
    return {
        "path": path,
        "noise": noise,
        "ranges": ranges,
        "header_lengths": header_lengths,
        "acqend_offsets": [
            offset + header + scans
            for (offset, _), header, scans in zip(ranges, header_lengths, scan_lengths)
        ],
    }


def kms_covariance(
    correlation: float = 0.6, phase: float = 0.3, channels: int = 8
) -> np.ndarray:
    """Build a positive-definite complex covariance with structured correlation.

    Args:
        correlation: Magnitude of neighbouring-channel correlation.
        phase: Phase increment per channel offset in radians.
        channels: Channel count.

    Returns:
        Covariance with variances increasing from 0.5 to 2.0.
    """
    index = np.arange(channels)
    offset = index[:, None] - index[None, :]
    rho = correlation ** np.abs(offset) * np.exp(1j * phase * offset)
    deviation = np.sqrt(np.linspace(0.5, 2.0, channels))
    return deviation[:, None] * rho * deviation[None, :]


def coil_yaps(
    blocks: Sequence[tuple[int, Sequence[tuple[int, str, str]]]], *, style: str = "tuple"
) -> dict[Any, Any]:
    """Build a coil-select ``MeasYaps`` mapping in one key style.

    Args:
        blocks: ``(block, [(adc, coil, element), ...])`` in insertion order.
        style: ``tuple``, ``dotted``, or ``bracket`` key style.

    Returns:
        Mapping with quoted strings, float numbers, and attribute keys.
    """
    yaps: dict[Any, Any] = {}
    for block, entries in blocks:
        prefix = ("sCoilSelectMeas", "aRxCoilSelectData", str(block), "asList")
        for index, (adc, coil, element) in enumerate(entries):
            fields = {
                ("lADCChannelConnected",): float(adc),
                ("lRxChannelConnected",): float(adc + 10 * block),
                ("sCoilElementID", "tCoilID"): f'"{coil}"',
                ("sCoilElementID", "tElement"): f'"{element}"',
            }
            for field_parts, value in fields.items():
                yaps[styled_key((*prefix, str(index), *field_parts), style)] = value
        yaps[styled_key((*prefix, "__attribute__", "size"), style)] = float(len(entries))
    return yaps


def styled_key(parts: tuple[str, ...], style: str) -> Any:
    """Render key components as a tuple, dotted, or bracketed key.

    Args:
        parts: Key components.
        style: ``tuple``, ``dotted``, or ``bracket``.

    Returns:
        Header key in the requested style.
    """
    if style == "tuple":
        return parts
    if style == "dotted":
        return ".".join(parts)
    text = ""
    for part in parts:
        text += f"[{part}]" if part.isdigit() else (f".{part}" if text else part)
    return text


def array_entries(elements: Sequence[str] = ARRAY_ELEMENTS) -> list[tuple[int, str, str]]:
    """Return receive-array entries with ADC channels ``1..N``.

    Args:
        elements: Element names in list order.

    Returns:
        ``(adc, coil, element)`` entries.
    """
    return [(index + 1, ARRAY_COIL, element) for index, element in enumerate(elements)]


BODY_ENTRIES = [(1, BODY_COIL, "B1"), (2, BODY_COIL, "B2")]


class MultiRaidTableTests(unittest.TestCase):
    """Verify multi-raid measurement-table parsing."""

    def test_table_entries_match_the_written_layout(self) -> None:
        """Parse counts, offsets, lengths, identifiers, and protocol names.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_synthetic_twix(Path(temporary))
            table = read_multiraid_table_from_file(fixture["path"])
            header = fixture["path"].read_bytes()[:10240]
        self.assertEqual(len(table), 2)
        self.assertEqual(read_multiraid_table(header), table)
        for entry, (offset, length), (measurement_id, file_id, protocol) in zip(
            table, fixture["ranges"], MEASUREMENT_TABLE
        ):
            self.assertIsInstance(entry, RaidMeasurement)
            self.assertEqual((entry.offset, entry.length), (offset, length))
            self.assertEqual(entry.measurement_id, measurement_id)
            self.assertEqual(entry.file_id, file_id)
            self.assertEqual(entry.protocol_name, protocol)
        self.assertEqual([entry.index for entry in table], [0, 1])
        records = json.dumps([entry.to_json() for entry in table])
        self.assertNotIn(SYNTHETIC_PATIENT.decode("latin-1"), records)
        self.assertEqual(
            set(table[0].to_json()),
            {"index", "measurement_id", "file_id", "offset", "length", "protocol_name"},
        )

    def test_invalid_tables_are_rejected(self) -> None:
        """Reject VB-like preambles, bad counts, short buffers, and overlaps.

        Returns:
            None.
        """
        valid = bytearray(10240)
        struct.pack_into("<II", valid, 0, 0, 2)
        struct.pack_into("<IIQQ", valid, 8, 1, 1, 10240, 1024)
        struct.pack_into("<IIQQ", valid, 160, 2, 2, 11264, 1024)
        self.assertEqual(len(read_multiraid_table(bytes(valid))), 2)
        cases = []
        vb_like = bytearray(valid)
        struct.pack_into("<I", vb_like, 0, 12000)
        cases.append(bytes(vb_like))
        for count in (0, 65):
            bad_count = bytearray(valid)
            struct.pack_into("<I", bad_count, 4, count)
            cases.append(bytes(bad_count))
        cases.append(bytes(valid[:100]))
        overlap = bytearray(valid)
        struct.pack_into("<Q", overlap, 160 + 8, 10240 + 512)
        cases.append(bytes(overlap))
        inside_table = bytearray(valid)
        struct.pack_into("<Q", inside_table, 8 + 8, 64)
        cases.append(bytes(inside_table))
        for case in cases:
            with self.assertRaises(ValueError):
                read_multiraid_table(case)
        with self.assertRaises(TypeError):
            read_multiraid_table([0, 1, 2])  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as temporary:
            truncated = Path(temporary) / "truncated.dat"
            truncated.write_bytes(bytes(valid) + bytes(1024))
            with self.assertRaises(ValueError):
                read_multiraid_table_from_file(truncated)


class ScanWalkTests(unittest.TestCase):
    """Verify MDH scan walking, role classification, and termination."""

    def test_measurement_walks_classify_roles_and_counters(self) -> None:
        """Summarize noise, image, and refscan roles of both measurements.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_synthetic_twix(Path(temporary))
            table = read_multiraid_table_from_file(fixture["path"])
            noise_walk = walk_measurement(fixture["path"], table[0])
            acquisition_walk = walk_measurement(fixture["path"], table[1])

        self.assertEqual(noise_walk["scans"], NOISE_LINE_COUNT)
        self.assertTrue(noise_walk["acqend_found"])
        self.assertEqual(noise_walk["termination"], "acqend")
        self.assertEqual(noise_walk["stop_offset"], fixture["acqend_offsets"][0])
        self.assertEqual(noise_walk["measurement_header_bytes"], fixture["header_lengths"][0])
        self.assertEqual(noise_walk["measurement"], table[0].to_json())
        self.assertEqual(noise_walk["dma_length_mismatches"], 0)
        expected_noise_flags = {name: 0 for name in EVAL_INFO_BITS}
        expected_noise_flags.update(
            NOISEADJSCAN=NOISE_LINE_COUNT,
            RAWDATACORRECTION=len(RAW_CORRECTION_NOISE_LINES),
            SYNCDATA=1,
            ACQEND=1,
        )
        self.assertEqual(noise_walk["flag_counts"], expected_noise_flags)
        self.assertEqual(
            noise_walk["roles"],
            {
                "noise": {
                    "lines": NOISE_LINE_COUNT,
                    "channel_id_sequences": [
                        {
                            "samples": NOISE_SAMPLES,
                            "channel_ids": list(CHANNEL_IDS),
                            "lines": NOISE_LINE_COUNT,
                        }
                    ],
                    "raw_data_correction_lines": len(RAW_CORRECTION_NOISE_LINES),
                    "lin_range": [0, NOISE_LINE_COUNT - 1],
                    "par_range": [0, 0],
                    "unique_lin_par_pairs": NOISE_LINE_COUNT,
                }
            },
        )

        self.assertEqual(acquisition_walk["scans"], 12 + 8 + 16)
        self.assertTrue(acquisition_walk["acqend_found"])
        self.assertEqual(acquisition_walk["stop_offset"], fixture["acqend_offsets"][1])
        self.assertEqual(acquisition_walk["dma_length_mismatches"], 1)
        self.assertEqual(acquisition_walk["flag_counts"]["ONLINE"], 12)
        self.assertEqual(acquisition_walk["flag_counts"]["PATREFSCAN"], 24)
        self.assertEqual(acquisition_walk["flag_counts"]["RAWDATACORRECTION"], 0)
        self.assertEqual(acquisition_walk["flag_counts"]["SYNCDATA"], 0)
        roles = acquisition_walk["roles"]
        self.assertEqual(
            list(roles),
            ["image", *(f"refscan_set{index}" for index in range(5))],
        )
        self.assertEqual(roles["image"]["lines"], 12)
        self.assertEqual(roles["image"]["lin_range"], [0, 3])
        self.assertEqual(roles["image"]["par_range"], [0, 2])
        self.assertEqual(roles["image"]["unique_lin_par_pairs"], 12)
        for set_index in range(4):
            role = roles[f"refscan_set{set_index}"]
            self.assertEqual(role["lines"], 2)
            self.assertEqual(role["lin_range"], [10, 11])
            self.assertEqual(role["par_range"], [5, 5])
        set4 = roles["refscan_set4"]
        self.assertEqual(set4["lines"], 16)
        self.assertEqual(set4["lin_range"], [0, 3])
        self.assertEqual(set4["par_range"], [0, 3])
        self.assertEqual(set4["unique_lin_par_pairs"], 16)
        for role in roles.values():
            self.assertEqual(role["raw_data_correction_lines"], 0)
            self.assertEqual(
                role["channel_id_sequences"],
                [
                    {
                        "samples": ACQUISITION_SAMPLES,
                        "channel_ids": list(CHANNEL_IDS),
                        "lines": role["lines"],
                    }
                ],
            )
        json.dumps(noise_walk)
        json.dumps(acquisition_walk)

    def test_all_buffer_types_give_identical_walks(self) -> None:
        """Accept bytes, bytearray, memoryview, uint8 arrays, and memory maps.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_synthetic_twix(Path(temporary))
            content = fixture["path"].read_bytes()
            offset, length = fixture["ranges"][0]
            start = offset + fixture["header_lengths"][0]
            end = offset + length
            reference = walk_scan_headers(content, start, end)
            mapped = np.memmap(fixture["path"], dtype=np.uint8, mode="r")
            candidates = [
                bytearray(content),
                memoryview(content),
                np.frombuffer(content, dtype=np.uint8),
                mapped,
            ]
            for candidate in candidates:
                self.assertEqual(walk_scan_headers(candidate, start, end), reference)
            del mapped, candidates
        self.assertEqual(reference["roles"]["noise"]["lines"], NOISE_LINE_COUNT)

    def test_role_precedence_follows_the_documented_order(self) -> None:
        """Classify noise, refscan sets, phase correction, feedback, and images.

        Returns:
            None.
        """
        scans = [
            scan_bytes(BITS["NOISEADJSCAN"] | BITS["PATREFSCAN"], 2, (0,)),
            scan_bytes(BITS["PATREFANDIMASCAN"], 2, (0,), counters={"Set": 2}),
            scan_bytes(BITS["PATREFSCAN"] | BITS["PHASCOR"], 2, (0,), counters={"Set": 3}),
            scan_bytes(BITS["PHASCOR"] | BITS["RTFEEDBACK"], 2, (0,)),
            scan_bytes(BITS["RTFEEDBACK"], 2, (0,)),
            scan_bytes(BITS["HPFEEDBACK"], 2, (0,)),
            scan_bytes(BITS["ONLINE"], 2, (0,)),
            scan_bytes(BITS["PHASESTABSCAN"], 2, (0,)),
        ]
        buffer = b"".join(scans)
        walk = walk_scan_headers(buffer, 0, len(buffer))
        self.assertEqual(
            {role: summary["lines"] for role, summary in walk["roles"].items()},
            {
                "noise": 1,
                "refscan_set2": 1,
                "refscan_set3": 1,
                "phasecor": 1,
                "feedback": 2,
                "image": 2,
            },
        )
        self.assertFalse(walk["acqend_found"])
        self.assertEqual(walk["termination"], "measurement_end")

    def test_walk_stops_safely_at_every_boundary(self) -> None:
        """Report termination reasons without reading beyond the range.

        Returns:
            None.
        """
        scan = scan_bytes(BITS["ONLINE"], 4, (0, 1))
        cases = {
            "zero_dma_length": bytes(192) + scan,
            "incomplete_scan_header": scan + bytes(100),
            "scan_beyond_end": scan + scan[:-8],
            "sync_data_beyond_end": scan + sync_bytes(64)[:-1],
            "invalid_sync_data_length": scan + sync_bytes(64, dma_length=100),
            "measurement_end": scan + scan,
        }
        expected_scans = {
            "zero_dma_length": 0,
            "incomplete_scan_header": 1,
            "scan_beyond_end": 1,
            "sync_data_beyond_end": 1,
            "invalid_sync_data_length": 1,
            "measurement_end": 2,
        }
        for termination, buffer in cases.items():
            walk = walk_scan_headers(buffer, 0, len(buffer))
            self.assertEqual(walk["termination"], termination)
            self.assertEqual(walk["scans"], expected_scans[termination])
            self.assertEqual(walk["flag_counts"]["ONLINE"], expected_scans[termination])
            self.assertEqual(walk["flag_counts"]["SYNCDATA"], 0)
            self.assertFalse(walk["acqend_found"])
        bounded = walk_scan_headers(scan + scan, 0, len(scan))
        self.assertEqual((bounded["scans"], bounded["stop_offset"]), (1, len(scan)))
        shifted = walk_scan_headers(bytes(7) + scan, 7, 7 + len(scan))
        self.assertEqual(shifted["roles"]["image"]["channel_id_sequences"][0]["channel_ids"], [0, 1])
        acqend = walk_scan_headers(scan + acqend_bytes() + scan, 0, 2 * len(scan) + 192)
        self.assertEqual((acqend["scans"], acqend["termination"]), (1, "acqend"))

    def test_invalid_walk_arguments_are_rejected(self) -> None:
        """Reject invalid ranges, offsets, and buffer layouts.

        Returns:
            None.
        """
        buffer = scan_bytes(BITS["ONLINE"], 2, (0,))
        for start, end in ((10, 5), (0, len(buffer) + 1), (-1, 5)):
            with self.assertRaises(ValueError):
                walk_scan_headers(buffer, start, end)
        with self.assertRaises(TypeError):
            walk_scan_headers(buffer, True, len(buffer))
        with self.assertRaises(TypeError):
            walk_scan_headers(list(buffer), 0, len(buffer))
        for array in (
            np.zeros(4, dtype=np.uint16),
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros(8, dtype=np.uint8)[::2],
        ):
            with self.assertRaises(ValueError):
                walk_scan_headers(array, 0, 1)


class ChannelIdentityTests(unittest.TestCase):
    """Verify exact MDH channel-ID sequence comparisons."""

    def _walks(
        self, refscan_channel_ids: Mapping[int, Sequence[int]] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Walk both measurements of one synthetic file.

        Args:
            refscan_channel_ids: Optional channel-ID override per refscan set.

        Returns:
            Noise and acquisition walk summaries.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_synthetic_twix(
                Path(temporary), refscan_channel_ids=refscan_channel_ids
            )
            table = read_multiraid_table_from_file(fixture["path"])
            return (
                walk_measurement(fixture["path"], table[0]),
                walk_measurement(fixture["path"], table[1]),
            )

    def test_identical_sequential_channel_ids_pass(self) -> None:
        """Accept one identical sequential sequence across every role.

        Returns:
            None.
        """
        report = channel_identity_report(*self._walks())
        self.assertTrue(report["identical"])
        self.assertTrue(report["sequential"])
        self.assertEqual(report["reference_channel_ids"], list(CHANNEL_IDS))
        self.assertEqual(report["channel_count"], len(CHANNEL_IDS))
        self.assertTrue(report["required_roles_present"])
        self.assertEqual(report["missing_roles"], [])
        self.assertEqual(report["differences"], [])
        self.assertEqual(
            sorted(report["role_sequences"]),
            sorted(["noise", "image", *(f"refscan_set{index}" for index in range(5))]),
        )
        json.dumps(report)

    def test_permuted_refscan_set_is_listed_as_a_difference(self) -> None:
        """Detect a channel permutation confined to one refscan set.

        Returns:
            None.
        """
        noise_walk, acquisition_walk = self._walks({2: (0, 2, 1, 3)})
        self.assertEqual(
            acquisition_walk["roles"]["refscan_set2"]["channel_id_sequences"][0]["channel_ids"],
            [0, 2, 1, 3],
        )
        report = channel_identity_report(noise_walk, acquisition_walk)
        self.assertFalse(report["identical"])
        self.assertTrue(report["sequential"])
        self.assertEqual(report["role_sequences"]["refscan_set2"], [[0, 2, 1, 3]])
        self.assertEqual(len(report["differences"]), 1)
        self.assertIn("refscan_set2", report["differences"][0])
        self.assertIn("position 1", report["differences"][0])

    def test_missing_required_roles_are_reported(self) -> None:
        """Report required roles absent from both walks.

        Returns:
            None.
        """
        noise_walk, acquisition_walk = self._walks()
        without_image = copy.deepcopy(acquisition_walk)
        del without_image["roles"]["image"]
        report = channel_identity_report(noise_walk, without_image)
        self.assertFalse(report["required_roles_present"])
        self.assertEqual(report["missing_roles"], ["image"])
        self.assertTrue(report["identical"])
        extra = channel_identity_report(
            noise_walk,
            acquisition_walk,
            required_roles=("noise", "refscan_set4", "refscan_set9"),
        )
        self.assertEqual(extra["missing_roles"], ["refscan_set9"])
        with self.assertRaises(TypeError):
            channel_identity_report(noise_walk, acquisition_walk, required_roles="noise")
        with self.assertRaises(ValueError):
            channel_identity_report({}, acquisition_walk)

    def test_identical_non_sequential_ids_are_flagged_as_not_sequential(self) -> None:
        """Separate exact identity from sequential numbering.

        Returns:
            None.
        """
        ids = (4, 5, 6, 7)
        noise = scan_bytes(BITS["NOISEADJSCAN"], 2, ids)
        image = scan_bytes(BITS["ONLINE"], 3, ids)
        noise_walk = walk_scan_headers(noise, 0, len(noise))
        image_walk = walk_scan_headers(image, 0, len(image))
        report = channel_identity_report(noise_walk, image_walk, required_roles=("noise", "image"))
        self.assertTrue(report["identical"])
        self.assertFalse(report["sequential"])
        self.assertEqual(report["reference_channel_ids"], list(ids))
        empty = channel_identity_report({"roles": {}}, {"roles": {}}, required_roles=())
        self.assertFalse(empty["identical"])
        self.assertIsNone(empty["reference_channel_ids"])
        self.assertEqual(len(empty["differences"]), 1)


class CoilSelectTests(unittest.TestCase):
    """Verify block-aware header coil-selection comparisons."""

    def test_blocks_never_merge_into_the_array_block(self) -> None:
        """Keep reused body-coil ADC numbers out of block 0.

        Returns:
            None.
        """
        acquisition = coil_yaps([(0, array_entries())])
        for noise_blocks in (
            [(0, array_entries()), (1, BODY_ENTRIES)],
            [(1, BODY_ENTRIES), (0, array_entries())],
        ):
            noise = coil_yaps(noise_blocks)
            tables = coil_select_tables(noise)
            self.assertEqual(
                tables[0],
                {index + 1: f"{ARRAY_COIL}:{name}" for index, name in enumerate(ARRAY_ELEMENTS)},
            )
            self.assertEqual(tables[1], {1: f"{BODY_COIL}:B1", 2: f"{BODY_COIL}:B2"})
            report = compare_coil_select(noise, acquisition)
            self.assertTrue(report["identical"])
            self.assertEqual(report["block"], 0)
            self.assertEqual(report["differences"], [])
            self.assertEqual(report["noise_block_sizes"], {0: 4, 1: 2})
            self.assertEqual(report["acquisition_block_sizes"], {0: 4})
            self.assertTrue(report["noise_adc_channels_contiguous"])
            self.assertTrue(report["acquisition_adc_channels_contiguous"])
            # A block-merging comparison reproduces the historical false mismatch.
            merged: dict[int, str] = {}
            for table in tables.values():
                merged.update(table)
            self.assertNotEqual(merged, coil_select_tables(acquisition)[0])
        with self.assertRaises(ValueError):
            compare_coil_select(coil_yaps([(0, array_entries()), (1, BODY_ENTRIES)]), acquisition, block=1)

    def test_permuted_array_elements_are_detected(self) -> None:
        """List ADC channels whose block-0 elements differ.

        Returns:
            None.
        """
        noise = coil_yaps([(0, array_entries()), (1, BODY_ENTRIES)])
        permuted = coil_yaps([(0, array_entries(("A2", "A1", "A3", "A4")))])
        report = compare_coil_select(noise, permuted)
        self.assertFalse(report["identical"])
        self.assertEqual(
            report["differences"],
            [
                [1, f"{ARRAY_COIL}:A1", f"{ARRAY_COIL}:A2"],
                [2, f"{ARRAY_COIL}:A2", f"{ARRAY_COIL}:A1"],
            ],
        )
        missing = coil_yaps([(0, array_entries()[:3])])
        report = compare_coil_select(noise, missing)
        self.assertEqual(report["differences"], [[4, f"{ARRAY_COIL}:A4", None]])
        self.assertTrue(report["acquisition_adc_channels_contiguous"])
        gap = coil_yaps([(0, [(1, ARRAY_COIL, "A1"), (2, ARRAY_COIL, "A2"), (4, ARRAY_COIL, "A4")])])
        self.assertFalse(compare_coil_select(noise, gap)["acquisition_adc_channels_contiguous"])

    def test_key_styles_agree_and_attribute_keys_are_ignored(self) -> None:
        """Parse tuple, dotted, and bracketed keys; ignore container metadata.

        Returns:
            None.
        """
        blocks = [(0, array_entries()), (1, BODY_ENTRIES)]
        expected = coil_select_tables(coil_yaps(blocks, style="tuple"))
        for style in ("dotted", "bracket"):
            self.assertEqual(coil_select_tables(coil_yaps(blocks, style=style)), expected)
        mixed = coil_yaps([(0, array_entries())], style="tuple")
        mixed.update(coil_yaps([(1, BODY_ENTRIES)], style="dotted"))
        self.assertEqual(coil_select_tables(mixed), expected)
        decorated = coil_yaps(blocks)
        decorated[
            ("sCoilSelectMeas", "aRxCoilSelectData", "2", "asList", "0", "__attribute__", "size")
        ] = 1.0
        decorated["sCoilSelectMeas.aRxCoilSelectData.3.asList.0.__attribute__.size"] = 1.0
        decorated[("sCoilSelectMeas", "aRxCoilSelectData", "x", "asList", "0", "lADCChannelConnected")] = 9.0
        decorated["sCoilSelectMeas.aRxCoilSelectData.0.asList.y.lADCChannelConnected"] = 9.0
        decorated[("sCoilSelectMeas", "aRxCoilSelectData", "0", "ucNoiseDecorrMode")] = "0x2"
        self.assertEqual(coil_select_tables(decorated), expected)

    def test_invalid_coil_tables_are_rejected(self) -> None:
        """Reject absent ADC channels, duplicates, missing elements, and blocks.

        Returns:
            None.
        """
        base = ("sCoilSelectMeas", "aRxCoilSelectData", "0", "asList")
        without_adc = coil_yaps([(0, array_entries())])
        del without_adc[(*base, "2", "lADCChannelConnected")]
        duplicate = coil_yaps([(0, [(1, ARRAY_COIL, "A1"), (1, ARRAY_COIL, "A2")])])
        without_element = coil_yaps([(0, array_entries())])
        del without_element[(*base, "1", "sCoilElementID", "tElement")]
        fractional = coil_yaps([(0, array_entries())])
        fractional[(*base, "0", "lADCChannelConnected")] = 1.5
        for yaps in (without_adc, duplicate, without_element, fractional):
            with self.assertRaises(ValueError):
                coil_select_tables(yaps)
        valid = coil_yaps([(0, array_entries())])
        with self.assertRaises(TypeError):
            compare_coil_select(valid, valid, block=True)
        with self.assertRaises(ValueError):
            compare_coil_select(valid, valid, block=-1)
        with self.assertRaises(TypeError):
            coil_select_tables([("key", 1)])  # type: ignore[arg-type]


class MetadataTests(unittest.TestCase):
    """Verify recorded, never-applied scaling metadata."""

    def test_metadata_is_recorded_without_applying_scaling(self) -> None:
        """Record dwell, oversampling, decorrelation, and scale factors.

        Returns:
            None.
        """
        yaps = {
            ("tProtocolName",): '"synthetic_protocol"',
            ("sRXSPEC", "alDwellTime", "0"): 4000.0,
            ("sCoilSelectMeas", "aRxCoilSelectData", "0", "ucNoiseDecorrMode"): "0x2",
            "sCoilSelectMeas.aFFT_SCALE.2.flFactor": 1.75,
            "sCoilSelectMeas.aFFT_SCALE.2.bValid": "true",
            ("sCoilSelectMeas", "aFFT_SCALE", "0", "flFactor"): 1.25,
            ("sCoilSelectMeas", "aFFT_SCALE", "0", "bValid"): 1.0,
            ("sCoilSelectMeas", "aFFT_SCALE", "1", "flFactor"): 1.5,
            ("sCoilSelectMeas", "aFFT_SCALE", "__attribute__", "size"): 3.0,
        }
        meas = {
            "flReadoutOSFactor": 2.0,
            "dRawDataCorrectionFactorRe": "",
            "dRawDataCorrectionFactorIm": "",
        }
        metadata = measurement_metadata(yaps, meas)
        self.assertEqual(
            metadata,
            {
                "protocol_name": "synthetic_protocol",
                "dwell_ns": 4000.0,
                "readout_oversampling_factor": 2.0,
                "noise_decorrelation_mode": 2,
                "raw_data_correction_factors_present": False,
                "raw_data_correction_factors": {
                    "dRawDataCorrectionFactorRe": [],
                    "dRawDataCorrectionFactorIm": [],
                },
                "fft_scale_factors": [1.25, 1.5, 1.75],
                "fft_scale_valid_flags": [True, None, True],
                "fft_scale_block": None,
                "fft_scale_by_block": {},
                "fft_scale_applied": False,
                "raw_data_correction_applied": False,
            },
        )
        json.dumps(metadata)
        corrected = measurement_metadata(
            yaps,
            {
                "dRawDataCorrectionFactorRe": "1.0 0.98",
                "dRawDataCorrectionFactorIm": "0.0 0.01",
            },
        )
        self.assertTrue(corrected["raw_data_correction_factors_present"])
        self.assertEqual(
            corrected["raw_data_correction_factors"],
            {"dRawDataCorrectionFactorRe": [1.0, 0.98], "dRawDataCorrectionFactorIm": [0.0, 0.01]},
        )
        self.assertFalse(corrected["raw_data_correction_applied"])
        self.assertIsNone(corrected["readout_oversampling_factor"])
        without_meas = measurement_metadata(yaps)
        self.assertIsNone(without_meas["readout_oversampling_factor"])
        self.assertFalse(without_meas["raw_data_correction_factors_present"])
        self.assertEqual(
            without_meas["raw_data_correction_factors"],
            {"dRawDataCorrectionFactorRe": None, "dRawDataCorrectionFactorIm": None},
        )
        empty = measurement_metadata({})
        self.assertIsNone(empty["protocol_name"])
        self.assertIsNone(empty["dwell_ns"])
        self.assertIsNone(empty["noise_decorrelation_mode"])
        self.assertEqual(empty["fft_scale_factors"], [])
        self.assertIsNone(empty["fft_scale_block"])
        self.assertEqual(empty["fft_scale_by_block"], {})

    def test_fft_scale_factor_validation(self) -> None:
        """Order factors by index and reject ambiguous or malformed entries.

        Returns:
            None.
        """
        ordered = {
            ("sCoilSelectMeas", "aFFT_SCALE", "1", "flFactor"): 2.5,
            ("sCoilSelectMeas", "aFFT_SCALE", "0", "flFactor"): "1.5",
        }
        self.assertEqual(fft_scale_factors(ordered), [1.5, 2.5])
        gap = {("sCoilSelectMeas", "aFFT_SCALE", "0", "flFactor"): 1.0,
               ("sCoilSelectMeas", "aFFT_SCALE", "2", "flFactor"): 1.0}
        prefixes = {("sCoilSelectMeas", "aFFT_SCALE", "0", "flFactor"): 1.0,
                    ("sOther", "aFFT_SCALE", "1", "flFactor"): 1.0}
        nonfinite = {("sCoilSelectMeas", "aFFT_SCALE", "0", "flFactor"): "nan"}
        text = {("sCoilSelectMeas", "aFFT_SCALE", "0", "flFactor"): "large"}
        for yaps in (gap, prefixes, nonfinite, text):
            with self.assertRaises(ValueError):
                fft_scale_factors(yaps)
        with self.assertRaises(ValueError):
            measurement_metadata({("sRXSPEC", "alDwellTime", "0"): "slow"})

    def test_fft_scale_is_read_per_coil_select_block(self) -> None:
        """Keep each coil-select block's FFT-scale factors separate.

        The measured adjustment scan stores block-0 receive-array factors and
        block-1 body-coil factors, with hexadecimal ``bValid`` flags. Block 0
        is reported, every block is recorded, and ambiguous or malformed
        layouts are still refused.

        Returns:
            None.
        """
        array = ("sCoilSelectMeas", "aRxCoilSelectData", "0", "aFFT_SCALE")
        body = ("sCoilSelectMeas", "aRxCoilSelectData", "1", "aFFT_SCALE")
        yaps = {
            array + ("0", "flFactor"): 4.5,
            array + ("0", "bValid"): "0x1",
            array + ("1", "flFactor"): "4.75",
            array + ("1", "bValid"): "0x0",
            array + ("__attribute__", "size"): 2.0,
            body + ("0", "flFactor"): 180.75,
            body + ("0", "bValid"): "0x1",
            body + ("1", "flFactor"): 172.5,
        }
        self.assertEqual(fft_scale_factors(yaps), [4.5, 4.75])
        blocks = {
            0: {"factors": [4.5, 4.75], "valid_flags": [True, False]},
            1: {"factors": [180.75, 172.5], "valid_flags": [True, None]},
        }
        self.assertEqual(fft_scale_factors_by_block(yaps), blocks)
        metadata = measurement_metadata(yaps)
        self.assertEqual(metadata["fft_scale_factors"], [4.5, 4.75])
        self.assertEqual(metadata["fft_scale_valid_flags"], [True, False])
        self.assertEqual(metadata["fft_scale_block"], 0)
        self.assertEqual(metadata["fft_scale_by_block"], blocks)
        self.assertFalse(metadata["fft_scale_applied"])
        json.dumps(metadata)
        # A single block keeps the previous single-prefix result.
        array_only = {key: value for key, value in yaps.items() if key[2] == "0"}
        self.assertEqual(measurement_metadata(array_only)["fft_scale_by_block"], {0: blocks[0]})

        third = ("sCoilSelectMeas", "aRxCoilSelectData", "2", "aFFT_SCALE")
        refused = {
            "blocks without block 0": {body + ("0", "flFactor"): 1.0, third + ("0", "flFactor"): 1.0},
            "a block and another prefix": {
                array + ("0", "flFactor"): 1.0,
                ("sCoilSelectMeas", "aFFT_SCALE", "0", "flFactor"): 1.0,
            },
            "gap in block 1": {array + ("0", "flFactor"): 1.0, body + ("1", "flFactor"): 1.0},
            "non-finite block-1 factor": {array + ("0", "flFactor"): 1.0, body + ("0", "flFactor"): "nan"},
            "non-numeric block-1 flag": {
                array + ("0", "flFactor"): 1.0,
                body + ("0", "flFactor"): 1.0,
                body + ("0", "bValid"): "maybe",
            },
        }
        for name, header in refused.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                measurement_metadata(header)
        for flag, expected in (("0x1", True), ("0x0", False), ("1", True), (0.0, False), ("true", True)):
            with self.subTest(flag=flag):
                header = {array + ("0", "flFactor"): 1.0, array + ("0", "bValid"): flag}
                self.assertEqual(measurement_metadata(header)["fft_scale_valid_flags"], [expected])
        for flag in ("maybe", "nan", float("inf")):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                measurement_metadata({array + ("0", "flFactor"): 1.0, array + ("0", "bValid"): flag})


class NoiseArrayTests(unittest.TestCase):
    """Verify noise-line layout, covariance, and correlation statistics."""

    def test_mapvbvd_counter_axes_flatten_column_major(self) -> None:
        """Flatten trailing counters with the leading counter fastest.

        Returns:
            None.
        """
        columns, channels, lins, averages = 3, 2, 4, 2
        array = np.zeros((columns, channels, lins, averages), dtype=np.complex64)
        for c, h, lin, ave in np.ndindex(array.shape):
            array[c, h, lin, ave] = c + 10 * h + 100 * lin + 1000 * ave
        lines = noise_lines_from_mapvbvd(array)
        self.assertEqual(lines.shape, (lins * averages, columns, channels))
        self.assertEqual(lines.dtype, np.complex64)
        self.assertTrue(lines.flags.c_contiguous)
        for line in range(lins * averages):
            lin, ave = line % lins, line // lins
            expected = np.add.outer(np.arange(columns), 10 * np.arange(channels))
            np.testing.assert_array_equal(lines[line].real, expected + 100 * lin + 1000 * ave)
        single = noise_lines_from_mapvbvd(np.ones((3, 2)))
        self.assertEqual(single.shape, (1, 3, 2))
        for invalid in (np.ones(3), np.ones((3, 0, 2))):
            with self.assertRaises(ValueError):
                noise_lines_from_mapvbvd(invalid)

    def test_noise_covariance_contract(self) -> None:
        """Demean, stay Hermitian, scale without bias, and reject bad input.

        Returns:
            None.
        """
        rng = np.random.default_rng(7)
        values = rng.standard_normal((500, 3)) + 1j * rng.standard_normal((500, 3))
        shifted = values + np.array([5.0 + 2.0j, -3.0j, 1.5])
        original = shifted.copy()
        covariance = noise_covariance(shifted)
        np.testing.assert_array_equal(shifted, original)
        self.assertEqual(covariance.dtype, np.complex128)
        self.assertTrue(np.array_equal(covariance, covariance.conj().T))
        np.testing.assert_allclose(covariance, np.cov(values, rowvar=False), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            noise_covariance(shifted, demean=False), shifted.T @ shifted.conj() / 500, rtol=1e-12
        )
        alternating = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        np.testing.assert_allclose(noise_covariance(alternating), [[4.0 / 3.0]])
        np.testing.assert_allclose(noise_covariance(alternating, demean=False), [[1.0]])
        self.assertEqual(noise_covariance(np.ones((4, 3)) * 1j).shape, (3, 3))
        invalid_inputs = [
            np.ones((3, 3)),
            np.ones(5),
            np.ones((5, 0)),
            np.array([[np.nan, 0.0], [1.0, 1.0], [2.0, 0.0]]),
            np.array([[np.inf, 0.0], [1.0, 1.0], [2.0, 0.0]]),
            np.ones((5, 2), dtype=bool),
        ]
        for invalid in invalid_inputs:
            with self.assertRaises(ValueError):
                noise_covariance(invalid)

    def test_correlation_statistics_on_known_matrices(self) -> None:
        """Recover known correlations, condition numbers, and scale ratios.

        Returns:
            None.
        """
        channels, correlation = 4, 0.25
        rho = (1 - correlation) * np.eye(channels) + correlation * np.ones((channels, channels))
        phases = np.exp(1j * np.array([0.0, 0.4, -1.1, 2.0]))
        rotated = phases[:, None] * rho * phases.conj()[None, :]
        for matrix in (rho, rotated):
            stats = correlation_statistics(matrix)
            self.assertEqual(stats["channel_count"], channels)
            for key in (
                "median_abs_offdiagonal_correlation",
                "p95_abs_offdiagonal_correlation",
                "max_abs_offdiagonal_correlation",
            ):
                self.assertAlmostEqual(stats[key], correlation, places=12)
            self.assertAlmostEqual(stats["condition_number"], 1.75 / 0.75, places=12)
            self.assertAlmostEqual(stats["min_eigenvalue"], 0.75, places=12)
            self.assertAlmostEqual(stats["max_eigenvalue"], 1.75, places=12)
            self.assertAlmostEqual(stats["channel_std_ratio_max_min"], 1.0, places=12)
            self.assertTrue(stats["positive_definite"])
        deviation = np.sqrt(np.array([1.0, 2.0, 4.0, 9.0]))
        scaled = deviation[:, None] * rotated * deviation[None, :]
        stats = correlation_statistics(scaled)
        self.assertAlmostEqual(stats["max_abs_offdiagonal_correlation"], correlation, places=12)
        self.assertAlmostEqual(stats["channel_std_ratio_max_min"], 3.0, places=12)
        self.assertAlmostEqual(stats["condition_number"], np.linalg.cond(scaled), places=9)
        mixed = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.3], [0.1, 0.3, 1.0]])
        stats = correlation_statistics(mixed)
        self.assertAlmostEqual(stats["median_abs_offdiagonal_correlation"], 0.3, places=12)
        self.assertAlmostEqual(stats["p95_abs_offdiagonal_correlation"], 0.75, places=12)
        self.assertAlmostEqual(stats["max_abs_offdiagonal_correlation"], 0.8, places=12)
        singular = correlation_statistics(np.ones((2, 2)))
        self.assertFalse(singular["positive_definite"])
        self.assertEqual(singular["condition_number"], float("inf"))
        for invalid in (
            np.array([[1.0, 0.5], [0.1, 1.0]]),
            np.ones((1, 1)),
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[1.0, np.nan], [np.nan, 1.0]]),
        ):
            with self.assertRaises(ValueError):
                correlation_statistics(invalid)


class CovarianceComparisonTests(unittest.TestCase):
    """Verify noise splits and descriptive covariance compatibility."""

    def test_split_rules_are_deterministic(self) -> None:
        """Split by alternating indices or by leading/trailing halves.

        Returns:
            None.
        """
        lines = np.arange(7 * 2 * 3).reshape(7, 2, 3)
        estimation, validation, record = split_noise_lines(lines)
        np.testing.assert_array_equal(estimation, lines[[0, 2, 4, 6]])
        np.testing.assert_array_equal(validation, lines[[1, 3, 5]])
        self.assertEqual(record["rule"], "alternate")
        self.assertEqual(
            (record["line_count"], record["estimation_line_count"], record["validation_line_count"]),
            (7, 4, 3),
        )
        repeated = split_noise_lines(lines, "alternate")
        np.testing.assert_array_equal(repeated[0], estimation)
        self.assertEqual(repeated[2], record)
        estimation, validation, record = split_noise_lines(lines, "halves")
        np.testing.assert_array_equal(estimation, lines[:4])
        np.testing.assert_array_equal(validation, lines[4:])
        self.assertEqual(record["estimation_slice"], {"start": 0, "stop": 4, "step": 1})
        self.assertEqual(record["validation_slice"], {"start": 4, "stop": 7, "step": 1})
        estimation[...] = -1
        self.assertEqual(int(lines.min()), 0)
        json.dumps(record)
        with self.assertRaises(ValueError):
            split_noise_lines(lines, "random")
        with self.assertRaises(ValueError):
            split_noise_lines(lines[:1])

    def test_halves_from_one_covariance_are_compatible(self) -> None:
        """Accept both split rules for samples drawn from one covariance.

        Returns:
            None.
        """
        reference = kms_covariance()
        rng = np.random.default_rng(20260929)
        white = (
            rng.standard_normal((40_000, 8)) + 1j * rng.standard_normal((40_000, 8))
        ) / np.sqrt(2.0)
        lines = (white @ np.linalg.cholesky(reference).T).reshape(400, 100, 8)
        for rule in ("alternate", "halves"):
            estimation, validation, record = split_noise_lines(lines, rule)
            self.assertEqual(record["estimation_line_count"], 200)
            comparison = compare_covariances(
                noise_covariance(estimation.reshape(-1, 8)),
                noise_covariance(validation.reshape(-1, 8)),
            )
            result = covariance_compatibility(comparison)
            self.assertTrue(result["compatible"], result["criteria"])
            self.assertEqual(result["failed_criteria"], [])
            self.assertAlmostEqual(result["trace_ratio"], 1.0, delta=0.05)
            self.assertEqual(result["limits"], DEFAULT_COVARIANCE_COMPATIBILITY_LIMITS)
            self.assertIn("not establish an absolute", result["interpretation"])
            json.dumps(comparison)
            json.dumps(result)

    def test_permutation_scaling_and_correlation_changes(self) -> None:
        """Reject permutation and recorrelation; accept a pure 0.8 scaling.

        Returns:
            None.
        """
        reference = kms_covariance()
        order = np.arange(8)[::-1]
        permuted = covariance_compatibility(
            compare_covariances(reference, reference[np.ix_(order, order)])
        )
        self.assertFalse(permuted["compatible"])
        self.assertIn("per_channel_variance_ratio_spread", permuted["failed_criteria"])
        self.assertIn("max_abs_correlation_difference", permuted["failed_criteria"])

        comparison = compare_covariances(reference, 0.8 * reference)
        self.assertAlmostEqual(comparison["trace_ratio"], 0.8, places=12)
        self.assertAlmostEqual(comparison["normalized_shape_relative_difference"], 0.0, places=12)
        self.assertAlmostEqual(comparison["generalized_eigenvalues"]["spread"], 1.0, places=10)
        self.assertAlmostEqual(comparison["generalized_eigenvalues"]["log_rms"], 0.0, places=10)
        self.assertAlmostEqual(
            comparison["normalized_eigenvalue_spectra"]["max_relative_difference"], 0.0, places=10
        )
        self.assertAlmostEqual(
            comparison["condition_numbers"]["reference"],
            comparison["condition_numbers"]["candidate"],
            places=8,
        )
        scaled = covariance_compatibility(comparison)
        self.assertTrue(scaled["compatible"])
        self.assertAlmostEqual(scaled["trace_ratio"], 0.8, places=12)
        self.assertNotIn("trace_ratio", scaled["criteria"])

        recorrelated = compare_covariances(reference, kms_covariance(correlation=0.3))
        result = covariance_compatibility(recorrelated)
        self.assertFalse(result["compatible"])
        self.assertIn("max_abs_correlation_difference", result["failed_criteria"])
        self.assertAlmostEqual(recorrelated["per_channel_variance_ratio"]["spread"], 1.0, places=12)
        self.assertGreater(recorrelated["correlation_difference"]["max_abs"], 0.25)
        relaxed = covariance_compatibility(
            recorrelated,
            {"max_abs_correlation_difference": 0.5, "normalized_shape_relative_difference": 1.0,
             "generalized_eigenvalue_spread": 100.0},
        )
        self.assertTrue(relaxed["compatible"], relaxed["criteria"])

    def test_invalid_comparisons_are_rejected(self) -> None:
        """Reject bad limits, incomplete comparisons, and invalid matrices.

        Returns:
            None.
        """
        comparison = compare_covariances(np.eye(3), 2 * np.eye(3))
        for limits in ({"trace_ratio": 1.0}, {"generalized_eigenvalue_spread": -1.0},
                       {"generalized_eigenvalue_spread": float("nan")}):
            with self.assertRaises(ValueError):
                covariance_compatibility(comparison, limits)
        incomplete = dict(comparison)
        del incomplete["correlation_difference"]
        with self.assertRaises(ValueError):
            covariance_compatibility(incomplete)
        with self.assertRaises(ValueError):
            compare_covariances(np.eye(3), np.eye(4))
        with self.assertRaises(ValueError):
            compare_covariances(np.ones((3, 3)), np.eye(3))
        with self.assertRaises(ValueError):
            compare_covariances(np.eye(3), np.ones((3, 3)))


class ScalingAndSourceTests(unittest.TestCase):
    """Verify the white-noise ratio and the module's execution boundary."""

    def test_white_noise_variance_ratio_uses_dwell_ratio(self) -> None:
        """Return reference dwell divided by target dwell.

        Returns:
            None.
        """
        self.assertEqual(expected_white_noise_variance_ratio(4000, 5000), 0.8)
        self.assertEqual(expected_white_noise_variance_ratio(5000.0, 4000.0), 1.25)
        for arguments in ((0, 5000), (4000, -1), (float("nan"), 5000), (True, 5000)):
            with self.assertRaises(ValueError):
                expected_white_noise_variance_ratio(*arguments)
        self.assertIn(
            "approximation",
            " ".join((expected_white_noise_variance_ratio.__doc__ or "").split()),
        )

    def test_module_source_has_no_external_execution(self) -> None:
        """Keep process launches, printing, and eager mapVBVD imports out.

        Returns:
            None.
        """
        source = Path(twix_noise.__file__).read_text(encoding="utf-8")
        # Fragments keep this audit list from matching source-wide searches.
        for token in ("sub" + "process", "os." + "system", "ba" + "rt "):
            self.assertNotIn(token, source.lower())
        tree = ast.parse(source)
        top_level_modules: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_modules.add(node.module.split(".")[0])
        self.assertNotIn("mapvbvd", top_level_modules)
        self.assertNotIn("torch", top_level_modules)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("print", calls)
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        self.assertTrue(functions)
        for node in functions:
            self.assertTrue(ast.get_docstring(node), node.name)


@unittest.skipUnless(MAPVBVD_AVAILABLE, "mapvbvd is not installed")
class MapvbvdIntegrationTests(unittest.TestCase):
    """Verify the lazy mapVBVD loaders on the synthetic multi-raid file."""

    def test_headers_keep_coil_select_blocks_separate(self) -> None:
        """Parse real mapVBVD tuple keys and compare block 0 only.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_synthetic_twix(Path(temporary))
            with contextlib.redirect_stderr(io.StringIO()):
                headers = load_measurement_headers(fixture["path"])
        self.assertEqual(len(headers), 2)
        self.assertEqual(set(headers[0]), {"MeasYaps", "Meas"})
        noise_yaps = headers[0]["MeasYaps"]
        self.assertTrue(any("__attribute__" in key for key in noise_yaps if isinstance(key, tuple)))
        tables = coil_select_tables(noise_yaps)
        self.assertEqual(sorted(tables), [0, 1])
        self.assertEqual(tables[1], {1: f"{BODY_COIL}:B1", 2: f"{BODY_COIL}:B2"})
        report = compare_coil_select(noise_yaps, headers[1]["MeasYaps"])
        self.assertTrue(report["identical"])
        self.assertEqual(report["noise_block_sizes"], {0: 4, 1: 2})
        noise_metadata = measurement_metadata(noise_yaps, headers[0]["Meas"])
        acquisition_metadata = measurement_metadata(headers[1]["MeasYaps"], headers[1]["Meas"])
        self.assertEqual(noise_metadata["protocol_name"], "synthetic_adjust")
        self.assertEqual(noise_metadata["dwell_ns"], 4000.0)
        self.assertEqual(acquisition_metadata["dwell_ns"], 5000.0)
        self.assertEqual(noise_metadata["readout_oversampling_factor"], 2.0)
        self.assertEqual(noise_metadata["noise_decorrelation_mode"], 2)
        self.assertEqual(noise_metadata["fft_scale_factors"], list(NOISE_FFT_SCALE))
        self.assertEqual(acquisition_metadata["fft_scale_factors"], list(ACQUISITION_FFT_SCALE))
        self.assertEqual(noise_metadata["fft_scale_valid_flags"], [True] * 4)
        self.assertEqual(noise_metadata["fft_scale_block"], 0)
        self.assertEqual(
            noise_metadata["fft_scale_by_block"],
            {
                0: {"factors": list(NOISE_FFT_SCALE), "valid_flags": [True] * 4},
                1: {"factors": list(BODY_FFT_SCALE), "valid_flags": [True] * 2},
            },
        )
        self.assertEqual(sorted(acquisition_metadata["fft_scale_by_block"]), [0])
        self.assertFalse(noise_metadata["raw_data_correction_factors_present"])
        self.assertAlmostEqual(
            expected_white_noise_variance_ratio(
                noise_metadata["dwell_ns"], acquisition_metadata["dwell_ns"]
            ),
            0.8,
        )

    def test_measurement0_noise_lines_are_raw_and_in_acquisition_order(self) -> None:
        """Load every written noise line exactly once without scaling.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_synthetic_twix(Path(temporary))
            with contextlib.redirect_stderr(io.StringIO()):
                result = load_measurement0_noise(fixture["path"])
        np.testing.assert_array_equal(result["lines"], fixture["noise"])
        self.assertEqual(result["measurement_index"], 0)
        self.assertEqual(result["data_type"], "noise")
        self.assertFalse(result["remove_oversampling"])
        self.assertFalse(result["regridding_applied"])
        self.assertEqual(result["line_order"], "acquisition")
        self.assertEqual(result["measurement_count"], 2)
        self.assertFalse(result["contract"]["absolute_noise_calibration"])
        self.assertFalse(result["metadata"]["fft_scale_applied"])
        self.assertFalse(result["metadata"]["raw_data_correction_applied"])
        self.assertEqual(result["metadata"]["dwell_ns"], 4000.0)
        samples = result["lines"].reshape(-1, len(CHANNEL_IDS))
        stats = correlation_statistics(noise_covariance(samples))
        self.assertTrue(stats["positive_definite"])

    def test_single_measurement_file_is_rejected(self) -> None:
        """Require a multi-raid file for measurement-0 noise.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_synthetic_twix(Path(temporary), include_noise_measurement=False)
            self.assertEqual(len(read_multiraid_table_from_file(fixture["path"])), 1)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(ValueError):
                    load_measurement0_noise(fixture["path"])
                self.assertEqual(len(load_measurement_headers(fixture["path"])), 1)


if __name__ == "__main__":
    unittest.main()
