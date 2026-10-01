"""Read-only Siemens VD/VE TWIX noise, channel-identity, and covariance checks.

This module inspects multi-raid Siemens VD/VE TWIX files without modifying
them. It parses the multi-raid measurement table, walks scan headers to
classify MDH roles and exact channel-ID sequences, compares header coil
selection block by block, records scaling metadata, loads measurement-0 noise
lines through mapVBVD, and computes descriptive complex noise-covariance
statistics.

Scientific contract:

* The noise source is the measurement-0 adjustment scan (for example
  AdjCoilSens) of a multi-raid file. Its dwell time can differ from the dwell
  time of the integrated-refscan ACS or imaging measurement, and its FFT-scale
  metadata can differ from those of the imaging measurement.
* FFT-scale factors and raw-data-correction factors are recorded only. No
  function in this module applies either of them to any sample.
* A dwell-time ratio yields only an expected variance ratio under a white-noise
  approximation. It never establishes an absolute ACS noise calibration.
* Covariance compatibility is judged on trace-normalized shape criteria with
  pre-registered descriptive limits. The trace ratio is reported but is never a
  pass criterion, and a passing result is not an absolute calibration.
* Header coil selection is compared per ``aRxCoilSelectData`` block. Blocks
  are never merged, because non-array blocks (for example body-coil reference
  elements of an adjustment scan) can reuse the list indices and ADC channel
  numbers of the receive-array block. FFT-scale factors are read per block in
  the same way: block 0 is reported and every block is recorded.

Only :func:`read_multiraid_table_from_file`, :func:`walk_measurement`,
:func:`load_measurement_headers`, and :func:`load_measurement0_noise` read
files, and all of them are read-only. ``mapvbvd`` is imported lazily inside the
two loader functions.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

EVAL_INFO_BITS: dict[str, int] = {
    "ACQEND": 0,
    "RTFEEDBACK": 1,
    "HPFEEDBACK": 2,
    "ONLINE": 3,
    "OFFLINE": 4,
    "SYNCDATA": 5,
    "RAWDATACORRECTION": 10,
    "REFPHASESTABSCAN": 14,
    "PHASESTABSCAN": 15,
    "PHASCOR": 21,
    "PATREFSCAN": 22,
    "PATREFANDIMASCAN": 23,
    "REFLECT": 24,
    "NOISEADJSCAN": 25,
}
LOOP_COUNTER_NAMES: tuple[str, ...] = (
    "Lin",
    "Ave",
    "Sli",
    "Par",
    "Eco",
    "Phs",
    "Rep",
    "Set",
    "Seg",
    "Ida",
    "Idb",
    "Idc",
    "Idd",
    "Ide",
)

# Pre-registered descriptive limits for comparing trace-normalized noise
# covariances. They screen for gross channel-order, per-channel-scaling, and
# correlation-pattern changes. Passing them never proves an absolute noise
# calibration, and none of them constrains the absolute (trace) scale.
DEFAULT_COVARIANCE_COMPATIBILITY_LIMITS: dict[str, float] = {
    "normalized_shape_relative_difference": 0.10,
    "per_channel_variance_ratio_spread": 1.20,
    "max_abs_correlation_difference": 0.10,
    "generalized_eigenvalue_spread": 1.50,
}

_MULTIRAID_HEADER_BYTES = 10240
_MULTIRAID_PREAMBLE_BYTES = 8
_MULTIRAID_ENTRY_BYTES = 152
_MULTIRAID_ENTRY = struct.Struct("<IIQQ")
_PROTOCOL_NAME_START = 88
_PROTOCOL_NAME_STOP = 152
_MAXIMUM_MULTIRAID_MEASUREMENTS = 64
_MAXIMUM_VD_HEADER_SIZE_WORD = 10000
_SCAN_HEADER_BYTES = 192
_CHANNEL_HEADER_BYTES = 32
_CHANNEL_ID_OFFSET = 24
_COMPLEX64_BYTES = 8
_DMA_LENGTH_MASK = 0x01FFFFFF
# Flags/DMA word, 36 skipped bytes, both eval-info words, samples, used
# channels, and the 14 loop counters of one 192-byte VD/VE scan header.
_SCAN_HEADER = struct.Struct("<I36xIIHH14H")
_LIN_INDEX = LOOP_COUNTER_NAMES.index("Lin")
_PAR_INDEX = LOOP_COUNTER_NAMES.index("Par")
_SET_INDEX = LOOP_COUNTER_NAMES.index("Set")
_BIT = {name: 1 << position for name, position in EVAL_INFO_BITS.items()}
_REFSCAN_MASK = _BIT["PATREFSCAN"] | _BIT["PATREFANDIMASCAN"]
_FEEDBACK_MASK = _BIT["RTFEEDBACK"] | _BIT["HPFEEDBACK"]
_COIL_SELECT_PREFIX = ("sCoilSelectMeas", "aRxCoilSelectData")
_FFT_SCALE_COMPONENT = "aFFT_SCALE"
_BRACKET_INDEX = re.compile(r"\[([0-9]+)\]")
_NUMERIC_INDEX = re.compile(r"[0-9]+")
_HERMITIAN_RELATIVE_TOLERANCE = 1e-8
_COMPATIBILITY_INTERPRETATION = (
    "Descriptive covariance-shape compatibility gate on trace-normalized "
    "matrices with pre-registered limits. It does not establish an absolute "
    "noise calibration; the trace ratio is reported only and is never a pass "
    "criterion."
)


@dataclass(frozen=True)
class RaidMeasurement:
    """One entry of a Siemens VD/VE multi-raid measurement table.

    The patient-name field of the table is intentionally never decoded or
    stored.

    Attributes:
        index: Zero-based position in the multi-raid table.
        measurement_id: Measurement identifier stored in the table.
        file_id: File identifier stored in the table.
        offset: Byte offset of the measurement inside the file.
        length: Byte length of the measurement, including its header.
        protocol_name: NUL-terminated latin-1 protocol name.
    """

    index: int
    measurement_id: int
    file_id: int
    offset: int
    length: int
    protocol_name: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-native record of this measurement-table entry.

        Returns:
            Mapping with the table index, identifiers, byte range, and
            protocol name.
        """
        return {
            "index": self.index,
            "measurement_id": self.measurement_id,
            "file_id": self.file_id,
            "offset": self.offset,
            "length": self.length,
            "protocol_name": self.protocol_name,
        }


@dataclass
class _RoleAccumulator:
    """Accumulate MDH statistics for one scan role during a header walk.

    Attributes:
        lines: Number of scans assigned to the role.
        raw_data_correction_lines: Scans that carry the raw-data-correction
            flag.
        sequences: Line count per ``(samples, channel IDs)`` combination, in
            order of appearance.
        lin_par_pairs: Distinct ``(Lin, Par)`` counter pairs.
    """

    lines: int = 0
    raw_data_correction_lines: int = 0
    sequences: dict[tuple[int, tuple[int, ...]], int] = field(default_factory=dict)
    lin_par_pairs: set[tuple[int, int]] = field(default_factory=set)

    def add(
        self,
        samples: int,
        channel_ids: tuple[int, ...],
        lin: int,
        par: int,
        raw_data_correction: bool,
    ) -> None:
        """Record one scan of this role.

        Args:
            samples: Samples per channel in the scan.
            channel_ids: MDH channel IDs in stored channel order.
            lin: Lin loop counter.
            par: Par loop counter.
            raw_data_correction: Whether the raw-data-correction flag is set.
        """
        self.lines += 1
        self.raw_data_correction_lines += int(raw_data_correction)
        key = (samples, channel_ids)
        self.sequences[key] = self.sequences.get(key, 0) + 1
        self.lin_par_pairs.add((lin, par))

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-native summary of this role.

        Returns:
            Line counts, distinct channel-ID sequences, raw-data-correction
            line count, inclusive Lin/Par ranges, and distinct Lin/Par pairs.
        """
        lins = [lin for lin, _ in self.lin_par_pairs]
        pars = [par for _, par in self.lin_par_pairs]
        return {
            "lines": self.lines,
            "channel_id_sequences": [
                {"samples": samples, "channel_ids": list(channel_ids), "lines": count}
                for (samples, channel_ids), count in self.sequences.items()
            ],
            "raw_data_correction_lines": self.raw_data_correction_lines,
            "lin_range": [min(lins), max(lins)],
            "par_range": [min(pars), max(pars)],
            "unique_lin_par_pairs": len(self.lin_par_pairs),
        }


def read_multiraid_table(header: bytes) -> list[RaidMeasurement]:
    """Parse the multi-raid measurement table of a Siemens VD/VE file.

    The preamble holds a header-size word and the measurement count. Entry
    ``i`` occupies 152 bytes starting at byte ``8 + 152 * i``: measurement
    ID, file ID, 64-bit offset, 64-bit length, a 64-byte patient-name field
    (never decoded), and a 64-byte NUL-terminated protocol name.

    Args:
        header: Leading file bytes containing the preamble and all entries;
            normally the complete 10240-byte multi-raid header.

    Returns:
        Measurement entries in table order.

    Raises:
        TypeError: If ``header`` is not a supported bytes-like object.
        ValueError: If the preamble does not describe a VD/VE multi-raid
            table, the buffer is too short, or entry byte ranges are invalid.
    """
    view = _byte_view(header)
    if len(view) < _MULTIRAID_PREAMBLE_BYTES:
        raise ValueError("Multi-raid header is shorter than its 8-byte preamble.")
    header_size_word, count = struct.unpack_from("<II", view, 0)
    # mapVBVD identifies VD/VE files by a small header-size word and at most 64
    # table entries; VB files instead start with a large header length.
    if header_size_word >= _MAXIMUM_VD_HEADER_SIZE_WORD:
        raise ValueError(
            "File preamble does not describe a VD/VE multi-raid table "
            f"(header-size word {header_size_word})."
        )
    if not 1 <= count <= _MAXIMUM_MULTIRAID_MEASUREMENTS:
        raise ValueError(
            f"Multi-raid measurement count must lie in [1, "
            f"{_MAXIMUM_MULTIRAID_MEASUREMENTS}]; found {count}."
        )
    table_end = _MULTIRAID_PREAMBLE_BYTES + _MULTIRAID_ENTRY_BYTES * count
    if len(view) < table_end:
        raise ValueError(
            f"Multi-raid header holds {len(view)} bytes but {count} entries "
            f"require {table_end} bytes."
        )
    measurements: list[RaidMeasurement] = []
    for index in range(count):
        start = _MULTIRAID_PREAMBLE_BYTES + _MULTIRAID_ENTRY_BYTES * index
        measurement_id, file_id, offset, length = _MULTIRAID_ENTRY.unpack_from(
            view, start
        )
        name_bytes = bytes(
            view[start + _PROTOCOL_NAME_START : start + _PROTOCOL_NAME_STOP]
        )
        protocol_name = name_bytes.split(b"\0", 1)[0].decode("latin-1")
        if offset < table_end:
            raise ValueError(
                f"Measurement {index} offset {offset} overlaps the multi-raid table."
            )
        if length < 4:
            raise ValueError(
                f"Measurement {index} length {length} cannot hold its header length."
            )
        measurements.append(
            RaidMeasurement(
                index=index,
                measurement_id=int(measurement_id),
                file_id=int(file_id),
                offset=int(offset),
                length=int(length),
                protocol_name=protocol_name,
            )
        )
    ordered = sorted(measurements, key=lambda item: item.offset)
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier.offset + earlier.length > later.offset:
            raise ValueError(
                f"Measurements {earlier.index} and {later.index} have overlapping "
                "byte ranges."
            )
    return measurements


def read_multiraid_table_from_file(path: str | Path) -> list[RaidMeasurement]:
    """Read and validate the multi-raid measurement table of a TWIX file.

    Args:
        path: Siemens VD/VE TWIX file.

    Returns:
        Measurement entries in table order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the table is invalid or a measurement extends beyond
            the end of the file.

    Side Effects:
        Reads at most the leading 10240 bytes of the file without modifying
        it.
    """
    source = Path(path).expanduser()
    file_size = source.stat().st_size
    with source.open("rb") as stream:
        header = stream.read(_MULTIRAID_HEADER_BYTES)
    measurements = read_multiraid_table(header)
    for measurement in measurements:
        if measurement.offset + measurement.length > file_size:
            raise ValueError(
                f"Measurement {measurement.index} extends beyond the file end "
                f"({measurement.offset + measurement.length} > {file_size} bytes)."
            )
    return measurements


def walk_scan_headers(buffer: Any, start: int, end: int) -> dict[str, Any]:
    """Walk VD/VE scan headers and summarize MDH roles and channel IDs.

    Each regular scan is a 192-byte header followed by ``used channels``
    blocks of a 32-byte channel header plus ``samples`` complex64 values; the
    walk advances by that computed size. SYNCDATA scans carry no channel data
    and are skipped by their 25-bit DMA length. An ACQEND scan ends the walk,
    and the walk never reads beyond ``end``.

    Roles follow a fixed precedence: NOISEADJSCAN gives ``"noise"``;
    PATREFSCAN or PATREFANDIMASCAN gives ``"refscan_set<Set>"``; PHASCOR gives
    ``"phasecor"``; RTFEEDBACK or HPFEEDBACK gives ``"feedback"``; every other
    scan is ``"image"``. Unlike mapVBVD's separate ``refscanPC`` stream, a
    refscan that also carries PHASCOR is therefore counted in its refscan set,
    and phase-stabilization scans without other role flags count as images.

    Args:
        buffer: ``bytes``, ``bytearray``, ``memoryview``, or a one-dimensional
            C-contiguous ``numpy.uint8`` array or memory map.
        start: Byte offset of the scan header where the walk begins.
        end: Exclusive byte offset that the walk must not cross, normally the
            end of the measurement.

    Returns:
        JSON-native summary with ``start``, ``end``, ``scans`` (regular scans
        walked), ``acqend_found``, ``termination`` (``acqend``,
        ``measurement_end``, ``incomplete_scan_header``, ``zero_dma_length``,
        ``invalid_sync_data_length``, ``sync_data_beyond_end``, or
        ``scan_beyond_end``), ``stop_offset``, ``flag_counts`` (every
        eval-info bit counted over the regular, SYNCDATA, and ACQEND headers
        that were accepted), ``dma_length_mismatches`` (regular scans whose
        DMA length differs from the computed size), and ``roles``. Each role
        reports ``lines``, ``channel_id_sequences`` (distinct
        samples/channel-ID combinations with line counts),
        ``raw_data_correction_lines``, inclusive ``lin_range`` and
        ``par_range``, and ``unique_lin_par_pairs``.

    Raises:
        TypeError: If ``buffer`` or the offsets have unsupported types.
        ValueError: If the buffer layout or byte range is invalid.
    """
    view = _byte_view(buffer)
    start = _nonnegative_offset(start, "start")
    end = _nonnegative_offset(end, "end")
    if not start <= end <= len(view):
        raise ValueError(
            f"Scan walk range [{start}, {end}) is invalid for a {len(view)}-byte buffer."
        )
    flag_counts = {name: 0 for name in EVAL_INFO_BITS}
    roles: dict[str, _RoleAccumulator] = {}
    scans = 0
    dma_length_mismatches = 0
    acqend_found = False
    position = start
    while True:
        if position + _SCAN_HEADER_BYTES > end:
            termination = (
                "measurement_end" if position == end else "incomplete_scan_header"
            )
            break
        word, mask, _, samples, channels, *counters = _SCAN_HEADER.unpack_from(
            view, position
        )
        dma_length = word & _DMA_LENGTH_MASK
        if mask & _BIT["ACQEND"]:
            _count_flags(flag_counts, mask)
            acqend_found = True
            termination = "acqend"
            break
        if dma_length == 0:
            # mapVBVD also treats a zero DMA length as the end of scan data;
            # continuing would misread zero padding as empty scans.
            termination = "zero_dma_length"
            break
        if mask & _BIT["SYNCDATA"]:
            if dma_length < _SCAN_HEADER_BYTES:
                termination = "invalid_sync_data_length"
                break
            if position + dma_length > end:
                termination = "sync_data_beyond_end"
                break
            _count_flags(flag_counts, mask)
            position += dma_length
            continue
        stride = _CHANNEL_HEADER_BYTES + _COMPLEX64_BYTES * samples
        size = _SCAN_HEADER_BYTES + channels * stride
        if position + size > end:
            termination = "scan_beyond_end"
            break
        _count_flags(flag_counts, mask)
        if channels:
            channel_ids = tuple(
                np.ndarray(
                    (channels,),
                    dtype="<u2",
                    buffer=view,
                    offset=position + _SCAN_HEADER_BYTES + _CHANNEL_ID_OFFSET,
                    strides=(stride,),
                ).tolist()
            )
        else:
            channel_ids = ()
        role = _scan_role(mask, counters[_SET_INDEX])
        accumulator = roles.get(role)
        if accumulator is None:
            accumulator = roles[role] = _RoleAccumulator()
        accumulator.add(
            samples,
            channel_ids,
            counters[_LIN_INDEX],
            counters[_PAR_INDEX],
            bool(mask & _BIT["RAWDATACORRECTION"]),
        )
        if dma_length != size:
            dma_length_mismatches += 1
        scans += 1
        position += size
    return {
        "start": start,
        "end": end,
        "scans": scans,
        "acqend_found": acqend_found,
        "termination": termination,
        "stop_offset": position,
        "flag_counts": flag_counts,
        "dma_length_mismatches": dma_length_mismatches,
        "roles": {role: accumulator.to_json() for role, accumulator in roles.items()},
    }


def walk_measurement(path: str | Path, measurement: RaidMeasurement) -> dict[str, Any]:
    """Walk the scan headers of one multi-raid measurement.

    Scans start at ``offset + hdr_len``, where ``hdr_len`` is the uint32 at the
    measurement offset, and the walk stops at ``offset + length`` at the
    latest.

    Args:
        path: Siemens VD/VE TWIX file.
        measurement: Table entry returned by
            :func:`read_multiraid_table_from_file` for the same file.

    Returns:
        The :func:`walk_scan_headers` summary plus ``measurement`` (the
        table entry) and ``measurement_header_bytes``.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If ``measurement`` is not a :class:`RaidMeasurement`.
        ValueError: If the measurement range or header length is invalid.

    Side Effects:
        Maps the file read-only with ``numpy.memmap``; nothing is written.
    """
    if not isinstance(measurement, RaidMeasurement):
        raise TypeError("walk_measurement requires a RaidMeasurement entry.")
    source = Path(path).expanduser()
    file_size = source.stat().st_size
    end = measurement.offset + measurement.length
    if measurement.length < 4 or end > file_size:
        raise ValueError(
            f"Measurement {measurement.index} byte range does not fit the "
            f"{file_size}-byte file."
        )
    data = np.memmap(source, dtype=np.uint8, mode="r")
    try:
        (header_bytes,) = struct.unpack_from("<I", data, measurement.offset)
        scan_start = measurement.offset + header_bytes
        if header_bytes < 4 or scan_start > end:
            raise ValueError(
                f"Measurement {measurement.index} header length {header_bytes} "
                "is inconsistent with its byte range."
            )
        walk = walk_scan_headers(data, scan_start, end)
    finally:
        del data
    return {
        **walk,
        "measurement": measurement.to_json(),
        "measurement_header_bytes": int(header_bytes),
    }


def channel_identity_report(
    noise_walk: Mapping[str, Any],
    acquisition_walk: Mapping[str, Any],
    *,
    required_roles: Sequence[str] = ("noise", "image", "refscan_set4"),
) -> dict[str, Any]:
    """Check that every MDH channel-ID sequence matches one exact reference.

    The reference is the noise walk's ``"noise"`` sequence when present;
    otherwise it is the earliest listed sequence of either walk. Sequences
    are compared in exact stored order, independently of sample counts.

    Args:
        noise_walk: :func:`walk_scan_headers` or :func:`walk_measurement`
            summary of the noise measurement.
        acquisition_walk: Summary of the imaging/refscan measurement.
        required_roles: Roles that must occur in at least one walk.

    Returns:
        ``identical`` (every sequence of every role of both walks equals the
        reference), ``sequential`` (reference is ``0..N-1``),
        ``reference_channel_ids``, ``channel_count``, ``role_sequences``
        (distinct sequences per role across both walks),
        ``required_roles_present``, ``missing_roles``, and ``differences``
        (human-readable mismatch descriptions).

    Raises:
        TypeError: If ``required_roles`` is a single string.
        ValueError: If a walk lacks a valid ``roles`` mapping.
    """
    if isinstance(required_roles, str):
        raise TypeError("required_roles must be a sequence of role names.")
    walks = (
        ("noise measurement", _walk_roles(noise_walk, "noise")),
        ("acquisition measurement", _walk_roles(acquisition_walk, "acquisition")),
    )
    observed: list[tuple[str, str, int, tuple[int, ...]]] = []
    present_roles: set[str] = set()
    for label, roles in walks:
        present_roles.update(roles)
        for role in sorted(roles):
            for entry in roles[role].get("channel_id_sequences", []):
                channel_ids = tuple(int(value) for value in entry["channel_ids"])
                observed.append((label, role, int(entry.get("samples", -1)), channel_ids))
    reference = next(
        (
            channel_ids
            for label, role, _, channel_ids in observed
            if label == "noise measurement" and role == "noise"
        ),
        observed[0][3] if observed else None,
    )
    differences: list[str] = []
    role_sequences: dict[str, list[list[int]]] = {}
    for label, role, samples, channel_ids in observed:
        sequences = role_sequences.setdefault(role, [])
        if list(channel_ids) not in sequences:
            sequences.append(list(channel_ids))
        if channel_ids != reference:
            differences.append(
                _sequence_difference(label, role, samples, channel_ids, reference)
            )
    if not observed:
        differences.append("No channel-ID sequences were found in either walk.")
    missing_roles = [role for role in required_roles if role not in present_roles]
    return {
        "identical": bool(observed) and not differences,
        "sequential": bool(reference) and list(reference) == list(range(len(reference))),
        "reference_channel_ids": list(reference) if reference is not None else None,
        "channel_count": len(reference) if reference is not None else 0,
        "role_sequences": {role: role_sequences[role] for role in sorted(role_sequences)},
        "required_roles_present": not missing_roles,
        "missing_roles": missing_roles,
        "differences": differences,
    }


def coil_select_tables(meas_yaps: Mapping[Any, Any]) -> dict[int, dict[int, str]]:
    """Build one ADC-channel-to-element table per coil-select block.

    Entries are read from ``sCoilSelectMeas.aRxCoilSelectData[b].asList[i]``.
    Every block ``b`` is kept separate: block 0 is the receive-array
    selection, while other blocks can reuse list indices and ADC channel
    numbers. Element labels are ``"<tCoilID>:<tElement>"`` (or ``tElement``
    alone without a coil ID), with surrounding ASCCONV quotes removed. Tuple
    keys, dotted keys, and bracketed dotted keys are accepted; keys with an
    ``__attribute__`` component or non-numeric block/list indices are ignored.

    Args:
        meas_yaps: mapVBVD ``MeasYaps``-compatible mapping.

    Returns:
        Mapping ``block -> {ADC channel -> element label}`` sorted by block
        and ADC channel.

    Raises:
        TypeError: If ``meas_yaps`` is not a mapping.
        ValueError: If an entry lacks a valid ADC channel or element name, or
            one block assigns the same ADC channel twice.
    """
    entries: dict[int, dict[int, dict[str, Any]]] = {}
    for parts, value in _normalized_header(meas_yaps).items():
        if len(parts) < 6 or parts[:2] != _COIL_SELECT_PREFIX or parts[3] != "asList":
            continue
        if not (_NUMERIC_INDEX.fullmatch(parts[2]) and _NUMERIC_INDEX.fullmatch(parts[4])):
            continue
        block_entries = entries.setdefault(int(parts[2]), {})
        block_entries.setdefault(int(parts[4]), {})[".".join(parts[5:])] = value
    tables: dict[int, dict[int, str]] = {}
    for block in sorted(entries):
        table: dict[int, str] = {}
        for list_index in sorted(entries[block]):
            fields = entries[block][list_index]
            location = f"coil-select block {block} list entry {list_index}"
            adc_channel = _channel_number(fields.get("lADCChannelConnected"), location)
            element = _clean_text(fields.get("sCoilElementID.tElement"))
            if not element:
                raise ValueError(f"The {location} has no sCoilElementID.tElement.")
            coil_id = _clean_text(fields.get("sCoilElementID.tCoilID"))
            if adc_channel in table:
                raise ValueError(
                    f"Coil-select block {block} assigns ADC channel {adc_channel} twice."
                )
            table[adc_channel] = f"{coil_id}:{element}" if coil_id else element
        tables[block] = dict(sorted(table.items()))
    return tables


def compare_coil_select(
    noise_yaps: Mapping[Any, Any],
    acquisition_yaps: Mapping[Any, Any],
    *,
    block: int = 0,
) -> dict[str, Any]:
    """Compare one coil-select block of two headers without merging blocks.

    Args:
        noise_yaps: ``MeasYaps`` mapping of the noise measurement.
        acquisition_yaps: ``MeasYaps`` mapping of the imaging measurement.
        block: ``aRxCoilSelectData`` block to compare; block 0 is the
            receive-array selection.

    Returns:
        ``block``, ``identical``, ``noise_adc_channels_contiguous``,
        ``acquisition_adc_channels_contiguous``, ``noise_block_sizes`` and
        ``acquisition_block_sizes`` (entries per block, all blocks), and
        ``differences`` as ``[adc, noise_element, acquisition_element]``
        triples, with ``None`` for an absent ADC channel.

    Raises:
        TypeError: If ``block`` is not an integer.
        ValueError: If ``block`` is negative or absent from either header, or
            a coil-select table is invalid.
    """
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)):
        raise TypeError("Coil-select block must be an integer.")
    block = int(block)
    if block < 0:
        raise ValueError("Coil-select block must be nonnegative.")
    noise_tables = coil_select_tables(noise_yaps)
    acquisition_tables = coil_select_tables(acquisition_yaps)
    for label, tables in (("noise", noise_tables), ("acquisition", acquisition_tables)):
        if block not in tables:
            raise ValueError(f"Coil-select block {block} is absent from the {label} header.")
    noise_map = noise_tables[block]
    acquisition_map = acquisition_tables[block]
    differences = [
        [adc, noise_map.get(adc), acquisition_map.get(adc)]
        for adc in sorted(set(noise_map) | set(acquisition_map))
        if noise_map.get(adc) != acquisition_map.get(adc)
    ]
    return {
        "block": block,
        "identical": not differences,
        "noise_adc_channels_contiguous": _is_contiguous(noise_map),
        "acquisition_adc_channels_contiguous": _is_contiguous(acquisition_map),
        "noise_block_sizes": {key: len(value) for key, value in noise_tables.items()},
        "acquisition_block_sizes": {
            key: len(value) for key, value in acquisition_tables.items()
        },
        "differences": differences,
    }


def fft_scale_factors(meas_yaps: Mapping[Any, Any]) -> list[float]:
    """Return the receive-array FFT-scale factors in index order; never apply them.

    Factors are read from keys ending in ``aFFT_SCALE[i].flFactor``. VD/VE
    headers store them per coil-select block, under
    ``sCoilSelectMeas.aRxCoilSelectData[b].aFFT_SCALE[i]``, and blocks are
    never merged. When several blocks hold factors, the block-0
    receive-array factors are returned and :func:`fft_scale_factors_by_block`
    records every block. Factors stored under one other prefix are returned
    as stored. They are measurement-specific metadata that this module
    records only.

    Args:
        meas_yaps: mapVBVD ``MeasYaps``-compatible mapping.

    Returns:
        Finite factors ordered by index ``0..N-1``; empty if none is stored.

    Raises:
        TypeError: If ``meas_yaps`` is not a mapping.
        ValueError: If factors appear under several key prefixes that are not
            all coil-select blocks including block 0, indices are not
            contiguous from zero, or a factor is not finite.
    """
    return _fft_factor_list(_fft_scale_selection(meas_yaps)[1], "FFT-scale")


def fft_scale_factors_by_block(meas_yaps: Mapping[Any, Any]) -> dict[int, dict[str, list[Any]]]:
    """Return the FFT-scale factors and ``bValid`` flags of every coil-select block.

    Each ``sCoilSelectMeas.aRxCoilSelectData[b]`` block is validated on its own
    and never merged with another, because a non-array block (for example the
    body-coil reference selection of an adjustment scan) has its own factors.
    The factors are recorded only.

    Args:
        meas_yaps: mapVBVD ``MeasYaps``-compatible mapping.

    Returns:
        Mapping ``block -> {"factors": [...], "valid_flags": [...]}`` sorted by
        block; empty when no factor is stored under a coil-select block.

    Raises:
        TypeError: If ``meas_yaps`` is not a mapping.
        ValueError: If two key prefixes name the same block, a block's indices
            are not contiguous from zero, a factor is not finite, or a
            ``bValid`` flag is not Boolean-like.
    """
    blocks = _coil_select_fft_groups(_fft_scale_groups(meas_yaps))
    result: dict[int, dict[str, list[Any]]] = {}
    for block in sorted(blocks):
        label = f"Coil-select block {block} FFT-scale"
        factors = _fft_factor_list(blocks[block], label)
        result[block] = {
            "factors": factors,
            "valid_flags": _fft_flag_list(blocks[block], len(factors), label),
        }
    return result


def measurement_metadata(
    meas_yaps: Mapping[Any, Any],
    meas_header: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Record noise-relevant acquisition metadata without applying scaling.

    Args:
        meas_yaps: mapVBVD ``MeasYaps``-compatible mapping.
        meas_header: Optional mapVBVD ``Meas`` mapping, which carries the
            readout oversampling factor and raw-data-correction factors.

    Returns:
        ``protocol_name``, ``dwell_ns`` (``sRXSPEC.alDwellTime[0]``),
        ``readout_oversampling_factor``, ``noise_decorrelation_mode`` (block-0
        ``ucNoiseDecorrMode``), ``raw_data_correction_factors_present``,
        ``raw_data_correction_factors`` (parsed real/imaginary field values,
        ``None`` when a field is absent), ``fft_scale_factors`` (see
        :func:`fft_scale_factors`), ``fft_scale_valid_flags`` (``bValid`` per
        factor, ``None`` if unrecorded), ``fft_scale_block`` (coil-select
        block of those factors, ``None`` for another prefix or when none is
        stored), ``fft_scale_by_block`` (see
        :func:`fft_scale_factors_by_block`), and the constant flags
        ``fft_scale_applied=False`` and ``raw_data_correction_applied=False``.
        Absent scalar fields are ``None``; no default is assumed.

    Raises:
        TypeError: If a header is not a mapping.
        ValueError: If a recorded numeric field or flag is malformed, or the
            FFT-scale layout is ambiguous.
    """
    yaps = _normalized_header(meas_yaps)
    meas = _normalized_header(meas_header)
    protocol_name = _clean_text(
        yaps.get(("tProtocolName",), meas.get(("tProtocolName",)))
    )
    raw_correction: dict[str, list[float] | str | None] = {}
    for field_name in ("dRawDataCorrectionFactorRe", "dRawDataCorrectionFactorIm"):
        value = meas.get((field_name,), yaps.get((field_name,)))
        raw_correction[field_name] = _number_list(value)
    fft_block, fft_entries = _fft_scale_selection(meas_yaps)
    factors = _fft_factor_list(fft_entries, "FFT-scale")
    return {
        "protocol_name": protocol_name or None,
        "dwell_ns": _optional_float(
            yaps.get(("sRXSPEC", "alDwellTime", "0")), "sRXSPEC.alDwellTime[0]"
        ),
        "readout_oversampling_factor": _optional_float(
            meas.get(("flReadoutOSFactor",)), "flReadoutOSFactor"
        ),
        "noise_decorrelation_mode": _header_code(
            yaps.get((*_COIL_SELECT_PREFIX, "0", "ucNoiseDecorrMode"))
        ),
        "raw_data_correction_factors_present": any(
            bool(value) for value in raw_correction.values()
        ),
        "raw_data_correction_factors": raw_correction,
        "fft_scale_factors": factors,
        "fft_scale_valid_flags": _fft_flag_list(fft_entries, len(factors), "FFT-scale"),
        "fft_scale_block": fft_block,
        "fft_scale_by_block": fft_scale_factors_by_block(meas_yaps),
        "fft_scale_applied": False,
        "raw_data_correction_applied": False,
    }


def load_measurement_headers(path: str | Path) -> list[dict[str, Any]]:
    """Load the parsed ``MeasYaps`` and ``Meas`` headers of every measurement.

    Args:
        path: Siemens TWIX file.

    Returns:
        One ``{"MeasYaps": dict, "Meas": dict}`` record per measurement in
        file order; a single-measurement file yields one record.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If ``mapvbvd`` is unavailable.

    Side Effects:
        Reads the file read-only through mapVBVD with MDH parsing disabled.
    """
    import mapvbvd

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = mapvbvd.mapVBVD(str(source), quiet=True, bReadMDH=False)
    measurements = list(root) if isinstance(root, (list, tuple)) else [root]
    records: list[dict[str, Any]] = []
    for measurement in measurements:
        header = measurement["hdr"]
        records.append(
            {
                "MeasYaps": dict(header.get("MeasYaps", {})),
                "Meas": dict(header.get("Meas", {})),
            }
        )
    return records


def noise_lines_from_mapvbvd(array: np.ndarray) -> np.ndarray:
    """Reorder a mapVBVD noise array into noise lines.

    mapVBVD uses MATLAB-style column-major dimension order, so all trailing
    counter axes are flattened in column-major order: the leading counter
    axis (normally Lin) varies fastest along the returned line axis.

    Args:
        array: Array shaped ``(Col, Cha, counters...)``; a two-dimensional
            ``(Col, Cha)`` array is one line.

    Returns:
        C-contiguous array shaped ``(Nlines, Ncol, Ncha)`` with the input
        dtype.

    Raises:
        ValueError: If the array has fewer than two axes or no samples.
    """
    values = np.asarray(array)
    if values.ndim < 2:
        raise ValueError("mapVBVD noise data must have at least (Col, Cha) axes.")
    if values.size == 0:
        raise ValueError("mapVBVD noise data contain no samples.")
    columns, channels = values.shape[:2]
    stacked = values.reshape((columns, channels, -1), order="F")
    return np.ascontiguousarray(np.transpose(stacked, (2, 0, 1)))


def load_measurement0_noise(path: str | Path) -> dict[str, Any]:
    """Load raw measurement-0 noise-adjustment lines of a multi-raid file.

    Readout oversampling is kept and regridding is disabled. Lines are read
    with ``twix_map_obj.unsorted()`` in acquisition order, because the
    counter-sorted accessor ``obj['']`` averages repeated counter tuples and
    zero-fills unacquired ones, both of which would bias a covariance
    estimate. FFT-scale and raw-data-correction factors are recorded in the
    metadata and never applied.

    Args:
        path: Multi-raid Siemens VD/VE TWIX file.

    Returns:
        ``lines`` shaped ``(Nlines, Ncol, Ncha)``, ``measurement_index=0``,
        ``data_type="noise"``, ``remove_oversampling=False``,
        ``regridding_applied=False``, ``line_order="acquisition"``,
        ``reader``, ``measurement_count``, ``metadata`` from
        :func:`measurement_metadata`, and ``contract`` describing the noise
        source and the absence of any absolute calibration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If ``mapvbvd`` is unavailable.
        ValueError: If the file is not multi-raid, measurement 0 has no noise
            data, or mapVBVD reports an incomplete read.

    Side Effects:
        Reads the file read-only through mapVBVD.
    """
    import mapvbvd

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = mapvbvd.mapVBVD(str(source), quiet=True)
    if not isinstance(root, (list, tuple)) or len(root) < 2:
        raise ValueError(
            "Measurement-0 noise requires a multi-raid TWIX file with at least "
            "two measurements."
        )
    measurement = root[0]
    if "noise" not in measurement:
        raise ValueError("Measurement 0 contains no noise-adjustment data.")
    noise = measurement["noise"]
    noise.flagRemoveOS = False
    noise.regrid = False
    lines = noise_lines_from_mapvbvd(np.asarray(noise.unsorted()))
    if bool(getattr(noise, "isBrokenFile", False)):
        raise ValueError("mapVBVD reported an incomplete measurement-0 noise read.")
    if lines.shape[0] != int(noise.NAcq):
        raise ValueError(
            f"Read {lines.shape[0]} noise lines but mapVBVD indexed {int(noise.NAcq)}."
        )
    header = measurement["hdr"]
    return {
        "lines": lines,
        "measurement_index": 0,
        "data_type": "noise",
        "remove_oversampling": False,
        "regridding_applied": False,
        "line_order": "acquisition",
        "reader": "mapvbvd twix_map_obj.unsorted",
        "measurement_count": len(root),
        "metadata": measurement_metadata(header["MeasYaps"], header.get("Meas")),
        "contract": {
            "source": "measurement-0 noise-adjustment scan",
            "dwell_time_may_differ_from_acquisition": True,
            "fft_scale_applied": False,
            "raw_data_correction_applied": False,
            "absolute_noise_calibration": False,
        },
    }


def noise_covariance(samples: np.ndarray, *, demean: bool = True) -> np.ndarray:
    """Estimate an unbiased complex channel covariance from noise samples.

    The convention matches ``numpy.cov``: ``C[i, j] = E[x_i conj(x_j)]``.
    With ``demean=True`` channel means are removed and the ``1/(N-1)``
    estimator is used; with ``demean=False`` the mean is taken as known zero
    and the unbiased estimator is ``1/N``.

    Args:
        samples: Array shaped ``(N, Nc)`` with one row per sample and one
            column per channel.
        demean: Whether to subtract the per-channel sample mean.

    Returns:
        Exactly Hermitian complex128 covariance shaped ``(Nc, Nc)``.

    Raises:
        ValueError: If the array is not two-dimensional and numeric, has no
            channels, has ``N <= Nc`` samples, or contains non-finite values.
    """
    values = np.asarray(samples)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("Noise samples must be a numeric (N, Nc) array.")
    count, channels = values.shape
    if channels < 1:
        raise ValueError("Noise samples must contain at least one channel.")
    if count <= channels:
        raise ValueError(
            f"Noise covariance needs more samples than channels; found N={count}, "
            f"Nc={channels}."
        )
    values = values.astype(np.complex128, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Noise samples contain non-finite values.")
    if demean:
        values -= values.mean(axis=0, keepdims=True)
    covariance = values.T @ values.conj() / (count - 1 if demean else count)
    return 0.5 * (covariance + covariance.conj().T)


def correlation_statistics(covariance: np.ndarray) -> dict[str, Any]:
    """Summarize channel correlation and conditioning of one covariance.

    Args:
        covariance: Hermitian covariance shaped ``(Nc, Nc)`` with ``Nc >= 2``
            and a positive diagonal.

    Returns:
        ``channel_count``; median, 95th-percentile (linear interpolation), and
        maximum absolute off-diagonal correlation; ``condition_number``
        (infinite unless positive definite); ``channel_std_ratio_max_min``;
        ``min_eigenvalue``; ``max_eigenvalue``; and ``positive_definite``
        (smallest eigenvalue above ``Nc * eps`` times the largest).

    Raises:
        ValueError: If the covariance is not a finite Hermitian matrix with
            at least two channels and a positive diagonal.
    """
    matrix = _validated_covariance(covariance, "Covariance")
    channels = matrix.shape[0]
    upper = np.triu_indices(channels, k=1)
    magnitudes = np.abs(_correlation_matrix(matrix)[upper])
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive_definite = _positive_definite(eigenvalues)
    deviations = np.sqrt(matrix.diagonal().real)
    return {
        "channel_count": channels,
        "median_abs_offdiagonal_correlation": float(np.median(magnitudes)),
        "p95_abs_offdiagonal_correlation": float(np.percentile(magnitudes, 95.0)),
        "max_abs_offdiagonal_correlation": float(np.max(magnitudes)),
        "condition_number": (
            float(eigenvalues[-1] / eigenvalues[0]) if positive_definite else math.inf
        ),
        "channel_std_ratio_max_min": float(np.max(deviations) / np.min(deviations)),
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
        "positive_definite": positive_definite,
    }


def split_noise_lines(
    lines: np.ndarray, rule: str = "alternate"
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split noise lines deterministically into estimation and validation sets.

    ``"alternate"`` interleaves the subsets in acquisition time, so slow drift
    affects both equally; ``"halves"`` separates them in time and therefore
    also probes temporal stability.

    Args:
        lines: Array whose leading axis indexes noise lines, normally shaped
            ``(Nlines, Ncol, Ncha)``.
        rule: ``"alternate"`` (even versus odd line indices) or ``"halves"``
            (leading ``ceil(N/2)`` versus trailing ``floor(N/2)`` lines).

    Returns:
        Estimation lines, validation lines (both C-contiguous copies), and a
        JSON-native record of the rule, index slices, and line counts.

    Raises:
        ValueError: If fewer than two lines are supplied or the rule is
            unknown.
    """
    values = np.asarray(lines)
    if values.ndim < 1 or values.shape[0] < 2:
        raise ValueError("Splitting noise requires at least two lines.")
    count = int(values.shape[0])
    if rule == "alternate":
        estimation = values[0::2].copy()
        validation = values[1::2].copy()
        description = (
            "Estimation uses even line indices and validation uses odd line indices."
        )
        estimation_slice = {"start": 0, "stop": count, "step": 2}
        validation_slice = {"start": 1, "stop": count, "step": 2}
    elif rule == "halves":
        split = (count + 1) // 2
        estimation = values[:split].copy()
        validation = values[split:].copy()
        description = (
            "Estimation uses the leading ceil(N/2) lines and validation uses the "
            "trailing floor(N/2) lines."
        )
        estimation_slice = {"start": 0, "stop": split, "step": 1}
        validation_slice = {"start": split, "stop": count, "step": 1}
    else:
        raise ValueError(f"Unknown noise split rule {rule!r}; use 'alternate' or 'halves'.")
    return estimation, validation, {
        "rule": rule,
        "description": description,
        "line_count": count,
        "estimation_line_count": int(estimation.shape[0]),
        "validation_line_count": int(validation.shape[0]),
        "estimation_slice": estimation_slice,
        "validation_slice": validation_slice,
    }


def compare_covariances(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Compare two noise covariances at matrix level after trace normalization.

    Both matrices are divided by their traces before every shape metric, so
    the comparison is insensitive to a global scale, which is reported
    separately as ``trace_ratio``.

    Args:
        reference: Positive-definite Hermitian reference covariance ``R``.
        candidate: Positive-definite Hermitian candidate covariance ``C`` of
            the same shape and channel order.

    Returns:
        ``channel_count``; ``trace_ratio`` = ``tr(C) / tr(R)``;
        ``normalized_shape_relative_difference`` =
        ``||C/trC - R/trR||_F / ||R/trR||_F``; ``per_channel_variance_ratio``
        (``min``, ``median``, ``max``, ``spread`` = max/min of normalized
        diagonal ratios); ``correlation_difference`` (``max_abs`` and
        ``median_abs`` off-diagonal correlation differences, and
        ``offdiagonal_magnitude_pearson``, ``None`` when undefined);
        ``generalized_eigenvalues`` of ``R^-1/2 C R^-1/2`` after trace
        normalization (``min``, ``max``, ``spread``, and ``log_rms`` of the
        natural logarithms); ``normalized_eigenvalue_spectra`` (descending
        trace-normalized spectra and their maximum relative difference); and
        ``condition_numbers``.

    Raises:
        ValueError: If either matrix is invalid or not positive definite, or
            the shapes differ.
    """
    reference_matrix = _validated_covariance(reference, "Reference covariance")
    candidate_matrix = _validated_covariance(candidate, "Candidate covariance")
    if reference_matrix.shape != candidate_matrix.shape:
        raise ValueError(
            f"Covariance shapes differ: {reference_matrix.shape} versus "
            f"{candidate_matrix.shape}."
        )
    reference_trace = float(np.sum(reference_matrix.diagonal().real))
    candidate_trace = float(np.sum(candidate_matrix.diagonal().real))
    reference_normalized = reference_matrix / reference_trace
    candidate_normalized = candidate_matrix / candidate_trace
    reference_eigenvalues, reference_vectors = np.linalg.eigh(reference_normalized)
    candidate_eigenvalues = np.linalg.eigvalsh(candidate_normalized)
    for label, eigenvalues in (
        ("Reference", reference_eigenvalues),
        ("Candidate", candidate_eigenvalues),
    ):
        if not _positive_definite(eigenvalues):
            raise ValueError(f"{label} covariance is not positive definite.")

    variance_ratio = (
        candidate_normalized.diagonal().real / reference_normalized.diagonal().real
    )
    upper = np.triu_indices(reference_matrix.shape[0], k=1)
    reference_correlation = _correlation_matrix(reference_matrix)[upper]
    candidate_correlation = _correlation_matrix(candidate_matrix)[upper]
    correlation_difference = np.abs(candidate_correlation - reference_correlation)

    # Whitening by the reference exposes shape differences along every
    # direction of channel space, including ones invisible on the diagonal.
    inverse_sqrt = (reference_vectors / np.sqrt(reference_eigenvalues)) @ (
        reference_vectors.conj().T
    )
    whitened = inverse_sqrt @ candidate_normalized @ inverse_sqrt
    generalized = np.linalg.eigvalsh(0.5 * (whitened + whitened.conj().T))
    if generalized[0] <= 0:
        raise ValueError("Generalized covariance eigenvalues are not positive.")
    reference_spectrum = reference_eigenvalues[::-1]
    candidate_spectrum = candidate_eigenvalues[::-1]
    return {
        "channel_count": int(reference_matrix.shape[0]),
        "trace_ratio": candidate_trace / reference_trace,
        "normalized_shape_relative_difference": float(
            np.linalg.norm(candidate_normalized - reference_normalized)
            / np.linalg.norm(reference_normalized)
        ),
        "per_channel_variance_ratio": {
            "min": float(np.min(variance_ratio)),
            "median": float(np.median(variance_ratio)),
            "max": float(np.max(variance_ratio)),
            "spread": float(np.max(variance_ratio) / np.min(variance_ratio)),
        },
        "correlation_difference": {
            "max_abs": float(np.max(correlation_difference)),
            "median_abs": float(np.median(correlation_difference)),
            "offdiagonal_magnitude_pearson": _pearson(
                np.abs(reference_correlation), np.abs(candidate_correlation)
            ),
        },
        "generalized_eigenvalues": {
            "min": float(generalized[0]),
            "max": float(generalized[-1]),
            "spread": float(generalized[-1] / generalized[0]),
            "log_rms": float(np.sqrt(np.mean(np.square(np.log(generalized))))),
        },
        "normalized_eigenvalue_spectra": {
            "reference": [float(value) for value in reference_spectrum],
            "candidate": [float(value) for value in candidate_spectrum],
            "max_relative_difference": float(
                np.max(np.abs(candidate_spectrum - reference_spectrum) / reference_spectrum)
            ),
        },
        "condition_numbers": {
            "reference": float(reference_eigenvalues[-1] / reference_eigenvalues[0]),
            "candidate": float(candidate_eigenvalues[-1] / candidate_eigenvalues[0]),
        },
    }


def covariance_compatibility(
    comparison: Mapping[str, Any],
    limits: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Apply pre-registered descriptive shape limits to a covariance comparison.

    Every criterion passes when its value is finite and at most its limit.
    The trace ratio is copied into the result for reporting only and never
    influences ``compatible``; a compatible result is not an absolute noise
    calibration.

    Args:
        comparison: Output of :func:`compare_covariances`.
        limits: Optional overrides for
            :data:`DEFAULT_COVARIANCE_COMPATIBILITY_LIMITS`.

    Returns:
        ``compatible``, per-criterion ``criteria`` records (``value``,
        ``limit``, ``passed``), ``failed_criteria``, the applied ``limits``,
        the reported ``trace_ratio``, and a short ``interpretation``.

    Raises:
        ValueError: If a limit name is unknown, a limit is not a finite
            nonnegative number, or the comparison lacks a required value.
    """
    applied = dict(DEFAULT_COVARIANCE_COMPATIBILITY_LIMITS)
    if limits is not None:
        unknown = sorted(set(limits) - set(applied))
        if unknown:
            raise ValueError(f"Unknown covariance compatibility limits: {unknown}.")
        for name, value in limits.items():
            limit = _optional_float(value, f"Limit {name}")
            if limit is None or limit < 0:
                raise ValueError(f"Limit {name} must be a finite nonnegative number.")
            applied[name] = limit
    sources = {
        "normalized_shape_relative_difference": ("normalized_shape_relative_difference",),
        "per_channel_variance_ratio_spread": ("per_channel_variance_ratio", "spread"),
        "max_abs_correlation_difference": ("correlation_difference", "max_abs"),
        "generalized_eigenvalue_spread": ("generalized_eigenvalues", "spread"),
    }
    criteria: dict[str, dict[str, Any]] = {}
    for name, path in sources.items():
        value = _comparison_value(comparison, path)
        criteria[name] = {
            "value": value,
            "limit": applied[name],
            "passed": bool(math.isfinite(value) and value <= applied[name]),
        }
    failed = [name for name, record in criteria.items() if not record["passed"]]
    return {
        "compatible": not failed,
        "criteria": criteria,
        "failed_criteria": failed,
        "limits": applied,
        "trace_ratio": _comparison_value(comparison, ("trace_ratio",)),
        "interpretation": _COMPATIBILITY_INTERPRETATION,
    }


def expected_white_noise_variance_ratio(
    reference_dwell_ns: float, target_dwell_ns: float
) -> float:
    """Return the target/reference noise-variance ratio for white noise.

    Under a white-noise approximation the receiver bandwidth scales as
    ``1 / dwell``, so the per-sample variance ratio is
    ``reference_dwell_ns / target_dwell_ns``. This is an approximation only:
    it ignores receiver filter shape, oversampling, digital and FFT scaling,
    and gain differences between measurements, and it never establishes an
    absolute ACS noise calibration.

    Args:
        reference_dwell_ns: Dwell time of the reference (noise) data in ns.
        target_dwell_ns: Dwell time of the target (for example ACS) data in ns.

    Returns:
        Expected ratio of target to reference per-sample variance.

    Raises:
        ValueError: If either dwell time is not a finite positive number.
    """
    values = []
    for label, value in (
        ("Reference dwell", reference_dwell_ns),
        ("Target dwell", target_dwell_ns),
    ):
        number = _optional_float(value, label)
        if number is None or number <= 0:
            raise ValueError(f"{label} must be a finite positive number of ns.")
        values.append(number)
    return values[0] / values[1]


def _byte_view(buffer: Any) -> memoryview:
    """Return a flat unsigned-byte view of a supported buffer.

    Args:
        buffer: ``bytes``, ``bytearray``, ``memoryview``, or a one-dimensional
            C-contiguous ``numpy.uint8`` array or memory map.

    Returns:
        One-dimensional memoryview with unsigned-byte items.

    Raises:
        TypeError: If the object is not a supported buffer type.
        ValueError: If an array or view is not a flat contiguous byte buffer.
    """
    if isinstance(buffer, np.ndarray):
        if buffer.ndim != 1 or buffer.dtype != np.uint8 or not buffer.flags.c_contiguous:
            raise ValueError("Array buffers must be one-dimensional C-contiguous uint8 data.")
        view = memoryview(buffer)
    elif isinstance(buffer, (bytes, bytearray, memoryview)):
        view = memoryview(buffer)
    else:
        raise TypeError(f"Unsupported TWIX buffer type: {type(buffer).__name__}.")
    if not view.c_contiguous:
        raise ValueError("TWIX buffers must be C-contiguous.")
    if view.ndim != 1 or view.format != "B":
        view = view.cast("B")
    return view


def _nonnegative_offset(value: Any, label: str) -> int:
    """Validate one byte offset.

    Args:
        value: Candidate offset.
        label: Name used in error messages.

    Returns:
        Offset as a Python integer.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is negative.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Scan walk {label} must be an integer byte offset.")
    if value < 0:
        raise ValueError(f"Scan walk {label} must be nonnegative.")
    return int(value)


def _count_flags(flag_counts: dict[str, int], mask: int) -> None:
    """Increment the count of every eval-info bit set in one mask word.

    Args:
        flag_counts: Mutable counts keyed by eval-info bit name.
        mask: Eval-info mask word 1 of one scan header.
    """
    for name, bit in _BIT.items():
        if mask & bit:
            flag_counts[name] += 1


def _scan_role(mask: int, set_counter: int) -> str:
    """Classify one regular scan by its eval-info mask.

    Args:
        mask: Eval-info mask word 1.
        set_counter: Set loop counter used to label refscan roles.

    Returns:
        ``noise``, ``refscan_set<Set>``, ``phasecor``, ``feedback``, or
        ``image``, following the documented precedence.
    """
    if mask & _BIT["NOISEADJSCAN"]:
        return "noise"
    if mask & _REFSCAN_MASK:
        return f"refscan_set{set_counter}"
    if mask & _BIT["PHASCOR"]:
        return "phasecor"
    if mask & _FEEDBACK_MASK:
        return "feedback"
    return "image"


def _walk_roles(walk: Mapping[str, Any], label: str) -> Mapping[str, Mapping[str, Any]]:
    """Return the validated role mapping of one scan-walk summary.

    Args:
        walk: Scan-walk summary.
        label: Walk name used in error messages.

    Returns:
        Mapping from role name to role summary.

    Raises:
        ValueError: If the summary or its roles are malformed.
    """
    roles = walk.get("roles") if isinstance(walk, Mapping) else None
    if not isinstance(roles, Mapping) or not all(
        isinstance(value, Mapping) for value in roles.values()
    ):
        raise ValueError(f"The {label} walk lacks a valid 'roles' mapping.")
    return roles


def _sequence_difference(
    label: str,
    role: str,
    samples: int,
    channel_ids: tuple[int, ...],
    reference: tuple[int, ...] | None,
) -> str:
    """Describe how one channel-ID sequence deviates from the reference.

    Args:
        label: Walk name.
        role: Scan role.
        samples: Samples per channel of the sequence.
        channel_ids: Observed channel IDs.
        reference: Reference channel IDs.

    Returns:
        Human-readable mismatch description.
    """
    reference = reference or ()
    position = next(
        (
            index
            for index, (observed, expected) in enumerate(zip(channel_ids, reference))
            if observed != expected
        ),
        min(len(channel_ids), len(reference)),
    )
    return (
        f"{label} role {role!r} ({samples} samples) channel IDs differ from the "
        f"reference at position {position}: {list(channel_ids)} versus "
        f"{list(reference)}."
    )


def _key_parts(key: Any) -> tuple[str, ...] | None:
    """Split one header key into its components.

    Args:
        key: mapVBVD tuple key, dotted string key, or bracketed dotted key
            such as ``a[0].b``.

    Returns:
        Key components as strings, or ``None`` for unsupported key types.
    """
    if isinstance(key, tuple):
        return tuple(str(part) for part in key)
    if isinstance(key, str):
        return tuple(part for part in _BRACKET_INDEX.sub(r".\1", key).split(".") if part)
    return None


def _normalized_header(mapping: Mapping[Any, Any] | None) -> dict[tuple[str, ...], Any]:
    """Index a header mapping by normalized key components.

    Tuple keys take precedence over string keys that normalize to the same
    components, because mapVBVD stores ASCCONV entries as tuples. Keys with an
    ``__attribute__`` component describe container metadata and are ignored.

    Args:
        mapping: ``MeasYaps``- or ``Meas``-compatible mapping, or ``None``.

    Returns:
        Mapping from key-component tuples to raw header values.

    Raises:
        TypeError: If ``mapping`` is neither ``None`` nor a mapping.
    """
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise TypeError("TWIX header data must be a mapping.")
    normalized: dict[tuple[str, ...], Any] = {}
    string_items: list[tuple[tuple[str, ...], Any]] = []
    for key, value in mapping.items():
        parts = _key_parts(key)
        if not parts or any("__attribute__" in part for part in parts):
            continue
        if isinstance(key, tuple):
            normalized[parts] = value
        else:
            string_items.append((parts, value))
    for parts, value in string_items:
        normalized.setdefault(parts, value)
    return normalized


def _clean_text(value: Any) -> str | None:
    """Convert one header value to text without surrounding ASCCONV quotes.

    Args:
        value: Raw header value.

    Returns:
        Stripped text, or ``None`` when the value is absent.
    """
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    while len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def _optional_float(value: Any, label: str) -> float | None:
    """Convert one optional numeric header value to a finite float.

    Args:
        value: Raw header value, possibly quoted text.
        label: Field name used in error messages.

    Returns:
        Finite float, or ``None`` for an absent or empty value.

    Raises:
        ValueError: If the value is Boolean, non-numeric, or not finite.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric, not Boolean.")
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    else:
        text = _clean_text(value)
        if not text:
            return None
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"{label} is not numeric: {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; found {number}.")
    return number


def _header_code(value: Any) -> int | float | str | None:
    """Normalize a header code that may be decimal, hexadecimal, or text.

    Args:
        value: Raw header value such as ``2.0`` or ``"0x2"``.

    Returns:
        Integer when the value is integral or Boolean, float for other
        numbers, stripped text when not numeric, or ``None`` when absent or
        empty.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return int(number) if number.is_integer() else number
    text = _clean_text(value)
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _number_list(value: Any) -> list[float] | str | None:
    """Parse a whitespace-separated numeric header field for recording.

    Args:
        value: Raw header value, number, or empty string.

    Returns:
        List of floats (empty for an empty field), the verbatim text when it
        is not numeric, or ``None`` when the field is absent.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return [float(value)]
    text = _clean_text(value) or ""
    try:
        return [float(token) for token in text.split()]
    except ValueError:
        return text


def _channel_number(value: Any, location: str) -> int:
    """Validate one connected ADC channel number from coil selection.

    Args:
        value: Raw ``lADCChannelConnected`` value.
        location: Entry description used in error messages.

    Returns:
        Nonnegative integer ADC channel number.

    Raises:
        ValueError: If the value is absent, non-numeric, negative, or not an
            integer.
    """
    number = _optional_float(value, f"lADCChannelConnected of the {location}")
    if number is None:
        raise ValueError(f"The {location} has no lADCChannelConnected value.")
    if number < 0 or not number.is_integer():
        raise ValueError(f"The {location} has an invalid ADC channel {value!r}.")
    return int(number)


def _is_contiguous(table: Mapping[int, str]) -> bool:
    """Report whether the ADC channel numbers of one block have no gaps.

    Args:
        table: ADC-channel-to-element mapping of one block.

    Returns:
        ``True`` when the table is nonempty and its keys form one integer run.
    """
    keys = sorted(table)
    return bool(keys) and keys == list(range(keys[0], keys[-1] + 1))


def _fft_scale_groups(meas_yaps: Mapping[Any, Any]) -> dict[tuple[str, ...], dict[int, dict[str, Any]]]:
    """Group ``aFFT_SCALE[i].<field>`` header values by key prefix and index.

    Args:
        meas_yaps: ``MeasYaps``-compatible mapping.

    Returns:
        Mapping ``prefix -> {index -> {field name -> raw value}}``, where the
        prefix holds every key component before ``aFFT_SCALE``.

    Raises:
        TypeError: If ``meas_yaps`` is not a mapping.
    """
    groups: dict[tuple[str, ...], dict[int, dict[str, Any]]] = {}
    for parts, value in _normalized_header(meas_yaps).items():
        if _FFT_SCALE_COMPONENT not in parts:
            continue
        position = parts.index(_FFT_SCALE_COMPONENT)
        if len(parts) != position + 3 or not _NUMERIC_INDEX.fullmatch(parts[position + 1]):
            continue
        group = groups.setdefault(parts[:position], {})
        group.setdefault(int(parts[position + 1]), {})[parts[position + 2]] = value
    return groups


def _coil_select_block(prefix: tuple[str, ...]) -> int | None:
    """Return the coil-select block named by an FFT-scale key prefix.

    Args:
        prefix: Key components before ``aFFT_SCALE``.

    Returns:
        Block ``b`` for ``sCoilSelectMeas.aRxCoilSelectData[b]``, else ``None``.
    """
    if len(prefix) == 3 and prefix[:2] == _COIL_SELECT_PREFIX and _NUMERIC_INDEX.fullmatch(prefix[2]):
        return int(prefix[2])
    return None


def _coil_select_fft_groups(
    groups: Mapping[tuple[str, ...], dict[int, dict[str, Any]]],
) -> dict[int, dict[int, dict[str, Any]]]:
    """Key the FFT-scale groups stored under coil-select blocks by block.

    Args:
        groups: Output of :func:`_fft_scale_groups`.

    Returns:
        Mapping ``block -> {index -> {field name -> raw value}}``; groups
        under other prefixes are omitted.

    Raises:
        ValueError: If two key prefixes name the same block.
    """
    blocks: dict[int, dict[int, dict[str, Any]]] = {}
    for prefix, entries in groups.items():
        block = _coil_select_block(prefix)
        if block is None:
            continue
        if block in blocks:
            raise ValueError(f"FFT-scale entries name coil-select block {block} twice.")
        blocks[block] = entries
    return blocks


def _fft_scale_selection(
    meas_yaps: Mapping[Any, Any],
) -> tuple[int | None, dict[int, dict[str, Any]]]:
    """Select the FFT-scale entries that describe the receive array.

    Entries under one prefix are used as stored. When several prefixes hold
    entries, every prefix must be a coil-select block and block 0 must be
    among them; the block-0 entries are selected and no blocks are merged.

    Args:
        meas_yaps: ``MeasYaps``-compatible mapping.

    Returns:
        ``(block, entries)``: the coil-select block of the selected entries
        (``None`` for another prefix or when none is stored), and the mapping
        ``index -> {field name -> raw value}``.

    Raises:
        TypeError: If ``meas_yaps`` is not a mapping.
        ValueError: If entries occur under several prefixes that are not all
            coil-select blocks including block 0, or two prefixes name the
            same block.
    """
    groups = _fft_scale_groups(meas_yaps)
    blocks = _coil_select_fft_groups(groups)
    if len(groups) <= 1:
        prefix, entries = next(iter(groups.items()), ((), {}))
        return _coil_select_block(prefix), entries
    if len(blocks) != len(groups) or 0 not in blocks:
        raise ValueError(
            "FFT-scale entries occur under several header prefixes that are not all "
            f"coil-select blocks including block 0: {sorted(groups)}."
        )
    return 0, blocks[0]


def _fft_factor_list(entries: Mapping[int, Mapping[str, Any]], label: str) -> list[float]:
    """Validate and order the ``flFactor`` values of one FFT-scale group.

    Args:
        entries: Mapping ``index -> {field name -> raw value}``.
        label: Group name used in error messages.

    Returns:
        Finite factors ordered by index ``0..N-1``.

    Raises:
        ValueError: If indices are not contiguous from zero or a factor is
            empty or not finite.
    """
    factors = {
        index: fields["flFactor"] for index, fields in entries.items() if "flFactor" in fields
    }
    if sorted(factors) != list(range(len(factors))):
        raise ValueError(f"{label} factor indices are not contiguous from zero: {sorted(factors)}.")
    result: list[float] = []
    for index in range(len(factors)):
        value = _optional_float(factors[index], f"{label} factor {index}")
        if value is None:
            raise ValueError(f"{label} factor {index} is empty.")
        result.append(value)
    return result


def _fft_flag_list(
    entries: Mapping[int, Mapping[str, Any]], count: int, label: str
) -> list[bool | None]:
    """Return the recorded ``bValid`` flag of each factor in one FFT-scale group.

    Flags may be Booleans, ``true``/``false`` text, or finite decimal or
    hexadecimal codes such as ``0x1``, as mapVBVD returns them.

    Args:
        entries: Mapping ``index -> {field name -> raw value}``.
        count: Number of factors in the group.
        label: Group name used in error messages.

    Returns:
        One Boolean per factor index, or ``None`` when no flag is recorded.

    Raises:
        ValueError: If a recorded flag is not Boolean-like.
    """
    flags: list[bool | None] = []
    for index in range(count):
        value = entries.get(index, {}).get("bValid")
        text = _clean_text(value)
        if value is None or not text:
            flags.append(None)
        elif isinstance(value, (bool, np.bool_)):
            flags.append(bool(value))
        elif text.lower() in {"true", "false"}:
            flags.append(text.lower() == "true")
        else:
            code = _header_code(value)
            if isinstance(code, str) or (isinstance(code, float) and not math.isfinite(code)):
                raise ValueError(f"{label} bValid {index} is not Boolean-like: {value!r}.")
            flags.append(bool(code))
    return flags


def _validated_covariance(matrix: np.ndarray, label: str) -> np.ndarray:
    """Validate a covariance matrix and return its exact Hermitian part.

    Args:
        matrix: Candidate square covariance.
        label: Name used in error messages.

    Returns:
        complex128 Hermitian matrix.

    Raises:
        ValueError: If the matrix is not numeric, square with at least two
            channels, finite, Hermitian within tolerance, or has a
            nonpositive diagonal.
    """
    values = np.asarray(matrix)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or values.shape[0] < 2
        or not np.issubdtype(values.dtype, np.number)
    ):
        raise ValueError(f"{label} must be a numeric square matrix with >= 2 channels.")
    values = values.astype(np.complex128)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values.")
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    if float(np.max(np.abs(values - values.conj().T))) > _HERMITIAN_RELATIVE_TOLERANCE * scale:
        raise ValueError(f"{label} is not Hermitian.")
    hermitian = 0.5 * (values + values.conj().T)
    if np.any(hermitian.diagonal().real <= 0):
        raise ValueError(f"{label} must have a positive diagonal.")
    return hermitian


def _positive_definite(eigenvalues: np.ndarray) -> bool:
    """Apply the numerical positive-definiteness rule to sorted eigenvalues.

    Args:
        eigenvalues: Ascending real eigenvalues of a Hermitian matrix.

    Returns:
        ``True`` when the smallest eigenvalue exceeds ``N * eps`` times the
        positive largest eigenvalue.
    """
    largest = float(eigenvalues[-1])
    tolerance = eigenvalues.size * np.finfo(np.float64).eps * largest
    return bool(largest > 0 and float(eigenvalues[0]) > tolerance)


def _correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """Convert a validated covariance to its complex correlation matrix.

    Args:
        covariance: Hermitian covariance with a positive diagonal.

    Returns:
        Complex correlation matrix with an exact unit diagonal.
    """
    deviations = np.sqrt(covariance.diagonal().real)
    correlation = covariance / np.outer(deviations, deviations)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return the Pearson correlation of two equally long real vectors.

    Args:
        left: Reference values.
        right: Candidate values.

    Returns:
        Pearson coefficient, or ``None`` with fewer than two values or zero
        variance in either vector.
    """
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _comparison_value(comparison: Mapping[str, Any], path: tuple[str, ...]) -> float:
    """Read one numeric value from a covariance comparison.

    Args:
        comparison: Output of :func:`compare_covariances`.
        path: Nested key path.

    Returns:
        Value as float; non-finite values are preserved.

    Raises:
        ValueError: If the value is absent or not numeric.
    """
    value: Any = comparison
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Covariance comparison lacks {'.'.join(path)}.")
        value = value[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"Covariance comparison value {'.'.join(path)} is not numeric.")
    return float(value)
