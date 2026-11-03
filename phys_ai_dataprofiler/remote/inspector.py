"""
Metadata-only inspectors that work over a `RangeFile` (local or remote).

They produce the same result types as the download-based inspectors
(`McapInspectionResult`, `MatInspectionResult`, `RichProfile`) but read only
the bytes that hold metadata:

MCAP  header (start), footer + summary section (end) and, for timing stats,
      the MessageIndex records that follow each chunk. Message payloads are
      never read, so for a typical file this is well under 1% of its size.
      Files with no summary (truncated recordings, or written without one)
      are walked record by record instead: chunk headers and message indexes
      are read, and a chunk body only when it is the sole source of a
      channel definition or of timestamps. The walk is capped by a budget.

MAT   128-byte header, then a walk over the top-level data elements reading
      only each element's tag and the array header (flags, dims, name, struct
      fields). For compressed (v7) elements the first few KB are inflated
      just far enough to reach the name. Then, within a byte budget, the
      floating-point variables are fetched and decoded to count NaNs.
"""

from __future__ import annotations

import io
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path


from phys_ai_dataprofiler.inspectors.mcap import (
    MCAP_MAGIC,
    ChannelInfo,
    FooterInfo,
    HeaderInfo,
    McapInspectionResult,
    SchemaInfo,
)
from phys_ai_dataprofiler.inspectors.mat import (
    MatFileHeader,
    MatFileInspector,
    MatInspectionResult,
    VariableInfo,
    H5PY_AVAILABLE,
)
from phys_ai_dataprofiler.inspectors.rich import Finding, RichProfile, mat_channel_profile, profile_mcap_channels
from phys_ai_dataprofiler.remote.range_file import RangeFile

try:
    from mcap.reader import SeekingReader, FOOTER_SIZE, MAGIC_SIZE
    from mcap.stream_reader import StreamReader, breakup_chunk
    from mcap.records import Channel, Chunk, Footer, Message, MessageIndex, Schema
    from mcap.data_stream import ReadDataStream
    from mcap.opcode import Opcode
    from mcap.exceptions import EndOfFile
    MCAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    MCAP_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Format sniffing
# --------------------------------------------------------------------------- #

def sniff_format(rf: RangeFile, name: str = "") -> str:
    head = rf.read_range(0, 128)
    if head[: len(MCAP_MAGIC)] == MCAP_MAGIC:
        return "mcap"
    if head[:4] == b"PAR1":
        return "parquet"
    if MatFileInspector._detect_format(head) in ("v5", "v7.3"):
        return "mat"
    ext = Path(name).suffix.lower()
    if ext == ".mcap":
        return "mcap"
    if ext == ".mat":
        return "mat"
    if ext == ".parquet":
        return "parquet"
    return "unknown"


# --------------------------------------------------------------------------- #
# MCAP
# --------------------------------------------------------------------------- #

def _read_footer(rf: RangeFile):
    rf.seek(-(FOOTER_SIZE + MAGIC_SIZE), io.SEEK_END)
    rec = next(StreamReader(rf, skip_magic=True).records)
    return rec if isinstance(rec, Footer) else None


def _parse_message_indexes(buf: bytes) -> list[MessageIndex]:
    out: list[MessageIndex] = []
    try:
        for rec in StreamReader(io.BytesIO(buf), skip_magic=True).records:
            if isinstance(rec, MessageIndex):
                out.append(rec)
    except EndOfFile:
        pass
    return out


def _channel_info(ch) -> ChannelInfo:
    return ChannelInfo(id=ch.id, topic=ch.topic, message_encoding=ch.message_encoding,
                       schema_id=ch.schema_id, metadata=ch.metadata)


def _schema_info(sc) -> SchemaInfo:
    return SchemaInfo(id=sc.id, name=sc.name, encoding=sc.encoding, data=sc.data)


def inspect_mcap_remote(
    rf: RangeFile,
    msgidx_budget_bytes: int = 64 * 1024 * 1024,
    workers: int = 16,
    log=None,
) -> tuple[McapInspectionResult, RichProfile]:
    """Header/footer/summary plus timing stats from message indexes only."""
    log = log or (lambda m: None)
    t0 = time.time()
    url = getattr(rf, "url", str(getattr(rf, "path", "")))
    size = rf.size
    magic = rf.read_range(0, len(MCAP_MAGIC))

    if not MCAP_AVAILABLE:
        res = McapInspectionResult(url=url, is_mcap=magic == MCAP_MAGIC, magic=magic,
                                   size_bytes=size, error="mcap library not available")
        return res, _empty_profile(url, "mcap", size, [Finding("error", "missing_library", None, "mcap library not available")], t0)

    if magic != MCAP_MAGIC:
        res = McapInspectionResult(url=url, is_mcap=False, magic=magic, size_bytes=size)
        return res, _empty_profile(url, "mcap", size, [Finding("error", "bad_magic", None, "not an MCAP file")], t0)

    header_info = footer_info = None
    findings: list[Finding] = []

    try:
        rf.seek(0)
        reader = SeekingReader(rf)
        header = reader.get_header()
        header_info = HeaderInfo(profile=header.profile, library=header.library)
    except Exception as exc:
        res = McapInspectionResult(url=url, is_mcap=True, magic=magic, size_bytes=size,
                                   error=f"Failed to parse with mcap library: {exc}")
        return res, _empty_profile(url, "mcap", size, [Finding("error", "parse_error", None, str(exc))], t0)

    def build(**kw):
        return _mcap_result(rf, url, header_info, footer_info, findings=findings, t0=t0, **kw)

    if rf.read_range(max(0, size - len(MCAP_MAGIC)), size) != MCAP_MAGIC:
        findings.append(Finding("warn", "truncated", None,
                                "no closing magic or footer: the recording was cut off (crash, or still being "
                                "written); profiled by walking its records"))
        walk = _walk_mcap(rf, msgidx_budget_bytes, log, findings)
        return build(chans=walk.chans, schemas=walk.schemas, segments=walk.segments(), counts=None,
                     duration_s=walk.duration_s, indexed=False)

    try:
        footer = _read_footer(rf)
        if footer is not None:
            footer_info = FooterInfo(summary_start=footer.summary_start,
                                     summary_offset_start=footer.summary_offset_start,
                                     summary_crc=footer.summary_crc)
            if footer.summary_start:
                # One request for the whole summary section instead of many tiny ones.
                rf.prefetch(footer.summary_start, size)

        summary = reader.get_summary()
    except Exception as exc:
        res = McapInspectionResult(url=url, is_mcap=True, magic=magic, size_bytes=size,
                                   header=header_info, footer=footer_info,
                                   error=f"Failed to parse with mcap library: {exc}")
        return res, _empty_profile(url, "mcap", size, [Finding("error", "parse_error", None, str(exc))], t0)

    if summary is None:
        findings.append(Finding("info", "unindexed_mcap", None,
                                "no summary section; profiled by walking its records"))
        walk = _walk_mcap(rf, msgidx_budget_bytes, log, findings)
        return build(chans=walk.chans, schemas=walk.schemas, segments=walk.segments(), counts=None,
                     duration_s=walk.duration_s, indexed=False)

    chans = {cid: _channel_info(ch) for cid, ch in summary.channels.items()}
    schemas = {sid: _schema_info(sc) for sid, sc in summary.schemas.items()}
    st = summary.statistics
    counts = st.channel_message_counts if st is not None and st.channel_message_counts else None
    duration_s = None
    if st is not None and st.message_end_time >= st.message_start_time and st.message_count:
        duration_s = (st.message_end_time - st.message_start_time) / 1e9
    elif summary.chunk_indexes:
        starts = [c.message_start_time for c in summary.chunk_indexes]
        ends = [c.message_end_time for c in summary.chunk_indexes]
        duration_s = (max(ends) - min(starts)) / 1e9

    if not summary.chunk_indexes and (st is None or st.message_count):
        # Unchunked file: no message indexes exist, so timestamps come from walking message headers.
        # The summary stays authoritative for channels, counts and duration.
        walk = _walk_mcap(rf, msgidx_budget_bytes, log, findings)
        return build(chans=chans, schemas=schemas, segments=walk.segments(), counts=counts,
                     duration_s=duration_s if duration_s is not None else walk.duration_s, indexed=True)

    # ---- timestamps from message indexes -----------------------------------
    ranges: list[tuple[int, int]] = []
    for ci in summary.chunk_indexes:
        if ci.message_index_offsets and ci.message_index_length:
            start = min(ci.message_index_offsets.values())
            ranges.append((start, start + ci.message_index_length))
    total_idx = sum(e - s for s, e in ranges)
    sampled = False
    if total_idx > msgidx_budget_bytes and ranges:
        # Keep an evenly spaced subset of chunks that fits the budget.
        keep = max(1, int(len(ranges) * msgidx_budget_bytes / total_idx))
        step = len(ranges) / keep
        ranges = [ranges[int(i * step)] for i in range(keep)]
        sampled = True

    # One run per channel, or, when chunks are sampled, one run per chunk: sampled chunks
    # are not adjacent, so the jump between them is not a gap.
    segments: dict[int, list[list[int]]] = {}
    if ranges:
        log(f"    fetching {len(ranges)} message-index block(s), {sum(e - s for s, e in ranges) / 1024:.0f} KB")
        flat: dict[int, list[int]] = {}
        for buf in rf.fetch_many(ranges, workers=workers):
            for mi in _parse_message_indexes(buf):
                ts = [t for t, _ in mi.records]
                if not ts:
                    continue
                if sampled:
                    segments.setdefault(mi.channel_id, []).append(sorted(ts))
                else:
                    flat.setdefault(mi.channel_id, []).extend(ts)
        if not sampled:
            segments = {cid: [sorted(ts)] for cid, ts in flat.items()}

    if not ranges and summary.chunk_indexes:
        findings.append(Finding("info", "no_message_index", None,
                                "chunks carry no message indexes; rates are averages, jitter/gaps unavailable"))
    if sampled:
        findings.append(Finding("info", "sampled_message_index", None,
                                f"message indexes sampled ({len(ranges)} chunks) to stay under "
                                f"{msgidx_budget_bytes // (1024 * 1024)} MB; jitter/gap stats are estimates"))
    return build(chans=chans, schemas=schemas, segments=segments, counts=counts,
                 duration_s=duration_s, indexed=True)


def _mcap_result(
    rf: RangeFile, url: str, header_info, footer_info, *,
    chans: dict[int, ChannelInfo], schemas: dict[int, SchemaInfo], segments: dict[int, list[list[int]]],
    counts: dict[int, int] | None, duration_s: float | None, indexed: bool,
    findings: list[Finding], t0: float,
) -> tuple[McapInspectionResult, RichProfile]:
    """Build the result and profile from channels, timestamp runs and (when known) summary counts."""
    channel_stats: dict[int, dict] = {}
    for cid in sorted(chans.keys() | segments.keys()):
        ch = chans.get(cid)
        schema = schemas.get(ch.schema_id) if ch else None
        segs = segments.get(cid, [])
        count = (counts or {}).get(cid)
        if count is None:
            count = sum(len(s) for s in segs)
        if count == 0 and indexed:
            continue
        channel_stats[cid] = {
            "topic": ch.topic if ch else f"<channel {cid}>",
            "schema": schema.name if schema else "unknown",
            "segments": segs,
            "count": count,
        }
    profiles, ch_findings = profile_mcap_channels(channel_stats)

    # Channels without per-message timestamps still get an average rate.
    if duration_s:
        for p in profiles:
            if p.rate_hz is None and p.msg_count > 1:
                p.rate_hz = p.msg_count / duration_s
    unnamed = [cid for cid in segments if cid not in chans]
    if unnamed:
        ch_findings.append(Finding("warn", "unknown_channels", None,
                                   f"{len(unnamed)} channel id(s) have messages but no channel record was read"))

    res = McapInspectionResult(
        url=url, is_mcap=True, magic=MCAP_MAGIC, size_bytes=rf.size,
        header=header_info, footer=footer_info,
        channels=list(chans.values()) or None, schemas=list(schemas.values()) or None,
        topics=sorted({c.topic for c in chans.values()}) or None,
    )
    prof = RichProfile(
        path=url, format="mcap", size_bytes=rf.size, indexed=indexed, duration_s=duration_s,
        time_base="absolute_epoch_ns" if duration_s is not None else None,
        channels=profiles, findings=findings + ch_findings, inspect_cost_s=time.time() - t0,
    )
    return res, prof


@dataclass
class _McapWalk:
    schemas: dict[int, SchemaInfo] = field(default_factory=dict)
    chans: dict[int, ChannelInfo] = field(default_factory=dict)
    timestamps: dict[int, list[int]] = field(default_factory=dict)
    chunk_bounds: list[tuple[int, int]] = field(default_factory=list)

    def segments(self) -> dict[int, list[list[int]]]:
        return {cid: [sorted(ts)] for cid, ts in self.timestamps.items()}

    @property
    def duration_s(self) -> float | None:
        bounds = self.chunk_bounds + [(min(ts), max(ts)) for ts in self.timestamps.values() if ts]
        if not bounds:
            return None
        return (max(e for _, e in bounds) - min(s for s, _ in bounds)) / 1e9


def _walk_mcap(rf: RangeFile, budget: int, log, findings: list[Finding]) -> _McapWalk:
    """Walk the records of an MCAP that has no usable summary or no chunks.

    Reads each record's 9-byte opcode/length, chunk headers (for time bounds)
    and the MessageIndex records after each chunk (for timestamps). A chunk
    body is fetched and decompressed only when it holds the definition of a
    channel not seen yet, or when the chunk has no message index. Unchunked
    files are read message header by message header. Stops when `budget`
    bytes have been read; fidelity notices are appended to `findings`.
    """
    size = rf.size
    start_bytes = rf.bytes_fetched
    walk = _McapWalk()
    n_chunks = undecoded = 0
    decode_error = None
    stopped = None
    off = len(MCAP_MAGIC)

    def over_budget(extra: int = 0) -> bool:
        return rf.bytes_fetched - start_bytes + extra > budget

    def record_header(p: int) -> tuple[int, int] | None:
        if p + 9 > size:
            return None
        hdr = rf.read_range(p, p + 9)
        return hdr[0], struct.unpack("<Q", hdr[1:9])[0]

    def read_record(cls, start: int, end: int):
        return cls.read(ReadDataStream(io.BytesIO(rf.read_range(start, end))))

    def take(records, messages: bool) -> None:
        for r in records:
            if isinstance(r, Schema):
                walk.schemas[r.id] = _schema_info(r)
            elif isinstance(r, Channel):
                walk.chans[r.id] = _channel_info(r)
            elif messages and isinstance(r, Message):
                walk.timestamps.setdefault(r.channel_id, []).append(r.log_time)

    def read_message_indexes(p: int) -> tuple[int, set[int]]:
        """Read the MessageIndex records starting at p. Returns (offset after them, channels indexed)."""
        seen: set[int] = set()
        while (nxt := record_header(p)) is not None and nxt[0] == Opcode.MESSAGE_INDEX:
            if p + 9 + nxt[1] > size:
                break
            mi = read_record(MessageIndex, p + 9, p + 9 + nxt[1])
            if mi.records:
                seen.add(mi.channel_id)
                walk.timestamps.setdefault(mi.channel_id, []).extend(t for t, _ in mi.records)
            p += 9 + nxt[1]
        return p, seen

    log("    walking records")
    while True:
        if over_budget():
            stopped = "budget"
            break
        rec = record_header(off)
        if rec is None:
            break
        op, length = rec
        body, end = off + 9, off + 9 + length
        if op in (Opcode.DATA_END, Opcode.FOOTER):
            break
        if end > size:
            stopped = "truncated"
            break
        if op in (Opcode.SCHEMA, Opcode.CHANNEL):
            take([read_record(Schema if op == Opcode.SCHEMA else Channel, body, end)], messages=False)
        elif op == Opcode.MESSAGE and length >= 22:
            channel_id, _seq, log_time = struct.unpack("<HIQ", rf.read_range(body, body + 14))
            walk.timestamps.setdefault(channel_id, []).append(log_time)
        elif op == Opcode.CHUNK:
            n_chunks += 1
            ch_start, ch_end = struct.unpack("<QQ", rf.read_range(body, body + 16))
            if ch_end >= ch_start:
                walk.chunk_bounds.append((ch_start, ch_end))
            # Message indexes for this chunk follow it directly.
            next_off, seen = read_message_indexes(end)
            if not seen or seen - walk.chans.keys():
                if over_budget(length):
                    undecoded += 1
                else:
                    try:
                        take(breakup_chunk(read_record(Chunk, body, end)), messages=not seen)
                    except Exception as exc:
                        undecoded += 1
                        decode_error = decode_error or str(exc)
            end = next_off
        off = end

    if stopped == "budget":
        findings.append(Finding("info", "partial_scan", None,
                                f"walk stopped at {off / size:.0%} of the file to stay under "
                                f"{budget // (1024 * 1024)} MB; statistics not in the summary cover that part only"))
    elif stopped == "truncated":
        findings.append(Finding("warn", "truncated", None,
                                f"last record at offset {off} runs past the end of the file; "
                                "messages in it are not counted"))
    if undecoded:
        detail = f"{undecoded} of {n_chunks} chunk(s) could not be decoded"
        detail += f" ({decode_error})" if decode_error else " within the budget"
        findings.append(Finding("info", "chunks_not_decoded", None,
                                detail + "; their channel names or timestamps may be missing"))
    return walk


# --------------------------------------------------------------------------- #
# MAT (v5 / v7 container; v7.3 via h5py when available)
# --------------------------------------------------------------------------- #

mi_INT8, mi_UINT8, mi_INT16, mi_UINT16, mi_INT32, mi_UINT32 = 1, 2, 3, 4, 5, 6
mi_MATRIX, mi_COMPRESSED, mi_UTF8 = 14, 15, 16

MX_CLASS = {
    1: "cell", 2: "struct", 3: "object", 4: "char", 5: "sparse",
    6: "float64", 7: "float32", 8: "int8", 9: "uint8", 10: "int16", 11: "uint16",
    12: "int32", 13: "uint32", 14: "int64", 15: "uint64",
    16: "function_handle", 17: "opaque",
}

_PROBE_INITIAL = 4096
_PROBE_MAX = 4 * 1024 * 1024


class _NeedMoreData(Exception):
    pass


def _tag(buf: bytes, p: int, e: str) -> tuple[int, int, int, int]:
    """Read a sub-element tag at p. Returns (type, nbytes, data_offset, next_offset)."""
    if p + 4 > len(buf):
        raise _NeedMoreData
    word = struct.unpack(e + "I", buf[p:p + 4])[0]
    small_n = (word >> 16) if e == "<" else (word & 0xFFFF)
    small_t = (word & 0xFFFF) if e == "<" else (word >> 16)
    if small_n:  # small data element: type+length packed in 4 bytes, data in next 4
        return small_t, small_n, p + 4, p + 8
    if p + 8 > len(buf):
        raise _NeedMoreData
    n = struct.unpack(e + "I", buf[p + 4:p + 8])[0]
    return word, n, p + 8, p + 8 + ((n + 7) // 8) * 8


def _parse_matrix_header(buf: bytes, e: str) -> dict:
    """Parse the leading sub-elements of a miMATRIX element (buf starts at its tag)."""
    mtype, nbytes, p, _ = _tag(buf, 0, e)
    if mtype != mi_MATRIX:
        raise ValueError(f"expected miMATRIX, got type {mtype}")
    info: dict = {"stored_bytes": nbytes}
    if nbytes == 0:  # empty element (e.g. function handle placeholder)
        info.update(name="", class_code=0, dims=(), complex=False, logical=False, nzmax=0, field_names=None)
        return info

    # Array flags
    _, n, d, p = _tag(buf, p, e)
    if d + 8 > len(buf):
        raise _NeedMoreData
    flags, nzmax = struct.unpack(e + "II", buf[d:d + 8])
    info["class_code"] = flags & 0xFF
    info["complex"] = bool(flags & 0x0800)
    info["global"] = bool(flags & 0x0400)
    info["logical"] = bool(flags & 0x0200)
    info["nzmax"] = nzmax

    # Dimensions
    _, n, d, p = _tag(buf, p, e)
    if d + n > len(buf):
        raise _NeedMoreData
    info["dims"] = tuple(struct.unpack(e + f"{n // 4}i", buf[d:d + n]))

    # Name
    _, n, d, p = _tag(buf, p, e)
    if d + n > len(buf):
        raise _NeedMoreData
    info["name"] = buf[d:d + n].decode("latin-1").rstrip("\x00")

    info["field_names"] = None
    if info["class_code"] in (2, 3):  # struct / object
        if info["class_code"] == 3:  # class name precedes the fields
            _, n, d, p = _tag(buf, p, e)
            if d + n > len(buf):
                raise _NeedMoreData
            info["class_name"] = buf[d:d + n].decode("latin-1").rstrip("\x00")
        _, n, d, p = _tag(buf, p, e)            # field name length
        if d + n > len(buf):
            raise _NeedMoreData
        flen = struct.unpack(e + "i", buf[d:d + 4])[0]
        _, n, d, p = _tag(buf, p, e)            # field names
        if d + n > len(buf):
            raise _NeedMoreData
        raw = buf[d:d + n]
        names = [raw[i:i + flen].decode("latin-1").rstrip("\x00") for i in range(0, n, flen)] if flen else []
        info["field_names"] = [x for x in names if x]
    return info


def _read_element_header(rf: RangeFile, off: int, mtype: int, nbytes: int, e: str) -> dict:
    """Read just enough of a top-level element to parse its array header."""
    probe = min(nbytes, _PROBE_INITIAL)
    while True:
        raw = rf.read_range(off + 8, off + 8 + probe)
        try:
            if mtype == mi_COMPRESSED:
                inflated = zlib.decompressobj().decompress(raw, max(_PROBE_MAX, 64 * 1024))
                info = _parse_matrix_header(inflated, e)
                info["compressed_bytes"] = nbytes
            else:
                # Uncompressed: the outer tag *is* the miMATRIX tag.
                info = _parse_matrix_header(struct.pack(e + "II", mtype, nbytes) + raw, e)
            return info
        except (_NeedMoreData, struct.error, IndexError, zlib.error):
            if probe >= nbytes or probe >= _PROBE_MAX:
                raise
            probe = min(nbytes, probe * 4, _PROBE_MAX)


def _variable_from_header(info: dict) -> VariableInfo:
    cls = info["class_code"]
    dtype = MX_CLASS.get(cls, f"class_{cls}")
    if info.get("logical"):
        dtype = "bool"
    elif info.get("complex") and dtype in ("float64", "float32"):
        dtype = "complex128" if dtype == "float64" else "complex64"
    size = info.get("stored_bytes", 0)
    name = info["name"]
    if not name and cls == 9:
        # MATLAB's unnamed uint8 "subsystem data" element; scipy uses this label.
        name = "__function_workspace__"
    return VariableInfo(
        name=name,
        dtype=dtype,
        shape=tuple(info["dims"]),
        size_bytes=size,
        is_sparse=cls == 5,
        is_complex=bool(info.get("complex")),
        is_struct=cls in (2, 3),
        is_cell=cls == 1,
        field_names=info.get("field_names"),
    )


_NAN_DTYPES = ("float64", "float32", "complex128", "complex64", "double", "single")


def _nan_candidate(v: VariableInfo) -> bool:
    """Dense floating-point variables, the only ones that can hold NaN."""
    return v.dtype in _NAN_DTYPES and not (v.is_sparse or v.is_struct or v.is_cell)


def _count_nan(value) -> int | None:
    try:
        import numpy as np
        arr = np.asarray(value)
        if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating)):
            return None
        return int(np.count_nonzero(np.isnan(arr)))
    except Exception:
        return None


def _mat_nan_scan(items: list, cost, scan, budget: int, log) -> tuple[dict[str, int], list[Finding]]:
    """Run `scan` on the candidates that fit `budget` (first fit, by `cost`). Returns (counts, findings).

    `scan(picked) -> (counts, error)`. A budget of 0 disables the scan.
    """
    if not items:
        return {}, []
    if budget <= 0:
        return {}, [Finding("info", "nan_check_skipped", None,
                            "NaN scan disabled (--nan-budget-mb 0); float variables were not read")]
    picked, used = [], 0
    for item in items:
        if used + cost(item) <= budget:
            picked.append(item)
            used += cost(item)
    if picked:
        log(f"    NaN scan: {len(picked)} float variable(s), {used / 1024:.0f} KB")
    try:
        counts, error = scan(picked)
    except Exception as exc:
        counts, error = {}, f"NaN scan failed: {exc}"
    findings = [Finding("warn", "nan_values", name, f"{n} NaN values found") for name, n in counts.items() if n]
    if len(picked) < len(items):
        findings.append(Finding("info", "nan_check_partial", None,
                                f"NaN scan covered {len(picked)} of {len(items)} float variable(s); the rest exceed "
                                f"the {budget // (1024 * 1024)} MB NaN budget (--nan-budget-mb)"))
    if error:
        findings.append(Finding("warn", "nan_check_failed", None, error))
    return counts, findings


def _scan_v5_elements(rf: RangeFile, head: bytes, picked: list[tuple[int, int, VariableInfo]],
                      workers: int) -> tuple[dict[str, int], str | None]:
    """Fetch whole v5/v7 elements and count NaNs with scipy (the 128-byte header makes each a valid file)."""
    import scipy.io
    counts: dict[str, int] = {}
    error = None
    # Small variables are often already in the block cache from the header walk; fetch_many reuses them.
    datas = rf.fetch_many([(off, off + 8 + nbytes) for off, nbytes, _ in picked], workers=workers)
    for (_, _, var), raw in zip(picked, datas):
        try:
            mat = scipy.io.loadmat(io.BytesIO(head + raw), squeeze_me=False)
            values = [v for k, v in mat.items() if not k.startswith("__")]
            n = _count_nan(values[0]) if values else None
            if n is not None:
                counts[var.name] = n
        except Exception as exc:
            error = error or f"{var.name}: {exc}"
    return counts, error


def _scan_hdf5_datasets(rf: RangeFile, picked: list[VariableInfo]) -> tuple[dict[str, int], str | None]:
    """Read float datasets of a v7.3 file through h5py (range reads) and count NaNs."""
    import h5py
    counts: dict[str, int] = {}
    error = None
    rf.seek(0)
    with h5py.File(rf, "r") as h5:
        for v in picked:
            try:
                data = h5[v.name][()]
                if data.dtype.names and {"real", "imag"} <= set(data.dtype.names):
                    data = data["real"] + 1j * data["imag"]
                n = _count_nan(data)
                if n is not None:
                    counts[v.name] = n
            except Exception as exc:
                error = error or f"{v.name}: {exc}"
    return counts, error


def inspect_mat_remote(rf: RangeFile, log=None, max_variables: int = 100_000,
                       nan_budget_bytes: int = 32 * 1024 * 1024, workers: int = 16,
                       ) -> tuple[MatInspectionResult, RichProfile]:
    log = log or (lambda m: None)
    t0 = time.time()
    url = getattr(rf, "url", str(getattr(rf, "path", "")))
    size = rf.size
    head = rf.read_range(0, 128)
    fmt = MatFileInspector._detect_format(head)

    def fail(msg: str, is_mat: bool = True, header=None, sev="error", ftype="parse_error"):
        res = MatInspectionResult(url=url, is_matfile=is_mat, file_format=fmt, size_bytes=size,
                                  header=header, error=msg)
        return res, _empty_profile(url, "mat", size, [Finding(sev, ftype, None, msg)], t0)

    if fmt is None:
        return fail("Not a valid MATLAB file", is_mat=False)

    if fmt == "v7.3":
        if not H5PY_AVAILABLE:
            return fail("h5py library required for MAT-File v7.3 format")
        try:
            data = MatFileInspector._parse_hdf5_mat(rf)  # h5py reads only HDF5 metadata
        except Exception as exc:
            return fail(f"Failed to parse MAT v7.3 over HTTP: {exc}")
        if data.get("error"):
            return fail(data["error"])
        res = MatInspectionResult(url=url, is_matfile=True, file_format=fmt, size_bytes=size, **data)
        counts, findings = _mat_nan_scan(
            [v for v in res.variables or [] if _nan_candidate(v)], lambda v: v.size_bytes,
            lambda picked: _scan_hdf5_datasets(rf, picked), nan_budget_bytes, log)
        return res, _mat_profile(url, size, res.variables or [], t0, findings, counts)

    if fmt == "v4":
        return fail("MAT v4 files are not supported in remote mode", ftype="unsupported")

    # ---- v5 / v7 ---------------------------------------------------------------
    endian_tag = head[126:128]
    e = "<" if endian_tag == b"IM" else ">" if endian_tag == b"MI" else "<"
    description = head[:116].decode("ascii", errors="ignore").replace("\x00", "").strip()
    version = struct.unpack(e + "H", head[124:126])[0]
    header = MatFileHeader(description=description, version=f"0x{version:04x}",
                           endian="little" if e == "<" else "big")

    variables: list[VariableInfo] = []
    names: list[str] = []
    findings: list[Finding] = []
    float_elements: list[tuple[int, int, VariableInfo]] = []
    off = 128
    while off + 8 <= size and len(variables) < max_variables:
        mtype, nbytes = struct.unpack(e + "II", rf.read_range(off, off + 8))
        if mtype not in (mi_MATRIX, mi_COMPRESSED):
            # Try 8-byte alignment once, then give up on the walk.
            aligned = ((off + 7) // 8) * 8
            if aligned != off and aligned + 8 <= size:
                mtype2, nbytes2 = struct.unpack(e + "II", rf.read_range(aligned, aligned + 8))
                if mtype2 in (mi_MATRIX, mi_COMPRESSED):
                    off, mtype, nbytes = aligned, mtype2, nbytes2
                else:
                    findings.append(Finding("warn", "walk_stopped", None,
                                            f"unexpected element type {mtype} at offset {off}; stopped listing"))
                    break
            else:
                findings.append(Finding("warn", "walk_stopped", None,
                                        f"unexpected element type {mtype} at offset {off}; stopped listing"))
                break
        if off + 8 + nbytes > size:
            findings.append(Finding("warn", "truncated", None,
                                    f"element at offset {off} claims {nbytes} bytes past end of file"))
            break
        try:
            info = _read_element_header(rf, off, mtype, nbytes, e)
            var = _variable_from_header(info)
            variables.append(var)
            names.append(var.name)
            if _nan_candidate(var):
                float_elements.append((off, nbytes, var))
        except Exception as exc:
            findings.append(Finding("warn", "element_parse_error", None,
                                    f"could not parse element header at offset {off}: {exc}"))
        off += 8 + nbytes
        if len(variables) % 200 == 0 and variables:
            log(f"    ... {len(variables)} variables so far")

    counts, nan_findings = _mat_nan_scan(
        float_elements, lambda el: el[1] + 8,
        lambda picked: _scan_v5_elements(rf, head, picked, workers), nan_budget_bytes, log)
    findings += nan_findings

    res = MatInspectionResult(
        url=url, is_matfile=True, file_format=fmt, size_bytes=size, header=header,
        variables=variables or None, variable_names=sorted(names) or None,
    )
    prof = _mat_profile(url, size, variables, t0, findings, counts)
    return res, prof


def _mat_profile(url: str, size: int, variables: list[VariableInfo], t0: float,
                 findings: list[Finding], nan_counts: dict[str, int]) -> RichProfile:
    channels = [mat_channel_profile(v.name, tuple(v.shape), v.dtype, nan_counts.get(v.name, 0)) for v in variables]
    return RichProfile(path=url, format="mat", size_bytes=size, indexed=False, duration_s=None,
                       time_base=None, channels=channels, findings=findings,
                       inspect_cost_s=time.time() - t0)


def _empty_profile(url: str, fmt: str, size: int, findings: list[Finding], t0: float) -> RichProfile:
    return RichProfile(path=url, format=fmt, size_bytes=size, indexed=False, duration_s=None,
                       time_base=None, channels=[], findings=findings, inspect_cost_s=time.time() - t0)


# --------------------------------------------------------------------------- #
# CLI for quick checks:  python -m phys_ai_dataprofiler.remote.inspector <local file>
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json
    import sys
    from dataclasses import asdict
    from phys_ai_dataprofiler.remote.range_file import LocalRangeFile

    path = sys.argv[1]
    rf = LocalRangeFile(path)
    fmt = sniff_format(rf, path)
    if fmt == "mcap":
        res, prof = inspect_mcap_remote(rf, log=lambda m: print(m, file=sys.stderr))
    elif fmt == "mat":
        res, prof = inspect_mat_remote(rf, log=lambda m: print(m, file=sys.stderr))
    else:
        sys.exit(f"unknown format: {path}")
    print(json.dumps({"basic": asdict(res), "rich": asdict(prof),
                      "fetch": {"requests": rf.requests, "bytes": rf.bytes_fetched, "size": rf.size}},
                     indent=2, default=lambda o: repr(o)))
