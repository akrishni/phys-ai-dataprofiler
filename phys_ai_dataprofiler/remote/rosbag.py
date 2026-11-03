"""
Metadata-only inspectors for ROS 1 bags and ROS 2 SQLite bags over a `RangeFile`.

ROS 1 (.bag, format 2.0)
    The 4 KB bag header gives `index_pos`; from there to EOF sit the Connection
    and ChunkInfo records (topics, types, per-chunk message counts and time
    ranges). Each chunk is followed by IndexData records, 12 bytes per message,
    which hold every message timestamp. Their byte range is computed from the
    ChunkInfo counts, so each chunk costs one request and chunk payloads (the
    only compressed part) are never read. Over the budget, chunks are sampled.
    Bags whose index was never written (recorder killed) are walked record by
    record; chunks are decompressed only to learn topic names.

ROS 2 (.db3, rosbag2 SQLite storage)
    A minimal read-only SQLite B-tree reader: page 1 gives the schema, the
    `topics` and `metadata` tables are read whole, then the `messages` table
    B-tree is descended level by level. Leaf cells start with the row's
    topic_id and timestamp; large payloads spill to overflow pages, which are
    never read. Over the budget, runs of consecutive leaves are sampled and
    the min/max timestamps come from `timestamp_idx`. The rosbag2
    `metadata.yaml` (embedded table, or the sidecar file next to the .db3)
    supplies authoritative counts and duration when present.

ROS 2 bags in MCAP storage are plain MCAP files and go through the MCAP inspector.
"""

from __future__ import annotations

import bz2
import struct
import time
from dataclasses import dataclass, field

from phys_ai_dataprofiler.inspectors.rich import Finding, RichProfile, profile_mcap_channels
from phys_ai_dataprofiler.remote.range_file import RangeFile

try:
    import lz4.frame as _lz4
except ImportError:  # pragma: no cover
    _lz4 = None

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None


BAG1_MAGIC = b"#ROSBAG V2.0\n"
BAG_MAGIC_PREFIX = b"#ROSBAG V"
SQLITE_MAGIC = b"SQLite format 3\x00"

MAX_RECORD_HEADER = 1 << 20          # sanity cap on a single bag record header
MAX_CHUNK_INFLATE = 256 << 20        # cap on one decompressed chunk (unindexed walk only)
MAX_SQLITE_RECORD = 16 << 20         # cap on a fully read SQLite row (topics, metadata tables)


@dataclass
class BagConnection:
    id: int
    topic: str
    type: str
    message_count: int | None = None
    md5sum: str | None = None                  # ROS 1
    serialization_format: str | None = None    # ROS 2
    callerid: str | None = None
    latching: bool | None = None
    message_definition: str | None = None


@dataclass
class BagInspectionResult:
    url: str
    is_bag: bool
    format: str                       # "bag1" or "bag2"
    size_bytes: int
    version: str | None = None        # "2.0" for ROS 1, rosbag2 metadata version for ROS 2
    indexed: bool = False
    error: str | None = None
    connections: list[BagConnection] | None = None
    topics: list[str] | None = None
    message_count: int | None = None
    start_ns: int | None = None
    end_ns: int | None = None
    duration_s: float | None = None
    # ROS 1
    chunk_count: int | None = None
    compression: list[str] | None = None
    # ROS 2
    storage: str | None = None
    ros_distro: str | None = None
    page_size: int | None = None
    metadata_source: str | None = None   # "embedded", "sidecar" or None
    leaves_total: int | None = None
    leaves_read: int | None = None


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #

def _url(rf: RangeFile) -> str:
    return getattr(rf, "url", str(getattr(rf, "path", "")))


def _profile(rf: RangeFile, fmt: str, indexed: bool, conns: dict[int, BagConnection],
             segments: dict[int, list[list[int]]], duration_s: float | None,
             findings: list[Finding], t0: float, complete: bool) -> RichProfile:
    stats: dict[int, dict] = {}
    for cid in sorted(conns.keys() | segments.keys()):
        c = conns.get(cid)
        segs = segments.get(cid, [])
        count = c.message_count if c and c.message_count is not None else sum(len(s) for s in segs)
        if count == 0 and complete:
            continue
        stats[cid] = {"topic": c.topic if c else f"<connection {cid}>",
                      "schema": c.type if c else "unknown", "segments": segs, "count": count}
    channels, ch_findings = profile_mcap_channels(stats)
    if duration_s:
        for p in channels:
            if p.rate_hz is None and p.msg_count > 1:
                p.rate_hz = p.msg_count / duration_s
    return RichProfile(path=_url(rf), format=fmt, size_bytes=rf.size, indexed=indexed,
                       duration_s=duration_s, time_base="ros_time_ns" if duration_s is not None else None,
                       channels=channels, findings=findings + ch_findings, inspect_cost_s=time.time() - t0)


def _sample_evenly(items: list, keep: int) -> list:
    if keep >= len(items):
        return items
    step = len(items) / keep
    return [items[int(i * step)] for i in range(keep)]


def _coalesce(ranges: list[tuple[int, int]], max_gap: int = 4096,
              max_len: int = 8 << 20) -> list[tuple[int, int]]:
    """Merge sorted [start, end) ranges into fewer requests when the bytes between them are few.

    A gap is bridged only if it is under `max_gap` and no larger than the ranges on
    either side, so merging never more than doubles the bytes read.
    """
    out: list[list[int]] = []
    for s, e in sorted(ranges):
        gap = s - out[-1][1] if out else None
        if (out and gap <= max_gap and gap <= min(e - s, out[-1][1] - out[-1][0])
                and e - out[-1][0] <= max_len):
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _fetch_pieces(rf: RangeFile, ranges: list[tuple[int, int]], workers: int) -> dict[tuple[int, int], bytes]:
    """Fetch [start, end) ranges with nearby ones merged; returns bytes per requested range."""
    merged = _coalesce(ranges)
    blobs = rf.fetch_many(merged, workers=workers)
    out: dict[tuple[int, int], bytes] = {}
    j = 0
    for s, e in sorted(ranges):
        while not (merged[j][0] <= s and e <= merged[j][1]):
            j += 1
        out[(s, e)] = blobs[j][s - merged[j][0]: e - merged[j][0]]
    return out


# --------------------------------------------------------------------------- #
# ROS 1 bag
# --------------------------------------------------------------------------- #

OP_MSG_DATA, OP_BAG_HEADER, OP_INDEX_DATA, OP_CHUNK, OP_CHUNK_INFO, OP_CONNECTION = 2, 3, 4, 5, 6, 7


def _bag_fields(buf: bytes) -> dict[bytes, bytes]:
    out: dict[bytes, bytes] = {}
    p = 0
    while p + 4 <= len(buf):
        n = struct.unpack_from("<I", buf, p)[0]
        p += 4
        if n > len(buf) - p:
            raise ValueError("bag header field runs past its record")
        k, sep, v = buf[p:p + n].partition(b"=")
        if not sep:
            raise ValueError("bag header field without '='")
        out[k] = v
        p += n
    return out


def _u32(v: bytes) -> int:
    return struct.unpack("<I", v)[0]


def _u64(v: bytes) -> int:
    return struct.unpack("<Q", v)[0]


def _ros_time(v: bytes) -> int:
    sec, nsec = struct.unpack("<II", v)
    return sec * 1_000_000_000 + nsec


def _iter_records(buf: bytes, base: int = 0):
    """Yield (offset, fields, data) for each complete record in buf."""
    p = 0
    while p + 4 <= len(buf):
        hl = struct.unpack_from("<I", buf, p)[0]
        if hl > MAX_RECORD_HEADER or p + 8 + hl > len(buf):
            return
        fields = _bag_fields(buf[p + 4: p + 4 + hl])
        dl = struct.unpack_from("<I", buf, p + 4 + hl)[0]
        if p + 8 + hl + dl > len(buf):
            return
        yield base + p, fields, buf[p + 8 + hl: p + 8 + hl + dl]
        p += 8 + hl + dl


def _connection(fields: dict[bytes, bytes], data: bytes) -> BagConnection:
    conn = _u32(fields[b"conn"])
    topic = fields.get(b"topic", b"").decode("utf-8", "replace")
    try:
        info = _bag_fields(data)
    except ValueError:
        info = {}
    latching = info.get(b"latching")
    return BagConnection(
        id=conn, topic=topic or info.get(b"topic", b"").decode("utf-8", "replace"),
        type=info.get(b"type", b"unknown").decode("utf-8", "replace"),
        md5sum=info.get(b"md5sum", b"").decode() or None,
        callerid=info.get(b"callerid", b"").decode("utf-8", "replace") or None,
        latching=(latching == b"1") if latching is not None else None,
        message_definition=info.get(b"message_definition", b"").decode("utf-8", "replace") or None,
    )


def _index_timestamps(fields: dict[bytes, bytes], data: bytes) -> tuple[int, list[int]]:
    conn = _u32(fields[b"conn"])
    count = min(_u32(fields[b"count"]), len(data) // 12)
    ts = [sec * 1_000_000_000 + nsec for sec, nsec, _ in struct.iter_unpack("<III", data[: count * 12])]
    return conn, ts


def _index_region_size(conn_counts: dict[int, int]) -> int:
    """Bytes of the IndexData records rosbag writes after a chunk: one per connection.

    Header fields op(1), ver(4), conn(4), count(4) with their 4-byte length and
    'name=' prefixes total 47 bytes; with the two length words and 12 bytes per
    message that is 55 + 12 * count.
    """
    return sum(55 + 12 * n for n in conn_counts.values())


@dataclass
class _ChunkInfo:
    pos: int
    start_ns: int
    end_ns: int
    counts: dict[int, int] = field(default_factory=dict)


def inspect_bag1_remote(rf: RangeFile, msgidx_budget_bytes: int = 64 << 20, workers: int = 16,
                        log=None) -> tuple[BagInspectionResult, RichProfile]:
    log = log or (lambda m: None)
    t0 = time.time()
    url, size = _url(rf), rf.size
    findings: list[Finding] = []
    res = BagInspectionResult(url=url, is_bag=False, format="bag1", size_bytes=size)

    def fail(msg: str, ftype: str = "parse_error"):
        res.error = msg
        return res, _profile(rf, "bag1", False, {}, {}, None, findings + [Finding("error", ftype, None, msg)],
                             t0, True)

    head = rf.read_range(0, 4096 + 13)
    if not head.startswith(BAG_MAGIC_PREFIX):
        return fail("not a ROS bag (bad magic)", "bad_magic")
    if not head.startswith(BAG1_MAGIC):
        version = head[len(BAG_MAGIC_PREFIX):head.find(b"\n")].decode("ascii", "replace")
        res.is_bag, res.version = True, version
        return fail(f"ROS bag format V{version} is not supported (only V2.0)", "unsupported_version")
    res.is_bag, res.version = True, "2.0"

    try:
        # Only the record header matters; its padding (data part) may extend past what was read.
        hl = struct.unpack_from("<I", head, 13)[0]
        if hl > MAX_RECORD_HEADER:
            return fail("first record is not a bag header")
        if 17 + hl > len(head):
            head = rf.read_range(0, 17 + hl)
        bh = _bag_fields(head[17:17 + hl])
        if bh.get(b"op") != bytes([OP_BAG_HEADER]):
            return fail("first record is not a bag header")
        index_pos = _u64(bh[b"index_pos"])
        chunk_count = _u32(bh[b"chunk_count"])
    except (ValueError, KeyError, struct.error) as exc:
        return fail(f"unreadable bag header: {exc}")

    if index_pos == 0 or index_pos >= size:
        findings.append(Finding("warn", "unindexed_bag", None,
                                "bag index was never written (recording interrupted, or not closed); "
                                "profiled by walking its records" if index_pos == 0 else
                                f"index_pos {index_pos} is past the end of the file ({size} B): truncated; "
                                "profiled by walking its records"))
        return _walk_bag1(rf, res, msgidx_budget_bytes, log, findings, t0)

    # ---- index section: connections and chunk infos ------------------------------
    rf.prefetch(index_pos, size)
    conns: dict[int, BagConnection] = {}
    chunks: list[_ChunkInfo] = []
    try:
        for _, f, data in _iter_records(rf.read_range(index_pos, size), index_pos):
            op = f.get(b"op", b"\x00")[0]
            if op == OP_CONNECTION:
                c = _connection(f, data)
                conns.setdefault(c.id, c)
            elif op == OP_CHUNK_INFO:
                ci = _ChunkInfo(pos=_u64(f[b"chunk_pos"]), start_ns=_ros_time(f[b"start_time"]),
                                end_ns=_ros_time(f[b"end_time"]))
                for conn, n in struct.iter_unpack("<II", data[: (len(data) // 8) * 8]):
                    ci.counts[conn] = n
                chunks.append(ci)
    except (ValueError, KeyError, struct.error) as exc:
        return fail(f"unreadable bag index: {exc}")
    chunks.sort(key=lambda c: c.pos)
    if chunk_count and len(chunks) != chunk_count:
        findings.append(Finding("warn", "chunk_count_mismatch", None,
                                f"bag header says {chunk_count} chunks, index lists {len(chunks)}"))

    totals: dict[int, int] = {}
    for ci in chunks:
        for conn, n in ci.counts.items():
            totals[conn] = totals.get(conn, 0) + n
    for cid, c in conns.items():
        c.message_count = totals.get(cid, 0)

    res.indexed = True
    res.chunk_count = len(chunks)
    res.connections = [conns[k] for k in sorted(conns)]
    res.topics = sorted({c.topic for c in conns.values()})
    res.message_count = sum(totals.values())
    if chunks:
        res.start_ns = min(c.start_ns for c in chunks)
        res.end_ns = max(c.end_ns for c in chunks)
        res.duration_s = (res.end_ns - res.start_ns) / 1e9

    # ---- per-message timestamps from IndexData after each chunk ---------------------
    ranges: list[tuple[int, int, _ChunkInfo]] = []
    for i, ci in enumerate(chunks):
        end = chunks[i + 1].pos if i + 1 < len(chunks) else index_pos
        ranges.append((end - _index_region_size(ci.counts), end, ci))
    total_idx = sum(e - s for s, e, _ in ranges)
    sampled = False
    if total_idx > msgidx_budget_bytes and ranges:
        ranges = _sample_evenly(ranges, max(1, int(len(ranges) * msgidx_budget_bytes / total_idx)))
        sampled = True

    segments: dict[int, list[list[int]]] = {}
    flat: dict[int, list[int]] = {}
    compressions: set[str] = set()
    fallback: list[_ChunkInfo] = []
    if ranges:
        log(f"    fetching {len(ranges)} index-data block(s), {sum(e - s for s, e, _ in ranges) / 1024:.0f} KB")
        blobs = _fetch_pieces(rf, [(s, e) for s, e, _ in ranges], workers)
        for s, e, ci in ranges:
            got = _parse_index_region(blobs[(s, e)], ci)
            if got is None:
                fallback.append(ci)
                continue
            _add_timestamps(got, segments, flat, sampled)
    if fallback:
        # Non-standard IndexData headers: find each chunk's end from its own header instead.
        heads = rf.fetch_many([(ci.pos, min(ci.pos + 512, size)) for ci in fallback], workers=workers)
        regions = []
        for ci, hb in zip(fallback, heads):
            try:
                hl = struct.unpack_from("<I", hb, 0)[0]
                ch = _bag_fields(hb[4:4 + hl])
                dl = struct.unpack_from("<I", hb, 4 + hl)[0]
                compressions.add(ch.get(b"compression", b"?").decode())
                start = ci.pos + 8 + hl + dl
                nxt = next((c.pos for c in chunks if c.pos > ci.pos), index_pos)
                regions.append((start, nxt, ci))
            except (struct.error, ValueError):
                continue
        for (s, e, ci), buf in zip(regions, rf.fetch_many([(s, e) for s, e, _ in regions], workers=workers)):
            got = _parse_index_region(buf, ci, strict=False)
            if got:
                _add_timestamps(got, segments, flat, sampled)
    if not sampled:
        segments = {cid: [sorted(ts)] for cid, ts in flat.items()}

    # Chunk compression: one tiny read of the first chunk header (often already cached).
    if chunks and not compressions:
        try:
            hb = rf.read_range(chunks[0].pos, chunks[0].pos + 256)
            hl = struct.unpack_from("<I", hb, 0)[0]
            compressions.add(_bag_fields(hb[4:4 + hl]).get(b"compression", b"?").decode())
        except (struct.error, ValueError):
            pass
    res.compression = sorted(compressions) or None

    if sampled:
        findings.append(Finding("info", "sampled_message_index", None,
                                f"index data sampled ({len(ranges)} of {len(chunks)} chunks) to stay under "
                                f"{msgidx_budget_bytes // (1 << 20)} MB; jitter/gap stats are estimates"))
    return res, _profile(rf, "bag1", True, conns, segments, res.duration_s, findings, t0, True)


def _parse_index_region(buf: bytes, ci: _ChunkInfo, strict: bool = True) -> dict[int, list[int]] | None:
    """Timestamps per connection from the IndexData records of one chunk; None if the bytes don't match."""
    out: dict[int, list[int]] = {}
    try:
        for _, f, data in _iter_records(buf):
            if f.get(b"op") != bytes([OP_INDEX_DATA]):
                if strict:
                    return None
                continue
            conn, ts = _index_timestamps(f, data)
            out.setdefault(conn, []).extend(ts)
    except (ValueError, KeyError, struct.error):
        return None
    if strict and {k: len(v) for k, v in out.items()} != {k: v for k, v in ci.counts.items() if v}:
        return None
    return out


def _add_timestamps(got: dict[int, list[int]], segments, flat, sampled: bool) -> None:
    for conn, ts in got.items():
        if not ts:
            continue
        if sampled:
            segments.setdefault(conn, []).append(sorted(ts))
        else:
            flat.setdefault(conn, []).extend(ts)


def _decompress_chunk(compression: str, data: bytes, size: int) -> bytes | None:
    if size > MAX_CHUNK_INFLATE:
        return None
    if compression == "none":
        return data
    if compression == "bz2":
        d = bz2.BZ2Decompressor()
        return d.decompress(data, max_length=size)
    if compression == "lz4" and _lz4 is not None:
        out = _lz4.decompress(data)
        return out if len(out) <= MAX_CHUNK_INFLATE else None
    return None


def _walk_bag1(rf: RangeFile, res: BagInspectionResult, budget: int, log, findings: list[Finding],
               t0: float) -> tuple[BagInspectionResult, RichProfile]:
    """Walk an unindexed bag: chunk headers and IndexData records, chunk bodies only for topic names."""
    size = rf.size
    p = 13
    read = 0
    conns: dict[int, BagConnection] = {}
    flat: dict[int, list[int]] = {}
    counts: dict[int, int] = {}
    pending: list[tuple[int, int, str, int]] = []   # chunk data (start, len, compression, size) to decode
    compressions: set[str] = set()
    n_chunks = 0
    stopped = None
    while p + 8 <= size:
        if read > budget:
            stopped = "budget"
            break
        try:
            hl = struct.unpack("<I", rf.read_range(p, p + 4))[0]
            if hl > MAX_RECORD_HEADER or p + 8 + hl > size:
                stopped = "truncated"
                break
            hb = rf.read_range(p + 4, p + 8 + hl)
            f = _bag_fields(hb[:hl])
            dl = struct.unpack("<I", hb[hl:hl + 4])[0]
        except (struct.error, ValueError):
            stopped = "truncated"
            break
        data_start, end = p + 8 + hl, p + 8 + hl + dl
        read += 8 + hl
        if end > size:
            stopped = "truncated"
            break
        op = f.get(b"op", b"\x00")[0]
        if op == OP_CHUNK:
            n_chunks += 1
            comp = f.get(b"compression", b"none").decode()
            compressions.add(comp)
            pending.append((data_start, dl, comp, _u32(f.get(b"size", b"\0\0\0\0"))))
        elif op == OP_INDEX_DATA:
            data = rf.read_range(data_start, end)
            read += dl
            conn, ts = _index_timestamps(f, data)
            flat.setdefault(conn, []).extend(ts)
            counts[conn] = counts.get(conn, 0) + len(ts)
        elif op == OP_CONNECTION:
            c = _connection(f, rf.read_range(data_start, end))
            read += dl
            conns.setdefault(c.id, c)
        p = end
    if stopped == "truncated" or (stopped is None and p < size):
        findings.append(Finding("warn", "truncated", None, f"bag ends mid-record at byte {p}"))
    if stopped == "budget":
        findings.append(Finding("info", "partial_scan", None,
                                f"walk stopped at byte {p} of {size} after {budget // (1 << 20)} MB budget"))

    # Topic names live in connection records inside chunks; decode chunks until every id is named.
    for start, dl, comp, usize in pending:
        if set(counts) <= set(conns) or read > budget:
            break
        raw = _decompress_chunk(comp, rf.read_range(start, start + dl), usize)
        read += dl
        if raw is None:
            findings.append(Finding("info", "chunks_not_decoded", None,
                                    f"cannot decode '{comp}' chunks here; some topic names unknown"))
            break
        for _, cf, cdata in _iter_records(raw):
            if cf.get(b"op") == bytes([OP_CONNECTION]):
                c = _connection(cf, cdata)
                conns.setdefault(c.id, c)
    if not flat and pending:
        findings.append(Finding("warn", "no_message_index", None,
                                "chunks have no index data; message timestamps unavailable"))

    for cid, c in conns.items():
        c.message_count = counts.get(cid, 0)
    all_ts = [t for ts in flat.values() for t in ts]
    res.chunk_count = n_chunks
    res.compression = sorted(compressions) or None
    res.connections = [conns[k] for k in sorted(conns)]
    res.topics = sorted({c.topic for c in conns.values()})
    res.message_count = sum(counts.values())
    if all_ts:
        res.start_ns, res.end_ns = min(all_ts), max(all_ts)
        res.duration_s = (res.end_ns - res.start_ns) / 1e9
    segments = {cid: [sorted(ts)] for cid, ts in flat.items()}
    return res, _profile(rf, "bag1", False, conns, segments, res.duration_s, findings, t0, stopped is None)


# --------------------------------------------------------------------------- #
# SQLite B-tree reader (read-only, just enough for rosbag2)
# --------------------------------------------------------------------------- #

class SqliteError(ValueError):
    pass


def _varint(buf: bytes, p: int) -> tuple[int, int]:
    v = 0
    for i in range(8):
        b = buf[p + i]
        v = (v << 7) | (b & 0x7F)
        if b < 0x80:
            return v, p + i + 1
    return (v << 8) | buf[p + 8], p + 9


def _serial_len(t: int) -> int:
    if t < 12:
        return (0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0)[t]
    return (t - 12) // 2 if t % 2 == 0 else (t - 13) // 2


def _decode_value(t: int, raw: bytes, enc: str):
    if t == 0:
        return None
    if 1 <= t <= 6:
        return int.from_bytes(raw, "big", signed=True)
    if t == 7:
        return struct.unpack(">d", raw)[0]
    if t == 8:
        return 0
    if t == 9:
        return 1
    if t >= 12 and t % 2 == 0:
        return bytes(raw)
    if t >= 13:
        return raw.decode(enc, "replace")
    raise SqliteError(f"reserved serial type {t}")


def _parse_record(payload: bytes, enc: str, want: int | None = None) -> list:
    """Decode a record; with `want`, stop after that many columns (payload may be a prefix)."""
    hsize, p = _varint(payload, 0)
    types = []
    while p < hsize:
        t, p = _varint(payload, p)
        types.append(t)
    out, q = [], hsize
    for t in types[: want if want is not None else len(types)]:
        n = _serial_len(t)
        if q + n > len(payload):
            raise SqliteError("record value runs past the available payload")
        out.append(_decode_value(t, payload[q:q + n], enc))
        q += n
    return out


@dataclass
class _Page:
    no: int
    kind: int                # 2 interior index, 5 interior table, 10 leaf index, 13 leaf table
    cells: list[int]         # offsets into data
    right: int | None
    data: bytes


class _Sqlite:
    def __init__(self, rf: RangeFile) -> None:
        self.rf = rf
        h = rf.read_range(0, 100)
        if h[:16] != SQLITE_MAGIC:
            raise SqliteError("not an SQLite database")
        ps = struct.unpack(">H", h[16:18])[0]
        self.page_size = 65536 if ps == 1 else ps
        if self.page_size < 512 or self.page_size & (self.page_size - 1):
            raise SqliteError(f"invalid page size {ps}")
        self.usable = self.page_size - h[20]
        self.wal = h[18] == 2 or h[19] == 2
        self.page_count = struct.unpack(">I", h[28:32])[0] or rf.size // self.page_size
        self.enc = {1: "utf-8", 2: "utf-16-le", 3: "utf-16-be"}.get(struct.unpack(">I", h[56:60])[0], "utf-8")

    def page_range(self, no: int) -> tuple[int, int]:
        return (no - 1) * self.page_size, no * self.page_size

    def parse_page(self, no: int, data: bytes) -> _Page:
        if len(data) < self.page_size:
            raise SqliteError(f"page {no} is beyond the end of the file (truncated database)")
        h = 100 if no == 1 else 0
        kind = data[h]
        if kind not in (2, 5, 10, 13):
            raise SqliteError(f"page {no} is not a B-tree page (type {kind})")
        ncells = struct.unpack_from(">H", data, h + 3)[0]
        interior = kind in (2, 5)
        right = struct.unpack_from(">I", data, h + 8)[0] if interior else None
        ptr = h + (12 if interior else 8)
        cells = list(struct.unpack_from(f">{ncells}H", data, ptr))
        return _Page(no, kind, cells, right, data)

    def page(self, no: int) -> _Page:
        s, e = self.page_range(no)
        return self.parse_page(no, self.rf.read_range(s, e))

    def _local_size(self, payload: int, table_leaf: bool) -> int:
        u = self.usable
        x = u - 35 if table_leaf else ((u - 12) * 64 // 255) - 23
        if payload <= x:
            return payload
        m = ((u - 12) * 32 // 255) - 23
        k = m + ((payload - m) % (u - 4))
        return k if k <= x else m

    def table_leaf_cell(self, pg: _Page, off: int) -> tuple[int, int, bytes, int | None]:
        """(rowid, payload_size, local_payload, first_overflow_page)."""
        d = pg.data
        size, p = _varint(d, off)
        rowid, p = _varint(d, p)
        local = self._local_size(size, True)
        ovf = struct.unpack_from(">I", d, p + local)[0] if local < size else None
        return rowid, size, d[p:p + local], ovf

    def index_cell_payload(self, pg: _Page, off: int) -> bytes:
        d = pg.data
        p = off + 4 if pg.kind == 2 else off
        size, p = _varint(d, p)
        return d[p:p + self._local_size(size, False)]

    def full_payload(self, local: bytes, size: int, ovf: int | None) -> bytes:
        if ovf is None:
            return local
        if size > MAX_SQLITE_RECORD:
            raise SqliteError(f"row of {size} bytes exceeds the read cap")
        out = bytearray(local)
        seen = set()
        while ovf and len(out) < size:
            if ovf in seen or ovf > self.page_count:
                raise SqliteError("corrupt overflow chain")
            seen.add(ovf)
            pg = self.rf.read_range(*self.page_range(ovf))
            ovf = struct.unpack(">I", pg[:4])[0]
            out += pg[4: 4 + min(self.usable - 4, size - len(out))]
        return bytes(out)

    def table_rows(self, root: int, max_pages: int = 4096):
        """Every (rowid, record) of a small table, reading overflow chains."""
        stack, pages = [root], 0
        rows = []
        while stack:
            no = stack.pop()
            pages += 1
            if pages > max_pages:
                raise SqliteError("table too large to read whole")
            pg = self.page(no)
            if pg.kind == 13:
                for off in pg.cells:
                    rowid, size, local, ovf = self.table_leaf_cell(pg, off)
                    rows.append((rowid, _parse_record(self.full_payload(local, size, ovf), self.enc)))
            elif pg.kind == 5:
                kids = [struct.unpack_from(">I", pg.data, off)[0] for off in pg.cells] + [pg.right]
                stack.extend(reversed(kids))
            else:
                raise SqliteError(f"page {no} is not a table page")
        return sorted(rows, key=lambda r: r[0])

    def index_edge(self, root: int, last: bool) -> bytes | None:
        """Payload of the first (or last) entry of an index B-tree."""
        no = root
        for _ in range(64):
            pg = self.page(no)
            if pg.kind == 10:
                if not pg.cells:
                    return None
                return self.index_cell_payload(pg, pg.cells[-1 if last else 0])
            if pg.kind != 2:
                raise SqliteError("not an index page")
            no = pg.right if last else struct.unpack_from(">I", pg.data, pg.cells[0])[0]
        raise SqliteError("index too deep")

    def table_last_rowid(self, root: int) -> int | None:
        no = root
        for _ in range(64):
            pg = self.page(no)
            if pg.kind == 13:
                return self.table_leaf_cell(pg, pg.cells[-1])[0] if pg.cells else None
            no = pg.right
        raise SqliteError("table too deep")


def _columns(sql: str) -> list[str]:
    body = sql[sql.find("(") + 1: sql.rfind(")")]
    cols, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            cols.append(cur)
            cur = ""
        else:
            cur += ch
    cols.append(cur)
    names = []
    for c in cols:
        tok = c.strip().split()
        if tok and tok[0].upper() not in ("PRIMARY", "UNIQUE", "CHECK", "FOREIGN", "CONSTRAINT"):
            names.append(tok[0].strip('"`[]'))
    return names


def _bag2_metadata(text: str) -> dict | None:
    if _yaml is None or not text:
        return None
    try:
        doc = _yaml.safe_load(text)
    except Exception:
        return None
    if isinstance(doc, dict) and "rosbag2_bagfile_information" in doc:
        doc = doc["rosbag2_bagfile_information"]
    return doc if isinstance(doc, dict) else None


def _ns(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d if isinstance(d, int) else None


def inspect_bag2_remote(rf: RangeFile, msgidx_budget_bytes: int = 64 << 20, workers: int = 16,
                        log=None, sidecar_yaml=None, file_name: str | None = None,
                        ) -> tuple[BagInspectionResult, RichProfile]:
    """ROS 2 SQLite bag: schema and topics, then per-topic timestamps from the messages table leaves.

    `sidecar_yaml` is the text of the bag's metadata.yaml, or a callable returning it (called only
    when the database has no embedded metadata, so the extra request is skipped when not needed).
    `file_name` is this .db3's base name, used to pick its entry out of a multi-file bag's metadata.
    """
    log = log or (lambda m: None)
    t0 = time.time()
    url, size = _url(rf), rf.size
    findings: list[Finding] = []
    res = BagInspectionResult(url=url, is_bag=False, format="bag2", size_bytes=size, storage="sqlite3")

    def fail(msg: str, ftype: str = "parse_error"):
        res.error = msg
        return res, _profile(rf, "bag2", False, {}, {}, None, findings + [Finding("error", ftype, None, msg)],
                             t0, True)

    try:
        db = _Sqlite(rf)
    except SqliteError as exc:
        return fail(str(exc), "bad_magic")
    res.page_size = db.page_size
    if db.wal:
        findings.append(Finding("warn", "wal_mode", None,
                                "database is in WAL mode; rows still in a -wal file are not visible"))

    try:
        master = {r[1]: {"type": r[0], "root": r[3], "sql": r[4] or ""}
                  for _, r in db.table_rows(1) if len(r) >= 5}
    except SqliteError as exc:
        return fail(f"unreadable SQLite schema: {exc}")
    if "messages" not in master or "topics" not in master:
        return fail("SQLite file without rosbag2 'messages' and 'topics' tables", "not_a_bag")
    res.is_bag = True

    conns: dict[int, BagConnection] = {}
    try:
        tcols = _columns(master["topics"]["sql"])
        for rowid, rec in db.table_rows(master["topics"]["root"]):
            row = dict(zip(tcols, rec))
            tid = row.get("id") if row.get("id") is not None else rowid
            conns[tid] = BagConnection(id=tid, topic=str(row.get("name") or ""), type=str(row.get("type") or "unknown"),
                                       serialization_format=row.get("serialization_format"))
        if "schema" in master:
            for _, rec in db.table_rows(master["schema"]["root"]):
                scols = _columns(master["schema"]["sql"])
                res.ros_distro = dict(zip(scols, rec)).get("ros_distro")
        meta_text = None
        if "metadata" in master:
            mcols = _columns(master["metadata"]["sql"])
            rows = db.table_rows(master["metadata"]["root"])
            if rows:
                meta_text = dict(zip(mcols, rows[-1][1])).get("metadata")
    except SqliteError as exc:
        return fail(f"unreadable rosbag2 tables: {exc}")

    meta = _bag2_metadata(meta_text) if meta_text else None
    res.metadata_source = "embedded" if meta else None
    if meta is None and sidecar_yaml:
        if callable(sidecar_yaml):
            try:
                sidecar_yaml = sidecar_yaml()
            except Exception as exc:
                findings.append(Finding("info", "sidecar_unreadable", None, f"metadata.yaml not read: {exc}"))
                sidecar_yaml = None
        meta = _bag2_metadata(sidecar_yaml) if sidecar_yaml else None
        res.metadata_source = "sidecar" if meta else None
    if meta:
        res.version = str(meta.get("version")) if meta.get("version") is not None else None
        res.ros_distro = res.ros_distro or meta.get("ros_distro")
        res.storage = meta.get("storage_identifier") or res.storage
        if meta.get("compression_mode") == "message" and meta.get("compression_format"):
            findings.append(Finding("info", "message_compression", None,
                                    f"payloads are {meta['compression_format']}-compressed per message"))

    # Counts the metadata can vouch for: only when it describes this one file.
    meta_counts: dict[str, int] | None = None
    meta_file = None
    if meta:
        files = meta.get("files") or []
        paths = meta.get("relative_file_paths") or []
        if file_name:
            meta_file = next((f for f in files if isinstance(f, dict)
                              and str(f.get("path", "")).split("/")[-1] == file_name), None)
        if len(paths) <= 1:
            meta_counts = {}
            for t in meta.get("topics_with_message_count") or []:
                tm = (t or {}).get("topic_metadata") or {}
                if tm.get("name") is not None and isinstance(t.get("message_count"), int):
                    meta_counts[tm["name"]] = t["message_count"]

    # ---- messages table: descend level by level, sampling runs when over budget ---------
    mcols = _columns(master["messages"]["sql"])
    try:
        i_topic, i_ts = mcols.index("topic_id"), mcols.index("timestamp")
    except ValueError:
        return fail(f"messages table lacks topic_id/timestamp columns: {mcols}", "not_a_bag")
    want = max(i_topic, i_ts) + 1

    budget = max(msgidx_budget_bytes, 4 * db.page_size)
    spent = 0
    level: list[tuple[int, int]] = [(master["messages"]["root"], 0)]   # (page, run id)
    leaves: list[tuple[_Page, int]] = []
    sampled = False
    next_run = 1
    try:
        while level:
            cost = len(level) * db.page_size
            if spent + cost > budget:
                # Keep evenly spaced runs of up to 8 consecutive pages (same parent run).
                groups: list[list[tuple[int, int]]] = []
                for no, run in level:
                    if groups and groups[-1][-1][1] == run and len(groups[-1]) < 8:
                        groups[-1].append((no, run))
                    else:
                        groups.append([(no, run)])
                room = max(1, (budget - spent) // db.page_size)
                keep = _sample_evenly(groups, max(1, int(len(groups) * room / len(level))))
                level = []
                for g in keep:
                    level.extend((no, next_run) for no, _ in g)
                    next_run += 1
                sampled = True
            blobs = _fetch_pieces(rf, [db.page_range(no) for no, _ in level], workers)
            spent += len(level) * db.page_size
            nxt: list[tuple[int, int]] = []
            for no, run in level:
                pg = db.parse_page(no, blobs[db.page_range(no)])
                if pg.kind == 13:
                    leaves.append((pg, run))
                elif pg.kind == 5:
                    kids = [struct.unpack_from(">I", pg.data, off)[0] for off in pg.cells] + [pg.right]
                    nxt.extend((k, run) for k in kids)
                else:
                    raise SqliteError(f"page {no} in messages table is not a table page")
            level = nxt
    except SqliteError as exc:
        return fail(f"unreadable messages table: {exc}")
    log(f"    read {len(leaves)} leaf page(s) of the messages table, {spent / 1024:.0f} KB")

    seg_runs: dict[int, dict[int, list[int]]] = {}
    counts: dict[int, int] = {}
    payload_bytes: dict[int, int] = {}
    bad_cells = 0
    for pg, run in leaves:
        for off in pg.cells:
            try:
                _, psize, local, _ = db.table_leaf_cell(pg, off)
                rec = _parse_record(local, db.enc, want)
            except (SqliteError, IndexError, struct.error):
                bad_cells += 1
                continue
            tid, ts = rec[i_topic], rec[i_ts]
            if not isinstance(tid, int) or not isinstance(ts, int):
                bad_cells += 1
                continue
            seg_runs.setdefault(tid, {}).setdefault(run, []).append(ts)
            counts[tid] = counts.get(tid, 0) + 1
            payload_bytes[tid] = payload_bytes.get(tid, 0) + psize
    if bad_cells:
        findings.append(Finding("warn", "unreadable_rows", None, f"{bad_cells} message row(s) could not be decoded"))
    segments = {tid: [sorted(ts) for ts in runs.values()] for tid, runs in seg_runs.items()}
    rows_read = sum(counts.values())

    # ---- counts, time range ------------------------------------------------------------
    start_ns = end_ns = None
    if not sampled:
        for cid, c in conns.items():
            c.message_count = counts.get(cid, 0)
        res.message_count = rows_read
        all_ts = [t for runs in seg_runs.values() for ts in runs.values() for t in ts]
        if all_ts:
            start_ns, end_ns = min(all_ts), max(all_ts)
    else:
        if meta_counts is not None:
            for c in conns.values():
                c.message_count = meta_counts.get(c.topic, 0)
            res.message_count = sum(meta_counts.values())
        else:
            try:
                total = db.table_last_rowid(master["messages"]["root"])
            except SqliteError:
                total = None
            if total and rows_read:
                for cid, c in conns.items():
                    c.message_count = round(counts.get(cid, 0) / rows_read * total)
                res.message_count = total
                findings.append(Finding("info", "estimated_counts", None,
                                        f"per-topic counts estimated from {rows_read:,} sampled rows "
                                        f"of ~{total:,} (no metadata for this file)"))
        idx = next((n for n, m in master.items() if m["type"] == "index"
                    and "messages" in m["sql"] and "timestamp" in m["sql"]), None)
        if idx:
            try:
                lo = db.index_edge(master[idx]["root"], last=False)
                hi = db.index_edge(master[idx]["root"], last=True)
                start_ns = _parse_record(lo, db.enc, 1)[0] if lo else None
                end_ns = _parse_record(hi, db.enc, 1)[0] if hi else None
            except SqliteError:
                pass
        findings.append(Finding("info", "sampled_message_index", None,
                                f"messages table sampled ({len(leaves)} leaf pages, {rows_read:,} rows) to stay "
                                f"under {msgidx_budget_bytes // (1 << 20)} MB; jitter/gap stats are estimates"))
    if meta_counts is not None and not sampled:
        mismatch = [c.topic for c in conns.values()
                    if c.topic in meta_counts and meta_counts[c.topic] != c.message_count]
        if mismatch:
            findings.append(Finding("warn", "metadata_mismatch", None,
                                    f"metadata message counts differ from the table for {len(mismatch)} topic(s), "
                                    f"e.g. {mismatch[0]}"))
    if start_ns is None and meta_file:
        start_ns = _ns(meta_file, "starting_time", "nanoseconds_since_epoch")
        dur = _ns(meta_file, "duration", "nanoseconds")
        end_ns = start_ns + dur if start_ns is not None and dur is not None else None

    res.start_ns, res.end_ns = start_ns, end_ns
    if start_ns is not None and end_ns is not None:
        res.duration_s = (end_ns - start_ns) / 1e9
    res.indexed = True
    res.connections = [conns[k] for k in sorted(conns)]
    res.topics = sorted({c.topic for c in conns.values()})
    res.leaves_read = len(leaves)
    if not sampled:
        res.leaves_total = len(leaves)
    if res.message_count == 0:
        findings.append(Finding("warn", "empty_file", None, "bag has no messages"))
    return res, _profile(rf, "bag2", True, conns, segments, res.duration_s, findings, t0, not sampled)
