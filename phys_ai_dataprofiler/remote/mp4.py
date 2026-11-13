"""
Metadata-only inspector for MP4 (and QuickTime .mov) video over a `RangeFile`.

An MP4 keeps its index in the `moov` box, at the start of the file (fast-start) or at the end. For each
track it holds the codec and frame size (`stsd`), the timescale (`mdhd`), the duration of every frame
(`stts`, run-length encoded, so a constant-rate video costs 8 bytes however long it is) and the
keyframes (`stss`). The inspector walks the top-level boxes by their 8-16 byte headers, skips `mdat`
(the frames themselves), and inside `moov` reads only the boxes above. The per-frame size and offset
tables (`stsz`, `stco`), which make up most of `moov`, are skipped. Frames are never decoded.

Fragmented MP4 (the `moov` holds no samples; each `moof` describes the fragment after it) is walked
fragment by fragment, reading only each `moof`, within a byte budget. Over the budget the remaining
fragments are not read and the report says so.

Timestamps are the container's timeline as the encoder wrote it. LeRobot and most recorders encode at a
fixed rate, so a frame dropped at capture time shows up as a gap only when the writer kept the real
capture timestamps.

LeRobot datasets keep each camera in its own MP4s, under `videos/<camera>/chunk-000/file-000.mp4` (v3)
or `videos/chunk-000/<camera>/episode_000000.mp4` (v2). `video_stream_name` turns such a path into the
camera name, used as the stream's topic, and a group key for files that cover the same time.
"""

from __future__ import annotations

import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

from phys_ai_dataprofiler.inspectors.rich import ChannelProfile, Finding, RichProfile, timing_stats
from phys_ai_dataprofiler.remote.range_file import RangeFile

MP4_EXTENSIONS = (".mp4", ".m4v", ".mov")
# Boxes that can open an ISO-BMFF / QuickTime file.
FIRST_BOXES = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"styp", b"sidx"}
CONTAINERS = {"moov", "trak", "mdia", "minf", "stbl", "edts", "mvex"}
LEAVES = {"mvhd", "tkhd", "mdhd", "hdlr", "stsd", "stts", "stss", "trex"}
CODECS = {"av01": "av1", "avc1": "h264", "avc3": "h264", "hvc1": "hevc", "hev1": "hevc", "vp09": "vp9",
          "vp08": "vp8", "mp4v": "mpeg4", "jpeg": "mjpeg", "mjpa": "mjpeg", "mjp2": "mjpeg2000",
          "apch": "prores", "apcn": "prores", "apcs": "prores", "apco": "prores", "ap4h": "prores",
          "mp4a": "aac", "opus": "opus", "Opus": "opus", "ac-3": "ac3", "fLaC": "flac"}
CODEC_ALIASES = {"avc": "h264", "libx264": "h264", "h.264": "h264", "hevc": "hevc", "h265": "hevc",
                 "libx265": "hevc", "libsvtav1": "av1", "libaom-av1": "av1", "libdav1d": "av1"}
MAX_TOP_LEVEL_BOXES = 1_000_000
CHUNK_DIR = re.compile(r"^chunk-\d+$")


@dataclass
class Mp4Track:
    track_id: int | None
    handler: str | None                # "vide", "soun", "meta", ...
    fourcc: str | None                 # sample entry type, e.g. "av01"
    codec: str | None                  # "av1", "h264", ...
    width: int | None = None
    height: int | None = None
    timescale: int | None = None
    duration_s: float | None = None
    frame_count: int | None = None
    rate_hz: float | None = None
    keyframes: int | None = None       # None when there is no sync table (every frame is a keyframe)
    max_keyframe_interval: int | None = None   # frames between keyframes, worst case


@dataclass
class Mp4InspectionResult:
    url: str
    is_mp4: bool
    size_bytes: int
    stream_name: str | None = None
    brand: str | None = None
    compatible_brands: list[str] = field(default_factory=list)
    top_level: list[str] = field(default_factory=list)   # box types in file order (runs collapsed)
    moov_at_start: bool | None = None
    fragmented: bool = False
    fragments_read: int | None = None
    duration_s: float | None = None
    bitrate_kbps: float | None = None
    tracks: list[Mp4Track] = field(default_factory=list)
    declared: dict | None = None       # what the dataset's metadata says about this camera, if known
    error: str | None = None


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def video_stream_name(path: str) -> tuple[str, str]:
    """(camera name, recording group) for a video path.

    LeRobot v2 `videos/chunk-000/<cam>/episode_000042.mp4`: one file per camera per episode, so the
    cameras of an episode form a group, `videos/chunk-000/*/episode_000042.mp4`.
    LeRobot v3 `videos/<cam>/chunk-000/file-000.mp4`: each file packs many episodes, and cameras split
    into files at different points (file-000 of two cameras need not hold the same episodes), so all of
    a dataset's videos form one group, `videos/*`, and each camera is the sum of its files.
    Other layouts use the file name as the camera and each file as its own recording.
    """
    parts = path.split("/")
    if "videos" in parts[:-1]:
        i = parts.index("videos")
        for j in range(i + 1, len(parts) - 1):
            if not CHUNK_DIR.match(parts[j]):
                if parts[-1].startswith("episode_"):
                    return parts[j], "/".join(parts[:j] + ["*"] + parts[j + 1:])
                return parts[j], "/".join(parts[:i + 1] + ["*"])
    return PurePosixPath(path).stem, path


def recording_seconds(items) -> float:
    """Total recorded time from (path, format, duration_s) of profiled files.

    The cameras of a recording cover the same time, so videos count once per group (see
    `video_stream_name`): each camera's files are summed, and the group takes its longest camera. A
    LeRobot dataset's Parquet table covers the same frames as its videos, so when any Parquet file has
    a duration the videos add nothing; otherwise they are the timeline.
    """
    items = [(p, f, d) for p, f, d in items if d]
    other = sum(float(d) for p, f, d in items if f != "mp4")
    if any(f == "parquet" for p, f, d in items):
        return other
    groups: dict[str, dict[str, float]] = {}
    for p, f, d in items:
        if f == "mp4":
            cam, g = video_stream_name(p)
            cams = groups.setdefault(g, {})
            cams[cam] = cams.get(cam, 0.0) + float(d)
    return other + sum(max(c.values()) for c in groups.values())


def canonical_codec(name: str | None) -> str | None:
    if not name:
        return None
    n = str(name).strip()
    return CODECS.get(n) or CODEC_ALIASES.get(n.lower()) or n.lower()


def is_mp4_head(head: bytes) -> bool:
    return len(head) >= 8 and head[4:8] in FIRST_BOXES


# --------------------------------------------------------------------------- #
# Box reading
# --------------------------------------------------------------------------- #

class _Truncated(Exception):
    pass


def _header(rf: RangeFile, pos: int, end: int) -> tuple[str, int, int]:
    """(type, total size, header size) of the box at pos; raises _Truncated past `end`."""
    h = rf.read_range(pos, min(pos + 16, end))
    if len(h) < 8:
        raise _Truncated(f"box header at {pos} cut off")
    size, typ = struct.unpack(">I4s", h[:8])
    hdr = 8
    if size == 1:
        if len(h) < 16:
            raise _Truncated(f"box header at {pos} cut off")
        size = struct.unpack(">Q", h[8:16])[0]
        hdr = 16
    elif size == 0:
        size = end - pos
    if size < hdr:
        raise ValueError(f"invalid box size {size} at offset {pos}")
    return typ.decode("latin-1"), size, hdr


def _children(rf: RangeFile, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        typ, size, hdr = _header(rf, pos, end)
        yield typ, pos, size, hdr
        pos += size


def _full(payload: bytes) -> tuple[int, int, bytes]:
    """(version, flags, body) of a FullBox payload."""
    if len(payload) < 4:
        raise ValueError("full box too short")
    return payload[0], int.from_bytes(payload[1:4], "big"), payload[4:]


def _timescale_duration(payload: bytes) -> tuple[int, int]:
    """(timescale, duration) from an mvhd or mdhd payload."""
    v, _, b = _full(payload)
    if v == 1:
        return struct.unpack_from(">IQ", b, 16)
    return struct.unpack_from(">II", b, 8)


@dataclass
class _Trak:
    track_id: int | None = None
    handler: str | None = None
    fourcc: str | None = None
    width: int | None = None
    height: int | None = None
    timescale: int | None = None
    media_duration: int | None = None
    stts: list[tuple[int, int]] | None = None
    stts_truncated: bool = False
    stss: np.ndarray | None = None
    trex_duration: int | None = None


def _parse_leaf(t: _Trak, typ: str, payload: bytes, budget_left: int) -> int:
    """Fill `t` from one leaf box; returns bytes used against the budget."""
    if typ == "tkhd":
        v, _, b = _full(payload)
        t.track_id = struct.unpack_from(">I", b, 16 if v == 1 else 8)[0]
    elif typ == "mdhd":
        t.timescale, t.media_duration = _timescale_duration(payload)
    elif typ == "hdlr":
        _, _, b = _full(payload)
        t.handler = b[4:8].decode("latin-1")
    elif typ == "stsd":
        _, _, b = _full(payload)
        if struct.unpack_from(">I", b, 0)[0] >= 1 and len(b) >= 12:
            e = b[4:]
            t.fourcc = e[4:8].decode("latin-1")
            if t.handler == "vide" and len(e) >= 36:
                t.width, t.height = struct.unpack_from(">HH", e, 32)
    elif typ == "stts":
        _, _, b = _full(payload)
        n = struct.unpack_from(">I", b, 0)[0]
        have = (len(b) - 4) // 8
        t.stts_truncated = have < n
        arr = np.frombuffer(b[4:4 + 8 * min(n, have)], dtype=">u4").reshape(-1, 2)
        t.stts = [(int(c), int(d)) for c, d in arr]
    elif typ == "stss":
        _, _, b = _full(payload)
        n = struct.unpack_from(">I", b, 0)[0]
        t.stss = np.frombuffer(b[4:4 + 4 * n], dtype=">u4").astype(np.int64)
    return len(payload)


def _walk_moov(rf: RangeFile, start: int, end: int, budget: int) -> tuple[int | None, int | None, list[_Trak], bool, int]:
    """(movie timescale, movie duration, tracks, fragmented, bytes of tables read) for one moov box."""
    tracks: list[_Trak] = []
    trex: dict[int, int] = {}
    movie = [None, None]
    fragmented = False
    used = 0

    def walk(s: int, e: int, trak: _Trak | None) -> None:
        nonlocal fragmented, used
        for typ, pos, size, hdr in _children(rf, s, e):
            if typ == "trak":
                t = _Trak()
                tracks.append(t)
                walk(pos + hdr, pos + size, t)
            elif typ == "mvex":
                fragmented = True
                walk(pos + hdr, pos + size, None)
            elif typ in CONTAINERS:
                walk(pos + hdr, pos + size, trak)
            elif typ == "mvhd":
                movie[0], movie[1] = _timescale_duration(rf.read_range(pos + hdr, pos + size))
            elif typ == "trex":
                _, _, b = _full(rf.read_range(pos + hdr, pos + size))
                tid, _, dur = struct.unpack_from(">III", b, 0)
                trex[tid] = dur
            elif typ in LEAVES and trak is not None:
                cap = pos + size
                if typ in ("stts", "stss"):
                    cap = min(cap, pos + hdr + max(8, budget - used))
                used += _parse_leaf(trak, typ, rf.read_range(pos + hdr, cap), budget - used)
                if typ == "stss" and cap < pos + size:
                    trak.stss = None     # partial sync table: keyframe stats unknown
                if typ == "stts" and cap < pos + size:
                    trak.stts_truncated = True

    walk(start, end, None)
    for t in tracks:
        if t.track_id in trex:
            t.trex_duration = trex[t.track_id]
    return movie[0], movie[1], tracks, fragmented, used


def _parse_moof(buf: bytes, base: int) -> list[tuple[int, int | None, list[int], list[bool]]]:
    """Per traf: (track_id, base decode time or None, sample durations, keyframe flags)."""
    out = []

    def boxes(b: bytes, s: int, e: int):
        p = s
        while p + 8 <= e:
            size, typ = struct.unpack_from(">I4s", b, p)
            hdr = 8
            if size == 1:
                size = struct.unpack_from(">Q", b, p + 8)[0]
                hdr = 16
            elif size == 0:
                size = e - p
            if size < hdr or p + size > e:
                raise ValueError(f"bad box in moof at {base + p}")
            yield typ.decode("latin-1"), p + hdr, p + size
            p += size

    for typ, s, e in boxes(buf, 0, len(buf)):
        if typ != "traf":
            continue
        tid, tdt, durs, keys = None, None, [], []
        dflt_dur, dflt_flags = None, None
        for t2, s2, e2 in boxes(buf, s, e):
            p = buf[s2:e2]
            v, fl, b = p[0], int.from_bytes(p[1:4], "big"), p[4:]
            if t2 == "tfhd":
                tid = struct.unpack_from(">I", b, 0)[0]
                o = 4
                if fl & 0x1:
                    o += 8
                if fl & 0x2:
                    o += 4
                if fl & 0x8:
                    dflt_dur = struct.unpack_from(">I", b, o)[0]
                    o += 4
                if fl & 0x10:
                    o += 4
                if fl & 0x20:
                    dflt_flags = struct.unpack_from(">I", b, o)[0]
            elif t2 == "tfdt":
                tdt = struct.unpack_from(">Q" if v == 1 else ">I", b, 0)[0]
            elif t2 == "trun":
                n = struct.unpack_from(">I", b, 0)[0]
                o = 4
                if fl & 0x1:
                    o += 4
                first_flags = None
                if fl & 0x4:
                    first_flags = struct.unpack_from(">I", b, o)[0]
                    o += 4
                for k in range(n):
                    d, sf = dflt_dur, dflt_flags
                    if fl & 0x100:
                        d = struct.unpack_from(">I", b, o)[0]
                        o += 4
                    if fl & 0x200:
                        o += 4
                    if fl & 0x400:
                        sf = struct.unpack_from(">I", b, o)[0]
                        o += 4
                    if fl & 0x800:
                        o += 4
                    if k == 0 and first_flags is not None:
                        sf = first_flags
                    durs.append(d)
                    keys.append(sf is None or not (sf & 0x10000))
        out.append((tid, tdt, durs, keys))
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _track_times(t: _Trak) -> np.ndarray | None:
    """Frame start times in track ticks, from stts."""
    if not t.stts:
        return None
    counts = np.array([c for c, _ in t.stts], dtype=np.int64)
    deltas = np.array([d for _, d in t.stts], dtype=np.int64)
    per = np.repeat(deltas, counts)
    return np.concatenate(([0], np.cumsum(per)))     # n frames + 1 (the end time)


def _keyframe_stats(sync: np.ndarray | None, frames: int) -> tuple[int | None, int | None]:
    if sync is None:
        return None, None
    if not len(sync):
        return 0, frames or None
    gaps = np.diff(np.concatenate((sync, [frames + 1])))
    return len(sync), int(max(gaps.max(), sync[0] - 1))


def inspect_mp4(rf: RangeFile, stream_name: str | None = None, budget_bytes: int = 64 << 20,
                declared: dict | None = None, log=None) -> tuple[Mp4InspectionResult, RichProfile]:
    """Profile an MP4 from its box headers and index. `declared` is the dataset's own description of
    this camera (fps, width, height, codec), checked against what the file says."""
    log = log or (lambda m: None)
    t0 = time.time()
    url = getattr(rf, "url", str(getattr(rf, "path", "")))
    size = rf.size
    res = Mp4InspectionResult(url=url, is_mp4=False, size_bytes=size, stream_name=stream_name, declared=declared)
    findings: list[Finding] = []
    channels: list[ChannelProfile] = []

    def done(duration=None, indexed=True):
        return res, RichProfile(path=url, format="mp4", size_bytes=size, indexed=indexed, duration_s=duration,
                                time_base="relative_s" if duration is not None else None, channels=channels,
                                findings=findings, inspect_cost_s=time.time() - t0)

    def fail(msg: str, ftype: str, indexed: bool = False):
        res.error = msg
        findings.append(Finding("error", ftype, None, msg))
        return done(indexed=indexed)

    if not is_mp4_head(rf.read_range(0, 16)):
        return fail("not an MP4/QuickTime file (no ftyp or known first box)", "bad_magic")
    res.is_mp4 = True

    # ---- top-level boxes -------------------------------------------------------------
    moov = None
    moofs: list[tuple[int, int]] = []
    mdat_bytes = 0
    truncated = None
    try:
        pos, n = 0, 0
        while pos < size and n < MAX_TOP_LEVEL_BOXES:
            typ, bsize, hdr = _header(rf, pos, size)
            n += 1
            if not res.top_level or res.top_level[-1] != typ:
                res.top_level.append(typ)
            if pos + bsize > size:
                truncated = f"{typ} box at offset {pos} runs {pos + bsize - size} bytes past the end of the file"
                if typ != "mdat":
                    break
            if typ == "ftyp":
                b = rf.read_range(pos + hdr, min(pos + bsize, pos + hdr + 256))
                res.brand = b[:4].decode("latin-1")
                res.compatible_brands = [b[i:i + 4].decode("latin-1") for i in range(8, len(b) - 3, 4)]
            elif typ == "moov" and moov is None:
                moov = (pos + hdr, min(pos + bsize, size))
                res.moov_at_start = mdat_bytes == 0
            elif typ == "moof":
                moofs.append((pos, min(pos + bsize, size)))
            elif typ == "mdat":
                mdat_bytes += min(bsize, size - pos) - hdr
            pos += bsize
    except _Truncated as exc:
        truncated = str(exc)
    except ValueError as exc:
        return fail(f"unreadable box structure: {exc}", "parse_error")

    if moov is None:
        return fail("no moov box: the file's index was never written (recording interrupted, or the file is cut "
                    "off), so it cannot be played or seeked" + (f"; {truncated}" if truncated else ""), "no_moov")
    if truncated:
        findings.append(Finding("error", "truncated", None, truncated))

    try:
        movie_ts, movie_dur, traks, fragmented, used = _walk_moov(rf, moov[0], moov[1], budget_bytes)
    except (ValueError, struct.error, _Truncated) as exc:
        return fail(f"unreadable moov box: {exc}", "parse_error")
    res.fragmented = fragmented or bool(moofs)

    # ---- fragmented: read each moof within the budget ------------------------------
    frag: dict[int, dict] = {}
    if moofs:
        read = n_read = 0
        for s, e in moofs:
            if read + (e - s) > max(budget_bytes - used, 0) and n_read > 0:
                findings.append(Finding("info", "partial_scan", None,
                                        f"read {n_read} of {len(moofs)} fragments to stay under "
                                        f"{budget_bytes / (1 << 20):g} MB (--msgidx-budget-mb); frame counts and timing are estimates"))
                break
            buf = rf.read_range(s, e)
            read += len(buf)
            n_read += 1
            try:
                trafs = _parse_moof(buf[8:] if struct.unpack_from(">I", buf, 0)[0] != 1 else buf[16:], s)
            except (ValueError, struct.error) as exc:
                findings.append(Finding("warn", "parse_error", None, f"unreadable fragment at {s}: {exc}"))
                continue
            for tid, tdt, durs, keys in trafs:
                d = frag.setdefault(tid, {"segments": [], "keys": [], "next": 0})
                start = tdt if tdt is not None else d["next"]
                d["segments"].append((start, durs))
                d["keys"].extend(keys)
                d["next"] = start + sum(x or 0 for x in durs)
        res.fragments_read = n_read

    # ---- tracks ---------------------------------------------------------------------
    total_dur = None
    sampled = any(f.type == "partial_scan" for f in findings)
    for t in traks:
        codec = canonical_codec(t.fourcc)
        tr = Mp4Track(track_id=t.track_id, handler=t.handler, fourcc=t.fourcc, codec=codec,
                      width=t.width, height=t.height, timescale=t.timescale)
        res.tracks.append(tr)
        if t.handler != "vide" or not t.timescale:
            continue
        segments_ns: list[np.ndarray] = []
        frames = 0
        if t.stts:
            ticks = _track_times(t)
            frames = len(ticks) - 1
            tr.duration_s = float(ticks[-1]) / t.timescale
            segments_ns.append((ticks[:-1] * 1_000_000_000) // t.timescale)
            tr.keyframes, tr.max_keyframe_interval = _keyframe_stats(t.stss, frames)
            if t.stts_truncated:
                sampled = True
                findings.append(Finding("info", "partial_scan", stream_name,
                                        "frame duration table larger than the budget; timing from its first part"))
        elif t.track_id in frag:
            d = frag[t.track_id]
            end_ticks = 0
            for start, durs in d["segments"]:
                if any(x is None for x in durs):
                    durs = [x if x is not None else (t.trex_duration or 0) for x in durs]
                ticks = start + np.concatenate(([0], np.cumsum(np.asarray(durs, dtype=np.int64))))
                segments_ns.append((ticks[:-1] * 1_000_000_000) // t.timescale)
                frames += len(durs)
                end_ticks = int(ticks[-1])
            first = d["segments"][0][0] if d["segments"] else 0
            tr.duration_s = (end_ticks - first) / t.timescale if d["segments"] else None
            keys = np.flatnonzero(np.asarray(d["keys"], dtype=bool)) + 1
            tr.keyframes, tr.max_keyframe_interval = _keyframe_stats(keys if len(d["keys"]) else None, frames)
            # Contiguous fragments form one timeline; keep them in one run so gaps between them count.
            if segments_ns and not sampled:
                segments_ns = [np.concatenate(segments_ns)]
        tr.frame_count = frames
        rate, jitter, gaps, max_gap = timing_stats([s for s in segments_ns if len(s)]) if segments_ns else (None, None, 0, None)
        tr.rate_hz = rate
        if tr.duration_s is not None:
            total_dur = max(total_dur or 0.0, tr.duration_s)
        topic = stream_name or "video"
        if sum(1 for x in traks if x.handler == "vide") > 1:
            topic = f"{topic}#{t.track_id}"
        channels.append(ChannelProfile(
            topic=topic, schema=f"video/{codec or t.fourcc or 'unknown'}", modality="video", codec=codec,
            resolution=[t.width, t.height] if t.width and t.height else None, dims=None, msg_count=frames,
            rate_hz=rate, jitter_ms=jitter, gaps_over_3x=gaps, max_gap_ms=max_gap))
        if gaps and (max_gap or 0) > 100:
            findings.append(Finding("warn", "dropped_frames", topic,
                                    f"{gaps} gaps > 3x the mean frame interval; worst {max_gap:.0f} ms"))
        if frames == 0:
            findings.append(Finding("warn", "empty_track", topic, "video track has no frames"))

    if not channels:
        findings.append(Finding("warn", "no_video_track", None,
                                "no video track (tracks: " + (", ".join(f"{t.handler}/{t.fourcc}" for t in res.tracks)
                                                              or "none") + ")"))
    if total_dur is None and movie_ts and movie_dur:
        total_dur = movie_dur / movie_ts
    res.duration_s = total_dur
    if total_dur:
        res.bitrate_kbps = round(mdat_bytes * 8 / total_dur / 1000, 1)

    # ---- the dataset's own description of this camera --------------------------------
    vt = next((t for t in res.tracks if t.handler == "vide"), None)
    if declared and vt is not None:
        topic = channels[0].topic if channels else stream_name
        fps = declared.get("fps")
        if fps and vt.rate_hz and abs(vt.rate_hz - fps) > 0.01 * fps:
            findings.append(Finding("warn", "fps_mismatch", topic,
                                    f"video runs at {vt.rate_hz:.2f} fps, dataset metadata says {fps:g}"))
        w, h = declared.get("width"), declared.get("height")
        if w and h and vt.width and vt.height and (int(w), int(h)) != (vt.width, vt.height):
            findings.append(Finding("warn", "resolution_mismatch", topic,
                                    f"video is {vt.width}x{vt.height}, dataset metadata says {int(w)}x{int(h)}"))
        dc = canonical_codec(declared.get("codec"))
        if dc and vt.codec and dc != vt.codec:
            findings.append(Finding("info", "codec_mismatch", topic,
                                    f"video codec is {vt.codec}, dataset metadata says {dc}"))
    log(f"    mp4: {len(res.tracks)} track(s), moov {'at start' if res.moov_at_start else 'at end'}"
        f"{', fragmented' if res.fragmented else ''}")
    return done(total_dur, indexed=True)


def declared_videos(info: dict) -> dict[str, dict]:
    """Per-camera fps/width/height/codec from a LeRobot meta/info.json (v2.x and v3)."""
    out: dict[str, dict] = {}
    for key, feat in (info.get("features") or {}).items():
        if not isinstance(feat, dict) or feat.get("dtype") != "video":
            continue
        vi = feat.get("info") or feat.get("video_info") or {}
        shape = feat.get("shape") or []
        names = feat.get("names") or []
        h = vi.get("video.height")
        w = vi.get("video.width")
        if (h is None or w is None) and len(shape) == 3:
            # shape is [height, width, channels] unless names say otherwise
            order = [str(n).lower() for n in names] if isinstance(names, list) else []
            if order[:2] == ["channels", "height"] or order[:1] == ["channel"]:
                h, w = shape[1], shape[2]
            else:
                h, w = shape[0], shape[1]
        out[key] = {"fps": vi.get("video.fps") or info.get("fps"), "width": w, "height": h,
                    "codec": vi.get("video.codec")}
    return out
