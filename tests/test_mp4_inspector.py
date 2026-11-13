#!/usr/bin/env python3
"""Checks for the MP4 inspector and LeRobot video handling, against files with known answers.

    python tests/test_mp4_inspector.py          # offline checks (MP4s are built here, no ffmpeg needed)
    python tests/test_mp4_inspector.py --hub    # also profiles LeRobot videos on the Hub

The Hub checks compare frame counts with each dataset's meta/info.json (ffprobe gives the same numbers).
Exits non-zero on the first failed check.
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.evaluate import FAIL, PASS, UNKNOWN, evaluate  # noqa: E402
from phys_ai_dataprofiler.remote.inspector import sniff_format  # noqa: E402
from phys_ai_dataprofiler.remote.mp4 import (  # noqa: E402
    declared_videos, inspect_mp4, recording_seconds, video_stream_name,
)
from phys_ai_dataprofiler.remote.range_file import LocalRangeFile  # noqa: E402
from phys_ai_dataprofiler.requirements import parse_requirements  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="mp4_test_"))


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


# --------------------------------------------------------------------------- #
# A minimal MP4 writer: enough boxes for a valid index, frames are dummy bytes
# --------------------------------------------------------------------------- #

def box(typ: str, *parts: bytes) -> bytes:
    body = b"".join(parts)
    return struct.pack(">I4s", 8 + len(body), typ.encode()) + body


def full(typ: str, version: int, flags: int, *parts: bytes) -> bytes:
    return box(typ, bytes([version]) + flags.to_bytes(3, "big"), *parts)


def video_entry(fourcc: str, w: int, h: int) -> bytes:
    # VisualSampleEntry: 6 reserved + data_reference_index, 16 pre-defined/reserved, width, height, ...
    return box(fourcc, b"\0" * 6 + struct.pack(">H", 1) + b"\0" * 16 + struct.pack(">HH", w, h)
               + struct.pack(">IIIH", 0x00480000, 0x00480000, 0, 1) + b"\0" * 32 + struct.pack(">Hh", 24, -1))


def trak(track_id: int, handler: str, fourcc: str, timescale: int, deltas: list[int], sync: list[int] | None,
         w: int = 0, h: int = 0, fragmented: bool = False) -> bytes:
    dur = sum(deltas)
    tkhd = full("tkhd", 0, 3, struct.pack(">IIII", 0, 0, track_id, 0), struct.pack(">I", dur), b"\0" * 60,
                struct.pack(">II", w << 16, h << 16))
    mdhd = full("mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, timescale, dur, 0x55c4, 0))
    hdlr = full("hdlr", 0, 0, b"\0" * 4 + handler.encode() + b"\0" * 12 + b"t\0")
    entry = video_entry(fourcc, w, h) if handler == "vide" else box(fourcc, b"\0" * 28)
    stsd = full("stsd", 0, 0, struct.pack(">I", 1), entry)
    runs: list[list[int]] = []
    for d in ([] if fragmented else deltas):
        if runs and runs[-1][1] == d:
            runs[-1][0] += 1
        else:
            runs.append([1, d])
    stts = full("stts", 0, 0, struct.pack(">I", len(runs)), b"".join(struct.pack(">II", c, d) for c, d in runs))
    n = 0 if fragmented else len(deltas)
    tables = [stsd, stts]
    if sync is not None and not fragmented:
        tables.append(full("stss", 0, 0, struct.pack(">I", len(sync)), b"".join(struct.pack(">I", s) for s in sync)))
    tables += [full("stsc", 0, 0, struct.pack(">I", 1), struct.pack(">III", 1, max(n, 1), 1)),
               full("stsz", 0, 0, struct.pack(">II", 0, n), b"".join(struct.pack(">I", 10) for _ in range(n))),
               full("stco", 0, 0, struct.pack(">II", 1, 0))]
    minf = box("minf", full("vmhd" if handler == "vide" else "smhd", 0, 1, b"\0" * 8), box("stbl", *tables))
    return box("trak", tkhd, box("mdia", mdhd, hdlr, minf))


def mp4(path: Path, deltas: list[int], timescale: int = 15360, sync: list[int] | None = None, w: int = 640,
        h: int = 480, fourcc: str = "av01", faststart: bool = True, audio: bool = False) -> Path:
    mvhd = full("mvhd", 0, 0, struct.pack(">IIII", 0, 0, 1000, int(sum(deltas) / timescale * 1000)), b"\0" * 80)
    traks = [trak(1, "vide", fourcc, timescale, deltas, sync, w, h)]
    if audio:
        traks.append(trak(2, "soun", "mp4a", 48000, [1024] * 10, None))
    moov = box("moov", mvhd, *traks)
    mdat = box("mdat", b"\x00" * (10 * len(deltas)))
    ftyp = box("ftyp", b"isom", struct.pack(">I", 512), b"isomiso2av01mp41")
    path.write_bytes(ftyp + (moov + mdat if faststart else mdat + moov))
    return path


def fragmented_mp4(path: Path, frags: list[list[int]], timescale: int = 12800, w: int = 64, h: int = 48) -> Path:
    """One moof+mdat per entry of `frags` (each a list of frame durations); first frame of each is a keyframe."""
    mvhd = full("mvhd", 0, 0, struct.pack(">IIII", 0, 0, 1000, 0), b"\0" * 80)
    mvex = box("mvex", full("trex", 0, 0, struct.pack(">IIIII", 1, 1, 0, 0, 0)))
    out = box("ftyp", b"iso5", struct.pack(">I", 512), b"iso5iso6mp41") + box(
        "moov", mvhd, trak(1, "vide", "avc1", timescale, [], None, w, h, fragmented=True), mvex)
    t = 0
    for seq, durs in enumerate(frags, 1):
        tfhd = full("tfhd", 0, 0x020000, struct.pack(">I", 1))
        tfdt = full("tfdt", 1, 0, struct.pack(">Q", t))
        samples = b"".join(struct.pack(">III", d, 10, 0 if k == 0 else 0x10000) for k, d in enumerate(durs))
        trun = full("trun", 0, 0x000701, struct.pack(">Ii", len(durs), 0), samples)
        out += box("moof", full("mfhd", 0, 0, struct.pack(">I", seq)), box("traf", tfhd, tfdt, trun))
        out += box("mdat", b"\0" * (10 * len(durs)))
        t += sum(durs)
    path.write_bytes(out)
    return path


def profile(path: Path, name: str = "cam", block_size: int = 4096, **kw):
    rf = LocalRangeFile(path, block_size=block_size)
    try:
        res, prof = inspect_mp4(rf, name, **kw)
        return res, prof, rf.bytes_fetched
    finally:
        rf.close()


def types(prof) -> set[str]:
    return {f.type for f in prof.findings}


# --------------------------------------------------------------------------- #
# Inspector
# --------------------------------------------------------------------------- #

def test_inspector() -> None:
    # 30 fps at timescale 15360 (512 ticks per frame), 300 frames, keyframe every 30 frames
    p = mp4(TMP / "fast.mp4", [512] * 300, sync=list(range(1, 301, 30)))
    rf = LocalRangeFile(p)
    check(sniff_format(rf, "x.bin") == "mp4", "sniffed as MP4 from its first box, whatever the extension")
    rf.close()
    res, prof, _ = profile(p)
    c = prof.channels[0]
    check(res.moov_at_start and not res.fragmented and res.error is None, "fast-start MP4: moov before mdat")
    check(c.msg_count == 300 and abs(c.rate_hz - 30) < 1e-6 and c.gaps_over_3x == 0 and abs(prof.duration_s - 10) < 1e-9,
          f"300 frames at 30 fps over 10 s: {c.msg_count}, {c.rate_hz}, {prof.duration_s}")
    check(c.modality == "video" and c.codec == "av1" and c.resolution == [640, 480] and c.topic == "cam",
          "video stream with codec, resolution and the camera name as topic")
    t = res.tracks[0]
    check(t.keyframes == 10 and t.max_keyframe_interval == 30, f"keyframes from stss: {t.keyframes}, every {t.max_keyframe_interval}")

    res, prof, fetched = profile(mp4(TMP / "end.mp4", [512] * 30000, faststart=False))
    check(not res.moov_at_start and prof.channels[0].msg_count == 30000, "moov at the end is found after mdat")
    size = (TMP / "end.mp4").stat().st_size
    check(fetched < 0.6 * size, f"mdat is skipped, not read ({fetched} of {size} B)")
    check(res.tracks[0].keyframes is None, "no stss: every frame is a keyframe (reported as None)")

    # variable frame durations: a 0.5 s hole in a 20 fps stream
    deltas = [768] * 20 + [7680] + [768] * 29          # timescale 15360: 768 = 50 ms
    res, prof, _ = profile(mp4(TMP / "gap.mp4", deltas))
    c = prof.channels[0]
    check(c.gaps_over_3x == 1 and abs(c.max_gap_ms - 500) < 1e-6 and "dropped_frames" in types(prof),
          f"a 500 ms hole in the timeline is a dropped-frames finding: {c.gaps_over_3x}, {c.max_gap_ms}")

    res, prof, _ = profile(mp4(TMP / "av.mp4", [512] * 30, audio=True))
    check(len(res.tracks) == 2 and len(prof.channels) == 1 and res.tracks[1].handler == "soun",
          "audio track listed, but only the video track is a stream")

    res, prof, _ = profile(mp4(TMP / "h264.mp4", [1000] * 25, timescale=25000, fourcc="avc1", w=1280, h=720))
    check(prof.channels[0].codec == "h264" and prof.channels[0].resolution == [1280, 720], "avc1 is reported as h264")

    # declared metadata disagrees
    res, prof, _ = profile(TMP / "fast.mp4", declared={"fps": 15, "width": 320, "height": 240, "codec": "h264"})
    check({"fps_mismatch", "resolution_mismatch", "codec_mismatch"} <= types(prof),
          "fps, resolution and codec checked against the dataset's metadata")
    res, prof, _ = profile(TMP / "fast.mp4", declared={"fps": 30, "width": 640, "height": 480, "codec": "libsvtav1"})
    check(not types(prof), "matching metadata gives no findings (encoder names are mapped to codecs)")


def test_damaged() -> None:
    whole = mp4(TMP / "whole.mp4", [512] * 200, faststart=False).read_bytes()
    (TMP / "cut.mp4").write_bytes(whole[:len(whole) // 2])
    res, prof, _ = profile(TMP / "cut.mp4")
    check(res.is_mp4 and "no_moov" in types(prof) and "never written" in res.error and not prof.channels,
          "cut before the index: no_moov error, nothing guessed")

    fast = mp4(TMP / "fast2.mp4", [512] * 200).read_bytes()
    (TMP / "cut2.mp4").write_bytes(fast[:-500])
    res, prof, _ = profile(TMP / "cut2.mp4")
    check(res.error is None and "truncated" in types(prof) and prof.channels[0].msg_count == 200,
          "fast-start file cut in mdat: index still read, truncation flagged")

    (TMP / "junk.mp4").write_bytes(b"\x00\x00\x00\x10notabox" + b"\x00" * 100)
    res, prof, _ = profile(TMP / "junk.mp4")
    check(not res.is_mp4 and "bad_magic" in types(prof), "not an MP4: bad_magic, no crash")

    (TMP / "badsize.mp4").write_bytes(box("ftyp", b"isom\0\0\0\0") + struct.pack(">I4s", 4, b"moov"))
    res, prof, _ = profile(TMP / "badsize.mp4")
    check(res.error is not None and not prof.channels, "box size smaller than its header: error, no loop")

    (TMP / "empty.mp4").write_bytes(box("ftyp", b"isom\0\0\0\0") + box("moov", full("mvhd", 0, 0, b"\0" * 96)))
    res, prof, _ = profile(TMP / "empty.mp4")
    check(res.error is None and "no_video_track" in types(prof), "moov without tracks: no_video_track finding")


def test_fragmented() -> None:
    p = fragmented_mp4(TMP / "frag.mp4", [[512] * 25] * 4)
    res, prof, _ = profile(p)
    c = prof.channels[0]
    check(res.fragmented and res.fragments_read == 4 and c.msg_count == 100 and abs(c.rate_hz - 25) < 1e-6,
          f"fragmented MP4: frames and rate from the moof boxes ({c.msg_count}, {c.rate_hz})")
    check(res.tracks[0].keyframes == 4 and res.tracks[0].max_keyframe_interval == 25, "keyframes from trun sample flags")
    check(abs(prof.duration_s - 4) < 1e-9 and c.gaps_over_3x == 0, "fragments joined into one timeline, no gap")
    res, prof, _ = profile(p, budget_bytes=400)
    check("partial_scan" in types(prof) and 0 < res.fragments_read < 4, "fragment walk stops at the budget, says so")
    gap = fragmented_mp4(TMP / "fraggap.mp4", [[512] * 25, [512] * 24 + [512 * 10], [512] * 25])
    res, prof, _ = profile(gap)
    check(prof.channels[0].gaps_over_3x == 1, "a long frame inside a fragment is a gap")


# --------------------------------------------------------------------------- #
# LeRobot layout, durations and requirements
# --------------------------------------------------------------------------- #

def test_lerobot() -> None:
    check(video_stream_name("videos/observation.images.wrist/chunk-000/file-003.mp4")
          == ("observation.images.wrist", "videos/*"),
          "LeRobot v3 path: camera, and all videos one group (cameras split files at different episodes)")
    check(video_stream_name("videos/chunk-001/observation.images.top/episode_000042.mp4")
          == ("observation.images.top", "videos/chunk-001/*/episode_000042.mp4"), "LeRobot v2 path: camera and episode")
    check(video_stream_name("clips/run1.mp4") == ("run1", "clips/run1.mp4"), "other layouts: one recording per file")

    info = {"fps": 30, "features": {
        "observation.images.up": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"],
                                  "info": {"video.fps": 30, "video.codec": "av1"}},
        "observation.images.v2": {"dtype": "video", "shape": [3, 96, 128], "names": ["channels", "height", "width"],
                                  "video_info": {"video.fps": 10.0, "video.codec": "h264"}},
        "observation.state": {"dtype": "float32", "shape": [6]}}}
    dv = declared_videos(info)
    check(dv == {"observation.images.up": {"fps": 30, "width": 640, "height": 480, "codec": "av1"},
                 "observation.images.v2": {"fps": 10.0, "width": 128, "height": 96, "codec": "h264"}},
          f"meta/info.json cameras, v3 and v2 field names: {dv}")

    items = [("videos/a/chunk-000/file-000.mp4", "mp4", 100.0), ("videos/b/chunk-000/file-000.mp4", "mp4", 99.0),
             ("videos/a/chunk-000/file-001.mp4", "mp4", 50.0)]
    check(recording_seconds(items) == 150.0, "v3: each camera's files summed, longest camera counted once")
    v2 = [("videos/chunk-000/a/episode_000000.mp4", "mp4", 10.0), ("videos/chunk-000/b/episode_000000.mp4", "mp4", 10.0),
          ("videos/chunk-000/a/episode_000001.mp4", "mp4", 7.0), ("videos/chunk-000/b/episode_000001.mp4", "mp4", 7.5)]
    check(recording_seconds(v2) == 17.5, "v2: one group per episode, its longest camera")
    uneven = [("videos/a/chunk-000/file-000.mp4", "mp4", 100.0), ("videos/a/chunk-000/file-001.mp4", "mp4", 50.0),
              ("videos/b/chunk-000/file-000.mp4", "mp4", 150.0)]
    check(recording_seconds(uneven) == 150.0, "v3 cameras split into different numbers of files: still 150 s")
    check(recording_seconds(items + [("data/chunk-000/file-000.parquet", "parquet", 148.0)]) == 148.0,
          "with a timed Parquet table, the videos add no time")


def rep_file(path, fmt, channels, duration, valid=True, findings=(), basic=None):
    return {"repo_path": path, "detected_format": fmt, "valid": valid, "error": None if valid else "broken",
            "summary": {"duration_s": duration}, "basic": basic or {},
            "rich": {"channels": channels, "findings": [{"type": t} for t in findings], "indexed": True}}


def chan(topic, modality, rate, gap=None):
    return {"topic": topic, "modality": modality, "schema": "x", "msg_count": 1000, "rate_hz": rate,
            "jitter_ms": 0.0, "max_gap_ms": gap or 1000 / rate, "gaps_over_3x": 0, "nan_count": 0}


def lerobot_report(cam_rate=30.0, wrist=True, broken_wrist=False):
    eps = {"count": 50, "complete": True, "seconds_min": 5.0}
    files = [rep_file("data/chunk-000/file-000.parquet", "parquet",
                      [chan("observation.state", "proprio", 30), chan("action", "action", 30)], 3600.0,
                      basic={"episodes": eps})]
    for chunk in ("file-000", "file-001"):
        files.append(rep_file(f"videos/observation.images.top/chunk-000/{chunk}.mp4", "mp4",
                              [chan("observation.images.top", "video", cam_rate)], 1800.0))
        if wrist:
            files.append(rep_file(f"videos/observation.images.wrist/chunk-000/{chunk}.mp4", "mp4",
                                  [] if broken_wrist else [chan("observation.images.wrist", "video", cam_rate)],
                                  None if broken_wrist else 1800.0, valid=not broken_wrist))
    return {"repo_id": "org/ds", "files": files, "files_skipped": [],
            "totals": {"duration_s": 3600.0}}


def verdict(report, doc):
    return evaluate(report, parse_requirements(doc, "test"))


def status(report, doc, cid=None):
    v = verdict(report, doc)
    return v.status if cid is None else next(c.status for c in v.checks if c.id == cid)


def test_requirements() -> None:
    cam = {"name": "cam", "match": {"modality": "video"}, "min_rate_hz": 25}
    state = {"name": "state", "match": {"modality": "proprio"}, "min_rate_hz": 25}
    r = lerobot_report()
    v = verdict(r, {"streams": [cam, state], "dataset": {"min_total_hours": 1, "min_episodes": 50}})
    check(v.status == PASS, "LeRobot: camera judged on the videos, state on the table, both pass")
    hours = next(c for c in v.checks if c.id == "dataset:min_total_hours")
    check(hours.status == PASS and hours.observed.startswith("1.00 h"),
          f"hours from the table, not table + every camera file: {hours.observed}")
    cam_check = next(c for c in v.checks if c.id == "stream:cam")
    check("1 of 1 video set" in cam_check.observed and "2 files" in cam_check.files[0]["detail"],
          f"v3 videos are judged as one set, each camera over its files: {cam_check.files[0]['detail']}")

    check(status(lerobot_report(cam_rate=15), {"streams": [cam]}) == FAIL, "slow cameras fail")
    two = {"name": "two cams", "match": {"modality": "video"}, "min_count": 2}
    check(status(lerobot_report(), {"streams": [two]}) == PASS, "two cameras in separate files meet min_count 2")
    check(status(lerobot_report(wrist=False), {"streams": [two]}) == FAIL, "one camera does not meet min_count 2")
    check(status(lerobot_report(broken_wrist=True), {"streams": [two]}) == UNKNOWN,
          "an unreadable camera file makes min_count 2 unknown, not fail")
    wrist = {"name": "wrist", "match": {"modality": "video", "topic": "*wrist*"}}
    check(status(lerobot_report(wrist=False), {"streams": [wrist]}) == FAIL, "a named camera that is absent fails")
    check(status(lerobot_report(), {"streams": [wrist]}) == PASS, "a named camera that is present passes")

    table_only = lerobot_report()
    table_only["files"] = table_only["files"][:1]
    v = verdict(table_only, {"streams": [cam]})
    check(v.status == UNKNOWN and "keep .mp4 in --ext" in (v.checks[0].resolve or ""),
          "table without its videos: camera unknown, and the hint says to profile the videos")

    videos_only = lerobot_report()
    videos_only["files"] = videos_only["files"][1:]
    check(status(videos_only, {"dataset": {"min_episodes": 10}}, "dataset:min_episodes") == UNKNOWN,
          "v3 videos alone do not give an episode count")
    check(status(videos_only, {"dataset": {"min_total_hours": 1}}) == PASS,
          "v3 videos alone: hours from one camera per recording (2 x 30 min)")


def v2_report(rates_b=(30.0, 30.0)):
    files = []
    for ep, rb in enumerate(rates_b):
        files.append(rep_file(f"videos/chunk-000/a/episode_{ep:06d}.mp4", "mp4", [chan("a", "video", 30.0)], 10.0))
        files.append(rep_file(f"videos/chunk-000/b/episode_{ep:06d}.mp4", "mp4", [chan("b", "video", rb)], 10.0))
    return {"repo_id": "org/v2", "files": files, "files_skipped": [], "totals": {"duration_s": 20.0}}


def test_more_requirements() -> None:
    two = {"name": "two", "match": {"modality": "video"}, "min_count": 2, "min_rate_hz": 25}
    v = verdict(v2_report(), {"streams": [two], "dataset": {"min_episodes": 2, "min_total_hours": 0.005}})
    check(v.status == PASS and "2 of 2 recording" in v.checks[0].observed,
          f"v2: each episode's cameras together, coverage per episode: {v.checks[0].observed}")
    eps = next(c for c in v.checks if c.id == "dataset:min_episodes")
    check(eps.status == PASS and eps.observed.startswith("2 "), f"v2 videos alone: one episode per group ({eps.observed})")
    check(status(v2_report((30.0, 10.0)), {"streams": [two]}) == FAIL, "v2: a slow camera in one episode fails it")
    half = dict(two, coverage=0.5)
    check(status(v2_report((30.0, 10.0)), {"streams": [half]}) == PASS, "coverage applies across v2 episodes")
    r = lerobot_report()
    r["files"][3]["rich"]["channels"][0]["rate_hz"] = 5.0       # one of top's two files is slow
    cam = {"name": "cam", "match": {"modality": "video"}, "min_count": "all", "min_rate_hz": 25}
    check(status(r, {"streams": [cam]}) == FAIL, "v3: one slow file of a camera fails it")
    check(status(r, {"streams": [dict(cam, coverage=0.5)]}) == PASS, "coverage applies across a v3 camera's files")


def test_hub() -> None:
    from phys_ai_dataprofiler.remote.range_file import HfRangeFile
    for repo, path, frames, fps in (("lerobot/pusht", "videos/observation.image/chunk-000/file-000.mp4", 25650, 10),
                                    ("lerobot/svla_so101_pickplace",
                                     "videos/observation.images.up/chunk-000/file-000.mp4", 11939, 30)):
        rf = HfRangeFile(repo, path)
        _, prof = inspect_mp4(rf, video_stream_name(path)[0])
        c = prof.channels[0]
        check(c.msg_count == frames and abs(c.rate_hz - fps) < 0.01 and rf.bytes_fetched < 0.05 * rf.size,
              f"Hub {repo}: {c.msg_count} frames at {c.rate_hz:.2f} fps (meta/info.json: {frames} @ {fps}); "
              f"read {rf.bytes_fetched // 1024} KB of {rf.size // 1024 // 1024} MB")


def main(hub: bool = False) -> None:
    test_inspector()
    test_damaged()
    test_fragmented()
    test_lerobot()
    test_requirements()
    test_more_requirements()
    if hub:
        test_hub()
    print("all checks passed")


if __name__ == "__main__":
    main(hub="--hub" in sys.argv)
