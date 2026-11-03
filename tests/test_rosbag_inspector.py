#!/usr/bin/env python3
"""Checks for the ROS 1 / ROS 2 (SQLite) bag inspectors against bags with known contents.

    python tests/test_rosbag_inspector.py          # offline checks (needs `rosbags` to write test bags)
    python tests/test_rosbag_inspector.py --hub    # also profiles small public bags on the Hub

Every offline case is compared with a full read through `rosbags`, so the
range-read numbers must equal what a complete scan gives.
Exits non-zero on the first failed check.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader as Reader1, Writer as Writer1
from rosbags.rosbag2 import Reader as Reader2, Writer as Writer2
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.inspectors.rich import profile_mcap_channels  # noqa: E402
from phys_ai_dataprofiler.remote.inspector import sniff_format  # noqa: E402
from phys_ai_dataprofiler.remote.range_file import LocalRangeFile  # noqa: E402
from phys_ai_dataprofiler.remote.rosbag import inspect_bag1_remote, inspect_bag2_remote  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="rosbag_test_"))
TS1 = get_typestore(Stores.ROS1_NOETIC)
TS2 = get_typestore(Stores.ROS2_HUMBLE)


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def findings(prof) -> set[str]:
    return {f.type for f in prof.findings}


def stats(prof) -> dict:
    return {c.topic: (c.msg_count, round(c.rate_hz or 0, 2), c.gaps_over_3x, round(c.max_gap_ms or 0, 1))
            for c in prof.channels}


def schedule(n: int, gap: bool = True):
    """(topic index, timestamp) for n messages: /imu and /camera/image alternating at 100 Hz combined,
    with one 0.5 s dropout halfway through."""
    t = 1_700_000_000_000_000_000
    for k in range(n):
        t += 10_000_000 + (500_000_000 if gap and k == n // 2 else 0)
        yield k % 2, t


TOPICS = [("/imu", "sensor_msgs/msg/Imu"), ("/camera/image", "sensor_msgs/msg/Image")]


def bag1(name: str, n: int = 4000, compression=None, chunk_threshold: int = 4000, payload: int = 60,
         gap: bool = True) -> Path:
    path = TMP / f"{name}.bag"
    w = Writer1(path)
    w.chunk_threshold = chunk_threshold
    if compression:
        w.set_compression(compression)
    rng = np.random.default_rng(0)  # a fresh payload per message, so compressed chunks stay realistic
    with w:
        conns = [w.add_connection(t, ty, typestore=TS1) for t, ty in TOPICS]
        for i, t in schedule(n, gap):
            w.write(conns[i], t, rng.bytes(payload))
    return path


def bag2(name: str, n: int = 4000, payload: int = 60, gap: bool = True, big_every: int = 0) -> Path:
    """rosbag2 SQLite bag directory; returns the .db3 path. big_every>0 adds 20 KB payloads (overflow pages)."""
    d = TMP / name
    if d.exists():
        shutil.rmtree(d)
    data = np.random.default_rng(0).bytes(payload)
    big = np.random.default_rng(1).bytes(20_000)
    with Writer2(d, version=9) as w:
        conns = [w.add_connection(t, ty, typestore=TS2) for t, ty in TOPICS]
        for k, (i, t) in enumerate(schedule(n, gap)):
            w.write(conns[i], t, big if big_every and k % big_every == 0 else data)
    return next(d.glob("*.db3"))


def reference(path: Path) -> dict:
    """Full-read statistics through rosbags and the shared timing code."""
    reader = Reader1(path) if path.suffix == ".bag" else Reader2(path.parent)
    with reader as r:
        per: dict[int, dict] = {}
        ids = {}
        for conn, ts, _ in r.messages():
            key = ids.setdefault(conn.topic, len(ids))
            s = per.setdefault(key, {"topic": conn.topic, "schema": conn.msgtype, "segments": [[]], "count": 0})
            s["segments"][0].append(ts)
            s["count"] += 1
    for s in per.values():
        s["segments"][0].sort()
    chans, _ = profile_mcap_channels(per)
    return {c.topic: (c.msg_count, round(c.rate_hz or 0, 2), c.gaps_over_3x, round(c.max_gap_ms or 0, 1))
            for c in chans}


def run(fn, path: Path, **kw):
    rf = LocalRangeFile(path, block_size=4096)
    res, prof = fn(rf, **kw)
    return res, prof, rf


def schema_name(prof, topic):
    return next(c.schema for c in prof.channels if c.topic == topic)


def main() -> None:
    # ---------------- ROS 1 ----------------
    p = bag1("plain")
    rf = LocalRangeFile(p)
    check(sniff_format(rf, "x.bin") == "bag1", "bag1: sniffed from magic bytes")
    res, prof, rf = run(inspect_bag1_remote, p)
    ref = reference(p)
    check(res.error is None and res.indexed, "bag1: indexed bag parses")
    check(stats(prof) == ref, f"bag1: stats equal full read {stats(prof)}")
    check(prof.channels[0].gaps_over_3x == 1 or any(c.gaps_over_3x for c in prof.channels),
          "bag1: the 0.5 s dropout is detected")
    check("dropped_frames" in findings(prof), "bag1: dropped_frames finding")
    check(res.chunk_count > 20 and res.compression == ["none"], f"bag1: {res.chunk_count} chunks, uncompressed")
    check(schema_name(prof, "/imu") == "sensor_msgs/Imu", "bag1: message type kept (ROS 1 naming)")
    check(rf.bytes_fetched < 0.25 * rf.size, f"bag1: read {rf.bytes_fetched} of {rf.size} B")

    for comp in (Writer1.CompressionFormat.BZ2, Writer1.CompressionFormat.LZ4):
        p = bag1(f"comp_{comp.name.lower()}", compression=comp, payload=400, chunk_threshold=40_000)
        res, prof, rf = run(inspect_bag1_remote, p)
        check(stats(prof) == reference(p) and res.compression == [comp.name.lower()],
              f"bag1 {comp.name}: stats equal full read, compression reported")
        check(rf.bytes_fetched < 0.25 * rf.size, f"bag1 {comp.name}: read {rf.bytes_fetched} of {rf.size} B")

    p = bag1("sampled", n=20000)
    res, prof, rf = run(inspect_bag1_remote, p, msgidx_budget_bytes=20_000)
    ref = reference(p)
    check("sampled_message_index" in findings(prof), "bag1 sampled: finding reported")
    check({t: v[0] for t, v in stats(prof).items()} == {t: v[0] for t, v in ref.items()},
          "bag1 sampled: counts still exact (from chunk infos)")
    check(all(abs(stats(prof)[t][1] - ref[t][1]) < 1 for t in ref), "bag1 sampled: rates within 1 Hz")

    # Unindexed: the recorder was killed before writing the index.
    p = bag1("plain_copy")
    raw = bytearray(p.read_bytes())
    res_full, _, _ = run(inspect_bag1_remote, p)
    cut = TMP / "unindexed.bag"
    head_end = 13 + 4 + int.from_bytes(raw[13:17], "little")
    idx = raw.find(b"index_pos=", 13, head_end) + len(b"index_pos=")
    index_pos = int.from_bytes(raw[idx: idx + 8], "little")
    raw[idx: idx + 8] = (0).to_bytes(8, "little")
    cut.write_bytes(raw[:index_pos])
    res, prof, _ = run(inspect_bag1_remote, cut)
    check("unindexed_bag" in findings(prof) and not res.indexed, "bag1 unindexed: reported")
    check(stats(prof) == reference(p), "bag1 unindexed: walk equals full read of the original")

    trunc = TMP / "truncated.bag"
    trunc.write_bytes(bytes(raw[: index_pos // 2]))
    res, prof, _ = run(inspect_bag1_remote, trunc)
    check("truncated" in findings(prof) and res.message_count > 0,
          f"bag1 truncated mid-chunk: {res.message_count} messages recovered, truncation reported")

    bad = TMP / "v12.bag"
    bad.write_bytes(b"#ROSBAG V1.2\n" + b"\0" * 64)
    res, prof, _ = run(inspect_bag1_remote, bad)
    check(res.error and "V1.2" in res.error and "unsupported_version" in findings(prof),
          "bag1: old V1.2 bags are named, not misparsed")

    # ---------------- ROS 2 SQLite ----------------
    p = bag2("small")
    check(sniff_format(LocalRangeFile(p), "x.bin") == "bag2", "bag2: sniffed from SQLite magic + tables")
    res, prof, rf = run(inspect_bag2_remote, p)
    ref = reference(p)
    check(res.error is None and res.metadata_source == "embedded", "bag2: parses, embedded metadata found")
    check(stats(prof) == ref, f"bag2: stats equal full read {stats(prof)}")
    check("dropped_frames" in findings(prof), "bag2: dropped_frames finding")
    check(res.ros_distro == "rosbags" and res.version == "9", "bag2: distro and metadata version")

    p = bag2("deep", n=60000, big_every=50)
    con = sqlite3.connect(p)
    depth_pages = con.execute("select count(*) from dbstat where name='messages' and pagetype='internal'").fetchone()[0] \
        if con.execute("select 1 from pragma_compile_options where compile_options like 'ENABLE_DBSTAT%'").fetchone() else None
    con.close()
    res, prof, rf = run(inspect_bag2_remote, p)
    ref = reference(p)
    check(stats(prof) == ref, "bag2 multi-level B-tree with overflow rows: stats equal full read"
          + (f" ({depth_pages} interior pages)" if depth_pages else ""))
    check(rf.bytes_fetched < 0.5 * rf.size, f"bag2: overflow pages skipped, read {rf.bytes_fetched} of {rf.size} B")

    res, prof, rf = run(inspect_bag2_remote, p, msgidx_budget_bytes=200_000)
    check("sampled_message_index" in findings(prof), "bag2 sampled: finding reported")
    check({t: v[0] for t, v in stats(prof).items()} == {t: v[0] for t, v in ref.items()},
          "bag2 sampled: counts exact from metadata")
    check(all(abs(stats(prof)[t][1] - ref[t][1]) < 1 for t in ref), "bag2 sampled: rates within 1 Hz")
    with Reader2(p.parent) as r:
        check(abs(res.duration_s - r.duration / 1e9) < 1e-6, "bag2 sampled: duration from timestamp_idx edges")
    check(rf.bytes_fetched <= 300_000, f"bag2 sampled: read {rf.bytes_fetched} B")

    # No embedded metadata (older rosbag2): the sidecar fills in, else counts are estimated.
    noemb = TMP / "noemb.db3"
    shutil.copy(p, noemb)
    con = sqlite3.connect(noemb)
    con.execute("drop table metadata")
    con.commit()
    con.execute("vacuum")
    con.close()
    res, prof, _ = run(inspect_bag2_remote, noemb, msgidx_budget_bytes=200_000)
    check("estimated_counts" in findings(prof) and res.metadata_source is None,
          "bag2 without metadata: counts estimated and labelled")
    total = sum(v[0] for v in ref.values())
    check(abs(res.message_count - total) <= 1, f"bag2 without metadata: total {res.message_count} vs {total}")
    sidecar = (p.parent / "metadata.yaml").read_text()
    res, prof, _ = run(inspect_bag2_remote, noemb, msgidx_budget_bytes=200_000, sidecar_yaml=sidecar,
                       file_name=p.name)
    check(res.metadata_source == "sidecar" and
          {t: v[0] for t, v in stats(prof).items()} == {t: v[0] for t, v in ref.items()},
          "bag2 with sidecar metadata.yaml: exact counts")

    other = TMP / "other.db3"
    con = sqlite3.connect(other)
    con.execute("create table t(x)")
    con.commit()
    con.close()
    res, prof, _ = run(inspect_bag2_remote, other)
    check(res.error and not res.is_bag, "bag2: unrelated SQLite file is rejected")
    check(sniff_format(LocalRangeFile(other), "data.db3") == "unknown", "bag2: unrelated SQLite not sniffed as bag")

    if "--hub" in sys.argv:
        hub()
    print("all checks passed")


def hub() -> None:
    from phys_ai_dataprofiler.remote.range_file import HfRangeFile
    for repo, path, fn in [
        ("liuxh010/IndoorLocalization_UWB_Camera_LiDAR_IMU", "LiDAR_UWB-2024-02-28-12-56-09.bag", inspect_bag1_remote),
        ("UniflexAI/rosbag_lekiwi_exp_result", "my_bag_0.db3", inspect_bag2_remote),
    ]:
        rf = HfRangeFile(repo, path)
        res, prof = fn(rf)
        print(f"  {repo}/{path}: {len(prof.channels)} topics, {res.message_count} msgs, "
              f"{res.duration_s} s; read {rf.bytes_fetched} of {rf.size} B in {rf.requests} requests")
        check(res.error is None and prof.channels, f"hub {repo}: parsed")


if __name__ == "__main__":
    main()
