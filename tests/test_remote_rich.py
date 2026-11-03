#!/usr/bin/env python3
"""Checks that remote mode (range reads, nothing downloaded) matches the full-read RichInspector.

    python tests/test_remote_rich.py

Covers: MCAP with and without a summary, unchunked, truncated, sampled message
indexes; MAT v5/v7 NaN counts within and over the NaN budget.
Exits non-zero on the first failed check.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
from mcap.writer import IndexType, Writer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.inspectors.rich import RichInspector  # noqa: E402
from phys_ai_dataprofiler.remote.inspector import inspect_mat_remote, inspect_mcap_remote  # noqa: E402
from phys_ai_dataprofiler.remote.range_file import LocalRangeFile  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="remote_rich_"))


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def findings(prof) -> set[str]:
    return {f.type for f in prof.findings}


def stats(prof) -> dict:
    return {c.topic: (c.msg_count, round(c.rate_hz or 0, 2), c.gaps_over_3x, round(c.max_gap_ms or 0, 1))
            for c in prof.channels}


def mcap_file(name: str, n: int = 3000, gap: bool = True, payload: int = 40, **kw) -> Path:
    """Two channels at 50 Hz each, one 0.5 s dropout halfway through."""
    path = TMP / f"{name}.mcap"
    with path.open("wb") as f:
        w = Writer(f, chunk_size=4000, **kw)
        w.start("test", "test")
        s = w.register_schema("sensor_msgs/Imu", "jsonschema", b"{}")
        chans = [w.register_channel("/imu", "json", s), w.register_channel("/cam/image", "json", s)]
        data = np.random.default_rng(0).bytes(payload)  # incompressible, so chunk sizes are realistic
        t = 0
        for k in range(n):
            t += 10_000_000 + (500_000_000 if gap and k == n // 2 else 0)
            w.add_message(chans[k % 2], log_time=t, data=data, publish_time=t)
        w.finish()
    return path


def remote(path: Path, **kw):
    rf = LocalRangeFile(path)
    res, prof = inspect_mcap_remote(rf, **kw)
    return res, prof, rf


def main() -> None:
    # 1. indexed MCAP: identical to a full read
    full = mcap_file("full")
    _, prof, _ = remote(full)
    check(stats(prof) == stats(RichInspector(str(full)).inspect()) and "dropped_frames" in findings(prof),
          f"indexed MCAP matches full read: {stats(prof)}")

    # 2. sampled message indexes: no gap invented at the seams between sampled chunks
    steady = mcap_file("steady", n=5000, gap=False)
    _, prof, _ = remote(steady, msgidx_budget_bytes=3000)
    imu = next(c for c in prof.channels if c.topic == "/imu")
    check("sampled_message_index" in findings(prof) and "dropped_frames" not in findings(prof)
          and imu.gaps_over_3x == 0 and abs(imu.rate_hz - 50) < 0.5,
          f"sampled indexes on a steady stream: {imu.rate_hz:.1f} Hz, {imu.gaps_over_3x} gaps")

    # 3. no summary section: walked record by record, same answer as a full read
    nosum = mcap_file("nosummary", index_types=IndexType.NONE, use_statistics=False,
                      use_summary_offsets=False, repeat_channels=False, repeat_schemas=False)
    res, prof, _ = remote(nosum)
    check(stats(prof) == stats(RichInspector(str(nosum)).inspect()) and res.topics == ["/cam/image", "/imu"]
          and "unindexed_mcap" in findings(prof) and prof.duration_s is not None,
          f"unindexed MCAP walked: {stats(prof)}")

    # 4. unchunked: no message indexes exist, timestamps from message headers; image-sized
    #    payloads (larger than a cache block) are skipped
    flat = mcap_file("unchunked", n=200, payload=300_000, use_chunking=False)
    _, prof, rf = remote(flat)
    check(stats(prof) == stats(RichInspector(str(flat)).inspect()) and rf.bytes_fetched < 0.5 * rf.size,
          f"unchunked MCAP: {stats(prof)}, read {rf.bytes_fetched / rf.size:.0%} (payloads skipped)")

    # 5. truncated recording: the full-read inspector fails, remote mode profiles what is there
    trunc = TMP / "trunc.mcap"
    trunc.write_bytes(full.read_bytes()[: int(full.stat().st_size * 0.6)])
    try:
        RichInspector(str(trunc)).inspect()
        local_failed = False
    except Exception:
        local_failed = True
    res, prof, _ = remote(trunc)
    counts = {c.topic: c.msg_count for c in prof.channels}
    check(res.error is None and "truncated" in findings(prof) and all(800 < n < 1500 for n in counts.values()),
          f"truncated MCAP profiled (full read {'fails' if local_failed else 'works'}): {counts}")

    # 6. walk budget (counts bytes fetched): stops early and says so
    big_mcap = mcap_file("nosummary_big", payload=2000, index_types=IndexType.NONE, use_statistics=False,
                    use_summary_offsets=False, repeat_channels=False, repeat_schemas=False)
    _, prof, rf = remote(big_mcap, msgidx_budget_bytes=1 << 20)
    check("partial_scan" in findings(prof) and rf.bytes_fetched < 0.5 * rf.size,
          f"walk budget respected: read {rf.bytes_fetched // 1024} KB of {rf.size // 1024} KB, {sorted(findings(prof))}")

    # 7. MAT NaN counts over range reads, v5 and v7 (compressed)
    for compressed in (False, True):
        path = TMP / f"nan_{compressed}.mat"
        sig = np.random.default_rng(0).standard_normal((500, 3))
        sig[3, 1] = sig[7, 2] = np.nan
        big = np.random.default_rng(1).standard_normal(400_000)  # incompressible, ~3 MB stored
        big[5] = np.nan
        scipy.io.savemat(path, {"joint_pos": sig, "f32": np.array([np.nan, 1], dtype=np.float32),
                                "ints": np.arange(4), "big": big}, do_compression=compressed)
        local = {c.topic: c.nan_count for c in RichInspector(str(path)).inspect().channels}
        _, prof = inspect_mat_remote(LocalRangeFile(path))
        got = {c.topic: c.nan_count for c in prof.channels}
        check(got == local and "nan_values" in findings(prof),
              f"MAT {'v7' if compressed else 'v5'} NaN counts match full read: {got}")

        rf = LocalRangeFile(path)
        _, prof = inspect_mat_remote(rf, nan_budget_bytes=100_000)
        got = {c.topic: c.nan_count for c in prof.channels}
        check(got["joint_pos"] == 2 and got["big"] == 0 and "nan_check_partial" in findings(prof)
              and rf.bytes_fetched < 0.2 * rf.size,
              f"MAT {'v7' if compressed else 'v5'} NaN budget skips the big variable, read "
              f"{rf.bytes_fetched // 1024} KB of {rf.size // 1024} KB")

        _, prof = inspect_mat_remote(LocalRangeFile(path), nan_budget_bytes=0)
        check("nan_check_skipped" in findings(prof), "MAT NaN scan can be disabled")


if __name__ == "__main__":
    main()
