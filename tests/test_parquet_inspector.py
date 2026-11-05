#!/usr/bin/env python3
"""Checks for the Parquet inspector against synthetic files with known answers.

    python tests/test_parquet_inspector.py          # offline checks
    python tests/test_parquet_inspector.py --hub    # also profiles two public Hub datasets

Exits non-zero on the first failed check.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.remote.parquet import inspect_parquet  # noqa: E402
from phys_ai_dataprofiler.remote.range_file import LocalRangeFile  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="pq_test_"))


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def findings(prof) -> set[str]:
    return {f.type for f in prof.findings}


def lerobot_like() -> Path:
    """3 episodes at 10 Hz: ep0 clean (50 frames), ep1 has a 0.4 s gap (40), ep2 short (5). 2 NaNs."""
    rows = []
    for ep, n, gap_at in ((0, 50, None), (1, 40, 20), (2, 5, None)):
        t = np.arange(n) * 0.1
        if gap_at is not None:
            t[gap_at:] += 0.4
        rows += [(ep, k, float(t[k])) for k in range(n)]
    n = len(rows)
    rng = np.random.default_rng(0)
    state = rng.random((n, 3), dtype=np.float32)
    state[3, 1] = np.nan
    state[60, 0] = np.inf
    tbl = pa.table({
        "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.ravel()), 3),
        "action": pa.array([list(r) for r in rng.random((n, 2), dtype=np.float32)], type=pa.list_(pa.float32())),
        "timestamp": pa.array([r[2] for r in rows], type=pa.float32()),
        "frame_index": pa.array([r[1] for r in rows], type=pa.int64()),
        "episode_index": pa.array([r[0] for r in rows], type=pa.int64()),
        "task": pa.array(["pick the cube"] * n),
    })
    path = TMP / "lerobot_like.parquet"
    pq.write_table(tbl, path, row_group_size=20)
    return path


def epoch_ns_log() -> Path:
    """Single stream of int64 epoch-ns timestamps at 100 Hz, no episode column, one 200 ms dropout."""
    t = 1_700_000_000_000_000_000 + np.arange(1000, dtype=np.int64) * 10_000_000
    t[500:] += 200_000_000
    path = TMP / "epoch_ns.parquet"
    pq.write_table(pa.table({"log_time": pa.array(t), "x": pa.array(np.zeros(1000))}), path, row_group_size=100)
    return path


def timestamp_typed() -> Path:
    t = pa.array(np.datetime64("2025-01-01") + np.arange(300) * np.timedelta64(20, "ms"))
    path = TMP / "ts_typed.parquet"
    pq.write_table(pa.table({"time": t, "speed": pa.array(np.ones(300))}), path)
    return path


def text_only() -> Path:
    path = TMP / "text.parquet"
    pq.write_table(pa.table({"id": pa.array(range(10)), "text": pa.array([f"row {i}" for i in range(10)]),
                             "meta": pa.array([{"a": i, "b": "x"} for i in range(10)])}), path)
    return path


def write(name: str, columns: dict, **kw) -> Path:
    path = TMP / name
    pq.write_table(pa.table(columns), path, **kw)
    return path


def inspect(path: Path, **kw):
    return inspect_parquet(LocalRangeFile(path), **kw)


def edge_cases() -> None:
    # Integer timestamp units. Relative ns/us counters pass 1e9/1e8 after a few seconds, so magnitude
    # alone would call them seconds; the step size has to decide.
    for label, values, unit in [
        ("relative ns, 200 s", np.arange(20_000, dtype=np.int64) * 10_000_000, "ns"),
        ("relative ns, 5 s", np.arange(500, dtype=np.int64) * 10_000_000, "ns"),
        ("relative us, 200 s", np.arange(20_000, dtype=np.int64) * 10_000, "us"),
        ("relative ms, 2 h", np.arange(720_000, dtype=np.int64) * 10, "ms"),
        ("epoch s", 1_700_000_000 + np.arange(100, dtype=np.int64), "s"),
        ("epoch ms", 1_700_000_000_000 + np.arange(100, dtype=np.int64) * 10, "ms"),
        ("epoch us", 1_700_000_000_000_000 + np.arange(100, dtype=np.int64) * 10_000, "us"),
    ]:
        res, _ = inspect(write(f"unit_{unit}_{len(values)}.parquet", {"t": pa.array(values)}))
        expect = 1.0 if label == "epoch s" else 100.0
        check(res.timestamp_unit == unit and abs(res.frame_rate_hz - expect) < 0.01,
              f"int timestamps, {label}: unit {res.timestamp_unit}, {res.frame_rate_hz:.2f} Hz (want {unit}, {expect:g} Hz)")

    res, _ = inspect(write("float_epoch.parquet", {"timestamp": pa.array(1.7e9 + np.arange(100) * 0.02)}))
    check(res.timestamp_unit == "s" and abs(res.frame_rate_hz - 50) < 0.01, "float epoch seconds: 50 Hz")
    res, _ = inspect(write("duration.parquet", {"time": pa.array(np.arange(100) * np.timedelta64(5, "ms"))}))
    check(res.timestamp_unit == "ms" and abs(res.frame_rate_hz - 200) < 0.01, "duration[ms] column: 200 Hz")

    # Column choice
    res, _ = inspect(write("string_ts.parquet", {"timestamp": pa.array(["a", "b"]), "time": pa.array([0.0, 0.5])}))
    check(res.timestamp_column == "time", "a non-numeric 'timestamp' column is passed over for 'time'")
    res, _ = inspect(write("upper.parquet", {"Timestamp": pa.array(np.arange(10) * 0.1)}))
    check(res.timestamp_column == "Timestamp", "timestamp column found case-insensitively")
    res, _ = inspect(write("float_ep.parquet", {"timestamp": pa.array(np.arange(10) * 0.1),
                                                 "episode_index": pa.array(np.zeros(10))}))
    check(res.episode_column is None, "a float episode_index is not used as an episode column")

    # Degenerate tables
    res, prof = inspect(write("zero_rows.parquet", {"timestamp": pa.array([], type=pa.float64()),
                                                     "x": pa.array([], type=pa.float32())}), nan_check=True)
    check(res.is_parquet and res.num_rows == 0 and res.error is None and prof.duration_s is None
          and res.frame_rate_hz is None, "zero-row table: schema only, no timing, no error")
    res, prof = inspect(write("one_row.parquet", {"timestamp": pa.array([1.0]), "x": pa.array([np.nan])}),
                        nan_check=True)
    check(res.frame_rate_hz is None and res.episodes["count"] == 1 and prof.duration_s == 0
          and "nan_values" in findings(prof), "one-row table: no rate, one zero-length episode, NaN still found")
    res, prof = inspect(write("all_null_ts.parquet", {"timestamp": pa.array([None, None], type=pa.float64())}))
    check(res.timestamp_column is None and res.frame_rate_hz is None, "all-null timestamp column: no timing")
    res, prof = inspect(write("some_null_ts.parquet",
                              {"timestamp": pa.array([0.0, 0.1, None, 0.3, float("nan"), 0.5, 0.6])}))
    # The 5 valid stamps leave intervals 0.1, 0.2, 0.2, 0.1 s: mean 0.15 s.
    check("null_timestamps" in findings(prof) and abs(res.frame_rate_hz - 1 / 0.15) < 0.01,
          f"null and NaN timestamps dropped and reported, rate from the rest: {res.frame_rate_hz:.2f} Hz")

    # Time order
    rng = np.random.default_rng(0)
    _, prof = inspect(write("shuffled.parquet", {"timestamp": pa.array(rng.permutation(1000) * 0.01)}))
    check("timestamps_unsorted" in findings(prof), "shuffled timestamps without episodes: reported as unsorted")
    _, prof = inspect(write("backstep.parquet", {"timestamp": pa.array([0.0, 0.1, 0.2, 0.15, 0.3, 0.4]),
                                                 "episode_index": pa.array([0] * 6)}))
    check("non_monotonic_time" in findings(prof), "time stepping back inside an episode: non_monotonic_time")
    t = np.arange(10) * 0.1
    res, prof = inspect(write("interleaved.parquet", {"timestamp": pa.array(np.repeat(t, 2)),
                                                       "episode_index": pa.array([0, 1] * 10)}))
    check(res.episodes["count"] == 2 and abs(res.frame_rate_hz - 10) < 0.01
          and "non_monotonic_time" not in findings(prof),
          "rows of two episodes interleaved: grouped by episode, 10 Hz, nothing flagged")
    res, prof = inspect(write("resets.parquet", {"timestamp": pa.array(np.tile(np.arange(50) * 0.02, 3))}))
    check(res.episodes["count"] == 3 and res.episodes["source"] == "timestamp resets"
          and abs(res.frame_rate_hz - 50) < 0.01, "no episode column: time resets split episodes")

    # NaN scan over nested lists with nulls at each level
    nested = pa.array([[[1.0, float("nan")]], [[float("inf")]], None, [[None]], [None]],
                      type=pa.list_(pa.list_(pa.float64())))
    res, _ = inspect(write("nested.parquet", {"a": nested}), nan_check=True)
    check(res.columns[0].nan_count == 2, "NaN/inf counted inside list<list<float>> with nulls")
    big = write("nan_sampled.parquet", {"x": pa.array(rng.random(200_000))}, row_group_size=10_000)
    _, prof = inspect(big, nan_check=True, nan_budget_bytes=200_000)
    check("nan_check_sampled" in findings(prof) and "nan_check_skipped" not in findings(prof),
          "NaN budget smaller than the float data: scan sampled and labelled")

    # Codecs and dictionary pages: the column range must include the dictionary page
    p = write("zstd_dict.parquet", {"timestamp": pa.array(np.arange(1000) * 0.01),
                                    "episode_index": pa.array(np.arange(1000) // 100)},
              compression="zstd", use_dictionary=True, row_group_size=250)
    res, _ = inspect(p)
    check(res.columns[0].compression == "ZSTD" and res.episodes["count"] == 10 and abs(res.frame_rate_hz - 100) < 0.01,
          "zstd + dictionary-encoded episode column decodes from range reads")

    # Damaged files
    g = TMP / "garbage_footer.parquet"
    g.write_bytes(b"PAR1" + np.random.default_rng(2).bytes(200) + (50).to_bytes(4, "little") + b"PAR1")
    res, prof = inspect(g)
    check(res.error and "footer" in res.error and "parse_error" in findings(prof),
          "valid magic, garbage footer: error, not a crash")
    e = TMP / "encrypted.parquet"
    e.write_bytes(b"PAR1" + b"\0" * 20 + b"PARE")
    res, prof = inspect(e)
    check(res.error and "encrypted" in res.error and "encrypted" in findings(prof), "encrypted footer is named")
    m = TMP / "magic_only.parquet"
    m.write_bytes(b"PAR1")
    res, _ = inspect(m)
    check(res.error is not None, "4-byte file: error, not a crash")
    good = lerobot_like().read_bytes()
    cut = TMP / "cut_tail.parquet"
    cut.write_bytes(good[:-100])
    res, prof = inspect(cut)
    check(not res.is_parquet and "bad_magic" in findings(prof), "tail cut off: rejected by the missing end magic")


def main(hub: bool) -> None:
    # 1. LeRobot-style episodes, metadata + timing only
    p = lerobot_like()
    rf = LocalRangeFile(p)
    res, prof = inspect_parquet(rf)
    check(res.is_parquet and res.num_rows == 95 and res.num_row_groups == 5, "footer: 95 rows in 5 row groups")
    check(res.timestamp_column == "timestamp" and res.timestamp_unit == "s" and res.episode_column == "episode_index",
          "detects LeRobot timestamp (s) and episode_index columns")
    check(res.episodes["count"] == 3 and res.episodes["frames_min"] == 5 and res.episodes["frames_max"] == 50,
          f"episode counts: {res.episodes}")
    check(abs(res.episodes["seconds_max"] - 4.9) < 1e-3, "episode durations from timestamps (ep0 = 4.9 s)")
    check({"dropped_frames", "short_episode", "nan_check_skipped"} <= findings(prof),
          f"findings without NaN scan: {sorted(findings(prof))}")
    state = next(c for c in res.columns if c.name == "observation.state")
    check(state.list_size == 3 and state.nan_count is None, "fixed-size list column: size 3, NaN not checked")
    ts = next(c for c in res.columns if c.name == "timestamp")
    check(ts.null_count == 0 and abs(float(ts.max) - 4.9) < 1e-3, "column statistics from footer (nulls, max)")

    # 2. same file, NaN scan on
    res, prof = inspect_parquet(LocalRangeFile(p), nan_check=True)
    state = next(c for c in res.columns if c.name == "observation.state")
    check(state.nan_count == 2 and "nan_values" in findings(prof), "NaN/inf scan finds the 2 bad values")

    # 3. budget forces row-group sampling
    res, prof = inspect_parquet(LocalRangeFile(p), column_budget_bytes=1)
    check(res.row_groups_read == 1 and "sampled_row_groups" in findings(prof) and res.episodes["complete"] is False,
          "tiny budget samples 1 row group and says so")

    # 4. byte cost on a small file: everything fits in one cache block, so nothing is read twice
    rf = LocalRangeFile(p)
    inspect_parquet(rf)
    check(rf.bytes_fetched <= p.stat().st_size, f"small file read once: {rf.bytes_fetched} of {p.stat().st_size} B")

    # 5. byte cost on a larger file: timing must skip the payload columns
    big = TMP / "big.parquet"
    n = 200_000
    rng = np.random.default_rng(1)
    pq.write_table(pa.table({
        "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(rng.random(n * 14, dtype=np.float32)), 14),
        "timestamp": pa.array(np.arange(n) / 30.0, type=pa.float32()),
        "episode_index": pa.array(np.arange(n) // 300, type=pa.int64()),
    }), big, row_group_size=20_000)
    rf = LocalRangeFile(big)
    res, _ = inspect_parquet(rf)
    payload = next(c.compressed_bytes for c in res.columns if c.name == "observation.state")
    check(rf.bytes_fetched < 0.25 * big.stat().st_size and res.episodes["count"] == -(-n // 300),
          f"large file: read {rf.bytes_fetched // 1024} KB of {big.stat().st_size // 1024} KB, skipping "
          f"{payload // 1024} KB of state; {res.episodes['count']} episodes at {res.frame_rate_hz:.1f} Hz")

    # 6. epoch-ns integers, no episodes
    res, prof = inspect_parquet(LocalRangeFile(epoch_ns_log()))
    check(res.timestamp_column == "log_time" and res.timestamp_unit == "ns", "int64 epoch-ns timestamps detected")
    check(abs(res.frame_rate_hz - 100) < 5 and "dropped_frames" in findings(prof),
          f"100 Hz stream with a dropout: rate {res.frame_rate_hz:.1f} Hz, findings {sorted(findings(prof))}")

    # 7. Arrow timestamp type
    res, _ = inspect_parquet(LocalRangeFile(timestamp_typed()))
    check(res.timestamp_column == "time" and abs(res.frame_rate_hz - 50) < 0.01, f"timestamp[ms] column: {res.frame_rate_hz:.2f} Hz")

    # 8. no time column, nested struct column
    res, prof = inspect_parquet(LocalRangeFile(text_only()))
    check(res.is_parquet and res.timestamp_column is None and prof.duration_s is None and len(res.columns) == 3,
          "text table: schema only, no timing, struct column counted once")

    # 9. not Parquet
    bad = TMP / "bad.parquet"
    bad.write_bytes(b"PAR1" + b"\x00" * 100)
    res, _ = inspect_parquet(LocalRangeFile(bad))
    check(not res.is_parquet and "truncated" in (res.error or ""), f"truncated file rejected: {res.error}")

    edge_cases()

    if not hub:
        return
    from phys_ai_dataprofiler.remote.range_file import HfRangeFile

    rf = HfRangeFile("lerobot/pusht", "data/chunk-000/file-000.parquet")
    res, prof = inspect_parquet(rf)
    check(res.episodes["count"] == 206 and abs(res.frame_rate_hz - 10) < 0.01,
          f"Hub LeRobot pusht: {res.episodes['count']} episodes at {res.frame_rate_hz:.1f} Hz (info.json says 206 @ 10 fps)")

    rf = HfRangeFile("nvidia/Nemotron-Personas-Vietnam", "data/train-00000-of-00011.parquet")
    res, prof = inspect_parquet(rf)
    check(res.num_rows and len(res.columns) == 21 and rf.bytes_fetched < 0.01 * rf.size,
          f"Hub text dataset: {res.num_rows:,} rows, {len(res.columns)} columns, read {rf.bytes_fetched // 1024} KB "
          f"of {rf.size // 1024 // 1024} MB")


if __name__ == "__main__":
    main(hub="--hub" in sys.argv)
