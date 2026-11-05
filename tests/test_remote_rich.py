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

    mcap_edge_cases()
    mat_edge_cases()


def tiny_mcap(name: str, messages: list[tuple[int, int]], topics=("/a",)) -> Path:
    """MCAP with the given (channel index, log_time) messages, written in that order."""
    path = TMP / f"{name}.mcap"
    with path.open("wb") as f:
        w = Writer(f)
        w.start("test", "test")
        s = w.register_schema("S", "jsonschema", b"{}")
        ids = [w.register_channel(t, "json", s) for t in topics]
        for ci, t in messages:
            w.add_message(ids[ci], log_time=t, publish_time=t, data=b"{}")
        w.finish()
    return path


def mcap_edge_cases() -> None:
    ms = 10_000_000
    cases = [
        ("no messages", tiny_mcap("empty", []), {}),
        ("one message", tiny_mcap("one", [(0, 5)]), {"/a": (1, 0, 0, 0)}),
        # /b is registered but never written to: it must not appear as a 0-message stream with a rate
        ("silent channel", tiny_mcap("silent", [(0, k * ms) for k in range(100)], ("/a", "/b")),
         {"/a": (100, 100.0, 0, 10.0)}),
        # written out of time order: intervals come from sorted timestamps, never negative
        ("out of order", tiny_mcap("ooo", [(0, k * ms) for k in (3, 1, 2, 0, 4)]), {"/a": (5, 100.0, 0, 10.0)}),
    ]
    for label, path, want in cases:
        res, prof, _ = remote(path)
        full = stats(RichInspector(str(path)).inspect())
        check(res.error is None and stats(prof) == want == full, f"MCAP {label}: {stats(prof)} equals full read")
    _, prof, _ = remote(cases[0][1])
    check(prof.duration_s is None and not prof.channels, "MCAP with no messages: no duration, no streams")

    # Starts more than 1 s apart are reported; within a second they are not.
    late = tiny_mcap("late", [(0, k * ms) for k in range(10)] + [(1, 3_000_000_000 + k * ms) for k in range(10)],
                     ("/a", "/b"))
    _, prof, _ = remote(late)
    check("start_offset" in findings(prof), "a stream starting 3 s after the first: start_offset")
    check("start_offset" not in findings(remote(cases[2][1])[1]), "streams starting together: no start_offset")

    junk = TMP / "junk.mcap"
    junk.write_bytes(np.random.default_rng(3).bytes(500))
    res, prof, _ = remote(junk)
    check(not res.is_mcap and "bad_magic" in findings(prof), "random bytes named .mcap: rejected, not parsed")


def mat_edge_cases() -> None:
    import scipy.sparse

    path = TMP / "types.mat"
    scipy.io.savemat(path, {
        "st": {"a": np.array([np.nan]), "b": 2},          # NaN inside a struct: not scanned
        "cl": np.array([1, "x"], dtype=object),
        "sp": scipy.sparse.csr_matrix(np.array([[np.nan, 0], [0, 1.0]])),
        "cx": np.array([1 + np.nan * 1j, 2]),
        "empty": np.zeros((0, 3)),
        "txt": "hello",
        "allnan": np.full(10, np.nan),
        "flags": np.array([True, False]),
    })
    # The full-read inspector used to raise on the sparse matrix (np.isnan on a scipy sparse object).
    local = RichInspector(str(path)).inspect()
    local_nan = {c.topic: c.nan_count for c in local.channels}
    check(local_nan["sp"] == 1 and local_nan["cx"] == 1 and local_nan["allnan"] == 10,
          f"full read handles sparse, complex and all-NaN variables: {local_nan}")
    res, prof = inspect_mat_remote(LocalRangeFile(path))
    got = {c.topic: (c.schema, c.nan_count) for c in prof.channels}
    check(res.error is None and len(got) == 8, f"remote MAT lists every variable type: {sorted(got)}")
    check(got["st"][0] == "struct" and got["cl"][0] == "cell" and got["sp"][0] == "sparse",
          "struct, cell and sparse are named as such")
    check(got["cx"][1] == 1 and got["allnan"][1] == 10 and got["empty"][1] == 0,
          "complex NaN counted, all-NaN counted, empty array is 0")

    raw = path.read_bytes()
    cut = TMP / "types_cut.mat"
    cut.write_bytes(raw[:-20])
    res, prof = inspect_mat_remote(LocalRangeFile(cut))
    check(res.error is None and "truncated" in findings(prof) and len(res.variable_names) == 7,
          f"truncated MAT: earlier variables kept, last one reported truncated ({res.variable_names})")

    empty = TMP / "no_vars.mat"
    scipy.io.savemat(empty, {})
    res, prof = inspect_mat_remote(LocalRangeFile(empty))
    check(res.is_matfile and res.error is None and not prof.channels and not prof.findings,
          "MAT with no variables: valid, empty")

    v4 = TMP / "v4.mat"
    scipy.io.savemat(v4, {"a": np.ones(3)}, format="4")
    res, prof = inspect_mat_remote(LocalRangeFile(v4))
    check(res.error and "unsupported" in findings(prof), "MAT v4 is reported unsupported, not misparsed")


if __name__ == "__main__":
    main()
