#!/usr/bin/env python3
"""Checks for the block-cached RangeFile that every remote inspector reads through.

    python tests/test_range_file.py

Every read is compared with the same slice of the bytes on disk, across block
boundaries, past the end of the file, after prefetch and under cache eviction.
Exits non-zero on the first failed check.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.remote.range_file import LocalRangeFile  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="range_file_test_"))
DATA = os.urandom(10_000)          # 9 full 1 KB blocks and a partial last block
BLOCK = 1024


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def rf(**kw) -> LocalRangeFile:
    path = TMP / "data.bin"
    if not path.exists():
        path.write_bytes(DATA)
    return LocalRangeFile(path, block_size=kw.pop("block_size", BLOCK), **kw)


def test_read_range() -> None:
    f = rf()
    cases = [(0, 1), (BLOCK - 1, BLOCK + 1), (500, 3 * BLOCK + 7), (9_990, 20_000), (-5, 3), (0, len(DATA))]
    check(all(f.read_range(s, e) == DATA[max(0, s):min(e, len(DATA))] for s, e in cases),
          "read_range matches the file across block edges, past EOF and from a negative start")
    check(f.read_range(500, 500) == b"" and f.read_range(600, 500) == b"" and f.read_range(20_000, 30_000) == b"",
          "empty, reversed and beyond-EOF ranges give no bytes")

    f = rf()
    f.read_range(10, 20)
    f.read_range(30, 40)
    check(f.requests == 1 and f.bytes_fetched == BLOCK, "two reads inside one block cost one request")
    f.read_range(BLOCK - 2, BLOCK + 2)
    check(f.requests == 2, "a read straddling a block edge fetches only the block not yet cached")

    f = rf()
    f.read_range(0, 5 * BLOCK)
    check(f.requests == 1 and f.bytes_fetched == 5 * BLOCK and f.cached_range(0, 10) is None,
          "reads over 2 blocks bypass the cache in one request")

    f = rf()
    f.read_range(9 * BLOCK, len(DATA))
    check(f.cached_range(9 * BLOCK, len(DATA)) == DATA[9 * BLOCK:], "the short final block is cached")


def test_cache_eviction() -> None:
    f = rf(max_blocks=3)
    for b in range(6):
        f.read_range(b * BLOCK, b * BLOCK + 1)
    check(len(f._blocks) == 3 and f.cache_bytes == 3 * BLOCK, "cache holds at most max_blocks")
    check(f.cached_range(0, 1) is None and f.cached_range(5 * BLOCK, 5 * BLOCK + 1) is not None,
          "oldest blocks are evicted first")
    f.read_range(3 * BLOCK, 3 * BLOCK + 1)             # touch block 3, so block 4 is now the oldest
    f.read_range(0, 1)
    check(f.cached_range(3 * BLOCK, 3 * BLOCK + 1) is not None and f.cached_range(4 * BLOCK, 4 * BLOCK + 1) is None,
          "a cache hit refreshes the block (LRU, not FIFO)")
    check(f.read_range(0, len(DATA)) == DATA, "reads stay correct after eviction")


def test_prefetch() -> None:
    f = rf()
    f.prefetch(100, 5_000)                            # rounds down to block 0, ends mid block 4
    n = f.requests
    check(f.read_range(0, 4 * BLOCK) == DATA[:4 * BLOCK] and f.requests == n,
          "prefetched whole blocks serve reads with no new request")
    check(f.cached_range(4 * BLOCK, 4 * BLOCK + 1) is None, "a partial block at the end of a prefetch is not cached")
    f.prefetch(BLOCK, 3 * BLOCK)
    check(f.requests == n, "prefetch of already cached blocks sends nothing")
    before = f.bytes_fetched
    f.prefetch(0, 6 * BLOCK)                          # blocks 0-3 cached: only blocks 4-5 are fetched
    check(f.requests == n + 1 and f.bytes_fetched - before == 2 * BLOCK,
          "prefetch trims cached blocks at the front and fetches the rest in one request")

    f = rf()
    f.prefetch(0, len(DATA))
    check(f.cached_range(0, len(DATA)) == DATA, "prefetch of the whole file caches the short final block too")
    out = f.fetch_many([(0, 10), (5_000, 5_010), (9_999, 10_000)])
    check(out == [DATA[0:10], DATA[5_000:5_010], DATA[9_999:]] and f.requests == 1,
          "fetch_many serves fully cached ranges for free")


def test_fetch_many() -> None:
    f = rf()
    ranges = [(i * 97, i * 97 + 50) for i in range(60)]
    check(f.fetch_many(ranges) == [DATA[s:e] for s, e in ranges], "fetch_many keeps order with many workers")
    check(f.fetch_many([]) == [] and f.fetch_many([(5, 9)]) == [DATA[5:9]], "fetch_many with none or one range")
    check(f.requests == 61 and f.bytes_fetched == 60 * 50 + 4, "fetch_many counts every request and byte")


def test_file_api() -> None:
    f = rf()
    check(f.seekable() and f.readable() and not f.writable(), "read-only seekable file")
    f.seek(-10, io.SEEK_END)
    check(f.read() == DATA[-10:] and f.read() == b"" and f.tell() == len(DATA), "SEEK_END then read to EOF")
    f.seek(100)
    f.seek(20, io.SEEK_CUR)
    check(f.read(5) == DATA[120:125] and f.tell() == 125, "SEEK_CUR and short read advance the position")
    f.seek(50_000)
    check(f.read(5) == b"" and f.tell() == 50_000, "seeking past EOF is allowed and reads nothing")
    f.seek(3)
    check(f.read(None) == DATA[3:] and (f.seek(3) or True) and f.read(-1) == DATA[3:],
          "read(None) and read(-1) read to the end")
    buf = bytearray(16)
    f.seek(len(DATA) - 4)
    check(f.readinto(buf) == 4 and bytes(buf[:4]) == DATA[-4:], "readinto near EOF returns the short count")
    for bad, what in (((-1,), "negative position"), ((0, 7), "invalid whence")):
        try:
            f.seek(*bad)
            check(False, f"seek with {what} raises")
        except ValueError:
            check(True, f"seek with {what} raises ValueError")
    for op in (lambda: f.write(b"x"), lambda: f.truncate()):
        try:
            op()
            check(False, "write/truncate raise")
        except io.UnsupportedOperation:
            pass
    check(True, "write and truncate raise UnsupportedOperation")
    with rf() as g:
        g.read_range(0, 10)
    check(g.closed and not g._blocks, "context manager closes the file and drops the cache")


def test_empty_file() -> None:
    p = TMP / "empty.bin"
    p.write_bytes(b"")
    f = LocalRangeFile(p)
    check(f.size == 0 and f.read() == b"" and f.read_range(0, 10) == b"" and f.fetch_many([]) == [],
          "empty file: every read is empty, nothing raises")
    f.prefetch(0, 100)
    check(f.requests == 0, "empty file: prefetch sends nothing")


def main() -> None:
    test_read_range()
    test_cache_eviction()
    test_prefetch()
    test_fetch_many()
    test_file_api()
    test_empty_file()
    print("all checks passed")


if __name__ == "__main__":
    main()
