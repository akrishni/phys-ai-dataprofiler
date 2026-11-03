"""
Seekable, byte-range backed file objects.

`HfRangeFile` exposes a file on the Hugging Face Hub as a seekable binary file
without downloading it: every read becomes an HTTP Range request against the
resolved (CDN) URL. A small block cache absorbs the many tiny reads that
parsers such as the mcap StreamReader issue, and `fetch_many()` pulls a list of
scattered ranges in parallel for things like MCAP message indexes.

`LocalRangeFile` offers the same interface over a local file so the remote
inspectors can be tested and used offline.
"""

from __future__ import annotations

import io
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

try:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    from huggingface_hub.utils._http import get_session
    HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HF_AVAILABLE = False


DEFAULT_BLOCK_SIZE = 64 * 1024
DEFAULT_MAX_BLOCKS = 128          # 8 MB of cache at the default block size
DEFAULT_WORKERS = 16


class RangeFile:
    """Base class: seekable file API on top of `_fetch(start, end)`."""

    size: int

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE, max_blocks: int = DEFAULT_MAX_BLOCKS) -> None:
        self._pos = 0
        self._block_size = block_size
        self._max_blocks = max_blocks
        self._blocks: "OrderedDict[int, bytes]" = OrderedDict()
        self._lock = threading.Lock()
        self.requests = 0
        self.bytes_fetched = 0
        self.fetch_time_s = 0.0
        self._closed = False

    # -- to be provided by subclasses -------------------------------------

    def _fetch(self, start: int, end: int) -> bytes:  # end exclusive
        raise NotImplementedError

    # -- raw access ---------------------------------------------------------

    def read_range(self, start: int, end: int) -> bytes:
        """Return bytes [start, end) going through the block cache."""
        start = max(0, start)
        end = min(end, self.size)
        if end <= start:
            return b""
        # Large reads bypass the cache, unless it already holds them (e.g. after prefetch).
        if end - start > 2 * self._block_size:
            return self._get(start, end)
        first = start // self._block_size
        last = (end - 1) // self._block_size
        out = bytearray()
        for b in range(first, last + 1):
            blk = self._block(b)
            b_start = b * self._block_size
            lo = max(start, b_start) - b_start
            hi = min(end, b_start + self._block_size) - b_start
            out += blk[lo:hi]
        return bytes(out)

    def prefetch(self, start: int, end: int) -> None:
        """Pull [start, end) in one request and seed the block cache with it."""
        start = max(0, (start // self._block_size) * self._block_size)
        end = min(end, self.size)
        # Skip blocks at either end that are already cached.
        with self._lock:
            while start < end and start // self._block_size in self._blocks:
                start += self._block_size
            while end > start and (end - 1) // self._block_size in self._blocks:
                end = ((end - 1) // self._block_size) * self._block_size
        if end <= start:
            return
        data = self._fetch_counted(start, end)
        with self._lock:
            for off in range(0, len(data), self._block_size):
                b = (start + off) // self._block_size
                blk = data[off: off + self._block_size]
                # Only whole blocks (or the final block of the file) are cacheable.
                if len(blk) == self._block_size or start + off + len(blk) >= self.size:
                    self._blocks[b] = blk
                    self._blocks.move_to_end(b)
            self._evict()

    def cached_range(self, start: int, end: int) -> bytes | None:
        """Return bytes [start, end) if every block they span is already cached, else None."""
        start = max(0, start)
        end = min(end, self.size)
        if end <= start:
            return b""
        first, last = start // self._block_size, (end - 1) // self._block_size
        with self._lock:
            blocks = [self._blocks.get(b) for b in range(first, last + 1)]
        if any(b is None for b in blocks):
            return None
        data = b"".join(blocks)
        off = start - first * self._block_size
        return data[off: off + (end - start)]

    def fetch_many(self, ranges: Sequence[tuple[int, int]], workers: int = DEFAULT_WORKERS) -> list[bytes]:
        """Fetch many [start, end) ranges concurrently; preserves order. Fully cached ranges cost nothing."""
        if not ranges:
            return []
        if len(ranges) == 1:
            return [self._get(*ranges[0])]
        with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as ex:
            return list(ex.map(lambda r: self._get(r[0], r[1]), ranges))

    @property
    def cache_bytes(self) -> int:
        """How many bytes the block cache can hold."""
        return self._block_size * self._max_blocks

    # -- file-like API --------------------------------------------------------

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._closed

    def flush(self) -> None:
        return None

    def write(self, data) -> int:
        raise io.UnsupportedOperation("read-only")

    def truncate(self, size=None) -> int:
        raise io.UnsupportedOperation("read-only")

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new < 0:
            raise ValueError("negative seek position")
        self._pos = new
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if self._pos >= self.size:
            return b""
        end = self.size if n is None or n < 0 else min(self.size, self._pos + n)
        data = self.read_range(self._pos, end)
        self._pos += len(data)
        return data

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self._blocks.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals --------------------------------------------------------------

    def _get(self, start: int, end: int) -> bytes:
        """Bytes [start, end) from the cache when every block is there, else one uncached request."""
        cached = self.cached_range(start, end)
        return cached if cached is not None else self._fetch_counted(start, end)

    def _fetch_counted(self, start: int, end: int) -> bytes:
        t0 = time.time()
        data = self._fetch(start, end)
        with self._lock:
            self.requests += 1
            self.bytes_fetched += len(data)
            self.fetch_time_s += time.time() - t0
        return data

    def _block(self, index: int) -> bytes:
        with self._lock:
            blk = self._blocks.get(index)
            if blk is not None:
                self._blocks.move_to_end(index)
                return blk
        start = index * self._block_size
        blk = self._fetch_counted(start, min(start + self._block_size, self.size))
        with self._lock:
            self._blocks[index] = blk
            self._blocks.move_to_end(index)
            self._evict()
        return blk

    def _evict(self) -> None:
        while len(self._blocks) > self._max_blocks:
            self._blocks.popitem(last=False)


class LocalRangeFile(RangeFile):
    """Same interface over a local file (for tests and offline use)."""

    def __init__(self, path: str | Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self._fh = self.path.open("rb")
        self._fh_lock = threading.Lock()

    def _fetch(self, start: int, end: int) -> bytes:
        with self._fh_lock:
            self._fh.seek(start)
            return self._fh.read(end - start)

    def close(self) -> None:
        super().close()
        try:
            self._fh.close()
        except Exception:
            pass


class HfRangeFile(RangeFile):
    """A Hugging Face Hub file read through HTTP Range requests.

    The repo path is resolved once to its final download URL (usually a CDN
    URL for LFS/Xet files). Reads go straight to that URL so each request is a
    single round trip. Auth headers are only attached when the final URL is on
    the Hub itself, so the token is never sent to a third-party CDN.
    """

    def __init__(
        self,
        repo_id: str,
        filename: str,
        token: str | None = None,
        repo_type: str = "dataset",
        revision: str | None = None,
        timeout: float = 60.0,
        retries: int = 3,
        **kwargs,
    ) -> None:
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub is not installed")
        super().__init__(**kwargs)
        self.repo_id = repo_id
        self.filename = filename
        self.token = token
        self.repo_type = repo_type
        self.revision = revision
        self.timeout = timeout
        self.retries = retries
        self._session = get_session()
        self._resolve_url = hf_hub_url(repo_id, filename, repo_type=repo_type, revision=revision)
        self._location: str | None = None
        self._headers: dict = {}
        self.size = 0
        self._resolve()

    @property
    def url(self) -> str:
        return f"hf://{self.repo_type}s/{self.repo_id}/{self.filename}"

    def _resolve(self) -> None:
        md = get_hf_file_metadata(self._resolve_url, token=self.token, timeout=self.timeout)
        self._location = md.location or self._resolve_url
        self.size = int(md.size or 0)
        same_host = self._location.split("/")[2] == self._resolve_url.split("/")[2]
        self._headers = build_hf_headers(token=self.token) if same_host else {}

    def _fetch(self, start: int, end: int) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self._session.get(
                    self._location,
                    headers={**self._headers, "Range": f"bytes={start}-{end - 1}"},
                    follow_redirects=True,
                    timeout=self.timeout,
                )
                if r.status_code in (401, 403) and attempt == 0:
                    # Signed CDN URLs expire; re-resolve once and retry.
                    self._resolve()
                    continue
                if r.status_code == 206:
                    return r.content
                if r.status_code == 200:
                    # Server ignored Range; slice the body (small files only).
                    return r.content[start:end]
                raise RuntimeError(f"HTTP {r.status_code} fetching bytes {start}-{end - 1}")
            except Exception as exc:  # network hiccup; back off and retry
                last_exc = exc
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"range fetch failed after {self.retries + 1} attempts: {last_exc}")
