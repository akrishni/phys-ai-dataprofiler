"""
Metadata-first Parquet inspector over a RangeFile (local or remote).

What it reads, in order:

1. The footer. Its length is in the last 8 bytes; it holds the schema, row
   groups, and per-column statistics (row and null counts, min/max, sizes,
   codecs). Typically tens of KB, whatever the file size.
2. Optionally, the few small columns needed for timing: a timestamp column and
   an episode index column (the LeRobot convention). Their byte ranges are known
   from the footer, so the cost is computed before reading and capped by a
   budget. Over budget, evenly spaced row groups are sampled. Ranges are fetched
   in parallel, then decoded by pyarrow from memory.
3. Optionally (nan_check, on by default in the CLI), float columns, to count
   NaN/inf values, which Parquet statistics do not record. Capped by
   nan_budget_bytes; over budget, row groups are sampled.

Large payload columns (images, text, embeddings) are never read.
"""

from __future__ import annotations

import bisect
import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from phys_ai_dataprofiler.inspectors.rich import ChannelProfile, Finding, RichProfile
from phys_ai_dataprofiler.remote.range_file import RangeFile

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYARROW_AVAILABLE = False

PARQUET_MAGIC = b"PAR1"
PARQUET_ENCRYPTED_MAGIC = b"PARE"
TIMESTAMP_CANDIDATES = ("timestamp", "log_time", "timestamps", "time", "stamp", "ts", "t")
EPISODE_CANDIDATES = ("episode_index", "episode_id", "episode")
DEFAULT_COLUMN_BUDGET = 16 * 1024 * 1024
DEFAULT_NAN_BUDGET = 256 * 1024 * 1024
_MERGE_GAP = 4096          # coalesce column-chunk ranges closer than this
_DISPLAY_MAX = 60


@dataclass
class ParquetColumnInfo:
    name: str
    type: str
    leaves: int
    compression: str | None
    compressed_bytes: int
    uncompressed_bytes: int
    null_count: int | None
    min: Any = None
    max: Any = None
    list_size: int | None = None
    nan_count: int | None = None


@dataclass
class ParquetInspectionResult:
    url: str
    is_parquet: bool
    size_bytes: int | None
    error: str | None = None
    num_rows: int | None = None
    num_row_groups: int | None = None
    created_by: str | None = None
    format_version: str | None = None
    columns: list[ParquetColumnInfo] | None = None
    metadata_keys: list[str] | None = None
    timestamp_column: str | None = None
    timestamp_unit: str | None = None
    episode_column: str | None = None
    frame_rate_hz: float | None = None
    episodes: dict | None = None
    row_groups_read: int | None = None


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #

def _is_list(t) -> bool:
    return (pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
            or (hasattr(pa.types, "is_list_view") and pa.types.is_list_view(t))
            or (hasattr(pa.types, "is_large_list_view") and pa.types.is_large_list_view(t)))


def _is_flat(t) -> bool:
    return not (pa.types.is_struct(t) or _is_list(t) or pa.types.is_map(t))


def _leaf_count(t) -> int:
    if pa.types.is_struct(t):
        return sum(_leaf_count(t.field(i).type) for i in range(t.num_fields)) or 1
    if pa.types.is_map(t):
        return _leaf_count(t.key_type) + _leaf_count(t.item_type)
    if _is_list(t):
        return _leaf_count(t.value_type)
    return 1


def _is_float_like(t) -> bool:
    if pa.types.is_floating(t):
        return True
    if _is_list(t):
        return _is_float_like(t.value_type)
    return False


def _display(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    if isinstance(v, str):
        return v if len(v) <= _DISPLAY_MAX else v[: _DISPLAY_MAX - 3] + "..."
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, float):
        return float(f"{v:.6g}")
    return v


def _type_str(t) -> str:
    s = str(t)
    return s if len(s) <= _DISPLAY_MAX else s[: _DISPLAY_MAX - 3] + "..."


def _modality(name: str, t) -> str:
    n = name.lower()
    if any(k in n for k in ("image", "img", "rgb", "depth", "camera", "video")):
        return "image"
    if n in TIMESTAMP_CANDIDATES or "timestamp" in n:
        return "timestamps"
    if n.endswith("index") or n.endswith("_id") or n == "id":
        return "index"
    if "action" in n:
        return "action"
    if any(k in n for k in ("state", "joint", "qpos", "qvel", "pose", "position", "effort", "gripper")):
        return "proprio"
    if any(k in n for k in ("imu", "accel", "gyro")):
        return "imu"
    if "reward" in n:
        return "reward"
    if any(k in n for k in ("done", "success", "terminal", "truncated")):
        return "flag"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "text"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return "binary"
    return "data"


def _leaf_map(pf, md) -> list[tuple[str, Any, list[int]]]:
    """[(top-level column name, arrow type or None, leaf column indices)]."""
    fields = list(pf.schema_arrow)
    counts = [_leaf_count(f.type) for f in fields]
    out = []
    if sum(counts) == md.num_columns:
        i = 0
        for f, n in zip(fields, counts):
            out.append((f.name, f.type, list(range(i, i + n))))
            i += n
    else:  # schema we could not map; fall back to one entry per leaf column
        for i in range(md.num_columns):
            out.append((md.schema.column(i).path, None, [i]))
    return out


def _column_infos(md, leaf_map) -> list[ParquetColumnInfo]:
    infos = []
    for name, t, leaves in leaf_map:
        comp = unc = 0
        codecs: set[str] = set()
        flat = t is not None and _is_flat(t) and len(leaves) == 1
        nulls: int | None = 0 if flat else None
        mn = mx = None
        minmax_ok = flat
        for rg in range(md.num_row_groups):
            g = md.row_group(rg)
            for li in leaves:
                c = g.column(li)
                comp += c.total_compressed_size or 0
                unc += c.total_uncompressed_size or 0
                codecs.add(str(c.compression))
                st = c.statistics
                if nulls is not None:
                    if st is not None and st.has_null_count:
                        nulls += st.null_count
                    else:
                        nulls = None
                if minmax_ok:
                    if st is not None and st.has_min_max:
                        try:
                            mn = st.min if mn is None else min(mn, st.min)
                            mx = st.max if mx is None else max(mx, st.max)
                        except Exception:
                            minmax_ok = False
                    else:
                        minmax_ok = False
        list_size = t.list_size if t is not None and pa.types.is_fixed_size_list(t) else None
        infos.append(ParquetColumnInfo(
            name=name,
            type=_type_str(t) if t is not None else "leaf",
            leaves=len(leaves),
            compression="/".join(sorted(codecs)) if codecs else None,
            compressed_bytes=comp,
            uncompressed_bytes=unc,
            null_count=nulls,
            min=_display(mn) if minmax_ok else None,
            max=_display(mx) if minmax_ok else None,
            list_size=list_size,
        ))
    return infos


# --------------------------------------------------------------------------- #
# Reading selected columns cheaply
# --------------------------------------------------------------------------- #

class _SparseFile(io.RawIOBase):
    """File-like view over a RangeFile that serves reads from prefetched ranges."""

    def __init__(self, rf: RangeFile, chunks: list[tuple[int, int, bytes]]) -> None:
        super().__init__()
        self.rf = rf
        self.size = rf.size
        self._pos = 0
        self._chunks = sorted(chunks, key=lambda c: c[0])
        self._starts = [c[0] for c in self._chunks]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if self._pos >= self.size:
            return b""
        end = self.size if n is None or n < 0 else min(self.size, self._pos + n)
        data = self._read_range(self._pos, end)
        self._pos += len(data)
        return data

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)

    def _read_range(self, start: int, end: int) -> bytes:
        i = bisect.bisect_right(self._starts, start) - 1
        if i >= 0:
            s, e, data = self._chunks[i]
            if start < e:
                head = data[start - s: min(end, e) - s]
                return head if end <= e else head + self.rf.read_range(e, end)
        return self.rf.read_range(start, end)


def _chunk_range(c) -> tuple[int, int]:
    start = c.data_page_offset
    if c.has_dictionary_page and c.dictionary_page_offset:
        start = min(start, c.dictionary_page_offset)
    return start, start + (c.total_compressed_size or 0)


def _plan_row_groups(md, leaves: list[int], budget: int) -> tuple[list[int], int, bool]:
    """Pick row groups so the bytes for `leaves` stay under budget. Returns (rgs, bytes, sampled)."""
    n = md.num_row_groups
    per_rg = [sum(md.row_group(r).column(li).total_compressed_size or 0 for li in leaves) for r in range(n)]
    total = sum(per_rg)
    if total <= budget or n <= 1:
        return list(range(n)), total, False
    keep = max(1, int(n * budget / total))
    step = n / keep
    rgs = sorted({int(i * step) for i in range(keep)})
    return rgs, sum(per_rg[r] for r in rgs), True


def _read_columns(rf: RangeFile, md, leaves: list[int], rgs: list[int], workers: int):
    """Fetch the column chunks for (rgs x leaves) in parallel and decode them."""
    ranges = sorted(_chunk_range(md.row_group(r).column(li)) for r in rgs for li in leaves)
    merged: list[list[int]] = []
    for s, e in ranges:
        if merged and s <= merged[-1][1] + _MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    datas = rf.fetch_many([(s, e) for s, e in merged], workers=workers)
    sparse = _SparseFile(rf, [(s, e, d) for (s, e), d in zip(merged, datas)])
    pf = pq.ParquetFile(sparse, metadata=md)
    try:
        return pf.reader.read_row_groups(rgs, column_indices=leaves, use_threads=False)
    except Exception:
        return pf.read_row_groups(rgs, use_threads=False)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

_UNIT_NS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
_EPOCH_S = (946_684_800, 4_102_444_800)   # 2000-01-01 .. 2100-01-01 in Unix seconds


def _to_ns(arr) -> tuple[np.ndarray | None, str | None, np.ndarray | None]:
    """Convert a timestamp column to int64 ns. Returns (values, unit, valid_mask)."""
    arr = arr.combine_chunks() if hasattr(arr, "combine_chunks") else arr
    t = arr.type
    valid = np.asarray(pc.is_valid(arr).to_numpy(zero_copy_only=False), dtype=bool)
    if pa.types.is_timestamp(t) or pa.types.is_duration(t):
        v = np.asarray(arr.cast(pa.int64()).to_numpy(zero_copy_only=False))
        v = np.where(valid, v, 0).astype(np.int64)
        return v * _UNIT_NS[t.unit], t.unit, valid
    if pa.types.is_integer(t) or pa.types.is_floating(t):
        is_int = pa.types.is_integer(t)
        if is_int:
            # Keep integers exact: epoch-ns values (~1.7e18) do not fit in float64.
            raw = np.asarray(arr.cast(pa.int64(), safe=False).to_numpy(zero_copy_only=False))
            v = np.where(valid, raw, 0).astype(np.int64)
            vv = v[valid]
        else:
            fv = np.asarray(arr.cast(pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
            valid &= np.isfinite(fv)
            vv = fv[valid]
        if not valid.any():
            return None, None, None
        lo, m = float(np.min(vv)), float(np.max(np.abs(vv)))
        d = np.diff(vv)
        step = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        # Epoch timestamps (years 2000-2100, steps under an hour) give the unit away by magnitude.
        # Relative integer time does not (ns since start passes 1e9 after one second), so its
        # unit comes from the step size.
        epoch = next((u for u, scale in _UNIT_NS.items()
                      if _EPOCH_S[0] * 1e9 / scale <= lo and m <= _EPOCH_S[1] * 1e9 / scale
                      and step * scale <= 3600e9), None)
        if m >= 1e17:
            unit = "ns"
        elif epoch:
            unit = epoch
        elif not is_int:
            unit = "us" if m >= 1e14 else "ms" if m >= 1e11 else "s"   # float seconds (LeRobot) unless huge
        else:                   # relative integer time; infer unit from step size
            unit = "ns" if step >= 1e6 else "us" if step >= 1e3 else "ms"
        if is_int:
            return v * _UNIT_NS[unit], unit, valid
        ns = np.where(valid, fv, 0.0) * _UNIT_NS[unit]
        return np.round(ns).astype(np.int64), unit, valid
    return None, None, None


def _timing(ts_ns: np.ndarray, ep: np.ndarray | None) -> dict:
    """Interval statistics within segments (episodes, or runs of increasing time)."""
    n = len(ts_ns)
    if ep is not None:
        order = np.lexsort((np.arange(n), ep))
        ts, eps = ts_ns[order], ep[order]
        same = eps[1:] == eps[:-1]
        seg_id = np.concatenate([[0], np.cumsum(~same)])
    else:
        ts = ts_ns
        d_all = np.diff(ts)
        same = d_all > 0
        seg_id = np.concatenate([[0], np.cumsum(~same)])
    d = np.diff(ts)
    within = d[same]
    non_monotonic = int((within <= 0).sum()) if ep is not None else int((~same).sum())
    pos = within[within > 0] / 1e6  # ms
    out: dict[str, Any] = {"segments": int(seg_id[-1]) + 1 if n else 0, "non_monotonic": non_monotonic}
    if len(pos):
        mean = float(pos.mean())
        out.update(rate_hz=1000.0 / mean, jitter_ms=float(pos.std()), max_gap_ms=float(pos.max()),
                   gaps_over_3x=int((pos > 3 * mean).sum()), median_interval_ms=float(np.median(pos)))
    # per-segment lengths
    if n:
        _, first, counts = np.unique(seg_id, return_index=True, return_counts=True)
        last = first + counts - 1
        dur = (ts[last] - ts[first]) / 1e9
        out.update(frames=counts, seconds=dur)
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def inspect_parquet(
    rf: RangeFile,
    column_budget_bytes: int = DEFAULT_COLUMN_BUDGET,
    nan_check: bool = False,
    nan_budget_bytes: int = DEFAULT_NAN_BUDGET,
    workers: int = 16,
    log=None,
) -> tuple[ParquetInspectionResult, RichProfile]:
    log = log or (lambda m: None)
    t0 = time.time()
    url = getattr(rf, "url", str(getattr(rf, "path", "")))
    size = rf.size
    findings: list[Finding] = []

    def done(res: ParquetInspectionResult, channels=None, duration=None, indexed=True) -> tuple:
        prof = RichProfile(path=url, format="parquet", size_bytes=size, indexed=indexed,
                           duration_s=duration, time_base="relative_s" if duration is not None else None,
                           channels=channels or [], findings=findings, inspect_cost_s=time.time() - t0)
        return res, prof

    if not PYARROW_AVAILABLE:
        findings.append(Finding("error", "missing_library", None, "pyarrow is required for Parquet"))
        return done(ParquetInspectionResult(url=url, is_parquet=False, size_bytes=size,
                                            error="pyarrow not installed"), indexed=False)

    head, tail = rf.read_range(0, 4), rf.read_range(max(0, size - 4), size)
    if tail == PARQUET_ENCRYPTED_MAGIC:
        findings.append(Finding("error", "encrypted", None, "encrypted Parquet footer"))
        return done(ParquetInspectionResult(url=url, is_parquet=True, size_bytes=size,
                                            error="encrypted Parquet is not supported"), indexed=False)
    if head != PARQUET_MAGIC or tail != PARQUET_MAGIC:
        findings.append(Finding("error", "bad_magic", None, "missing PAR1 magic at start or end"))
        return done(ParquetInspectionResult(url=url, is_parquet=False, size_bytes=size,
                                            error="Not a valid Parquet file (truncated or not Parquet)"), indexed=False)

    try:
        rf.seek(0)
        pf = pq.ParquetFile(rf)
        md = pf.metadata
    except Exception as exc:
        findings.append(Finding("error", "parse_error", None, str(exc)))
        return done(ParquetInspectionResult(url=url, is_parquet=True, size_bytes=size,
                                            error=f"Failed to read Parquet footer: {exc}"), indexed=False)

    lmap = _leaf_map(pf, md)
    cols = _column_infos(md, lmap)
    by_name = {name: (t, leaves) for name, t, leaves in lmap}
    lower = {name.lower(): name for name, t, leaves in lmap
             if t is not None and _is_flat(t) and len(leaves) == 1}
    res = ParquetInspectionResult(
        url=url, is_parquet=True, size_bytes=size,
        num_rows=md.num_rows, num_row_groups=md.num_row_groups,
        created_by=md.created_by, format_version=str(md.format_version),
        columns=cols,
        metadata_keys=sorted(k.decode("utf-8", "replace") for k in (md.metadata or {}).keys()),
    )

    # ---- timing from a timestamp column (and episode index if present) ---------
    ts_col = next((lower[c] for c in TIMESTAMP_CANDIDATES if c in lower
                   and (pa.types.is_integer(by_name[lower[c]][0]) or pa.types.is_floating(by_name[lower[c]][0])
                        or pa.types.is_timestamp(by_name[lower[c]][0]) or pa.types.is_duration(by_name[lower[c]][0]))), None)
    ep_col = next((lower[c] for c in EPISODE_CANDIDATES if c in lower and pa.types.is_integer(by_name[lower[c]][0])), None)
    float_cols = [c for c in cols if by_name[c.name][0] is not None and _is_float_like(by_name[c.name][0])]

    ts_leaves = by_name[ts_col][1] + (by_name[ep_col][1] if ep_col else []) if ts_col and md.num_rows else []
    nan_leaves = [li for c in float_cols for li in by_name[c.name][1]] if nan_check else []
    ts_plan = _plan_row_groups(md, ts_leaves, column_budget_bytes) if ts_leaves else None
    nan_plan = _plan_row_groups(md, nan_leaves, nan_budget_bytes) if nan_leaves else None

    # Timing and NaN reads overlap (a float timestamp is both). When together they cover most of a
    # small file, one request for the whole file is cheaper than fetching the same columns twice.
    planned = (ts_plan[1] if ts_plan else 0) + (nan_plan[1] if nan_plan else 0)
    if planned > size // 2 and size <= rf.cache_bytes:
        rf.prefetch(0, size)
    timing: dict = {}
    sampled = False
    if ts_plan:
        leaves = ts_leaves
        rgs, nbytes, sampled = ts_plan
        log(f"    reading {ts_col}{' + ' + ep_col if ep_col else ''} from {len(rgs)}/{md.num_row_groups} "
            f"row group(s), {nbytes / 1024:.0f} KB")
        try:
            tbl = _read_columns(rf, md, leaves, rgs, workers)
            ts_ns, unit, valid = _to_ns(tbl.column(ts_col))
            ep = None
            if ep_col and ts_ns is not None:
                ep = np.asarray(tbl.column(ep_col).combine_chunks().fill_null(-1).to_numpy(zero_copy_only=False), dtype=np.int64)
            if ts_ns is not None:
                if valid is not None and not valid.all():
                    findings.append(Finding("warn", "null_timestamps", ts_col, f"{int((~valid).sum())} null or non-finite timestamps"))
                    ts_ns = ts_ns[valid]
                    ep = ep[valid] if ep is not None else None
                timing = _timing(ts_ns, ep)
                res.timestamp_column, res.timestamp_unit, res.episode_column = ts_col, unit, ep_col
                res.row_groups_read = len(rgs)
        except Exception as exc:
            findings.append(Finding("warn", "timing_read_failed", ts_col, f"could not read {ts_col}: {exc}"))

    duration = None
    if timing:
        res.frame_rate_hz = timing.get("rate_hz")
        frames, secs = timing.get("frames"), timing.get("seconds")
        if frames is not None and len(frames):
            duration = float(np.sum(secs))
            res.episodes = {
                "source": ep_col or "timestamp resets",
                "count": int(len(frames)),
                "frames_min": int(frames.min()), "frames_median": float(np.median(frames)), "frames_max": int(frames.max()),
                "seconds_min": round(float(secs.min()), 3), "seconds_median": round(float(np.median(secs)), 3),
                "seconds_max": round(float(secs.max()), 3),
                "complete": not sampled,
            }
            if not sampled and len(frames) >= 3:
                med = float(np.median(secs))
                short = int((secs < 0.25 * med).sum()) if med > 0 else 0
                if short:
                    findings.append(Finding("info", "short_episode", ep_col,
                                            f"{short} of {len(frames)} episodes shorter than 25% of the median ({med:.1f} s)"))
        if timing.get("gaps_over_3x") and (timing.get("max_gap_ms") or 0) > 100:
            findings.append(Finding("warn", "dropped_frames", ts_col,
                                    f"{timing['gaps_over_3x']} gaps > 3x the mean interval; worst {timing['max_gap_ms']:.0f} ms"))
        if ep_col and timing.get("non_monotonic"):
            findings.append(Finding("warn", "non_monotonic_time", ts_col,
                                    f"{timing['non_monotonic']} non-increasing timestamps within episodes"))
        if not ep_col and timing.get("segments", 1) > max(1, md.num_rows // 100):
            findings.append(Finding("warn", "timestamps_unsorted", ts_col,
                                    f"timestamp decreases {timing['segments'] - 1} times; file may not be time-ordered"))
        if sampled:
            findings.append(Finding("info", "sampled_row_groups", ts_col,
                                    f"timing from {res.row_groups_read}/{md.num_row_groups} row groups to stay under "
                                    f"{column_budget_bytes // (1024 * 1024)} MB; rates and gaps are estimates"))

    # ---- NaN / inf scan on float columns (within nan_budget_bytes) -------------
    if nan_plan:
        rgs, _, nan_sampled = nan_plan
        try:
            tbl = _read_columns(rf, md, nan_leaves, rgs, workers)
            for c in float_cols:
                arr = tbl.column(c.name).combine_chunks()
                while _is_list(arr.type):
                    arr = pc.list_flatten(arr)
                bad = int(pc.sum(pc.invert(pc.is_finite(arr.drop_null()))).as_py() or 0)
                c.nan_count = bad
                if bad:
                    findings.append(Finding("warn", "nan_values", c.name, f"{bad} NaN or inf values"))
            if nan_sampled:
                findings.append(Finding("info", "nan_check_sampled", None,
                                        f"NaN scan covered {len(rgs)}/{md.num_row_groups} row groups"))
        except Exception as exc:
            findings.append(Finding("warn", "nan_check_failed", None, str(exc)))
    elif float_cols:
        findings.append(Finding("info", "nan_check_skipped", None,
                                "Parquet statistics do not count NaN; NaN scan disabled (--nan-budget-mb 0)"))

    # ---- per-column stream profiles ------------------------------------------------
    channels = []
    for c in cols:
        t = by_name[c.name][0]
        flat_nulls = c.null_count if c.null_count is not None else 0
        channels.append(ChannelProfile(
            topic=c.name,
            schema=c.type,
            modality=_modality(c.name, t) if t is not None else "data",
            codec=c.compression,
            resolution=[c.list_size] if c.list_size else None,
            dims=c.list_size,
            msg_count=(md.num_rows - flat_nulls) if md.num_rows is not None else 0,
            rate_hz=timing.get("rate_hz"),
            jitter_ms=timing.get("jitter_ms"),
            gaps_over_3x=timing.get("gaps_over_3x", 0) or 0,
            max_gap_ms=timing.get("max_gap_ms"),
            nan_count=c.nan_count or 0,
        ))
    return done(res, channels, duration)
