"""
Per-episode video spans of a LeRobot v3 dataset, from `meta/episodes/chunk-*/file-*.parquet`.

LeRobot v3 packs many episodes into each camera's MP4, so a video's own timeline says nothing about where
one episode ends. The episodes table records it: for every episode and camera, `videos/<camera>/
from_timestamp` and `to_timestamp`, the episode's time span inside that camera's file, next to `length`,
the episode's number of table rows. Only those columns are read; the per-episode `stats/...` columns,
which make up most of the file on large datasets, are skipped.

LeRobot v2 keeps one MP4 per camera per episode, so the video span is the file's own duration and this
table is not needed (it is `meta/episodes.jsonl` there, without video spans).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from phys_ai_dataprofiler.remote.parquet import _read_columns
from phys_ai_dataprofiler.remote.range_file import LocalRangeFile

EPISODES_PREFIX = "meta/episodes/"


def episode_tables(paths: list[str]) -> list[str]:
    return sorted(p for p in paths if p.startswith(EPISODES_PREFIX) and p.endswith(".parquet"))


def _wanted(names: list[str]) -> list[str]:
    return [n for n in names if n in ("episode_index", "length")
            or (n.startswith("videos/") and n.endswith(("/from_timestamp", "/to_timestamp")))]


def read_spans(open_range, paths: list[str], budget_bytes: int, workers: int = 8, log=None) -> dict:
    """Video spans per camera and episode from the episodes tables at `paths`.

    `open_range(path) -> RangeFile`. Reads files in parallel, column chunks only, up to `budget_bytes` of
    them; beyond that, files are taken evenly spaced and the result is marked incomplete.

    Returns {"cameras": {camera: {episode: [from_s, to_s]}}, "length": {episode: rows},
             "complete": bool, "files_read": int, "files_total": int, "bytes": int, "error": str | None}.
    """
    log = log or (lambda m: None)
    out: dict = {"cameras": {}, "length": {}, "complete": True, "files_read": 0, "files_total": len(paths),
                 "bytes": 0, "error": None}
    if not paths:
        out["complete"] = False
        out["error"] = "no meta/episodes tables"
        return out

    def plan(path: str):
        rf = open_range(path)
        md = pq.ParquetFile(rf).metadata
        names = [md.schema.column(i).path for i in range(md.num_columns)]
        cols = _wanted(names)
        leaves = [names.index(c) for c in cols]
        need = sum(md.row_group(r).column(i).total_compressed_size or 0
                   for r in range(md.num_row_groups) for i in leaves)
        return path, rf, md, cols, need

    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as ex:
        plans = list(ex.map(plan, paths))
    total = sum(p[4] for p in plans)
    chosen = plans
    if total > budget_bytes:
        keep = max(1, int(len(plans) * budget_bytes / total))
        step = len(plans) / keep
        chosen = [plans[int(i * step)] for i in range(keep)]
        out["complete"] = False
    log(f"    reading video spans from {len(chosen)}/{len(plans)} episodes table(s), "
        f"{sum(p[4] for p in chosen) / 1024:.0f} KB")

    def read(p):
        _, rf, md, cols, _ = p
        # Just the needed column chunks, as the Parquet inspector reads its timestamp column.
        names = [md.schema.column(i).path for i in range(md.num_columns)]
        tbl = _read_columns(rf, md, [names.index(c) for c in cols], list(range(md.num_row_groups)), workers)
        return tbl, rf

    with ThreadPoolExecutor(max_workers=min(workers, len(chosen))) as ex:
        results = list(ex.map(read, chosen))
    for tbl, rf in results:
        out["bytes"] += rf.bytes_fetched
        out["files_read"] += 1
        cols = tbl.column_names
        eps = tbl.column("episode_index").to_pylist()
        if "length" in cols:
            for e, n in zip(eps, tbl.column("length").to_pylist(), strict=True):
                out["length"][int(e)] = int(n)
        cams = {c[len("videos/"):-len("/from_timestamp")] for c in cols if c.endswith("/from_timestamp")}
        for cam in sorted(cams):
            fr = tbl.column(f"videos/{cam}/from_timestamp").to_pylist()
            to = tbl.column(f"videos/{cam}/to_timestamp").to_pylist() if f"videos/{cam}/to_timestamp" in cols else None
            if to is None:
                continue
            d = out["cameras"].setdefault(cam, {})
            for e, a, b in zip(eps, fr, to, strict=True):
                if a is not None and b is not None:
                    d[int(e)] = [float(a), float(b)]
        rf.close()
    return out


def read_spans_local(root, budget_bytes: int, log=None) -> dict:
    """The same for a dataset on disk."""
    root = Path(root)
    paths = episode_tables([p.relative_to(root).as_posix() for p in root.rglob("*.parquet")])
    return read_spans(lambda p: LocalRangeFile(root / p), paths, budget_bytes, log=log)
