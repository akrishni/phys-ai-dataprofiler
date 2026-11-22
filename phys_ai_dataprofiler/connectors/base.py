"""
Pieces shared by every connector (Hugging Face Hub, local filesystem, ...).

A connector feeds files to the remote inspectors. Informally, it provides:

    connect() -> str | None         validate access; return a username or None
    list_files() -> list[RemoteFile]          every file under the target, with size
    open_range(remote, block_size) -> RangeFile   byte-range access, nothing downloaded
    read_text(path, max_bytes) -> str              read one small file whole
    download(remote, local_dir, force) -> Path     fetch/locate a file on local disk
    .video_info: dict[str, dict]      per-camera fps/size/codec, set by load_video_info()
    .resolved_sha: str | None         commit/blob id, if the source has one

`select_files` and `load_video_info` only use this interface, so they work the same for any connector.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from typing import Iterable


@dataclass
class RemoteFile:
    path: str                     # path inside the repo/tree
    size: int | None              # bytes, if known up front
    sha: str | None = None        # blob id / lfs sha, when the source has one


def _interleave_kinds(files: list[RemoteFile]) -> list[RemoteFile]:
    """Order files so a --max-files sample takes from every kind of file in the repo.

    A LeRobot repo sorts `data/*.parquet` before `videos/<camera>/*.mp4`, so the first N paths would be
    tables only and every camera would go unprofiled. Kinds are the top-level folder plus, under
    `videos/`, the camera; each kind keeps its own path order and kinds take turns.
    """
    from phys_ai_dataprofiler.remote.mp4 import video_stream_name
    kinds: dict[str, list[RemoteFile]] = {}
    for f in files:
        top = f.path.split("/", 1)[0] if "/" in f.path else ""
        kind = top + ("/" + video_stream_name(f.path)[0] if f.path.lower().endswith(".mp4") else "")
        kinds.setdefault(kind, []).append(f)
    out: list[RemoteFile] = []
    queues = list(kinds.values())
    i = 0
    while queues:
        q = queues[i % len(queues)]
        out.append(q.pop(0))
        if not q:
            queues.remove(q)
        else:
            i += 1
    return out


def select_files(
    files: Iterable[RemoteFile],
    extensions: Iterable[str],
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    explicit: Iterable[str] | None = None,
    max_files: int | None = None,
    max_file_bytes: int | None = None,
    max_total_bytes: int | None = None,
) -> tuple[list[RemoteFile], list[dict]]:
    """Pick which files to inspect. Returns (selected, skipped_with_reason)."""
    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions)
    explicit_set = set(explicit or [])
    include = list(include or [])
    exclude = list(exclude or [])

    selected: list[RemoteFile] = []
    skipped: list[dict] = []
    total = 0

    ordered = sorted(files, key=lambda x: x.path)
    if max_files is not None and not explicit_set:
        ordered = _interleave_kinds(ordered)
    for f in ordered:
        if explicit_set:
            if f.path not in explicit_set:
                continue
        else:
            if not f.path.lower().endswith(exts):
                continue
            if include and not any(fnmatch.fnmatch(f.path, p) for p in include):
                skipped.append({"path": f.path, "size": f.size, "reason": "not matched by --include"})
                continue
            if exclude and any(fnmatch.fnmatch(f.path, p) for p in exclude):
                skipped.append({"path": f.path, "size": f.size, "reason": "matched --exclude"})
                continue

        if max_file_bytes is not None and f.size is not None and f.size > max_file_bytes:
            skipped.append({"path": f.path, "size": f.size, "reason": "exceeds --max-file-mb"})
            continue
        if max_total_bytes is not None and f.size is not None and total + f.size > max_total_bytes:
            skipped.append({"path": f.path, "size": f.size, "reason": "would exceed --max-total-mb"})
            continue
        if max_files is not None and len(selected) >= max_files:
            skipped.append({"path": f.path, "size": f.size, "reason": "exceeds --max-files"})
            continue

        selected.append(f)
        total += f.size or 0

    missing = explicit_set - {f.path for f in selected} - {s["path"] for s in skipped}
    for path in sorted(missing):
        skipped.append({"path": path, "size": None, "reason": "not found in repo"})

    selected.sort(key=lambda x: x.path)
    return selected, skipped


def load_video_info(connector, files: Iterable[RemoteFile]) -> dict[str, dict]:
    """Read a LeRobot `meta/info.json`, if the source has one, for what each camera should look like.

    Sets `connector.video_info` and returns it. Works for any connector that implements `read_text`
    and has a `.log` callable and a `.video_info` attribute.
    """
    from phys_ai_dataprofiler.remote.mp4 import declared_videos
    if not any(f.path == "meta/info.json" for f in files):
        return {}
    try:
        connector.video_info = declared_videos(json.loads(connector.read_text("meta/info.json")))
    except Exception as exc:
        connector.log(f"Could not read meta/info.json: {exc}")
    return connector.video_info
