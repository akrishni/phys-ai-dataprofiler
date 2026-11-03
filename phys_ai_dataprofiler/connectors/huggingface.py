#!/usr/bin/env python3
"""
Hugging Face Hub connector for the MCAP / MATLAB file inspectors.

Connects to the Hub with an access token, lists the files in a repo
(dataset by default), runs the inspectors on the MCAP / MAT files, and writes
a JSON + Markdown report.

Two modes:

  remote   (default) Nothing is downloaded. Files are read through HTTP Range
           requests and only the bytes that hold metadata are fetched: the
           MCAP header, footer, summary and message indexes; the MAT header
           and per-variable headers. Typically well under 1% of the file.
           Data-dependent checks (MAT NaN counts, unindexed MCAPs) are not
           available in this mode unless the file is small enough for
           --full-read-max-mb, in which case it is fetched whole.

  download Files are downloaded to --download-dir and inspected locally with
           the full-read inspectors (previous behaviour).

Token resolution order:
  1. --token on the command line
  2. HF_TOKEN / HUGGINGFACE_HUB_TOKEN environment variables
  3. the token cached by `hf auth login` (huggingface_hub default)

Typical use:

    phys-ai-dataprofiler org/dataset --token hf_xxx
    phys-ai-dataprofiler org/dataset --dry-run
    phys-ai-dataprofiler org/dataset --include "episodes/*.mcap" --max-files 20
    phys-ai-dataprofiler org/dataset --full-read-max-mb 50     # small files read whole
    phys-ai-dataprofiler org/dataset --mode download --max-total-mb 2000
    phys-ai-dataprofiler org/dataset --mode download --offline  # reuse earlier downloads
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


from phys_ai_dataprofiler.inspectors.mcap import McapFileInspector, MCAP_MAGIC
from phys_ai_dataprofiler.inspectors.mat import MatFileInspector, HDF5_MAGIC, MATFILE_V5_MAGIC
from phys_ai_dataprofiler.inspectors.rich import RichInspector
from phys_ai_dataprofiler.remote.range_file import HfRangeFile
from phys_ai_dataprofiler.remote.inspector import inspect_mcap_remote, inspect_mat_remote, sniff_format
from phys_ai_dataprofiler.remote.parquet import inspect_parquet
from phys_ai_dataprofiler.remote.rosbag import inspect_bag1_remote, inspect_bag2_remote
from phys_ai_dataprofiler.remote.range_file import LocalRangeFile

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    HF_AVAILABLE = True
except ImportError:  # pragma: no cover - reported at runtime
    HF_AVAILABLE = False


DEFAULT_EXTENSIONS = (".mcap", ".mat", ".parquet", ".bag", ".db3")
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class RemoteFile:
    path: str                     # path inside the repo
    size: int | None              # bytes, if the Hub reported it
    sha: str | None = None        # blob id / lfs sha


@dataclass
class FileReport:
    repo_path: str
    local_path: str | None
    size_bytes: int | None
    detected_format: str | None            # "mcap", "mat", "parquet", "bag1", "bag2", "unknown"
    valid: bool
    mode: str = "remote"                   # "remote" or "download"
    bytes_fetched: int | None = None       # bytes actually read from the Hub
    requests: int | None = None            # HTTP requests issued (remote mode)
    download_s: float | None = None
    inspect_s: float | None = None
    error: str | None = None
    basic: dict | None = None              # McapInspectionResult / MatInspectionResult
    rich: dict | None = None               # RichProfile
    summary: dict = field(default_factory=dict)


@dataclass
class RunReport:
    repo_id: str
    repo_type: str
    revision: str | None
    resolved_sha: str | None
    authenticated_as: str | None
    generated_at: str
    mode: str
    download_dir: str | None
    files_listed: int
    files_selected: int
    files_downloaded: int
    files_skipped: list[dict]
    total_bytes_selected: int
    files: list[FileReport]
    totals: dict
    run_cost_s: float
    verdict: dict | None = None            # set when --requirements is given


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert inspector results (dataclasses, bytes, Paths,
    numpy scalars, tuples) into something json.dumps can handle."""
    if _depth > 20:
        return repr(obj)
    if isinstance(obj, str):
        # MAT v5 headers are NUL/space padded to 116 bytes; keep reports clean.
        return obj.replace("\x00", "").rstrip()
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, bytes):
        # Schema payloads can be large; keep a bounded, readable preview.
        text = obj.decode("utf-8", errors="replace")
        preview = text if len(text) <= 512 else text[:512] + "..."
        return {"bytes": len(obj), "hex_prefix": obj[:16].hex(), "preview": preview}
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v, _depth + 1) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v, _depth + 1) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return _to_jsonable(obj.tolist(), _depth + 1)
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return repr(obj)


def _human_size(n: int | None) -> str:
    if n is None:
        return "-"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{n} B"


def _pct(part: int | None, whole: int | None) -> str:
    if not part or not whole:
        return "-"
    return f"{part / whole * 100:.2f}%"


def _sniff_local_format(path: Path) -> str:
    """Detect the container format from magic bytes, falling back to the extension."""
    try:
        rf = LocalRangeFile(path)
    except OSError:
        return "unknown"
    try:
        return sniff_format(rf, path.name)
    finally:
        rf.close()


def resolve_token(explicit: str | None) -> str | None:
    """Return the token to use, or None to fall back to the HF cached login."""
    if explicit:
        return explicit
    for var in TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None


# --------------------------------------------------------------------------- #
# Connector
# --------------------------------------------------------------------------- #

class HuggingFaceConnector:
    """Connects to the Hugging Face Hub and feeds repo files to the inspectors."""

    def __init__(
        self,
        repo_id: str,
        token: str | None = None,
        repo_type: str = "dataset",
        revision: str | None = None,
        log=None,
    ) -> None:
        if not HF_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub is not installed. Run: pip install huggingface_hub"
            )
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.token = resolve_token(token)
        self.api = HfApi(token=self.token)
        self.log = log or (lambda msg: print(msg, file=sys.stderr))
        self.whoami: dict | None = None
        self.resolved_sha: str | None = None

    # -- connection ---------------------------------------------------------

    def connect(self) -> str | None:
        """Validate the token and that the repo is reachable.
        Returns the authenticated username (None for anonymous access)."""
        user = None
        if self.token:
            try:
                info = self.api.whoami()
                user = info.get("name")
                self.log(f"Authenticated to Hugging Face as: {user}")
            except HfHubHTTPError as exc:
                raise RuntimeError(f"Token rejected by Hugging Face: {exc}") from exc
        else:
            try:
                info = self.api.whoami()
                user = info.get("name")
                self.log(f"Using cached Hugging Face login: {user}")
            except Exception:
                self.log("No token provided; accessing the Hub anonymously (public repos only).")

        try:
            info = self.api.repo_info(
                self.repo_id, repo_type=self.repo_type, revision=self.revision
            )
            self.resolved_sha = getattr(info, "sha", None)
            self.log(
                f"Repo {self.repo_type}/{self.repo_id} "
                f"@ {self.revision or 'main'} (sha {self.resolved_sha})"
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Repo '{self.repo_id}' not found as {self.repo_type}. "
                f"Check --repo-type and that the token can see it."
            ) from exc
        except GatedRepoError as exc:
            raise RuntimeError(
                f"Repo '{self.repo_id}' is gated; accept its terms on the Hub "
                f"with this account first."
            ) from exc
        except RevisionNotFoundError as exc:
            raise RuntimeError(f"Revision '{self.revision}' not found.") from exc

        self.whoami = {"name": user} if user else None
        return user

    # -- listing ------------------------------------------------------------

    def list_files(self) -> list[RemoteFile]:
        """List every file in the repo with its size."""
        files: list[RemoteFile] = []
        tree = self.api.list_repo_tree(
            self.repo_id,
            recursive=True,
            repo_type=self.repo_type,
            revision=self.revision,
        )
        for entry in tree:
            if not hasattr(entry, "size"):  # RepoFolder
                continue
            lfs = getattr(entry, "lfs", None)
            sha = None
            if lfs is not None:
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            files.append(RemoteFile(path=entry.path, size=entry.size,
                                    sha=sha or getattr(entry, "blob_id", None)))
        return files

    @staticmethod
    def select_files(
        files: Iterable[RemoteFile],
        extensions: Iterable[str] = DEFAULT_EXTENSIONS,
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

        for f in sorted(files, key=lambda x: x.path):
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

        return selected, skipped

    # -- access -------------------------------------------------------------

    def open_range(self, remote: RemoteFile, block_size: int) -> HfRangeFile:
        """Open a repo file for byte-range reads (nothing is downloaded)."""
        return HfRangeFile(
            self.repo_id, remote.path, token=self.token, repo_type=self.repo_type,
            revision=self.revision, block_size=block_size,
        )

    def read_text(self, path: str, max_bytes: int = 4 << 20) -> str:
        """Read a small repo file (e.g. a rosbag2 metadata.yaml) without downloading it to disk."""
        rf = HfRangeFile(self.repo_id, path, token=self.token, repo_type=self.repo_type,
                         revision=self.revision)
        try:
            if rf.size > max_bytes:
                raise RuntimeError(f"{path} is {rf.size} B, over the {max_bytes} B cap")
            return rf.read_range(0, rf.size).decode("utf-8", "replace")
        finally:
            rf.close()

    def download(self, remote: RemoteFile, local_dir: Path, force: bool = False) -> Path:
        """Download one file into local_dir, preserving its repo path."""
        local_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=remote.path,
            repo_type=self.repo_type,
            revision=self.revision,
            local_dir=str(local_dir),
            token=self.token,
            force_download=force,
        )
        return Path(path)


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #

def _fill_from_results(report: FileReport, fmt: str, res, prof) -> None:
    """Populate a FileReport from a basic inspection result and a RichProfile."""
    report.basic = _to_jsonable(res)
    if fmt == "mcap":
        report.valid = bool(res.is_mcap and not res.error)
        report.error = res.error
        report.summary = {
            "profile": res.header.profile if res.header else None,
            "library": res.header.library if res.header else None,
            "channels": len(res.channels or []),
            "schemas": len(res.schemas or []),
            "topics": res.topics or [],
        }
    elif fmt == "parquet":
        report.valid = bool(res.is_parquet and not res.error)
        report.error = res.error
        report.summary = {
            "columns": len(res.columns or []),
            "rows": res.num_rows,
            "row_groups": res.num_row_groups,
            "timestamp_column": res.timestamp_column,
            "episode_column": res.episode_column,
            "frame_rate_hz": res.frame_rate_hz,
            "episodes": (res.episodes or {}).get("count"),
        }
    elif fmt in ("bag1", "bag2"):
        report.valid = bool(res.is_bag and not res.error)
        report.error = res.error
        report.summary = {
            "bag_version": res.version,
            "storage": res.storage,
            "channels": len(res.connections or []),
            "topics": res.topics or [],
            "chunks": res.chunk_count,
            "compression": res.compression,
        }
    else:
        report.valid = bool(res.is_matfile and not res.error)
        report.error = res.error
        report.summary = {
            "mat_format": res.file_format,
            "variables": len(res.variables or []),
            "variable_names": res.variable_names or [],
        }
    if prof is not None:
        report.rich = _to_jsonable(prof)
        report.summary.update({
            "duration_s": prof.duration_s,
            "messages": (report.summary.get("rows") if fmt == "parquet"
                         else sum(c.msg_count for c in prof.channels)),
            "findings": len(prof.findings),
            "warnings": sum(1 for f in prof.findings if f.severity == "warn"),
            "errors": sum(1 for f in prof.findings if f.severity == "error"),
        })


def inspect_local_file(repo_path: str, local_path: Path, size_hint: int | None, deep: bool = True) -> FileReport:
    """Full-read inspection of a file on disk (download mode)."""
    t0 = time.time()
    fmt = _sniff_local_format(local_path)
    size = local_path.stat().st_size if local_path.exists() else size_hint
    report = FileReport(repo_path=repo_path, local_path=str(local_path), size_bytes=size,
                        detected_format=fmt, valid=False, mode="download",
                        bytes_fetched=size)
    try:
        if fmt == "mcap":
            res = McapFileInspector(str(local_path)).inspect()
        elif fmt == "mat":
            res = MatFileInspector(str(local_path)).inspect()
        elif fmt in ("parquet", "bag1", "bag2"):
            rf = LocalRangeFile(local_path, block_size=1024 * 1024)
            try:
                if fmt == "parquet":
                    res, prof = inspect_parquet(rf, column_budget_bytes=1 << 62, nan_check=True)
                elif fmt == "bag1":
                    res, prof = inspect_bag1_remote(rf, msgidx_budget_bytes=1 << 62)
                else:
                    sidecar = local_path.parent / "metadata.yaml"
                    res, prof = inspect_bag2_remote(
                        rf, msgidx_budget_bytes=1 << 62, file_name=local_path.name,
                        sidecar_yaml=(lambda: sidecar.read_text()) if sidecar.exists() else None)
            finally:
                rf.close()
            _fill_from_results(report, fmt, res, prof)
            report.inspect_s = round(time.time() - t0, 3)
            return report
        else:
            report.error = "Not recognised as MCAP, MAT, Parquet or ROS bag (magic bytes and extension both unknown)"
            report.inspect_s = round(time.time() - t0, 3)
            return report
        prof = None
        if deep:
            try:
                prof = RichInspector(str(local_path)).inspect()
            except Exception as exc:
                report.rich = {"error": f"Rich inspection failed: {exc}"}
        _fill_from_results(report, fmt, res, prof)
        if deep and prof is None and report.rich is None:
            report.rich = {"error": "Rich inspection failed"}
    except Exception as exc:
        report.error = f"Inspection failed: {exc}"
    report.inspect_s = round(time.time() - t0, 3)
    return report


def _sidecar_path(repo_path: str) -> str:
    """The rosbag2 metadata.yaml that sits next to a .db3 file."""
    return repo_path.rsplit("/", 1)[0] + "/metadata.yaml" if "/" in repo_path else "metadata.yaml"


def inspect_remote_file(connector: HuggingFaceConnector, remote: RemoteFile, args, log) -> FileReport:
    """Metadata-only inspection over HTTP Range requests (remote mode)."""
    t0 = time.time()
    report = FileReport(repo_path=remote.path, local_path=None, size_bytes=remote.size,
                        detected_format=None, valid=False, mode="remote")
    rf = None
    nan_budget = int(args.nan_budget_mb * 1024 * 1024)
    try:
        rf = connector.open_range(remote, block_size=args.block_kb * 1024)
        report.size_bytes = rf.size or remote.size
        fmt = sniff_format(rf, remote.path)
        report.detected_format = fmt
        if fmt == "mcap":
            res, prof = inspect_mcap_remote(
                rf, msgidx_budget_bytes=int(args.msgidx_budget_mb * 1024 * 1024),
                workers=args.workers, log=log,
            )
            _fill_from_results(report, fmt, res, prof)
        elif fmt == "mat":
            res, prof = inspect_mat_remote(rf, log=log, nan_budget_bytes=nan_budget, workers=args.workers)
            _fill_from_results(report, fmt, res, prof)
        elif fmt == "parquet":
            res, prof = inspect_parquet(
                rf, column_budget_bytes=int(args.column_budget_mb * 1024 * 1024),
                nan_check=nan_budget > 0, nan_budget_bytes=nan_budget,
                workers=args.workers, log=log,
            )
            _fill_from_results(report, fmt, res, prof)
        elif fmt == "bag1":
            res, prof = inspect_bag1_remote(
                rf, msgidx_budget_bytes=int(args.msgidx_budget_mb * 1024 * 1024), workers=args.workers, log=log)
            _fill_from_results(report, fmt, res, prof)
        elif fmt == "bag2":
            res, prof = inspect_bag2_remote(
                rf, msgidx_budget_bytes=int(args.msgidx_budget_mb * 1024 * 1024), workers=args.workers, log=log,
                sidecar_yaml=lambda: connector.read_text(_sidecar_path(remote.path)),
                file_name=remote.path.rsplit("/", 1)[-1])
            _fill_from_results(report, fmt, res, prof)
        else:
            report.error = "Not recognised as MCAP, MAT, Parquet or ROS bag (magic bytes and extension both unknown)"
    except Exception as exc:
        report.error = f"Remote inspection failed: {exc}"
    finally:
        if rf is not None:
            report.bytes_fetched = rf.bytes_fetched
            report.requests = rf.requests
            rf.close()
    report.inspect_s = round(time.time() - t0, 3)
    return report


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def build_totals(files: list[FileReport]) -> dict:
    by_format: dict[str, int] = {}
    for f in files:
        key = f.detected_format or "unknown"
        by_format[key] = by_format.get(key, 0) + 1
    size = sum(f.size_bytes or 0 for f in files)
    fetched = sum(f.bytes_fetched or 0 for f in files)
    return {
        "files": len(files),
        "valid": sum(1 for f in files if f.valid),
        "invalid_or_errored": sum(1 for f in files if not f.valid),
        "bytes": size,
        "bytes_fetched": fetched,
        "fetched_pct": round(fetched / size * 100, 3) if size else None,
        "requests": sum(f.requests or 0 for f in files),
        "by_format": by_format,
        "by_mode": {m: sum(1 for f in files if f.mode == m) for m in ("remote", "download") if any(f.mode == m for f in files)},
        "messages": sum(f.summary.get("messages", 0) or 0 for f in files),
        "duration_s": round(sum(f.summary.get("duration_s", 0) or 0 for f in files), 3),
        "findings": sum(f.summary.get("findings", 0) or 0 for f in files),
        "warnings": sum(f.summary.get("warnings", 0) or 0 for f in files),
        "errors": sum(f.summary.get("errors", 0) or 0 for f in files)
                  + sum(1 for f in files if f.error),
    }


def render_markdown(run: RunReport, verdict_md: str | None = None) -> str:
    L: list[str] = []
    L.append(f"# Inspection report: `{run.repo_type}/{run.repo_id}`")
    L.append("")
    L.append(f"- Generated: {run.generated_at}")
    L.append(f"- Revision: `{run.revision or 'main'}` (sha `{run.resolved_sha or '-'}`)")
    L.append(f"- Authenticated as: {run.authenticated_as or 'anonymous'}")
    L.append(f"- Mode: {run.mode}" + (" (HTTP byte-range reads, nothing downloaded)" if run.mode == "remote" else ""))
    if run.download_dir:
        L.append(f"- Download dir: `{run.download_dir}`")
    L.append(f"- Files in repo: {run.files_listed}; selected: {run.files_selected}; "
             f"downloaded: {run.files_downloaded}; skipped: {len(run.files_skipped)}")
    L.append(f"- Selected size: {_human_size(run.total_bytes_selected)}")
    L.append(f"- Run time: {run.run_cost_s:.1f} s")
    L.append("")
    if verdict_md:
        L.append(verdict_md)
        L.append("")

    t = run.totals
    L.append("## Totals")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Files inspected | {t['files']} |")
    L.append(f"| Valid | {t['valid']} |")
    L.append(f"| Invalid / errored | {t['invalid_or_errored']} |")
    L.append(f"| By format | {', '.join(f'{k}: {v}' for k, v in t['by_format'].items()) or '-'} |")
    L.append(f"| By mode | {', '.join(f'{k}: {v}' for k, v in t['by_mode'].items()) or '-'} |")
    L.append(f"| Total size | {_human_size(t['bytes'])} |")
    L.append(f"| Bytes read from Hub | {_human_size(t['bytes_fetched'])} ({_pct(t['bytes_fetched'], t['bytes'])}) in {t['requests']} requests |")
    L.append(f"| Messages (MCAP, bags) / rows (MAT, Parquet) | {t['messages']:,} |")
    L.append(f"| Total duration | {t['duration_s']} s |")
    L.append(f"| Findings | {t['findings']} (warn {t['warnings']}, error {t['errors']}) |")
    L.append("")

    L.append("## Files")
    L.append("")
    L.append("| File | Size | Read | Format | Valid | Channels/Vars | Duration | Msgs/Rows | Findings | Error |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for f in run.files:
        s = f.summary
        chans = s.get("channels", s.get("variables", "-"))
        dur = f"{s['duration_s']:.1f} s" if s.get("duration_s") else "-"
        msgs = f"{s['messages']:,}" if s.get("messages") is not None else "-"
        err = (f.error or "").replace("|", "\\|")
        if len(err) > 80:
            err = err[:77] + "..."
        read = f"{_human_size(f.bytes_fetched)} ({_pct(f.bytes_fetched, f.size_bytes)})" if f.bytes_fetched is not None else "-"
        L.append(
            f"| `{f.repo_path}` | {_human_size(f.size_bytes)} | {read} | {f.detected_format} | "
            f"{'yes' if f.valid else 'no'} | {chans} | {dur} | {msgs} | "
            f"{s.get('findings', '-')} | {err or '-'} |"
        )
    L.append("")

    for f in run.files:
        L.append(f"### `{f.repo_path}`")
        L.append("")
        L.append(f"- Mode: {f.mode}; read {_human_size(f.bytes_fetched)} of {_human_size(f.size_bytes)}"
                 + (f" in {f.requests} requests" if f.requests else "")
                 + f"; inspect time {f.inspect_s}s")
        if f.detected_format == "mcap" and f.basic:
            L.append(f"- Profile: `{f.summary.get('profile') or '-'}`, library: `{f.summary.get('library') or '-'}`")
            topics = f.summary.get("topics") or []
            if topics:
                L.append(f"- Topics ({len(topics)}):")
                for topic in topics[:50]:
                    L.append(f"  - `{topic}`")
                if len(topics) > 50:
                    L.append(f"  - ... {len(topics) - 50} more")
            schemas = (f.basic or {}).get("schemas") or []
            if schemas:
                L.append(f"- Schemas ({len(schemas)}):")
                for sc in schemas[:50]:
                    L.append(f"  - `{sc.get('name')}` ({sc.get('encoding')}, {sc.get('data', {}).get('bytes', 0)} bytes)")
        elif f.detected_format == "mat" and f.basic:
            hdr = (f.basic or {}).get("header") or {}
            L.append(f"- MAT format: {f.summary.get('mat_format')}, version {hdr.get('version', '-')}, "
                     f"endian {hdr.get('endian', '-')}")
            if hdr.get("description"):
                L.append(f"- Description: {hdr['description']}")
            variables = (f.basic or {}).get("variables") or []
            if variables:
                size_note = "stored bytes" if f.mode == "remote" else "bytes"
                L.append("")
                L.append(f"| Variable | dtype | shape | {size_note} | flags |")
                L.append("|---|---|---|---|---|")
                for v in variables[:200]:
                    flags = [k[3:] for k in ("is_sparse", "is_complex", "is_struct", "is_cell") if v.get(k)]
                    if v.get("field_names"):
                        flags.append(f"fields: {', '.join(v['field_names'][:8])}" + (" ..." if len(v['field_names']) > 8 else ""))
                    L.append(f"| `{v['name']}` | {v['dtype']} | {tuple(v['shape'])} | {v['size_bytes']} | {', '.join(flags) or '-'} |")
                if len(variables) > 200:
                    L.append(f"| ... {len(variables) - 200} more | | | | |")

        if f.detected_format in ("bag1", "bag2") and f.basic:
            b = f.basic
            if f.detected_format == "bag1":
                L.append(f"- ROS 1 bag V{b.get('version') or '?'}, {'indexed' if b.get('indexed') else 'no index'}; "
                         f"{b.get('chunk_count') or 0} chunk(s), compression {', '.join(b.get('compression') or ['-'])}")
            else:
                L.append(f"- ROS 2 bag, storage `{b.get('storage') or '-'}`, metadata v{b.get('version') or '-'} "
                         f"({b.get('metadata_source') or 'none found'}), distro `{b.get('ros_distro') or '-'}`, "
                         f"page size {b.get('page_size')}; {b.get('leaves_read')} leaf page(s) read")
            if b.get("message_count") is not None:
                L.append(f"- Messages: {b['message_count']:,}"
                         + (f" over {b['duration_s']:.1f} s" if b.get("duration_s") else ""))

        if f.detected_format == "parquet" and f.basic:
            b = f.basic
            L.append(f"- Rows: {b.get('num_rows') or 0:,} in {b.get('num_row_groups')} row group(s); "
                     f"writer `{b.get('created_by') or '-'}`, format {b.get('format_version') or '-'}")
            if b.get("timestamp_column"):
                rate = f"{b['frame_rate_hz']:.2f} Hz" if b.get("frame_rate_hz") else "-"
                L.append(f"- Time: column `{b['timestamp_column']}` ({b.get('timestamp_unit')}), rate {rate}"
                         + (f", episodes from `{b['episode_column']}`" if b.get("episode_column") else "")
                         + (f", {b['row_groups_read']}/{b['num_row_groups']} row groups read" if b.get("row_groups_read") is not None else ""))
            ep = b.get("episodes")
            if ep:
                L.append(f"- Episodes{'' if ep.get('complete') else ' (sampled)'}: {ep['count']}, "
                         f"frames min/median/max {ep['frames_min']}/{ep['frames_median']:g}/{ep['frames_max']}, "
                         f"seconds {ep['seconds_min']}/{ep['seconds_median']}/{ep['seconds_max']}")
            if b.get("metadata_keys"):
                L.append(f"- Key-value metadata: {', '.join('`' + k + '`' for k in b['metadata_keys'][:10])}")
            pcols = b.get("columns") or []
            if pcols:
                L.append("")
                L.append("| Column | Type | Nulls | Min | Max | NaN/inf | Compressed | Codec |")
                L.append("|---|---|---|---|---|---|---|---|")
                for c in pcols[:200]:
                    def cell(v):
                        return "-" if v is None else str(v).replace("|", "\\|")
                    L.append(f"| `{c['name']}` | {cell(c['type'])} | {cell(c.get('null_count'))} | {cell(c.get('min'))} | "
                             f"{cell(c.get('max'))} | {cell(c.get('nan_count'))} | {_human_size(c.get('compressed_bytes'))} | "
                             f"{cell(c.get('compression'))} |")
                if len(pcols) > 200:
                    L.append(f"| ... {len(pcols) - 200} more | | | | | | | |")

        rich = f.rich or {}
        channels = rich.get("channels") or []
        if channels and f.detected_format in ("mcap", "bag1", "bag2"):
            L.append("")
            L.append("| Topic | Schema | Modality | Count | Rate (Hz) | Jitter (ms) | Gaps>3x | Max gap (ms) |")
            L.append("|---|---|---|---|---|---|---|---|")
            for c in channels[:200]:
                rate = f"{c['rate_hz']:.2f}" if c.get("rate_hz") is not None else "-"
                jit = f"{c['jitter_ms']:.2f}" if c.get("jitter_ms") is not None else "-"
                mg = f"{c['max_gap_ms']:.1f}" if c.get("max_gap_ms") is not None else "-"
                L.append(f"| `{c['topic']}` | {c['schema']} | {c.get('modality') or '-'} | {c['msg_count']:,} | "
                         f"{rate} | {jit} | {c.get('gaps_over_3x', 0)} | {mg} |")
        elif channels and f.detected_format == "mat":
            nan_rows = [c for c in channels if c.get("nan_count")]
            if nan_rows:
                L.append("")
                L.append("| Variable | NaNs |")
                L.append("|---|---|")
                for c in nan_rows:
                    L.append(f"| `{c['topic']}` | {c['nan_count']} |")
        findings = rich.get("findings") or []
        if findings:
            L.append("")
            L.append("Findings:")
            for fd in findings:
                topic = f" `{fd['topic']}`" if fd.get("topic") else ""
                L.append(f"- **{fd['severity']}** {fd['type']}{topic}: {fd['detail']}")
        if isinstance(rich, dict) and rich.get("error"):
            L.append(f"- Rich inspection: {rich['error']}")
        if f.error:
            L.append(f"- **Error:** {f.error}")
        L.append("")

    if run.files_skipped:
        L.append("## Skipped files")
        L.append("")
        L.append("| File | Size | Reason |")
        L.append("|---|---|---|")
        for s in run.files_skipped:
            L.append(f"| `{s['path']}` | {_human_size(s.get('size'))} | {s['reason']} |")
        L.append("")

    return "\n".join(L)


def print_console_summary(run: RunReport, log) -> None:
    log("")
    log("=" * 110)
    log(f"HUGGING FACE INSPECTION SUMMARY  {run.repo_type}/{run.repo_id}  [{run.mode}]")
    log("=" * 110)
    t = run.totals
    log(f"Files inspected: {t['files']}   valid: {t['valid']}   errored: {t['invalid_or_errored']}   "
        f"size: {_human_size(t['bytes'])}   read: {_human_size(t['bytes_fetched'])} ({_pct(t['bytes_fetched'], t['bytes'])})   "
        f"findings: {t['findings']}")
    log("")
    log(f"{'File':<44} {'Size':>10} {'Read':>10} {'Fmt':>5} {'Valid':>5} {'Ch/Var':>7} {'Dur':>9} {'Find':>5} {'Time':>6}")
    log("-" * 110)
    for f in run.files:
        name = f.repo_path if len(f.repo_path) <= 42 else "..." + f.repo_path[-39:]
        s = f.summary
        dur = f"{s['duration_s']:.1f}s" if s.get("duration_s") else "-"
        log(f"{name:<44} {_human_size(f.size_bytes):>10} {_human_size(f.bytes_fetched):>10} {f.detected_format or '-':>5} "
            f"{'yes' if f.valid else 'no':>5} {str(s.get('channels', s.get('variables', s.get('columns', '-')))):>7} "
            f"{dur:>9} {str(s.get('findings', '-')):>5} {str(f.inspect_s) + 's':>6}")
    if run.files_skipped:
        log("")
        log(f"Skipped {len(run.files_skipped)} file(s); see report for reasons.")
    log("=" * 110)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

class NothingToProfile(RuntimeError):
    """The repo has no files this tool can profile."""


def run(args: argparse.Namespace) -> RunReport:
    log = (lambda msg: None) if args.quiet else (lambda msg: print(msg, file=sys.stderr))
    t_start = time.time()

    download_dir = Path(args.download_dir).resolve()
    repo_slug = args.repo_id.replace("/", "__")
    local_root = download_dir / f"{args.repo_type}__{repo_slug}"
    full_read_bytes = int(args.full_read_max_mb * 1024 * 1024) if args.full_read_max_mb else 0

    connector: HuggingFaceConnector | None = None
    user = None
    resolved_sha = None
    skipped: list[dict] = []
    reports: list[FileReport] = []
    files_listed = 0
    selected: list[RemoteFile] = []
    downloaded = 0

    if args.offline:
        if not local_root.exists():
            raise RuntimeError(f"--offline given but nothing downloaded at {local_root}")
        exts = tuple(e if e.startswith(".") else f".{e}" for e in args.ext)
        local_files = sorted(p for p in local_root.rglob("*")
                             if p.is_file() and not p.name.startswith(".")
                             and p.suffix.lower() in exts)
        files_listed = len(local_files)
        for p in local_files:
            selected.append(RemoteFile(path=p.relative_to(local_root).as_posix(), size=p.stat().st_size))
        log(f"Offline mode: inspecting {len(selected)} file(s) under {local_root}")
        mode = "download"
    else:
        mode = args.mode
        connector = HuggingFaceConnector(
            repo_id=args.repo_id, token=args.token, repo_type=args.repo_type,
            revision=args.revision, log=log,
        )
        user = connector.connect()
        resolved_sha = connector.resolved_sha

        remote_files = connector.list_files()
        files_listed = len(remote_files)
        log(f"Repo contains {files_listed} file(s), "
            f"{_human_size(sum(f.size or 0 for f in remote_files))} total")

        selected, skipped = HuggingFaceConnector.select_files(
            remote_files,
            extensions=args.ext,
            include=args.include,
            exclude=args.exclude,
            explicit=args.file,
            max_files=args.max_files,
            max_file_bytes=int(args.max_file_mb * 1024 * 1024) if args.max_file_mb else None,
            max_total_bytes=int(args.max_total_mb * 1024 * 1024) if args.max_total_mb else None,
        )
        log(f"Selected {len(selected)} file(s), "
            f"{_human_size(sum(f.size or 0 for f in selected))}; skipped {len(skipped)}; mode={mode}")
        if not selected and not args.file:
            from collections import Counter
            exts = Counter((Path(f.path).suffix.lower() or "(no extension)") for f in remote_files)
            found = ", ".join(f"{n} {e}" for e, n in exts.most_common(8))
            wanted = " ".join(args.ext)
            if not skipped:
                raise NothingToProfile(
                    f"No {wanted} files in {args.repo_type}/{args.repo_id}. It contains: {found}. "
                    f"Use --ext to choose other extensions, or --file for a specific path.")
        for f in selected:
            log(f"  + {f.path}  ({_human_size(f.size)})")

        if args.dry_run:
            log("Dry run: not inspecting.")
            return RunReport(
                repo_id=args.repo_id, repo_type=args.repo_type, revision=args.revision,
                resolved_sha=resolved_sha, authenticated_as=user,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                mode=mode, download_dir=None, files_listed=files_listed,
                files_selected=len(selected), files_downloaded=0, files_skipped=skipped,
                total_bytes_selected=sum(f.size or 0 for f in selected),
                files=[], totals=build_totals([]), run_cost_s=round(time.time() - t_start, 2),
            )

    for i, remote in enumerate(selected, 1):
        prefix = f"[{i}/{len(selected)}]"
        use_download = (
            mode == "download"
            or connector is None
            or (full_read_bytes and remote.size is not None and remote.size <= full_read_bytes)
        )

        if not use_download:
            log(f"{prefix} remote-inspecting {remote.path} ({_human_size(remote.size)})")
            rep = inspect_remote_file(connector, remote, args, log)
            reports.append(rep)
            status = "ok" if rep.valid else f"PROBLEM: {rep.error}"
            log(f"    {rep.detected_format} {status}; read {_human_size(rep.bytes_fetched)} "
                f"({_pct(rep.bytes_fetched, rep.size_bytes)}) in {rep.requests} requests, {rep.inspect_s}s")
            continue

        local_path = local_root / remote.path
        dl_s = None
        if connector is not None:
            why = "full read" if mode == "remote" else "downloading"
            log(f"{prefix} {why} {remote.path} ({_human_size(remote.size)})")
            t0 = time.time()
            try:
                local_path = connector.download(remote, local_root, force=args.force_download)
                if remote.path.lower().endswith(".db3"):
                    try:
                        connector.download(RemoteFile(_sidecar_path(remote.path), None), local_root,
                                           force=args.force_download)
                    except Exception:
                        pass  # the bag may embed its metadata, or have none
                dl_s = round(time.time() - t0, 2)
                downloaded += 1
            except Exception as exc:
                reports.append(FileReport(
                    repo_path=remote.path, local_path=None, size_bytes=remote.size,
                    detected_format=None, valid=False, mode="download",
                    error=f"Download failed: {exc}",
                ))
                log(f"    download failed: {exc}")
                continue

        log(f"{prefix} inspecting {remote.path}")
        rep = inspect_local_file(remote.path, Path(local_path), remote.size, deep=not args.no_rich)
        rep.download_s = dl_s
        reports.append(rep)
        status = "ok" if rep.valid else f"PROBLEM: {rep.error}"
        log(f"    {rep.detected_format} {status} ({rep.inspect_s}s)")

        # In remote mode the full-read fallback is temporary unless asked to keep it.
        cleanup = args.delete_after or (mode == "remote" and not args.keep_downloads and connector is not None)
        if cleanup and Path(local_path).exists():
            try:
                Path(local_path).unlink()
                rep.local_path = None
            except OSError:
                pass

    return RunReport(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        resolved_sha=resolved_sha,
        authenticated_as=user,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        mode=mode,
        download_dir=str(local_root) if downloaded or args.offline else None,
        files_listed=files_listed,
        files_selected=len(selected),
        files_downloaded=downloaded,
        files_skipped=skipped,
        total_bytes_selected=sum(f.size or 0 for f in selected),
        files=reports,
        totals=build_totals(reports),
        run_cost_s=round(time.time() - t_start, 2),
    )


def write_reports(run_report: RunReport, report_dir: Path, basename: str | None, log,
                  verdict_md: str | None = None) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = basename or f"{run_report.repo_type}__{run_report.repo_id.replace('/', '__')}__{stamp}"
    json_path = report_dir / f"{base}.json"
    md_path = report_dir / f"{base}.md"
    json_path.write_text(json.dumps(_to_jsonable(run_report), indent=2))
    md_path.write_text(render_markdown(run_report, verdict_md))
    log(f"JSON report: {json_path}")
    log(f"Markdown report: {md_path}")
    return json_path, md_path


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Profile MCAP, ROS bag, MAT and Parquet files in a Hugging Face repo and write a report. "
                    "Other commands: check, compare, requirements (run '<command> -h').",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("repo_id", help="Hub repo id, e.g. org/dataset-name")
    p.add_argument("--token", help="Hugging Face access token (else HF_TOKEN env or cached login)")
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    p.add_argument("--revision", help="Branch, tag or commit sha (default: main)")

    mode = p.add_argument_group("access mode")
    mode.add_argument("--mode", default="remote", choices=["remote", "download"],
                      help="remote: byte-range reads, nothing downloaded (default); download: fetch files to disk")
    mode.add_argument("--full-read-max-mb", type=float, default=0,
                      help="In remote mode, files at or below this size are fetched whole for full-fidelity "
                           "checks. Default 0 = never.")
    mode.add_argument("--msgidx-budget-mb", type=float, default=64,
                      help="Max MCAP message-index bytes to fetch per file for timing stats; beyond this, chunks are "
                           "sampled. Also caps the record walk of MCAPs that have no summary (unindexed or truncated)")
    mode.add_argument("--nan-budget-mb", type=float, default=32,
                      help="Max bytes of float data (MAT variables, Parquet float columns) to read per file for "
                           "NaN/inf counts; beyond this the scan is partial and says so. 0 disables the scan")
    mode.add_argument("--column-budget-mb", type=float, default=16,
                      help="Max bytes of Parquet timestamp/episode columns to read per file for timing; "
                           "beyond this, row groups are sampled")
    mode.add_argument("--workers", type=int, default=16, help="Parallel range requests per file")
    mode.add_argument("--block-kb", type=int, default=64, help="Block size for cached range reads")

    sel = p.add_argument_group("file selection")
    sel.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTENSIONS),
                     help="Extensions to consider (default: .mcap .mat .parquet .bag .db3)")
    sel.add_argument("--include", action="append", help="Glob on repo path; repeatable")
    sel.add_argument("--exclude", action="append", help="Glob on repo path to skip; repeatable")
    sel.add_argument("--file", action="append", help="Exact repo path to fetch; repeatable (bypasses --ext)")
    sel.add_argument("--max-files", type=int, help="Stop after this many files")
    sel.add_argument("--max-file-mb", type=float, help="Skip files larger than this")
    sel.add_argument("--max-total-mb", type=float, help="Cap the total selected size")

    rq = p.add_argument_group("requirements")
    rq.add_argument("--requirements", "-r", metavar="FILE|preset:NAME",
                    help="Check the profile against a requirements YAML file or built-in preset and add a "
                         "pass/fail/unknown verdict to the report. Exit code: 0 pass, 1 fail, 4 unknown")

    io = p.add_argument_group("storage and output")
    io.add_argument("--download-dir", default="./hf_downloads", help="Where downloaded files are stored")
    io.add_argument("--report-dir", default="./reports", help="Where the JSON/Markdown reports go")
    io.add_argument("--report-name", help="Report basename (default: repo + timestamp)")
    io.add_argument("--keep-downloads", action="store_true",
                    help="Remote mode: keep files fetched via --full-read-max-mb instead of deleting them")
    io.add_argument("--force-download", action="store_true", help="Download mode: re-download even if cached")
    io.add_argument("--delete-after", action="store_true", help="Download mode: delete each file after inspecting it")
    io.add_argument("--offline", action="store_true",
                    help="Skip the Hub; inspect files already in --download-dir for this repo")
    io.add_argument("--dry-run", action="store_true", help="List what would be inspected and exit")
    io.add_argument("--no-rich", action="store_true", help="Download mode: skip the deep (message-level) inspector")
    io.add_argument("--print-json", action="store_true", help="Also print the JSON report to stdout")
    io.add_argument("--quiet", action="store_true", help="Suppress progress output on stderr")
    return p


SUBCOMMANDS = ("check", "compare", "requirements")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        from phys_ai_dataprofiler import cli
        return cli.main(argv)
    args = build_parser().parse_args(argv)
    return profile_main(args)


def profile_main(args: argparse.Namespace) -> int:
    """Profile one repo; with --requirements, add a verdict. Shared by the default command and `check`."""
    log = (lambda msg: None) if args.quiet else (lambda msg: print(msg, file=sys.stderr))
    req = None
    if getattr(args, "requirements", None):
        from phys_ai_dataprofiler.requirements import RequirementsError, load_requirements
        try:
            req = load_requirements(args.requirements)    # fail fast, before any network traffic
        except RequirementsError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _apply_requirements_scope(args, req)
    try:
        run_report = run(args)
    except NothingToProfile as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

    verdict = verdict_md = None
    if req is not None and not args.dry_run:
        from phys_ai_dataprofiler.evaluate import evaluate, render_verdict_markdown, verdict_to_dict
        verdict = evaluate(_to_jsonable(run_report), req)
        run_report.verdict = verdict_to_dict(verdict)
        verdict_md = render_verdict_markdown(verdict)

    if not args.dry_run:
        print_console_summary(run_report, log)
        if verdict is not None:
            from phys_ai_dataprofiler.evaluate import print_verdict
            print_verdict(verdict, log)
    if not getattr(args, "no_report", False):
        write_reports(run_report, Path(args.report_dir), args.report_name, log, verdict_md)
    if args.print_json:
        print(json.dumps(_to_jsonable(run_report), indent=2))

    if verdict is not None:
        return verdict.exit_code
    if not args.dry_run and run_report.totals["invalid_or_errored"] > 0:
        return 1
    return 0


def _apply_requirements_scope(args: argparse.Namespace, req) -> None:
    """files.include/exclude in the requirements narrow what gets profiled, on top of --include/--exclude."""
    if req.files_include and not args.include and not args.file:
        args.include = list(req.files_include)
    if req.files_exclude:
        args.exclude = list(args.exclude or []) + list(req.files_exclude)


if __name__ == "__main__":
    sys.exit(main())
