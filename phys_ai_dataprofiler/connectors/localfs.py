"""
Local filesystem connector: feeds a directory (or a single file) on disk to the inspectors.

Uses the same byte-range inspection path as the Hugging Face connector, through `LocalRangeFile`, so
large local files (multi-GB `.mcap` / `.mat`) are profiled from their headers and indexes rather than
read in full.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from phys_ai_dataprofiler.connectors.base import RemoteFile
from phys_ai_dataprofiler.remote.range_file import LocalRangeFile


class LocalFsConnector:
    """Connects to a directory or file on local disk."""

    def __init__(self, root: Path, log=None) -> None:
        self.root = Path(root)
        self.single_file: Path | None = None
        self.log = log or (lambda msg: None)
        self.video_info: dict[str, dict] = {}
        self.resolved_sha: str | None = None

    # -- connection ---------------------------------------------------------

    def connect(self) -> str | None:
        """Validate the path exists. Returns None (there is no "authenticated as" for local files)."""
        if not self.root.exists():
            raise RuntimeError(f"Path '{self.root}' does not exist")
        if self.root.is_file():
            self.single_file = self.root
            self.root = self.root.parent
        self.log(f"Local path {self.single_file or self.root}")
        return None

    # -- listing --------------------------------------------------------------

    def list_files(self) -> list[RemoteFile]:
        """List every file under the target, with its size."""
        if self.single_file is not None:
            st = self.single_file.stat()
            return [RemoteFile(path=self.single_file.name, size=st.st_size, sha=None)]
        files: list[RemoteFile] = []
        for p in self.root.rglob("*"):
            if not p.is_file() or p.name.startswith("."):
                continue
            files.append(RemoteFile(path=p.relative_to(self.root).as_posix(), size=p.stat().st_size, sha=None))
        return files

    # -- access -------------------------------------------------------------

    def open_range(self, remote: RemoteFile, block_size: int) -> LocalRangeFile:
        """Open a file under the target for byte-range reads."""
        return LocalRangeFile(self.root / remote.path, block_size=block_size)

    def read_text(self, path: str, max_bytes: int = 4 << 20) -> str:
        """Read a small file (e.g. a rosbag2 metadata.yaml) whole."""
        full = self.root / path
        size = full.stat().st_size
        if size > max_bytes:
            raise RuntimeError(f"{path} is {size} B, over the {max_bytes} B cap")
        return full.read_text(encoding="utf-8", errors="replace")

    def load_video_info(self, files: Iterable[RemoteFile]) -> dict[str, dict]:
        from phys_ai_dataprofiler.connectors.base import load_video_info
        return load_video_info(self, files)

    def download(self, remote: RemoteFile, local_dir: Path, force: bool = False) -> Path:
        """Nothing to fetch; the file is already on disk."""
        return self.root / remote.path
