#!/usr/bin/env python3
"""Checks for the local filesystem connector: listing, single-file targets, and an end-to-end
CLI run against a local directory (byte-range reads, no HTTP, nothing deleted).

    python tests/test_localfs_connector.py

No network. Exits non-zero on the first failed check.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phys_ai_dataprofiler.connectors.localfs import LocalFsConnector  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="localfs_test_"))


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def test_list_files() -> None:
    root = TMP / "tree"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.mcap").write_bytes(b"x")
    (root / "b.mat").write_bytes(b"y")
    (root / ".hidden.mcap").write_bytes(b"z")

    c = LocalFsConnector(root)
    c.connect()
    files = sorted(c.list_files(), key=lambda f: f.path)
    check([f.path for f in files] == ["a/x.mcap", "b.mat"], f"hidden files skipped, posix paths: {[f.path for f in files]}")
    check(all(f.sha is None and f.size is not None for f in files), "size known, no sha for local files")


def test_connect_errors_and_single_file() -> None:
    missing = LocalFsConnector(TMP / "does_not_exist")
    try:
        missing.connect()
        check(False, "connect() on a missing path should raise")
    except RuntimeError as exc:
        check("does not exist" in str(exc), f"missing path: clear error: {exc}")

    single = TMP / "single.mcap"
    single.write_bytes(b"x")
    c = LocalFsConnector(single)
    user = c.connect()
    check(user is None, "connect() returns None (no 'authenticated as' concept)")
    files = c.list_files()
    check([f.path for f in files] == ["single.mcap"], "single-file target lists just that file")
    rf = c.open_range(files[0], block_size=64 * 1024)
    check(rf.read_range(0, 1) == b"x", "open_range reads the single file")
    rf.close()


def cli(*args, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "phys_ai_dataprofiler", *args], capture_output=True, text=True,
                          cwd=cwd or TMP, env={**os.environ, "PYTHONPATH": str(ROOT)})


def test_cli_end_to_end() -> None:
    data_dir = TMP / "dataset"
    data_dir.mkdir()
    shutil.copy(ROOT / "samples" / "sample.mcap", data_dir / "sample.mcap")
    shutil.copy(ROOT / "samples" / "sample_v5.mat", data_dir / "sample_v5.mat")
    report_dir = TMP / "reports"

    r = cli(str(data_dir), "--ext", ".mcap", ".mat", "--report-dir", str(report_dir), "--report-name", "local_run")
    check(r.returncode == 0, f"local directory profile exits 0: {r.stderr}")
    check((data_dir / "sample.mcap").exists() and (data_dir / "sample_v5.mat").exists(),
          "source files are untouched")

    rep = json.loads((report_dir / "local_run.json").read_text())
    check(rep["repo_type"] == "local", f"report records a local run: {rep['repo_type']}")
    check(rep["mode"] == "local", f"mode is 'local': {rep['mode']}")
    check(len(rep["files"]) == 2 and all(f["valid"] for f in rep["files"]), f"both files profiled and valid: {rep['files']}")
    check(all(f["bytes_fetched"] is not None and f["bytes_fetched"] <= f["size_bytes"] for f in rep["files"]),
          "byte-range reads recorded, not a full-file read beyond the file's own size")
    md = (report_dir / "local_run.md").read_text()
    check(str(data_dir) in md, "markdown report names the local path")
    check("Authenticated as" not in md and "Revision:" not in md,
          "Hub-only fields are omitted for a local run")


def test_single_file_target() -> None:
    f = TMP / "lone.mcap"
    shutil.copy(ROOT / "samples" / "sample.mcap", f)
    r = cli(str(f), "--report-dir", str(TMP / "reports2"))
    check(r.returncode == 0, f"single local file profile exits 0: {r.stderr}")


def test_delete_after_does_not_delete_source() -> None:
    data_dir = TMP / "dataset_delete"
    data_dir.mkdir()
    shutil.copy(ROOT / "samples" / "sample.mcap", data_dir / "sample.mcap")
    r = cli(str(data_dir), "--ext", ".mcap", "--delete-after", "--report-dir", str(TMP / "reports3"))
    check(r.returncode == 0, f"run with --delete-after exits 0: {r.stderr}")
    check((data_dir / "sample.mcap").exists(), "--delete-after never deletes the original local file")


def test_requirements_check_against_local_dir() -> None:
    data_dir = TMP / "dataset_req"
    data_dir.mkdir()
    shutil.copy(ROOT / "samples" / "sample.mcap", data_dir / "sample.mcap")
    req = TMP / "req.yaml"
    req.write_text("streams:\n  - name: any\n    match: {topic: '*'}\n    required: false\n")
    r = cli("check", str(data_dir), "-r", str(req), "--report-dir", str(TMP / "reports4"))
    check(r.returncode in (0, 1, 4), f"check against a local dir runs end to end: {r.returncode} {r.stderr}")


def main() -> None:
    test_list_files()
    test_connect_errors_and_single_file()
    test_cli_end_to_end()
    test_single_file_target()
    test_delete_after_does_not_delete_source()
    test_requirements_check_against_local_dir()
    print("all checks passed")


if __name__ == "__main__":
    main()
