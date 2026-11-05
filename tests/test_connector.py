#!/usr/bin/env python3
"""Checks for the Hub connector's offline pieces: file selection, format sniffing, JSON conversion,
totals, and the CLI's error paths for bad inputs.

    python tests/test_connector.py

No network. Exits non-zero on the first failed check.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phys_ai_dataprofiler.connectors.huggingface import (  # noqa: E402
    FileReport, HuggingFaceConnector, RemoteFile, _human_size, _sidecar_path, _sniff_local_format, _to_jsonable,
    build_totals, inspect_local_file)
from phys_ai_dataprofiler.remote.inspector import sniff_format  # noqa: E402
from phys_ai_dataprofiler.remote.range_file import LocalRangeFile  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="connector_test_"))


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def reasons(skipped) -> dict[str, str]:
    return {s["path"]: s["reason"] for s in skipped}


def test_select_files() -> None:
    files = [RemoteFile("b.mcap", 100, None), RemoteFile("a.MCAP", 50, None), RemoteFile("c.txt", 1, None),
             RemoteFile("d/e.mat", 10**9, None), RemoteFile("f.parquet", None, None), RemoteFile("g.mcap", 60, None)]
    sel, skipped = HuggingFaceConnector.select_files(files, extensions=["mcap", ".mat", ".parquet"])
    check([f.path for f in sel] == ["a.MCAP", "b.mcap", "d/e.mat", "f.parquet", "g.mcap"] and not skipped,
          "extensions match case-insensitively, with or without the dot; output sorted by path")

    sel, skipped = HuggingFaceConnector.select_files(files, extensions=["mcap", "mat", "parquet"],
                                                     max_file_bytes=10**6, max_total_bytes=200, max_files=3)
    check([f.path for f in sel] == ["a.MCAP", "b.mcap", "f.parquet"], f"limits applied in path order: {[f.path for f in sel]}")
    check(reasons(skipped) == {"d/e.mat": "exceeds --max-file-mb", "g.mcap": "would exceed --max-total-mb"},
          "each skipped file says which limit it hit")
    check(any(f.path == "f.parquet" for f in sel), "a file with unknown size is never rejected by a size limit")

    _, skipped = HuggingFaceConnector.select_files(files, extensions=["mcap"], max_files=1)
    check(reasons(skipped) == {"b.mcap": "exceeds --max-files", "g.mcap": "exceeds --max-files"}, "max_files")

    sel, skipped = HuggingFaceConnector.select_files(files, extensions=["mcap"], include=["*.mcap"], exclude=["g*"])
    check([f.path for f in sel] == ["b.mcap"] and reasons(skipped) == {"a.MCAP": "not matched by --include",
                                                                         "g.mcap": "matched --exclude"},
          "include globs are case-sensitive; exclude wins over include")

    sel, skipped = HuggingFaceConnector.select_files(files, explicit=["c.txt", "zzz.mcap"])
    check([f.path for f in sel] == ["c.txt"] and reasons(skipped) == {"zzz.mcap": "not found in repo"},
          "explicit files ignore extensions; missing ones are reported")
    sel, skipped = HuggingFaceConnector.select_files(files, explicit=["d/e.mat"], max_file_bytes=10)
    check(not sel and reasons(skipped) == {"d/e.mat": "exceeds --max-file-mb"},
          "size limits still apply to explicit files, and they are not also reported missing")
    sel, skipped = HuggingFaceConnector.select_files([], extensions=["mcap"])
    check(sel == [] and skipped == [], "empty repo: nothing selected, nothing skipped")


def test_sniff() -> None:
    junk = TMP / "junk.bin"
    junk.write_bytes(np.random.default_rng(0).bytes(500))
    rf = LocalRangeFile(junk)
    got = {n: sniff_format(rf, n) for n in ("a.mcap", "a.MAT", "a.parquet", "a.bag", "a.db3", "a", "")}
    check(got == {"a.mcap": "mcap", "a.MAT": "mat", "a.parquet": "parquet", "a.bag": "bag1", "a.db3": "unknown",
                  "a": "unknown", "": "unknown"}, f"unknown bytes fall back to the extension: {got}")
    mat = TMP / "renamed.mcap"
    scipy.io.savemat(mat, {"x": np.ones(3)})
    check(_sniff_local_format(mat) == "mat", "magic bytes beat a misleading extension")
    empty = TMP / "empty.parquet"
    empty.write_bytes(b"")
    check(_sniff_local_format(empty) == "parquet", "empty file: extension only")
    check(_sniff_local_format(TMP / "missing.mcap") == "unknown", "missing file: unknown, not an exception")

    rep = inspect_local_file("junk.bin", junk, None)
    check(not rep.valid and "Not recognised" in (rep.error or ""), "unrecognised local file: error in the report")
    bad = TMP / "bad.parquet"
    bad.write_bytes(b"PAR1" + b"\0" * 64)
    rep = inspect_local_file("bad.parquet", bad, None)
    check(not rep.valid, "corrupt Parquet in download mode: not valid, no exception")


def test_jsonable() -> None:
    @dataclass
    class D:
        blob: bytes
        items: tuple

    out = _to_jsonable({1: D(b"\x00abc" * 200, (np.float32(1.5), np.int64(3), np.arange(3), Path("/a"), {"s"})),
                        "hdr": "MATLAB 5.0 MAT-file\x00\x00   "})
    check(json.loads(json.dumps(out)) == out, "result is plain JSON")
    check(set(out) == {"1", "hdr"}, "dict keys become strings")
    blob = out["1"]["blob"]
    check(blob["bytes"] == 800 and blob["hex_prefix"] == "00616263" * 4 and len(blob["preview"]) == 515,
          "bytes become a bounded preview with length and hex prefix")
    check(out["1"]["items"] == [1.5, 3, [0, 1, 2], "/a", ["s"]], "numpy scalars/arrays, Path, set and tuple converted")
    check(out["hdr"] == "MATLAB 5.0 MAT-file", "NUL padding and trailing spaces stripped from strings")
    deep: list = []
    cur = deep
    for _ in range(30):
        cur.append([])
        cur = cur[0]
    node, depth = _to_jsonable(deep), 0
    while isinstance(node, list):
        node, depth = node[0], depth + 1
    check(depth == 21 and isinstance(node, str), "deeply nested input is cut off at depth 20 (rest as repr)")
    check(_to_jsonable(np.bool_(True)) is True and _to_jsonable(object()).startswith("<object"),
          "numpy bool becomes a Python bool; unknown objects fall back to repr")


def test_small_helpers() -> None:
    check([_human_size(n) for n in (None, 0, 1023, 1024, 1536 * 1024, 5 * 1024**5)]
          == ["-", "0 B", "1023 B", "1.0 KB", "1.5 MB", "5120.0 TB"], "human sizes, capped at TB")
    check(_sidecar_path("a/b/c.db3") == "a/b/metadata.yaml" and _sidecar_path("c.db3") == "metadata.yaml",
          "rosbag2 sidecar path, nested and at the repo root")

    def fr(fmt, valid, mode, **summary):
        r = FileReport(repo_path="x", local_path=None, size_bytes=100, detected_format=fmt, valid=valid, mode=mode)
        r.bytes_fetched, r.requests, r.summary = 10, 2, summary
        return r

    t = build_totals([fr("mcap", True, "remote", duration_s=1.5, messages=10),
                      fr(None, False, "download", duration_s=None),
                      fr("mcap", True, "remote", findings=2, warnings=1)])
    check(t["files"] == 3 and t["valid"] == 2 and t["invalid_or_errored"] == 1, "totals count valid and invalid files")
    check(t["by_format"] == {"mcap": 2, "unknown": 1} and t["by_mode"] == {"remote": 2, "download": 1},
          "totals by format (None counted as unknown) and by mode")
    check(t["duration_s"] == 1.5 and t["messages"] == 10 and t["fetched_pct"] == 10.0,
          "missing or None summary fields count as zero")
    check(build_totals([])["fetched_pct"] is None, "no files: fetched_pct is None, not a division by zero")


def cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "phys_ai_dataprofiler", *args], capture_output=True, text=True,
                          cwd=TMP, env={**os.environ, "PYTHONPATH": str(ROOT)})


def test_cli_errors() -> None:
    (TMP / "r.yaml").write_text("streams:\n  - name: any\n    match: {topic: '*'}\n")
    (TMP / "not_report.json").write_text('{"x": 1}')
    (TMP / "broken.json").write_text("{bad")
    for args, want in [
        (("check", "--report", "not_report.json", "-r", "r.yaml"), "not a profiler run report"),
        (("check", "--report=not_report.json", "-r", "r.yaml"), "not a profiler run report"),
        (("check", "--report", "broken.json", "-r", "r.yaml"), "cannot read report"),
        (("check", "--report", "missing.json", "-r", "r.yaml"), "cannot read report"),
        (("check", "--report", "not_report.json", "-r", "missing.yaml"), "file not found"),
        (("requirements", "show", "nope"), "no preset named 'nope'"),
    ]:
        r = cli(*args)
        check(r.returncode == 2 and want in r.stderr and "Traceback" not in r.stderr,
              f"{' '.join(args)}: exit 2, '{want}'")
    r = cli("requirements", "show", "preset:visual-odometry")
    check(r.returncode == 0 and "name: visual-odometry" in r.stdout, "requirements show accepts the preset: prefix")
    empty_rep = TMP / "empty_report.json"
    empty_rep.write_text(json.dumps({"repo_id": "x", "files": []}))
    r = cli("check", "--report", str(empty_rep), "-r", "r.yaml")
    check(r.returncode == 4, "a report with no files gives unknown (exit 4)")


def main() -> None:
    test_select_files()
    test_sniff()
    test_jsonable()
    test_small_helpers()
    test_cli_errors()
    print("all checks passed")


if __name__ == "__main__":
    main()
