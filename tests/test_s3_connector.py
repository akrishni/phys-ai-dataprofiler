#!/usr/bin/env python3
"""Checks for the S3 connector: URI parsing, listing, byte-range reads, and an end-to-end CLI run
against a moto-mocked bucket (no real AWS account or network access needed).

    python tests/test_s3_connector.py

Exits non-zero on the first failed check. Reported as failed, not skipped, if `boto3`/`moto` are
missing (same convention as the other optional-dependency tests).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import boto3  # noqa: E402
from moto import mock_aws  # noqa: E402

from phys_ai_dataprofiler.connectors.s3 import S3Connector, parse_s3_uri  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="s3_test_"))
BUCKET = "test-dataprofiler-bucket"
REGION = "us-east-1"


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def test_parse_s3_uri() -> None:
    cases = [
        ("s3://bucket/a/b/", ("bucket", "a/b/")),
        ("s3://bucket/a/b", ("bucket", "a/b/")),
        ("s3://bucket", ("bucket", "")),
        ("s3://bucket/", ("bucket", "")),
        ("s3://bucket//a//", ("bucket", "a/")),
    ]
    for uri, want in cases:
        got = parse_s3_uri(uri)
        check(got == want, f"parse_s3_uri({uri!r}) == {want!r}, got {got!r}")
    for bad in ("not-s3://x", "s3://"):
        try:
            parse_s3_uri(bad)
            check(False, f"parse_s3_uri({bad!r}) should raise")
        except ValueError:
            check(True, f"parse_s3_uri({bad!r}) raises ValueError")


def _put_sample_files(client, prefix: str) -> None:
    client.put_object(Bucket=BUCKET, Key=prefix, Body=b"")  # a "directory marker" object
    client.upload_file(str(ROOT / "samples" / "sample.mcap"), BUCKET, prefix + "sample.mcap")
    client.upload_file(str(ROOT / "samples" / "sample_v5.mat"), BUCKET, prefix + "sample_v5.mat")


@mock_aws
def test_list_and_connect() -> None:
    client = boto3.client("s3", region_name=REGION)
    client.create_bucket(Bucket=BUCKET)
    _put_sample_files(client, "episodes/")

    c = S3Connector(BUCKET, "episodes/", region=REGION)
    user = c.connect()
    check(user is not None, f"connect() resolves a caller identity under moto: {user}")

    files = sorted(c.list_files(), key=lambda f: f.path)
    check([f.path for f in files] == ["sample.mcap", "sample_v5.mat"],
          f"directory-marker object skipped, paths relative to prefix: {[f.path for f in files]}")
    check(all(f.sha is not None and f.size is not None for f in files), "size and ETag-derived sha present")

    missing = S3Connector("this-bucket-does-not-exist-12345", region=REGION)
    try:
        missing.connect()
        check(False, "connect() on a missing bucket should raise")
    except RuntimeError as exc:
        check("not found" in str(exc), f"missing bucket: clear error: {exc}")


@mock_aws
def test_open_range_matches_full_read() -> None:
    client = boto3.client("s3", region_name=REGION)
    client.create_bucket(Bucket=BUCKET)
    _put_sample_files(client, "run1/")

    c = S3Connector(BUCKET, "run1/", region=REGION)
    c.connect()
    files = {f.path: f for f in c.list_files()}
    full = (ROOT / "samples" / "sample.mcap").read_bytes()

    rf = c.open_range(files["sample.mcap"], block_size=64 * 1024)
    check(rf.size == len(full), f"S3RangeFile reports the right size: {rf.size} vs {len(full)}")
    check(rf.read_range(0, rf.size) == full, "byte-range read over S3 matches a full local read")
    mid = len(full) // 2
    check(rf.read_range(mid, mid + 10) == full[mid:mid + 10], "an arbitrary sub-range matches too")
    rf.close()


@mock_aws
def test_cli_end_to_end() -> None:
    client = boto3.client("s3", region_name=REGION)
    client.create_bucket(Bucket=BUCKET)
    _put_sample_files(client, "ep/")
    report_dir = TMP / "reports_s3"

    # moto intercepts boto3 calls process-wide via env vars it sets up; the CLI subprocess needs the
    # same moto mock active, so run the CLI in-process instead of as a real subprocess for this test.
    from phys_ai_dataprofiler.connectors.huggingface import build_parser, run, write_reports
    args = build_parser().parse_args([
        f"s3://{BUCKET}/ep/", "--ext", ".mcap", ".mat", "--aws-region", REGION,
        "--report-dir", str(report_dir), "--report-name", "s3_run", "--quiet",
    ])
    run_report = run(args)
    check(run_report.totals["invalid_or_errored"] == 0, f"both files profiled without error: {run_report.totals}")
    check(run_report.repo_type == "s3" and run_report.repo_id == f"s3://{BUCKET}/ep/",
          f"report records the s3 target: {run_report.repo_type} {run_report.repo_id}")
    check(run_report.mode == "remote", f"default mode is remote (byte-range): {run_report.mode}")
    check(all(f.bytes_fetched is not None and f.bytes_fetched <= (f.size_bytes or 0) for f in run_report.files),
          "byte-range reads recorded, not a full download beyond the object's own size")

    write_reports(run_report, report_dir, "s3_run", lambda m: None)
    rep = json.loads((report_dir / "s3_run.json").read_text())
    check(len(rep["files"]) == 2, f"two files in the written report: {rep['files']}")


@mock_aws
def test_download_mode() -> None:
    client = boto3.client("s3", region_name=REGION)
    client.create_bucket(Bucket=BUCKET)
    _put_sample_files(client, "dl/")
    download_dir = TMP / "s3_downloads"

    from phys_ai_dataprofiler.connectors.huggingface import build_parser, run
    args = build_parser().parse_args([
        f"s3://{BUCKET}/dl/", "--ext", ".mcap", "--mode", "download", "--aws-region", REGION,
        "--download-dir", str(download_dir), "--report-dir", str(TMP / "reports_s3_dl"), "--quiet",
    ])
    run_report = run(args)
    check(run_report.mode == "download", f"mode is download: {run_report.mode}")
    check(run_report.files_downloaded == 1, f"one file downloaded: {run_report.files_downloaded}")
    check(run_report.totals["invalid_or_errored"] == 0, f"downloaded file profiled without error: {run_report.totals}")


def main() -> None:
    test_parse_s3_uri()
    test_list_and_connect()
    test_open_range_matches_full_read()
    test_cli_end_to_end()
    test_download_mode()
    print("all checks passed")


if __name__ == "__main__":
    main()
