"""
S3 connector: feeds objects under an S3 bucket/prefix to the inspectors.

Uses the same byte-range inspection path as the other connectors, through `S3RangeFile`
(`GetObject` with a `Range` header), so large objects (multi-GB `.mcap` / `.mat`) are profiled from
their headers and indexes rather than downloaded in full.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from phys_ai_dataprofiler.connectors.base import RemoteFile
from phys_ai_dataprofiler.remote.range_file import S3RangeFile

try:
    import boto3
    import botocore.exceptions
    BOTO3_AVAILABLE = True
except ImportError:  # pragma: no cover - reported at runtime
    BOTO3_AVAILABLE = False


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an `s3://bucket/prefix` URI into (bucket, prefix). `prefix` has no leading slash and,
    if non-empty, a trailing slash (so it can be prepended straight to a relative key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri}")
    rest = uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"s3:// URI has no bucket: {uri}")
    prefix = prefix.strip("/")
    if prefix:
        prefix += "/"
    return bucket, prefix


class S3Connector:
    """Connects to an S3 bucket (optionally scoped to a prefix) and feeds its objects to the inspectors."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        profile: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        log=None,
    ) -> None:
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 is not installed. Run: pip install boto3")
        self.bucket = bucket
        self.prefix = prefix
        self.log = log or (lambda msg: None)
        self.video_info: dict[str, dict] = {}
        self.resolved_sha: str | None = None
        self._session = boto3.Session(profile_name=profile)
        self.client = self._session.client("s3", region_name=region, endpoint_url=endpoint_url)

    # -- connection ---------------------------------------------------------

    def connect(self) -> str | None:
        """Validate the bucket/prefix is reachable. Returns the caller identity's ARN, if resolvable."""
        try:
            self.client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix, MaxKeys=1)
        except botocore.exceptions.NoCredentialsError as exc:
            raise RuntimeError(
                "No AWS credentials found. Set --aws-profile, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
                "or run from an environment with an IAM role."
            ) from exc
        except botocore.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchBucket", "404"):
                raise RuntimeError(f"Bucket '{self.bucket}' not found (or not visible with these credentials)") from exc
            if code in ("AccessDenied", "403"):
                raise RuntimeError(f"Access denied to bucket '{self.bucket}' with these credentials") from exc
            raise RuntimeError(f"Could not list s3://{self.bucket}/{self.prefix}: {exc}") from exc

        user = None
        try:
            sts = self._session.client("sts")
            user = sts.get_caller_identity().get("Arn")
        except Exception:
            pass
        self.log(f"Authenticated to AWS as: {user}" if user else "Using default AWS credential resolution")
        self.log(f"s3://{self.bucket}/{self.prefix}")
        return user

    # -- listing --------------------------------------------------------------

    def list_files(self) -> list[RemoteFile]:
        """List every object under the bucket/prefix, with its size."""
        files: list[RemoteFile] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") and obj.get("Size", 0) == 0:
                    continue  # a "directory marker" object, not a real file
                files.append(RemoteFile(
                    path=key[len(self.prefix):], size=obj.get("Size"),
                    sha=(obj.get("ETag") or "").strip('"') or None,
                ))
        return files

    # -- access -------------------------------------------------------------

    def open_range(self, remote: RemoteFile, block_size: int) -> S3RangeFile:
        """Open an object under the bucket/prefix for byte-range reads (nothing is downloaded)."""
        return S3RangeFile(self.client, self.bucket, self.prefix + remote.path, block_size=block_size)

    def read_text(self, path: str, max_bytes: int = 4 << 20) -> str:
        """Read a small object (e.g. a rosbag2 metadata.yaml) without downloading it to disk."""
        rf = S3RangeFile(self.client, self.bucket, self.prefix + path)
        try:
            if rf.size > max_bytes:
                raise RuntimeError(f"{path} is {rf.size} B, over the {max_bytes} B cap")
            return rf.read_range(0, rf.size).decode("utf-8", "replace")
        finally:
            rf.close()

    def load_video_info(self, files: Iterable[RemoteFile]) -> dict[str, dict]:
        from phys_ai_dataprofiler.connectors.base import load_video_info
        return load_video_info(self, files)

    def download(self, remote: RemoteFile, local_dir: Path, force: bool = False) -> Path:
        """Download one object into local_dir, preserving its path relative to the prefix."""
        dest = local_dir / remote.path
        if force or not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, self.prefix + remote.path, str(dest))
        return dest
