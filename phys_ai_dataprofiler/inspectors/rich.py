"""
Rich file inspector with detailed profiling and quality analysis.

Provides deep inspection including:
- Timing and rate analysis
- Data quality checks (gaps, NaNs, dropped frames)
- Modality detection
- Statistical summaries
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

try:
    from mcap.reader import make_reader
    MCAP_AVAILABLE = True
except ImportError:
    MCAP_AVAILABLE = False

try:
    import scipy.io
    import scipy.sparse
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class ChannelProfile:
    topic: str
    schema: str
    modality: str | None
    codec: str | None
    resolution: List[int] | None
    dims: int | None
    msg_count: int
    rate_hz: float | None
    jitter_ms: float | None
    gaps_over_3x: int
    max_gap_ms: float | None
    nan_count: int = 0


@dataclass
class Finding:
    severity: str  # "error", "warn", "info"
    type: str
    topic: str | None
    detail: str


@dataclass
class RichProfile:
    path: str
    format: str
    size_bytes: int
    indexed: bool
    duration_s: float | None
    time_base: str | None
    channels: List[ChannelProfile]
    findings: List[Finding]
    inspect_cost_s: float


def timing_stats(segments: List[List[int]]) -> tuple:
    """Rate, jitter and gap statistics from runs of nanosecond timestamps.

    Each run is a sorted, contiguous stretch of a recording. Intervals are only
    taken inside a run, so the jump between runs (e.g. sampled chunks) is not
    counted as a gap; a full recording is a single run.

    Returns (rate_hz, jitter_ms, gaps_over_3x, max_gap_ms); all None/0 when
    there are no intervals.
    """
    import numpy as np

    diffs = [np.diff(np.asarray(seg, dtype=np.int64)) for seg in segments if len(seg) > 1]
    if not diffs:
        return None, None, 0, None
    intervals = np.concatenate(diffs) / 1e6
    avg_interval = float(intervals.mean())
    rate_hz = 1000.0 / avg_interval if avg_interval > 0 else 0
    jitter_ms = float(intervals.std())
    gaps_over_3x = int(np.count_nonzero(intervals > 3 * avg_interval))
    return rate_hz, jitter_ms, gaps_over_3x, float(intervals.max())


def mat_channel_profile(name: str, shape: tuple, dtype: str, nan_count: int) -> ChannelProfile:
    """ChannelProfile for one MATLAB variable; shared by the local and remote MAT inspectors."""
    return ChannelProfile(
        topic=name,
        schema=dtype,
        modality=RichInspector._detect_mat_modality(name, shape, dtype),
        codec=None,
        resolution=list(shape) if len(shape) <= 2 else None,
        dims=shape[1] if len(shape) == 2 else None,
        msg_count=shape[0] if len(shape) > 0 else 1,
        rate_hz=None,
        jitter_ms=None,
        gaps_over_3x=0,
        max_gap_ms=None,
        nan_count=nan_count,
    )


def profile_mcap_channels(channel_stats: Dict[int, dict]) -> tuple:
    """Build ChannelProfiles and Findings from per-channel timestamp lists.

    channel_stats maps channel id -> {"topic", "schema", "segments", "count"},
    where "segments" is a list of sorted runs of timestamps (see timing_stats).
    Shared by the local (full read) and remote (message-index) inspectors so
    both produce identical statistics for the same timestamps.
    """
    channels: List[ChannelProfile] = []
    findings: List[Finding] = []
    channel_start_times: Dict[str, int] = {}

    for channel_id, stats in channel_stats.items():
        topic = stats["topic"]
        schema = stats["schema"]
        segments = [seg for seg in stats["segments"] if seg]
        msg_count = stats["count"]

        if segments:
            channel_start_times[topic] = min(seg[0] for seg in segments)

        rate_hz, jitter_ms, gaps_over_3x, max_gap_ms = timing_stats(segments)

        modality = RichInspector._detect_modality(schema, topic)

        # Detect codec and resolution for video
        codec = None
        resolution = None
        if "video" in modality.lower() or "compressed" in schema.lower():
            codec = "h264"  # Could parse from schema
            resolution = [640, 480]  # Would need to parse from messages

        # Detect dimensions for arrays
        dims = None
        if "joint" in topic.lower() or "state" in schema.lower():
            dims = 7  # Would need to parse from messages

        channels.append(ChannelProfile(
            topic=topic,
            schema=schema,
            modality=modality,
            codec=codec,
            resolution=resolution,
            dims=dims,
            msg_count=msg_count,
            rate_hz=rate_hz,
            jitter_ms=jitter_ms,
            gaps_over_3x=gaps_over_3x,
            max_gap_ms=max_gap_ms,
            nan_count=0,  # Would need to parse message data
        ))

        if gaps_over_3x > 0 and max_gap_ms and max_gap_ms > 100:
            findings.append(Finding(
                severity="warn",
                type="dropped_frames",
                topic=topic,
                detail=f"{gaps_over_3x} gaps > 100 ms"
            ))

    # Check for timing offsets between channels
    if len(channel_start_times) > 1:
        topics = sorted(channel_start_times.keys(), key=lambda t: channel_start_times[t])
        first_topic = topics[0]
        first_time = channel_start_times[first_topic]

        for topic in topics[1:]:
            offset_s = (channel_start_times[topic] - first_time) / 1e9
            if offset_s > 1.0:
                findings.append(Finding(
                    severity="info",
                    type="start_offset",
                    topic=None,
                    detail=f"{topic} starts {offset_s:.1f} s after {first_topic}"
                ))

    return channels, findings


class RichInspector:
    """Deep inspection with quality analysis."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.path = Path(file_path)

    def inspect(self) -> RichProfile:
        """Perform rich inspection of the file."""
        start_time = time.time()

        ext = self.path.suffix.lower()

        if ext == ".mcap":
            profile = self._inspect_mcap()
        elif ext == ".mat":
            profile = self._inspect_mat()
        else:
            # Try to detect format
            profile = self._inspect_unknown()

        profile.inspect_cost_s = time.time() - start_time
        return profile

    def _inspect_mcap(self) -> RichProfile:
        """Rich inspection of MCAP file."""
        if not MCAP_AVAILABLE:
            return RichProfile(
                path=str(self.path),
                format="mcap",
                size_bytes=self.path.stat().st_size,
                indexed=False,
                duration_s=None,
                time_base=None,
                channels=[],
                findings=[Finding("error", "missing_library", None,
                         "mcap library not available")],
                inspect_cost_s=0,
            )

        size_bytes = self.path.stat().st_size
        channels = []
        findings = []

        with open(self.path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()

            # Get basic info
            indexed = summary is not None

            # Analyze messages for timing and quality
            channel_stats = defaultdict(lambda: {
                "timestamps": [],
                "count": 0,
                "schema": "",
                "topic": "",
            })

            first_timestamp = None
            last_timestamp = None

            for schema, channel, message in reader.iter_messages():
                log_time = message.log_time

                if first_timestamp is None:
                    first_timestamp = log_time
                last_timestamp = log_time

                stats = channel_stats[channel.id]
                stats["count"] += 1
                stats["timestamps"].append(log_time)
                stats["topic"] = channel.topic
                stats["schema"] = schema.name if schema else "unknown"

            # Calculate duration
            duration_s = None
            if first_timestamp and last_timestamp:
                duration_s = (last_timestamp - first_timestamp) / 1e9

            # Analyze each channel (shared with the remote inspector)
            for stats in channel_stats.values():
                stats["segments"] = [stats.pop("timestamps")]
            channels, findings = profile_mcap_channels(channel_stats)

        return RichProfile(
            path=str(self.path),
            format="mcap",
            size_bytes=size_bytes,
            indexed=indexed,
            duration_s=duration_s,
            time_base="absolute_epoch_ns",
            channels=channels,
            findings=findings,
            inspect_cost_s=0,
        )

    def _inspect_mat(self) -> RichProfile:
        """Rich inspection of MATLAB file."""
        if not SCIPY_AVAILABLE:
            return RichProfile(
                path=str(self.path),
                format="mat",
                size_bytes=self.path.stat().st_size,
                indexed=False,
                duration_s=None,
                time_base=None,
                channels=[],
                findings=[Finding("error", "missing_library", None,
                         "scipy library not available")],
                inspect_cost_s=0,
            )

        size_bytes = self.path.stat().st_size
        channels = []
        findings = []

        # Load MAT file
        mat_data = scipy.io.loadmat(str(self.path), struct_as_record=False, squeeze_me=False)

        # Analyze variables
        for name, value in mat_data.items():
            if name.startswith("__"):
                continue

            # Extract shape and type info
            if hasattr(value, "shape"):
                shape = value.shape
                dtype = str(value.dtype)

                # Check for NaNs
                # Sparse matrices: only the stored entries can be NaN (np.isnan rejects the matrix itself).
                values = value.data if scipy.sparse.issparse(value) else value
                nan_count = 0
                if np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.complexfloating):
                    nan_count = int(np.count_nonzero(np.isnan(values)))

                channels.append(mat_channel_profile(name, shape, dtype, nan_count))

                # Generate findings
                if nan_count > 0:
                    findings.append(Finding(
                        severity="warn",
                        type="nan_values",
                        topic=name,
                        detail=f"{nan_count} NaN values found"
                    ))

        return RichProfile(
            path=str(self.path),
            format="mat",
            size_bytes=size_bytes,
            indexed=False,
            duration_s=None,
            time_base=None,
            channels=channels,
            findings=findings,
            inspect_cost_s=0,
        )

    def _inspect_unknown(self) -> RichProfile:
        """Handle unknown file types."""
        return RichProfile(
            path=str(self.path),
            format="unknown",
            size_bytes=self.path.stat().st_size if self.path.exists() else 0,
            indexed=False,
            duration_s=None,
            time_base=None,
            channels=[],
            findings=[Finding("error", "unknown_format", None,
                     f"Unknown file format: {self.path.suffix}")],
            inspect_cost_s=0,
        )

    @staticmethod
    def _detect_modality(schema: str, topic: str) -> str:
        """Detect data modality from schema and topic."""
        schema_lower = schema.lower()
        topic_lower = topic.lower()

        if "video" in schema_lower or "image" in schema_lower or "cam" in topic_lower:
            return "video"
        elif "joint" in topic_lower or "joint" in schema_lower:
            return "proprio"
        elif "imu" in topic_lower or "odom" in topic_lower:
            return "imu"
        elif "lidar" in topic_lower or "scan" in schema_lower:
            return "lidar"
        elif "gps" in topic_lower:
            return "gps"
        else:
            return "other"

    @staticmethod
    def _detect_mat_modality(name: str, shape: tuple, dtype: str) -> str:
        """Detect data modality from MATLAB variable."""
        name_lower = name.lower()

        if "image" in name_lower or "img" in name_lower:
            return "image"
        elif "joint" in name_lower or "pos" in name_lower:
            return "proprio"
        elif "time" in name_lower or "stamp" in name_lower:
            return "timestamps"
        elif len(shape) == 2 and shape[1] > 1:
            return "timeseries"
        else:
            return "data"

    def to_json(self, profile: RichProfile) -> str:
        """Convert profile to JSON string."""
        # Convert dataclasses to dicts
        data = {
            "path": profile.path,
            "format": profile.format,
            "size_bytes": profile.size_bytes,
            "indexed": profile.indexed,
            "duration_s": profile.duration_s,
            "time_base": profile.time_base,
            "channels": [asdict(c) for c in profile.channels],
            "findings": [asdict(f) for f in profile.findings],
            "inspect_cost_s": profile.inspect_cost_s,
        }
        return json.dumps(data, indent=2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m phys_ai_dataprofiler.inspectors.rich <file>")
        sys.exit(1)

    inspector = RichInspector(sys.argv[1])
    profile = inspector.inspect()
    print(inspector.to_json(profile))
