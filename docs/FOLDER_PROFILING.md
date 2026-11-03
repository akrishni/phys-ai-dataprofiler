# Folder Profiling

Rich folder profiling for MCAP and MATLAB files with detailed channel statistics, timing analysis, and data quality findings.

## Features

### Rich Inspection
- **Channel-level Statistics**: Message counts, rates, jitter, gaps
- **Timing Analysis**: Duration, timestamps, offsets between channels
- **Data Quality**: Dropped frames, NaN detection, gap analysis
- **Modality Detection**: Auto-detect video, proprio, IMU, GPS, etc.
- **Format Details**: Codec, resolution, dimensions

### Output Format
Structured JSON with:
- File metadata (path, size, duration, indexed)
- Per-channel statistics (topic, schema, modality, msg_count, rate_hz, jitter_ms, gaps)
- Data quality findings (warnings, errors, info)
- Inspection cost timing

## Usage

### Basic Usage

```bash
# Profile a folder
python scripts/profile_folder_rich.py /path/to/data

# Profile current directory
python scripts/profile_folder_rich.py .

# Only MCAP files
python scripts/profile_folder_rich.py /path/to/data --ext .mcap

# Save to file
python scripts/profile_folder_rich.py /path/to/data --output profiles.json

# Pretty print
python scripts/profile_folder_rich.py /path/to/data --pretty
```

### Command Options

```
--no-recursive      Don't search subdirectories
--ext .mcap         Filter by extension (.mcap or .mat)
--output FILE       Save JSON output to file
--pretty            Pretty-print JSON output
--no-summary        Skip summary table (JSON only)
```

## Output Example

```json
{
  "path": "episode_0042/episode.mcap",
  "format": "mcap",
  "size_bytes": 812000000,
  "indexed": true,
  "duration_s": 94.2,
  "time_base": "absolute_epoch_ns",
  "channels": [
    {
      "topic": "/cam/wrist",
      "schema": "foxglove.CompressedVideo",
      "modality": "video",
      "codec": "h264",
      "resolution": [640, 480],
      "msg_count": 2826,
      "rate_hz": 30.0,
      "jitter_ms": 1.8,
      "gaps_over_3x": 4,
      "max_gap_ms": 210
    },
    {
      "topic": "/joint_states",
      "schema": "sensor_msgs/JointState",
      "modality": "proprio",
      "dims": 7,
      "msg_count": 9420,
      "rate_hz": 100.0,
      "nan_count": 0
    }
  ],
  "findings": [
    {
      "severity": "warn",
      "type": "dropped_frames",
      "topic": "/cam/wrist",
      "detail": "4 gaps > 100 ms"
    },
    {
      "severity": "info",
      "type": "start_offset",
      "detail": "/cam/wrist starts 1.4 s after /joint_states"
    }
  ],
  "inspect_cost_s": 0.9
}
```

## Summary Table

When running with default settings, a summary table is printed to stderr:

```
====================================================================
FOLDER PROFILE SUMMARY
====================================================================

Files: 15
Total Size: 0.81 GB
Total Duration: 142.5 s
Total Channels: 285
Total Messages: 125,430

⚠️  Warnings: 12

File                             Size   Duration   Channels   Msgs      Findings
--------------------------------------------------------------------------------
episode_0001.mcap              54.2M      94.2s         19      23,542         3
episode_0002.mcap              61.8M     102.8s         19      25,680         2
...
====================================================================
```

## Detected Modalities

The inspector automatically detects data modalities:

| Modality | Detection Criteria | Examples |
|----------|-------------------|----------|
| `video` | Schema contains "video", "image", "compressed"; topic contains "cam" | /cam/wrist, foxglove.CompressedVideo |
| `proprio` | Topic/schema contains "joint", "state" | /joint_states, /joint_effort |
| `imu` | Topic contains "imu", "odom" | /imu/data, /odometry |
| `lidar` | Topic contains "lidar", schema contains "scan" | /scan, /velodyne |
| `gps` | Topic contains "gps" | /gps/fix, /gps/position |
| `timestamps` | Variable name contains "time", "stamp" | timestamps, time_vec |
| `timeseries` | 2D arrays with multiple columns | sensor_data (100x3) |
| `other` | Everything else | /instruction, /metadata |

## Data Quality Findings

### Finding Types

**Errors** (severity: `error`):
- `missing_library`: Required parsing library not available
- `unknown_format`: File format not recognized

**Warnings** (severity: `warn`):
- `dropped_frames`: Large gaps detected (>3x expected interval)
- `nan_values`: NaN values found in numeric data

**Info** (severity: `info`):
- `start_offset`: Channels start at different times

### Gap Detection

For each channel with > 1 message:
- Calculate average message interval
- Detect gaps > 3× average interval
- Report if max gap > 100ms and count > 0

### Timing Analysis

- Duration: `(last_timestamp - first_timestamp) / 1e9`
- Rate: `1000.0 / avg_interval_ms`
- Jitter: Standard deviation of intervals

## Integration Examples

### Python

```python
from tools.rich_inspector import RichInspector

inspector = RichInspector("data.mcap")
profile = inspector.inspect()

# Access results
print(f"Duration: {profile.duration_s}s")
print(f"Channels: {len(profile.channels)}")

for channel in profile.channels:
    print(f"{channel.topic}: {channel.msg_count} messages @ {channel.rate_hz}Hz")

for finding in profile.findings:
    print(f"[{finding.severity}] {finding.detail}")
```

### Batch Processing

```python
from pathlib import Path
from tools.rich_inspector import RichInspector
import json

profiles = []
for mcap_file in Path("data").glob("*.mcap"):
    profile = RichInspector(str(mcap_file)).inspect()
    profiles.append(profile)

# Save all profiles
with open("profiles.json", "w") as f:
    json.dump([format_profile(p) for p in profiles], f, indent=2)
```

### Filtering

```bash
# Find files with dropped frames
python scripts/profile_folder_rich.py data/ --pretty | \
  jq '.[] | select(.findings[] | select(.type == "dropped_frames"))'

# Get total message count
python scripts/profile_folder_rich.py data/ --pretty | \
  jq '[.[] | .channels[] | .msg_count] | add'

# List video topics
python scripts/profile_folder_rich.py data/ --pretty | \
  jq '.[] | .channels[] | select(.modality == "video") | .topic'
```

## Performance

- **Inspection Cost**: Shown in `inspect_cost_s` field
- **Memory**: Metadata only, messages not loaded
- **Large Files**: Efficiently processes multi-GB files
- **Parallel**: Use `xargs` for parallel folder processing

Example parallel processing:
```bash
find data/ -name "*.mcap" | \
  xargs -P 4 -I {} python -m phys_ai_dataprofiler.inspectors.rich {} > {}.json
```

## Comparison with Basic Inspectors

| Feature | Basic Inspector | Rich Inspector |
|---------|----------------|----------------|
| Header/Footer | ✓ | ✓ |
| Channel List | ✓ | ✓ |
| Message Counts | ✗ | ✓ |
| Rate/Jitter | ✗ | ✓ |
| Gap Detection | ✗ | ✓ |
| NaN Detection | ✗ | ✓ |
| Modality Detection | ✗ | ✓ |
| Quality Findings | ✗ | ✓ |
| Timing Analysis | ✗ | ✓ |
| Folder Profiling | Basic | Rich JSON |

## Limitations

- **Video Codec**: Currently hardcoded to h264, needs parsing
- **Resolution**: Fixed [640, 480], needs message parsing
- **Dimensions**: Limited detection from schema hints
- **NaN Detection**: Only for MATLAB files currently
- **Message Data**: Not parsed, statistics from metadata only

## Future Enhancements

- [ ] Parse video codec/resolution from message data
- [ ] Deep message inspection for value ranges
- [ ] Timestamp synchronization analysis
- [ ] Compression ratio statistics
- [ ] Schema validation
- [ ] Interactive visualization
- [ ] Database export (SQLite, PostgreSQL)
- [ ] Real-time monitoring mode
