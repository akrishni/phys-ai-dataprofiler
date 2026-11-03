# File Inspectors

## Repository layout

```
phys_ai_dataprofiler/            importable package (pip install -e .)
  inspectors/  mcap.py  mat.py  rich.py      full-read inspectors for local files
  remote/      range_file.py  inspector.py   byte-range file objects + metadata-only inspectors
               parquet.py                    Parquet: footer, column stats, LeRobot-style timing
  connectors/  huggingface.py                Hub connector and the `phys-ai-dataprofiler` CLI
scripts/                    demos, sample-file generators, folder profilers
tests/                      runnable checks against the sample files
samples/                    small MCAP / MAT files used by scripts and tests
docs/                       design, test-dataset list, lab-data sources, older notes
```

Large downloaded datasets, generated reports and the virtualenv live outside this project, in
`~/pers_workspace/adhoc/` (`data/`, `reports/`, `venv311/`). The virtualenv runs Python 3.11 and has this
project installed in editable mode with the `h5py` (MAT v7.3) and `rosbags` extras. Activate it with
`source ~/pers_workspace/adhoc/venv311/bin/activate`. To reinstall:
`pip install -e ~/pers_workspace/phys-ai-dataprofiler[mat73,rosbag]`. Use Python 3.13 or older for a new
environment: `h5py` has no wheel for 3.15 yet.

Tools to inspect binary file formats and extract metadata, structure, and content information.

## Available Inspectors

1. **MCAP File Inspector** - Inspect MCAP (Message Capture) files
2. **MATLAB File Inspector** - Inspect MATLAB .mat files

## Requirements and verdicts

Say what your model needs in a YAML file, and the profiler answers pass, fail or unknown for each
requirement, with evidence and, for unknowns, how to settle them. Checks are deterministic rules, with no
model call. Full reference: [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

```bash
phys-ai-dataprofiler check org/dataset -r preset:teleop-imitation        # exit 0 pass, 1 fail, 4 unknown
phys-ai-dataprofiler compare org/a org/b -r my_requirements.yaml          # rank candidate datasets
phys-ai-dataprofiler check --report reports/run.json -r my_requirements.yaml   # re-check offline
```

```yaml
streams:
  - name: camera
    match: {modality: video}
    min_rate_hz: 25
  - name: joint state
    match: {topic: ["/joint_states", "observation.state"]}
    min_rate_hz: 50
    max_gap_ms: 100
dataset:
  min_total_hours: 20
  min_episodes: 500
```

A number that was sampled and lies within 10% of the threshold, a NaN count the format cannot provide,
or a file that failed to profile makes the result **unknown**, never a guessed pass.

## ROS bag support

`.bag` (ROS 1, format V2.0) and `.db3` (ROS 2, SQLite storage) files are profiled from byte ranges, like
MCAP. ROS 2 bags in MCAP storage are plain `.mcap` files and use the MCAP path. Code:
`phys_ai_dataprofiler/remote/rosbag.py`.

- **ROS 1:** the bag header points to the index at the end of the file (connections plus per-chunk counts
  and time ranges). Per-message timestamps come from the IndexData records after each chunk, 12 bytes per
  message, one request per chunk. Chunk payloads are never read, so compression (bz2, lz4) makes no
  difference. Over `--msgidx-budget-mb`, chunks are sampled; counts stay exact. Bags with no index
  (recorder killed) are walked record by record and reported `unindexed_bag`; truncated bags are reported
  `truncated`. Old V1.2 bags are identified but not parsed.
- **ROS 2 SQLite:** a small read-only SQLite reader descends the `messages` table B-tree. Each row's
  `topic_id` and `timestamp` sit at the start of its leaf cell, so message payloads and their overflow
  pages are never read. Counts and duration come from the `metadata` table, or from the `metadata.yaml`
  next to the `.db3` when the database has none. Over budget, runs of leaf pages are sampled and the
  start and end times come from `timestamp_idx`. Without any metadata, sampled per-topic counts are
  estimates, reported as `estimated_counts`.

Timing statistics use the same code as MCAP. Tests compare every case against a full read through
`rosbags`: `python tests/test_rosbag_inspector.py` (add `--hub` for two public bags).

| Dataset | Result | Read |
|---|---|---|
| `Adapting/nuscenes-rosbags` `103.bag` (ROS 1) | 14 topics, 1,585 chunks, 19.5 s | 533 KB of 5.8 GB |
| `dvar1ch/ros2-bag-datasets` (ROS 2 SQLite, 2 files) | `/image` at 30 Hz, metadata v9, jazzy | 1.5 MB of 668 MB |
| `liuxh010/IndoorLocalization_...` (ROS 1), `UniflexAI/rosbag_lekiwi_exp_result` (ROS 2) | per-topic stats identical to a full read | 146 KB of 891 KB; 2.1 MB of 18.6 MB |

## Parquet support

Parquet keeps its schema, row groups and per-column statistics in a footer at the end of the file, so the
profiler reads the footer first (typically under 100 KB, whatever the file size). It reports rows, row groups,
every column's type, null count, min and max, compressed size and codec.

If a time column is present (`timestamp`, `log_time`, `time`, ...), it also reads that column, and
`episode_index` when present (the LeRobot convention). Their byte ranges are known from the footer, so the cost
is computed up front and capped by `--column-budget-mb` (default 16); beyond that, row groups are sampled and
the report says so. From these it derives frame rate, jitter, gaps, episode count and lengths, and flags
dropped frames, short episodes and non-monotonic time. Image, text and state columns are never read.

Units are inferred: float seconds (LeRobot), integer epoch s/ms/us/ns, and Arrow timestamp types.
NaN/inf counts are not in Parquet statistics, so float columns are read too, capped by `--nan-budget-mb`
(default 32); beyond that, row groups are sampled and the report says so. `--nan-budget-mb 0` turns this off.

| Dataset | Result | Read |
|---|---|---|
| `lerobot/pusht` (LeRobot v3) | 206 episodes at 10 Hz, matching `meta/info.json` | 205 KB of 659 KB |
| `nvidia/Nemotron-Personas-Vietnam` (text) | 90,910 rows, 21 columns per file | 143 KB of 194 MB |

Repos with nothing to profile now say so and exit with code 3, listing the file types they do contain.

---

# MCAP File Inspector

A tool to inspect MCAP (Message Capture) files and extract header, footer, channels, topics, and schema information.

## Features

- **Header Inspection**: Extracts profile and library information
- **Footer Inspection**: Extracts summary start offset, summary offset start, and CRC
- **Channel Discovery**: Lists all channels with their topics, encoding, schema IDs, and metadata
- **Topic Listing**: Provides a sorted list of all unique topics
- **Schema Information**: Extracts schema details including name, encoding, and data
- **Local and Remote Files**: Supports both local file paths and remote URLs
- **Fallback Support**: Works with or without the mcap library installed

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install mcap
```

## Usage

### As a module

```python
from phys_ai_dataprofiler.inspectors.mcap import McapFileInspector

# Inspect a local file
inspector = McapFileInspector("samples/sample.mcap")
result = inspector.inspect()

print(f"Topics: {result.topics}")
print(f"Channels: {len(result.channels)}")
print(f"Header Profile: {result.header.profile}")
print(f"Footer Summary Start: {result.footer.summary_start}")
```

### As a command-line tool

```bash
# Using the basic inspect_file script
python -m phys_ai_dataprofiler.inspectors.mcap samples/sample.mcap

# Using the test script with formatted output
python tests/test_mcap_inspector.py samples/sample.mcap
```

### Remote URLs

```bash
python tests/test_mcap_inspector.py https://example.com/data.mcap
```

## Data Structures

### McapInspectionResult
- `url`: File URL or path
- `is_mcap`: Boolean indicating if file is a valid MCAP
- `magic`: Magic bytes from file header
- `size_bytes`: File size in bytes
- `path`: Local path (if applicable)
- `error`: Error message (if any)
- `header`: HeaderInfo object
- `footer`: FooterInfo object  
- `channels`: List of ChannelInfo objects
- `schemas`: List of SchemaInfo objects
- `topics`: Sorted list of topic strings
- `mcap_available`: Boolean indicating if mcap library is available

### HeaderInfo
- `profile`: MCAP profile string
- `library`: Library that created the file

### FooterInfo
- `summary_start`: Offset where summary section starts
- `summary_offset_start`: Offset where summary offsets start
- `summary_crc`: CRC checksum of summary section

### ChannelInfo
- `id`: Channel ID
- `topic`: Topic name
- `message_encoding`: Message encoding (e.g., "json", "protobuf")
- `schema_id`: Associated schema ID
- `metadata`: Dictionary of channel metadata

### SchemaInfo
- `id`: Schema ID
- `name`: Schema name
- `encoding`: Schema encoding type
- `data`: Raw schema data (bytes)

## Example Output

```
Inspecting: samples/sample.mcap

URL: samples/sample.mcap
Is MCAP: True
Size: 1042 bytes
MCAP Library Available: True

=== Header ===
Profile: test
Library: sample_creator

=== Footer ===
Summary Start: 397
Summary Offset Start: 849
Summary CRC: 544820573

=== Topics (2) ===
  - /sensors/gps
  - /sensors/imu

=== Channels (2) ===
  Channel 1:
    Topic: /sensors/gps
    Encoding: json
    Schema ID: 1
    Metadata: {'description': 'GPS sensor data'}
  Channel 2:
    Topic: /sensors/imu
    Encoding: json
    Schema ID: 1
    Metadata: {'description': 'IMU sensor data'}

=== Schemas (1) ===
  Schema 1:
    Name: example.Point
    Encoding: jsonschema
    Data size: 84 bytes
```

## Creating Test Files

Use the included `create_sample_mcap.py` script to create test MCAP files:

```bash
python scripts/create_sample_mcap.py output.mcap
```

## Implementation Details

The inspector uses the official [mcap](https://github.com/foxglove/mcap) library to parse MCAP files:
- `make_reader()` for high-level summary and header access
- `StreamReader` for low-level footer record extraction
- Supports both file handles and BytesIO for remote content

For remote files, the inspector first checks the magic bytes using HTTP Range requests, then downloads the full file for parsing if valid.

## License

This is a demonstration tool for MCAP file inspection.

---

# MATLAB File Inspector

A tool to inspect MATLAB .mat files and extract header information, variable metadata, and data types.

## Features

- **Header Inspection**: Extracts description, version, and endianness information
- **Variable Discovery**: Lists all variables with their names, types, shapes, and sizes
- **Type Detection**: Identifies scalars, vectors, matrices, complex numbers, sparse matrices, structs, and cell arrays
- **Format Support**: Handles MAT-File v4, v5, v7, and v7.3 (HDF5-based) formats
- **Local and Remote Files**: Supports both local file paths and remote URLs
- **Fallback Support**: Works with limited functionality if scipy/h5py are not installed

## Installation

1. Create a virtual environment (if not already created):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install scipy numpy       # For v4/v5/v7 format support
pip install h5py              # For v7.3 (HDF5) format support (requires HDF5 library)
```

## Usage

### As a module

```python
from phys_ai_dataprofiler.inspectors.mat import MatFileInspector

# Inspect a local file
inspector = MatFileInspector("data.mat")
result = inspector.inspect()

print(f"Format: {result.file_format}")
print(f"Variables: {result.variable_names}")
for var in result.variables:
    print(f"{var.name}: {var.dtype} {var.shape}")
```

### As a command-line tool

```bash
# Using the basic matfile_inspector script
python -m phys_ai_dataprofiler.inspectors.mat data.mat

# Using the test script with formatted output
python tests/test_mat_inspector.py data.mat
```

### Remote URLs

```bash
python tests/test_mat_inspector.py https://example.com/data.mat
```

## Data Structures

### MatInspectionResult
- `url`: File URL or path
- `is_matfile`: Boolean indicating if file is a valid MAT-File
- `file_format`: Format version ("v4", "v5", "v7", or "v7.3")
- `size_bytes`: File size in bytes
- `path`: Local path (if applicable)
- `error`: Error message (if any)
- `header`: MatFileHeader object
- `variables`: List of VariableInfo objects
- `variable_names`: Sorted list of variable names
- `scipy_available`: Boolean indicating if scipy is available
- `h5py_available`: Boolean indicating if h5py is available

### MatFileHeader
- `description`: File description string (includes platform and creation time)
- `version`: Version number in hex format
- `endian`: Endianness ("little", "big", or "unknown")
- `created`: Creation timestamp (when available)

### VariableInfo
- `name`: Variable name
- `dtype`: Data type string
- `shape`: Tuple of dimensions
- `size_bytes`: Size in bytes
- `is_sparse`: Boolean indicating sparse matrix
- `is_complex`: Boolean indicating complex numbers
- `is_struct`: Boolean indicating MATLAB struct
- `is_cell`: Boolean indicating cell array
- `field_names`: List of field names (for structs)

## Example Output

```
Inspecting: samples/sample_v5.mat

URL: samples/sample_v5.mat
Is MAT-File: True
File Format: v5
Size: 1544 bytes
SciPy Available: True
h5py Available: False

=== Header ===
Description: MATLAB 5.0 MAT-file Platform: posix, Created on: Sun Oct 11 18:59:29 2026
Version: 0x0100
Endian: little

=== Variables (12) ===
  - col_vector
  - complex_data
  - double_data
  - float_data
  - int_array
  - logical_array
  - matrix_2d
  - matrix_3d
  - row_vector
  - scalar_value
  - text_string
  - uint_array

=== Variable Details ===

  scalar_value:
    Type: int64
    Shape: (1, 1)
    Size: 8 bytes

  complex_data:
    Type: complex128
    Shape: (1, 3)
    Size: 48 bytes
    Flags: complex

  matrix_3d:
    Type: float64
    Shape: (2, 3, 4)
    Size: 192 bytes
```

## Supported MAT-File Formats

### v4 Format
- Legacy format from MATLAB 4
- Limited data type support
- Requires scipy

### v5 Format (Default)
- MATLAB 5 and later (default save format in older versions)
- Supports all standard MATLAB types
- Requires scipy

### v7 Format
- Compressed v5 format
- Same structure as v5 but with compression
- Requires scipy

### v7.3 Format (HDF5-based)
- MATLAB R2006b and later
- Based on HDF5 file format
- Required for variables larger than 2GB
- Requires h5py and HDF5 library

## Creating Test Files

Use the included `create_sample_mat.py` script to create test MAT files:

```bash
python scripts/create_sample_mat.py samples/sample_v5.mat samples/sample_v7.mat
```

## Implementation Details

The inspector uses two libraries depending on the MAT-File format:
- **scipy.io.loadmat**: For v4, v5, and v7 formats
- **h5py**: For v7.3 (HDF5-based) format

For remote files, the inspector first checks the file header using HTTP Range requests to determine the format, then downloads the full file for parsing if valid.

## Comparison with MCAP Inspector

Both inspectors follow a similar architecture:

| Feature | MCAP Inspector | MATLAB Inspector |
|---------|----------------|------------------|
| File Format | MCAP (robotics) | MAT-File (MATLAB) |
| Header Info | ✓ | ✓ |
| Footer Info | ✓ | N/A |
| Variable/Channel List | ✓ | ✓ |
| Type Information | ✓ | ✓ |
| Remote Support | ✓ | ✓ |
| Compression Support | ✓ | ✓ |

---

# Hugging Face Connector

`phys_ai_dataprofiler/connectors/huggingface.py` (CLI: `phys-ai-dataprofiler`, or `python -m phys_ai_dataprofiler`) connects to the Hugging Face Hub with an access token,
finds the MCAP / MAT files in a repo, inspects them and writes a JSON and a
Markdown report. By default **nothing is downloaded**: files are read through
HTTP Range requests and only the bytes holding metadata are fetched.

| What | MCAP | MAT (v5 / v7) |
|---|---|---|
| Bytes read | header, footer, summary section, MessageIndex records | 128-byte header + each variable's tag and array header, then float variables up to `--nan-budget-mb` |
| Typical share of file | well under 1% (163 MB file: ~0.5 MB) | headers: one 64 KB block per few variables; plus the float data that fits the NaN budget |
| Available | profile, library, topics, schemas, channels, per-channel count, rate, jitter, gaps, duration | format, header, every variable with name, class/dtype, dims, stored size, complex/sparse/struct/cell flags, struct field names, NaN counts |
| No summary / truncated | walked record by record: chunk headers and message indexes, chunk bodies only for channel definitions or when a chunk has no index; capped by `--msgidx-budget-mb` | - |
| Needs a full read | - | v4 files, v7.3 without `h5py` |

Remote mode gives the same streams, rates, gaps, findings and NaN counts as
the full-read `RichInspector` (checked by `tests/test_remote_rich.py`). It also
profiles truncated MCAPs, which the full read cannot open. Anything partial
(budget reached, chunks sampled) is reported as a finding. Use
`--full-read-max-mb N` to fetch files at or below that size whole instead
(deleted afterwards unless `--keep-downloads`), or `--mode download` to go
back to downloading everything.

## Installation

```bash
source ~/pers_workspace/adhoc/venv311/bin/activate
pip install huggingface_hub
```

## Token

Resolved in this order: `--token`, then `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`,
then the login cached by `hf auth login`. Public repos work with no token.
The token is only ever sent to huggingface.co, never to the CDN hosts that
serve the file bytes.

## Usage

```bash
# Inspect every .mcap/.mat file in a dataset repo without downloading
phys-ai-dataprofiler org/dataset --token hf_xxx

# See what would be inspected first
phys-ai-dataprofiler org/dataset --dry-run

# Limit the selection: by glob, by count, by size
phys-ai-dataprofiler org/dataset --include "episodes/*.mcap" --max-files 5

# Exact files
phys-ai-dataprofiler org/dataset --file logs/run1.mcap --file calib/cam.mat

# Larger NaN budget (default 32 MB of float data per file), or none
phys-ai-dataprofiler org/dataset --nan-budget-mb 256
phys-ai-dataprofiler org/dataset --nan-budget-mb 0

# Fetch small files whole instead of range-reading them
phys-ai-dataprofiler org/dataset --full-read-max-mb 50

# Classic download mode, and re-running on an earlier download
phys-ai-dataprofiler org/dataset --mode download --max-total-mb 2000
phys-ai-dataprofiler org/dataset --mode download --offline

# Model repos, specific revisions
phys-ai-dataprofiler org/model --repo-type model --revision v1.2
```

Tuning for remote mode: `--workers` (parallel range requests, default 16),
`--msgidx-budget-mb` (cap on MCAP message-index bytes per file; beyond it
chunks are sampled and the report says so; also caps the record walk of MCAPs
without a summary), `--nan-budget-mb` (float bytes read per file for NaN
counts), `--block-kb` (cached read block).

Reports go to `./reports/<repo>__<timestamp>.{json,md}` (override with
`--report-dir`, `--report-name`) and include, per file, how many bytes were
read and in how many requests. Exit code is 1 when any file failed, so it is
CI-friendly.

## Modules

- `phys_ai_dataprofiler/remote/range_file.py`: `HfRangeFile` (seekable file over Hub byte ranges,
  block cache, parallel `fetch_many`) and `LocalRangeFile` (same API on disk).
- `phys_ai_dataprofiler/remote/inspector.py`: `inspect_mcap_remote` / `inspect_mat_remote`,
  returning the same `McapInspectionResult`, `MatInspectionResult` and
  `RichProfile` types as the full-read inspectors. Run it on a local file with
  `python -m phys_ai_dataprofiler.remote.inspector file.mcap` to see what it extracts.
- `phys_ai_dataprofiler/inspectors/rich.py` exposes `timing_stats` and `profile_mcap_channels`,
  shared by both paths so remote and local statistics are identical.

```python
from phys_ai_dataprofiler.connectors.huggingface import HuggingFaceConnector
from phys_ai_dataprofiler.remote.inspector import inspect_mcap_remote, inspect_mat_remote, sniff_format

hf = HuggingFaceConnector("org/dataset", token="hf_xxx")
hf.connect()
selected, skipped = hf.select_files(hf.list_files(), max_files=10)
for f in selected:
    rf = hf.open_range(f, block_size=64 * 1024)
    fmt = sniff_format(rf, f.path)
    result, profile = inspect_mcap_remote(rf) if fmt == "mcap" else inspect_mat_remote(rf)
    print(f.path, rf.bytes_fetched, "of", rf.size, "bytes read")
```

## License

These are demonstration tools for file inspection.
