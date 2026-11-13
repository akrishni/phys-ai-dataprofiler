# Full-read inspectors (local files)

These are the original inspectors in `phys_ai_dataprofiler/inspectors/`. They **read the whole file**
and are meant for files already on disk. They are not what the `phys-ai-dataprofiler` command uses to
profile the Hub: that goes through the byte-range readers in `phys_ai_dataprofiler/remote/`, which fetch
only metadata (see the [README](../README.md)). The test suite uses these inspectors as the reference
that the range readers are checked against (`tests/test_remote_rich.py`).

Install the package with `pip install -e ".[mat73,dev]"` (see Development in the README); that brings in
`mcap`, `scipy`, `numpy` and, for MAT v7.3, `h5py`.

## MCAP inspector

A tool to inspect MCAP (Message Capture) files and extract header, footer, channels, topics, and schema information.

### Features

- **Header Inspection**: Extracts profile and library information
- **Footer Inspection**: Extracts summary start offset, summary offset start, and CRC
- **Channel Discovery**: Lists all channels with their topics, encoding, schema IDs, and metadata
- **Topic Listing**: Provides a sorted list of all unique topics
- **Schema Information**: Extracts schema details including name, encoding, and data
- **Local files, or URLs downloaded in full**
- **Fallback Support**: Works with or without the mcap library installed

### Usage

#### As a module

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

#### As a command-line tool

```bash
# Using the basic inspect_file script
python -m phys_ai_dataprofiler.inspectors.mcap samples/sample.mcap

# Using the test script with formatted output
python tests/test_mcap_inspector.py samples/sample.mcap
```

#### URLs (downloaded in full)

```bash
python tests/test_mcap_inspector.py https://example.com/data.mcap
```

### Data Structures

#### McapInspectionResult
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

#### HeaderInfo
- `profile`: MCAP profile string
- `library`: Library that created the file

#### FooterInfo
- `summary_start`: Offset where summary section starts
- `summary_offset_start`: Offset where summary offsets start
- `summary_crc`: CRC checksum of summary section

#### ChannelInfo
- `id`: Channel ID
- `topic`: Topic name
- `message_encoding`: Message encoding (e.g., "json", "protobuf")
- `schema_id`: Associated schema ID
- `metadata`: Dictionary of channel metadata

#### SchemaInfo
- `id`: Schema ID
- `name`: Schema name
- `encoding`: Schema encoding type
- `data`: Raw schema data (bytes)

### Example Output

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

### Creating Test Files

Use the included `create_sample_mcap.py` script to create test MCAP files:

```bash
python scripts/create_sample_mcap.py output.mcap
```

### Implementation Details

The inspector uses the official [mcap](https://github.com/foxglove/mcap) library to parse MCAP files:
- `make_reader()` for high-level summary and header access
- `StreamReader` for low-level footer record extraction
- Supports both file handles and BytesIO for remote content

Given a URL, this inspector reads the magic bytes with a Range request and then **downloads the whole file** to parse it. For Hub datasets use the `phys-ai-dataprofiler` command instead, which reads only the metadata bytes.

## MATLAB inspector

A tool to inspect MATLAB .mat files and extract header information, variable metadata, and data types.

### Features

- **Header Inspection**: Extracts description, version, and endianness information
- **Variable Discovery**: Lists all variables with their names, types, shapes, and sizes
- **Type Detection**: Identifies scalars, vectors, matrices, complex numbers, sparse matrices, structs, and cell arrays
- **Format Support**: Handles MAT-File v4, v5, v7, and v7.3 (HDF5-based) formats
- **Local files, or URLs downloaded in full**
- **Fallback Support**: Works with limited functionality if scipy/h5py are not installed

### Usage

#### As a module

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

#### As a command-line tool

```bash
# Using the basic matfile_inspector script
python -m phys_ai_dataprofiler.inspectors.mat data.mat

# Using the test script with formatted output
python tests/test_mat_inspector.py data.mat
```

#### URLs (downloaded in full)

```bash
python tests/test_mat_inspector.py https://example.com/data.mat
```

### Data Structures

#### MatInspectionResult
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

#### MatFileHeader
- `description`: File description string (includes platform and creation time)
- `version`: Version number in hex format
- `endian`: Endianness ("little", "big", or "unknown")
- `created`: Creation timestamp (when available)

#### VariableInfo
- `name`: Variable name
- `dtype`: Data type string
- `shape`: Tuple of dimensions
- `size_bytes`: Size in bytes
- `is_sparse`: Boolean indicating sparse matrix
- `is_complex`: Boolean indicating complex numbers
- `is_struct`: Boolean indicating MATLAB struct
- `is_cell`: Boolean indicating cell array
- `field_names`: List of field names (for structs)

### Example Output

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

### Supported MAT-File Formats

#### v4 Format
- Legacy format from MATLAB 4
- Limited data type support
- Requires scipy

#### v5 Format (Default)
- MATLAB 5 and later (default save format in older versions)
- Supports all standard MATLAB types
- Requires scipy

#### v7 Format
- Compressed v5 format
- Same structure as v5 but with compression
- Requires scipy

#### v7.3 Format (HDF5-based)
- MATLAB R2006b and later
- Based on HDF5 file format
- Required for variables larger than 2GB
- Requires h5py and HDF5 library

### Creating Test Files

Use the included `create_sample_mat.py` script to create test MAT files:

```bash
python scripts/create_sample_mat.py samples/sample_v5.mat samples/sample_v7.mat
```

### Implementation Details

The inspector uses two libraries depending on the MAT-File format:
- **scipy.io.loadmat**: For v4, v5, and v7 formats
- **h5py**: For v7.3 (HDF5-based) format

Given a URL, this inspector reads the 128-byte header with a Range request and then **downloads the whole file** to parse it. For Hub datasets use the `phys-ai-dataprofiler` command instead, which reads only variable headers and float data within the NaN budget.

### Comparison with MCAP Inspector

Both inspectors follow a similar architecture:

| Feature | MCAP Inspector | MATLAB Inspector |
|---------|----------------|------------------|
| File Format | MCAP (robotics) | MAT-File (MATLAB) |
| Header Info | ✓ | ✓ |
| Footer Info | ✓ | N/A |
| Variable/Channel List | ✓ | ✓ |
| Type Information | ✓ | ✓ |
| URLs (full download) | ✓ | ✓ |
| Compression Support | ✓ | ✓ |
