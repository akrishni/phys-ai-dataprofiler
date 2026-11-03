# File Inspectors - Implementation Summary

## Overview

Created two comprehensive file inspection tools for binary file formats:
1. **MCAP File Inspector** - For robotics data capture files
2. **MATLAB File Inspector** - For MATLAB data files

Both inspectors follow similar architecture patterns and provide detailed metadata extraction.

## Files Created

### Core Inspectors
- `phys_ai_dataprofiler/inspectors/mcap.py` - MCAP file inspector
- `phys_ai_dataprofiler/inspectors/mat.py` - MATLAB file inspector

### Test Scripts
- `test_inspector.py` - Test script for MCAP inspector
- `test_matfile_inspector.py` - Test script for MATLAB inspector
- `compare_inspectors.py` - Unified comparison tool for both inspectors

### Sample Data Generators
- `create_sample_mcap.py` - Creates sample MCAP files for testing
- `create_sample_mat.py` - Creates sample MAT files for testing

### Sample Files
- `samples/sample.mcap` - Sample MCAP file with 2 channels/topics
- `samples/sample_v5.mat` - MAT-File v5 with 12 variables
- `samples/sample_v7.mat` - MAT-File v7 (compressed) with sensor data

### Documentation
- `README.md` - Complete documentation for both inspectors
- `SUMMARY.md` - This file

## Architecture

Both inspectors share a common design pattern:

```
Inspector Class
├── __init__(url)
├── inspect() → Result
├── _inspect_local(path) → Result
├── _inspect_remote(url) → Result
├── _parse_with_library() → dict
└── Helper methods
```

### Common Features

1. **Dual Source Support**
   - Local file paths
   - Remote URLs (with HTTP Range request optimization)

2. **Magic Byte Detection**
   - Quick format validation before full parsing
   - Minimal data transfer for remote files

3. **Library-based Parsing**
   - MCAP: Uses official `mcap` library
   - MATLAB: Uses `scipy` (v4/v5/v7) and `h5py` (v7.3)

4. **Graceful Degradation**
   - Works with or without optional libraries
   - Clear error messages when libraries missing

5. **Rich Metadata Extraction**
   - Headers, footers, variables/channels
   - Type information, shapes, sizes
   - Format-specific details

## MCAP Inspector Details

### Dependencies
- `mcap` - Official MCAP library
- `lz4` - Compression support
- `zstandard` - Compression support

### Extracted Information
- **Header**: profile, library
- **Footer**: summary offsets, CRC checksums
- **Channels**: ID, topic, encoding, schema, metadata
- **Schemas**: ID, name, encoding, data
- **Topics**: Unique list of all topics

### Implementation Highlights
- Uses `make_reader()` for high-level access
- Uses `StreamReader` for footer record extraction
- Supports both seeking and non-seeking file handles

## MATLAB Inspector Details

### Dependencies
- `scipy` - For v4/v5/v7 formats
- `numpy` - Array operations
- `h5py` - For v7.3 HDF5-based format (optional)

### Extracted Information
- **Header**: description, version, endianness, creation time
- **Variables**: name, dtype, shape, size
- **Type Flags**: sparse, complex, struct, cell array
- **Struct Fields**: Field names for structured arrays

### Implementation Highlights
- Format auto-detection from magic bytes
- Handles all MAT-File versions (v4, v5, v7, v7.3)
- Extracts detailed type information including:
  - Scalars, vectors, matrices (any dimension)
  - Complex numbers
  - Sparse matrices
  - Structs and cell arrays
  - Multiple numeric types (int8-int64, uint8-uint64, float32, float64)

## Usage Examples

### MCAP Inspection
```python
from tools.mcapfile_inspector import McapFileInspector

inspector = McapFileInspector("data.mcap")
result = inspector.inspect()

print(f"Topics: {result.topics}")
print(f"Channels: {len(result.channels)}")
```

### MATLAB Inspection
```python
from phys_ai_dataprofiler.inspectors.mat import MatFileInspector

inspector = MatFileInspector("data.mat")
result = inspector.inspect()

print(f"Format: {result.file_format}")
print(f"Variables: {result.variable_names}")
```

### Auto-detection
```python
from compare_inspectors import inspect_file

inspect_file("unknown_file.bin")
```

## Testing

All inspectors have been tested with:
- ✅ Valid files (both local and remote)
- ✅ Invalid/corrupted files
- ✅ Non-matching file types
- ✅ Missing dependencies
- ✅ Empty files
- ✅ Various data types and structures

## Performance Considerations

### Local Files
- Single-pass reading for small files
- Efficient seeking for large files
- Minimal memory footprint (metadata only)

### Remote Files
- **Optimization**: Range requests for headers/footers
- **Trade-off**: Full download required for complete parsing
- Future: Could implement streaming parsers for better efficiency

## Future Enhancements

### MCAP Inspector
- [ ] Streaming support for large remote files
- [ ] Message count statistics
- [ ] Attachment extraction
- [ ] Metadata records parsing

### MATLAB Inspector
- [ ] Partial variable loading (for large files)
- [ ] HDF5 streaming support
- [ ] Better struct field parsing
- [ ] MATLAB object support
- [ ] Function handle inspection

### Both
- [ ] Progress callbacks for large files
- [ ] Batch inspection of multiple files
- [ ] Export to JSON/YAML
- [ ] Comparison/diff between files
- [ ] GUI interface

## Dependencies Matrix

| Feature | MCAP | MATLAB (v5) | MATLAB (v7.3) |
|---------|------|-------------|---------------|
| Core Python | ✓ | ✓ | ✓ |
| numpy | - | ✓ | ✓ |
| scipy | - | ✓ | - |
| mcap | ✓ | - | - |
| h5py | - | - | ✓ |
| lz4 | ✓ | - | - |
| zstandard | ✓ | - | - |

## Code Statistics

- **Total Lines**: ~1200 (both inspectors + tests)
- **MCAP Inspector**: ~260 lines
- **MATLAB Inspector**: ~450 lines
- **Test Scripts**: ~250 lines
- **Sample Generators**: ~150 lines
- **Documentation**: ~500 lines

## Known Limitations

### MCAP Inspector
- Remote files require full download (no streaming)
- Large attachments load fully into memory

### MATLAB Inspector
- v7.3 format requires h5py + HDF5 library installation
- Struct field data not fully parsed (only field names)
- Cell array contents not deeply inspected
- MATLAB objects not supported

## Conclusion

Both inspectors provide production-ready functionality for inspecting their respective file formats. They follow best practices for:
- Error handling and reporting
- Resource management
- Library usage
- API design
- Documentation

The shared architecture makes it easy to add additional file format inspectors following the same pattern.
