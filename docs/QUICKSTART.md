# Quick Start Guide

Get up and running with the file inspectors in under 5 minutes!

## 1. Setup (1 minute)

```bash
# Navigate to the project directory
cd ~/pers_workspace/phys-ai-dataprofiler

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install mcap scipy numpy
```

## 2. Create Sample Files (30 seconds)

```bash
# Create sample MCAP file
python scripts/create_sample_mcap.py

# Create sample MATLAB files
python scripts/create_sample_mat.py
```

## 3. Try the Inspectors (1 minute)

### Option A: Run the Demo
```bash
python scripts/demo.py
```

### Option B: Inspect Individual Files
```bash
# MCAP file
python tests/test_mcap_inspector.py samples/sample.mcap

# MATLAB file
python tests/test_mat_inspector.py samples/sample_v5.mat
```

### Option C: Compare Multiple Files
```bash
python scripts/compare_inspectors.py samples/sample.mcap samples/sample_v5.mat
```

## 4. Use in Your Code (2 minutes)

### MCAP Files

```python
from tools.mcapfile_inspector import McapFileInspector

# Inspect a file
inspector = McapFileInspector("data.mcap")
result = inspector.inspect()

# Access the data
if result.is_mcap:
    print(f"Topics: {result.topics}")
    print(f"Channels: {len(result.channels)}")
    
    for channel in result.channels:
        print(f"  {channel.topic}: {channel.message_encoding}")
```

### MATLAB Files

```python
from phys_ai_dataprofiler.inspectors.mat import MatFileInspector

# Inspect a file
inspector = MatFileInspector("data.mat")
result = inspector.inspect()

# Access the data
if result.is_matfile:
    print(f"Format: {result.file_format}")
    print(f"Variables: {result.variable_names}")
    
    for var in result.variables:
        print(f"  {var.name}: {var.dtype} {var.shape}")
```

## Common Use Cases

### 1. Quick File Validation
```python
# Check if file is valid without parsing
from tools.mcapfile_inspector import McapFileInspector

result = McapFileInspector("unknown.file").inspect()
if result.is_mcap:
    print("✓ Valid MCAP file")
else:
    print("✗ Not a valid MCAP file")
```

### 2. Extract Metadata Only
```python
# Get just the header info
result = MatFileInspector("data.mat").inspect()
if result.header:
    print(f"Created: {result.header.description}")
    print(f"Format: {result.file_format}")
```

### 3. Remote File Inspection
```python
# Inspect a file from URL
url = "https://example.com/data.mcap"
result = McapFileInspector(url).inspect()
print(f"Remote file size: {result.size_bytes} bytes")
```

### 4. Batch Processing
```python
import glob

# Inspect all MCAP files in directory
for filepath in glob.glob("*.mcap"):
    result = McapFileInspector(filepath).inspect()
    if result.is_mcap and not result.error:
        print(f"{filepath}: {len(result.topics)} topics")
```

### 5. Data Type Analysis
```python
# Analyze variable types in MAT file
result = MatFileInspector("data.mat").inspect()
if result.variables:
    complex_vars = [v for v in result.variables if v.is_complex]
    sparse_vars = [v for v in result.variables if v.is_sparse]
    
    print(f"Complex variables: {len(complex_vars)}")
    print(f"Sparse matrices: {len(sparse_vars)}")
```

## Troubleshooting

### "mcap library not available"
```bash
pip install mcap lz4 zstandard
```

### "scipy library required"
```bash
pip install scipy numpy
```

### "h5py library required for MAT-File v7.3"
```bash
# Install HDF5 first
brew install hdf5  # macOS
# or apt-get install libhdf5-dev  # Ubuntu
# or see: https://www.hdfgroup.org/downloads/hdf5/

# Then install h5py
pip install h5py
```

### "Not a valid MATLAB file"
- Check if file is actually a .mat file
- Try opening with MATLAB/Octave to verify
- File might be corrupted

### Remote file inspection fails
- Check internet connection
- Verify URL is accessible
- Server must support HTTP Range requests for optimal performance

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [SUMMARY.md](SUMMARY.md) for implementation details
- Browse the source code in the `phys_ai_dataprofiler/` package
- Create your own file format inspector using the same pattern!

## Tips

1. **Performance**: For large remote files, download once and inspect locally
2. **Memory**: Inspectors only load metadata, not the full data
3. **Errors**: Always check `result.error` before accessing other fields
4. **Compatibility**: Both inspectors work with Python 3.9+

## Need Help?

Check these files for more information:
- `README.md` - Full documentation
- `SUMMARY.md` - Implementation details
- `demo.py` - Working examples
- Source code in `phys_ai_dataprofiler/` - Inline documentation

Happy inspecting! 🔍
