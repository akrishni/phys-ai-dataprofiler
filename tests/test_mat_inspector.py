#!/usr/bin/env python3
"""Checks for the full-read MATLAB inspector.

    python tests/test_mat_inspector.py              # checks against samples/ (exits non-zero on failure)
    python tests/test_mat_inspector.py FILE_OR_URL  # print what the inspector finds in one file
"""

import sys
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.inspectors.mat import MatFileInspector


def print_result(result):
    """Pretty print the inspection result."""
    print(f"URL: {result.url}")
    print(f"Is MAT-File: {result.is_matfile}")
    print(f"File Format: {result.file_format}")
    print(f"Size: {result.size_bytes} bytes" if result.size_bytes else "Size: Unknown")
    print(f"SciPy Available: {result.scipy_available}")
    print(f"h5py Available: {result.h5py_available}")

    if result.error:
        print(f"\nError: {result.error}")
        return

    if result.header:
        print("\n=== Header ===")
        print(f"Description: {result.header.description}")
        print(f"Version: {result.header.version}")
        print(f"Endian: {result.header.endian}")
        if result.header.created:
            print(f"Created: {result.header.created}")

    if result.variable_names:
        print(f"\n=== Variables ({len(result.variable_names)}) ===")
        for name in result.variable_names:
            print(f"  - {name}")

    if result.variables:
        print(f"\n=== Variable Details ===")
        for var in result.variables:
            print(f"\n  {var.name}:")
            print(f"    Type: {var.dtype}")
            print(f"    Shape: {var.shape}")
            print(f"    Size: {var.size_bytes} bytes")

            flags = []
            if var.is_sparse:
                flags.append("sparse")
            if var.is_complex:
                flags.append("complex")
            if var.is_struct:
                flags.append("struct")
            if var.is_cell:
                flags.append("cell")

            if flags:
                print(f"    Flags: {', '.join(flags)}")

            if var.field_names:
                print(f"    Fields: {', '.join(var.field_names)}")


ROOT = Path(__file__).resolve().parents[1]


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def run_checks() -> None:
    r = MatFileInspector(str(ROOT / "samples" / "sample_v5.mat")).inspect()
    check(r.is_matfile and r.error is None and r.file_format == "v5", "sample_v5.mat parses as v5")
    check(len(r.variable_names or []) == 12, f"12 variables, got {len(r.variable_names or [])}")
    v = {x.name: x for x in r.variables or []}
    check(v["matrix_2d"].dtype == "float64" and tuple(v["matrix_2d"].shape) == (3, 4), "2-D double matrix")
    check(tuple(v["matrix_3d"].shape) == (2, 3, 4), "3-D shape kept")
    check(v["complex_data"].dtype == "complex128", "complex type")
    check(v["int_array"].dtype == "int32" and v["uint_array"].dtype == "uint16", "integer types")
    check(v["float_data"].dtype == "float32", "single precision")

    r = MatFileInspector(str(ROOT / "samples" / "sample_v7.mat")).inspect()
    check(r.is_matfile and r.error is None, "sample_v7.mat (compressed) parses")
    v = {x.name: x for x in r.variables or []}
    check(sorted(v) == ["device_id", "location", "sample_rate", "sensor_data", "timestamps"], "v7 variable names")
    check(tuple(v["sensor_data"].shape) == (100, 3) and tuple(v["timestamps"].shape) == (1, 100), "v7 shapes")

    bad = MatFileInspector(str(ROOT / "samples" / "test.txt")).inspect()
    check(not bad.is_matfile, "a text file is not a MAT file")
    print("all checks passed")


def main():
    if len(sys.argv) < 2:
        run_checks()
        return
    url = sys.argv[1]
    print(f"Inspecting: {url}\n")
    print_result(MatFileInspector(url).inspect())


if __name__ == "__main__":
    main()
