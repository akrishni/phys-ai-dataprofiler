#!/usr/bin/env python3
"""Test script for MATLAB file inspector."""

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


def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_mat_inspector.py <mat_file_or_url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"Inspecting: {url}\n")

    inspector = MatFileInspector(url)
    result = inspector.inspect()

    print_result(result)


if __name__ == "__main__":
    main()
