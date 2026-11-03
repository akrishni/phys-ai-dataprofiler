#!/usr/bin/env python3
"""Compare file inspectors for different formats."""

import sys
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.inspectors.mcap import McapFileInspector
from phys_ai_dataprofiler.inspectors.mat import MatFileInspector


def inspect_file(file_path: str):
    """Auto-detect file type and inspect."""
    path = Path(file_path)

    print(f"{'='*70}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*70}\n")

    # Check file extension for hints
    ext = path.suffix.lower()

    if ext == ".mcap":
        print("Detected MCAP file (by extension)")
        inspector = McapFileInspector(file_path)
        result = inspector.inspect()

        print(f"Valid MCAP: {result.is_mcap}")
        if result.is_mcap and not result.error:
            print(f"Profile: {result.header.profile if result.header else 'N/A'}")
            print(f"Topics: {len(result.topics) if result.topics else 0}")
            print(f"Channels: {len(result.channels) if result.channels else 0}")
            print(f"Schemas: {len(result.schemas) if result.schemas else 0}")
        elif result.error:
            print(f"Error: {result.error}")

    elif ext == ".mat":
        print("Detected MATLAB file (by extension)")
        inspector = MatFileInspector(file_path)
        result = inspector.inspect()

        print(f"Valid MAT-File: {result.is_matfile}")
        if result.is_matfile and not result.error:
            print(f"Format: {result.file_format}")
            print(f"Variables: {len(result.variables) if result.variables else 0}")
            if result.variables:
                total_size = sum(v.size_bytes for v in result.variables)
                print(f"Total data size: {total_size} bytes")
        elif result.error:
            print(f"Error: {result.error}")
    else:
        print(f"Unknown file extension: {ext}")
        print("Trying both inspectors...\n")

        # Try MCAP first
        print("--- Trying MCAP Inspector ---")
        mcap_result = McapFileInspector(file_path).inspect()
        print(f"Is MCAP: {mcap_result.is_mcap}")

        # Try MAT
        print("\n--- Trying MATLAB Inspector ---")
        mat_result = MatFileInspector(file_path).inspect()
        print(f"Is MAT-File: {mat_result.is_matfile}")

        if not mcap_result.is_mcap and not mat_result.is_matfile:
            print("\nFile format not recognized by either inspector")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_inspectors.py <file1> [file2] [file3] ...")
        print("\nExample:")
        print("  python scripts/compare_inspectors.py samples/sample.mcap samples/sample_v5.mat")
        sys.exit(1)

    for file_path in sys.argv[1:]:
        inspect_file(file_path)
        print()


if __name__ == "__main__":
    main()
