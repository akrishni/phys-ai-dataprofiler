#!/usr/bin/env python3
"""
Demonstration of MCAP and MATLAB file inspectors.

This script showcases the capabilities of both file inspectors
by analyzing sample files and displaying their metadata.
"""

import sys
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.inspectors.mcap import McapFileInspector
from phys_ai_dataprofiler.inspectors.mat import MatFileInspector


def demo_mcap():
    """Demonstrate MCAP file inspection."""
    print("=" * 80)
    print("MCAP FILE INSPECTION DEMO")
    print("=" * 80)

    inspector = McapFileInspector("samples/sample.mcap")
    result = inspector.inspect()

    print(f"\n📁 File: {result.url}")
    print(f"✓ Valid MCAP: {result.is_mcap}")
    print(f"📊 Size: {result.size_bytes:,} bytes")

    if result.header:
        print(f"\n📋 Header Information:")
        print(f"   Profile: {result.header.profile}")
        print(f"   Library: {result.header.library}")

    if result.footer:
        print(f"\n📊 Footer Information:")
        print(f"   Summary Start: {result.footer.summary_start}")
        print(f"   Summary Offset: {result.footer.summary_offset_start}")
        print(f"   CRC: 0x{result.footer.summary_crc:08x}")

    if result.topics:
        print(f"\n📡 Topics ({len(result.topics)}):")
        for topic in result.topics:
            print(f"   • {topic}")

    if result.channels:
        print(f"\n📺 Channels ({len(result.channels)}):")
        for channel in result.channels:
            print(f"   Channel {channel.id}:")
            print(f"      Topic: {channel.topic}")
            print(f"      Encoding: {channel.message_encoding}")
            print(f"      Schema: #{channel.schema_id}")

    if result.schemas:
        print(f"\n🗂️  Schemas ({len(result.schemas)}):")
        for schema in result.schemas:
            print(f"   Schema {schema.id}:")
            print(f"      Name: {schema.name}")
            print(f"      Encoding: {schema.encoding}")
            print(f"      Size: {len(schema.data)} bytes")

    print()


def demo_matlab():
    """Demonstrate MATLAB file inspection."""
    print("=" * 80)
    print("MATLAB FILE INSPECTION DEMO")
    print("=" * 80)

    inspector = MatFileInspector("samples/sample_v5.mat")
    result = inspector.inspect()

    print(f"\n📁 File: {result.url}")
    print(f"✓ Valid MAT-File: {result.is_matfile}")
    print(f"📊 Format: {result.file_format}")
    print(f"💾 Size: {result.size_bytes:,} bytes")

    if result.header:
        print(f"\n📋 Header Information:")
        print(f"   Description: {result.header.description[:60]}...")
        print(f"   Version: {result.header.version}")
        print(f"   Endian: {result.header.endian}")

    if result.variables:
        print(f"\n📊 Variables ({len(result.variables)}):")

        # Categorize variables
        scalars = [v for v in result.variables if v.shape == (1, 1)]
        vectors = [v for v in result.variables if len(v.shape) == 2 and min(v.shape) == 1 and v.shape != (1, 1)]
        matrices = [v for v in result.variables if len(v.shape) >= 2 and min(v.shape) > 1]
        arrays_3d = [v for v in result.variables if len(v.shape) > 2]
        complex_vars = [v for v in result.variables if v.is_complex]

        print(f"   Scalars: {len(scalars)}")
        print(f"   Vectors: {len(vectors)}")
        print(f"   Matrices (2D): {len(matrices)}")
        print(f"   Arrays (3D+): {len(arrays_3d)}")
        print(f"   Complex: {len(complex_vars)}")

        print(f"\n🔢 Variable Details:")
        for var in result.variables[:5]:  # Show first 5
            flags = []
            if var.is_complex:
                flags.append("complex")
            if var.is_sparse:
                flags.append("sparse")

            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"   • {var.name}: {var.dtype} {var.shape}{flag_str}")
            print(f"     Size: {var.size_bytes} bytes")

        if len(result.variables) > 5:
            print(f"   ... and {len(result.variables) - 5} more variables")

        # Statistics
        total_size = sum(v.size_bytes for v in result.variables)
        print(f"\n📈 Statistics:")
        print(f"   Total data size: {total_size:,} bytes")
        print(f"   Average var size: {total_size // len(result.variables):,} bytes")

    print()


def demo_comparison():
    """Compare file formats side-by-side."""
    print("=" * 80)
    print("FORMAT COMPARISON")
    print("=" * 80)

    mcap_result = McapFileInspector("samples/sample.mcap").inspect()
    mat_result = MatFileInspector("samples/sample_v5.mat").inspect()

    print(f"\n{'Property':<25} {'MCAP':<25} {'MATLAB':<25}")
    print("-" * 75)
    print(f"{'File Size':<25} {mcap_result.size_bytes:>20,} B  {mat_result.size_bytes:>20,} B")
    print(f"{'Primary Entities':<25} {f'{len(mcap_result.channels or [])} channels':>25} {f'{len(mat_result.variables or [])} variables':>25}")
    print(f"{'Has Header':<25} {'✓' if mcap_result.header else '✗':>25} {'✓' if mat_result.header else '✗':>25}")
    print(f"{'Has Footer':<25} {'✓' if mcap_result.footer else '✗':>25} {'N/A':>25}")
    print(f"{'Compression Support':<25} {'✓':>25} {'✓ (v7)':>25}")
    print(f"{'Remote URL Support':<25} {'✓':>25} {'✓':>25}")

    print()


def main():
    """Run all demos."""
    try:
        demo_mcap()
        demo_matlab()
        demo_comparison()

        print("=" * 80)
        print("✅ Demo completed successfully!")
        print("=" * 80)
        print()
        print("Try the inspectors yourself:")
        print("  python tests/test_mcap_inspector.py samples/sample.mcap")
        print("  python tests/test_mat_inspector.py samples/sample_v5.mat")
        print("  python scripts/compare_inspectors.py samples/sample.mcap samples/sample_v5.mat")
        print()

    except FileNotFoundError as e:
        print(f"❌ Error: Sample file not found - {e}")
        print("Run the following to create sample files:")
        print("  python create_sample_mcap.py")
        print("  python create_sample_mat.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
