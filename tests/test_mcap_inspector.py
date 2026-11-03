#!/usr/bin/env python3
"""Test script for MCAP file inspector."""

import sys
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.inspectors.mcap import McapFileInspector


def print_result(result):
    """Pretty print the inspection result."""
    print(f"URL: {result.url}")
    print(f"Is MCAP: {result.is_mcap}")
    print(f"Size: {result.size_bytes} bytes" if result.size_bytes else "Size: Unknown")
    print(f"MCAP Library Available: {result.mcap_available}")

    if result.error:
        print(f"Error: {result.error}")
        return

    if result.header:
        print("\n=== Header ===")
        print(f"Profile: {result.header.profile}")
        print(f"Library: {result.header.library}")

    if result.footer:
        print("\n=== Footer ===")
        print(f"Summary Start: {result.footer.summary_start}")
        print(f"Summary Offset Start: {result.footer.summary_offset_start}")
        print(f"Summary CRC: {result.footer.summary_crc}")

    if result.topics:
        print(f"\n=== Topics ({len(result.topics)}) ===")
        for topic in result.topics:
            print(f"  - {topic}")

    if result.channels:
        print(f"\n=== Channels ({len(result.channels)}) ===")
        for channel in result.channels:
            print(f"  Channel {channel.id}:")
            print(f"    Topic: {channel.topic}")
            print(f"    Encoding: {channel.message_encoding}")
            print(f"    Schema ID: {channel.schema_id}")
            if channel.metadata:
                print(f"    Metadata: {channel.metadata}")

    if result.schemas:
        print(f"\n=== Schemas ({len(result.schemas)}) ===")
        for schema in result.schemas:
            print(f"  Schema {schema.id}:")
            print(f"    Name: {schema.name}")
            print(f"    Encoding: {schema.encoding}")
            print(f"    Data size: {len(schema.data)} bytes")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_inspector.py <mcap_file_or_url>")
        sys.exit(1)

    url = sys.argv[1]
    print(f"Inspecting: {url}\n")

    inspector = McapFileInspector(url)
    result = inspector.inspect()

    print_result(result)


if __name__ == "__main__":
    main()
