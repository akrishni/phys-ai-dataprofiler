#!/usr/bin/env python3
"""Checks for the full-read MCAP inspector.

    python tests/test_mcap_inspector.py              # checks against samples/ (exits non-zero on failure)
    python tests/test_mcap_inspector.py FILE_OR_URL  # print what the inspector finds in one file
"""

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


ROOT = Path(__file__).resolve().parents[1]


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def run_checks() -> None:
    r = McapFileInspector(str(ROOT / "samples" / "sample.mcap")).inspect()
    check(r.is_mcap and r.error is None, "sample.mcap is recognised and parses")
    check(r.header is not None and (r.header.profile, r.header.library) == ("test", "sample_creator"),
          "header profile and library")
    check(r.topics == ["/sensors/gps", "/sensors/imu"], f"topics {r.topics}")
    check(len(r.channels or []) == 2 and all(c.message_encoding for c in r.channels), "two channels with encodings")
    check([s.name for s in r.schemas or []] == ["example.Point"], "schema name")
    check(r.footer is not None and r.footer.summary_start > 0, "footer points to a summary section")

    bad = McapFileInspector(str(ROOT / "samples" / "test.txt")).inspect()
    check(not bad.is_mcap, "a text file is not an MCAP")
    missing = McapFileInspector(str(ROOT / "samples" / "does_not_exist.mcap")).inspect()
    check(not missing.is_mcap and missing.error, "a missing file reports an error instead of raising")
    print("all checks passed")


def main():
    if len(sys.argv) < 2:
        run_checks()
        return
    url = sys.argv[1]
    print(f"Inspecting: {url}\n")
    print_result(McapFileInspector(url).inspect())


if __name__ == "__main__":
    main()
