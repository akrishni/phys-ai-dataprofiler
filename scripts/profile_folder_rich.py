#!/usr/bin/env python3
"""
Rich folder profiler that generates detailed JSON profiles for each file.

Output format matches the example provided, with channel-level statistics,
quality findings, and timing analysis.
"""

import sys
import json
from pathlib import Path
from typing import List

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.inspectors.rich import RichInspector, RichProfile


def profile_folder(folder_path: str, recursive: bool = True, ext_filter: str = None) -> List[RichProfile]:
    """Profile all files in a folder with rich inspection."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if folder.is_file():
        # Single file
        return [RichInspector(str(folder)).inspect()]

    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    # Find all files
    if recursive:
        if ext_filter:
            files = list(folder.rglob(f"*.{ext_filter.lstrip('.')}"))
        else:
            files = list(folder.rglob("*"))
    else:
        if ext_filter:
            files = list(folder.glob(f"*.{ext_filter.lstrip('.')}"))
        else:
            files = list(folder.glob("*"))

    # Filter to only regular files and relevant extensions
    valid_exts = {".mcap", ".mat"}
    files = [f for f in files if f.is_file() and (not ext_filter or f.suffix in valid_exts)]

    # Profile each file
    profiles = []
    for filepath in sorted(files):
        print(f"Profiling: {filepath.name}...", file=sys.stderr)
        try:
            profile = RichInspector(str(filepath)).inspect()
            profiles.append(profile)
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)

    return profiles


def format_profile(profile: RichProfile) -> dict:
    """Format profile as dictionary for JSON output."""
    return {
        "path": profile.path,
        "format": profile.format,
        "size_bytes": profile.size_bytes,
        "indexed": profile.indexed,
        "duration_s": profile.duration_s,
        "time_base": profile.time_base,
        "channels": [
            {
                "topic": ch.topic,
                "schema": ch.schema,
                "modality": ch.modality,
                "codec": ch.codec,
                "resolution": ch.resolution,
                "dims": ch.dims,
                "msg_count": ch.msg_count,
                "rate_hz": ch.rate_hz,
                "jitter_ms": ch.jitter_ms,
                "gaps_over_3x": ch.gaps_over_3x,
                "max_gap_ms": ch.max_gap_ms,
                "nan_count": ch.nan_count,
            }
            for ch in profile.channels
        ],
        "findings": [
            {
                "severity": f.severity,
                "type": f.type,
                "topic": f.topic,
                "detail": f.detail,
            }
            for f in profile.findings
        ],
        "inspect_cost_s": round(profile.inspect_cost_s, 2),
    }


def print_summary_table(profiles: List[RichProfile]):
    """Print a summary table of all files."""
    print("\n" + "="*100, file=sys.stderr)
    print("FOLDER PROFILE SUMMARY", file=sys.stderr)
    print("="*100, file=sys.stderr)

    total_size = sum(p.size_bytes for p in profiles)
    total_duration = sum(p.duration_s for p in profiles if p.duration_s)
    total_channels = sum(len(p.channels) for p in profiles)
    total_messages = sum(sum(ch.msg_count for ch in p.channels) for p in profiles)

    print(f"\nFiles: {len(profiles)}", file=sys.stderr)
    print(f"Total Size: {total_size / 1e9:.2f} GB", file=sys.stderr)
    print(f"Total Duration: {total_duration:.1f} s", file=sys.stderr)
    print(f"Total Channels: {total_channels}", file=sys.stderr)
    print(f"Total Messages: {total_messages:,}", file=sys.stderr)

    # Warnings and errors
    errors = sum(1 for p in profiles for f in p.findings if f.severity == "error")
    warnings = sum(1 for p in profiles for f in p.findings if f.severity == "warn")

    if errors > 0:
        print(f"\n⚠️  Errors: {errors}", file=sys.stderr)
    if warnings > 0:
        print(f"⚠️  Warnings: {warnings}", file=sys.stderr)

    # Per-file summary
    print(f"\n{'File':<40} {'Size':>10} {'Duration':>10} {'Channels':>10} {'Msgs':>12} {'Findings':>10}", file=sys.stderr)
    print("-"*100, file=sys.stderr)

    for profile in profiles:
        filename = Path(profile.path).name
        if len(filename) > 37:
            filename = filename[:34] + "..."

        size_str = f"{profile.size_bytes / 1e6:.1f}M"
        duration_str = f"{profile.duration_s:.1f}s" if profile.duration_s else "-"
        channels_str = str(len(profile.channels))
        msgs_str = f"{sum(ch.msg_count for ch in profile.channels):,}"
        findings_str = f"{len(profile.findings)}"

        print(f"{filename:<40} {size_str:>10} {duration_str:>10} {channels_str:>10} {msgs_str:>12} {findings_str:>10}", file=sys.stderr)

    print("="*100 + "\n", file=sys.stderr)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Rich profiling of MCAP and MATLAB files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile current directory
  python scripts/profile_folder_rich.py .

  # Profile specific folder
  python scripts/profile_folder_rich.py /path/to/data

  # Only MCAP files
  python scripts/profile_folder_rich.py /path/to/data --ext .mcap

  # Non-recursive
  python scripts/profile_folder_rich.py /path/to/data --no-recursive

  # Save to file
  python scripts/profile_folder_rich.py /path/to/data --output profiles.json

  # Pretty print
  python scripts/profile_folder_rich.py /path/to/data --pretty
        """
    )

    parser.add_argument(
        "folder",
        help="Folder path to profile (or single file)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories"
    )
    parser.add_argument(
        "--ext",
        help="Filter by extension (.mcap or .mat)"
    )
    parser.add_argument(
        "--output",
        help="Save JSON output to file"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip summary table (only JSON)"
    )

    args = parser.parse_args()

    try:
        profiles = profile_folder(
            args.folder,
            recursive=not args.no_recursive,
            ext_filter=args.ext
        )

        if not profiles:
            print("No files found to profile", file=sys.stderr)
            sys.exit(1)

        # Format profiles as JSON
        output_data = [format_profile(p) for p in profiles]

        # Print summary table
        if not args.no_summary:
            print_summary_table(profiles)

        # Output JSON
        json_str = json.dumps(output_data, indent=2 if args.pretty else None)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_str)
            print(f"✓ Profiles saved to: {args.output}", file=sys.stderr)
        else:
            print(json_str)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
