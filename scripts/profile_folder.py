#!/usr/bin/env python3
"""
Folder Profiler for MCAP and MATLAB files.

Recursively inspects all files in a directory and produces
aggregate statistics and summaries.
"""

import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phys_ai_dataprofiler.inspectors.mcap import McapFileInspector
from phys_ai_dataprofiler.inspectors.mat import MatFileInspector


@dataclass
class FileProfile:
    path: str
    file_type: str  # "mcap", "mat", "unknown"
    size_bytes: int
    is_valid: bool
    error: str | None
    metadata: Dict[str, Any]


@dataclass
class FolderStats:
    total_files: int
    total_size_bytes: int
    mcap_files: int
    mat_files: int
    unknown_files: int
    valid_files: int
    failed_files: int
    profiles: List[FileProfile]


def inspect_file(filepath: Path) -> FileProfile:
    """Inspect a single file and return its profile."""
    ext = filepath.suffix.lower()
    size = filepath.stat().st_size

    # Try MCAP first
    if ext == ".mcap" or ext == "":
        result = McapFileInspector(str(filepath)).inspect()
        if result.is_mcap:
            metadata = {
                "format": "mcap",
                "profile": result.header.profile if result.header else None,
                "library": result.header.library if result.header else None,
                "topics": len(result.topics) if result.topics else 0,
                "channels": len(result.channels) if result.channels else 0,
                "schemas": len(result.schemas) if result.schemas else 0,
                "topic_list": result.topics,
            }
            return FileProfile(
                path=str(filepath),
                file_type="mcap",
                size_bytes=size,
                is_valid=not result.error,
                error=result.error,
                metadata=metadata,
            )

    # Try MATLAB
    if ext == ".mat" or ext == "":
        result = MatFileInspector(str(filepath)).inspect()
        if result.is_matfile:
            metadata = {
                "format": "mat",
                "version": result.file_format,
                "variables": len(result.variables) if result.variables else 0,
                "variable_names": result.variable_names,
                "variable_types": {},
            }

            # Categorize variables
            if result.variables:
                types = defaultdict(int)
                for var in result.variables:
                    types[var.dtype] += 1
                    if var.is_complex:
                        types["complex"] += 1
                    if var.is_sparse:
                        types["sparse"] += 1
                metadata["variable_types"] = dict(types)

            return FileProfile(
                path=str(filepath),
                file_type="mat",
                size_bytes=size,
                is_valid=not result.error,
                error=result.error,
                metadata=metadata,
            )

    # Unknown file type
    return FileProfile(
        path=str(filepath),
        file_type="unknown",
        size_bytes=size,
        is_valid=False,
        error="Unknown file type",
        metadata={},
    )


def profile_folder(folder_path: str, recursive: bool = True, pattern: str = "*") -> FolderStats:
    """Profile all files in a folder."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    # Find all files
    if recursive:
        files = list(folder.rglob(pattern))
    else:
        files = list(folder.glob(pattern))

    # Filter to only regular files
    files = [f for f in files if f.is_file()]

    # Profile each file
    profiles = []
    mcap_count = 0
    mat_count = 0
    unknown_count = 0
    valid_count = 0
    failed_count = 0
    total_size = 0

    for filepath in files:
        profile = inspect_file(filepath)
        profiles.append(profile)

        total_size += profile.size_bytes

        if profile.file_type == "mcap":
            mcap_count += 1
        elif profile.file_type == "mat":
            mat_count += 1
        else:
            unknown_count += 1

        if profile.is_valid:
            valid_count += 1
        else:
            failed_count += 1

    return FolderStats(
        total_files=len(files),
        total_size_bytes=total_size,
        mcap_files=mcap_count,
        mat_files=mat_count,
        unknown_files=unknown_count,
        valid_files=valid_count,
        failed_files=failed_count,
        profiles=profiles,
    )


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def print_summary(stats: FolderStats, folder_path: str):
    """Print folder profile summary."""
    print("=" * 100)
    print(f"FOLDER PROFILE: {folder_path}")
    print("=" * 100)

    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'─' * 100}")
    print(f"Total Files:          {stats.total_files:>6}")
    print(f"Total Size:           {format_size(stats.total_size_bytes):>12}")
    print(f"Valid Files:          {stats.valid_files:>6} ({stats.valid_files/stats.total_files*100:.1f}%)" if stats.total_files > 0 else "Valid Files:          0")
    print(f"Failed Files:         {stats.failed_files:>6}")

    # File type breakdown
    print(f"\n📁 FILE TYPE BREAKDOWN")
    print(f"{'─' * 100}")
    print(f"MCAP Files:           {stats.mcap_files:>6}")
    print(f"MATLAB Files:         {stats.mat_files:>6}")
    print(f"Unknown/Other:        {stats.unknown_files:>6}")

    # MCAP-specific statistics
    mcap_profiles = [p for p in stats.profiles if p.file_type == "mcap" and p.is_valid]
    if mcap_profiles:
        print(f"\n📡 MCAP FILES ANALYSIS")
        print(f"{'─' * 100}")

        total_topics = sum(p.metadata.get("topics", 0) for p in mcap_profiles)
        total_channels = sum(p.metadata.get("channels", 0) for p in mcap_profiles)
        total_schemas = sum(p.metadata.get("schemas", 0) for p in mcap_profiles)

        print(f"Total Topics:         {total_topics:>6}")
        print(f"Total Channels:       {total_channels:>6}")
        print(f"Total Schemas:        {total_schemas:>6}")
        print(f"Avg Topics/File:      {total_topics/len(mcap_profiles):>6.1f}")
        print(f"Avg Channels/File:    {total_channels/len(mcap_profiles):>6.1f}")

        # Unique topics across all files
        all_topics = set()
        for p in mcap_profiles:
            if p.metadata.get("topic_list"):
                all_topics.update(p.metadata["topic_list"])

        if all_topics:
            print(f"\nUnique Topics Across All Files ({len(all_topics)}):")
            for topic in sorted(all_topics):
                print(f"  • {topic}")

    # MATLAB-specific statistics
    mat_profiles = [p for p in stats.profiles if p.file_type == "mat" and p.is_valid]
    if mat_profiles:
        print(f"\n🔢 MATLAB FILES ANALYSIS")
        print(f"{'─' * 100}")

        total_vars = sum(p.metadata.get("variables", 0) for p in mat_profiles)

        # Version breakdown
        versions = defaultdict(int)
        for p in mat_profiles:
            version = p.metadata.get("version", "unknown")
            versions[version] += 1

        print(f"Total Variables:      {total_vars:>6}")
        print(f"Avg Variables/File:   {total_vars/len(mat_profiles):>6.1f}")

        print(f"\nFormat Versions:")
        for version, count in sorted(versions.items()):
            print(f"  v{version}: {count} file(s)")

        # Aggregate variable types
        all_types = defaultdict(int)
        for p in mat_profiles:
            for dtype, count in p.metadata.get("variable_types", {}).items():
                all_types[dtype] += count

        if all_types:
            print(f"\nVariable Types (aggregate):")
            for dtype, count in sorted(all_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {dtype:<20} {count:>6}")

    # File listing
    print(f"\n📋 FILE LISTING")
    print(f"{'─' * 100}")
    print(f"{'Path':<60} {'Type':<8} {'Size':<12} {'Status':<10} {'Details'}")
    print(f"{'─' * 100}")

    for profile in sorted(stats.profiles, key=lambda p: (p.file_type, p.path)):
        rel_path = Path(profile.path).name
        if len(rel_path) > 57:
            rel_path = "..." + rel_path[-54:]

        status = "✓ Valid" if profile.is_valid else "✗ Failed"

        # Format details based on file type
        details = ""
        if profile.file_type == "mcap" and profile.is_valid:
            topics = profile.metadata.get("topics", 0)
            channels = profile.metadata.get("channels", 0)
            details = f"{topics} topics, {channels} channels"
        elif profile.file_type == "mat" and profile.is_valid:
            vars = profile.metadata.get("variables", 0)
            version = profile.metadata.get("version", "?")
            details = f"{vars} vars, v{version}"
        elif profile.error:
            details = profile.error[:40]

        print(f"{rel_path:<60} {profile.file_type.upper():<8} {format_size(profile.size_bytes):<12} {status:<10} {details}")

    # Failed files section
    failed_profiles = [p for p in stats.profiles if not p.is_valid]
    if failed_profiles:
        print(f"\n❌ FAILED FILES ({len(failed_profiles)})")
        print(f"{'─' * 100}")
        for profile in failed_profiles:
            print(f"  {Path(profile.path).name}")
            print(f"    Error: {profile.error}")

    print(f"\n{'=' * 100}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile all MCAP and MATLAB files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile current directory
  python profile_folder.py .

  # Profile specific folder recursively
  python profile_folder.py /path/to/data

  # Non-recursive profiling
  python profile_folder.py /path/to/data --no-recursive

  # Only .mcap files
  python profile_folder.py /path/to/data --pattern "*.mcap"
        """
    )

    parser.add_argument(
        "folder",
        help="Folder path to profile"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories"
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="File pattern to match (default: *)"
    )
    parser.add_argument(
        "--output",
        help="Save output to file instead of printing"
    )

    args = parser.parse_args()

    try:
        print("Profiling folder...")
        stats = profile_folder(
            args.folder,
            recursive=not args.no_recursive,
            pattern=args.pattern
        )

        if args.output:
            # Redirect output to file
            import sys
            original_stdout = sys.stdout
            with open(args.output, 'w') as f:
                sys.stdout = f
                print_summary(stats, args.folder)
                sys.stdout = original_stdout
            print(f"✓ Profile saved to: {args.output}")
        else:
            print_summary(stats, args.folder)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
