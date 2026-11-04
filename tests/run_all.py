#!/usr/bin/env python3
"""Run every test script and report which failed.

    python tests/run_all.py            # offline checks only (what CI runs on every push)
    python tests/run_all.py --hub      # also the checks that profile small public Hub datasets

Each tests/test_*.py is a standalone script that exits non-zero on its first
failed check. This runs them one by one in this Python, prints a summary, and
exits 1 if any failed. A test that needs an optional dependency (rosbags for
the bag tests) is reported as failed, not skipped, when that dependency is
missing.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUB_CAPABLE = {"test_parquet_inspector.py", "test_rosbag_inspector.py"}


def main() -> int:
    hub = "--hub" in sys.argv
    scripts = sorted(HERE.glob("test_*.py"))
    results = []
    for script in scripts:
        args = [sys.executable, str(script)] + (["--hub"] if hub and script.name in HUB_CAPABLE else [])
        print(f"\n===== {script.name}{' --hub' if len(args) > 2 else ''}", flush=True)
        t0 = time.time()
        proc = subprocess.run(args, cwd=HERE.parent)
        results.append((script.name, proc.returncode, time.time() - t0))

    print("\n===== summary")
    for name, code, secs in results:
        print(f"  {'ok  ' if code == 0 else 'FAIL'}  {name}  ({secs:.1f} s)" + (f"  exit {code}" if code else ""))
    failed = [r for r in results if r[1] != 0]
    print(f"{len(results) - len(failed)} of {len(results)} test scripts passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
