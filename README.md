# phys-ai-dataprofiler

[![tests](https://github.com/akrishni/phys-ai-dataprofiler/actions/workflows/tests.yml/badge.svg)](https://github.com/akrishni/phys-ai-dataprofiler/actions/workflows/tests.yml)

Profile physical-AI and lab datasets on the Hugging Face Hub **without downloading them**, and check
them against what your model needs.

It reads only the bytes that hold metadata (headers, indexes, footers) through HTTP range requests,
usually well under 1% of each file, and reports streams, message counts, rates, jitter, gaps, episodes,
NaN counts and problems such as truncated or unindexed recordings. A requirements file turns that into a
pass / fail / unknown verdict, and `compare` ranks candidate datasets side by side.

| Format | What is read |
|---|---|
| MCAP (incl. ROS 2 MCAP storage) | header, footer, summary, message indexes |
| ROS 1 bag (V2.0) | bag index and per-chunk index data; compressed payloads are skipped |
| ROS 2 bag (SQLite `.db3`) | schema, topics, metadata and the leaf pages of the messages table |
| MATLAB `.mat` (v5, v7, v7.3) | header and each variable's header; float data within a NaN budget |
| Parquet / LeRobot | footer, column statistics, timestamp and episode columns |

Measured on public Hub data: a 5.8 GB ROS 1 bag profiled from 533 KB, a 2.6 GB MAT v7.3 file from 808 KB.

## Install

```bash
pip install phys-ai-dataprofiler
pip install "phys-ai-dataprofiler[mat73]"      # MAT v7.3 (HDF5) files; needs Python 3.13 or older for h5py
```

Python 3.10 or newer. Public datasets need no account; for gated or private ones set `HF_TOKEN` (or run
`hf auth login`). The token is only sent to huggingface.co, never to the CDN hosts that serve file bytes.

## Quick start

```bash
phys-ai-dataprofiler org/dataset --dry-run              # what files are there, and how big
phys-ai-dataprofiler org/dataset --max-files 10         # profile a sample -> reports/<repo>__<time>.md/.json

phys-ai-dataprofiler requirements list                  # built-in presets
phys-ai-dataprofiler requirements show teleop-imitation > req.yaml   # start your own, edit the numbers
phys-ai-dataprofiler check org/dataset -r req.yaml      # verdict; exit 0 pass, 1 fail, 4 unknown
phys-ai-dataprofiler compare org/a org/b -r req.yaml    # rank candidates against the same requirements
```

Nothing is downloaded unless you ask for it (`--mode download`, or `--full-read-max-mb` for small files).
Nothing is ever written to the Hub.

## Requirements and verdicts

Say what your model needs in a YAML file, and the profiler answers pass, fail or unknown for each
requirement, with evidence and, for unknowns, how to settle them. Checks are deterministic rules, with no
model call. Full reference: [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

```bash
phys-ai-dataprofiler check org/dataset -r preset:teleop-imitation        # exit 0 pass, 1 fail, 4 unknown
phys-ai-dataprofiler compare org/a org/b -r my_requirements.yaml          # rank candidate datasets
phys-ai-dataprofiler check --report reports/run.json -r my_requirements.yaml   # re-check offline
```

```yaml
streams:
  - name: camera
    match: {modality: video}
    min_rate_hz: 25
  - name: joint state
    match: {topic: ["/joint_states", "observation.state"]}
    min_rate_hz: 50
    max_gap_ms: 100
dataset:
  min_total_hours: 20
  min_episodes: 500
```

A number that was sampled and lies within 10% of the threshold, a NaN count the format cannot provide,
or a file that failed to profile makes the result **unknown**, never a guessed pass.

## ROS bag support

`.bag` (ROS 1, format V2.0) and `.db3` (ROS 2, SQLite storage) files are profiled from byte ranges, like
MCAP. ROS 2 bags in MCAP storage are plain `.mcap` files and use the MCAP path. Code:
`phys_ai_dataprofiler/remote/rosbag.py`.

- **ROS 1:** the bag header points to the index at the end of the file (connections plus per-chunk counts
  and time ranges). Per-message timestamps come from the IndexData records after each chunk, 12 bytes per
  message, one request per chunk. Chunk payloads are never read, so compression (bz2, lz4) makes no
  difference. Over `--msgidx-budget-mb`, chunks are sampled; counts stay exact. Bags with no index
  (recorder killed) are walked record by record and reported `unindexed_bag`; truncated bags are reported
  `truncated`. Old V1.2 bags are identified but not parsed.
- **ROS 2 SQLite:** a small read-only SQLite reader descends the `messages` table B-tree. Each row's
  `topic_id` and `timestamp` sit at the start of its leaf cell, so message payloads and their overflow
  pages are never read. Counts and duration come from the `metadata` table, or from the `metadata.yaml`
  next to the `.db3` when the database has none. Over budget, runs of leaf pages are sampled and the
  start and end times come from `timestamp_idx`. Without any metadata, sampled per-topic counts are
  estimates, reported as `estimated_counts`.

Timing statistics use the same code as MCAP. Tests compare every case against a full read through
`rosbags`: `python tests/test_rosbag_inspector.py` (add `--hub` for two public bags).

| Dataset | Result | Read |
|---|---|---|
| `Adapting/nuscenes-rosbags` `103.bag` (ROS 1) | 14 topics, 1,585 chunks, 19.5 s | 533 KB of 5.8 GB |
| `dvar1ch/ros2-bag-datasets` (ROS 2 SQLite, 2 files) | `/image` at 30 Hz, metadata v9, jazzy | 1.5 MB of 668 MB |
| `liuxh010/IndoorLocalization_...` (ROS 1), `UniflexAI/rosbag_lekiwi_exp_result` (ROS 2) | per-topic stats identical to a full read | 146 KB of 891 KB; 2.1 MB of 18.6 MB |

## Parquet support

Parquet keeps its schema, row groups and per-column statistics in a footer at the end of the file, so the
profiler reads the footer first (typically under 100 KB, whatever the file size). It reports rows, row groups,
every column's type, null count, min and max, compressed size and codec.

If a time column is present (`timestamp`, `log_time`, `time`, ...), it also reads that column, and
`episode_index` when present (the LeRobot convention). Their byte ranges are known from the footer, so the cost
is computed up front and capped by `--column-budget-mb` (default 16); beyond that, row groups are sampled and
the report says so. From these it derives frame rate, jitter, gaps, episode count and lengths, and flags
dropped frames, short episodes and non-monotonic time. Image, text and state columns are never read.

Units are inferred: float seconds (LeRobot), integer epoch s/ms/us/ns, and Arrow timestamp types.
NaN/inf counts are not in Parquet statistics, so float columns are read too, capped by `--nan-budget-mb`
(default 32); beyond that, row groups are sampled and the report says so. `--nan-budget-mb 0` turns this off.

| Dataset | Result | Read |
|---|---|---|
| `lerobot/pusht` (LeRobot v3) | 206 episodes at 10 Hz, matching `meta/info.json` | 205 KB of 659 KB |
| `nvidia/Nemotron-Personas-Vietnam` (text) | 90,910 rows, 21 columns per file | 143 KB of 194 MB |

## Usage

```bash
# Choose files: by glob, count, size, or exact path
phys-ai-dataprofiler org/dataset --include "episodes/*.mcap" --max-files 5
phys-ai-dataprofiler org/dataset --max-file-mb 500 --max-total-mb 5000
phys-ai-dataprofiler org/dataset --file logs/run1.mcap --file calib/cam.mat

# Read budgets per file (beyond them, data is sampled and the report says so)
phys-ai-dataprofiler org/dataset --nan-budget-mb 256        # float data scanned for NaNs; 0 turns it off
phys-ai-dataprofiler org/dataset --msgidx-budget-mb 128     # MCAP / ROS 1 message-index bytes
phys-ai-dataprofiler org/dataset --column-budget-mb 64      # Parquet time and episode columns

# Fetch small files whole, or download everything
phys-ai-dataprofiler org/dataset --full-read-max-mb 50      # deleted afterwards unless --keep-downloads
phys-ai-dataprofiler org/dataset --mode download --max-total-mb 2000
phys-ai-dataprofiler org/dataset --mode download --offline  # re-run on an earlier download

# Model repos, other revisions
phys-ai-dataprofiler org/model --repo-type model --revision v1.2
```

Other options: `--workers` (parallel range requests, default 16), `--block-kb` (read block size),
`--token` (else `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or the `hf auth login` cache). Run with `--help` for
the full list.

Reports go to `./reports/<repo>__<timestamp>.{json,md}` (`--report-dir`, `--report-name`) and record, per
file, how many bytes were read in how many requests. Truncated MCAPs and unindexed bags, which a full read
cannot open, are walked record by record and flagged. Exit codes: 0 ok, 1 a file failed (or a `check`
failed), 2 usage or connection error, 3 nothing to profile, 4 a `check` was unknown.

From Python:

```python
from phys_ai_dataprofiler.connectors.huggingface import HuggingFaceConnector
from phys_ai_dataprofiler.remote.inspector import inspect_mcap_remote, inspect_mat_remote, sniff_format

hf = HuggingFaceConnector("org/dataset")
hf.connect()
selected, skipped = hf.select_files(hf.list_files(), max_files=10)
for f in selected:
    rf = hf.open_range(f, block_size=64 * 1024)
    fmt = sniff_format(rf, f.path)
    if fmt in ("mcap", "mat"):
        result, profile = inspect_mcap_remote(rf) if fmt == "mcap" else inspect_mat_remote(rf)
        print(f.path, rf.bytes_fetched, "of", rf.size, "bytes read")
```

More documentation in `docs/`: [requirements reference](docs/REQUIREMENTS.md), and the
[full-read inspectors](docs/LOCAL_INSPECTORS.md) for local files, which the range readers are tested against.

## Development

```bash
git clone https://github.com/akrishni/phys-ai-dataprofiler
cd phys-ai-dataprofiler
pip install -e ".[mat73,rosbag,dev]"
python tests/run_all.py                  # every test script, with a summary
python tests/run_all.py --hub            # also profile small public Hub datasets
python tests/test_rosbag_inspector.py    # or run one script on its own
```

CI (`.github/workflows/tests.yml`) runs `tests/run_all.py` on every push and pull request, on Python 3.10
to 3.13 on Linux and 3.13 on macOS, and builds the package and smoke-tests the installed CLI. The Hub
checks need network access and are only run on demand: Actions tab, "tests", "Run workflow", tick
"hub".

```
phys_ai_dataprofiler/
  connectors/huggingface.py   Hub connector and the default CLI command
  cli.py                      check, compare, requirements subcommands
  requirements.py evaluate.py compare.py   requirements files, verdicts, comparisons
  presets/                    built-in requirements files
  remote/                     byte-range readers: MCAP, MAT, Parquet, ROS bags
  inspectors/                 full-read inspectors for local files
tests/  samples/  scripts/  docs/
```

## License and citation

MIT License, copyright (c) 2026 Anand Krishniyer. See [LICENSE](LICENSE).

If you use this tool in research, please cite it; GitHub's "Cite this repository" button (from
[CITATION.cff](CITATION.cff)) gives a ready-made reference.
