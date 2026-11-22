# phys-ai-dataprofiler

[![tests](https://github.com/akrishni/phys-ai-dataprofiler/actions/workflows/tests.yml/badge.svg)](https://github.com/akrishni/phys-ai-dataprofiler/actions/workflows/tests.yml)

Profile physical-AI and lab datasets on the Hugging Face Hub **without downloading them**, and check
them against what your model needs.

It reads only the bytes that hold metadata (headers, indexes, footers) through HTTP range requests, and
reports streams, message counts, rates, jitter, gaps, episodes, NaN counts, and problems such as
truncated or unindexed recordings. LeRobot datasets are covered as a whole: state and actions from the
Parquet tables, each camera from its MP4s. A requirements file turns that into a
pass / fail / unknown verdict, and `compare` ranks candidate datasets side by side.

| Format | What is read |
|---|---|
| MCAP (incl. ROS 2 MCAP storage) | header, footer, summary, message indexes |
| ROS 1 bag (V2.0) | bag index and per-chunk index data; compressed payloads are skipped |
| ROS 2 bag (SQLite `.db3`) | schema, topics, metadata and the leaf pages of the messages table |
| MATLAB `.mat` (v5, v7, v7.3) | header and each variable's header; float data within a NaN budget |
| Parquet / LeRobot | footer, column statistics, timestamp and episode columns |
| MP4 / LeRobot videos | the `moov` index: codec, resolution, per-frame timestamps, keyframes; frames are never decoded |

On large files that is usually well under 1%: a 5.8 GB ROS 1 bag is profiled from 533 KB, a 2.4 GB MAT v7.3
file from 232 KB, a 493 MB camera video from 444 KB. Small files cost proportionally more, because headers and indexes have a minimum size and
the NaN scan reads float data up to `--nan-budget-mb` (default 32 MB per file): a 659 KB LeRobot file is
read in full. All figures in this README were measured with default settings.

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

`dataset.min_stream_overlap_pct` checks that the required streams record together. It's computed per
recording: per file for MCAP and bags, per episode for LeRobot (table against each camera's video). It
catches a camera that stops halfway, or a video cut shorter than its episode, which each stream's own
checks miss. See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md#stream-overlap).

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
| `Adapting/nuscenes-rosbags` `103.bag` (ROS 1) | 14 topics, 1,585 chunks, 19.5 s | 533 KB of 5.8 GB (0.01%) |
| `dvar1ch/ros2-bag-datasets` (ROS 2 SQLite, 2 files) | `/image` at 30 Hz, metadata v9, jazzy | 1.5 MB of 668 MB (0.2%) |
| `UniflexAI/rosbag_lekiwi_exp_result` (ROS 2) | per-topic stats identical to a full read | 2.0 MB of 17.7 MB (11%) |
| `liuxh010/IndoorLocalization_...` (ROS 1) | per-topic stats identical to a full read | 142 KB of 870 KB (16%) |

## Parquet support

Parquet keeps its schema, row groups and per-column statistics in a footer at the end of the file, so the
profiler reads the footer first (typically under 100 KB, whatever the file size). It reports rows, row groups,
every column's type, null count, min and max, compressed size and codec.

If a time column is present (`timestamp`, `log_time`, `time`, ...), it also reads that column, and
`episode_index` when present (the LeRobot convention). Their byte ranges are known from the footer, so the cost
is computed up front and capped by `--column-budget-mb` (default 16); beyond that, row groups are sampled and
the report says so. From these it derives frame rate, jitter, gaps, episode count and lengths, and flags
dropped frames, short episodes and non-monotonic time. Image and text columns are never read.

Units are inferred: float seconds (LeRobot), integer epoch s/ms/us/ns, and Arrow timestamp types.
NaN/inf counts are not in Parquet statistics, so float columns are read too, capped by `--nan-budget-mb`
(default 32); beyond that, row groups are sampled, or for a file stored as one row group the smallest float
columns are scanned whole and the rest left out, and the report says so. `--nan-budget-mb 0` turns this off.

| Dataset | Result | Read |
|---|---|---|
| `nvidia/Nemotron-Personas-Vietnam` (text, one file) | 90,910 rows, 21 columns | 143 KB of 195 MB (0.07%) |
| `lerobot/pusht` (LeRobot v3) | 206 episodes at 10 Hz, matching `meta/info.json` | 659 KB of 659 KB (100%) |
| same, `--nan-budget-mb 0` | same episodes and rate; no NaN counts | 194 KB of 659 KB (29%) |

`pusht` is small and almost all float state and action columns, so the default NaN scan reads all of it.
On a dataset of that size the full read is cheap; the budget only limits the cost on large files.

## Video (MP4) support

LeRobot stores each camera as MP4 files next to the Parquet tables, `videos/<camera>/chunk-000/file-000.mp4`
(v3) or `videos/chunk-000/<camera>/episode_000000.mp4` (v2). An MP4 keeps its index in one `moov` box, at
the start or the end of the file. The profiler finds it from the 8-byte box headers, skipping the frame
data, and reads only the boxes that give the codec, resolution, timescale, every frame's duration and
the keyframes; the per-frame size and offset tables are skipped too. Frames are never decoded. Fragmented
MP4 is walked one `moof` at a time within `--msgidx-budget-mb`.

Each camera becomes a `video` stream named after its folder (`observation.images.wrist`), with frame
count, rate, jitter and gaps, so `match: {modality: video, topic: "*wrist*"}` works as it does for a
ROS image topic. When the repo has a LeRobot `meta/info.json`, each video is checked against the fps,
resolution and codec it declares. A file whose index was never written (recording cut off) is reported
`no_moov`; one cut off after the index is reported `truncated`.

In a requirements check, LeRobot cameras are judged on the videos and state and actions on the tables,
and hours are counted once, not once per camera; see [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).
With `--max-files N`, the sample takes files from the tables and from every camera in turn.

| Dataset | Result | Read |
|---|---|---|
| `lerobot/droid_1.0.1` `exterior_1_left` (v3) | 96,183 frames at 15 Hz, 320x180 AV1, 6,412 s | 444 KB of 493 MB (0.09%) |
| `lerobot/aloha_sim_insertion_human` `top` (v3) | 25,000 frames at 50 Hz, 640x480 AV1, 500 s | 238 KB of 84 MB (0.28%) |
| `lerobot/pusht` `observation.image` (v3) | 25,650 frames at 10 Hz, 96x96 AV1, matching `meta/info.json` and ffprobe | 201 KB of 6.6 MB (3%) |

Frame timestamps are the container's timeline as the encoder wrote it. Most recorders, LeRobot included,
encode at a fixed rate, so a frame dropped at capture time shows as a gap only if the writer kept real
capture timestamps; the Parquet `timestamp` column is the better check for capture gaps.

## Local files

The same command also works on a local directory or file, with no network and no token:

```bash
phys-ai-dataprofiler ./my-dataset --ext .mcap .mat          # a directory
phys-ai-dataprofiler ./my-dataset/run_042.mcap                # a single file
phys-ai-dataprofiler check ./my-dataset -r preset:teleop-imitation
```

If the argument names an existing path on disk, it's treated as local instead of a Hub repo id; a
directory is walked recursively, skipping dot-files. Local files are read the same way as remote ones,
through byte-range reads (`LocalRangeFile`), so a large `.mcap` or `.mat` file is still profiled from its
header and index rather than read in full. `--mode`, `--download-dir` and `--keep-downloads` have no
effect on a local target.

## Usage

```bash
# Choose files: by glob, count, size, or exact path (--ext .parquet to skip videos)
phys-ai-dataprofiler org/dataset --include "episodes/*.mcap" --max-files 5
phys-ai-dataprofiler org/dataset --max-file-mb 500 --max-total-mb 5000
phys-ai-dataprofiler org/dataset --file logs/run1.mcap --file calib/cam.mat

# Read budgets per file (beyond them, data is sampled and the report says so)
phys-ai-dataprofiler org/dataset --nan-budget-mb 256        # float data scanned for NaNs; 0 turns it off
phys-ai-dataprofiler org/dataset --msgidx-budget-mb 128     # MCAP / ROS 1 message indexes, MP4 fragments
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
  connectors/huggingface.py   Hub connector, local-filesystem dispatch, and the default CLI command
  connectors/localfs.py       local directory/file connector
  connectors/base.py          pieces shared by every connector (file selection, LeRobot video info)
  cli.py                      check, compare, requirements subcommands
  requirements.py evaluate.py compare.py   requirements files, verdicts, comparisons
  presets/                    built-in requirements files
  remote/                     byte-range readers: MCAP, MAT, Parquet, ROS bags, MP4
  inspectors/                 full-read inspectors for local files
tests/  samples/  scripts/  docs/
```

## License and citation

MIT License, copyright (c) 2026 Anand Krishniyer. See [LICENSE](LICENSE).

If you use this tool in research, please cite it; GitHub's "Cite this repository" button (from
[CITATION.cff](CITATION.cff)) gives a ready-made reference.
