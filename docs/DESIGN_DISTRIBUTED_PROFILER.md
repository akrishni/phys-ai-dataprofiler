# Zero-Copy Dataset Profiler for Physical-AI Datasets

Design document. Status: draft v2, 2026-10-17 (v1 had a shared profile registry; removed on review).
Scope: MCAP, ROS 1 bag, ROS 2 bag (MCAP and SQLite storage), MATLAB MAT (v5/v7, v7.3), plus LeRobot v2/v3
sidecars since that is the dominant Hub-native robotics format. Primary source: the Hugging Face Hub; the
reader layer is generic over any HTTP(S)/S3/GCS endpoint that honours Range requests.

---

## 1. Problem

A data scientist is deciding whether a physical-AI dataset (tens of GB to 100+ TB of robot logs) is usable for
the model they are building. They need to know what sensors are present, at what rates, how long, how many
episodes, whether there are gaps or drops, and whether the schema matches their pipeline. Today that means
downloading the files and running something like `mcap info` on each one, which is slow, costs disk and egress,
and often is not even possible on a laptop for a multi-terabyte set.

Goals:

1. **No copies.** Never materialise a dataset file on disk to profile it.
2. **Runs on the user's laptop.** No service, no shared state, no account beyond a Hub token if the dataset is
   private. The tool is a CLI plus an optional local viewer.
3. **Cheap enough to be casual.** A profile should cost milliseconds of CPU and kilobytes of network per file so
   a user can profile several candidate datasets in a coffee break.
4. **Parallel at the file level.** A dataset of thousands of files is profiled by fanning out range requests
   from one machine; the design keeps that parallelism safe under Hub rate limits.
5. **Answers the user's question.** The output is a fit-for-purpose verdict against the user's stated model
   requirements, backed by the profile, not just a table of numbers.
6. **Honest about fidelity.** Every number says what it was derived from (sidecar, index, sample, full scan)
   so the user knows what a metadata-only profile can and cannot claim.

Non-goals (v1): a shared profile store or registry, publishing results back to the Hub, visualising message
content, decoding video, dataset curation or modification.

## 2. Key insight: the formats were built for this

Every target format keeps the information a profiler needs in a small, separately addressable region. Reading
that region over HTTP Range requests yields the same answers as a full scan, except for data-dependent checks.

| Format | Where the metadata lives | Per-message timestamps | Measured on this machine |
|---|---|---|---|
| MCAP | Footer (last 37 B) points to a Summary section at the tail: Schema, Channel, Statistics, ChunkIndex records. SummaryOffset records allow fetching one group only. | MessageIndex records right after each chunk (channel_id, [log_time, offset]); ChunkIndex gives their offsets and lengths. | 163 MB public file: 577 KB read (0.35%), 217 requests, 6.7 s. 109 MB local file: 0.49%, channel stats identical to a full scan on all 39 channels. |
| ROS 1 bag v2.0 | Bag header (first 4 KB) holds `index_pos`; Connection and ChunkInfo records sit from `index_pos` to EOF. | IndexData records right after each chunk (12 B per message). Chunk payloads, the only compressed part, are skipped. | Synthetic 1.6 MB bag: 3.3% read, 27 requests, all 4,500 timestamps recovered. |
| ROS 2 bag | `metadata.yaml` sidecar: per-topic counts, start time, duration, storage id. MCAP storage is the default since Iron and behaves as above. | SQLite storage: `timestamp_idx` is a covering index for timestamps only; per-topic timing needs table leaf pages. | Not yet measured. |
| MAT v5/v7 | 128-byte header, then a chain of data elements; each element's tag gives its length so payloads are skipped. Compressed (v7) elements are inflated only until the name is reached. | n/a (no time axis in the container) | Variable list identical to scipy on both sample files; one 64 KB block per few variables. |
| MAT v7.3 (HDF5) | 128-byte MATLAB header, HDF5 superblock at offset 512, then object headers and B-trees. `h5py` over a block-cached range file lists top-level variables, struct fields and MATLAB classes. | n/a | 2.6 GB Attia 2020 batch file on the Hub: 808 KB read (0.03%), 4 requests, 1.6 s. Requires Python <= 3.13 for an `h5py` wheel today. |
| LeRobot v3 | `meta/info.json`, `meta/stats.json`, `meta/episodes/*.parquet` already hold schema, fps, codec, resolution, episode lengths and per-feature stats. | Parquet `timestamp` column, reachable via the Parquet footer. Videos never opened. | Not needed; the sidecars are the profile. |

The Hub cooperates: resolve URLs redirect to a CDN that answers `206 Partial Content` and the tree API returns
per-file `size`, `lfs.sha256` and `xetHash`. A content hash per file, available before reading a byte, lets the
laptop skip files it has already profiled, across runs and across datasets that share files.

What cannot come from an index: NaN counts, value ranges, image sanity, decoding errors inside payloads, and
anything about unindexed or truncated files. Those are handled by explicit, budgeted tiers (section 4.2).

## 3. Prior art and the gap

- `mcap info` (Foxglove CLI) reads remote MCAP lazily via the index but is single-file and prints text.
- Foxglove, Roboto.ai, Nominal, Formant profile data that has been **ingested into their platform**; the user
  must upload first, which is the copy this design avoids.
- `rosbags` (pure Python) reads bag1/bag2 indexes first but only from local paths.
- whylogs introduced mergeable profiles for distributed tabular profiling; ydata-profiling has a time-series mode.
- Croissant and HF dataset cards describe structure and provenance but have no quality or statistics layer.
- The HF Dataset Viewer computes statistics only for Parquet-convertible data; it is disabled on MCAP/bag/MAT repos.

Nothing lets an individual point a laptop at a remote robotics dataset, get a whole-dataset profile without
copying it, and check it against what their model needs. That is the gap this tool fills.

## 4. Architecture

Everything below runs in one process on the user's machine.

```
  requirements.yaml ─┐
                     ▼
  repo id ──▶ Discover ──▶ Plan ──▶ Profile (N files in parallel) ──▶ Merge ──▶ Evaluate ──▶ Report
               │             │            │                            │           │
          Hub tree API   local cache   RangeFile + format adapters   Dataset   verdict vs
          (size, hash)   (by hash)     (range reads only)            Profile   requirements
```

### 4.1 Discover

Input: a repo id (or any URL prefix). One paginated `tree?recursive=1&expand=1` call yields every file with
`path, size, lfs.sha256, xetHash`. Files are classified by extension and, when ambiguous, by an 8-to-128-byte
magic read. Sidecars (`metadata.yaml`, LeRobot `meta/*`) are detected and attached to the files they describe.
Selection flags (`--include`, `--exclude`, `--max-files`, `--sample N`) let the user profile a representative
subset first and the whole set only if the subset looks promising.

### 4.2 Plan

For each selected file the planner checks the local cache by content hash. Hits are reused unchanged. Misses are
assigned a **tier** and a **budget**:

| Tier | What is read | Yields | Default budget |
|---|---|---|---|
| T0 sidecar | `metadata.yaml`, LeRobot `meta/*` | topics, counts, duration, schema, fps, stats | always |
| T1 index | header + tail index + message indexes | everything in section 2 plus per-stream rate, jitter, gaps, start offsets | 64 MB of index bytes per file, else evenly spaced chunk sampling |
| T2 sample | T1 + decode N evenly spaced chunks | payload decodes, NaN/inf scan, value ranges, image dims, schema conformance | N = 8 chunks, max 64 MB per file |
| T3 full | sequential read of the whole file | everything; the only option for unindexed or truncated files | off by default; opt-in with a size cap |

The plan is printed before anything runs (`--dry-run`): files, tiers, estimated bytes and requests. The user
can raise or lower tiers for specific streams; for example "T2 on the joint-state topic only" to get NaN
counts where it matters.

### 4.3 Profile (the worker)

A worker is a pure function: `profile(file_ref, plan) -> FileProfile`. It opens a `RangeFile`, runs the format
adapter at the requested tier, and returns a profile plus a fetch receipt (bytes, requests, wall time). Workers
hold no state and need no disk.

Format adapters implement one interface:

```python
class Adapter(Protocol):
    def sniff(self, head: bytes, name: str) -> bool
    def profile(self, rf: RangeFile, tier: Tier, budget: Budget) -> FileProfile
```

Adapters in v1: `mcap`, `bag1`, `bag2` (metadata.yaml + mcap or sqlite), `mat` (v5/v7 walker, v7.3 via h5py
over RangeFile), `lerobot` (sidecars + Parquet footers). The existing `phys_ai_dataprofiler/remote/inspector.py` is the seed
for `mcap` and `mat`; the ROS 1 walk prototyped in this session becomes `bag1`.

Timing statistics come from one shared function over timestamp lists, so a profile from message indexes equals
a profile from a full scan on the same file. Verified on 39 channels of a 109 MB MCAP.

### 4.4 Laptop-scale parallelism

"Distributed" here means distributing range requests, not machines. Two levels:

- **Across files:** a process pool (default: CPU count) each handling one file at a time. Files are dealt
  largest-first so the long tail does not land at the end.
- **Within a file:** a thread pool of 16 to 32 concurrent range requests for scattered message indexes.
  Measured: 214 scattered 2 KB reads in 2.3 s with 32 threads against the Hub CDN.

A global token bucket caps requests per 5-minute window below the Hub's published limits (3,000 resolves
anonymous, 5,000 with a free account, 12,000 PRO) and backs off on 429. Resolve calls are one per file; the
subsequent range reads go to the signed CDN URL, which is refreshed on expiry (about one hour).

Everything is idempotent: a crashed or interrupted run resumes from the local cache and only profiles the files
it has not seen. Optional, not required: a `--workers host1,host2` mode that runs the same worker over SSH on a
few lab machines and streams FileProfiles back. No shared storage is involved; the laptop remains the merge point.

### 4.5 Local cache

`~/.cache/phys-ai-dataprofiler/<hash[:2]>/<hash>.json`, one FileProfile per content hash, plus a repo-level
inventory cache keyed by commit sha. Consequences:

- Re-running on the same dataset after a new commit only profiles the changed files.
- Two datasets that share files (forks, re-uploads, mirrors) share profiles.
- The cache is plain JSON the user can inspect, copy to a colleague, or delete. Sharing it is a user choice,
  never an automatic upload. There is no central store and the tool never writes to the Hub.

### 4.6 Profile schema (mergeable)

```jsonc
FileProfile {
  content_hash, size_bytes, format, format_version, tier, fetch: {bytes, requests, seconds},
  time: {start_ns, end_ns, duration_s, time_base},
  streams: [ { name, type/schema, encoding, count, rate_hz, jitter_ms, gaps_over_3x, max_gap_ms,
               first_ns, last_ns, modality, extra: {codec, resolution, dims, dtype, shape, nan_count?} } ],
  schemas: [ {name, encoding, hash} ],
  findings: [ {severity, code, stream?, detail} ],
  provenance: { adapter_version, profiled_at }
}
DatasetProfile = merge(FileProfiles) + inventory + {
  streams_by_name: union with per-file coverage (which files have it, rate distribution, total messages),
  total_duration, total_messages, episodes (from LeRobot meta or file boundaries),
  findings roll-up, fidelity: {files_by_tier, checks_skipped}
}
```

Merge is a fold over files: counts add, time ranges union, rates are recomputed from counts and durations,
jitter and gap fields are summarised as distributions, findings concatenate. Because merge is associative the
dataset profile can be updated incrementally as files finish.

### 4.7 Evaluate: requirements and verdict

This is the step that turns numbers into a decision. The user writes a small requirements file describing what
their model needs; defaults exist for common cases (imitation learning from teleop, visual odometry, IMU-only).

```yaml
# requirements.yaml
streams:
  - match: {modality: video, any_of: ["/cam*", "observation.images.*"]}
    min_rate_hz: 25
    min_resolution: [640, 480]
    required: true
  - match: {modality: proprio, any_of: ["/joint_states", "observation.state"]}
    min_rate_hz: 50
    max_gap_ms: 100
    required: true
  - match: {modality: imu}
    min_rate_hz: 100
    required: false
dataset:
  min_total_hours: 20
  min_episodes: 500
  min_episode_seconds: 5
  max_files_unindexed_pct: 5
  same_schema_across_files: true
```

Evaluation produces, per requirement, one of `pass`, `fail`, `unknown` (the profile tier could not establish
it) with the evidence: which streams matched, coverage in files and hours, rate distribution, worst gap. The
report leads with the verdict and the list of unknowns, then the profile, then the plan the user could run to
resolve the unknowns (for example "T2 on 20 sampled files would settle the NaN question for about 300 MB").

### 4.8 Quality rules (findings)

Rules run on profiles, not on files, so they are cheap and re-runnable when rules change:

- `dropped_frames`: gaps over 3x the median interval and over 100 ms on a stream.
- `rate_drift`: a stream's rate differs by more than 10% across files.
- `start_offset`: a stream starts more than 1 s after the earliest stream in the file.
- `missing_stream`: a stream present in most files is absent from some.
- `schema_mismatch`: same stream name, different schema hash across files.
- `short_episode`, `empty_file`, `clock_jump` (non-monotonic or epoch-1970 timestamps, seen in the scan).
- `unindexed_mcap`, `truncated`, `partial_scan`, `chunks_not_decoded`, `sampled_message_index`, `nan_check_partial`,
  `nan_check_skipped`: fidelity notices.
- `nan_values`, `constant_signal`, `out_of_range`: T2 only.

### 4.9 Report

- Terminal summary and a Markdown plus JSON report in a local folder (today's output, extended with the
  verdict section and dataset roll-up).
- Optional `--html` self-contained page with stream coverage timeline, rate and gap charts, opened locally.
  No hosting involved.
- `--compare a.json b.json` to put two candidate datasets side by side against the same requirements, since
  choosing between datasets is the common case.

## 5. Cost model (laptop)

Per MCAP file at T1: one resolve request, one or two for header and tail, one per chunk for message indexes.
Bytes: summary size plus 16 B per message. For the 163 MB test file that was 217 requests and 577 KB in 6.7 s
with 16 threads.

A 1 TB dataset of 1,000 such files: about 220k requests and 0.6 GB of transfer. With 8 processes times 16
threads that is roughly 15 to 30 minutes of wall time, bounded by request latency and the rate limit, not by
bandwidth or CPU. Chunk sampling caps per-file requests when files have tens of thousands of chunks, and the
`--sample` flag lets the user profile 50 files in a minute before committing to the full set. A MAT dataset of
4,900 small files costs one request per file, about 30 minutes single-threaded or 3 minutes with 16 threads.

## 6. Security

- Tokens are sent only to `huggingface.co`, never to CDN hosts. Verified in the current implementation.
- Dataset bytes are untrusted input. Adapters enforce record-size limits, bounded decompression (zlib, zstd,
  lz4, bz2 with output caps), and never execute code from files.
- Nothing leaves the laptop. The tool makes only GET requests to the Hub and the CDN. Profiles live in the
  user's cache directory.

## 7. Phased plan

| Phase | Deliverable | Depends on |
|---|---|---|
| 0 (done) | MCAP and MAT v5/v7 metadata-only inspection over Hub byte ranges; report with bytes-read accounting. | |
| 1 | `bag1` and `bag2` adapters; FileProfile schema and merge; local content-hash cache; dataset roll-up; process-pool fan-out with rate limiting. | ROS 1 walk prototype from this session. |
| 2 | Requirements file, evaluator, verdict report; `--sample` and `--compare`. | Phase 1 schema. |
| 3 | T2 sampling tier (payload decodes, NaN scan); `lerobot` adapter; quality rules engine. | Phase 1 schema. |
| 4 | Local HTML report with charts; MAT v7.3 via h5py over RangeFile; S3/GCS endpoints. | |
| 5 (optional) | SSH fan-out to lab machines with the laptop as merge point. | Phase 1 idempotent workers. |

## 8. Risks and open questions

- **Unindexed or truncated files** need a full read. The planner surfaces them, the verdict marks affected
  requirements `unknown`, and the user decides whether to pay for T3 on a size-capped subset.
- **HDF5 metadata is scattered** in general, but the one real MAT v7.3 file measured so far (2.6 GB, MATLAB
  struct arrays) needed only 4 block reads with a 256 KB block cache. Re-measure on files with thousands of
  variables before promising numbers.
- **SQLite bags** give global timing cheaply but per-topic timing only by touching table pages. Rely on
  `metadata.yaml` for counts and treat per-topic jitter as T2.
- **Rate limits** bound how aggressively a laptop can fan out. Whether CDN range reads count against the
  resolver bucket was observed once and must be re-verified; the token bucket assumes the conservative case.
- **Fidelity communication** is the main product risk: a metadata-only profile that reads like a full audit
  will mislead. The verdict's `unknown` state and the per-file tier list exist to prevent that.
- **Requirements expressiveness**: stream matching by name glob and modality covers the common cases; richer
  matching (by schema field, by frame id) can come later without changing the profile schema.

## 9. Test datasets on the Hub

See `docs/TEST_DATASETS.md` (generated from a Hub scan) for 39 public repos with MCAP, ROS 1 bag, ROS 2 SQLite bag
and MAT files, with sizes and the smallest file in each.

## Sources

- MCAP specification: https://mcap.dev/spec and the Python reader https://github.com/foxglove/mcap
- ROS 1 bag reader source (ros_comm `bag.py`, ternaris `rosbags`): https://gitlab.com/ternaris/rosbags
- rosbag2 SQLite schema: https://github.com/ros2/rosbag2 (`rosbag2_storage_sqlite3/src/.../sqlite_storage.cpp`)
- HDF5 file-like objects in h5py: https://docs.h5py.org/en/stable/high/file.html
- LeRobot dataset v3: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
- Hub filesystem API: https://huggingface.co/docs/huggingface_hub/guides/hf_file_system
- Xet deduplication and content hashes: https://huggingface.co/docs/hub/en/xet/deduplication
- Hub rate limits: https://huggingface.co/docs/hub/en/rate-limits
- `mcap` CLI remote reads: https://mcap.dev/guides/cli
- whylogs mergeable profiles: https://docs.whylabs.ai/docs/whylogs-overview
- Croissant specification: https://docs.mlcommons.org/croissant/docs/croissant-spec.html
