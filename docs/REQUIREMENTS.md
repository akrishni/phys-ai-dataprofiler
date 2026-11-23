# Requirements, verdicts and comparisons

A requirements file says what your model needs from a dataset. The profiler checks a profile against
it and answers **pass**, **fail** or **unknown** for each requirement, with the evidence. The same
profile and the same file always give the same answer: requirements are plain YAML, checked by rules,
with no model or network call involved in the check.

```bash
phys-ai-dataprofiler requirements list                         # built-in presets
phys-ai-dataprofiler requirements show teleop-imitation > req.yaml
phys-ai-dataprofiler requirements validate req.yaml            # check the file, profile nothing

phys-ai-dataprofiler org/dataset -r req.yaml                   # profile + verdict in the report
phys-ai-dataprofiler check org/dataset -r preset:visual-odometry   # same; for CI
phys-ai-dataprofiler check --report reports/run.json -r req.yaml   # re-check a saved report, offline
phys-ai-dataprofiler compare org/a org/b old_run.json -r req.yaml  # rank candidates
phys-ai-dataprofiler compare org/a org/b -r req.yaml -- --max-files 20   # options after -- go to each profile
```

Exit codes: **0** pass, **1** fail, **2** bad requirements file or usage, **3** nothing to profile,
**4** unknown. `compare` exits with the best candidate's status.

## The file

```yaml
version: 1                      # optional; this profiler reads version 1
name: my-policy                 # optional; defaults to the file name
description: what this is for   # optional

files:                          # optional: which repo files the requirements apply to
  include: ["episodes/*"]       # globs on the repo path; also narrows what gets profiled
  exclude: ["meta/*"]

streams:                        # what must be in each file
  - name: wrist camera          # required, unique; used in reports
    required: true              # default true; false = reported but never changes the verdict
    match:                      # at least one of topic, modality, schema
      modality: video           # one or a list; see modalities below
      topic: ["*wrist*", "observation.images.*"]   # globs, case-insensitive
      exclude_topic: "*camera_info*"
      schema: "sensor_msgs/*Image*"
    min_count: 1                # matching streams that must meet the thresholds; a number or "all"
    coverage: 1.0               # share of in-scope files that must satisfy this; 0.9 = 90%
    min_rate_hz: 25
    max_rate_hz: 120
    max_jitter_ms: 5
    max_gap_ms: 100             # worst interval between consecutive messages
    max_gaps: 0                 # intervals over 3x the mean
    max_nan: 0                  # NaN/inf values (MAT and Parquet only, see below)

dataset:                        # what the whole selection must add up to
  min_total_hours: 20
  min_episodes: 500
  min_episode_seconds: 5
  max_short_episodes_pct: 5     # with min_episode_seconds: share allowed to be shorter (default 0)
  same_schema_across_files: true
  max_invalid_files_pct: 0      # files that could not be read
  max_unindexed_files_pct: 5    # MCAP/bag files with no index, or truncated
  min_stream_overlap_pct: 95    # per recording, the required streams record together (see below)
```

The file is validated before any network traffic. Every problem is reported at once, with suggestions
for typos (`min_rate` → `min_rate_hz`) and a pointer when a field is in the wrong section.

**Modalities:** `video` (`image` is the same thing), `proprio`, `action`, `imu`, `lidar`, `gps`,
`reward`, `flag`, `timestamps`, `index`, `text`, `binary`, `timeseries`, `data`, `other`. They come from
topic, column and type names, so a `topic` glob is often the more reliable match on unfamiliar data.

**Not accepted, on purpose:** `min_resolution`, `codec`, `min_dims`. MP4 profiles report resolution and
codec from the file's index, but MCAP and bag image messages are never decoded, so these could not be
checked on most formats. They are rejected with that reason rather than accepted and left unknown.

**LeRobot videos.** LeRobot keeps state and actions in Parquet tables and each camera in its own MP4s.
With both in scope, a camera requirement is judged on the videos and every other requirement on the
tables. Cameras are matched by name (`observation.images.wrist` for `topic: "*wrist*"`), and several
cameras in separate files together meet `min_count: 2`. In LeRobot v3 every camera file packs many
episodes and cameras split files at different points, so all videos are judged as one set and each
camera passes when `coverage` of its files do; in v2 (one file per camera per episode) each episode is a
recording. Hours come from the table when it is profiled, otherwise one camera per recording, never the
sum of every camera file.

## How pass, fail and unknown are decided

- **pass:** the profile shows the requirement is met.
- **fail:** the profile shows it is not met, from numbers measured completely.
- **unknown:** the profile cannot settle it, and the report says how to settle it.

A requirement is unknown, never a guessed pass or fail, when:

- **The number was not measured.** NaN counts for MCAP and bags, whose payloads are never decoded. Gaps
  and jitter for a file with no per-message timestamps. A NaN scan that was turned off or covered only
  part of the data, unless it already found NaNs, which proves a fail.
- **The number is a sample estimate close to the threshold.** When message indexes or row groups were
  sampled to stay within budget, an estimate decides pass or fail only if it clears the threshold by 10%.
  Measured numbers get a 0.1% allowance for float rounding, so 9.9999 Hz meets `min_rate_hz: 10`.
- **The data might be there but was not seen.** A required stream is missing from a file whose stream
  list is incomplete (unnamed channels, undecoded chunks). Or a camera is missing from a Parquet table and
  none of the dataset's MP4s were profiled (`--ext` without `.mp4`, or `files.exclude` on `videos/`).
- **Not every file was looked at.** A file failed to profile, or `--max-files`/`--max-total-mb` left files
  out and the total so far is below the threshold. Once the profiled files already clear it, such as
  20 hours already counted, it is a pass.

**Overall verdict:** fail if any required check fails, otherwise unknown if any required check is unknown,
otherwise pass. Optional streams are reported but do not affect it.

**Coverage:** a stream check passes when it holds in at least `coverage` of the in-scope files. It is
unknown when the files still undecided could bring it there, and fail when they cannot.

**Episodes:** from the Parquet episode column (LeRobot) when there is one, otherwise one episode per
file with a time range. The report keeps only the shortest, median and longest episode per Parquet file,
so `min_episode_seconds` on LeRobot data passes only when the shortest episode clears it; otherwise it is
unknown.

## Stream overlap

`min_stream_overlap_pct` checks that the required streams cover the same time in each recording. Every
stream can pass its own rate, jitter and gap checks while the set is still unusable: a camera that stops
halfway, joint state that starts 10 s late, or topics from two different sessions in one file.

For each recording, each required stream's span runs from its first to its last message. When several
channels match one requirement, their spans combine. The overlap is the time when every span is active,
divided by the time when any is. A camera recording 2-12 s and joints recording 0-10 s overlap 8 of 12
seconds, 66.7%. The check passes when every recording reaches the threshold, and the evidence lists the
worst recordings with each stream's span.

- **What counts as a recording:**
  - **MCAP, ROS 1 bag, ROS 2 `.db3`:** each file. Spans come from the same message indexes the rate
    checks read, so nothing more is downloaded. When indexes are sampled, each stream's first and last
    chunk is still read, so its span stays exact.
  - **LeRobot:** each episode, comparing the table's span with each camera's video span. LeRobot v3
    packs many episodes into each video, so the per-episode video spans come from `meta/episodes`. It
    is read only when this key is set, and only its timing columns: 104 KB for `lerobot/pusht`, 6.6 MB
    for `lerobot/droid_1.0.1` (95,658 episodes). It's capped by `--episodes-budget-mb` (default 16).
    Both timelines start at the episode's first frame, so the check finds videos cut short, or longer
    than their episode.
- **Unknown, never a guessed pass:** when a stream's first or last message may not have been read (a
  walk stopped at `--msgidx-budget-mb`, sampled ROS 2 pages, a sampled `meta/episodes`). Also when a
  stream has no message times (MAT files), or a file couldn't be profiled. A required stream that is
  absent from a fully read file is 0%.
- **What it does not see:** a constant delay between streams. A camera whose frames arrive 40 ms late
  still overlaps the joints 99%, and its own timestamps look regular. Catching that needs the sensor's
  capture time as well as the recorder's, which the profiler does not read.

## Comparing datasets

`compare` evaluates each candidate against the same file and ranks them: fewer required failures first,
then fewer unknowns, then more passes, then more hours. Ties keep the order you gave. A target is a saved
JSON report, a Hub repo id, an `s3://bucket/prefix` URI, or a local directory/file path, which is
profiled with default settings plus any options after `--`. The output is a Markdown table with one
column per dataset, plus a "why not pass" list per candidate, and a JSON file with every verdict.

## Presets

| Preset | For |
|---|---|
| `teleop-imitation` | Imitation learning from teleoperation: camera and robot state at 10 Hz or more, 1 h, 50 episodes |
| `visual-odometry` | VO/VIO and SLAM: camera at 15 Hz or more without long gaps, optional IMU at 100 Hz or more, consistent types |
| `lidar-perception` | LiDAR at 5 Hz or more, 10 minutes |
| `timeseries-clean` | Lab and sensor data (MAT, Parquet): no NaN/inf in float data, every file readable |

The thresholds are modest starting points. Copy one with `requirements show` and tighten it for your model.
