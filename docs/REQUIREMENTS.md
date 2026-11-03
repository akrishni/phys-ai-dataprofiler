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
```

The file is validated before any network traffic. Every problem is reported at once, with suggestions
for typos (`min_rate` → `min_rate_hz`) and a pointer when a field is in the wrong section.

**Modalities:** `video` (`image` is the same thing), `proprio`, `action`, `imu`, `lidar`, `gps`,
`reward`, `flag`, `timestamps`, `index`, `text`, `binary`, `timeseries`, `data`, `other`. They come from
topic, column and type names, so a `topic` glob is often the more reliable match on unfamiliar data.

**Not accepted, on purpose:** `min_resolution`, `codec`, `min_dims`. The profiler reads indexes and
metadata and never decodes image or message payloads, so it cannot measure these. They are rejected
with that reason rather than accepted and left unknown forever.

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
  list is incomplete (unnamed channels, undecoded chunks). Or a camera is missing from a Parquet file;
  LeRobot keeps cameras in separate video files, which are not profiled yet.
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

## Comparing datasets

`compare` evaluates each candidate against the same file and ranks them: fewer required failures first,
then fewer unknowns, then more passes, then more hours. Ties keep the order you gave. A target is a saved
JSON report or a Hub repo id, which is profiled with default settings plus any options after `--`. The
output is a Markdown table with one column per dataset, plus a "why not pass" list per candidate, and a
JSON file with every verdict.

## Presets

| Preset | For |
|---|---|
| `teleop-imitation` | Imitation learning from teleoperation: camera and robot state at 10 Hz or more, 1 h, 50 episodes |
| `visual-odometry` | VO/VIO and SLAM: camera at 15 Hz or more without long gaps, optional IMU at 100 Hz or more, consistent types |
| `lidar-perception` | LiDAR at 5 Hz or more, 10 minutes |
| `timeseries-clean` | Lab and sensor data (MAT, Parquet): no NaN/inf in float data, every file readable |

The thresholds are modest starting points. Copy one with `requirements show` and tighten it for your model.
