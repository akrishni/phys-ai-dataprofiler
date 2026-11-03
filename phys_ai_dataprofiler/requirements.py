"""
Requirements files: what a model needs from a dataset, written as YAML.

A requirements file lists stream requirements (a camera at 25 Hz, joint states
with no gap over 100 ms) and dataset requirements (20 hours, 500 episodes). It
is loaded and validated here; `evaluate.py` checks a profile against it.

Evaluation is deterministic: the same profile and the same file always give
the same verdict. There is no free-text input and no model call anywhere.

    version: 1
    name: my-policy
    files:
      exclude: ["meta/*"]
    streams:
      - name: wrist camera
        match: {modality: video, topic: ["*wrist*"]}
        min_rate_hz: 25
      - name: joint state
        match: {modality: proprio}
        min_rate_hz: 50
        max_gap_ms: 100
    dataset:
      min_total_hours: 20
      min_episodes: 500

See docs/REQUIREMENTS.md for every field and how pass, fail and unknown are decided.
"""

from __future__ import annotations

import difflib
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
PRESET_DIR = Path(__file__).resolve().parent / "presets"
PRESET_PREFIX = "preset:"

# Modalities the inspectors assign. "image" (Parquet, MAT) and "video" (MCAP, bags) mean the same thing here.
MODALITIES = ("video", "image", "proprio", "action", "imu", "lidar", "gps", "reward", "flag",
              "timestamps", "index", "text", "binary", "timeseries", "data", "other")
MODALITY_ALIASES = {"image": "video"}

# Fields people will reach for that the profiler cannot measure yet. Rejected with a reason rather than
# accepted and silently left unknown forever.
NOT_MEASURABLE = {
    "min_resolution": "image resolution is not measured: the profiler reads indexes and metadata, it does not "
                      "decode image or video payloads",
    "max_resolution": "image resolution is not measured (payloads are not decoded)",
    "resolution": "image resolution is not measured (payloads are not decoded)",
    "codec": "video codec is not measured (payloads are not decoded)",
    "min_dims": "array dimensions of MCAP/bag messages are not measured (payloads are not decoded)",
}

STREAM_THRESHOLDS = ("min_rate_hz", "max_rate_hz", "max_jitter_ms", "max_gap_ms", "max_gaps", "max_nan")
STREAM_KEYS = ("name", "required", "match", "min_count", "coverage") + STREAM_THRESHOLDS
MATCH_KEYS = ("topic", "exclude_topic", "modality", "schema")
DATASET_KEYS = ("min_total_hours", "min_episodes", "min_episode_seconds", "max_short_episodes_pct",
                "same_schema_across_files", "max_invalid_files_pct", "max_unindexed_files_pct")
TOP_KEYS = ("version", "name", "description", "files", "streams", "dataset")
FILES_KEYS = ("include", "exclude")
INT_FIELDS = {"max_gaps", "max_nan", "min_episodes"}
PCT_FIELDS = {"max_short_episodes_pct", "max_invalid_files_pct", "max_unindexed_files_pct"}


class RequirementsError(ValueError):
    """A requirements file that cannot be used; `errors` lists every problem found."""

    def __init__(self, errors: list[str], source: str = "") -> None:
        self.errors = errors
        self.source = source
        head = f"invalid requirements{f' in {source}' if source else ''}"
        super().__init__(head + ":\n" + "\n".join(f"  - {e}" for e in errors))


def _canon_modality(m: str) -> str:
    m = m.lower()
    return MODALITY_ALIASES.get(m, m)


@dataclass
class Match:
    topic: list[str] = field(default_factory=list)
    exclude_topic: list[str] = field(default_factory=list)
    modality: list[str] = field(default_factory=list)
    schema: list[str] = field(default_factory=list)

    def matches(self, channel: dict) -> bool:
        topic = str(channel.get("topic") or "")
        if self.topic and not any(fnmatch.fnmatch(topic.lower(), g.lower()) for g in self.topic):
            return False
        if self.exclude_topic and any(fnmatch.fnmatch(topic.lower(), g.lower()) for g in self.exclude_topic):
            return False
        if self.modality and _canon_modality(str(channel.get("modality") or "")) not in \
                {_canon_modality(m) for m in self.modality}:
            return False
        schema = str(channel.get("schema") or "")
        if self.schema and not any(fnmatch.fnmatch(schema.lower(), g.lower()) for g in self.schema):
            return False
        return True

    def wants_video(self) -> bool:
        return "video" in {_canon_modality(m) for m in self.modality}

    def describe(self) -> str:
        parts = []
        if self.modality:
            parts.append("modality " + "|".join(self.modality))
        if self.topic:
            parts.append("topic " + "|".join(self.topic))
        if self.schema:
            parts.append("schema " + "|".join(self.schema))
        if self.exclude_topic:
            parts.append("not " + "|".join(self.exclude_topic))
        return ", ".join(parts)


@dataclass
class StreamRequirement:
    name: str
    match: Match
    required: bool = True
    min_count: int | str = 1          # an int, or "all": every matching stream must satisfy the thresholds
    coverage: float = 1.0             # fraction of files in scope that must satisfy the requirement
    min_rate_hz: float | None = None
    max_rate_hz: float | None = None
    max_jitter_ms: float | None = None
    max_gap_ms: float | None = None
    max_gaps: int | None = None
    max_nan: int | None = None

    def thresholds(self) -> list[tuple[str, float]]:
        return [(k, getattr(self, k)) for k in STREAM_THRESHOLDS if getattr(self, k) is not None]


@dataclass
class DatasetRequirements:
    min_total_hours: float | None = None
    min_episodes: int | None = None
    min_episode_seconds: float | None = None
    max_short_episodes_pct: float = 0.0
    same_schema_across_files: bool | None = None
    max_invalid_files_pct: float | None = None
    max_unindexed_files_pct: float | None = None

    def active(self) -> list[str]:
        out = []
        for k in DATASET_KEYS:
            if k == "max_short_episodes_pct":
                continue
            v = getattr(self, k)
            if v is not None and v is not False:
                out.append(k)
        return out


@dataclass
class Requirements:
    name: str
    source: str
    version: int = SCHEMA_VERSION
    description: str | None = None
    files_include: list[str] = field(default_factory=list)
    files_exclude: list[str] = field(default_factory=list)
    streams: list[StreamRequirement] = field(default_factory=list)
    dataset: DatasetRequirements = field(default_factory=DatasetRequirements)

    def in_scope(self, repo_path: str) -> bool:
        if self.files_include and not any(fnmatch.fnmatch(repo_path, g) for g in self.files_include):
            return False
        return not any(fnmatch.fnmatch(repo_path, g) for g in self.files_exclude)


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #

class _Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}" if where else msg)

    def unknown_keys(self, where: str, d: dict, allowed: tuple[str, ...], elsewhere: dict[str, str]) -> None:
        for k in d:
            if k in allowed:
                continue
            path = f"{where}.{k}" if where else str(k)
            if k in NOT_MEASURABLE:
                self.err(path, f"not supported: {NOT_MEASURABLE[k]}")
            elif k in elsewhere:
                self.err(path, f"belongs under {elsewhere[k]}")
            else:
                close = difflib.get_close_matches(str(k), allowed, n=1, cutoff=0.6)
                self.err(path, "unknown field" + (f"; did you mean '{close[0]}'?" if close else
                                                  f"; allowed: {', '.join(allowed)}"))

    def number(self, where: str, v: Any, integer: bool = False, pct: bool = False) -> float | int | None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            self.err(where, f"expected a {'whole ' if integer else ''}number, got {v!r}")
            return None
        if integer and (isinstance(v, float) and not v.is_integer()):
            self.err(where, f"expected a whole number, got {v!r}")
            return None
        if v < 0:
            self.err(where, f"must be 0 or more, got {v!r}")
            return None
        if pct and v > 100:
            self.err(where, f"is a percentage, must be 0 to 100, got {v!r}")
            return None
        return int(v) if integer else float(v)

    def boolean(self, where: str, v: Any) -> bool | None:
        if not isinstance(v, bool):
            self.err(where, f"expected true or false, got {v!r}")
            return None
        return v

    def globs(self, where: str, v: Any) -> list[str]:
        items = [v] if isinstance(v, str) else v
        if not isinstance(items, list) or not items or not all(isinstance(x, str) and x for x in items):
            self.err(where, f"expected a pattern or a list of patterns, got {v!r}")
            return []
        return list(items)


def _elsewhere_for_stream() -> dict[str, str]:
    d = {k: "dataset" for k in DATASET_KEYS}
    d.update({k: "match" for k in MATCH_KEYS})
    return d


def parse_requirements(doc: Any, source: str = "") -> Requirements:
    """Validate a loaded YAML document; raises RequirementsError listing every problem."""
    v = _Validator()
    if not isinstance(doc, dict):
        raise RequirementsError([f"expected a mapping at the top level, got {type(doc).__name__}"], source)

    stream_level = {k: "a stream entry (streams: - ...)" for k in STREAM_KEYS if k != "name"}
    stream_level.update({k: "a stream's match" for k in MATCH_KEYS})
    v.unknown_keys("", doc, TOP_KEYS, {**stream_level, **{k: "dataset" for k in DATASET_KEYS}})

    version = doc.get("version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        v.err("version", f"unsupported version {version!r}; this profiler reads version {SCHEMA_VERSION}")

    name = doc.get("name") or (Path(source).stem if source and not source.startswith(PRESET_PREFIX)
                               else source.removeprefix(PRESET_PREFIX) or "requirements")
    if not isinstance(name, str):
        v.err("name", f"expected text, got {name!r}")
        name = str(name)
    description = doc.get("description")
    if description is not None and not isinstance(description, str):
        v.err("description", "expected text")
        description = None

    req = Requirements(name=name, source=source, description=(description or "").strip() or None)

    files = doc.get("files")
    if files is not None:
        if not isinstance(files, dict):
            v.err("files", "expected a mapping with include and/or exclude")
        else:
            v.unknown_keys("files", files, FILES_KEYS, {})
            if "include" in files:
                req.files_include = v.globs("files.include", files["include"])
            if "exclude" in files:
                req.files_exclude = v.globs("files.exclude", files["exclude"])

    streams = doc.get("streams") or []
    if not isinstance(streams, list):
        v.err("streams", "expected a list of stream requirements (each starting with '- name: ...')")
        streams = []
    seen_names: set[str] = set()
    for i, s in enumerate(streams):
        where = f"streams[{i}]"
        if not isinstance(s, dict):
            v.err(where, "expected a mapping with at least name and match")
            continue
        v.unknown_keys(where, s, STREAM_KEYS, _elsewhere_for_stream())
        sname = s.get("name")
        if not isinstance(sname, str) or not sname.strip():
            v.err(f"{where}.name", "required: a short label such as 'wrist camera'")
            sname = f"stream {i + 1}"
        elif sname in seen_names:
            v.err(f"{where}.name", f"duplicate stream name '{sname}'")
        seen_names.add(sname)
        where = f"streams[{i}] ({sname})"

        m = s.get("match")
        match = Match()
        if not isinstance(m, dict) or not m:
            v.err(f"{where}.match", "required: at least one of topic, modality or schema "
                                    "(use topic: '*' to match every stream)")
        else:
            v.unknown_keys(f"{where}.match", m, MATCH_KEYS, {k: "the stream entry" for k in STREAM_KEYS})
            if "topic" in m:
                match.topic = v.globs(f"{where}.match.topic", m["topic"])
            if "exclude_topic" in m:
                match.exclude_topic = v.globs(f"{where}.match.exclude_topic", m["exclude_topic"])
            if "schema" in m:
                match.schema = v.globs(f"{where}.match.schema", m["schema"])
            if "modality" in m:
                mods = v.globs(f"{where}.match.modality", m["modality"])
                for mod in mods:
                    if mod.lower() not in MODALITIES:
                        close = difflib.get_close_matches(mod.lower(), MODALITIES, n=1, cutoff=0.6)
                        v.err(f"{where}.match.modality", f"unknown modality '{mod}'"
                              + (f"; did you mean '{close[0]}'?" if close else f"; known: {', '.join(MODALITIES)}"))
                match.modality = [mod.lower() for mod in mods]
            if not (match.topic or match.modality or match.schema):
                v.err(f"{where}.match", "needs at least one of topic, modality or schema")

        sr = StreamRequirement(name=sname, match=match)
        if "required" in s:
            b = v.boolean(f"{where}.required", s["required"])
            sr.required = True if b is None else b
        if "min_count" in s:
            mc = s["min_count"]
            if mc == "all":
                sr.min_count = "all"
            else:
                n = v.number(f"{where}.min_count", mc, integer=True)
                if n is not None and n < 1:
                    v.err(f"{where}.min_count", "must be 1 or more, or 'all'")
                elif n is not None:
                    sr.min_count = n
        if "coverage" in s:
            c = v.number(f"{where}.coverage", s["coverage"])
            if c is not None and not (0 < c <= 1):
                v.err(f"{where}.coverage", f"a fraction of files, must be above 0 and at most 1, got {c}")
            elif c is not None:
                sr.coverage = c
        for k in STREAM_THRESHOLDS:
            if k in s:
                setattr(sr, k, v.number(f"{where}.{k}", s[k], integer=k in INT_FIELDS))
        if sr.min_rate_hz is not None and sr.max_rate_hz is not None and sr.min_rate_hz > sr.max_rate_hz:
            v.err(where, f"min_rate_hz {sr.min_rate_hz} is above max_rate_hz {sr.max_rate_hz}")
        req.streams.append(sr)

    ds = doc.get("dataset")
    if ds is not None:
        if not isinstance(ds, dict):
            v.err("dataset", "expected a mapping")
        else:
            v.unknown_keys("dataset", ds, DATASET_KEYS, {k: "a stream entry" for k in STREAM_KEYS + MATCH_KEYS})
            d = req.dataset
            for k in DATASET_KEYS:
                if k not in ds:
                    continue
                if k == "same_schema_across_files":
                    d.same_schema_across_files = v.boolean(f"dataset.{k}", ds[k])
                else:
                    val = v.number(f"dataset.{k}", ds[k], integer=k in INT_FIELDS, pct=k in PCT_FIELDS)
                    if val is not None or k != "max_short_episodes_pct":
                        setattr(d, k, val if val is not None else getattr(d, k))
            if "max_short_episodes_pct" in ds and "min_episode_seconds" not in ds:
                v.err("dataset.max_short_episodes_pct", "only means something together with min_episode_seconds")

    if not req.streams and not req.dataset.active() and not v.errors:
        v.err("", "no requirements: add at least one entry under streams or dataset")
    if v.errors:
        raise RequirementsError(v.errors, source)
    return req


def load_requirements(spec: str) -> Requirements:
    """Load `preset:NAME` or a path to a YAML file."""
    if spec.startswith(PRESET_PREFIX):
        name = spec[len(PRESET_PREFIX):]
        text = preset_text(name)
        source = spec
    else:
        path = Path(spec)
        if not path.is_file():
            hint = (f"; built-in presets: {', '.join('preset:' + n for n, _ in list_presets())}"
                    if not path.suffix else "")
            raise RequirementsError([f"file not found: {spec}{hint}"], spec)
        text = path.read_text()
        source = str(path)
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise RequirementsError([f"not valid YAML{where}: {getattr(exc, 'problem', exc)}"], source) from None
    return parse_requirements(doc, source)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

def list_presets() -> list[tuple[str, str]]:
    out = []
    for p in sorted(PRESET_DIR.glob("*.yaml")):
        try:
            doc = yaml.safe_load(p.read_text()) or {}
            desc = " ".join(str(doc.get("description") or "").split())
        except yaml.YAMLError:
            desc = "(unreadable)"
        out.append((p.stem, desc))
    return out


def preset_text(name: str) -> str:
    path = PRESET_DIR / f"{name}.yaml"
    if not name or "/" in name or not path.is_file():
        all_names = [n for n, _ in list_presets()]
        names = ", ".join(all_names)
        close = ([n for n in all_names if name and n.startswith(name)]
                 or difflib.get_close_matches(name, all_names, n=1))
        raise RequirementsError([f"no preset named '{name}'" + (f"; did you mean '{close[0]}'?" if close else "")
                                 + f". Presets: {names}"], PRESET_PREFIX + name)
    return path.read_text()
