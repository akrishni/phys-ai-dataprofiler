"""
Check a profile against a requirements file: pass, fail or unknown, with evidence.

Input is a run report as written by the CLI (JSON dict) and a `Requirements`.
Output is a `Verdict`. Nothing here reads the network or a model; the same
report and requirements always give the same verdict.

Status rules
    pass     the profile shows the requirement is met.
    fail     the profile shows it is not met, from numbers it measured completely.
    unknown  the profile cannot settle it: the number was not measured (no NaN
             scan, no timestamps), was estimated from a sample and lies close to
             the threshold, or a file could not be read.

A sampled (estimated) number decides pass or fail only when it clears the
threshold by a margin (`SAMPLED_MARGIN`); otherwise the result is unknown. For
each unknown the verdict says what would settle it.

Overall: fail if any required check fails; otherwise unknown if any required
check is unknown; otherwise pass. Checks on optional streams are reported but
never change the overall result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phys_ai_dataprofiler.requirements import Requirements, StreamRequirement

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"
SAMPLED_MARGIN = 0.10          # 10%: sampled estimates must clear a threshold by this much
PRECISION = 0.001              # 0.1%: rounding allowance for measured (non-sampled) numbers
RANK = {PASS: 0, UNKNOWN: 1, FAIL: 2}

# Findings that mean timing numbers are estimates from part of the file.
SAMPLED_TIMING = {"sampled_message_index", "sampled_row_groups", "partial_scan"}
# Findings that mean a stream's message count is an estimate.
ESTIMATED_COUNTS = {"estimated_counts", "partial_scan"}
# Findings that mean the NaN scan did not cover all float data.
NAN_INCOMPLETE = {"nan_check_partial", "nan_check_sampled", "nan_check_failed"}
NAN_SKIPPED = {"nan_check_skipped"}
UNINDEXED = {"unindexed_mcap", "unindexed_bag", "truncated"}
# Findings that mean some streams may exist without having been seen or named.
INCOMPLETE_STREAM_LIST = {"unknown_channels", "chunks_not_decoded", "partial_scan", "walk_stopped"}
# Formats whose NaN count is a real measurement (MCAP and bag payloads are never decoded).
NAN_FORMATS = {"mat", "parquet"}


@dataclass
class Check:
    id: str                       # e.g. "stream:wrist camera" or "dataset:min_total_hours"
    label: str
    status: str
    required: bool
    expected: str
    observed: str
    evidence: list[str] = field(default_factory=list)
    resolve: str | None = None    # how to turn an unknown into pass/fail
    files: list[dict] = field(default_factory=list)   # per-file detail for stream checks
    headline: str = ""            # short "what was found" for compare tables


@dataclass
class Verdict:
    requirements: str
    requirements_source: str
    repo_id: str
    status: str
    passed: int
    failed: int
    unknown: int
    checks: list[Check]
    scope: dict
    notes: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return {PASS: 0, FAIL: 1, UNKNOWN: 4}[self.status]


# --------------------------------------------------------------------------- #
# Profile views
# --------------------------------------------------------------------------- #

@dataclass
class _FileView:
    path: str
    format: str | None
    valid: bool
    error: str | None
    channels: list[dict]
    finding_types: set[str]
    duration_s: float | None
    indexed: bool
    basic: dict

    @property
    def timing_sampled(self) -> bool:
        return bool(self.finding_types & SAMPLED_TIMING)

    @property
    def counts_estimated(self) -> bool:
        return bool(self.finding_types & ESTIMATED_COUNTS)


def _views(report: dict, req: Requirements) -> tuple[list[_FileView], list[str]]:
    views, out_of_scope = [], []
    for f in report.get("files") or []:
        path = f.get("repo_path") or ""
        if not req.in_scope(path):
            out_of_scope.append(path)
            continue
        rich = f.get("rich") if isinstance(f.get("rich"), dict) else {}
        views.append(_FileView(
            path=path, format=f.get("detected_format"), valid=bool(f.get("valid")), error=f.get("error"),
            channels=list(rich.get("channels") or []),
            finding_types={x.get("type") for x in rich.get("findings") or [] if isinstance(x, dict)},
            duration_s=(f.get("summary") or {}).get("duration_s"),
            indexed=bool(rich.get("indexed")), basic=f.get("basic") or {},
        ))
    return views, out_of_scope


# --------------------------------------------------------------------------- #
# Threshold comparison
# --------------------------------------------------------------------------- #

def _fmt(v: float | int | None, unit: str = "") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        v = round(v, 2) if abs(v) < 1000 else round(v)
    return f"{v:,}{unit}" if isinstance(v, int) else f"{v:g}{unit}"


UNITS = {"min_rate_hz": " Hz", "max_rate_hz": " Hz", "max_jitter_ms": " ms", "max_gap_ms": " ms",
         "max_gaps": "", "max_nan": ""}
CHANNEL_FIELD = {"min_rate_hz": "rate_hz", "max_rate_hz": "rate_hz", "max_jitter_ms": "jitter_ms",
                 "max_gap_ms": "max_gap_ms", "max_gaps": "gaps_over_3x", "max_nan": "nan_count"}


def compare(kind: str, value: float | None, limit: float, estimated: bool) -> str:
    """pass/fail/unknown for one number against one threshold; kind is 'min' or 'max'."""
    if value is None:
        return UNKNOWN
    if not estimated:
        # Measured numbers carry float rounding (10 Hz from float32 timestamps reads 9.9999 Hz).
        tol = PRECISION * abs(limit)
        ok = value >= limit - tol if kind == "min" else value <= limit + tol
        return PASS if ok else FAIL
    margin = SAMPLED_MARGIN * max(abs(limit), 1e-9)
    if kind == "min":
        if value >= limit + margin:
            return PASS
        if value < limit - margin:
            return FAIL
    else:
        if value <= limit - margin:
            return PASS
        if value > limit + margin:
            return FAIL
    return UNKNOWN


def _channel_value(fv: _FileView, ch: dict, key: str) -> tuple[float | None, bool, str | None]:
    """(value, is_estimate, reason when value is None) for one requirement field on one channel."""
    fld = CHANNEL_FIELD[key]
    if key == "max_nan":
        if fv.format not in NAN_FORMATS:
            return None, False, f"NaN values are not counted for {fv.format} (message payloads are not decoded)"
        if fv.finding_types & NAN_SKIPPED:
            return None, False, "NaN scan was off (--nan-budget-mb 0)"
        nan = ch.get("nan_count")
        if fv.finding_types & NAN_INCOMPLETE:
            # A partial scan that already found NaNs proves a lower bound; zero proves nothing.
            if nan and nan > 0:
                return float(nan), False, None
            return None, False, "NaN scan covered only part of the float data (raise --nan-budget-mb)"
        return float(nan or 0), False, None
    if key == "max_gaps" or key == "max_gap_ms" or key == "max_jitter_ms":
        if (ch.get("msg_count") or 0) > 1 and ch.get(fld) is None:
            return None, False, "no per-message timestamps (file has no message index)"
    v = ch.get(fld)
    if v is None:
        if (ch.get("msg_count") or 0) <= 1:
            return None, False, "fewer than two messages"
        return None, False, "no timing for this stream"
    return float(v), fv.timing_sampled and key != "max_nan", None


def _channel_status(sr: StreamRequirement, fv: _FileView, ch: dict) -> tuple[str, list[str], list[str]]:
    """Status of one channel against every threshold; (status, failing/unknown notes, measured values)."""
    worst = PASS
    notes, measured = [], []
    for key, limit in sr.thresholds():
        value, est, why = _channel_value(fv, ch, key)
        kind = "min" if key.startswith("min_") else "max"
        st = compare(kind, value, limit, est)
        shown = _fmt(value, UNITS[key]) + (" (est.)" if est and value is not None else "")
        measured.append(f"{key.removeprefix('min_').removeprefix('max_')} {shown}")
        if st != PASS:
            op = ">=" if kind == "min" else "<="
            if value is None:
                notes.append(f"{key}: unknown, {why}")
            elif st == UNKNOWN:
                notes.append(f"{key}: {shown} is estimated and within {SAMPLED_MARGIN:.0%} of {_fmt(limit)}")
            else:
                notes.append(f"{key}: {shown}, needs {op} {_fmt(limit, UNITS[key])}")
        if RANK[st] > RANK[worst]:
            worst = st
    return worst, notes, measured


# --------------------------------------------------------------------------- #
# Stream checks
# --------------------------------------------------------------------------- #

def _stream_file_status(sr: StreamRequirement, fv: _FileView) -> dict:
    if not fv.valid:
        return {"file": fv.path, "status": UNKNOWN, "matched": [], "detail": f"file not profiled: {fv.error or 'invalid'}"}
    matched = [c for c in fv.channels if sr.match.matches(c)]
    if not matched:
        if not fv.channels:
            return {"file": fv.path, "status": UNKNOWN, "matched": [], "detail": "no streams were read from this file"}
        if fv.finding_types & INCOMPLETE_STREAM_LIST:
            return {"file": fv.path, "status": UNKNOWN, "matched": [],
                    "detail": "no matching stream among those read, but the stream list is incomplete "
                              "(unnamed channels or undecoded chunks)"}
        if fv.format == "parquet" and sr.match.wants_video():
            return {"file": fv.path, "status": UNKNOWN, "matched": [],
                    "detail": "no image column; LeRobot-style datasets keep cameras in separate video files, "
                              "which are not profiled"}
        return {"file": fv.path, "status": FAIL, "matched": [], "detail": "no matching stream"}
    per = []
    for ch in matched:
        st, notes, measured = _channel_status(sr, fv, ch)
        per.append({"topic": ch.get("topic"), "status": st, "notes": notes, "measured": measured,
                    "count": ch.get("msg_count"), "rate_hz": ch.get("rate_hz"), "max_gap_ms": ch.get("max_gap_ms")})
    need = len(per) if sr.min_count == "all" else sr.min_count
    n_pass = sum(1 for p in per if p["status"] == PASS)
    n_maybe = sum(1 for p in per if p["status"] == UNKNOWN)
    if n_pass >= need:
        st = PASS
    elif n_pass + n_maybe >= need:
        st = UNKNOWN
    else:
        st = FAIL
    if st == PASS:
        best = [p for p in per if p["status"] == PASS][: max(1, need)]
        detail = "; ".join(f"{p['topic']} ({', '.join(p['measured']) or 'present'})" for p in best)
    else:
        bad = [p for p in per if p["status"] != PASS] or per
        detail = "; ".join(f"{p['topic']}: {'; '.join(p['notes'])}" for p in bad[:3])
        if len(matched) < need:
            detail = f"{len(matched)} matching stream(s), needs {need}. " + detail
    return {"file": fv.path, "status": st, "matched": [p["topic"] for p in per], "detail": detail,
            "rates_hz": [p["rate_hz"] for p in per if p["rate_hz"] is not None],
            "max_gaps_ms": [p["max_gap_ms"] for p in per if p["max_gap_ms"] is not None]}


def _range(vals: list[float], unit: str) -> str:
    lo, hi = min(vals), max(vals)
    return _fmt(lo, unit) if abs(hi - lo) < 0.01 * max(abs(hi), 1) else f"{_fmt(lo)}-{_fmt(hi, unit)}"


def _headline(details: list[dict]) -> str:
    """Short summary of what matched, for compare tables: topics, rate range, worst gap."""
    topics = sorted({t for d in details for t in d.get("matched", [])})
    if not topics:
        return "no matching stream"
    rates = [r for d in details for r in d.get("rates_hz", [])]
    gaps = [g for d in details for g in d.get("max_gaps_ms", [])]
    parts = [topics[0] if len(topics) == 1 else f"{len(topics)} streams"]
    if rates:
        parts.append(_range(rates, " Hz"))
    if gaps:
        parts.append(f"worst gap {_fmt(max(gaps), ' ms')}")
    return ", ".join(parts)


def _resolve_hint(details: list[dict]) -> str | None:
    text = " ".join(d["detail"] for d in details if d["status"] == UNKNOWN)
    hints = []
    if "estimated" in text:
        hints.append("raise --msgidx-budget-mb (MCAP, bags) or --column-budget-mb (Parquet) so timing is not sampled")
    if "NaN scan covered only part" in text or "NaN scan was off" in text:
        hints.append("raise --nan-budget-mb")
    if "not counted for" in text:
        hints.append("NaN counts need MAT or Parquet data; for MCAP/bags drop max_nan or check a converted copy")
    if "no message index" in text:
        hints.append("profile the unindexed files with --full-read-max-mb")
    if "file not profiled" in text or "no streams were read" in text:
        hints.append("fix or exclude the files that failed to profile (files.exclude)")
    if "stream list is incomplete" in text:
        hints.append("raise --msgidx-budget-mb so the record walk names every channel")
    if "separate video files" in text:
        hints.append("cameras in video files are not profiled yet; mark this stream required: false, "
                     "or match on the dataset's own camera columns")
    return "; ".join(dict.fromkeys(hints)) or None


def _check_stream(sr: StreamRequirement, views: list[_FileView]) -> Check:
    details = [_stream_file_status(sr, fv) for fv in views]
    n = len(details)
    n_pass = sum(1 for d in details if d["status"] == PASS)
    n_fail = sum(1 for d in details if d["status"] == FAIL)
    need = sr.coverage * n
    if n == 0:
        status = UNKNOWN
    elif n_pass >= need - 1e-9:
        status = PASS
    elif n - n_fail >= need - 1e-9:
        status = UNKNOWN
    else:
        status = FAIL

    thresholds = ", ".join(f"{k} {_fmt(v, UNITS[k])}" for k, v in sr.thresholds())
    expected = (f"{'every' if sr.min_count == 'all' else (str(sr.min_count) + ' or more')} stream(s) matching "
                f"[{sr.match.describe()}]" + (f" with {thresholds}" if thresholds else "")
                + (f", in {sr.coverage:.0%} of files" if sr.coverage < 1 else " in every file"))
    observed = f"satisfied in {n_pass} of {n} file(s)" + (f", failed in {n_fail}" if n_fail else "")
    evidence = []
    shown = [d for d in details if d["status"] != PASS][:5] or details[:3]
    for d in shown:
        evidence.append(f"{d['status'].upper()} {d['file']}: {d['detail']}")
    hidden = len([d for d in details if d["status"] != PASS]) - len([d for d in shown if d["status"] != PASS])
    if hidden > 0:
        evidence.append(f"... {hidden} more file(s) not passing")
    if sr.match.wants_video():
        evidence.append("note: image resolution and codec are not checked (payloads are not decoded)")
    return Check(id=f"stream:{sr.name}", label=sr.name, status=status, required=sr.required,
                 expected=expected, observed=observed, evidence=evidence,
                 resolve=_resolve_hint(details) if status == UNKNOWN else None, files=details,
                 headline=_headline(details))


# --------------------------------------------------------------------------- #
# Dataset checks
# --------------------------------------------------------------------------- #

def _episodes(views: list[_FileView]) -> tuple[list[float] | None, int | None, bool, str]:
    """(per-episode seconds if known, episode count, complete, source).

    Parquet files with an episode column give counts, but the report keeps only
    min/median/max lengths per file, so per-episode seconds are None there.
    Note: an episode split across two Parquet files is counted in both.
    """
    lerobot = [fv for fv in views if fv.format == "parquet" and (fv.basic.get("episodes") or {}).get("count")]
    if lerobot:
        count = sum(int(fv.basic["episodes"]["count"]) for fv in lerobot)
        complete = all(fv.basic["episodes"].get("complete", False) for fv in lerobot)
        return None, count, complete, f"Parquet episode column ({len(lerobot)} file(s))"
    timed = [fv for fv in views if fv.valid and fv.duration_s]
    if not timed:
        return None, None, False, "no file with a time range"
    return [float(fv.duration_s) for fv in timed], len(timed), len(timed) == len(views), \
        "one episode per file (no episode column)"


def _check_dataset(req: Requirements, views: list[_FileView], report: dict) -> list[Check]:
    d = req.dataset
    checks: list[Check] = []
    n = len(views)
    invalid = [fv for fv in views if not fv.valid]
    durations = [fv.duration_s for fv in views if fv.valid and fv.duration_s]
    missing_time = [fv.path for fv in views if fv.valid and not fv.duration_s]
    total_h = sum(durations) / 3600 if durations else 0.0
    selection_partial = bool(report.get("files_skipped")) and any(
        s.get("reason", "").startswith(("exceeds --max", "would exceed")) for s in report.get("files_skipped") or [])

    def add(key, status, expected, observed, evidence=None, resolve=None):
        checks.append(Check(id=f"dataset:{key}", label=key, status=status, required=True, expected=expected,
                            observed=observed, evidence=evidence or [], resolve=resolve, headline=observed))

    if d.min_total_hours is not None:
        observed = f"{total_h:.2f} h over {len(durations)} file(s)"
        if total_h >= d.min_total_hours:
            add("min_total_hours", PASS, f">= {_fmt(d.min_total_hours)} h", observed)
        elif invalid or missing_time or selection_partial:
            why = []
            if invalid:
                why.append(f"{len(invalid)} file(s) not profiled")
            if missing_time:
                why.append(f"{len(missing_time)} file(s) without a time range")
            if selection_partial:
                why.append("--max-files/--max-file-mb/--max-total-mb left files out")
            add("min_total_hours", UNKNOWN, f">= {_fmt(d.min_total_hours)} h", observed,
                ["below the threshold so far, but " + ", ".join(why)],
                "profile every file in scope (drop the --max-* limits) and fix unreadable files")
        else:
            add("min_total_hours", FAIL, f">= {_fmt(d.min_total_hours)} h", observed)

    if d.min_episodes is not None or d.min_episode_seconds is not None:
        secs, count, complete, source = _episodes(views)
        if d.min_episodes is not None:
            exp = f">= {d.min_episodes:,} episodes"
            if count is None:
                add("min_episodes", UNKNOWN, exp, "episodes not found", [source],
                    "episodes come from a Parquet episode column, or one per timed file")
            elif count >= d.min_episodes:
                add("min_episodes", PASS, exp, f"{count:,} ({source})")
            elif not complete or selection_partial:
                why, fix = [], []
                if selection_partial:
                    why.append("--max-* limits left files out")
                    fix.append("profile every file in scope (drop the --max-* limits)")
                if not complete:
                    if source.startswith("Parquet"):
                        why.append("the episode column was sampled")
                        fix.append("raise --column-budget-mb")
                    else:
                        why.append("some files have no time range")
                        fix.append("fix or exclude files without timing")
                add("min_episodes", UNKNOWN, exp, f"{count:,} so far ({source})",
                    ["below the threshold so far, but " + " and ".join(why)], "; ".join(fix))
            else:
                add("min_episodes", FAIL, exp, f"{count:,} ({source})")
        if d.min_episode_seconds is not None:
            exp = (f"episodes >= {_fmt(d.min_episode_seconds)} s"
                   + (f" (up to {_fmt(d.max_short_episodes_pct)}% may be shorter)" if d.max_short_episodes_pct else ""))
            if secs is None:
                # LeRobot: only per-file min is in the report, which still proves pass when min clears it.
                mins = [(fv.basic.get("episodes") or {}).get("seconds_min") for fv in views
                        if fv.format == "parquet" and fv.basic.get("episodes")]
                if mins and all(m is not None and m >= d.min_episode_seconds for m in mins):
                    add("min_episode_seconds", PASS, exp, f"shortest episode {min(mins):g} s ({source})")
                else:
                    add("min_episode_seconds", UNKNOWN, exp,
                        f"shortest episode {min(m for m in mins if m is not None):g} s" if any(
                            m is not None for m in mins) else "episode lengths unknown",
                        ["per-episode lengths are summarised per file (min/median/max), so the share of short "
                         "episodes cannot be computed exactly"] if mins else [source],
                        "check the dataset's episode metadata (e.g. LeRobot meta/episodes) for exact lengths")
            else:
                short = [s for s in secs if s < d.min_episode_seconds]
                pct = 100 * len(short) / len(secs)
                obs = f"{len(short)} of {len(secs)} shorter ({pct:.1f}%); shortest {min(secs):.1f} s ({source})"
                if pct <= d.max_short_episodes_pct:
                    add("min_episode_seconds", PASS, exp, obs)
                elif not complete:
                    add("min_episode_seconds", UNKNOWN, exp, obs, ["some files have no time range"])
                else:
                    add("min_episode_seconds", FAIL, exp, obs)

    if d.same_schema_across_files:
        by_topic: dict[str, set[str]] = {}
        for fv in views:
            for c in fv.channels:
                by_topic.setdefault(str(c.get("topic")), set()).add(str(c.get("schema")))
        clash = {t: s for t, s in by_topic.items() if len(s) > 1}
        topic_sets = {frozenset(str(c.get("topic")) for c in fv.channels) for fv in views if fv.valid and fv.channels}
        exp = "each stream has one type in every file"
        if not by_topic:
            add("same_schema_across_files", UNKNOWN, exp, "no streams read")
        elif clash:
            ev = [f"{t}: {', '.join(sorted(s))}" for t, s in list(clash.items())[:5]]
            add("same_schema_across_files", FAIL, exp, f"{len(clash)} stream(s) change type", ev)
        else:
            ev = [f"{len(topic_sets)} different stream sets across files"] if len(topic_sets) > 1 else []
            add("same_schema_across_files", UNKNOWN if invalid else PASS, exp,
                f"{len(by_topic)} stream(s), one type each", ev + (
                    [f"{len(invalid)} file(s) not profiled"] if invalid else []))

    if d.max_invalid_files_pct is not None:
        pct = 100 * len(invalid) / n if n else 0.0
        st = PASS if pct <= d.max_invalid_files_pct else FAIL
        add("max_invalid_files_pct", st, f"<= {_fmt(d.max_invalid_files_pct)}% of files unreadable",
            f"{len(invalid)} of {n} ({pct:.1f}%)", [f"{fv.path}: {fv.error}" for fv in invalid[:5]])

    if d.max_unindexed_files_pct is not None:
        unidx = [fv for fv in views if fv.finding_types & UNINDEXED]
        pct = 100 * len(unidx) / n if n else 0.0
        st = PASS if pct <= d.max_unindexed_files_pct else FAIL
        add("max_unindexed_files_pct", st, f"<= {_fmt(d.max_unindexed_files_pct)}% unindexed or truncated",
            f"{len(unidx)} of {n} ({pct:.1f}%)", [fv.path for fv in unidx[:5]])
    return checks


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def evaluate(report: dict, req: Requirements) -> Verdict:
    views, out_of_scope = _views(report, req)
    checks = [_check_stream(sr, views) for sr in req.streams] + _check_dataset(req, views, report)
    notes = []
    if not views:
        notes.append("no profiled files are in scope for these requirements"
                     + (f" ({len(out_of_scope)} excluded by files.include/exclude)" if out_of_scope else ""))
    skipped = report.get("files_skipped") or []
    limited = [s for s in skipped if str(s.get("reason", "")).startswith(("exceeds --max", "would exceed"))]
    if limited:
        notes.append(f"{len(limited)} file(s) were not profiled because of --max-files/--max-file-mb/--max-total-mb; "
                     "the verdict covers the profiled files only")

    required = [c for c in checks if c.required]
    if not views:
        status = UNKNOWN
    elif any(c.status == FAIL for c in required):
        status = FAIL
    elif any(c.status == UNKNOWN for c in required):
        status = UNKNOWN
    else:
        status = PASS
    return Verdict(
        requirements=req.name, requirements_source=req.source, repo_id=report.get("repo_id", ""),
        status=status,
        passed=sum(1 for c in checks if c.status == PASS),
        failed=sum(1 for c in checks if c.status == FAIL),
        unknown=sum(1 for c in checks if c.status == UNKNOWN),
        checks=checks,
        scope={"files_in_scope": len(views), "files_out_of_scope": len(out_of_scope),
               "files_invalid": sum(1 for v in views if not v.valid), "files_skipped": len(skipped)},
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

BADGE = {PASS: "PASS", FAIL: "FAIL", UNKNOWN: "UNKNOWN"}


def render_verdict_markdown(v: Verdict, level: int = 2) -> str:
    h = "#" * level
    L = [f"{h} Verdict: **{BADGE[v.status]}** against `{v.requirements}`", ""]
    L.append(f"{v.passed} passed, {v.failed} failed, {v.unknown} unknown; "
             f"{v.scope['files_in_scope']} file(s) in scope. Requirements: `{v.requirements_source}`.")
    for n in v.notes:
        L.append(f"- Note: {n}")
    L.append("")
    L.append("| Check | Status | Required | Expected | Observed |")
    L.append("|---|---|---|---|---|")
    for c in v.checks:
        L.append(f"| {c.label.replace('|', '/')} | {BADGE[c.status]} | {'yes' if c.required else 'no'} | "
                 f"{c.expected.replace('|', '/')} | {c.observed.replace('|', '/')} |")
    L.append("")
    for c in v.checks:
        if not c.evidence and not c.resolve:
            continue
        L.append(f"{h}# {c.label}: {BADGE[c.status]}")
        L.append("")
        for e in c.evidence:
            L.append(f"- {e}")
        if c.resolve:
            L.append(f"- To settle: {c.resolve}")
        L.append("")
    return "\n".join(L)


def print_verdict(v: Verdict, log) -> None:
    log("")
    log(f"VERDICT: {BADGE[v.status]}  ({v.requirements}; {v.passed} pass, {v.failed} fail, {v.unknown} unknown)")
    for c in v.checks:
        tag = "" if c.required else " (optional)"
        log(f"  {BADGE[c.status]:<8} {c.label}{tag}: {c.observed}")
        if c.status != PASS:
            for e in c.evidence[:2]:
                log(f"           {e}")
            if c.resolve:
                log(f"           to settle: {c.resolve}")
    for n in v.notes:
        log(f"  note: {n}")


def verdict_to_dict(v: Verdict) -> dict[str, Any]:
    from dataclasses import asdict
    return {**asdict(v), "exit_code": v.exit_code}
