#!/usr/bin/env python3
"""Checks for requirements files, the evaluator, compare and the CLI commands.

    python tests/test_requirements.py

Builds run reports in memory (and one real report from a local MCAP via the CLI's
offline mode), so no network is needed. Exits non-zero on the first failed check.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phys_ai_dataprofiler.compare import compare_reports, render_compare_markdown  # noqa: E402
from phys_ai_dataprofiler.evaluate import (  # noqa: E402
    FAIL, PASS, UNKNOWN, compare, evaluate, render_verdict_markdown, verdict_to_dict)
from phys_ai_dataprofiler.requirements import (  # noqa: E402
    RequirementsError, list_presets, load_requirements, parse_requirements)

TMP = Path(tempfile.mkdtemp(prefix="req_test_"))


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def errors_of(doc) -> list[str]:
    try:
        parse_requirements(doc, "t.yaml")
    except RequirementsError as exc:
        return exc.errors
    return []


def ch(topic, modality="other", schema="X", rate=None, gap=None, gaps=0, count=100, nan=0, jitter=None):
    return {"topic": topic, "schema": schema, "modality": modality, "msg_count": count, "rate_hz": rate,
            "jitter_ms": jitter, "gaps_over_3x": gaps, "max_gap_ms": gap, "nan_count": nan}


def file(path, channels, fmt="mcap", duration=60.0, valid=True, findings=(), basic=None, error=None):
    return {"repo_path": path, "detected_format": fmt, "valid": valid, "error": error, "basic": basic or {},
            "summary": {"duration_s": duration},
            "rich": {"indexed": True, "channels": channels,
                     "findings": [{"severity": "info", "type": t, "topic": None, "detail": ""} for t in findings]}}


def report(files, skipped=(), repo="org/ds"):
    return {"repo_id": repo, "files": files, "files_skipped": list(skipped),
            "totals": {"files": len(files), "duration_s": sum(f["summary"]["duration_s"] or 0 for f in files),
                       "invalid_or_errored": sum(not f["valid"] for f in files), "bytes": 1, "bytes_fetched": 1}}


def status(rep, doc, cid=None):
    v = evaluate(rep, parse_requirements(doc, "t.yaml"))
    if cid is None:
        return v.status
    return next(c.status for c in v.checks if c.id == cid)


CAM = {"name": "cam", "match": {"modality": "video"}, "min_rate_hz": 25}


def test_parsing() -> None:
    for name, _ in list_presets():
        req = load_requirements(f"preset:{name}")
        check(req.streams or req.dataset.active(), f"preset {name} loads")
    check(len(list_presets()) >= 4, "four or more presets ship")

    e = errors_of({"streams": [{"name": "c", "match": {"modality": "video"}, "min_rate": 25}]})
    check(any("did you mean 'min_rate_hz'" in x for x in e), "typo gets a suggestion")
    e = errors_of({"streams": [{"name": "c", "match": {"modality": "video"}, "min_resolution": [640, 480]}]})
    check(any("not supported" in x and "decode" in x for x in e), "unmeasurable field rejected with the reason")
    e = errors_of({"streams": [{"name": "c", "match": {"modality": "vidoe"}}]})
    check(any("did you mean 'video'" in x for x in e), "unknown modality gets a suggestion")
    e = errors_of({"streams": [{"name": "c", "match": {"modality": "video"}, "min_rate_hz": "fast"}]})
    check(any("expected a number" in x for x in e), "non-numeric threshold rejected")
    e = errors_of({"streams": [{"name": "c", "match": {"modality": "video"}, "min_rate_hz": 30, "max_rate_hz": 10}]})
    check(any("above max_rate_hz" in x for x in e), "min above max rejected")
    e = errors_of({"min_total_hours": 5})
    check(any("belongs under dataset" in x for x in e), "dataset field at top level points to dataset")
    e = errors_of({"streams": [{"name": "c", "modality": "video"}]})
    check(any("belongs under match" in x for x in e) and any("match: required" in x for x in e),
          "match field at stream level points to match")
    e = errors_of({"streams": [{"name": "a", "match": {"topic": "*"}}, {"name": "a", "match": {"topic": "*"}}]})
    check(any("duplicate" in x for x in e), "duplicate stream names rejected")
    e = errors_of({"version": 2, "streams": [CAM]})
    check(any("unsupported version" in x for x in e), "unknown schema version rejected")
    e = errors_of({"dataset": {"max_invalid_files_pct": 150}})
    check(any("percentage" in x for x in e), "percentage over 100 rejected")
    e = errors_of({})
    check(any("no requirements" in x for x in e), "empty file rejected")
    e = errors_of({"streams": [{"match": {"topic": 3}}], "dataset": {"min_episodes": 2.5}})
    check(len(e) >= 3, f"every problem reported at once ({len(e)} errors)")

    bad = TMP / "bad.yaml"
    bad.write_text("streams:\n  - name: x\n    match: {modality: video\n")
    try:
        load_requirements(str(bad))
        check(False, "broken YAML raises")
    except RequirementsError as exc:
        check("line" in str(exc), "broken YAML error names the line")
    try:
        load_requirements("preset:teleop")
        check(False, "unknown preset raises")
    except RequirementsError as exc:
        check("did you mean 'teleop-imitation'" in str(exc), "unknown preset gets a suggestion")


def test_compare_margin() -> None:
    check(compare("min", 30, 25, False) == PASS and compare("min", 24.9, 25, False) == FAIL, "exact: plain comparison")
    check(compare("min", 25, 25, False) == PASS, "exact: threshold itself passes")
    check(compare("min", 9.99999, 10, False) == PASS, "exact: float rounding at the threshold passes (9.99999 Hz)")
    check(compare("min", 9.9, 10, False) == FAIL, "exact: 1% under still fails")
    check(compare("max", 0, 0, False) == PASS and compare("max", 1, 0, False) == FAIL, "zero limit has no tolerance")
    check(compare("min", 28, 25, True) == PASS, "estimate clearing the 10% margin passes")
    check(compare("min", 26, 25, True) == UNKNOWN, "estimate within the margin is unknown")
    check(compare("min", 20, 25, True) == FAIL, "estimate far below fails")
    check(compare("max", 80, 100, True) == PASS and compare("max", 105, 100, True) == UNKNOWN, "max side margin")
    check(compare("min", None, 25, False) == UNKNOWN, "missing value is unknown")


def test_streams() -> None:
    good = report([file("a.mcap", [ch("/cam/front", "video", rate=30, gap=40)]),
                   file("b.mcap", [ch("/cam/front", "video", rate=29, gap=45)])])
    check(status(good, {"streams": [CAM]}) == PASS, "camera at 30 Hz meets 25 Hz")
    slow = report([file("a.mcap", [ch("/cam/front", "video", rate=12)])])
    v = evaluate(slow, parse_requirements({"streams": [CAM]}, "t"))
    check(v.status == FAIL and "needs >= 25 Hz" in v.checks[0].evidence[0], "slow camera fails with evidence")
    check(v.exit_code == 1, "fail exit code is 1")

    sampled = report([file("a.mcap", [ch("/cam", "video", rate=26)], findings=["sampled_message_index"])])
    v = evaluate(sampled, parse_requirements({"streams": [CAM]}, "t"))
    check(v.status == UNKNOWN and v.exit_code == 4, "sampled rate near threshold is unknown, exit 4")
    check("msgidx-budget-mb" in (v.checks[0].resolve or ""), "unknown says which budget to raise")
    sampled_ok = report([file("a.mcap", [ch("/cam", "video", rate=30)], findings=["sampled_message_index"])])
    check(status(sampled_ok, {"streams": [CAM]}) == PASS, "sampled rate well above passes")

    missing = report([file("a.mcap", [ch("/imu", "imu", rate=200)])])
    check(status(missing, {"streams": [CAM]}) == FAIL, "absent required stream fails")
    opt = {"streams": [{**CAM, "required": False}, {"name": "imu", "match": {"modality": "imu"}}]}
    v = evaluate(missing, parse_requirements(opt, "t"))
    check(v.status == PASS and v.checks[0].status == FAIL, "optional failure does not change the overall result")

    unnamed = report([file("a.mcap", [ch("/imu", "imu", rate=200)], findings=["unknown_channels"])])
    check(status(unnamed, {"streams": [CAM]}) == UNKNOWN, "absent stream with an incomplete stream list is unknown")

    lerobot = report([file("data/c0.parquet", [ch("observation.state", "proprio", rate=10)], fmt="parquet")])
    check(status(lerobot, {"streams": [CAM]}) == UNKNOWN, "Parquet without image columns: camera unknown, not fail")

    alias = report([file("d.parquet", [ch("observation.images.top", "image", rate=30)], fmt="parquet")])
    check(status(alias, {"streams": [CAM]}) == PASS, "'image' modality satisfies a 'video' match")

    globs = report([file("a.mcap", [ch("/zed/left/image_raw", "video", rate=30), ch("/zed/left/camera_info", "video", rate=30)])])
    req = {"streams": [{"name": "c", "match": {"topic": "*image*", "exclude_topic": "*info*"}, "min_count": "all",
                        "min_rate_hz": 25}]}
    v = evaluate(globs, parse_requirements(req, "t"))
    check(v.status == PASS and v.checks[0].files[0]["matched"] == ["/zed/left/image_raw"], "topic glob and exclude")

    two = report([file("a.mcap", [ch("/cam1", "video", rate=30), ch("/cam2", "video", rate=10)])])
    check(status(two, {"streams": [{**CAM, "min_count": 2}]}) == FAIL, "min_count 2 needs two streams passing")
    check(status(two, {"streams": [{**CAM, "min_count": 1}]}) == PASS, "min_count 1 needs one")
    check(status(two, {"streams": [{**CAM, "min_count": "all"}]}) == FAIL, "min_count all needs every match")

    cov = report([file(f"{i}.mcap", [ch("/cam", "video", rate=30 if i < 9 else 5)]) for i in range(10)])
    check(status(cov, {"streams": [CAM]}) == FAIL, "coverage 1.0: one bad file fails")
    check(status(cov, {"streams": [{**CAM, "coverage": 0.9}]}) == PASS, "coverage 0.9: 9 of 10 passes")

    gap = report([file("a.bag", [ch("/joint_states", "proprio", rate=100, gap=250, gaps=3)], fmt="bag1")])
    js = {"name": "js", "match": {"modality": "proprio"}, "max_gap_ms": 100, "max_gaps": 5}
    v = evaluate(gap, parse_requirements({"streams": [js]}, "t"))
    check(v.status == FAIL and "max_gap_ms: 250 ms" in v.checks[0].evidence[0], "worst gap over the limit fails")
    noidx = report([file("a.mcap", [ch("/js", "proprio", rate=100, gap=None, count=500)])])
    check(status(noidx, {"streams": [js]}) == UNKNOWN, "no per-message timestamps: gap is unknown")

    nan = {"name": "f", "match": {"topic": "*"}, "min_count": "all", "max_nan": 0}
    check(status(report([file("a.mat", [ch("v", "data", nan=0)], fmt="mat")]), {"streams": [nan]}) == PASS,
          "MAT with a full NaN scan and no NaN passes")
    check(status(report([file("a.mat", [ch("v", "data", nan=3)], fmt="mat")]), {"streams": [nan]}) == FAIL,
          "MAT with NaNs fails")
    check(status(report([file("a.mat", [ch("v", "data", nan=0)], fmt="mat", findings=["nan_check_partial"])]),
                 {"streams": [nan]}) == UNKNOWN, "partial NaN scan with zero found is unknown")
    check(status(report([file("a.mat", [ch("v", "data", nan=2)], fmt="mat", findings=["nan_check_partial"])]),
                 {"streams": [nan]}) == FAIL, "partial NaN scan that found NaNs still fails")
    v = evaluate(report([file("a.mcap", [ch("/x", "data")])]), parse_requirements({"streams": [nan]}, "t"))
    check(v.status == UNKNOWN and "not counted for mcap" in v.checks[0].evidence[0],
          "MCAP NaN requirement is unknown, never a false pass")

    broken = report([file("a.mcap", [ch("/cam", "video", rate=30)]), file("b.mcap", [], valid=False, error="HTTP 500")])
    check(status(broken, {"streams": [CAM]}) == UNKNOWN, "unreadable file makes a full-coverage check unknown")
    scoped = parse_requirements({"files": {"exclude": "b.*"}, "streams": [CAM]}, "t")
    check(evaluate(broken, scoped).status == PASS, "files.exclude takes a file out of scope")
    check(evaluate(report([]), parse_requirements({"streams": [CAM]}, "t")).status == UNKNOWN,
          "no files in scope is unknown")


def test_dataset() -> None:
    files = [file(f"{i}.mcap", [ch("/cam", "video", rate=30)], duration=1800.0) for i in range(4)]
    rep = report(files)
    check(status(rep, {"dataset": {"min_total_hours": 2}}, "dataset:min_total_hours") == PASS, "2 h of 2 h passes")
    check(status(rep, {"dataset": {"min_total_hours": 3}}, "dataset:min_total_hours") == FAIL, "2 h of 3 h fails")
    lim = report(files, skipped=[{"path": "x.mcap", "reason": "exceeds --max-files"}])
    check(status(lim, {"dataset": {"min_total_hours": 3}}, "dataset:min_total_hours") == UNKNOWN,
          "short only because --max-files left files out: unknown")
    check(status(lim, {"dataset": {"min_total_hours": 1}}, "dataset:min_total_hours") == PASS,
          "already over the threshold despite --max-files: pass")

    check(status(rep, {"dataset": {"min_episodes": 4}}, "dataset:min_episodes") == PASS, "one episode per file")
    ep = {"count": 206, "complete": True, "seconds_min": 4.2, "seconds_median": 12, "seconds_max": 30}
    pq = report([file("data/c0.parquet", [ch("observation.state", "proprio", rate=10)], fmt="parquet",
                      basic={"episodes": ep})])
    check(status(pq, {"dataset": {"min_episodes": 200}}, "dataset:min_episodes") == PASS, "Parquet episode column")
    check(status(pq, {"dataset": {"min_episodes": 300}}, "dataset:min_episodes") == FAIL, "too few episodes fails")
    pq_s = report([file("c.parquet", [ch("s", "proprio")], fmt="parquet", basic={"episodes": {**ep, "complete": False}})])
    check(status(pq_s, {"dataset": {"min_episodes": 300}}, "dataset:min_episodes") == UNKNOWN,
          "sampled episode column: unknown")
    check(status(pq, {"dataset": {"min_episode_seconds": 4}}, "dataset:min_episode_seconds") == PASS,
          "shortest Parquet episode over the limit passes")
    check(status(pq, {"dataset": {"min_episode_seconds": 5}}, "dataset:min_episode_seconds") == UNKNOWN,
          "shortest under the limit: share unknown from summaries")

    short = report([file(f"{i}.bag", [ch("/x")], fmt="bag1", duration=d) for i, d in enumerate([2, 10, 12, 15])])
    check(status(short, {"dataset": {"min_episode_seconds": 5}}, "dataset:min_episode_seconds") == FAIL,
          "one short file-episode fails with 0% allowed")
    check(status(short, {"dataset": {"min_episode_seconds": 5, "max_short_episodes_pct": 25}},
                 "dataset:min_episode_seconds") == PASS, "25% short allowed passes")

    same = report([file("a.bag", [ch("/js", schema="sensor_msgs/JointState")]),
                   file("b.bag", [ch("/js", schema="sensor_msgs/JointState")])])
    diff = report([file("a.bag", [ch("/js", schema="sensor_msgs/JointState")]),
                   file("b.bag", [ch("/js", schema="custom/JointState")])])
    rq = {"dataset": {"same_schema_across_files": True}}
    check(status(same, rq) == PASS and status(diff, rq) == FAIL, "schema drift across files fails")

    inv = report([file("a.mcap", [ch("/x")]), file("b.mcap", [], valid=False, error="boom")])
    check(status(inv, {"dataset": {"max_invalid_files_pct": 0}}) == FAIL, "unreadable files over the limit fail")
    check(status(inv, {"dataset": {"max_invalid_files_pct": 50}}) == PASS, "50% allowed passes")
    un = report([file("a.mcap", [ch("/x")], findings=["unindexed_mcap"]), file("b.mcap", [ch("/x")])])
    check(status(un, {"dataset": {"max_unindexed_files_pct": 0}}) == FAIL, "unindexed files counted")


def test_compare() -> None:
    req = parse_requirements({"streams": [CAM], "dataset": {"min_total_hours": 1}}, "t")
    a = report([file("a.mcap", [ch("/cam", "video", rate=12)], duration=7200)], repo="org/slow")
    b = report([file("a.mcap", [ch("/cam", "video", rate=30)], duration=7200)], repo="org/good")
    c = report([file("a.mcap", [ch("/cam", "video", rate=26)], duration=7200, findings=["sampled_message_index"])],
               repo="org/maybe")
    ranked = compare_reports([("slow", a), ("maybe", c), ("good", b)], req)
    check([x.label for x in ranked] == ["good", "maybe", "slow"], "ranking: pass, then unknown, then fail")
    md = render_compare_markdown(ranked, req)
    check("1. `good`" in md and "why not pass" in md and "30 Hz" in md, "comparison table shows rates and reasons")
    again = compare_reports([("slow", a), ("maybe", c), ("good", b)], req)
    check(render_compare_markdown(again, req) == md, "same inputs give the same comparison")


def cli(*args, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "phys_ai_dataprofiler", *args], capture_output=True, text=True,
                          cwd=cwd or TMP, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})


def test_cli() -> None:
    r = cli("requirements", "list")
    check(r.returncode == 0 and "preset:teleop-imitation" in r.stdout, "requirements list")
    r = cli("requirements", "show", "visual-odometry")
    check(r.returncode == 0 and yaml.safe_load(r.stdout)["name"] == "visual-odometry", "requirements show prints YAML")
    bad = TMP / "typo.yaml"
    bad.write_text("streams:\n  - name: cam\n    match: {modality: video}\n    min_rate: 25\n")
    r = cli("requirements", "validate", str(bad))
    check(r.returncode == 2 and "min_rate_hz" in r.stderr, "validate reports the typo, exit 2")
    r = cli("check", "org/ds", "-r", str(bad))
    check(r.returncode == 2 and "min_rate_hz" in r.stderr, "check refuses bad requirements before any network call")

    # A real report from a local MCAP through the offline download path.
    repo = "local/sample"
    root = TMP / "hf_downloads" / f"dataset__{repo.replace('/', '__')}"
    root.mkdir(parents=True)
    shutil.copy(ROOT / "samples" / "sample.mcap", root / "sample.mcap")
    req = TMP / "req.yaml"
    req.write_text(yaml.safe_dump({"name": "any-stream", "streams": [{"name": "any", "match": {"topic": "*"}}],
                                   "dataset": {"max_invalid_files_pct": 0}}))
    r = cli(repo, "--offline", "--ext", ".mcap", "-r", str(req), "--report-name", "run")
    check(r.returncode == 0 and "VERDICT: PASS" in r.stderr, f"profile with -r: verdict printed, exit {r.returncode}")
    rep = json.loads((TMP / "reports" / "run.json").read_text())
    check(rep["verdict"]["status"] == "pass", "verdict stored in the JSON report")
    check("Verdict: **PASS**" in (TMP / "reports" / "run.md").read_text(), "verdict at the top of the Markdown")

    strict = TMP / "strict.yaml"
    strict.write_text(yaml.safe_dump({"streams": [{"name": "cam", "match": {"modality": "video"}, "min_rate_hz": 25}]}))
    r = cli("check", "--report", "reports/run.json", "-r", str(strict), "-o", "verdict.json")
    check(r.returncode == 1 and json.loads((TMP / "verdict.json").read_text())["status"] == "fail",
          "check --report: no network, exit 1 on fail, JSON verdict written")
    r = cli("compare", "reports/run.json", "reports/run.json", "-r", str(req), "-o", "cmp.md")
    check(r.returncode == 0 and "Dataset comparison" in (TMP / "cmp.md").read_text(), "compare two saved reports")
    r = cli("compare", "reports/run.json", "-r", str(req))
    check(r.returncode == 2, "compare needs two targets")


def test_parsing_edges() -> None:
    for doc in ([], None, "streams", 3):
        try:
            parse_requirements(doc, "t.yaml")
            check(False, f"top-level {type(doc).__name__} rejected")
        except RequirementsError as exc:
            check("mapping at the top level" in str(exc), f"top-level {type(doc).__name__} rejected")
    S = {"name": "s", "match": {"topic": "*"}}
    cases = [
        ({"streams": "cam"}, "expected a list"),
        ({"streams": ["cam"]}, "expected a mapping"),
        ({"files": ["a/*"], "streams": [S]}, "include and/or exclude"),
        ({"files": {"exlude": "b"}, "streams": [S]}, "did you mean 'exclude'"),
        ({"streams": [{"name": "s", "match": {"topic": []}}]}, "pattern or a list of patterns"),
        ({"streams": [{"name": "s", "match": {"topic": ""}}]}, "pattern or a list of patterns"),
        ({"streams": [{"name": "", "match": {"topic": "*"}}]}, "name: required"),
        ({"streams": [{**S, "min_count": 0}]}, "1 or more"),
        ({"streams": [{**S, "min_count": "some"}]}, "whole number"),
        ({"streams": [{**S, "coverage": 0}]}, "above 0"),
        ({"streams": [{**S, "coverage": 1.5}]}, "at most 1"),
        ({"streams": [{**S, "max_gaps": 2.5}]}, "whole number"),
        ({"streams": [{**S, "min_rate_hz": True}]}, "expected a number"),
        ({"streams": [{**S, "min_rate_hz": -1}]}, "0 or more"),
        ({"streams": [{**S, "required": "yes"}]}, "true or false"),
        ({"streams": [{"name": "s", "match": {"topic": "*", "min_rate_hz": 3}}]}, "belongs under the stream entry"),
        ({"dataset": {"min_episodes": 3, "match": {}}}, "belongs under a stream entry"),
        ({"dataset": {"max_short_episodes_pct": 10}}, "together with min_episode_seconds"),
        ({"dataset": {"same_schema_across_files": "yes"}}, "true or false"),
        ({"dataset": {"same_schema_across_files": False}}, "no requirements"),
        ({"dataset": "lots"}, "expected a mapping"),
        ({"name": 5, "streams": [S]}, "expected text"),
        ({"codec": "h264", "streams": [S]}, "not supported"),
    ]
    for doc, want in cases:
        e = errors_of(doc)
        check(any(want in x for x in e), f"rejected with '{want}': {str(doc)[:60]}")

    ok = parse_requirements({"streams": [{**S, "min_count": 2.0, "min_rate_hz": 10, "max_rate_hz": 10,
                                          "match": {"modality": "VIDEO"}}]}, "/some/dir/my-policy.yaml")
    s = ok.streams[0]
    check(s.min_count == 2 and isinstance(s.min_count, int), "min_count 2.0 is accepted as the integer 2")
    check(s.match.modality == ["video"], "modality is case-insensitive")
    check(ok.name == "my-policy", "name falls back to the file stem")
    check(parse_requirements({"streams": [S]}, "preset:abc").name == "abc", "name falls back to the preset name")

    for spec, want in (("nofile", "built-in presets"), ("nofile.yaml", "file not found"), ("preset:", "no preset named ''"),
                       ("preset:../presets/teleop-imitation", "no preset named"), ("preset:visual", "visual-odometry")):
        try:
            load_requirements(spec)
            check(False, f"load_requirements({spec!r}) raises")
        except RequirementsError as exc:
            check(want in str(exc), f"load_requirements({spec!r}): {want}")
    empty = TMP / "empty.yaml"
    empty.write_text("")
    try:
        load_requirements(str(empty))
        check(False, "empty YAML file raises")
    except RequirementsError as exc:
        check("mapping" in str(exc), "empty YAML file rejected as not a mapping")


def test_evaluate_edges() -> None:
    one = report([file("a.mcap", [ch("/cam", "video", rate=None, count=1)])])
    v = evaluate(one, parse_requirements({"streams": [CAM]}, "t"))
    check(v.status == UNKNOWN and "fewer than two messages" in v.checks[0].evidence[0],
          "single-message stream: rate unknown, says why")
    fast = report([file("a.mcap", [ch("/cam", "video", rate=60, jitter=5, gap=40)])])
    check(status(fast, {"streams": [{**CAM, "max_rate_hz": 40}]}) == FAIL, "max_rate_hz exceeded fails")
    check(status(fast, {"streams": [{"name": "j", "match": {"topic": "*"}, "max_jitter_ms": 2}]}) == FAIL,
          "max_jitter_ms exceeded fails")
    check(status(fast, {"streams": [{"name": "j", "match": {"topic": "*"}, "max_jitter_ms": 10}]}) == PASS,
          "max_jitter_ms met passes")

    # Sampled timing, zero-limit gap counts: an estimate of 0 cannot prove 0, an estimate of 1 proves > 0.
    js = {"name": "js", "match": {"modality": "proprio"}, "max_gaps": 0}
    smp0 = report([file("a.bag", [ch("/js", "proprio", rate=100, gap=12, gaps=0)], findings=["sampled_message_index"])])
    smp1 = report([file("a.bag", [ch("/js", "proprio", rate=100, gap=500, gaps=1)], findings=["sampled_message_index"])])
    check(status(smp0, {"streams": [js]}) == UNKNOWN, "sampled: no gaps seen is unknown for max_gaps 0")
    check(status(smp1, {"streams": [js]}) == FAIL, "sampled: a gap seen fails max_gaps 0")

    skipped = report([file("a.parquet", [ch("x", "proprio")], fmt="parquet", findings=["nan_check_skipped"])])
    v = evaluate(skipped, parse_requirements({"streams": [{"name": "f", "match": {"topic": "*"}, "max_nan": 0}]}, "t"))
    check(v.status == UNKNOWN and "nan-budget-mb" in (v.checks[0].resolve or ""), "NaN scan off: unknown, how to fix")

    js_schema = report([file("a.mcap", [ch("/js", schema="sensor_msgs/JointState")])])
    check(status(js_schema, {"streams": [{"name": "j", "match": {"schema": "*jointstate"}}]}) == PASS,
          "schema glob is case-insensitive")
    check(status(js_schema, {"streams": [{"name": "j", "match": {"topic": "/JS"}}]}) == PASS,
          "topic glob is case-insensitive")

    two = report([file("data/a.mcap", [ch("/cam", "video", rate=30)]), file("meta/b.mcap", [])])
    check(status(two, {"files": {"include": "data/*"}, "streams": [CAM]}) == PASS, "files.include narrows scope")
    v = evaluate(two, parse_requirements({"files": {"include": "nothing/*"}, "streams": [CAM]}, "t"))
    check(v.status == UNKNOWN and "2 excluded" in v.notes[0], "nothing in scope: unknown, says how many were excluded")

    unk_opt = report([file("a.mcap", [ch("/imu", "imu", rate=200)], findings=["unknown_channels"])])
    v = evaluate(unk_opt, parse_requirements({"streams": [{**CAM, "required": False},
                                                          {"name": "imu", "match": {"modality": "imu"}}]}, "t"))
    check(v.status == PASS and v.checks[0].status == UNKNOWN, "optional unknown does not change the overall result")

    good = [file(f"{i}.mcap", [ch("/cam", "video", rate=30)]) for i in range(8)]
    r = report(good + [file("x.mcap", [], valid=False, error="e"), file("y.mcap", [ch("/cam", "video", rate=5)])])
    check(status(r, {"streams": [{**CAM, "coverage": 0.9}]}) == UNKNOWN,
          "coverage 0.9: 8 pass, 1 unreadable, 1 fail is unknown (depends on the unreadable file)")
    r = report(good + [file("x.mcap", [ch("/cam", "video", rate=5)]), file("y.mcap", [ch("/cam", "video", rate=5)])])
    check(status(r, {"streams": [{**CAM, "coverage": 0.9}]}) == FAIL, "coverage 0.9: 8 of 10 fails")

    many = report([file(f"{i}.mcap", [ch("/cam", "video", rate=5)]) for i in range(8)])
    ev = evaluate(many, parse_requirements({"streams": [CAM]}, "t")).checks[0].evidence
    check(sum(e.startswith("FAIL") for e in ev) == 5 and any("3 more file(s)" in e for e in ev),
          "evidence lists 5 failing files and counts the rest")

    pipes = report([file("a.mcap", [ch("/cam", "video", rate=30)])])
    v = evaluate(pipes, parse_requirements({"streams": [{"name": "c", "match": {"topic": ["/a", "/cam"]}}]}, "t"))
    row = next(line for line in render_verdict_markdown(v).splitlines() if line.startswith("| c "))
    check(row.count("|") == 6, "glob alternatives (a|b) do not break the Markdown table")
    d = json.loads(json.dumps(verdict_to_dict(v)))
    check(d["exit_code"] == 0 and d["status"] == "pass", "verdict round-trips through JSON with its exit code")

    # Reports from older versions or failed files: missing or null sections must not raise.
    sparse = {"repo_id": "x", "files": [{"repo_path": "a.mcap", "valid": True, "rich": None}, {"repo_path": "b.mcap"}]}
    v = evaluate(sparse, parse_requirements({"streams": [CAM], "dataset": {"min_total_hours": 1,
                                             "same_schema_across_files": True, "max_invalid_files_pct": 50}}, "t"))
    check(v.status == UNKNOWN and len(v.checks) == 4, "report with null rich and missing fields: unknown, no crash")

    timeless = report([file("a.mcap", [ch("/x")], duration=None), file("b.mcap", [ch("/x")], duration=600)])
    for key, val in (("min_total_hours", 1), ("min_episodes", 5), ("min_episode_seconds", 700)):
        check(status(timeless, {"dataset": {key: val}}, f"dataset:{key}") == UNKNOWN,
              f"{key} short because a file has no time range: unknown")
    check(status(report([]), {"dataset": {"max_invalid_files_pct": 0}}) == UNKNOWN, "no files at all: unknown")
    check(status(report([file("a.mcap", [])]), {"dataset": {"same_schema_across_files": True}}) == UNKNOWN,
          "same_schema with no streams read: unknown")
    check(status(report([file("a.bag", [ch("/js", schema="A")]), file("b.bag", [], valid=False, error="x")]),
                 {"dataset": {"same_schema_across_files": True}}) == UNKNOWN,
          "same_schema with an unreadable file: unknown, not pass")
    no_min = report([file("c.parquet", [ch("s", "proprio")], fmt="parquet",
                          basic={"episodes": {"count": 3, "complete": True, "seconds_min": None}})])
    check(status(no_min, {"dataset": {"min_episode_seconds": 4}}) == UNKNOWN,
          "Parquet episodes without seconds_min: unknown, no crash")


def test_compare_edges() -> None:
    req = parse_requirements({"streams": [CAM]}, "t")
    a = report([file("a.mcap", [ch("/cam", "video", rate=30)], duration=3600)], repo="org/a")
    b = report([file("a.mcap", [ch("/cam", "video", rate=30)], duration=7200)], repo="org/b")
    empty = {"repo_id": "org/none", "files": [], "totals": {}, "files_skipped": []}
    ranked = compare_reports([("a", a), ("none", empty), ("b", b)], req)
    check([c.label for c in ranked] == ["b", "a", "none"], "equal verdicts rank by hours; a failed profile ranks last")
    tie = compare_reports([("x", a), ("y", a)], req)
    check([c.label for c in tie] == ["x", "y"], "exact ties keep the input order")
    piped = parse_requirements({"streams": [{**CAM, "name": "cam|front"}]}, "t")
    md = render_compare_markdown(compare_reports([("a|1", a), ("b", b)], piped), piped)
    rows = [line for line in md.splitlines() if line.startswith("| ")]
    check(all(r.count("|") == 4 for r in rows), "labels and stream names containing | do not add table columns")
    v = evaluate(a, piped)
    row = next(line for line in render_verdict_markdown(v).splitlines() if line.startswith("| cam"))
    check(row.count("|") == 6, "a stream name containing | does not break the verdict table")


def main() -> None:
    test_parsing()
    test_parsing_edges()
    test_compare_margin()
    test_streams()
    test_evaluate_edges()
    test_dataset()
    test_compare()
    test_compare_edges()
    test_cli()
    print("all checks passed")


if __name__ == "__main__":
    main()
