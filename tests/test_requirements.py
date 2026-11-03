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
from phys_ai_dataprofiler.evaluate import FAIL, PASS, UNKNOWN, compare, evaluate  # noqa: E402
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


def main() -> None:
    test_parsing()
    test_compare_margin()
    test_streams()
    test_dataset()
    test_compare()
    test_cli()
    print("all checks passed")


if __name__ == "__main__":
    main()
