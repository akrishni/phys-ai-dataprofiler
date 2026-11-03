"""
Subcommands next to the default `phys-ai-dataprofiler org/dataset` profile:

    phys-ai-dataprofiler check org/dataset -r req.yaml        profile + verdict; exit 0 pass, 1 fail, 4 unknown
    phys-ai-dataprofiler check --report run.json -r req.yaml  verdict from an existing report, no network
    phys-ai-dataprofiler compare a.json b.json -r req.yaml    rank saved reports against the same requirements
    phys-ai-dataprofiler compare org/a org/b -r req.yaml      profile each repo, then rank
    phys-ai-dataprofiler requirements list                    built-in presets
    phys-ai-dataprofiler requirements show teleop-imitation   print a preset (start your own from it)
    phys-ai-dataprofiler requirements validate req.yaml       check a file without profiling anything

Exit codes: 0 pass (or success), 1 fail, 2 usage or requirements error,
3 nothing to profile, 4 unknown (the profile could not settle every required check).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from phys_ai_dataprofiler.compare import compare_reports, compare_to_dict, print_compare, render_compare_markdown
from phys_ai_dataprofiler.evaluate import evaluate, print_verdict, render_verdict_markdown, verdict_to_dict
from phys_ai_dataprofiler.requirements import (
    RequirementsError,
    list_presets,
    load_requirements,
    preset_text,
)

PROG = "phys-ai-dataprofiler"


def _log(quiet: bool):
    return (lambda m: None) if quiet else (lambda m: print(m, file=sys.stderr))


def _load_req_or_exit(spec: str):
    try:
        return load_requirements(spec)
    except RequirementsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


def _read_report(path: str) -> dict:
    try:
        rep = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: cannot read report {path}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if not isinstance(rep, dict) or "files" not in rep:
        print(f"Error: {path} is not a profiler run report (no 'files')", file=sys.stderr)
        raise SystemExit(2)
    return rep


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #

def cmd_check(argv: list[str]) -> int:
    from phys_ai_dataprofiler.connectors.huggingface import build_parser, profile_main

    if "--report" in argv or any(a.startswith("--report=") for a in argv):
        p = argparse.ArgumentParser(prog=f"{PROG} check",
                                    description="Verdict for an existing run report (no network).")
        p.add_argument("--report", required=True, help="JSON run report written by an earlier profile")
        p.add_argument("--requirements", "-r", required=True, metavar="FILE|preset:NAME")
        p.add_argument("--output", "-o", help="Write the verdict here (.md or .json; default: print only)")
        p.add_argument("--quiet", action="store_true")
        a = p.parse_args(argv)
        req = _load_req_or_exit(a.requirements)
        v = evaluate(_read_report(a.report), req)
        print_verdict(v, _log(a.quiet))
        if a.output:
            out = Path(a.output)
            out.write_text(json.dumps(verdict_to_dict(v), indent=2) if out.suffix == ".json"
                           else render_verdict_markdown(v, level=1))
            _log(a.quiet)(f"Verdict: {out}")
        return v.exit_code

    p = build_parser(prog=f"{PROG} check")
    p.description = ("Profile a repo and check it against requirements. Exit code 0 pass, 1 fail, 4 unknown. "
                     "With --report FILE, check an existing report instead (no network).")
    args = p.parse_args(argv)
    if not args.requirements:
        p.error("check needs --requirements FILE or --requirements preset:NAME")
    return profile_main(args)


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

def cmd_compare(argv: list[str]) -> int:
    from phys_ai_dataprofiler.connectors.huggingface import build_parser, run, NothingToProfile, _to_jsonable

    p = argparse.ArgumentParser(
        prog=f"{PROG} compare",
        description="Rank datasets against the same requirements. Each target is a saved JSON run report "
                    "or a Hub repo id (profiled now with the default settings plus any options after --).",
        epilog="Example: compare lerobot/pusht org/other -r preset:teleop-imitation -- --max-files 20",
    )
    p.add_argument("targets", nargs="+", help="report.json files and/or Hub repo ids (at least two)")
    p.add_argument("--requirements", "-r", required=True, metavar="FILE|preset:NAME")
    p.add_argument("--output", "-o", help="Write the comparison here (.md or .json; default: "
                                          "reports/compare_<requirements>_<time>.md and .json)")
    p.add_argument("--quiet", action="store_true")
    if "--" in argv:
        i = argv.index("--")
        argv, profile_opts = argv[:i], argv[i + 1:]
    else:
        profile_opts = []
    a = p.parse_args(argv)
    if len(a.targets) < 2:
        p.error("compare needs at least two targets")
    log = _log(a.quiet)
    req = _load_req_or_exit(a.requirements)

    reports: list[tuple[str, dict]] = []
    for t in a.targets:
        if t.endswith(".json") or Path(t).is_file():
            reports.append((Path(t).stem, _read_report(t)))
            continue
        log(f"== profiling {t}")
        args = build_parser().parse_args([t, *profile_opts, *(["--quiet"] if a.quiet else [])])
        if req.files_include and not args.include and not args.file:
            args.include = list(req.files_include)
        if req.files_exclude:
            args.exclude = list(args.exclude or []) + list(req.files_exclude)
        try:
            rep = run(args)
        except NothingToProfile as exc:
            log(f"   {exc}")
            rep = None
        except RuntimeError as exc:
            log(f"   Error: {exc}")
            rep = None
        if rep is None:
            reports.append((t, {"repo_id": t, "files": [], "totals": {}, "files_skipped": []}))
        else:
            reports.append((t, _to_jsonable(rep)))

    cands = compare_reports(reports, req)
    print_compare(cands, log)
    md = render_compare_markdown(cands, req)
    data = compare_to_dict(cands, req)
    if a.output:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2) if out.suffix == ".json" else md)
        log(f"Comparison: {out}")
    else:
        d = Path("reports")
        d.mkdir(exist_ok=True)
        base = d / f"compare_{req.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base.with_suffix(".md").write_text(md)
        base.with_suffix(".json").write_text(json.dumps(data, indent=2))
        log(f"Comparison: {base}.md and .json")
    best = cands[0].verdict.status
    return {"pass": 0, "fail": 1, "unknown": 4}[best]


# --------------------------------------------------------------------------- #
# requirements
# --------------------------------------------------------------------------- #

def cmd_requirements(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog=f"{PROG} requirements", description="Presets and requirements files.")
    sp = p.add_subparsers(dest="action", required=True)
    sp.add_parser("list", help="List built-in presets")
    s = sp.add_parser("show", help="Print a preset's YAML (redirect to a file to start your own)")
    s.add_argument("preset")
    v = sp.add_parser("validate", help="Check a requirements file without profiling anything")
    v.add_argument("spec", metavar="FILE|preset:NAME")
    a = p.parse_args(argv)

    if a.action == "list":
        for name, desc in list_presets():
            print(f"preset:{name}\n    {desc}")
        return 0
    if a.action == "show":
        try:
            print(preset_text(a.preset.removeprefix("preset:")), end="")
        except RequirementsError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    try:
        req = load_requirements(a.spec)
    except RequirementsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"OK: {req.name}: {len(req.streams)} stream requirement(s)"
          f" ({sum(s.required for s in req.streams)} required), "
          f"{len(req.dataset.active())} dataset requirement(s)")
    for s in req.streams:
        th = ", ".join(f"{k}={val:g}" for k, val in s.thresholds())
        print(f"  stream '{s.name}'{'' if s.required else ' (optional)'}: [{s.match.describe()}]"
              + (f" {th}" if th else ""))
    for k in req.dataset.active():
        print(f"  dataset {k} = {getattr(req.dataset, k)}")
    return 0


def main(argv: list[str]) -> int:
    cmd, rest = argv[0], argv[1:]
    try:
        return {"check": cmd_check, "compare": cmd_compare, "requirements": cmd_requirements}[cmd](rest)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
