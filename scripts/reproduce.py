"""Re-analyse the hardware jobs in a scratch folder and compare them with runs/.

    python scripts/reproduce.py analyze [--out build/reproduce] [--job NAME ...] [--jobs N]
    python scripts/reproduce.py compare [--out build/reproduce] [--job NAME ...] [--strict]
    python scripts/reproduce.py combine [--out build/recombine]

analyze copies each job folder of runs/, without its analysis.json, to OUT/runs/<job>,
runs heavyhex's analyze on the copy, and pools the copies with combine once every job
has been re-analysed. Nothing under runs/ is written: `heavyhex experiment analyze` and
`combine` would overwrite the stored files, and an OUT inside runs/, or whose runs/ is the
repository's own, is refused. The fifteen jobs take about 15 minutes with four workers,
each needing 1-2 GB of memory. combine only pools the stored analyses.

compare sets each re-analysed job, and combined.json, against runs/ in four tiers:

  A  decoded from the chip's shots, and the record (patches, rep, frozen prediction):
     equal.
  B  fitted to the chip's shots (the idle test's f, the fits, the drift halves, the
     round covariance): equal to 1e-9 relative, or 1e-15 absolute near zero.
  C  sampled by the simulator (every predicted_* block): each predicted logical error,
     detector rate and <X> within 5.5 standard errors of the stored one, and their mean
     squared deviation, in standard errors, at most 1.3 (it is about 1 for two
     independent samples); what is derived from the samples (agreement, leakage, f read
     off the memory runs, the detectors beyond chance) is not compared. 5.5 standard
     errors keeps a false alarm over the fifteen jobs' 53,760 values below 0.3%.
  D  the decisions in combined.json: each agreement test's chi2 within 25% relative,
     about five times what the simulator's noise in the predictions moves it (3-5%
     between two 200,000-shot samples), and each verdict, and whether the model matches
     the device in each basis, equal unless the stored chi2 lies within that 25% of the
     bound. Such a borderline test (X eps5, at 19.8 against 18.31) is printed as a note,
     with whether its verdict came back. The per-repetition Lambda verdicts are tier A.

A job that was asked for but not re-analysed counts as a failure. combined.json is
compared when no --job is named.

--strict asks for every value to be equal and each file identical byte for byte, line
endings and code.analyzed (the commit doing the re-analysis) aside. That holds on the
environment that analysed the runs, Linux x86-64 with requirements-lock.txt. On another
platform the C library's exp can differ in the last bit, which moves a few of the noise
model's probabilities by an ulp and reshuffles stim's seeded sampling, so tier C agrees
only within the simulator's noise.

Exit status: 0 when everything compared passes, 1 when something fails, 2 on a usage
error.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
ANALYSIS, COMBINED = "analysis.json", "combined.json"
DEFAULT_OUT = REPO / "build" / "reproduce"
RELATIVE = 1e-9  # tier B
ABSOLUTE = 1e-15  # tier B, for values at zero up to float noise
STANDARD_ERRORS = 5.5  # tier C, each value
MEAN_SQUARE = 1.3  # tier C, over a file's values (about 1 for independent samples)
ULPS = 4  # combine over the stored analyses: scipy's t and chi2 quantiles move by an ulp
DECISION_RELATIVE = 0.25  # tier D: each agreement test's chi2, and the band around its bound

JsonPath = tuple[Any, ...]  # keys and indices from the top of a JSON document


@functools.cache
def heavyhex_run() -> Any:
    """heavyhex.experiment.run, from this checkout's src/ whatever else is installed."""
    src = REPO / "src"
    if not sys.path or sys.path[0] != str(src):
        sys.path.insert(0, str(src))
    from heavyhex.experiment import run

    if src.resolve() not in Path(run.__file__).resolve().parents:
        raise RuntimeError(f"heavyhex was imported from {run.__file__}, not from {src}")
    return run


def jobs(runs: Path = RUNS) -> list[str]:
    """The jobs that ran on the chip: the folders of runs/ with the chip's shots."""
    return sorted(path.parent.name for path in runs.glob("*/shots.npz"))


def scratch(out: Path) -> Path:
    """OUT resolved, or a ValueError if writing OUT/runs would touch the stored runs/."""
    out, runs = Path(out).resolve(), RUNS.resolve()
    if out == runs.parent or out == runs or runs in out.parents:
        raise ValueError(f"{out} would write into {runs}; choose a scratch folder")
    return out


# --- re-analysis --------------------------------------------------------------------------


def analyze_job(name: str, out: Path, runs: Path = RUNS) -> Path:
    """Copy one job to OUT/runs/<name> without its analysis, and analyse the copy there."""
    run = heavyhex_run()
    source, copy = runs / name, scratch(out) / "runs" / name
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name != ANALYSIS and not path.name.startswith("rehearsal"):
            shutil.copy2(path, copy / path.name)
    # analyze records the commit doing the work; git status then only reads the repository.
    os.environ.setdefault("GIT_OPTIONAL_LOCKS", "0")
    saved, run.RUNS = run.RUNS, copy.parent
    try:
        run.analyze(copy)
    finally:
        run.RUNS = saved
    return copy / ANALYSIS


def combine_in(out: Path) -> Path:
    """Pool the analyses in OUT/runs into OUT/runs/combined.json."""
    run = heavyhex_run()
    runs = scratch(out) / "runs"
    saved, run.RUNS = run.RUNS, runs
    try:
        run.combine()
    finally:
        run.RUNS = saved
    return runs / COMBINED


def copy_stored(out: Path, runs: Path = RUNS) -> None:
    """Copy every job's run.json and stored analysis.json to OUT/runs, for combine_in."""
    out = scratch(out)
    for name in jobs(runs):
        copy = out / "runs" / name
        copy.mkdir(parents=True, exist_ok=True)
        for file in ("run.json", ANALYSIS):
            shutil.copy2(runs / name / file, copy / file)


def _selected(names: list[str] | None) -> list[str]:
    """The jobs named on the command line, once each, or every job; a ValueError for others."""
    known = jobs()
    chosen = list(dict.fromkeys(names or known))
    unknown = [name for name in chosen if name not in known]
    if unknown:
        raise ValueError(f"no such job in {RUNS}: {', '.join(unknown)}")
    return chosen


def _analyze_command(args: argparse.Namespace, out: Path, names: list[str]) -> int:
    failed = []
    if args.jobs == 1:
        for name in names:
            start = time.time()
            analyze_job(name, out)
            print(f"{name}: {time.time() - start:.0f} s", flush=True)
    else:

        def one(name: str) -> tuple[str, subprocess.CompletedProcess, float]:
            start = time.time()
            command = [sys.executable, __file__, "analyze", "--out", str(out), "--jobs", "1"]
            done = subprocess.run(
                [*command, "--job", name, "--no-combine"], capture_output=True, text=True
            )
            return name, done, time.time() - start

        with ThreadPoolExecutor(args.jobs) as pool:
            for name, done, seconds in pool.map(one, names):
                print(f"{name}: {seconds:.0f} s{'' if done.returncode == 0 else ', FAILED'}")
                if done.returncode:
                    failed.append(name)
                    print(done.stdout + done.stderr, file=sys.stderr)
    if failed:
        return 1
    if not args.no_combine:
        missing = [name for name in jobs() if not (out / "runs" / name / ANALYSIS).exists()]
        if missing:
            print(f"combine waits for {len(missing)} more jobs: {', '.join(missing)}")
        else:
            print(f"pooled into {combine_in(out)}")
    return 0


def _combine_command(args: argparse.Namespace, out: Path, names: list[str]) -> int:
    copy_stored(out)
    print(f"pooled the stored analyses into {combine_in(out)}")
    return 0


# --- comparison ---------------------------------------------------------------------------

# How a value is compared: ("exact",), ("ulp",), ("close",), ("sampled", shots),
# ("derived",) for the keys and lengths only, or ("skip",).
Rule = tuple[Any, ...]
EXACT: Rule = ("exact",)
ULP: Rule = ("ulp",)
CLOSE: Rule = ("close",)
DERIVED: Rule = ("derived",)
SKIP: Rule = ("skip",)


@dataclass
class Report:
    """How one file compares with its stored copy."""

    identical: bool = False  # LF-normalised bytes, code.analyzed aside
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # tier D: verdicts too close to call
    worst_relative: float = 0.0  # tier B
    worst_standard_errors: float = 0.0  # tier C
    squares: list[float] = field(default_factory=list)  # tier C, each value's z**2

    @property
    def mean_square(self) -> float:
        return sum(self.squares) / len(self.squares) if self.squares else 0.0

    @property
    def passed(self) -> bool:
        return not self.failures and self.mean_square <= MEAN_SQUARE


def strict_rule(path: JsonPath) -> Rule:
    """Every value equal, except who re-analysed the job."""
    return SKIP if path[:2] == ("code", "analyzed") else EXACT


def analysis_rule(path: JsonPath) -> Rule:
    """The tier of a value of a job's analysis.json."""
    run = heavyhex_run()
    top = path[0] if path else None
    if top == "code":
        return strict_rule(path)
    if top in ("decoupled_dephasing", "idle_test", "dephasing_by_qubit", "drift"):
        return CLOSE
    if top == "observed":
        if path[1:2] == ("fits",) or path[3:4] == ("round_covariance",):
            return CLOSE
        return EXACT  # the decoded shots: logical errors, detector rates, <X>
    shots = {
        "predicted_at_measured_dephasing": run.SIMULATED_SHOTS,
        "predicted_from_calibration_after": run.SIMULATED_SHOTS,
        "predicted_with_each_qubits_dephasing": run.SIDE_SHOTS,
    }
    if top in shots:
        if path[1:2] == ("settings",) and path[3:4] in (
            ("logical_error",),
            ("detector_rates",),
            ("x",),
        ):
            return ("sampled", shots[top])
        if path[1:2] == ("settings",) and path[3:4] == ("detectors",):
            return EXACT
        return DERIVED  # fits, rates by round, covariance: from the sampled values
    if top in ("agreement", "leakage", "dephasing_from_memory", "detectors_beyond_chance"):
        return DERIVED
    if top == "worst_detectors":
        return SKIP  # how many there are depends on the samples
    return EXACT  # shots_from, patches, rep, frozen_prediction, chance_excess, and anything new


def combined_rule(path: JsonPath) -> Rule:
    """The tier of a value of combined.json, pooled from re-analysed jobs."""
    top = path[0] if path else None
    if top == "reps":
        key = path[2] if len(path) > 2 else None
        if key == "decoupled_dephasing":
            return CLOSE
        if key == "tests":
            return CLOSE if path[4:5] in (("observed",), ("observed_uncertainty",)) else DERIVED
        if key in ("leakage", "detectors_beyond_chance"):
            return DERIVED
        return EXACT  # day, rep, pairing, folders, spots, drift, verdict
    if top == "pooled_x":
        return DERIVED if path[1:2] == ("predicted",) else CLOSE
    if top in ("model", "matches"):
        return DERIVED
    return EXACT  # shots_from, days, incomplete


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def _leaf(got: Any, want: Any, path: JsonPath, rule: Rule, report: Report) -> None:
    kind = rule[0]
    if kind == "derived":
        return
    where = "/".join(map(str, path))
    numbers = _number(got) and _number(want) and math.isfinite(got) and math.isfinite(want)
    if kind == "exact" or not numbers or got == want or type(got) is not type(want):
        if not _same(got, want):
            report.failures.append(f"{where}: {got!r} != {want!r}")
        elif kind == "sampled" and numbers:
            report.squares.append(0.0)  # an equal value counts in the mean square too
        return
    if kind == "ulp":
        if abs(got - want) > ULPS * math.ulp(max(abs(got), abs(want))):
            report.failures.append(f"{where}: {got!r} != {want!r} (over {ULPS} ulp)")
    elif kind == "close":
        off = abs(got - want) / max(abs(got), abs(want))
        report.worst_relative = max(report.worst_relative, off)
        if off > RELATIVE and abs(got - want) > ABSOLUTE:
            report.failures.append(f"{where}: {got!r} != {want!r} ({off:.1e} relative)")
    elif kind == "sampled":
        shots = rule[1]
        if path[3] == "x":  # <X> = 1 - 2p: compare p, which leaves z as it is
            got, want = (1 - got) / 2, (1 - want) / 2
        if not (0 <= got <= 1 and 0 <= want <= 1):
            report.failures.append(f"{where}: {got!r} or {want!r} is not a probability")
            return
        middle = (got + want) / 2
        # Binomial, with at least one count's worth of variance.
        sigma = math.sqrt(2 * max(middle * (1 - middle), 1 / shots) / shots)
        z = abs(got - want) / sigma
        report.worst_standard_errors = max(report.worst_standard_errors, z)
        report.squares.append(z * z)
        if z > STANDARD_ERRORS:
            report.failures.append(f"{where}: {got!r} vs {want!r} ({z:.1f} standard errors)")


def walk(got: Any, want: Any, rule_of: Callable[[JsonPath], Rule], report: Report) -> None:
    """Compare two parsed JSON documents value by value, each by its rule."""

    def visit(a: Any, b: Any, path: JsonPath) -> None:
        rule = rule_of(path)
        if rule == SKIP:
            return
        where = "/".join(map(str, path)) or "(top)"
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                report.failures.append(f"{where}: keys {sorted(a.keys() ^ b.keys())} differ")
            for key in a.keys() & b.keys():
                visit(a[key], b[key], (*path, key))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                report.failures.append(f"{where}: {len(a)} items, stored {len(b)}")
            for k, (x, y) in enumerate(zip(a, b, strict=False)):
                visit(x, y, (*path, k))
        elif isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
            report.failures.append(f"{where}: {type(a).__name__} != {type(b).__name__}")
        else:
            _leaf(a, b, path, rule, report)

    visit(got, want, ())


def _lf(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _written(document: Any) -> bytes:
    """The bytes heavyhex writes for a JSON document (run._write), with LF line endings."""
    return (json.dumps(document, indent=1) + "\n").encode()


def compare_analysis(got: Path, want: Path, *, strict: bool = False) -> Report:
    """A re-analysed job's analysis.json against the stored one."""
    raw, stored = _lf(got), _lf(want)
    new, old = json.loads(raw), json.loads(stored)
    report = Report()
    as_written = _written(new) == raw  # nothing but what analyze writes
    if isinstance(new.get("code"), dict) and isinstance(old.get("code"), dict):
        new["code"]["analyzed"] = old["code"].get("analyzed")  # who re-analysed it
    report.identical = as_written and _written(new) == stored
    walk(new, old, strict_rule if strict else analysis_rule, report)
    if strict and not report.identical and report.passed:
        report.failures.append("equal values, but the bytes differ")
    return report


def compare_combined(
    got: Path, want: Path, *, strict: bool = False, rule: Callable[[JsonPath], Rule] | None = None
) -> Report:
    """A combined.json against the stored one: by tier, or by `rule` if one is given."""
    report = Report(identical=_lf(got) == _lf(want))
    rule_of = strict_rule if strict else (rule or combined_rule)
    new, old = json.loads(got.read_text()), json.loads(want.read_text())
    walk(new, old, rule_of, report)
    if not strict and rule is None:
        compare_decisions(new, old, report)
    if strict and not report.identical and report.passed:
        report.failures.append("equal values, but the bytes differ")
    return report


def compare_decisions(new: dict, old: dict, report: Report) -> None:
    """Tier D: the model-agreement tests and verdicts of a combined.json.

    A verdict whose stored chi2 lies within DECISION_RELATIVE of its bound can flip with the
    simulator's noise, so a change there is not a failure; every such test is still listed in
    report.notes, with whether its verdict came back, and so is the basis verdict it feeds.
    """
    near_bound = {"X": False, "Z": False}
    for name, want in old.get("model", {}).items():
        got = new.get("model", {}).get(name)
        if got is None:
            continue  # the walk reports the missing key
        off = abs(got["chi2"] - want["chi2"]) / max(abs(got["chi2"]), abs(want["chi2"]))
        near = abs(want["chi2"] - want["bar"]) <= DECISION_RELATIVE * want["bar"]
        near_bound[name.split()[0]] |= near
        if off > DECISION_RELATIVE:
            report.failures.append(f"model/{name}/chi2: {got['chi2']!r} != {want['chi2']!r}")
        if got["agrees"] != want["agrees"] and not near:
            report.failures.append(f"model/{name}/agrees: {got['agrees']} != {want['agrees']}")
        if near:
            outcome = (
                "verdict reproduced"
                if got["agrees"] == want["agrees"]
                else f"verdict changed: agrees {got['agrees']}, stored {want['agrees']}"
            )
            report.notes.append(
                f"model/{name}: chi2 {got['chi2']:.2f} (stored {want['chi2']:.2f}) against the "
                f"bound {want['bar']:.2f}, borderline; {outcome}"
            )
    for basis, want in old.get("matches", {}).items():
        got = new.get("matches", {}).get(basis)
        if got is None:
            continue
        if got != want and not near_bound.get(basis, False):
            report.failures.append(f"matches/{basis}: {got} != {want}")
        elif got != want:
            report.notes.append(f"matches/{basis}: {got}, stored {want}, through a borderline test")


def _compare_command(args: argparse.Namespace, out: Path, names: list[str]) -> int:
    if not (out / "runs").is_dir():
        print(f"nothing to compare in {out}: run analyze first", file=sys.stderr)
        return 2
    rows = [
        (name, out / "runs" / name / ANALYSIS, RUNS / name / ANALYSIS, compare_analysis)
        for name in names
    ]
    if not args.job:
        rows.append((COMBINED, out / "runs" / COMBINED, RUNS / COMBINED, compare_combined))
    print(f"{'':28} {'bytes':9} {'tiers':6} {'B, rel':>8} {'C, s.e.':>8} {'C, z^2':>7}")
    failed = compared = 0
    for name, got, want, compare in rows:
        if not got.exists():
            print(f"{name:28} not re-analysed")
            failed += 1
            continue
        compared += 1
        report = compare(got, want, strict=args.strict)
        failed += not report.passed
        print(
            f"{name:28} {'identical' if report.identical else 'differ':9} "
            f"{'pass' if report.passed else 'FAIL':6} "
            f"{report.worst_relative:8.1e} {report.worst_standard_errors:8.2f} "
            f"{report.mean_square:7.2f}"
        )
        if report.mean_square > MEAN_SQUARE:
            print(f"    predictions off by {report.mean_square:.2f} squared standard errors")
        for line in report.failures[:10]:
            print(f"    {line}")
        if len(report.failures) > 10:
            print(f"    ... and {len(report.failures) - 10} more")
        for line in report.notes:
            print(f"    note: {line}")
    print(f"{compared} files compared, {failed} failed{' (strict)' if args.strict else ''}")
    return 1 if failed else 0


def _workers(text: str) -> int:
    count = int(text)
    if count < 1:
        raise argparse.ArgumentTypeError("at least 1")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reproduce.py", description=__doc__.split("\n\n")[0] if __doc__ else None
    )
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="re-analyse jobs in OUT/runs, then combine")
    analyze.add_argument("--out", default=str(DEFAULT_OUT), help="the scratch folder")
    analyze.add_argument("--job", action="append", help="a folder name in runs/ (repeatable)")
    analyze.add_argument(
        "--jobs", type=_workers, default=min(4, os.cpu_count() or 1), help="workers"
    )
    analyze.add_argument("--no-combine", action="store_true", help=argparse.SUPPRESS)
    compare = sub.add_parser("compare", help="compare OUT/runs with runs/ by tier")
    compare.add_argument("--out", default=str(DEFAULT_OUT), help="the scratch folder")
    compare.add_argument("--job", action="append", help="compare only this job (repeatable)")
    compare.add_argument("--strict", action="store_true", help="every value and byte equal")
    combine = sub.add_parser("combine", help="pool the stored analyses into OUT/runs")
    combine.add_argument("--out", default=str(REPO / "build" / "recombine"))
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code or 0)
    try:
        out, names = scratch(Path(args.out)), _selected(getattr(args, "job", None))
    except ValueError as error:
        print(f"reproduce.py: {error}", file=sys.stderr)
        return 2
    commands = {
        "analyze": _analyze_command,
        "compare": _compare_command,
        "combine": _combine_command,
    }
    return commands[args.command](args, out, names)


if __name__ == "__main__":
    sys.exit(main())
