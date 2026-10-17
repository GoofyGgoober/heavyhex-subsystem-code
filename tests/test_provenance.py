"""The record in runs/: every job names the code that prepared and analysed it, and that
code is still the code in this tree.

The fifteen jobs ran from tag v0.2.0. Each run.json, job.json and analysis.json stores the
git hash of src/ and of pyproject.toml (heavyhex.experiment.run.commit), and
`heavyhex experiment analyze` says so when the code differs from a job's, so src/ and
pyproject.toml stay byte-identical to v0.2.0. The jobs retain their recorded commit IDs;
docs/commit-map.tsv resolves those IDs to this repository's commits.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

TAG = "v0.2.0"
FROZEN = ("src", "pyproject.toml")  # heavyhex.experiment.run.CODE
COMMIT_MAP = Path("docs") / "commit-map.tsv"
FILES = (
    "run.json",
    "calibration.json",
    "calibration_after.json",
    "circuits.qpy.gz",
    "prediction.json",
    "job.json",
    "shots.npz",
    "analysis.json",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def _git(repo: Path, *words: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *words],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},  # only reads the repository
    )


def git(repo: Path, *words: str) -> str:
    """git's answer; a failure fails the test."""
    done = _git(repo, *words)
    assert done.returncode == 0, f"git {' '.join(words)}: {done.stderr.strip()}"
    return done.stdout.strip()


def rewritten(repo: Path) -> dict[str, str]:
    """Each commit runs/ records -> the same commit after the rewrite."""
    lines = (repo / COMMIT_MAP).read_text().splitlines()
    rows = (line.split("\t") for line in lines if line and not line.startswith(("#", "old\t")))
    return dict(rows)


@pytest.fixture(scope="module")
def history(repo_root: Path) -> Path:
    """The checkout, with git, the full history and the tag; a skip where they're missing
    (an archive, a shallow clone)."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    if _git(repo_root, "rev-parse", "--is-inside-work-tree").returncode:
        pytest.skip("not a git checkout")
    if _git(repo_root, "rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        pytest.skip("a shallow clone lacks the commits the jobs record")
    if _git(repo_root, "rev-parse", "-q", "--verify", f"refs/tags/{TAG}").returncode:
        pytest.skip(f"no tag {TAG}")
    return repo_root


@pytest.fixture(scope="module")
def recorded(run_folders: list[Path]) -> list[str]:
    """The [src tree, pyproject.toml blob] pair every job records."""
    return read(run_folders[0] / "run.json")["code"]["code"]


def test_every_job_folder_is_complete(run_folders):
    assert len(run_folders) == 15
    for folder in run_folders:
        # A local rehearsal's files are simulated and kept out of git (.gitignore).
        found = sorted(p.name for p in folder.iterdir() if not p.name.startswith("rehearsal"))
        assert found == sorted(FILES), folder.name


def test_every_job_records_the_same_unchanged_code(run_folders, recorded):
    for folder in run_folders:
        prepared = read(folder / "run.json")["code"]
        sent = read(folder / "job.json")["code"]
        analysis = read(folder / "analysis.json")["code"]
        for code in (prepared, sent, analysis["analyzed"]):
            assert code["code"] == recorded and code["changed"] is False, folder.name
        assert analysis["prepared"] == prepared, folder.name


def test_the_analysis_compares_with_the_prediction_frozen_before_the_run(run_folders):
    for folder in run_folders:
        frozen = read(folder / "prediction.json")
        stored = read(folder / "analysis.json")["frozen_prediction"]
        assert stored == {f: prediction["fits"] for f, prediction in frozen.items()}, folder.name


@pytest.mark.git
def test_the_recorded_code_is_tag_v020(history, recorded):
    assert git(history, "rev-parse", *(f"{TAG}:{path}" for path in FROZEN)).split() == recorded


@pytest.mark.git
def test_each_recorded_commit_holds_the_recorded_code(history, run_folders, recorded):
    commits = set()
    for folder in run_folders:
        analysis = read(folder / "analysis.json")["code"]
        for code in (analysis["prepared"], read(folder / "job.json")["code"], analysis["analyzed"]):
            commits.add(code["commit"])
    now = rewritten(history)
    assert commits <= now.keys(), sorted(commits - now.keys())
    for commit in sorted(commits):
        found = git(history, "rev-parse", *(f"{now[commit]}:{path}" for path in FROZEN)).split()
        assert found == recorded, commit
        assert not _git(history, "merge-base", "--is-ancestor", now[commit], "HEAD").returncode, (
            commit
        )


@pytest.mark.git
def test_the_frozen_code_is_unchanged_in_this_tree(history, recorded):
    """src/ and pyproject.toml as at v0.2.0, committed and in the working tree."""
    assert git(history, "rev-parse", *(f"HEAD:{path}" for path in FROZEN)).split() == recorded
    # Line endings and the executable bit, as another OS sees a shared checkout, don't count.
    neutral = ("-c", "core.autocrlf=input", "-c", "core.filemode=false")
    assert git(history, *neutral, "status", "--porcelain", "--", *FROZEN) == ""


@pytest.mark.filterwarnings("ignore:.*weak parts")
def test_each_job_ran_where_its_days_plan_put_it(run_folders):
    """The day's plan, redone from the job's own calibration, gives its spots and qubits.

    The circuits sent are taken as data: rebuilding circuits.qpy.gz needs the live
    target of the day, which FakeFez does not reproduce.
    """
    from heavyhex.patches.placement import Calibration, fez_plan, fez_qubits

    for folder in run_folders:
        run = read(folder / "run.json")
        calibration = Calibration.load(folder / "calibration.json")
        rep = run["rep"]
        job = fez_plan(calibration)[rep["rep"] - 1][rep["job"] - 1]
        assert {str(d): [list(s.origin), list(s.direction)] for d, s in job.items()} == rep[
            "spots"
        ], folder.name
        for d, spot in job.items():
            assert fez_qubits(d, calibration, spot) == run["chips"][str(d)], folder.name
