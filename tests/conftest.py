"""Shared fixtures, and the options for the slow re-analysis of the fifteen hardware jobs.

    pytest                              everything but the re-analysis
    pytest --reproduce                  also re-analyse every job (about 15 min, 4 workers)
    pytest --reproduce --strict-floats  ... and ask for byte-identical files, as on the
                                        environment that analysed the runs (requirements-lock.txt)

HEAVYHEX_REPRODUCE=1 and HEAVYHEX_STRICT=1 do the same as the two options, and
HEAVYHEX_WORKERS sets how many jobs are re-analysed at once (default 4).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("heavyhex")
    group.addoption(
        "--reproduce",
        action="store_true",
        default=os.environ.get("HEAVYHEX_REPRODUCE") == "1",
        help="re-analyse the fifteen hardware jobs and compare them with runs/",
    )
    group.addoption(
        "--strict-floats",
        action="store_true",
        default=os.environ.get("HEAVYHEX_STRICT") == "1",
        help="compare every value exactly and every file byte for byte",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "reproduce: re-analyses the hardware jobs (--reproduce)")
    config.addinivalue_line("markers", "git: needs a git checkout with its tags")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--reproduce"):
        return
    skip = pytest.mark.skip(
        reason="re-analysing the 15 jobs takes about 15 min with 4 workers; run with --reproduce"
    )
    for item in items:
        if "reproduce" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def strict(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--strict-floats"))


@pytest.fixture(scope="session")
def run_folders(repo_root: Path) -> list[Path]:
    """The jobs that ran on the chip, in runs/; skips where runs/ isn't shipped."""
    found = sorted(path.parent for path in (repo_root / "runs").glob("ibm_fez-*/shots.npz"))
    if not found:
        pytest.skip("no runs/ folder with the chip's shots")
    return found
