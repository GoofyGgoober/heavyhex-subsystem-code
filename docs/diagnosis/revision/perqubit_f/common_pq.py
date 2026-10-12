"""Shared loading for the per-qubit-f revision check. Exploratory, post hoc; reads the
committed run folders only, changes nothing in src/ or runs/, uses no QPU.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNS = REPO / "runs"


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def job_folders() -> list[Path]:
    """The fifteen hardware job folders (offline/rehearsal folders skipped, as combine() does)."""
    found = []
    for folder in sorted(RUNS.glob("ibm_fez-2026-10-*-r*")):
        if not (folder / "analysis.json").exists():
            continue
        info = read(folder / "run.json")
        if not info.get("rep") or info.get("offline"):
            continue
        found.append(folder)
    return found


def day_number(day: str) -> int:
    """Folder dates 2026-10-04..08 are the paper's days 1-5."""
    return int(day[-2:]) - 3
