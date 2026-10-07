"""How well the run's rules decide, simulated at the run's shot counts.

Each rep's chip result is drawn from its frozen prediction: every memory circuit's
logical error P(n) comes back with the binomial noise of SHOTS_PER_CIRCUIT shots
(times `scatter`, for noise beyond shot noise), and the prediction with that of
SIMULATED_SHOTS. Both are fitted as the run fits them, and the reps are pooled and
tested as run.combine does. The measured f is taken as known exactly, and IBM as
not recalibrating during a job, so the bounds are as narrow as the run allows.

Reports, for 5 and 10 reps:
- how often pooled Λ in X memory is called either way when it is exactly 1;
- how often it calls d = 5 worse at the predicted Λ;
- how often the X-memory model test fails when the model is exactly right;
- the gap in Λ the test catches 80% of the time.

    python docs/power.py runs/<rep 1 folder> runs/<rep 2 folder> [runs/<rep 2's other job>] ...

Folders of a split rep's two jobs go next to each other; a rep's patches are
found from the folders' predictions.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from scipy.stats import t as student_t

from heavyhex.experiment import run
from heavyhex.experiment.circuits import ROUNDS, SETTINGS
from heavyhex.simulation.noisy import fit_per_round

N = np.array(ROUNDS, dtype=float)


def curves(folders: list[Path], end: str) -> list[dict[tuple[str, str], tuple[float, float]]]:
    """Each rep's (A, ε) for every (basis, patch), fitted to the frozen prediction at f = end."""
    reps: dict[tuple[str, int], dict] = {}
    for folder in folders:
        info = json.loads((folder / "run.json").read_text())
        prediction = json.loads((folder / "prediction.json").read_text())[end]
        rows = prediction["settings"]
        patches = run.patches_of(info)
        found = reps.setdefault((info["rep"]["day"], info["rep"]["rep"]), {})
        for basis in "XZ":
            mine = [r for s, r in zip(SETTINGS, rows) if s.kind == "memory" and s.basis == basis]
            for k, d in enumerate(patches):
                p = np.array([r["logical_error"][k] for r in mine])
                slope, offset = np.polyfit(N, np.log(1 - 2 * p), 1)
                found[basis, d] = (float(np.exp(offset)), float((1 - np.exp(slope)) / 2))
    return list(reps.values())


def logical_error(a: float, eps: float) -> np.ndarray:
    return (1 - a * (1 - 2 * eps) ** N) / 2


def drawn(p: np.ndarray, shots: int, rng: np.random.Generator, scatter: float = 1.0) -> float:
    """ε fitted to P(n) as `shots` shots return it."""
    seen = rng.binomial(shots, p) / shots
    seen = np.clip(p + scatter * (seen - p), 0.0, 0.5)
    return fit_per_round(ROUNDS, seen.tolist(), shots)


def rep_test(truth: dict, model: dict, rng: np.random.Generator, scatter: float) -> dict:
    """One rep's observed and predicted ε₃, ε₅ and Λ in X memory, each with its σ."""
    found = {}
    for d in ("3", "5"):
        seen = drawn(logical_error(*truth["X", d]), run.SHOTS_PER_CIRCUIT, rng, scatter)
        want = drawn(logical_error(*model["X", d]), run.SIMULATED_SHOTS, rng)
        found[f"eps{d}"] = (seen.per_round, seen.uncertainty, want.per_round, want.uncertainty)
    (o3, s3, p3, n3), (o5, s5, p5, n5) = found["eps3"], found["eps5"]
    lam, pred = o3 / o5, p3 / p5
    found["lambda"] = (
        lam,
        lam * sqrt((s3 / o3) ** 2 + (s5 / o5) ** 2),
        pred,
        pred * sqrt((n3 / p3) ** 2 + (n5 / p5) ** 2),
    )
    return found


def campaign(truths: list, models: list, reps: int, rng, scatter: float = 1.0) -> dict:
    """Pooled Λ and its call, and whether each X test of run.combine passes."""
    tests = [
        rep_test(truths[k % len(truths)], models[k % len(models)], rng, scatter)
        for k in range(reps)
    ]
    lam = np.array([t["lambda"][0] for t in tests])
    weights = 1 / np.array([t["lambda"][1] for t in tests]) ** 2
    pooled = float((weights * lam).sum() / weights.sum())
    spread = float((weights * (lam - pooled) ** 2).sum() / (reps - 1))
    sigma = sqrt(max(spread, 1.0) / weights.sum())  # as run.combine widens it
    critical = float(student_t.ppf(0.95, reps - 1))
    passes = {}
    for name in ("lambda", "eps3", "eps5"):
        z = [(o - p) / sqrt(s**2 + n**2) for o, s, p, n in (t[name] for t in tests)]
        passes[name] = sum(v * v for v in z) < chi2.ppf(0.95, reps)
    return {"call": run.verdict(pooled, sigma, critical), "sigma": sigma, "passes": passes}


def scaled(truths: list, factor: float) -> list:
    """The truths with ε₃ scaled, so Λ moves by the same factor."""
    return [{**t, ("X", "3"): (t["X", "3"][0], t["X", "3"][1] * factor)} for t in truths]


def at_one(truths: list) -> list:
    """The truths with ε₃ = ε₅, so Λ = 1."""
    return [{**t, ("X", "3"): (t["X", "3"][0], t["X", "5"][1])} for t in truths]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    for end in map(str, run.ENDS):
        models = curves(args.folders, end)
        lam = [m["X", "3"][1] / m["X", "5"][1] for m in models]
        print(f"f = {end}: predicted Λ per rep {', '.join(f'{v:.2f}' for v in lam)}")
        for reps in (5, 10):

            def rate(truths, check, scatter=1.0):
                hits = sum(
                    check(campaign(truths, models, reps, rng, scatter)) for _ in range(args.trials)
                )
                return hits / args.trials

            one = rep_test(models[0], models[0], rng, 1.0)["lambda"][1]
            sigma = campaign(models, models, reps, rng)["sigma"]
            print(f"  {reps} reps: σ(Λ) {one:.3f} for one rep, {sigma:.3f} pooled")
            for scatter in (1.0, 2.0, 4.0):
                called = rate(at_one(models), lambda c: c["call"] != "undecided", scatter)
                print(f"    Λ = 1, scatter {scatter:.0f}x shot noise: called {called:.1%}")
            worse = rate(models, lambda c: c["call"] == "d = 5 worse")
            print(f"    at the predicted Λ: d = 5 called worse {worse:.1%}")
            fails = rate(models, lambda c: not all(c["passes"].values()))
            print(f"    model exactly right: the X tests fail {fails:.1%}")
            for gap in np.arange(0.02, 0.31, 0.02):
                factor = 1 + gap / float(np.mean(lam))
                caught = rate(scaled(models, factor), lambda c: not c["passes"]["lambda"])
                if caught >= 0.8:
                    print(f"    a gap in Λ of {gap:.2f} is caught {caught:.0%} of the time")
                    break


if __name__ == "__main__":
    main()
