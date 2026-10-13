"""Where the log-linear fit's bias comes from (App. stats).

fits.py's parametric bootstrap shows that the planned weighted-log fit underestimates the
device's eps at distance 5. Here the same bootstrap (binomial draws of N = 2500 shots from
each curve's binomial fit, 2,000 replicates) is refitted twice: with the planned weights,
the inverse of Var ln(1 - 2P) at the observed P, and with the same weights at the true P.
Both drop the round counts with observed P >= 1/2, so the difference is the weights alone.
Bias is the mean over replicates minus the true eps, averaged over the ten curves of each
basis and distance. Seconds.

    python docs/analysis/fit_bias.py      # writes results/fit_bias.json
"""

from __future__ import annotations

import numpy as np

from heavyhex.experiment.circuits import ROUNDS

from common import result, write_json
from fitting import wls_batch

REPLICATES = 2000
SEED = 7


def wls_true_weights(rounds, p_hat: np.ndarray, p_true: np.ndarray, shots: int) -> np.ndarray:
    """The weighted-log fit of each row of p_hat, with weights at p_true; eps per row."""
    n = np.broadcast_to(np.asarray(rounds, float), p_hat.shape)
    keep = (p_hat > 0) & (p_hat < 0.5)
    y = np.log(1 - 2 * np.where(keep, p_hat, 0.25))
    sigma = 2 * np.sqrt(p_true * (1 - p_true) / shots) / (1 - 2 * p_true)
    w = np.where(keep, np.broadcast_to(1 / sigma**2, p_hat.shape), 0.0)
    S, Sx, Sy = w.sum(1), (w * n).sum(1), (w * y).sum(1)
    Sxx, Sxy = (w * n * n).sum(1), (w * n * y).sum(1)
    ok = keep.sum(1) >= 2
    det = np.where(ok, S * Sxx - Sx**2, 1.0)
    b = np.where(ok, (S * Sxy - Sx * Sy) / det, -np.inf)
    return np.where(ok, (1 - np.exp(b)) / 2, 0.5)


def main() -> None:
    fits = result("fits.json")
    rounds = np.array(ROUNDS)
    rng = np.random.default_rng(SEED)
    by_kind: dict[str, dict[str, list[float]]] = {}
    for key, curve in fits["curves"].items():
        basis, distance = key.split()[-2], key.split()[-1]
        N = curve["N"]
        A, eps = curve["binomial"]["A"], curve["binomial"]["eps"]
        p_true = (1 - A * (1 - 2 * eps) ** rounds) / 2
        p_hat = rng.binomial(N, p_true, size=(REPLICATES, len(rounds))) / N
        planned, _ = wls_batch(rounds, p_hat, N)
        true_weights = wls_true_weights(rounds, p_hat, p_true, N)
        kind = by_kind.setdefault(f"{basis} {distance}", {"planned": [], "true_weights": []})
        kind["planned"].append(float(planned.mean() - eps))
        kind["true_weights"].append(float(true_weights.mean() - eps))
    out = {
        "replicates": REPLICATES,
        "seed": SEED,
        "bias_pp": {
            kind: {
                way: {"mean": 100 * float(np.mean(v)), "range": [100 * min(v), 100 * max(v)]}
                for way, v in ways.items()
            }
            for kind, ways in sorted(by_kind.items())
        },
    }
    write_json("fit_bias.json", out)
    for kind, ways in out["bias_pp"].items():
        print(
            kind,
            f"planned {ways['planned']['mean']:+.2f} pp,",
            f"weights at the true P {ways['true_weights']['mean']:+.2f} pp",
        )


if __name__ == "__main__":
    main()
