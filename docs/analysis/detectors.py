"""Detector classes, edge types of the decoder's graph, and the p_ij estimator.

Shared by detection_events.py, pij.py, pij_decode.py and corr_matching.py. Nothing here reads
runs/ or writes results.

Detector classes: an X stabilizer of weight 4 (bulk) or 2 (boundary), a Z stabilizer (weight
2d), a flag, a relay. Labels are (patch, round, name) as the analysis stores them; round r of
an n-round circuit is t = r + 1 in the paper, and r = n is the final layer from the data
readout.

An edge of the matching graph joins two detectors, or one detector and the boundary. Its type
is round-independent: (patch, earlier name, later name, round difference, position), where the
position says whether the edge touches round 0 ("first"), the final layer ("final"), both, or
neither ("bulk"). Every round runs the same gates, so edges of one type have nearly the same
probability in the model. Classes: space-like (same round, two stabilizers), time-like (same
stabilizer, consecutive rounds: its mid-circuit measurement and reset), diagonal (two
stabilizers, consecutive rounds), boundary (one detector), and flag/relay (any edge that
touches a flag or relay detector, boundary edges included).

The p_ij estimator: for detection events x_i, x_j,

    p_ij = 1/2 - 1/2 sqrt(1 - 4 (<x_i x_j> - <x_i><x_j>) / (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>)),

the probability of the independent mechanisms that flip both, exactly, if every mechanism is
independent; and a boundary edge from the remainder of p_i,

    1 - 2 p_i,boundary = (1 - 2<x_i>) / prod_j (1 - 2 p_ij)   over the graph neighbours j of i.

The same formulas on the model's exact moments give the model as the estimator sees it
(model_pairs): pieces of mechanisms that fire three or more detectors land on every pair they
contain.

The distance-5 Z-memory stabilizers fire in about 40% (model) to 48% (device) of rounds. There
the denominator above is close to zero and 2500 shots do not determine p_ij, so edges that
touch them are not learned (saturated): they keep the model's p, and the boundary estimate of
a neighbouring flag or relay uses the model's p for that edge.

Learned priors for the decoder (priors): each edge of a circuit's graph takes the mean of its
type over the learning circuits; a type that only the one-round circuit has (round 1 to the
final layer) takes the mean of its "first" and "final" variants; an edge whose type was not
learned keeps the calibrated p.
"""

from __future__ import annotations

import numpy as np
import pymatching
import stim

from matching import mechanisms

CLASSES = ("space-like", "time-like", "diagonal", "boundary", "flag/relay")
DETECTOR_CLASSES = ("X weight 4", "X weight 2", "Z", "flag", "relay")
P_MIN, P_MAX = 1e-4, 0.499  # learned priors are clipped to this range before decoding


def detector_class(name: str) -> str:
    if name.startswith("z_flag"):
        return "flag"
    if name.startswith("relay"):
        return "relay"
    if name[0] == "X":
        return "X weight 4" if name.count("X") == 4 else "X weight 2"
    return "Z"


def is_readout(name: str) -> bool:
    """A flag or relay: a single readout, not a comparison of two rounds."""
    return " " in name


def edge_type(a, b, rounds: int) -> tuple[tuple, str, str]:
    """(type key, class, sub-class) of the edge between detector labels a and b (b None: boundary)."""
    if b is not None and (int(b[1]), b[2]) < (int(a[1]), a[2]):
        a, b = b, a
    ra = int(a[1])
    rb = ra if b is None else int(b[1])
    first, final = ra == 0, rb == rounds
    pos = "first+final" if first and final else "first" if first else "final" if final else "bulk"
    key = (str(a[0]), a[2], "" if b is None else b[2], rb - ra, pos)
    names = [a[2]] if b is None else [a[2], b[2]]
    readouts = sum(is_readout(n) for n in names)
    if readouts:
        sub = (
            "flag/relay boundary"
            if b is None
            else "flag/relay pair"
            if readouts == 2
            else "flag/relay-stabilizer"
        )
        return key, "flag/relay", sub
    if b is None:
        return key, "boundary", "boundary"
    if a[2] == b[2]:
        cls = "time-like" if rb - ra == 1 else "other"
    else:
        cls = "space-like" if rb == ra else "diagonal" if rb - ra == 1 else "other"
    return key, cls, cls


def key_str(key: tuple) -> str:
    return "|".join(map(str, key))


def graph(matching: pymatching.Matching) -> dict:
    """The merged matching graph as arrays: u, v (-1 for the boundary), p, observable sets."""
    u, v, p, obs = [], [], [], []
    for a, b, attr in matching.edges():
        u.append(a)
        v.append(-1 if b is None else b)
        p.append(attr["error_probability"])
        obs.append(tuple(sorted(attr["fault_ids"])))
    return {"u": np.array(u), "v": np.array(v), "p": np.array(p), "obs": obs}


def with_priors(g: dict, p: np.ndarray, num_detectors: int, num_observables: int):
    """The same graph (edges, boundary edges, observables) with edge probabilities p."""
    lines = []
    for a, b, q, obs in zip(g["u"], g["v"], np.clip(p, P_MIN, P_MAX), g["obs"]):
        dets = f"D{a}" if b < 0 else f"D{a} D{b}"
        lines.append(f"error({q:.12g}) {dets} {' '.join(f'L{k}' for k in obs)}".rstrip())
    lines += [f"detector D{d}" for d in range(num_detectors)]
    lines += [f"logical_observable L{k}" for k in range(num_observables)]
    m = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel("\n".join(lines)))
    assert m.num_edges == len(g["u"]), (m.num_edges, len(g["u"]))
    return m


def pij(mi, mj, mij):
    """The p_ij estimator from <x_i>, <x_j>, <x_i x_j> (any broadcastable shapes)."""
    den = 1 - 2 * mi - 2 * mj + 4 * mij
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = 1 - 4 * (mij - mi * mj) / den
        return np.where(den > 0, 0.5 - 0.5 * np.sqrt(np.clip(arg, 0, None)), np.nan)


def all_pairs(dets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """p_ij for every pair of detectors (D x D) from detection events (shots x D), and the
    estimator's denominator 1 - 2<x_i> - 2<x_j> + 4<x_i x_j>; near zero, p_ij is undetermined."""
    x = dets.astype(float)
    m = x.mean(0)
    mm = x.T @ x / x.shape[0]
    return pij(m[:, None], m[None, :], mm), 1 - 2 * m[:, None] - 2 * m[None, :] + 4 * mm


def boundary(m, p_edges, u, v):
    """p_i,boundary for every detector, from its rate m (..., D) and the estimates of its graph
    edges p_edges (..., E) between u and v (v >= 0 only)."""
    logs = np.zeros(m.shape)
    term = np.log(np.clip(1 - 2 * p_edges, 1e-12, None))  # NaN (not estimable) propagates
    for ends in (u, v):
        np.add.at(logs.T, ends, term.T)  # .T: detectors on the first axis, with or without B
    return 0.5 - 0.5 * (1 - 2 * m) * np.exp(-logs)


def estimate(dets: np.ndarray, g: dict, weights: np.ndarray | None = None) -> np.ndarray:
    """Every graph edge's p from detection events (shots x D): p_ij for the edges between two
    detectors, the remainder of p_i for the boundary edges. With weights (B x shots, each row
    summing to shots), one row per bootstrap replicate (B x E); without, one row (E)."""
    x = dets.astype(np.float32)
    pair = g["v"] >= 0
    u, v = g["u"][pair], g["v"][pair]
    prod = x[:, u] * x[:, v]
    n = x.shape[0]
    if weights is None:
        m, mij = x.mean(0), prod.mean(0)
    else:
        w = weights.astype(np.float32)
        m, mij = (w @ x) / n, (w @ prod) / n
    m, mij = m.astype(float), mij.astype(float)
    p_pair = pij(m[..., u], m[..., v], mij)
    out = np.empty((*m.shape[:-1], len(g["u"])))
    out[..., pair] = p_pair
    out[..., ~pair] = boundary(m, p_pair, u, v)[..., g["u"][~pair]]
    return out


def model_pairs(model: stim.Circuit) -> np.ndarray:
    """The model's p_ij for every pair of detectors (D x D), and its detection rates on the
    diagonal: what the estimator returns on the model with infinitely many shots."""
    probs, dets, _obs, _comps = mechanisms(model)
    n = model.num_detectors
    logs = np.zeros((n, n))
    for p, ds in zip(probs, dets):
        if ds:
            ix = np.array(ds)
            logs[np.ix_(ix, ix)] += np.log1p(-2 * p)
    return 0.5 - 0.5 * np.exp(logs)


def model_estimate(pairs: np.ndarray, g: dict) -> np.ndarray:
    """estimate() on the model's exact moments: each graph edge as the estimator sees it."""
    pair = g["v"] >= 0
    u, v = g["u"][pair], g["v"][pair]
    out = np.empty(len(g["u"]))
    out[pair] = pairs[u, v]
    out[~pair] = boundary(np.diag(pairs).copy(), pairs[u, v], u, v)[g["u"][~pair]]
    return out


def numbered(labels) -> list[tuple]:
    """Labels with the two relays of one gauge told apart: a repeated (patch, round, name)
    gets #k, k counting from 0 in detector order (as common.readout_series does)."""
    seen: dict = {}
    out = []
    for patch, r, name in labels:
        k = seen.get((str(patch), int(r), name), 0)
        seen[(str(patch), int(r), name)] = k + 1
        out.append((str(patch), int(r), f"{name}#{k}" if name.startswith("relay") else name))
    return out


def types(g: dict, labels, rounds: int) -> tuple[list[tuple], list[str], list[str]]:
    """Type key, class and sub-class of every graph edge."""
    labels = numbered(labels)
    keys, cls, sub = [], [], []
    for a, b in zip(g["u"], g["v"]):
        k, c, s = edge_type(labels[a], None if b < 0 else labels[b], rounds)
        keys.append(k)
        cls.append(c)
        sub.append(s)
    return keys, cls, sub


def saturated(label) -> bool:
    """A distance-5 Z-memory stabilizer: too close to 50% for p_ij from 2500 shots."""
    return str(label[0]) == "5" and detector_class(label[2]) == "Z"


def rates(dets: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    x = dets.astype(np.float32)
    return (x.mean(0) if w is None else (w @ x) / x.shape[0]).astype(float)


def learned(dets: np.ndarray, g: dict, labels, model_p: np.ndarray, w=None) -> np.ndarray:
    """estimate(), with the edges that touch a saturated detector set to NaN and the boundary
    edges recomputed with the model's p (model_estimate) on those edges."""
    sat = np.array([saturated(lab) for lab in labels])
    fixed = sat[g["u"]] | np.where(g["v"] >= 0, sat[np.maximum(g["v"], 0)], False)
    est = estimate(dets, g, w)
    pair = g["v"] >= 0
    p_pair = np.where(fixed[pair], model_p[pair], est[..., pair])
    est[..., ~pair] = boundary(rates(dets, w), p_pair, g["u"][pair], g["v"][pair])[
        ..., g["u"][~pair]
    ]
    est[..., fixed] = np.nan
    return est


def type_id(basis: str, key: tuple) -> str:
    """'basis patch name_a|name_b|dr|pos', as pij.json stores the types."""
    return f"{basis} {key[0]} {key_str(key[1:])}"


def by_type(ids: list[str], values: np.ndarray, into: dict) -> None:
    """Collect each edge's value under its type id (into: id -> list of values)."""
    for t, v in zip(ids, values):
        into.setdefault(t, []).append(v)


def type_means(collected: dict) -> dict[str, float]:
    """Mean over a type's instances; types with any instance not estimated are left out."""
    return {t: float(np.mean(v)) for t, v in collected.items() if np.all(np.isfinite(np.array(v)))}


def priors(
    g: dict, ids: list[str], values: dict[str, float], scaled: bool = False, boundary: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Each edge's learned p (see the module docstring) and whether it kept the calibrated p.

    values: the learned p of each type, or with scaled=True the ratio learned / model (both as
    the estimator sees them), which then multiplies the edge's calibrated p. Scaling keeps the
    model's split of mechanisms that fire three or more detectors, which the pair estimator
    cannot see, so on the model's own shots it returns the calibrated priors up to noise.
    boundary=False keeps every boundary edge at its calibrated p."""
    p = g["p"].copy()
    kept = np.ones(len(p), bool)
    for e, t in enumerate(ids):
        if not boundary and g["v"][e] < 0:
            continue
        v = values.get(t)
        if v is None and t.endswith("|first+final"):
            both = [values.get(t.replace("first+final", s)) for s in ("first", "final")]
            both = [x for x in both if x is not None]
            v = float(np.mean(both)) if both else None
        if v is not None:
            p[e], kept[e] = (g["p"][e] * v if scaled else v), False
    return p, kept


def multinomial_weights(shots: int, B: int, seed: int) -> np.ndarray:
    """B bootstrap replicates of the shots as multiplicities (B x shots)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(shots, np.full(shots, 1 / shots), size=B).astype(np.float32)
