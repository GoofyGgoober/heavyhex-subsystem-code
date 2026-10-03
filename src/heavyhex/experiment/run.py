"""Prepare, submit and analyze the runs. Each job keeps everything in its own folder.

The run is RUN_DAYS days. Each day has one rep per clean place for the d=5
patch (at most MAX_REPS), with the d=3 patch beside it in the same job, or, if no
clean d=3 place fits beside it, in a job of its own straight after.

prepare_day: plan the day's reps from the day's calibration and prepare a folder
for each job. prepare: place one job's patches, translate the circuits for the
chip, and freeze the simulator's prediction. rehearse: fake the chip's shots with
the simulator, into rehearsal.npz. submit: send the circuits (uses the QPU, at
most QPU_SECONDS_LIMIT per job and QPU_TOTAL_SECONDS over every job), then save
the shots and IBM's calibration as the job finished. analyze: decode one job's
shots and compare with the prediction. combine: Λ for each rep and over all of them.

The job lists every circuit's pieces in BLOCKS blocks, each in its own shuffled
order, and every piece takes the same number of shots. IBM runs a job shot by shot
across all its circuits, so the first and second halves of each piece's shots are
the job's first and second halves in time: comparing them shows drift during the job.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from math import log, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..circuits.flagged import memory_circuit_flagged
from ..patches.operators import build_operators
from ..patches.placement import MAX_REPS, Calibration, fez_plan, fez_qubits
from ..simulation.noisy import add_detectors, fit_per_round
from ..simulation.scheduled import noisy_scheduled
from .circuits import (
    DISTANCES,
    ROUNDS,
    SETTINGS,
    Setting,
    distance,
    for_fez,
    job_order,
    logical_circuit,
    piece_shots,
)

if TYPE_CHECKING:
    import numpy as np
    import stim
    from qiskit import QuantumCircuit

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs"
# Simulated shots per circuit. At 50,000 the noise in Λ is about 0.01 and biases it low.
SIMULATED_SHOTS = 200_000
SIDE_SHOTS = 50_000  # for side predictions that no decision rests on
ENDS = (0.0, 1.0)  # decoupled_dephasing at the two ends of the prediction
# How far errors in the CZ, readout and single-qubit inputs move the f read off the
# memory runs: 0.06-0.13 between IBM's 3 October 2026 calibration and the day before's
# (docs/calibration_scatter.py).
CALIBRATION_SCATTER_F = 0.1
# A data qubit whose idle test pins its own f no better than this is left to the median.
LOOSEST_F = 0.5
# A rise in detection rate over the rounds, beyond the prediction, by more than this
# many standard errors counts as leakage or heating. Set before the run.
LEAKAGE_BAR = 3.0
# A change between the first and second halves of the job's shots beyond this many
# standard errors, in any of the four ε, counts as drift. Set before the run; a drift-free
# rehearsal reached 3.1.
DRIFT_BAR = 3.5
# Days with at least one rep; a day with no clean place for the d=5 patch doesn't count.
RUN_DAYS = 5
# Shots per memory circuit in each job; each idle readout gets half.
SHOTS_PER_CIRCUIT = 2500
# A job may use this much QPU time and no more, and every job together QPU_TOTAL_SECONDS:
# IBM cancels a job at its limit, whatever has happened by then, a job's limit is cut to
# what the total has left, and a run folder that had a job is never sent again. Set by
# the author.
QPU_SECONDS_LIMIT = 50
QPU_TOTAL_SECONDS = 300
# Jobs across every run folder: RUN_DAYS days of MAX_REPS reps, each at most two jobs.
QPU_JOBS_LIMIT = 20
# IBM's rule for a sampler job's QPU time: about 2 s to load it, then the repetition delay
# and the circuit for every shot (quantum.cloud.ibm.com/docs/guides/estimate-job-run-time).
LOAD_SECONDS = 2.0
REP_DELAY = 250e-6  # IBM's default, when the backend doesn't say


SHOTS, REHEARSAL = "shots.npz", "rehearsal.npz"


def prepare_day(
    calibration: Calibration, target: Any, *, offline: bool = False, reps: int = MAX_REPS
) -> list[tuple[Path, dict]]:
    """Plan the day's reps and prepare a folder for each job, in the order they are sent."""
    plan = fez_plan(calibration, reps)
    if not plan:
        raise ValueError(
            f"no clean place for the d=5 patch on {calibration.backend} today; the day is skipped"
        )
    day = f"{datetime.now():%Y-%m-%d}"
    prepared = []
    for r, jobs in enumerate(plan, start=1):
        pairing = "paired" if len(jobs) == 1 else "split"
        for k, spots in enumerate(jobs, start=1):
            chips = {str(d): fez_qubits(d, calibration, spot) for d, spot in sorted(spots.items())}
            alone = None if pairing == "paired" else next(iter(spots))
            folder = day_folder(calibration.backend, day, r, alone, offline=offline)
            rep = {
                "day": day,
                "rep": r,
                "pairing": pairing,
                "job": k,  # its place in the rep's order
                "spots": {str(d): [list(s.origin), list(s.direction)] for d, s in spots.items()},
            }
            prediction = prepare(folder, calibration, target, offline=offline, chips=chips, rep=rep)
            prepared.append((folder, prediction))
    return prepared


def day_folder(
    backend: str, day: str, rep: int, alone: int | None = None, *, offline: bool = False
) -> Path:
    """runs/<backend>-<day>-r<rep>, with -d<d> for a patch that runs alone."""
    return RUNS / (
        f"{backend}-{day}-r{rep}{f'-d{alone}' if alone else ''}{'-offline' if offline else ''}"
    )


def prepare(
    folder: Path,
    calibration: Calibration,
    target: Any,
    *,
    offline: bool = False,
    chips: dict[str, list[int]] | None = None,
    rep: dict | None = None,
) -> dict:
    """Write a job's calibration, circuits and prediction into a new folder.

    chips gives each of the job's patches its chip qubits; both patches at their
    best pair of spots if it isn't given. rep records the job's day and rep.
    """
    from qiskit import qpy

    chips = chips or {str(d): fez_qubits(d, calibration) for d in DISTANCES}
    circuits = [for_fez(logical_circuit(s, chips, target.num_qubits)[0], target) for s in SETTINGS]
    folder.mkdir(parents=True)
    calibration.save(folder / "calibration.json")
    with gzip.open(folder / "circuits.qpy.gz", "wb") as f:
        qpy.dump(circuits, f)
    run = {
        "backend": calibration.backend,
        "dt": target.dt,
        "offline": offline,
        "prepared": _now(),
        "code": commit(),
        "chips": chips,
        "rep": rep,
        "settings": [asdict(s) for s in SETTINGS],
        "order": job_order(SETTINGS),
        "seconds": [c.estimate_duration(target, unit="s") for c in circuits],
    }
    _write(folder / "run.json", run)
    prediction = {
        str(f): predict(circuits, calibration, target.dt, decoupled_dephasing=f) for f in ENDS
    }
    _write(folder / "prediction.json", prediction)
    return prediction


def predict(
    circuits: list[QuantumCircuit],
    calibration: Calibration,
    dt: float,
    *,
    decoupled_dephasing: float,
    dephasing_by_qubit: dict[int, float] | None = None,
    shots: int | None = None,
) -> dict:
    """What the simulator expects of each circuit, then the fits."""
    shots = shots or SIMULATED_SHOTS
    settings = []
    for setting, circuit in zip(SETTINGS, circuits):
        model, records, labels = simulated(
            setting, circuit, calibration, dt, decoupled_dephasing, dephasing_by_qubit
        )
        if setting.decoded:
            sampler = model.compile_detector_sampler(seed=setting.rounds)
            detectors, flips = sampler.sample(shots, separate_observables=True)
            failed = _decode(model, detectors) != flips
            settings.append(_decoded(setting, labels, detectors, failed))
        else:
            measured = model.compile_sampler(seed=setting.rounds).sample(shots)
            settings.append({"x": x_values(by_register(measured, records))})
    return {"settings": settings, **_fits(settings, shots, _patches(settings))}


def simulated(
    setting: Setting,
    circuit: QuantumCircuit,
    calibration: Calibration,
    dt: float,
    decoupled_dephasing: float,
    dephasing_by_qubit: dict[int, float] | None = None,
) -> tuple[stim.Circuit, dict[tuple[str, int], int], list]:
    """The stim circuit for a translated circuit; memory gets detectors and one logical per patch.

    Returns the circuit, the measurement of each (register, bit), and detector labels
    as (patch, round, stabilizer). Logical k is the circuit's k-th patch.
    """
    model, records = noisy_scheduled(
        circuit,
        calibration,
        dt,
        decoupled_dephasing=decoupled_dephasing,
        dephasing_by_qubit=dephasing_by_qubit,
    )
    labels: list = []
    if setting.decoded:
        present = [p for p in setting.patches if (f"d{p}", 0) in records]
        for k, patch in enumerate(present):
            _, schedule = memory_circuit_flagged(
                build_operators(distance(patch)), rounds=setting.rounds, basis=setting.basis
            )
            mine = {(reg[0], i): m for (reg, i), m in records.items() if reg[1:] == patch}
            found: list = []
            add_detectors(model, schedule, mine, observable=k, labels=found)
            labels += [(patch, *label) for label in found]
    return model, records, labels


def submit(folder: Path, backend: Any, shots: int) -> None:
    """Send the job's circuits and save every shot. Uses QPU time on a real backend.

    shots is per memory circuit; each idle readout gets half. Refuses a folder that
    already has a job: analyze fetches its shots. On the QPU the job is capped at
    job_limit(): QPU_SECONDS_LIMIT, or what QPU_TOTAL_SECONDS has left if that is
    less. It isn't sent if it's estimated to need more, it's cancelled before it runs
    if IBM's estimate, read once as it's queued, is more, and IBM cancels it if it
    uses more. job.json claims the folder before anything is sent, so even a job
    whose reply from IBM is lost is never sent twice.
    """
    from qiskit_ibm_runtime import SamplerV2

    run = _read(folder / "run.json")
    check_settings(run)
    on_qpu = not _is_local(backend)
    if run["offline"] and on_qpu:
        raise ValueError("this run was prepared offline; prepare it again before using the QPU")
    refuse_resubmit(folder)
    limit = None
    if on_qpu:
        refuse_another_job()
        limit = job_limit()
    each = piece_shots(shots)
    estimate = qpu_seconds(run, shots, backend) if on_qpu else 0.0
    if limit is not None and estimate > limit:
        raise ValueError(
            f"the job would need about {estimate:.0f} s of QPU time, over its {limit} s limit "
            f"({QPU_SECONDS_LIMIT} s a job, {QPU_TOTAL_SECONDS} s in all, "
            f"{qpu_committed():.0f} s already counted); not sent"
        )
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = each
    tag = f"heavyhex:{folder.name}"  # finds the job in IBM's list if its reply is lost
    if on_qpu:
        sampler.options.max_execution_time = limit  # IBM cancels the job past it
        sampler.options.environment.job_tags = [tag]
        # IBM's own decoupling and twirling would change the circuit the prediction is for.
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
    circuits = load_circuits(folder)
    record = {
        "job": None,  # until IBM's reply gives the job's id
        "tag": tag,
        "backend": backend.name,
        "shots": shots,
        "shots_per_piece": each,
        "submitted": _now(),
        "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": commit(),
        "local": not on_qpu,
        "qpu_seconds_limit": limit,
        "qpu_seconds_estimate": estimate if on_qpu else None,
    }
    with open(folder / "job.json", "x") as f:  # claims the folder, never over an earlier job
        f.write(json.dumps(record, indent=1) + "\n")
    job = sampler.run([circuits[i] for i, _ in run["order"]])
    record["job"] = job.job_id()
    _write(folder / "job.json", record)
    if on_qpu:
        try:
            ibm = (job.usage_estimation or {}).get("quantum_seconds")
        except Exception:  # only a second look; IBM enforces the limit either way
            ibm = None
        record["qpu_seconds_ibm_estimate"] = ibm
        _write(folder / "job.json", record)
        if ibm is not None and limit is not None and ibm > limit:
            try:
                job.cancel()
                outcome = "cancelled before it ran"
            except Exception as error:
                outcome = f"cancelling failed ({error}), so IBM stops it at the limit"
            with suppress(Exception):
                finish(folder, job)
            raise ValueError(
                f"IBM estimates the job at {ibm:.0f} s of QPU time, over its "
                f"{limit} s limit; {outcome}"
            )
    try:
        result = job.result()
    except Exception:
        if on_qpu:
            with suppress(Exception):
                finish(folder, job)  # what a cancelled or failed job used, and when
        raise
    save_shots(folder, result)
    if on_qpu:
        finish(folder, job)


def qpu_seconds(run: dict, shots: int, backend: Any = None) -> float:
    """The job's QPU time by IBM's rule: loading, then rep delay and circuit per shot."""
    each = piece_shots(shots)
    configuration = getattr(backend, "configuration", None)
    rest = getattr(configuration(), "default_rep_delay", None) if configuration else None
    rest = rest or REP_DELAY
    return LOAD_SECONDS + each * sum(run["seconds"][i] + rest for i, _ in run["order"])


def _qpu_jobs() -> list[tuple[str, dict]]:
    """The run folders' jobs that went, or may have gone, to the QPU."""
    found = []
    for path in sorted(RUNS.glob("*/job.json")):
        record = json.loads(path.read_text() or "{}")
        if not record.get("local"):  # an unreadable or older record counts as the QPU's
            found.append((path.parent.name, record))
    return found


def qpu_committed() -> float:
    """QPU seconds the jobs so far count against QPU_TOTAL_SECONDS: what IBM counted
    for a job it has finished counting, and the job's whole limit otherwise.
    """
    total = 0.0
    for _, record in _qpu_jobs():
        usage = record.get("usage") or {}
        used = next(
            (usage[key] for key in ("qpu_charge_time_seconds", "quantum_seconds") if key in usage),
            None,
        )
        limit = record.get("qpu_seconds_limit") or QPU_SECONDS_LIMIT
        total += used if used is not None and usage.get("status") == "complete" else limit
    return total


def job_limit() -> int:
    """The next job's QPU limit: QPU_SECONDS_LIMIT, or what QPU_TOTAL_SECONDS has left."""
    return max(int(min(QPU_SECONDS_LIMIT, QPU_TOTAL_SECONDS - qpu_committed())), 0)


def refuse_another_job() -> None:
    """Refuse once the jobs reach QPU_JOBS_LIMIT or the QPU time QPU_TOTAL_SECONDS."""
    sent = _qpu_jobs()
    if len(sent) >= QPU_JOBS_LIMIT:
        raise ValueError(
            f"{len(sent)} jobs have gone, and {QPU_JOBS_LIMIT} were approved in all; "
            "nothing more goes to the QPU without the author's new approval"
        )
    if job_limit() <= 0:
        raise ValueError(
            f"the jobs so far count {qpu_committed():.0f} s of the {QPU_TOTAL_SECONDS} s "
            "approved in all; nothing more goes to the QPU without the author's new approval"
        )


def usage_pending(folder: Path) -> bool:
    """Whether IBM is still counting the QPU time of the folder's job."""
    record = _read(folder / "job.json")
    if record.get("qpu_seconds_limit") is None or record.get("job") is None:
        return False  # a local run, or a job IBM's reply never named
    return (record.get("usage") or {}).get("status") != "complete"


def refuse_resubmit(folder: Path) -> None:
    if (folder / "job.json").exists() or (folder / SHOTS).exists():
        raise ValueError(
            f"{folder} already has a job or the chip's shots; "
            f"heavyhex experiment analyze --run {folder} fetches and analyzes them"
        )


def check_settings(run: dict) -> None:
    """Refuse a run prepared with other settings than today's code has."""
    if run.get("settings") != [asdict(s) for s in SETTINGS] or run.get("order") != [
        list(piece) for piece in job_order(SETTINGS)
    ]:
        raise ValueError("this run was prepared with other settings; prepare it again")


def finish(folder: Path, job: Any) -> None:
    """When the job ran, and IBM's calibration in force as it finished (read-only)."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    record = _read(folder / "job.json")
    metrics = job.metrics() or {}
    stamps = metrics.get("timestamps", {})
    record.update({f"{name}_utc": stamps.get(name) for name in ("created", "running", "finished")})
    record["usage"] = metrics.get("usage")  # IBM's count of the QPU time; "status" says if final
    record["status"] = str(job.status())
    with suppress(Exception):
        record["reason"] = job.error_message()  # why IBM stopped it, if it did
    _write(folder / "job.json", record)
    if stamps.get("finished"):
        when = datetime.fromisoformat(stamps["finished"].replace("Z", "+00:00"))
        device = QiskitRuntimeService().backend(record["backend"])
        Calibration.from_device(device, at=when).save(folder / "calibration_after.json")


def rehearse(
    folder: Path,
    shots: int,
    decoupled_dephasing: float,
    seed: int = 0,
    dephasing_by_qubit: dict[int, float] | None = None,
) -> None:
    """Write simulated shots, as the chip would return them, to rehearsal.npz.

    analyze(folder, rehearsal=True) reads them; the chip's own shots go to shots.npz.
    """
    import numpy as np

    run = _read(folder / "run.json")
    check_settings(run)
    calibration = Calibration.load(folder / "calibration.json")
    circuits = load_circuits(folder)
    models: dict[int, tuple] = {}
    saved = {}
    for k, (i, block) in enumerate(run["order"]):
        if i not in models:
            setting, circuit = SETTINGS[i], circuits[i]
            models[i] = simulated(
                setting, circuit, calibration, run["dt"], decoupled_dephasing, dephasing_by_qubit
            )[:2]
        model, records = models[i]
        measured = model.compile_sampler(seed=seed + k).sample(piece_shots(shots))
        for name, bits in by_register(measured, records).items():
            saved[f"{k}/{i}/{block}/{name}"] = bits
    np.savez_compressed(folder / REHEARSAL, **saved)


def fetch(folder: Path) -> None:
    """Get the shots of a submitted job from IBM (read-only). Never sends anything."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    record = _read(folder / "job.json")
    if record.get("job") is None:
        raise ValueError(
            f"a job was sent at {record.get('submitted_utc')} UTC but IBM's reply never came. "
            "Find it, read-only, with QiskitRuntimeService().jobs(job_tags="
            f"[{record.get('tag')!r}]) and put its id in job.json; this folder is not sent again"
        )
    job = QiskitRuntimeService().job(record["job"])
    status = str(job.status())
    if status in ("CANCELLED", "ERROR"):
        reason = None
        with suppress(Exception):
            reason = job.error_message()
            finish(folder, job)
        raise ValueError(
            f"job {record['job']} ended {status} without shots ({reason}); "
            "this folder is not sent again"
        )
    save_shots(folder, job.result())
    if not (folder / "calibration_after.json").exists():
        finish(folder, job)


def save_shots(folder: Path, result: Any) -> None:
    """Every piece's bits, keyed by piece, circuit, block and register."""
    import numpy as np

    order = _read(folder / "run.json")["order"]
    shots = {
        f"{k}/{i}/{block}/{name}": bits.to_bool_array(order="little")
        for k, ((i, block), pub) in enumerate(zip(order, result))
        for name, bits in pub.data.items()
    }
    np.savez_compressed(folder / SHOTS, **shots)


def analyze(
    folder: Path, decoupled_dephasing: float | None = None, *, rehearsal: bool = False
) -> dict:
    """Decode the shots and set them against the prediction.

    The idle test gives decoupled_dephasing, unless one is given: the median over
    the data qubits, used for every qubit. The memory shots are decoded, and the
    simulator rerun, with that value. Each data qubit's own value, a prediction
    from the calibration after the run, and the two halves of the job are side checks.
    rehearsal analyzes rehearsal.npz instead of the chip's shots.
    """
    import numpy as np

    run = _read(folder / "run.json")
    check_settings(run)
    patches = patches_of(run)
    calibration = Calibration.load(folder / "calibration.json")
    circuits = load_circuits(folder)
    source = REHEARSAL if rehearsal else SHOTS
    halves_of_shots = [load_shots(folder, source=source, half=half) for half in (0, 1)]
    shots = load_shots(folder, source=source)

    measured_f = _idle_dephasing(shots, circuits, calibration, run["dt"])
    if decoupled_dephasing is None and not np.isfinite(measured_f["median"]):
        raise ValueError("the idle test gave no dephasing; pass one to analyze")
    f = measured_f["median"] if decoupled_dephasing is None else decoupled_dephasing
    f = max(float(f), 0.0)  # above 1 is physical (more noise than the echo T2), below 0 is not
    table = []
    if decoupled_dephasing is None:
        table = _dephasing_table(measured_f, run["chips"], calibration, f)

    observed, by_half = [], [[], []]
    for i, (setting, circuit) in enumerate(zip(SETTINGS, circuits)):
        if not setting.decoded:
            observed.append({"x": x_values(shots[i])})
            for part in by_half:
                part.append({})
            continue
        model, records, labels = simulated(setting, circuit, calibration, run["dt"], f)
        failed, detectors = [], []
        for part, half in zip(by_half, halves_of_shots):
            measurements = in_record_order(half[i], records)
            fired, flips = model.compile_m2d_converter().convert(
                measurements=measurements, separate_observables=True
            )
            failed.append(_decode(model, fired) != flips)
            detectors.append(fired)
            part.append({"logical_error": failed[-1].mean(axis=0).tolist()})
        observed.append(
            _decoded(setting, labels, np.concatenate(detectors), np.concatenate(failed))
        )
    count = _memory_shots(shots)
    seen = {"settings": observed, **_fits(observed, count, patches)}
    halves = [
        _fits(part, _memory_shots(half), patches)["fits"]
        for part, half in zip(by_half, halves_of_shots)
    ]

    expected = predict(circuits, calibration, run["dt"], decoupled_dephasing=f)
    after = None
    after_missing = (folder / "job.json").exists() and not (
        folder / "calibration_after.json"
    ).exists()
    filled: list[int] = []
    if (folder / "calibration_after.json").exists():
        later, filled = _filled(
            Calibration.load(folder / "calibration_after.json"), calibration, run["chips"]
        )
        after = predict(circuits, later, run["dt"], decoupled_dephasing=f)
    own = None
    if table:
        own = predict(
            circuits,
            calibration,
            run["dt"],
            decoupled_dephasing=f,
            dephasing_by_qubit={row["qubit"]: row["dephasing"] for row in table},
            shots=SIDE_SHOTS,
        )
    frozen = _read(folder / "prediction.json")
    analysis = {
        "shots_from": source,
        "patches": patches,
        "rep": run.get("rep"),
        "code": {"prepared": run.get("code"), "analyzed": commit()},
        "calibration_after_missing": after_missing,
        "calibration_after_filled": filled,
        "decoupled_dephasing": f,
        "idle_test": measured_f,
        "dephasing_by_qubit": table,
        "observed": seen,
        "drift": _drift(halves),
        "predicted_at_measured_dephasing": expected,
        "predicted_from_calibration_after": after,
        "predicted_with_each_qubits_dephasing": own,
        "frozen_prediction": {f_end: p["fits"] for f_end, p in frozen.items()},
        "agreement": _agreement(
            seen["fits"],
            expected["fits"],
            after and after["fits"],
            _slopes(frozen, expected, f),
            measured_f["median_uncertainty"] if decoupled_dephasing is None else 0.0,
        ),
        "dephasing_from_memory": _dephasing_from_memory(seen, frozen, expected, f, measured_f),
        "leakage": _leakage(observed, expected["settings"], count),
        **_worst_detectors(
            observed, expected["settings"], count, after["settings"] if after else None
        ),
    }
    _write(folder / ("rehearsal_analysis.json" if rehearsal else "analysis.json"), analysis)
    return analysis


def combine(*, rehearsal: bool = False) -> dict:
    """Λ for every rep, and over all of them, from the job folders' analyses.

    A rep's Λ sets its d=3 job's ε₃ against its d=5 job's ε₅, the same job when the
    rep is paired; a split rep's bound adds its two jobs' variances. In X memory, Λ
    pooled over the reps, each weighted by its inverse variance, is decided by the
    same one-sided rule as one rep; and the simulation matches the chip if, for each
    of Λ, ε₃ and ε₅, the sum over reps of ((observed - predicted) / (bound / 2))² is
    below the 95% point of χ² with a degree per rep. Z memory is reported, not decided.
    """
    from scipy.stats import chi2

    name = "rehearsal_analysis.json" if rehearsal else "analysis.json"
    reps: dict[tuple[str, int], dict] = {}
    for folder in sorted(path.parent for path in RUNS.glob(f"*/{name}")):
        info = _read(folder / "run.json")
        if not info.get("rep") or (info.get("offline") and not rehearsal):
            continue
        analysis = _read(folder / name)
        entry = reps.setdefault(
            (info["rep"]["day"], info["rep"]["rep"]),
            {
                "day": info["rep"]["day"],
                "rep": info["rep"]["rep"],
                "pairing": info["rep"]["pairing"],
                "folders": [],
                "spots": {},
                "decoupled_dephasing": {},
                "tests": {},
                "drift": False,
                "leakage": False,
                "detectors_beyond_chance": 0,
            },
        )
        entry["folders"].append(folder.name)
        entry["spots"].update(info["rep"]["spots"])
        entry["decoupled_dephasing"][folder.name] = analysis["decoupled_dephasing"]
        entry["tests"].update(analysis["agreement"]["tests"])
        entry["drift"] |= any(row["drift"] for row in analysis["drift"].values())
        entry["leakage"] |= any(row["rises"] for row in analysis["leakage"].values())
        entry["detectors_beyond_chance"] += analysis["detectors_beyond_chance"]
    complete = []
    for entry in reps.values():
        tests = entry["tests"]
        if not all(f"{b} eps{d}" in tests for b in "XZ" for d in DISTANCES):
            continue  # a split rep with one job still to analyze
        for basis in "XZ":
            if f"{basis} lambda" not in tests:
                tests[f"{basis} lambda"] = _ratio_test(
                    tests[f"{basis} eps3"], tests[f"{basis} eps5"]
                )
        row = tests["X lambda"]
        entry["verdict"] = verdict(row["observed"], row["observed_uncertainty"])
        complete.append(entry)
    complete.sort(key=lambda entry: (entry["day"], entry["rep"]))
    pooled: dict = {}
    model: dict = {}
    if complete:
        rows = [entry["tests"]["X lambda"] for entry in complete]
        weights = [1 / row["observed_uncertainty"] ** 2 for row in rows]
        lam = sum(w * row["observed"] for w, row in zip(weights, rows)) / sum(weights)
        sigma = 1 / sqrt(sum(weights))
        pooled = {
            "lambda": lam,
            "lambda_uncertainty": sigma,
            "verdict": verdict(lam, sigma),
            "predicted": sum(w * row["predicted"] for w, row in zip(weights, rows)) / sum(weights),
        }
        for test in ("X lambda", "X eps3", "X eps5", "Z lambda", "Z eps3", "Z eps5"):
            z = [
                (row["observed"] - row["predicted"]) / (row["bound"] / 2)
                for row in (entry["tests"][test] for entry in complete)
            ]
            stat, bar = sum(v * v for v in z), float(chi2.ppf(0.95, len(z)))
            model[test] = {
                "chi2": stat,
                "reps": len(z),
                "bar": bar,
                "p_value": float(chi2.sf(stat, len(z))),
                "agrees": stat < bar,
            }
    combined = {
        "shots_from": REHEARSAL if rehearsal else SHOTS,
        "days": len({entry["day"] for entry in complete}),
        "reps": complete,
        "pooled_x": pooled,
        "model": model,
        "matches": {
            basis: all(model[f"{basis} {q}"]["agrees"] for q in ("lambda", "eps3", "eps5"))
            for basis in "XZ"
            if model
        },
        "incomplete": [entry["folders"] for entry in reps.values() if entry not in complete],
    }
    _write(RUNS / ("rehearsal_combined.json" if rehearsal else "combined.json"), combined)
    return combined


def _ratio_test(e3: dict, e5: dict) -> dict:
    """The Λ test of a split rep, from its two jobs' ε tests, as independent."""
    observed = e3["observed"] / e5["observed"]
    predicted = e3["predicted"] / e5["predicted"]

    def spread(key: str, against: str) -> float:
        return sqrt(sum((row[key] / row[against]) ** 2 for row in (e3, e5)))

    sigma = observed * spread("observed_uncertainty", "observed")
    noise = predicted * spread("simulator_noise", "predicted")
    from_f = predicted * spread("from_f", "predicted")
    gap = predicted * spread("calibration_gap", "predicted")
    bound = 2 * sqrt(sigma**2 + noise**2 + from_f**2 + gap**2 / 2)
    return {
        "observed": observed,
        "predicted": predicted,
        "observed_uncertainty": sigma,
        "simulator_noise": noise,
        "from_f": from_f,
        "calibration_gap": gap,
        "bound": bound,
        "agrees": abs(observed - predicted) < bound,
    }


CODE = ("src", "pyproject.toml")  # what the prediction depends on


def commit() -> dict | None:
    """The git commit; the hash of the code in it, which committing a run folder
    doesn't change; and whether the code in the working tree differs from it.
    """

    def git(*words: str) -> str:
        return subprocess.run(
            ["git", *words], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        code = git("rev-parse", *(f"HEAD:{path}" for path in CODE)).split()
        changed = git("status", "--porcelain", "--", *CODE)
        return {"commit": git("rev-parse", "HEAD"), "code": code, "changed": bool(changed)}
    except (OSError, subprocess.CalledProcessError):
        return None


def uncommitted(folder: Path) -> list[str]:
    """What keeps the run folder off the record: files git doesn't have as they are,
    or commits not yet pushed. Empty once the folder is committed and pushed.
    """
    folder = folder.resolve()
    try:
        listed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", str(folder)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(folder / "run.json")],
            cwd=REPO,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return [str(folder)]
    found = [line[3:] for line in listed]
    if tracked.returncode:
        found.append(str((folder / "run.json").relative_to(REPO)))
    # Commits the upstream lacks, as of the last fetch; no upstream counts as unpushed.
    ahead = subprocess.run(
        ["git", "rev-list", "@{u}..HEAD", "--", str(folder)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if ahead.returncode or ahead.stdout.strip():
        found.append("(not pushed)")
    return sorted(set(found))


def load_circuits(folder: Path) -> list[QuantumCircuit]:
    from qiskit import qpy

    with gzip.open(folder / "circuits.qpy.gz", "rb") as f:
        return qpy.load(f)


def load_shots(
    folder: Path, source: str = SHOTS, half: int | None = None
) -> list[dict[str, np.ndarray]]:
    """Per circuit, each register's bits as shots x bits, from every piece of the job.

    half 0 or 1 keeps the first or second half of every piece's shots, in the order
    they ran: IBM runs the job shot by shot across all its pieces.
    """
    import numpy as np

    saved = np.load(folder / source)
    parts: list[dict[str, list[tuple[int, np.ndarray]]]] = [{} for _ in SETTINGS]
    for key in saved.files:
        k, i, _, name = key.split("/")
        bits = saved[key]
        if half is not None:
            middle = len(bits) // 2
            bits = bits[:middle] if half == 0 else bits[middle:]
        parts[int(i)].setdefault(name, []).append((int(k), bits))
    return [
        {name: np.concatenate([bits for _, bits in sorted(found)]) for name, found in part.items()}
        for part in parts
    ]


def _decode(model: stim.Circuit, detectors: np.ndarray) -> np.ndarray:
    import pymatching

    matching = pymatching.Matching.from_detector_error_model(
        model.detector_error_model(decompose_errors=True)
    )
    return matching.decode_batch(detectors).astype(bool)


def by_register(measured: np.ndarray, records: dict) -> dict[str, np.ndarray]:
    """Simulated shots (shots x measurements) as the chip returns them: per register."""
    import numpy as np

    sizes: dict[str, int] = {}
    for name, i in records:
        sizes[name] = max(sizes.get(name, 0), i + 1)
    bits = {name: np.zeros((len(measured), size), dtype=bool) for name, size in sizes.items()}
    for (name, i), k in records.items():
        bits[name][:, i] = measured[:, k]
    return bits


def in_record_order(bits: dict[str, np.ndarray], records: dict) -> np.ndarray:
    import numpy as np

    count = next(iter(bits.values())).shape[0]
    out = np.zeros((count, len(records)), dtype=bool)
    for (name, i), k in records.items():
        out[:, k] = bits[name][:, i]
    return out


def x_values(bits: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """<X> of each data qubit, per patch, from the final readout (<Y> for a Y readout)."""
    return {
        str(d): [float(1 - 2 * bits[f"d{d}"][:, q].mean()) for q in range(d * d)]
        for d in DISTANCES
        if f"d{d}" in bits
    }


def patches_of(run: dict) -> list[str]:
    """The patches a job runs, "3" before "5"."""
    return [p for p in map(str, DISTANCES) if p in run["chips"]]


def _patches(rows: list[dict]) -> list[str]:
    """The patches of decoded rows, in the order of their logicals."""
    labels = next(row["detectors"] for row in rows if "detectors" in row)
    return list(dict.fromkeys(str(patch) for patch, _, _ in labels))


def _decoded(setting: Setting, labels: list, detectors: np.ndarray, failed: np.ndarray) -> dict:
    """A decoded circuit's logical error and detector rates, observed or simulated.

    For the longest memory runs, also the covariance across shots of each patch's
    mean stabilizer detection per round, which the leakage check needs.
    """
    import numpy as np

    rates = detectors.mean(axis=0).tolist()
    found = {
        "logical_error": failed.mean(axis=0).tolist(),
        "detector_rates": rates,
        "detectors": labels,
        "by_round": _by_round(labels, rates),
    }
    if setting.kind == "memory" and setting.rounds == max(ROUNDS):
        found["round_covariance"] = {}
        for patch in dict.fromkeys(str(p) for p, _, _ in labels):
            columns = [
                [
                    k
                    for k, (p, r, name) in enumerate(labels)
                    if str(p) == patch and r == n and " " not in name
                ]
                for n in range(1, setting.rounds)
            ]
            means = np.stack([detectors[:, ks].mean(axis=1) for ks in columns], axis=1)
            found["round_covariance"][patch] = np.atleast_2d(np.cov(means, rowvar=False)).tolist()
    return found


def _memory_shots(shots: list[dict[str, np.ndarray]]) -> int:
    i = next(i for i, s in enumerate(SETTINGS) if s.kind == "memory")
    return next(iter(shots[i].values())).shape[0]


def _idle_dephasing(shots: list, circuits: list, calibration: Calibration, dt: float) -> dict:
    """Each data qubit's share of the dephasing, from its <X> and <Y> over the idle test.

    A frequency offset turns X into Y, so sqrt(<X>^2 + <Y>^2) falls from dephasing
    alone; its angle gives the offset, as a phase per round.
    """
    import numpy as np

    idle = [i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "X"]
    in_y = {s.rounds: i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "Y"}
    rounds = np.array([SETTINGS[i].rounds for i in idle])
    xs = [x_values(shots[i]) for i in idle]
    ys = [x_values(shots[in_y[n]]) for n in rounds]
    ends = [[_exact_x(circuits[i], calibration, dt, f) for i in idle] for f in ENDS]
    found: dict[str, dict[str, list[float]]] = {
        key: {} for key in ("per_qubit", "uncertainty", "x_only", "phase_per_round")
    }
    for d in xs[0]:
        for found_d in found.values():
            found_d[d] = []
        for q in range(int(d) ** 2):
            x = np.array([v[d][q] for v in xs])
            y = np.array([v[d][q] for v in ys])
            count_x = np.array([len(shots[i][f"d{d}"]) for i in idle])
            count_y = np.array([len(shots[in_y[n]][f"d{d}"]) for n in rounds])
            # Shot noise adds about (1 - v^2) / shots to each v^2; take it back out.
            r2 = x**2 + y**2 - (1 - x**2) / count_x - (1 - y**2) / count_y
            r = np.sqrt(np.maximum(r2, 0))
            log_x0, log_x1 = (np.log([e[d][q] for e in end]) for end in ends)
            share, sigma = _fit_share(r, log_x0, log_x1, (count_x + count_y) / 2)
            found["per_qubit"][d].append(share)
            found["uncertainty"][d].append(sigma)
            found["x_only"][d].append(_fit_share(x, log_x0, log_x1, count_x)[0])
            found["phase_per_round"][d].append(_phase(rounds, x, y, r))
    pairs = [
        (s, u)
        for d in found["per_qubit"]
        for s, u in zip(found["per_qubit"][d], found["uncertainty"][d])
        if np.isfinite(s) and np.isfinite(u)
    ]
    every = [s for s, _ in pairs]
    x_only = [s for d in found["x_only"] for s in found["x_only"][d] if np.isfinite(s)]
    return {
        **found,
        "uniformity": _uniformity(np.array(every), np.array([u for _, u in pairs])),
        "median": float(np.median(every)) if every else float("nan"),
        # Standard error of a median: about 1.25 that of a mean.
        "median_uncertainty": float(1.25 * np.std(every) / sqrt(len(every)))
        if len(every) > 1
        else float("nan"),
        "x_only_median": float(np.median(x_only)) if x_only else float("nan"),
    }


def _phase(rounds: np.ndarray, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> float:
    """A qubit's phase per round, from the angle of (<X>, <Y>) over the idle lengths.

    Unwrapped one length at a time against the slope so far, since the lengths
    double; NaN if the angles don't then lie on a line to within pi/2.
    """
    import numpy as np

    keep = r > 0.05  # noise in the angle goes as 1 / (r sqrt(shots))
    n, angle, weight = rounds[keep], np.arctan2(y, x)[keep], r[keep]
    if len(n) < 2:
        return float("nan")
    slope = (np.angle(np.exp(1j * (angle[1] - angle[0])))) / (n[1] - n[0])
    for k in range(1, len(n)):
        guess = angle[k - 1] + slope * (n[k] - n[k - 1])
        angle[k] += 2 * np.pi * np.round((guess - angle[k]) / (2 * np.pi))
        slope, start = np.polyfit(n[: k + 1], angle[: k + 1], 1, w=weight[: k + 1])
    if np.abs(angle - (start + slope * n)).max() > np.pi / 2:
        return float("nan")
    return float(slope)


def _uniformity(shares: np.ndarray, sigmas: np.ndarray) -> dict:
    """How well one f fits every data qubit: chi^2 about their weighted mean.

    A p-value near 0 says the qubits differ by more than the idle test's noise.
    """
    import numpy as np
    from scipy.stats import chi2

    if len(shares) < 2:
        return {}
    weight = 1 / sigmas**2
    middle = float((weight * shares).sum() / weight.sum())
    stat = float((((shares - middle) / sigmas) ** 2).sum())
    low, high = np.percentile(shares, [25, 75])
    return {
        "qubits": len(shares),
        "weighted_mean": middle,
        "quartiles": [float(low), float(high)],
        "chi2_per_dof": stat / (len(shares) - 1),
        "p_value": float(chi2.sf(stat, len(shares) - 1)),
    }


def _exact_x(
    circuit: QuantumCircuit, calibration: Calibration, dt: float, f: float
) -> dict[str, list[float]]:
    """<X> of each data qubit, exact under the simulator's noise.

    The product of 1 - 2p over the errors that flip its readout.
    """
    import stim

    model, records = noisy_scheduled(circuit, calibration, dt, decoupled_dephasing=f)
    present = [d for d in DISTANCES if (f"d{d}", 0) in records]
    data = [(d, q) for d in present for q in range(d * d)]
    for k, (d, q) in enumerate(data):
        last = stim.target_rec(records[f"d{d}", q] - model.num_measurements)
        model.append("OBSERVABLE_INCLUDE", [last], k)
    x = dict.fromkeys(data, 1.0)
    for error in model.detector_error_model().flattened():
        if error.type == "error":
            for target in error.targets_copy():
                if target.is_logical_observable_id():
                    x[data[target.val]] *= 1 - 2 * error.args_copy()[0]
    return {str(d): [x[d, q] for q in range(d * d)] for d in present}


def _fit_share(
    seen: np.ndarray, log_x0: np.ndarray, log_x1: np.ndarray, shots: np.ndarray
) -> tuple[float, float]:
    """The share f, and its standard error, from log seen = a + log_x0 - f (log_x0 - log_x1).

    A weighted line through the idle lengths where seen is still above 0.05; the
    offset a takes up readout error the simulator gets wrong. With one length, a = 0.
    NaN where the test can't pin f to LOOSEST_F, as when T2 is close to 2 T1 and
    IBM's T2 adds next to nothing to the decay.
    """
    import numpy as np

    keep = seen > 0.05
    extra = (log_x0 - np.log(np.where(keep, seen, 1)))[keep]  # decay beyond T1 alone
    full = (log_x0 - log_x1)[keep]  # what IBM's T2 adds
    weight = (shots * seen**2 / (1 - np.minimum(seen, 0.999) ** 2))[keep]  # 1 / var(log seen)
    nan = (float("nan"), float("nan"))
    if not keep.any():
        return nan
    if keep.sum() == 1:
        if full[0] <= 0:
            return nan
        share, sigma = extra[0] / full[0], 1 / (full[0] * sqrt(weight[0]))
    else:
        w, wx, wy = weight.sum(), (weight * full).sum(), (weight * extra).sum()
        spread = w * (weight * full**2).sum() - wx**2
        if spread <= 0:
            return nan
        share = (w * (weight * full * extra).sum() - wx * wy) / spread
        sigma = sqrt(w / spread)
    return (float(share), float(sigma)) if sigma <= LOOSEST_F else nan


def _dephasing_table(
    measured: dict, chips: dict[str, list[int]], calibration: Calibration, median: float
) -> list[dict]:
    """Each data qubit's own share, highest first; the median where the idle test can't tell."""
    import numpy as np

    rows = [
        {
            "patch": int(d),
            "data": f"Q{q + 1}",
            "qubit": chips[d][q],  # data qubit q is the flagged circuit's qubit q
            "dephasing": max(share, 0.0) if np.isfinite(share) else median,
            "uncertainty": measured["uncertainty"][d][q],
            "measured": bool(np.isfinite(share)),
            "phase_per_round": measured["phase_per_round"][d][q],
            "t2_us": calibration.t2_us[chips[d][q]],
        }
        for d, shares in measured["per_qubit"].items()
        for q, share in enumerate(shares)
    ]
    return sorted(rows, key=lambda row: -row["dephasing"])


def _by_round(labels: list, rates: list) -> dict[str, list[float]]:
    """Mean firing rate of each patch's stabilizer detectors, round by round.

    A rise over the rounds that the prediction lacks points to leakage or heating.
    """
    sums: dict[str, dict[int, list[float]]] = {}
    for (patch, r, name), rate in zip(labels, rates):
        if " " not in name:  # stabilizers, not flags or relays
            sums.setdefault(str(patch), {}).setdefault(r, []).append(rate)
    return {patch: [sum(v) / len(v) for _, v in sorted(by.items())] for patch, by in sums.items()}


def _worst_detectors(
    observed: list, expected: list, shots: int, after: list | None = None, many: int = 10
) -> dict:
    """The detectors that fire more often than predicted beyond chance, most excessive first.

    With N detectors in all, chance alone reaches about sqrt(2 ln N) standard errors.
    A detector counts only if it is beyond that against the prediction from the
    calibration before the run and, when there is one, after it too, so that
    IBM's recalibrations in between don't fill the list; they are ranked by how
    many times more often than predicted they fire.
    """
    import numpy as np

    memory = [i for i, s in enumerate(SETTINGS) if s.kind == "memory"]
    count = sum(len(observed[i]["detector_rates"]) for i in memory)
    bar = sqrt(2 * log(max(count, 2)))
    rows = []
    for i in memory:
        got = np.array(observed[i]["detector_rates"])
        wants = [np.array(sim[i]["detector_rates"]) for sim in (expected, after) if sim]
        noise = 1 / shots + 1 / SIMULATED_SHOTS  # the chip's shots and the simulator's
        excess = np.min(
            [(got - want) / np.sqrt(np.maximum(want * (1 - want), 1e-6) * noise) for want in wants],
            axis=0,
        )
        ratio = got / np.maximum(wants[0], 1e-6)
        for k in np.flatnonzero(excess > bar):
            rows.append(
                {
                    "setting": str(SETTINGS[i]),
                    "detector": observed[i]["detectors"][k],
                    "observed": float(got[k]),
                    "predicted": float(wants[0][k]),
                    "ratio": float(ratio[k]),
                    "excess": float(excess[k]),
                }
            )
    rows.sort(key=lambda row: -row["ratio"])
    return {
        "chance_excess": bar,
        "detectors_beyond_chance": len(rows),
        "worst_detectors": rows[:many],
    }


def _fits(settings: list[dict], shots: int, patches: list[str]) -> dict:
    """Error per round per basis and patch, and Λ when the job has both patches.

    Logical k of each row is patches[k].
    """
    fits: dict = {}
    for basis in "XZ":
        rows = [r for s, r in zip(SETTINGS, settings) if s.kind == "memory" and s.basis == basis]
        per_round = {}
        for k, d in enumerate(patches):
            fit = fit_per_round(ROUNDS, [r["logical_error"][k] for r in rows], shots)
            per_round[d] = [fit.per_round, fit.uncertainty]
        fits[basis] = {
            "per_round": per_round,
            **ratio(per_round),
            "from_round_2": _window(rows, shots, patches),
        }
    return {"fits": fits}


def ratio(per_round: dict[str, list[float]]) -> dict:
    """Λ = ε₃ / ε₅, its standard error, and the decision: one-sided 5% each way.

    Empty unless both patches are there.
    """
    if not {"3", "5"} <= set(per_round):
        return {}
    (e3, u3), (e5, u5) = per_round["3"], per_round["5"]
    lam = e3 / e5
    sigma = lam * sqrt((u3 / e3) ** 2 + (u5 / e5) ** 2)  # one standard error
    return {"lambda": lam, "lambda_uncertainty": sigma, "verdict": verdict(lam, sigma)}


def verdict(lam: float, sigma: float) -> str:
    if lam - 1.645 * sigma > 1:
        return "d = 5 better"
    if lam + 1.645 * sigma < 1:
        return "d = 5 worse"
    return "undecided"


def _window(rows: list[dict], shots: int, patches: list[str], first: int = 2) -> dict:
    """ε and Λ fitted from round `first` on: the first round, compared with the
    prepared state, differs from the rest. Exploratory, nothing is decided on it.
    """
    keep = [k for k, n in enumerate(ROUNDS) if n >= first]
    if len(keep) < 2:
        return {}
    per_round = {}
    for k, d in enumerate(patches):
        fit = fit_per_round(
            [ROUNDS[j] for j in keep], [rows[j]["logical_error"][k] for j in keep], shots
        )
        per_round[d] = [fit.per_round, fit.uncertainty]
    found = {"per_round": per_round, **ratio(per_round)}
    found.pop("verdict", None)
    return found


def _quantities(fits: dict, basis: str) -> dict[str, tuple[float, float]]:
    """What the agreement rule compares: (value, standard error) of Λ in X, ε₃ and ε₅."""
    found = {f"eps{d}": tuple(value) for d, value in fits[basis]["per_round"].items()}
    if basis == "X" and "lambda" in fits[basis]:
        found["lambda"] = (fits[basis]["lambda"], fits[basis]["lambda_uncertainty"])
    return found


def _filled(
    later: Calibration, prepared: Calibration, chips: dict[str, list[int]]
) -> tuple[Calibration, list[int]]:
    """The calibration after the job, with the prepared values for any of the run's
    qubits it leaves out or marks broken, so the after-prediction neither fails nor
    quietly uses a perfect part. Returns the qubits filled in.
    """
    from dataclasses import replace

    from ..patches.placement import broken_qubit

    used = sorted({q for chip in chips.values() for q in chip})
    gaps = [q for q in used if broken_qubit(q, later)]
    if not gaps:
        return later, []

    def fill(values: dict | None, before: dict | None) -> dict | None:
        if values is None or before is None:
            return values
        return {**values, **{q: before[q] for q in gaps if q in before}}

    return replace(
        later,
        readout=fill(later.readout, prepared.readout),
        t1_us=fill(later.t1_us, prepared.t1_us),
        t2_us=fill(later.t2_us, prepared.t2_us),
        single_qubit=fill(later.single_qubit, prepared.single_qubit),
        broken_qubits=tuple(q for q in later.broken_qubits if q not in gaps),
    ), gaps


def _slopes(frozen: dict, expected: dict, f: float) -> dict[str, float]:
    """How fast each compared quantity moves with f at the measured f.

    From the curve through the frozen predictions at f = 0 and 1 and the
    prediction at the measured f (a line if f is near an end).
    """
    import numpy as np

    known = [(0.0, frozen[str(ENDS[0])]["fits"]), (1.0, frozen[str(ENDS[1])]["fits"])]
    if 0.05 < f < 0.95:
        known.insert(1, (f, expected["fits"]))
    at = np.array([point for point, _ in known])
    found = {}
    for basis in "XZ":
        if all(basis in fits for _, fits in known):
            values = [_quantities(fits, basis) for _, fits in known]
            for name in values[0]:
                curve = np.polyfit(at, [v[name][0] for v in values], len(known) - 1)
                found[f"{basis} {name}"] = float(np.polyval(np.polyder(curve), f))
    return found


def _agreement(
    observed: dict,
    predicted: dict,
    after: dict | None,
    slopes: dict[str, float] | None = None,
    sigma_f: float = 0.0,
) -> dict:
    """Does the simulation match the chip: |obs - pred| < 2 sqrt(Var_obs + Var_pred)?

    Var_pred adds the simulator's shot noise; the measured f's uncertainty times the
    quantity's slope in f; and, for the calibration, Δ²/2, Δ being the gap between the
    predictions from IBM's calibrations before and after the job (zero if IBM didn't
    recalibrate in between). X memory matches if its Λ, ε₃ and ε₅ all do; Z memory is
    judged on its own.
    """
    found: dict = {}
    for basis in "XZ":
        if basis not in observed:
            continue
        seen, want = _quantities(observed, basis), _quantities(predicted, basis)
        later = _quantities(after, basis) if after else {}
        for name, (value, sigma) in seen.items():
            guess, noise = want[name]
            gap = abs(guess - later[name][0]) if later else 0.0
            from_f = (slopes or {}).get(f"{basis} {name}", 0.0) * sigma_f
            bound = 2 * sqrt(sigma**2 + noise**2 + from_f**2 + gap**2 / 2)
            found[f"{basis} {name}"] = {
                "observed": value,
                "predicted": guess,
                "observed_uncertainty": sigma,
                "simulator_noise": noise,
                "from_f": from_f,
                "calibration_gap": gap,
                "bound": bound,
                "agrees": abs(value - guess) < bound,
            }
    verdicts = {
        basis: all(row["agrees"] for name, row in found.items() if name.startswith(f"{basis} "))
        for basis in "XZ"
        if any(name.startswith(f"{basis} ") for name in found)
    }
    return {"tests": found, "matches": verdicts}


def _dephasing_from_memory(
    observed: dict, frozen: dict, expected: dict, f: float, idle: dict
) -> dict:
    """f read off the X-memory runs, by the curve through the predictions at f = 0, f and 1.

    The frozen predictions give the ends and the prediction at the idle test's f
    the middle; ε₃ and the detector rates bend with f, so a straight line between
    the ends would read f high. From ε₃, and from the stabilizer detectors' rates;
    each agrees with the idle test if it is within twice their combined
    uncertainty, calibration scatter included. Exploratory: nothing is decided on it.
    """
    import numpy as np

    low, high = (frozen[str(end)] for end in ENDS)
    known = [(0.0, low), (1.0, high)]
    if 0.05 < f < 0.95:  # too near an end, the curve is no better than the line
        known.insert(1, (f, expected))
    at = np.array([point for point, _ in known])
    degree = len(known) - 1
    grid = np.linspace(-0.5, 2.0, 2501)

    small = min(observed["fits"]["X"]["per_round"], key=int)  # d=3 when the job has it
    eps, sigma = observed["fits"]["X"]["per_round"][small]
    curve = np.polyfit(at, [p["fits"]["X"]["per_round"][small][0] for _, p in known], degree)
    from_eps = float(grid[np.argmin(np.abs(np.polyval(curve, grid) - eps))])
    sigma_eps = float(sigma / abs(np.polyval(np.polyder(curve), from_eps)))

    rows: list[list[list[float]]] = [[] for _ in known]
    got: list[float] = []
    for i, setting in enumerate(SETTINGS):
        if setting.kind == "memory" and setting.basis == "X":
            labels = observed["settings"][i]["detectors"]
            stabilizers = [k for k, (_, _, name) in enumerate(labels) if " " not in name]
            got += [observed["settings"][i]["detector_rates"][k] for k in stabilizers]
            for into, (_, p) in zip(rows, known):
                into.append([p["settings"][i]["detector_rates"][k] for k in stabilizers])
    rates = np.array([np.concatenate(r) for r in rows])  # points x detectors
    curves = np.polyfit(at, rates, degree)  # coefficients x detectors
    misfit = [((np.polyval(curves, g) - np.array(got)) ** 2).sum() for g in grid]
    from_rates = float(grid[int(np.argmin(misfit))])
    f_idle, sigma_idle = idle["median"], idle["median_uncertainty"]

    def agrees(value: float, spread: float) -> bool:
        return bool(
            abs(value - f_idle) < 2 * sqrt(spread**2 + sigma_idle**2 + CALIBRATION_SCATTER_F**2)
        )

    return {
        "from_eps": small,
        "from_eps3": from_eps,
        "from_eps3_uncertainty": sigma_eps,
        "from_detector_rates": from_rates,
        "idle_test": f_idle,
        "idle_test_uncertainty": sigma_idle,
        "calibration_scatter": CALIBRATION_SCATTER_F,
        "eps3_agrees": agrees(from_eps, sigma_eps),
        "detector_rates_agree": agrees(from_rates, 0.0),
    }


def _drift(halves: list[dict]) -> dict:
    """ε per round in each half of the job's shots, and the second half's change in standard errors."""
    found = {}
    for basis in "XZ":
        for d in halves[0][basis]["per_round"]:
            (early, u1), (late, u2) = (fits[basis]["per_round"][d] for fits in halves[:2])
            change = (late - early) / sqrt(u1**2 + u2**2)
            found[f"{basis} eps{d}"] = {
                "early": early,
                "late": late,
                "change": change,
                "drift": abs(change) > DRIFT_BAR,
            }
    return found


def _leakage(observed: list, expected: list, shots: int) -> dict:
    """In the longest memory runs, does the detection rate rise over the rounds more than predicted?

    The least-squares slope of (observed - predicted) mean stabilizer rate against
    the round, over the rounds that compare two stabilizer readouts (not the first
    or the last), in standard errors from the covariance of the per-round means
    across shots, the chip's and the simulator's; above LEAKAGE_BAR counts as
    leakage or heating.
    """
    import numpy as np

    longest = max(ROUNDS)
    found: dict = {}
    if longest < 3:  # a slope needs two rounds between the first and the last
        return found
    rounds = np.arange(1, longest)
    c = (rounds - rounds.mean()) / ((rounds - rounds.mean()) ** 2).sum()
    for i, setting in enumerate(SETTINGS):
        if setting.kind != "memory" or setting.rounds != longest:
            continue
        for patch, got in observed[i]["by_round"].items():
            gap = np.array(got[1:longest]) - np.array(expected[i]["by_round"][patch][1:longest])
            noise = (
                np.array(observed[i]["round_covariance"][patch]) / shots
                + np.array(expected[i]["round_covariance"][patch]) / SIMULATED_SHOTS
            )
            slope = float(c @ gap)
            z = slope / sqrt(float(c @ noise @ c))
            found[f"{setting.basis} d={patch}"] = {
                "slope_per_round": slope,
                "standard_errors": z,
                "rises": z > LEAKAGE_BAR,
            }
    return found


def _is_local(backend: Any) -> bool:
    return type(backend).__module__.startswith("qiskit_aer")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n")
