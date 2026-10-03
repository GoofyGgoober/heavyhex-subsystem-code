#!/usr/bin/env bash
# One run day of the Fez experiment, unattended, as agreed on 3-4 October 2026:
# wait for IBM's evening calibration, prepare the day, put the prediction on record
# (commit and push), send the day's jobs in order, then commit the shots, analyze,
# combine, commit and push. Windows Task Scheduler starts it at 20:00 for days 2-5.
#
# It sends nothing without the author's approval file, stops after the registered
# five run days, never starts a second day on one date or too late to finish before
# midnight, stops on any failure, and resumes where it stopped when run again.
# The QPU caps (50 s a job, 300 s and 20 jobs in all) are enforced by heavyhex itself.
#
#   scripts/daily_run.sh            the real run: sends the day's jobs
#   scripts/daily_run.sh --dry-run  offline calibration, rehearsed shots: no QPU, no commits
set -euo pipefail

cd "$(dirname "$0")/.."
hh=.venv/bin/heavyhex
py=.venv/bin/python
backend=ibm_fez
run_days=5
cal_after=${CAL_AFTER:-20:00}        # IBM's evening calibration lands at 20:05 local time
cal_deadline=${CAL_DEADLINE:-21:30}  # past this, use the latest calibration there is
latest_start=${LATEST_START:-22:30}  # a day is prepared and sent on one calendar date
approval=${HEAVYHEX_APPROVAL:-$HOME/.config/heavyhex/qpu-approval.json}
today=$(date +%F)
dry=false
[ "${1:-}" = --dry-run ] && dry=true
mode=$($dry && echo dry || echo real)

mkdir -p "$HOME/heavyhex-logs"
exec > >(tee -a "$HOME/heavyhex-logs/$today-$mode.log") 2>&1
say() { echo "$(date '+%F %T')  $*"; }
trap 'say "STOPPED: line $LINENO failed (exit $?)"' ERR
say "=== $mode run, $today ==="

offline() { if $dry; then echo --offline; fi; }
rehearsal() { if $dry; then echo --rehearsal; fi; }
quiet() { grep -v "WARNING\|Warning\|weak:\|chips = {" || true; }

# Today's job folders in the order they are sent: (rep, job) from run.json, not by name.
folders() {
  $py - "$today" "$mode" <<'PY'
import json, pathlib, sys
day, dry = sys.argv[1], sys.argv[2] == "dry"
found = []
for f in pathlib.Path("runs").glob(f"ibm_fez-{day}-r*"):
    if f.name.endswith("-offline") == dry and (f / "run.json").exists():
        rep = json.loads((f / "run.json").read_text())["rep"]
        found.append((rep["rep"], rep["job"], f.as_posix()))
sys.stdout.write("".join(f"{path}\n" for *_, path in sorted(found)))  # nothing at all if none
PY
}

# Dates that have the chip's shots, i.e. run days that happened.
run_dates() {
  local d
  for d in runs/"$backend"-*-r*/; do
    case $d in *-offline/) continue ;; esac
    if [ -f "$d/shots.npz" ]; then basename "$d" | cut -d- -f3-5; fi
  done | sort -u
}

# IBM's latest calibration of the backend, read-only.
calibrated() {
  $py -c "from qiskit_ibm_runtime import QiskitRuntimeService as S
print(S().backend('$backend').properties().last_update_date.astimezone().isoformat(timespec='seconds'))" 2>/dev/null
}

wait_for_calibration() {
  local want deadline at
  want=$(date -d "$today $cal_after" +%s)
  deadline=$(date -d "$today $cal_deadline" +%s)
  while :; do
    at=$(calibrated) || at=""
    if [ -n "$at" ] && [ "$(date -d "$at" +%s)" -ge "$want" ]; then
      say "IBM calibrated $backend at $at"
      return
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      say "no calibration after $cal_after by $cal_deadline; using IBM's latest (${at:-unreachable})"
      return
    fi
    say "waiting for IBM's calibration after $cal_after (latest ${at:-unreachable})"
    sleep 120
  done
}

# Commit the run's record and figures (never the paper or anything else), then push.
commit() {
  if $dry; then say "dry run, not committed: $1"; return; fi
  git add -A runs docs/figures ':!runs/*-offline'
  if git diff --cached --quiet; then say "nothing new to commit: $1"; return; fi
  printf '%s\n\n%s\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n' "$1" "$2" | git commit -q -F -
  git -c credential.helper= -c credential.helper='!gh auth git-credential' push -q origin HEAD
  say "committed and pushed: $1"
}

qpu_used() { $py -c "from heavyhex.experiment import run; print(f'{run.qpu_committed():.0f}')" 2>/dev/null; }
field() { $py -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
day_name() { date -d "$today" '+%-d %B'; }

# --- guards (the real run only) ---------------------------------------------
if ! $dry; then
  git pull -q --ff-only
  if [ ! -f "$approval" ]; then say "no QPU approval at $approval; nothing is sent"; exit 1; fi
  first=$(field "$approval" first_day)
  last=$(field "$approval" last_day)
  approved=$(field "$approval" run_days)
  if [[ $today < $first || $today > $last ]]; then
    say "the approval covers $first to $last, not $today; nothing is sent"
    exit 0
  fi
  done_days=$(run_dates | grep -cvx "$today" || true)
  used=$(run_dates | awk -v s="$first" -v t="$today" '$0 >= s && $0 != t' | wc -l)
  if [ "$done_days" -ge "$run_days" ]; then say "all $run_days run days are done"; exit 0; fi
  if [ "$used" -ge "$approved" ]; then say "the $approved approved days are used; nothing more is sent"; exit 0; fi
  say "run day $((done_days + 1)) of $run_days; QPU used so far $(qpu_used) s of 300"
fi

# --- prepare (once a day, again if IBM recalibrates before the first job) ----
prepare_today() {
  $hh calibrate $(offline) 2>&1 | quiet
  set +e
  out=$($hh experiment prepare $(offline) 2>&1)
  rc=$?
  set -e
  quiet <<<"$out"
  if [ $rc -ne 0 ]; then
    if grep -q "no clean place for the d=5 patch" <<<"$out"; then
      commit "Skip $(day_name): no clean place for the d=5 patch" \
        "IBM's calibration left no clean place for the d=5 patch on $backend, so by the registered rules the day is skipped."
      exit 0
    fi
    say "prepare failed (exit $rc)"
    exit $rc
  fi
  if $dry; then
    $py docs/figures/figure1.py --out /tmp/figure1-dry-run.png 2>&1 | quiet | tail -1
  else
    $py docs/figures/draw_blueprint.py 2>&1 | quiet | tail -1
    $py docs/figures/figure1.py --out docs/figures/figure1.png 2>&1 | quiet | tail -1
  fi
  mapfile -t jobs < <(folders)
  at=$(field "${jobs[0]}/calibration.json" calibrated)
  commit "Prepare $(day_name)'s ${#jobs[@]} jobs from IBM's $(date -u -d "$at" '+%H:%M') UTC calibration" \
    "The day's folders, predictions and figures, on record before any job is sent."
}

mapfile -t jobs < <(folders)
if [ ${#jobs[@]} -eq 0 ]; then
  if [ "$(date +%s)" -ge "$(date -d "$today $latest_start" +%s)" ]; then
    say "too late to prepare and send before midnight; the day is not started"
    exit 0
  fi
  if $dry; then say "IBM's latest calibration (read-only check): $(calibrated || echo unreachable)"
  else wait_for_calibration; fi
  prepare_today
fi

# Before the first job only: if IBM's numbers changed since the prepare, prepare again
# (as on day 1). Once a job has gone, the day finishes as prepared.
if ! $dry; then
  for again in 1 2 3; do
    if compgen -G "runs/$backend-$today-r*/job.json" >/dev/null; then break; fi  # a job has gone
    prepared=$(field "${jobs[0]}/calibration.json" calibrated)
    now=$(calibrated) || now=""
    if [ -z "$now" ] || [ "$(date -d "$now" +%s)" -eq "$(date -d "$prepared" +%s)" ]; then break; fi
    if [ "$again" -eq 3 ]; then say "IBM keeps recalibrating ($prepared -> $now); stopping"; exit 1; fi
    say "IBM recalibrated after the prepare ($prepared -> $now); preparing again"
    rm -rf "${jobs[@]}"
    prepare_today
  done
fi
say "jobs in order: ${jobs[*]}"

# --- send -------------------------------------------------------------------
sent=0
for f in "${jobs[@]}"; do
  if $dry; then
    if [ ! -f "$f/rehearsal.npz" ]; then $hh experiment rehearse --run "$f" 2>&1 | quiet | tail -3; fi
    continue
  fi
  if [ -f "$f/shots.npz" ]; then say "$f already has its shots"; continue; fi
  if [ -f "$f/job.json" ]; then say "$f had a job but no shots; it is never resubmitted"; continue; fi
  echo yes | $hh experiment submit --run "$f" 2>&1 | quiet | tail -6
  sent=$((sent + 1))
done
if [ $sent -gt 0 ]; then
  commit "Shots of $(day_name)'s jobs on $backend" \
    "Each job's record, shots and IBM's calibration as it finished. QPU used in all: $(qpu_used) s of 300."
fi

# --- analyze, combine --------------------------------------------------------
for f in "${jobs[@]}"; do
  if $dry; then result=rehearsal_analysis.json; shots=rehearsal.npz
  else result=analysis.json; shots=shots.npz; fi
  if [ -f "$f/$result" ]; then continue; fi
  if [ ! -f "$f/$shots" ]; then say "$f has no shots to analyze"; continue; fi
  $hh experiment analyze --run "$f" $(rehearsal) 2>&1 | quiet | tail -4
done
$hh experiment combine $(rehearsal) 2>&1 | quiet | tail -12
commit "Analyze $(day_name)'s jobs and combine the reps" "Each job's analysis and the reps pooled over the days so far."

if $dry; then
  rm -rf runs/*-offline
  git checkout -q -- docs/figures 2>/dev/null || true
  say "dry run cleaned up; working tree changes left: $(git status --short | wc -l)"
fi
say "=== done ==="
