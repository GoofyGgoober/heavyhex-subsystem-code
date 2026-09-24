# QPU usage

- Do not submit jobs to or otherwise consume real QPU resources without the user's direct, explicit permission for the specific run or clearly defined batch of runs.
- This applies to both paid usage and free-quota usage, including smoke tests and diagnostic jobs.
- Saved credentials, account access, or requests to develop, compile, or test hardware support do not constitute permission to use a QPU.
- Local simulation, local compilation, and read-only account/backend queries that consume no QPU time are allowed without QPU-use permission.

# Calibration

- IBM recalibrates Fez about once a day, and where the patches go depends on it. Run `heavyhex calibrate` (read-only) before any QPU run, then redraw with `python docs/figures/draw_blueprint.py`. Hardware code must get its qubits from `fez_qubits()`, which pulls a fresh calibration if the last one isn't from today and refuses broken parts.

# Subagents

- All subagents must use Grok 4.6 High Fast (`cursor-grok-4.6-high-fast`).
- When launching a Task or other subagent, set `model` to `cursor-grok-4.6-high-fast`. Do not inherit another model and do not pick a different slug unless the user explicitly overrides this for that run.
