# Proposal

## Why

The agents that built the model library and the numerics audit in `organize-model-library` found defects in the core:
- bugs that give wrong results with no error;
- bugs that raise a raw error from deep inside the code;
- performance defects that make ordinary runs tens to thousands of times slower than they need to be.

This change fixes every defect it could reproduce against the current code. Each fix comes with a regression test that
fails before the fix. Missing features found in the same log are left to the changes that own them.

## What Changes

Boundaries and steppers:
- Boundary laws apply after every internal solver step, not once per frame. Today a particle with a long frame leaves
  a reflecting box and stays outside.
- `IMEX2` restarts its two-step history when a call does not continue the state and step of the previous call. Today a
  second call from the same state gives a different result.
- `SSPRK` limits its step by the diffusive stability limit of Laplacian rates on finite-volume fields, as well as by
  the CFL condition. Today viscous Burgers grows past its initial maximum.
- `ScipyODE` keeps the probe of its first step inside the recorded history of delays shorter than that step. Today it
  fails with a raw `IndexError` for `τ = 1e-4`.

Conditions and equations:
- `where` with equal branches simplifies to that branch, so the derivative of `where(c, a, b)` with respect to a
  symbol that neither branch contains is zero. Today equations with `where` or drivers on the right-hand side fail with
  "not linear in the highest derivative".
- The NumPy backend evaluates `where` without warnings from the branch it discards. Today `where(x > 0, 1/x, 0)`
  emits a `RuntimeWarning` at `x = 0`.

Inputs to laws and components:
- **BREAKING:** a law whose value is a NumPy array raises a named error, and a NumPy scalar becomes a Sage number. Today
  `np.float64(2) * v * v` silently gives the element-wise product of a vector instead of its dot product.
- `Jump` increments that are not numbers raise a named error that names the jump, the state, and the increment. Today
  a parameter symbol raises a raw `TypeError` from Sage.
- **BREAKING:** an `Observable` whose expression cannot be pickled raises a named error when it is created. Today the
  system fails to pickle later, for example in a process pool.
- A one-member group reads per-member values and locations given as one-entry sequences, as larger groups do. Today
  `location=[cell]` raises `unhashable type: 'list'` and `n=[3.0]` gives the agent the list.
- An agent type whose state name clashes with an attribute Mesa reads raises a named error when it is defined. Today
  a state `position` breaks every snapshot with a raw `TypeError`.

Agent runs:
- The Mesa stepper advances Mesa's clock, so `model.time` in rules follows the run. Today it stays at 0.

Runs:
- **BREAKING:** parameter values set with `update_parameters` stay in force for later runs of the same simulation. A
  new `Simulation.extend(until)` continues a finished run from its last frame. Today a second `run` resets the
  parameters silently and always starts again at time 0.
- The default frame duration of continuous forms stays 0.01 for short runs and grows in 1, 2, 5 steps so a run has at
  most 1000 frames. Today `run(until=1000)` creates 100001 frames.
- `ScipyODE` keeps one integrator across frames and restarts it only when the parameters, the state, or a breakpoint
  forces it. Today it restarts `solve_ivp` every frame.

Found while applying:
- System observables read the time as `s.t`. Today `s.t` raises "No symbol named 't'".
- A delay that is an expression of parameters, such as `delayed(x, 2 * s.tau)`, takes its smallest value from the
  parameter values. Today it raises "smallest value is unknown".
- Periodic breakpoints are found without an end time. Today a persistent integrator with `pulse_train` overflows.
- Equations whose right sides hold Python floats, such as the height of `step(t, 1)`, solve for their highest
  derivatives. Today Sage's check of exact rings fails on the float.

Compiled code:
- The Numba backend compiles the kernels that event loops call per event as serial code. Today `parallel=True` makes
  the compiled Gillespie loop about 20 times slower per run, and about 1000 times slower per event.
- Compiled kernels read their variables through slices and shapes computed once at compile time. Today `np.prod` in
  `Entry.size`, `Entry.columns`, and `Entry.view` takes about 40% of a NumPy drift evaluation.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: per-step boundaries, conditions and their derivatives, equations with conditions, jump
  increments, observable pickling, and one-member groups.
- `symbolic-analysis`: NumPy values in laws.
- `agent-based-modeling`: Mesa's clock and reserved agent state names.
- `simulation`: parameters that persist across runs, extending runs, default frame durations, and integrators that
  persist across frames.
- `hybrid-dynamics`: delays shorter than the first step.
- `pde-solvers`: the `IMEX2` restart and diffusion-limited `SSPRK` steps.
- `model-compilation`: serial event loops and compiled layouts.

## Impact

- `skcomplex.core.model`: `_functions.py` (`where`), `_laws.py` (value checks of laws and `Jump`),
  `_observables.py`, and `_components.py` (reserved agent state names).
- `skcomplex.core.compile`: `_layout.py`, `_kernels.py`, `_printing.py` (`where` in NumPy),
  `_numba_backend.py`, `_fields.py` (finite-volume stability limits), and `_compiler.py` (equation linearity).
- `skcomplex.core.solvers`: `_steppers.py`, `_stiff.py`, `_hybrid.py`, `_agents.py`, and `_cosimulation.py`.
- `skcomplex.core.runtime`: `_simulations.py`.
- Tests, docstring examples, and the user guide that use lambda observables move to module-level functions or
  symbolic expressions. The documentation test runs each document as a registered module, so its functions pickle.
  The changelog entries for the three breaking changes are left to the release, since the parallel apply runs
  forbid edits of the changelog.
- No new dependency.
