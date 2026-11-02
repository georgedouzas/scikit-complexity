# Proposal

## Why

Applying `add-systems-biology-models` and `add-statistical-physics-models` found a second round of core defects:
- inputs and laws that crash compilation with a raw error;
- choices that silently ignore what the user or the element type declared;
- costs that grow faster than the model.

This change fixes every item of that log that reproduces on the current code. Each fix comes with a regression test
that fails before the fix. It does not repeat the fixes of `fix-core-defects`.

## What Changes

Drivers and laws:
- `series` with any number of points compiles. Today 200 points raise `SyntaxError: too many nested parentheses`, and
  1000 points raise `RecursionError` while the NumPy code is printed.
- An interaction law may divide by a reduction, as in `s.cells.x / count(over=s.cells)`. Today compiling it fails with
  Sage's `ValueError: power::eval(): division by zero`.
- The member expression of a group observable may hold reductions, with or without weights, such as
  `m.x * total(m.x, weights=W)`. Today a plain reduction raises "The function total has no NumPy equivalent", and a
  weighted one raises a named error about an unknown symbol.

Roles and jumps:
- A plain role binds a group member such as `sites[3]`, as a variadic role already does. Today it raises a
  `RoleError`.
- A group `Jump` may change a member at an offset, `Jump(rate, {s.n: -1, shift_member(s.n, 1): 1})`, so a conserved
  hop on every link of a ring or a line is one group law and one kernel, not one interaction per link. Today the target
  is rejected.
- Compiling a system grows linearly with its number of components. Today the offset of each new variable sums the
  sizes of every earlier one, so compiling 800 one-link interactions takes 8.5 s, against 1.6 s for 400.

Visuals:
- `Element.lattice` keeps a visual that the element type declares, and draws an image only when the type declares
  none. Today it always draws an image.
- Every time series view of the default blueprint shows the whole run. Today the views that a points visual declares
  for one-dimensional positions and other states keep Rerun's default range.

Solvers:
- **BREAKING:** the default ODE solver is SciPy's `LSODA`, which detects stiffness and switches between Adams and BDF
  methods, instead of `RK45`. Today `Simulation(create('nanoparticle_corona')).run(until=10800)` runs for minutes,
  while the catalog run with its registered `LSODA` solver takes 0.2 s. Trajectories of existing ODE runs with the
  default solver change within the solver tolerance.

Benchmarks:
- The real-time benchmarks of the 100 000-sphere gas and the 1000 × 1000 Game of Life are rerun on an idle machine.
  This is a task, not a defect.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `system-modeling`: reductions in any position of interaction laws, reductions in member expressions of observables,
  group members bound to plain roles, jumps at member offsets, and linear compile time.
- `hybrid-dynamics`: series inputs of any length.
- `lattice-dynamics`: lattice groups that keep the visual of their element type.
- `rerun-visualization`: time series views over the whole run.
- `simulation`: the default ODE solver detects stiffness.

## Impact

- `skcomplex.core.model`: `_functions.py` (`series`) and `_components.py` (`Element.lattice`).
- `skcomplex.core.compile`: `_compiler.py` (`_outside_reductions`, `_declare`, role resolution of member references,
  and offset targets of jumps), `_observables.py` (reductions in member expressions), and `_members.py` (offset index
  tables reused by jumps).
- `skcomplex.core.solvers`: `_stiff.py` and `_steppers.py` (the default ODE stepper and the scatter of offset jump
  increments in the NumPy and Numba Gillespie loops).
- `skcomplex.core.viz`: `_blueprints.py`.
- `skcomplex.models`: `zero_range_process` and `asep` move to one group law with offset jumps, and keep their
  trajectories' paths. `dissipative_zero_range` keeps its per-link tree, which offsets cannot address.
- The changelog entry for the new default ODE solver is left to the release, since the parallel apply runs forbid
  edits of the changelog.
- No new dependency. SciPy's `LSODA`, NumPy's index arrays, and Rerun's time axis cover the fixes.
