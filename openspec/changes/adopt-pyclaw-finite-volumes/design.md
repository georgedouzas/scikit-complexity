# Design

## Context

A finite-volume field is `FieldElement(element=..., space=LatticeSpace(...), discretization='finite-volume')`. Its
conserved states take the flux laws of `ConservationInteraction` and `AdvectionInteraction`, or `FluxLaw`s the element
returns itself. `FieldFunctionsPass._compile_volumes` gathers every `FluxLaw` of one field into one
`FiniteVolumeKernel` in `skcomplex.engine.problems`:
- the fluxes are printed in shared symbols per state, so the Rusanov speed is the spectral radius of the Jacobian of
  the whole system,
- parameters and states that no flux conserves are read per cell, piecewise constant at the interfaces,
- the numerical flux is Rusanov, or Godunov for one state with uniform parameters,
- the limiter is minmod, van Leer, or none, and every flux law of the field must ask for the same scheme.

`SSPRKSolver` steps the problem with a CFL step from `FiniteVolumeKernel.max_speed`, and source rates on the conserved
states add to the same drift. A conservation law with `riemann=None` lowers to `RateLaw(u, -divergence(flux))` and
never reaches a `FiniteVolumeKernel`, so it stays with the spectral and finite-difference solvers.

`ExternalSolver` is the contract for third-party libraries: `_get_name`, `_get_form`, `_unsupported`, `_build`, and
`_advance`, with `_prepare` raising one `BackendError` that lists everything the library cannot run. `LammpsSolver`
is the one external solver today. `PassCompiler` checks the solver passed against its `SupportSpec` and reports every
mismatch as an `IssueSpec` in its `report_`, which `Simulation.check` raises.

The kernel still spends its time where the audit found it: `_speed` calls `np.linalg.eigvals` on the stacked
Jacobians, and `_sides` and `_run` call `np.take(values, range(...))`.

## Goals / Non-Goals

**Goals:**

- Run the conservation laws of a lattice finite-volume field in PyClaw from the same compiled kernel, with no change to
  the model.
- Produce Riemann solvers with proper waves, so a generated solver is close to a shipped one in accuracy, including
  for fluxes that read per-cell parameters.
- Keep the in-house path for every platform without the extra, and make it several times faster.

**Non-Goals:**

- Finite volumes on meshes, adaptive refinement with AMRClaw, or GeoClaw's bathymetry. Fields keep their lattices.
- Conservation laws with `riemann=None`, which are rates of spectral and finite-difference fields.
- Generated Riemann solvers in Fortran. A generated solver is vectorised NumPy and runs only where PyClaw accepts
  Python kernels.
- Making PyClaw a hard dependency. That waits until it ships wheels for every supported platform and Python version.

## Decisions

### Re-measure before building

The audit timed the scheme before one kernel per field and per-cell parameters existed. The first task times, on one
machine after a warm-up call:
- Burgers on 3200 cells to time 2 and the dam break on 1600 cells to time 0.5 with `SSPRKSolver` on the current code,
- the same runs with the closed-form speeds and the slicing of the hot path,
- PyClaw's classic solver with its shipped Riemann solvers and with a Python Roe solver on the same problems.

The numbers go into this section, and the speed thresholds of the scenarios are set from them: the PyClaw scenarios
ask for at least half of the measured ratio, and the hot path scenario for at least half of its measured speed-up. If
PyClaw is no longer at least ten times faster than the fixed native path, the change stops after the hot path and
records why.

### Dependency placement

The Project Profile says that a library `skcomplex.engine` uses is a runtime dependency, and that an optional extra
holds only libraries a model alone needs. PyClaw cannot be a runtime dependency while it ships one wheel. The change
proposes one clause for the profile: an engine library that publishes no wheel for a supported platform or Python
version MAY be an optional extra, when an in-house path covers the same systems, the engine imports it lazily in one
module, and the compile report names the extra when it is missing. The amendment adds a rule, so it needs the
maintainer's approval, and the tasks put it first.

Alternatives: a hard dependency fails to install on Linux and Python 3.12. Vendoring the Fortran kernels breaks the
rule against vendoring. Moving the solver to `skcomplex.models` would put a general solver in the model library.

### Optional extra and lazy import

`pyproject.toml` declares `clawpack = ["clawpack>=5.14.0"]` under `[project.optional-dependencies]`. `PyClawSolver`
imports `clawpack.pyclaw` and `clawpack.riemann` inside `_build`, and the import-outside-top-level rule of `ruff` is
ignored for that one module in `pyproject.toml`, so no inline suppression is written. deptry learns the optional
import.

`BaseSolver` gains the hook `_get_missing(self) -> str`, which returns the empty string by default. `PyClawSolver`
returns the fix when `importlib.util.find_spec('clawpack')` finds nothing: install `scikit-complexity[clawpack]` or
the conda-forge package `clawpack`, or use `SSPRKSolver` or `FiniteVolumeSolver`. `PassCompiler` adds the issue
`solver-unavailable` with that fix to the report of the passed solver, so `Simulation.check` raises before
compilation.

### The PyClaw solver

`PyClawSolver(riemann='auto', limiter=None, cfl=0.9, source_split=2, random_state=None, until=inf)` in
`skcomplex.engine.solvers` subclasses `ExternalSolver` with the ODE form, and declares:

```text
SupportSpec(time={ContinuousTime}, space={LatticeSpace}, laws={FluxLaw, RateLaw, BoundaryLaw},
            orders={OrderKind.INTEGER})
```

`PassCompiler` compares this declaration with the system, so a mesh space, a jump law, or an agent group is reported as
`solver-unsupported`. `_unsupported` lists the finer conditions the declaration cannot express, which `_prepare`
raises as one `BackendError`: a problem without `FiniteVolumeKernel`s, a rate law on a state no flux conserves, and a
field whose generated solver needs a Python kernel in a dimension where PyClaw has none.

`_build` creates one `pyclaw.Domain`, `State`, `Solution`, and solver per kernel, with the lattice's cell centres,
`ClawSolver1D`, `ClawSolver2D`, or `ClawSolver3D`, and copies the conserved states in. `limiter=None` takes the
kernel's limiter: minmod and van Leer map to PyClaw's TVD limiters, and `None` gives first order. `_advance` writes
uniform parameters into `problem_data` and per-cell values into `state.aux`, calls `evolve_to_time(solution, until)`
for every field in turn, and copies `state.q` back into the state vector in the layout of the compiled problem, so
frames, trajectories, and Rerun logs keep their names. The chosen Riemann solver of each field is exposed as
`riemann_`, a mapping from the field's path to `'shipped'`, `'roe'`, `'hll'`, or `'rusanov'`.

### One symbolic source for the eigen-structure

`_compile_volumes` already builds the canonical flux columns in shared symbols before printing them. The kernel keeps
them as `expressions`, one column of Sage expressions per dimension, with the state symbols and the parameter symbols.
The closed-form wave speeds of the native kernel and the Riemann lowering of PyClaw both read this field, so the
eigen-decomposition is derived in one place.

### Lowering a flux system to a Riemann solver

A private module of `PyClawSolver` turns the flux columns of a kernel into a Riemann solver in three tiers:
- Shipped solvers. The columns are compared with templates after substituting the field's states in order: `a u` with
  `advection_1D` and its 2D and 3D forms, `u²/2` with `burgers_1D`, `[q, q²/h + g h²/2]` with
  `shallow_roe_with_efix_1D`, the two-dimensional shallow-water flux with `shallow_roe_with_efix_2D`, and the Euler
  flux with `euler_with_efix_1D` and its 2D form. A template matches when the difference simplifies to zero in Sage and
  every coefficient it reads, such as `g` or `gamma`, is uniform, and the coefficient becomes a `problem_data` entry.
- Generated Roe-type f-wave solvers. For up to three states, Sage computes the eigenvalues and right eigenvectors of
  the flux Jacobian. When all are closed forms, they compile to NumPy functions evaluated at the interface average,
  with per-cell values averaged from the `aux` arrays of the two cells, and the solver splits the flux difference
  `f(q_r) - f(q_l)` into f-waves, which conserves for any linearisation and for fluxes that vary in space. A
  Harten-Hyman entropy fix splits transonic waves. A scalar flux uses the Roe speed `(f(r) - f(l)) / (r - l)`, so the
  kernel's `riemann='godunov'` keeps its exact shocks and rarefactions for convex scalar fluxes.
- Generated HLL solvers. When Sage gives no closed-form eigen-decomposition, or the field has more than three states,
  the solver uses two HLL waves with speed bounds from the closed-form eigenvalues of the kernel, or from numerical
  eigenvalues of the interface Jacobians when those are missing. Rusanov stays as the last resort and is never the
  default.

`riemann` on the solver overrides the choice with `'shipped'`, `'roe'`, `'hll'`, or `'rusanov'`. A forced shipped
solver on a field with per-cell values is unsupported. Generated solvers set `kernel_language='Python'`. PyClaw's
classic Python kernels cover one dimension, so a two- or three-dimensional field without a shipped solver is
unsupported. The task that adds the extra confirms this limit against PyClaw 5.14.0 and widens it if the classic 2D
solver accepts Python kernels.

Alternatives: generating Rusanov for every flux was 2 to 6 times less accurate in the audit. Generating Fortran per
model needs a compiler at run time.

### Boundaries

The kernel holds one boundary per conserved state. When every state of a field has the same boundary, `periodic` maps
to `BC.periodic` and `zero-flux` to `BC.extrap`. Fixed edges and mixed boundaries per state map to `BC.custom` with a
ghost-cell fill per state: periodic wraps, zero flux copies the edge cell, and fixed writes the constant. This matches
the kernel's `_pad`, so both paths see the same ghost cells.

### Sources and live parameters

`FieldFunctionsPass._with_volumes` adds the flux differences to the drift. The pass also keeps the drift before that
addition on `ProblemSpec` as the source rates, which hold the `source` of every `ConservationInteraction` and every
other rate law on the conserved states. `PyClawSolver` evaluates them in `step_source` on the state reshaped to the
field's layout and integrates them with a second-order Runge-Kutta step under Strang splitting (`source_split=2`).
Parameters are read from the parameter vector at every `_advance`, so a parameter changed between frames reaches
`problem_data`, the `aux` arrays, and the generated solver before the next frame.

### Choosing the path

`FiniteVolumeSolver(path='auto', cfl=0.9, order=3, dt=0.01, random_state=None, until=inf)` in
`skcomplex.engine.solvers` is a `Solver` with the support of `SSPRKSolver` on finite-volume fields:

```text
SupportSpec(time={ContinuousTime}, space={LatticeSpace}, laws={FluxLaw, RateLaw, BoundaryLaw},
            orders={OrderKind.INTEGER})
```

In `_prepare` it builds `PyClawSolver` when `path='auto'`, `clawpack` imports, and `PyClawSolver._unsupported` finds
nothing, and `SSPRKSolver` otherwise, and delegates `_run` to it. `path='pyclaw'` forces PyClaw, so its
`_get_missing` reports the missing extra, and `path='native'` forces `SSPRKSolver`. The choice is exposed as `path_`,
`'pyclaw'` or `'native'`. The argument is `path`, not `backend`, since `Simulation` already takes a `backend` that
names the expression backend.

This makes the default depend on the installed packages. The parity requirement bounds the difference, `path_`
makes the choice visible, and a user pins a path with `path`. Alternative: keeping `SSPRKSolver` as the default leaves
the fast path off for everyone who installs the extra.

### The in-house hot path

`_compile_volumes` asks Sage for the eigenvalues of the flux Jacobian of every dimension once, from the kept
`expressions`. For one state the eigenvalue is the derivative, for two states the closed form from the trace and the
determinant, and for more states Sage's symbolic eigenvalues when they come back as explicit expressions. They print to
one function per dimension of the states and the per-cell values, and `_speed` takes the largest absolute value.
`np.linalg.eigvals` stays only for Jacobians with no closed form, and the kernel records which one it uses as
`closed_form`. `_sides` and `_run` build slice tuples per axis once and replace `np.take(values, range(...))` with
basic slicing, which returns views.

## Risks / Trade-offs

- [The amendment is refused] → The change keeps only the hot path and records PyClaw as blocked until it ships wheels
  for every supported platform.
- [The extra installs only on macOS 15 with Apple silicon and Python 3.13 from PyPI] → The extra is optional, the
  fallback runs everywhere, the docs give the conda-forge route, and tests that need PyClaw skip without it while a
  dedicated CI job runs them.
- [A Sage eigen-decomposition with branch cuts, such as square roots of negative values in non-hyperbolic states] →
  The generated solver checks for non-real speeds at each step and raises `SolverError` naming the field and the
  first bad cell.
- [The default solver depends on the environment] → Results stay within the parity tolerance, and `path_` shows the
  choice.
- [Generated Python solvers are slower than shipped Fortran ones] → The re-measurement includes a Python Roe solver,
  and every library flux of a known form with uniform parameters maps to a shipped solver.

## Migration Plan

The library is pre-1.0. `SSPRKSolver` keeps its interface, and models that pass `solver=SSPRKSolver()` keep the
in-house path. Library defaults move to `FiniteVolumeSolver`, which falls back to the in-house path without the extra.

## Open Questions

- None. The solver choice stays inside `FiniteVolumeSolver`, since `Simulation` picks no solver by availability, and
  the discretization is the `discretization` argument of `FieldElement`.
