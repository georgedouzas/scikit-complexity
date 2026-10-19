# Design

## Context

A field with `discretization='finite-volume'` on a `LatticeSpace` declares `Flux(target, flux, limiter, riemann)` laws.
`compile_volumes` turns them into a `FiniteVolume`, whose `rates` adds MUSCL reconstruction with a minmod or van Leer
limiter and a Rusanov or exact Godunov flux to the drift, and `SSPRK` steps the drift with a CFL step from
`max_speed`. Rate laws on the same states add sources through the same drift. `ExternalBackend` is the contract for
handing a problem to a third-party library: `check` lists what the library cannot run, `build` builds the library
model, and `advance` moves it from frame to frame. LAMMPS is the one backend today.

The audit numbers and the platform limits of PyClaw are in the proposal. PyClaw 5.14.0 is BSD-3. It supports periodic,
extrapolation, wall, and custom boundaries, parameters through `problem_data`, frame stepping through
`evolve_to_time`, sources through `step_source`, and two and three dimensions with its Fortran kernels.

## Goals / Non-Goals

**Goals:**

- Run the conservation laws of a lattice finite-volume field in PyClaw from the same `Flux` laws, with no change to the
  model.
- Produce Riemann solvers with proper waves, so a generated solver is close to a shipped one in accuracy.
- Keep the in-house path for every platform without the extra, and make it several times faster.

**Non-Goals:**

- Finite volumes on meshes, adaptive refinement with AMRClaw, or GeoClaw's bathymetry. Fields keep their lattices.
- Generated Riemann solvers in Fortran. A generated solver is vectorised NumPy and runs only where PyClaw accepts Python
  kernels.
- Making PyClaw a hard dependency. That waits until it ships wheels for every supported platform and Python version.

## Decisions

### Optional extra and lazy import

`pyproject.toml` declares `clawpack = ["clawpack>=5.14.0"]` under `[project.optional-dependencies]`. The backend module
imports `clawpack.pyclaw` and `clawpack.riemann` inside `build`, the one lazy import the constitution allows, with a
suppression whose reason names the extra. When the import fails, the backend raises `BackendError` with the message
that the PyClaw backend needs `pip install 'scikit-complexity[clawpack]'` or `conda install -c conda-forge clawpack`,
and that `FiniteVolumeSolver` or `SSPRK` runs without it. deptry learns the optional import the way it knows pandas.

Alternatives: a hard dependency fails to install on Linux and Python 3.12. Vendoring the Fortran kernels breaks the
rule against vendoring.

### The PyClaw backend

`PyClawBackend(riemann='auto', limiter=None, cfl=0.9, order=2, source_split=2)` subclasses `ExternalBackend` with the
ODE form. `check` accepts a problem whose states all belong to lattice finite-volume fields with flux laws, and whose
other laws on those states are rate laws. It reports every other element, every field on a mesh, and every field whose
generated solver needs a Python kernel in a dimension where PyClaw has none. `build` creates one `pyclaw.Domain`,
`State`, and `Solution` per field with the lattice's cell centers, chooses `ClawSolver1D`, `ClawSolver2D`, or
`ClawSolver3D`, and copies the state vector in. `limiter=None` takes the field's `Flux.limiter`: minmod and van Leer map
to PyClaw's TVD limiters, and `None` gives first order. `advance` writes the parameter vector into `problem_data`,
calls `evolve_to_time(solution, until)`, and copies `state.q` back into the state vector in the layout of the compiled
problem, so frames, trajectories, and Rerun logs keep their names.

Several finite-volume fields in one problem each get their own solver and advance in turn between frames, since they
share no fluxes.

### Lowering fluxes to Riemann solvers

A private module turns the flux columns of a field, one per dimension, into a Riemann solver in three tiers:
- Shipped solvers. The flux columns are compared with templates after substituting the field's states in order:
  `a u` with `advection_1D` and its 2D and 3D forms, `u²/2` with `burgers_1D`, `[q, q²/h + g h²/2]` with
  `shallow_roe_with_efix_1D`, the two-dimensional shallow-water flux with `shallow_roe_with_efix_2D`, and the Euler
  flux with `euler_with_efix_1D` and its 2D form. A template matches when the difference simplifies to zero in Sage,
  and the template's coefficient, such as `g` or `gamma`, becomes a `problem_data` entry read from the field's
  parameters.
- Generated Roe-type f-wave solvers. For up to three states, Sage computes the eigenvalues and right eigenvectors of
  the flux Jacobian. When all are closed forms, they compile to NumPy functions evaluated at the interface average,
  and the solver splits the flux difference `f(q_r) - f(q_l)` into f-waves. The f-wave split conserves for any
  linearisation, which an average-state Roe split does not. A Harten-Hyman entropy fix splits transonic waves. A scalar
  flux uses the Roe speed `(f(r) - f(l)) / (r - l)`, so `riemann='godunov'` and convex scalar fluxes keep their exact
  shocks and rarefactions.
- Generated HLL solvers. When Sage gives no closed-form eigen-decomposition, or the field has more than three states,
  the solver uses two HLL waves with speed bounds from the compiled eigenvalues, or from numerical eigenvalues of the
  interface Jacobians when those are missing. Rusanov stays as the last resort and is never the default.

`riemann` on the backend overrides the choice with `'shipped'`, `'roe'`, `'hll'`, or `'rusanov'`, and the chosen tier
is exposed as `riemann_` per field. Generated solvers set `kernel_language='Python'`. PyClaw's classic Python kernels
cover one dimension, so a two- or three-dimensional field without a shipped solver is reported by `check`. The first
task confirms this limit against PyClaw 5.14.0 and widens it if the classic 2D solver accepts Python kernels.

Alternatives: generating Rusanov for every flux was 2 to 6 times less accurate in the audit. Generating Fortran per
model needs a compiler at run time.

### Boundaries

When every state of a field has the same boundary, `periodic` maps to `BC.periodic` and `zero-flux` to `BC.extrap`.
Fixed edges and mixed boundaries per state map to `BC.custom` with a ghost-cell fill per state: periodic wraps, zero
flux copies the edge cell, and fixed writes the constant. This matches the in-house `_pad`, so both paths see the same
ghost cells.

### Sources and live parameters

Rate laws on the conserved states compile into the problem's drift as today. The backend removes the flux part,
evaluates the rest in `step_source` on the state reshaped to the field's layout, and integrates it with a second-order
Runge-Kutta step under Strang splitting (`source_split=2`). Parameters are read from the parameter vector at every
`advance`, so a parameter changed between frames reaches the shipped solver's `problem_data` and the generated solver's
compiled functions before the next frame.

### Choosing the path

`FiniteVolumeSolver(backend='auto', cfl=0.9, ...)` is a `Stepper` registered as `'finite-volume'`. With `'auto'` it
takes the PyClaw backend when `clawpack` imports and `check` passes, and `SSPRK` otherwise. `'pyclaw'` forces PyClaw
and raises the missing-extra error, and `'native'` forces `SSPRK`. The choice is exposed as `backend_`. The
library's finite-volume models switch their default solver from `SSPRK()` to `FiniteVolumeSolver()`.

This makes the default depend on the installed packages. The parity requirement bounds the difference, `backend_`
makes the choice visible, and a user pins a path with `backend`. Alternative: keeping `SSPRK` as the default leaves the
fast path off for everyone who installs the extra.

### The in-house hot path

`compile_volumes` asks Sage for the eigenvalues of every flux Jacobian once. For one state the eigenvalue is the
derivative, for two states the closed form from the trace and the determinant, and for more states Sage's symbolic
eigenvalues when they come back as explicit expressions. They compile to one NumPy function per dimension, and `_speed`
takes the largest absolute value. `np.linalg.eigvals` stays only for Jacobians with no closed form, and the compiled
problem records which one it uses. `rates` builds slice tuples per axis once and replaces `np.take(values, range(...))`
with basic slicing, which returns views. The same compiled eigenvalues feed the generated PyClaw solvers, so the
eigen-decomposition lives in one place.

## Risks / Trade-offs

- [The extra installs only on macOS 15 with Apple silicon and Python 3.13 from PyPI] → The extra is optional, the
  fallback runs everywhere, the docs give the conda-forge route, and tests that need PyClaw skip without it while a
  dedicated CI job runs them.
- [A Sage eigen-decomposition with branch cuts, such as square roots of negative values in non-hyperbolic states] →
  The generated solver checks for non-real speeds at each step and raises `SolverError` naming the field and the
  first bad cell.
- [The default solver depends on the environment] → Results stay within the parity tolerance, and `backend_` shows the
  choice.
- [Generated Python solvers are slower than shipped Fortran ones] → They were still 20 to 150 times faster than the
  in-house scheme in the audit, and every library flux with a known form maps to a shipped solver.

## Migration Plan

The library is pre-1.0. `SSPRK` keeps its interface, and models that pass `solver=SSPRK()` keep the in-house path.
Library defaults move to `FiniteVolumeSolver`, which falls back to the in-house path without the extra.
