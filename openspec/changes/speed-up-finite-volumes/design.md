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

`SSPRKSolver` steps the problem with a CFL step from `FiniteVolumeKernel.max_speed`. The kernel spends its time where
the audit found it: `_speed` calls `np.linalg.eigvals` on the stacked Jacobians, and `_sides` and `_run` call
`np.take(values, range(...))`, which copies.

## Goals / Non-Goals

**Goals:**

- Make the in-house kernel several times faster with the same results within rounding.

**Non-Goals:**

- PyClaw, in any form. On 2026-11-01 the maintainer refused a dependency placement exception for a library with one
  wheel, so PyClaw waits until it ships wheels for every supported platform and Python version, and then enters as a
  runtime dependency in its own change.
- New numerical fluxes, Roe or HLL solvers, meshes, or adaptive refinement.
- Conservation laws with `riemann=None`, which are rates of spectral and finite-difference fields.

## Decisions

### Measure before and after

The first task times Burgers on 3200 cells to time 2 and the dam break on 1600 cells to time 0.5 with `SSPRKSolver`, on
one machine after a warm-up call, and records the times and L1 errors here. The last hot-path task times them again,
and the faster dam break scenario asks for at least half of the measured speed-up, so the threshold comes from a
measurement and not from a guess.

### One symbolic source for the eigen-structure

`_compile_volumes` already builds the canonical flux columns in shared symbols before printing them. The kernel keeps
them as `expressions`, one column of Sage expressions per dimension, with the state and parameter symbols, so the wave
speeds derive from the same flux the kernel integrates.

### Closed-form speeds and slicing

`_compile_volumes` asks Sage for the eigenvalues of the flux Jacobian of every dimension once, from `expressions`. For
one state the eigenvalue is the derivative, for two states the closed form from the trace and the determinant, and for
more states Sage's symbolic eigenvalues when they come back as explicit expressions. They print to one function per
dimension of the states and the per-cell values, and `_speed` takes the largest absolute value. `np.linalg.eigvals`
stays only for Jacobians with no closed form, and the kernel records which one it uses as `closed_form`. `_sides` and
`_run` build slice tuples per axis once and replace `np.take(values, range(...))` with basic slicing, which returns
views.

Alternative: caching numerical eigenvalues between stages. Rejected because states change every stage, so the cache
would never hit.

## Risks / Trade-offs

- [A closed form with branch cuts, such as square roots of negative values in non-hyperbolic states] → The kernel
  checks for non-real speeds and raises `SolverError` naming the field and the first bad cell.
- [Sage returns eigenvalues as implicit roots for large systems] → The kernel keeps `np.linalg.eigvals` for that field
  and reports it through `closed_form`.
- [Views from slicing are written in place] → The reconstruction only reads its sides, and the update writes to a new
  array, as now.

## Migration Plan

`SSPRKSolver` and `FiniteVolumeKernel` keep their interfaces. Models need no change.

## Open Questions

- None.
