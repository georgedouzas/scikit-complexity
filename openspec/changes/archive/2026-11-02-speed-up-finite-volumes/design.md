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

### Keep the kernel in house

A library audit on 2026-11-01 looked for a maintained Python finite-volume library that could run a compiled
`FiniteVolumeKernel`, that is an arbitrary system of conservation laws with per-cell parameters on a lattice, and
publishes wheels for Python 3.12 and 3.13 on Linux and macOS. None fits, so the plan stays the in-house kernel:
- FiPy 4.0.3 discretises implicit diffusion, convection, and source terms of its own term objects on its own meshes and
  owns the solver loop. It has no Riemann solvers or limiters for nonlinear hyperbolic systems, and it cannot take a
  flux system compiled elsewhere.
- ANUGA 4.1.0, Apache-2.0, solves only the shallow-water equations on triangular meshes, and publishes no Linux aarch64
  wheels.
- PyFR 3.1, BSD, is a high-order flux reconstruction code for the Euler, Navier-Stokes, and artificial compressibility
  equations on unstructured meshes, driven by configuration and mesh files. Its systems are fixed in its code, so a
  field's flux cannot be plugged in.
- dune-fem 2.12.0.4 is GPL, publishes no wheels, and builds the DUNE C++ stack from source on install.
- pystencils 2.1 is GPL-3.0-or-later and generates stencil kernels from SymPy, which the Project Profile bans. It
  provides no numerical fluxes, limiters, or time stepping, so it would replace only the slicing this change fixes.
- Devito 4.8.23, MIT, is a finite-difference stencil compiler built on SymPy, with no finite-volume or Riemann
  discretisation.
- jax-cfd 0.2.1 has not been released since 2024-05-13, targets incompressible Navier-Stokes on staggered grids, and
  would add JAX as a runtime dependency.

PyClaw is the one library that fits the numerics, and it is out of scope for the platform reason under Non-Goals.

### Measure before and after

The first task times Burgers on 3200 cells to time 2 and the dam break on 1600 cells to time 0.5 with `SSPRKSolver`, on
one machine after a warm-up call, and records the times and L1 errors here. The last hot-path task times them again,
and the faster dam break scenario asks for at least half of the measured speed-up, so the threshold comes from a
measurement and not from a guess.

The measurements ran on 2026-11-01 on an Apple M2 with 8 cores, Python 3.13, and one thread for NumPy, BLAS, and Numba.
Other agents shared the machine, so every time comes with the one-minute load average at the start of the run. The L1
error of Burgers is against the entropy solution from the characteristics, and the L1 error of the dam break is the
error of the depth against Stoker's exact solution. Each run follows a warm-up run on 64 cells.

Before, on the kernel with `np.linalg.eigvals` and `np.take`:

| Run | Burgers, 3200 cells | Dam break, 1600 cells |
| --- | --- | --- |
| Load 77 to 173 | 43.0 s, 43.3 s, 41.6 s | 37.6 s, 38.8 s, 44.1 s |
| Load 13 to 61 | 24.3 s, 25.2 s, 24.3 s | 22.0 s, 19.9 s, 21.9 s |
| Load 6 to 8 | 22.6 s, 23.1 s, 21.0 s | 20.3 s, 20.9 s, 21.2 s |
| L1 error | 2.096e-3 | 6.683e-3 |

The runs at the highest load differ by 17% for the dam break, so the baseline is the run at load 6 to 8, where every
time lies within 10% of the median of its three runs: 22.6 s for Burgers and 20.9 s for the dam break.

After, with closed-form speeds and slicing, interleaved with the runs before at the same loads:

| Run | Burgers, 3200 cells | Dam break, 1600 cells |
| --- | --- | --- |
| Load 66 to 77 | 2.83 s, 2.47 s, 2.50 s | 0.65 s, 0.62 s, 0.63 s |
| Load 9 | 3.44 s, 2.95 s, 2.64 s | 0.72 s, 0.69 s, 0.70 s |
| L1 error | 2.096e-3 | 6.683e-3 |

The errors agree to the digits shown. Against the baseline, Burgers runs about 8 times faster and the dam break about
30 times faster, 20.9 s against 0.70 s. Burgers has one state, so its speed-up comes from slicing alone, while the dam
break also drops `np.linalg.eigvals`, which took three quarters of its time in a profile. With slicing in both runs,
the closed-form speeds alone run the dam break 29 to 33 times faster than the numerical eigenvalues at load 5 to 6,
with solutions equal within 2.3e-14. The faster dam break scenario therefore asks for a speed-up of at least 15, half of
the measured 30, and compares closed-form speeds against numerical eigenvalues through `set_params(speeds=())`,
since the `np.take` path no longer exists.

### One symbolic source for the eigen-structure

`_compile_volumes` already builds the canonical flux columns in shared symbols before printing them. The kernel keeps
them as `expressions`, one column of Sage expressions per dimension, with the state and parameter symbols, so the wave
speeds derive from the same flux the kernel integrates.

### Closed-form speeds and slicing

`_compile_volumes` asks Sage for the eigenvalues of the flux Jacobian of every dimension once, from `expressions`. For
one state the eigenvalue is the derivative, for two states the closed form from the trace and the determinant, and for
more states Sage's symbolic eigenvalues when they come back as explicit expressions free of the imaginary unit, since
radicals with complex intermediates turn into NaN in real floating point. They print to one function per
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
