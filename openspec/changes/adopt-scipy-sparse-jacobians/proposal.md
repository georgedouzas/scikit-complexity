# Proposal

## Why

`ScipyODE` passes neither `jac` nor `jac_sparsity` to `scipy.integrate.solve_ivp`, so its implicit methods build a
dense Jacobian by finite differences with one drift evaluation per state. The audit in `organize-model-library`
measured Fisher-KPP by finite differences on 4000 points, which is stiff with a largest eigenvalue of about 1600, up to
`T = 10`:

| Path | Time | Relative error |
|---|---|---|
| SciPy BDF with `jac_sparsity` | 0.20 s | 5.3e-7 |
| SciPy Radau with `jac_sparsity` | 0.46 s | 2.2e-9 |
| SciPy BDF with the dense Jacobian, today | 27.2 s | |
| SciPy LSODA with the dense Jacobian | 20.6 s | |
| In-house Crank-Nicolson, `dt = 0.01` | 0.35 s | 9.2e-6 |

SciPy already implements sparse finite-difference Jacobians by column grouping and sparse LU in BDF and Radau. The
framework only has to tell SciPy which states each rate reads, which the compiled symbolic laws already know. On
spectral Kuramoto-Sivashinsky ETDRK4 still wins, 0.072 s against BDF's 0.44 s, so the in-house split steppers stay.

## What Changes

- Derive the Jacobian sparsity pattern of a compiled ODE problem from its symbolic laws: per-member reads, shared
  reads, field neighbour tables, the sparse matrices of finite-difference and finite element operators,
  finite-volume reconstruction stencils, and dense blocks where couplings are only known at run time.
- Derive an exact sparse Jacobian from the laws differentiated by Sage, compiled with the problem's backend.
- Give `ScipyODE` a `jacobian` parameter, `'auto'` by default: BDF and Radau get the sparsity pattern, LSODA gets the
  half bandwidths of a banded pattern, `'exact'` passes the exact sparse Jacobian, and `'dense'` keeps today's path.
  Explicit methods ignore it.
- Pass the same structure in the hybrid path that restarts the integrator at breakpoints and events and records dense
  output for delays.
- Recommend BDF and Radau as the stiff path for finite-difference and finite element fields: the stability warning of
  explicit steppers names them, and an adaptive explicit method that spans more than 1000 stable steps in one call
  warns once.
- No new dependency. SciPy and Sage already are dependencies.

## Capabilities

### New Capabilities

- `ode-jacobians`: the Jacobian sparsity pattern and the exact sparse Jacobian of compiled ODE problems, and their use
  by the SciPy integrators, including the hybrid path.

### Modified Capabilities

- `spatial-fields`: the stability warning names the implicit SciPy methods, and adaptive explicit methods warn on stiff
  fields.

## Impact

- `skcomplex.core.compile` gains `jacobian_pattern` and `jacobian_function`, and `DerivedRead` keeps the sparse matrix
  of its operator when it has one.
- `skcomplex.core.solvers.ScipyODE` gains the `jacobian` parameter and passes the structure in `advance` and
  `_advance_hybrid`.
- `skcomplex.core.solvers.Stepper` changes the stability warning text.
- The user guide's solver page recommends BDF and Radau for stiff fields and the split steppers for spectral fields.
- Benchmarks gain Fisher-KPP on 4000 points with the dense, sparse, and exact paths.
- Later option, out of scope: `diffrax` for gradients through the integrator or GPUs, which needs a `jax.numpy`
  printer and was slower on CPU in the audit.
