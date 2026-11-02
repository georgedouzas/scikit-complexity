# Proposal

## Why

The audit in `organize-model-library` measured the framework's finite volumes against PyClaw. Burgers' equation on
3200 cells took 23 s at an L1 error of 1.7e-3, while PyClaw's classic solver with a monotonized central limiter took
0.18 s at 6.7e-4. The dam break on 1600 cells took 30.5 s, while PyClaw took 0.05 s at half the error. A Rusanov solver
generated naively for PyClaw was 2 to 6 times less accurate than a Roe solver, so a lowering has to produce proper
waves. The constitution requires a capability that a mature library provides to be taken from that library.

Those numbers predate the current lowering. `ConservationInteraction` and `AdvectionInteraction` now lower the fluxes
of a field to `FluxLaw`s, `FieldFunctionsPass` compiles every flux law of one finite-volume field into one
`FiniteVolumeKernel` in shared symbols, with per-cell parameters and the Rusanov and Godunov numerical fluxes, and
`riemann=None` lowers a flux to the rate `-divergence(flux)` for spectral and finite-difference fields. This change
therefore re-measures the timings on the current code as its first task and fixes the hot path of the in-house kernel,
which still calls `np.linalg.eigvals` and `np.take` with a Python range.

PyClaw 5.14.0 ships a wheel only for Python 3.13 on macOS 15 with Apple silicon. Linux, Python 3.12, and Intel macOS
need a source build with meson and gfortran, or the conda-forge package. The Project Profile makes every library that
`skcomplex.engine` uses a runtime dependency, and on 2026-11-01 the maintainer refused a placement exception for it, so
PyClaw waits for its own change once it ships wheels for every supported platform. A library audit on the same day
found no other finite-volume library that fits, and the design records why each one fails.

## What Changes

- Re-measure Burgers on 3200 cells and the dam break on 1600 cells with `SSPRKSolver` on the current kernel, record the
  numbers in the design, and set the speed threshold of the specs from them.
- Keep the symbolic flux columns of a field on its `FiniteVolumeKernel`, so the closed-form eigenvalues derive from the
  flux the kernel integrates.
- Fix the in-house hot path: wave speeds come from closed-form eigenvalues compiled once, with `np.linalg.eigvals`
  kept only where no closed form exists, and the reconstruction slices arrays instead of calling `np.take`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `pde-solvers`: the in-house finite-volume kernel computes wave speeds from closed-form eigenvalues and slices its
  arrays.

## Impact

- `FieldFunctionsPass._compile_volumes` keeps the symbolic flux columns and compiles closed-form wave speeds, and
  `FiniteVolumeKernel` uses them and slices instead of calling `np.take`.
- `SSPRKSolver` and `FiniteVolumeKernel` keep their interfaces, and no dependency is added. The finite-volume models of
  `skcomplex.models` need no change, and their results stay the same within rounding.
- The changelog records the faster kernel.
