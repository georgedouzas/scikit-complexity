# Proposal

## Why

The audit in `organize-model-library` measured the framework's finite volumes against PyClaw. Burgers' equation on
3200 cells took 23 s at an L1 error of 1.7e-3, while PyClaw's classic solver with a monotonized central limiter took
0.18 s at 6.7e-4. The dam break on 1600 cells took 30.5 s, while PyClaw took 0.05 s at half the error. Even PyClaw with
a Roe Riemann solver written in pure Python ran 20 to 150 times faster than the in-house scheme. A Rusanov solver
generated naively for PyClaw was 2 to 6 times less accurate than a Roe solver, so the lowering has to produce proper
waves. The constitution requires a capability that a mature library provides to be taken from that library, so the
conservation laws of finite-volume fields should run in PyClaw.

PyClaw ships wheels only for Python 3.13 on macOS 15 with Apple silicon. Linux, Python 3.12, and Intel macOS need a
source build with meson and gfortran, or the conda-forge package. PyClaw therefore enters as an optional extra, and the
in-house scheme stays as the fallback. The audit also found that 78% of the in-house time goes to `np.linalg.eigvals` of
the flux Jacobian and 14% to `np.take` with a Python range, so the fallback's hot path is fixed too.

## What Changes

- Add an optional `clawpack` extra that installs `clawpack>=5.14.0`, and keep it out of the default install.
- Add a PyClaw backend that runs the conservation laws of lattice finite-volume fields. It lowers every field's symbolic
  `Flux` laws to a PyClaw Riemann solver:
  - known fluxes, such as linear advection, Burgers, shallow water in one and two dimensions, and the Euler equations,
    map to PyClaw's shipped Fortran solvers, with their parameters passed through `problem_data`;
  - other fluxes of up to three states get a generated Roe-type f-wave solver from a Sage eigen-decomposition of the
    flux Jacobian, with an entropy fix;
  - fluxes whose eigen-decomposition has no closed form get a generated HLL solver, with Rusanov as the last resort.
- Map field boundaries to PyClaw boundary conditions. Periodic edges become periodic, zero-flux edges become
  extrapolation, fixed edges and mixed boundaries per state become custom ghost-cell fills.
- Step frames with `evolve_to_time`, apply rate laws on the same states as sources through `step_source`, and pass
  live parameter changes to `problem_data` before the next frame.
- Add a finite-volume solver that picks the PyClaw backend when the extra is installed and the field is supported, and
  the in-house scheme otherwise. It records which path it took. Library finite-volume models default to it.
- Raise a named error that names the `clawpack` extra and the conda-forge package when a user asks for the PyClaw
  backend without it.
- Fix the in-house hot path: wave speeds come from closed-form or symbolic eigenvalues compiled once, with the numerical
  eigenvalues kept only where no closed form exists, and the reconstruction slices arrays instead of calling `np.take`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `external-backends`: a PyClaw backend for finite-volume fields behind an optional extra, with flux lowering,
  boundaries, sources, live parameters, parity with the native path, and an error for a missing extra.
- `pde-solvers`: finite volumes pick PyClaw when available and keep the in-house scheme as the fallback, whose wave
  speeds come from closed-form eigenvalues.

## Impact

- `skcomplex.core.backends` gains `PyClawBackend` and `FiniteVolumeSolver`, and a private module that lowers `Flux`
  laws to Riemann solvers. `clawpack` is imported lazily, with a suppression that names the optional extra.
- `skcomplex.core.compile` compiles the symbolic eigenvalues and eigenvectors of the flux Jacobians once, and
  `FiniteVolume` uses them in `_speed` and slices in `rates`.
- `pyproject.toml` gains the `clawpack` extra and the deptry entry for the optional import. CI gains a job on macOS
  with Apple silicon and Python 3.13 that installs the extra. Tests that need PyClaw skip without it.
- The finite-volume models in `skcomplex.models.physics.fluids`, `skcomplex.models.engineering.traffic`, and the
  wave-breaking model default to `FiniteVolumeSolver`. Their results change within the discretisation error.
- The user guide documents the extra, its installation on each platform, the backend choice, and the parity
  tolerances, and the changelog records the new solver and the faster fallback.
