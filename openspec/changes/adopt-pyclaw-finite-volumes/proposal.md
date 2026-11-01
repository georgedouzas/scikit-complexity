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
`riemann=None` lowers a flux to the rate `-divergence(flux)` for spectral and finite-difference fields. `SSPRKSolver`
declares its `SupportSpec`, `ExternalSolver` is the contract for third-party solvers, and the compiler reports what a
solver does not support as `IssueSpec`s. This change therefore drops all of that from its scope, re-measures the
timings on the current code as its first task, and keeps three things: a PyClaw solver that lowers the compiled kernel
of a field, the hot path of the in-house kernel, which still calls `np.linalg.eigvals` and `np.take` with a Python
range, and a finite-volume solver that picks between them.

PyClaw 5.14.0 ships a wheel only for Python 3.13 on macOS 15 with Apple silicon. Linux, Python 3.12, and Intel macOS
need a source build with meson and gfortran, or the conda-forge package. The Project Profile makes every library that
`skcomplex.engine` uses a runtime dependency, which would break the install everywhere else. This change therefore
asks for a narrow amendment of the dependency placement rule, so PyClaw can enter as an optional extra with the
in-house kernel as the fallback, and stops before implementation if the maintainer refuses it.

## What Changes

- Re-measure Burgers on 3200 cells and the dam break on 1600 cells with `SSPRKSolver` on the current kernel and with
  PyClaw's classic solver, record the numbers in the design, and set the speed thresholds of the specs from them.
- Amend the dependency placement rule of the Project Profile: an engine library that publishes no wheel for some
  supported platform or Python version MAY be an optional extra, when an in-house path covers the same systems and the
  engine imports the library lazily.
- Add an optional `clawpack` extra that installs `clawpack>=5.14.0`, and keep it out of the default install.
- Add `PyClawSolver`, an `ExternalSolver` in `skcomplex.engine.solvers`, that runs the `FiniteVolumeKernel`s of a
  problem in PyClaw. It lowers the flux system of each field to a Riemann solver:
  - known fluxes, such as linear advection, Burgers, shallow water in one and two dimensions, and the Euler equations,
    map to PyClaw's shipped Fortran solvers, with uniform parameters passed through `problem_data`,
  - other fluxes of up to three states, and every flux that reads per-cell parameters, get a generated Roe-type
    f-wave solver from a Sage eigen-decomposition of the flux Jacobian, with an entropy fix, reading per-cell values
    from PyClaw's `aux` arrays,
  - fluxes whose eigen-decomposition has no closed form get a generated HLL solver, with Rusanov as the last resort.
- Map field boundaries to PyClaw boundary conditions, step frames with `evolve_to_time`, apply the source rates of
  the conserved states through `step_source`, and pass live parameter changes before the next frame.
- Keep the symbolic flux columns of a field on its `FiniteVolumeKernel`, so the closed-form eigenvalues of the native
  kernel and the Riemann lowering of PyClaw derive from one place.
- Fix the in-house hot path: wave speeds come from closed-form eigenvalues compiled once, with `np.linalg.eigvals`
  kept only where no closed form exists, and the reconstruction slices arrays instead of calling `np.take`.
- Add `FiniteVolumeSolver(path='auto')` in `skcomplex.engine.solvers`, which runs `PyClawSolver` when the extra is
  installed and supports the problem and `SSPRKSolver` otherwise, and records the path it took. The library's
  finite-volume models default to it.
- Report a solver whose library is missing as an `IssueSpec` of the compile report, whose fix names the `clawpack`
  extra and the conda-forge package.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `external-backends`: a PyClaw solver for finite-volume fields behind an optional extra, with flux lowering,
  per-cell parameters, boundaries, sources, live parameters, parity with the native path, and an issue for a missing
  extra.
- `pde-solvers`: finite volumes run in PyClaw when available and in the in-house kernel otherwise, whose wave speeds
  come from closed-form eigenvalues.

## Impact

- `skcomplex.engine.solvers` gains `PyClawSolver` and `FiniteVolumeSolver`, with private modules for the Riemann
  lowering. `clawpack` is imported lazily inside `PyClawSolver._build`, with the import-outside-top-level rule ignored
  for that one module in `pyproject.toml` rather than by an inline suppression.
- `BaseSolver` gains the trivial hook `_get_missing`, which returns the install fix of a solver whose library does not
  import, and `PassCompiler` reports it as the issue `solver-unavailable`.
- `FieldFunctionsPass._compile_volumes` keeps the symbolic flux columns and compiles closed-form wave speeds, and
  `FiniteVolumeKernel` uses them and slices instead of calling `np.take`. `ProblemSpec` keeps the source rates of the
  conserved states apart from the flux differences, for `step_source`.
- `pyproject.toml` gains the `clawpack` extra and the deptry entry for the optional import. CI gains a job on macOS
  with Apple silicon and Python 3.13 that installs the extra. Tests that need PyClaw skip without it.
- The constitution's Project Profile gains the amended dependency placement rule and the `clawpack` extra, with a
  version bump.
- The finite-volume models `burgers`, `shallow_water`, `wave_breaking`, and `discrete_velocity_boltzmann` of
  `skcomplex.models.physics.fluids` and `lighthill_whitham_richards`, `payne`, and `kerner_konhauser` of
  `skcomplex.models.engineering.traffic` default to `FiniteVolumeSolver`. Their results change within the
  discretisation error.
- The user guide documents the extra, its installation on each platform, the solver choice, and the parity
  tolerances, and the changelog records the new solvers and the faster fallback.
