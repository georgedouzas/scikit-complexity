# Tasks

## 1. Measurement and approval

- [ ] 1.1 Time Burgers on 3200 cells to time 2 and the dam break on 1600 cells to time 0.5 with `SSPRKSolver` on the
  current `FiniteVolumeKernel`, and with PyClaw's classic solver with shipped and Python Roe Riemann solvers, after a
  warm-up call on one machine, record the times and L1 errors in design.md, set the speed thresholds of the scenarios
  from them, and verify PyClaw is still at least ten times faster or stop the change after section 2
- [ ] 1.2 Propose the dependency placement clause and the `clawpack` extra as an amendment of the Project Profile, with
  the version bump, and verify the maintainer approves it before section 3 starts

## 2. In-house hot path

- [ ] 2.1 Keep the canonical flux columns of a field as `expressions` on `FiniteVolumeKernel` in
  `FieldFunctionsPass._compile_volumes`, and verify the Burgers shock scenario still passes
- [ ] 2.2 Compile closed-form eigenvalues of the flux Jacobians from `expressions`, the derivative for one state, the
  trace and determinant for two, and Sage's eigenvalues for more, use them in `FiniteVolumeKernel._speed`, keep
  `np.linalg.eigvals` only without a closed form, record `closed_form`, and verify the same speeds and no closed form
  scenarios
- [ ] 2.3 Replace `np.take(..., range(...))` in `FiniteVolumeKernel._sides` and `_run` with slice tuples built once per
  axis, and verify the faster dam break scenario

## 3. Extra and solver

- [ ] 3.1 Add the `clawpack` extra, the deptry entry, and the per-module ignore of the import-outside-top-level rule to
  `pyproject.toml`, confirm against PyClaw 5.14.0 which dimensions accept Python Riemann kernels, and verify
  `pip install -e '.[clawpack]'` imports `clawpack.pyclaw` on macOS with Apple silicon and Python 3.13
- [ ] 3.2 Add the hook `_get_missing` to `BaseSolver` and the issue `solver-unavailable` to `PassCompiler`, and verify
  the missing extra scenario
- [ ] 3.3 Add `PyClawSolver` to `skcomplex.engine.solvers` with its `SupportSpec`, `_unsupported`, `_build`, and
  `_advance` through `evolve_to_time`, and verify the trajectory and unsupported problem scenarios
- [ ] 3.4 Map the kernel boundaries to periodic, extrapolation, and custom ghost-cell fills, and verify the periodic
  conservation and mixed boundaries scenarios
- [ ] 3.5 Keep the source rates of the conserved states on `ProblemSpec`, apply them through `step_source` with Strang
  splitting, pass parameters through `problem_data` and `aux` at every frame, and verify the damped Burgers and gravity
  change scenarios

## 4. Riemann solvers

- [ ] 4.1 Match flux columns with uniform coefficients against the templates of PyClaw's shipped advection, Burgers,
  shallow-water, and Euler solvers, and verify the known fluxes scenario
- [ ] 4.2 Generate Roe-type f-wave solvers with an entropy fix from the eigen-decomposition of `expressions`, reading
  per-cell values from `aux`, and verify the generated Roe solver and per-cell parameters scenarios
- [ ] 4.3 Generate HLL solvers with a Rusanov last resort, and verify the HLL fallback scenario

## 5. Solver choice and parity

- [ ] 5.1 Add `FiniteVolumeSolver` to `skcomplex.engine.solvers` with its `SupportSpec`, `path` of `'auto'`,
  `'pyclaw'`, and `'native'`, and `path_`, and verify the default install, fallback without the extra, pinned path, and
  finite volumes on a network scenarios
- [ ] 5.2 Switch `burgers`, `shallow_water`, `wave_breaking`, `discrete_velocity_boltzmann`,
  `lighthill_whitham_richards`, `payne`, and `kerner_konhauser` to `FiniteVolumeSolver()` in their registered defaults,
  and verify their model tests pass
- [ ] 5.3 Add the parity tests for Burgers and the dam break at 800 and 1600 cells, skipped without PyClaw, and verify
  the backend parity scenario
- [ ] 5.4 Add the Burgers and dam break speed and accuracy tests to the benchmark suite, and verify the Burgers speed
  and dam break speed scenarios on macOS with Apple silicon

## 6. Documentation and gate

- [ ] 6.1 Document the extra, its installation through PyPI and conda-forge, the solver choice, the Riemann solver
  tiers, and the parity tolerances in the user guide, add the changelog entry, and verify the documentation build runs
  the examples
- [ ] 6.2 Add a CI job on macOS with Apple silicon and Python 3.13 that installs the extra and runs the PyClaw tests,
  and verify it passes
- [ ] 6.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
