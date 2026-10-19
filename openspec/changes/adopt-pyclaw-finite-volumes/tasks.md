# Tasks

## 1. In-house hot path

- [ ] 1.1 Compile closed-form eigenvalues of the flux Jacobians in `compile_volumes`, one per state for scalars, from
  the trace and determinant for two states, and from Sage for more, keeping `np.linalg.eigvals` only without a closed
  form, and verify the same speeds and no closed form scenarios
- [ ] 1.2 Replace `np.take(..., range(...))` in `FiniteVolume.rates` with slice tuples built once per axis, and verify
  the faster dam break scenario and that the Burgers shock scenario still passes

## 2. Extra and backend

- [ ] 2.1 Add the `clawpack` extra to `pyproject.toml` and the deptry entry, confirm against PyClaw 5.14.0 which
  dimensions accept Python Riemann kernels, and verify `pip install -e '.[clawpack]'` imports `clawpack.pyclaw` on
  macOS with Apple silicon and Python 3.13
- [ ] 2.2 Add `PyClawBackend` with `check`, `build`, and `advance` through `evolve_to_time`, the lazy import, and the
  missing-extra `BackendError`, and verify the trajectory, unsupported problem, and missing extra scenarios
- [ ] 2.3 Map the field boundaries to periodic, extrapolation, and custom ghost-cell fills, and verify the periodic
  conservation and mixed boundaries scenarios
- [ ] 2.4 Apply rate laws through `step_source` with Strang splitting and parameters through `problem_data` at every
  frame, and verify the damped Burgers and gravity change scenarios

## 3. Riemann solvers

- [ ] 3.1 Match fluxes against the templates of PyClaw's shipped advection, Burgers, shallow-water, and Euler solvers,
  and verify the known fluxes scenario
- [ ] 3.2 Generate Roe-type f-wave solvers with an entropy fix from the compiled eigen-decomposition, and verify the
  generated Roe solver scenario
- [ ] 3.3 Generate HLL solvers with a Rusanov last resort, and verify the HLL fallback scenario

## 4. Solver choice and parity

- [ ] 4.1 Add `FiniteVolumeSolver` with `'auto'`, `'pyclaw'`, and `'native'` and the `backend_` attribute, register it
  as `'finite-volume'`, switch the library finite-volume models to it, and verify the default install, fallback
  without the extra, and pinned path scenarios
- [ ] 4.2 Add the parity tests for Burgers and the dam break at 800 and 1600 cells, skipped without PyClaw, and verify
  the backend parity scenario
- [ ] 4.3 Add the Burgers and dam break speed and accuracy tests to the benchmark suite, and verify the Burgers speed
  and dam break speed scenarios on macOS with Apple silicon

## 5. Documentation and gate

- [ ] 5.1 Document the extra, its installation through PyPI and conda-forge, the solver choice, the Riemann solver
  tiers, and the parity tolerances in the user guide, add the changelog entry, and verify the documentation build
  runs the examples
- [ ] 5.2 Add a CI job on macOS with Apple silicon and Python 3.13 that installs the extra and runs the PyClaw tests,
  and verify it passes
- [ ] 5.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
