# Tasks

## 1. Measurement

- [x] 1.1 Time Burgers on 3200 cells to time 2 and the dam break on 1600 cells to time 0.5 with `SSPRKSolver` on the
  current `FiniteVolumeKernel` after a warm-up call on one machine, record the times and L1 errors in design.md, and
  verify the numbers are reproducible within 10% over three runs

## 2. Hot path

- [ ] 2.1 Keep the canonical flux columns of a field as `expressions` on `FiniteVolumeKernel` in
  `FieldFunctionsPass._compile_volumes`, and verify the Burgers shock unchanged scenario
- [ ] 2.2 Compile closed-form eigenvalues of the flux Jacobians from `expressions`, the derivative for one state, the
  trace and determinant for two, and Sage's eigenvalues for more, use them in `FiniteVolumeKernel._speed`, keep
  `np.linalg.eigvals` only without a closed form, record `closed_form`, and verify the same speeds and no closed form
  scenarios
- [ ] 2.3 Replace `np.take(..., range(...))` in `FiniteVolumeKernel._sides` and `_run` with slice tuples built once per
  axis, re-time both runs, record the speed-up in design.md, set the threshold of the faster dam break scenario to
  half of it, and verify the scenario

## 3. Documentation and gate

- [ ] 3.1 Add the changelog entry, and verify the finite-volume models' tests pass
- [ ] 3.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
