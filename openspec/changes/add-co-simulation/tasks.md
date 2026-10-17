# Tasks

## 1. Partitioning

- [ ] 1.1 Build the partition tree when one form does not fit, assign laws by target, and raise the subsystem-specific
  error for mixed leaves, and verify the partitioning and form inference tests pass
- [ ] 1.2 Turn cross-partition symbols into inputs passed as `u` to the lowered functions on both backends, and verify
  the held input test passes

## 2. Aggregates and agent inputs

- [ ] 2.1 Add `Aggregate` with count, sum, mean, and custom reductions over element and agent groups, and verify the
  agent count test passes
- [ ] 2.2 Expose inputs to agent rules through `model.inputs`, and verify the agent resource test passes

## 3. Scheduler

- [ ] 3.1 Add `PartitionedProblem` and the `CoSimulation` stepper with Gauss-Seidel and Jacobi schemes and `sync_dt`
  validation, and verify the interval and ordering tests pass
- [ ] 3.2 Route frames, trajectories, live parameters, and random states across partitions, and verify the mixed
  trajectory and reproducibility tests pass
- [ ] 3.3 Add the split Lotka-Volterra convergence test, and verify the error at least halves with each halving of
  `sync_dt`

## 4. Model library and documentation

- [ ] 4.1 Add the farmer and resource harvesting model, and verify the collapse and sustainable effort tests pass
- [ ] 4.2 Document partitions, inputs, aggregates, and the scheduler with its splitting error in the user guide, and
  verify the documentation test passes
- [ ] 4.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass

## Workflow follow-up

- Archive the change after review.
