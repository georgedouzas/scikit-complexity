# Tasks

## 1. Prerequisite and baselines

- [ ] 1.1 Check that `add-device-execution` is applied, so the JAX backend fills `AcceleratedSpec`, and stop if it is
  not
- [ ] 1.2 Time the current ensemble of 100000 stochastic SIR runs with `executor='process'` and the current
  `orbit_diagram` of the logistic map over 1000000 values with `vectorized=True`, write the times and the machine in
  the benchmark docstrings, and verify they repeat within 10% over three runs

## 2. Batched loops

- [ ] 2.1 Add the private hook `_get_batched` to `BaseBackend` with the NumPy loop over runs as its default, add
  `batched` to `LoweredFunctionsSpec`, and verify the batch-of-initial-states scenario in
  `tests/core/engine/backends/test_base.py`
- [ ] 2.2 Compile the Numba loops with `nogil=True`, override `_get_batched` with a thread pool of one run per call and
  one generator per run, and verify the Numba part of the backends-agree-on-a-batch scenario in
  `tests/core/engine/backends/test_numba.py`
- [ ] 2.3 Override `_get_batched` in the JAX backend with `jax.vmap` of every loop over states, parameters, and keys,
  and verify the JAX part of the backends-agree-on-a-batch scenario in `tests/core/engine/backends/test_jax.py`
- [ ] 2.4 Verify that a batch of Gillespie runs on the JAX backend ends every run at its own last event, with runs whose
  epidemics die out early beside runs that last

## 3. Batched runtime

- [ ] 3.1 Add `core/engine/runtime/_batches.py` with `_simulate_batch`, which advances the batch frame by frame with the
  solver's batched loop, evaluates the schedules of every run before each frame, copies the batch once per frame, and
  returns one trajectory per run, and verify it through the public experiments in task 5.1
- [ ] 3.2 Raise `ValidationError` in `_simulate_batch` for a solver without an accelerated loop, check the solver in
  `_run_experiment` before batching, run such a structure one by one with one `BackendWarning` that names the solver
  and the solvers that batch, and verify the solver-without-a-loop scenario through `ensemble` in
  `tests/core/experiments/test_ensemble.py`
- [ ] 3.3 Size batches from the bytes of one run and the free memory of the device or the host, wrap out-of-memory
  errors in `ExperimentError`, and verify the automatic-size scenario with a fake memory report and the
  too-large-a-batch scenario on the CPU device

## 4. Random streams

- [ ] 4.1 Give every run of a batch the `SeedSequence` that `_run_experiment` spawns for it, as a key or a generator,
  and verify the batch-sizes scenario in `tests/core/experiments/test_runs.py`

## 5. Experiments

- [ ] 5.1 Add `vectorized` and `batch_size` to `ensemble`, `sweep`, and `compare_scenarios`, group the tasks by
  structure into batches in `_run_experiment`, and verify the vectorised-sweep, reaction-networks-batch, and
  scenarios-with-schedules-batch scenarios
- [ ] 5.2 Move `orbit_diagram` and `basin_stability` to batched runs, add `batch_size`, and verify the bistable-system
  and system-of-groups scenarios in `tests/core/experiments/test_basin_stability.py` and the existing orbit diagram
  tests unchanged
- [ ] 5.3 Remove `core/experiments/_replicate.py` and `tests/core/experiments/test_replicate.py`, rewrite the two
  replication tests of `tests/core/components/elements/test_declared.py` and the member test of
  `tests/core/engine/runtime/test_trajectories.py` as batched ensembles through the public API, and verify no module
  imports `_replicate`
- [ ] 5.4 Verify that a batched ensemble on a thread executor and on a process executor gives the result of the serial
  executor

## 6. Benchmarks

- [ ] 6.1 Add the benchmark of the ensemble of a stochastic reaction network, and verify its target locally
- [ ] 6.2 Add the benchmark of the orbit diagram on a GPU, marked `gpu`, and verify its target on the GPU runner of
  `add-device-execution`, or record that none was available
- [ ] 6.3 Add the gate benchmarks, one per experiment of runs, that time `vectorized=True` against `vectorized=False`
  on the NumPy and Numba backends and the CPU device of JAX, at 1, 10, 1000, and 100000 runs, for a map of single
  elements, an ODE system of groups, and a stochastic reaction network, with the GPU cases marked `gpu`, and verify
  the small-ensemble-does-not-lose scenario
- [ ] 6.4 Run the gate benchmarks on an idle machine and on the GPU runner, write the outcome of every case in the
  benchmark docstrings and the design, and verify that batched runs take at most the unbatched time in every case,
  within the 10% spread of three repeats, or fix the losing cases first and run the gate again

## 7. Documentation

- [ ] 7.1 Add a batched runs section to `docs/overview/user_guide/run_experiments.md` with `vectorized`, `batch_size`,
  the solvers that batch, the fallback warning, and a runnable ensemble example, and verify the markdown tests run it
- [ ] 7.2 Update the docstrings of `orbit_diagram` and `basin_stability`, which describe replication into members, and
  verify the doctests pass

## 8. Batched runs by default

- [ ] 8.1 Check that task 6.4 passed on every case, and stop if one loses
- [ ] 8.2 BREAKING: make `vectorized=True` the default of `ensemble`, `sweep`, `compare_scenarios`, `basin_stability`,
  and `orbit_diagram`, update their docstrings and the batched runs section of the user guide, commit it as a breaking
  change (`feat!:`) whose body names the fallback warning, the memory of a batch, and `vectorized=False` as the way
  back, so the changelog lists it, and verify the batched-by-default scenario and that every existing seeded
  experiment test passes unchanged

## 9. Integration

- [ ] 9.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules with the thread
  limits set to 1, doctests, and the strict docs build) and verify every session passes
- [ ] 9.2 Run `openspec validate add-batched-runs --strict` and verify it passes
