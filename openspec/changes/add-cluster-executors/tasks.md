# Tasks

## 1. Baselines and dependencies

- [ ] 1.1 Time the current ensemble of 10000 stochastic SIR runs with `executor='process'` and four processes, count
  its compiles, write both in the benchmark docstring, and verify the time repeats within 10% over three runs
- [ ] 1.2 Add `distributed>=2026.8.0`, `ray>=2.59.0`, `zarr>=3.4.0`, and `xarray>=2026.9.0` (unless
  `add-learning-foundation` already did) to the runtime dependencies in `pyproject.toml` and the Project Profile with
  their release dates and licences, and verify `pdm install` resolves on Python 3.12, 3.13, and 3.14 and all four
  import
- [ ] 1.3 Add session fixtures for a Dask `LocalCluster` of two workers and a local Ray runtime of two CPUs in
  `tests/core/experiments/conftest.py`, and verify both start and stop cleanly in one test run

## 2. Executors

- [ ] 2.1 Add `RayExecutor` in `core/experiments/_executors.py` with `submit` and `shutdown`, export it from
  `skcomplex.core.experiments`, and verify the ray-not-started scenario in `tests/core/experiments/test_executors.py`
- [ ] 2.2 Move the submission of `_run_experiment` and `_map_calls` to one path that submits futures and collects them
  with `as_completed`, ordered by run index at the end, and verify the existing serial, thread, and process tests pass
  unchanged
- [ ] 2.3 Verify the local-Dask-cluster, refits-on-Dask, and local-Ray-runtime scenarios

## 3. Chunks

- [ ] 3.1 Add `chunk_size` to `ensemble`, `sweep`, `compare_scenarios`, `basin_stability`, and `orbit_diagram`, split
  the runs of each structure into chunks that compile once, and verify the one-compile-per-chunk scenario in
  `tests/core/experiments/test_runs.py`
- [ ] 3.2 Make a chunk of `add-batched-runs` one batch when `vectorized=True`, if that change is applied, and verify a
  vectorised ensemble on the Dask fixture equals the serial result
- [ ] 3.3 Set the Numba threads of a chunk on a process or remote worker from the CPUs the executor grants, and verify
  the thread-limit-on-a-worker scenario
- [ ] 3.4 Verify that an ensemble of 1000 runs gives equal results on the serial executor, a thread pool, a process
  pool, the Dask fixture, and the Ray fixture, with chunk sizes 1, 7, and 1000

## 4. Stores

- [ ] 4.1 Add `core/experiments/_stores.py` with the writer that creates the arrays from the first chunk and writes
  every chunk by run index, add `store` to `ensemble`, `sweep`, and `compare_scenarios`, and verify the existing-store
  scenario in `tests/core/experiments/test_stores.py`
- [ ] 4.2 Read the series of a stored result in `get_series` and `get_ensemble`, and verify they equal those of the
  same experiment in memory for an ensemble of 50 runs
- [ ] 4.3 Verify the result-larger-than-memory scenario with a memory limit set by `resource.setrlimit` in a child
  process, marked slow

## 5. Datasets

- [ ] 5.1 Add `convert_to_dataset` to `ExperimentResult` for results in memory and in a store, and verify the
  ensemble-as-a-dataset and stored-result scenarios in `tests/core/experiments/test_results.py`

## 6. Failures

- [ ] 6.1 Turn `KilledWorker` of Dask into `ExperimentError` with the named message, and verify the killed-Dask-worker
  scenario on a `LocalCluster` with a small memory limit
- [ ] 6.2 Turn `WorkerCrashedError`, `NodeDiedError`, and `ObjectLostError` of Ray into `ExperimentError`, and verify
  the dead-Ray-worker scenario by killing the worker process of a running task
- [ ] 6.3 Verify the error-of-a-run scenario, so errors that a run raises itself propagate unchanged

## 7. Benchmarks

- [ ] 7.1 Add the benchmark of the ensemble of 10000 stochastic SIR runs on a Dask `LocalCluster` of four workers
  against the baseline of task 1.1, and verify it takes at most the baseline time

## 8. Documentation

- [ ] 8.1 Add a clusters section to `docs/overview/user_guide/run_experiments.md` with Dask, `RayExecutor`,
  `chunk_size`, `store`, `convert_to_dataset`, the thread limits, and the named failures, with runnable examples on a
  local Dask cluster and a local Ray runtime, and verify the markdown tests run them
- [ ] 8.2 Add recipes, as code that the tests do not run, for Dask on SLURM and PBS through dask-jobqueue, for
  executorlib on SLURM, for `LocalCUDACluster` of dask-cuda, for the Ray cluster launcher on a cloud and on SLURM, and
  for KubeRay, and verify the docs build
- [ ] 8.3 Update the docstrings of every experiment that takes `executor` to name Dask and `RayExecutor`, and verify
  the doctests and pydoclint pass

## 9. Integration

- [ ] 9.1 Run `nox` (formatting, ruff, mypy, pydoclint, deptry, pip-audit, import contracts, tests of the touched
  modules with the thread limits set to 1, doctests, and the strict docs build) and verify every session passes
- [ ] 9.2 Run `openspec validate add-cluster-executors --strict` and verify it passes
