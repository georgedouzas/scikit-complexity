## Context

- `_run_experiment` in `core/experiments/_runs.py` checks `executor` against `('serial', 'thread', 'process')` or
  `concurrent.futures.Executor`, builds one `_TaskSpec` per run with its `SeedSequence`, and calls
  `pool.map(functools.partial(_run_task, problems=problems), tasks)`. The dictionary of compiled problems is shared in
  serial and thread runs, and every process pickles an empty copy, so process workers compile once per task.
- `BrokenProcessPool` becomes `ExperimentError` with the advice of the main guard. No other failure is named.
- `ExperimentResult` holds a tuple of `_RunResultSpec`, each with its times, series, and scalars in memory. It offers
  `rows`, `get_column`, `get_series`, `get_ensemble`, `get_summary`, and `convert_to_frame`.
- `_map_calls` runs whole fits on an executor for the refits of identifiability and individual calibration.
- `distributed` 2026.8.0 is installed through pyabc, and `xarray` 2026.9.0 through dit and pydmd. Neither is declared.
- `add-learning-foundation` declares `xarray` for `read_runs`, and `add-learning-bridges`, `add-diversity-search`, and
  `add-experimental-design` run their folds and batches on the executor of the experiment.

## Goals / Non-Goals

**Goals:**
- Dask and Ray executors on one machine, a cluster, or a cloud, tested on local clusters.
- One compile per chunk of runs on any executor.
- Results larger than the memory of one machine.
- Results that do not depend on the executor, and failures of remote workers that name what to do.
- A decision on each new dependency.

**Non-Goals:**
- Checkpoint and resume of experiments, which the user did not ask for.
- Starting or stopping clusters. A cluster is an application concern, as the constitution says, so the user starts it
  and passes its executor. The docs show how.
- Writing the series from the workers straight to a shared store. The caller writes them as chunks arrive, which works
  without a shared file system. Writes from workers can follow for object stores.
- Streaming the results of `calibrate`, `basin_stability`, and `orbit_diagram`, whose results are small.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `RayExecutor` | 2, an executor the experiments run by type | new private module `core/experiments/_executors.py`, exported from `skcomplex.core.experiments` | every field with large ensembles or sweeps: epidemiology, climate, ecology, finance, statistical physics, and the rollouts of reinforcement learning |
| Chunks, thread limits, and named failures | 2, procedures of the experiment runner | `core/experiments/_executors.py` | the same fields |
| Zarr stores of results | 2, an option of experiments and their result | new private module `core/experiments/_stores.py` | the same fields |
| `convert_to_dataset` | 2, a method of the result of experiments | `core/experiments/_results.py` | the same fields, and every user of xarray tools |

No new package. An executor is already a concept of the experiments, and a store is where a result lives.

### Dependencies

Checked on PyPI on 2026-11-11. The rule of the Project Profile decides: a library that the core imports is a runtime
dependency, whatever its weight or licence. `skcomplex.core.experiments` imports all four below.

| Library | Version | Released | Licence | Status |
| --- | --- | --- | --- | --- |
| distributed (with dask) | 2026.8.0 | 2026-08-24 | BSD-3-Clause | Added as `distributed>=2026.8.0`. Already installed through pyabc. Imported for `KilledWorker` and the client executor in tests |
| ray | 2.60.0 | 2026-11-04 | Apache-2.0 | Added as `ray>=2.59.0` (2026-10-02), since 2.60.0 is a week old. Wheels for CPython 3.12 to 3.14 on Linux x86_64 and aarch64 and macOS arm64, about 79 MB |
| zarr | 3.4.2 | 2026-10-27 | MIT | Added as `zarr>=3.4.0` (2026-09-15). The chunked array store of the results |
| xarray | 2026.9.0 | 2026-09-29 | Apache-2.0 | Added as `xarray>=2026.9.0`, as `add-learning-foundation` also does. Whichever change lands first declares it |
| dask-jobqueue | 0.9.0 | 2024-08-22 | BSD-3-Clause | Not a dependency. The user installs it to start a Dask cluster on SLURM or PBS. Its last release is past the 24 month limit, though its repository had commits in December 2025, so it could not be a runtime dependency without an exception in the Project Profile |

Alternatives surveyed for SLURM and PBS:

| Library | Version and date | Licence | Verdict |
| --- | --- | --- | --- |
| executorlib | 1.11.0, 2026-10-23 | BSD-3-Clause | Its `SlurmClusterExecutor` and `FluxJobExecutor` are `concurrent.futures` executors, so they work with `executor` with no code here. The docs show it beside dask-jobqueue as the maintained choice for SLURM |
| submitit | 1.5.4, 2025-12-17 | MIT | SLURM only, and its executor is not a `concurrent.futures` executor |
| parsl | 2026.11.9, 2026-11-09 | Apache-2.0 | Covers SLURM, PBS Pro, and LSF, but its executors are not `concurrent.futures` executors, so it would need an adapter |
| psij-python | 0.9.11, 2025-05-30 | MIT | A job submission interface, not an executor of functions |

Ray has no `concurrent.futures` executor in `ray.util`, which holds a multiprocessing pool, a joblib backend, and a Dask
scheduler, none of them an `Executor`. `RayExecutor` is therefore the adapter the constitution allows: about 40 lines
that call `ray.remote(function).options(...).remote(*arguments).future()`. The multiprocessing `Pool` of Ray was
rejected because the experiments need futures to stream chunks as they complete.

Weight: Ray adds about 79 MB and a gRPC stack to every install. The rule makes it a runtime dependency. An extra would
break the rule, and the open question asks the user whether the rule should bend here.

### Executors

- `executor` keeps its four kinds. A Dask executor is `client.get_executor(resources=...)`, and a Ray executor is
  `RayExecutor(...)`. No new strings such as `'dask'` or `'ray'`, so there is one way to pass each.
- `RayExecutor` stores `num_cpus`, `num_gpus`, and `resources`. `submit` checks that Ray is initialised and raises
  `ExperimentError` naming `ray.init` otherwise. `shutdown` cancels pending tasks when asked to, and leaves the Ray
  runtime running, since the user started it.
- `_map_calls` and `_run_experiment` share one submission path, so identifiability refits and individual calibrations
  run on Dask and Ray too.

### Chunks

- `chunk_size=None` keeps one run per chunk for `'serial'` and `'thread'`, which share the cache of compiled problems,
  and otherwise splits the runs of each structure into at most 256 chunks, enough to keep 64 workers busy four times
  over. An integer is used as given. With `vectorized=True` a chunk is one batch of `add-batched-runs`, and
  `batch_size` sets it.
- A chunk compiles its structure once and runs its runs in order. It returns its run results, or writes nothing and
  returns them to the caller when a store is used.
- Thread limits: a chunk on a process or remote worker calls `numba.set_num_threads` with the CPUs the executor grants,
  `num_cpus` of `RayExecutor`, and one for a process pool or a Dask executor, whose workers already hold one thread
  per core. The JAX backend on a worker sees the GPUs the worker holds: Ray sets `CUDA_VISIBLE_DEVICES` for a task with
  `num_gpus`, and the docs show the `LocalCUDACluster` of dask-cuda, which starts one Dask worker per GPU.
- The caller submits every chunk and collects them with `concurrent.futures.as_completed`, so a store receives chunks
  as they finish and the memory holds only the chunks in flight. The results are ordered by run index at the end.

### Stores

- `store` is a path or a `zarr` store. The first finished chunk creates one array per kept series with shape
  `(runs, frames, *shape)`, chunked by the runs of a chunk, and the arrays `times` and `seed`. Every chunk writes its
  rows by run index. The scalar rows stay in memory and are also written as one array per column at the end.
- A stored result holds `store` beside `runs`, whose series are empty. `get_series` and `get_ensemble` read the arrays
  from the store, and `convert_to_dataset` opens it with `xarray.open_zarr`, which reads lazily through dask.
- `convert_to_dataset` of an in-memory result builds the same dataset from its runs, with the dimensions `run` and
  `time` and the design values, scenario, replicate, and seed as coordinates of `run`.
- A store that already holds arrays raises `ExperimentError` naming the path, so a run never overwrites a result.

### Random streams across nodes

The `SeedSequence` of every run is spawned on the caller before any submission, in a fixed order, and travels with its
task. A result therefore equals the serial result on any executor, any chunk size, and any number of workers, and the
tests check it on Dask and Ray.

### Named failures

`_executors.py` turns `distributed.scheduler.KilledWorker`, and `ray.exceptions.WorkerCrashedError`,
`ray.exceptions.NodeDiedError`, and `ray.exceptions.ObjectLostError` into `ExperimentError`. The message names the
executor, the first run of the chunk, and what to check: the memory of the workers, `chunk_size`, and the retry
settings of Dask (`distributed.scheduler.allowed-failures`) or Ray (`max_retries`). An exception that a run raises
itself propagates unchanged, as today.

### Probes

The targets are ratios to the current code on the same machine, measured in task 1.1.

- An ensemble of 10000 stochastic SIR runs on a Dask `LocalCluster` of four workers takes at most the time of
  `executor='process'` with four processes today, which compiles once per run.
- A sweep of the logistic map over 1000000 values of `r` with 1000 kept steps, 8 GB of series, runs with a store on a
  machine limit of 2 GB of memory for the caller and gives the same series as a sweep of 10000 values in memory.
- An ensemble of 1000 runs gives equal results on the serial executor, a thread pool, a process pool, a Dask
  `LocalCluster`, and a local Ray runtime.

## Risks / Trade-offs

- [Ray is heavy] → It is the user's explicit choice, and the rule makes it a runtime dependency. The open question
  offers the alternative.
- [Ray and Dask tests are slow to start] → One fixture per test session starts each local cluster once. The tests that
  start one are marked slow when they pass 10 seconds.
- [Clusters cannot be tested in CI] → The SLURM, PBS, and cloud recipes are documented as code that the tests do not
  run, and the local clusters test the same executors.
- [Pickling systems for remote workers] → Systems already pickle for process pools. Dask and Ray use cloudpickle, which
  also pickles factories defined in a notebook.
- [A store on a network file system can be slow] → Chunks are written by the caller in whole Zarr chunks, so no chunk is
  written twice.

## Migration Plan

Nothing breaks. `executor` keeps its values, `chunk_size=None` keeps one run per chunk on the serial and thread
executors, and `store=None` keeps results in memory. Process pools compile once per chunk instead of once per run, so
they get faster with the same results.

## Open Questions

- Ray as a runtime dependency adds about 79 MB to every install. Keep it there, as the rule says, or allow an extra
  `ray` as an exception to the rule?
- Should the project test SLURM in CI with the Docker image of a SLURM cluster, as dask-jobqueue does?
