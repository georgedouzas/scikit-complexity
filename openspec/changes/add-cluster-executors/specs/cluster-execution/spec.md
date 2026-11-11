# Spec Delta

## Purpose

Lets a user run the experiments of a system on Dask and Ray clusters, from one machine to HPC clusters and clouds, in
chunks that compile once, with results larger than memory streamed to a Zarr store and the same results as a serial
run.

## ADDED Requirements

### Requirement: Experiments run on Dask
Every experiment that takes `executor` SHALL accept the executor of a Dask client and SHALL return the result of the
serial executor.

#### Scenario: Local Dask cluster
- **WHEN** an ensemble of 200 stochastic SIR runs with seed 7 executes on `client.get_executor()` of a Dask
  `LocalCluster` of two worker processes
- **THEN** the result equals the result of `executor='serial'`

#### Scenario: Refits on Dask
- **WHEN** the refits under noise of an identifiability experiment run on the executor of a Dask client
- **THEN** the result equals the result of `executor='serial'`

### Requirement: Experiments run on Ray
`skcomplex.core.experiments` SHALL offer `RayExecutor(num_cpus=1, num_gpus=0, resources=None)`, a
`concurrent.futures.Executor` that runs every submitted function as a Ray task with those resources, and every
experiment that takes `executor` SHALL accept it and return the result of the serial executor. Submitting before Ray is
initialised SHALL raise `ExperimentError`, whose message names `ray.init`.

#### Scenario: Local Ray runtime
- **WHEN** an ensemble of 200 stochastic SIR runs with seed 7 executes on `RayExecutor()` after `ray.init(num_cpus=2)`
- **THEN** the result equals the result of `executor='serial'`

#### Scenario: Ray not started
- **WHEN** a user passes `RayExecutor()` before calling `ray.init`
- **THEN** the experiment raises `ExperimentError`, whose message names `ray.init`

### Requirement: Runs travel in chunks that compile once
The experiments of runs SHALL take `chunk_size` and SHALL send the runs of one compiled structure in chunks, each
compiled once on its worker, and a result SHALL not depend on the chunk size. On a process or remote worker, a chunk
SHALL run Numba with the CPUs the executor grants, one by default.

#### Scenario: One compile per chunk
- **WHEN** a sweep of 100 values runs on a process pool with `chunk_size=25`
- **THEN** the system is compiled 4 times, and the result equals the result with `chunk_size=1`

#### Scenario: Thread limit on a worker
- **WHEN** a chunk runs on a process pool on a machine with eight cores
- **THEN** Numba runs it with one thread

### Requirement: Results stream to a Zarr store
`ensemble`, `sweep`, and `compare_scenarios` SHALL take `store`, a path or a Zarr store. With a store, the series of
every finished chunk SHALL be written to the store as the chunk arrives, only the scalar rows SHALL stay in memory, and
the result SHALL read its series from the store. A store that already holds a result SHALL raise `ExperimentError`,
whose message names the path.

#### Scenario: Result larger than memory
- **WHEN** a sweep of the logistic map over 1000000 values of `r` keeps 1000 values per run with a store, in a caller
  limited to 2 GB of memory, in a test marked `slow`
- **THEN** it finishes, and the series of the first 10000 runs equal those of a sweep of the same values in memory

#### Scenario: Existing store
- **WHEN** a user runs an ensemble into a path that already holds a stored result
- **THEN** it raises `ExperimentError`, whose message names the path

### Requirement: Remote worker failures raise a named error
A Dask worker that is killed, or a Ray worker or node that dies, SHALL raise `ExperimentError`, whose message names the
executor, the first run of the failed chunk, the memory of the workers, `chunk_size`, and the retry setting of the
executor. An error that a run raises itself SHALL propagate unchanged.

#### Scenario: Killed Dask worker
- **WHEN** a run of an ensemble on a Dask `LocalCluster` makes its worker exceed its memory limit until Dask gives up
- **THEN** the experiment raises `ExperimentError`, whose message names `chunk_size` and
  `distributed.scheduler.allowed-failures`

#### Scenario: Error of a run
- **WHEN** a factory of systems raises `ValidationError` for one row of a sweep on `RayExecutor()`
- **THEN** the experiment raises `ValidationError`
