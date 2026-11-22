## Why

The user wants many runs on HPC clusters (SLURM and PBS) and in the cloud, through both Dask and Ray, behind the
`executor` argument that the experiments already take, with both tested. Today `executor` takes `'serial'`,
`'thread'`, `'process'`, or any `concurrent.futures` executor, and three things stop it from scaling past one machine:

- Ray offers no `concurrent.futures` executor, so it cannot be passed at all.
- Every run is one task, and with an executor that is not serial or threaded every task compiles its system again,
  since the cache of compiled problems does not cross processes. On a cluster the compile time of a symbolic system can
  exceed the run time.
- Every result returns to the caller and stays in memory, so an ensemble or a sweep larger than the memory of one
  machine cannot finish. A remote worker that dies raises an error of Dask or Ray that names nothing the user can act
  on.

This change makes Dask and Ray executors first-class, sends tasks in chunks that compile once, streams large results
to a Zarr store as they arrive, and names the failures of remote workers.

## What Changes

- **Dask.** The executor of a Dask client, `client.get_executor()`, works with every experiment, on a local cluster,
  on SLURM or PBS through `dask-jobqueue`, which the user installs, or in the cloud. Tests run it on a
  `LocalCluster`.
- **Ray.** New class `RayExecutor(num_cpus=1, num_gpus=0, resources=None)` in `skcomplex.core.experiments`, a
  `concurrent.futures.Executor` whose `submit` runs a function as a Ray task with those resources and returns the
  future of its object reference. It works on a local Ray runtime, on a cluster started by the Ray cluster launcher on
  a cloud or on SLURM, and on KubeRay. Tests run it on a local Ray runtime. Ray comes with the optional extra `ray`,
  `pip install 'scikit-complexity[ray]'`, and `RayExecutor` imports it on first use, raising `MissingExtraError`
  that names the extra when it is missing.
- **Chunks.** The experiments of runs gain `chunk_size=None`. A chunk is a list of runs of one compiled structure sent
  as one task, so a worker compiles once per chunk. With `vectorized=True` of `add-batched-runs` a chunk is a batch.
  Inside a chunk on a process or remote worker, Numba runs with the threads the executor grants, one by default, so
  workers do not oversubscribe their cores.
- **Streaming to a store.** `ensemble`, `sweep`, and `compare_scenarios` gain `store=None`, a path or a Zarr store.
  With a store, the series of every finished chunk are written to Zarr arrays with one row per run as the chunk
  arrives, only the scalar rows stay in memory, and the `ExperimentResult` reads its series from the store.
  `ExperimentResult.convert_to_dataset()` returns the result as an xarray dataset, lazily for a stored result.
- **Random streams across nodes.** Every run keeps its spawned `SeedSequence`, which travels with its task, so a result
  on Dask or Ray equals the serial result. Tests check it.
- **Named failures.** A Dask worker that is killed or a Ray worker or node that dies raises `ExperimentError`, whose
  message names the executor, the chunk, and the memory and retry settings to check.
- **Dependencies.** `distributed`, `zarr`, and `xarray` become runtime dependencies, since
  `skcomplex.core.experiments` imports them. Ray becomes the optional extra `ray`, an explicit exception to the rule
  that libraries the core uses are runtime dependencies, decided by the user on 2026-11-11 because of its size of about
  79 MB. The extra joins the nox sessions and the install lines of every workflow. `dask-jobqueue` stays with the user.
- **`import_extra` moves down to core.** The helper that imports an extra and raises `MissingExtraError` moves from
  `skcomplex/systems/_extras.py` to `skcomplex/core/errors.py`, beside the error, so the core can use it without
  importing the systems. `skcomplex.systems` keeps exporting it.
- **SLURM in CI.** A CI job starts a small SLURM cluster in Docker, from the maintained image
  `giovtorres/slurm-docker-cluster`, and runs a distributed ensemble on Dask and on Ray clusters started on it.

## Capabilities

### New Capabilities

- `cluster-execution`: experiments on Dask and Ray clusters, chunks of runs, Zarr stores of results, and named
  failures of remote workers.

### Modified Capabilities

- `experiments`: the named error of dead workers covers remote workers, and results convert to xarray datasets.

## Impact

- Core: new private modules `core/experiments/_executors.py` (`RayExecutor`, the chunks, the thread limits, and the
  failure names) and `core/experiments/_stores.py` (the Zarr writer and reader). `_runs.py` submits chunks and collects
  them as they complete. `_results.py` gains the store and `convert_to_dataset`. `core/errors.py` gains `import_extra`,
  and `systems/_extras.py` is removed.
- Dependencies: `distributed>=2026.8.0` (2026.8.0, 2026-08-24, BSD-3-Clause, already installed through pyabc),
  `zarr>=3.4.0` (3.4.2, 2026-10-27, MIT), and `xarray>=2026.9.0` (2026-09-29, Apache-2.0, already installed through
  dit and pydmd), which `add-learning-foundation` also declares, as runtime dependencies, and the extra
  `ray = ["ray>=2.59.0"]` (2.60.0, 2026-11-04, Apache-2.0). `executorlib` joins a development group `slurm` used only
  by the SLURM job.
- CI: the extra `ray` in the `EXTRAS` of `noxfile.py` and the `pdm install` lines of `ci.yml`, `release.yml`, and
  `docs.yml`, a check of the missing extra in the default install job, and a job `slurm` in `ci.yml` with its compose
  file and job scripts under `.github/slurm/`.
- Docs: a section on clusters in the run experiments page of the user guide, with Dask on SLURM and PBS through
  dask-jobqueue, Ray through its cluster launcher, and cloud recipes, shown as code that the tests do not run.
- Order: in parallel with `add-device-execution`, since it touches only the experiments. `add-batched-runs` composes
  with it, one batch per chunk. `add-multi-device-sharding` reads its stores for runs too large for one host.
