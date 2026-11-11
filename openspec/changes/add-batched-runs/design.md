## Context

- `_run_experiment` in `core/experiments/_runs.py` builds one `_TaskSpec` per row, scenario, and replicate, spawns one
  `SeedSequence` per row and replicate, compiles one problem per structure (source and model parameters), and maps
  `_run_task` over the tasks on the executor. Each task calls `_simulate`, which steps one state.
- `_replicate.py` turns a system of single elements into a system of groups and runs it once. `orbit_diagram` and
  `basin_stability` use it when `vectorized=True`, and read members with `Trajectory.get_member`. It rejects anything
  but plain single elements, and a replicated `ReactionInteraction` fails its role check.
- After `add-device-execution`, the JAX backend fills `AcceleratedSpec` with loops on a device, and every loop takes
  `(t, x, p, until, ...)` and a key when it draws. The Numba loops take a NumPy generator when they draw.
- `GradientDescentSolver` and calibration already use `jax.vmap` of an unrolled run over batches of parameters.

## Goals / Non-Goals

**Goals:**
- One batched run path for every experiment of runs and every kind of system whose solver has an accelerated loop.
- Results that do not depend on batching or batch size.
- Batch sizes that fit the memory of the device or the host.
- Remove the replication and its limits.

**Non-Goals:**
- Batches split over several devices. `add-multi-device-sharding` does it.
- Batches spread over cluster nodes. `add-cluster-executors` sends one batch per task, which composes with this change
  without more work.
- Batched calibration. `calibrate` keeps its own vectorised likelihoods, and ABC-SMC keeps the sampler of pyabc.
- Batches with adaptive solvers. Diffrax can batch adaptive solves with `vmap`, but every run then takes the steps of
  the slowest one. It can follow when a user needs it.

## Decisions

### Placement

| Component | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| Batched run of one compiled problem | 2, a runtime procedure the engine runs on any problem | new private module `core/engine/runtime/_batches.py` | every field with ensembles or sweeps: epidemiology (outbreak ensembles), ecology (extinction risk), climate (initial condition ensembles), finance (Monte Carlo paths), dynamical systems (orbit diagrams, basins) |
| Batched loops of the JAX and Numba backends | 2, backend lowerings that `_batches.py` reads | `core/engine/backends/_jax.py`, `core/engine/backends/_numba.py` | the same fields |
| `vectorized` and `batch_size` of the experiments | 2, options of experiments over any system | `core/experiments/_runs.py` and the experiment modules | the same fields |

No new package and no new concept. A batch is a set of runs, which experiments already are.

### Dependencies

None. `jax.vmap` and `concurrent.futures.ThreadPoolExecutor` are already in use.

### A batch of runs

- `_batches.py` holds `_simulate_batch(problem, solver, states, parameters, seeds, schedules, until, dt)`. `states`
  has shape `(runs, n)`, `parameters` shape `(runs, m)`, and `seeds` holds one `SeedSequence` per run. It returns one
  `Trajectory` per run on a shared time grid.
- It advances the batch frame by frame, as `Simulation` advances one state. Before every frame it evaluates the
  schedules of every run into the parameter rows, so scenarios with changes at times batch. After every frame it copies
  the batch to the host once.
- The solver must have an accelerated loop for the problem's form. `_simulate_batch` raises `ValidationError` for the
  others, and `_run_experiment` checks the solver before it batches, so an experiment never reaches that error: it
  runs the runs of such a structure one by one and warns once with `BackendWarning`, whose message names the solver and
  `RK4Solver`, `VelocityVerletSolver`, `Yoshida4Solver`, `EulerMaruyamaSolver`, `GillespieSolver`, and
  `IterationSolver`. `add-large-system-kernels` adds the loops of `LatticeSolver` and `MonteCarloSolver`, which then
  batch with no change here.
- Rejected: a batch as one long state vector of a replicated problem. It is what `_replicate` does at the level of the
  system, and it couples runs through every reduction over a group, which then needs per-run masks.

### Batched loops per backend

- JAX: `jax.vmap` of each loop of `_jax_loops.py` over `x`, `p`, and the key, with `t` and `until` shared. Gillespie
  runs keep their own event times, since `vmap` of a `while_loop` runs until every run ends and masks the finished ones.
- Numba: the loops of `_numba_loops.py` gain `nogil=True`, and the batch runs them on a `ThreadPoolExecutor` with one
  run per call and one generator per run. Nested `prange` inside a loop runs serially in a thread, so a batch of large
  groups uses threads over runs instead of threads over members. Rejected: a new `prange` over runs in the source of
  every loop, which would need a typed list of generators that Numba does not support.
- NumPy: a loop over the runs, the reference that the batched results are tested against.
- `BaseBackend` gains the private hook `_get_batched(loop)` with the NumPy loop as its default, and the JAX and Numba
  backends override it. The batched loops are kept in `LoweredFunctionsSpec.batched`, an `AcceleratedSpec` whose loops
  take a leading run axis.

### Random streams and equality

- Every run keeps the `SeedSequence` that `_run_experiment` spawns for it today. The JAX backend turns it into the key
  of the run, as `add-device-execution` defines, and the Numba and NumPy backends into its generator. A run therefore
  draws the same numbers alone, in any batch, and at any batch size.
- The equality is exact on the NumPy and Numba backends and within 1e-10 on the JAX backend, where XLA may fuse a
  batched sum in another order.

### Batch sizes

- `batch_size=None` sizes batches automatically. The bytes of one run are the state and parameters times the workspace
  factor of the loop (five for RK4, three for velocity Verlet, two for the others), plus the kept frames of its outputs
  on the host, all in the precision of the backend.
- The free bytes come from `device.memory_stats()` on a GPU, `bytes_limit` less `bytes_in_use`, and from the physical
  memory of the host through `os.sysconf` on the CPU. Half of the free device memory and a quarter of the host memory
  are used, and a batch holds at least one run.
- An integer `batch_size` is used as given. A batch that does not fit raises the out-of-memory error of the backend,
  which `_simulate_batch` re-raises as `ExperimentError` naming `batch_size`.

### Experiments

- `ensemble`, `sweep`, and `compare_scenarios` gain `vectorized: bool = False` and `batch_size: int | None = None`.
  `basin_stability` and `orbit_diagram` keep `vectorized` and gain `batch_size`.
- With `vectorized=True`, `_run_experiment` groups the tasks by the key of their structure, splits each group into
  batches, and maps `_run_batch` over the batches on the executor, one batch per task. Outputs and analyses run per run
  on the host, as today.
- `orbit_diagram` and `basin_stability` read their runs from the batched trajectories and no longer call
  `Trajectory.get_member`.
- `_replicate.py` and its tests are removed. `tests/core/components/elements/test_declared.py` and
  `tests/core/engine/runtime/test_trajectories.py` import the private `_replicate`, which the constitution forbids,
  and their two tests move to batched ensembles of declared elements and of the Lotka-Volterra system.

### Batched runs by default

Decided by the user on 2026-11-11: `vectorized=True` becomes the default of the experiments of runs once the
benchmarks show that batched runs never lose, with a reported serial fallback for systems that cannot batch.

- The gate is a set of benchmarks marked `benchmark`, one per experiment of runs, each timed with `vectorized=True` and
  `vectorized=False` on the same machine. They cover the NumPy and Numba backends and the CPU device of JAX, with the
  GPU device on the GPU runner of `add-device-execution`, at 1, 10, 1000, and 100000 runs, for a map of single
  elements, an ODE system of groups, and a stochastic reaction network. Batched runs never lose when every case takes
  at most the unbatched time, within the 10% spread of three repeats.
- The flip waits for the gate. A losing case is fixed first, for instance by a run count below which the experiment
  runs one by one, and the outcome of every case is written in the benchmark docstrings and in this design.
- The fallback stays explicit after the flip. A structure whose solver has no accelerated loop runs one by one with the
  `BackendWarning` above, whether `vectorized=True` is the default or passed.
- The flip is BREAKING. Seeded results are unchanged, exactly on the NumPy and Numba backends and within 1e-10 on the
  JAX backend, as the equality above requires. What changes is behaviour: the fallback warning appears for systems that
  ran silently before, and a batch holds more runs in memory at once. The breaking commit, and so the changelog, says
  both and names `vectorized=False` as the way back.

### Probes

The targets are ratios to the current code on the same machine, measured in task 1.1.

- Stochastic SIR reaction network, 1000 people, `GillespieSolver`, 100000 replicates by `ensemble`. With
  `vectorized=True` on the Numba backend it takes at most the time of `executor='process'` on all cores today. On
  `JaxBackend(device='gpu')` it takes at most a fifth of that time.
- `orbit_diagram` of the logistic map over 1000000 values of `r`, 1000 transient steps and 100 kept. With the batched
  JAX loops on the CPU it takes at most twice the time of the replicated Numba run of today. On a GPU it takes at most
  a fifth of it.
- `basin_stability` of the Kuramoto system with 100 oscillators, a system of groups that replication rejected, over
  10000 initial states, runs with `vectorized=True` and gives the same fractions as the unbatched run.

## Risks / Trade-offs

- [Analyses run per run on the host] → They do today. Outputs that are reductions on the device come with
  `add-large-system-kernels`.
- [Runs of different lengths in one Gillespie batch] → A batch lasts as long as its longest run. The docs advise
  smaller batches for runs whose event counts vary widely.
- [Thread batches of the Numba backend lose the parallelism over members] → Batches of runs already use every core,
  and a single huge run is the case for `vectorized=False`.
- [Removing `_replicate` drops a path some user script may import] → It is private, and the public `vectorized`
  argument keeps its meaning.

## Migration Plan

`vectorized=True` of `orbit_diagram` and `basin_stability` keeps its results within 1e-12 for the systems replication
accepted. The examples of both experiments run unchanged. When the default flips, scripts keep their seeded results,
and a script that needs runs one by one, or wants no fallback warning, passes `vectorized=False`.

## Open Questions

None. The user decided on 2026-11-11 that batched runs become the default once the benchmarks show they never lose.
