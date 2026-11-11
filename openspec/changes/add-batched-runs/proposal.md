## Why

Many runs matter as much to the user as huge single runs: ensembles, sweeps, scenarios, basin stability, orbit
diagrams, calibration, inference, and reinforcement learning rollouts. Today the experiments run one task per run, and
each task steps its own state from Python. The only vectorised path, `vectorized=True` of `basin_stability` and
`orbit_diagram`, rebuilds the system with every single element replaced by a group, through the private
`_replicate`. That path has three limits:

- It rejects systems with child systems, groups, fields, or agents, so most large systems cannot use it.
- A replicated `ReactionInteraction` fails its role check, as the archived change `fix-error-reporting-defects`
  recorded.
- Only two experiments offer it, and the result reads members of one trajectory rather than runs.

A batch of runs of one compiled problem needs no new system. It is the same problem with a leading run axis on its
states, parameters, and random keys. On the JAX backend `jax.vmap` runs the batch as one device program. On the Numba
backend the loops run on threads, one run per thread. This change adds batched runs to every experiment of runs and
retires the replication.

## What Changes

- **Batched runs in the engine.** A private runtime function runs a batch of states, parameters, keys, and schedules
  of one compiled problem with a solver that has an accelerated loop, and returns one trajectory per run. The JAX
  backend batches its loops with `jax.vmap`. The Numba backend compiles its loops with `nogil=True` and runs them on a
  thread pool, each run with its own generator. The NumPy backend loops over the runs, as the reference.
- **`vectorized` for every experiment of runs.** `ensemble`, `sweep`, and `compare_scenarios` gain
  `vectorized=False` and `batch_size=None`, and `basin_stability` and `orbit_diagram` keep `vectorized` and gain
  `batch_size`. With `vectorized=True`, runs that share a compiled structure run in batches, and the result equals the
  result of the runs one by one on the same backend.
- **Memory-aware batch sizes.** `batch_size=None` sizes a batch from the bytes one run needs, its state, parameters,
  loop workspace, and kept frames, and from the free memory of the device, or of the host for the CPU.
- **Every kind of system.** Single elements, groups, child systems, fields, and reaction networks batch, since the batch
  never rebuilds the system. Lattices, networks, and agents in their vectorised form batch once
  `add-large-system-kernels` gives their solvers loops. A solver without an accelerated loop, such as a SciPy integrator
  or Mesa, runs its runs one by one, and the experiment warns with `BackendWarning`, which names the solver and the
  solvers that batch, so a fallback is reported and never silent.
- **A benchmark gate.** Benchmarks time every experiment of runs with and without batches, on every backend and at
  small and large run counts. Batched runs must never lose before they become the default.
- **BREAKING: `vectorized=True` becomes the default** of `ensemble`, `sweep`, `compare_scenarios`, `basin_stability`,
  and `orbit_diagram` once the gate passes, as the last task of the change. Seeded results stay identical on the NumPy
  and Numba backends and within 1e-10 on the JAX backend, but users see the fallback warning for systems that cannot
  batch, and batches hold more memory at once. `vectorized=False` keeps runs one by one.
- **BREAKING (private only): `_replicate` is removed.** The replication module and its tests go, and the tests that
  imported it read batched runs through the public experiments instead.

## Capabilities

### New Capabilities

- `batched-runs`: runs of one compiled problem over a leading axis of states, parameters, and keys, on every backend.

### Modified Capabilities

- `experiments`: `vectorized` and `batch_size` for every experiment of runs, results equal to the unbatched runs,
  basin stability of systems with groups, the reported fallback, and batched runs as the default.

## Impact

- Core: a new private module `core/engine/runtime/_batches.py`. `core/engine/backends/_jax.py` and
  `core/engine/backends/_numba.py` gain the batched loops. `core/experiments/_runs.py` groups tasks into batches.
  `core/experiments/_replicate.py` and `tests/core/experiments/test_replicate.py` are removed.
- Dependencies: none.
- Docs: a batched runs section in the run experiments page of the user guide, and the changelog entry of the new
  default.
- Drafts that gain from it without changes: `add-state-estimation`, whose ensemble and particle filters advance many
  members, `add-learning-foundation`, whose `train_policy` runs rollouts through the runner of `ensemble`,
  `add-off-policy-learning`, `add-diversity-search`, and `add-experimental-design`, whose batches go through the same
  runner.
- Order: after `add-device-execution`, whose loops it batches. Before `add-multi-device-sharding`, which splits a batch
  over devices.
