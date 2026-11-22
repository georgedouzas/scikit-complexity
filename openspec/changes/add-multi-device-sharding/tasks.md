# Tasks

## 1. Prerequisites

- [ ] 1.1 Check that `add-device-execution`, `add-batched-runs`, `add-large-system-kernels`, and
  `add-cluster-executors` are applied, and stop if one is not
- [ ] 1.2 Add a fixture in `tests/core/engine/backends/conftest.py` that runs a test body in a subprocess with
  `XLA_FLAGS=--xla_force_host_platform_device_count=4`, and verify `jax.devices('cpu')` lists four devices in it

## 2. Several devices

- [ ] 2.1 Accept a sequence of devices as `device` of `JaxBackend`, build `mesh_` in
  `core/engine/backends/_jax_sharding.py`, and verify the emulated-devices scenario in
  `tests/core/engine/backends/test_jax_sharding.py`
- [ ] 2.2 Check the devices against `jax.process_count()`, and verify the devices-of-one-process-in-a-cluster-run
  scenario with two local processes

## 3. Batches over devices

- [ ] 3.1 Pad and shard the run axis of a batch over the mesh, drop the padded runs, scale the automatic batch size by
  the device count, and verify the ensemble-on-four-devices scenario

## 4. One system over devices

- [ ] 4.1 Shard groups of at least 1000 members along their members from the layout of the problem, replicate the rest,
  and verify a Kuramoto run on four devices equals the run on one device within 1e-10
- [ ] 4.2 Split lattices by rows and exchange halos with `jax.lax.ppermute` inside `jax.shard_map` in the `lattice`
  loop, once per colour, and verify the lattice-on-four-devices scenario
- [ ] 4.3 Run network, mesh, and particle gathers and sums with sharded inputs, and verify the
  network-and-flock-on-four-devices scenario
- [ ] 4.4 Reduce observables across devices into replicated scalars, and verify that a run with `record_states=False`
  on four devices copies one value per observable and frame to the host

## 5. Several nodes

- [ ] 5.1 Verify the two-CPU-processes scenario with `jax.distributed.initialize` on a local coordinator and the gloo
  collectives, marked slow
- [ ] 5.2 Gather the addressable shards of every process into the trajectory of a sharded run without a store, and
  verify both processes of the two-process test hold the full spins of the last frame

## 6. Stores of trajectories

- [ ] 6.1 Add `store` to `simulate` and the recorder through the Zarr writer of `add-cluster-executors`, read states of
  a stored trajectory from the store, and verify the run-on-one-device-with-a-store scenario in
  `tests/core/engine/runtime/test_simulate.py`
- [ ] 6.2 Write the shards of every process at every frame, and verify the sharded-run-with-a-store scenario

## 7. Benchmarks

- [ ] 7.1 Add the scaling benchmarks of the ensemble over GPUs, of the weak scaling of a lattice, and of the strong
  scaling of the SIS network of 10000000 nodes, marked `benchmark` and `gpu`, each skipped with the reason "needs two
  GPUs or more, found N" on fewer GPUs, and verify the skip reason on a machine with one GPU or none
- [ ] 7.2 Check that the job `gpu` of `.github/workflows/gpu.yml` from `add-device-execution` runs these benchmarks
  through `nox -s gpu`, so they gate releases when the runner holds several GPUs, and verify their targets in a manual
  run of `gpu.yml` on a multi-GPU runner, or record in the design that none exists yet
- [ ] 7.3 Write in the release section of `CONTRIBUTING.md` that the multi-GPU targets need a runner with two GPUs or
  more and that the multi-node targets run on demand, with the command that runs them on a SLURM allocation, and
  verify the command matches the launch recipe of task 8.2

## 8. Documentation

- [ ] 8.1 Add a several devices and nodes section to `docs/overview/user_guide/simulate.md` with a sequence of devices,
  what splits and what is copied, `store`, and `jax.distributed.initialize`, with a runnable example on emulated CPU
  devices, and verify the markdown tests run it
- [ ] 8.2 Add launch recipes for SLURM with `srun`, for Open MPI with `mpirun`, and for a Ray cluster, as code that the
  tests do not run, and the pointer to the LAMMPS backend for the largest particle systems, and verify the docs build

## 9. Integration

- [ ] 9.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules with the thread
  limits set to 1, doctests, and the strict docs build) and verify every session passes
- [ ] 9.2 Run `openspec validate add-multi-device-sharding --strict` and verify it passes
