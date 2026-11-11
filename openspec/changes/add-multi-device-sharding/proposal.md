## Why

The user targets a multi-GPU node and GPU clusters as well as one GPU, for many runs and for huge single runs alike.
After `add-device-execution`, `add-batched-runs`, and `add-large-system-kernels`, a batch of runs or a large system
runs on one device. Two limits remain:

- A batch can use only one GPU of a node, so a node of eight GPUs runs at the speed of one unless the user splits the
  experiment by hand.
- A single system must fit in the memory of one device. A lattice of 16384 by 16384 sites in double precision with its
  workspace, or a flock of tens of millions of agents, does not.

JAX shards arrays over the devices of a node and, with one process per node, over the devices of a cluster, and its
random keys give the same draws whatever the sharding. This change spreads batches and large systems over several
devices and nodes, and streams the frames of runs larger than one host to a Zarr store.

## What Changes

- **Several devices.** `JaxBackend(device=...)` also takes a sequence of devices, such as `jax.devices('gpu')`. A
  backend with several devices builds a one-axis `jax.sharding.Mesh` over them.
- **Batches over devices.** A batch of `add-batched-runs` on several devices splits its run axis over them. Runs do not
  talk to each other, so the batch needs no communication, and its results equal those on one device.
- **One system over devices.** A single run on several devices splits the member axis of every group of at least 1000
  members over them, and keeps smaller parts on every device. Lattices split by rows and exchange one row of halo with
  their neighbours by `jax.lax.ppermute` inside `jax.shard_map`. Networks, meshes, and particles leave the
  communication of their gathers and sums to the sharding of XLA, which gathers the members a device needs.
- **Several nodes.** With one process per node, started by `srun`, `mpirun`, or the Ray and Dask recipes, and
  `jax.distributed.initialize()` called by the user in each, `jax.devices()` lists the devices of every node and the
  same backend spans the cluster. Every process runs the same script, as multi-controller JAX requires.
- **Frames of runs larger than a host.** `simulate` gains `store=None`. With a store each process writes the shards of
  the states it holds to a Zarr array per state as frames are recorded, and the trajectory reads them from the store.
  Observables are reduced on the devices and copied to every process.
- **Same draws on any sharding.** The keys of `add-device-execution` are split by the partitionable threefry generator
  of JAX, so a sharded run draws the same numbers as the run on one device.

## Capabilities

### New Capabilities

- `sharded-runs`: batches of runs and single systems split over the devices of a node and of a cluster.

### Modified Capabilities

- `simulation`: trajectories that stream their frames to a Zarr store.

## Impact

- Core: new private module `core/engine/backends/_jax_sharding.py` with the mesh, the shardings of states and
  parameters, and the halo exchange of lattices. `core/engine/backends/_jax.py` and `_jax_loops.py` place their inputs
  with those shardings. `core/engine/runtime/_simulate.py` and the recorder gain `store`, through the Zarr writer of
  `add-cluster-executors`.
- Dependencies: none new. `jax` and `zarr`, the latter added by `add-cluster-executors`.
- Tests: emulated CPU devices through `XLA_FLAGS=--xla_force_host_platform_device_count=4` in CI, two local processes
  with the gloo collectives of the CPU for the multi-process tests, and `gpu` tests for real GPUs.
- Docs: a section on several devices and nodes in the simulate page of the user guide, with SLURM and Open MPI launch
  recipes that the tests do not run.
- Order: after `add-batched-runs`, `add-large-system-kernels`, and `add-cluster-executors`, whose store it reuses.
