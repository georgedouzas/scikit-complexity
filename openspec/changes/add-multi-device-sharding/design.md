## Context

- After `add-device-execution`, `JaxBackend(device=...)` runs a problem on one device with loops on the device and keys
  derived from the random state of the run.
- After `add-batched-runs`, a batch of runs is a `jax.vmap` of those loops over a leading run axis.
- After `add-large-system-kernels`, lattices gather from neighbour tables, particles search cell lists, networks sum
  over edge lists, and observables are reduced on the device.
- After `add-cluster-executors`, `core/experiments/_stores.py` writes chunks of results to Zarr.
- JAX 0.12 offers `jax.sharding.Mesh` and `NamedSharding`, `jax.shard_map` with `jax.lax.ppermute` for explicit
  communication, `jax.distributed.initialize` for one process per node with automatic detection of SLURM and Open MPI,
  the gloo collectives for several CPU processes, and the partitionable threefry generator, on by default, which gives
  the same random bits whatever the sharding. The installed version confirms all of them.
- The LAMMPS backend already runs particle systems over MPI ranks, with the domain decomposition of LAMMPS.

## Goals / Non-Goals

**Goals:**
- Batches of runs over the devices of a node and of a cluster.
- One system larger than a device over several devices and nodes.
- Frames of runs larger than one host written to a store.
- The same results as on one device.

**Non-Goals:**
- Spatial domain decomposition of particles across devices. Particles gather the positions they need through XLA,
  which costs one all-gather of positions per step. LAMMPS already decomposes particle domains over MPI for the systems
  it runs, and the docs point to it for the largest particle runs.
- Automatic choice of devices. The user passes them.
- Starting the processes of a cluster. `srun`, `mpirun`, Ray, or Dask start them, as the docs show.
- Checkpoint and resume, which the user did not ask for.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| Mesh and shardings of states, parameters, and batches | 2, a backend lowering the engine runs on any problem | new private module `core/engine/backends/_jax_sharding.py` | every field with ensembles or large systems: statistical physics, active matter, epidemiology, climate, neuroscience |
| Halo exchange of lattices | 2, a backend lowering of `LatticeNeighborhood` reads | `core/engine/backends/_jax_sharding.py` | statistical physics, chemistry (pattern formation), ecology, cellular automata |
| `store` of `simulate` | 2, an option of the runtime | `core/engine/runtime/_simulate.py`, `_recorder.py` | every field with runs larger than a host |

No new package and no new concept. A set of devices is a setting of the backend, and a store is where a trajectory
lives.

### Dependencies

None new. The table records the libraries surveyed for communication across nodes, checked on PyPI on 2026-11-11.

| Library | Version and date | Licence | Verdict |
| --- | --- | --- | --- |
| jax (multi-controller) | 0.12.0.1, 2026-10-21 | Apache-2.0 | Used. NCCL on NVIDIA, RCCL on AMD, and gloo on CPUs, behind one API |
| mpi4jax | 0.9.1.post2, 2026-09-21 | MIT | Rejected here. MPI calls inside traced code, useful when a cluster has MPI but no NCCL fabric, which `add-domain-decomposition` covers for CPU clusters |

### Several devices

- `device` of `JaxBackend` takes a sequence of `jax.Device` beside the forms of `add-device-execution`. `prepare` builds
  `mesh_`, a `Mesh` of one axis named `'devices'`, and keeps `device_` as the first device for the parts that are not
  sharded.
- Rejected: a separate argument `devices`. Two arguments would say the same thing, and a sequence reads plainly as
  several devices.

### Batches over devices

- A batch on several devices pads its run axis to a multiple of the device count, with padded runs that start from the
  first run and are dropped at the end, and places states, parameters, and keys with
  `NamedSharding(mesh, P('devices'))`. `jit` then runs every device on its share with no communication.
- The memory-aware batch size of `add-batched-runs` multiplies by the device count.

### One system over devices

- The layout of the problem tells which entries of the state vector belong to which group. Every group of at least 1000
  members, the threshold the backends already use, is padded to a multiple of the device count and split along its
  members. Single elements, small groups, and parameters are replicated.
- Lattices split along their first axis. Inside `jax.shard_map`, each device sends its first and last rows to its
  neighbours with `jax.lax.ppermute` before every gather, so a step moves two rows per device instead of the whole
  lattice. Checkerboard and colored schedules exchange halos once per colour.
- Networks, meshes, and particles keep their gathers and `segment_sum` in `jit` with sharded inputs. XLA inserts the
  collectives, an all-gather of the members a device reads. This is correct for every space and costs memory
  proportional to the gathered members. Rejected: explicit partitions of graphs by METIS, which need a host library and
  repartitioning when edges change.
- Observables reduce across devices inside `jit` and come back as replicated scalars, so `record_states=False` runs copy
  a few numbers per frame whatever the device count.

### Several nodes

- Multi-controller JAX: one process per node or per GPU, each calling `jax.distributed.initialize()` before anything
  else, after which `jax.devices()` lists every device of the cluster and `jax.local_devices()` those of the process.
  The call stays with the user, because it starts a connection to a coordinator, an application concern by the
  constitution.
- Every process runs the same script and compiles the same problem. The backend checks that `jax.process_count()` is 1
  or that the devices it got span every process, and raises `BackendError` otherwise, since a mesh over the devices of
  one process in a multi-process run would deadlock the others.
- Recording a frame of a sharded state copies only the addressable shards of a process. Without a store, the trajectory
  of each process holds the whole state gathered with `multihost_utils.process_allgather`, which the docs advise only
  for small runs.

### Stores of trajectories

- `simulate(..., store=None)` takes a path or a Zarr store. With one, the recorder creates one array per state with
  shape `(frames, *shape)` chunked as the shards, and every process writes the shards it holds at every frame. The
  trajectory holds the times and observables in memory and reads states from the store.
- On one device and one process a store works the same way, so a long run of a large system on one GPU fits in host
  memory too.

### Determinism

- The partitionable threefry generator gives every member the same bits on one device and on any sharding, so a sharded
  run equals the run on one device up to the order of sums, within 1e-10.
- Halo exchanges move values, not sums, so a lattice run on several devices equals the run on one device exactly.

### Probes

Speed targets are scaling ratios on the same machine, measured on real GPUs. Emulated CPU devices share one machine's
cores, so in CI they check results only.

Decided by the user on 2026-11-11: a GPU runner gates releases on the GPU tests and benchmarks, through the job `gpu`
of `add-device-execution`, whose runner label is the repository variable `GPU_RUNNER`.

- The tests and benchmarks of several GPUs carry the `gpu` marker and run in that job. Each checks
  `len(jax.devices('gpu'))` and skips with the reason "needs two GPUs or more, found N" when the runner holds fewer,
  so a single-GPU runner shows the skip in the test report rather than passing it in silence.
- When the runner holds two GPUs or more, their results and scaling targets gate releases like the other GPU tests.
  The maintainer chooses the runner, so a release on a single-GPU runner proceeds with the multi-GPU skips listed.
- Multi-node targets, the two-node runs of `jax.distributed.initialize` on GPUs, stay on demand unless a multi-node
  runner exists. CI keeps checking the multi-process path with two local CPU processes.

- Batch: the ensemble of 100000 stochastic SIR runs of `add-batched-runs` on the GPUs of one node runs at least 0.8
  times the GPU count faster than on one of them.
- Weak scaling of one system: an Ising system of 8192 by 8192 sites on one GPU, and of 8192 times the GPU count by 8192
  on all of them, 100 checkerboard sweeps each. The second takes at most 1.25 times the first.
- Strong scaling of a network: the quenched mean field SIS system on a Barabási-Albert network of 10000000 nodes on one
  GPU and on four takes on four at most half the time on one.
- Results: on four emulated CPU devices, the ensemble, the Ising run, the SIS run, and a Vicsek run equal their runs on
  one device, exactly for the lattice and within 1e-10 otherwise.

## Risks / Trade-offs

- [XLA gathers for particles and networks cost memory and bandwidth] → Correct for every space, and the probes measure
  the cost. Explicit partitions can follow for spaces whose probes miss their targets.
- [A multi-process script that diverges between processes deadlocks] → The backend checks the devices against the
  processes and raises before compiling. The docs state that every process must run the same code.
- [No multi-GPU runner in CI] → Emulated CPU devices and two local CPU processes test the results in CI. The multi-GPU
  targets run in the release GPU job when its runner holds several GPUs and are skipped with their reason otherwise,
  and the multi-node targets run on demand.
- [Padding adds work] → At most one member or run per device.

## Migration Plan

Nothing breaks. A single device behaves as in `add-device-execution`, and `store=None` keeps trajectories in memory.

## Open Questions

None. The user decided on 2026-11-11 that the GPU runner gates releases, multi-GPU targets included when the runner
holds several GPUs, and that multi-node targets stay on demand unless a runner exists.
