## Why

The user targets HPC clusters, and some clusters have many CPU nodes and no GPUs. A lattice or a field on a lattice
larger than the memory of one machine can then only run if its sites are split over the nodes, each node holding a
slab and exchanging a halo of boundary sites with its neighbours at every step. This is domain decomposition with MPI,
the standard method of lattice and field codes on CPU clusters.

`add-multi-device-sharding` already splits lattices over GPUs and nodes through multi-controller JAX. This change covers
the same need for the NumPy and Numba backends on CPU clusters. It is the least urgent of the changes for massive
simulations: GPU clusters are covered by sharding, particle systems by the LAMMPS backend, and many-run workloads by the
cluster executors. It is drafted last for that reason and can wait until a user needs a CPU-only run larger than a
machine.

## What Changes

- **Decomposed runs.** `simulate` gains `communicator=None`, an `mpi4py` communicator. With one, every rank of an MPI
  run, started by `mpiexec` or `srun`, simulates the slab of every lattice along its first axis, and the run is
  the run of the whole system.
- **Slabs that compile alone.** Every rank compiles the system with the shape of its slab plus a halo as wide as the
  reach of the neighbourhoods, so no rank builds the whole lattice. Initial values that are numbers or functions of the
  site are evaluated per slab, and full arrays are sliced.
- **Halo exchange.** Before every evaluation of a drift or an update, and once per colour of checkerboard and colored
  schedules, every rank sends its boundary rows to its neighbours and receives their rows into its halo with
  `Sendrecv`, wrapping around for periodic boundaries. Steppers then run one internal step at a time, since the Numba
  loops cannot call MPI.
- **Observables and frames.** Observables that are sums, means, minima, or maxima over lattice sites reduce over the
  ranks with `Allreduce`, so every rank holds them. Other observables of lattice sites raise `ValidationError`. Frames
  of states go to the store of `add-multi-device-sharding`, each rank writing its slab, or stay with each rank as the
  frames of its slab.
- **Dependency.** `mpi4py` becomes a runtime dependency. `mpich`, already a runtime dependency for LAMMPS, provides the
  MPI library and `mpiexec`.

## Capabilities

### New Capabilities

- `domain-decomposition`: runs of lattices and fields on lattices split over MPI ranks with halo exchange.

### Modified Capabilities

None.

## Impact

- Core: new private module `core/engine/runtime/_decomposition.py` with the slabs, the halo exchange, and the reductions
  of observables. `core/engine/runtime/_simulate.py` takes `communicator`. The lattice compile pass takes the shape of a
  slab.
- Dependencies: `mpi4py>=4.1.0` (4.1.2, 2026-05-16, BSD-3-Clause).
- Tests: runs under `mpiexec -n 4` of the installed MPICH in a subprocess, marked slow.
- Docs: a domain decomposition section in the simulate page of the user guide, with an `mpiexec` and an `srun` recipe.
- Order: last of the changes for massive simulations, after `add-multi-device-sharding`, whose store it reuses. The
  least urgent.
