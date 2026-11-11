## Context

- `LatticeSpace` has a shape and a boundary, `'periodic'`, `'fixed'`, or `'reflecting'`, and `LatticeNeighborhood`
  gives the offsets of the neighbours of a site. Groups and fields compile independently of their size.
- The Numba backend writes lattice stencils in `_numba_stencils.py` and runs the steps of a frame inside compiled loops,
  which cannot call MPI. The forward functions in `functions_` evaluate one drift or update per call.
- `mpich` is a runtime dependency, because the LAMMPS wheel links an MPI library it does not bundle. It installs
  `mpiexec`.
- `add-multi-device-sharding` splits lattices over GPUs and nodes with halo exchange in JAX, and adds `simulate(...,
  store=...)`, which writes the shards of every process to Zarr.
- The LAMMPS backend decomposes particle domains over MPI ranks for the systems it runs.

## Goals / Non-Goals

**Goals:**
- Lattices and fields on lattices larger than one machine on CPU clusters, with the NumPy and Numba backends.
- The same results as a run on one rank for deterministic systems, and the same distributions for stochastic ones.
- No rank holds the whole lattice.

**Non-Goals:**
- Particles, networks, and meshes. Particles run over MPI through the LAMMPS backend, and networks and meshes need graph
  partitioning, which can follow if a user needs it.
- Implicit and spectral steppers, whose global solves and FFTs need distributed linear algebra.
- Decomposition along more than one axis. Slabs along the first axis suffice up to hundreds of ranks for lattices of
  tens of thousands of rows.
- GPU clusters, which `add-multi-device-sharding` covers.
- Checkpoint and resume, which the user did not ask for.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `communicator` of `simulate`, slabs, halo exchange, and reductions of observables | 2, a runtime procedure the engine runs on any lattice problem | new private module `core/engine/runtime/_decomposition.py` | statistical physics (Ising, Potts, contact process), chemistry (reaction-diffusion), ecology (spatial populations), climate (gridded models), materials science (phase field) |
| Slab shape of the lattice compile pass | 2, a compile pass of lattices | `core/engine/compile/passes/_lattices.py` | the same fields |

No new package. A communicator is a setting of a run, as a clock is.

### Dependencies

Checked on PyPI on 2026-11-11.

| Library | Version and date | Licence | Status |
| --- | --- | --- | --- |
| mpi4py | 4.1.2, 2026-05-16 | BSD-3-Clause | Added as `mpi4py>=4.1.0`. The MPI bindings of Python, built against the MPICH that `mpich` installs |
| numba-mpi | 1.3.1, 2026-05-27 | GPL-3.0-only | Rejected. It calls MPI from compiled loops, which would keep the Numba loops of a frame, but the halo exchange of a large slab costs far more than the Python call per step it saves, and it adds a second copyleft licence |
| mpi4jax | 0.9.1.post2, 2026-09-21 | MIT | Rejected. The JAX backend uses the collectives of multi-controller JAX in `add-multi-device-sharding` |
| mpi4py-fft | 2.0.6, 2024-05-13 | BSD-2-Clause | Rejected. Spectral steppers are out of scope, and its last release is near the 24 month limit |

`mpi4py` is imported by `skcomplex.core.engine`, so it is a runtime dependency by the rule of the Project Profile.

### Slabs

- With `communicator`, rank `r` of `n` owns the rows `[r * N // n, (r + 1) * N // n)` of every lattice of `N` rows.
  Every lattice of the system must have the same number of rows, and a system whose lattices differ raises
  `ValidationError`.
- Every rank compiles the system with each lattice replaced by its slab plus `h` halo rows on each side, where `h` is
  the largest reach of its neighbourhoods along the first axis. The halo rows are states that no law writes, so the
  compiled problem is the problem of a smaller lattice with a fixed boundary that the exchange fills.
- Initial values that are numbers, or functions of the coordinates of a site, are evaluated on the rows of the slab. A
  full array is sliced, which needs the array to fit on every rank, and the docs say so.

### Halo exchange

- The rank wraps the drift and update functions of the problem: `exchange(x)` then the local function. `exchange` sends
  the first and last `h` owned rows to the ranks above and below with `Sendrecv` and writes the received rows into the
  halo. A periodic lattice wraps from the last rank to the first, and fixed and reflecting boundaries fill the outer
  halos of the first and last ranks as the lattice does.
- Solvers run one internal step per call of the forward functions, so the exchange happens before every stage of RK4
  and every colour of a Monte Carlo sweep. The accelerated loops are skipped when a communicator is set.
- Rejected: halos exchanged once per frame with a halo as wide as the steps of a frame. It saves messages but grows the
  halo with the frame and gives wrong values for implicit couplings.

### Observables and frames

- Every observable that is a sum, a mean, a minimum, or a maximum over the sites of a lattice reduces locally and then
  over the ranks with `Allreduce`. An observable of single elements is evaluated on every rank alike. Any other
  observable of lattice sites raises `ValidationError` that names the reductions that decompose.
- With the `store` of `add-multi-device-sharding`, every rank writes its owned rows of every frame, and the store holds
  the whole lattice. Without a store, each rank keeps the frames of its own slab in its trajectory.

### Random streams

- Rank `r` draws from the child `r` of `SeedSequence(random_state).spawn(n)`. A stochastic decomposed run is therefore
  reproducible for a fixed number of ranks, and agrees with the run on one rank in distribution. Rejected: one stream
  per site, which would make the run independent of the number of ranks but costs a counter-based draw per site and step
  in the Numba loops.

### Probes

The targets are ratios measured with `mpiexec` on one machine and, when available, on a cluster. Decided by the user
on 2026-11-11: multi-node targets stay on demand unless a multi-node runner exists. The weak scaling on four ranks of
one machine is the benchmark that runs with the others, and the runs on 16 and 64 ranks run on demand on a cluster.

- `GrayScottSystem` on 4096 by 4096 sites, 100 RK4 frames, deterministic: the run on four ranks equals the run on one
  rank within 1e-12.
- Weak scaling: Gray-Scott on 4096 by 4096 sites on one rank, and on 4096 times the rank count by 4096 on 4, 16, and
  64 ranks of a cluster. The time per frame grows by at most 1.25 times.
- `IsingSystem` on 2048 by 2048 sites at temperature 2, on four ranks: the magnetisation agrees with the run on one rank
  within three standard errors.
- Memory: on four ranks, no rank allocates more than a third of the memory of the run on one rank.

## Risks / Trade-offs

- [MPI tests are slow and flaky to start] → The tests run `mpiexec -n 4` of the installed MPICH in a subprocess, once
  per scenario, marked slow, on CI runners of Linux and macOS.
- [A Python call per internal step] → Each call does work proportional to a slab of millions of sites, so the call
  costs little beside it. The weak scaling probe measures it.
- [Results depend on the number of ranks for stochastic runs] → Stated in the docs and the spec, with agreement in
  distribution.
- [Least urgent] → The change waits until a user needs CPU-only runs larger than a machine. Nothing else depends on it.

## Migration Plan

Nothing breaks. `communicator=None` keeps every run as it is.

## Open Questions

None. The user decided on 2026-11-11 that multi-node targets stay on demand unless a multi-node runner exists.
