## Why

Huge single runs matter to the user as much as many runs: millions of agents, particles, lattice sites, network nodes,
and field cells. After `add-device-execution` the JAX backend runs whole frames on a GPU, but only for systems without
a space. Every large system of the library has one:

- Lattices, such as the Ising system and the contact process, and fields on lattices, such as Gray-Scott, read their
  neighbours through `LatticeNeighborhood`.
- Particles and flocks, such as the Lennard-Jones fluid and the Vicsek system, find their neighbours within a radius,
  in a cone, or among the nearest, which the Numba backend does with cell grids in `_numba_neighbours.py`.
- Networks, such as the quenched mean field SIS system and Kuramoto oscillators on networks, sum over the edges of
  graphs with millions of nodes.
- Agent groups in their vectorised form compile to maps, transitions, and samples with neighbour reductions, which are
  the only way to run millions of agents, since Mesa steps agents one by one in Python.

A large run also copies its whole state to the host at every recorded frame, even when the user needs only an order
parameter. This change lowers spaces, neighbourhoods, networks, and the vectorised agent laws to JAX kernels, and keeps
observables on the device.

## What Changes

- **Lattices on the device.** `LatticeNeighborhood` reads lower to gathers from a table of neighbour indices, and a new
  loop `lattice` of `AcceleratedSpec` runs the local rules, array updates, and checkerboard Monte Carlo updates of
  `LatticeSolver` and `MonteCarloSolver` on the device, with one key per step.
- **Neighbour search on the device.** `RadiusNeighborhood`, with its cone, and `NearestNeighborhood` lower to a JAX
  cell list: members sorted by cell, a fixed number of slots per cell, and candidate pairs from the cells around each
  member, rebuilt every step inside the loop. A cell that overflows its slots raises `BackendError` after the frame,
  naming the setting that raises the capacity.
- **Networks and meshes as sparse sums.** `AdjacencyNeighborhood` reads and reductions over edges lower to
  `jax.ops.segment_sum` over the edge lists of the space, so a network with millions of nodes runs in memory linear in
  its edges. The explicit operators of fields on meshes lower the same way from the sparse matrices scikit-fem
  assembles.
- **Agents in their vectorised form.** `TransitionLaw` and `SampleLaw` lower to categorical draws per member, so agent
  interactions in their symbolic form run on the device. Rule and match laws, which need sequential random orders, raise
  `BackendError` that names the vectorised form.
- **Reductions that stay on the device.** Observables are traced and evaluated on the device after every frame.
  `simulate` gains `record_states=True`. With `record_states=False` a run copies only the observables to the host, and
  experiments whose outputs are all observables set it themselves.
- **Probes.** The Ising system on a lattice of 4096 by 4096, the Vicsek system of 1000000 particles, the Lennard-Jones
  fluid of 1000000 particles, the quenched mean field SIS system on a Barabási-Albert network of 1000000 nodes, and
  Gray-Scott on a lattice of 2048 by 2048, each against the Numba backend on the same machine.

## Capabilities

### New Capabilities

- `device-kernels`: lattices, neighbour search, networks, meshes, and vectorised agent laws lowered to device kernels,
  with observables reduced on the device.

### Modified Capabilities

- `simulation`: runs that record only observables.

## Impact

- Core: new private module `core/engine/backends/_jax_neighbours.py`, the JAX counterpart of `_numba_neighbours.py`.
  `core/engine/backends/_jax.py` and `_jax_loops.py` gain the lowerings and the `lattice` loop. `AcceleratedSpec` in
  `core/specs/_accelerated_spec.py` gains `lattice`. `core/engine/runtime/_simulate.py` and the recorder gain
  `record_states`. `LatticeSolver` and `MonteCarloSolver` read the `lattice` loop when a backend fills it.
- Dependencies: none. The cell list is written on `jax.numpy`, after the survey in the design.
- Docs: a large systems section in the simulate page of the user guide.
- Order: after `add-device-execution`. In parallel with `add-batched-runs`. Before `add-multi-device-sharding`, which
  splits these kernels over devices.
- Drafts: `add-neural-particle-simulators` reuses the neighbour search of the Numba backend for its graph networks on
  the CPU, and its design keeps GPUs out. The JAX cell list here can serve its rollouts later without a change to it.
