# Spec Delta

## Purpose

Lets a user run large systems with a space on a device: lattices, fields on lattices, particles and flocks with
neighbour search, networks, explicit fields on meshes, and agent groups whose laws are symbolic, with observables
reduced on the device.

## ADDED Requirements

### Requirement: Lattices run on the device
The JAX backend SHALL lower reads of `LatticeNeighborhood` to gathers from a table of neighbour indices and SHALL fill
a `lattice` loop of the accelerated loops of the problem, which `LatticeSolver` reads for update laws on a lattice and
`MetropolisSolver` reads for its sweeps, for the schedules `'synchronous'`, `'checkerboard'`, and `'colored'`. The
schedules `'random-sequential'` and `'random-site'` SHALL raise `BackendError`, whose message names the parallel
schedules.

#### Scenario: Ising magnetisation
- **WHEN** `IsingSystem` of 256 by 256 sites at temperature 2 with `backend=JaxBackend(device='cpu')` runs 2000
  checkerboard sweeps of `MetropolisSolver` over 20 seeds, after 500 sweeps of warm-up
- **THEN** the mean absolute magnetisation per site is within three standard errors of Onsager's value 0.911

#### Scenario: Random sequential updates
- **WHEN** a lattice system whose local rule has `schedule='random-sequential'` and `backend=JaxBackend()` is prepared
- **THEN** it raises `BackendError`, whose message names `'checkerboard'` and `'colored'`

### Requirement: Neighbours are found on the device
The JAX backend SHALL lower reads and reductions of `RadiusNeighborhood`, with or without a cone, and of
`NearestNeighborhood` to a cell list on the device, in periodic and bounded boxes, and a cell that overflows its
capacity SHALL raise `BackendError` after the frame, whose message names `capacity`.

#### Scenario: Same neighbours as the Numba search
- **WHEN** the radius, cone, and nearest neighbours of 10000 members at random, clustered, and periodic positions are
  found on the JAX backend and on the Numba backend
- **THEN** both give the same sets of neighbours

#### Scenario: Overflow
- **WHEN** a flock whose members gather into one cell runs with `backend=JaxBackend(device='cpu', capacity=8)`
- **THEN** it raises `BackendError`, whose message names `capacity`

#### Scenario: Ordered flock
- **WHEN** `VicsekSystem` of 4000 particles at density 4 and noise 0.1 with `backend=JaxBackend(device='cpu')` runs
  500 steps
- **THEN** its polarisation exceeds 0.9

### Requirement: Networks and meshes run as sparse sums
The JAX backend SHALL lower reads and reductions of `AdjacencyNeighborhood` and the explicit operators of fields on
a `CellComplexSpace` mesh to sums over edge lists, in memory linear in the edges, and implicit steppers on meshes SHALL
raise
`BackendError`, whose message names the NumPy and Numba backends.

#### Scenario: Same drift as the Numba backend
- **WHEN** a quenched mean field SIS system on a Barabási-Albert network of 100000 nodes is evaluated on the JAX and
  Numba backends at the same state
- **THEN** the drifts agree within 1e-12 relative error in double precision

#### Scenario: Epidemic threshold
- **WHEN** the same system runs to time 200 on the JAX backend with the infection rate at half and at twice the
  threshold `1 / Lambda`
- **THEN** the prevalence falls below 1e-3 in the first run and stays above 1e-2 in the second

### Requirement: Symbolic agent laws run on the device
The JAX backend SHALL lower update laws on the members of a group whose values hold categorical, Bernoulli, uniform, or
normal draws to draws per member, and SHALL raise `BackendError` for an update law whose value is a Python rule,
`rule(agent, system)` or `rule(system)`, whose message names symbolic update laws.

#### Scenario: Transitions in distribution
- **WHEN** an element of 100000 members with an update law that draws one of two states runs 50 steps on the JAX and
  NumPy backends
- **THEN** the fractions in each state agree within three standard errors

#### Scenario: Agent rule
- **WHEN** a system whose agents move by an update law with the rule `rule(agent, system)` is prepared with
  `backend=JaxBackend()`
- **THEN** it raises `BackendError`, whose message names symbolic update laws

### Requirement: Large systems are benchmarked against the Numba backend
The benchmarks SHALL time large systems with a space on the JAX backend against the Numba backend on the same machine.

#### Scenario: Large Ising lattice
- **WHEN** `IsingSystem` of 4096 by 4096 sites runs 100 checkerboard sweeps on the Numba backend and on the JAX
  backend, in a test marked `benchmark`
- **THEN** the run on the CPU device takes at most twice the Numba time, and the run on a GPU, marked `gpu`, at most a
  tenth

#### Scenario: Million-particle flock
- **WHEN** a Vicsek system of 1000000 particles at density 1 runs 100 steps on both backends, in a test marked
  `benchmark`
- **THEN** the run on the CPU device takes at most three times the Numba time, and the run on a GPU, marked `gpu`, at
  most a fifth

#### Scenario: Million-node network
- **WHEN** a quenched mean field SIS system on a Barabási-Albert network of 1000000 nodes runs 100 RK4 frames on both
  backends, in a test marked `benchmark`
- **THEN** the run on the CPU device takes at most twice the Numba time, and the run on a GPU, marked `gpu`, at most a
  fifth
