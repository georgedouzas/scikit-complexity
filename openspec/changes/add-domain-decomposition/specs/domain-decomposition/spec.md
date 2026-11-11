# Spec Delta

## Purpose

Lets a user run a lattice or a field on a lattice larger than one machine on a CPU cluster, split over MPI ranks with
halo exchange, on the NumPy and Numba backends.

## ADDED Requirements

### Requirement: Runs split over MPI ranks
`simulate` SHALL take `communicator`, an `mpi4py` communicator. With one, every rank SHALL compile and run the slab of
every lattice along its first axis with a halo as wide as the reach of its neighbourhoods, exchanged before every
evaluation of a drift or an update and once per colour of a checkerboard or colored schedule, and no rank SHALL build
the whole lattice. A system whose lattices have different numbers of rows SHALL raise `ValidationError`.

#### Scenario: Deterministic field on four ranks
- **WHEN** a Gray-Scott system of 512 by 512 sites runs 50 RK4 frames under `mpiexec -n 4` with
  `communicator=MPI.COMM_WORLD`, in a test marked `slow`
- **THEN** the slabs of the four ranks, put together, equal the run on one rank within 1e-12

#### Scenario: Periodic boundary
- **WHEN** a diffusion field on a periodic lattice runs on four ranks
- **THEN** the field equals the run on one rank within 1e-12, and the rows of the first and last ranks exchange values

#### Scenario: Lattices of different heights
- **WHEN** a system with lattices of 100 and 200 rows runs with a communicator
- **THEN** it raises `ValidationError`, whose message names both shapes

#### Scenario: Memory per rank
- **WHEN** a Gray-Scott system of 4096 by 4096 sites starts on four ranks
- **THEN** no rank allocates more than a third of the memory of the run on one rank

### Requirement: Observables reduce over ranks
In a decomposed run, observables that are sums, means, minima, or maxima over the sites of a lattice SHALL reduce over
the ranks so every rank holds the same values, and other observables of lattice sites SHALL raise `ValidationError`,
whose message names the reductions that decompose.

#### Scenario: Ising magnetisation on four ranks
- **WHEN** an Ising system of 512 by 512 sites at temperature 2 runs 1000 checkerboard sweeps on four ranks over 10
  seeds, in a test marked `slow`
- **THEN** every rank holds the same magnetisation series, and its mean agrees with the run on one rank within three
  standard errors

#### Scenario: Observable that does not decompose
- **WHEN** a decomposed run declares an observable of the site in the middle of a lattice
- **THEN** it raises `ValidationError`, whose message names sums, means, minima, and maxima

### Requirement: Decomposed stochastic runs are reproducible for a rank count
Rank `r` of `n` SHALL draw from child `r` of the `SeedSequence` of the random state spawned into `n` children, so two
decomposed runs with the same random state and number of ranks SHALL be equal.

#### Scenario: Same seed and ranks
- **WHEN** a contact process of 256 by 256 sites runs twice on four ranks with random state 5
- **THEN** both runs are equal

### Requirement: Decomposed runs scale weakly
The benchmarks SHALL time decomposed runs whose sites grow with the number of ranks.

#### Scenario: Weak scaling of a field
- **WHEN** Gray-Scott runs 100 RK4 frames on 4096 by 4096 sites on one rank and on 4096 times four by 4096 sites on
  four ranks of one machine with at least four cores, in a test marked `benchmark`
- **THEN** the time per frame on four ranks is at most 1.25 times the time on one rank
