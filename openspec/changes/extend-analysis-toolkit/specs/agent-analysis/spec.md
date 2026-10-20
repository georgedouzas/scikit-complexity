# Spec Delta

## Purpose

Measures agent snapshots, positions, velocities, headings, phases, and categorical states, in continuous, grid, and
periodic spaces: order parameters, velocity and energy laws, neighbour statistics, displacement, and segregation.

## ADDED Requirements

### Requirement: Order parameters of headings and phases
The analysis layer SHALL compute the polar and nematic order of headings or velocities, the milling order of positions
and velocities about their centre, and the complex order parameter of phases, for one snapshot or every frame.

#### Scenario: Vicsek ordering
- **WHEN** a user computes the polarization of 1000 Vicsek agents after relaxation at noise 0.1 and at noise 5
- **THEN** the low noise gives above 0.9 and the high noise below `3 / sqrt(1000)`

#### Scenario: Same order parameter on oscillators and headings
- **WHEN** a user computes the complex order parameter of Kuramoto phases at twice the critical coupling, and of
  Vicsek headings at noise 0.1
- **THEN** both moduli exceed 0.6, and each equals the polarization of unit vectors at those angles within 1e-12

#### Scenario: Milling
- **WHEN** a user computes the milling and polar orders of a D'Orsogna swarm started turning one way, in its mill state
- **THEN** the milling order exceeds 0.8 and the polar order lies below 0.2

### Requirement: Velocity and energy laws of agents
The analysis layer SHALL compute the kinetic temperature of agent velocities in any dimension, and compare the speed
distribution with the Maxwell-Boltzmann law of that dimension at that temperature by a goodness-of-fit test.

#### Scenario: Maxwell-Boltzmann in two and three dimensions
- **WHEN** a user compares the speeds of a two-dimensional and a three-dimensional ideal gas after relaxation with the
  law of their kinetic temperatures
- **THEN** neither rejects the law at the 0.05 level, and the kinetic temperature is constant over frames within 1e-10

#### Scenario: Langevin particles
- **WHEN** a user computes the kinetic temperature of 2000 underdamped Langevin particles with bath temperature 1.5
- **THEN** it lies within 3 percent of 1.5 and the speeds do not reject the Maxwell-Boltzmann law

### Requirement: Neighbour statistics of point sets
The analysis layer SHALL compute nearest-neighbour distance distributions, the Clark-Evans ratio, and the radial
distribution function of positions in open or periodic boxes, and SHALL coarse-grain agents into density and
polarization fields for field analysis.

#### Scenario: Random and ordered positions
- **WHEN** a user computes the Clark-Evans ratio of 2000 uniform random points and of a triangular lattice of points
- **THEN** the random points give 1 within 0.05 and the lattice gives above 2

#### Scenario: Radial distribution of a fluid
- **WHEN** a user computes the radial distribution function of an ideal gas and of a dense Lennard-Jones fluid
- **THEN** the gas gives 1 within 0.05 at every distance, and the fluid has its first peak within 5 percent of
  `2^(1/6) sigma`

#### Scenario: Coarse-grained density
- **WHEN** a user coarse-grains 10,000 agents into a 32 × 32 density field
- **THEN** the field sums to 10,000, and its structure factor comes from the same function as for a lattice

### Requirement: Displacement of agents
The analysis layer SHALL compute mean squared displacements of agents from unwrapped positions over all lags, their
scaling exponent over a chosen lag range, and the diffusion coefficient in the diffusive range.

#### Scenario: Diffusion on two kinds of space
- **WHEN** a user computes the mean squared displacement of 1000 Brownian particles in a periodic box and of 1000 random
  walkers on a periodic grid
- **THEN** both exponents lie within 0.05 of 1, and neither shows a plateau at the box size

#### Scenario: Active particles
- **WHEN** a user computes the mean squared displacement of two-dimensional active Brownian particles with speed `v`
  and rotational diffusion `D_r`
- **THEN** the exponent lies within 0.1 of 2 for lags below `0.1 / D_r` and within 0.1 of 1 above `10 / D_r`, with the
  long-time diffusion coefficient within 10 percent of `v^2 / (2 D_r)`

### Requirement: Segregation of categorical states
The analysis layer SHALL compute the dissimilarity index, the mean share of like neighbours, and the interface density
of categorical states, of agents on grids or networks and of lattice fields.

#### Scenario: Same segregation index on two domains
- **WHEN** a user computes the mean share of like neighbours of a Schelling model after relaxation with tolerance 0.5,
  and of an Ising lattice quenched to temperature 1
- **THEN** both exceed 0.75, while a random arrangement of the same composition gives within 0.02 of the global share
