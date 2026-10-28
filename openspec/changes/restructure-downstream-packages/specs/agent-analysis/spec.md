## MODIFIED Requirements

### Requirement: Neighbour statistics of point sets
The analysis layer SHALL compute nearest-neighbour distance distributions, the Clark-Evans ratio, and the radial
distribution function of positions in open or periodic boxes, and SHALL coarse-grain agents into density and
polarization fields for field analysis, or over the values of one of their states into a density over those values.

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

#### Scenario: Density over a state
- **WHEN** a user coarse-grains the wealth of 1,000 traders over 20 given wealth values
- **THEN** the densities are the fractions of traders nearest to every value, and they sum to 1
