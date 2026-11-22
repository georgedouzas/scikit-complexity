## ADDED Requirements

### Requirement: Fixed points come in a fixed order
`stability` SHALL return fixed points sorted by their values in the order of the state names, the same in every
process, and exact solutions SHALL convert to floats without solver noise.

#### Scenario: Logistic map
- **WHEN** the fixed points of the logistic map are found in two fresh processes
- **THEN** both return the same points in the same order, and the point at 0 is exactly 0.0

### Requirement: Aperiodic orbits have period 0
`orbit_diagram` SHALL report a period of 0 for a parameter value whose kept values never repeat within the rounding.

#### Scenario: Chaotic logistic map
- **WHEN** an orbit diagram of the logistic map keeps 64 values at r equal to 3.9
- **THEN** its period is 0, while r equal to 3.2 gives 2

### Requirement: Lattices coarse grain over blocks
`coarse_grain` SHALL coarse grain a lattice configuration over blocks of a given side, by majority with a tie rule or
by mean, for one snapshot or every frame of a trajectory.

#### Scenario: Majority over 3 by 3 blocks
- **WHEN** a 9 by 9 spin lattice coarse grains with `blocks=3` and `rule='majority'`
- **THEN** the result is 3 by 3 and every cell holds the sign of the sum of its block

#### Scenario: Mean over blocks
- **WHEN** a lattice coarse grains with `rule='mean'`
- **THEN** every cell holds the mean of its block

#### Scenario: Blocks that do not divide the lattice
- **WHEN** a 10 by 10 lattice coarse grains with `blocks=3`
- **THEN** an error names the side and the block size

### Requirement: Orbit diagrams scan many values quickly
`orbit_diagram` SHALL scan the logistic map over 1200 parameter values with 1000 transient steps in a few seconds.

#### Scenario: Scan time
- **WHEN** the scan runs on a laptop
- **THEN** it finishes in under 10 seconds
