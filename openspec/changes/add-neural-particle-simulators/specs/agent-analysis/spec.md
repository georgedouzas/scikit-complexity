## ADDED Requirements

### Requirement: Particles are coarse-grained to beads
`coarse_grain` SHALL take `mapping`, a matrix of beads by particles with rows that sum to 1 or a list of groups of
particle indices with optional masses, and SHALL then return the positions of the beads at every frame, the mapped
forces when forces are given, and the box. Periodic positions SHALL be unwrapped within every group before mapping.

#### Scenario: Centres of mass
- **WHEN** a trajectory of 100 dimers is coarse-grained with one group per dimer and equal masses
- **THEN** every bead lies at the midpoint of its dimer, also for dimers split by a periodic side

#### Scenario: Rows that do not sum to one
- **WHEN** a mapping matrix has a row whose sum is not 1
- **THEN** `AnalysisError` is raised and names the row

### Requirement: Distribution distances between particle sets
`distribution_distance(source, reference, kind=...)` SHALL return, at every frame, the distance between the positions of
`source` and of `reference` as distributions: `'mmd'` the maximum mean discrepancy with a Gaussian kernel,
`'sinkhorn'` the debiased Sinkhorn divergence, and `'wasserstein'` the exact 2-Wasserstein distance between sets of equal
size, across periodic sides when a box is given.

#### Scenario: Identical sets
- **WHEN** a particle set is compared with itself under every kind
- **THEN** every distance is 0 within 1e-8

#### Scenario: Shifted sets
- **WHEN** a set of 500 points is compared with the same set shifted by 0.1 in open space
- **THEN** the Wasserstein distance is 0.1 within 1e-8, and the Sinkhorn divergence with a small regularisation lies
  within 5% of its square

#### Scenario: Unequal sizes for the exact distance
- **WHEN** the exact Wasserstein distance is asked for sets of different sizes
- **THEN** `AnalysisError` is raised and suggests the Sinkhorn divergence
