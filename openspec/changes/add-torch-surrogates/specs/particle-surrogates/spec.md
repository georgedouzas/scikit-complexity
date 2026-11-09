## ADDED Requirements

### Requirement: Force matching fits equivariant network potentials
`force_matching` with `kind='equivariant'` SHALL fit a many-body equivariant network potential plus a repulsive prior
to the mapped forces, with a kind per bead, and the `system` of its result SHALL move the beads by the gradient of that
potential.

#### Scenario: Network on Lennard-Jones forces
- **WHEN** force matching with `kind='equivariant'` runs with the identity mapping on frames and exact forces of a
  Lennard-Jones fluid
- **THEN** the relative force error on held-out frames is below 0.05

#### Scenario: Many-body beads
- **WHEN** the dimer fluid of the force-matching probe is matched with `kind='equivariant'` and with `kind='linear'`
- **THEN** the relative force error of the equivariant fit is at most that of the linear fit

#### Scenario: Wrong number of bead kinds
- **WHEN** the bead kinds differ in length from the number of beads of the mapping
- **THEN** `AnalysisError` is raised and names both lengths
