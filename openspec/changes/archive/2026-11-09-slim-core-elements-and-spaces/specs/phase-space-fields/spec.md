## MODIFIED Requirements

### Requirement: Phase spaces are products of position and class axes
`PhaseSpace` SHALL be the product of a position lattice and one or more class axes, each a uniform grid over an
interval with zero-flux edges or a periodic grid. It SHALL be a lattice space, so a `FieldElement` of
`NonnegativeElement`, or of a realization of it, on it holds one value `x` per point of the product, and its laws are
written for one point. The compile report SHALL hold a phase space larger than its size limit, with the size it would
take.

#### Scenario: Size of a phase space
- **WHEN** a user creates a field over a periodic position lattice of 64 sites and a class axis of 128 points on
  `[-6, 6]`
- **THEN** the field holds 8192 values, and its initial values can be given as an array over position and class

#### Scenario: Periodic class axis
- **WHEN** a field over a periodic class axis of 32 angles is advected along that axis by a constant drift for one full
  turn
- **THEN** the profile returns to its initial values within the discretization error of the scheme
