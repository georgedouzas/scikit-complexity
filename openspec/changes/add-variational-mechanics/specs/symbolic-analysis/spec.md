# Spec Delta

## ADDED Requirements

### Requirement: Poisson brackets decide conservation
The analysis layer SHALL compute Poisson brackets over the canonical pairs of a system's canonical interactions and
decide whether an expression is conserved, symbolically, or numerically when simplification fails, reporting which
method decided.

#### Scenario: Energy and angular momentum
- **WHEN** a user checks the energy and the angular momentum of the Kepler problem
- **THEN** both are reported conserved, and the x-component of the momentum is reported not conserved with its rate of
  change

### Requirement: Symmetries are checked by the analysis
The analysis layer SHALL take symmetries as infinitesimal transformations of a system's coordinates, given to its
functions rather than declared on the system, check them against the function of the system's canonical or variational
interaction, and return the Noether charge of a continuous symmetry.

#### Scenario: Rotation invariance
- **WHEN** a user checks rotation symmetry in the plane against the Kepler problem
- **THEN** the check confirms invariance, and the Noether charge equals the angular momentum
