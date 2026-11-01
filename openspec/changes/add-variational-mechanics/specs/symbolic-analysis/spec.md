# Spec Delta

## ADDED Requirements

### Requirement: Poisson brackets decide conservation
`InvariantAnalysis` SHALL compute the Poisson brackets of expressions given as `ObservableSpec`s over the canonical
pairs of a system's `HamiltonianInteraction`s, and SHALL decide whether each expression is conserved, from its Poisson
bracket with the Hamiltonian for a Hamiltonian system and from its Lie derivative along the vector field otherwise. It
SHALL decide symbolically, or numerically at random states when simplification does not decide, and SHALL report which
method decided.

#### Scenario: Energy and angular momentum
- **WHEN** a user checks the energy, the angular momentum, and the first momentum component of the Kepler problem
- **THEN** the energy and the angular momentum are reported conserved, and the momentum component is reported not
  conserved with its rate of change

#### Scenario: Jacobi integral of a Lagrangian system
- **WHEN** a user checks the Jacobi integral of a restricted three-body problem written with `EulerLagrangeInteraction`
- **THEN** it is reported conserved from the Lie derivative along the vector field

#### Scenario: Brackets without canonical pairs
- **WHEN** a user asks for the Poisson brackets of a system with no `HamiltonianInteraction`
- **THEN** `AnalysisError` is raised that names `HamiltonianInteraction`

### Requirement: Symmetries are checked by the analysis
`InvariantAnalysis` SHALL take a symmetry as a `SymmetrySpec` whose generator maps coordinates to their infinitesimal
variations, given to the analysis rather than declared on the system, SHALL check it against the Hamiltonian of the
system's `HamiltonianInteraction`s or the summed Lagrangian of its `EulerLagrangeInteraction`s, and SHALL return the
Noether charge of the symmetry.

#### Scenario: Rotation invariance
- **WHEN** a user checks the rotation of the plane against the Kepler problem
- **THEN** the analysis confirms invariance, and the Noether charge equals the angular momentum

#### Scenario: Broken symmetry
- **WHEN** a user checks a translation of the first coordinate against Hénon-Heiles
- **THEN** the analysis reports no invariance and returns the first-order change of the Hamiltonian as the remainder
