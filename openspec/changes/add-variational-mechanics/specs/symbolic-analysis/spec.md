# Spec Delta

## ADDED Requirements

### Requirement: Poisson brackets decide conservation
The analysis layer SHALL compute Poisson brackets over a system's canonical pairs and decide whether an observable is
conserved, symbolically, or numerically when simplification fails, reporting which method decided.

#### Scenario: Energy and angular momentum
- **WHEN** a user checks the energy and the angular momentum of the Kepler problem
- **THEN** both are reported conserved, and the x-component of the momentum is reported not conserved with its rate of
  change
