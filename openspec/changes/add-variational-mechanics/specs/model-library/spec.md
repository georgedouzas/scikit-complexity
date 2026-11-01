# Spec Delta

## ADDED Requirements

### Requirement: Hamiltonian mechanics models
The library SHALL build the Kepler problem `two_body_orbit`, the restricted three-body problem, and Hénon-Heiles on
`HamiltonianInteraction`, and their catalog entries SHALL pass the Hamiltonian as an observable with an invariant
tolerance, so the `InvariantObserver` checks its drift.

#### Scenario: Kepler orbit
- **WHEN** `two_body_orbit` runs for ten periods of a bound orbit with `Yoshida4Solver`
- **THEN** the orbit closes on itself within 1e-6 relative error and the energy drift stays below 1e-8

#### Scenario: Hénon-Heiles below the escape energy
- **WHEN** `henon_heiles` runs from an energy of 1/12 to time 1000
- **THEN** the orbit stays bounded and the energy stays within its invariant tolerance

### Requirement: Variational mechanics models are recorded in the model inventory
Every model this change rebuilds SHALL keep its name and field path in the model inventory, and SHALL list the
`variational-mechanics` primitive.

#### Scenario: Inventory models rebuilt
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are rebuilt on `HamiltonianInteraction` under their names and field paths, list the primitive
  `variational-mechanics`, and are marked implemented:
  - `two_body_orbit` in `physics/celestial_mechanics`
  - `restricted_three_body` in `physics/celestial_mechanics`
  - `henon_heiles` in `physics/nonlinear_oscillators`
