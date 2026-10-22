# Spec Delta

## ADDED Requirements

### Requirement: Mechanics models
The library SHALL provide simple, double, and coupled pendulums, the Kepler problem, the restricted three-body problem,
and Hénon-Heiles as registered factories built from canonical or variational interactions under continuous time, and
the Chirikov standard map as a symplectic map under discrete time. Their catalog entries SHALL pass the relative drift
of the energy as an observable.

#### Scenario: Kepler orbit
- **WHEN** the Kepler problem runs for ten periods of a bound orbit with the Yoshida solver
- **THEN** the orbit closes on itself within 1e-6 relative error and the energy drift stays below 1e-8

### Requirement: Variational mechanics models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are rebuilt on the new formalism under their names and field paths, and marked implemented:
  - `pendulum` in `physics/mechanics`
  - `double_pendulum` in `physics/mechanics`
  - `coupled_pendulums` in `physics/mechanics`
  - `two_body_orbit` in `physics/celestial_mechanics`
  - `restricted_three_body` in `physics/celestial_mechanics`
  - `standard_map` in `mathematics/dynamical_systems`
  - `henon_heiles` in `physics/nonlinear_oscillators`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
