# Spec Delta

## ADDED Requirements

### Requirement: Fractional models
The library SHALL provide fractional relaxation, a fractional damped oscillator, fractional Van der Pol, and fractional
Lorenz as registered factories with the order as an argument.

#### Scenario: Slower than exponential
- **WHEN** fractional relaxation runs with orders 1 and 0.5 from the same state
- **THEN** the order 0.5 trajectory stays above the exponential one at long times

### Requirement: Fractional dynamics models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `fractional_relaxation` in `mathematics/fractional_dynamics`
  - `fractional_oscillator` in `mathematics/fractional_dynamics`
  - `fractional_van_der_pol` in `mathematics/fractional_dynamics`
  - `fractional_lorenz` in `mathematics/fractional_dynamics`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
