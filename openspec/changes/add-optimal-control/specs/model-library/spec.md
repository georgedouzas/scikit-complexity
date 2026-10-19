# Spec Delta

## ADDED Requirements

### Requirement: Control models
The library SHALL provide optimal harvesting, SIR vaccination, Ramsey optimal growth, a PID-stabilised inverted
pendulum, and a linear-quadratic regulated oscillator as registered factories.

#### Scenario: Vaccination lowers the peak
- **WHEN** the SIR vaccination problem is solved with a bounded vaccination rate and a cost on infections
- **THEN** the optimally controlled epidemic peak is lower than the uncontrolled one

### Requirement: Models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `sir_vaccination` in `biology/epidemiology`
  - `optimal_consumption` in `economics/macroeconomics`
  - `optimal_harvesting` in `economics/resources`
  - `pid_inverted_pendulum` in `engineering/control`
  - `lqr_oscillator` in `engineering/control`
  - `optimal_chemotherapy` in `biology/oncology`
  - `optimal_immunotherapy` in `biology/oncology`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
