# Spec Delta

## ADDED Requirements

### Requirement: Kinetic models of active particles
The library SHALL provide wealth and welfare dynamics, criminality against security forces, tumour-immune
competition, and opinion formation as registered kinetic factories.

#### Scenario: Immune response
- **WHEN** the tumour-immune model runs with an immune activity above its threshold
- **THEN** the tumour density decays, and below the threshold it grows

### Requirement: Kinetic models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `immune_competition` in `biology/oncology`
  - `kinetic_wealth_welfare` in `economics/econophysics`
  - `kinetic_opinion` in `social/opinion_dynamics`
  - `evacuation` in `social/crowds`
  - `criminality` in `social/crowds`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
