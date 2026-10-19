# Spec Delta

## ADDED Requirements

### Requirement: Chemical Langevin equations of reaction networks
The chemistry module SHALL describe binding and dissociation reactions as chemical Langevin equations, in which one
Wiener process per reaction drives every species the reaction changes, and SHALL provide the chemical Langevin
equation of the enzyme network `E + S <-> C -> E + P` as a registered factory.

#### Scenario: Binding conserves a difference
- **WHEN** `A + B -> C` runs as a chemical Langevin equation with `EulerMaruyama`
- **THEN** `A - B` stays at its initial value within 1e-9 relative error at every frame, while `A` fluctuates

#### Scenario: Enzyme totals under noise
- **WHEN** the chemical Langevin network runs for its default time with a fixed random state
- **THEN** `E + C` and `S + C + P` stay at their initial values within 1e-9 relative error at every frame

#### Scenario: Two descriptions at once
- **WHEN** a binding reaction is both stochastic and in Langevin mode
- **THEN** compiling raises a named error that names the interaction

### Requirement: Correlated geometric Brownian motions
The finance module SHALL provide two assets whose prices follow geometric Brownian motions driven by Wiener processes
with a given correlation, as a registered factory.

#### Scenario: Correlated returns
- **WHEN** the model runs with correlation 0.7 over 10 000 frames
- **THEN** the sample correlation of the two assets' log returns per frame is 0.7 within 0.05

### Requirement: Models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `chemical_langevin_network` in `chemistry/kinetics`
  - `correlated_geometric_brownian_motion` in `economics/finance`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
