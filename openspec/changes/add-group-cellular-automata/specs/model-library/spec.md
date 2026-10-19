# Spec Delta

## ADDED Requirements

### Requirement: Group automata models
The library SHALL provide the majority automaton on the free group, the lamplighter automaton, the discrete Laplacian
as a linear automaton, the reversible Q2R Ising automaton, and the golden mean shift as registered factories.

#### Scenario: Q2R conserves energy
- **WHEN** the Q2R automaton runs on a periodic lattice
- **THEN** the Ising energy of the configuration stays constant at every step

### Requirement: Models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `free_group_automaton` in `computer_science/automata`
  - `lamplighter_automaton` in `computer_science/automata`
  - `heisenberg_automaton` in `computer_science/automata`
  - `baumslag_solitar_automaton` in `computer_science/automata`
  - `linear_ring_automaton` in `computer_science/automata`
  - `subshifts` in `computer_science/automata`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
