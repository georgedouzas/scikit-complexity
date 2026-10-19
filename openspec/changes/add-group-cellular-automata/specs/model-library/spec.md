# Spec Delta

## ADDED Requirements

### Requirement: Group automata models
The library SHALL provide the majority automaton on the free group, the lamplighter automaton, the discrete Laplacian
as a linear automaton, the reversible Q2R Ising automaton, the golden mean shift, and the majority automaton on the
Grigorchuk group as registered factories, each with a gallery example whose thumbnail is its Rerun screenshot and
which shows an analysis plot from its run.

#### Scenario: Q2R conserves energy
- **WHEN** the Q2R automaton runs on a periodic lattice
- **THEN** the Ising energy of the configuration stays constant at every step

#### Scenario: Grigorchuk automaton
- **WHEN** a user creates `grigorchuk_automaton` by name and runs it with its default parameters
- **THEN** it runs the majority rule with the memory set `{1, a, b, c, d}` on a ball of radius 8 with 271 sites, every
  interior cell takes the majority value of itself and its four neighbours, and the space reports the ball sizes 1, 5,
  11, 23, 40, 68, 108, 176, and 271 for radii 0 to 8

#### Scenario: Grigorchuk gallery plot
- **WHEN** a user opens the gallery example of `grigorchuk_automaton`
- **THEN** below the Rerun screenshot of the Cayley ball it plots the ball sizes for radii 0 to 10, and the text states
  that the local log-log slope rises while `log |B(n)| / n` falls, so the growth is faster than polynomial and slower
  than exponential

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
  - `grigorchuk_automaton` in `computer_science/automata` (CC 6.9)

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
