# Spec Delta

## ADDED Requirements

### Requirement: Lattice models
The library SHALL provide elementary automata, Game of Life, majority rule, Langton's ant, Nagel-Schreckenberg traffic,
the hodgepodge machine, the Bak-Tang-Wiesenfeld sandpile, the Drossel-Schwabl forest fire, Olami-Feder-Christensen
earthquakes, and the Ising, Potts, and XY models as registered factories.

#### Scenario: Traffic jams
- **WHEN** the Nagel-Schreckenberg model runs at densities 0.1 and 0.5 with slowdown probability 0.3
- **THEN** the mean flow at density 0.5 is lower than at the critical density near 0.15, and jams appear as backward
  moving clusters of stopped cars
