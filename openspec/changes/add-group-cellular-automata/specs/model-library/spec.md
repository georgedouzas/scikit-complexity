# Spec Delta

## ADDED Requirements

### Requirement: Group automata models
The library SHALL provide the majority automaton on the free group, the lamplighter automaton, the discrete Laplacian
as a linear automaton, the reversible Q2R Ising automaton, and the golden mean shift as registered factories.

#### Scenario: Q2R conserves energy
- **WHEN** the Q2R automaton runs on a periodic lattice
- **THEN** the Ising energy of the configuration stays constant at every step
