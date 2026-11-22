## ADDED Requirements

### Requirement: Ensembles vary the seed of each replicate's system
An ensemble of a system that has a `random_state` parameter SHALL give every replicate a system seed spawned from the
experiment's seed, so replicates start from different initial states and stay reproducible.

#### Scenario: Replicates start apart
- **WHEN** an ensemble of 40 Curie Weiss runs at low temperature runs with seed 3
- **THEN** the initial spins differ between replicates and both signs of the magnetisation appear

#### Scenario: Ensembles stay reproducible
- **WHEN** the same ensemble runs twice with seed 3
- **THEN** both return identical results
