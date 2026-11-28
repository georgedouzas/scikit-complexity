## ADDED Requirements

### Requirement: Pólya urns draw once per step in discrete time
`PolyaUrnSystem(time='discrete')` SHALL draw exactly one ball in every urn per step, and its docstring SHALL say that
in continuous time the draws happen at the times of a Poisson process of rate 1.

#### Scenario: Two reds in two draws
- **WHEN** many urns with one red and one blue ball run two discrete steps
- **THEN** the share with two reds is close to 1/3

### Requirement: Galton Watson offspring laws and generation sizes
`GaltonWatsonSystem` SHALL take a Poisson, geometric, or finite offspring law, and SHALL track the size of every
generation as one count when `individuals=False`.

#### Scenario: Geometric offspring
- **WHEN** a Galton Watson process with geometric offspring of mean 0.8 runs many times
- **THEN** the mean size of generation 3 is close to 0.8 cubed

#### Scenario: Cheap supercritical runs
- **WHEN** 400 runs with Poisson offspring of mean 2.5 over 12 generations run with `individuals=False`
- **THEN** they finish in a few seconds and their mean sizes match those of the individuals

#### Scenario: Finite law
- **WHEN** the offspring probabilities are `[0.25, 0.5, 0.25]`
- **THEN** the mean size of generation 1 from one individual is close to 1

### Requirement: West Brown Enquist runs scale with the vessels
`WestBrownEnquistSystem` SHALL run in a time that grows roughly with the number of vessels as its depth grows.

#### Scenario: One level deeper
- **WHEN** a network of branching ratio 2 runs at depth 7 and at depth 8
- **THEN** the second run takes less than four times as long as the first
