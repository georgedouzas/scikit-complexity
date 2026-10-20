# Spec Delta

## ADDED Requirements

### Requirement: Models on lattice geometries and graphs
The library SHALL provide, as registered factories:
- the Ising model on a graph under `physics/statistical_mechanics` (Argun et al. 2021 §2.1);
- the Drossel-Schwabl forest fire on triangular lattices and on graphs under `physics/criticality` (Argun et al. 2021
  ch. 3);
- the FHP lattice gas under `physics/fluids` (Dilão 2023 §5.16);
- the Bak-Sneppen model of coevolution with the extremal update under `biology/evolution` (Gros 2015 §6.6, Thurner et
  al. 2018 §5.4.4, Jensen 2023 §4.2).

#### Scenario: FHP conserves mass and momentum
- **WHEN** the FHP lattice gas runs for 500 steps on a periodic 64 × 64 triangular lattice from random occupations
- **THEN** the particle number and both components of the total momentum are the same at every step as at the start

#### Scenario: Bak-Sneppen self-organises
- **WHEN** Bak-Sneppen runs on a ring of 200 species for 10⁶ extremal updates after a transient of 10⁶ updates
- **THEN** the median fitness exceeds 0.75, and the largest minimum recorded in the events lies between 0.6 and 0.72,
  around the critical value 0.667

#### Scenario: Ising on a graph orders
- **WHEN** the Ising model on a graph runs on a random 4-regular graph of 1000 nodes at temperatures 1.0 and 6.0
  with ensembles
- **THEN** the mean absolute magnetisation is above 0.9 at temperature 1.0 and below 0.1 at temperature 6.0

#### Scenario: Forest fire on a graph
- **WHEN** the forest fire runs on a connected Watts-Strogatz graph of 500 nodes, fully forested, with no growth and no
  lightning, from one burning node
- **THEN** every node burns, and each node starts burning at the step equal to its graph distance from the first one
