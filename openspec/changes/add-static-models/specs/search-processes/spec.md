# Spec Delta

## Purpose

Lets a user model optimisers as processes: a population of candidate solutions moving on an objective landscape, one
generation per time step.

## ADDED Requirements

### Requirement: Searches declare variables, bounds, objective, and algorithm
Elements SHALL declare a search by its decision variables with bounds, an objective law over a symbolic expression
compiled like any law, an algorithm, and a population size. The algorithms SHALL include a genetic algorithm,
differential evolution, particle swarm optimisation, an evolution strategy, and simulated annealing. A search SHALL
require discrete time, and validation SHALL report it under any other time.

#### Scenario: Unknown algorithm
- **WHEN** a search names an algorithm that is not provided
- **THEN** construction raises a named error that lists the provided algorithms

#### Scenario: Search under static time
- **WHEN** a search is placed in a system with `StaticTime()`
- **THEN** validation reports the search and suggests `DiscreteTime()`

### Requirement: Generations are time steps
A run SHALL advance a search by one generation per step of discrete time. The frame SHALL hold the candidates as the
members of a group and their objective values, and the catalog entry SHALL declare the reductions `best` and `mean` as
observables. Runs with the same seed SHALL be identical.

#### Scenario: Differential evolution on Rastrigin
- **WHEN** differential evolution with 40 candidates searches the two-dimensional Rastrigin function on `[-5.12, 5.12]`
  with `random_state=1`
- **THEN** the best value is below 1e-3 by frame 50, and it never increases from one frame to the next

### Requirement: Annealing follows a temperature schedule
Simulated annealing SHALL accept a worse candidate with the Metropolis probability of the sampling law at the current
temperature, and the temperature SHALL follow a schedule over frames.

#### Scenario: Cooling
- **WHEN** simulated annealing runs with a temperature that falls geometrically to 1e-3
- **THEN** the fraction of accepted worse moves falls below 0.05 in the last tenth of the run
