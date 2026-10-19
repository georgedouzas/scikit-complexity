# Spec Delta

## Purpose

Lets a user model optimisers as processes: a population of candidate solutions moving on an objective landscape, one
generation per frame.

## ADDED Requirements

### Requirement: Searches declare variables, bounds, objective, and algorithm
Elements SHALL declare a search by its decision variables with bounds, a symbolic objective compiled like any law,
an algorithm, and a population size. The algorithms SHALL include a genetic algorithm, differential evolution,
particle swarm optimisation, an evolution strategy, and simulated annealing.

#### Scenario: Unknown algorithm
- **WHEN** a search names an algorithm that is not provided
- **THEN** construction raises a named error that lists the provided algorithms

### Requirement: Generations are frames
A run SHALL advance a search by one generation per frame. The frame SHALL hold the candidates as the members of a
group, their objective values, and the observables `best` and `mean`. Runs with the same seed SHALL be identical.

#### Scenario: Differential evolution on Rastrigin
- **WHEN** differential evolution with 40 candidates searches the two-dimensional Rastrigin function on `[-5.12, 5.12]`
  with `random_state=1`
- **THEN** the best value is below 1e-3 by frame 50, and it never increases from one frame to the next

### Requirement: Annealing follows a temperature schedule
Simulated annealing SHALL accept a worse candidate with the Metropolis probability at the current temperature, and
the temperature SHALL follow a schedule over frames.

#### Scenario: Cooling
- **WHEN** simulated annealing runs with a temperature that falls geometrically to 1e-3
- **THEN** the fraction of accepted worse moves falls below 0.05 in the last tenth of the run
